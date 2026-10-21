"""Independent partition, integer-count target, protocol and saved-logit audit."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits
import run_screen as screen

ROOT = screen.ROOT
d = screen.d


def reference_group(assignment, train, observed, classes):
    groups = assignment[train]
    counts = np.zeros((int(assignment.max())+1, classes), dtype=np.int64)
    np.add.at(counts, (groups, observed), 1)
    histogram = counts[groups].copy()
    histogram[np.arange(len(train)), observed] -= 1
    support = histogram.sum(1)
    r = np.full(histogram.shape, 1/classes, dtype=np.float64)
    np.divide(histogram, support[:, None], out=r, where=support[:, None] > 0)
    maximum = histogram.max(1)
    unique = (histogram == maximum[:, None]).sum(1) == 1
    q = r.copy()
    q[unique] *= .5
    q[np.flatnonzero(unique), histogram[unique].argmax(1)] += .5
    alpha = np.maximum(0., (classes*r.max(1)-1)/(classes-1))
    alpha[support < 2] = 0.
    target = (1-alpha[:, None])*np.eye(classes)[observed]+alpha[:, None]*q
    return dict(r=r, q=q, alpha=alpha, support=support, target=target)


def close(actual, expected):
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=2e-6, rtol=0), np.max(np.abs(actual-expected))


def verify_bfs(name):
    graph = d.load_graph(name, 0)
    folder = ROOT/'partitions'/name
    for path, digest in screen.read(folder/'frozen.json').items():
        assert d.sha(path) == digest
    assignment = np.load(folder/'BFS.npy')
    balls = screen.read(folder/'BFS_balls.json')
    n = graph.num_nodes
    edge = graph.edge_index.numpy()
    keep = edge[0] != edge[1]
    u, v = edge[:, keep]
    adjacency = sparse.csr_matrix((np.ones(2*len(u)),
        (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))
    adjacency.data[:] = 1.
    active = np.ones(n, dtype=bool)
    degrees = np.diff(adjacency.indptr).astype(np.int64)
    for ball in balls:
        members = np.flatnonzero(assignment == ball['ball_id'])
        assert len(members) == ball['size'] and len(members) <= int(np.sqrt(n))
        candidates = np.flatnonzero(active)
        center = int(candidates[np.argmax(degrees[candidates])])
        assert center == ball['center'] and int(degrees[center]) == ball['center_residual_degree']
        local = adjacency[members][:, members]
        components, _ = sparse.csgraph.connected_components(local, directed=False)
        assert components == 1
        active[members] = False
        for node in members:
            neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]
            degrees[neighbors[active[neighbors]]] -= 1
        degrees[members] = -1
    assert not active.any()
    return dict(dataset=name, balls=len(balls), status='passed')


def verify_case(folder):
    name = folder.parent.name
    seed = int(folder.name[4:])
    graph = d.load_graph(name, seed)
    tr = graph.train_index.numpy()
    with np.load(folder/'split.npz') as saved:
        for key, expected in [('train', tr), ('val', graph.val_index.numpy()), ('test', graph.test_index.numpy())]:
            assert np.array_equal(saved[key], expected)
    with np.load(folder/'assignments.npz') as saved:
        assignments = {k: saved[k].copy() for k in saved.files}
    for assignment in assignments.values():
        assert assignment.shape == (graph.num_nodes,) and assignment.min() >= 0
    for indices in [np.arange(graph.num_nodes), tr]:
        assert np.array_equal(np.bincount(assignments['RANDOM'][indices], minlength=assignments['BFS'].max()+1),
                              np.bincount(assignments['BFS'][indices], minlength=assignments['BFS'].max()+1))
    relation_meta = screen.read(folder/'relations.json')
    relations = {arm: sparse.load_npz(folder/(arm+'_relation.npz')) for arm in relation_meta}
    for arm, relation in relations.items():
        assert not np.any(relation.diagonal())
        assert d.fingerprint(relation.indptr, relation.indices, relation.data) == relation_meta[arm]['hash']
        if arm != 'H_KNN':
            expected = d.group_weights(assignments[arm][tr])
            assert (expected != relation).nnz == 0
    assert np.array_equal(np.diff(relations['BFS'].indptr), np.diff(relations['H_KNN'].indptr))
    metrics = screen.read(folder/'metrics.json')
    for noise, rate in screen.NOISE.items():
        data = screen.inject_train_label_noise(graph, 'uniform' if noise == 'clean' else noise, rate, seed)
        parent = folder/noise
        observed = data.noisy_y[tr].numpy()
        clean = data.y[tr].numpy()
        protocol = screen.read(parent/'protocol.json')
        assert d.fingerprint(tr, data.val_index.numpy(), data.test_index.numpy()) == protocol['split_hash']
        assert d.fingerprint(observed) == protocol['noise_hash']
        assert torch.equal(data.y[data.val_mask], data.noisy_y[data.val_mask])
        assert torch.equal(data.y[data.test_mask], data.noisy_y[data.test_mask])
        for arm, digest in screen.read(parent/'frozen_targets.json').items():
            path = parent/(arm+'.npz')
            assert d.sha(path) == digest
            with np.load(path) as saved:
                values = {k: saved[k].copy() for k in saved.files}
            if arm in assignments:
                reference = reference_group(assignments[arm], tr, observed, data.num_classes)
                for key, expected in reference.items():
                    close(values[key], expected)
            elif arm == 'CE':
                close(values['target'], np.eye(data.num_classes)[observed])
            else:
                # KNN has a different candidate set; count every unique voter directly.
                relation = relations[arm]
                hist = np.zeros_like(values['q'], dtype=np.float64)
                for node in range(len(tr)):
                    ids = relation.indices[relation.indptr[node]:relation.indptr[node+1]]
                    hist[node] = np.bincount(observed[ids], minlength=data.num_classes)
                support = hist.sum(1)
                r = np.full(hist.shape, 1/data.num_classes, dtype=np.float64)
                np.divide(hist, support[:, None], out=r, where=support[:, None] > 0)
                close(values['r'], r)
                alpha = np.maximum(0., (data.num_classes*r.max(1)-1)/(data.num_classes-1))
                alpha[support < 2] = 0.
                close(values['alpha'], alpha)
                q = r.copy()
                unique = (hist == hist.max(1)[:, None]).sum(1) == 1
                q[unique] *= .5
                q[np.flatnonzero(unique), hist[unique].argmax(1)] += .5
                close(values['q'], q)
                close(values['target'], (1-alpha[:, None])*np.eye(data.num_classes)[observed]+alpha[:, None]*q)
            close(values['target'].sum(1), np.ones(len(tr)))
            assert values['target'].min() >= 0 and np.isfinite(values['target']).all()
            brier = float(np.square(values['target']-np.eye(data.num_classes)[clean]).sum(1).mean())
            row = [r for r in metrics if r['arm'] == arm and r['noise'] == noise]
            assert len(row) == 1 and abs(row[0]['target_brier']-brier) < 2e-6
    return dict(dataset=name, seed=seed, target_records=len(metrics), status='passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--students-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    for manifest in ['source_manifest.json', 'student_source_manifest.json']:
        if (ROOT/manifest).exists():
            for path, digest in screen.read(ROOT/manifest).items():
                assert d.sha(path) == digest, path
    started = time.time()
    if args.students_only:
        prior = screen.read(ROOT/'audit/report.json')
        assert prior['status'] == 'passed'
        partitions, cases = prior['partitions'], prior['cases']
    else:
        with threadpool_limits(limits=4):
            partitions = [verify_bfs(name) for name in screen.NAMES]
            cases = []
            for folder in sorted((ROOT/'runs').glob('*/seed*')):
                print('AUDIT', folder.relative_to(ROOT), flush=True)
                cases.append(verify_case(folder))
        assert len(cases) == 15 and sum(r['target_records'] for r in cases) == 315
    students = []
    for file in sorted((ROOT/'students').glob('*/seed*/*/*/result.json')):
        result = screen.read(file)
        config = screen.read(file.parent/'config.json')
        assert d.sha(config['target_path']) == result['target_hash'] == config['target_hash']
        graph = d.load_graph(result['dataset'], result['seed'])
        logits = torch.load(file.parent/'logits.pt', map_location='cpu', weights_only=True)
        for indices, key in [(graph.val_index, 'validation_accuracy'), (graph.test_index, 'test_accuracy')]:
            accuracy = float((logits[indices].argmax(1) == graph.y[indices]).float().mean())
            assert abs(accuracy-result[key]) < 1e-8
        students.append(dict(dataset=result['dataset'], seed=result['seed'], noise=result['noise'],
                             arm=result['arm'], status='passed'))
    d.write(ROOT/'audit/report.json', dict(status='passed', partitions=partitions, cases=cases,
        students=students, verified_seconds=time.time()-started, auditor_source_hash=d.sha(__file__)))
    print('PASS partitions=%d protocol_cases=%d target_rows=%d students=%d' %
          (len(partitions), len(cases), sum(r['target_records'] for r in cases), len(students)), flush=True)


if __name__ == '__main__':
    main()
