"""Freeze label-free partitions/relations and original LCSS targets before evaluation."""
from pathlib import Path
import json
import logging
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
MECHANISM = ROOT.parent/'heterophily_mechanism_20261014'
sys.path.insert(0, str(MECHANISM))
import diagnostic as d
from gblcss.data import inject_train_label_noise
from gblcss.granular_ball_topology import build_topology_balls, random_support_matched_assignment

NAMES = ['corafull', 'roman', 'coauthorcs']
SEEDS = list(range(5))
NOISE = {'clean': 0., 'uniform': .4, 'pair': .4}
ARMS = ['CE', 'OLD64', 'BFS', 'H_GEOM', 'X_GEOM', 'RANDOM', 'H_KNN']
H = dict(d.CACHED_H, roman=MECHANISM/'representations/Roman-empire/grace/H.pt')
GEOMETRY = dict(n_init=1, max_iter=50, tol=1e-4, init='k-means++',
                bisecting_strategy='biggest_inertia', random_state=0)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prior_folder(name, seed):
    return MECHANISM/'runs'/name/'grace'/f'seed{seed}'


def geometric_partition(values, count, sparse_input=False):
    values = values.astype(np.float32, copy=False)
    if sparse_input:
        values = sparse.csr_matrix(values)
    values = normalize(values, norm='l2', copy=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = BisectingKMeans(n_clusters=count, **GEOMETRY)
        assignments = model.fit_predict(values)
    return assignments.astype(np.int64), dict(requested_balls=count,
        actual_balls=int(np.unique(assignments).size), inertia=float(model.inertia_),
        sparse_input=sparse_input, feature_dimension=values.shape[1],
        warnings=[str(w.message) for w in caught])


def freeze_sources():
    paths = [ROOT/'PLAN.md', Path(__file__),
             WORKSPACE/'GB-LCSS_code/src/gblcss/granular_ball_topology.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/targets.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/data.py',
             WORKSPACE/'GB-LCSS_code/src/gblcss/representation.py', Path(d.__file__)]
    for name in NAMES:
        paths.extend([d.DATA[name], H[name]])
        for seed in SEEDS:
            old = prior_folder(name, seed)
            paths.extend([old/'assignments.npz', old/'split.npz'])
            for noise in NOISE:
                paths.extend([old/noise/'protocol.json', old/noise/'frozen_targets.json', old/noise/'GB64.npz'])
    manifest = {str(p.resolve()): d.sha(p) for p in paths}
    existing = ROOT/'source_manifest.json'
    if existing.exists() and read(existing) != manifest:
        raise ValueError('Sources changed: use a new experiment version')
    d.write(existing, manifest)
    return manifest


def build_partitions(name):
    folder = ROOT/'partitions'/name
    if (folder/'stats.json').exists():
        frozen = read(folder/'frozen.json')
        for p, digest in frozen.items():
            if d.sha(p) != digest:
                raise ValueError('Frozen partition changed: '+p)
        return
    graph = d.load_graph(name, 0)
    h = d.load_frozen_representation(H[name], graph.num_nodes).numpy()
    folder.mkdir(parents=True, exist_ok=True)
    logging.info('PARTITION %s N=%d Xdim=%d', name, graph.num_nodes, graph.x.shape[1])
    started = time.perf_counter()
    result = build_topology_balls(graph.edge_index, graph.num_nodes)
    stats = {'BFS': dict(result.summary, seconds=time.perf_counter()-started)}
    count = int(np.unique(result.assignment).size)
    assignments = {'BFS': result.assignment}
    d.write(folder/'BFS_balls.json', result.balls)
    for arm, values, use_sparse in [('H_GEOM', h, False), ('X_GEOM', graph.x.numpy(), True)]:
        logging.info('GEOMETRY %s %s K=%d', name, arm, count)
        started = time.perf_counter()
        assignment, details = geometric_partition(values, count, use_sparse)
        assignments[arm] = assignment
        stats[arm] = dict(details, seconds=time.perf_counter()-started)
        logging.info('GEOMETRY_DONE %s %s balls=%d seconds=%.2f', name, arm,
                     details['actual_balls'], stats[arm]['seconds'])
    for arm, assignment in assignments.items():
        np.save(folder/(arm+'.npy'), assignment)
    d.write(folder/'stats.json', stats)
    files = [folder/(a+'.npy') for a in assignments]+[folder/'stats.json', folder/'BFS_balls.json']
    d.write(folder/'frozen.json', {str(p.resolve()): d.sha(p) for p in files})


def support_statistics(assignment, train):
    labels, sizes = np.unique(assignment, return_counts=True)
    counts = np.bincount(assignment[train], minlength=int(labels.max())+1)[labels]
    return dict(actual_balls=len(labels), min_size=int(sizes.min()), median_size=float(np.median(sizes)),
                max_size=int(sizes.max()), singleton_balls=int((sizes == 1).sum()),
                zero_training_balls=int((counts == 0).sum()), unsupported_balls=int((counts < 3).sum()),
                mean_training_count=float(counts.mean()), min_training_count=int(counts.min()),
                training_count_quantiles=np.quantile(counts, [0, .25, .5, .75, 1]).tolist())


def run_case(name, seed):
    folder = ROOT/'runs'/name/f'seed{seed}'
    if (folder/'metrics.json').exists():
        return
    graph = d.load_graph(name, seed)
    tr = graph.train_index.numpy()
    old = prior_folder(name, seed)
    partition = ROOT/'partitions'/name
    assignments = {a: np.load(partition/(a+'.npy')) for a in ['BFS', 'H_GEOM', 'X_GEOM']}
    with np.load(old/'assignments.npz') as prior:
        assignments['OLD64'] = prior['GB64'].copy()
    assignments['RANDOM'] = random_support_matched_assignment(assignments['BFS'], tr, seed)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'split.npz', train=tr, val=graph.val_index.numpy(), test=graph.test_index.numpy())
    np.savez_compressed(folder/'assignments.npz', **assignments)
    d.write(folder/'partition_stats.json', {a: support_statistics(v, tr) for a, v in assignments.items()})
    started = time.perf_counter()
    relations = {a: d.group_weights(v[tr]) for a, v in assignments.items()}
    h = d.load_frozen_representation(H[name], graph.num_nodes).numpy()
    relations['H_KNN'] = d.knn_weights(h[tr], np.diff(relations['BFS'].indptr), 'cpu')
    for arm, relation in relations.items():
        sparse.save_npz(folder/(arm+'_relation.npz'), relation)
    d.write(folder/'relations.json', {a: dict(hash=d.fingerprint(w.indptr, w.indices, w.data),
        nnz=int(w.nnz), self_votes=int(np.count_nonzero(w.diagonal()))) for a, w in relations.items()})
    d.write(folder/'timing.json', dict(relations_seconds=time.perf_counter()-started,
                                      knn_device='cpu', cached_grace=True))
    rows = []
    for noise, rate in NOISE.items():
        data = inject_train_label_noise(graph, 'uniform' if noise == 'clean' else noise, rate, seed)
        observed = data.noisy_y[tr].numpy()
        classes = data.num_classes
        old_protocol = read(old/noise/'protocol.json')
        split_hash = d.fingerprint(tr, data.val_index.numpy(), data.test_index.numpy())
        noise_hash = d.fingerprint(observed)
        if split_hash != old_protocol['split_fingerprint'] or noise_hash != old_protocol['noisy_train_fingerprint']:
            raise ValueError('Historical split/noise mismatch')
        distributions = {arm: d.vote(w, observed, classes) for arm, w in relations.items()}
        onehot = np.eye(classes)[observed]
        zeros = np.zeros(len(tr))
        distributions['CE'] = (onehot, onehot, zeros, zeros, zeros)
        condition = folder/noise
        condition.mkdir(exist_ok=True)
        np.save(condition/'observed_train.npy', observed)
        frozen = {}
        for arm, (r, q, alpha, support, effective) in distributions.items():
            target = d.mix(q, alpha, observed).astype(np.float32)
            file = condition/(arm+'.npz')
            np.savez_compressed(file, r=r.astype(np.float32), q=q.astype(np.float32),
                alpha=alpha.astype(np.float32), target=target, support=support, effective=effective)
            frozen[arm] = d.sha(file)
        with np.load(old/noise/'GB64.npz') as prior, np.load(condition/'OLD64.npz') as current:
            if not np.array_equal(prior['target'], current['target']):
                raise ValueError('Historical OLD64 target differs')
        d.write(condition/'frozen_targets.json', frozen)
        d.write(condition/'protocol.json', dict(dataset=name, seed=seed, noise=noise, rate=rate,
            split_hash=split_hash, noise_hash=noise_hash, historical_protocol=str(old/noise/'protocol.json'),
            val_clean=bool(torch.equal(data.y[data.val_mask], data.noisy_y[data.val_mask])),
            test_clean=bool(torch.equal(data.y[data.test_mask], data.noisy_y[data.test_mask]))))
        # Only after all targets of this condition are frozen is clean Y evaluated.
        clean = data.y[tr].numpy()
        for arm, (_, q, alpha, support, _) in distributions.items():
            with np.load(condition/(arm+'.npz')) as artifact:
                metric = d.target_metrics(artifact['target'], q, alpha, clean, observed)
            positive = margin_mean = None
            if arm != 'CE':
                clean_r = d.vote(relations[arm], clean, classes)[0]
                competitor = clean_r.copy()
                competitor[np.arange(len(tr)), clean] = -np.inf
                margin = clean_r[np.arange(len(tr)), clean]-competitor.max(1)
                valid = support >= 2
                positive = float(np.mean(valid & (margin > 0)))
                margin_mean = float(np.mean(margin[valid])) if valid.any() else None
            high = alpha >= .75
            rows.append(dict(dataset=name, seed=seed, noise=noise, arm=arm, **metric,
                positive_clean_margin_fraction=positive, mean_clean_margin=margin_mean,
                mean_support=float(np.mean(support)), eligible_rate=float(np.mean(support >= 2)),
                high_alpha_wrong_q_fraction=float(np.mean(high & (q.argmax(1) != clean))),
                high_alpha_wrong_q_conditional=float(np.mean(q.argmax(1)[high] != clean[high])) if high.any() else None))
        logging.info('FROZEN %s seed%d %s', name, seed, noise)
    d.write(folder/'metrics.json', rows)
    summarize()


def summarize():
    rows = [row for file in sorted((ROOT/'runs').glob('*/seed*/metrics.json')) for row in read(file)]
    d.csv_write(ROOT/'target_metrics.csv', rows)
    if not rows:
        return
    frame = pd.DataFrame(rows)
    metrics = ['target_brier', 'q_accuracy', 'q_true_probability', 'clean_harm_rate',
               'noise_recovery_rate', 'eligible_rate', 'positive_clean_margin_fraction',
               'mean_support', 'high_alpha_wrong_q_fraction']
    summary = frame.groupby(['dataset', 'noise', 'arm'])[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT/'target_summary.csv', index=False)
    lookup = {(r['dataset'], r['seed'], r['noise'], r['arm']): r for r in rows}
    paired = []
    for row in rows:
        if row['arm'] != 'BFS':
            continue
        for reference in ['CE', 'OLD64', 'H_GEOM', 'X_GEOM', 'RANDOM', 'H_KNN']:
            other = lookup[row['dataset'], row['seed'], row['noise'], reference]
            paired.append(dict(dataset=row['dataset'], seed=row['seed'], noise=row['noise'], reference=reference,
                brier_delta=row['target_brier']-other['target_brier'],
                q_accuracy_delta=row['q_accuracy']-other['q_accuracy']))
    d.csv_write(ROOT/'paired_target_deltas.csv', paired)


def main():
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT/'screen_raw.log', encoding='utf-8')])
    manifest = freeze_sources()
    config = dict(datasets=NAMES, protocol_seeds=SEEDS, noise=NOISE, arms=ARMS,
        bfs_capacity='floor_sqrt_N', geometric_control=GEOMETRY, geometry_seed=0,
        geometry_sparse_X=True, min_other_votes=2, beta=.5, relation_device='cpu',
        existing_ssl_seeds=[0], stage='target_diagnostic', downstream_gate='PLAN.md')
    d.write(ROOT/'config.json', config)
    status = dict(status='running', started=time.time())
    d.write(ROOT/'screen_status.json', status)
    try:
        with threadpool_limits(limits=4):
            for name in NAMES:
                build_partitions(name)
                for seed in SEEDS:
                    run_case(name, seed)
        for p, digest in manifest.items():
            if d.sha(p) != digest:
                raise ValueError('Source changed during screen: '+p)
        status.update(status='completed', elapsed_seconds=time.time()-status['started'],
            target_records=len(pd.read_csv(ROOT/'target_metrics.csv')))
    except Exception as exc:
        status.update(status='failed', error=repr(exc))
        d.write(ROOT/'screen_status.json', status)
        raise
    d.write(ROOT/'screen_status.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
