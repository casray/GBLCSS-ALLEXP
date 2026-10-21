from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
SOURCE = ROOT.parent / 'gb_dist_protocol_seeds_20261007'
DATASETS = ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho']
DATA_FILES = {
    'cora': PROJECT / 'external/NoisyGL/data/Cora/processed/data.pt',
    'citeseer': PROJECT / 'external/NoisyGL/data/CiteSeer/processed/data.pt',
    'corafull': PROJECT / 'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
    'coauthorcs': PROJECT / 'external/NoisyGL/data/CS/processed/data.pt',
    'amazonpho': PROJECT / 'external/NoisyGL/data/Photo/processed/data.pt',
}
KM_CONFIG = dict(init='k-means++', n_init=10, random_state=0,
                 max_iter=300, tol=1e-4, algorithm='lloyd')
RANDOM_SEEDS = [100, 101, 102]
BUDGETS = [.1, .25, .5]
EXCEPTIONS = [0., .1, .25, .5, 1.]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def assignment_hash(a):
    a = np.ascontiguousarray(a, dtype='<i8')
    header = f'dtype={a.dtype.str};shape={a.shape[0]};order=C\n'.encode('ascii')
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def export_csv(name, rows):
    if not rows:
        return
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def moments(x, a):
    x, a = np.asarray(x), np.asarray(a, dtype=np.int64)
    assert x.ndim == 2 and a.shape == (len(x),)
    sizes = np.bincount(a)
    assert sizes.min() > 0
    means = np.empty((len(sizes), x.shape[1]), dtype=np.float64)
    ss = np.empty(len(sizes), dtype=np.float64)
    for j in range(len(sizes)):
        block = x[a == j]
        means[j] = block.mean(axis=0, dtype=np.float64)
        squared = np.einsum('ij,ij->', block, block, dtype=np.float64)
        ss[j] = max(0., squared - sizes[j] * np.dot(means[j], means[j]))
    return sizes, means, ss


def allocate(sizes, budget, weights):
    sizes = np.asarray(sizes, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    assert len(sizes) <= budget <= sizes.sum()
    assert weights.shape == sizes.shape and np.all(weights >= 0)
    allocation = np.ones(len(sizes), dtype=np.int64)
    while allocation.sum() < budget:
        left = int(budget - allocation.sum())
        capacity = sizes - allocation
        active_weights = np.where(capacity > 0, weights, 0.)
        if active_weights.sum() == 0:
            active_weights = capacity.astype(float)
        quota = left * active_weights / active_weights.sum()
        addition = np.minimum(capacity, np.floor(quota).astype(np.int64))
        if addition.sum() == 0:
            rank = np.argsort(-np.where(capacity > 0, quota, -1.), kind='stable')
            addition[rank[:min(left, int((capacity > 0).sum()))]] = 1
        allocation += addition
    assert allocation.sum() == budget and np.all(allocation <= sizes)
    return allocation


def sampling_mse(sizes, ss, allocation):
    sizes, allocation = np.asarray(sizes), np.asarray(allocation)
    variance = np.divide(ss, sizes - 1, out=np.zeros(len(sizes)), where=sizes > 1)
    return float(np.sum((sizes / sizes.sum())**2 * (1-allocation/sizes) * variance / allocation))


def random_groups(a, seed):
    return np.random.default_rng(seed).permutation(a)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def probabilities(path):
    p = torch.load(path, map_location='cpu', weights_only=True).numpy().astype(np.float64)
    assert np.isfinite(p).all() and np.min(p) >= 0
    assert np.max(np.abs(p.sum(1) - 1)) < 1e-5
    return p / p.sum(1, keepdims=True)


def comparison(p, q):
    difference = p-q
    kl = np.sum(p*(np.log(np.maximum(p, 1e-12))-np.log(np.maximum(q, 1e-12))), axis=1)
    return dict(mse=float(np.mean(np.sum(difference**2, axis=1))),
                kl=float(np.mean(kl)), agreement_pct=float(np.mean(p.argmax(1) == q.argmax(1))*100))


def compression_curve(p, a, fractions):
    sizes, means, _ = moments(p, a)
    approximate = means[a]
    errors = np.sum((p-approximate)**2, axis=1)
    order = np.argsort(-errors, kind='stable')
    rows = []
    for fraction in fractions:
        count = int(np.floor(len(p)*fraction))
        q = approximate.copy()
        q[order[:count]] = p[order[:count]]
        storage = len(p)*4 + means.size*4 + count*(4+p.shape[1]*4)
        rows.append(dict(exception_fraction=fraction, exception_count=count,
                         bytes=int(storage), ratio_to_fp32=storage/p.size/4,
                         ratio_to_fp16=storage/p.size/2, **comparison(p, q)))
    return rows


def low_variance_attribute_ratio(sizes, teacher_ss, x_ss, x_global_var):
    order = np.argsort(teacher_ss/sizes, kind='stable')
    # Fractional boundary weight makes this diagnostic cover exactly 25% of nodes.
    remaining, numerator = sizes.sum()*.25, 0.
    for j in order:
        take = min(remaining, sizes[j])
        numerator += take*x_ss[j]/sizes[j]
        remaining -= take
        if remaining <= 0:
            break
    return float(numerator/(sizes.sum()*.25)/x_global_var) if x_global_var > 0 else None


def freeze():
    assert not (ROOT/'frozen_inputs.json').exists(), 'do not overwrite a frozen run'
    paths = [ROOT/'PLAN.md', ROOT/'diagnose.py', ROOT/'test_diagnostic.py',
             SOURCE/'frozen_inputs.json', SOURCE/'summary.json']
    source_hashes = read(SOURCE/'frozen_inputs.json')['hashes']
    for dataset in DATASETS:
        paths.append(DATA_FILES[dataset])
        for rate in (.4, .6):
            for seed in (1, 2):
                folder = SOURCE/'runs'/dataset/f'uniform{rate}'/f'seed{seed}'
                meta = read(folder/'protocol.json')
                paths += [folder/'protocol.json', folder/'protocol.pt',
                          folder/'teacher/probabilities.pt', folder/'GB0/probabilities.pt',
                          folder/'GB0/epoch200_probabilities.pt', Path(meta['representation_source']),
                          Path(meta['partition_source'])]
    hashes = {}
    known = {str(Path(p).resolve()): value for p, value in source_hashes.items()}
    for path in sorted(set(paths)):
        path = path.resolve()
        value = sha(path)
        if str(path) in known:
            assert value == known[str(path)], f'changed original input: {path}'
        hashes[str(path)] = value
    write(ROOT/'frozen_inputs.json', dict(hashes=hashes, teacher_output_labels='not read',
          evaluation_labels_used=False, datasets=DATASETS, protocol_seeds=[1,2],
          noise_rates=[.4,.6], random_seeds=RANDOM_SEEDS, budgets=BUDGETS, exceptions=EXCEPTIONS,
          kmeans=KM_CONFIG, threads=4))
    write(ROOT/'environment.json', dict(executable=sys.executable, python=sys.version,
          numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__,
          platform=platform.platform(), compute='CPU offline statistics, no training'))
    print(f'FROZEN {len(hashes)} inputs and code files', flush=True)


def verify():
    for path, expected in read(ROOT/'frozen_inputs.json')['hashes'].items():
        assert sha(path) == expected, f'input changed: {path}'


def run():
    assert not (ROOT/'execution.json').exists(), 'completed diagnostics must not be overwritten'
    verify()
    torch.set_num_threads(4)
    all_metrics, all_curves, all_balls, all_sampling, all_allocations = [], [], [], [], []
    groups_cache = {}
    started = time.perf_counter()
    for dataset in DATASETS:
        stored = torch.load(DATA_FILES[dataset], map_location='cpu', weights_only=False)
        raw = stored[0]
        attributes = raw['x'] if isinstance(raw, dict) else raw.x
        x = normalize(attributes.numpy())
        del stored, raw, attributes
        x_global_var = float(moments(x, np.zeros(len(x), dtype=int))[2].sum()/len(x))
        for seed in (1, 2):
            base = SOURCE/'runs'/dataset/'uniform0.4'/f'seed{seed}'
            meta = read(base/'protocol.json')
            # The protocol archive contains labels; only the assignment field is used.
            archive = torch.load(base/'protocol.pt', map_location='cpu', weights_only=True)
            gb = archive['assignment'].numpy().astype(np.int64)
            del archive
            assert assignment_hash(gb) == meta['assignment_hash']
            original = np.load(meta['partition_source'])['assignment']
            np.testing.assert_array_equal(gb, original)
            assert len(gb) == len(x)
            k = len(np.unique(gb))
            assert np.array_equal(np.unique(gb), np.arange(k))
            cache_key = (meta['representation_source'], k)
            if cache_key not in groups_cache:
                tensor = torch.load(meta['representation_source'], map_location='cpu', weights_only=True)
                if isinstance(tensor, dict):
                    tensor = next(tensor[key] for key in ('H','h','embeddings','embedding') if key in tensor)
                h = normalize(tensor.numpy())
                assert len(h) == len(x)
                tick = time.perf_counter()
                with threadpool_limits(limits=4):
                    model = KMeans(n_clusters=k, **KM_CONFIG).fit(h)
                km = model.labels_.astype(np.int64)
                groups_cache[cache_key] = km
                target = ROOT/'partitions'/f'{dataset}_K{k}'
                target.mkdir(parents=True, exist_ok=True)
                np.save(target/'assignment.npy', km)
                write(target/'config.json', dict(count=k, seconds=time.perf_counter()-tick,
                      embedding_source=meta['representation_source'], inertia=float(model.inertia_),
                      assignment_hash=assignment_hash(km), kmeans=KM_CONFIG))
                del h, tensor, model
                print(f'KM {dataset} K={k} saved', flush=True)
            groups = [('GB', -1, gb), ('KM', -1, groups_cache[cache_key])]
            groups += [('Random', rs, random_groups(gb, rs)) for rs in RANDOM_SEEDS]
            x_stats = {(name, rs): moments(x, a)[2] for name, rs, a in groups}
            for rate in (.4, .6):
                folder = SOURCE/'runs'/dataset/f'uniform{rate}'/f'seed{seed}'
                current = read(folder/'protocol.json')
                assert current['assignment_hash'] == meta['assignment_hash']
                p = probabilities(folder/'teacher/probabilities.pt')
                student = probabilities(folder/'GB0/probabilities.pt')
                final_student = probabilities(folder/'GB0/epoch200_probabilities.pt')
                assert p.shape == student.shape == final_student.shape and len(p) == len(x)
                _, _, total_ss = moments(p, np.zeros(len(p), dtype=int))
                total_ss = float(total_ss.sum())
                teacher_class = p.argmax(1)
                for method, rs, a in groups:
                    key = dict(dataset=dataset, rate=rate, seed=seed, method=method, random_seed=rs)
                    sizes, means, ss = moments(p, a)
                    x_ss = x_stats[(method, rs)]
                    substitute = means[a]
                    agreement = (teacher_class == substitute.argmax(1))
                    class_agreement = [agreement[teacher_class == label].mean()*100
                                       for label in np.unique(teacher_class)]
                    all_metrics.append(dict(**key, nodes=len(p), classes=p.shape[1], groups=len(sizes),
                        **comparison(p, substitute),
                        macro_teacher_class_agreement_pct=float(np.mean(class_agreement)),
                        explained_teacher_variance=1-float(ss.sum()/total_ss) if total_ss > 0 else None,
                        attribute_within_global_ratio=float(x_ss.sum()/len(x)/x_global_var),
                        low_teacher_variance_attribute_ratio=low_variance_attribute_ratio(sizes,ss,x_ss,x_global_var),
                        teacher_student_disagree_pct=float(np.mean(teacher_class != student.argmax(1))*100),
                        fp16_teacher_agreement_pct=float(np.mean(p.astype(np.float16).argmax(1)==teacher_class)*100)))
                    for j in range(len(sizes)):
                        mask = a == j
                        all_balls.append(dict(**key, ball=j, nodes=int(sizes[j]),
                            teacher_variance=float(ss[j]/sizes[j]),
                            normalized_x_variance=float(x_ss[j]/sizes[j]),
                            student_disagree_pct=float(np.mean(teacher_class[mask] != student[mask].argmax(1))*100),
                            epoch200_disagree_pct=float(np.mean(teacher_class[mask] != final_student[mask].argmax(1))*100)))
                    all_curves += [dict(**key, **row) for row in compression_curve(p,a,EXCEPTIONS)]
                    variance = np.divide(ss, sizes-1, out=np.zeros(len(sizes)), where=sizes>1)
                    for fraction in BUDGETS:
                        budget = max(len(sizes), int(np.floor(len(p)*fraction)))
                        uniform = float((1-budget/len(p))*(total_ss/(len(p)-1))/budget)
                        for scheme, weights in [('size', sizes), ('adaptive', sizes*np.sqrt(variance))]:
                            m = allocate(sizes, budget, weights)
                            error = sampling_mse(sizes, ss, m)
                            all_sampling.append(dict(**key, fraction=fraction, budget=budget, scheme=scheme,
                                sampling_mean_mse=error, uniform_mean_mse=uniform,
                                ratio_to_uniform=error/uniform if uniform>0 else None))
                            all_allocations += [dict(**key, fraction=fraction, scheme=scheme, ball=j,
                                nodes=int(sizes[j]), allocated=int(m[j])) for j in range(len(sizes))]
                print(f'DONE {dataset} uniform{rate} seed{seed}', flush=True)
        del x
    for name, rows in [('group_metrics.csv',all_metrics),('compression_curves.csv',all_curves),
                       ('ball_metrics.csv',all_balls),('sampling_metrics.csv',all_sampling),
                       ('allocations.csv',all_allocations)]:
        export_csv(name, rows)
    verify()
    write(ROOT/'execution.json', dict(status='ok', protocol_cases=20, grouping_rows=len(all_metrics),
          seconds=time.perf_counter()-started, input_hash_check='passed', training_runs=0,
          true_label_metrics=0))
    print('COMPLETE: input hashes unchanged, no model trained, no true-label metrics', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze','run'])
    args = parser.parse_args()
    with (ROOT/f'{args.mode}.log').open('a', encoding='utf-8') as handle:
        class Tee:
            def __init__(self, original): self.original = original
            def write(self, value):
                self.original.write(value)
                handle.write(value)
                handle.flush()
            def flush(self): self.original.flush(); handle.flush()
        sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
        try:
            freeze() if args.mode == 'freeze' else run()
        except Exception:
            traceback.print_exc()
            raise
