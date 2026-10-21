"""Fail-closed, CPU-only independent audit of frozen experiment artifacts.

This script never imports the target generator or student evaluator. Graph loading
and the frozen original noise injector are reused only to recover the protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
import torch

import common as c

ROOT = Path(__file__).resolve().parent
ARRAY_TOL = 1.2e-7
METRIC_TOL = 1e-12
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON value {value}: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(2**20), b''):
                digest.update(block)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def manifest(path, required=None):
    records = read(path)
    require(isinstance(records, dict) and records, f'Empty/invalid manifest: {path}')
    if required is not None:
        wanted = {str(Path(p).resolve()) for p in required}
        require(set(records) == wanted, f'Manifest file set differs: {path}')
    for filename, digest in records.items():
        require(Path(filename).is_absolute(), f'Manifest path is not absolute: {filename}')
        require(isinstance(digest, str) and len(digest) == 64, f'Invalid SHA: {filename}')
        require(sha(filename) == digest, f'SHA mismatch: {filename}')
    return records


def compare_arrays(actual, expected, description, tolerance=ARRAY_TOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f'{description}: shape {actual.shape} != {expected.shape}')
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{description}: nonfinite array')
    difference = float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)), initial=0))
    require(difference <= tolerance, f'{description}: max absolute difference {difference} > {tolerance}')
    return difference


def compare_records(actual, expected, description, tolerance=METRIC_TOL):
    require(set(actual) == set(expected), f'{description}: fields differ: {set(actual) ^ set(expected)}')
    for key, wanted in expected.items():
        found = actual[key]
        if isinstance(wanted, dict):
            compare_records(found, wanted, f'{description}.{key}', tolerance)
        elif isinstance(wanted, list):
            require(isinstance(found, list) and len(found) == len(wanted), f'{description}.{key}: list differs')
            for i, (left, right) in enumerate(zip(found, wanted)):
                if isinstance(right, dict):
                    compare_records(left, right, f'{description}.{key}[{i}]', tolerance)
                else:
                    require(left == right, f'{description}.{key}[{i}] differs')
        elif isinstance(wanted, (float, int)) and not isinstance(wanted, bool):
            require(isinstance(found, (float, int)) and not isinstance(found, bool), f'{description}.{key}: numeric type differs')
            require(math.isfinite(float(found)) and math.isfinite(float(wanted)), f'{description}.{key}: nonfinite')
            require(abs(float(found)-float(wanted)) <= tolerance, f'{description}.{key}: {found} != {wanted}')
        else:
            require(found == wanted and (not isinstance(wanted, bool) or type(found) is bool), f'{description}.{key}: {found} != {wanted}')


def independent_targets(groups, observed, classes, ce=False):
    """Count each ball separately, remove the node's own observed class, then mix."""
    observed = np.asarray(observed, dtype=np.int64)
    require(observed.ndim == 1 and len(observed) > 0, 'Empty/invalid observed labels')
    require(classes >= 2 and np.all((observed >= 0) & (observed < classes)), 'Invalid observed class')
    n = len(observed)
    onehot = np.eye(classes, dtype=np.float64)[observed]
    if ce:
        return dict(target=onehot.astype(np.float32), r=onehot.astype(np.float32),
                    q=onehot.astype(np.float32), alpha=np.zeros(n, np.float32), support=np.zeros(n, np.int64))
    groups = np.asarray(groups)
    require(groups.shape == observed.shape, 'Group/label shape mismatch')
    r = np.empty((n, classes), dtype=np.float64)
    support = np.empty(n, dtype=np.int64)
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        counts = np.bincount(observed[ids], minlength=classes)
        votes = np.repeat(counts[None, :], len(ids), axis=0)
        votes[np.arange(len(ids)), observed[ids]] -= 1
        support[ids] = len(ids)-1
        r[ids] = votes/(len(ids)-1) if len(ids) > 1 else np.full((1, classes), 1/classes)
    maximum = r.max(axis=1)
    alpha = np.clip((classes*maximum-1)/(classes-1), 0, 1)
    alpha[support < 2] = 0
    q = r.copy()
    unique = np.count_nonzero(np.isclose(r, maximum[:, None], atol=1e-12, rtol=0), axis=1) == 1
    q[unique] *= .5
    q[np.flatnonzero(unique), np.argmax(r[unique], axis=1)] += .5
    target = (1-alpha[:, None])*onehot + alpha[:, None]*q
    return dict(target=target.astype(np.float32), r=r.astype(np.float32), q=q.astype(np.float32),
                alpha=alpha.astype(np.float32), support=support)


def independent_metrics(logits, labels, binary=False):
    """Use sklearn for predictions and preserve training's float32 probability ties."""
    values = torch.as_tensor(logits).detach().cpu()
    labels = np.asarray(labels)
    require(values.ndim == 2 and len(values) == len(labels) and torch.isfinite(values).all().item(), 'Invalid logits')
    predictions = values.argmax(1).numpy()
    # The original accuracy averages an indicator in float32. Quantize the
    # independently counted sklearn fraction to that same stored precision.
    result = dict(accuracy=float(np.float32(accuracy_score(labels, predictions))),
                  macro_f1=float(f1_score(labels, predictions, average='macro', zero_division=0)))
    if binary:
        require(values.shape[1] == 2 and set(np.unique(labels)) == {0, 1}, 'Invalid binary evaluation split')
        probability = torch.softmax(values, dim=1).numpy()[:, 1]
        result.update(roc_auc=float(roc_auc_score(labels, probability)),
                      average_precision=float(average_precision_score(labels, probability)))
    return result


def check_history(history, best_epoch, validation, epochs, *, primary_metric='accuracy'):
    require(primary_metric in {'accuracy', 'roc_auc'}, 'Unknown history primary metric')
    require(isinstance(history, list) and len(history) == epochs, 'History length differs from fixed training epochs')
    scores = []
    for epoch, record in enumerate(history, start=1):
        require(set(record) == {'epoch', 'validation_score', 'loss'} and record['epoch'] == epoch, 'Invalid history epoch/fields')
        score, loss = float(record['validation_score']), float(record['loss'])
        require(math.isfinite(score) and 0 <= score <= 1 and math.isfinite(loss), 'Nonfinite/invalid training history')
        scores.append(score)
    first_max = int(np.argmax(scores))+1
    require(type(best_epoch) is int and best_epoch == first_max, f'Best epoch is not first maximum: {best_epoch} != {first_max}')
    # History accuracy is a GPU float32 reduction, whereas the checkpoint is
    # evaluated on CPU. These can differ by one ULP even for identical class
    # predictions. This tolerance does not affect the exact first argmax above.
    tolerance = ARRAY_TOL if primary_metric == 'accuracy' else METRIC_TOL
    require(abs(float(validation)-scores[first_max-1]) <= tolerance, 'Saved best logits validation differs from history maximum')


def canonical_adjacency(graph):
    edges = graph.edge_index.numpy()
    edges = edges[:, edges[0] != edges[1]]
    rows = np.concatenate([edges[0], edges[1]])
    cols = np.concatenate([edges[1], edges[0]])
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(graph.num_nodes, graph.num_nodes)).tocsr()
    matrix.sum_duplicates()
    matrix.data.fill(1)
    matrix.sort_indices()
    return matrix


def normalized(array):
    values = np.asarray(array, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm != 0).astype(np.float32)


def check_views(name, graph, adjacency):
    folder = ROOT/'views'/name
    manifest(folder/'manifest.json', [folder/(x+'.npy') for x in ['X', 'R', 'SHUFFLE', 'permutation']]+[c.DATA[name]])
    expected_input = dict(dataset=name, data_sha256=sha(c.DATA[name]), features_sha256=fingerprint(graph.x.numpy()),
                          edges_sha256=fingerprint(graph.edge_index.numpy()), label_usage='none')
    compare_records(read(folder/'input.json'), expected_input, f'{name} view inputs')
    n, width = graph.x.shape
    permutation = np.load(folder/'permutation.npy', allow_pickle=False)
    expected_perm = np.random.default_rng(c.CONFIG['residual']['shuffle_seed']).permutation(n).astype(np.int64)
    require(permutation.dtype == np.int64 and np.array_equal(permutation, expected_perm), f'{name}: shuffle permutation differs')
    sample = np.sort(np.random.default_rng(20261015).choice(n, size=min(64, n), replace=False))
    sampled_residual = {}
    features = graph.x.numpy()
    for node in np.unique(np.concatenate([sample, permutation[sample]])):
        adjacent = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]
        neighbor = features[adjacent].astype(np.float64).sum(axis=0)/len(adjacent) if len(adjacent) else np.zeros(width)
        sampled_residual[int(node)] = normalized((neighbor-features[node])[None, :])[0]
    u = normalized(features[sample])
    v = np.stack([sampled_residual[int(i)] for i in sample])
    vp = np.stack([sampled_residual[int(i)] for i in permutation[sample]])
    factor = np.float32(1/np.sqrt(2))
    expected = dict(X=u, R=np.concatenate([u*factor, v*factor], axis=1),
                    SHUFFLE=np.concatenate([u*factor, vp*factor], axis=1))
    for key in expected:
        saved = np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False)
        shape = (n, width) if key == 'X' else (n, 2*width)
        require(saved.shape == shape and saved.dtype == np.float32, f'{name}/{key}: shape/dtype differs')
        for start in range(0, n, 1024):
            require(np.isfinite(saved[start:start+1024]).all(), f'{name}/{key}: nonfinite view')
        compare_arrays(saved[sample], expected[key], f'{name}/{key} independently reconstructed 64 rows')
    return len(sample)


def check_partitions(name, seed, graph, adjacency):
    folder = ROOT/'partitions'/name/f'seed{seed}'
    manifest(folder/'manifest.json', [folder/'assignments.npz', folder/'split.npz', folder/'stats.json',
                                    c.DATA[name], c.H[name], ROOT/'views'/name/'manifest.json'])
    tr = graph.train_index.numpy()
    with np.load(folder/'split.npz', allow_pickle=False) as split:
        require(set(split.files) == {'train', 'val', 'test'}, 'Partition split fields differ')
        for key, index in [('train', tr), ('val', graph.val_index.numpy()), ('test', graph.test_index.numpy())]:
            require(np.array_equal(split[key], index), f'{name}/seed{seed}: saved {key} split changed')
    with np.load(folder/'assignments.npz', allow_pickle=False) as archive:
        assignments = {key: archive[key].copy() for key in archive.files}
    require(set(assignments) == set(c.ARMS)-{'CE'}, 'Partition arm keys differ')
    stats = read(folder/'stats.json')
    require(set(stats) == set(assignments), 'Partition statistics arm keys differ')
    n = graph.num_nodes
    for arm, group in assignments.items():
        require(group.shape == (n,) and group.dtype.kind in 'iu', f'{name}/{arm}: assignment shape/dtype')
        ids = np.unique(group)
        require(np.array_equal(ids, np.arange(len(ids))), f'{name}/{arm}: ball IDs not contiguous/complete')
        sizes = np.bincount(group)
        counts = np.bincount(group[tr], minlength=len(sizes))
        expected = dict(actual_balls=len(sizes), singletons=int((sizes == 1).sum()),
                        train_count_quantiles=np.quantile(counts, [0, .25, .5, .75, 1]).tolist(),
                        size_quantiles=np.quantile(sizes, [0, .25, .5, .75, 1]).tolist(),
                        eligible_rate=float(np.mean(counts[group[tr]] >= 3)))
        compare_records({key: stats[arm][key] for key in expected}, expected, f'{name}/{arm} partition statistics')
        require(stats[arm]['num_balls'] == len(sizes), f'{name}/{arm}: declared ball count differs')
        if arm.endswith('_TREE'):
            require(len(ids) <= math.ceil(math.sqrt(n)), f'{name}/{arm}: exceeds tree budget')
            require(len(ids) == 1 or (sizes.min() >= 16 and counts.min() >= 16), f'{name}/{arm}: leaf support constraint violated')
            require(len(ids) != 1 or len(tr) < 32 or n < 32 or stats[arm]['stop_reason'] != 'target_reached', f'{name}/{arm}: suspicious singleton root')
        elif arm == 'OLD64':
            require(len(ids) <= 64, f'{name}: OLD64 budget exceeded')
        elif arm == 'BFS':
            require(int(sizes.max()) <= math.isqrt(n), f'{name}: BFS capacity violated')
            # Traverse only within each saved ball. Every ball must be connected;
            # global graph connectivity is irrelevant to this independent test.
            visited = np.zeros(n, dtype=bool)
            components = np.zeros(len(sizes), dtype=np.int64)
            for start in range(n):
                if visited[start]:
                    continue
                ball = int(group[start]); components[ball] += 1
                stack = [start]; visited[start] = True
                while stack:
                    node = stack.pop()
                    neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node+1]]
                    for neighbor in neighbors:
                        if not visited[neighbor] and group[neighbor] == ball:
                            visited[neighbor] = True; stack.append(int(neighbor))
            require(np.all(components == 1), f'{name}: disconnected BFS ball')
    return assignments


def diagnostic_metrics(values, clean, observed, groups=None):
    t, q, alpha, support = (values[k] for k in ['target', 'q', 'alpha', 'support'])
    truth = np.eye(t.shape[1])[clean]
    brier = np.square(t-truth).sum(axis=1)
    initially_clean = observed == clean
    prediction, qprediction = t.argmax(1), q.argmax(1)
    def mean_on(value, mask):
        return float(np.mean(value[mask])) if mask.any() else None
    result = dict(target_brier=float(brier.mean()), q_brier=float(np.square(q-truth).sum(1).mean()),
                  q_accuracy=float(np.mean(qprediction == clean)), target_accuracy=float(np.mean(prediction == clean)),
                  mean_alpha=float(alpha.mean()), q_true_probability=float(q[np.arange(len(clean)), clean].mean()),
                  clean_brier=mean_on(brier, initially_clean), noisy_brier=mean_on(brier, ~initially_clean),
                  clean_harm_rate=mean_on(prediction != clean, initially_clean), noise_recovery_rate=mean_on(prediction == clean, ~initially_clean),
                  correction_coverage=float(np.mean(prediction != observed)), observed_error=float(np.mean(~initially_clean)))
    valid = support >= 2
    margin_fraction = margin_mean = None
    if groups is not None:
        # Keep full precision for clean-margin diagnostics: the generator uses
        # unrounded leave-one-out probabilities before constructing artifacts.
        clean_r = np.empty_like(truth)
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            votes = np.repeat(np.bincount(clean[ids], minlength=t.shape[1])[None, :], len(ids), axis=0)
            votes[np.arange(len(ids)), clean[ids]] -= 1
            clean_r[ids] = votes/(len(ids)-1) if len(ids) > 1 else 1/t.shape[1]
        competitors = clean_r.copy()
        competitors[np.arange(len(clean)), clean] = -np.inf
        margins = clean_r[np.arange(len(clean)), clean]-competitors.max(1)
        margin_fraction = float(np.mean(valid & (margins > 0)))
        margin_mean = mean_on(margins, valid)
    result.update(eligible_rate=float(valid.mean()), eligible_q_accuracy=mean_on(qprediction == clean, valid),
                  positive_clean_margin_fraction=margin_fraction, mean_clean_margin=margin_mean,
                  mean_support=float(support.mean()), high_alpha_wrong_q_fraction=float(np.mean((alpha >= .75) & (qprediction != clean))))
    return result


def gate_from_metrics(rows, datasets):
    lookup = {(r['dataset'], int(r['seed']), r['noise'], r['arm']): r for r in rows}
    result = {}
    for name in sorted(datasets):
        checks = []
        for noise in sorted(set(c.conditions(name))-{'clean'}):
            for ref in c.CONFIG['gate']['references']:
                delta = np.array([lookup[name, seed, noise, 'R_TREE']['target_brier']-lookup[name, seed, noise, ref]['target_brier'] for seed in c.SEEDS])
                improved = int(np.count_nonzero(delta < 0))
                passed = bool(delta.mean() <= c.CONFIG['gate']['mean_brier_delta_max'] and improved >= c.CONFIG['gate']['minimum_improved_seeds'])
                checks.append(dict(noise=noise, reference=ref, mean_delta=float(delta.mean()), std=float(delta.std(ddof=1)), improved_seeds=improved, passed=passed))
        result[name] = dict(passed=all(check['passed'] for check in checks), checks=checks)
    return result


def key_of(row):
    return row['dataset'], int(row['seed']), row['noise'], row['arm']


def validate_target_keys(rows, datasets):
    expected = {(name, seed, noise, arm) for name in datasets for seed in c.SEEDS
                for noise in c.conditions(name) for arm in c.ARMS}
    actual = [key_of(row) for row in rows]
    require(len(actual) == len(expected) and len(set(actual)) == len(actual) and set(actual) == expected,
            f'Target key matrix incomplete/duplicated/unexpected: found {len(actual)}, expected {len(expected)}')
    if set(datasets) == set(c.CONFIG['datasets']):
        require(len(actual) == 420, 'Complete audit requires exactly 420 targets')
    return len(expected)


def source_files():
    modules = ['granular_ball.py', 'granular_ball_adaptive.py', 'granular_ball_trajectory.py',
               'granular_ball_topology.py', 'targets.py', 'data.py', 'representation.py']
    return [ROOT/name for name in ['PLAN.md', 'config.json', 'common.py', 'features.py', 'group_targets.py', 'run.py']]+[
        c.WORKSPACE/'GB-LCSS_code/src/gblcss'/name for name in modules]+[Path(c.d.__file__)]


def preparation_files():
    return [ROOT/name for name in ['prepare.py', 'common.py', 'PLAN.md', 'config.json', 'dataset_manifest.json']]+[
        c.MECHANISM/'prepare_representations.py', c.WORKSPACE/'GB-LCSS_code/src/gblcss/grace.py',
        c.WORKSPACE/'GB-LCSS_code/src/gblcss/data.py']+[c.DATA[name] for name in c.CONFIG['datasets']]


def verify_csv(path, rows, datasets):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        saved = [r for r in reader if set(datasets) == set(c.CONFIG['datasets']) or r['dataset'] in datasets]
        fields = set(reader.fieldnames or [])
    expected = {key_of(row): row for row in rows}
    actual = {key_of(row): row for row in saved}
    require(len(actual) == len(saved) and actual.keys() == expected.keys(), f'{path.name}: stale/duplicate/missing keys')
    require(fields == set().union(*(set(r) for r in rows)), f'{path.name}: CSV columns differ')
    for key, row in expected.items():
        converted = {}
        for field, wanted in row.items():
            text = actual[key][field]
            if wanted is None:
                require(text == '', f'{path.name}/{key}/{field}: expected empty value')
                converted[field] = None
            elif isinstance(wanted, bool):
                require(text in {'True', 'False'}, f'{path.name}: malformed bool')
                converted[field] = text == 'True'
            elif isinstance(wanted, (int, float)):
                converted[field] = float(text)
            else:
                converted[field] = text
        compare_records(converted, row, f'{path.name}/{key}')


def verify_summary_csv(path, rows, datasets):
    """Rebuild each mean/std/count from the verified raw records."""
    frame = pd.DataFrame(rows)
    numeric = [key for key in frame.select_dtypes('number').columns if key != 'seed']
    expected = frame.groupby(['dataset', 'noise', 'arm'])[numeric].agg(['mean', 'std', 'count'])
    expected.columns = ['_'.join(parts) for parts in expected.columns]
    expected = expected.reset_index()
    saved = pd.read_csv(path)
    if set(datasets) != set(c.CONFIG['datasets']):
        saved = saved[saved.dataset.isin(datasets)]
    keys = ['dataset', 'noise', 'arm']
    require(set(saved.columns) == set(expected.columns), f'{path.name}: summary columns differ')
    require(not saved.duplicated(keys).any(), f'{path.name}: duplicate summary keys')
    left = saved.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    require(left.index.equals(right.index), f'{path.name}: summary keys differ')
    for column in right.columns:
        actual, wanted = left[column].to_numpy(), right[column].to_numpy()
        missing = pd.isna(wanted)
        require(np.array_equal(pd.isna(actual), missing), f'{path.name}/{column}: missing summary values differ')
        compare_arrays(actual[~missing], wanted[~missing], f'{path.name}/{column}', METRIC_TOL)


def independent_pairs(rows):
    """Rebuild every planned comparison directly from the verified run records."""
    lookup = {key_of(row): row for row in rows}
    require(len(lookup) == len(rows), 'Duplicate student records while rebuilding pairs')
    pairs = []
    for (name, seed, noise, arm), row in sorted(lookup.items()):
        if arm not in {'R_TREE', 'X_TREE', 'BFS'}:
            continue
        for reference in ['CE', 'OLD64', 'H_TREE', 'X_TREE', 'SHUFFLE_TREE']:
            other = lookup.get((name, seed, noise, reference))
            if other is None or reference == arm:
                continue
            require(row['primary_metric'] == other['primary_metric'], 'Paired primary metrics differ')
            delta = 100*(row['test_primary']-other['test_primary'])
            require(math.isfinite(delta), 'Nonfinite paired student delta')
            pairs.append(dict(dataset=name, seed=seed, noise=noise, arm=arm, reference=reference,
                              primary_metric=row['primary_metric'], delta_pp=delta))
    return pairs


def verify_paired_csvs(folder, rows, datasets):
    pairs = independent_pairs(rows)
    require(pairs, 'No planned student comparisons could be rebuilt')
    folder = Path(folder)
    path = folder/'paired_student_deltas.csv'
    fields = {'dataset', 'seed', 'noise', 'arm', 'reference', 'primary_metric', 'delta_pp'}
    key_fields = ['dataset', 'seed', 'noise', 'arm', 'reference', 'primary_metric']
    def pair_key(row):
        return tuple(int(row[key]) if key == 'seed' else row[key] for key in key_fields)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        saved = [row for row in reader if set(datasets) == set(c.CONFIG['datasets']) or row['dataset'] in datasets]
        require(set(reader.fieldnames or []) == fields, 'Paired delta CSV fields differ')
    actual = {pair_key(row): row for row in saved}
    expected = {pair_key(row): row for row in pairs}
    require(len(actual) == len(saved) and actual.keys() == expected.keys(), 'Paired delta CSV has stale/duplicate/missing keys')
    for key, wanted in expected.items():
        found = dict(actual[key], seed=int(actual[key]['seed']), delta_pp=float(actual[key]['delta_pp']))
        compare_records(found, wanted, f'paired_student_deltas.csv/{key}')

    path = folder/'paired_student_summary.csv'
    group_fields = ['dataset', 'noise', 'arm', 'reference', 'primary_metric']
    expected_summary = pd.DataFrame(pairs).groupby(group_fields).delta_pp.agg(['mean', 'std', 'count'])
    saved_summary = pd.read_csv(path)
    if set(datasets) != set(c.CONFIG['datasets']):
        saved_summary = saved_summary[saved_summary.dataset.isin(datasets)]
    require(set(saved_summary.columns) == set(group_fields+['mean', 'std', 'count']), 'Paired summary CSV fields differ')
    require(not saved_summary.duplicated(group_fields).any(), 'Duplicate paired summary keys')
    actual_summary = saved_summary.set_index(group_fields).sort_index()
    expected_summary = expected_summary.sort_index()
    require(actual_summary.index.equals(expected_summary.index), 'Paired summary CSV has stale/missing keys')
    for column in ['mean', 'std', 'count']:
        found = actual_summary[column].to_numpy()
        wanted = expected_summary[column].to_numpy()
        missing = pd.isna(wanted)
        require(np.array_equal(pd.isna(found), missing), f'Paired summary {column} missing values differ')
        compare_arrays(found[~missing], wanted[~missing], f'paired_student_summary.csv/{column}', METRIC_TOL)
    return len(pairs)


def check_dataset(name, record):
    require(Path(record['path']).resolve() == Path(c.DATA[name]).resolve(), f'{name}: dataset path differs')
    require(sha(c.DATA[name]) == record['sha256'], f'{name}: dataset SHA changed')
    wanted_url = f"https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/{c.CONFIG['official_commit']}/data/{name}.npz" if name in c.CONFIG['new_datasets'] else None
    require(record['source_url'] == wanted_url, f'{name}: source URL/commit differs')
    graph = c.load_graph(name, 0)
    require(record['nodes'] == graph.num_nodes and record['features'] == graph.x.shape[1] and record['classes'] == graph.num_classes, f'{name}: dataset dimensions differ')
    require(record['class_counts'] == np.bincount(graph.y.numpy()).tolist(), f'{name}: dataset class counts differ')
    adjacency = canonical_adjacency(graph)
    require(record['edges_undirected'] == adjacency.nnz//2, f'{name}: edge count differs')
    rows, cols = adjacency.nonzero()
    keep = rows < cols; y = graph.y.numpy()
    degree = np.diff(adjacency.indptr)
    chance = float(np.square(np.bincount(y, weights=degree)/degree.sum()).sum())
    homophily = float(np.mean(y[rows[keep]] == y[cols[keep]]))
    compare_records({k: record[k] for k in ['edge_homophily', 'adjusted_homophily']},
                    dict(edge_homophily=homophily, adjusted_homophily=(homophily-chance)/(1-chance)), f'{name} dataset homophily')
    require([r['seed'] for r in record['splits']] == c.SEEDS, f'{name}: manifest split seed set differs')
    return graph, adjacency


def check_representation(name, graph, partial):
    raw = torch.load(c.H[name], map_location='cpu', weights_only=False)
    if isinstance(raw, dict):
        raw = raw.get('H', raw.get('h', raw.get('embedding', raw.get('embeddings'))))
    require(isinstance(raw, torch.Tensor) and raw.ndim == 2 and raw.shape[0] == graph.num_nodes and raw.shape[1] > 0, f'{name}: invalid representation shape')
    require(torch.isfinite(raw).all().item(), f'{name}: nonfinite representation')
    if name in c.CONFIG['new_datasets']:
        folder = Path(c.H[name]).parent
        config, summary = read(folder/'config.json'), read(folder/'summary.json')
        require(config['action'] == 'train-grace' and config['epochs'] == c.CONFIG['grace']['epochs'] and config['seed'] == c.CONFIG['grace']['seed'] and config['chunk_size'] == c.CONFIG['grace']['chunk_size'], f'{name}: GRACE protocol differs')
        require(config['data_sha256'] == sha(c.DATA[name]) and config['script_sha256'] == sha(c.MECHANISM/'prepare_representations.py'), f'{name}: GRACE input/source SHA differs')
        require(config['label_usage'] == 'none; fixed last epoch, no evaluation', f'{name}: GRACE label usage differs')
        fixed = dict(hidden_dim=256, projection_dim=256, layers=2, learning_rate=.001,
                     weight_decay=1e-5, edge_drop_rates=[.1, .4], feature_mask_rates=[.1, 0],
                     temperature=.7, activation='relu',
                     feature_preprocessing='per-row min shift and sum normalization, original GB-LCSS GRACE')
        compare_records({key: config[key] for key in fixed}, fixed, f'{name} fixed GRACE hyperparameters')
        require(summary['status'] == 'completed' and summary['epochs'] == config['epochs'] and summary['shape'] == list(raw.shape) and summary['representation_sha256'] == sha(c.H[name]), f'{name}: invalid GRACE completion summary')
        history = [json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))) for line in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(history) == config['epochs'], f'{name}: incomplete SSL history')
        require(all(r['epoch'] == i+1 and math.isfinite(r['loss']) and math.isfinite(r['seconds']) for i, r in enumerate(history)), f'{name}: invalid SSL history')
    return dict(path=str(Path(c.H[name]).resolve()), sha256=sha(c.H[name]), shape=list(raw.shape), newly_trained=name in c.CONFIG['new_datasets'])


def audit_targets(datasets, partial):
    manifest(ROOT/'source_manifest.json', source_files())
    manifest(ROOT/'preparation_manifest.json', preparation_files())
    data_manifest = read(ROOT/'dataset_manifest.json')
    require(data_manifest['commit'] == c.CONFIG['official_commit'] and set(data_manifest['datasets']) == set(c.CONFIG['datasets']), 'Dataset manifest scope/commit differs')
    if not partial:
        require(read(ROOT/'prepare_status.json')['status'] == 'completed', 'Preparation is not completed')
        require(set(read(ROOT/'representation_manifest.json')) == set(c.CONFIG['datasets']), 'Representation manifest dataset set differs')
    expected = {(name, seed, noise, arm) for name in datasets for seed in c.SEEDS for noise in c.conditions(name) for arm in c.ARMS}
    rows, representations, sampled = [], {}, {}
    largest_error = 0.
    for name in datasets:
        graph, adjacency = check_dataset(name, data_manifest['datasets'][name])
        representations[name] = check_representation(name, graph, partial)
        if not partial:
            saved = read(ROOT/'representation_manifest.json')[name]
            compare_records(saved, {k: representations[name][k] for k in ['path', 'sha256', 'newly_trained']}, f'{name} representation manifest')
        sampled[name] = check_views(name, graph, adjacency)
        print(f'AUDIT {name}: views/representation verified', flush=True)
        for seed in c.SEEDS:
            graph = c.load_graph(name, seed)
            tr, va, te = (getattr(graph, key+'_index').numpy() for key in ['train', 'val', 'test'])
            split_record = data_manifest['datasets'][name]['splits'][seed]
            compare_records(split_record, dict(seed=seed, hash=fingerprint(tr, va, te), sizes=[len(tr), len(va), len(te)]), f'{name}/seed{seed} dataset split')
            joined = np.concatenate([tr, va, te])
            require(len(joined) == len(np.unique(joined)) and np.all((joined >= 0) & (joined < graph.num_nodes)), f'{name}/seed{seed}: split overlap/range')
            assignments = check_partitions(name, seed, graph, adjacency)
            parent = ROOT/'targets'/name/f'seed{seed}'
            saved_rows = read(parent/'metrics.json')
            require(len(saved_rows) == 3*len(c.ARMS) and len({key_of(row) for row in saved_rows}) == len(saved_rows), f'{name}/seed{seed}: incomplete/duplicate metrics')
            lookup = {key_of(row): row for row in saved_rows}
            for noise, (kind, rate) in c.conditions(name).items():
                noisy_graph, _ = c.load_case(name, seed, noise)
                observed = noisy_graph.noisy_y[tr].numpy()
                protocol = dict(dataset=name, seed=seed, noise=noise, kind=kind, rate=rate,
                                split_hash=fingerprint(tr, va, te), noise_hash=fingerprint(observed),
                                data_sha256=sha(c.DATA[name]), val_clean=True, test_clean=True)
                require(torch.equal(noisy_graph.y[va], noisy_graph.noisy_y[va]) and torch.equal(noisy_graph.y[te], noisy_graph.noisy_y[te]), f'{name}: evaluation labels were corrupted')
                folder = parent/noise
                manifest(folder/'frozen_targets.json', [folder/(arm+'.npz') for arm in c.ARMS]+[folder/'observed.npy', folder/'protocol.json'])
                compare_records(read(folder/'protocol.json'), protocol, f'{name}/seed{seed}/{noise} protocol')
                require(np.array_equal(np.load(folder/'observed.npy', allow_pickle=False), observed), f'{name}/seed{seed}/{noise}: observed noise labels differ')
                for arm in c.ARMS:
                    group = None if arm == 'CE' else assignments[arm][tr]
                    rebuilt = independent_targets(group, observed, graph.num_classes, ce=arm == 'CE')
                    with np.load(folder/(arm+'.npz'), allow_pickle=False) as archive:
                        values = {key: archive[key].copy() for key in archive.files}
                    require(set(values) == set(rebuilt), f'{name}/{arm}: target archive fields differ')
                    for field in values:
                        tol = 0 if field == 'support' else ARRAY_TOL
                        largest_error = max(largest_error, compare_arrays(values[field], rebuilt[field], f'{name}/seed{seed}/{noise}/{arm}/{field}', tol))
                        require(values[field].dtype == (np.int64 if field == 'support' else np.float32), f'{name}/{arm}/{field}: dtype differs')
                    row = dict(dataset=name, seed=seed, noise=noise, arm=arm, **diagnostic_metrics(values, graph.y[tr].numpy(), observed, group))
                    compare_records(lookup[(name, seed, noise, arm)], row, f'{name}/seed{seed}/{noise}/{arm} diagnostics')
                    rows.append(row)
            print(f'AUDIT {name} seed{seed}: partitions and 21 targets verified', flush=True)
    validate_target_keys(rows, datasets)
    actual_paths = {(p.parts[-4], int(p.parts[-3][4:]), p.parts[-2], p.stem) for p in (ROOT/'targets').glob('*/seed*/*/*.npz') if not partial or p.parts[-4] in datasets}
    require(actual_paths == expected, 'Unexpected/missing target artifacts')
    verify_csv(ROOT/'target_metrics.csv', rows, datasets)
    verify_summary_csv(ROOT/'target_summary.csv', rows, datasets)
    gate = gate_from_metrics(rows, datasets)
    if not partial:
        saved_gate = read(ROOT/'gate.json')
        compare_records(saved_gate, dict(datasets=gate, selected_datasets=[n for n, v in gate.items() if v['passed']], target_rows=len(rows), source='all frozen metrics.json records, not cached CSV'), 'Frozen target gate')
        require(len(rows) == 420, 'Complete audit requires exactly 420 targets')
    manifest(ROOT/'source_manifest.json', source_files())
    manifest(ROOT/'preparation_manifest.json', preparation_files())
    return dict(target_rows=len(rows), expected_target_rows=len(expected), maximum_target_absolute_error=largest_error,
                view_rows_independently_rebuilt=sampled, representations=representations, gate=gate), rows


def audit_students(target_rows, gate, datasets):
    student_files = [ROOT/name for name in ['run_students.py', 'metric_training.py', 'common.py', 'PLAN.md', 'config.json', 'gate.json', 'source_manifest.json']]+[
        c.WORKSPACE/'GB-LCSS_code/src/gblcss/models.py', c.WORKSPACE/'GB-LCSS_code/src/gblcss/training.py']
    manifest(ROOT/'student_source_manifest.json', student_files)
    selected = {name for name, value in gate.items() if value['passed']}
    expected = {(name, seed, noise, arm) for name in datasets for seed in c.SEEDS for noise in c.conditions(name)
                for arm in c.CONFIG['baseline_arms']+(c.CONFIG['candidate_arms'] if name in selected else [])}
    actual_paths = {(p.parts[-5], int(p.parts[-4][4:]), p.parts[-3], p.parts[-2]): p for p in (ROOT/'students').glob('*/seed*/*/*/result.json') if set(datasets) == set(c.CONFIG['datasets']) or p.parts[-5] in datasets}
    require(set(actual_paths) == expected, f'Student key matrix incomplete/unexpected: found {len(actual_paths)}, expected {len(expected)}')
    rows = []
    for name in datasets:
        for seed in c.SEEDS:
            for noise in c.conditions(name):
                graph, _ = c.load_case(name, seed, noise)
                for arm in c.ARMS:
                    key = (name, seed, noise, arm)
                    if key not in expected:
                        continue
                    folder = actual_paths[key].parent
                    row, config = read(folder/'result.json'), read(folder/'config.json')
                    require(key_of(row) == key, f'{folder}: result key differs')
                    primary = c.CONFIG['classification']['primary'].get(name, 'accuracy')
                    target_path = ROOT/'targets'/name/f'seed{seed}'/noise/(arm+'.npz')
                    target_hash = sha(target_path)
                    require(config['device'].startswith(('cuda', 'cpu')), f'{folder}: invalid device')
                    wanted_config = dict(dataset=name, seed=seed, noise=noise, arm=arm, target_hash=target_hash,
                                         protocol=read(target_path.parent/'protocol.json'), primary_metric=primary,
                                         training=c.CONFIG['training'], device=config['device'], model_input='original X only', checkpoint='clean validation '+primary)
                    compare_records(config, wanted_config, f'{folder} config')
                    require(row['target_hash'] == target_hash and row['logits_hash'] == sha(folder/'logits.pt'), f'{folder}: student target/logits SHA changed')
                    require(type(row['newly_trained']) is bool and row['newly_trained'], f'{folder}: student was not newly trained')
                    require(row['primary_metric'] == primary, f'{folder}: student primary metric differs')
                    require(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] > 0, f'{folder}: invalid training duration')
                    logits = torch.load(folder/'logits.pt', map_location='cpu', weights_only=True)
                    require(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32 and tuple(logits.shape) == (graph.num_nodes, graph.num_classes) and torch.isfinite(logits).all().item(), f'{folder}: invalid student logits')
                    recomputed = {}
                    for split, labels in [('train', graph.noisy_y), ('val', graph.noisy_y), ('test', graph.y)]:
                        mask = getattr(graph, split+'_mask')
                        metrics = independent_metrics(logits[mask], labels[mask].numpy(), binary=graph.num_classes == 2)
                        recomputed.update({split+'_'+k: v for k, v in metrics.items()})
                    recomputed.update(test_primary=recomputed['test_'+primary], validation_primary=recomputed['val_'+primary])
                    compare_records({k: row[k] for k in recomputed}, recomputed, f'{folder} independent sklearn metrics')
                    check_history(read(folder/'history.json'), row['best_epoch'], recomputed['validation_primary'], c.CONFIG['training']['epochs'], primary_metric=primary)
                    rows.append(row)
            print(f'AUDIT {name} seed{seed}: saved student logits/history verified', flush=True)
    verify_csv(ROOT/'student_results.csv', rows, datasets)
    verify_summary_csv(ROOT/'student_summary.csv', rows, datasets)
    pair_count = verify_paired_csvs(ROOT, rows, datasets)
    require(len(rows) == len(expected), 'Student record count differs')
    manifest(ROOT/'student_source_manifest.json', student_files)
    manifest(ROOT/'source_manifest.json', source_files())
    manifest(ROOT/'preparation_manifest.json', preparation_files())
    return dict(student_runs=len(rows), expected_student_runs=len(expected), paired_student_comparisons=pair_count,
                selected_datasets=sorted(selected), metrics_recomputed='sklearn from saved float32 logits; first maximum validation epoch checked')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=c.CONFIG['datasets'])
    parser.add_argument('--allow-partial', action='store_true', help='Explicit development audit; never writes the full completion report')
    args = parser.parse_args()
    datasets = args.datasets or c.CONFIG['datasets']
    require(len(datasets) == len(set(datasets)), 'Duplicate requested datasets')
    partial = args.allow_partial or set(datasets) != set(c.CONFIG['datasets'])
    filename = ('partial_' if partial else '')+('targets.json' if args.targets_only else 'report.json')
    output = ROOT/'audit'/filename
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', partial=partial, datasets=datasets, audit_source_sha256=sha(__file__),
                  sources={str(Path(__file__).resolve()): sha(__file__)}, target_tolerance=ARRAY_TOL, metric_tolerance=METRIC_TOL,
                  history_accuracy_tolerance=ARRAY_TOL,
                  history_accuracy_tolerance_reason='GPU float32 history mean versus CPU checkpoint mean can differ by one ULP; epoch selection remains exact; ROC-AUC and saved CPU metrics retain 1e-12 tolerance')
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    save()
    try:
        torch.set_num_threads(c.CONFIG['threads'])
        target_report, rows = audit_targets(datasets, partial)
        report.update(target_report)
        if not args.targets_only:
            report.update(audit_students(rows, target_report['gate'], datasets))
        for filename in ['source_manifest.json', 'preparation_manifest.json']+([] if args.targets_only else ['student_source_manifest.json']):
            report['sources'].update(read(ROOT/filename))
        report.update(status='passed', hashed_files=len({key[0] for key in HASH_CACHE}))
        save()
        print(json.dumps(dict(status='passed', partial=partial, report=str(output), target_rows=len(rows), student_runs=report.get('student_runs')), ensure_ascii=False), flush=True)
    except Exception as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
