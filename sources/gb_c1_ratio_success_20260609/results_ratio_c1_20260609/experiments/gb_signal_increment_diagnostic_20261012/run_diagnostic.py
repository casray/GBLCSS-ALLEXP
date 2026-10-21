from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import sklearn
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
SOURCE = ROOT.parent / 'gb_dist_protocol_seeds_20261007'
DATASETS = ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho']
RATES = [.4, .6]
SEEDS = [1, 2]
DATA_FILES = {
    'cora': PROJECT / 'external/NoisyGL/data/Cora/processed/data.pt',
    'citeseer': PROJECT / 'external/NoisyGL/data/CiteSeer/processed/data.pt',
    'corafull': PROJECT / 'gb_graph_aware_minival_6_1/data/CoraFull/cora/processed/data_undirected.pt',
    'coauthorcs': PROJECT / 'external/NoisyGL/data/CS/processed/data.pt',
    'amazonpho': PROJECT / 'external/NoisyGL/data/Photo/processed/data.pt',
}
KM_CONFIG = dict(init='k-means++', n_init=10, random_state=0, max_iter=300, tol=1e-4)
BASE_NAMES = ['teacher_observed', 'teacher_max', 'teacher_entropy', 'teacher_margin',
              'teacher_observed_deficit', 'peer_prototype_observed_agreement',
              'peer_teacher_observed', 'peer_teacher_max', 'prototype_observed_cosine',
              'prototype_observed_deficit', 'prototype_observed_agreement']
KNN_NAMES = ['observed_share', 'max_share', 'entropy', 'observed_deficit',
             'mean_cosine', 'std_cosine']
REGION_NAMES = ['log_size', 'log_other_train_support', 'observed_share', 'max_share',
                'entropy', 'observed_deficit', 'relative_radius_distance', 'radial_percentile',
                'relative_other_center_margin', 'other_teacher_observed', 'no_other_train']
ARMS = {
    'Base': ['base'], 'Base+kNN': ['base', 'knn'],
    **{f'Base+{g}': ['base', g] for g in ('GB', 'KM', 'BKM', 'Random')},
    **{f'Base+kNN+{g}': ['base', 'knn', g] for g in ('GB', 'KM', 'BKM', 'Random')},
}
SETTINGS = dict(datasets=DATASETS, rates=RATES, seeds=SEEDS, knn_k=20, peer_k=20,
                peer_non_two_hop_factor=.5, coverage=.2, logistic_C=1.,
                grouping=KM_CONFIG, random_seed=100, threads=4, gpu_chunk=256,
                cv='leave-one-dataset-out', auc_gate=.02, precision_gate=.02,
                positive_dataset_gate=3, seconds_budget=3600,
                primary_target='training_annotation_corruption', stage='exploratory')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def assignment_hash(a):
    a = np.ascontiguousarray(a, dtype='<i8')
    header = f'dtype={a.dtype.str};shape={a.shape[0]};order=C\n'.encode('ascii')
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def export_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-12))).sum(1) / np.log(p.shape[1])


def check_probability(p):
    if p.ndim != 2 or not np.isfinite(p).all() or p.min() < 0:
        raise ValueError('invalid probabilities')
    if np.max(np.abs(p.sum(1) - 1)) > 1e-5:
        raise ValueError('probabilities do not sum to one')


def vote_statistics(votes, observed):
    rows = np.arange(len(observed))
    own = votes[rows, observed]
    maximum = votes.max(1)
    return np.column_stack([own, maximum, entropy(votes), maximum - own])


def leave_one_out_votes(assignment, train, noisy, classes):
    count = int(np.max(assignment)) + 1
    train_groups = assignment[train]
    counts = np.zeros((count, classes), dtype=np.float64)
    np.add.at(counts, (train_groups, noisy[train]), 1.)
    remaining = counts[train_groups] - np.eye(classes)[noisy[train]]
    support = remaining.sum(1)
    votes = np.divide(remaining, support[:, None], out=np.full_like(remaining, 1. / classes),
                      where=support[:, None] > 0)
    return votes, support


def region_features(h, assignment, train, noisy, teacher):
    classes = teacher.shape[1]
    observed = noisy[train]
    groups = assignment[train]
    count = int(np.max(assignment)) + 1
    sizes = np.bincount(assignment, minlength=count)
    if np.any(sizes == 0):
        raise ValueError('empty/noncontiguous region')
    centers = np.zeros((count, h.shape[1]), dtype=np.float64)
    np.add.at(centers, assignment, h)
    centers /= sizes[:, None]
    distances = np.linalg.norm(h - centers[assignment], axis=1)
    radii = np.bincount(assignment, weights=distances, minlength=count) / sizes
    ranks = np.empty(len(h), dtype=np.float64)
    for j in range(count):
        members = np.flatnonzero(assignment == j)
        ranks[members] = rankdata(distances[members], method='average') / len(members)
    center_distances = np.sqrt(np.maximum(
        (h[train].astype(np.float64) ** 2).sum(1)[:, None]
        + (centers ** 2).sum(1)[None, :] - 2 * h[train] @ centers.T, 0))
    center_distances[np.arange(len(train)), groups] = np.inf
    alternative = center_distances.min(1) if count > 1 else distances[train]
    scale = np.maximum(radii[groups], 1e-12)
    votes, support = leave_one_out_votes(assignment, train, noisy, classes)
    mean_teacher = np.zeros((count, classes), dtype=np.float64)
    np.add.at(mean_teacher, groups, teacher[train])
    other_teacher = np.divide(mean_teacher[groups] - teacher[train], support[:, None],
                              out=np.full((len(train), classes), 1. / classes),
                              where=support[:, None] > 0)
    features = np.column_stack([
        np.log1p(sizes[groups]), np.log1p(support), vote_statistics(votes, observed),
        distances[train] / scale, ranks[train], (alternative - distances[train]) / scale,
        other_teacher[np.arange(len(train)), observed], (support == 0).astype(float),
    ])
    if features.shape[1] != len(REGION_NAMES) or not np.isfinite(features).all():
        raise ValueError('invalid region features')
    return features


@torch.no_grad()
def exact_neighbors(h, train, edge_index, k=20, chunk=256):
    device = torch.device('cuda')
    z = torch.as_tensor(h, device=device)
    references = z[torch.as_tensor(train, device=device)]
    k_train, k_peer = min(k, len(train) - 1), min(k, len(h) - 1)
    if min(k_train, k_peer) < 1:
        raise ValueError('not enough reference nodes')
    source, target = np.asarray(edge_index, dtype=np.int64)
    adjacency = sp.csr_matrix((np.ones(len(source), dtype=np.int8), (source, target)),
                              shape=(len(h), len(h)))
    adjacency = ((adjacency + adjacency.T) != 0).astype(bool).tocsr()
    knn_idx, knn_cos, peers = [], [], []
    for offset in range(0, len(train), chunk):
        nodes = train[offset:offset + chunk]
        query = z[torch.as_tensor(nodes, device=device)]
        local_similarity = query @ references.T
        local_similarity[torch.arange(len(nodes), device=device),
                         torch.arange(offset, offset + len(nodes), device=device)] = -torch.inf
        cosine, indices = torch.topk(local_similarity, k_train, dim=1)
        knn_idx.append(train[indices.cpu().numpy()])
        knn_cos.append(cosine.cpu().numpy())
        similarity = query @ z.T
        weighted = similarity * SETTINGS['peer_non_two_hop_factor']
        for row, node in enumerate(nodes):
            direct = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
            reachable = set(int(x) for x in direct)
            for neighbor in direct:
                reachable.update(int(x) for x in adjacency.indices[
                    adjacency.indptr[neighbor]:adjacency.indptr[neighbor + 1]])
            reachable.discard(int(node))
            if reachable:
                candidates = torch.as_tensor(sorted(reachable), device=device)
                weighted[row, candidates] = similarity[row, candidates]
        weighted[torch.arange(len(nodes), device=device), torch.as_tensor(nodes, device=device)] = -torch.inf
        peers.append(torch.topk(weighted, k_peer, dim=1).indices.cpu().numpy())
    result = (np.concatenate(knn_idx), np.concatenate(knn_cos), np.concatenate(peers))
    if np.any(result[0] == train[:, None]) or np.any(result[2] == train[:, None]):
        raise ValueError('current node retrieved as its own peer')
    return result


def base_features(h, train, noisy, teacher, peer_indices):
    observed = noisy[train]
    classes = teacher.shape[1]
    counts = np.bincount(observed, minlength=classes)
    prototypes = np.zeros((classes, h.shape[1]), dtype=np.float64)
    np.add.at(prototypes, observed, h[train])
    prototypes /= np.maximum(counts[:, None], 1)
    prototypes = normalize(prototypes)
    similarities = h @ prototypes.T
    similarities[:, counts == 0] = -1.
    pseudo = similarities.argmax(1)
    p = teacher[train]
    sorted_p = np.sort(p, axis=1)
    own = p[np.arange(len(train)), observed]
    peer_teacher = teacher[peer_indices].mean(1)
    own_similarity = similarities[train, observed]
    features = np.column_stack([
        own, p.max(1), entropy(p), sorted_p[:, -1] - sorted_p[:, -2], p.max(1) - own,
        (pseudo[peer_indices] == observed[:, None]).mean(1),
        peer_teacher[np.arange(len(train)), observed], peer_teacher.max(1),
        own_similarity, similarities[train].max(1) - own_similarity,
        (pseudo[train] == observed).astype(float),
    ])
    if features.shape[1] != len(BASE_NAMES) or not np.isfinite(features).all():
        raise ValueError('invalid baseline features')
    return features


def knn_features(train, noisy, indices, cosine, classes):
    votes = np.eye(classes)[noisy[indices]].mean(1)
    return np.column_stack([vote_statistics(votes, noisy[train]), cosine.mean(1), cosine.std(1)])


def coverage_metrics(labels, scores, fraction=.2):
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores)
    count = max(1, int(np.ceil(len(labels) * fraction)))
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    above, tied = scores > threshold, scores == threshold
    remaining = count - int(above.sum())
    expected_true = labels[above].sum() + remaining * labels[tied].mean()
    expected_clean = count - expected_true
    return dict(precision20=float(expected_true / count),
                recall20=float(expected_true / max(int(labels.sum()), 1)),
                clean_flag_rate20=float(expected_clean / max(int((labels == 0).sum()), 1)),
                actual_coverage=float(count / len(labels)))


def metrics(labels, scores):
    return dict(auc=float(roc_auc_score(labels, scores)),
                ap=float(average_precision_score(labels, scores)),
                logloss=float(log_loss(labels, scores, labels=[0, 1])),
                brier=float(brier_score_loss(labels, scores)), **coverage_metrics(labels, scores))


def fit_predict(train_features, train_labels, train_weights, eval_features):
    scaler = StandardScaler().fit(train_features, sample_weight=train_weights)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000)
    model.fit(scaler.transform(train_features), train_labels, sample_weight=train_weights)
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError('diagnostic logistic regression did not converge')
    return model.predict_proba(scaler.transform(eval_features))[:, 1], model, scaler


def cell_folder(dataset, rate, seed):
    return SOURCE / 'runs' / dataset / f'uniform{rate}' / f'seed{seed}'


def freeze():
    if (ROOT / 'input_manifest.json').exists():
        verify()
        return
    inputs = list(ROOT.glob('*.py')) + [ROOT / 'PLAN.md']
    original = read(SOURCE / 'frozen_inputs.json')['hashes']
    for dataset in DATASETS:
        inputs.append(DATA_FILES[dataset])
        for rate in RATES:
            for seed in SEEDS:
                cell = cell_folder(dataset, rate, seed)
                meta = read(cell / 'protocol.json')
                inputs.extend([cell / 'protocol.json', cell / 'protocol.pt',
                               cell / 'teacher/probabilities.pt', cell / 'teacher/config.json',
                               cell / 'GB0/probabilities.pt', Path(meta['representation_source']),
                               Path(meta['partition_source'])])
        km = ROOT.parent / 'gb_regional_kmeans_20261007/partitions' / dataset
        if (km / 'assignment.npy').exists():
            inputs.extend([km / 'assignment.npy', km / 'config.json'])
    hashes = {}
    for path in sorted(set(p.resolve() for p in inputs)):
        value = sha(path)
        if str(path) in original and original[str(path)] != value:
            raise ValueError(f'original source fingerprint mismatch: {path}')
        hashes[str(path)] = value
    write(ROOT / 'input_manifest.json', dict(settings=SETTINGS, hashes=hashes,
          feature_names=dict(base=BASE_NAMES, knn=KNN_NAMES, region=REGION_NAMES), arms=ARMS))
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          platform=platform.platform(), torch=torch.__version__, hip=torch.version.hip,
          gpu=torch.cuda.get_device_name(0), numpy=np.__version__, sklearn=sklearn.__version__,
          feature_compute='GPU exact cosine neighbors; CPU group/prototype statistics',
          model_compute='CPU logistic regression', supervised_model_runs=0))
    logging.info('FROZEN %s inputs and source files', len(hashes))


def verify():
    for path, expected in read(ROOT / 'input_manifest.json')['hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'frozen input changed: {path}')


def partition_pair(dataset, h, count, embedding_path):
    folder = ROOT / 'partitions' / f'{dataset}_K{count}'
    if (folder / 'config.json').exists():
        meta = read(folder / 'config.json')
        if meta['embedding_sha256'] != sha(embedding_path):
            raise ValueError('cached partition embedding changed')
        return {arm: np.load(folder / f'{arm}.npy') for arm in ('KM', 'BKM')}
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    old = ROOT.parent / 'gb_regional_kmeans_20261007/partitions' / dataset
    reuse = False
    if (old / 'config.json').exists():
        meta = read(old / 'config.json')
        reuse = meta['count'] == count and meta['embedding_sha256'] == sha(embedding_path)
    if reuse:
        km = np.load(old / 'assignment.npy').astype(np.int64)
    else:
        km = KMeans(n_clusters=count, algorithm='lloyd', **KM_CONFIG).fit_predict(h).astype(np.int64)
    logging.info('KM %s K=%s reuse=%s seconds=%.2f', dataset, count, reuse, time.perf_counter()-tick)
    tick = time.perf_counter()
    bkm = BisectingKMeans(n_clusters=count, bisecting_strategy='biggest_inertia',
                          **KM_CONFIG).fit_predict(h).astype(np.int64)
    for arm, assignment in [('KM', km), ('BKM', bkm)]:
        if len(assignment) != len(h) or len(np.unique(assignment)) != count:
            raise ValueError('partition size mismatch')
        np.save(folder / f'{arm}.npy', assignment)
    write(folder / 'config.json', dict(embedding_sha256=sha(embedding_path),
          region_count=count, km_reused=reuse, bkm_seconds=time.perf_counter()-tick,
          assignment_hashes={arm: assignment_hash(a) for arm, a in [('KM', km), ('BKM', bkm)]}))
    logging.info('BKM %s K=%s seconds=%.2f', dataset, count, time.perf_counter()-tick)
    return dict(KM=km, BKM=bkm)


def prepare_features(started):
    for dataset in DATASETS:
        raw = torch.load(DATA_FILES[dataset], map_location='cpu', weights_only=False)[0]
        edge_index = (raw['edge_index'] if isinstance(raw, dict) else raw.edge_index).numpy()
        base_meta = read(cell_folder(dataset, .4, 1) / 'protocol.json')
        embedding_path = Path(base_meta['representation_source'])
        tensor = torch.load(embedding_path, map_location='cpu', weights_only=True)
        h = normalize(tensor['H'].numpy() if isinstance(tensor, dict) else tensor.numpy())
        del raw, tensor
        for seed in SEEDS:
            proto = torch.load(cell_folder(dataset, .4, seed) / 'protocol.pt', weights_only=True)
            train = proto['train'].numpy()
            gb = proto['assignment'].numpy()
            meta = read(cell_folder(dataset, .4, seed) / 'protocol.json')
            if assignment_hash(gb) != meta['assignment_hash'] or len(gb) != len(h):
                raise ValueError('GB cache does not match protocol')
            with np.load(meta['partition_source']) as source:
                np.testing.assert_array_equal(source['assignment'], gb)
            count = len(np.unique(gb))
            groups = dict(GB=gb, **partition_pair(dataset, h, count, embedding_path),
                          Random=np.random.default_rng(100).permutation(gb))
            dest = ROOT / 'features' / dataset / f'seed{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            neighbors_path = dest / 'neighbors.npz'
            if neighbors_path.exists():
                with np.load(neighbors_path) as cached:
                    np.testing.assert_array_equal(cached['train'], train)
                    indices, cosine, peers = cached['knn'], cached['cosine'], cached['peers']
            else:
                tick = time.perf_counter()
                indices, cosine, peers = exact_neighbors(h, train, edge_index)
                np.savez_compressed(neighbors_path, train=train, knn=indices, cosine=cosine, peers=peers)
                logging.info('GPU NEIGHBORS %s seed%s seconds=%.2f', dataset, seed, time.perf_counter()-tick)
            for rate in RATES:
                if time.perf_counter() - started > SETTINGS['seconds_budget']:
                    raise TimeoutError('offline run exceeded registered 60-minute budget')
                output = dest / f'uniform{rate}.npz'
                if output.exists():
                    logging.info('REUSE FEATURES %s %.1f seed%s', dataset, rate, seed)
                    continue
                cell = cell_folder(dataset, rate, seed)
                protocol = torch.load(cell / 'protocol.pt', weights_only=True)
                np.testing.assert_array_equal(protocol['train'].numpy(), train)
                np.testing.assert_array_equal(protocol['assignment'].numpy(), gb)
                noisy = protocol['noisy'].numpy()
                teacher = torch.load(cell / 'teacher/probabilities.pt', weights_only=True).numpy().astype(np.float64)
                check_probability(teacher)
                if len(teacher) != len(h):
                    raise ValueError('teacher node count mismatch')
                features = dict(base=base_features(h, train, noisy, teacher, peers),
                                knn=knn_features(train, noisy, indices, cosine, teacher.shape[1]))
                for arm, assignment in groups.items():
                    features[arm] = region_features(h, assignment, train, noisy, teacher)
                np.savez_compressed(output, train=train, **features)
                write(output.with_suffix('.json'), dict(dataset=dataset, seed=seed, rate=rate,
                      rows=len(train), features={k: list(v.shape) for k, v in features.items()},
                      current_node_excluded_from_votes=True, clean_labels_used_in_features=False,
                      classes=teacher.shape[1], assignment_hash=assignment_hash(gb)))
                logging.info('FEATURES %s %.1f seed%s n=%s', dataset, rate, seed, len(train))
        torch.cuda.empty_cache()


def load_records():
    records = []
    for dataset in DATASETS:
        for rate in RATES:
            for seed in SEEDS:
                path = ROOT / 'features' / dataset / f'seed{seed}' / f'uniform{rate}.npz'
                with np.load(path) as stored:
                    features = {k: stored[k].copy() for k in ('base', 'knn', 'GB', 'KM', 'BKM', 'Random')}
                    train = stored['train'].copy()
                # Evaluation responses are loaded only after feature construction has ended.
                proto = torch.load(cell_folder(dataset, rate, seed) / 'protocol.pt', weights_only=True)
                labels = (proto['clean'][train] != proto['noisy'][train]).numpy().astype(np.int64)
                if len(np.unique(labels)) != 2:
                    raise ValueError('one-class diagnostic response')
                records.append(dict(dataset=dataset, rate=rate, seed=seed, train=train,
                                    features=features, labels=labels))
    return records


def fold_indices(records, heldout):
    training = [i for i, r in enumerate(records) if r['dataset'] != heldout]
    evaluation = [i for i, r in enumerate(records) if r['dataset'] == heldout]
    if not training or not evaluation:
        raise ValueError('empty dataset fold')
    return training, evaluation


def evaluate():
    records = load_records()
    rows, folds = [], []
    for heldout in DATASETS:
        training, evaluation = fold_indices(records, heldout)
        labels = np.concatenate([records[j]['labels'] for j in training])
        weights = np.concatenate([np.full(len(records[j]['labels']), 1. / len(records[j]['labels']))
                                   for j in training])
        weights *= len(weights) / weights.sum()
        folds.append(dict(heldout=heldout, training_datasets=sorted(set(records[j]['dataset'] for j in training)),
                          training_cells=len(training), evaluation_cells=len(evaluation)))
        for arm, names in ARMS.items():
            xtrain = np.concatenate([np.column_stack([records[j]['features'][name] for name in names])
                                      for j in training])
            sizes = [len(records[j]['labels']) for j in evaluation]
            xeval = np.concatenate([np.column_stack([records[j]['features'][name] for name in names])
                                     for j in evaluation])
            prediction, model, scaler = fit_predict(xtrain, labels, weights, xeval)
            dest = ROOT / 'evaluation' / heldout / arm.replace('+', '_')
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'fit.npz', coefficient=model.coef_, intercept=model.intercept_,
                                scale=scaler.scale_, mean=scaler.mean_)
            offset = 0
            for j, size in zip(evaluation, sizes):
                record = records[j]
                scores = prediction[offset:offset + size]
                offset += size
                key = dict(dataset=record['dataset'], rate=record['rate'], seed=record['seed'], arm=arm)
                rows.append(dict(**key, nodes=size, corrupted=int(record['labels'].sum()),
                                 **metrics(record['labels'], scores)))
                np.savez_compressed(dest / f"u{record['rate']}_seed{record['seed']}.npz",
                                    train=record['train'], target=record['labels'], score=scores)
            logging.info('FOLD %s ARM %s done', heldout, arm)
    export_csv(ROOT / 'per_cell_metrics.csv', rows)
    write(ROOT / 'folds.json', folds)
    summarize(rows)


def summarize(rows):
    fields = ['auc', 'ap', 'logloss', 'brier', 'precision20', 'recall20', 'clean_flag_rate20']
    dataset_rows = []
    for dataset in DATASETS:
        for arm in ARMS:
            cells = [r for r in rows if r['dataset'] == dataset and r['arm'] == arm]
            dataset_rows.append(dict(dataset=dataset, arm=arm, cells=len(cells),
                                     **{f: float(np.mean([r[f] for r in cells])) for f in fields}))
    export_csv(ROOT / 'dataset_metrics.csv', dataset_rows)
    summaries = []
    for arm in ARMS:
        selected = [r for r in dataset_rows if r['arm'] == arm]
        summaries.append(dict(arm=arm, **{f: float(np.mean([r[f] for r in selected])) for f in fields},
                              auc_dataset_std=float(np.std([r['auc'] for r in selected], ddof=1))))
    export_csv(ROOT / 'macro_metrics.csv', summaries)
    lookup = {(r['dataset'], r['rate'], r['seed'], r['arm']): r for r in rows}
    comparisons = [('Base+kNN+GB', 'Base+kNN'), ('Base+kNN+GB', 'Base+kNN+KM'),
                   ('Base+kNN+GB', 'Base+kNN+BKM'), ('Base+GB', 'Base'),
                   ('Base+GB', 'Base+KM'), ('Base+GB', 'Base+BKM')]
    paired, aggregates = [], []
    for first, second in comparisons:
        deltas = []
        for dataset in DATASETS:
            per_dataset = []
            for rate in RATES:
                for seed in SEEDS:
                    a, b = lookup[dataset, rate, seed, first], lookup[dataset, rate, seed, second]
                    row = dict(dataset=dataset, rate=rate, seed=seed, first=first, second=second,
                               **{f: a[f] - b[f] for f in fields})
                    paired.append(row)
                    per_dataset.append(row)
            deltas.append({f: float(np.mean([r[f] for r in per_dataset])) for f in fields})
        aggregates.append(dict(first=first, second=second,
              **{f: float(np.mean([r[f] for r in deltas])) for f in fields},
              positive_datasets=int(sum(d['auc'] > 0 for d in deltas)),
              auc_paired_dataset_std=float(np.std([d['auc'] for d in deltas], ddof=1))))
    export_csv(ROOT / 'paired_deltas.csv', paired)
    export_csv(ROOT / 'macro_deltas.csv', aggregates)
    primary = aggregates[0]
    g1 = primary['auc'] >= .02 and primary['positive_datasets'] >= 3
    g2 = all(r['auc'] >= .02 and r['positive_datasets'] >= 3 for r in aggregates[1:3])
    g3 = primary['precision20'] >= .02 and primary['clean_flag_rate20'] <= 0
    decision = 'confirm_fresh_data' if g1 and g2 and g3 else (
        'regional_only_not_gb_specific' if g1 and g3 else 'stop_current_confidence_branch')
    write(ROOT / 'decision.json', dict(stage='exploratory_screen', G1=g1, G2=g2, G3=g3,
          decision=decision, primary_delta=primary, gb_vs_km=aggregates[1], gb_vs_bkm=aggregates[2],
          externally_confirmed=False, diagnostic_probe_uses_clean_training_flags=True,
          deployable_algorithm_established=False, teacher_training_started=False))
    logging.info('DECISION %s G1=%s G2=%s G3=%s delta_auc=%.6f', decision, g1, g2, g3, primary['auc'])


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / 'raw_log.txt', encoding='utf-8')])
    if (ROOT / 'execution.json').exists():
        raise RuntimeError('completed experiment must not be overwritten')
    if not torch.cuda.is_available():
        raise RuntimeError('registered GPU runtime is unavailable')
    torch.set_num_threads(4)
    started = time.perf_counter()
    freeze()
    verify()
    with threadpool_limits(limits=4):
        prepare_features(started)
        evaluate()
    verify()
    write(ROOT / 'output_manifest.json', {str(p.relative_to(ROOT)): sha(p)
          for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.npz', '.npy', '.csv')})
    write(ROOT / 'execution.json', dict(status='ok', protocol_cells=20, outer_dataset_folds=5,
          arms=len(ARMS), diagnostic_fits=5 * len(ARMS), seconds=time.perf_counter()-started,
          input_hashes_unchanged=True, teacher_training_runs=0, student_training_runs=0,
          feature_source='cached GRACE + ERASE; GPU neighbors', tests_read_as_responses=False))
    logging.info('COMPLETE seconds=%.2f inputs unchanged', time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        try:
            run()
        except Exception:
            logging.exception('FAILED; completed cells are retained; thresholds unchanged')
            raise
