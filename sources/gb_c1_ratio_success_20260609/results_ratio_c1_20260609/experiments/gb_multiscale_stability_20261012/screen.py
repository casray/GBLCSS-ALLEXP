from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import sklearn
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PROJECT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / 'GB-LCSS_GitHub/src'))
from gblcss.data import GraphData, inject_train_label_noise, percent_split_indices

GRACE_ROOT = PROJECT / 'gb_graph_aware_minival_6_1'
sys.path.insert(0, str(GRACE_ROOT / 'src'))
from noisy_label_gb_grace.upstream_grace import build_grace_model

DATASETS = ['cora', 'citeseer']
SEEDS = [0, 1, 2]
KINDS = ['uniform', 'pair']
BUDGETS = [16, 32, 64]
VIEW_SEEDS = [101, 202]
COVERAGES = [.1, .2, .4, .6, 1.]
DATA_FILES = {name: PROJECT / f'external/NoisyGL/data/{display}/processed/data.pt'
              for name, display in [('cora', 'Cora'), ('citeseer', 'CiteSeer')]}
CHECKPOINTS = {
    'cora': GRACE_ROOT / 'checkpoints/grace/Cora/ratio_train0p2_val0p1_split0_pretrain0/checkpoint.pt',
    'citeseer': PROJECT / 'gb_noisy_label_validation/checkpoints/grace/CiteSeer/ratio_train0p2_val0p1_split0_pretrain0/checkpoint.pt',
}
H_FILES = {name: GRACE_ROOT / f'checkpoints/embeddings_v2/{name}/grace/H.pt' for name in DATASETS}
CLUSTER = dict(init='k-means++', n_init=10, random_state=0, max_iter=300, tol=1e-4)
SETTINGS = dict(datasets=DATASETS, protocol_seeds=SEEDS, kinds=KINDS, noise_rate=.4,
                budgets=BUDGETS, view_seeds=VIEW_SEEDS, min_child_size=32, split_reduction=.05,
                min_voters=2, support_constant=2., coverages=COVERAGES,
                primary_candidate_coverage=.2, cross_method_training_coverage=.05,
                precision_gate=.03, unique_gate=.02, positive_units=3, max_seconds=1800,
                parent_smoothing=False, true_labels_in_scoring=False, stage='exploratory')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def export(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def raw_graph(name):
    raw = torch.load(DATA_FILES[name], weights_only=False, map_location='cpu')[0]
    if isinstance(raw, dict):
        return raw['x'].float(), raw['edge_index'].long(), raw['y'].long()
    return raw.x.float(), raw.edge_index.long(), raw.y.long()


def gb_tree(h):
    leaves = [np.arange(len(h))]
    tree = [dict(parent=-1, children=[], members=np.arange(len(h)))]
    leaf_ids = [0]
    cuts, iteration = {}, 0

    def radius(indices):
        points = h[indices]
        return float(np.linalg.norm(points - points.mean(0), axis=1).mean())

    def assignment():
        result = np.empty(len(h), dtype=np.int64)
        for j, members in enumerate(leaves):
            result[members] = j
        return result

    while len(leaves) < max(BUDGETS):
        candidates = [(radius(indices), pos) for pos, indices in enumerate(leaves)
                      if len(indices) >= 2 * SETTINGS['min_child_size']]
        accepted = False
        for old_radius, pos in sorted(candidates, reverse=True):
            members = leaves[pos]
            label = KMeans(n_clusters=2, **{**CLUSTER, 'random_state': iteration + pos}).fit_predict(h[members])
            children = [members[label == c] for c in (0, 1)]
            if min(len(c) for c in children) < SETTINGS['min_child_size']:
                continue
            new_radius = sum(len(c) * radius(c) for c in children) / len(members)
            if old_radius > 1e-12 and new_radius > (1 - SETTINGS['split_reduction']) * old_radius:
                continue
            parent = leaf_ids[pos]
            child_ids = [len(tree), len(tree) + 1]
            tree[parent]['children'] = child_ids
            tree.extend(dict(parent=parent, children=[], members=c) for c in children)
            leaves.append(children[1])
            leaves[pos] = children[0]
            leaf_ids[pos] = child_ids[0]
            leaf_ids.append(child_ids[1])
            accepted = True
            break
        if not accepted:
            break
        iteration += 1
        if len(leaves) in BUDGETS:
            cuts[len(leaves)] = assignment()
    for budget in BUDGETS:
        if budget not in cuts:
            cuts[budget] = assignment()
    return cuts, tree


def nested(first, second):
    return all(len(np.unique(first[second == g])) == 1 for g in np.unique(second))


def scope_members(assignment, train):
    groups = np.asarray(assignment)[train]
    members = groups[:, None] == groups[None, :]
    np.fill_diagonal(members, False)
    return members


@torch.no_grad()
def knn_order(h, train):
    features = torch.as_tensor(h, device='cuda')
    reference = features[torch.as_tensor(train, device='cuda')]
    result = []
    for start in range(0, len(train), 256):
        similarity = reference[start:start + 256] @ reference.T
        rows = torch.arange(similarity.shape[0], device='cuda')
        similarity[rows, rows + start] = -torch.inf
        result.append(torch.argsort(similarity, dim=1, descending=True).cpu().numpy()[:, :-1])
    return np.concatenate(result)


def matched_knn_members(order, counts):
    result = np.zeros((len(order), len(order)), dtype=bool)
    for row, count in enumerate(counts):
        result[row, order[row, :int(count)]] = True
    np.fill_diagonal(result, False)
    return result


def summarize_votes(scopes, observed, classes):
    scopes = np.asarray(scopes, dtype=bool)
    if scopes.ndim != 3 or scopes.shape[1:] != (len(observed), len(observed)):
        raise ValueError('invalid voter scopes')
    if np.any(np.diagonal(scopes, axis1=1, axis2=2)):
        raise ValueError('current node supplies a vote')
    support = scopes.sum(2)
    active = support >= SETTINGS['min_voters']
    for j in range(len(scopes)):
        for previous in range(j):
            same = np.all(scopes[j] == scopes[previous], axis=1)
            active[j, same & active[previous]] = False
    votes = scopes.astype(np.float64) @ np.eye(classes)[observed]
    votes = np.divide(votes, support[:, :, None], out=np.zeros_like(votes), where=support[:, :, None] > 0)
    count = active.sum(0)
    mean = np.divide((votes * active[:, :, None]).sum(0), count[:, None],
                     out=np.full((len(observed), classes), 1. / classes), where=count[:, None] > 0)
    midpoint = .5 * (votes + mean[None])
    log = lambda value: np.log(np.maximum(value, 1e-12))
    js = .5 * ((votes * (log(votes) - log(midpoint))).sum(2)
               + (mean[None] * (log(mean[None]) - log(midpoint))).sum(2))
    disagreement = np.divide((js * active).sum(0), count, out=np.zeros(len(observed)), where=count > 0)
    stability = np.clip(1 - disagreement / np.log(2), 0, 1)
    union = np.any(scopes & active[:, :, None], axis=0).sum(1)
    ordered = np.sort(mean, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    prediction = mean.argmax(1)
    candidate = (count > 0) & (margin > 1e-12) & (prediction != observed)
    base_score = margin * union / (union + SETTINGS['support_constant'])
    return dict(probability=mean, prediction=prediction, candidate=candidate, stability=stability,
                margin=margin, unique_support=union, distinct_scopes=count,
                Margin=margin, NoStability=base_score, Full=base_score * stability, StabilityOnly=stability)


def selection(scores, candidate, fraction=None, total_fraction=None):
    nodes = np.flatnonzero(candidate)
    desired = int(np.ceil(len(nodes) * fraction)) if fraction is not None else int(np.ceil(len(scores) * total_fraction))
    weights = np.zeros(len(scores))
    if desired < 1 or len(nodes) < desired:
        return weights, False
    boundary = np.partition(scores[nodes], len(nodes) - desired)[len(nodes) - desired]
    above = candidate & (scores > boundary)
    ties = candidate & (scores == boundary)
    weights[above] = 1.
    weights[ties] = (desired - above.sum()) / ties.sum()
    return weights, True


def correction_metrics(clean, observed, prediction, weights):
    correct, initially_wrong = prediction == clean, observed != clean
    size = float(weights.sum())
    fixed = float((weights * (correct & initially_wrong)).sum())
    harmed = float((weights * (~correct & ~initially_wrong)).sum())
    return dict(selected=size, precision=float((weights * correct).sum() / size) if size > 0 else None,
                recall=fixed / max(int(initially_wrong.sum()), 1),
                clean_corruption=harmed / max(int((~initially_wrong).sum()), 1),
                actual_training_coverage=size / len(clean), net_accuracy_change=(fixed - harmed) / len(clean),
                fixed=fixed, harmed=harmed)


def freeze():
    files = list(ROOT.glob('*.py')) + [ROOT / 'PLAN.md',
            GRACE_ROOT / 'src/noisy_label_gb_grace/upstream_grace.py',
            GRACE_ROOT / 'upstream/GRACE/model.py', WORKSPACE / 'GB-LCSS_GitHub/src/gblcss/data.py']
    for name in DATASETS:
        files.extend([DATA_FILES[name], H_FILES[name], CHECKPOINTS[name]])
        for seed in SEEDS:
            files.append(ROOT.parent / f'main_protocol_replay_20260922/matrix/{name}/default/seed{seed}.json')
    write(ROOT / 'input_manifest.json', dict(settings=SETTINGS, hashes={str(p.resolve()): sha(p) for p in files}))
    write(ROOT / 'environment.json', dict(executable=sys.executable, python=sys.version,
          torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0),
          numpy=np.__version__, sklearn=sklearn.__version__, compute='GPU frozen forwards/knn; CPU clustering/votes'))


def verify():
    for path, value in read(ROOT / 'input_manifest.json')['hashes'].items():
        if sha(path) != value:
            raise ValueError(f'original source/input changed: {path}')


@torch.no_grad()
def representations(name):
    x, edges, _ = raw_graph(name)
    stored = torch.load(CHECKPOINTS[name], map_location='cpu', weights_only=False)
    model = build_grace_model(SimpleNamespace(num_features=x.shape[1]), stored['config'], torch.device('cuda'))
    model.load_state_dict(stored['model_state_dict'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = x - x.min()
    x = (x / x.sum(1, keepdim=True).clamp_min(1)).cuda()
    edges = edges.cuda()
    original = F.normalize(model(x, edges), dim=1).cpu()
    old = torch.load(H_FILES[name], map_location='cpu', weights_only=True)['H']
    error = float((original - old).abs().max())
    if error > 1e-5:
        raise ValueError(f'canonical GRACE mismatch: {name}, {error}')
    values = [normalize(old.numpy())]
    config = stored['config']
    view_logs = []
    for index, seed in enumerate(VIEW_SEEDS, 1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        edge_rate, feature_rate = config[f'drop_edge_rate_{index}'], config[f'drop_feature_rate_{index}']
        # Match the original upstream augmentation operation, including RNG order.
        augmented_edges = dropout_adj(edges, p=edge_rate, training=True)[0]
        dropped = torch.rand(x.shape[1], device=x.device) < feature_rate
        augmented_x = x.clone()
        augmented_x[:, dropped] = 0
        h = F.normalize(model(augmented_x, augmented_edges), dim=1).cpu()
        values.append(normalize(h.numpy()))
        view_logs.append(dict(view=index, seed=seed, edge_rate=edge_rate, feature_rate=feature_rate,
                              retained_edges=augmented_edges.shape[1], dropped_dimensions=int(dropped.sum()),
                              canonical_mean_cosine=float(F.cosine_similarity(h, old).mean())))
    dest = ROOT / 'representations' / name
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'views.npz', **{f'view{j}': h for j, h in enumerate(values)})
    write(dest / 'config.json', dict(checkpoint=str(CHECKPOINTS[name]), canonical_max_abs_error=error,
                                  views=view_logs, checkpoint_loaded_strictly=True))
    return values


def partitions(name, views):
    methods = {arm: [] for arm in ('GB', 'KM', 'BKM')}
    diagnostics = []
    for view, h in enumerate(views):
        gb, tree = gb_tree(h)
        dest = ROOT / 'partitions' / name / f'view{view}'
        dest.mkdir(parents=True, exist_ok=True)
        tree_values = {f'members_{j}': t['members'] for j, t in enumerate(tree)}
        np.savez_compressed(dest / 'gb_tree.npz', **tree_values)
        write(dest / 'gb_tree.json', [{k: v for k, v in t.items() if k != 'members'} for t in tree])
        counts = [len(np.unique(gb[b])) for b in BUDGETS]
        all_cuts = dict(GB=[gb[b] for b in BUDGETS], KM=[], BKM=[])
        cache = {}
        for count in counts:
            if count not in cache:
                cache[count] = dict(KM=KMeans(n_clusters=count, algorithm='lloyd', **CLUSTER).fit_predict(h),
                                    BKM=BisectingKMeans(n_clusters=count, bisecting_strategy='biggest_inertia',
                                                       **CLUSTER).fit_predict(h))
            for arm in ('KM', 'BKM'):
                all_cuts[arm].append(cache[count][arm])
        np.savez_compressed(dest / 'assignments.npz', **{f'{arm}_{b}': a for arm, cuts in all_cuts.items()
                                                       for b, a in zip(BUDGETS, cuts)})
        for arm, cuts in all_cuts.items():
            methods[arm].append(cuts)
            diagnostics.append(dict(dataset=name, view=view, method=arm, counts=counts,
                                    nested_16_32=nested(cuts[0], cuts[1]), nested_32_64=nested(cuts[1], cuts[2])))
        logging.info('PARTITIONS %s view%s counts=%s', name, view, counts)
    for arm, cuts in methods.items():
        diagnostics.append(dict(dataset=name, view='cross_view', method=arm,
                                counts=[], nested_16_32=None, nested_32_64=None,
                                **{f'ari_budget{b}': adjusted_rand_score(cuts[1][j], cuts[2][j])
                                   for j, b in enumerate(BUDGETS)}))
    write(ROOT / 'partitions' / name / 'diagnostics.json', diagnostics)
    return methods


def protocol(name, seed, kind):
    x, edges, clean = raw_graph(name)
    train, val, test = percent_split_indices(len(clean), seed)
    masks = []
    for indices in (train, val, test):
        mask = torch.zeros(len(clean), dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    data = GraphData(name, x, edges, clean, *masks, train_index=torch.as_tensor(train),
                     val_index=torch.as_tensor(val), test_index=torch.as_tensor(test))
    inject_train_label_noise(data, kind, .4, seed)
    split_hash = fingerprint(train, val, test)
    noise_hash = fingerprint(clean.numpy(), data.noisy_y.numpy())
    reference = read(ROOT.parent / f'main_protocol_replay_20260922/matrix/{name}/default/seed{seed}.json')
    assert split_hash == reference['split_hash']
    if kind == 'uniform':
        assert noise_hash == reference['noise_hash']
    assert torch.equal(clean[~masks[0]], data.noisy_y[~masks[0]])
    dest = ROOT / 'protocols' / name / f'{kind}_seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'labels.npz', train=train, observed=data.noisy_y[train].numpy(),
                        clean=clean[train].numpy())
    write(dest / 'config.json', dict(dataset=name, seed=seed, kind=kind, rate=.4,
                                   split_hash=split_hash, noise_hash=noise_hash, train_nodes=len(train),
                                   classes=data.num_classes, true_labels_not_used_to_score=True))
    return train, data.noisy_y[train].numpy(), data.num_classes


def compute_scores():
    scope_diagnostics = []
    for name in DATASETS:
        views = representations(name)
        methods = partitions(name, views)
        for seed in SEEDS:
            train, _, _ = protocol(name, seed, 'uniform')
            groups = {arm: [[scope_members(a, train) for a in cuts] for cuts in v]
                      for arm, v in methods.items()}
            orders = [knn_order(h, train) for h in views]
            groups['kNN'] = [[matched_knn_members(orders[j], m.sum(1)) for m in groups['GB'][j]]
                              for j in range(len(views))]
            for arm, v in groups.items():
                left, right = v[1][-1], v[2][-1]
                union = np.logical_or(left, right).sum(1)
                jac = np.divide(np.logical_and(left, right).sum(1), union,
                                out=np.ones(len(train)), where=union > 0)
                scope_diagnostics.append(dict(dataset=name, seed=seed, method=arm,
                                              mean_fine_cross_view_voter_jaccard=float(jac.mean()),
                                              mean_fine_support_view1=float(left.sum(1).mean()),
                                              mean_fine_support_view2=float(right.sum(1).mean())))
            for kind in KINDS:
                current_train, observed, classes = protocol(name, seed, kind)
                np.testing.assert_array_equal(train, current_train)
                dest = ROOT / 'scores' / name / f'{kind}_seed{seed}'
                dest.mkdir(parents=True, exist_ok=True)
                for arm, v in groups.items():
                    configurations = dict(FullEnsemble=v[1] + v[2], CanonicalFine=[v[0][-1]],
                                          CanonicalMultiscale=v[0], TwinFine=[v[1][-1], v[2][-1]])
                    for variant, scopes in configurations.items():
                        score = summarize_votes(scopes, observed, classes)
                        np.savez_compressed(dest / f'{arm}_{variant}.npz', train=train, observed=observed, **score)
                logging.info('SCORES %s %s seed%s ready', name, kind, seed)
        del views, methods
        torch.cuda.empty_cache()
    export(ROOT / 'scope_diagnostics.csv', scope_diagnostics)


def evaluate():
    rows, calibration, auc_rows = [], [], []
    for name in DATASETS:
        for kind in KINDS:
            for seed in SEEDS:
                label_path = ROOT / 'protocols' / name / f'{kind}_seed{seed}' / 'labels.npz'
                with np.load(label_path) as labels:
                    clean = labels['clean'].copy()
                dest = ROOT / 'scores' / name / f'{kind}_seed{seed}'
                for arm in ('GB', 'KM', 'BKM', 'kNN'):
                    for variant in ('FullEnsemble', 'CanonicalFine', 'CanonicalMultiscale', 'TwinFine'):
                        with np.load(dest / f'{arm}_{variant}.npz') as score:
                            observed, prediction, candidate = score['observed'], score['prediction'], score['candidate']
                            rankings = ['Margin', 'NoStability', 'Full', 'StabilityOnly'] if variant == 'FullEnsemble' else ['NoStability']
                            for ranking in rankings:
                                values = score[ranking]
                                correct = prediction[candidate] == clean[candidate]
                                auc = float(roc_auc_score(correct, values[candidate])) if len(np.unique(correct)) == 2 else None
                                auc_rows.append(dict(dataset=name, kind=kind, seed=seed, method=arm,
                                                     variant=variant, ranking=ranking, candidate_count=int(candidate.sum()),
                                                     candidate_correctness_auc=auc, mean_stability=float(score['stability'][candidate].mean())
                                                     if candidate.any() else None))
                                for fraction in COVERAGES:
                                    weights, sufficient = selection(values, candidate, fraction=fraction)
                                    rows.append(dict(dataset=name, kind=kind, seed=seed, method=arm,
                                         variant=variant, ranking=ranking, coverage_type='candidate_fraction',
                                         fraction=fraction, sufficient=sufficient, candidate_count=int(candidate.sum()),
                                         **correction_metrics(clean, observed, prediction, weights)))
                                if variant == 'FullEnsemble' and ranking == 'Full':
                                    weights, sufficient = selection(values, candidate, total_fraction=.05)
                                    rows.append(dict(dataset=name, kind=kind, seed=seed, method=arm,
                                         variant=variant, ranking=ranking, coverage_type='training_fraction',
                                         fraction=.05, sufficient=sufficient, candidate_count=int(candidate.sum()),
                                         **correction_metrics(clean, observed, prediction, weights)))
                            if variant == 'FullEnsemble':
                                nodes = np.flatnonzero(candidate)
                                if len(nodes):
                                    margin_order = nodes[np.argsort(score['margin'][nodes], kind='stable')]
                                    for j, block in enumerate(np.array_split(margin_order, 4)):
                                        if not len(block):
                                            continue
                                        stable_order = block[np.argsort(score['stability'][block], kind='stable')]
                                        for side, subset in zip(('low', 'high'), np.array_split(stable_order, 2)):
                                            if not len(subset):
                                                continue
                                            calibration.append(dict(dataset=name, kind=kind, seed=seed, method=arm,
                                                 margin_quartile=j, stability_half=side, nodes=len(subset),
                                                 mean_margin=float(score['margin'][subset].mean()),
                                                 mean_stability=float(score['stability'][subset].mean()),
                                                 correction_precision=float((prediction[subset] == clean[subset]).mean())))
    export(ROOT / 'coverage_curves.csv', rows)
    export(ROOT / 'candidate_auc.csv', auc_rows)
    export(ROOT / 'conditional_stability.csv', calibration)
    decide(rows)


def decide(rows):
    primary = [r for r in rows if r['variant'] == 'FullEnsemble' and r['coverage_type'] == 'candidate_fraction'
               and r['fraction'] == .2]
    export(ROOT / 'primary_cells.csv', primary)
    fields = ['precision', 'recall', 'clean_corruption', 'net_accuracy_change', 'actual_training_coverage']
    units = []
    for name in DATASETS:
        for kind in KINDS:
            for method in ('GB', 'KM', 'BKM', 'kNN'):
                for ranking in ('Margin', 'NoStability', 'Full', 'StabilityOnly'):
                    subset = [r for r in primary if (r['dataset'], r['kind'], r['method'], r['ranking']) ==
                              (name, kind, method, ranking)]
                    valid = all(r['sufficient'] for r in subset)
                    row = dict(dataset=name, kind=kind, method=method, ranking=ranking, valid=valid)
                    for f in fields:
                        values = [r[f] for r in subset if r[f] is not None]
                        row[f] = float(np.mean(values)) if len(values) == 3 else None
                        row[f + '_seed_std'] = float(np.std(values, ddof=1)) if len(values) == 3 else None
                    units.append(row)
    export(ROOT / 'primary_unit_summary.csv', units)
    lookup = {(r['dataset'], r['kind'], r['method'], r['ranking']): r for r in units}
    deltas = []
    for name in DATASETS:
        for kind in KINDS:
            full, baseline = lookup[name, kind, 'GB', 'Full'], lookup[name, kind, 'GB', 'NoStability']
            valid = full['valid'] and baseline['valid']
            deltas.append(dict(dataset=name, kind=kind, valid=valid,
                          **{f: full[f] - baseline[f] if valid else None for f in fields}))
    export(ROOT / 'mechanism_deltas.csv', deltas)
    valid = all(r['valid'] for r in deltas)
    mean = {f: float(np.mean([r[f] for r in deltas])) if valid else None for f in fields}
    positive = sum(r['precision'] > 0 for r in deltas if r['valid'])
    g1 = valid and mean['precision'] >= .03 and positive >= 3 and mean['clean_corruption'] <= 0
    matched = [r for r in rows if r['coverage_type'] == 'training_fraction']
    export(ROOT / 'matched_coverage_cells.csv', matched)
    paired = []
    for other in ('BKM', 'kNN', 'KM'):
        precision_deltas = []
        complete = True
        unit_positive = 0
        for name in DATASETS:
            for kind in KINDS:
                group = [r for r in matched if r['dataset'] == name and r['kind'] == kind]
                a = sorted([r for r in group if r['method'] == 'GB'], key=lambda r: r['seed'])
                b = sorted([r for r in group if r['method'] == other], key=lambda r: r['seed'])
                ok = all(r['sufficient'] for r in a+b) and len(a) == len(b) == 3
                if ok:
                    difference = float(np.mean([x['precision']-y['precision'] for x,y in zip(a,b)]))
                    precision_deltas.append(difference)
                    unit_positive += difference > 0
                else:
                    difference = None
                    complete = False
                paired.append(dict(dataset=name, kind=kind, comparator=other, complete=ok, precision_delta=difference))
        paired.append(dict(dataset='macro', kind='both', comparator=other, complete=complete,
                           precision_delta=float(np.mean(precision_deltas)) if complete else None,
                           positive_units=int(unit_positive)))
    keys = ['dataset', 'kind', 'comparator', 'complete', 'precision_delta', 'positive_units']
    export(ROOT / 'matched_coverage_deltas.csv', [{k:r.get(k) for k in keys} for r in paired])
    main = [r for r in paired if r['dataset'] == 'macro' and r['comparator'] in ('BKM','kNN')]
    g2 = all(r['complete'] and r['precision_delta'] >= .02 and r['positive_units'] >= 3 for r in main)
    decision = 'external_confirmation' if g1 and g2 else ('generic_stability_only' if g1 else 'stop_stability_implementation')
    write(ROOT / 'decision.json', dict(G1=g1, G2=g2, decision=decision,
          stability_increment=mean, positive_units=int(positive), grouping_comparisons=main,
          stage='exploratory', externally_confirmed=False, classifier_training_started=False))
    logging.info('DECISION %s G1=%s G2=%s precision_increment=%s', decision, g1, g2, mean['precision'])


def run():
    if (ROOT / 'execution.json').exists() or (ROOT / 'input_manifest.json').exists():
        raise RuntimeError('refusing to overwrite frozen experiment')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / 'raw_log.txt', encoding='utf-8')])
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    torch.set_num_threads(4)
    started = time.perf_counter()
    freeze()
    with threadpool_limits(limits=4):
        compute_scores()
        if time.perf_counter() - started > SETTINGS['max_seconds']:
            raise TimeoutError('registered time budget exceeded')
        # All scoring is committed before any true correction-quality metrics are computed.
        score_files = list((ROOT / 'scores').rglob('*.npz'))
        write(ROOT / 'score_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in score_files})
        evaluate()
    verify()
    write(ROOT / 'execution.json', dict(status='ok', protocol_cells=12, datasets=2,
          frozen_encoder_checkpoints=2, augmented_views=4, new_model_training_runs=0,
          score_files=len(score_files), seconds=time.perf_counter()-started, inputs_unchanged=True))
    logging.info('COMPLETE %.2f seconds', time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        try:
            run()
        except Exception:
            logging.exception('execution failed; original files retained')
            raise
