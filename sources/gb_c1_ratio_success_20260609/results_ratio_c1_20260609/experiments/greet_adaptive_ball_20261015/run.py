"""Controlled re-partitioning of frozen GRACE/GREET with identical adaptive rules."""
from pathlib import Path
import argparse
import json
import logging
import math
import shutil
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
PREVIOUS = ROOT.parent / 'greet_half_pilot_20261015'
MECHANISM = ROOT.parent / 'heterophily_mechanism_20261014'
sys.path.insert(0, str(MECHANISM))
import diagnostic_v2 as d
sys.path.insert(0, str(PREVIOUS))
from target_tools import ball_bundle as make_balls
from gblcss.granular_ball_adaptive import build_adaptive_granular_balls
from gblcss.representation import load_frozen_representation
from gblcss.data import inject_train_label_noise, set_seed
from gblcss.training import train_x_only_mlp

CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
ARMS = ['CE'] + [space + '_' + rule for space in CONFIG['spaces'] for rule in CONFIG['rules']]
H_PATHS = dict(GRACE=dict(d.CACHED_H, roman=MECHANISM / 'representations/Roman-empire/grace/H.pt'),
               GREET={n: PREVIOUS / 'representations' / n / 'H.pt' for n in CONFIG['datasets']})


def build_partition(h, train, budget, seed):
    return build_adaptive_granular_balls(h, target_balls=budget, train_index=train,
                                       min_train_per_ball=16, random_state=seed)


def bundle_from_neighbors(weights, observed, classes):
    r, q, alpha, support, _ = d.vote(weights, observed, classes)
    return dict(r=r, q=q, alpha=alpha, support=support, target=d.mix(q, alpha, observed))


def write(path, value):
    d.write(path, value)


def status(stage, **extra):
    write(ROOT / 'status.json', dict(status='running', stage=stage,
          updated_local=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))


def freeze_sources():
    files = [ROOT / n for n in ['run.py', 'audit_report.py', 'test_protocol.py', 'config.json', 'PLAN.md']]
    files += [Path(d.__file__), PREVIOUS / 'target_tools.py', PREVIOUS / 'methods.py']
    files += [WORKSPACE / 'GB-LCSS_code/src/gblcss' / n for n in
              ['granular_ball_adaptive.py', 'data.py', 'representation.py', 'training.py', 'models.py']]
    inputs = {str(p.resolve()): d.sha(p) for p in files}
    for name in CONFIG['datasets']:
        inputs[str(d.DATA[name].resolve())] = d.sha(d.DATA[name])
        for space in CONFIG['spaces']:
            p = H_PATHS[space][name]
            inputs[str(p.resolve())] = d.sha(p)
        for seed in CONFIG['protocol_seeds']:
            old = MECHANISM / 'runs' / name / 'grace' / f'seed{seed}'
            for p in [old / 'assignments.npz', PREVIOUS / 'partitions' / name / f'seed{seed}' / 'assignments.npz']:
                inputs[str(p.resolve())] = d.sha(p)
            for condition in CONFIG['conditions']:
                for p in [old / condition / 'protocol.json', old / condition / 'GB64.npz']:
                    inputs[str(p.resolve())] = d.sha(p)
    manifest = ROOT / 'source_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != inputs:
        raise ValueError('Frozen sources or inputs changed; start a new version')
    write(manifest, inputs)
    for p in files:
        target = ROOT / 'source_snapshot' / p.name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and d.sha(target) != d.sha(p):
            raise ValueError('Frozen source snapshot changed: ' + str(p))
        if not target.exists():
            shutil.copyfile(p, target)


def verify_sources():
    for p, digest in json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8')).items():
        if d.sha(p) != digest:
            raise ValueError('Frozen input modified: ' + p)


def load_case(name, seed, condition):
    graph = d.load_graph(name, seed)
    graph = inject_train_label_noise(graph, 'uniform' if condition == 'clean' else condition,
                                    CONFIG['conditions'][condition], seed)
    tr = graph.train_index.numpy()
    observed = graph.noisy_y[tr].numpy()
    folder = MECHANISM / 'runs' / name / 'grace' / f'seed{seed}' / condition
    prior = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    split_hash = d.fingerprint(tr, graph.val_index.numpy(), graph.test_index.numpy())
    noise_hash = d.fingerprint(observed)
    if split_hash != prior['split_fingerprint'] or noise_hash != prior['noisy_train_fingerprint']:
        raise ValueError('Historical protocol mismatch')
    protocol = dict(dataset=name, seed=seed, noise=condition, split_hash=split_hash, noise_hash=noise_hash,
                    val_clean=bool(torch.equal(graph.noisy_y[graph.val_index], graph.y[graph.val_index])),
                    test_clean=bool(torch.equal(graph.noisy_y[graph.test_index], graph.y[graph.test_index])),
                    source_manifest_sha256=d.sha(ROOT / 'source_manifest.json'))
    return graph, protocol


def geometry_stats(h, assignment, train):
    sizes = np.bincount(assignment); support = np.bincount(assignment[train], minlength=len(sizes))
    total_sse = 0.
    for group in range(len(sizes)):
        z = h[assignment == group].astype(np.float64)
        total_sse += float(np.square(z - z.mean(0)).sum())
    return dict(actual_balls=len(sizes), min_size=int(sizes.min()), max_size=int(sizes.max()),
                median_size=float(np.median(sizes)), min_train_support=int(support.min()),
                median_train_support=float(np.median(support)), max_train_support=int(support.max()),
                mean_other_votes=float((support[assignment[train]] - 1).mean()),
                normalized_input_sse=total_sse)


def partitions(name, seed, space, graph, device):
    folder = ROOT / 'partitions' / name / f'seed{seed}' / space
    if (folder / 'complete.json').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    h = load_frozen_representation(H_PATHS[space][name], graph.num_nodes).numpy()
    train = graph.train_index.numpy()
    old_path = (MECHANISM / 'runs' / name / 'grace' / f'seed{seed}' / 'assignments.npz' if space == 'GRACE'
                else PREVIOUS / 'partitions' / name / f'seed{seed}' / 'assignments.npz')
    with np.load(old_path) as prior:
        assignments = {'OLD64': prior['GB64' if space == 'GRACE' else 'GB'].copy()}
    stats = {'OLD64': dict(source=str(old_path))}
    for rule, budget in [('ADAPT64', 64), ('SQRT16', math.ceil(math.sqrt(graph.num_nodes)))]:
        start = time.time()
        logging.info('PARTITION %s seed%d %s %s budget%d', name, seed, space, rule, budget)
        result = build_partition(h, train, budget, seed)
        assignments[rule] = result.assignment
        stats[rule] = dict(**result.summary, elapsed_seconds=time.time() - start)
        write(folder / (rule + '_trace.json'), result.split_trace)
        write(folder / (rule + '_balls.json'), result.balls)
    count = len(np.unique(assignments['SQRT16']))
    assignments['KM_MATCH'] = KMeans(n_clusters=count, n_init=10, random_state=seed).fit_predict(h)
    for rule, assignment in assignments.items():
        stats.setdefault(rule, {}).update(geometry_stats(h, assignment, train))
    counts = np.bincount(assignments['SQRT16'][train])[assignments['SQRT16'][train]] - 1
    weights = d.knn_weights(h[train], counts, device=device)
    if weights.diagonal().any() or not np.array_equal(np.diff(weights.indptr), counts):
        raise ValueError('kNN self-vote or support mismatch')
    sparse.save_npz(folder / 'knn.npz', weights)
    np.savez_compressed(folder / 'assignments.npz', **assignments)
    write(folder / 'stats.json', stats)
    write(folder / 'complete.json', dict(h_path=str(H_PATHS[space][name]), h_sha256=d.sha(H_PATHS[space][name]),
          train_hash=d.fingerprint(train), assignment_sha256=d.sha(folder / 'assignments.npz'),
          knn_sha256=d.sha(folder / 'knn.npz'), unique_voter_match=True))


def get_bundles(name, seed, observed, classes, train):
    onehot = np.eye(classes)[observed]
    bundles = {'CE': dict(r=onehot, q=onehot, alpha=np.zeros(len(train)), target=onehot,
                          support=np.zeros(len(train), dtype=np.int64))}
    for space in CONFIG['spaces']:
        folder = ROOT / 'partitions' / name / f'seed{seed}' / space
        with np.load(folder / 'assignments.npz') as a:
            for rule in ['OLD64', 'ADAPT64', 'SQRT16', 'KM_MATCH']:
                bundles[space + '_' + rule] = make_balls(a[rule][train], observed, classes)
        weights = sparse.load_npz(folder / 'knn.npz')
        bundles[space + '_KNN_MATCH'] = bundle_from_neighbors(weights, observed, classes)
    return bundles


def summarize():
    for kind in ['target', 'student']:
        pattern = 'targets/*/seed*/*/metrics.json' if kind == 'target' else 'students/*/seed*/*/*/result.json'
        rows = []
        for p in ROOT.glob(pattern):
            value = json.loads(p.read_text(encoding='utf-8'))
            rows.extend(value if isinstance(value, list) else [value])
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame.to_csv(ROOT / (kind + '_results.csv'), index=False, encoding='utf-8-sig')
        metric = 'target_brier' if kind == 'target' else 'test_accuracy'
        frame.groupby(['dataset', 'noise', 'arm'])[metric].agg(['count', 'mean', 'std']).reset_index().to_csv(
            ROOT / (kind + '_summary.csv'), index=False, encoding='utf-8-sig')


def targets(name, seed):
    for condition in CONFIG['conditions']:
        folder = ROOT / 'targets' / name / f'seed{seed}' / condition
        if (folder / 'metrics.json').exists():
            continue
        graph, protocol = load_case(name, seed, condition)
        train = graph.train_index.numpy(); observed = graph.noisy_y[train].numpy()
        bundles = get_bundles(name, seed, observed, graph.num_classes, train)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'observed_train.npy', observed)
        frozen = {}
        for arm, bundle in bundles.items():
            np.savez_compressed(folder / (arm + '.npz'), **bundle)
            frozen[arm] = d.sha(folder / (arm + '.npz'))
        write(folder / 'frozen_targets.json', frozen)
        write(folder / 'protocol.json', protocol)
        historical = MECHANISM / 'runs' / name / 'grace' / f'seed{seed}' / condition / 'GB64.npz'
        with np.load(historical) as prior:
            if not np.allclose(prior['target'], bundles['GRACE_OLD64']['target'], atol=2e-6, rtol=0):
                raise ValueError('Historical GRACE targets not reproduced')
        # Clean labels enter diagnostics only after target artifacts have been frozen.
        clean = graph.y[train].numpy()
        clean_bundles = get_bundles(name, seed, clean, graph.num_classes, train)
        rows = []
        for arm, bundle in bundles.items():
            metric = d.target_metrics(bundle['target'], bundle['q'], bundle['alpha'], clean, observed)
            r = clean_bundles[arm]['r']; other = r.copy(); other[np.arange(len(train)), clean] = -np.inf
            margin = r[np.arange(len(train)), clean] - other.max(1)
            valid = bundle['support'] >= 2
            rows.append(dict(dataset=name, seed=seed, noise=condition, arm=arm, **metric,
                        mean_support=float(bundle['support'].mean()), support_coverage=float(valid.mean()),
                        positive_clean_margin_fraction=float(np.mean(valid & (margin > 0))) if arm != 'CE' else None,
                        mean_clean_margin=float(margin[valid].mean()) if valid.any() else None))
        write(folder / 'metrics.json', rows)
        summarize()
        logging.info('TARGETS %s seed%d %s arms%d', name, seed, condition, len(bundles))


def students(name, seed, device):
    for condition in CONFIG['conditions']:
        graph, protocol = load_case(name, seed, condition)
        parent = ROOT / 'targets' / name / f'seed{seed}' / condition
        saved_protocol = json.loads((parent / 'protocol.json').read_text(encoding='utf-8'))
        if saved_protocol != protocol:
            raise ValueError('Frozen student protocol mismatch')
        frozen = json.loads((parent / 'frozen_targets.json').read_text(encoding='utf-8'))
        for arm in ARMS:
            folder = ROOT / 'students' / name / f'seed{seed}' / condition / arm
            if (folder / 'result.json').exists():
                continue
            target_path = parent / (arm + '.npz')
            if d.sha(target_path) != frozen[arm]:
                raise ValueError('Frozen target modified')
            with np.load(target_path) as a:
                local = a['target'].copy()
            full = torch.nn.functional.one_hot(graph.noisy_y, graph.num_classes).float()
            full[graph.train_index] = torch.from_numpy(local).float()
            write(folder / 'config.json', dict(**protocol, arm=arm, training=CONFIG['training'], device=device,
                  target_path=str(target_path), target_sha256=frozen[arm]))
            status('students', dataset=name, seed=seed, noise=condition, arm=arm)
            set_seed(seed); start = time.time()
            result = train_x_only_mlp(graph, full, device=device, **CONFIG['training'])
            torch.save(result.logits, folder / 'logits.pt')
            row = dict(dataset=name, seed=seed, noise=condition, arm=arm,
                       train_accuracy=result.train_accuracy, validation_accuracy=result.validation_accuracy,
                       test_accuracy=result.test_accuracy, best_epoch=result.best_epoch,
                       elapsed_seconds=time.time() - start, target_sha256=frozen[arm])
            write(folder / 'result.json', row)
            summarize()
            logging.info('STUDENT %s seed%d %s %s test%.6f', name, seed, condition, arm, result.test_accuracy)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / 'raw.log', encoding='utf-8')])
    try:
        freeze_sources()
        with threadpool_limits(limits=4):
            for name in CONFIG['datasets']:
                for seed in CONFIG['protocol_seeds']:
                    graph, _ = load_case(name, seed, 'clean')
                    for space in CONFIG['spaces']:
                        status('partitions', dataset=name, seed=seed, space=space)
                        partitions(name, seed, space, graph, args.device)
                    status('targets', dataset=name, seed=seed)
                    targets(name, seed)
            verify_sources()
            for name in CONFIG['datasets']:
                for seed in CONFIG['protocol_seeds']:
                    students(name, seed, args.device)
        verify_sources(); summarize()
        from audit_report import audit_and_report
        audit_and_report()
        write(ROOT / 'status.json', dict(status='completed', stage='all',
              updated_local=time.strftime('%Y-%m-%d %H:%M:%S'), students=CONFIG['expected_targets_and_students']))
    except Exception as exc:
        write(ROOT / 'status.json', dict(status='failed', error=repr(exc), updated_local=time.strftime('%Y-%m-%d %H:%M:%S')))
        raise


if __name__ == '__main__':
    main()
