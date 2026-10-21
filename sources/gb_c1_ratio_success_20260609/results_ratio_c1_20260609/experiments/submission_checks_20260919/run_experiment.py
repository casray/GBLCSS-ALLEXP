"""Fixed-protocol sensitivity runs using the unchanged submission implementation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUTPUT = HERE
DEVICE = 'cpu'


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def historical_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def arms(cfg):
    result = [{'name': 'default', 'source': 'grace', 'override': {}}]
    for key, values in cfg['grids'].items():
        result.extend({'name': f'{key}_{v}', 'source': 'grace', 'override': {key: v}}
                      for v in values)
    result.extend({'name': n, 'source': n, 'override': {}}
                  for n in ['mlp_ce', 'shuffled', 'raw_x', 'a2x'])
    return result


def brier_decomposition(targets, noisy, clean, train, classes):
    target = targets[train].double()
    y, z = clean[train], noisy[train]
    truth = F.one_hot(y, classes).double()
    errors = (target - truth).square().sum(1)
    corrupt = y != z
    n = len(y)
    original = 2.0 * corrupt.double().mean().item()
    gain = (2.0 - errors[corrupt]).sum().item() / n
    cost = errors[~corrupt].sum().item() / n
    predicted = target.argmax(1)
    result = dict(target_brier=errors.mean().item(), noisy_brier=original,
                  brier_gain=original-errors.mean().item(), corrupted_gain=gain, clean_cost=cost,
                  target_accuracy=(predicted == y).double().mean().item(),
                  corrupted_count=int(corrupt.sum()), clean_count=int((~corrupt).sum()),
                  corrected_corrupt=int(((predicted == y) & corrupt).sum()),
                  damaged_clean=int(((predicted != y) & ~corrupt).sum()))
    assert abs(result['brier_gain'] - gain + cost) < 1e-7
    return result


def validate_pairs(rows):
    groups = {}
    for row in rows:
        key = row['dataset'], int(row['seed'])
        pair = row['split_hash'], row['noise_hash']
        assert groups.setdefault(key, pair) == pair, f'protocol drift: {key}'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def snapshot(cfg):
    root = Path(cfg['code_root'])
    dest = HERE / 'source_snapshot'
    for folder in ['src', 'configs']:
        for p in (root / folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.yaml'}:
                q = dest / p.relative_to(root)
                if q.exists():
                    assert q.read_bytes() == p.read_bytes(), f'source changed: {p}'
                else:
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, q)
    sys.path.insert(0, str(dest / 'src'))
    return root, dest


def load_graph(key, seed, cfg):
    from torch_geometric.datasets import Planetoid, CitationFull
    from gblcss.data import GraphData, percent_split_indices, inject_train_label_noise
    root = Path(cfg['data_root'])
    display = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'corafull': 'Cora'}[key]
    filename = 'data_undirected.pt' if key == 'corafull' else 'data.pt'
    if not (root / display / 'processed' / filename).exists():
        raise FileNotFoundError(f'local cached dataset missing: {key}; download not implicit')
    raw = (CitationFull(root=root, name='cora') if key == 'corafull'
           else Planetoid(root=root, name=display))[0]
    indices = [torch.as_tensor(x) for x in percent_split_indices(raw.num_nodes, seed)]
    masks = []
    for index in indices:
        mask = torch.zeros(raw.num_nodes, dtype=torch.bool)
        mask[index] = True
        masks.append(mask)
    graph = GraphData(key, raw.x.float(), raw.edge_index.long(), raw.y.long(), *masks,
                      train_index=indices[0], val_index=indices[1], test_index=indices[2])
    inject_train_label_noise(graph, cfg['noise_type'], cfg['noise_rate'], seed)
    assert torch.equal(graph.y[~graph.train_mask], graph.noisy_y[~graph.train_mask])
    return graph


def group_diagnostics(graph, assignment, targets, alpha, min_support):
    train = graph.train_index.numpy()
    a = assignment[train]
    clean = graph.y[graph.train_index].numpy()
    noisy = graph.noisy_y[graph.train_index].numpy()
    groups = np.unique(assignment)
    purity = 0
    positive_margin = 0
    support = np.zeros(len(train), dtype=int)
    for group in groups:
        positions = np.flatnonzero(a == group)
        if not len(positions):
            continue
        counts = np.bincount(clean[positions], minlength=graph.num_classes)
        purity += int(counts.max())
        support[positions] = len(positions)-1
        for pos in positions:
            others = counts.copy()
            others[clean[pos]] -= 1
            own = others[clean[pos]]
            others[clean[pos]] = -1
            positive_margin += int(own > others.max())
    return dict(num_balls=len(groups), min_group_nodes=int(np.bincount(assignment).min()),
                mean_group_nodes=float(len(assignment)/len(groups)),
                train_purity_weighted=purity/len(train),
                positive_clean_margin_fraction=positive_margin/len(train),
                mean_other_votes=float(support.mean()),
                eligible_support_fraction=float((support >= min_support).mean()),
                alpha_coverage=float((alpha[graph.train_index] > 0).float().mean()),
                mean_alpha=float(alpha[graph.train_index].mean()))


def prepare(key, seed, cfg, matrix, base, root):
    from gblcss.data import set_seed
    from gblcss.granular_ball import build_granular_balls, random_size_matched_assignment
    from gblcss.targets import build_leave_one_out_targets
    from gblcss.representation import load_frozen_representation
    from gblcss.ablation_targets import _representation_source
    from threadpoolctl import threadpool_limits
    graph = load_graph(key, seed, cfg)
    hp = root / 'representations' / key / 'grace/H.pt'
    h = load_frozen_representation(hp, graph.num_nodes)
    split_hash = historical_hash(graph.train_index, graph.val_index, graph.test_index)
    noise_hash = historical_hash(graph.y, graph.noisy_y)
    if (key, seed) == ('cora', 0):
        assert split_hash == 'fe241a5bd04c1aba7c5312aa410fc4c653a2c14dd68e58e9015b5271eb4c67f9'
        assert noise_hash == 'eb786f64ad9bf24efea0826e657708af4a1f081023c1eca47b0106fc2765d3a1'
    common = dict(dataset=key, seed=seed, noise_type=cfg['noise_type'], noise_rate=cfg['noise_rate'],
                  split_hash=split_hash, noise_hash=noise_hash,
                  representation_hash=hashlib.sha256(hp.read_bytes()).hexdigest())
    destination = OUTPUT / 'runs' / key / f'seed{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    partitions = {}
    representations = {'grace': h}
    archive = list(csv.DictReader((root / 'results/figure_data/fig4_target_quality_per_seed.csv').open(encoding='utf-8-sig')))
    expected = next(x for x in archive if x['dataset'] == key and int(x['seed']) == seed
                    and x['noise_type'] == cfg['noise_type'] and float(x['noise_rate']) == cfg['noise_rate'])
    for arm in matrix:
        path = destination / (arm['name']+'.json')
        if path.exists():
            recorded = json.loads(path.read_text(encoding='utf8'))
            assert recorded['split_hash'] == split_hash and recorded['noise_hash'] == noise_hash
            if arm['name'] == 'default':
                grouping_key = ('grace', recorded['max_balls'], recorded['min_ball_size'], recorded['split_improvement'])
                partitions[grouping_key] = SimpleNamespace(assignment=np.load(path.with_suffix('.npz'))['assignment'])
            continue
        started = time.perf_counter()
        method = {**base['method'], **arm['override']}
        source = arm['source']
        set_seed(seed)
        if source == 'mlp_ce':
            target = F.one_hot(graph.noisy_y, graph.num_classes).float()
            alpha = torch.zeros(graph.num_nodes)
            assignment = np.zeros(graph.num_nodes, dtype=np.int64)
            details = {}
        else:
            rep = source if source in ['raw_x', 'a2x'] else 'grace'
            if rep not in representations:
                representations[rep] = _representation_source(
                    graph, 'raw_features' if rep == 'raw_x' else 'propagated_features')
                # Match the existing ablation preparation before the common builder.
                representations[rep] = F.normalize(representations[rep], dim=1, eps=1e-12)
            grouping_key = (rep, method['max_balls'], method['min_ball_size'], method['split_improvement'])
            if grouping_key not in partitions:
                with threadpool_limits(limits=cfg['grouping_threads']):
                    partitions[grouping_key] = build_granular_balls(
                        representations[rep], max_balls=method['max_balls'],
                        min_ball_size=method['min_ball_size'], split_improvement=method['split_improvement'],
                        random_state=seed)
            assignment = partitions[grouping_key].assignment
            if source == 'shuffled':
                assignment = random_size_matched_assignment(assignment, seed + 17)
            result = build_leave_one_out_targets(
                graph.noisy_y, assignment, graph.train_index, graph.num_classes,
                min_train_per_ball=method['min_train_per_ball'], beta=method['beta'])
            target, alpha = result.soft_target, result.alpha
            details = result.diagnostics
        assert torch.isfinite(target).all() and torch.allclose(target.sum(1), torch.ones(graph.num_nodes), atol=1e-6)
        assert torch.equal(target[~graph.train_mask], F.one_hot(graph.noisy_y[~graph.train_mask], graph.num_classes).float())
        quality = brier_decomposition(target, graph.noisy_y, graph.y, graph.train_index, graph.num_classes)
        assert abs(quality['noisy_brier'] - float(expected['noisy_onehot_brier'])) < 2e-5
        archive_brier = float(expected['soft_target_brier'])
        if arm['name'] == 'default':
            assert abs(quality['target_brier'] - archive_brier) < 2e-5, (key, seed, quality['target_brier'], archive_brier)
        row = {**common, 'arm': arm['name'], 'source': source, **method, **details, **quality,
               **group_diagnostics(graph, assignment, target, alpha, method['min_train_per_ball']),
               'target_hash': fingerprint(target), 'assignment_hash': fingerprint(assignment),
               'archived_default_brier': archive_brier,
               'default_archive_brier_delta': quality['target_brier']-archive_brier if arm['name']=='default' else None,
               'target_seconds': time.perf_counter()-started, 'status': 'prepared'}
        if source == 'mlp_ce':
            for field in ['num_balls', 'min_group_nodes', 'mean_group_nodes', 'train_purity_weighted',
                          'positive_clean_margin_fraction', 'mean_other_votes', 'eligible_support_fraction']:
                row[field] = None
        np.savez_compressed(path.with_suffix('.npz'), targets=target.numpy(), alpha=alpha.numpy(), assignment=assignment)
        write_json(path, row)
        print(json.dumps({'stage':'prepared','dataset':key,'seed':seed,'arm':arm['name'],
                          'balls':row['num_balls'],'brier':row['target_brier'],
                          'seconds':round(row['target_seconds'],2)}), flush=True)
    return graph


def train(key, seed, cfg, matrix, base):
    from gblcss.data import set_seed
    from gblcss import training
    graph = load_graph(key, seed, cfg)
    destination = OUTPUT / 'runs' / key / f'seed{seed}'
    completed = {}
    for p in destination.glob('*.json'):
        row = json.loads(p.read_text(encoding='utf8'))
        if row.get('status') == 'complete':
            assert row['epochs'] == cfg['epochs'] and row['threads'] == cfg['threads']
            assert row.get('device', 'cpu') == DEVICE
            completed.setdefault(row['target_hash'], row)
    for arm in matrix:
        path = destination / (arm['name']+'.json')
        row = json.loads(path.read_text(encoding='utf8'))
        if row['status'] == 'complete':
            continue
        if row['target_hash'] in completed:
            other = completed[row['target_hash']]
            for k in ['test_accuracy', 'validation_accuracy', 'train_accuracy', 'best_epoch', 'epochs', 'threads', 'device', 'gpu_peak_allocated_mib']:
                row[k] = other[k]
            row.update(status='complete', training_seconds=0.0, reused_from=other['arm'])
        else:
            target = torch.from_numpy(np.load(path.with_suffix('.npz'))['targets'])
            set_seed(seed)
            loss_values = []
            original_loss = training.soft_cross_entropy

            def record_loss(logits, values):
                assert str(logits.device) == DEVICE
                loss = original_loss(logits, values)
                loss_values.append(float(loss.detach()))
                return loss

            training.soft_cross_entropy = record_loss
            if DEVICE.startswith('cuda'):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                result = training.train_x_only_mlp(
                    graph, target, **base['model'], learning_rate=base['training']['learning_rate'],
                    weight_decay=base['training']['weight_decay'], epochs=cfg['epochs'], device=DEVICE)
            finally:
                training.soft_cross_entropy = original_loss
            assert len(loss_values) == cfg['epochs'] and np.isfinite(loss_values).all()
            row.update(test_accuracy=result.test_accuracy, validation_accuracy=result.validation_accuracy,
                       train_accuracy=result.train_accuracy, best_epoch=result.best_epoch, epochs=cfg['epochs'],
                       threads=cfg['threads'], training_seconds=time.perf_counter()-start,
                       device=DEVICE,
                       gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if DEVICE.startswith('cuda') else 0,
                       status='complete', reused_from='')
            write_csv(path.with_name(path.stem+'_loss.csv'), [dict(epoch=i+1, loss=v) for i,v in enumerate(loss_values)])
            np.savez_compressed(path.with_name(path.stem+'_predictions.npz'), logits=result.logits.numpy())
        write_json(path, row)
        completed.setdefault(row['target_hash'], row)
        print(json.dumps({'stage':'trained','dataset':key,'seed':seed,'arm':arm['name'],
                          'accuracy':round(row['test_accuracy']*100,3),
                          'seconds':round(row['training_seconds'],2),'reused_from':row['reused_from']}), flush=True)


def summarize():
    import pandas as pd
    rows = [json.loads(p.read_text(encoding='utf8')) for p in (OUTPUT/'runs').glob('*/*/*.json')]
    validate_pairs(rows)
    write_csv(OUTPUT/'target_diagnostics.csv', rows)
    complete = [x for x in rows if x['status']=='complete']
    write_csv(OUTPUT/'per_seed_results.csv', complete)
    if complete:
        frame = pd.DataFrame(complete)
        references = frame[frame.arm=='default'][['dataset','seed','test_accuracy','target_brier']]
        joined = frame.merge(references, on=['dataset','seed'], suffixes=('', '_reference'), validate='many_to_one')
        joined['delta_pp'] = 100*(joined.test_accuracy-joined.test_accuracy_reference)
        joined['brier_delta'] = joined.target_brier-joined.target_brier_reference
        joined.to_csv(OUTPUT/'paired_results.csv',index=False,encoding='utf-8-sig')
        agg = joined.groupby(['dataset','arm'],sort=False).agg(
            seeds=('seed','nunique'), accuracy_mean=('test_accuracy','mean'), accuracy_sd=('test_accuracy','std'),
            delta_pp=('delta_pp','mean'), paired_sd_pp=('delta_pp','std'),
            brier_mean=('target_brier','mean'), brier_delta=('brier_delta','mean'),
            balls_mean=('num_balls','mean'), balls_min=('num_balls','min'), balls_max=('num_balls','max'),
            coverage_mean=('alpha_coverage','mean'), purity_mean=('train_purity_weighted','mean'),
            corrupted_gain=('corrupted_gain','mean'), clean_cost=('clean_cost','mean'))
        agg.to_csv(OUTPUT/'summary.csv',encoding='utf-8-sig')
    print(json.dumps({'prepared':len(rows),'complete':len(complete)}), flush=True)


def main():
    global DEVICE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['prepare','train','all','summarize'],default='all')
    parser.add_argument('--datasets',default=None)
    parser.add_argument('--seeds',default=None)
    parser.add_argument('--arms',default=None)
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    args = parser.parse_args()
    DEVICE = args.device
    OUTPUT = HERE/'gpu_reference_matched' if DEVICE.startswith('cuda') else HERE
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if DEVICE.startswith('cuda'):
        assert torch.cuda.is_available(), 'GPU requested; CPU fallback is not permitted'
        torch.cuda.set_device(0)
    cfg = json.loads((HERE/'config.json').read_text(encoding='utf8'))
    root, source = snapshot(cfg)
    from gblcss.experiment import load_config
    import sklearn, torch_geometric
    torch.set_num_threads(cfg['threads'])
    torch.set_num_interop_threads(1)
    base = load_config(source/'configs/main/gblcss.yaml')
    matrix = arms(cfg)
    if args.arms:
        requested = set(args.arms.split(','))
        assert requested <= {a['name'] for a in matrix}
        matrix = [a for a in matrix if a['name'] in requested]
    datasets = args.datasets.split(',') if args.datasets else cfg['datasets']
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else cfg['seeds']
    assert set(datasets)<=set(cfg['datasets']) and set(seeds)<=set(cfg['seeds'])
    environment = dict(python=sys.version,torch=torch.__version__,sklearn=sklearn.__version__,
                       pyg=torch_geometric.__version__,platform=platform.platform(),device=DEVICE,
                       hip=torch.version.hip,gpu=torch.cuda.get_device_name(0) if DEVICE.startswith('cuda') else None,
                       threads=cfg['threads'],config=cfg,base_config=base)
    env_path = OUTPUT/'environment.json'
    if env_path.exists():
        assert json.loads(env_path.read_text(encoding='utf8')) == environment
    else:
        write_json(env_path,environment)
    if args.mode == 'summarize':
        summarize()
        return
    for key in datasets:
        for seed in seeds:
            if args.mode in ['prepare','all']:
                prepare(key,seed,cfg,matrix,base,root)
            if args.mode in ['train','all']:
                train(key,seed,cfg,matrix,base)
            summarize()


if __name__ == '__main__':
    main()
