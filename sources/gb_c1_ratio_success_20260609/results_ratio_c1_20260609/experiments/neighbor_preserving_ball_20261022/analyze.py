"""Post-grid audit; test labels are never consulted by the fitting program."""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from run import ROOT, OLD, FUSION, CONFIG, read, write, sha, csv_write, initialize, verify
from run import load_dataset, raw_data, MLP, state_hash, g


def logs(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def smoke_audit():
    verify()
    checks = []
    for route in CONFIG['routes']:
        for arm in CONFIG['arms']:
            path = ROOT/'smoke'/'cora'/'uniform0.4'/(route+'_'+arm)
            result = read(path/'result.json')
            assert result['status'] == 'ok' and result['epochs'] == 3
            if arm in ('CE','OLD64'):
                old = FUSION/'runs'/'cora'/'seed0'/'uniform0.4'/(route+'_'+('CE' if arm=='CE' else 'LCSS'))
                actual, expected = logs(path/'epochs.jsonl'), logs(old/'epochs.jsonl')[:3]
                assert [r['validation_correct'] for r in actual] == [r['validation_correct'] for r in expected]
                delta = max(abs(a['loss']-b['loss']) for a,b in zip(actual,expected))
                if delta > 1e-7:
                    raise RuntimeError('Historical first-three-epoch loss mismatch')
                checks.append(dict(route=route, arm=arm, loss_difference=delta))
    write(ROOT/'smoke_audit.json', dict(passed=True, fits=16, historical_bridges=checks))
    print(json.dumps(read(ROOT/'smoke_audit.json')), flush=True)


def macro_f1(y, pred, classes):
    cm = np.bincount(y*classes+pred, minlength=classes*classes).reshape(classes, classes)
    denominator = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*cm.diagonal(), denominator, out=np.zeros(classes), where=denominator>0).mean())


def full_audit():
    initialize()
    verify()
    progress = read(ROOT/'progress.json')
    if progress['status'] != 'completed' or progress['completed'] != CONFIG['expected_cells']:
        raise RuntimeError('Complete the entire fixed grid before evaluating test predictions')
    rows, target_rows, node_rows, checks = [], [], [], []
    for name in CONFIG['datasets']:
        data = load_dataset(name)
        raw = raw_data(name)
        geo = torch.load(ROOT/'inputs'/name/'geometry.pt', weights_only=True)
        train, valid, test = (geo[k].numpy() for k in ('train','valid','test'))
        y = raw.y.numpy().astype(np.int64)
        tree = torch.load(ROOT/'inputs'/name/'tree.pt', weights_only=True)
        recovered, value = g.select_cut(tree['nodes'], tree['costs'].numpy())
        assert recovered == tree['np_leaves'] and abs(value-tree['cost']) < 1e-10
        for leaf in tree['fine_leaves']:
            assert len(tree['nodes'][leaf]['members']) >= CONFIG['min_nodes']
            assert tree['nodes'][leaf]['train_count'] >= CONFIG['min_train']
        for condition in CONFIG['conditions']:
            cache = torch.load(ROOT/'inputs'/name/(condition+'.pt'), weights_only=True)
            observed = cache['observed'].numpy()
            corrupted = observed != y[train]
            onehot = np.eye(data['classes'])[y[train]]
            observed_onehot = np.eye(data['classes'])[observed]
            for arm in CONFIG['arms']:
                target = cache['targets'][arm].numpy().astype(float)
                brier = ((target-onehot)**2).sum(1)
                baseline_brier = ((observed_onehot-onehot)**2).sum(1)
                maximum = target.max(1)
                target_correct = target.argmax(1) == y[train]
                for stratum, mask in [('all',np.ones(len(train),bool)),('clean',~corrupted),('corrupted',corrupted)]:
                    n = int(mask.sum())
                    target_rows.append(dict(dataset=name,condition=condition,arm=arm,stratum=stratum,n=n,
                        brier=float(brier[mask].mean()) if n else None,
                        brier_gain=float((baseline_brier[mask]-brier[mask]).mean()) if n else None,
                        argmax_accuracy=float(target_correct[mask].mean()) if n else None,
                        damaged_fraction=float((brier[mask]>baseline_brier[mask]+1e-7).mean()) if n else None,
                        tied_fraction=float(((target==maximum[:,None]).sum(1)[mask]>1).mean()) if n else None,
                        support_mean=float(cache['supports'][arm][mask].mean()) if n else None,
                        alpha_mean=float(cache['alphas'][arm][mask].mean()) if n else None))
                for route in CONFIG['routes']:
                    path = ROOT/'runs'/name/condition/(route+'_'+arm)
                    result = read(path/'result.json')
                    assert result['status'] == 'ok'
                    source = path if not result['reused'] else __import__('pathlib').Path(result['source_folder'])
                    assert sha(source/'student.pt') == result['checkpoint_sha256']
                    assert sha(source/'probabilities.pt') == result['probabilities_sha256']
                    probabilities = torch.load(source/'probabilities.pt', weights_only=True)
                    state = torch.load(source/'student.pt', weights_only=True)
                    model = MLP(data['x'].shape[1], CONFIG['hidden'], data['classes'], 2, CONFIG['dropout']).cuda().eval()
                    model.load_state_dict(state)
                    with torch.no_grad():
                        reloaded = model(data['x']).softmax(1).cpu()
                    delta = float((reloaded-probabilities).abs().max())
                    if delta != 0:
                        raise RuntimeError('Checkpoint predictions not reproduced exactly')
                    log = logs(source/'epochs.jsonl')
                    assert len(log) == CONFIG['epochs' if route=='R0' else 'head_epochs']
                    best = max(r['validation_correct'] for r in log)
                    first = next(r['epoch'] for r in log if r['validation_correct']==best)
                    pred = probabilities.numpy().argmax(1)
                    assert first == result['best_epoch'] and best == result['valid_correct']
                    assert int((pred[valid]==y[valid]).sum()) == best
                    anchor = read(FUSION/'runs'/name/'seed0'/condition/(route+'_CE')/'result.json')
                    assert result['initial_hash']==anchor['initial_hash']
                    assert result['initial_rng_hash']==anchor['initial_rng_hash']
                    if route=='R2':
                        original = torch.load(OLD/'pretraining'/name/'model.pt', map_location='cpu', weights_only=True)
                        for key,tensor in state.items():
                            if not key.startswith('layers.1.'):
                                assert torch.equal(tensor, original['backbone.'+key])
                    if not result['reused']:
                        assert result['target_hash'] == state_hash(dict(target=cache['targets'][arm]))
                    accuracy = 100*float((pred[test]==y[test]).mean())
                    rows.append(dict(dataset=name,condition=condition,seed=0,route=route,arm=arm,
                                     accuracy=accuracy,macro_f1=100*macro_f1(y[test],pred[test],data['classes']),
                                     validation=100*best/len(valid),best_epoch=first,seconds=result['seconds'],
                                     reused=result['reused'],source=str(source)))
                    checks.append(dict(dataset=name,condition=condition,route=route,arm=arm,
                                       checkpoint_difference=delta,selection_verified=True,initialization_verified=True))
        # Offline class mixing is diagnostic only, computed after the fixed full grid.
        for arm, assignment in geo['partitions'].items():
            a = assignment.numpy()[train]
            for ball in np.unique(a):
                selected = a==ball
                node_rows.append(dict(dataset=name,arm=arm,ball=int(ball),train_count=int(selected.sum()),
                    clean_classes=int(len(np.unique(y[train][selected])))))
    csv_write(ROOT/'results.csv', rows)
    csv_write(ROOT/'target_diagnostics.csv', target_rows)
    csv_write(ROOT/'ball_diagnostics.csv', node_rows)
    lookup = {(r['dataset'],r['condition'],r['route'],r['arm']):r['accuracy'] for r in rows}
    summaries, paired, gates = [], [], {}
    for route in CONFIG['routes']:
        for arm in CONFIG['arms']:
            for domain in ['noisy','clean']:
                selected = [r for r in rows if r['route']==route and r['arm']==arm and (r['condition']=='clean')==(domain=='clean')]
                summaries.append(dict(route=route,arm=arm,domain=domain,cells=len(selected),seeds=1,seed_std=None,
                                      accuracy=float(np.mean([r['accuracy'] for r in selected])),
                                      macro_f1=float(np.mean([r['macro_f1'] for r in selected]))))
        differences = {}
        for reference in ['CE','OLD64','TREE64','FINE','KM','KNN32','KNNMATCH']:
            noisy, clean = [], []
            for name in CONFIG['datasets']:
                for condition in CONFIG['conditions']:
                    delta = lookup[name,condition,route,'NP']-lookup[name,condition,route,reference]
                    paired.append(dict(dataset=name,condition=condition,route=route,reference=reference,np_delta_pp=delta))
                    (clean if condition=='clean' else noisy).append(delta)
            differences[reference] = dict(noisy_macro=float(np.mean(noisy)),noisy_wins=sum(v>1e-10 for v in noisy),
                noisy_nonnegative=sum(v>=-1e-10 for v in noisy),noisy_min=float(min(noisy)),clean_min=float(min(clean)))
        clean_safe = differences['CE']['clean_min']>=-1-1e-10
        cf = lookup['corafull','clean',route,'NP']-lookup['corafull','clean',route,'OLD64']
        repair_items = dict(clean_safe=clean_safe,corafull_clean_gain=cf>=1-1e-10,
            noisy_macro=differences['OLD64']['noisy_macro']>=-1e-10,
            nonnegative_cells=differences['OLD64']['noisy_nonnegative']>=6,
            worst_cell=differences['OLD64']['noisy_min']>=-1-1e-10)
        incremental_items = {'clean_safe':clean_safe}
        for ref in ['TREE64','FINE','KM']:
            incremental_items[ref+'_macro'] = differences[ref]['noisy_macro']>=.5-1e-10
            incremental_items[ref+'_wins'] = differences[ref]['noisy_wins']>=6
        gates[route] = dict(repair_pass=all(repair_items.values()),incremental_pass=all(incremental_items.values()),
                           repair_items=repair_items,incremental_items=incremental_items,
                           corafull_clean_delta_old64=cf,differences=differences)
    csv_write(ROOT/'summary.csv', summaries)
    csv_write(ROOT/'paired_differences.csv', paired)
    write(ROOT/'gates.json', dict(routes=gates,seed_count=1,confirmatory=False,
          decision='candidate_for_separate_confirmation' if gates['R0']['repair_pass'] or gates['R0']['incremental_pass'] else 'stop_seed0'))
    verify()
    write(ROOT/'audit.json', dict(passed=True,total_cells=len(rows),new=sum(not r['reused'] for r in rows),
          reused=sum(r['reused'] for r in rows),checks=checks,analysis_sha256=sha(__file__),
          frozen_files=len(read(ROOT/'frozen_inputs.json')['sha256'])))
    print(json.dumps(dict(summary=summaries,gates=read(ROOT/'gates.json')),indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['smoke','full'])
    args = parser.parse_args()
    smoke_audit() if args.phase=='smoke' else full_audit()
