"""Post-grid metrics, paired decisions and checkpoint verification; no tuning."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run import CONFIG, OLD, load_dataset, raw_data, read, write, sha, csv_write, initialize, verify
from gblcss.models import MLP


def mean(values):
    return float(np.mean(values))


def std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def main():
    initialize()
    verify()
    files = sorted((ROOT/'runs').glob('*/*/*/*/result.json'))
    if len(files) != CONFIG['expected_fits'] or read(ROOT/'progress.json')['status'] != 'completed':
        raise RuntimeError('Evaluate tests only after the complete frozen grid')
    old_rows = list(__import__('csv').DictReader((OLD/'results.csv').open(encoding='utf-8-sig')))
    anchors = {(r['dataset'], r['condition'], r['route'], r['rule']):r for r in old_rows
               if r['phase'] == 'main' and r['legacy'] == 'False'}
    records, checks, initial = [], [], defaultdict(set)
    for number, file in enumerate(files, 1):
        row = read(file)
        if row['status'] != 'ok':
            raise RuntimeError('Incomplete run: '+str(file))
        folder = file.parent
        data = load_dataset(row['dataset'])
        assets = torch.load(ROOT/'inputs'/row['dataset']/f'seed{row["seed"]}'/'assets.pt', weights_only=True)
        truth = raw_data(row['dataset']).y.cpu().long()
        p = torch.load(folder/'probabilities.pt', weights_only=True)
        model = MLP(data['x'].shape[1], CONFIG['hidden'], data['classes'], 2, CONFIG['dropout']).cuda().eval()
        checkpoint = torch.load(folder/'student.pt', weights_only=True)
        model.load_state_dict(checkpoint)
        with torch.no_grad():
            reloaded = model(data['x']).softmax(1).cpu()
        difference = float((p-reloaded).abs().max())
        if difference != 0 or sha(folder/'student.pt') != row['checkpoint_sha256'] or sha(folder/'probabilities.pt') != row['probabilities_sha256']:
            raise RuntimeError('Checkpoint or prediction verification failed: '+str(folder))
        if not torch.isfinite(p).all() or not torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5):
            raise RuntimeError('Invalid saved probability')
        valid, test = assets['valid'], assets['test']
        pred = p.argmax(1)
        correct = int((pred[valid] == truth[valid]).sum())
        if correct != row['valid_correct']:
            raise RuntimeError('Validation metric mismatch')
        test_correct = int((pred[test] == truth[test]).sum())
        accuracy = test_correct/len(test)*100
        # Independent NumPy argmax/count, separate from the PyTorch metric.
        np_accuracy = float(np.mean(np.argmax(p[test].numpy(), axis=1) == truth[test].numpy())*100)
        if abs(accuracy-np_accuracy) > 1e-10:
            raise RuntimeError('Independent accuracy mismatch')
        one = torch.nn.functional.one_hot(truth[test], data['classes']).float()
        record = {k:row[k] for k in ('dataset', 'seed', 'condition', 'route', 'rule', 'best_epoch', 'seconds', 'mean_weight')}
        record.update(test_percent=accuracy, validation_percent=100*correct/len(valid), test_correct=test_correct,
                      test_total=len(test), macro_f1_percent=float(f1_score(truth[test].numpy(), pred[test].numpy(), average='macro', zero_division=0)*100),
                      test_brier=float((p[test]-one).square().sum(1).mean()),
                      test_nll=float(-p[test].gather(1, truth[test, None]).clamp_min(1e-12).log().mean()),
                      folder=str(folder))
        anchor_diff = None
        if row['seed'] == 0 and row['rule'] in ('CE', 'LCSS'):
            anchor = anchors[(row['dataset'], row['condition'], row['route'], 'T0' if row['rule']=='CE' else 'T1')]
            anchor_p = torch.load(Path(anchor['folder'])/'probabilities.pt', weights_only=True)
            anchor_diff = float((p-anchor_p).abs().max())
            if anchor_diff != 0 or row['best_epoch'] != int(anchor['best_epoch']):
                raise RuntimeError('Historical seed0 anchor mismatch: '+str(folder))
        initial[(row['dataset'], row['seed'], row['condition'], row['route'])].add((row['initial_hash'], row['initial_rng_hash']))
        records.append(record)
        checks.append(dict(dataset=row['dataset'], seed=row['seed'], condition=row['condition'], route=row['route'],
                           rule=row['rule'], reload_max_difference=difference, anchor_max_difference=anchor_diff))
        if number % 50 == 0:
            print(f'AUDIT {number}/{len(files)}', flush=True)
    if any(len(v) != 1 for v in initial.values()):
        raise RuntimeError('Loss arms did not share initialization/RNG')
    csv_write(ROOT/'results.csv', records)
    groups = defaultdict(list)
    for row in records:
        groups[(row['route'], row['rule'], row['dataset'], row['condition'])].append(row)
    summary = []
    for (route, rule, dataset, condition), values in sorted(groups.items()):
        summary.append(dict(route=route, rule=rule, dataset=dataset, condition=condition, seeds=len(values),
                            test_mean=mean([r['test_percent'] for r in values]),
                            test_std=std([r['test_percent'] for r in values]),
                            macro_f1_mean=mean([r['macro_f1_percent'] for r in values]),
                            validation_mean=mean([r['validation_percent'] for r in values])))
    csv_write(ROOT/'summary.csv', summary)
    lookup = {(r['route'], r['dataset'], r['condition'], r['seed'], r['rule']):r for r in records}
    paired = []
    for route in CONFIG['routes']:
        for method in ('W_LCSS', 'SOFT_GCE', 'W_MASS'):
            for baseline in ('CE', 'LCSS', 'GCE', 'W_MASS', 'MASS'):
                if method == baseline:
                    continue
                for dataset in CONFIG['datasets']:
                    for condition in CONFIG['conditions']:
                        diffs = [lookup[(route, dataset, condition, s, method)]['test_percent']-
                                 lookup[(route, dataset, condition, s, baseline)]['test_percent'] for s in CONFIG['seeds']]
                        paired.append(dict(route=route, method=method, baseline=baseline, dataset=dataset, condition=condition,
                                           mean_difference=mean(diffs), paired_std=std(diffs), positive_seeds=sum(v>0 for v in diffs),
                                           **{f'seed{s}_difference':diff for s,diff in zip(CONFIG['seeds'], diffs)}))
    csv_write(ROOT/'paired_deltas.csv', paired)
    macros = []
    for route in CONFIG['routes']:
        for rule in CONFIG['rules']:
            for scope in ('noisy', 'clean'):
                per_seed = []
                for seed in CONFIG['seeds']:
                    values = [r['test_percent'] for r in records if r['route']==route and r['rule']==rule and r['seed']==seed
                              and ((r['condition']!='clean') if scope=='noisy' else (r['condition']=='clean'))]
                    per_seed.append(mean(values))
                macros.append(dict(route=route, rule=rule, scope=scope, mean=mean(per_seed), paired_seed_std=std(per_seed),
                                   **{f'seed{s}':v for s,v in zip(CONFIG['seeds'],per_seed)}))
    csv_write(ROOT/'macro_summary.csv', macros)
    decisions = {}
    for route in CONFIG['routes']:
        comparisons = {}
        for baseline, margin in [('LCSS',1.), ('GCE',1.), ('W_MASS',0.)]:
            cells = [r for r in paired if r['route']==route and r['method']=='W_LCSS' and r['baseline']==baseline and r['condition']!='clean']
            delta = mean([r['mean_difference'] for r in cells])
            positive_cells = sum(r['mean_difference']>0 for r in cells)
            positive_datasets = sum(mean([r['mean_difference'] for r in cells if r['dataset']==d])>0 for d in CONFIG['datasets'])
            seed_macro = [mean([r[f'seed{s}_difference'] for r in cells]) for s in CONFIG['seeds']]
            comparisons[baseline] = dict(mean_difference=delta, paired_seed_std=std(seed_macro), positive_cells=positive_cells,
                                         positive_datasets=positive_datasets, positive_seed_macros=sum(x>0 for x in seed_macro),
                                         passed=(delta>=margin if margin else delta>0) and positive_cells>=6 and positive_datasets>=2 and sum(x>0 for x in seed_macro)>=2)
        clean = [r for r in paired if r['route']==route and r['method']=='W_LCSS' and r['baseline']=='CE' and r['condition']=='clean']
        safety = all(r['mean_difference']>=-1 for r in clean)
        passed = safety and all(r['passed'] for r in comparisons.values())
        decisions[route] = dict(comparisons=comparisons, clean_safety_passed=safety,
                               clean_differences={r['dataset']:r['mean_difference'] for r in clean},
                               full_gate_passed=passed, decision='eligible for separate confirmation, no automatic combination' if passed else 'stop current weighted fusion specification')
    write(ROOT/'decision.json', dict(primary='R0', secondary='R2', routes=decisions,
          soft_membership_started=False, adaptive_generation_started=False,
          ssl_independence='fixed historical GRACE; protocol/student seeds0/1/2; exploratory reused datasets'))
    verify()
    write(ROOT/'audit.json', dict(status='passed', fits=len(records), checkpoint_reloads=len(checks),
          seed0_exact_probability_and_epoch_anchors=sum(r['anchor_max_difference'] is not None for r in checks),
          shared_initial_state_groups=len(initial), all_frozen_hashes_unchanged=True, checks=checks))
    print(json.dumps(dict(macros=macros, decisions=decisions), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
