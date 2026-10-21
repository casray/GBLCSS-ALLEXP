import csv
import json
import statistics
from collections import defaultdict
import numpy as np
from replay_gate import HERE, NOISYGL, rows, save

folder = HERE / 'matrix'
records = []
for path in folder.glob('*/*/seed*.json'):
    if path.name.endswith(('.config.json', '.diagnostics.json')):
        continue
    row = json.loads(path.read_text(encoding='utf-8'))
    diag = json.loads(path.with_suffix('.diagnostics.json').read_text(encoding='utf-8'))
    repaired = HERE/'fixed_group_repairs'/path.relative_to(folder)
    if repaired.exists():
        row = json.loads(repaired.read_text(encoding='utf-8'))
        diag = json.loads(repaired.with_suffix('.diagnostics.json').read_text(encoding='utf-8'))
    if row['dataset']=='amazonpho' and row['arm']=='default' and int(row['seed'])==1:
        recovery = HERE/'assignment_recovery'
        row = json.loads((recovery/'result.json').read_text(encoding='utf-8'))
        diag = json.loads((recovery/'amazonpho_seed1.diagnostics.json').read_text(encoding='utf-8'))
        row['recovery_source'] = str(recovery)
    assert len(diag['losses']) == 200, path
    row.update({k: v for k, v in diag.items() if k != 'losses'})
    target_path = path.with_suffix('.npz')
    if repaired.exists():
        target_path = repaired.with_suffix('.npz')
    if row['dataset']=='amazonpho' and row['arm']=='default' and int(row['seed'])==1:
        target_path = HERE/'assignment_recovery/amazonpho_seed1.npz'
    if target_path.exists():
        with np.load(target_path) as data:
            clean, noisy = data['clean'], data['noisy']
            membership = data['assignment'][data['train_index']]
            classes = data['targets'].shape[1]
            support = int(row['arm'].split('_')[1]) if row['arm'].startswith('smin_') else 2
            positive = covered = 0
            for group in np.unique(membership):
                positions = np.flatnonzero(membership==group)
                cc = np.bincount(clean[positions], minlength=classes)
                nc = np.bincount(noisy[positions], minlength=classes)
                for pos in positions:
                    votes = cc.copy()
                    votes[clean[pos]] -= 1
                    own = votes[clean[pos]]
                    votes[clean[pos]] = -1
                    positive += int(own > votes.max())
                    others = nc.copy()
                    others[noisy[pos]] -= 1
                    n = len(positions)-1
                    covered += int(n >= support and others.max()*classes > n)
            row['positive_margin_fraction'] = positive/len(clean)
            row['consensus_coverage'] = covered/len(clean)
    records.append(row)
groups = defaultdict(list)
for row in records:
    groups[row['dataset'], row['arm']].append(row)
summary = []
for (dataset, arm), group in groups.items():
    assert sorted(int(r['seed']) for r in group) == list(range(5)), (dataset, arm)
    values = [100*float(r['test']) for r in group]
    entry = dict(dataset=dataset, arm=arm, mean=statistics.mean(values), std=statistics.stdev(values))
    for field in ('target_brier','observed_brier','corrected_error_gain','clean_error_cost','net_brier_gain','num_balls','positive_margin_fraction','consensus_coverage'):
        if all(isinstance(r.get(field), (float, int)) for r in group):
            entry[field] = statistics.mean(r[field] for r in group)
    summary.append(entry)
defaults = {r['dataset']: r for r in summary if r['arm']=='default'}
for entry in summary:
    group = groups[entry['dataset'], entry['arm']]
    base = {int(r['seed']): r for r in groups[entry['dataset'], 'default']}
    deltas = [100*(float(r['test'])-float(base[int(r['seed'])]['test'])) for r in group]
    entry['paired_delta_mean_pp'] = statistics.mean(deltas)
    entry['paired_delta_std_pp'] = statistics.stdev(deltas)
save(folder/'summary.json', summary)
save(folder/'final_per_seed.json', records)
keys = sorted({k for r in summary for k in r})
with (folder/'summary.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(summary)
reference = rows(NOISYGL/'runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv')
comparison = []
for row in records:
    if row['arm'] != 'default':
        continue
    ref = next(r for r in reference if r['dataset']==row['dataset'] and r['method'].endswith('_loo') and int(r['seed'])==int(row['seed']))
    comparison.append(dict(dataset=row['dataset'],seed=int(row['seed']),delta_pp=100*(float(row['test'])-float(ref['test'])),
        split_matches=row['split_hash']==ref['split_hash'],noise_matches=row['noise_hash']==ref['noise_hash'],assignment_matches=row['ball_assignment_hash']==ref['ball_assignment_hash']))
save(folder/'default_comparison.json', comparison)
print(json.dumps(dict(rows=len(records), configurations=len(summary), exact_default_scores=sum(abs(r['delta_pp'])<1e-5 for r in comparison), exact_default_assignments=sum(r['assignment_matches'] for r in comparison)), indent=2))
for r in summary:
    if r['arm']=='default':
        print(f"{r['dataset']}: {r['mean']:.2f} +/- {r['std']:.2f}")
