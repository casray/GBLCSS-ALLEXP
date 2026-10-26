"""Independent saved-artifact audit, run only after the complete grid."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    config = read(ROOT/'config.json')
    if read(ROOT/'progress.json')['status'] != 'completed' or read(ROOT/'audit.json')['status'] != 'passed':
        raise RuntimeError('Complete the grid and checkpoint audit first')
    rows = list(csv.DictReader((ROOT/'results.csv').open(encoding='utf-8-sig')))
    summaries = list(csv.DictReader((ROOT/'summary.csv').open(encoding='utf-8-sig')))
    paired = list(csv.DictReader((ROOT/'paired_deltas.csv').open(encoding='utf-8-sig')))
    if len(rows) != config['expected_fits']:
        raise RuntimeError('Incomplete results table')
    lookup = {(r['route'],r['rule'],r['dataset'],r['condition'],int(r['seed'])):r for r in rows}
    max_summary_error = max_pair_error = 0.
    frozen_encoder_checks = 0
    weight_rows = []
    for r in rows:
        folder = Path(r['folder'])
        state = read(folder/'result.json')
        history = [json.loads(line) for line in (folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = config['head_epochs'] if r['route']=='R2' else config['epochs']
        if len(history) != expected or [h['epoch'] for h in history] != list(range(1,expected+1)):
            raise RuntimeError('Invalid training trajectory: '+str(folder))
        counts = [h['validation_correct'] for h in history]
        selected = int(np.argmax(counts))+1
        if selected != int(r['best_epoch']) or max(counts) != state['valid_correct']:
            raise RuntimeError('Not the first strict validation optimum')
        if abs(np.mean([h['mean_weight'] for h in history])-float(r['mean_weight'])) > 1e-12:
            raise RuntimeError('Mean weight mismatch')
        for p,key in [('student.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            if hash_file(folder/p) != state[key]:
                raise RuntimeError('Saved artifact hash mismatch')
        if r['route']=='R2':
            enc = torch.load(state['encoder_source'], weights_only=True)
            stu = torch.load(folder/'student.pt', weights_only=True)
            for k,v in stu.items():
                if not k.startswith('layers.1.') and not torch.equal(v,enc['backbone.'+k]):
                    raise RuntimeError('Frozen encoder/BN differs in selected checkpoint')
            frozen_encoder_checks += 1
        if r['rule'].startswith('W_'):
            cache = torch.load(ROOT/'inputs'/r['dataset']/f'seed{r["seed"]}'/(r['condition']+'.pt'), weights_only=True)
            assets = torch.load(ROOT/'inputs'/r['dataset']/f'seed{r["seed"]}'/'assets.pt', weights_only=True)
            p = torch.load(folder/'probabilities.pt', weights_only=True)[assets['train']].double()
            target = cache['target'].double()
            if r['rule']=='W_MASS':
                own = target.gather(1,cache['observed'][:,None])
                target = ((1-own)/(target.shape[1]-1)).expand_as(target).clone()
                target.scatter_(1,cache['observed'][:,None],own)
            weight = (p*target).sum(1).pow(config['gce_q'])
            quantiles = np.quantile(weight.numpy(),[0,.25,.5,.75,1])
            weight_rows.append(dict(dataset=r['dataset'],condition=r['condition'],seed=int(r['seed']),route=r['route'],rule=r['rule'],
                                    checkpoint_weight_mean=float(weight.mean()), uniform_prediction_scale=float(target.shape[1]**(-config['gce_q'])),
                                    **{label:float(v) for label,v in zip(['min','q25','median','q75','max'],quantiles)}))
    for r in summaries:
        values = [float(lookup[(r['route'],r['rule'],r['dataset'],r['condition'],s)]['test_percent']) for s in config['seeds']]
        error = max(abs(np.mean(values)-float(r['test_mean'])),abs(np.std(values,ddof=1)-float(r['test_std'])))
        max_summary_error=max(max_summary_error,error)
    for r in paired:
        values = [float(lookup[(r['route'],r['method'],r['dataset'],r['condition'],s)]['test_percent'])-
                  float(lookup[(r['route'],r['baseline'],r['dataset'],r['condition'],s)]['test_percent']) for s in config['seeds']]
        error = max(abs(np.mean(values)-float(r['mean_difference'])),abs(np.std(values,ddof=1)-float(r['paired_std'])))
        max_pair_error=max(max_pair_error,error)
        if sum(v>0 for v in values) != int(r['positive_seeds']):
            raise RuntimeError('Wrong seed win count')
    if max(max_summary_error,max_pair_error)>1e-10:
        raise RuntimeError('Summary arithmetic mismatch')
    frozen=read(ROOT/'frozen_inputs.json')['sha256']
    for p,d in frozen.items():
        if hash_file(p)!=d:
            raise RuntimeError('Historical/frozen input changed')
    recovery = read(ROOT/'numerical_recovery_v2.json')
    for p,d in recovery['sha256'].items():
        if hash_file(p)!=d:
            raise RuntimeError('Numerical recovery source changed')
    stable_rows = [r for r in rows if r['rule'] in ('GCE','SOFT_GCE')]
    for r in stable_rows:
        artifact = read(Path(r['folder'])/'result.json')
        if artifact.get('numerical_implementation') != 'v2_exp_q_log_p':
            raise RuntimeError('Mixed GCE numerical implementations in formal table')
    macro_rows = []
    for route in config['routes']:
        for rule in config['rules']:
            for scope in ('noisy','clean'):
                values = [np.mean([float(r['test_percent']) for r in rows if r['route']==route and r['rule']==rule
                                   and int(r['seed'])==s and ((r['condition']!='clean') if scope=='noisy' else (r['condition']=='clean'))])
                          for s in config['seeds']]
                macro_rows.append(dict(route=route,rule=rule,scope=scope,mean=float(np.mean(values)),
                                       seed_macro_std=float(np.std(values,ddof=1)),
                                       **{f'seed{s}':float(v) for s,v in zip(config['seeds'],values)}))
    with (ROOT/'macro_summary_verified.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(macro_rows[0])); writer.writeheader(); writer.writerows(macro_rows)
    decisions = read(ROOT/'decision.json')['routes']
    decision_checks = 0
    for route in config['routes']:
        component_passes = []
        for baseline, margin in [('LCSS',1.),('GCE',1.),('W_MASS',0.)]:
            matrix = np.array([[float(lookup[(route,'W_LCSS',d,c,s)]['test_percent'])-
                                float(lookup[(route,baseline,d,c,s)]['test_percent']) for s in config['seeds']]
                               for d in config['datasets'] for c in config['conditions'] if c!='clean'])
            average = float(matrix.mean())
            positive_cells = int((matrix.mean(1)>0).sum())
            positive_datasets = int((matrix.reshape(3,3,3).mean((1,2))>0).sum())
            positive_seeds = int((matrix.mean(0)>0).sum())
            expected = (average>=margin if margin else average>0) and positive_cells>=6 and positive_datasets>=2 and positive_seeds>=2
            saved = decisions[route]['comparisons'][baseline]
            if abs(average-saved['mean_difference'])>1e-10 or (positive_cells,positive_datasets,positive_seeds,expected)!=(saved['positive_cells'],saved['positive_datasets'],saved['positive_seed_macros'],saved['passed']):
                raise RuntimeError('Decision component mismatch')
            component_passes.append(expected)
            decision_checks += 1
        clean_deltas = [np.mean([float(lookup[(route,'W_LCSS',d,'clean',s)]['test_percent'])-
                                  float(lookup[(route,'CE',d,'clean',s)]['test_percent']) for s in config['seeds']]) for d in config['datasets']]
        safety = all(v>=-1 for v in clean_deltas)
        if safety != decisions[route]['clean_safety_passed'] or (safety and all(component_passes)) != decisions[route]['full_gate_passed']:
            raise RuntimeError('Full decision mismatch')
    with (ROOT/'checkpoint_weight_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(weight_rows[0]))
        writer.writeheader(); writer.writerows(weight_rows)
    output=dict(status='passed', trajectories=len(rows), first_strict_validation_optima=len(rows),
                frozen_encoder_checkpoint_checks=frozen_encoder_checks, cell_summaries=len(summaries), paired_rows=len(paired),
                maximum_summary_error=max_summary_error, maximum_pair_error=max_pair_error,
                frozen_sources_verified=len(frozen), audit_source_sha256=hash_file(__file__),
                stable_gce_runs_checked=len(stable_rows), numerical_recovery_sources_verified=len(recovery['sha256']),
                independent_gate_components=decision_checks,
                weight_quantile_rows=len(weight_rows), weights_read_from_selected_evaluation_checkpoint_not_training_dropout=True)
    (ROOT/'independent_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
