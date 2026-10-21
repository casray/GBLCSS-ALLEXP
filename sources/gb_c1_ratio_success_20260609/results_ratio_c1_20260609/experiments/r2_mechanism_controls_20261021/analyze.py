"""Post-grid scoring and independent artifact checks; not imported by training."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'representation_supervision_20261021'
sys.path.insert(0, str(PREVIOUS))
from core import state_hash
from protocol import read, write, sha, raw_data, load_dataset
from gblcss.models import MLP


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save_csv(name, rows):
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_targets(z, counts):
    counts = np.asarray(counts, dtype=np.float64)
    n, classes = counts.shape
    support = counts.sum(1)
    r = counts/np.maximum(1, support)[:, None]
    maximum = counts.max(1)
    a = np.clip((maximum/np.maximum(1, support)-1/classes)/(1-1/classes), 0, 1)
    a[support < 2] = 0
    strength = np.where((counts == maximum[:, None]).sum(1)>1, 0., .5)
    q = (1-strength[:, None])*r+strength[:, None]*np.eye(classes)[counts.argmax(1)]
    return (1-a[:, None])*np.eye(classes)[z]+a[:, None]*q


def check_targets(config):
    records, maximum_diff = [], 0.
    for name in config['datasets']:
        assets = torch.load(PREVIOUS/'inputs'/name/'assets.pt', weights_only=True)
        groups = torch.load(ROOT/'inputs'/name/'groups.pt', weights_only=True)
        assignments = {k:v.numpy() for k,v in groups['assignment'].items()}
        train = assets['train'].numpy()
        support = groups['knn_support'].numpy()
        neighbors = groups['knn'].numpy()
        classes = assets['classes']
        require(np.array_equal(assignments['GB'], assets['assignment'].numpy()), 'Historical GB changed')
        require(np.array_equal(np.bincount(assignments['GB']), np.bincount(assignments['RANDOM'])), 'Random sizes changed')
        require(np.array_equal(np.bincount(assignments['GB'][train]), np.bincount(assignments['RANDOM'][train])), 'Random train counts changed')
        require(len(np.unique(assignments['KM']))==len(np.unique(assignments['GB'])), 'KM region count differs')
        counts = np.bincount(assignments['GB'][train])
        require(np.array_equal(support, counts[assignments['GB'][train]]-1), 'KNN support differs')
        for n in range(len(train)):
            used = neighbors[n, :support[n]]
            require(n not in used, 'Self vote in KNN')
            require(len(np.unique(used)) == support[n], 'Duplicate KNN voter')
        order = torch.load(ROOT/'encoders'/name/'SHUFFLED_H/permutation.pt', weights_only=True).numpy()
        require(np.array_equal(np.sort(order), np.arange(len(assets['h']))), 'Bad H permutation')
        require(not np.array_equal(order, np.arange(len(order))), 'H not shuffled')
        for condition in config['conditions']:
            source = torch.load(PREVIOUS/'inputs'/name/(condition+'.pt'), weights_only=True)
            targets = torch.load(ROOT/'inputs'/name/(condition+'.pt'), weights_only=True)
            z = source['observed'].numpy()
            for rule in ('GB','KM','RANDOM','KNN'):
                if rule=='KNN':
                    votes = np.stack([np.bincount(z[neighbors[n, :support[n]]], minlength=classes) for n in range(len(z))])
                    local_support = support
                else:
                    a = assignments[rule][train]
                    votes_all = np.zeros((int(a.max())+1, classes))
                    np.add.at(votes_all, (a, z), 1)
                    votes = votes_all[a]-np.eye(classes)[z]
                    local_support = votes.sum(1)
                expected = independent_targets(z, votes)
                actual = targets[rule].numpy()
                diff = float(np.max(np.abs(expected-actual)))
                maximum_diff = max(maximum_diff, diff)
                require(diff < 4e-7, 'Independent target mismatch')
                require(np.allclose(actual.sum(1), 1, atol=1e-6), 'Target invalid')
                records.append(dict(dataset=name, condition=condition, rule=rule,
                                    regions=None if rule=='KNN' else len(np.unique(assignments[rule])),
                                    mean_votes=float(local_support.mean()), min_votes=int(local_support.min()),
                                    max_votes=int(local_support.max()), voting_coverage=float(np.mean(local_support>=2)),
                                    changed_target_fraction=float(np.mean(actual[np.arange(len(z)), z]<1-1e-7)),
                                    mean_off_observed_mass=float(np.mean(1-actual[np.arange(len(z)), z])),
                                    target_reference_max_difference=diff))
            mass = targets['MASS'].numpy()
            gb = targets['GB'].numpy()
            require(np.allclose(mass[np.arange(len(z)), z], gb[np.arange(len(z)), z], atol=1e-7), 'MASS own class differs')
            off_mask = np.ones_like(mass, dtype=bool)
            off_mask[np.arange(len(z)), z] = False
            off = mass[off_mask].reshape(len(z), -1)
            require(np.all(np.max(off, 1)-np.min(off, 1)<1e-7), 'MASS off-label not uniform')
    save_csv('target_diagnostics.csv', records)
    return dict(independent_targets_verified=len(records), max_target_difference=maximum_diff,
                exact_knn_support_and_self_exclusion=True, random_total_and_train_sizes_preserved=True,
                h_rows_permuted_bijectively=True, mass_control_exact_observed_probability=True)


def score_and_check(config):
    rows, artifacts = [], {}
    anchor_max = 0.
    for name in config['datasets']:
        data = load_dataset(name)
        assets = torch.load(PREVIOUS/'inputs'/name/'assets.pt', weights_only=True)
        test, valid = assets['test'].numpy(), assets['valid'].numpy()
        clean = raw_data(name).y.numpy()
        first_initial, first_classifier = set(), set()
        for condition in config['conditions']:
            cells = [('REAL', r) for r in config['real_rules']]
            cells += [(route, r) for route in config['control_routes'] for r in config['control_rules']]
            for route, rule in cells:
                folder = ROOT/'runs'/name/condition/(route+'_'+rule)
                result = read(folder/'result.json')
                require(result['status']=='ok', 'Incomplete fit')
                state = torch.load(folder/'student.pt', weights_only=True)
                require(sha(folder/'student.pt')==result['student_sha256'], 'Student artifact changed')
                require(sha(folder/'probabilities.pt')==result['probability_sha256'], 'Probabilities changed')
                probability = torch.load(folder/'probabilities.pt', weights_only=True)
                model = MLP(data['x'].shape[1], 64, data['classes'], 2, .5).cuda().eval()
                model.load_state_dict(state, strict=True)
                with torch.no_grad():
                    recalculated = model(data['x']).softmax(1).cpu()
                require(torch.equal(recalculated, probability), 'Reloaded X-only predictions differ')
                source = torch.load(result['encoder_source'], weights_only=True)
                for key, value in state.items():
                    if not key.startswith('layers.1.'):
                        require(torch.equal(value, source['backbone.'+key]), 'Frozen encoder/BN changed')
                initial_classifier = {k.removeprefix('backbone.layers.1.'):v for k,v in source.items() if k.startswith('backbone.layers.1.')}
                require(state_hash(initial_classifier)==result['classifier_hash'], 'Classifier initialization metadata differs')
                first_initial.add(result['initial_hash'])
                first_classifier.add(result['classifier_hash'])
                trace = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
                require(len(trace)==100, 'Head budget differs')
                counts = [r['valid_correct'] for r in trace]
                require(int(np.argmax(counts))+1==result['best_epoch'], 'Not first strict validation maximum')
                p = probability.numpy().astype(np.float64)
                require(np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5), 'Invalid probabilities')
                predicted = p.argmax(1)
                val_acc = float(np.mean(predicted[valid]==clean[valid]))
                require(abs(val_acc-result['validation'])<1e-12, 'Validation mismatch')
                test_acc = float(np.mean(predicted[test]==clean[test]))
                nll = float(-np.log(np.maximum(p[test, clean[test]], 1e-30)).mean())
                brier = float((np.square(p[test]).sum(1)-2*p[test, clean[test]]+1).mean())
                if route=='REAL' and rule in ('CE','GB'):
                    old = PREVIOUS/'main'/name/condition/('R2_T0' if rule=='CE' else 'R2_T1')
                    old_p = torch.load(old/'probabilities.pt', weights_only=True)
                    diff = float((old_p-probability).abs().max())
                    anchor_max = max(anchor_max, diff)
                    require(diff==0, 'Anchor predictions changed')
                    require(read(old/'result.json')['best_epoch']==result['best_epoch'], 'Anchor epoch changed')
                rows.append(dict(dataset=name, condition=condition, route=route, rule=rule, seed=0,
                                 test_percent=100*test_acc, validation_percent=100*val_acc,
                                 final_validation_percent=100*result['final_validation'], nll=nll, brier=brier,
                                 best_epoch=result['best_epoch'], head_seconds=result['seconds'], seed_std=None,
                                 folder=str(folder)))
                for leaf in ('student.pt', 'probabilities.pt', 'epochs.jsonl', 'result.json'):
                    artifacts[str(folder/leaf)] = sha(folder/leaf)
        require(len(first_initial)==1 and len(first_classifier)==1, 'Initial model/head not paired')
        load_dataset.cache_clear()
        torch.cuda.empty_cache()
    require(len(rows)==config['expected_fits'], 'Wrong formal matrix size')
    save_csv('results.csv', rows)
    return rows, dict(reloaded_checkpoints=len(rows), metric_rows=len(rows), anchor_cells=24,
                      anchor_max_probability_difference=anchor_max, first_strict_validation_max_verified=True,
                      exact_encoder_and_bn_freeze=True, same_initial_classifier=True, updates_per_head=100,
                      all_exports_original_x_only_mlp=True, artifact_sha256=artifacts)


def summarize(rows, config):
    by = {(r['dataset'],r['condition'],r['route'],r['rule']):r for r in rows}
    noisy = [(n,c) for n in config['datasets'] for c in config['conditions'] if c!='clean']
    clean = [(n,'clean') for n in config['datasets']]
    comparisons = []
    comparisons_spec = [
        ('GB_minus_CE','REAL','GB','REAL','CE'),
        ('GB_minus_LS','REAL','GB','REAL','LS'),
        ('GB_minus_GCE','REAL','GB','REAL','GCE'),
        ('GB_minus_BOOT','REAL','GB','REAL','BOOT'),
        ('GB_minus_MASS','REAL','GB','REAL','MASS'),
        ('GB_minus_KNN','REAL','GB','REAL','KNN'),
        ('GB_minus_KM','REAL','GB','REAL','KM'),
        ('GB_minus_RANDOM','REAL','GB','REAL','RANDOM'),
        ('REAL_CE_minus_RANDOM_ENCODER_CE','REAL','CE','RANDOM_ENCODER','CE'),
        ('REAL_CE_minus_SHUFFLED_H_CE','REAL','CE','SHUFFLED_H','CE'),
        ('RANDOM_ENCODER_GB_minus_CE','RANDOM_ENCODER','GB','RANDOM_ENCODER','CE'),
        ('SHUFFLED_H_GB_minus_CE','SHUFFLED_H','GB','SHUFFLED_H','CE')]
    stats = {}
    for title, ar, a, br, b in comparisons_spec:
        ds = {}
        all_delta = []
        for n,c in noisy+clean:
            av,bv=by[n,c,ar,a]['test_percent'],by[n,c,br,b]['test_percent']
            comparisons.append(dict(comparison=title,dataset=n,condition=c,a_test=av,b_test=bv,delta_pp=av-bv))
            if c!='clean':
                ds.setdefault(n,[]).append(av-bv)
                all_delta.append(av-bv)
        stats[title] = dict(mean_noisy_delta_pp=float(np.mean(all_delta)), positive_cells=int(np.sum(np.array(all_delta)>0)),
                            cells=len(all_delta), dataset_means={n:float(np.mean(v)) for n,v in ds.items()},
                            clean_deltas={n:by[n,'clean',ar,a]['test_percent']-by[n,'clean',br,b]['test_percent'] for n in config['datasets']})
    summary = []
    arms = [('REAL',r) for r in config['real_rules']]+[(r,s) for r in config['control_routes'] for s in config['control_rules']]
    for route, rule in arms:
        for scope, keys in [('noisy',noisy),('clean',clean)]:
            selected=[by[n,c,route,rule] for n,c in keys]
            summary.append(dict(route=route,rule=rule,scope=scope,cells=len(keys),seed_count=1,seed_std=None,
                                test_mean_percent=float(np.mean([r['test_percent'] for r in selected])),
                                validation_mean_percent=float(np.mean([r['validation_percent'] for r in selected]))))
    regularizers=['LS','GCE','BOOT']
    validation_means={r:float(np.mean([by[n,c,'REAL',r]['validation_percent'] for n,c in noisy])) for r in regularizers}
    best=max(regularizers, key=lambda r:validation_means[r])
    envelope=[by[n,c,'REAL','GB']['test_percent']-max(by[n,c,'REAL',r]['test_percent'] for r in regularizers) for n,c in noisy]

    def gate(key, threshold=.5):
        s=stats[key]
        return bool(s['mean_noisy_delta_pp']>=threshold and s['positive_cells']>=6 and sum(v>0 for v in s['dataset_means'].values())>=2)

    def transfer(key):
        s=stats[key]
        return bool(s['mean_noisy_delta_pp']>=1 and sum(v>0 for v in s['dataset_means'].values())>=2)

    safety=all(v>=-1 for v in stats['GB_minus_CE']['clean_deltas'].values())
    decision=dict(selected_generic_regularizer=best, selection_uses='noisy validation macro only',
                  generic_validation_means=validation_means,
                  transfer_vs_random=transfer('REAL_CE_minus_RANDOM_ENCODER_CE'),
                  transfer_vs_shuffle=transfer('REAL_CE_minus_SHUFFLED_H_CE'),
                  local_direction_gate=gate('GB_minus_MASS'),
                  generic_regularizer_gate=gate('GB_minus_'+best),
                  km_gate=gate('GB_minus_KM'),
                  random_group_gate=stats['GB_minus_RANDOM']['mean_noisy_delta_pp']>=.5,
                  knn_gate=gate('GB_minus_KNN'),
                  clean_safety_gate=safety,
                  per_cell_test_regularizer_envelope_mean_pp=float(np.mean(envelope)),
                  envelope_is_descriptive_test_oracle_not_selector=True,
                  no_automatic_expansion=True, seed_count=1, seed_std=None)
    decision['full_gb_continuation_gate']=bool(decision['local_direction_gate'] and decision['generic_regularizer_gate']
          and decision['km_gate'] and decision['random_group_gate'] and decision['knn_gate'] and safety)
    save_csv('paired_deltas.csv',comparisons)
    save_csv('summary.csv',summary)
    write(ROOT/'comparisons.json',stats)
    write(ROOT/'decision.json',decision)
    lines=['# R2 controls: numerical summary', '', 'Single protocol/student seed0; no seed standard deviation or significance claims.', '',
           '| Route | Rule | Noisy accuracy % | Clean accuracy % |','|---|---|---:|---:|']
    for route,rule in arms:
        s=[r for r in summary if r['route']==route and r['rule']==rule]
        values={r['scope']:r['test_mean_percent'] for r in s}
        lines.append(f'| {route} | {rule} | {values["noisy"]:.4f} | {values["clean"]:.4f} |')
    lines+=['','## Paired comparisons','']
    for title,s in stats.items():
        lines.append(f'- {title}: {s["mean_noisy_delta_pp"]:+.4f}pp; positive {s["positive_cells"]}/9; dataset means {s["dataset_means"]}; clean {s["clean_deltas"]}')
    lines+=['','## Frozen gates','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```',
            '', 'Interpretation and final research decision are in 综合结论与接续.md.']
    (ROOT/'numerical_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return decision


def main():
    torch.set_num_threads(4)
    config=read(ROOT/'config.json')
    progress=read(ROOT/'progress.json')
    require(progress['status']=='complete' and progress['completed']==156, 'Do not score before full grid')
    frozen=read(ROOT/'frozen_inputs.json')['sha256']
    for path,expected in frozen.items():
        require(sha(path)==expected,'Frozen source changed: '+path)
    target_audit=check_targets(config)
    rows,model_audit=score_and_check(config)
    decision=summarize(rows,config)
    audit=dict(status='passed', frozen_files_verified=len(frozen), **target_audit, **model_audit)
    write(ROOT/'audit.json',audit)
    print(json.dumps(dict(status='complete',audit={k:v for k,v in audit.items() if k!='artifact_sha256'},decision=decision),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
