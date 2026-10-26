"""Post-freeze test evaluation and seed-paired descriptive statistics."""
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from protocol import ROOT, EXP, CONFIG, common, read, save, sha, now, csv_save, track, load, canonical
from analysis_stats import mean_std, seed_macro
from fit import f1_score
from run import cells, folder_for, decision_path
from targets import evidence, groups

METRICS = ('test_accuracy','validation','test_macro_f1')


def require_global_freeze(path=None):
    path = path or ROOT/'selections_frozen.json'
    if not path.exists():
        raise RuntimeError('Test label access requires frozen global selections')
    frozen = read(path)
    for relative,digest in frozen['selections'].items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError('Selection changed after global freeze')
    return sha(path)


def make_pairs(rows):
    bycell = defaultdict(dict)
    for r in rows:
        bycell[tuple(r[k] for k in ('stage','dataset','seed','backbone','condition'))][r['arm']] = r
    result = []
    for (stage,name,seed,backbone,cond),arms in bycell.items():
        selected = 'GROUP_SELECTED' if stage=='D1' else 'PROBE'
        if selected not in arms:
            continue
        for ref in (['CE','SUPPORT_L1.0','CELL_GLOBAL_VAL','EXACT_MASS_GLOBAL','MASS_MATCHED_SHUFFLE','GCE_Q0.7','KM_L1.0']
                    if stage=='D1' else ['CE','FULL_VAL','FIXED_S32','ONLY_K','ONLY_STRENGTH','SC','LEGACY_SUPPORT32','GCE_Q0.7','KM_L1.0']):
            if ref not in arms:
                continue
            result.append(dict(stage=stage,dataset=name,seed=seed,backbone=backbone,condition=cond,
                selected=selected,reference=ref,
                **{m+'_diff_pp':100*(arms[selected][m]-arms[ref][m]) for m in METRICS}))
    return result


def summarize(rows, metrics, identity, subset=(0,1,2)):
    groupset = defaultdict(list)
    for row in rows:
        if row['seed'] in subset:
            groupset[tuple(row[k] for k in identity)].append(row)
    summaries = []
    for key,values in groupset.items():
        result = dict(zip(identity,key))
        result.update(seeds=','.join(map(str,subset)),n_seeds=len({r['seed'] for r in values}))
        for metric in metrics:
            perseed = seed_macro(values,metric)
            mean,std = mean_std(perseed)
            result[metric+'_mean'],result[metric+'_std_ddof1'] = mean,std
            for s in subset:
                selected = [r[metric] for r in values if r['seed']==s]
                result[metric+f'_seed{s}'] = float(np.mean(selected)) if selected else None
        summaries.append(result)
    return summaries


def with_macros(rows):
    out = list(rows)
    for row in rows:
        if row['condition']!='clean':
            out.append(dict(row,condition='NOISY_MACRO'))
            out.append(dict(row,condition='NOISY_MACRO',dataset='ALL'))
        else:
            out.append(dict(row,dataset='ALL'))
    return out


def analyze():
    freeze = require_global_freeze()
    save(ROOT/'testing_started.json',dict(started=now(),selection_freeze_sha256=freeze,
         policy='all model and parameter choices frozen before this test reader'))
    records = read(ROOT/'fit_registry.json')
    rows,selected,recalls,costs,regrets,strata,frequency = [],[],[],[],[],[],[]
    scored = {}
    for name in CONFIG['datasets']:
        raw = torch.load(track(common.DATA_FILES[name]),map_location='cpu',weights_only=False)[0]
        y = (raw['y'] if isinstance(raw,dict) else raw.y).long()
        classes = int(y.max())+1
        for seed in CONFIG['seeds']:
            geo = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/f'seed{seed}/geometry.pt')
            tr,vi,te = [geo[k] for k in ('train','valid','test')]
            counts = torch.bincount(y[tr],minlength=classes)
            freqgroup = torch.empty(classes,dtype=torch.long)
            for group,ids in enumerate(np.array_split(torch.argsort(counts,stable=True).numpy(),3)):
                freqgroup[torch.as_tensor(ids)] = group
            current = [r for r in records if r['dataset']==name and r['seed']==seed]
            probs = {}
            for item in current:
                folder = ROOT/item['folder']
                result = read(folder/'result.json')
                if result['epoch']!=200:
                    raise RuntimeError('Incomplete full grid before test scoring')
                prob = torch.load(folder/'probabilities.pt',weights_only=True)
                if prob.shape!=(len(y),classes) or not torch.isfinite(prob).all() or not torch.allclose(prob.sum(1),torch.ones(len(y)),atol=1e-6):
                    raise RuntimeError('Invalid probabilities')
                pred = prob.argmax(1)
                row = dict(item,validation=result['validation'],valid_nll=result['valid_nll'],best_epoch=result['best_epoch'],
                    test_accuracy=float((pred[te]==y[te]).double().mean()),test_macro_f1=f1_score(pred[te],y[te],classes),
                    train_clean_accuracy=float((pred[tr]==y[tr]).double().mean()),seconds=result['seconds'])
                rows.append(row)
                scored[item['folder']] = row
                probs[item['folder']] = prob
                costs.append(dict(item,epochs=200,seconds=result['seconds'],target_ref=result['target_ref'],
                                  checkpoint_bytes=(folder/'checkpoint.pt').stat().st_size,probability_bytes=(folder/'probabilities.pt').stat().st_size))
                for c in range(classes):
                    mask = y[te]==c
                    recalls.append(dict(item,class_id=c,train_true_count=int(counts[c]),frequency_group=int(freqgroup[c]),
                                        test_count=int(mask.sum()),recall=float((pred[te][mask]==c).double().mean()) if mask.any() else None))
            def add(stage,backbone,cond,arm,source_stage,source_arm):
                folder = canonical(folder_for(source_stage,name,seed,backbone,cond,source_arm))
                row = dict(scored[folder],stage=stage,arm=arm,source_stage=source_stage,source_arm=source_arm)
                selected.append(row)
                pred = probs[folder].argmax(1)
                for k in range(3):
                    mask = freqgroup[y[te]]==k
                    frequency.append(dict(stage=stage,dataset=name,seed=seed,backbone=backbone,condition=cond,arm=arm,
                        frequency_group=k,count=int(mask.sum()),accuracy=float((pred[te][mask]==y[te][mask]).double().mean()) if mask.any() else None))
                return row,probs[folder]
            for backbone in CONFIG['backbones']:
                for cond in CONFIG['conditions']:
                    d1 = read(decision_path('D1',name,seed,backbone,cond))
                    d1arms = {a:a for a in ['CE','SUPPORT_L0.25','SUPPORT_L0.5','SUPPORT_L1.0','GCE_Q0.7','KM_L1.0','EXACT_MASS_GLOBAL','MASS_MATCHED_SHUFFLE']}
                    d1arms['GROUP_SELECTED'] = d1['selected']
                    gl = d1['best_cell_global_lambda']
                    d1arms['CELL_GLOBAL_VAL'] = 'CE' if gl==0 else f'SUPPORT_L{gl}'
                    pack = load(EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/f'seed{seed}'/f'{cond}.pt')
                    e = evidence(pack['observed'],geo['legacy32'][tr],classes)
                    grp = groups(e)
                    corrupted = pack['observed']!=y[tr]
                    one = F.one_hot(y[tr],classes).float()
                    for arm,source in d1arms.items():
                        row,prob = add('D1',backbone,cond,arm,'D1',source)
                        if arm not in ('GROUP_SELECTED','CELL_GLOBAL_VAL','EXACT_MASS_GLOBAL','MASS_MATCHED_SHUFFLE','CE','SUPPORT_L1.0'):
                            continue
                        result = read(ROOT/row['folder']/'result.json')
                        target = torch.load(ROOT/result['target_ref'],weights_only=True)
                        brier = ((target-one)**2).sum(1)-((e['own']-one)**2).sum(1)
                        truth = (target-e['own']).gather(1,y[tr,None]).squeeze(1)
                        moved = 1-target.gather(1,pack['observed'][:,None]).squeeze(1)
                        masks = {'original_clean':~corrupted,'original_corrupted':corrupted}
                        masks.update({f'group_{k}':grp==k for k in range(3)})
                        masks.update({f'true_frequency_{k}':freqgroup[y[tr]]==k for k in range(3)})
                        masks.update({f'group_{k}_{label}':(grp==k)&mask for k in range(3)
                                      for label,mask in [('clean',~corrupted),('corrupted',corrupted)]})
                        for label,mask in masks.items():
                            if mask.any():
                                strata.append(dict(dataset=name,seed=seed,backbone=backbone,condition=cond,arm=arm,stratum=label,
                                    count=int(mask.sum()),fraction=float(mask.double().mean()),
                                    support=float(e['support'][mask].mean()),original_mass=float(e['m'][mask].mean()),
                                    actual_mass=float(moved[mask].mean()),brier_change=float(brier[mask].mean()),
                                    true_class_mass_change=float(truth[mask].mean()),
                                    train_clean_accuracy=float((prob[tr].argmax(1)[mask]==y[tr][mask]).double().mean())))
                    probe = read(decision_path('C1',name,seed,backbone,cond))
                    full = read(decision_path('C1',name,seed,backbone,cond,'_full'))
                    c1arms = {key:full[key] for key in ['FULL_VAL','ONLY_K','ONLY_STRENGTH','FIXED_S32','SC']}
                    c1arms['PROBE'] = probe['PROBE']
                    c1rows = {}
                    for arm,source in c1arms.items():
                        c1rows[arm],_ = add('C1',backbone,cond,arm,'D1' if source=='CE' else 'C1',source)
                    for arm,source in [('CE','CE'),('LEGACY_SUPPORT32','SUPPORT_L1.0'),('GCE_Q0.7','GCE_Q0.7'),('KM_L1.0','KM_L1.0')]:
                        c1rows[arm],_ = add('C1',backbone,cond,arm,'D1',source)
                    gap = 100*(c1rows['PROBE']['test_accuracy']-c1rows['FULL_VAL']['test_accuracy'])
                    regrets.append(dict(dataset=name,seed=seed,backbone=backbone,condition=cond,
                        PROBE=probe['PROBE'],FULL_VAL=full['FULL_VAL'],test_gap_pp=gap,
                        validation_gap_pp=100*(c1rows['PROBE']['validation']-c1rows['FULL_VAL']['validation']),
                        within_0_5pp=abs(gap)<=.5,within_1pp=abs(gap)<=1,within_2pp=abs(gap)<=2,
                        probe_updates=1000,ce_updates=200,full_updates=3600,actual_candidate_updates=3600,
                        probe_candidate_seconds=full['probe_candidate_seconds'],ce_seconds=full['probe_ce_seconds'],
                        full_candidate_seconds=full['full_candidate_seconds'],
                        candidate_time_ratio=full['probe_candidate_seconds']/full['full_candidate_seconds']))
            del probs
    paired = make_pairs(selected)
    diffmetrics = [m+'_diff_pp' for m in METRICS]
    ids = ['stage','dataset','backbone','condition','arm']
    pids = ['stage','dataset','backbone','condition','selected','reference']
    csv_save(ROOT/'results.csv',rows)
    csv_save(ROOT/'selected_results.csv',selected)
    csv_save(ROOT/'summary.csv',summarize(with_macros(selected),METRICS,ids))
    csv_save(ROOT/'seeds12_summary.csv',summarize(with_macros(selected),METRICS,ids,(1,2)))
    csv_save(ROOT/'paired_differences.csv',paired)
    csv_save(ROOT/'paired_summary.csv',summarize(with_macros(paired),diffmetrics,pids))
    csv_save(ROOT/'seeds12_paired_summary.csv',summarize(with_macros(paired),diffmetrics,pids,(1,2)))
    csv_save(ROOT/'per_class_recall.csv',recalls)
    csv_save(ROOT/'frequency_group_test.csv',frequency)
    csv_save(ROOT/'d1_strata.csv',strata)
    csv_save(ROOT/'costs.csv',costs)
    csv_save(ROOT/'c1_regret.csv',regrets)
    csv_save(ROOT/'c1_costs.csv',[{k:v for k,v in r.items() if k not in ('test_gap_pp','validation_gap_pp')} for r in regrets])
    save(ROOT/'scored_folders.json',scored)
    save(ROOT/'analysis_complete.json',dict(finished=now(),full_fit_rows=len(rows),selected_rows=len(selected),
        paired_rows=len(paired),selection_freeze_sha256=freeze,seed_std='ddof1',unit='paired protocol/student seed'))
