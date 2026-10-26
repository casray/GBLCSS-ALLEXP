"""Posthoc clean-label scoring, only after all fit-stage choices are frozen."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from fit_pipeline import ROOT, EXP, CONFIG, r2, read, sha, save_json, save_csv


def metrics(y,p,classes):
    cm=torch.bincount(y*classes+p,minlength=classes*classes).reshape(classes,classes).double()
    den=cm.sum(0)+cm.sum(1)
    f1=torch.where(den>0,2*cm.diag()/den,0.)
    return dict(accuracy=float((y==p).double().mean()),macro_f1=float(f1.mean()))


def selector_row(original, label, support):
    return {**original, 'arm':label, 'selected_support':support}


def main():
    for stage in ('E1','E2','E4'):
        if not (ROOT/f'{stage}_fits_complete.json').exists():
            raise RuntimeError('Do not inspect test results before all stage selections are frozen')
    if (ROOT/'analysis_complete.json').exists():
        print('Analysis already complete; read saved results')
        return
    choices={str(p):sha(p) for p in ROOT.glob('*decision.json')}
    if (ROOT/'test_release.json').exists():
        if read(ROOT/'test_release.json')['choices'] != choices:
            raise RuntimeError('Selection changed since the first test release')
    else:
        save_json(ROOT/'test_release.json',dict(time=datetime.now().astimezone().isoformat(),choices=choices,
            selection_uses_test=False,oracle='optimistic retrospective support choice; never used for gates'))
    labels={n:r2.raw_data(n).y.long().cpu() for n in CONFIG['datasets']}
    rows=[]
    predictions={}
    class_rows=[]
    for path in sorted((ROOT/'fits').glob('**/result.json')):
        r=read(path)
        if r['status']!='complete':
            raise RuntimeError('Incomplete fit found')
        stage=path.relative_to(ROOT/'fits').parts[0]
        prob=torch.load(path.parent/'probabilities.pt',map_location='cpu',weights_only=True)
        if sha(path.parent/'probabilities.pt')!=r['probabilities_sha256']:
            raise RuntimeError('Prediction changed')
        geo=torch.load(ROOT/'diagnostics/e1'/r['dataset']/f'seed{r["seed"]}'/'geometry.pt',weights_only=True)
        test=geo['test']; y=labels[r['dataset']][test];p=prob.argmax(1)[test]
        row=dict(stage=stage,**{k:r[k] for k in ['dataset','seed','condition','arm','validation','best_epoch','seconds','epochs']},
                 **metrics(y,p,prob.shape[1]),folder=str(path.parent))
        rows.append(row)
        predictions[(stage,r['dataset'],r['seed'],r['condition'],r['arm'])]=p
        for c in range(prob.shape[1]):
            m=y==c
            class_rows.append(dict(stage=stage,dataset=r['dataset'],seed=r['seed'],condition=r['condition'],arm=r['arm'],
                class_index=c,count=int(m.sum()),accuracy=float((p[m]==y[m]).float().mean()) if m.any() else None))
    indexed={(r['stage'],r['dataset'],r['seed'],r['condition'],r['arm']):r for r in rows}
    selector_rows=[]
    decision=read(ROOT/('E1_full_decision.json' if (ROOT/'E1_full_decision.json').exists() else 'E1_pilot_decision.json'))
    for choice in decision['selections']:
        n,s,c=choice['dataset'],choice['seed'],choice['condition']
        available={k:indexed['E1',n,s,c,f'S{k}'] for k in CONFIG['support_grid']}
        oracle=max(available,key=lambda k:(available[k]['accuracy'],k))
        for label,k in [('SC',choice['SC']),('VAL',choice['VAL']),('ORACLE',oracle),('FIXED32',32)]:
            selector_rows.append(selector_row(available[k],label,k))
    rows.extend(selector_rows)
    save_csv(ROOT/'results.csv',rows)
    save_csv(ROOT/'per_class_results.csv',class_rows)
    save_csv(ROOT/'E1_selector_results.csv',selector_rows)
    groups=defaultdict(list)
    for r in rows:
        groups[r['stage'],r['dataset'],r['condition'],r['arm']].append(r)
    summary=[]
    for key,rs in groups.items():
        row=dict(zip(['stage','dataset','condition','arm'],key),seeds=len(rs))
        for m in ('accuracy','macro_f1','validation','seconds'):
            vals=[r[m] for r in rs]
            row[m+'_mean']=float(np.mean(vals))
            row[m+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summary.append(row)
    save_csv(ROOT/'summary.csv',summary)
    paired=[]
    for r in rows:
        if r['stage']=='E1' and r['arm'] in ('SC','VAL','FIXED32','ORACLE'):
            for base in ('CE','GCE','LEGACY32','OLD64','KM','KNN'):
                b=indexed['E1',r['dataset'],r['seed'],r['condition'],base]
                paired.append(dict(stage='E1',dataset=r['dataset'],seed=r['seed'],condition=r['condition'],
                    arm=r['arm'],baseline=base,accuracy_delta_pp=100*(r['accuracy']-b['accuracy']),
                    macro_f1_delta_pp=100*(r['macro_f1']-b['macro_f1'])))
        if r['stage']=='E4' and r['arm'] in ('B','D'):
            for base in ('A200','A300','C_full','C_head'):
                b=indexed['E4',r['dataset'],r['seed'],r['condition'],base]
                paired.append(dict(stage='E4',dataset=r['dataset'],seed=r['seed'],condition=r['condition'],
                    arm=r['arm'],baseline=base,accuracy_delta_pp=100*(r['accuracy']-b['accuracy']),
                    macro_f1_delta_pp=100*(r['macro_f1']-b['macro_f1'])))
    save_csv(ROOT/'paired_deltas.csv',paired)
    strata=[]
    cells=sorted({(r['dataset'],r['seed'],r['condition']) for r in rows if r['stage']=='E4'})
    for name,seed,condition in cells:
        geo=torch.load(ROOT/'diagnostics/e1'/name/f'seed{seed}'/'geometry.pt',weights_only=True)
        test=geo['test']; y=labels[name][test]
        teacher_folder=(ROOT/'clean_teachers'/name/f'seed{seed}' if condition=='clean' else
            EXP/'teacher_support_lcss_20261022/runs'/name/f'seed{seed}'/condition/'SUPPORT_LCSS')
        tp=torch.load(teacher_folder/'probabilities.pt',weights_only=True).argmax(1)[test]
        a=predictions['E4',name,seed,condition,'A200'];c=predictions['E4',name,seed,condition,'C_head']
        for arm in CONFIG['pilot']['E4_arms']:
            p=predictions['E4',name,seed,condition,arm]
            masks=dict(teacher_correct=tp==y,teacher_wrong=tp!=y,
                       R2_only_correct=(c==y)&(a!=y),KD_only_correct=(a==y)&(c!=y))
            for label,mask in masks.items():
                strata.append(dict(dataset=name,seed=seed,condition=condition,arm=arm,stratum=label,nodes=int(mask.sum()),
                    accuracy=float((p[mask]==y[mask]).double().mean()) if mask.any() else None,
                    teacher_agreement=float((p[mask]==tp[mask]).double().mean()) if mask.any() else None,
                    R2_KD_oracle_union=float(((c==y)|(a==y)).double().mean())))
    save_csv(ROOT/'E4_teacher_strata.csv',strata)
    for p,digest in choices.items():
        if sha(Path(p))!=digest:
            raise RuntimeError('Selection modified during posthoc analysis')
    save_json(ROOT/'analysis_complete.json',dict(rows=len(rows),fit_rows=len(indexed),test_release_sha256=sha(ROOT/'test_release.json'),
        mean_std_policy='sample std for >=2 seeds; null for seed0 only',complete=datetime.now().astimezone().isoformat()))
    print(f'POSTHOC COMPLETE: {len(indexed)} fits, {len(selector_rows)} selector rows')


if __name__=='__main__':
    main()
