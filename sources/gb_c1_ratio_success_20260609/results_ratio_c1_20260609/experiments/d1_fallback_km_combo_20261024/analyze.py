"""Post-freeze scoring with matched-seed, seed-first descriptive summaries."""
from collections import defaultdict
import json
import statistics

import torch
import torch.nn.functional as F

from protocol import ROOT, CONFIG, common, read, save, sha, track, now, csv_save, load, geometry_path
from rules import freeze_required
from targets import evidence, groups
from fit import f1_score

COMPARISONS=[('FALLBACK','LEGACY_EXPANDED','LEGACY_GLOBAL'),
             ('FALLBACK','LEGACY_EXPANDED','LEGACY_FORCED'),('FALLBACK','LEGACY_EXPANDED','CE'),
             ('KM_MATCH','LEGACY_GLOBAL','KM_GLOBAL'),('KM_MATCH','LEGACY_EXPANDED','KM_EXPANDED'),
             ('KM_GROUP','KM_EXPANDED','KM_GLOBAL'),
             ('COMBO','ADAPTIVE_EXPANDED','FIXED_GLOBAL'),('COMBO','ADAPTIVE_EXPANDED','FIXED_EXPANDED'),
             ('COMBO','ADAPTIVE_EXPANDED','ADAPTIVE_GLOBAL'),('COMBO','ADAPTIVE_EXPANDED','ORIGINAL_C1'),
             ('REFINEMENT','ADAPTIVE_GLOBAL','ORIGINAL_C1'),('COMBO','ADAPTIVE_EXPANDED','SINGLE_MODULE_VAL'),
             ('FIXED_GROUP','FIXED_EXPANDED','FIXED_GLOBAL'),('SUPPORT','ADAPTIVE_GLOBAL','FIXED_GLOBAL')]


def summarize(rows, identity, metrics, seeds):
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['seed'] not in seeds:
            continue
        variants=[r,dict(r,dataset='ALL')] if r['condition']=='clean' else [r,dict(r,condition='NOISY_MACRO'),dict(r,condition='NOISY_MACRO',dataset='ALL')]
        for v in variants:
            by[tuple(v[k] for k in identity)][r['seed']].append(v)
    out=[]
    for key,groupset in by.items():
        result=dict(zip(identity,key),seeds=','.join(map(str,seeds)),n_seeds=len(groupset))
        assert set(groupset)==set(seeds)
        for m in metrics:
            means={s:statistics.mean(v[m] for v in rs) for s,rs in groupset.items()}
            result[m+'_mean']=statistics.mean(means.values())
            result[m+'_std_ddof1']=statistics.stdev(means.values())
            result.update({m+f'_seed{s}':v for s,v in means.items()})
        out.append(result)
    return out


def validate_freeze():
    digest=freeze_required(ROOT/'selections_frozen.json')
    frozen=read(ROOT/'selections_frozen.json')
    assert sha(ROOT/'reuse_manifest.json')==frozen['reuse_manifest_sha256']
    for p,h in frozen['choices'].items():
        assert sha(ROOT/p)==h
    return digest


def analyze():
    freeze=validate_freeze()
    save(ROOT/'testing_started.json',dict(started=now(),freeze_sha256=freeze,retrospective=True))
    refs=read(ROOT/'reuse_manifest.json')
    rawrows,selected,pairs,recalls,strata,costs=[],[],[],[],[],[]
    choices=[read(p) for p in (ROOT/'selections').rglob('*.json')]
    for n in CONFIG['datasets']:
        data=torch.load(track(common.DATA_FILES[n]),map_location='cpu',weights_only=False)[0]
        y=(data['y'] if isinstance(data,dict) else data.y).long()
        classes=int(y.max())+1
        for s in CONFIG['seeds']:
            geo=load(geometry_path(n,s))
            tr,te=geo['train'],geo['test']
            km=load(ROOT.parent/'r2_body_comparison_20261023/geometry'/n/f'seed{s}.pt')
            for c in CONFIG['conditions']:
                from protocol import cell_inputs
                pack=cell_inputs(n,s,c)
                es={p:evidence(pack['observed'],part[tr],classes) for p,part in dict(LEGACY=geo['legacy32'],KM=km,
                    **{f'S{k}':v for k,v in geo['partitions'].items()}).items()}
                scored,probabilities={},{}
                for key,item in refs.items():
                    if (item['dataset'],item['seed'],item['condition'])!=(n,s,c):
                        continue
                    folder=__import__('pathlib').Path(item['folder'])
                    r=read(folder/'result.json')
                    assert sha(folder/'result.json')==item['result_sha256']
                    prob=torch.load(folder/'probabilities.pt',weights_only=True)
                    assert prob.shape==(len(y),classes) and torch.isfinite(prob).all()
                    assert torch.allclose(prob.sum(1),torch.ones(len(y)),atol=1e-6)
                    pred=prob.argmax(1)
                    metrics=dict(test_accuracy=float((pred[te]==y[te]).double().mean()),
                        test_macro_f1=f1_score(pred[te],y[te],classes),validation=r['validation'],valid_macro_f1=r['valid_macro_f1'])
                    row=dict(ref=key,**item,**metrics)
                    rawrows.append(row)
                    scored[key]=row
                    probabilities[key]=prob
                for cell in [v for v in choices if (v['dataset'],v['seed'],v['condition'])==(n,s,c)]:
                    b=cell['backbone']
                    for selector,methods in cell['choices'].items():
                        for method,key in methods.items():
                            row=dict(scored[key],selector=selector,method=method)
                            selected.append(row)
                            pred=probabilities[key].argmax(1)
                            for cl in range(classes):
                                mask=y[te]==cl
                                recalls.append(dict(dataset=n,seed=s,backbone=b,condition=c,selector=selector,method=method,
                                    class_id=cl,train_true_count=int((y[tr]==cl).sum()),test_count=int(mask.sum()),
                                    recall=float((pred[te][mask]==cl).double().mean()) if mask.any() else None))
                            e=es[refs[key]['partition']]
                            group=groups(e)
                            corrupted=pack['observed']!=y[tr]
                            target=torch.load(refs[key]['target_path'],weights_only=True)
                            one=F.one_hot(y[tr],classes).float()
                            brier=((target-one)**2).sum(1)-((e['own']-one)**2).sum(1)
                            truegain=(target-e['own']).gather(1,y[tr,None]).squeeze(1)
                            masks=dict(clean=~corrupted,corrupted=corrupted)
                            masks.update({f'group{k}_{label}':(group==k)&mask for k in range(3) for label,mask in [('clean',~corrupted),('corrupted',corrupted)]})
                            for label,mask in masks.items():
                                if mask.any():
                                    strata.append(dict(dataset=n,seed=s,backbone=b,condition=c,selector=selector,method=method,
                                        partition=refs[key]['partition'],stratum=label,count=int(mask.sum()),
                                        support=float(e['support'][mask].mean()),brier_change=float(brier[mask].mean()),
                                        true_class_mass_change=float(truegain[mask].mean()),
                                        train_accuracy=float((pred[tr][mask]==y[tr][mask]).double().mean())))
                        for stage,a,d in COMPARISONS:
                            if a not in methods or d not in methods:
                                continue
                            aa,dd=scored[methods[a]],scored[methods[d]]
                            pairs.append(dict(stage=stage,dataset=n,seed=s,backbone=b,condition=c,selector=selector,
                                method=a,reference=d,**{m+'_diff_pp':100*(aa[m]-dd[m]) for m in ['test_accuracy','test_macro_f1','validation','valid_macro_f1']}))
                        if b=='R0':
                            costs.append(dict(dataset=n,seed=s,condition=c,selector=selector,
                                adaptive_support=cell['meta']['adaptive_support'],original_probe=cell['meta']['PROBE'],
                                **{k:v for k,v in cell['meta']['adaptive_cost'].items() if k!='prefix_epochs'},
                                fixed_global_updates=800,fixed_expanded_updates=1600,matched_KM_GB_global=800,matched_KM_GB_expanded=1600))
                del probabilities
    csv_save(ROOT/'results.csv',rawrows)
    csv_save(ROOT/'selected_results.csv',selected)
    csv_save(ROOT/'paired_differences.csv',pairs)
    csv_save(ROOT/'per_class_recall.csv',recalls)
    csv_save(ROOT/'strata.csv',strata)
    csv_save(ROOT/'deployment_costs.csv',costs)
    for prefix,seeds in [('',[0,1,2]),('seeds12_',[1,2])]:
        csv_save(ROOT/(prefix+'summary.csv'),summarize(selected,['dataset','backbone','condition','selector','method'],
            ['test_accuracy','test_macro_f1','validation','valid_macro_f1'],seeds))
        csv_save(ROOT/(prefix+'paired_summary.csv'),summarize(pairs,['stage','dataset','backbone','condition','selector','method','reference'],
            ['test_accuracy_diff_pp','test_macro_f1_diff_pp','validation_diff_pp','valid_macro_f1_diff_pp'],seeds))
    save(ROOT/'analysis_complete.json',dict(finished=now(),raw_rows=len(rawrows),selected_rows=len(selected),pairs=len(pairs),
        retrospective_reuse=True,freeze_sha256=freeze,std='ddof1',unit='protocol/student seed'))
