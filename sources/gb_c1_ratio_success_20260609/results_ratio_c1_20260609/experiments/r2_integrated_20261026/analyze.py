"""Post-freeze test and diagnostic analysis; seed-first paired summaries."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from protocol import ROOT,EXP,CONFIG,read,save,sha,track,now,csv_save,common,load,geometry_path,cell_inputs
from targets import groups
from fit import f1_score

METRICS=['test_accuracy','test_macro_f1','validation','valid_macro_f1']
COMPARISONS=[('ADAPTIVE_EXPANDED',b) for b in ['SINGLE_MODULE_VAL','CE','GCE','OLD64',
    'LEGACY_SUPPORT32','FIXED_S32','FIXED_GLOBAL','FIXED_EXPANDED','ADAPTIVE_GLOBAL']]+[
    ('FIXED_EXPANDED','FIXED_GLOBAL'),('ADAPTIVE_GLOBAL','FIXED_GLOBAL')]


def summarize(rows,identity,metrics,seeds,expand=True,require_all=True):
    buckets=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['seed'] not in seeds:continue
        variants=[r]
        if expand:
            if r['condition']=='clean':variants+=[dict(r,dataset='ALL')]
            else:variants+=[dict(r,condition='NOISY_MACRO'),dict(r,dataset='ALL',condition='NOISY_MACRO')]
        for v in variants:buckets[tuple(v[k] for k in identity)][r['seed']].append(v)
    out=[]
    for key,byseed in buckets.items():
        if require_all:assert set(byseed)==set(seeds)
        row=dict(zip(identity,key),seeds=','.join(map(str,seeds)),n_available_seeds=len(byseed))
        for m in metrics:
            means={s:statistics.mean(r[m] for r in rs if r[m] is not None) for s,rs in byseed.items()
                if any(r[m] is not None for r in rs)}
            row[m+'_mean']=statistics.mean(means.values()) if means else None
            row[m+'_std_ddof1']=statistics.stdev(means.values()) if len(means)>1 else None
            row.update({m+f'_seed{s}':v for s,v in means.items()})
        out.append(row)
    return out


def validate_freeze():
    p=ROOT/'selections_frozen.json'
    if not p.exists():raise RuntimeError('Test requires frozen choices')
    f=read(p)
    assert sha(ROOT/'reuse_manifest.json')==f['reuse_manifest_sha256']
    for kind in ('choices','races'):
        for name,digest in f[kind].items():assert sha(ROOT/name)==digest
    return sha(p)


def analyze():
    frozen=validate_freeze()
    save(ROOT/'testing_started.json',dict(started=now(),freeze_sha256=frozen,retrospective=True))
    from run import prepare
    from protocol import graph_inputs
    refs=read(ROOT/'reuse_manifest.json')
    raw,selected,pairs,recalls,strata,costs,bridges=[],[],[],[],[],[],[]
    for n in CONFIG['datasets']:
        data=torch.load(track(common.DATA_FILES[n]),map_location='cpu',weights_only=False)[0]
        y=(data['y'] if isinstance(data,dict) else data.y).long()
        classes=int(y.max())+1
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            tr,te=g['geometry']['train'],g['geometry']['test']
            for c in CONFIG['conditions']:
                v,es,_=prepare(n,s,c,g)
                choice=read(ROOT/'selections'/n/f'seed{s}'/f'{c}.json')
                rows,probs={},{}
                for key,ref in refs.items():
                    if (ref['dataset'],ref['seed'],ref['condition'])!=(n,s,c) or ref['epoch']!=200:continue
                    folder=Path(ref['folder']);r=read(folder/'result.json')
                    assert sha(folder/'result.json')==ref['result_sha256']
                    p=torch.load(folder/'probabilities.pt',weights_only=True)
                    assert p.shape==(len(y),classes) and torch.isfinite(p).all()
                    assert torch.allclose(p.sum(1),torch.ones(len(y)),atol=1e-6)
                    pred=p.argmax(1)
                    row=dict(ref=key,**ref,test_accuracy=float((pred[te]==y[te]).double().mean()),
                        test_macro_f1=f1_score(pred[te],y[te],classes),validation=r['validation'],
                        valid_macro_f1=r['valid_macro_f1'],valid_nll=r['valid_nll'],best_epoch=r['best_epoch'])
                    raw.append(row);rows[key]=row;probs[key]=p
                    if ref['arm']=='OLD64':
                        old=EXP/'r2_body_comparison_20261023/fits'/n/f'seed{s}'/c/'OLD64'
                        prev=read(track(old/'result.json'))
                        op=torch.load(track(old/'probabilities.pt'),weights_only=True)
                        assert sha(old/'probabilities.pt')==prev['probabilities_sha256']
                        bridges.append(dict(dataset=n,seed=s,condition=c,max_prob_difference=float((p-op).abs().max()),
                            best_epoch_new=r['best_epoch'],best_epoch_old=prev['best_epoch'],
                            old64_refit_reason='older schema missing full input/RNG hash contract'))
                for method,key in choice['choices'].items():
                    row=dict(rows[key],method=method);selected.append(row)
                    pred=probs[key].argmax(1)
                    for cl in range(classes):
                        mask=y[te]==cl
                        recalls.append(dict(dataset=n,seed=s,condition=c,method=method,class_id=cl,
                            train_count=int((y[tr]==cl).sum()),test_count=int(mask.sum()),
                            recall=float((pred[te][mask]==cl).double().mean()) if mask.any() else None))
                    e=es[refs[key]['partition']];group=groups(e)
                    corrupted=v['observed']!=y[tr]
                    target=torch.load(refs[key]['target_path'],weights_only=True)
                    one=F.one_hot(y[tr],classes).float()
                    brier=((target-one)**2).sum(1)-((e['own']-one)**2).sum(1)
                    truegain=(target-e['own']).gather(1,y[tr,None]).squeeze(1)
                    masks=dict(clean=~corrupted,corrupted=corrupted)
                    masks.update({f'group{k}_{label}':(group==k)&mask for k in range(3)
                        for label,mask in [('clean',~corrupted),('corrupted',corrupted)]})
                    for label,mask in masks.items():
                        if not mask.any():continue
                        strata.append(dict(dataset=n,seed=s,condition=c,method=method,stratum=label,
                            partition=refs[key]['partition'],count=int(mask.sum()),support=float(e['support'][mask].mean()),
                            brier_change=float(brier[mask].mean()),true_class_mass_change=float(truegain[mask].mean()),
                            train_accuracy=float((pred[tr][mask]==y[tr][mask]).double().mean())))
                for a,b in COMPARISONS:
                    ra,rb=rows[choice['choices'][a]],rows[choice['choices'][b]]
                    pairs.append(dict(dataset=n,seed=s,condition=c,method=a,reference=b,
                        **{m+'_diff_pp':100*(ra[m]-rb[m]) for m in METRICS}))
                phys=[r for r in read(ROOT/'physical_fits.json').values() if (r['dataset'],r['seed'],r['condition'])==(n,s,c)]
                costs.append(dict(dataset=n,seed=s,condition=c,joint_candidate_updates=2250,joint_with_ce=2450,
                    global_candidate_updates=1000,global_with_ce=1200,fixed_global=800,fixed_expanded=1600,
                    stronger_single_union_without_duplicate_ce_upper=2600,
                    new_unique=len(phys),new_full=sum(r['epoch']==200 for r in phys),
                    new_updates=sum(r['epoch'] for r in phys),new_training_seconds=sum(r['seconds'] for r in phys),
                    historical_full_global_grid_paid_updates=3600,
                    historical_replay_new_updates=0,budget_ratio_joint_to_global=2450/1200))
    for name,data in [('results',raw),('selected_results',selected),('paired_differences',pairs),
        ('per_class_recall',recalls),('strata',strata),('costs',costs),('old64_bridges',bridges)]:
        csv_save(ROOT/(name+'.csv'),data)
    for prefix,seeds in [('',[0,1,2]),('seeds12_',[1,2])]:
        csv_save(ROOT/(prefix+'summary.csv'),summarize(selected,['dataset','condition','method'],METRICS,seeds))
        csv_save(ROOT/(prefix+'paired_summary.csv'),summarize(pairs,['dataset','condition','method','reference'],[m+'_diff_pp' for m in METRICS],seeds))
        csv_save(ROOT/(prefix+'per_class_recall_summary.csv'),summarize(recalls,
            ['dataset','condition','method','class_id'],['recall'],seeds,expand=False,require_all=False))
        csv_save(ROOT/(prefix+'strata_summary.csv'),summarize(strata,
            ['dataset','condition','method','stratum'],['support','brier_change','true_class_mass_change','train_accuracy'],seeds,expand=False,require_all=False))
    refpath=track(EXP/'d1_fallback_km_combo_20261024/selected_results.csv')
    with refpath.open(encoding='utf-8-sig',newline='') as f:
        reference=[r for r in csv.DictReader(f) if r['backbone']=='R0' and r['selector']=='accuracy'
            and r['method'] in ['FIXED_GLOBAL','FIXED_EXPANDED','ADAPTIVE_GLOBAL','ADAPTIVE_EXPANDED','SINGLE_MODULE_VAL']]
    csv_save(ROOT/'R0_readonly_reference.csv',reference)
    save(ROOT/'analysis_complete.json',dict(finished=now(),selected=len(selected),paired=len(pairs),
        per_class_rows=len(recalls),strata_rows=len(strata),freeze_sha256=frozen,std='ddof1',
        repeat_unit='protocol/student seed; fixed historical SSL',R0_reference='different historical tie/search; descriptive only'))
