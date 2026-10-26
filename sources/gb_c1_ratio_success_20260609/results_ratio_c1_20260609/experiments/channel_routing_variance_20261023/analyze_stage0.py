"""Posthoc stage0 comparisons; test gate is explicitly exploratory."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch

from run_stage0 import ROOT, OLD, CONFIG, f, old_folder
from metrics import accuracy_f1, pair_metrics, stage0_gate


def main():
    if not (ROOT/'fits_complete.json').exists():
        raise RuntimeError('Finish all fits before test scoring')
    if (ROOT/'stage0_decision.json').exists():
        print('Stage0 already scored; read immutable decision')
        return
    start_sources={str(p):f.sha(p) for p in [Path(__file__),ROOT/'metrics.py',ROOT/'PLAN.md',ROOT/'config.json']}
    f.save_json(ROOT/'stage0_test_release.json',dict(time=datetime.now().astimezone().isoformat(),
        sources=start_sources,explicit_test_gate=True,exploratory=True,source_choices_frozen=True))
    records,pairs,gaps,audits=[],[],[],[]
    pair_list=[('A0','C0'),('A0','A1'),('C0','C1'),('A0','C1'),('A1','C0'),('A1','C1')]
    for name in CONFIG['datasets']:
        raw=f.r2.raw_data(name)
        f.record_input(f.r2.DATA_FILES[name])
        labels=raw.y.long().cpu().numpy()
        geo=f.load(OLD/'diagnostics/e1'/name/'seed0/geometry.pt')
        for condition in CONFIG['conditions']:
            pred,prob={},{}
            for arm in CONFIG['arms']:
                for seed in (0,1):
                    folder=old_folder(name,condition,arm) if seed==0 else ROOT/'fits'/name/'protocol0/student1'/condition/arm
                    result=f.read(folder/'result.json')
                    f.record_input(folder/'result.json')
                    ps=f.load(folder/'probabilities.pt').numpy()
                    if f.sha(folder/'probabilities.pt')!=result['probabilities_sha256'] or f.sha(folder/'student.pt')!=result['checkpoint_sha256']:
                        raise RuntimeError('Artifact hash mismatch')
                    f.record_input(folder/'student.pt')
                    key=('A' if arm=='A300' else 'C')+str(seed)
                    prob[key]=ps
                    pred[key]=ps.argmax(1)
                    if not np.isfinite(ps).all() or not np.allclose(ps.sum(1),1,atol=1e-6):
                        raise RuntimeError('Invalid probability')
                    epochs=[json.loads(l) for l in (folder/'epochs.jsonl').read_text().splitlines()]
                    f.record_input(folder/'epochs.jsonl')
                    first=max(epochs,key=lambda r:r['validation_correct'])
                    if len(epochs)!=CONFIG['arms'][arm] or first['epoch']!=result['best_epoch']:
                        raise RuntimeError('Epoch budget/first validation maximum mismatch')
                    vi=geo['valid'].numpy()
                    if np.sum(pred[key][vi]==labels[vi])!=result['valid_correct']:
                        raise RuntimeError('Saved validation probability mismatch')
                    for split in ('valid','test'):
                        idx=geo[split].numpy()
                        records.append(dict(dataset=name,condition=condition,arm=arm,student_seed=seed,protocol_seed=0,
                            split=split,nodes=len(idx),best_epoch=result['best_epoch'],**accuracy_f1(labels[idx],pred[key][idx],ps.shape[1])))
                    audits.append(dict(dataset=name,condition=condition,arm=arm,student_seed=seed,
                        model_probability_hashes=True,validation_reproduced=True,first_maximum=True,epoch_budget=True))
            by_split={}
            for split in ('valid','test'):
                idx=geo[split].numpy();y=labels[idx]
                cell={}
                for a,c in pair_list:
                    pair=a+'_'+c
                    row=dict(dataset=name,condition=condition,split=split,pair=pair,
                        **pair_metrics(y,pred[a][idx],pred[c][idx]))
                    avg=accuracy_f1(y,((prob[a][idx]+prob[c][idx])/2).argmax(1),prob[a].shape[1])
                    row.update(average_accuracy=avg['accuracy'],average_macro_f1=avg['macro_f1'])
                    pairs.append(row);cell[pair]=row
                cross=cell['A0_C0']; same=max(cell['A0_A1']['union'],cell['C0_C1']['union'])
                cross_mean=float(np.mean([cell[p]['union'] for p in ['A0_C0','A0_C1','A1_C0','A1_C1']]))
                gaps.append(dict(dataset=name,condition=condition,split=split,
                    cross_union=cross['union'],same_A_union=cell['A0_A1']['union'],same_C_union=cell['C0_C1']['union'],
                    gap_pp=100*(cross['union']-same),cross_four_mean=cross_mean,cross_four_gap_pp=100*(cross_mean-same),
                    cross_headroom=cross['headroom'],same_A_headroom=cell['A0_A1']['headroom'],same_C_headroom=cell['C0_C1']['headroom']))
    f.save_csv(ROOT/'model_results.csv',records)
    f.save_csv(ROOT/'pair_results.csv',pairs)
    f.save_csv(ROOT/'stage0_gaps.csv',gaps)
    grouped=defaultdict(list)
    for r in records:
        grouped[r['dataset'],r['condition'],r['arm'],r['split']].append(r)
    summary=[]
    for key,rs in grouped.items():
        row=dict(zip(['dataset','condition','arm','split'],key),protocol_seeds=1,student_seeds=len(rs))
        for field in ('accuracy','macro_f1'):
            row[field+'_mean']=float(np.mean([r[field] for r in rs]))
            row[field+'_std']=float(np.std([r[field] for r in rs],ddof=1)) if len(rs)>1 else None
        summary.append(row)
    f.save_csv(ROOT/'model_summary.csv',summary)
    decision=stage0_gate([r for r in gaps if r['split']=='test'],
        primary=CONFIG['gate']['primary_datasets'],threshold=CONFIG['gate']['min_gap_pp'],required=CONFIG['gate']['min_positive_datasets'])
    decision.update(cross_pair='A0_C0',same_reference='per-condition max(A0_A1,C0_C1)',
        frozen_R2_encoder=True,claim='conditional initialization control only; no causal attribution or independent confirmation',
        mean_cross_pair_sensitivity={n:float(np.mean([r['cross_four_gap_pp'] for r in gaps if r['split']=='test' and r['dataset']==n and r['condition']!='clean'])) for n in CONFIG['datasets']})
    f.save_json(ROOT/'stage0_decision.json',decision)
    f.save_json(ROOT/'stage0_audit.json',dict(passed=True,fit_records=len(audits),checks=audits,
        test_release=f.sha(ROOT/'stage0_test_release.json'),decision_sha256=f.sha(ROOT/'stage0_decision.json')))
    if not decision['passed']:
        f.save_json(ROOT/'routing_status.json',dict(status='not_run',reason='stage0 failed frozen +2pp on 2/3 gate',
            new_neural_fits=0,new_router_fits=0,fused_distillation=False))
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    main()
