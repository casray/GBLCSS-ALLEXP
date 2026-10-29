"""Independent saved-prediction scores, paired statistics and scope audit."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
import numpy as np
import torch

W=Path(r'C:\Users\Admin\Desktop\实验8去噪5.19-\gb_c1_ratio_success_20260609\results_ratio_c1_20260609')
sys.path.insert(0,str(W/'experiments/graphecl_gb_context_20261028/budget'))
from reuse import ROOT, labels_after_lock, supervised, load, read, save, sha


def independent_scores(rows):
    cache={};supervision={};max_acc=max_f1=max_brier=0.
    for row in rows:
        key=(row['dataset'],row['seed'])
        if key not in cache:
            lab=labels_after_lock(*key)
            cache[key]=(lab['y'].numpy(),lab['test'].numpy())
        skey=(*key,row['condition'])
        if skey not in supervision:supervision[skey]=supervised(*skey)
        y,te=cache[key];data=supervision[skey];train=data['train'].numpy();classes=data['classes']
        folder=Path(row['source']);pred=load(folder/'probabilities.pt').numpy().argmax(1)
        truth=y[te];p=pred[te];conf=np.bincount(truth*classes+p,minlength=classes*classes).reshape(classes,classes)
        tp=np.diag(conf).astype(float);den=conf.sum(0)+conf.sum(1)
        f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den!=0).mean()
        acc=(p==truth).mean();target=load(folder/'target.pt').numpy().astype(np.float64)
        brier=np.mean(np.square(target).sum(1)-2*target[np.arange(len(train)),y[train]]+1)
        max_acc=max(max_acc,abs(float(acc)-row['test_accuracy']))
        max_f1=max(max_f1,abs(float(f1)-row['test_macro_f1']))
        max_brier=max(max_brier,abs(float(brier)-row['target_brier']))
    assert max_acc<1e-12 and max_f1<1e-12 and max_brier<1e-12
    return dict(accuracy_error=max_acc,macro_f1_error=max_f1,brier_error=max_brier)


def main():
    torch.set_num_threads(4)
    rows=read(ROOT/'analysis_budget/scores.json');summary=read(ROOT/'analysis_budget/summary.json');pairs=read(ROOT/'analysis_budget/paired.json')
    assert len(rows)==492 and len({(r['dataset'],r['seed'],r['arm'],r['condition'],r['head']) for r in rows})==492
    assert set(r['dataset'] for r in rows)=={'cora','citeseer','corafull','coauthorcs','amazonpho'}
    metrics=['test_accuracy','test_macro_f1','validation_accuracy','target_brier'];groups={}
    for r in rows:
        groups.setdefault((r['dataset'],r['arm'],r['head'],r['seed']),[]).append(r)
    error=0.
    for r in summary:
        for metric in metrics:
            vals=[statistics.mean(v[metric] for v in groups[r['dataset'],r['arm'],r['head'],s]) for s in [0,1]]
            error=max(error,abs(statistics.mean(vals)-r[metric]['mean']),abs(statistics.stdev(vals)-r[metric]['sd']))
    lookup={(r['dataset'],r['seed'],r['arm'],r['condition'],r['head']):r for r in rows}
    pair_error=0.
    for r in pairs:
        comphead=('CE' if r['head']=='CE' else 'fixed') if r['comparator']=='ecl' else r['head']
        for metric in metrics:
            ds=[100*statistics.mean(lookup[r['dataset'],s,r['arm'],c,r['head']][metric]-lookup[r['dataset'],s,r['comparator'],c,comphead][metric] for c in ['uniform0.4','uniform0.6','pair0.4']) for s in [0,1]]
            pair_error=max(pair_error,abs(statistics.mean(ds)-r[metric]['mean_pp']),abs(statistics.stdev(ds)-r[metric]['sd_pp']))
    assert error<1e-12 and pair_error<1e-10
    diagnostic=independent_scores(rows)
    audit=dict(status='passed',rows=len(rows),summary_max_error=error,paired_pp_max_error=pair_error,**diagnostic,
               scores_sha256=sha(ROOT/'analysis_budget/scores.json'),script_sha256=sha(Path(__file__)))
    save(ROOT/'analysis_budget/independent_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
