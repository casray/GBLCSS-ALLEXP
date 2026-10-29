"""Independent NumPy/sklearn scoring and per-node LOO target audit."""
import sys,json,hashlib,statistics as st
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from run import ROOT,read,sha,load,ref,verify_signature
from data_bridge import analysis_labels,load_supervised

def main():
    torch.set_num_threads(4)
    receipt=read(ROOT/'receipt.json');final=read(ROOT/'formal/final.json');scores=read(ROOT/'analysis/scores.json')
    assert final['status']=='complete' and final['scored']==128 and len(scores)==128
    assert len({tuple(r[k] for k in ['dataset','seed','representation','condition','head']) for r in scores})==128
    verify_signature(receipt);verify_signature(read(ROOT/'features_locked.json'))
    assert final['audit_sha256']==sha(ROOT/'analysis/audit.json')
    max_score=0.;max_target=0.;case_cache={};max_summary=0.
    for rr in receipt['references']:
        ds,s=rr['dataset'],rr['seed'];truth=analysis_labels(ds,s)
        for c in ['clean','uniform0.4','uniform0.6','pair0.4']:
            data=load_supervised(ds,s,c);obs=data['observed'].numpy();classes=data['classes']
            groups=load(rr['assignment'])[data['train']].numpy();one=np.eye(classes)[obs];target=np.zeros_like(one)
            counts={g:np.bincount(obs[groups==g],minlength=classes) for g in np.unique(groups)}
            for i,(z,g) in enumerate(zip(obs,groups)):
                local=counts[g].copy();local[z]-=1;support=local.sum();maximum=local.max()
                alpha=max(0.,min(1.,(maximum/max(1,support)-1/classes)/(1-1/classes))) if support>=2 else 0.
                beta=0. if (local==maximum).sum()>1 else .5
                q=(1-beta)*local/max(1,support);q[local.argmax()]+=beta
                target[i]=(1-alpha)*one[i]+alpha*q
            case_cache[(ds,s,c)]=(truth,data,{'CE':one,'SUPPORT32':target})
    for r in scores:
        ds,s,c,head=(r[k] for k in ['dataset','seed','condition','head'])
        truth,data,expected=case_cache[(ds,s,c)];path=Path(r['source'])
        p=load(path/'probabilities.pt').numpy();t=load(path/'target.pt').numpy();pred=p.argmax(1)
        err=float(np.abs(t-expected[head]).max());max_target=max(max_target,err);assert err<2e-7
        y=truth['y'].numpy();idx=truth['test'].numpy();valid=data['valid'].numpy();train=truth['train'].numpy()
        metrics=dict(test_accuracy=accuracy_score(y[idx],pred[idx]),test_macro_f1=f1_score(y[idx],pred[idx],labels=np.arange(p.shape[1]),average='macro',zero_division=0),validation_accuracy=accuracy_score(data['valid_y'].numpy(),pred[valid]),target_brier=float(((t.astype(float)-np.eye(p.shape[1])[y[train]])**2).sum(1).mean()))
        for k,v in metrics.items():max_score=max(max_score,abs(v-r[k]));assert abs(v-r[k])<3e-7
        logs=[json.loads(x) for x in (path/'epochs.jsonl').read_text().splitlines()]
        values=[x['validation_correct'] for x in logs];assert len(values)==200 and values.index(max(values))+1==r['best_epoch']
    for row in read(ROOT/'analysis/summary.json'):
        subset=[r for r in scores if all(r[k]==row[k] for k in ['dataset','representation','head'])]
        for metric in ['test_accuracy','validation_accuracy','test_macro_f1','target_brier']:
            scale=1 if metric=='target_brier' else 100
            values=[scale*np.mean([r[metric] for r in subset if r['seed']==s and r['condition']!='clean']) for s in [0,1]]
            for k,v in [('mean',np.mean(values)),('sd',np.std(values,ddof=1))]:max_summary=max(max_summary,abs(v-row[metric][k]));assert abs(v-row[metric][k])<1e-10
    result=dict(status='passed',scored=128,source_hashes=len(receipt['signature']),independent_numpy_target_cases=len(case_cache),max_manual_target_error=max_target,max_independent_score_error=max_score,max_mean_sd_error=max_summary,review_script_sha256=sha(__file__))
    dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
