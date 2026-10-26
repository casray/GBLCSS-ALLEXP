"""Simple fusion and teacher-confidence reference on cached seed0 predictions."""
import json
from pathlib import Path

import numpy as np

from run_stage0 import ROOT,OLD,CONFIG,f,old_folder
from metrics import accuracy_f1,pair_metrics,route_metrics
from routing_kernels import choose_weight,choose_threshold


def cells():
    for name in CONFIG['datasets']:
        raw=f.r2.raw_data(name)
        y=raw.y.long().cpu().numpy()
        geo=f.load(OLD/'diagnostics/e1'/name/'seed0/geometry.pt')
        for condition in CONFIG['conditions']:
            a=f.load(old_folder(name,condition,'A300')/'probabilities.pt').numpy()
            c=f.load(old_folder(name,condition,'C_head')/'probabilities.pt').numpy()
            teacher=f.load(old_folder(name,condition,'A300')/'target.pt').numpy()
            yield dict(dataset=name,condition=condition,y=y,geo=geo,a=a,c=c,teacher=teacher,edges=raw.edge_index.cpu().numpy())


def evaluate(cell,pred,method,selection_validation=None,hard=False):
    y,a,c=cell['y'],cell['a'].argmax(1),cell['c'].argmax(1)
    row=dict(dataset=cell['dataset'],condition=cell['condition'],method=method)
    for split in ('valid','test'):
        idx=cell['geo'][split].numpy()
        met=accuracy_f1(y[idx],pred[idx],cell['a'].shape[1])
        row.update({split+'_'+k:v for k,v in met.items()})
        if split=='test':
            pair=pair_metrics(y[idx],a[idx],c[idx])
            row.update(oracle=pair['union'],best_single=pair['best_single'],gain_best_pp=100*(met['accuracy']-pair['best_single']))
            if hard:
                rm=route_metrics(y[idx],a[idx],c[idx],pred[idx]!=a[idx])
                row.update(rm)
                row['attainable_fraction']=(met['accuracy']-pair['best_single'])/pair['headroom'] if pair['headroom']>0 else None
            else:
                row['attainable_fraction']=None
    if selection_validation is not None:
        row['fullfit_valid_accuracy']=row['valid_accuracy']
        row['valid_accuracy']=selection_validation
    return row


def branch_gate(rows, baseline, method):
    index={(r['dataset'],r['condition'],r['method']):r for r in rows}
    noisy,clean={},{}
    for name in CONFIG['datasets']:
        ds=[]
        for condition in CONFIG['conditions']:
            n=index[name,condition,method]['valid_accuracy']
            b=(max(index[name,condition,k]['valid_accuracy'] for k in ('A0','C0')) if baseline=='best_single' else index[name,condition,baseline]['valid_accuracy'])
            if condition=='clean': clean[name]=100*(n-b)
            else: ds.append(100*(n-b))
        noisy[name]=float(np.mean(ds))
    passed=sum(noisy[n]>=.5 for n in CONFIG['gate']['primary_datasets'])>=2 and min(clean.values())>=-1
    return dict(passed=bool(passed),method=method,baseline=baseline,split='validation',
        noisy_gain_pp=noisy,clean_gain_pp=clean,threshold_pp=.5,required_primary_graphs=2)


def main():
    if not f.read(ROOT/'stage0_decision.json')['passed']:
        raise RuntimeError('Stage0 gate failed')
    if (ROOT/'L1_decision.json').exists():
        print('L1 already complete');return
    f.save_json(ROOT/'L1_sources.json',{str(p):f.sha(p) for p in [Path(__file__),ROOT/'routing_kernels.py',ROOT/'metrics.py',ROOT/'PLAN.md',ROOT/'config.json']})
    decisions=[]
    for cell in cells():
        vi=cell['geo']['valid'].numpy();y=cell['y'][vi]
        a,c=cell['a'],cell['c']
        w=choose_weight(y,a[vi],c[vi])
        # score=1-confidence; higher scores favor switching to C_head.
        threshold=choose_threshold(y,a[vi].argmax(1),c[vi].argmax(1),1-cell['teacher'][vi].max(1))
        decisions.append(dict(dataset=cell['dataset'],condition=cell['condition'],weight=w,
            confidence_threshold=None if np.isinf(threshold) else threshold,threshold_none_means_never_switch=True))
    f.save_json(ROOT/'L1_selections.json',decisions)
    chosen={(r['dataset'],r['condition']):r for r in decisions}
    rows=[]
    for cell in cells():
        a,c=cell['a'],cell['c'];selection=chosen[cell['dataset'],cell['condition']]
        w=selection['weight'];t=selection['confidence_threshold']
        t=float('inf') if t is None else t
        pred=dict(A0=a.argmax(1),C0=c.argmax(1),MEAN=((a+c)/2).argmax(1),
            GLOBAL=(w*a+(1-w)*c).argmax(1),TCONF=np.where(1-cell['teacher'].max(1)>=t,c.argmax(1),a.argmax(1)))
        for method,p in pred.items():
            rows.append(evaluate(cell,p,method,hard=method=='TCONF'))
    f.save_csv(ROOT/'L1_results.csv',rows)
    decision=branch_gate(rows,'best_single','GLOBAL')
    f.save_json(ROOT/'L1_decision.json',decision)
    f.save_json(ROOT/'routing_status.json',dict(status='L2_eligible' if decision['passed'] else 'stopped_L1',
        stages_completed=['L1'],neural_fits=0,router_fits=0))
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
