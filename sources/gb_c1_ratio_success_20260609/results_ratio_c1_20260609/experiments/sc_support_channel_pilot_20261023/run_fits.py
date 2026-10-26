"""Validation-only stage orchestration, followed by locked posthoc analysis."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from fit_pipeline import (ROOT, EXP, CONFIG, data, cell, control_targets, historical_target,
    historical_encoder, fit, alignment, teacher, read, load, initialize, save_json,
    save_csv, sha, AlignedMLP, state_hash, set_seed)


def location(stage,name,seed,condition,arm):
    return ROOT/'fits'/stage/name/f'seed{seed}'/condition/arm


def smoke():
    if (ROOT/'smoke_complete.json').exists():
        return
    d=data('cora')
    geo,pack=cell('cora',0,'uniform0.4')
    ce=F.one_hot(pack['observed'],d['classes']).float()
    r=fit(ROOT/'smoke/R0','cora',0,'uniform0.4','CE',ce,3)
    ref=read(EXP/'gce_lcss_fusion_20261022/runs/cora/seed0/uniform0.4/R0_CE/result.json')
    for key in ('initial_hash','initial_rng_hash'):
        if r[key]!=ref[key]:
            raise RuntimeError('Historical initialization bridge failed: '+key)
    historical_log=EXP/'gce_lcss_fusion_20261022/runs/cora/seed0/uniform0.4/R0_CE/epochs.jsonl'
    old=[json.loads(line) for line in historical_log.read_text().splitlines()][:3]
    new=[json.loads(line) for line in (ROOT/'smoke/R0/epochs.jsonl').read_text().splitlines()]
    loss_error=max(abs(a['loss']-b['loss']) for a,b in zip(old,new))
    if loss_error>1e-6:
        raise RuntimeError(f'R0 loss trajectory differs: {loss_error}')
    enc=historical_encoder('cora',0)
    frozen=fit(ROOT/'smoke/frozen','cora',0,'uniform0.4','frozen',ce,3,encoder=enc,frozen=True)
    prob,_=teacher('cora',0,'uniform0.4')
    fit(ROOT/'smoke/KD','cora',0,'uniform0.4','KD',prob,3,all_nodes=True)
    fit(ROOT/'smoke/GCN','cora',0,'uniform0.4','GCN',historical_target('cora',0,'uniform0.4'),3,graph=True)
    save_json(ROOT/'smoke_complete.json',dict(passed=True,initialization_exact=True,
        R0_max_loss_error=loss_error,frozen_encoder_unchanged=frozen['frozen_encoder_unchanged'],
        deployment_difference=frozen['deployment_difference'],manifest=sha(ROOT/'fit_sources.json')))


def e1_cells(names,seeds):
    for name in names:
        for seed in seeds:
            for condition in CONFIG['conditions']:
                geo,pack=cell(name,seed,condition)
                targets={f'S{s}':t for s,t in pack['targets'].items()}
                targets.update(control_targets(name,seed,condition))
                targets['GCE']=targets['CE']
                for arm,target in targets.items():
                    fit(location('E1',name,seed,condition,arm),name,seed,condition,arm,target,200,gce=arm=='GCE')


def e1_decision(names,seeds,label):
    rows=[]
    for name in names:
        for seed in seeds:
            for condition in CONFIG['conditions']:
                _,pack=cell(name,seed,condition)
                rs={s:read(location('E1',name,seed,condition,f'S{s}')/'result.json') for s in CONFIG['support_grid']}
                sc=pack['selected_support']
                val=max(rs,key=lambda s:(rs[s]['valid_correct'],s))
                ce=read(location('E1',name,seed,condition,'CE')/'result.json')
                rows.append(dict(dataset=name,seed=seed,condition=condition,SC=sc,VAL=val,
                    SC_validation=rs[sc]['validation'],VAL_validation=rs[val]['validation'],
                    SC_minus_VAL_pp=100*(rs[sc]['validation']-rs[val]['validation']),
                    SC_minus_S32_pp=100*(rs[sc]['validation']-rs[32]['validation']),
                    SC_minus_CE_pp=100*(rs[sc]['validation']-ce['validation'])))
    uniform=[r for r in rows if r['condition'].startswith('uniform')]
    gains={n:float(np.mean([r['SC_minus_S32_pp'] for r in uniform if r['dataset']==n])) for n in names}
    clean={n:float(np.mean([r['SC_minus_CE_pp'] for r in rows if r['dataset']==n and r['condition']=='clean'])) for n in names}
    gap=float(np.mean([r['SC_minus_VAL_pp'] for r in uniform]))
    needed=2 if label=='pilot' else 3
    passed=gap>=-.5 and sum(v>0 for v in gains.values())>=needed and min(clean.values())>=-1
    decision=dict(stage='E1',scope=label,passed=passed,validation_only=True,uniform_SC_minus_VAL_pp=gap,
        uniform_SC_minus_fixed32_pp=gains,clean_SC_minus_CE_pp=clean,
        decision='expand' if passed else 'stop E1 expansion; no threshold changes',selections=rows)
    save_json(ROOT/f'E1_{label}_decision.json',decision)
    print('E1 GATE '+json.dumps({k:v for k,v in decision.items() if k!='selections'}),flush=True)
    return decision


def e1():
    e1_cells(CONFIG['pilot']['E1_datasets'],[0])
    decision=e1_decision(CONFIG['pilot']['E1_datasets'],[0],'pilot')
    if decision['passed']:
        e1_cells(CONFIG['datasets'],CONFIG['seeds'])
        full=e1_decision(CONFIG['datasets'],CONFIG['seeds'],'full')
        if full['passed']:
            for name in CONFIG['datasets']:
                for seed in CONFIG['seeds']:
                    for cond in CONFIG['conditions']:
                        _,p=cell(name,seed,cond)
                        targets=dict(SC=p['targets'][p['selected_support']],S32=p['targets'][32],
                                     CE=F.one_hot(p['observed'],data(name)['classes']).float())
                        for arm,t in targets.items():
                            fit(location('E1_GCN',name,seed,cond,arm),name,seed,cond,arm,t,200,graph=True)
    return decision


def alignment_metrics(name,seed,folder,prefix):
    d=data(name)
    geo,_=cell(name,seed,'clean')
    model=AlignedMLP(d['x'].shape[1],d['classes'],d['h'].shape[1],64,.5).cuda().eval()
    model.load_state_dict(load(folder/f'prefix{prefix}.pt'))
    with torch.no_grad():
        x=model.projection(model.encode(d['x']))[geo['valid'].cuda()].double()
        y=d['h'][geo['valid'].cuda()].double()
        cosine=float(F.cosine_similarity(x,y).mean())
        x=x-x.mean(0);y=y-y.mean(0)
        cka=float((x.T@y).square().sum()/((x.T@x).square().sum()*(y.T@y).square().sum()).sqrt().clamp_min(1e-30))
    return dict(dataset=name,seed=seed,prefix=prefix,validation_projected_cosine=cosine,
                validation_linear_CKA=cka,interpretation='in-sample all-node alignment, not held-out predictability')


def e2_heads(name,seed,prefixes,budget):
    folder=alignment(name,seed,budget)
    metrics=[]
    for prefix in prefixes:
        metrics.append(alignment_metrics(name,seed,folder,prefix))
        for condition in CONFIG['conditions']:
            _,pack=cell(name,seed,condition)
            targets=dict(CE=F.one_hot(pack['observed'],data(name)['classes']).float(),
                         SUPPORT=historical_target(name,seed,condition))
            for method,target in targets.items():
                arm=f'P{prefix}_{method}'
                fit(location('E2',name,seed,condition,arm),name,seed,condition,arm,target,100,
                    encoder=folder/f'prefix{prefix}.pt',frozen=True)
    save_csv(folder/'metrics.csv',metrics)


def e2():
    decisions=[]
    for name in CONFIG['pilot']['E2_datasets']:
        e2_heads(name,0,[100,300,600],600)
        gains=[]
        for prefix in (300,600):
            per_arm={}
            for method in ('CE','SUPPORT'):
                deltas=[]
                for cond in CONFIG['conditions']:
                    a=read(location('E2',name,0,cond,f'P100_{method}')/'result.json')['validation']
                    b=read(location('E2',name,0,cond,f'P{prefix}_{method}')/'result.json')['validation']
                    deltas.append(100*(b-a))
                per_arm[method]=float(np.mean(deltas))
            gain=float(np.mean(list(per_arm.values())))
            gains.append(dict(prefix=prefix,mean_validation_gain_pp=gain,per_arm=per_arm,
                              passed=gain>=1 and min(per_arm.values())>=0))
        chosen=next((r['prefix'] for r in gains if r['passed']),100)
        decisions.append(dict(dataset=name,chosen_prefix=chosen,candidates=gains,validation_only=True))
        save_json(ROOT/'E2_decision.json',decisions)
        print('E2 GATE '+json.dumps(decisions[-1]),flush=True)
        if chosen>100:
            for seed in (1,2):
                e2_heads(name,seed,[100,chosen],chosen)


def e4_cells(seeds):
    for name in CONFIG['datasets']:
        for seed in seeds:
            enc=historical_encoder(name,seed)
            for condition in CONFIG['conditions']:
                prob,tfolder=teacher(name,seed,condition)
                target=historical_target(name,seed,condition)
                results={}
                for arm in CONFIG['pilot']['E4_arms']:
                    initialized=arm not in ('A200','A300')
                    kd=arm in ('A200','A300','B','D')
                    frozen=arm in ('D','C_head')
                    results[arm]=fit(location('E4',name,seed,condition,arm),name,seed,condition,arm,
                        prob if kd else target,300 if arm=='A300' else 200,
                        encoder=enc if initialized else None,frozen=frozen,all_nodes=kd)
                if len({results[a]['initial_hash'] for a in ('B','D','C_full','C_head')})!=1:
                    raise RuntimeError('R2 initial encoder/head mismatch')
                a=(location('E4',name,seed,condition,'A200')/'epochs.jsonl').read_text().splitlines()
                b=(location('E4',name,seed,condition,'A300')/'epochs.jsonl').read_text().splitlines()[:200]
                if a!=b:
                    raise RuntimeError('A200/A300 first-200 trajectory mismatch')
                save_json(location('E4',name,seed,condition,'pair_audit.json'),dict(
                    initial_R2_exact=True,A200_A300_prefix_exact=True,teacher_path=str(tfolder),
                    teacher_sha256=sha(tfolder/'probabilities.pt')))


def e4_decision():
    rows=[]
    for name in CONFIG['datasets']:
        for cond in CONFIG['conditions']:
            results={a:read(location('E4',name,0,cond,a)/'result.json') for a in CONFIG['pilot']['E4_arms']}
            base=max(('A200','A300','C_full','C_head'),key=lambda a:results[a]['validation'])
            rows.append(dict(dataset=name,condition=cond,comparator=base,
                B=results['B']['validation'],D=results['D']['validation'],baseline=results[base]['validation']))
    selected=max(('B','D'),key=lambda a:np.mean([r[a] for r in rows]))
    noisy={n:float(np.mean([100*(r[selected]-r['baseline']) for r in rows if r['dataset']==n and r['condition']!='clean'])) for n in CONFIG['datasets']}
    clean={n:float(np.mean([100*(r[selected]-r['baseline']) for r in rows if r['dataset']==n and r['condition']=='clean'])) for n in CONFIG['datasets']}
    passed=sum(v>=1 for v in noisy.values())>=3 and min(clean.values())>=-1
    decision=dict(selected_candidate=selected,validation_only=True,passed=passed,
        noisy_gain_pp=noisy,clean_gain_pp=clean,selections=rows,decision='expand seeds1/2' if passed else 'stop E4 expansion')
    save_json(ROOT/'E4_decision.json',decision)
    print('E4 GATE '+json.dumps({k:v for k,v in decision.items() if k!='selections'}),flush=True)
    return decision


def e4():
    e4_cells([0])
    if e4_decision()['passed']:
        e4_cells([1,2])


def main(stage):
    lock=ROOT/'fit.lock'
    with lock.open('x',encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        initialize()
        smoke()
        for name,func in [('E1',e1),('E2',e2),('E4',e4)]:
            if stage not in ('all',name):
                continue
            done=ROOT/f'{name}_fits_complete.json'
            if not done.exists():
                save_json(ROOT/'fit_status.json',dict(stage=name,status='running',pid=os.getpid()))
                func()
                outputs={str(p):sha(p) for p in (ROOT/'fits'/name).glob('**/result.json')}
                save_json(done,dict(completed=datetime.now().astimezone().isoformat(),results=outputs))
            print(name+' FITS FINISHED',flush=True)
        save_json(ROOT/'fit_status.json',dict(stage=stage,status='complete',pid=os.getpid()))
    except Exception:
        failure=ROOT/f'failure_{datetime.now():%Y%m%d_%H%M%S}.json'
        save_json(failure,dict(stage=stage,traceback=traceback.format_exc()))
        save_json(ROOT/'fit_status.json',dict(stage=stage,status='failed',failure=str(failure)))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','E1','E2','E4'],default='all')
    args=parser.parse_args()
    main(args.stage)
