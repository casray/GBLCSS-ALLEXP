"""R2-only joint race with explicit retrospective uniform-prefix replay."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from protocol import (ROOT,EXP,CONFIG,INPUTS,PILOT,read,save,sha,track,load,now,append,
    graph_inputs,cell_inputs,encoder_path,state_hash,training_data,freeze_sources,
    canonical,disk_guard,csv_save,tree_hash,import_file)
from targets import evidence,shrink,groups
from fit import fit,Student,HIDDEN
from rules import candidates,weights,rank_key,prefix_score,survivors,require_joint_frozen

OLD=EXP/'d1_c1_confirmation_20261024'
GRID=candidates()
REFS,PHYSICAL,VERIFIED={},{},set()
original_selection=import_file('integrated_original_selection',PILOT/'selection.py')


def status(stage,**kwargs):
    v=dict(stage=stage,time=now(),**kwargs)
    save(ROOT/'status.json',v)
    append(ROOT/'events.jsonl',v)
    (ROOT/'HANDOFF.md').write_text('# R2 integrated execution\n\n'+json.dumps(v,indent=2)+
        '\n\n60 R2 cells; old directories read-only. No automatic further experiments.\n'
        'Head checkpoints require encoder_path + encoder_file_sha256.\n'
        'Uniform prefixes are retrospective; new D1 tie=1. Decisions precede fixed controls and test scoring.\n',encoding='utf-8')
    print(json.dumps(v),flush=True)


def cellid(n,s,c):
    return f'{n}/seed{s}/R2/{c}'


def oldfolder(n,s,c,arm):
    if arm in GRID:
        assert GRID[arm]['kind']=='uniform'
        return OLD/'fits/C1'/n/f'seed{s}'/'R2'/c/arm
    names={'CE':'CE','GCE':'GCE_Q0.7','LEGACY_SUPPORT32':'SUPPORT_L1.0'}
    return OLD/'fits/D1'/n/f'seed{s}'/'R2'/c/names[arm]


def prepare(n,s,c,g):
    pack=cell_inputs(n,s,c)
    geo=g['geometry']
    v=training_data(dict(x=g['x'],train=geo['train'],valid=geo['valid'],
        observed=pack['observed'],valid_y=pack['valid_y'],classes=pack['targets'][32].shape[1]))
    parts={str(k):p for k,p in geo['partitions'].items()}
    parts['LEGACY']=geo['legacy32']
    with np.load(track(EXP/'main_protocol_replay_20260922/matrix'/n/'default'/f'seed{s}.npz')) as data:
        parts['OLD64']=torch.from_numpy(data['assignment']).long()
    es={k:evidence(v['observed'],p[geo['train']],v['classes']) for k,p in parts.items()}
    assert max(float((es[str(k)]['native']-pack['targets'][k]).abs().max()) for k in CONFIG['supports'])<3e-7
    return v,es,parts


def gpu(v):
    return {k:x.cuda() if isinstance(x,torch.Tensor) else x for k,x in v.items()}


def ihash(v):
    return state_hash({k:x for k,x in v.items() if isinstance(x,torch.Tensor)})


def getlogs(folder):
    return [json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()]


def verify_old(n,s,c,arm,target,v):
    folder=oldfolder(n,s,c,arm)
    r=read(track(folder/'result.json'))
    assert r['target_hash']==state_hash({'target':target})
    assert r['input_hash']==ihash(v)
    assert r['encoder_file_sha256']==sha(encoder_path(n,s))
    assert r['epoch']==200 and r['backbone']=='R2' and r['encoder_frozen_unchanged']
    assert r['source_sha256']==sha(OLD/'sources.json')
    assert (r['optimizer'],r['lr'],r['weight_decay'],r['selection'])==('Adam',.01,.0005,'first_strict_max_validation_accuracy')
    ce=read(oldfolder(n,s,c,'CE')/'result.json')
    assert (r['initial_hash'],r['initial_rng_hash'],r['encoder_hash'])==(ce['initial_hash'],ce['initial_rng_hash'],ce['encoder_hash'])
    if str(folder) not in VERIFIED:
        for name,key in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            assert sha(track(folder/name))==r[key]
        logs=getlogs(folder)
        track(folder/'epochs.jsonl')
        assert [x['epoch'] for x in logs]==list(range(1,201))
        best=max(logs,key=lambda x:x['validation_correct'])
        assert (r['best_epoch'],r['valid_correct'],r['valid_nll'])==(best['epoch'],best['validation_correct'],best['validation_nll'])
        assert state_hash({'target':load(OLD/r['target_ref'])})==r['target_hash']
        VERIFIED.add(str(folder))
    return folder,r


def register(n,s,c,arm,folder,partition,historical):
    r=read(folder/'result.json')
    key=cellid(n,s,c)+'/'+arm
    record=dict(dataset=n,seed=s,backbone='R2',condition=c,arm=arm,partition=str(partition),
        folder=str(folder.resolve()),historical=historical,epoch=r['epoch'],
        target_path=str(((OLD if historical else ROOT)/r['target_ref']).resolve()),
        target_hash=r['target_hash'],input_hash=r['input_hash'],initial_hash=r['initial_hash'],
        source_sha256=r['source_sha256'],encoder_hash=r['encoder_hash'],
        result_sha256=sha(folder/'result.json'),checkpoint_format=r['checkpoint_format'])
    REFS[key]=record
    save(ROOT/'reuse_manifest.json',REFS)
    return r


def target_for(arm,es):
    if arm in GRID:
        spec=GRID[arm]
        e=es[str(spec['support'])]
        return shrink(e,weights(e,spec)),str(spec['support'])
    p='OLD64' if arm=='OLD64' else 'LEGACY'
    return (es[p]['own'] if arm in ('CE','GCE') else shrink(es[p],1)),p


def metadata_score(arm,es):
    if arm=='CE':
        return 0,0.
    spec=GRID[arm]
    return spec['support'],float(weights(es[str(spec['support'])],spec).mean())


def obtain(n,s,c,arm,until,v,es,g):
    target,partition=target_for(arm,es)
    historical=arm not in GRID or GRID[arm]['kind']=='uniform'
    historical=historical and arm!='OLD64'
    if historical:
        folder,r=verify_old(n,s,c,arm,target,v)
        register(n,s,c,arm,folder,partition,True)
    else:
        folder=ROOT/'fits'/cellid(n,s,c)/arm
        before=read(folder/'result.json')['epoch'] if (folder/'result.json').exists() else 0
        if before<until:
            status('TRAINING',dataset=n,seed=s,condition=c,arm=arm,until=until,physical=len(PHYSICAL))
        enc=encoder_path(n,s)
        r=fit(folder,'R2',v,target,g['encoder'],seed=s,until=until,
            metadata=dict(dataset=n,condition=c,arm=arm,stage='INTEGRATED',partition=partition,
                encoder_path=str(enc.resolve()),encoder_file_sha256=sha(enc)))
        ce=read(oldfolder(n,s,c,'CE')/'result.json')
        assert r['initial_hash']==ce['initial_hash'] and r['initial_rng_hash']==ce['initial_rng_hash']
        if until<200 and before<until:
            snap=folder/'prefixes'/str(until)
            snap.mkdir(parents=True,exist_ok=True)
            for filename in ('resume.pt','result.json'):
                shutil.copy2(folder/filename,snap/filename)
            save(snap/'manifest.json',dict(epoch=until,finished=now(),resume_sha256=sha(snap/'resume.pt'),
                result_sha256=sha(snap/'result.json'),log_prefix_sha256=sha(folder/'epochs.jsonl')))
        PHYSICAL[canonical(folder)]=dict(dataset=n,seed=s,condition=c,arm=arm,epoch=r['epoch'],seconds=r['seconds'])
        assert len(PHYSICAL)<=CONFIG['scientific_new_unique_upper']
        save(ROOT/'physical_fits.json',PHYSICAL)
        register(n,s,c,arm,folder,partition,False)
    if arm=='OLD64' or arm in ('GCE','LEGACY_SUPPORT32'):
        return r
    support,mean=metadata_score(arm,es)
    return prefix_score(getlogs(folder),until,arm,support,mean,len(v['valid']))


def original_row(row):
    return {('strength' if k=='mean_lambda' else k):v for k,v in row.items()}


def race(n,s,c,v,es,g,kind):
    path=ROOT/'races'/cellid(n,s,c)/(kind+'.json')
    if path.exists():
        return read(path)
    arms=list(GRID) if kind=='JOINT' else [a for a,d in GRID.items() if d['kind']=='uniform']
    rounds=[]
    for prefix,count in CONFIG['joint_race' if kind=='JOINT' else 'global_race']:
        rp=path.with_name(kind+f'_prefix{prefix}.json')
        if rp.exists():
            rd=read(rp)
        else:
            rows=[obtain(n,s,c,a,prefix,v,es,g) for a in arms]
            if prefix<200:
                kept=survivors(rows,count,prefix) if kind=='JOINT' else original_selection.probe_survivors([original_row(r) for r in rows],count,prefix)
            else:
                rows.append(obtain(n,s,c,'CE',200,v,es,g))
                winner=min(rows,key=rank_key if kind=='JOINT' else lambda r:original_selection.rank_key(original_row(r)))['arm']
                kept=[winner]
            rd=dict(prefix=prefix,scores=rows,survivors=kept,frozen=now(),
                retrospective_uniform_replay=True,new_groups='own optimizer/RNG real continuation')
            save(rp,rd)
        rounds.append(rd)
        arms=rd['survivors']
    result=dict(dataset=n,seed=s,condition=c,kind=kind,selected=arms[0],rounds=rounds,
        candidate_updates=2250 if kind=='JOINT' else 1000,ce_updates=200,frozen=now())
    if kind=='GLOBAL':
        prior=read(track(OLD/'selections/C1'/n/f'seed{s}'/'R2'/f'{c}.json'))
        assert result['selected']==prior['PROBE'],'Original C1 selected arm changed'
        assert all(a['survivors']==b['survivors'] for a,b in zip(rounds[:2],prior['rounds']))
        result['exact_original_selection_bridge']=True
    save(path,result)
    return result


def train():
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            for c in CONFIG['conditions']:
                selection=ROOT/'selections'/n/f'seed{s}'/f'{c}.json'
                if selection.exists():
                    continue
                cpu,es,_=prepare(n,s,c,g)
                v=gpu(cpu)
                gl=race(n,s,c,v,es,g,'GLOBAL')
                jo=race(n,s,c,v,es,g,'JOINT')
                jp=ROOT/'races'/cellid(n,s,c)/'JOINT.json'
                require_joint_frozen(jp)
                joint_hash=sha(jp)
                fixed=['CE']+[a for a,d in GRID.items() if d['support']==32]
                full=[obtain(n,s,c,a,200,v,es,g) for a in fixed]
                for a in ('GCE','LEGACY_SUPPORT32','OLD64'):
                    obtain(n,s,c,a,200,v,es,g)
                assert sha(jp)==joint_hash
                fixedgl=[r for r in full if r['arm']=='CE' or GRID[r['arm']]['kind']=='uniform']
                fx=min(full,key=rank_key)
                ag=next(r for r in gl['rounds'][-1]['scores'] if r['arm']==gl['selected'])
                choices=dict(CE='CE',GCE='GCE',OLD64='OLD64',LEGACY_SUPPORT32='LEGACY_SUPPORT32',
                    FIXED_GLOBAL=min(fixedgl,key=rank_key)['arm'],FIXED_EXPANDED=fx['arm'],
                    ADAPTIVE_GLOBAL=gl['selected'],ADAPTIVE_EXPANDED=jo['selected'],
                    SINGLE_MODULE_VAL=min([fx,ag],key=rank_key)['arm'],FIXED_S32='S32_L1.0')
                save(selection,dict(dataset=n,seed=s,backbone='R2',condition=c,frozen=now(),
                    choices={k:cellid(n,s,c)+'/'+a for k,a in choices.items()},fixed_scores=full,
                    joint_sha256=joint_hash,global_sha256=sha(jp.with_name('GLOBAL.json')),
                    fixed_controls_after_joint=True,rank='rules.rank_key; GLOBAL original pilot rank',
                    joint_candidate_updates=2250,global_candidate_updates=1000,ce_updates=200,
                    fixed_global_updates=800,fixed_expanded_updates=1600))
                status('CELL_COMPLETE',dataset=n,seed=s,condition=c,cells=len(list((ROOT/'selections').rglob('*.json'))))
            HIDDEN.clear()
            torch.cuda.empty_cache()
    save(ROOT/'selections_frozen.json',dict(frozen=now(),choices={canonical(p):sha(p) for p in (ROOT/'selections').rglob('*.json')},
        races={canonical(p):sha(p) for p in (ROOT/'races').rglob('*.json')},reuse_manifest_sha256=sha(ROOT/'reuse_manifest.json'),
        retrospective=True,test_access_before_freeze=False))


def preflight():
    status('PREFLIGHT')
    assert read(track(OLD/'complete.json'))['audit']=='passed'
    assert sha(ROOT/'fit.py')==sha(OLD/'fit.py')
    for manifest in ('sources.json','inputs.json'):
        for path,h in read(track(OLD/manifest)).items():
            assert sha(path)==h,path
    passports,profiles,details=[],[],[]
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            passport=next(p for p in read(track(OLD/'passports.json')) if p['dataset']==n and p['seed']==s)
            assert g['h_bridge']==passport['h_normalization']
            passports.append({k:x for k,x in passport.items() if not k.startswith('km')})
            for c in CONFIG['conditions']:
                v,es,parts=prepare(n,s,c,g)
                for arm in ['CE','GCE','LEGACY_SUPPORT32']+[a for a,d in GRID.items() if d['kind']=='uniform']:
                    t,p=target_for(arm,es)
                    folder,_=verify_old(n,s,c,arm,t,v)
                    register(n,s,c,arm,folder,p,True)
                for k,e in es.items():
                    profiles.append(dict(dataset=n,seed=s,condition=c,partition=k,regions=len(torch.unique(parts[k])),
                        mean_support=float(e['support'].mean()),min_support=float(e['support'].min()),
                        max_support=float(e['support'].max()),tied_fraction=float(e['tied'].double().mean()),
                        group_counts=[int((groups(e)==i).sum()) for i in range(3)]))
                ce=oldfolder(n,s,c,'CE')
                head=(ce/'checkpoint.pt').stat().st_size
                probs=(ce/'probabilities.pt').stat().st_size
                targetbytes=4*len(v['train'])*v['classes']
                projected=25*(probs+targetbytes+head*9+180000)
                details.append(dict(dataset=n,seed=s,condition=c,projected_bytes=projected))
    estimated=sum(x['projected_bytes'] for x in details)+512*1024**2
    free=disk_guard()
    save(ROOT/'storage_estimate.json',dict(free_bytes=free,projected_bytes=estimated,reserve_bytes=3*1024**3,
        strategy=CONFIG['storage'],details=details))
    assert free-estimated>=3*1024**3,'Storage estimate exceeds reserve'
    save(ROOT/'passports.json',passports)
    save(ROOT/'support_profiles.json',profiles)
    save(ROOT/'preflight_complete.json',dict(finished=now(),cells=60,verified_historical_folders=len(VERIFIED),
        new_fit_upper=1500,old64_refit_reason='old schema lacks full input/RNG contract',retrospective=True))


def smoke():
    checks=[]
    for n,s,pair in [('cora',0,True),('corafull',1,False),('coauthorcs',2,False)]:
        g=graph_inputs(n,s)
        v,es,_=prepare(n,s,'uniform0.4',g)
        v=gpu(v)
        target,_=target_for('S32_G_A0.0_C1.0',es)
        folder=ROOT/'smoke'/f'{n}_seed{s}'
        meta=dict(dataset=n,condition='uniform0.4',encoder_path=str(encoder_path(n,s)),encoder_file_sha256=sha(encoder_path(n,s)))
        r=fit(folder,'R2',v,target,g['encoder'],seed=s,until=2,metadata=meta)
        ce=read(oldfolder(n,s,'uniform0.4','CE')/'result.json')
        assert r['initial_hash']==ce['initial_hash'] and r['initial_rng_hash']==ce['initial_rng_hash']
        if pair:
            other=folder.with_name(folder.name+'_resume')
            for ep in [1,2]:
                fit(other,'R2',v,target,g['encoder'],seed=s,until=ep,metadata=meta)
            a=torch.load(folder/'resume.pt',weights_only=False)
            b=torch.load(other/'resume.pt',weights_only=False)
            assert all(tree_hash(a[k])==tree_hash(b[k]) for k in ('model','optimizer','rng','best_state'))
        st=Student('R2',v,g['encoder'],s)
        ck=torch.load(folder/'checkpoint.pt',weights_only=True)
        st.set_state(ck['model']);st.model.eval();st.check_frozen()
        with torch.no_grad():
            diff=float((st.forward().softmax(1).cpu()-torch.load(folder/'probabilities.pt',weights_only=True)).abs().max())
        assert diff==0
        checks.append(dict(dataset=n,seed=s,own_resume_pair=pair,paired_initialization=True,frozen_BN=True,reload_maxerror=diff))
        HIDDEN.clear()
    save(ROOT/'smoke_complete.json',dict(finished=now(),fits=4,epochs=2,checks=checks))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if (ROOT/'complete.json').exists():
        print('Complete; no training');return
    torch.set_num_threads(1)
    lock=ROOT/'run.lock'
    with lock.open('x') as f:
        json.dump(dict(pid=os.getpid(),started=now()),f)
    try:
        for filename,target in [('inputs.json',INPUTS),('reuse_manifest.json',REFS),('physical_fits.json',PHYSICAL)]:
            if (ROOT/filename).exists():
                target.update(read(ROOT/filename))
        freeze_sources()
        if not (ROOT/'preflight_complete.json').exists():
            preflight()
        if args.preflight:return
        while not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*1024**3:
            status('GPU_WAIT');time.sleep(60)
        save(ROOT/'environment.json',dict(python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            free_total_bytes=list(torch.cuda.mem_get_info()),pid=os.getpid(),started=now()))
        if not (ROOT/'smoke_complete.json').exists():smoke()
        if args.smoke:return
        if not (ROOT/'selections_frozen.json').exists():train()
        from analyze import analyze
        from audit import audit
        from report import report
        status('ANALYSIS');analyze()
        status('AUDIT');audit()
        report()
        save(ROOT/'complete.json',dict(finished=now(),scientific_unique=len(PHYSICAL),
            scientific_updates=sum(r['epoch'] for r in PHYSICAL.values()),smoke_fits=4,audit='passed'))
        status('COMPLETE',scientific_unique=len(PHYSICAL),audit='passed')
    except BaseException:
        failure=dict(time=now(),traceback=traceback.format_exc())
        append(ROOT/'failures.jsonl',failure);save(ROOT/'failure_latest.json',failure)
        status('INTERRUPTED',failure='failure_latest.json')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    sys.modules['run']=sys.modules[__name__]
    main()
