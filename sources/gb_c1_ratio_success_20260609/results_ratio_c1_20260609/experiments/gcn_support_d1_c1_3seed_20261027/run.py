"""Exact GCN supervision pilot extension; frozen old nine cells stay read-only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parent
EXP=CODE.parent
CONFIG=json.loads((CODE/'config.json').read_text(encoding='utf-8'))
ROOT=Path(CONFIG['runtime']);OLD=EXP/CONFIG['historical']


def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


p=imported('gcn_expansion_protocol',OLD/'protocol.py')
p.ROOT,p.CONFIG,p.INPUTS=ROOT,CONFIG,{}
sys.modules['protocol']=p
targets=imported('targets',OLD/'targets.py')
rules=imported('gcn_expansion_rules',OLD/'rules.py')
kernel=imported('kernels',OLD/'kernels.py')
fitmodule=imported('gcn_expansion_fit',OLD/'fit.py')
import numpy as np
import torch

GRID=rules.candidates();REFS={};PHYSICAL={}


def cid(n,s,c):return f'{n}/seed{s}/{c}'
def cells():return [(n,s,c) for n in CONFIG['datasets'] for s in CONFIG['seeds'] for c in CONFIG['conditions']]
def historical(n,s,c):return n in ('cora','citeseer','corafull') and s==0
def gpu(v):return {k:x.cuda() if isinstance(x,torch.Tensor) else x for k,x in v.items()}


def status(stage,**kw):
    row=dict(stage=stage,time=p.now(),pid=os.getpid(),**kw)
    p.save(ROOT/'status.json',row);p.append(ROOT/'events.jsonl',row)
    p.save(CODE/'status.json',row) if CODE==ROOT else (CODE/'status.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)


def freeze_sources():
    files=[CODE/'run.py',CODE/'launch.py',CODE/'config.json',CODE/'PLAN.md',*CODE.glob('tests/test_contract.py'),
        OLD/'protocol.py',OLD/'rules.py',OLD/'targets.py',OLD/'fit.py',OLD/'kernels.py',
        Path(sys.modules['core'].__file__),Path(p.common.__file__),Path(targets._kernel.__file__),
        Path(sys.modules['gblcss.models'].__file__),Path(sys.modules['gblcss.data'].__file__),
        EXP/'gb_strong_kd_screen_20261007/losses.py',*list((EXP/'gb_strong_kd_screen_20261007/vendor').glob('*.py'))]
    value={str(x.resolve()):p.sha(x) for x in files}
    if (ROOT/'sources.json').exists():assert p.read(ROOT/'sources.json')==value,'Frozen training source changed'
    else:p.save(ROOT/'sources.json',value)
    if not (ROOT/'config.resolved.json').exists():p.save(ROOT/'config.resolved.json',CONFIG)


def prepare(n,s,c,g):
    pack=p.cell_inputs(n,s,c);geo=g['geometry']
    v=p.training_data(dict(x=g['x'],train=geo['train'],valid=geo['valid'],observed=pack['observed'],
        valid_y=pack['valid_y'],classes=pack['targets'][32].shape[1]))
    parts={str(k):a for k,a in geo['partitions'].items()};parts['LEGACY']=geo['legacy32']
    with np.load(p.track(EXP/'main_protocol_replay_20260922/matrix'/n/'default'/f'seed{s}.npz')) as data:
        assert np.array_equal(data['train_index'],geo['train'].numpy())
        parts['OLD64']=torch.from_numpy(data['assignment']).long()
    es={k:targets.evidence(v['observed'],a[geo['train']],v['classes']) for k,a in parts.items()}
    assert max(float((es[str(k)]['native']-pack['targets'][k]).abs().max()) for k in CONFIG['supports'])<3e-7
    baselines={a:targets.shrink(es[k],1) for a,k in [('OLD64','OLD64'),('LEGACY_SUPPORT32','LEGACY')]}
    previous=EXP/'teacher_support_lcss_20261022/inputs'/n/f'seed{s}'/f'{c}.pt'
    bridge={}
    if previous.exists():
        cached=p.load(previous)
        assert torch.equal(cached['observed_train'],v['observed'])
        assert torch.equal(cached['train_index'],v['train']) and torch.equal(cached['valid_index'],v['valid'])
        for a,b in [('OLD64','OLD_LCSS'),('LEGACY_SUPPORT32','SUPPORT_LCSS')]:
            bridge[a]=float((baselines[a]-cached['targets'][b]).abs().max());assert bridge[a]<3e-7
            baselines[a]=cached['targets'][b].float()
    passport=dict(dataset=n,seed=s,condition=c,input_hash=p.state_hash({k:x for k,x in v.items() if isinstance(x,torch.Tensor)}),
        split_hash=p.state_hash({k:geo[k] for k in ('train','valid','test')}),target_bridge=bridge,noise_regenerated=False)
    path=ROOT/'passports'/n/f'seed{s}'/f'{c}.json'
    if path.exists():assert p.read(path)==passport
    else:p.save(path,passport)
    return v,es,baselines


def context(g):
    edges=g['edge_index'].cuda();n=len(g['x'])
    return dict(adjacency=torch.sparse_coo_tensor(edges,torch.ones(edges.shape[1],device='cuda'),(n,n)).coalesce())


def target_for(arm,es,baselines):
    if arm=='CE':return es['LEGACY']['own'],'LEGACY'
    if arm in baselines:return baselines[arm],'OLD64' if arm=='OLD64' else 'LEGACY'
    spec=GRID[arm];e=es[str(spec['support'])]
    return targets.shrink(e,rules.weights(e,spec)),str(spec['support'])


def register(folder,n,s,c,arm,partition,kind):
    result=p.read(folder/'result.json');key=f'{cid(n,s,c)}/{kind}/{arm}'
    REFS[key]=dict(dataset=n,seed=s,condition=c,kind=kind,arm=arm,partition=partition,
        folder=str(folder.resolve()),target_path=str((ROOT/result['target_ref']).resolve()),epoch=result['epoch'],
        result_sha256=p.sha(folder/'result.json'),initial_hash=result['initial_hash'],initial_rng_hash=result['initial_rng_hash'],historical=False)
    PHYSICAL[p.canonical(folder)]=dict(dataset=n,seed=s,condition=c,kind=kind,arm=arm,epoch=result['epoch'],seconds=result['seconds'])
    p.save(ROOT/'refs.json',REFS);p.save(ROOT/'physical_fits.json',PHYSICAL)
    return key,result


def logs(folder):return [json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()]


def teacher(n,s,c,arm,until,v,es,baselines,ctx):
    t,part=target_for(arm,es,baselines);folder=ROOT/'fits/teacher'/cid(n,s,c)/arm
    before=p.read(folder/'result.json')['epoch'] if (folder/'result.json').exists() else 0
    result=fitmodule.fit(folder,'GCN',v,t,ctx,seed=s,until=until,
        metadata=dict(dataset=n,condition=c,arm=arm,stage='TEACHER',partition=part,
            adjacency_hash=p.state_hash({'indices':ctx['adjacency'].indices(),'values':ctx['adjacency'].values()})))
    if before<until and until<200:
        snap=ROOT/'prefix_snapshots'/cid(n,s,c)/arm/str(until);snap.mkdir(parents=True,exist_ok=True)
        for file in ('resume.pt','result.json'):
            dest=snap/file
            if dest.exists():assert p.sha(dest)==p.sha(folder/file)
            else:shutil.copy2(folder/file,dest)
        p.save(snap/'manifest.json',dict(prefix=until,resume_sha256=p.sha(snap/'resume.pt'),result_sha256=p.sha(snap/'result.json')))
    register(folder,n,s,c,arm,part,'teacher')
    if arm not in GRID:return None
    spec=GRID[arm]
    return rules.prefix_score(logs(folder),until,arm,spec['support'],float(rules.weights(es[part],spec).mean()),len(v['valid']))


def ce_score(n,s,c,v):return rules.prefix_score(logs(ROOT/'fits/teacher'/cid(n,s,c)/'CE'),200,'CE',0,0.,len(v['valid']))


def race(n,s,c,v,es,baselines,ctx):
    folder=ROOT/'races'/cid(n,s,c);path=folder/'joint.json'
    if path.exists():return p.read(path)
    arms=list(GRID);rounds=[]
    for prefix,count in CONFIG['joint_race']:
        rp=folder/f'prefix{prefix}.json'
        if rp.exists():rd=p.read(rp)
        else:
            scores=[teacher(n,s,c,a,prefix,v,es,baselines,ctx) for a in arms]
            if prefix<200:kept=rules.survivors(scores,count,prefix)
            else:scores.append(ce_score(n,s,c,v));kept=[min(scores,key=rules.rank_key)['arm']]
            rd=dict(prefix=prefix,scores=scores,survivors=kept,frozen=p.now());p.save(rp,rd)
        rounds.append(rd);arms=rd['survivors']
    result=dict(dataset=n,seed=s,condition=c,selected=arms[0],rounds=rounds,frozen=p.now(),candidate_updates=2250)
    p.save(path,result);return result


def reuse_old():
    frozen=p.read(p.track(OLD/'selections_frozen.json'));oldrefs=p.read(p.track(OLD/'refs.json'))
    assert p.sha(OLD/'refs.json')==frozen['refs_sha256']
    assert p.sha(OLD/'teachers_frozen.json')==frozen['teacher_freeze_sha256']
    for name,digest in frozen['choices'].items():assert p.sha(p.track(OLD/name))==digest
    for key,ref in oldrefs.items():
        assert p.sha(p.track(Path(ref['folder'])/'result.json'))==ref['result_sha256']
        REFS[key]=dict(ref,historical=True)
    for n,s,c in cells():
        if not historical(n,s,c):continue
        for kind in ('teacher','student'):
            src=OLD/'selections'/kind/n/f'{c}.json';value=p.read(p.track(src))
            dest=ROOT/'selections'/kind/n/'seed0'/f'{c}.json'
            if dest.exists():assert p.read(dest)==value
            else:p.save(dest,value)
            for key in value['choices'].values():
                ref=REFS[key];result=p.read(Path(ref['folder'])/'result.json')
                for file,field in [('probabilities.pt','probabilities_sha256'),('checkpoint.pt','checkpoint_sha256')]:
                    assert p.sha(p.track(Path(ref['folder'])/file))==result[field]
    p.save(ROOT/'refs.json',REFS)
    p.save(ROOT/'historical_manifest.json',dict(source=str(OLD),old_selection_freeze_sha256=p.sha(OLD/'selections_frozen.json'),
        source_refs_sha256=p.sha(OLD/'refs.json'),cells=9,logical_students=54,physical_manifest=p.read(p.track(OLD/'physical_fits.json'))))


def preflight():
    if (ROOT/'preflight_complete.json').exists():return
    reuse_old();checks=[]
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=p.graph_inputs(n,s)
            for c in CONFIG['conditions']:
                v,es,b=prepare(n,s,c,g)
                checks.append(dict(dataset=n,seed=s,condition=c,nodes=len(v['x']),classes=v['classes'],historical=historical(n,s,c)))
    assert len(checks)==45
    p.disk_guard(24*1024**3)
    p.save(ROOT/'preflight_complete.json',dict(time=p.now(),cells=45,new_cells=36,checks=checks,teacher_trajectories=1620,logical_new_students=216))
    status('PREFLIGHT_COMPLETE',new_teacher_cells=36)


def smoke():
    if (ROOT/'smoke_complete.json').exists():return
    checks=[]
    for n,s,kind in [('cora',1,'GCN'),('cora',1,'DIST'),('coauthorcs',0,'GCN')]:
        g=p.graph_inputs(n,s);cpu,es,b=prepare(n,s,'uniform0.4',g);v=gpu(cpu)
        ctx=context(g) if kind=='GCN' else dict(teacher=torch.ones((len(v['x']),v['classes']),device='cuda')/v['classes'])
        t,_=target_for('S32_G_A0.0_C1.0',es,b)
        folder=ROOT/'smoke'/f'{n}_seed{s}_{kind}'
        fitmodule.fit(folder,kind,v,t,ctx,seed=s,until=2,metadata=dict(dataset=n,condition='uniform0.4',arm=kind))
        if n=='cora':
            other=folder.with_name(folder.name+'_resume')
            for ep in (1,2):fitmodule.fit(other,kind,v,t,ctx,seed=s,until=ep,metadata=dict(dataset=n,condition='uniform0.4',arm=kind))
            a,z=[torch.load(f/'resume.pt',map_location='cpu',weights_only=False) for f in (folder,other)]
            assert all(p.tree_hash(a[k])==p.tree_hash(z[k]) for k in ('model','optimizer','rng','best_state'))
        checks.append(dict(dataset=n,seed=s,kind=kind,resume_exact=n=='cora'))
    g=p.graph_inputs('cora',0);cpu,es,b=prepare('cora',0,'uniform0.4',g);v=gpu(cpu)
    for kind,arm in [('teacher','CE'),('student','JOINT')]:
        choice=p.read(ROOT/'selections'/kind/'cora/seed0/uniform0.4.json');ref=REFS[choice['choices'][arm]]
        oldfolder=Path(ref['folder']);result=p.read(oldfolder/'result.json')
        if kind=='teacher':ctx=context(g)
        else:
            tr=REFS[choice['teacher_choices'][arm]]
            ctx=dict(teacher=p.load(Path(tr['folder'])/'probabilities.pt').cuda().detach())
        k=kernel.Kernel('GCN' if kind=='teacher' else 'DIST',v,ctx,0)
        assert k.initial_hash==result['initial_hash'] and k.initial_rng==result['initial_rng_hash']
        for row in logs(oldfolder)[:2]:
            k.model.train();k.optimizer.zero_grad(set_to_none=True)
            value=k.objective(k.forward()) if kind=='student' else -(es['LEGACY']['own'].cuda()*k.forward(v['train']).log_softmax(1)).sum(1).mean()
            value.backward();k.optimizer.step();k.model.eval()
            with torch.no_grad():correct=int((k.forward(v['valid']).argmax(1)==v['valid_y']).sum())
            assert float(value.detach())==row['loss'] and correct==row['validation_correct']
    p.save(ROOT/'smoke_complete.json',dict(time=p.now(),checks=checks,trajectories=5,old_initialization_and_two_epoch_bridge_exact=True))
    status('SMOKE_COMPLETE')


def train_teachers():
    if (ROOT/'teachers_frozen.json').exists():return
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=p.graph_inputs(n,s);ctx=context(g)
            for c in CONFIG['conditions']:
                choice=ROOT/'selections/teacher'/n/f'seed{s}'/f'{c}.json'
                if choice.exists():continue
                status('TEACHER_CELL',dataset=n,seed=s,condition=c,complete=len(list((ROOT/'selections/teacher').rglob('*.json'))))
                cpu,es,b=prepare(n,s,c,g);v=gpu(cpu)
                teacher(n,s,c,'CE',200,v,es,b,ctx)
                selected=race(n,s,c,v,es,b,ctx);jp=ROOT/'races'/cid(n,s,c)/'joint.json';digest=p.sha(jp)
                fixed=[ce_score(n,s,c,v)]+[teacher(n,s,c,a,200,v,es,b,ctx) for a,spec in GRID.items() if spec['support']==32]
                for a in ('OLD64','LEGACY_SUPPORT32'):teacher(n,s,c,a,200,v,es,b,ctx)
                assert p.sha(jp)==digest
                selected_arms=dict(CE='CE',OLD64='OLD64',LEGACY_SUPPORT32='LEGACY_SUPPORT32',DEEP_S32='S32_L1.0',
                    FIXED_D1=min(fixed,key=rules.rank_key)['arm'],JOINT=selected['selected'])
                p.save(choice,dict(dataset=n,seed=s,condition=c,choices={a:f'{cid(n,s,c)}/teacher/{b}' for a,b in selected_arms.items()},
                    fixed_scores=fixed,joint_sha256=digest,frozen=p.now()))
                status('TEACHER_CELL_COMPLETE',dataset=n,seed=s,condition=c)
            del ctx;torch.cuda.empty_cache()
    p.save(ROOT/'teacher_refs_frozen.json',{k:r for k,r in REFS.items() if r['kind']=='teacher'})
    p.save(ROOT/'teachers_frozen.json',dict(frozen=p.now(),refs_sha256=p.sha(ROOT/'teacher_refs_frozen.json'),
        selections={p.canonical(x):p.sha(x) for x in (ROOT/'selections/teacher').rglob('*.json')},
        races={p.canonical(x):p.sha(x) for x in (ROOT/'races').rglob('*.json')},historical_manifest_sha256=p.sha(ROOT/'historical_manifest.json')))


def teacher_freeze_check():
    f=p.read(ROOT/'teachers_frozen.json')
    assert p.sha(ROOT/'teacher_refs_frozen.json')==f['refs_sha256']
    assert p.sha(ROOT/'historical_manifest.json')==f['historical_manifest_sha256']
    for group in ('selections','races'):
        for name,digest in f[group].items():assert p.sha(ROOT/name)==digest


def train_students():
    if (ROOT/'selections_frozen.json').exists():return
    teacher_freeze_check()
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=p.graph_inputs(n,s)
            for c in CONFIG['conditions']:
                out=ROOT/'selections/student'/n/f'seed{s}'/f'{c}.json'
                if out.exists():continue
                cpu,es,b=prepare(n,s,c,g);v=gpu(cpu)
                choices=p.read(ROOT/'selections/teacher'/n/f'seed{s}'/f'{c}.json')['choices']
                aliases={};selected={}
                for route,key in choices.items():
                    tf=Path(REFS[key]['folder']);digest=p.sha(tf/'probabilities.pt')
                    if digest in aliases:selected[route]=aliases[digest];continue
                    prob=p.load(tf/'probabilities.pt').cuda().detach()
                    folder=ROOT/'fits/student'/cid(n,s,c)/route
                    status('DIST_STUDENT',dataset=n,seed=s,condition=c,route=route)
                    fitmodule.fit(folder,'DIST',v,es['LEGACY']['own'],dict(teacher=prob),seed=s,until=200,
                        metadata=dict(dataset=n,condition=c,arm=route,stage='STUDENT',partition='LEGACY',
                            teacher_probability_sha256=digest,teacher_folder=str(tf)))
                    sk,_=register(folder,n,s,c,route,'LEGACY','student');selected[route]=sk;aliases[digest]=sk
                p.save(out,dict(dataset=n,seed=s,condition=c,choices=selected,teacher_choices=choices,frozen=p.now()))
            torch.cuda.empty_cache()
    p.save(ROOT/'selections_frozen.json',dict(frozen=p.now(),choices={p.canonical(x):p.sha(x) for x in (ROOT/'selections').rglob('*.json')},
        teacher_freeze_sha256=p.sha(ROOT/'teachers_frozen.json'),refs_sha256=p.sha(ROOT/'refs.json')))
    status('ALL_CHOICES_FROZEN',teacher_cells=45,student_logical_endpoints=270)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','smoke','fit'],default='fit')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(CONFIG['threads'])
    if (ROOT/'complete.json').exists() or (args.stage=='fit' and (ROOT/'fits_complete.json').exists()):print('Already complete; no duplicate fitting');return
    lock=ROOT/'run.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),time=p.now()),f)
    try:
        for name,value in [('inputs.json',p.INPUTS),('refs.json',REFS),('physical_fits.json',PHYSICAL)]:
            if (ROOT/name).exists():value.update(p.read(ROOT/name))
        freeze_sources();preflight()
        if args.stage=='preflight':return
        assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=8*1024**3,'GPU unavailable/busy'
        p.save(ROOT/'environment.json',dict(time=p.now(),pid=os.getpid(),python=sys.executable,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),free_total=list(torch.cuda.mem_get_info())))
        smoke()
        if args.stage=='smoke':return
        train_teachers();train_students();teacher_freeze_check()
        for name,digest in p.read(ROOT/'sources.json').items():assert p.sha(name)==digest
        for name,digest in p.INPUTS.items():assert p.sha(name)==digest
        p.save(ROOT/'fits_complete.json',dict(time=p.now(),status='FITTING_COMPLETE_TEST_DEFERRED',cells=45,new_teacher_cells=36,
            new_teacher_trajectories=sum(r['kind']=='teacher' for r in PHYSICAL.values()),
            new_physical_students=sum(r['kind']=='student' for r in PHYSICAL.values()),logical_students=270,
            scientific_updates=sum(r['epoch'] for r in PHYSICAL.values())))
        status('FITTING_COMPLETE_TEST_DEFERRED')
    except BaseException:
        failure=dict(time=p.now(),stage=args.stage,traceback=traceback.format_exc())
        p.save(ROOT/'failure_latest.json',failure);p.append(ROOT/'failures.jsonl',failure);status('FAILED');raise
    finally:lock.unlink()


if __name__=='__main__':main()
