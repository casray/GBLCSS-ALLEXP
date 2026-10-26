"""Frozen fallback / matched KM / same-tree combination execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

from protocol import (ROOT, EXP, CONFIG, INPUTS, read, save, sha, track, load, now, append,
                      graph_inputs, cell_inputs, encoder_path, state_hash, training_data,
                      freeze_sources, canonical, disk_guard, csv_save)
from targets import evidence, shrink, groups
from fit import fit, Student, HIDDEN
from rules import choose, support_from_probe, distinct_supports, refinement_cost

OLD = EXP/'d1_c1_confirmation_20261024'
REFS, PHYSICAL, TARGET_INDEX, VERIFIED = {}, {}, {}, set()
OLD_INDEX = {}
GROUP_ARMS = [f'GROUP_A{a}_C{b}' for a,b in CONFIG['group_pairs']]
GLOBALS = ['CE','L0.25','L0.5','L1.0']


def status(stage, **kw):
    value = dict(stage=stage,time=now(),**kw)
    save(ROOT/'status.json',value)
    append(ROOT/'events.jsonl',value)
    (ROOT/'HANDOFF.md').write_text('# D1 fallback / KM / same-tree combo\n\n'+json.dumps(value,indent=2)+
        '\n\nFrozen runtime. Historical directories read-only. Inspect failures and lock before recovery.\n',encoding='utf-8')
    print(json.dumps(value),flush=True)


def oldfolder(stage,n,s,b,c,arm):
    return OLD/'fits'/stage/n/f'seed{s}'/b/c/arm


def probe_path(n,s,b,c):
    return OLD/'selections/C1'/n/f'seed{s}'/b/f'{c}.json'


def prepare(n,s,c,g):
    pack = cell_inputs(n,s,c)
    geo = g['geometry']
    values = dict(x=g['x'],train=geo['train'],valid=geo['valid'],observed=pack['observed'],
                  valid_y=pack['valid_y'],classes=pack['targets'][32].shape[1])
    parts = dict(LEGACY=geo['legacy32'],KM=g['km'],**{f'S{k}':v for k,v in geo['partitions'].items()})
    es = {p:evidence(pack['observed'],part[geo['train']],values['classes']) for p,part in parts.items()}
    return values,es


def to_gpu(values):
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in values.items()}


def cell_hash(values):
    return state_hash({k:v for k,v in values.items() if isinstance(v,torch.Tensor)})


def verify_old(folder, target_hash=None, input_hash=None):
    folder = Path(folder)
    if str(folder) not in VERIFIED:
        r = read(track(folder/'result.json'))
        assert r['epoch']==200 and r['status']=='complete'
        assert r['source_sha256']==sha(OLD/'sources.json')
        assert r['optimizer']=='Adam' and r['lr']==.01 and r['weight_decay']==.0005
        assert r['selection']=='first_strict_max_validation_accuracy'
        ce = read(oldfolder('D1',r['dataset'],r['seed'],r['backbone'],r['condition'],'CE')/'result.json')
        assert r['initial_hash']==ce['initial_hash'] and r['initial_rng_hash']==ce['initial_rng_hash']
        for name,key in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            assert sha(track(folder/name))==r[key]
        logs = [json.loads(x) for x in track(folder/'epochs.jsonl').read_text().splitlines()]
        assert [v['epoch'] for v in logs]==list(range(1,201))
        best = max(logs,key=lambda v:v['validation_correct'])
        assert (best['epoch'],best['validation_correct'],best['validation_nll'],best['validation_macro_f1'])==(
            r['best_epoch'],r['valid_correct'],r['valid_nll'],r['valid_macro_f1'])
        target = load(OLD/r['target_ref'])
        assert state_hash({'target':target})==r['target_hash']
        VERIFIED.add(str(folder))
    r = read(folder/'result.json')
    assert target_hash is None or r['target_hash']==target_hash
    assert input_hash is None or r['input_hash']==input_hash
    return r


def init_old_index():
    for item in read(track(OLD/'fit_registry.json')):
        p = OLD/item['folder']
        r = read(p/'result.json')
        if r['q'] is None:
            key = (r['dataset'],r['seed'],r['backbone'],r['condition'],r['target_hash'])
            OLD_INDEX.setdefault(key,p)


def identify(n,s,b,c,p,arm):
    return '/'.join([n,f'seed{s}',b,c,p,arm])


def register(n,s,b,c,p,arm,folder,reason,values=None,target=None):
    folder = Path(folder)
    key = identify(n,s,b,c,p,arm)
    digest = state_hash({'target':target}) if target is not None else None
    ih = cell_hash(values) if values is not None else None
    historical = OLD in folder.parents
    r = verify_old(folder,digest,ih) if historical else read(folder/'result.json')
    if digest is not None:
        assert r['target_hash']==digest
    if ih is not None:
        assert r['input_hash']==ih
    target_path = (OLD if historical else ROOT)/r['target_ref']
    record = dict(dataset=n,seed=s,backbone=b,condition=c,partition=p,arm=arm,
                  folder=str(folder.resolve()),target_path=str(target_path.resolve()),historical=historical,
                  reason=reason,result_sha256=sha(folder/'result.json'),target_hash=r['target_hash'],
                  input_hash=r['input_hash'],initial_hash=r['initial_hash'],source_sha256=r['source_sha256'])
    if key in REFS:
        record['reason']=REFS[key]['reason']
        assert REFS[key]==record
    REFS[key]=record
    TARGET_INDEX[(n,s,b,c,r['target_hash'])]=folder
    save(ROOT/'reuse_manifest.json',REFS)
    return key,r


def existing(n,s,b,c,p,arm,target,values):
    if p=='LEGACY':
        oldarm = 'SUPPORT_'+arm if arm.startswith('L') else arm
        folder = oldfolder('D1',n,s,b,c,oldarm)
    elif p=='KM':
        folder = oldfolder('D1',n,s,b,c,'CE' if arm=='CE' else 'KM_L1.0')
    else:
        folder = oldfolder('D1',n,s,b,c,'CE') if arm=='CE' else oldfolder('C1',n,s,b,c,p+'_'+arm)
    return register(n,s,b,c,p,arm,folder,'historical_exact_provenance',values,target)


def execute(n,s,c,p,arm,target,values,g):
    key = (n,s,'R0',c,state_hash({'target':target}))
    alias = TARGET_INDEX.get(key,OLD_INDEX.get(key))
    if alias is not None:
        return register(n,s,'R0',c,p,arm,alias,'exact_target_input_init_alias',values,target)
    folder = ROOT/'fits'/n/f'seed{s}'/'R0'/c/p/arm
    status('TRAINING',dataset=n,seed=s,condition=c,partition=p,arm=arm,physical_complete=len(PHYSICAL))
    enc = encoder_path(n,s)
    r = fit(folder,'R0',values,target,g['encoder'],seed=s,metadata=dict(dataset=n,condition=c,partition=p,
        arm=arm,stage='FOLLOWUP',encoder_path=str(enc.resolve()),encoder_file_sha256=sha(enc)))
    ce = read(oldfolder('D1',n,s,'R0',c,'CE')/'result.json')
    assert r['initial_hash']==ce['initial_hash'] and r['initial_rng_hash']==ce['initial_rng_hash']
    PHYSICAL[canonical(folder)] = dict(dataset=n,seed=s,condition=c,partition=p,arm=arm,folder=canonical(folder))
    assert len(PHYSICAL)<=CONFIG['fit_upper_bound']
    save(ROOT/'physical_fits.json',PHYSICAL)
    return register(n,s,'R0',c,p,arm,folder,'new_physical_fit',values,target)


def pool_global(n,s,b,c,p,e,values,g=None):
    out = []
    for arm,l in zip(GLOBALS,[0.,.25,.5,1.]):
        target = shrink(e,l)
        if p=='KM' and l in (.25,.5):
            out.append(execute(n,s,c,p,arm,target,values,g))
        else:
            out.append(existing(n,s,b,c,p,arm,target,values))
    return out


def expand(n,s,b,c,p,e,globalpool,values,g=None):
    tie_index = min(range(4),key=lambda i:(-globalpool[i][1]['validation'],i))
    tie_lambda = [0.,.25,.5,1.][tie_index]
    group = groups(e)
    out = list(globalpool)
    for arm,(a,d) in zip(GROUP_ARMS,CONFIG['group_pairs']):
        w = torch.where(group==0,a,torch.where(group==1,d,tie_lambda)).double()
        target = shrink(e,w)
        out.append(existing(n,s,b,c,p,arm,target,values) if p=='LEGACY' else execute(n,s,c,p,arm,target,values,g))
    return out,tie_lambda


def freeze_cell(n,s,b,c,pools,meta):
    result = dict(dataset=n,seed=s,backbone=b,condition=c,meta=meta,
                  pools={name:[dict(ref=k,validation=r['validation'],valid_macro_f1=r['valid_macro_f1'],
                    valid_nll=r['valid_nll'],best_epoch=r['best_epoch']) for k,r in pool] for name,pool in pools.items()},
                  choices={selector:{name:choose(pool,selector) for name,pool in pools.items()} for selector in CONFIG['selectors']})
    path = ROOT/'selections'/n/f'seed{s}'/b/f'{c}.json'
    if path.exists():
        previous=read(path)
        assert {k:v for k,v in previous.items() if k!='frozen'}==result
    else:
        save(path,dict(result,frozen=now()))


def preflight():
    status('PREFLIGHT')
    assert read(track(OLD/'complete.json'))['audit']=='passed'
    for name in ['sources.json','inputs.json']:
        manifest = read(track(OLD/name))
        for path,digest in manifest.items():
            assert sha(path)==digest, path
    assert sha(OLD/'fit.py')==sha(ROOT/'fit.py'), 'Fit engine must remain byte identical'
    inventory,passports = [],[]
    projected = 0
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            passport=next(v for v in read(track(OLD/'passports.json')) if v['dataset']==n and v['seed']==s)
            assert g['h_bridge']==passport['h_normalization']
            assert state_hash({'partition':g['km']})==passport['km_hash']
            passports.append(passport)
            for c in CONFIG['conditions']:
                values,es=prepare(n,s,c,g)
                for b in CONFIG['backbones']:
                    gl=pool_global(n,s,b,c,'LEGACY',es['LEGACY'],values)
                    expand(n,s,b,c,'LEGACY',es['LEGACY'],gl,values)
                probe=read(track(probe_path(n,s,'R0',c)))
                supports=distinct_supports(probe['PROBE'])
                count=6+4*len(supports)
                inventory.append(dict(dataset=n,seed=s,condition=c,supports=supports,PROBE=probe['PROBE'],
                    km_logical_new=6,deep_logical_new=4*len(supports),new_fit_upper_after_support_dedup=count,
                    adaptive_cost=refinement_cost(probe,support_from_probe(probe['PROBE']))))
                for support in supports:
                    pool_global(n,s,'R0',c,f'S{support}',es[f'S{support}'],values)
                existing(n,s,'R0',c,'KM','CE',es['KM']['own'],values)
                existing(n,s,'R0',c,'KM','L1.0',shrink(es['KM'],1),values)
                ce=oldfolder('D1',n,s,'R0',c,'CE')
                size=(ce/'checkpoint.pt').stat().st_size+(ce/'probabilities.pt').stat().st_size
                target_bytes=4*len(values['train'])*values['classes']
                projected+=count*(size+target_bytes+180000)
    upper=sum(v['new_fit_upper_after_support_dedup'] for v in inventory)
    free=disk_guard()
    projected+=512*1024**2
    save(ROOT/'inventory.json',dict(cells=120,new_R0_cells=60,logical_fit_upper=upper,absolute_cap=840,
        exact_target_aliases='resolved before each fit; KM group targets depend on new KM global validation',cells_detail=inventory))
    save(ROOT/'storage_estimate.json',dict(free_bytes=free,projected_upper_bytes=projected,reserve_bytes=3*1024**3,
        logical_fit_upper=upper,historical_artifacts='external references only',target_dedup=True))
    assert free-projected>=3*1024**3, 'Insufficient storage: stop before fits'
    save(ROOT/'passports.json',passports)
    save(ROOT/'preflight_complete.json',dict(finished=now(),upper=upper,verified_reuse_folders=len(VERIFIED)))


def smoke():
    checks=[]
    for n,s,pair in [('cora',0,True),('corafull',1,False),('coauthorcs',2,False)]:
        g=graph_inputs(n,s)
        values,es=prepare(n,s,'uniform0.4',g)
        values=to_gpu(values)
        path=ROOT/'smoke'/f'{n}_seed{s}'
        meta=dict(dataset=n,condition='uniform0.4')
        r=fit(path,'R0',values,shrink(es['KM'],.5),g['encoder'],seed=s,until=2,metadata=meta)
        ce=read(oldfolder('D1',n,s,'R0','uniform0.4','CE')/'result.json')
        assert r['initial_hash']==ce['initial_hash'] and r['initial_rng_hash']==ce['initial_rng_hash']
        if pair:
            other=path.with_name(path.name+'_resume')
            for until in [1,2]:
                fit(other,'R0',values,shrink(es['KM'],.5),g['encoder'],seed=s,until=until,metadata=meta)
            from protocol import tree_hash
            a=torch.load(path/'resume.pt',weights_only=False)
            d=torch.load(other/'resume.pt',weights_only=False)
            assert all(tree_hash(a[k])==tree_hash(d[k]) for k in ['model','optimizer','rng','best_state'])
        checks.append(dict(dataset=n,seed=s,paired_initialization=True,own_resume=pair))
        del g,values,es
        torch.cuda.empty_cache()
    save(ROOT/'smoke_complete.json',dict(fits=4,epochs=2,checks=checks,finished=now()))


def train():
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            for c in CONFIG['conditions']:
                cpu,es=prepare(n,s,c,g)
                for b in CONFIG['backbones']:
                    legacy=pool_global(n,s,b,c,'LEGACY',es['LEGACY'],cpu)
                    lg,tie=expand(n,s,b,c,'LEGACY',es['LEGACY'],legacy,cpu)
                    forced=[(identify(n,s,b,c,'LEGACY',a),r) for _,r in lg[4:] for a in [r['arm']]]
                    pools=dict(CE=[legacy[0]],LEGACY_GLOBAL=legacy,LEGACY_EXPANDED=lg,LEGACY_FORCED=forced)
                    meta=dict(legacy_tie_lambda=tie,retrospective_fallback=True)
                    if b=='R0':
                        values=to_gpu(cpu)
                        km=pool_global(n,s,b,c,'KM',es['KM'],values,g)
                        kg,kt=expand(n,s,b,c,'KM',es['KM'],km,values,g)
                        pools.update(KM_GLOBAL=km,KM_EXPANDED=kg)
                        probe=read(track(probe_path(n,s,b,c)))
                        adaptive=support_from_probe(probe['PROBE'])
                        deep={}
                        deep_ties={}
                        for support in distinct_supports(probe['PROBE']):
                            p=f'S{support}'
                            glob=pool_global(n,s,b,c,p,es[p],values)
                            expanded,t=expand(n,s,b,c,p,es[p],glob,values,g)
                            deep[support]=(glob,expanded)
                            deep_ties[p]=t
                        pools.update(FIXED_GLOBAL=deep[32][0],FIXED_EXPANDED=deep[32][1],
                            ADAPTIVE_GLOBAL=[legacy[0]] if adaptive is None else deep[adaptive][0],
                            ADAPTIVE_EXPANDED=[legacy[0]] if adaptive is None else deep[adaptive][1])
                        anchor=legacy[0] if adaptive is None else next(v for v in deep[adaptive][0] if v[0].endswith('/'+probe['PROBE'].split('_',1)[1]))
                        pools['ORIGINAL_C1']=[anchor]
                        pools['FIXED_S32']=[deep[32][0][-1]]
                        pools['SINGLE_MODULE_VAL']=pools['ADAPTIVE_GLOBAL']+pools['FIXED_EXPANDED']
                        meta.update(adaptive_support=adaptive,PROBE=probe['PROBE'],probe_sha256=sha(probe_path(n,s,b,c)),
                            deep_ties=deep_ties,adaptive_cost=refinement_cost(probe,adaptive),
                            fixed_global_updates=800,fixed_expanded_updates=1600,
                            matched_legacy_km_global_updates=800,matched_legacy_km_expanded_updates=1600)
                    freeze_cell(n,s,b,c,pools,meta)
            del g,cpu,es
            HIDDEN.clear()
            torch.cuda.empty_cache()
    frozen=ROOT/'selections_frozen.json'
    if not frozen.exists():
        save(frozen,dict(frozen=now(),choices={canonical(p):sha(p) for p in (ROOT/'selections').rglob('*.json')},
            reuse_manifest_sha256=sha(ROOT/'reuse_manifest.json'),physical_fits=len(PHYSICAL)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if (ROOT/'complete.json').exists():
        return
    torch.set_num_threads(1)
    lock=ROOT/'run.lock'
    with lock.open('x') as f:
        json.dump(dict(pid=os.getpid(),started=now()),f)
    try:
        for name,target in [('inputs.json',INPUTS),('reuse_manifest.json',REFS),('physical_fits.json',PHYSICAL)]:
            if (ROOT/name).exists():
                target.update(read(ROOT/name))
        for r in REFS.values():
            TARGET_INDEX[(r['dataset'],r['seed'],r['backbone'],r['condition'],r['target_hash'])]=Path(r['folder'])
        freeze_sources()
        init_old_index()
        if not (ROOT/'preflight_complete.json').exists():
            preflight()
        if args.preflight:
            return
        while not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*1024**3:
            status('GPU_WAIT')
            time.sleep(60)
        save(ROOT/'environment.json',dict(python=__import__('sys').executable,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0),free_total=list(torch.cuda.mem_get_info()),started=now(),pid=os.getpid()))
        if not (ROOT/'smoke_complete.json').exists():
            smoke()
        if args.smoke:
            return
        if not (ROOT/'selections_frozen.json').exists():
            train()
        from analyze import analyze
        from audit import audit
        from report import report
        status('ANALYSIS')
        analyze()
        status('AUDIT')
        audit()
        report()
        save(ROOT/'complete.json',dict(finished=now(),scientific_fits=len(PHYSICAL),smoke_fits=4,audit='passed'))
        status('COMPLETE',scientific_fits=len(PHYSICAL),audit='passed')
    except BaseException:
        failure=dict(time=now(),traceback=traceback.format_exc())
        append(ROOT/'failures.jsonl',failure)
        save(ROOT/'failure_latest.json',failure)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
