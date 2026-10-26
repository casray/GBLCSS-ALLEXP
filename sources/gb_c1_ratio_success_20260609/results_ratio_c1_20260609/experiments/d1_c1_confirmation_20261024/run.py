"""Approved five-graph, three-seed D1 then causal C1 confirmation queue."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from protocol import (ROOT, EXP, CONFIG, INPUTS, WORKSPACE, save, read, load, track, sha, now,
                      output, append, csv_save, canonical, freeze_sources, graph_inputs, cell_inputs,
                      encoder_path, geometry_path, training_data, state_hash, disk_guard, tree_hash)
from targets import evidence, shrink, groups, global_mass, match_mass
from selection import rank_key, probe_survivors, score
from fit import fit, Student, HIDDEN

RECORDS = []


def cells():
    for name in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            for backbone in CONFIG['backbones']:
                for cond in CONFIG['conditions']:
                    yield name,seed,backbone,cond


def candidates():
    return {f'S{k}_L{l}':(k,l) for k in CONFIG['supports'] for l in CONFIG['strengths']}


def event(stage, **kwargs):
    value = dict(timestamp=now(), stage=stage, **kwargs)
    save(ROOT/'status.json',value)
    append(ROOT/'events.jsonl',value)
    output(ROOT/'HANDOFF.md').write_text(
        '# D1/C1 confirmation execution\n\nState: '+stage+'\n\n'+json.dumps(value,ensure_ascii=False,indent=2)+
        '\n\nResume: D:/codex_temp/noisygl-rocm721/Scripts/python.exe -B experiments/d1_c1_confirmation_20261024/run.py\n'
        'Only this new experiment directory is writable. Inspect failures.jsonl and run.lock before recovery.\n'
        'Selections are validation-only; test scoring is deferred until selections_frozen.json.\n',encoding='utf-8')
    print('STATUS '+json.dumps(value),flush=True)


def folder_for(stage,name,seed,backbone,cond,arm):
    return ROOT/'fits'/stage/name/f'seed{seed}'/backbone/cond/arm


def decision_path(stage,name,seed,backbone,cond,suffix=''):
    return ROOT/'selections'/stage/name/f'seed{seed}'/backbone/f'{cond}{suffix}.json'


def prepare(name,seed,cond,graph):
    pack = cell_inputs(name,seed,cond)
    geo,z = graph['geometry'],pack['observed']
    classes = pack['targets'][32].shape[1]
    values = training_data(dict(x=graph['x'].cuda(),train=geo['train'].cuda(),valid=geo['valid'].cuda(),
                               observed=z.cuda(),valid_y=pack['valid_y'].cuda(),classes=classes))
    es = {s:evidence(z,part[geo['train']],classes) for s,part in geo['partitions'].items()}
    es['LEGACY'] = evidence(z,geo['legacy32'][geo['train']],classes)
    es['KM'] = evidence(z,graph['km'][geo['train']],classes)
    path = ROOT/'evidence'/name/f'seed{seed}'/cond
    if not (path/'bridge.json').exists():
        source = track(EXP/'r2_support_5dataset_20261022/run.py')
        nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if isinstance(n,ast.FunctionDef) and n.name=='numpy_targets']
        ns = dict(np=np)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        differences = {}
        for key,part in [('LEGACY',geo['legacy32']),('KM',graph['km'])]:
            expected = torch.from_numpy(ns['numpy_targets'](z.numpy(),part[geo['train']].numpy(),classes,.5,2)).float()
            differences[key] = float((expected-es[key]['native']).abs().max())
        for s in CONFIG['supports']:
            differences[str(s)] = float((es[s]['native']-pack['targets'][s]).abs().max())
        differences['reconstruction'] = max(float((shrink(e,1)-e['native']).abs().max()) for e in es.values())
        if max(differences.values())>3e-7:
            raise RuntimeError('Historical target numerical bridge failed')
        save(path/'bridge.json',differences)
        profiles = []
        for s,e in es.items():
            profiles.append(dict(partition=str(s),regions=len(torch.unique(geo['partitions'][s] if isinstance(s,int)
                        else geo['legacy32'] if s=='LEGACY' else graph['km'])),
                        mean=float(e['support'].mean()),median=float(e['support'].median()),
                        minimum=float(e['support'].min()),maximum=float(e['support'].max()),
                        q90=float(torch.quantile(e['support'],.9)),mass_mean=float(e['m'].mean()),
                        tied_fraction=float(e['tied'].double().mean())))
        save(path/'profiles.json',profiles)
    return values,es,pack


def execute(stage,name,seed,backbone,cond,arm,values,target,encoder,until=200,q=None):
    folder = folder_for(stage,name,seed,backbone,cond,arm)
    event(stage,dataset=name,seed=seed,backbone=backbone,condition=cond,arm=arm,until=until,
          completed_full=sum((ROOT/r['folder']/'result.json').exists() and read(ROOT/r['folder']/'result.json')['epoch']==200 for r in RECORDS)
          if len(RECORDS)%30==0 else len(RECORDS))
    encpath = encoder_path(name,seed)
    result = fit(folder,backbone,values,target,encoder,seed=seed,until=until,q=q,
                 metadata=dict(dataset=name,condition=cond,arm=arm,stage=stage,
                               encoder_path=str(encpath.resolve()),encoder_file_sha256=INPUTS[str(encpath.resolve())]))
    item = dict(stage=stage,dataset=name,seed=seed,backbone=backbone,condition=cond,arm=arm,
                folder=canonical(folder),alias=False)
    if item['folder'] not in {r['folder'] for r in RECORDS}:
        RECORDS.append(item)
        save(ROOT/'fit_registry.json',RECORDS)
    return result


def preflight():
    passports, estimates = [],[]
    for name in CONFIG['datasets']:
        hs = []
        for seed in CONFIG['seeds']:
            event('PREFLIGHT',dataset=name,seed=seed)
            g = graph_inputs(name,seed,verify_km=True)
            geo = g['geometry']
            hs.append(geo['h_sha256'])
            conditions = []
            for cond in CONFIG['conditions']:
                p = cell_inputs(name,seed,cond)
                conditions.append(dict(condition=cond,observed_hash=state_hash({'z':p['observed']}),
                                  validation_hash=state_hash({'yv':p['valid_y']})))
            n,d,c = len(g['x']),g['x'].shape[1],p['targets'][32].shape[1]
            model_bytes = 4*(d*64+64*c+g['h_dim']*64+64*8)
            probability_bytes = 4*n*c
            target_bytes = 4*len(geo['train'])*c
            # Both backbones and all conditions; count targets without dedup for a conservative estimate.
            expected = 4*30*(model_bytes+4*(64*c+c)+2*probability_bytes+2*target_bytes+2*60000)
            estimates.append(dict(dataset=name,seed=seed,estimated_final_bytes=expected,nodes=n,features=d,classes=c))
            passports.append(dict(dataset=name,seed=seed,h_sha256=geo['h_sha256'],
                h_normalization=g['h_bridge'],
                h_nodes=n,encoder=str(encoder_path(name,seed)),encoder_sha256=sha(encoder_path(name,seed)),
                split_hash=state_hash({k:geo[k] for k in ('train','valid','test')}),
                partitions_hash=state_hash({str(k):v for k,v in geo['partitions'].items()}),
                legacy_hash=state_hash({'partition':geo['legacy32']}),km_hash=state_hash({'partition':g['km']}),
                km_exact_recomputed=True,conditions=conditions))
            del g
        if len(set(hs)) != 1:
            raise RuntimeError('Historical H changed across protocol seeds')
    estimated = sum(r['estimated_final_bytes'] for r in estimates)
    free = disk_guard()
    save(ROOT/'storage_estimate.json',dict(free_bytes=free,estimated_final_bytes=estimated,
         minimum_reserve_bytes=CONFIG['storage']['minimum_free_bytes'],details=estimates,
         strategy=CONFIG['storage'],fits=3600,historical_reuse=False))
    if free-estimated < CONFIG['storage']['minimum_free_bytes']:
        raise RuntimeError('Preflight conservative storage estimate exceeds safe capacity')
    save(ROOT/'passports.json',passports)
    save(ROOT/'preflight_complete.json',dict(finished=now(),graph_seed_passports=15,cells=120,km_exact=15))


def stage_d1():
    for name in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            graph = graph_inputs(name,seed)
            for backbone in CONFIG['backbones']:
                for cond in CONFIG['conditions']:
                    values,es,_ = prepare(name,seed,cond,graph)
                    e = es['LEGACY']
                    run = lambda arm,target,q=None: execute('D1',name,seed,backbone,cond,arm,values,target,graph['encoder'],q=q)
                    anchors = {0.:run('CE',e['own'])}
                    anchors.update({l:run(f'SUPPORT_L{l}',shrink(e,l)) for l in CONFIG['strengths']})
                    run('GCE_Q0.7',e['own'],.7)
                    run('KM_L1.0',shrink(es['KM'],1))
                    best_global = min(anchors,key=lambda l:(-anchors[l]['validation'],l))
                    group = groups(e)
                    rows,weights = [],{}
                    for a,b in CONFIG['group_pairs']:
                        arm = f'GROUP_A{a}_C{b}'
                        w = torch.where(group==0,a,torch.where(group==1,b,best_global)).double()
                        rows.append(score(arm,run(arm,shrink(e,w))))
                        weights[arm] = w
                    winner = min(rows,key=rank_key)['arm']
                    w = weights[winner]
                    mean = global_mass(e['m'],w)
                    perm = torch.randperm(len(w),generator=torch.Generator().manual_seed(20261024))
                    shuffled = match_mass(e['m'],w,w[perm])
                    choice = dict(dataset=name,seed=seed,backbone=backbone,condition=cond,selected=winner,
                        best_cell_global_lambda=best_global,mean_lambda=mean,frozen=now(),scores=rows,
                        total_mass=float((e['m']*w).sum()),global_mass=float((e['m']*mean).sum()),
                        shuffled_mass=float((e['m']*shuffled).sum()),
                        groups=[dict(group=k,count=int((group==k).sum()),
                            selected_lambda=float(w[group==k][0]) if (group==k).any() else None,
                            observed_class_counts=torch.bincount(e['z'][group==k],minlength=values['classes']).tolist()) for k in range(3)])
                    path = decision_path('D1',name,seed,backbone,cond)
                    if path.exists():
                        previous = read(path)
                        if {k:v for k,v in previous.items() if k!='frozen'} != {k:v for k,v in choice.items() if k!='frozen'}:
                            raise RuntimeError('Frozen D1 choice differs on resume')
                    else:
                        save(path,choice)
                    weights_path = output(ROOT/'evidence'/name/f'seed{seed}'/cond/f'd1_{backbone}.pt')
                    if not weights_path.exists():
                        torch.save(dict(weights=w,shuffled=shuffled,groups=group),weights_path)
                    run('EXACT_MASS_GLOBAL',shrink(e,mean))
                    run('MASS_MATCHED_SHUFFLE',shrink(e,shuffled))
            del graph,values,es
            HIDDEN.clear()
            torch.cuda.empty_cache()
    save(ROOT/'D1_complete.json',dict(finished=now(),full_fit_count=1440))


def require_probe(path):
    if not Path(path).exists():
        raise RuntimeError('FULL_VAL audit requires already frozen causal PROBE')
    return sha(path)


def stage_c1():
    for name in CONFIG['datasets']:
        for seed in CONFIG['seeds']:
            graph = graph_inputs(name,seed)
            for backbone in CONFIG['backbones']:
                for cond in CONFIG['conditions']:
                    values,es,pack = prepare(name,seed,cond,graph)
                    arms = candidates()
                    ce = read(folder_for('D1',name,seed,backbone,cond,'CE')/'result.json')
                    run = lambda arm,until: execute('C1',name,seed,backbone,cond,arm,values,
                                       shrink(es[arms[arm][0]],arms[arm][1]),graph['encoder'],until=until)
                    path = decision_path('C1',name,seed,backbone,cond)
                    if not path.exists():
                        survivors,rounds = list(arms),[]
                        for prefix,count in [(25,6),(75,2),(200,1)]:
                            roundpath = decision_path('C1',name,seed,backbone,cond,f'_prefix{prefix}')
                            if prefix<200 and roundpath.exists():
                                rd = read(roundpath)
                                rounds.append(rd)
                                survivors = rd['survivors']
                                continue
                            rows = [score(arm,run(arm,prefix),*arms[arm]) for arm in survivors]
                            if prefix<200:
                                survivors = probe_survivors(rows,count,prefix)
                                rd = dict(prefix=prefix,scores=rows,survivors=survivors,frozen=now())
                                save(roundpath,rd)
                                rounds.append(rd)
                            else:
                                rows.append(score('CE',ce))
                                save(path,dict(frozen=now(),PROBE=min(rows,key=rank_key)['arm'],rounds=rounds,
                                    final_scores=rows,candidate_updates=1000,ce_updates=200,
                                    probe_candidate_seconds=sum(read(folder_for('C1',name,seed,backbone,cond,a)/'result.json')['seconds'] for a in arms),
                                    true_prefix_execution=True,historical_reuse=False))
                    frozen = require_probe(path)
                    rows = [score('CE',ce)]
                    probe_seconds = read(path)['probe_candidate_seconds']
                    fullpath = decision_path('C1',name,seed,backbone,cond,'_full')
                    for arm,(s,l) in arms.items():
                        rows.append(score(arm,run(arm,200),s,l))
                    if sha(path) != frozen:
                        raise RuntimeError('Frozen PROBE changed during FULL_VAL audit')
                    if not fullpath.exists():
                        save(fullpath,dict(frozen=now(),probe_sha256=frozen,FULL_VAL=min(rows,key=rank_key)['arm'],
                            ONLY_K=min([r for r in rows if r['arm']=='CE' or r['strength']==1],key=rank_key)['arm'],
                            ONLY_STRENGTH=min([r for r in rows if r['arm']=='CE' or r['support']==32],key=rank_key)['arm'],
                            FIXED_S32='S32_L1.0',SC=f'S{pack["selected_support"]}_L1.0',scores=rows,
                            probe_candidate_seconds=probe_seconds,probe_ce_seconds=ce['seconds'],
                            full_candidate_seconds=sum(read(folder_for('C1',name,seed,backbone,cond,a)/'result.json')['seconds'] for a in arms),
                            full_candidate_updates=3600,probe_candidate_updates=1000,audit_complete=True))
            del graph,values,es
            HIDDEN.clear()
            torch.cuda.empty_cache()
    save(ROOT/'C1_complete.json',dict(finished=now(),candidate_fit_count=2160))


def smoke():
    checks = []
    for name,seed,backbone,pair in [('cora',0,'R0',True),('corafull',1,'R0',False),
                                  ('coauthorcs',2,'R0',False),('cora',1,'R2',True)]:
        graph = graph_inputs(name,seed)
        values,es,_ = prepare(name,seed,'uniform0.4',graph)
        label = f'{name}_seed{seed}_{backbone}'
        folder = ROOT/'smoke'/label
        meta = dict(dataset=name,condition='uniform0.4',encoder_path=str(encoder_path(name,seed)),
                    encoder_file_sha256=sha(encoder_path(name,seed)))
        fit(folder,backbone,values,es['LEGACY']['native'],graph['encoder'],seed=seed,until=2,metadata=meta)
        if pair:
            fit(folder.with_name(label+'_resume'),backbone,values,es['LEGACY']['native'],graph['encoder'],seed=seed,until=1,metadata=meta)
            fit(folder.with_name(label+'_resume'),backbone,values,es['LEGACY']['native'],graph['encoder'],seed=seed,until=2,metadata=meta)
            a = torch.load(folder/'probabilities.pt',weights_only=True)
            b = torch.load(folder.with_name(label+'_resume')/'probabilities.pt',weights_only=True)
            if not torch.equal(a,b):
                raise RuntimeError('Own model/optimizer/RNG smoke resume mismatch')
            first = torch.load(folder/'resume.pt',weights_only=False)
            resumed = torch.load(folder.with_name(label+'_resume')/'resume.pt',weights_only=False)
            for key in ('model','optimizer','rng','best_state'):
                if tree_hash(first[key]) != tree_hash(resumed[key]):
                    raise RuntimeError('Terminal smoke continuation differs: '+key)
        ck = torch.load(folder/'checkpoint.pt',weights_only=True)
        student = Student(backbone,values,graph['encoder'],seed)
        student.set_state(ck['model'])
        student.model.eval()
        with torch.no_grad():
            diff = float((student.forward().softmax(1).cpu()-torch.load(folder/'probabilities.pt',weights_only=True)).abs().max())
        if diff>1e-7:
            raise RuntimeError('Smoke checkpoint reload mismatch')
        checks.append(dict(dataset=name,seed=seed,backbone=backbone,reload_difference=diff,resume_pair=pair))
        del graph,values,es,student
        HIDDEN.clear()
        torch.cuda.empty_cache()
    save(ROOT/'smoke_complete.json',dict(fits=6,epochs_per_fit=2,checks=checks,finished=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if (ROOT/'complete.json').exists():
        print('Already complete; no training launched',flush=True)
        return
    torch.set_num_threads(1)
    lock = output(ROOT/'run.lock')
    with lock.open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),started=now()),stream)
    try:
        if (ROOT/'inputs.json').exists():
            INPUTS.update(read(ROOT/'inputs.json'))
        if (ROOT/'fit_registry.json').exists():
            RECORDS.extend(read(ROOT/'fit_registry.json'))
        freeze_sources()
        if not (ROOT/'preflight_complete.json').exists():
            preflight()
        if args.preflight:
            return
        while not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<8*1024**3:
            event('GPU_WAIT',reason='GPU unavailable or occupied; other jobs untouched')
            time.sleep(60)
        save(ROOT/'environment.json',dict(python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
             free_total_bytes=list(torch.cuda.mem_get_info()),pid=os.getpid(),started=now(),model='gpt-6.1-sol',effort='high'))
        if not (ROOT/'smoke_complete.json').exists():
            smoke()
        if args.smoke:
            return
        for stage,function in [('D1',stage_d1),('C1',stage_c1)]:
            if not (ROOT/f'{stage}_complete.json').exists():
                function()
        frozen = ROOT/'selections_frozen.json'
        if not frozen.exists():
            save(frozen,dict(frozen=now(),selections={canonical(p):sha(p) for p in (ROOT/'selections').rglob('*.json')},
                            full_fits=len(RECORDS),test_access_before_freeze=False))
        from analyze import analyze
        event('ANALYSIS')
        analyze()
        from audit import audit
        event('AUDIT')
        audit()
        from report import report
        report()
        save(ROOT/'complete.json',dict(finished=now(),full_fits=len(RECORDS),smoke_fits=6,audit=read(ROOT/'audit.json')['status']))
        event('COMPLETE',full_fits=len(RECORDS),audit='passed')
    except BaseException:
        failure = dict(time=now(),traceback=traceback.format_exc(),status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else None)
        append(ROOT/'failures.jsonl',failure)
        save(ROOT/'failure_latest.json',failure)
        event('INTERRUPTED',failure='failure_latest.json')
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    main()
