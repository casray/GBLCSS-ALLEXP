"""Reconstruct targets, causal selectors, paired init and sampled checkpoints."""
import csv
from pathlib import Path

import torch

from protocol import ROOT,CONFIG,read,save,sha,state_hash,now,csv_save,graph_inputs
from fit import Student,HIDDEN
from rules import rank_key,prefix_score,survivors,weights
from analyze import validate_freeze


def audit():
    from run import (GRID,REFS,prepare,gpu,cellid,getlogs,metadata_score,original_row,
        original_selection,target_for,oldfolder)
    freeze=validate_freeze()
    refs=read(ROOT/'reuse_manifest.json')
    reloads=[];target_count=0;race_count=0;prefix_count=0
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            for c in CONFIG['conditions']:
                cpu,es,_=prepare(n,s,c,g);v=gpu(cpu)
                cellrefs={k:r for k,r in refs.items() if (r['dataset'],r['seed'],r['condition'])==(n,s,c)}
                ce=read(oldfolder(n,s,c,'CE')/'result.json')
                for key,ref in cellrefs.items():
                    folder=Path(ref['folder']);r=read(folder/'result.json')
                    assert sha(folder/'result.json')==ref['result_sha256']
                    assert (r['initial_hash'],r['initial_rng_hash'],r['encoder_hash'])==(ce['initial_hash'],ce['initial_rng_hash'],ce['encoder_hash'])
                    target,_=target_for(ref['arm'],es)
                    assert state_hash({'target':target})==r['target_hash']
                    assert state_hash({'target':torch.load(ref['target_path'],weights_only=True)})==r['target_hash']
                    target_count+=1
                    logs=getlogs(folder)
                    assert [z['epoch'] for z in logs]==list(range(1,r['epoch']+1))
                    best=max(logs,key=lambda x:x['validation_correct'])
                    assert (best['epoch'],best['validation_correct'],best['validation_nll'])==(r['best_epoch'],r['valid_correct'],r['valid_nll'])
                    for filename,k in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
                        assert sha(folder/filename)==r[k]
                    if not ref['historical']:
                        assert r['source_sha256']==sha(ROOT/'sources.json')
                        for p in (folder/'prefixes').glob('*/manifest.json'):
                            snap=read(p);d=p.parent
                            assert sha(d/'resume.pt')==snap['resume_sha256'] and sha(d/'result.json')==snap['result_sha256']
                            ck=torch.load(d/'resume.pt',weights_only=False)
                            assert ck['epoch']==snap['epoch'] and ck['meta']['initial_hash']==ce['initial_hash']
                            rr=read(d/'result.json')
                            assert ck['best_epoch']==rr['best_epoch']
                            assert ck['optimizer']['state'] and ck['rng']['gpu'] is not None
                            prefix_count+=1
                for kind in ('GLOBAL','JOINT'):
                    race=read(ROOT/'races'/cellid(n,s,c)/(kind+'.json'))
                    expected=list(GRID) if kind=='JOINT' else [a for a,d in GRID.items() if d['kind']=='uniform']
                    for rd in race['rounds']:
                        prefix=rd['prefix'];rows=[]
                        for arm in expected+(['CE'] if prefix==200 else []):
                            ref=refs[cellid(n,s,c)+'/'+arm]
                            support,mean=metadata_score(arm,es)
                            rows.append(prefix_score(getlogs(Path(ref['folder'])),prefix,arm,support,mean,len(v['valid'])))
                        assert rows==rd['scores']
                        if prefix<200:
                            count=(14 if prefix==25 else 4) if kind=='JOINT' else (6 if prefix==25 else 2)
                            expected=survivors(rows,count,prefix) if kind=='JOINT' else original_selection.probe_survivors([original_row(r) for r in rows],count,prefix)
                        else:
                            expected=[min(rows,key=rank_key if kind=='JOINT' else lambda r:original_selection.rank_key(original_row(r)))['arm']]
                        assert expected==rd['survivors']
                    assert expected==[race['selected']]
                    race_count+=1
                choice=read(ROOT/'selections'/n/f'seed{s}'/f'{c}.json')
                jp=ROOT/'races'/cellid(n,s,c)/'JOINT.json';jo=read(jp)
                assert sha(jp)==choice['joint_sha256']
                finalgroup={r['arm'] for r in jo['rounds'][-1]['scores']}
                for arm in [a for a,d in GRID.items() if d['support']==32 and d['kind']=='group']:
                    folder=Path(refs[cellid(n,s,c)+'/'+arm]['folder'])
                    r=read(folder/'result.json')
                    if arm not in finalgroup:assert r['finished']>=jo['frozen']
                fixed=choice['fixed_scores'];fx=min(fixed,key=rank_key)
                gl=read(jp.with_name('GLOBAL.json'))
                ag=next(r for r in gl['rounds'][-1]['scores'] if r['arm']==gl['selected'])
                assert choice['choices']['FIXED_EXPANDED'].endswith('/'+fx['arm'])
                assert choice['choices']['SINGLE_MODULE_VAL'].endswith('/'+min([fx,ag],key=rank_key)['arm'])
                if c in ('clean','uniform0.6'):
                    key=choice['choices']['ADAPTIVE_EXPANDED']
                    folder=Path(refs[key]['folder']);ck=torch.load(folder/'checkpoint.pt',weights_only=True)
                    assert sha(ck['encoder_path'])==ck['encoder_file_sha256']
                    student=Student('R2',v,g['encoder'],s)
                    student.set_state(ck['model']);student.model.eval();student.check_frozen()
                    with torch.no_grad():
                        err=float((student.forward().softmax(1).cpu()-torch.load(folder/'probabilities.pt',weights_only=True)).abs().max())
                    assert err<=1e-7
                    reloads.append(dict(dataset=n,seed=s,condition=c,ref=key,max_error=err))
            HIDDEN.clear();torch.cuda.empty_cache()
    for filename in ('sources.json','inputs.json'):
        for path,digest in read(ROOT/filename).items():assert sha(path)==digest,path
    physical=read(ROOT/'physical_fits.json')
    assert len(physical)==1500 and all(r['epoch'] in (25,75,200) for r in physical.values())
    assert len(read(ROOT/'selections_frozen.json')['choices'])==60
    csv_save(ROOT/'checkpoint_reload_audit.csv',reloads)
    save(ROOT/'audit.json',dict(status='passed',finished=now(),physical_unique=len(physical),
        physical_full=sum(r['epoch']==200 for r in physical.values()),actual_new_updates=sum(r['epoch'] for r in physical.values()),
        targets=target_count,races=race_count,prefix_resume_snapshots=prefix_count,reload_count=len(reloads),
        reload_max_error=max(r['max_error'] for r in reloads),sources_inputs_unchanged=True,
        test_after_freeze=True,freeze_sha256=freeze,errors=[]))
