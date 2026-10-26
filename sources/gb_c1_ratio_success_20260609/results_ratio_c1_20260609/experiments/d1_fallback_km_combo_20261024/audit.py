"""Artifact, selector, target, initialization and frozen-protocol audit."""
import json
from pathlib import Path
import torch

from protocol import ROOT,CONFIG,read,save,sha,now,graph_inputs,state_hash,csv_save
from rules import choose,refinement_cost
from targets import shrink,groups
from fit import Student
from analyze import validate_freeze


def audit():
    from run import OLD,oldfolder,probe_path,prepare,to_gpu
    validate_freeze()
    for filename in ['sources.json','inputs.json']:
        for path,digest in read(ROOT/filename).items():
            assert sha(path)==digest, path
    refs=read(ROOT/'reuse_manifest.json')
    physical=read(ROOT/'physical_fits.json')
    assert 0<len(physical)<=read(ROOT/'inventory.json')['logical_fit_upper']<=840
    assert len(list((ROOT/'fits').rglob('result.json')))==len(physical)
    done={}
    for ref in refs.values():
        folder=Path(ref['folder'])
        if str(folder) in done:
            continue
        r=read(folder/'result.json')
        assert sha(folder/'result.json')==ref['result_sha256']
        assert r['epoch']==200 and r['status']=='complete'
        assert r['q'] is None
        for name,key in [('checkpoint.pt','checkpoint_sha256'),('probabilities.pt','probabilities_sha256')]:
            assert sha(folder/name)==r[key]
        logs=[json.loads(v) for v in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [v['epoch'] for v in logs]==list(range(1,201))
        top=max(logs,key=lambda v:v['validation_correct'])
        assert (r['best_epoch'],r['valid_correct'],r['valid_nll'],r['valid_macro_f1'])==(
            top['epoch'],top['validation_correct'],top['validation_nll'],top['validation_macro_f1'])
        ce=read(oldfolder('D1',r['dataset'],r['seed'],r['backbone'],r['condition'],'CE')/'result.json')
        assert (r['initial_hash'],r['initial_rng_hash'],r['input_hash'])==(ce['initial_hash'],ce['initial_rng_hash'],ce['input_hash'])
        assert state_hash({'target':torch.load(ref['target_path'],weights_only=True)})==r['target_hash']
        assert r['source_sha256']==sha((OLD if ref['historical'] else ROOT)/'sources.json')
        if not ref['historical']:
            assert not (folder/'resume.pt').exists()
        done[str(folder)]=r
    selections=[read(p) for p in (ROOT/'selections').rglob('*.json')]
    assert len(selections)==120
    checks,reloads=0,[]
    for n in CONFIG['datasets']:
        for s in CONFIG['seeds']:
            g=graph_inputs(n,s)
            for c in CONFIG['conditions']:
                values,es=prepare(n,s,c,g)
                for cell in [v for v in selections if (v['dataset'],v['seed'],v['condition'])==(n,s,c)]:
                    for selector,methods in cell['choices'].items():
                        for method,selected in methods.items():
                            pool=[(v['ref'],done[refs[v['ref']]['folder']]) for v in cell['pools'][method]]
                            assert choose(pool,selector)==selected
                            checks+=1
                    for method,size in [('LEGACY_GLOBAL',4),('LEGACY_EXPANDED',8),('LEGACY_FORCED',4)]:
                        assert len(cell['pools'][method])==size
                    if cell['backbone']=='R0':
                        for method,size in [('KM_GLOBAL',4),('KM_EXPANDED',8),('FIXED_GLOBAL',4),('FIXED_EXPANDED',8)]:
                            assert len(cell['pools'][method])==size
                        probe=read(probe_path(n,s,'R0',c))
                        assert sha(probe_path(n,s,'R0',c))==cell['meta']['probe_sha256']
                        support=cell['meta']['adaptive_support']
                        assert cell['meta']['adaptive_cost']==refinement_cost(probe,support)
                        if support is None:
                            assert cell['pools']['ADAPTIVE_EXPANDED']==cell['pools']['CE']
                        else:
                            assert all(refs[r['ref']]['partition']==f'S{support}' for r in cell['pools']['ADAPTIVE_EXPANDED'])
                    for key,ref in refs.items():
                        if (ref['dataset'],ref['seed'],ref['backbone'],ref['condition'])!=(n,s,cell['backbone'],c):
                            continue
                        e=es[ref['partition']]
                        arm=ref['arm']
                        if arm=='CE':
                            target=e['own']
                        elif arm.startswith('L'):
                            target=shrink(e,float(arm[1:]))
                        else:
                            a,d=[float(v[1:]) for v in arm.split('_')[1:]]
                            p=ref['partition']
                            tie=cell['meta']['legacy_tie_lambda'] if p=='LEGACY' else (
                                min([0.,.25,.5,1.],key=lambda l:(-done[refs[cell['pools']['KM_GLOBAL'][[0.,.25,.5,1.].index(l)]['ref']]['folder']]['validation'],l))
                                if p=='KM' else cell['meta']['deep_ties'][p])
                            gr=groups(e)
                            target=shrink(e,torch.where(gr==0,a,torch.where(gr==1,d,tie)).double())
                        assert state_hash({'target':target})==ref['target_hash'],key
                if c=='uniform0.4':
                    gpu=to_gpu(values)
                    for kind in ['KM','DEEP']:
                        samples=[r for r in physical.values() if r['dataset']==n and r['seed']==s and r['condition']==c and
                            (r['partition']=='KM' if kind=='KM' else r['partition'].startswith('S'))]
                        if not samples:
                            continue
                        folder=ROOT/samples[0]['folder']
                        student=Student('R0',gpu,g['encoder'],s)
                        student.set_state(torch.load(folder/'checkpoint.pt',weights_only=True)['model'])
                        student.model.eval()
                        with torch.no_grad():
                            error=float((student.forward().softmax(1).cpu()-torch.load(folder/'probabilities.pt',weights_only=True)).abs().max())
                        assert error<3e-7
                        reloads.append(dict(folder=samples[0]['folder'],error=error))
                        del student
                    del gpu
            del g,values,es
            torch.cuda.empty_cache()
    assert read(ROOT/'testing_started.json')['started']>=read(ROOT/'selections_frozen.json')['frozen']
    csv_save(ROOT/'checkpoint_reload_audit.csv',reloads)
    save(ROOT/'audit.json',dict(status='passed',finished=now(),new_physical_fits=len(physical),
        logical_references=len(refs),unique_verified_artifact_sets=len(done),cells=len(selections),selector_checks=checks,
        reconstructed_targets=len(refs),reloads=len(reloads),reload_max_error=max(r['error'] for r in reloads),
        historical_sources_unchanged=True,test_after_freeze=True,errors=[]))
