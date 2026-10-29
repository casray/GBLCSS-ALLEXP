"""Frozen SSL screen: prepare, smoke, all, status. No test-label access during fits."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from artifacts import ROOT, CODE, PREP, PROTOCOL as P, read, save, save_tensor, sha, signature, reusable, complete
from data_bridge import load_ssl, seed_all, state_hash

@contextmanager
def single_instance():
    # Windows byte-range lock is released by the OS even after a crash.
    import msvcrt
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runner.lock').open('a+b') as handle:
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        try: msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc: raise RuntimeError('Another SSL runner holds the lock') from exc
        try: yield
        finally: handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)

def rng_state():
    return dict(cpu=torch.get_rng_state(),gpu=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['gpu'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

def prepare():
    env=os.environ.copy()
    # These five pinned repos were downloaded by the sandbox account; only this
    # verifier subprocess trusts their exact paths, without changing global Git.
    repos=['GRACE','GCA','ProGCL','GraphECL','BGRL']
    env['GIT_CONFIG_COUNT']=str(len(repos))
    for i,repo in enumerate(repos):
        env[f'GIT_CONFIG_KEY_{i}']='safe.directory'
        env[f'GIT_CONFIG_VALUE_{i}']=(PREP.parent/'ssl_backbone_selection_20261026'/'upstream'/repo).as_posix()
    env['PYTHONIOENCODING']='utf-8'
    checked=subprocess.run([sys.executable,'-B',str(PREP/'prepare.py'),'verify'],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    if checked.returncode: raise RuntimeError(checked.stdout+checked.stderr)
    verification=json.loads(checked.stdout)
    if verification['source_and_input_check']!='pass': raise RuntimeError(verification)
    if not torch.cuda.is_available(): raise RuntimeError('Registered GPU unavailable')
    free=shutil.disk_usage(ROOT).free/1024**3
    if free<P['minimum_output_free_gib']:raise RuntimeError('Insufficient output free space')
    frozen=signature();receipt=ROOT/'receipt.json'
    if receipt.exists() and read(receipt)['signature']!=frozen:
        raise RuntimeError('Adapter signature changed; retain prior receipt and review before new fits')
    if not receipt.exists():
        save(receipt,dict(signature=frozen,output_root=str(ROOT.resolve()),code_root=str(CODE.resolve()),
             protocol_id=P['protocol_id'],gpu=torch.cuda.get_device_name(0),free_gib=free,
             input_verification=verification,created=time.strftime('%Y-%m-%d %H:%M:%S')))
    return read(receipt)

def export(model,x,edge,method,folder,epoch):
    rng=rng_state();training=model.training
    try:
        model.eval()
        with torch.no_grad():
            raw=(model.embedding(x) if method=='graphecl' else model(x,edge)).cpu()
            h=F.normalize(raw,dim=1)
        if not torch.isfinite(h).all():raise FloatingPointError('Export nonfinite')
        path=folder/f'H_epoch{epoch}.pt'
        save_tensor(path,dict(H=h,raw_H=raw,epoch=epoch))
        save_tensor(folder/f'model_epoch{epoch}.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
        return path
    finally:
        model.train(training);restore_rng(rng)

def ecl_loss_kwargs(cfg):
    return dict(tau=cfg.get('temperature',P['graphecl']['temperature']),lam=cfg.get('lam',P['graphecl']['lam']),lambda_loss=cfg.get('lambda_loss',P['graphecl']['lambda_loss']))

def train_ssl(dataset,seed,method,smoke=False):
    import contrastive_family as family
    import progcl
    import graphecl as ge
    prefix='engineering' if smoke else 'formal'
    folder=ROOT/prefix/'ssl'/dataset/f'seed{seed}'/method
    cfg=dict(P['graphecl']['dataset_configs'][dataset] if method=='graphecl' else P['family']['dataset_configs'][dataset])
    epochs=2 if smoke else (cfg['epochs'] if method=='graphecl' else P['family']['epochs'])
    expected=dict(dataset=dataset,seed=seed,method=method,epochs=epochs,config=cfg,receipt_sha256=sha(ROOT/'receipt.json'))
    final_path=folder/f'H_epoch{epochs}.pt'
    exports=[(method,final_path)]
    if not smoke and method=='grace' and dataset in ('cora','citeseer'):
        exports.append(('grace200',folder/'H_epoch200.pt'))
    if reusable(folder,expected): return exports
    folder.mkdir(parents=True,exist_ok=True)
    data=load_ssl(dataset,seed);x=data['x'].cuda();edge=data['edge_index'].cuda()
    seed_all(seed)
    model=(ge.Model(x.shape[1],cfg) if method=='graphecl' else family.Model(x.shape[1],cfg)).cuda()
    initial=state_hash(model.state_dict())
    if method in ('gca_degree','progcl_weight_degree'):
        grace_config=folder.parent/'grace/config.json'
        if grace_config.exists() and read(grace_config)['initial_hash']!=initial:
            raise RuntimeError('Family initialization differs from paired GRACE')
    if method=='progcl_weight_degree':
        other=folder.parent/'gca_degree/config.json'
        if other.exists() and read(other)['initial_hash']!=initial:
            raise RuntimeError('GCA/ProGCL initialization mismatch')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',P['family']['weight_decay']))
    # Separate model initialization from the paired augmentation RNG stream.
    seed_all(seed)
    weights=family.prepare_weights(x,edge) if method in ('gca_degree','progcl_weight_degree') else None
    if method=='graphecl':edge=ge.with_loops(edge,len(x));adj=ge.adjacency(edge,len(x),x.dtype)
    start_epoch=1;B=None;resume=folder/'resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['expected']!=expected:raise RuntimeError('Resume configuration mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        start_epoch=state['epoch']+1
        logpath=folder/'epochs.jsonl'
        if logpath.exists():
            oldlog=logpath.read_text(encoding='utf-8')
            (folder/f'epochs_before_resume_{time.time_ns()}.jsonl').write_text(oldlog,encoding='utf-8')
            kept=[line for line in oldlog.splitlines() if json.loads(line)['epoch']<=state['epoch']]
            logpath.write_text('\n'.join(kept)+'\n',encoding='utf-8')
    elif (folder/'config.json').exists():
        raise RuntimeError('Partial SSL attempt without checkpoint retained')
    save(folder/'config.json',dict(expected,initial_hash=initial,input=data['metadata'],augmentation_seed=seed))
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (folder/'epochs.jsonl').open('a',encoding='utf-8') as log:
        for epoch in range(start_epoch,epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            if method=='graphecl':
                h,q=model(x,adj);loss=ge.backward_loss(h,q,edge,chunk=P['family']['query_chunk'],**ecl_loss_kwargs(cfg))
            else:
                (x1,e1),(x2,e2)=family.augment_views(x,edge,cfg['edge_drop'],cfg['feature_drop'],weights)
                z1=model.project(model(x1,e1));z2=model.project(model(x2,e2))
                threshold=2 if smoke else 400
                if method=='progcl_weight_degree' and epoch>=threshold and B is None:
                    cached=(folder/'B.npy').exists()
                    B=progcl.weights_for_epoch(epoch,z1,z2,folder/'B.npy',epoch_start=threshold,chunk=512)
                    if cached and epoch==threshold:
                        recovered=np.random.randint(0,len(z1),100).tolist()
                        if recovered!=read(folder/'B.npy.json')['fit_columns']:
                            raise RuntimeError('Recovered ProGCL fit RNG differs from frozen B')
                loss=family.backward_loss(z1,z2,cfg['tau'],chunk=P['family']['query_chunk'],B=B)
            if not np.isfinite(loss):raise FloatingPointError(f'Nonfinite SSL loss at epoch {epoch}')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite SSL gradient at epoch {epoch}')
            optimizer.step()
            row=dict(epoch=epoch,loss=loss,elapsed_seconds=time.perf_counter()-started,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(row)+'\n');log.flush()
            if epoch in (1,399,400,401) or epoch%50==0 or epoch==epochs:
                save(folder/'progress.json',row)
                print(json.dumps(dict(dataset=dataset,seed=seed,method=method,**row)),flush=True)
            if method=='grace' and epoch==200 and dataset in ('cora','citeseer') and not smoke:
                export(model,x,edge,method,folder,epoch)
            if epoch%50==0 or epoch in (399,400) or epoch==epochs:
                save_tensor(resume,dict(expected=expected,epoch=epoch,model=model.state_dict(),
                            optimizer=optimizer.state_dict(),rng=rng_state()))
            if epoch==399 and method in ('gca_degree','progcl_weight_degree'):
                save(folder/'warmup399.json',dict(state_hash=state_hash(model.state_dict())))
                if method=='progcl_weight_degree':
                    other=folder.parent/'gca_degree/warmup399.json'
                    if other.exists() and read(other)['state_hash']!=state_hash(model.state_dict()):
                        raise RuntimeError('GCA/ProGCL paired warmup trajectory differs')
    export(model,x,edge,method,folder,epochs)
    # Reload on CPU, then replay on the registered GPU in eval mode.
    saved=torch.load(folder/f'model_epoch{epochs}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved);model.eval()
    with torch.no_grad():
        check=F.normalize((model.embedding(x) if method=='graphecl' else model(x,edge)).cpu(),dim=1)
    stored=torch.load(final_path,map_location='cpu',weights_only=True)['H']
    difference=float((check-stored).abs().max())
    if difference>1e-7:raise RuntimeError('SSL export reload mismatch')
    if smoke:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state_hash(state['model'])!=state_hash(saved):raise RuntimeError('Resume/model state mismatch')
        before=rng_state();restore_rng(state['rng']); draws=torch.rand(8,device='cuda')
        restore_rng(state['rng']);draws2=torch.rand(8,device='cuda');restore_rng(before)
        if not torch.equal(draws,draws2):raise RuntimeError('Resume RNG replay mismatch')
        if method=='progcl_weight_degree':
            restored=progcl.load_weights(folder/'B.npy')
            if not np.array_equal(restored[:3],B[:3]):raise RuntimeError('B reload mismatch')
        next_hashes=[]
        for replay_index in range(2):
            # Compare the live optimizer's next step against a fresh disk restore.
            if replay_index==1:
                checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
                model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
            model.train();optimizer.zero_grad(set_to_none=True)
            if method=='graphecl':
                hh,qq=model(x,adj);ge.backward_loss(hh,qq,edge,chunk=512,**ecl_loss_kwargs(cfg))
            else:
                (xx1,ee1),(xx2,ee2)=family.augment_views(x,edge,cfg['edge_drop'],cfg['feature_drop'],weights)
                zz1=model.project(model(xx1,ee1));zz2=model.project(model(xx2,ee2))
                family.backward_loss(zz1,zz2,cfg['tau'],chunk=512,B=B)
            optimizer.step();next_hashes.append(state_hash(model.state_dict()))
        if next_hashes[0]!=next_hashes[1]:raise RuntimeError('Resumed next update differs')
        restore_rng(before)
        save(folder/'reload_check.json',dict(export_max_error=difference,resume_state_match=True,rng_replay=True,next_update_hash=next_hashes[0]))
    save(folder/'result.json',dict(expected,status='complete',initial_hash=initial,
         seconds_current_attempt=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    required=['config.json','epochs.jsonl','result.json','resume.pt',f'H_epoch{epochs}.pt',f'model_epoch{epochs}.pt']
    if len(exports)>1:required+=['H_epoch200.pt','model_epoch200.pt']
    if method=='progcl_weight_degree':required+=['B.npy','B.npy.json']
    complete(folder,expected,required)
    del model,optimizer,x,edge,B;gc.collect();torch.cuda.empty_cache()
    return exports

def execute(smoke=False):
    from downstream import run_downstream
    import probe
    prepare();torch.set_num_threads(P['threads']);torch.use_deterministic_algorithms(True)
    prefix='engineering' if smoke else 'formal'
    if not smoke:
        gate=read(ROOT/'engineering/smoke_complete.json')
        if gate['receipt_sha256']!=sha(ROOT/'receipt.json'):raise RuntimeError('Smoke adapter signature differs')
    failures=[];completed=[]
    methods=[m for stage in P['stage_order'] for m in stage]
    for method in methods:
        for dataset in P['datasets']:
            for seed in ([0] if smoke else P['seeds']):
                task=dict(method=method,dataset=dataset,seed=seed)
                save(ROOT/prefix/'status.json',dict(status='running',task=task,completed=completed,failures=failures,pid=os.getpid()))
                try:
                    exports=train_ssl(dataset,seed,method,smoke)
                    for arm,path in exports:
                        run_downstream(dataset,seed,arm,path,smoke)
                        probe.run(path,dataset,seed,arm,smoke)
                    completed.append(task)
                except Exception:
                    error=dict(task=task,traceback=traceback.format_exc(),time=time.strftime('%Y-%m-%d %H:%M:%S'))
                    failures.append(error)
                    save(ROOT/prefix/'failures'/f'{method}_{dataset}_{seed}_{time.time_ns()}.json',error)
                    print(error['traceback'],flush=True)
                    if smoke:raise
                finally:gc.collect();torch.cuda.empty_cache()
    save(ROOT/prefix/'status.json',dict(status='fits_complete' if not failures else 'fits_complete_with_failures',completed=completed,failures=failures))
    if smoke:
        save(ROOT/'engineering/smoke_complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),tasks=completed))
    else:
        save(ROOT/'formal/fits_locked.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),completed=completed,failures=failures))
        from analyze import analyze
        analyze()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','smoke','all','status'])
    args=parser.parse_args()
    if args.command=='status':
        for path in ROOT.glob('*/status.json'):print(path,read(path))
    else:
        with single_instance():
            if args.command=='prepare':print(json.dumps(prepare(),ensure_ascii=False,indent=2))
            else:execute(args.command=='smoke')
