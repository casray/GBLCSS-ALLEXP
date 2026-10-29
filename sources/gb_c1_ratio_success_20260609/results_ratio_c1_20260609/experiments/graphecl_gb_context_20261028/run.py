"""Audited context preflight, shared-warmup SSL, and frozen native-head fitting."""
import argparse,gc,json,math,os,time,traceback,copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from runtime import *
import context as ctx

def status(stage,**kwargs):
    save(ROOT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs))
    print(stage,json.dumps(kwargs,ensure_ascii=False),flush=True)

def build_context(h,edge,seed,dest,stage,small=False):
    expected=dict(h_hash=state_hash({'H':h}),edge_hash=state_hash({'edge':edge}),seed=seed,stage=stage,small=small)
    if art.reusable(dest,expected):return load(dest/'context.pt')
    start=time.perf_counter();e=edge[:,edge[0]!=edge[1]].numpy();n=len(h)
    built=tree.build_tree(h.numpy(),np.arange(n),max_leaves=4 if small else CFG['max_leaves'],snapshot_leaves=4 if small else CFG['max_leaves'],min_nodes=2 if small else CFG['min_nodes'],min_train=1,seed=seed)
    assignment=built['fine_assignment'];r=ctx.context_scores(e,assignment,seed)
    cos=ctx.cosine_scores(h.numpy(),e);gb=ctx.center_scores(e,r['scores'],n,r['active'])
    cd=ctx.center_scores(e,cos,n,r['active']);sh=ctx.shuffle_delta(e,gb,seed)
    # ge.with_loops preserves original non-self edge order, then appends N self edges.
    cache={k:torch.from_numpy(np.r_[v,np.zeros(n)]).float() for k,v in [('gb',gb),('cos',cd),('shuffle',sh)]}
    cache.update(assignment=torch.from_numpy(assignment),scores=torch.from_numpy(r['scores']),cosine_scores=torch.from_numpy(cos),active=torch.from_numpy(r['active']),fold=torch.from_numpy(r['fold']))
    corr=float(spearmanr(r['scores'],cos).statistic) if len(e[0]) and np.ptp(r['scores'])>0 and np.ptp(cos)>0 else None
    stats=dict(stage=stage,nodes=n,edges=len(e[0]),balls=int(assignment.max())+1,active_fraction=float(r['active'].mean()),degree_le1_fraction=float((r['degree']<=1).mean()),spearman=corr,seconds=time.perf_counter()-start,doses={})
    for eta in [.25,1.]:
        change=np.abs(eta*gb)
        stats['doses'][str(eta)]=dict(relative_change_quantiles=np.quantile(change,[0,.25,.5,.75,.9,.99,1]).tolist(),fraction_above_10pct=float((change>.1).mean()),cos_change_mean=float(np.abs(eta*cd).mean()))
    stats['advisories']=dict(high_correlation=corr is not None and abs(corr)>.8,weak_low_dose=stats['doses']['0.25']['fraction_above_10pct']<.2)
    if not np.isfinite(gb).all() or np.max(abs(gb),initial=0)>1+1e-10:raise RuntimeError('Invalid context weights')
    if not small and np.max(abs(gb),initial=0)<1e-10:raise RuntimeError('No context intervention: retain preflight and inspect geometry')
    save_tensor(dest/'context.pt',cache);save(dest/'stats.json',stats)
    art.complete(dest,expected,['context.pt','stats.json'])
    return cache

def preflight():
    prepare();reports=[]
    for d in CFG['primary_datasets']:
        for s in CFG['seeds']:
            status('TERMINAL_PREFLIGHT',dataset=d,seed=s)
            cfg=dataset_config(d);h=load(old_ssl(d,s)/f"H_epoch{cfg['epochs']}.pt")['H'].float()
            data=features(d,s);p=ROOT/'preflight/terminal'/d/f'seed{s}'
            build_context(h,data['edge_index'],s,p,'terminal_offline')
            reports.append(dict(dataset=d,seed=s,**read(p/'stats.json')))
            gc.collect()
    save(ROOT/'preflight/terminal_summary.json',reports);status('TERMINAL_PREFLIGHT_COMPLETE',cells=len(reports))

def export_h(model,x,path):
    before=rng_state();training=model.training
    try:
        model.eval()
        with torch.no_grad():h=F.normalize(model.embedding(x).cpu(),dim=1)
        if not torch.isfinite(h).all():raise FloatingPointError('Nonfinite H')
        save_tensor(path,h);return h
    finally:model.train(training);restore_rng(before)

def grad_diagnostics(model,x,adj,edge,cache,cfg,dest):
    before=rng_state();training=model.training;state=copy.deepcopy(model.state_dict())
    model.train();model.zero_grad(set_to_none=True);g,t=model(x,adj);ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**loss_kwargs(cfg))
    baseline=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    base_norm=math.sqrt(sum(float(v.square().sum()) for v in baseline));rows=[]
    for kind in ['gb','cos','shuffle']:
        model.load_state_dict(state);restore_rng(before);model.zero_grad(set_to_none=True);g,t=model(x,adj)
        src,dst=edge;deg=torch.bincount(dst,minlength=len(x)).to(g.dtype)
        corr=-(cache[kind].to(x.device)/deg[dst]*(F.normalize(g,dim=1)[src]*F.normalize(t,dim=1)[dst]).sum(1)/cfg['temperature']).sum()/len(x)
        corr.backward();diff=[torch.zeros_like(p) if p.grad is None else p.grad.detach() for p in model.parameters()]
        norm=math.sqrt(sum(float(v.square().sum()) for v in diff));dot=sum(float((a*b).sum()) for a,b in zip(baseline,diff))
        rows.append(dict(kind=kind,gradient_ratio_at_eta1=norm/max(base_norm,1e-30),gradient_cosine=dot/max(base_norm*norm,1e-30),correction_at_eta1=float(corr.detach())))
    model.load_state_dict(state);model.zero_grad(set_to_none=True);model.train(training);restore_rng(before)
    save(dest,dict(baseline_gradient_norm=base_norm,rows=rows))

def train_endpoint(d,s,arm,small=False):
    stage='smoke' if small else 'formal';folder=ssl_dir(d,s,arm,stage);cfg=dataset_config(d)
    if small:cfg=dict(cfg,epochs=4)
    T=cfg['epochs'];mid=T//2;expected=dict(dataset=d,seed=s,arm=arm,config=cfg,receipt_sha256=sha(ROOT/'receipt.json'),stage=stage)
    if art.reusable(folder,expected):return folder
    data=features(d,s);x=data['x'].cuda();edge=ge.with_loops(data['edge_index'].cuda(),len(x));adj=ge.adjacency(edge,len(x),x.dtype)
    seed_all(s);model=ge.Model(x.shape[1],cfg).cuda();initial=state_hash(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);seed_all(s)
    shared=ROOT/stage/'warmup'/d/f'seed{s}';cache=None
    resume=folder/'resume.pt';start_epoch=1
    if resume.exists():
        checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
        if checkpoint['expected']!=expected:raise RuntimeError('Resume signature mismatch')
        start_epoch=checkpoint['epoch']+1
    elif arm!='ecl':
        checkpoint=torch.load(shared/'checkpoint.pt',map_location='cpu',weights_only=False);start_epoch=mid+1
        if checkpoint['epoch']!=mid:raise RuntimeError('Wrong shared warmup epoch')
    else:checkpoint=None
    if checkpoint is not None:
        restore_snapshot(model,optimizer,checkpoint)
    if start_epoch>mid:cache=load(shared/'context/context.pt')
    folder.mkdir(parents=True,exist_ok=True);save(folder/'config.json',dict(expected,initial_hash=initial))
    logpath=folder/'epochs.jsonl'
    if logpath.exists():
        old=logpath.read_text(encoding='utf-8');rows=[json.loads(v) for v in old.splitlines()]
        if any(r['epoch']>=start_epoch for r in rows):
            (folder/f'epochs_prior_resume_{time.time_ns()}.jsonl').write_text(old,encoding='utf-8')
            logpath.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start_epoch),encoding='utf-8')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with logpath.open('a',encoding='utf-8') as log:
        for epoch in range(start_epoch,T+1):
            model.train();optimizer.zero_grad(set_to_none=True);g,t=model(x,adj)
            eta=CFG['arms'][arm]*min(1,max(0,(epoch-mid)/math.ceil(.1*T)))
            if arm=='ecl' or eta==0:loss=ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**loss_kwargs(cfg))
            else:
                kind='gb' if arm.startswith('gb') else 'cos' if arm.startswith('cos') else 'shuffle'
                loss=ctx.backward_loss(g,t,edge,cache[kind].to(x.device),eta,chunk=CFG['chunk'],**loss_kwargs(cfg))
            if not math.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite SSL update')
            optimizer.step();row=dict(epoch=epoch,loss=loss,eta=eta,seconds_current_attempt=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n');log.flush()
            if epoch==mid and arm=='ecl':
                record=dict(expected=expected,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state())
                save_tensor(shared/'checkpoint.pt',record);h=export_h(model,x,shared/'H.pt')
                cache=build_context(h,data['edge_index'],s,shared/'context','midpoint_replay',small)
                grad_diagnostics(model,x,adj,edge,cache,cfg,shared/'gradient_diagnostics.json')
            if epoch%25==0 or epoch in (1,mid,T):
                save_tensor(resume,dict(expected=expected,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state()))
                status('SSL',dataset=d,seed=s,arm=arm,epoch=epoch,total=T,stage_group=stage)
    h=export_h(model,x,folder/'H.pt');save_tensor(folder/'model.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
    audit=dict(initial_hash=initial,seconds_current_attempt=time.perf_counter()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),source='new baseline replay' if arm=='ecl' else 'shared midpoint continuation')
    if arm=='ecl' and d!='roman' and not small:
        old=old_ssl(d,s);old_model=load(old/f'model_epoch{T}.pt');old_h=load(old/f'H_epoch{T}.pt')['H']
        audit.update(old_state_equal=state_hash(old_model)==state_hash(load(folder/'model.pt')),old_h_max_error=float((h-old_h).abs().max()))
        if not audit['old_state_equal'] or audit['old_h_max_error']>1e-7:raise RuntimeError('Baseline replay differs from legacy; inspect before intervention')
    # Saved endpoint forward replay, without changing the exported file.
    model.load_state_dict(load(folder/'model.pt'));model.eval()
    with torch.no_grad():audit['reload_error']=float((F.normalize(model.embedding(x).cpu(),dim=1)-h).abs().max())
    if audit['reload_error']>1e-7:raise RuntimeError('SSL reload mismatch')
    if small:
        ck=torch.load(resume,map_location='cpu',weights_only=False);next_hash=[]
        for _ in range(2):
            restore_snapshot(model,optimizer,ck);model.train();optimizer.zero_grad(set_to_none=True);g,t=model(x,adj)
            if arm=='ecl':ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**loss_kwargs(cfg))
            else:ctx.backward_loss(g,t,edge,cache['gb'].cuda(),CFG['arms'][arm],chunk=CFG['chunk'],**loss_kwargs(cfg))
            optimizer.step();next_hash.append(state_hash(model.state_dict()))
        if next_hash[0]!=next_hash[1]:raise RuntimeError('Resume next-step mismatch')
        audit['resume_step_equal']=True
    save(folder/'result.json',dict(expected,status='complete',audit=audit));art.complete(folder,expected,['config.json','epochs.jsonl','H.pt','model.pt','resume.pt','result.json'])
    del model,optimizer,x,edge,adj,g,t;gc.collect();torch.cuda.empty_cache();return folder

def fit_heads(d,s,arm,small=False):
    stage='smoke' if small else 'formal';hpath=ssl_dir(d,s,arm,stage)/'H.pt';h=F.normalize(load(hpath).float(),dim=1)
    for c in (CFG['conditions'][:1] if small else CFG['conditions']):
        data=supervised(d,s,c);seed_all(s);m=torch.nn.Linear(h.shape[1],data['classes']);initial={k:v.clone() for k,v in m.state_dict().items()}
        for head in (['CE','fixed'] if arm=='ecl' else ['CE','fixed','own']):
            if head=='CE':target=F.one_hot(data['observed'],data['classes']).float();target_source='observed training labels'
            else:
                origin='ecl' if head=='fixed' else arm;origin_path=ssl_dir(d,s,origin,stage)/'H.pt'
                ass=down._support_assignment(ROOT/stage/'targets'/d/f'seed{s}'/origin,F.normalize(load(origin_path).float(),dim=1),data['train'],d,s,32)
                target=down.build_support32_targets(data['observed'],ass[data['train']],classes=data['classes']);target_source=str(origin_path)
                if head=='fixed' and d!='roman' and not small:
                    if not torch.equal(target,load(old_head(d,s,c,'SUPPORT32')/'target.pt')):raise RuntimeError('Original SUPPORT32 target mismatch')
            path=head_dir(d,s,arm,c,head,stage)
            expected=dict(dataset=d,seed=s,arm=arm,condition=c,head=head,h_sha256=sha(hpath),initial_hash=state_hash(initial),target_hash=state_hash({'target':target}),target_source=target_source,epochs=2 if small else CFG['head_epochs'],lr=CFG['head_lr'],weight_decay=CFG['head_wd'],receipt_sha256=sha(ROOT/'receipt.json'))
            fit_linear(path,h,target,data,initial,expected)
            if arm=='ecl' and d!='roman' and not small:
                old=old_head(d,s,c,'CE' if head=='CE' else 'SUPPORT32');new=read(path/'result.json');prior=read(old/'result.json')
                vals=lambda p:[json.loads(v)['validation_correct'] for v in (p/'epochs.jsonl').read_text().splitlines()]
                err=float((load(path/'probabilities.pt')-load(old/'probabilities.pt')).abs().max())
                rec=dict(probability_error=err,validation_trajectory_equal=vals(path)==vals(old),best_epoch_equal=new['best_epoch']==prior['best_epoch'])
                save(path/'legacy_replay.json',rec)
                if err>2e-5 or not rec['validation_trajectory_equal'] or not rec['best_epoch_equal']:raise RuntimeError('Native head differs from legacy')

def smoke():
    prepare();status('SMOKE_START')
    for arm in ['ecl','gb025','gb100']:
        train_endpoint('cora',0,arm,True);fit_heads('cora',0,arm,True)
    # Roman data/split/noise contracts and Photo architecture have separate input checks.
    r=features('roman',0);sd=supervised('roman',0,'uniform0.4')
    assert not set(sd['train'].tolist())&set(sd['valid'].tolist())
    assert sd['observed'].shape==sd['train'].shape and torch.isfinite(r['x']).all()
    cfg=dataset_config('amazonpho');assert cfg['num_mlp']==2 and cfg['lam']==.1
    save(ROOT/'smoke/complete.json',dict(receipt_sha256=sha(ROOT/'receipt.json'),status='passed',roman_nodes=len(r['x']),photo_config=True));status('SMOKE_PASSED')

def execute():
    receipt=prepare();gate=read(ROOT/'smoke/complete.json')
    assert gate['status']=='passed' and gate['receipt_sha256']==sha(ROOT/'receipt.json')
    assert len(read(ROOT/'preflight/terminal_summary.json'))==10
    completed=[];failures=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            for arm in CFG['arms']:
                task=dict(dataset=d,seed=s,arm=arm)
                try:
                    train_endpoint(d,s,arm);fit_heads(d,s,arm);completed.append(task)
                    save(ROOT/'formal/completed.json',dict(completed=completed,failures=failures));status('ENDPOINT_COMPLETE',**task,count=len(completed),total=84)
                except Exception:
                    failure=dict(task=task,traceback=traceback.format_exc(),time=time.strftime('%Y-%m-%d %H:%M:%S'));failures.append(failure)
                    save(ROOT/'failures'/f'{d}_{s}_{arm}_{time.time_ns()}.json',failure);status('FAILED',**task,error=failure['traceback']);raise
                finally:gc.collect();torch.cuda.empty_cache()
    verify(receipt['signature'])
    save(ROOT/'formal/fits_locked.json',dict(status='complete',completed=completed,failures=failures,receipt_sha256=sha(ROOT/'receipt.json')))
    from analyze import analyze
    analyze();status('FINISHED',endpoints=len(completed),heads=720)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','smoke','all','status']);args=parser.parse_args()
    if args.command=='status':print(read(ROOT/'status.json'))
    else:
        torch.set_num_threads(CFG['threads']);torch.use_deterministic_algorithms(True)
        with single_instance():
            if args.command=='prepare':print('Frozen source files',len(prepare()['signature']))
            elif args.command=='preflight':preflight()
            elif args.command=='smoke':smoke()
            else:execute()
