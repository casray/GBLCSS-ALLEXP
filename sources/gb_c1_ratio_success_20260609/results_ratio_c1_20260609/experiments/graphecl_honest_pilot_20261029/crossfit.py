"""One-way, label-isolated short continuation; fixed H0 downstream readout."""
from common import *
from mechanisms import *
import argparse,copy,math,traceback
from contextlib import contextmanager


@contextmanager
def lock():
    import msvcrt
    with (ROOT/'crossfit.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def freeze_sources():
    prepare();p=ROOT/'crossfit/receipt.json'
    paths=[CODE/'crossfit.py',CODE/'analyze.py',CODE/'test_crossfit.py']
    if p.exists():old.verify(read(p)['signature'])
    else:save(p,dict(parent_receipt_sha=sha(ROOT/'receipt.json'),signature={str(q):sha(q) for q in paths}))


def local_packages(d,s):
    data=old.supervised(d,s,CFG['crossfit_condition']);train=data['train'].numpy();h=load(source_ssl(d,s)/'H.pt')
    parts=split_folds(train,s);packages=[]
    for f,positions in enumerate(parts):
        ids=train[positions];dest=ROOT/'crossfit/packages'/d/f'seed{s}'/f'fold{f}'
        a=support(dest/'partition',h,ids,d,s)
        package=fold_package(ids,data['observed'].numpy()[positions],a,data['classes'])
        package['positions']=positions
        changed=data['observed'].numpy().copy();outside=np.setdiff1d(np.arange(len(train)),positions);changed[outside]=(changed[outside]+1)%data['classes']
        check=fold_package(ids,changed[positions],a,data['classes'])
        assert all(np.array_equal(package[k],check[k]) for k in check)
        weights=positive_weights(package['target'],package['group'])
        if not (weights.sum(1)>0).any():raise RuntimeError('Fold has no active positive candidates')
        if (dest/'package.pt').exists():
            stored=load(dest/'package.pt');assert all(torch.equal(stored[k],torch.as_tensor(v)) for k,v in package.items())
        else:save_tensor(dest/'package.pt',{k:torch.as_tensor(v) for k,v in package.items()})
        save(dest/'isolation.json',dict(outside_labels_changed=len(outside),same_package=True,active_fraction=float((weights.sum(1)>0).mean()),
            positive_count_mean=float((weights>0).sum(1).mean()),positive_weight_std=float(weights[weights>0].std()),label_hash=state_hash({'labels':torch.from_numpy(package['observed'])})))
        packages.append(package)
    return data,h,packages


def aux_inputs(package,arm,s,fold,device='cuda'):
    target=package['target'].copy()
    if arm=='shuffle':target=target[np.random.default_rng(s+29100+fold).permutation(len(target))]
    w=positive_weights(target,package['group'])
    return torch.as_tensor(package['ids'],device=device),torch.as_tensor(w,device=device)


def norm_grads(model):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))


def calibrate(d,s,fold,package):
    path=ROOT/'crossfit/calibration'/d/f'seed{s}'/f'fold{fold}.json'
    if path.exists():return read(path)['weight']
    cfg,x,edge,adj=model_data(d,s);model=original_model(d,s,cfg,x);initial=copy.deepcopy(model.state_dict())
    model.train();model.zero_grad(set_to_none=True);g,t=model(x,adj)
    ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**old.loss_kwargs(cfg));base=norm_grads(model)
    model.load_state_dict(initial);model.zero_grad(set_to_none=True);g,t=model(x,adj)
    ids,w=aux_inputs(package,'real',s,fold);q=torch.arange(min(CFG['query_batch'],len(ids)),device='cuda')
    loss=soft_contrastive(t[ids],w,q,CFG['temperature']);loss.backward();aux=norm_grads(model)
    if not np.isfinite(aux) or aux<=1e-12:raise RuntimeError('Invalid auxiliary gradient')
    weight=min(CFG['lambda_cap'],CFG['aux_gradient_ratio']*base/aux)
    save(path,dict(weight=weight,base_gradient_norm=base,aux_gradient_norm=aux,actual_ratio=weight*aux/base,
                   package_hash=state_hash({k:torch.as_tensor(v) for k,v in package.items()})))
    del model,x,edge,adj,g,t,ids,w;gc.collect();torch.cuda.empty_cache();return weight


def one_step(model,optimizer,x,adj,edge,cfg,ids,weights,queries,weight):
    model.train();optimizer.zero_grad(set_to_none=True);g,t=model(x,adj)
    aux=t.sum()*0 if ids is None else soft_contrastive(t[ids],weights,queries,CFG['temperature'])
    if ids is not None and weight>0:(weight*aux).backward(retain_graph=True)
    loss=ge.backward_loss(g,t,edge,chunk=CFG['chunk'],**old.loss_kwargs(cfg))
    if not math.isfinite(loss) or not torch.isfinite(aux) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite training')
    optimizer.step();return loss,float(aux.detach())


def train_branch(d,s,arm,fold=None,package=None,epochs=None,stage='formal'):
    epochs=CFG['epochs'] if epochs is None else epochs
    dest=ROOT/'crossfit'/stage/'ssl'/d/f'seed{s}'/(arm if fold is None else f'{arm}_fold{fold}')
    weight=0. if package is None else calibrate(d,s,fold,package)
    package_hash=None if package is None else state_hash({k:torch.as_tensor(v) for k,v in package.items()})
    expected=dict(dataset=d,seed=s,arm=arm,fold=fold,epochs=epochs,weight=weight,package_hash=package_hash,
        original_model_sha=sha(source_ssl(d,s)/'model.pt'),receipt_sha=sha(ROOT/'crossfit/receipt.json'))
    if old.art.reusable(dest,expected):return dest
    cfg,x,edge,adj=model_data(d,s);seed_all(s);model=original_model(d,s,cfg,x)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']*CFG['lr_multiplier'],weight_decay=cfg['weight_decay'])
    ids,weights=(None,None) if package is None else aux_inputs(package,arm,s,fold)
    dest.mkdir(parents=True,exist_ok=True);start=1
    if (dest/'resume.pt').exists():
        ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
        assert ck['expected']==expected
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start=ck['epoch']+1
    save(dest/'config.json',expected);started=time.perf_counter();logpath=dest/'epochs.jsonl'
    if logpath.exists():
        rows=[json.loads(v) for v in logpath.read_text(encoding='utf-8').splitlines()]
        if any(r['epoch']>=start for r in rows):
            old.art.atomic(dest/f'prior_{time.time_ns()}.jsonl',lambda p:Path(p).write_bytes(logpath.read_bytes()))
            old.art.atomic(logpath,lambda p:Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start),encoding='utf-8'))
    with logpath.open('a',encoding='utf-8') as log:
        for epoch in range(start,epochs+1):
            queries=None if ids is None else torch.as_tensor(np.random.default_rng(s+29200+fold*100+epoch).choice(len(ids),min(len(ids),CFG['query_batch']),replace=False),device='cuda')
            loss,aux=one_step(model,optimizer,x,adj,edge,cfg,ids,weights,queries,weight)
            row=dict(epoch=epoch,ecl_loss=loss,aux_loss=aux,lambda_weight=weight,seconds_current_attempt=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n');log.flush()
            if epoch%5==0 or epoch==epochs:
                save_tensor(dest/'resume.pt',dict(expected=expected,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict()))
                status('CROSSFIT_TRAIN',dataset=d,seed=s,arm=arm,fold=fold,**row)
    model.eval()
    with torch.no_grad():h=F.normalize(model.embedding(x).cpu(),dim=1)
    save_tensor(dest/'H.pt',h);save_tensor(dest/'model.pt',model.state_dict())
    replay=original_model(d,s,cfg,x);replay.load_state_dict(load(dest/'model.pt'));replay.eval()
    with torch.no_grad():h2=F.normalize(replay.embedding(x).cpu(),dim=1)
    if not torch.equal(h,h2):raise RuntimeError('Encoder GPU replay mismatch')
    save(dest/'result.json',dict(expected,status='complete',gpu_embedding_bitwise=True,seconds_current_attempt=time.perf_counter()-started))
    old.art.complete(dest,expected,['config.json','epochs.jsonl','resume.pt','H.pt','model.pt','result.json'])
    del model,replay,x,edge,adj,optimizer,ids,weights;gc.collect();torch.cuda.empty_cache();return dest


def fold_target(d,s,name,h,data,positions):
    ids=data['train'].numpy()[positions]
    folder=ROOT/'crossfit/targets'/d/f'seed{s}'/name
    a=support(folder/'partition',h,ids,d,s)
    z=data['observed'].numpy()[positions];detail=vote_details(z,a[ids],data['classes'])
    reference=old.down.build_support32_targets(torch.from_numpy(z),torch.from_numpy(a[ids]),classes=data['classes'])
    error=float(np.abs(reference.numpy()-detail['target']).max())
    if error>2e-7:raise RuntimeError('Fold target mismatch')
    for row in range(min(10,len(z))):
        z2=z.copy();z2[row]=(z2[row]+1)%data['classes'];v2=vote_details(z2,a[ids],data['classes'])
        assert np.array_equal(detail['q'][row],v2['q'][row]) and detail['alpha'][row]==v2['alpha'][row]
    save_tensor(folder/'target.pt',reference);save(folder/'audit.json',dict(own_noise_invariance_checked=min(10,len(z)),target_rebuild_max_error=error))
    return reference


def run():
    freeze_sources()
    if not (ROOT/'crossfit/smoke/passed.json').exists():raise RuntimeError('Smoke check required')
    heads=[];ssls=[]
    for d in CFG['datasets']:
        for s in CFG['seeds']:
            status('CROSSFIT_CELL',dataset=d,seed=s)
            data,h0,packages=local_packages(d,s);n=len(data['train']);classes=data['classes'];c=CFG['crossfit_condition']
            b0=torch.empty(n,classes)
            for f,p in enumerate(packages):b0[p['positions']]=torch.as_tensor(p['target'])
            targets=dict(b0=b0)
            base=train_branch(d,s,'ecl');ssls.append(str(base));hb=load(base/'H.pt')
            targets['a20']=fold_target(d,s,'a20',hb,data,np.arange(n));targets['b20']=torch.empty(n,classes)
            for f,p in enumerate(packages):targets['b20'][p['positions']]=fold_target(d,s,f'b20_fold{f}',hb,data,p['positions'])
            for arm in ['real','shuffle']:
                combined=torch.empty(n,classes)
                for f,p in enumerate(packages):
                    source=train_branch(d,s,arm,f,p);ssls.append(str(source));other=packages[1-f]
                    combined[other['positions']]=fold_target(d,s,f'{arm}_vote_fold{1-f}',load(source/'H.pt'),data,other['positions'])
                targets[arm]=combined
            for arm,target in targets.items():
                folder=ROOT/'crossfit/formal/heads'/d/f'seed{s}'/arm
                fit_head(folder,h0,target,data,d,s,c,arm);verify_head(folder,h0);heads.append(str(folder))
    save(ROOT/'crossfit/fits_locked.json',dict(heads=heads,ssls=ssls,expected_ssl=20,expected_heads=20))
    status('CROSSFIT_FITS_COMPLETE',ssl=len(ssls),heads=len(heads))


def smoke():
    freeze_sources();d,s=CFG['datasets'][0],CFG['seeds'][0]
    data,h,packages=local_packages(d,s);p=packages[0]
    cfg,x,edge,adj=model_data(d,s);ids,w=aux_inputs(p,'real',s,0);queries=torch.arange(min(len(ids),CFG['query_batch']),device='cuda')
    # Same fold-local input produces identical full-model updates after unrelated-label mutation.
    states=[]
    weight=calibrate(d,s,0,p)
    for _ in range(2):
        seed_all(s);model=original_model(d,s,cfg,x);opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']*CFG['lr_multiplier'],weight_decay=cfg['weight_decay'])
        one_step(model,opt,x,adj,edge,cfg,ids,w,queries,weight);states.append(state_hash(model.state_dict()))
        del model,opt;gc.collect();torch.cuda.empty_cache()
    assert states[0]==states[1]
    del x,edge,adj,ids,w;gc.collect();torch.cuda.empty_cache()
    path=train_branch(d,s,'real',0,p,epochs=1,stage='smoke')
    save(ROOT/'crossfit/smoke/passed.json',dict(status='passed',identical_local_input_updates=True,model_hash=states[0],one_epoch_path=str(path)))
    status('CROSSFIT_SMOKE_PASSED')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','run']);args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    try:
        with lock():(smoke if args.mode=='smoke' else run)()
    except Exception as exc:
        status('CROSSFIT_FAILED',error=repr(exc));traceback.print_exc();raise
