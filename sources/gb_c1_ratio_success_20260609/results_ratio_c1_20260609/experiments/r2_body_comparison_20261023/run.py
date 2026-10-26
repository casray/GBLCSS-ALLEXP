"""Matched frozen-R2 supervision and author-core external controls."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
sys.path.insert(0,str(EXP/'sc_support_channel_pilot_20261023'))
import fit_pipeline as f
f.ROOT=ROOT
from core import AlignedMLP,state_hash
from gblcss.baselines import structure as st
sys.path.insert(0,str(EXP/'channel_routing_variance_20261023'))
from metrics import accuracy_f1

INPUTS={}
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as z:
        for b in iter(lambda:z.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load(path):
    INPUTS[str(path)]=sha(path)
    return torch.load(path,map_location='cpu',weights_only=True)
def loss(logits,z,target,arm):
    lp=logits.log_softmax(1)
    if arm=='GCE':return ((1-torch.exp(.7*lp.gather(1,z[:,None])))/.7).mean()
    return -(target*lp).sum(1).mean()
def model(x,classes,h_dim,seed,encoder):
    f.set_seed(seed)
    m=AlignedMLP(x.shape[1],classes,h_dim,64,.5).cuda()
    f.copy_encoder(m,encoder);m.freeze_encoder()
    return m
def fixed_data(name,seed):
    b=EXP/'sc_support_channel_pilot_20261023/diagnostics/e1'/name/f'seed{seed}'
    return b,load(b/'geometry.pt')
def targets(name,seed,cond,geo,pack,edges,classes,km):
    tr=geo['train'];z=pack['observed']
    labels=torch.full((len(geo['legacy32']),),-1,dtype=torch.long);labels[tr]=z
    vals={'CE':F.one_hot(z,classes).float()}
    with np.load(EXP/'main_protocol_replay_20260922/matrix'/name/'default'/f'seed{seed}.npz') as old:
        oldpart=torch.from_numpy(old['assignment']).long()
    for arm,part in [('OLD64',oldpart),('SUPPORT',geo['legacy32']),('KM',km)]:
        vals[arm]=torch.from_numpy(f.r2.numpy_targets(z.numpy(),part[tr].numpy(),classes,.5,2)).float()
    common=dict(num_nodes=len(labels),n_classes=classes)
    vals['POSTEL']=st.build_postel_targets(labels,tr,edges,**common)[0][tr]
    vals['SALS']=st.build_sals_targets(labels,tr,edges,**common)[0][tr]
    prop=st.als_label_propagation(labels,tr,edges,**common)[0]
    for v in vals.values():
        assert torch.isfinite(v).all() and torch.allclose(v.sum(1),torch.ones(len(tr)),atol=2e-6)
    return vals,prop
def fit_head(name,seed,cond,arm,x,hidden,enc,classes,geo,pack,target,prop,h_dim,epochs=None):
    folder=ROOT/'fits'/name/f'seed{seed}'/cond/arm
    if (folder/'result.json').exists():
        r=f.read(folder/'result.json')
        if r['status']!='complete':raise RuntimeError('Partial fit '+str(folder))
        assert sha(folder/'probabilities.pt')==r['probabilities_sha256'];return r
    folder.mkdir(parents=True,exist_ok=True)
    m=model(x,classes,h_dim,seed,enc);frozen=state_hash(m.encoder_state());initial=state_hash(m.deployment_state())
    tr,vi=geo['train'].cuda(),geo['valid'].cuda();z=pack['observed'].cuda();yv=pack['valid_y'].cuda();t=target.cuda()
    params=list(m.classifier_parameters());rel=None
    if arm=='ALS':
        # Isolated auxiliary RNG preserves the common classifier initialization.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed+735); rel=torch.nn.Parameter(torch.empty(classes,classes,device='cuda'));torch.nn.init.xavier_uniform_(rel)
        params.append(rel)
    opt=torch.optim.Adam(params,lr=.01,weight_decay=.0005)
    n=epochs or CFG['head_epochs'];best=-1;beststate=None;started=time.perf_counter()
    torch.save(target,folder/'target.pt')
    meta=dict(dataset=name,seed=seed,condition=cond,arm=arm,epochs=n,initial_hash=initial,encoder_hash=frozen,target_hash=state_hash({'target':target}))
    save(folder/'config.json',meta);save(folder/'result.json',dict(meta,status='running'))
    try:
        with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
            for ep in range(1,n+1):
                m.train_head();opt.zero_grad(set_to_none=True);logits=m.backbone.layers[-1](hidden[tr])
                if arm=='ALS':
                    soft=F.softmax(prop[tr]@rel.t(),dim=1)
                    alpha=st.als_pacing_alpha(ep-1,pacing='exponential',alpha_max=.1,rate=-.001,initial=.1)
                    value=st.als_eq5_loss(logits,z,soft,alpha=alpha,kl_weight=.001)
                else:value=loss(logits,z,t,arm)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite gradient')
                opt.step();m.eval()
                with torch.no_grad():correct=int((m.backbone.layers[-1](hidden[vi]).argmax(1)==yv).sum())
                if correct>best:best=correct;selected=ep;beststate=deepcopy(m.deployment_state());bestrel=None if rel is None else rel.detach().cpu().clone()
                log.write(json.dumps(dict(epoch=ep,loss=float(value.detach()),validation_correct=correct))+'\n')
        assert state_hash(m.encoder_state())==frozen
        m.backbone.load_state_dict(beststate);m.eval()
        with torch.no_grad():probs=m(x)[0].softmax(1).cpu()
        torch.save(beststate,folder/'student.pt');torch.save(probs,folder/'probabilities.pt')
        if bestrel is not None:torch.save(bestrel,folder/'relevance.pt')
        out=dict(meta,status='complete',best_epoch=selected,valid_correct=best,valid_total=len(vi),seconds=time.perf_counter()-started,
                 probabilities_sha256=sha(folder/'probabilities.pt'),checkpoint_sha256=sha(folder/'student.pt'),frozen_encoder=True)
        save(folder/'result.json',out);print(json.dumps(out),flush=True);return out
    except Exception:
        save(folder/'failure.json',dict(traceback=traceback.format_exc()));raise
def run_r2():
    for name in CFG['datasets']:
        raw=f.r2.raw_data(name);INPUTS[str(f.r2.DATA_FILES[name])]=sha(f.r2.DATA_FILES[name])
        h,hp,mp=f.r2.load_h(name,raw);INPUTS[str(hp)]=sha(hp)
        x=raw.x.float().cuda();classes=int(raw.y.max())+1
        for seed in CFG['seeds']:
            b,geo=fixed_data(name,seed);enc=load(f.historical_encoder(name,seed))
            m=model(x,classes,h.shape[1],seed,enc)
            with torch.no_grad():hidden=m.encode(x).detach()
            km_path=ROOT/'geometry'/name/f'seed{seed}.pt'
            if km_path.exists():km=load(km_path)
            else:
                with threadpool_limits(limits=1):km=torch.from_numpy(KMeans(len(torch.unique(geo['legacy32'])),n_init=10,random_state=seed).fit_predict(h.numpy())).long()
                km_path.parent.mkdir(parents=True,exist_ok=True);torch.save(km,km_path)
            for cond in CFG['conditions']:
                pack=load(b/f'{cond}.pt');vals,prop=targets(name,seed,cond,geo,pack,raw.edge_index.cpu(),classes,km)
                for arm in CFG['arms']:
                    save(ROOT/'status.json',dict(stage='r2',dataset=name,seed=seed,condition=cond,arm=arm))
                    fit_head(name,seed,cond,arm,x,hidden,enc,classes,geo,pack,vals.get(arm,vals['CE']),prop.cuda(),h.shape[1])
            save(ROOT/'inputs.json',INPUTS)
        del x,hidden,m;torch.cuda.empty_cache()
    save(ROOT/'r2_complete.json',dict(fits=480))
def main():
    torch.set_num_threads(CFG['threads'])
    if (ROOT/'complete.json').exists():print('Already complete');return
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<4*2**30:raise RuntimeError('GPU unavailable or busy')
    with (ROOT/'run.lock').open('x') as lock:lock.write(str(os.getpid()))
    try:
        save(ROOT/'environment.json',dict(torch=torch.__version__,python=sys.executable,gpu=torch.cuda.get_device_name(0)))
        sources=[*ROOT.glob('*.py'),ROOT/'PLAN.md',ROOT/'config.json',Path(st.__file__),Path(f.__file__),Path(f.r2.__file__)]
        manifest={str(p):sha(p) for p in sources}
        if (ROOT/'sources.json').exists():assert f.read(ROOT/'sources.json')==manifest
        else:save(ROOT/'sources.json',manifest)
        run_r2()
        from external import run_external
        run_external()
        save(ROOT/'inputs.json',INPUTS)
        from analyze import analyze
        analyze()
    except Exception:
        save(ROOT/'failure.json',dict(traceback=traceback.format_exc()));raise
    finally:(ROOT/'run.lock').unlink(missing_ok=True)
if __name__=='__main__':
    sys.modules['run']=sys.modules[__name__]
    main()
