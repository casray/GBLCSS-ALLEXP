"""Frozen H linear probes/native heads, selected using validation only."""
import time
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from artifacts import ROOT, PROTOCOL, save, save_tensor, sha, reusable, complete
from data_bridge import load_supervised, seed_all, state_hash
from downstream import build_support32_targets, _support_assignment

def fit(h, h_path, dataset, seed, arm, condition='clean', head='CE', smoke=False):
    base = ROOT/('engineering' if smoke else 'formal')/'linear'/dataset/f'seed{seed}'/arm
    folder = base/condition/head
    epochs = 2 if smoke else PROTOCOL['probe']['epochs']
    expected = dict(dataset=dataset,seed=seed,arm=arm,condition=condition,head=head,
                    h_sha256=sha(h_path),epochs=epochs,lr=.01,weight_decay=.0005)
    if reusable(folder,expected):
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f'Incomplete linear fit retained: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    data=load_supervised(dataset,seed,condition)
    target=F.one_hot(data['observed'],data['classes']).float()
    if head=='SUPPORT32':
        assignment=_support_assignment(base,h.cpu(),data['train'],dataset,seed,32)
        target=build_support32_targets(data['observed'],assignment[data['train']],classes=data['classes'])
    seed_all(seed)
    model=torch.nn.Linear(h.shape[1],data['classes']).cuda()
    initial=state_hash(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0005)
    hd=h.cuda(); td=target.cuda(); train=data['train'].cuda(); valid=data['valid'].cuda()
    vy=data['valid_y'].cuda(); best=-1; best_epoch=0; state=None
    save(folder/'config.json',expected)
    start=time.perf_counter()
    with (folder/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,epochs+1):
            optimizer.zero_grad(set_to_none=True)
            loss=-(td*F.log_softmax(model(hd[train]),dim=1)).sum(1).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Linear loss nonfinite')
            loss.backward(); optimizer.step()
            with torch.no_grad(): correct=int((model(hd[valid]).argmax(1)==vy).sum())
            if correct>best:
                best,best_epoch=correct,epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            log.write(json.dumps(dict(epoch=epoch,training_loss=float(loss.detach()),validation_correct=correct))+'\n')
            log.flush()
    model.load_state_dict(state)
    with torch.no_grad(): probabilities=model(hd).softmax(1).cpu()
    replay=int((probabilities[data['valid']].argmax(1)==data['valid_y']).sum())
    if replay!=best: raise RuntimeError('Linear checkpoint replay mismatch')
    save_tensor(folder/'student.pt',state);save_tensor(folder/'probabilities.pt',probabilities)
    save_tensor(folder/'target.pt',target)
    save(folder/'result.json',dict(expected,status='complete',best_epoch=best_epoch,valid_correct=best,
         initial_hash=initial,seconds=time.perf_counter()-start,selection='first strict maximum validation count'))
    complete(folder,expected,['student.pt','probabilities.pt','target.pt','config.json','epochs.jsonl','result.json'])
    return folder

def run(h_path,dataset,seed,arm,smoke=False):
    h=F.normalize(torch.load(h_path,map_location='cpu',weights_only=True)['H'].float(),dim=1)
    probe=fit(h,h_path,dataset,seed,arm,smoke=smoke)
    if arm=='graphecl':
        for condition in PROTOCOL['conditions']:
            for head in PROTOCOL['graphecl']['native_heads']:
                if condition=='clean' and head=='CE':
                    save(probe/'native_alias.json',dict(alias='primary clean H probe',h_sha256=sha(h_path),
                         config_sha256=sha(probe/'config.json')))
                else: fit(h,h_path,dataset,seed,arm,condition,head,smoke)

