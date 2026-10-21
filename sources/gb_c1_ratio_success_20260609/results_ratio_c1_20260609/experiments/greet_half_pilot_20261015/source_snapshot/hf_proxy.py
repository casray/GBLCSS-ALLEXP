"""HALF-inspired proxy, not HALF reproduction; held-out label loss is diagnostic input."""
from pathlib import Path
import json
import logging
import sys
import numpy as np
from scipy import sparse
from scipy.special import logsumexp
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(WORKSPACE/'GB-LCSS_code/src'))
from gblcss.models import MLP
from gblcss.data import set_seed


def filtered_features(x,edges,block=128):
    x=np.asarray(x,dtype=np.float32); edges=np.asarray(edges,dtype=np.int64)
    edges=edges[:,edges[0]!=edges[1]]; n=len(x)
    adj=sparse.csr_matrix((np.ones(edges.shape[1],dtype=np.float32),edges),shape=(n,n))
    adj.sum_duplicates(); adj.data[:]=1
    degree=np.asarray(adj.sum(1)).ravel()
    loops=adj+sparse.eye(n,dtype=np.float32,format='csr')
    inv=np.asarray(loops.sum(1)).ravel()**-.5
    operator=sparse.diags(inv)@loops@sparse.diags(inv)
    low=np.empty_like(x)
    for start in range(0,x.shape[1],block):low[:,start:start+block]=operator@x[:,start:start+block]
    norm=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    total=np.zeros(n,dtype=np.float64)
    for start in range(0,edges.shape[1],256):
        u,v=edges[:,start:start+256]
        sim=np.maximum((norm[u]*norm[v]).sum(1),0)
        np.add.at(total,u,sim)
    h=np.divide(total,degree,out=np.full(n,.5,dtype=np.float64),where=degree>0)
    return low,x-low,h


def fit_channel(features,fit_indices,fit_labels,hold_indices,n_classes,config,seed,folder,device):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    set_seed(seed)
    x=torch.as_tensor(features,dtype=torch.float32,device=device)
    fit=torch.as_tensor(fit_indices,device=device); hold=torch.as_tensor(hold_indices,device=device)
    labels=torch.as_tensor(fit_labels,dtype=torch.long,device=device)
    # Only fold-external labels are supplied to the fitting API.
    model=MLP(x.shape[1],64,n_classes,2,.5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.0005)
    snapshots={}; history=[]
    for epoch in range(1,config['epochs']+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x)[fit],labels)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite auxiliary loss')
        loss.backward(); optimizer.step()
        history.append(dict(epoch=epoch,loss=float(loss.detach())))
        if epoch in config['saved_epochs']:
            model.eval()
            with torch.no_grad():snapshots[epoch]=model(x)[hold].cpu().numpy()
    torch.save(model.state_dict(),folder/'checkpoint.pt')
    np.savez_compressed(folder/'fold.npz',fit_indices=fit_indices,hold_indices=hold_indices,fit_labels=fit_labels)
    (folder/'training.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    for epoch,logits in snapshots.items():np.save(folder/f'hold_logits_epoch{epoch}.npy',logits)
    del model,x,fit,hold,labels,optimizer
    return snapshots


def cross_fit_scores(features,observed,train_indices,n_classes,config,seed,folder,device):
    folder=Path(folder); observed=np.asarray(observed); train_indices=np.asarray(train_indices)
    rng=np.random.RandomState(seed+3298)
    folds=np.array_split(rng.permutation(len(train_indices)),config['folds'])
    if any(len(fold)==0 for fold in folds):raise ValueError('Too few training nodes for folds')
    losses={channel:{epoch:np.full(len(observed),np.nan) for epoch in config['saved_epochs']} for channel in features}
    for fold,held in enumerate(folds):
        fit=np.setdiff1d(np.arange(len(train_indices)),held)
        if np.intersect1d(train_indices[fit],train_indices[held]).size:raise ValueError('Crossfit overlap')
        for channel,x in features.items():
            logging.info('HF seed%d fold%d %s 30epoch',seed,fold,channel)
            logits=fit_channel(x,train_indices[fit],observed[fit],train_indices[held],n_classes,
                               config,seed*100+fold,folder/f'fold{fold}'/channel,device)
            for epoch,prediction in logits.items():
                losses[channel][epoch][held]=logsumexp(prediction,axis=1)-prediction[np.arange(len(held)),observed[held]]
    if any(not np.isfinite(score).all() for values in losses.values() for score in values.values()):
        raise ValueError('Missing crossfit predictions')
    return losses,folds
