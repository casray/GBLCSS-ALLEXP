"""Portable numerical building blocks; pretraining accepts attributes/edges only."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import torch
from torch import nn
from torch.nn import functional as F


def weighted_filters(edges, weights, n, highpass_alpha=.1):
    u,v=edges
    if bool((u==v).any()): raise ValueError('Supply loop-free edges')
    diagonal=torch.arange(n,device=edges.device)
    indices=torch.cat((edges,torch.stack((diagonal,diagonal))),1)
    low_degree=weights.new_ones(n).index_add(0,u,weights)
    high_degree=weights.new_ones(n).index_add(0,u,1-weights)
    low_inv=low_degree.rsqrt(); high_inv=high_degree.rsqrt()
    low=torch.cat((weights*low_inv[u]*low_inv[v],low_inv.square()))
    high=torch.cat((-highpass_alpha*(1-weights)*high_inv[u]*high_inv[v],weights.new_ones(n)))
    return indices,low,high


def propagate(x, indices, weights):
    result=x.new_zeros(x.shape)
    # Matrix convention A[u,v] multiplies x[v], matching upstream dense branch.
    return result.index_add(0,indices[0],x[indices[1]]*weights[:,None])


def return_features(edges, n, *, hops=16, walks=1024, seed=0, block=128):
    if min(hops,walks,block)<1: raise ValueError('Positive walk dimensions required')
    edges=np.asarray(edges,dtype=np.int64)
    adj=sparse.csr_matrix((np.ones(edges.shape[1]),edges),shape=(n,n))
    adj.sum_duplicates(); adj.sort_indices()
    degrees=np.diff(adj.indptr)
    result=np.zeros((n,hops),dtype=np.float64)
    if adj.nnz==0:return result.astype(np.float32)
    rng=np.random.RandomState(seed); root=np.arange(n)[:,None]
    for start in range(0,walks,block):
        width=min(block,walks-start)
        positions=np.broadcast_to(root,(n,width)).copy()
        for t in range(hops):
            current_degree=degrees[positions]
            valid=current_degree>0
            offsets=adj.indptr[positions]+(rng.random_sample(positions.shape)*current_degree).astype(np.int64)
            selected=adj.indices[np.minimum(offsets,adj.nnz-1)]
            positions=np.where(valid,selected,positions)
            result[:,t]+=((positions==root)&(degrees[:,None]>0)).sum(1)
    return (result/walks).astype(np.float32)


def rank_gate(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite, nonempty one-dimensional scores required')
    return (rankdata(scores,method='average')-.5)/len(scores)


def match_budget(alpha, gate, desired):
    alpha=np.asarray(alpha,dtype=np.float64); gate=np.asarray(gate,dtype=np.float64)
    if alpha.shape!=gate.shape or np.any(alpha<0) or np.any(gate<=0):
        raise ValueError('Nonnegative alpha and positive gates required')
    if desired<0 or desired>alpha.sum()+1e-10: raise ValueError('Unreachable budget')
    if desired==0:return np.zeros_like(alpha)
    lo,hi=0.,1./gate.min()
    for _ in range(80):
        mid=(lo+hi)/2
        if (alpha*np.minimum(mid*gate,1.)).sum()<desired:lo=mid
        else:hi=mid
    return alpha*np.minimum((lo+hi)/2*gate,1.)


def contrastive_loss(z1,z2,ids,knn,tau=.2):
    pos=(knn[ids,:,None]==ids[None,None,:]).any(1)|torch.eye(len(ids),dtype=torch.bool,device=ids.device)
    if bool(pos.all(1).any()): raise ValueError('Contrastive batch contains a row without negatives')
    sim=F.normalize(z1)@F.normalize(z2).T/tau
    log_denominator=torch.logsumexp(sim.masked_fill(pos,-torch.inf),dim=1,keepdim=True)
    return -((sim-log_denominator)*pos).sum(1).div(pos.sum(1)).mean()


class GreetPortable(nn.Module):
    def __init__(self,input_dim,half_dim=128,projection_dim=128,layers=2,se_dim=16,highpass_alpha=.1):
        super().__init__()
        self.low_encoder=nn.Linear(input_dim,half_dim)
        self.high_encoder=nn.Linear(input_dim,half_dim)
        self.low_projector=nn.Linear(half_dim,projection_dim)
        self.high_projector=nn.Linear(half_dim,projection_dim)
        self.node_discriminator=nn.Linear(input_dim+se_dim,128)
        self.edge_discriminator=nn.Linear(256,1)
        self.layers=layers; self.highpass_alpha=highpass_alpha

    def sample_weights(self,x,se,edges):
        h=F.relu(self.node_discriminator(torch.cat((x,se),1)))
        u,v=edges
        raw=(self.edge_discriminator(torch.cat((h[u],h[v]),1))+
             self.edge_discriminator(torch.cat((h[v],h[u]),1))).flatten()/2
        eps=(.0001-(1-.0001))*torch.rand_like(raw)+(1-.0001)
        return torch.sigmoid(raw+torch.log(eps)-torch.log1p(-eps))

    def channels(self,x,indices,low,high):
        a=F.relu(self.low_encoder(x)); b=F.relu(self.high_encoder(x))
        for _ in range(self.layers):
            a=propagate(a,indices,low); b=propagate(b,indices,high)
        return a,b

    def encode(self,x,indices,low,high):
        return torch.cat(self.channels(x,indices,low,high),1)

    def project(self,x,indices,low,high):
        a,b=self.channels(x,indices,low,high)
        return torch.cat((self.low_projector(a),self.high_projector(b)),1)

    def project_channel(self,x,indices,weights,channel):
        encoder,projector=(self.low_encoder,self.low_projector) if channel=='low' else (self.high_encoder,self.high_projector)
        h=F.relu(encoder(x))
        for _ in range(self.layers):h=propagate(h,indices,weights)
        return projector(h)

    @torch.no_grad()
    def export(self,x,se,edges,seed):
        devices=[x.device.index or 0] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            weights=self.sample_weights(x,se,edges)
            indices,low,high=weighted_filters(edges,weights,len(x),self.highpass_alpha)
            return self.encode(x,indices,low,high),weights

    def encoder_parameters(self):
        return list(self.low_encoder.parameters())+list(self.high_encoder.parameters())+list(self.low_projector.parameters())+list(self.high_projector.parameters())

    def discriminator_parameters(self):
        return list(self.node_discriminator.parameters())+list(self.edge_discriminator.parameters())
