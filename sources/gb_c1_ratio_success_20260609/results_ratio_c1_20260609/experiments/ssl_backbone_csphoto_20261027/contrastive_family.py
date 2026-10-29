"""Pinned GRACE/GCA kernels; full negatives with O(chunk*N) activation memory."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree,to_undirected

class Encoder(nn.Module):
    def __init__(self,in_dim,hidden,activation):
        super().__init__()
        self.conv=nn.ModuleList([GCNConv(in_dim,2*hidden),GCNConv(2*hidden,hidden)])
        self.activation=activation
    def forward(self,x,edge):
        for conv in self.conv:x=self.activation(conv(x,edge))
        return x

class Model(nn.Module):
    """forward gives preprojection H; project(H) gives contrastive Z."""
    def __init__(self,in_dim,cfg):
        super().__init__()
        if cfg.get('layers',2)!=2:raise ValueError('Frozen protocol requires two layers')
        if cfg['activation'] not in ('relu','prelu'):raise ValueError('Unknown activation')
        activation=F.relu if cfg['activation']=='relu' else nn.PReLU()
        h=cfg['hidden']; self.encoder=Encoder(in_dim,h,activation)
        self.fc1=nn.Linear(h,cfg['projection']);self.fc2=nn.Linear(cfg['projection'],h)
    def forward(self,x,edge):return self.encoder(x,edge)
    def project(self,h):return self.fc2(F.elu(self.fc1(h)))


def backward_loss(z1,z2,tau,chunk=512,B=None):
    """Backprop exact symmetric loss, returning float. Caller zeroes optimizer.

    B is detached N x (N-1), ordered by removing each row's diagonal.
    Each query block is freed immediately after backward. Leaf embedding
    gradients include both query and key roles, then propagate once to encoder.
    """
    if z1.shape!=z2.shape or len(z1)<2 or chunk<1 or tau<=0:raise ValueError('Invalid loss shape/config')
    n=len(z1)
    if B is not None and tuple(B.shape)!=(n,n-1):raise ValueError('Invalid B shape')
    a=F.normalize(z1,dim=1);b=F.normalize(z2,dim=1)
    u=a.detach().requires_grad_();v=b.detach().requires_grad_()
    total=0.
    for q,k in [(u,v),(v,u)]:
        for start in range(0,n,chunk):
            end=min(start+chunk,n); rows=torch.arange(end-start,device=q.device); cols=torch.arange(start,end,device=q.device)
            same=(q[start:end]@q.T/tau).exp();cross=(q[start:end]@k.T/tau).exp();pos=cross[rows,cols]
            if B is None:den=same.sum(1)+cross.sum(1)-same[rows,cols]
            else:
                bw=B[start:end]
                bw=bw.detach().to(device=q.device,dtype=torch.float64) if isinstance(bw,torch.Tensor) else torch.as_tensor(np.array(bw,copy=True),device=q.device,dtype=torch.float64)
                if not torch.isfinite(bw).all() or (bw<0).any() or (bw.mean(1)<=0).any():raise FloatingPointError('Invalid frozen ProGCL weights')
                mask=torch.ones_like(same,dtype=torch.bool);mask[rows,cols]=False
                den=pos+((same[mask].view(end-start,n-1)+cross[mask].view(end-start,n-1))*bw).sum(1)/bw.mean(1)
            loss=-(pos/den).log().sum()/(2*n)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite contrastive loss')
            total+=loss.item();loss.backward()
    torch.autograd.backward((a,b),(u.grad,v.grad))
    return total


def _importance(s):
    if not s.numel():return s
    den=s.max()-s.mean()
    return torch.ones_like(s) if den==0 else (s.max()-s)/den

def prepare_weights(x,edge):
    """Official degree importance; zero columns kept, equal scores uniform."""
    deg=degree(to_undirected(edge,num_nodes=len(x))[1],num_nodes=len(x))
    ew=_importance(deg[edge[1]].clamp_min(1).log())
    active=x.bool().any(0); score=x.bool().float().T@deg
    fw=torch.zeros_like(score)
    # An active feature supported only on isolated nodes receives finite minimal importance input.
    fw[active]=_importance(score[active].clamp_min(torch.finfo(score.dtype).tiny).log())
    return {'edge':ew,'feature':fw,'active':active}

def _prob(w,p,active=None):
    if active is None:active=torch.ones_like(w,dtype=torch.bool)
    out=torch.zeros_like(w)
    if active.any():
        v=w[active];mean=v.mean()
        out[active]=p if mean==0 else (v/mean*p).clamp(max=.7)
    return out.clamp(max=.7)

def _edge(edge,p,w):
    if not 0<=p<=1:raise ValueError('Invalid edge probability')
    if p==0:return edge
    keep=torch.rand(edge.shape[1],device=edge.device)>=p if w is None else torch.bernoulli(1-_prob(w,p)).bool()
    return edge[:,keep]

def _feature(x,p,weights):
    if not 0<=p<=1:raise ValueError('Invalid feature probability')
    if p==0:return x
    mask=torch.empty(x.shape[1],device=x.device).uniform_(0,1)<p if weights is None else torch.bernoulli(_prob(weights['feature'],p,weights['active'])).bool()
    out=x.clone();out[:,mask]=0;return out

def augment(x,edge,p_edge,p_feat,weights=None):
    """One view (edge then feature); use augment_views for official RNG order."""
    e=_edge(edge,p_edge,None if weights is None else weights['edge'])
    return _feature(x,p_feat,weights),e

def augment_views(x,edge,edge_probs,feature_probs,weights=None):
    """Official two-edge, two-feature order, including discarded uniform masks.

    Protocol p=0 short-circuits consume no RNG; both weighted methods share this.
    """
    edges=[_edge(edge,p,None if weights is None else weights['edge']) for p in edge_probs]
    features=[_feature(x,p,None) for p in feature_probs]
    if weights is not None:features=[_feature(x,p,weights) for p in feature_probs]
    return (features[0],edges[0]),(features[1],edges[1])
