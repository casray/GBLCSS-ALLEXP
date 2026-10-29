"""Label-free leave-one-out regional context and exact contrastive gradients."""
import numpy as np
import torch
import torch.nn.functional as F

def context_scores(edge, assignment, seed):
    edge=np.asarray(edge,dtype=np.int64);ball=np.asarray(assignment,dtype=np.int64)
    n=len(ball)
    if edge.ndim!=2 or edge.shape[0]!=2 or ball.ndim!=1 or n==0 or (ball<0).any():
        raise ValueError('Invalid graph or partition')
    if edge.size and ((edge<0).any() or (edge>=n).any() or (edge[0]==edge[1]).any()):
        raise ValueError('Context graph must be loop-free with valid endpoints')
    src,dst=edge;k=int(ball.max())+1;degree=np.bincount(dst,minlength=n)
    p=np.zeros((n,k),dtype=np.float64)
    np.add.at(p,(dst,ball[src]),1.)
    p/=np.maximum(degree[:,None],1)
    fold=np.full(n,-1,dtype=np.int64);rng=np.random.default_rng(seed+73001)
    sums=np.zeros((k,2,k));counts=np.zeros((k,2),dtype=np.int64)
    for b in range(k):
        ids=np.flatnonzero((ball==b)&(degree>0));ids=rng.permutation(ids)
        for f in (0,1):
            members=ids[f::2];fold[members]=f;counts[b,f]=len(members);sums[b,f]=p[members].sum(0)
    own=np.stack([fold==0,fold==1],1)
    remaining=counts[ball]-own
    active=(remaining.min(1)>=2)&(degree>=2)
    total=p.sum(0);nonisolated=int((degree>0).sum())
    result=np.zeros(len(src),dtype=np.float64)
    for start in range(0,len(src),65536):
        end=min(start+65536,len(src));i=dst[start:end];target=ball[src[start:end]]
        bg=(total[target]-p[i,target])/max(nonisolated-1,1)
        score=np.ones(len(i))
        for f in (0,1):
            th=(sums[ball[i],f,target]-own[i,f]*p[i,target])/np.maximum(remaining[i,f],1)
            th=np.maximum(th,0);bg=np.maximum(bg,0)
            score=np.minimum(score,th/(th+bg+1e-12))
        result[start:end]=np.where(active[i],score,0)
    return dict(scores=result,active=active,fold=fold,degree=degree,ball=ball)

def center_scores(edge,scores,n,active):
    src,dst=np.asarray(edge);scores=np.asarray(scores,dtype=np.float64)
    if len(scores)!=len(src) or not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('Scores must be finite in [0,1]')
    deg=np.bincount(dst,minlength=n);mean=np.bincount(dst,weights=scores,minlength=n)/np.maximum(deg,1)
    delta=(scores-mean[dst])*np.asarray(active)[dst]
    if np.max(np.abs(np.bincount(dst,weights=delta,minlength=n)),initial=0)>1e-9:
        raise RuntimeError('Centered row mass mismatch')
    return delta

def cosine_scores(h,edge):
    h=np.asarray(h,dtype=np.float64);h=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    src,dst=np.asarray(edge);out=np.empty(len(src))
    for start in range(0,len(src),8192):
        stop=min(start+8192,len(src));out[start:stop]=np.einsum('ij,ij->i',h[src[start:stop]],h[dst[start:stop]])
    return np.clip((out+1)/2,0,1)

def shuffle_delta(edge,delta,seed):
    dst=np.asarray(edge)[1];out=np.asarray(delta).copy();rng=np.random.default_rng(seed+91009)
    order=np.argsort(dst,kind='stable');breaks=np.r_[0,np.flatnonzero(np.diff(dst[order]))+1,len(order)]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        ids=order[lo:hi];out[ids]=rng.permutation(out[ids])
    return out

def dense_loss(h,t,edge,delta,eta,tau=.5,lam=.001,lambda_loss=1.):
    z,q=F.normalize(h,dim=1),F.normalize(t,dim=1);src,dst=edge;n=len(z)
    degree=torch.bincount(dst,minlength=n).to(z.dtype)
    a=torch.exp(z@z.T/tau).sum(1);b=lam*torch.exp(z@q.T/tau).sum(1)
    positive=(z[src]*q[dst]).sum(1)/tau
    return ((-positive*(1+eta*delta)+lambda_loss*torch.log(a[dst]+b[src]))/degree[dst]).sum()/n

def backward_loss(h,t,edge,delta,eta,chunk=512,tau=.5,lam=.001,lambda_loss=1.):
    z=F.normalize(h.detach(),dim=1).requires_grad_(True)
    q=F.normalize(t.detach(),dim=1).requires_grad_(True);n=len(z)
    with torch.no_grad():
        a=torch.cat([torch.exp(z[i:i+chunk]@z.T/tau).sum(1) for i in range(0,n,chunk)])
        b=torch.cat([lam*torch.exp(z[i:i+chunk]@q.T/tau).sum(1) for i in range(0,n,chunk)])
    a.requires_grad_(True);b.requires_grad_(True);src,dst=edge
    degree=torch.bincount(dst,minlength=n).to(z.dtype);positive=(z[src]*q[dst]).sum(1)/tau
    loss=((-positive*(1+eta*delta)+lambda_loss*torch.log(a[dst]+b[src]))/degree[dst]).sum()/n
    loss.backward()
    for i in range(0,n,chunk):
        ai=torch.exp(z[i:i+chunk]@z.T/tau).sum(1);bi=lam*torch.exp(z[i:i+chunk]@q.T/tau).sum(1)
        torch.autograd.backward((ai,bi),(a.grad[i:i+chunk],b.grad[i:i+chunk]))
    torch.autograd.backward((F.normalize(h,dim=1),F.normalize(t,dim=1)),(z.grad,q.grad))
    return float(loss.detach())
