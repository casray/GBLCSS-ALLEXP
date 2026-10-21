"""Exact equal-weight leave-one-out histograms without quadratic voter matrices."""
import numpy as np


def group_vote(groups, observed, n_classes):
    groups=np.asarray(groups)
    observed=np.asarray(observed,dtype=np.int64)
    if groups.ndim!=1 or observed.shape!=groups.shape or len(groups)==0:raise ValueError('Invalid groups')
    if n_classes<2 or np.any(observed<0) or np.any(observed>=n_classes):raise ValueError('Invalid labels')
    _,inverse=np.unique(groups,return_inverse=True)
    hist=np.zeros((int(inverse.max())+1,n_classes),dtype=np.float64)
    np.add.at(hist,(inverse,observed),1)
    local=hist[inverse]-np.eye(n_classes)[observed]
    support=(np.bincount(inverse)[inverse]-1).astype(np.int64)
    r=np.divide(local,support[:,None],out=np.full_like(local,1/n_classes),where=support[:,None]>0)
    maximum=r.max(1)
    alpha=np.clip((n_classes*maximum-1)/(n_classes-1),0,1)
    alpha[support<2]=0
    tied=np.isclose(r,maximum[:,None],atol=1e-12,rtol=0).sum(1)>1
    q=r.copy();unique=~tied
    q[unique]=.5*r[unique]+.5*np.eye(n_classes)[r[unique].argmax(1)]
    return r,q,alpha,support,support.astype(np.float64)
