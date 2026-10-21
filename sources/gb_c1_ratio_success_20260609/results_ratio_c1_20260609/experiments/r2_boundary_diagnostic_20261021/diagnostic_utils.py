"""Non-training statistics for the fixed GB versus kNN diagnostic."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def geometry(h, assignment):
    h=np.asarray(h,dtype=np.float64)
    _,inv,counts=np.unique(assignment,return_inverse=True,return_counts=True)
    centers=np.zeros((len(counts),h.shape[1]),dtype=np.float64)
    np.add.at(centers,inv,h)
    centers/=counts[:,None]
    distance=np.linalg.norm(h-centers[inv],axis=1)
    radius=np.bincount(inv,weights=distance)/counts
    return distance,radius[inv],distance/np.maximum(radius[inv],1e-12),counts[inv]


def crossing_fraction(assignment,neighbors,support):
    result=np.full(len(assignment),np.nan)
    for n in range(len(assignment)):
        if support[n]:
            result[n]=np.mean(assignment[neighbors[n,:support[n]]]!=assignment[n])
    return result


def quantile_bins(x,eligible):
    x=np.asarray(x,dtype=float)
    valid=np.asarray(eligible)&np.isfinite(x)
    if not valid.any():
        return np.full(len(x),-1),np.array([])
    edges=np.unique(np.quantile(x[valid],[0,.25,.5,.75,1]))
    bins=np.searchsorted(edges[1:-1],x,side='right')
    bins[~np.isfinite(x)]=-1
    return bins,edges


def target_metrics(target,truth,observed):
    target=np.asarray(target,dtype=np.float64)
    ids=np.arange(len(truth))
    true_p=target[ids,truth]
    other=target.copy()
    other[ids,truth]=-np.inf
    maximum=other.max(1)
    return dict(brier=np.square(target).sum(1)-2*true_p+1,true_probability=true_p,
                strict_wrong=maximum>true_p+1e-7,truth_tied=np.abs(maximum-true_p)<=1e-7,
                softened=target[ids,observed]<1-1e-7,off_observed_mass=1-target[ids,observed])


def pearson(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    x,y=x-x.mean(),y-y.mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return None if len(x)<3 or denominator<1e-12 else float(np.dot(x,y)/denominator)


def spearman(x,y):
    x,y=np.asarray(x),np.asarray(y)
    keep=np.isfinite(x)&np.isfinite(y)
    return pearson(rankdata(x[keep]),rankdata(y[keep])) if keep.sum()>=3 else None


def within_group_rank_correlation(x,y,groups):
    x,y,groups=np.asarray(x),np.asarray(y),np.asarray(groups)
    keep=np.isfinite(x)&np.isfinite(y)
    if keep.sum()<3:
        return None
    a,b=rankdata(x[keep]),rankdata(y[keep])
    _,inv,counts=np.unique(groups[keep],return_inverse=True,return_counts=True)
    a-=np.bincount(inv,weights=a)[inv]/counts[inv]
    b-=np.bincount(inv,weights=b)[inv]/counts[inv]
    return pearson(a,b)
