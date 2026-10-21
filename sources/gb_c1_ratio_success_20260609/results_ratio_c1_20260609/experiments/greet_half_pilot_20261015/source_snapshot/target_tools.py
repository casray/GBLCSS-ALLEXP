"""Independent leave-one-out reference and node-specific gate construction."""
import sys
from pathlib import Path
import numpy as np
from methods import rank_gate

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'GB-LCSS_code/src'))


def mix(q,alpha,observed):
    return (1-alpha[:,None])*np.eye(q.shape[1])[observed]+alpha[:,None]*q


def ball_bundle(groups,observed,n_classes,beta=.5,min_votes=2):
    groups=np.asarray(groups); observed=np.asarray(observed)
    n=len(groups); r=np.full((n,n_classes),1/n_classes); support=np.zeros(n,dtype=np.int64)
    for group in np.unique(groups):
        members=np.flatnonzero(groups==group); count=len(members)-1; support[members]=count
        if count:
            hist=np.bincount(observed[members],minlength=n_classes)
            r[members]=(hist[None,:]-np.eye(n_classes)[observed[members]])/count
    maximum=r.max(1); alpha=np.clip((n_classes*maximum-1)/(n_classes-1),0,1)
    alpha[support<min_votes]=0
    tied=(np.abs(r-maximum[:,None])<=1e-12).sum(1)>1
    q=r.copy(); unique=~tied
    q[unique]=(1-beta)*r[unique]+beta*np.eye(n_classes)[r[unique].argmax(1)]
    return dict(r=r,q=q,alpha=alpha,target=mix(q,alpha,observed),support=support)


def gated_bundle(base,observed,scores):
    alpha=base['alpha']*rank_gate(scores)
    return dict(q=base['q'],alpha=alpha,target=mix(base['q'],alpha,observed),support=base['support'])
