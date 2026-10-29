"""Pure, fold-local mechanisms; no dataset loading or global label access."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax


def split_folds(train,seed):
    train=np.asarray(train)
    if len(train)<4 or len(np.unique(train))!=len(train):raise ValueError('Invalid train IDs')
    order=np.random.default_rng(seed+29001).permutation(len(train))
    return np.sort(order[:len(order)//2]),np.sort(order[len(order)//2:])


def vote_details(labels,assignment,classes,beta=.5):
    labels=np.asarray(labels,dtype=np.int64);assignment=np.asarray(assignment,dtype=np.int64)
    if labels.shape!=assignment.shape or labels.ndim!=1 or not len(labels):raise ValueError('Mismatched labels and assignments')
    if labels.min()<0 or labels.max()>=classes or assignment.min()<0:raise ValueError('Invalid label/group')
    own=np.eye(classes,dtype=np.float64)[labels]
    counts=np.zeros((int(assignment.max())+1,classes),np.float64)
    np.add.at(counts,assignment,own);counts=counts[assignment]-own
    support=counts.sum(1);vote=counts/np.maximum(support[:,None],1)
    maximum=counts.max(1);winner=counts.argmax(1);tie=(counts==maximum[:,None]).sum(1)>1
    alpha=np.clip((maximum/np.maximum(support,1)-1/classes)/(1-1/classes),0,1)
    alpha[support<2]=0
    strength=np.where(tie,0.,beta)
    q=(1-strength[:,None])*vote+strength[:,None]*np.eye(classes)[winner]
    target=(1-alpha[:,None])*own+alpha[:,None]*q
    return dict(counts=counts,support=support,vote=vote,q=q,alpha=alpha,target=target)


def fold_package(ids,local_labels,full_assignment,classes):
    ids=np.asarray(ids,dtype=np.int64);local_labels=np.asarray(local_labels,dtype=np.int64)
    if len(ids)!=len(local_labels) or len(np.unique(ids))!=len(ids):raise ValueError('Fold-local labels required')
    group=np.asarray(full_assignment,dtype=np.int64)[ids]
    details=vote_details(local_labels,group,classes)
    return dict(ids=ids,observed=local_labels.copy(),group=group,target=details['target'].astype(np.float32))


def positive_weights(target,group):
    t=np.asarray(target,dtype=np.float64);g=np.asarray(group)
    w=np.maximum(t@t.T-1/t.shape[1],0)*(g[:,None]==g[None,:]);np.fill_diagonal(w,0)
    return (w/np.maximum(w.sum(1,keepdims=True),1e-30)).astype(np.float32)


def soft_contrastive(features,weights,queries,tau=.5):
    if tau<=0:raise ValueError('Positive temperature required')
    z=F.normalize(features,dim=1);w=weights[queries]
    active=w.sum(1)>0
    if not active.any():return features.sum()*0
    queries=queries[active];w=w[active]
    logits=z[queries]@z.T/tau
    mask=F.one_hot(queries,num_classes=len(z)).bool()
    logden=torch.logsumexp(logits.masked_fill(mask,-torch.inf),1)
    return (logden-(w*logits).sum(1)).mean()


def normalized(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def region_distribution(features,assignment,queries,tau=.5):
    x=normalized(features);a=np.asarray(assignment,dtype=np.int64);ids=np.asarray(queries,dtype=np.int64)
    k=int(a.max())+1;counts=np.bincount(a,minlength=k);sums=np.zeros((k,x.shape[1]))
    np.add.at(sums,a,x);centers=normalized(sums)
    logits=x[ids]@centers.T
    own=normalized(sums[a[ids]]-x[ids]);logits[np.arange(len(ids)),a[ids]]=(x[ids]*own).sum(1)
    logits[:,counts==0]=-1e6
    singleton=counts[a[ids]]<=1;logits[np.flatnonzero(singleton),a[ids[singleton]]]=-1e6
    return softmax(logits/tau,axis=1)


def relation_features(g,h,assignment,queries,tau=.5):
    q=region_distribution(g,assignment,queries,tau);p=region_distribution(h,assignment,queries,tau)
    kl=(q*(np.log(q.clip(1e-30))-np.log(p.clip(1e-30)))).sum(1)
    entropy=-(q*np.log(q.clip(1e-30))).sum(1)/max(np.log(q.shape[1]),1e-12)
    own=q[np.arange(len(queries)),np.asarray(assignment)[queries]]
    return np.c_[kl,entropy,own]


def support_changes(old,new):
    old=np.asarray(old);new=np.asarray(new);_,o=np.unique(old,return_inverse=True);_,n=np.unique(new,return_inverse=True)
    oc=np.bincount(o);nc=np.bincount(n);joint=np.bincount(o*(n.max()+1)+n,minlength=(o.max()+1)*(n.max()+1)).reshape(o.max()+1,n.max()+1)
    intersection=joint[o,n]-1;os=oc[o]-1;ns=nc[n]-1;union=os+ns-intersection
    jac=np.divide(intersection,union,out=np.ones(len(o),dtype=float),where=union>0)
    return dict(old_support=os,new_support=ns,intersection=intersection,jaccard=jac,changed=(intersection!=os)|(intersection!=ns))


def quality(target,truth):
    t=np.asarray(target,dtype=np.float64);y=np.asarray(truth,dtype=np.int64)
    return (t*t).sum(1)-2*t[np.arange(len(y)),y]+1


def true_margins(truth,assignment,classes):
    r=vote_details(truth,assignment,classes)['vote'];own=r[np.arange(len(r)),truth];other=r.copy();other[np.arange(len(r)),truth]=-np.inf
    return own-other.max(1),own
