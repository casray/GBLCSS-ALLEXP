"""Label-free features and validation-only routing choices."""
import numpy as np


def choose_weight(y, a, c):
    return float(max(np.linspace(0,1,21),key=lambda w:(int(np.sum((w*a+(1-w)*c).argmax(1)==y)),w)))


def choose_threshold(y, a, c, score):
    def quality(t):
        switched=(score>=t)&(a!=c)
        return int(np.sum(np.where(switched,c,a)==y)),-int(switched.sum()),t
    return float(max([*np.linspace(0,1,21),float('inf')],key=quality))


def entropy(p):
    p=np.asarray(p,dtype=np.float64)
    return -(p*np.log(np.clip(p,1e-15,1))).sum(1)/np.log(p.shape[1])


def js(a,c):
    a,c=np.asarray(a,dtype=np.float64),np.asarray(c,dtype=np.float64)
    m=(a+c)/2
    return .5*(a*(np.log(np.clip(a,1e-15,1))-np.log(np.clip(m,1e-15,1)))).sum(1)+.5*(c*(np.log(np.clip(c,1e-15,1))-np.log(np.clip(m,1e-15,1)))).sum(1)


def student_features(a,c):
    aa=np.sort(a,axis=1);cc=np.sort(c,axis=1)
    return np.column_stack([aa[:,-1],entropy(a),aa[:,-1]-aa[:,-2],
        cc[:,-1],entropy(c),cc[:,-1]-cc[:,-2],js(a,c),(a.argmax(1)==c.argmax(1)).astype(float)])


def region_features(assignment,teacher,c,train,z):
    _,group=np.unique(assignment,return_inverse=True)
    n=len(group);counts=np.bincount(group)
    agree=(teacher==c).astype(float)
    total=np.bincount(group,weights=agree)
    agreement=np.divide(total[group]-agree,counts[group]-1,out=np.zeros(n),where=counts[group]>1)
    mask=np.zeros(n,dtype=bool);mask[train]=True
    mismatch=np.zeros(n);mismatch[train]=(teacher[train]!=z)
    support=np.bincount(group[train],minlength=len(counts))[group]-mask
    wrong=np.bincount(group,weights=mismatch)[group]-mismatch
    conflict=np.divide(wrong,support,out=np.zeros(n),where=support>0)
    return np.column_stack([agreement,conflict,np.log1p(support),(support==0).astype(float)])
