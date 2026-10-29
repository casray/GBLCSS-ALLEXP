"""Label-free feature transforms and the registered fitting matrix."""
import torch
import torch.nn.functional as F

def tasks(config):
    return [dict(dataset=d,seed=s,representation=r,condition=c,head=h)
            for d in config['datasets'] for s in config['seeds']
            for r in config['representations'] for c in config['conditions'] for h in config['heads']]

def normalize_rows(h):
    if h.ndim!=2 or not torch.isfinite(h).all():raise ValueError('Invalid feature matrix')
    return F.normalize(h,dim=1)

def fit_pca(h,k):
    if h.ndim!=2 or not torch.isfinite(h).all():raise ValueError('Invalid feature matrix')
    if not 1<=k<=min(h.shape[0]-1,h.shape[1]):raise ValueError('Invalid PCA dimension')
    hd=h.double();mean=hd.mean(0);centered=hd-mean
    values,vectors=torch.linalg.eigh(centered.T@centered/len(h))
    values=values.flip(0).clamp_min(0);components=vectors.flip(1)[:,:k].contiguous()
    if float(values.sum())<=0:raise ValueError('PCA requires positive variance')
    pivots=components.abs().argmax(0)
    signs=components[pivots,torch.arange(k)].sign()
    components=components*signs
    return dict(mean=mean,components=components,eigenvalues=values,
                retained_fraction=float(values[:k].sum()/values.sum()))

def project_pca(h,state):
    return (h.double()-state['mean'])@state['components']
