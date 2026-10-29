"""Frozen ProGCL-weight full-branch math, float64 mixture and disk mmap B."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import beta
import torch
import torch.nn.functional as F

EPS=1e-12
class BetaMixture:
    def __init__(self):
        self.alphas=np.array([1.,2.]);self.betas=np.array([2.,1.]);self.weight=np.array([.95,.05])
    def likelihoods(self,x):
        return np.stack([self.weight[k]*beta.pdf(x,self.alphas[k],self.betas[k]) for k in range(2)])
    def posterior(self,x):
        r=self.likelihoods(np.clip(x,EPS,1-EPS));return r[0]/(r.sum(0)+EPS)
    def fit(self,x):
        x=np.clip(np.asarray(x,dtype=np.float64),EPS,1-EPS)
        for _ in range(10):
            r=np.maximum(self.likelihoods(x),EPS);r/=r.sum(0)
            for k in range(2):
                mean=(r[k]*x).sum()/r[k].sum();var=(r[k]*(x-mean)**2).sum()/r[k].sum()
                if not np.isfinite(var) or var<=0:raise FloatingPointError('Degenerate Beta fit variance')
                self.alphas[k]=mean*(mean*(1-mean)/var-1)
                self.betas[k]=self.alphas[k]*(1-mean)/mean
            if self.betas[1]<1:self.betas[1]=1.01
            self.weight=r.sum(1);self.weight/=self.weight.sum()
            if not all(np.isfinite(v).all() and (v>0).all() for v in [self.alphas,self.betas,self.weight]):raise FloatingPointError('Invalid Beta parameters')
        return self

def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def _tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def build_weights(z1,z2,path,chunk=512):
    """Fit once using current NumPy RNG (randint replacement, 100 columns).

    Writes .npy float64 Nx(N-1) plus .json metadata, returns read-only mmap.
    Input Z is projected representation. Caller records epoch/checkpoint RNG.
    Refuses overwrites, invalid weights and nonfinite/degenerate fits.
    """
    path=Path(path);meta_path=path.with_suffix(path.suffix+'.json')
    if path.exists() or meta_path.exists():raise FileExistsError(path)
    if z1.shape!=z2.shape or len(z1)<2 or chunk<1:raise ValueError('Invalid representation shape/chunk')
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        a=F.normalize(z1.detach(),dim=1);b=F.normalize(z2.detach(),dim=1);n=len(a)
        indices=np.random.randint(0,n,100)
        fit=((a@b[torch.as_tensor(indices,device=b.device)].T+1)/2).cpu().double().numpy().reshape(-1)
        mixture=BetaMixture().fit(fit)
        arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(n,n-1))
        blocks=[]
        try:
            for start in range(0,n,chunk):
                end=min(start+chunk,n)
                sim=((a[start:end]@b.T+1)/2).cpu().double().numpy()
                mask=np.ones(sim.shape,dtype=bool);mask[np.arange(end-start),np.arange(start,end)]=False
                cosine=sim[mask].reshape(end-start,n-1)
                value=mixture.posterior(cosine)*cosine
                if not np.isfinite(value).all() or (value<0).any() or (value.mean(1)<=0).any():raise FloatingPointError('Invalid frozen B rows')
                arr[start:end]=value
                blocks.append({'start':start,'end':end,'sha256':hashlib.sha256(value.tobytes()).hexdigest()})
            arr.flush()
        finally:del arr
    metadata={'shape':[n,n-1],'dtype':'float64','fit_columns':indices.tolist(),'alphas':mixture.alphas.tolist(),'betas':mixture.betas.tolist(),'mixture_weights':mixture.weight.tolist(),'iterations':10,'z1_sha256':_tensor_hash(z1),'z2_sha256':_tensor_hash(z2),'blocks':blocks,'sha256':_hash(path),'adaptation':'float64 fitting; posterior input clipped to [1e-12,1-1e-12]'}
    meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return load_weights(path)

def load_weights(path):
    """Verify completed weight file before mmap restoration; never refit."""
    path=Path(path);meta=json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf-8'))
    if _hash(path)!=meta['sha256']:raise ValueError('Frozen B hash mismatch')
    arr=np.load(path,mmap_mode='r',allow_pickle=False)
    if arr.dtype!=np.float64 or list(arr.shape)!=meta['shape']:raise ValueError('Frozen B schema mismatch')
    return arr

def weights_for_epoch(epoch,z1,z2,path,epoch_start=400,chunk=512):
    """399: None; 400: fit once or verified resume; 401+: verified load.

    Runner should retain returned mmap in memory to avoid repeated hash I/O.
    """
    if epoch<epoch_start:return None
    if epoch==epoch_start and not Path(path).exists():return build_weights(z1,z2,path,chunk)
    return load_weights(path)
