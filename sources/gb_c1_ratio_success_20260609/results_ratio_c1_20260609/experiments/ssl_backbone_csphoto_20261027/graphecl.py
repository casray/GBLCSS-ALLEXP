"""GraphECL full-neighborhood objective; sparse DGL norm=both equivalent."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def with_loops(edge, n):
    edge = edge[:, edge[0] != edge[1]]
    ids = torch.arange(n, device=edge.device)
    return torch.cat((edge, torch.stack((ids, ids))), 1)

def adjacency(edge, n, dtype):
    src, dst = edge
    outdeg = torch.bincount(src, minlength=n).to(dtype).clamp_min(1)
    indeg = torch.bincount(dst, minlength=n).to(dtype).clamp_min(1)
    val = outdeg[src].rsqrt() * indeg[dst].rsqrt()
    return torch.sparse_coo_tensor(torch.stack((dst,src)), val, (n,n)).coalesce()

class Conv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(inp,out))
        self.bias = nn.Parameter(torch.zeros(out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, x @ self.weight) + self.bias

class Model(nn.Module):
    def __init__(self, inp, cfg):
        super().__init__()
        h, layers = cfg['hidden'], cfg['layers']
        self.convs = nn.ModuleList([Conv(inp if i==0 else h,h) for i in range(layers)])
        self.encoder_target = nn.Sequential(nn.Linear(inp,h),nn.BatchNorm1d(h),nn.ReLU(),nn.Linear(h,h))
        depth = cfg.get('num_mlp', 1)
        if depth not in (1,2): raise ValueError('Supported projector depth: 1 or 2')
        self.projector = nn.Linear(h,h) if depth == 1 else nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,h))

    def forward(self, x, adj):
        z = x
        for i, conv in enumerate(self.convs):
            z = conv(z,adj)
            if i < len(self.convs)-1:
                z = F.relu(z)
        return z, self.projector(self.encoder_target(x))

    def embedding(self, x):
        return self.encoder_target(x)

def dense_loss(h, t, edge, tau=.5, lam=.001, lambda_loss=1.):
    z,q = F.normalize(h,dim=1),F.normalize(t,dim=1)
    src,dst = edge
    degree = torch.bincount(dst,minlength=len(z)).to(z.dtype)
    a = torch.exp(z@z.T/tau).sum(1)
    b = lam*torch.exp(z@q.T/tau).sum(1)
    positive = (z[src]*q[dst]).sum(1)/tau
    return ((-positive+lambda_loss*torch.log(a[dst]+b[src]))/degree[dst]).sum()/len(z)

def backward_loss(h, t, edge, chunk=512, tau=.5, lam=.001, lambda_loss=1.):
    # First obtain scalar row sums without storing their quadratic graphs.
    z = F.normalize(h.detach(),dim=1).requires_grad_(True)
    q = F.normalize(t.detach(),dim=1).requires_grad_(True)
    n = len(z)
    with torch.no_grad():
        a = torch.cat([torch.exp(z[i:i+chunk]@z.T/tau).sum(1) for i in range(0,n,chunk)])
        b = torch.cat([lam*torch.exp(z[i:i+chunk]@q.T/tau).sum(1) for i in range(0,n,chunk)])
    a.requires_grad_(True)
    b.requires_grad_(True)
    src,dst = edge
    degree = torch.bincount(dst,minlength=n).to(z.dtype)
    positive = (z[src]*q[dst]).sum(1)/tau
    loss = ((-positive+lambda_loss*torch.log(a[dst]+b[src]))/degree[dst]).sum()/n
    loss.backward()
    for i in range(0,n,chunk):
        ai = torch.exp(z[i:i+chunk]@z.T/tau).sum(1)
        bi = lam*torch.exp(z[i:i+chunk]@q.T/tau).sum(1)
        torch.autograd.backward((ai,bi),(a.grad[i:i+chunk],b.grad[i:i+chunk]))
    torch.autograd.backward((F.normalize(h,dim=1),F.normalize(t,dim=1)),(z.grad,q.grad))
    return float(loss.detach())
