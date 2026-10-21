"""Label-free context objectives and exact memory-bounded GRACE loss."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class WeightedGRACE(nn.Module):
    def __init__(self, input_dim: int, hidden: int, projection: int,
                 temperature: float, activation: str):
        super().__init__()
        self.convolutions = nn.ModuleList([GCNConv(input_dim, 2*hidden), GCNConv(2*hidden, hidden)])
        self.activation = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.projection_1 = nn.Linear(hidden, projection)
        self.projection_2 = nn.Linear(projection, hidden)
        self.temperature = temperature

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.convolutions:
            x = self.activation(layer(x, edge_index, edge_weight))
        return x

    def project(self, h):
        return self.projection_2(F.elu(self.projection_1(h)))


class _BlockLoss(torch.autograd.Function):
    """Recompute all negative probabilities in backward; never sample negatives."""
    @staticmethod
    def forward(ctx, first, second, temperature, chunk_size):
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError('views must have the same 2D shape')
        if temperature <= 0 or chunk_size < 1:
            raise ValueError('positive temperature and chunk size required')
        ctx.save_for_backward(first, second)
        ctx.temperature, ctx.chunk_size = temperature, chunk_size
        n = first.shape[0]
        result = first.new_zeros(())
        for anchor, keys in ((first, second), (second, first)):
            for start in range(0, n, chunk_size):
                end = min(start+chunk_size, n)
                row = torch.arange(end-start, device=first.device)
                column = torch.arange(start, end, device=first.device)
                within = anchor[start:end] @ anchor.T / temperature
                between = anchor[start:end] @ keys.T / temperature
                positive = between[row, column]
                within[row, column] = -torch.inf
                logden = torch.logaddexp(torch.logsumexp(within, 1), torch.logsumexp(between, 1))
                result += (logden-positive).sum()/(2*n)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        first, second = ctx.saved_tensors
        tau, chunk = ctx.temperature, ctx.chunk_size
        n = len(first)
        g1, g2 = torch.zeros_like(first), torch.zeros_like(second)
        for anchor, keys, ga, gk in ((first, second, g1, g2), (second, first, g2, g1)):
            for start in range(0, n, chunk):
                end = min(start+chunk, n)
                row = torch.arange(end-start, device=first.device)
                column = torch.arange(start, end, device=first.device)
                within = anchor[start:end] @ anchor.T / tau
                between = anchor[start:end] @ keys.T / tau
                within[row, column] = -torch.inf
                logden = torch.logaddexp(torch.logsumexp(within, 1), torch.logsumexp(between, 1))
                pw = (within-logden[:, None]).exp()
                pb = (between-logden[:, None]).exp()
                pb[row, column] -= 1
                scale = grad_output/(2*n*tau)
                ga[start:end] += scale * (pw @ anchor + pb @ keys)
                ga += scale * (pw.T @ anchor[start:end])
                gk += scale * (pb.T @ anchor[start:end])
        return g1, g2, None, None


def contrastive_loss(first, second, temperature, chunk_size=1024):
    return _BlockLoss.apply(F.normalize(first, dim=1), F.normalize(second, dim=1),
                            float(temperature), int(chunk_size))


def neighbor_mean(h, edges):
    source, target = edges
    keep = source != target
    source, target = source[keep], target[keep]
    count = torch.bincount(target, minlength=len(h))
    adjacency = torch.sparse_coo_tensor(torch.stack([target, source]),
        torch.ones(len(source), dtype=h.dtype, device=h.device), (len(h), len(h))).coalesce()
    mean = torch.sparse.mm(adjacency, h)/count.clamp_min(1)[:, None]
    return mean, count > 0


def sample_context_edges(edges, num_nodes, max_neighbors=16):
    if max_neighbors < 1:
        raise ValueError('max_neighbors must be positive')
    edges = edges[:, edges[0] != edges[1]]
    # Randomize within destination groups, then retain the first k per root.
    order = torch.randperm(edges.shape[1], device=edges.device)
    order = order[torch.argsort(edges[1, order], stable=True)]
    targets = edges[1, order]
    counts = torch.bincount(targets, minlength=num_nodes)
    offsets = torch.cumsum(counts, 0)-counts
    rank = torch.arange(len(order), device=edges.device)-offsets[targets]
    return edges[:, order[rank < max_neighbors]]


def vicreg(first, second, mask):
    if first.shape != second.shape or mask.shape != (len(first),):
        raise ValueError('context views and common mask must agree')
    count = int(mask.sum().item())
    if count < 2:
        return (first.sum()+second.sum())*0, {'valid_roots': count, 'invariance': 0.,
                                                   'variance': 0., 'covariance': 0.}
    first, second = first[mask], second[mask]
    invariance = (first-second).square().mean()
    variance = F.relu(1-torch.sqrt(first.var(0, correction=1)+1e-4)).mean()
    variance += F.relu(1-torch.sqrt(second.var(0, correction=1)+1e-4)).mean()
    covariance = first.new_zeros(())
    for x in (first, second):
        centered = x-x.mean(0)
        cov = centered.T @ centered/(count-1)
        covariance += (cov.square().sum()-cov.diag().square().sum())/x.shape[1]
    loss = 25*invariance+25*variance+covariance
    return loss, {'valid_roots': count, 'invariance': float(invariance.detach()),
                  'variance': float(variance.detach()), 'covariance': float(covariance.detach())}


def combine_space(h, c, context_mix, structural_mask):
    if not 0 <= context_mix <= 1:
        raise ValueError('context_mix must be in [0,1]')
    valid = structural_mask & (h.norm(dim=1)>1e-8) & (c.norm(dim=1)>1e-8)
    combined = torch.cat([math.sqrt(1-context_mix)*F.normalize(h, dim=1),
                          math.sqrt(context_mix)*F.normalize(c, dim=1)], dim=1)
    return combined, valid


def geometry(h):
    centered = h.double()-h.double().mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered/max(len(h)-1, 1)).clamp_min(0)
    weights = eigenvalues/eigenvalues.sum().clamp_min(1e-20)
    rank = torch.exp(-(weights*torch.log(weights.clamp_min(1e-20))).sum())
    return {'effective_rank': float(rank), 'mean_feature_std': float(centered.std(0).mean()),
            'mean_norm': float(h.norm(dim=1).mean()), 'finite': bool(torch.isfinite(h).all())}
