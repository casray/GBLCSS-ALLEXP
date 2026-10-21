"""Loss and matched-control kernels for the R2 mechanism experiment."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mass_matched(target, observed):
    own = target.gather(1, observed[:, None])
    out = ((1-own)/(target.shape[1]-1)).expand_as(target).clone()
    out.scatter_(1, observed[:, None], own)
    return out


def loss_for(logits, observed, target, rule, *, epsilon=.1, q=.7, label_weight=.8):
    log_p = logits.log_softmax(1)
    if rule == 'GCE':
        return ((1-log_p.gather(1, observed[:, None]).exp().pow(q))/q).mean()
    if rule == 'LS':
        target = (1-epsilon)*F.one_hot(observed, logits.shape[1])+epsilon/logits.shape[1]
    elif rule == 'BOOT':
        target = label_weight*F.one_hot(observed, logits.shape[1])+(1-label_weight)*log_p.detach().exp()
    elif rule not in ('CE', 'GB', 'MASS', 'KNN', 'KM', 'RANDOM'):
        raise ValueError(rule)
    return -(target*log_p).sum(1).mean()


def targets_from_counts(observed, counts, beta=.5, min_votes=2):
    support = counts.sum(1)
    r = counts/support.clamp_min(1)[:, None]
    maximum, winner = counts.max(1)
    classes = counts.shape[1]
    alpha = ((maximum/support.clamp_min(1)-1/classes)/(1-1/classes)).clamp(0, 1)
    alpha = torch.where(support >= min_votes, alpha, 0.)
    tied = (counts == maximum[:, None]).sum(1) > 1
    strength = torch.where(tied, 0., beta)
    q = (1-strength[:, None])*r+strength[:, None]*F.one_hot(winner, classes)
    target = (1-alpha[:, None])*F.one_hot(observed, classes)+alpha[:, None]*q
    if not torch.isfinite(target).all() or not torch.allclose(target.sum(1), torch.ones_like(alpha), atol=1e-6):
        raise ValueError('Invalid vote target')
    return target, alpha


def support_counts(assignments):
    _, inv, counts = torch.unique(assignments, return_inverse=True, return_counts=True)
    return counts[inv]-1


def neighbor_indices(h, support, batch_size=256):
    n, max_votes = len(h), int(support.max())
    if int(support.min()) < 0 or max_votes >= n:
        raise ValueError('Invalid leave-one-out neighbor count')
    result = torch.empty((n, max_votes), dtype=torch.long, device=h.device)
    h = F.normalize(h, dim=1)
    for start in range(0, n, batch_size):
        stop = min(start+batch_size, n)
        scores = h[start:stop]@h.T
        scores[torch.arange(stop-start, device=h.device), torch.arange(start, stop, device=h.device)] = -torch.inf
        result[start:stop] = scores.argsort(dim=1, descending=True, stable=True)[:, :max_votes]
    return result


def random_partition(assignment, train, seed):
    original = assignment.cpu()
    train = train.cpu()
    mask = torch.ones(len(original), dtype=torch.bool)
    mask[train] = False
    result = original.clone()
    generator = torch.Generator().manual_seed(seed)
    for ids in (train, mask.nonzero(as_tuple=True)[0]):
        result[ids] = original[ids[torch.randperm(len(ids), generator=generator)]]
    return result.to(assignment.device)


def shuffle_rows(h, seed):
    order = torch.randperm(len(h), generator=torch.Generator().manual_seed(seed)).to(h.device)
    return h[order], order
