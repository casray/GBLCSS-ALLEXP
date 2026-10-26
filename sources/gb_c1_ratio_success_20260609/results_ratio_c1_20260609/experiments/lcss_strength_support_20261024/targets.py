"""LCSS evidence and target interventions; CPU float64 construction."""
import numpy as np
import torch
import torch.nn.functional as F


def evidence(z, assignment, classes):
    z, assignment = z.cpu().long(), assignment.cpu().long()
    own = F.one_hot(z, classes).double()
    _, inverse = torch.unique(assignment, sorted=True, return_inverse=True)
    counts = own.new_zeros((int(inverse.max())+1, classes))
    counts.index_add_(0, inverse, own)
    counts = counts[inverse]-own
    support = counts.sum(1)
    vote = counts/support.clamp_min(1)[:, None]
    maximum, winner = counts.max(1)
    tied = (counts == maximum[:, None]).sum(1) > 1
    alpha = ((maximum/support.clamp_min(1)-1/classes)/(1-1/classes)).clamp(0, 1)
    alpha[support < 2] = 0
    beta = torch.where(tied, 0., .5)
    q = (1-beta[:, None])*vote+beta[:, None]*F.one_hot(winner, classes)
    qz = q.gather(1, z[:, None]).squeeze(1)
    m = alpha*(1-qz)
    d = q.clone()
    d.scatter_(1, z[:, None], 0)
    d = d/(1-qz).clamp_min(1e-30)[:, None]
    native = ((1-alpha[:, None])*own+alpha[:, None]*q).float()
    sorted_votes = vote.sort(1, descending=True).values
    return dict(z=z, own=own.float(), counts=counts, support=support, vote=vote,
                winner=winner, tied=tied, alpha=alpha, q=q, qz=qz, m=m, d=d,
                native=native, maximum=sorted_votes[:, 0], margin=sorted_votes[:, 0]-sorted_votes[:, 1])


def shrink(e, strength, direction=None):
    w = torch.as_tensor(strength, dtype=torch.float64, device=e['m'].device)
    if bool(((w < 0) | (w > 1)).any()):
        raise ValueError('Strength outside [0,1]')
    mass = e['m']*w
    direction = e['d'] if direction is None else direction
    result = ((1-mass[:, None])*e['own'].double()+mass[:, None]*direction).float()
    if not torch.isfinite(result).all() or not torch.allclose(result.sum(1), torch.ones(len(result), device=result.device), atol=4e-7):
        raise ValueError('Invalid target')
    return result


def mass_direction(e):
    return (1-e['own'].double())/(e['own'].shape[1]-1)


def match_mass(m, original, permuted):
    m, w, r = m.double(), original.double(), permuted.double()
    total, target, current = m.sum(), (m*w).sum(), (m*r).sum()
    if total == 0:
        return torch.zeros_like(r)
    if current > target:
        r = r*(target/current)
    elif current < target:
        r = r+(target-current)/(total-current)*(1-r)
    if not torch.allclose((m*r).sum(), target, atol=1e-9, rtol=1e-12):
        raise ValueError('Transferred mass mismatch')
    return r.clamp(0, 1)


def global_mass(m, w):
    return float((m.double()*w.double()).sum()/m.double().sum()) if m.sum() > 0 else 0.


def groups(e):
    agree = (e['winner'] == e['z']) & ~e['tied']
    return torch.where(e['tied'], 2, torch.where(agree, 0, 1))


def b2_targets(es):
    r = es[32]
    j = r['m'] > 1e-8
    for e in es.values():
        j &= (e['support'] >= 2) & (1-e['qz'] > 1e-8)
    result = {f'N{s}': e['native'].clone() for s, e in es.items()}
    for s in (8, 128):
        for prefix, mass, direction in [('D', r['m'], es[s]['d']),
                                         ('M', es[s]['m'], r['d']),
                                         ('DM', es[s]['m'], es[s]['d'])]:
            t = r['native'].clone()
            intervention = ((1-mass[:, None])*r['own'].double()+mass[:, None]*direction).float()
            t[j] = intervention[j]
            result[f'{prefix}{s}'] = t
    return result, j


def observable_stats(x, edge_index, train, e, k, strength):
    n, dim = x.shape
    c = e['own'].shape[1]
    freq = torch.bincount(e['z'], minlength=c).double()
    p = freq/freq.sum()
    entropy = float(-(p[p > 0]*p[p > 0].log()).sum()/np.log(c)) if c > 1 else 0.
    labels = torch.full((n,), -1, dtype=torch.long)
    labels[train] = e['z']
    src, dst = edge_index
    keep = (labels[src] >= 0) & (labels[dst] >= 0)
    hom = float((labels[src[keep]] == labels[dst[keep]]).double().mean()) if keep.any() else 0.
    values = [np.log1p(n), np.log1p(dim), c, len(train)/n, np.log1p(len(train)/c),
              entropy, float(freq.std(unbiased=False)/freq.mean()), np.log1p(len(src)/n), hom,
              float(not keep.any()), np.log2(k), float(strength), float(e['support'].mean()),
              float(e['support'].median()), float(torch.quantile(e['support'], .9)),
              float((e['support'] >= 2).double().mean()), float(e['maximum'].mean()),
              float(e['margin'].mean())]
    leaders = (e['counts'] == e['counts'].max(1).values[:, None]).double()
    leaders = leaders/leaders.sum(1)[:, None]
    sc = float(leaders.gather(1, e['z'][:, None]).mean())
    return values+[sc, float(e['m'].mean())]
