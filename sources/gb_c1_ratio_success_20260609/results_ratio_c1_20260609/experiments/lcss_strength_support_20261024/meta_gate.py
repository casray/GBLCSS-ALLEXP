"""One exact differentiable SGD virtual update for a frozen linear head."""
import torch
import torch.nn.functional as F


def static_features(e):
    f = torch.stack([e['maximum'], e['margin'], e['qz'], torch.log1p(e['support']), e['alpha']], 1)
    mean, sd = f.mean(0), f.std(0, unbiased=False).clamp_min(1e-8)
    return (f-mean)/sd


def gate_features(e, p, static=None):
    p = p.detach()
    static = static_features(e).to(p) if static is None else static.to(p)
    z = e['z'].to(p.device)
    q = e['q'].to(p)
    freq = torch.bincount(z, minlength=p.shape[1]).to(p)/len(z)
    dynamic = torch.stack([p.gather(1, z[:, None]).squeeze(1), (q*p).sum(1),
                           -(p*p.clamp_min(1e-30).log()).sum(1)/torch.log(p.new_tensor(p.shape[1])),
                           freq[z]], 1)
    return torch.cat([static, dynamic], 1).detach()


def gate_target(e, weights, direction=None):
    own = e['own'].to(weights)
    d = e['d'].to(weights) if direction is None else direction.to(weights)
    mass = e['m'].to(weights)*weights
    return (1-mass[:, None])*own+mass[:, None]*d


def meta_step(head, gate, optimizer, train_x, meta_x, meta_y, e, features, direction=None):
    optimizer.zero_grad(set_to_none=True)
    weights = gate(features).sigmoid().squeeze(1)
    target = gate_target(e, weights, direction)
    value = -(target*head(train_x).log_softmax(1)).sum(1).mean()
    params = tuple(head.parameters())
    grad = torch.autograd.grad(value, params, create_graph=True)
    virtual = [p-.01*g for p, g in zip(params, grad)]
    prediction = F.linear(meta_x, virtual[0], virtual[1])
    penalty = torch.cat([p.reshape(-1) for p in gate.parameters()]).square().mean()
    outer = F.cross_entropy(prediction, meta_y)+.01*penalty
    gate_grad = torch.autograd.grad(outer, tuple(gate.parameters()))
    norm = float(torch.stack([g.square().sum() for g in gate_grad]).sum().sqrt())
    for p, g in zip(gate.parameters(), gate_grad):
        p.grad = g
    optimizer.step()
    return float(outer.detach()), norm
