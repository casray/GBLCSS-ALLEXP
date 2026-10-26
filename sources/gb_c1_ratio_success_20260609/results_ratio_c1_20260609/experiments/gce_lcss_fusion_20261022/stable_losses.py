"""Algebraically identical GCE, evaluated in log space to avoid 0**q gradients."""
import torch
from kernels import objective


def stable_objective(logits, observed, target, rule, q=.7):
    if rule not in ('GCE', 'SOFT_GCE'):
        return objective(logits, observed, target, rule, q)
    if not 0 < q < 1:
        raise ValueError('This fixed experiment requires 0 < q < 1')
    log_p = logits.log_softmax(1)
    power = (q*log_p).exp()
    weight = torch.ones(len(logits), dtype=logits.dtype, device=logits.device)
    if rule == 'GCE':
        return ((1-power.gather(1, observed[:, None]))/q).mean(), weight
    return (target*(1-power)/q).sum(1).mean(), weight
