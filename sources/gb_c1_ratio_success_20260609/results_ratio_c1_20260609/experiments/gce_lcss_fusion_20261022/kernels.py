"""Fixed GCE/LCSS loss definitions and truth-only offline diagnostics."""
import torch
import torch.nn.functional as F


def mass_target(target, observed):
    own = target.gather(1, observed[:, None])
    mass = ((1-own)/(target.shape[1]-1)).expand_as(target).clone()
    mass.scatter_(1, observed[:, None], own)
    return mass


def objective(logits, observed, target, rule, q=.7):
    if not 0 < q < 1:
        raise ValueError('This fixed experiment requires 0 < q < 1')
    log_p = logits.log_softmax(1)
    weight = torch.ones(len(logits), dtype=logits.dtype, device=logits.device)
    if rule == 'GCE':
        return ((1-log_p.gather(1, observed[:, None]).exp().pow(q))/q).mean(), weight
    if rule == 'SOFT_GCE':
        return (target*(1-log_p.exp().pow(q))/q).sum(1).mean(), weight
    if rule == 'CE':
        target = F.one_hot(observed, logits.shape[1]).to(logits.dtype)
    elif rule in ('MASS', 'W_MASS'):
        target = mass_target(target, observed)
    elif rule not in ('LCSS', 'W_LCSS'):
        raise ValueError(rule)
    if rule.startswith('W_'):
        weight = (target*log_p.exp()).sum(1).detach().pow(q)
    return (weight*(-target*log_p).sum(1)).mean(), weight


def target_diagnostic(target, observed, clean, q=.7):
    if not len(target):
        return dict(count=0)
    top = target.max(1).values
    ties = (target-top[:, None]).abs() <= 1e-7
    unique = ties.sum(1) == 1
    predicted = target.argmax(1)
    row = torch.arange(len(target), device=target.device)
    one = F.one_hot(clean, target.shape[1]).to(target.dtype)
    sharp = target.double().pow(1/(1-q))
    sharp = sharp/sharp.sum(1, keepdim=True)
    brier = (target-one).square().sum(1)
    sharp_brier = (sharp-one).square().sum(1)
    def count(mask):
        return int(mask.sum())
    return dict(count=len(target), tie_count=count(~unique),
                observed_unique_count=count(unique & (predicted == observed)),
                clean_unique_count=count(unique & (predicted == clean)),
                other_unique_count=count(unique & (predicted != observed) & (predicted != clean)),
                observed_in_max_count=count(ties[row, observed]),
                clean_in_max_count=count(ties[row, clean]),
                argmax_observed_count=count(predicted == observed),
                argmax_clean_count=count(predicted == clean),
                mean_observed_probability=float(target[row, observed].mean()),
                mean_clean_probability=float(target[row, clean].mean()),
                mean_brier=float(brier.mean()), mean_sharpened_brier=float(sharp_brier.mean()),
                sharpened_brier_change=float((sharp_brier-brier).mean()))
