from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def eligible_mask(assignment, train_mask, min_votes=2):
    assignment = np.asarray(assignment, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    if assignment.shape != train_mask.shape or assignment.ndim != 1:
        raise ValueError('assignment and training mask must be equal-length vectors')
    valid = train_mask & (assignment >= 0)
    result = np.zeros_like(train_mask)
    ids, counts = np.unique(assignment[valid], return_counts=True)
    result[valid] = np.isin(assignment[valid], ids[counts >= min_votes + 1])
    return result


def shuffled_assignment(assignment, eligible, seed=137):
    shuffled = np.asarray(assignment, dtype=np.int64).copy()
    shuffled[eligible] = np.random.default_rng(seed).permutation(shuffled[eligible])
    return shuffled


def mix(teacher, anchor, weight):
    result = teacher.clone()
    active = weight > 0
    result[active] = ((1 - weight[active, None]) * teacher[active]
                      + weight[active, None] * anchor[active])
    return result


def build_arms(teacher, gb, random_gb, noisy_labels, alpha, eligible, strength=.5):
    if not 0 <= strength <= 1:
        raise ValueError('strength must be in [0, 1]')
    eligible = eligible.to(device=teacher.device, dtype=torch.bool)
    weight = strength * alpha * (1 - teacher.max(1).values)
    weight = torch.where(eligible, weight, torch.zeros_like(weight))
    constant = torch.zeros_like(weight)
    if eligible.any():
        constant[eligible] = weight[eligible].mean()
    zero = torch.zeros_like(weight)
    weights = dict(D0=zero, D1=weight.clone(), D2=constant, D3=weight.clone(), D4=weight.clone())
    anchors = dict(D1=F.one_hot(noisy_labels, teacher.shape[1]).float(), D2=gb, D3=gb, D4=random_gb)
    targets = dict(D0=teacher.clone())
    targets.update({arm: mix(teacher, anchors[arm], weights[arm]) for arm in anchors})
    for arm, target in targets.items():
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError(f'invalid probability in {arm}')
        if (target.sum(1) - 1).abs().max() > 1e-5:
            raise ValueError(f'probability rows do not sum to one in {arm}')
        if not torch.equal(target[~eligible], teacher[~eligible]):
            raise ValueError(f'ineligible node modified in {arm}')
    torch.testing.assert_close(constant.sum(), weight.sum(), rtol=1e-5, atol=1e-6)
    return targets, weights
