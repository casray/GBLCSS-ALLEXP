from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gb_knowledge_diagnostic_20261007'))
from diagnose import allocate, moments


def weighted_dist(logits, teacher, weights, population):
    if (logits.ndim != 2 or teacher.shape != logits.shape or weights.shape != (len(logits),)
            or population < len(logits) or not torch.isfinite(weights).all()
            or (weights <= 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.), atol=1e-6)):
        raise ValueError('invalid shapes, population, or normalized positive weights')
    student = logits.softmax(1)
    a, b = student - student.mean(1, keepdim=True), teacher - teacher.mean(1, keepdim=True)
    inter = 1 - ((a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1) + 1e-8) * weights).sum()
    a = student - (weights[:, None]*student).sum(0, keepdim=True)
    b = teacher - (weights[:, None]*teacher).sum(0, keepdim=True)
    covariance = (weights[:, None]*a*b).sum(0)
    norm = ((weights[:, None]*a.square()).sum(0).sqrt()
            * (weights[:, None]*b.square()).sum(0).sqrt())
    intra = 1 - (covariance/(norm + 1e-8/population)).mean()
    return inter + intra


def prepare(assignment, teacher, budget, method):
    n = len(teacher)
    if not 1 <= budget <= n or len(assignment) != n:
        raise ValueError('invalid budget or assignment length')
    if method == 'uniform':
        return {'groups': [np.arange(n)], 'allocation': np.array([budget]), 'n': n}
    if method not in ('size', 'adaptive'):
        raise ValueError('unknown allocation method')
    sizes, _, ss = moments(teacher, assignment)
    variance = np.divide(ss, sizes-1, out=np.zeros_like(ss), where=sizes > 1)
    weights = sizes if method == 'size' else sizes*np.sqrt(variance)
    allocation = allocate(sizes, budget, weights)
    return {'groups': [np.flatnonzero(assignment == j) for j in range(len(sizes))],
            'allocation': allocation, 'n': n}


def take(plan, priority):
    ids, weights = [], []
    for members, count in zip(plan['groups'], plan['allocation']):
        chosen = members[np.argsort(priority[members], kind='stable')[:count]]
        ids.append(chosen)
        weights.append(np.full(count, len(members)/(plan['n']*count)))
    return np.concatenate(ids), np.concatenate(weights)


def sample(assignment, teacher, budget, method, seed):
    plan = prepare(assignment, teacher, budget, method)
    ids, weights = take(plan, np.random.default_rng(seed).random(len(teacher)))
    return ids, weights, plan['allocation']
