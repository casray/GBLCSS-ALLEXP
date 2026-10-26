"""Noisy-label-only scoring of fixed leave-one-out candidates."""
from __future__ import annotations

import numpy as np


def loo_counts(labels, groups, classes):
    z, a = np.asarray(labels), np.asarray(groups)
    if z.ndim != 1 or a.shape != z.shape or len(z) == 0 or classes < 2:
        raise ValueError('Expected nonempty labels and matching assignments')
    if not np.issubdtype(z.dtype, np.integer) or (z < 0).any() or (z >= classes).any():
        raise ValueError('Invalid labels')
    _, inverse = np.unique(a, return_inverse=True)
    hist = np.bincount(inverse*classes+z,
                       minlength=(inverse.max()+1)*classes).reshape(-1, classes)
    counts = hist[inverse].copy()
    counts[np.arange(len(z)), z] -= 1
    return counts


def majority_probabilities(counts):
    counts = np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[1] < 2 or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts.sum(1) <= 0).any()):
        raise ValueError('Every query must have finite nonnegative votes and at least one donor')
    winners = counts == counts.max(1, keepdims=True)
    return winners / winners.sum(1, keepdims=True)


def agreement(probabilities, labels):
    p, z = np.asarray(probabilities), np.asarray(labels)
    if p.ndim != 2 or z.shape != (len(p),) or len(z) == 0:
        raise ValueError('Incompatible score inputs')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probability rows')
    if not np.issubdtype(z.dtype, np.integer) or (z < 0).any() or (z >= p.shape[1]).any():
        raise ValueError('Invalid score labels')
    return float(p[np.arange(len(z)), z].mean())


def choose_support(scores, tolerance=1e-12):
    if not scores or tolerance < 0 or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Scores must be nonempty and finite')
    best = max(scores.values())
    return int(max(s for s, value in scores.items() if best-value <= tolerance))
