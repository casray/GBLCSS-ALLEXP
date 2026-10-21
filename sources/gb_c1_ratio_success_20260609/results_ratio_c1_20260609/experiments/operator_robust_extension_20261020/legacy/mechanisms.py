"""NumPy-only training-label targets for fixed-budget coarsening controls."""
import numpy as np


def _validated_assignment(assignment):
    values = np.asarray(assignment)
    if (values.ndim != 1 or not values.size
            or not np.issubdtype(values.dtype, np.integer)):
        raise ValueError("assignment must be a nonempty integer vector")
    unique = np.unique(values)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("assignment IDs must be contiguous 0..M-1")
    return values.astype(np.int64, copy=False), len(unique)


def random_size_matched_assignment(assignment: np.ndarray, seed: int) -> np.ndarray:
    """Permute full-node memberships, preserving every ball's exact node count.

    The function receives neither labels nor masks. Individual seeds may leave
    some memberships unchanged; no rejection step conditions the random null.
    """
    assignment, _ = _validated_assignment(assignment)
    return np.random.default_rng(seed).permutation(assignment)


def build_targets(labels: np.ndarray, assignment: np.ndarray,
                  train_indices: np.ndarray, n_classes: int, mode: str,
                  target_beta: float = .5) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return train-ordered targets, inclusive ball q, train counts, diagnostics.

    A: noisy one-hot. B: inclusive ball empirical distribution. Bprime: pure
    leave-one-out distribution with own-label fallback for support one. LCSS:
    original GBLCSS confidence-weighted, tie-safe leave-one-out correction,
    requiring at least two remaining votes. Bprime is not complete GBLCSS.

    Only labels indexed by train_indices are inspected. Balls with no training
    support have zero q and zero count; target rows always sum to one.
    """
    assignment, num_balls = _validated_assignment(assignment)
    labels, train = np.asarray(labels), np.asarray(train_indices)
    if labels.ndim != 1 or len(labels) != len(assignment):
        raise ValueError("labels must be a vector matching assignment length")
    if (train.ndim != 1 or not train.size
            or not np.issubdtype(train.dtype, np.integer)):
        raise ValueError("train_indices must be a nonempty integer vector")
    if (np.any(train < 0) or np.any(train >= len(labels))
            or len(np.unique(train)) != len(train)):
        raise ValueError("train_indices must be unique valid node indices")
    if (not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, (bool, np.bool_))
            or n_classes < 2):
        raise ValueError("n_classes must be an integer >= 2")
    if mode not in ("A", "B", "Bprime", "LCSS"):
        raise ValueError("mode must be A, B, Bprime, or LCSS")
    if not np.isfinite(target_beta) or not 0. <= target_beta <= 1.:
        raise ValueError("target_beta must lie in [0,1]")

    # Index before validation: held-out label values are not target inputs.
    supported = labels[train]
    if (not np.issubdtype(supported.dtype, np.number)
            or np.iscomplexobj(supported) or not np.isfinite(supported).all()
            or np.any(supported < 0) or np.any(supported >= n_classes)
            or np.any(supported != np.floor(supported))):
        raise ValueError("training labels must be integer class IDs in [0,n_classes)")
    supported = supported.astype(np.int64)
    train_balls = assignment[train]
    histogram = np.zeros((num_balls, n_classes), dtype=np.int64)
    np.add.at(histogram, (train_balls, supported), 1)
    support = histogram.sum(axis=1)
    q = np.divide(histogram, support[:, None],
                  out=np.zeros(histogram.shape, dtype=np.float32),
                  where=support[:, None] > 0)
    onehot = np.eye(n_classes, dtype=np.float32)[supported]
    targets = onehot.copy()
    eligible_nodes = tied_nodes = 0
    if mode == "B":
        targets = q[train_balls].copy()
    elif mode in ("Bprime", "LCSS"):
        needed = 2 if mode == "Bprime" else 3
        rows = np.flatnonzero(support[train_balls] >= needed)
        if len(rows):
            remaining = histogram[train_balls[rows]].copy()
            remaining[np.arange(len(rows)), supported[rows]] -= 1
            denominator = support[train_balls[rows]] - 1
            distribution = remaining.astype(np.float32) / denominator[:, None].astype(np.float32)
            if mode == "Bprime":
                targets[rows] = distribution
            else:
                maximum = remaining.max(axis=1)
                # Compute confidence in float64 like the original Python formula,
                # then cast coefficients before float32 tensor-style arithmetic.
                alpha = np.clip((maximum / denominator - 1. / n_classes)
                                / (1. - 1. / n_classes), 0., 1.)
                active = alpha > 0.
                rows, remaining = rows[active], remaining[active]
                distribution, alpha = distribution[active], alpha[active]
                tied = (remaining == remaining.max(axis=1, keepdims=True)).sum(axis=1) > 1
                local = distribution.copy()
                unique = ~tied
                majority = np.eye(n_classes, dtype=np.float32)[remaining[unique].argmax(axis=1)]
                local[unique] = (np.float32(1. - target_beta) * distribution[unique]
                                 + np.float32(target_beta) * majority)
                targets[rows] = (np.asarray(1. - alpha, dtype=np.float32)[:, None] * onehot[rows]
                                 + alpha.astype(np.float32)[:, None] * local)
                eligible_nodes, tied_nodes = len(rows), int(tied.sum())

    positive = targets > 0
    entropy_terms = np.zeros_like(targets)
    entropy_terms[positive] = -targets[positive] * np.log(targets[positive])
    diagnostics = {
        "mode": mode,
        "num_balls": int(num_balls),
        "train_support_total": int(support.sum()),
        "train_support_min": int(support.min()),
        "train_support_max": int(support.max()),
        "train_support_mean": float(support.mean()),
        "empty_train_balls": int(np.count_nonzero(support == 0)),
        "singleton_train_balls": int(np.count_nonzero(support == 1)),
        "nonempty_train_balls": int(np.count_nonzero(support)),
        "target_entropy_mean": float(entropy_terms.sum(axis=1).mean()),
        "eligible_train_nodes": int(eligible_nodes),
        "tied_train_nodes": int(tied_nodes),
        "leave_one_out": int(mode in ("Bprime", "LCSS")),
        "target_beta": float(target_beta),
        "tie_mode": "distribution_only" if mode == "LCSS" else "not_applicable",
    }
    return targets, q, support.astype(np.float32), diagnostics
