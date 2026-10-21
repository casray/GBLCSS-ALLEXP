from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BallNoiseRateEstimate:
    weighted_estimate: float
    unweighted_estimate: float
    coverage: float
    eligible_count: int
    train_count: int
    effective_sample_size: float
    weight_sum: float
    per_ball: list[dict]


def _mask_to_indices(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        values = mask.detach().cpu().numpy()
    else:
        values = np.asarray(mask)
    if values.dtype == np.bool_:
        return np.flatnonzero(values).astype(np.int64)
    return values.astype(np.int64, copy=False).reshape(-1)


def estimate_ball_noise_rate_loo(
    assignment,
    noisy_label,
    train_mask,
    n_classes,
    min_train_votes=3,
) -> BallNoiseRateEstimate:
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    labels = torch.as_tensor(noisy_label).detach().cpu().long().numpy().reshape(-1)
    if assignment.shape[0] != labels.shape[0]:
        raise ValueError("assignment and noisy_label must have the same length")
    if int(n_classes) < 2:
        raise ValueError("n_classes must be at least 2")
    if int(min_train_votes) < 2:
        raise ValueError("min_train_votes must be at least 2")
    if labels.size and (labels.min() < 0 or labels.max() >= int(n_classes)):
        raise ValueError("noisy_label contains a class outside [0, n_classes)")

    train_idx = _mask_to_indices(train_mask)
    if train_idx.size and (train_idx.min() < 0 or train_idx.max() >= labels.shape[0]):
        raise ValueError("train_mask contains an out-of-range index")

    all_weights = []
    all_disagreements = []
    per_ball = []
    random_baseline = 1.0 / float(n_classes)
    denominator = 1.0 - random_baseline

    for ball_id in np.unique(assignment[train_idx]) if train_idx.size else []:
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_votes):
            continue
        counts = np.bincount(labels[ball_train], minlength=int(n_classes)).astype(np.int64)
        ball_weights = []
        ball_disagreements = []
        for node in ball_train:
            loo_counts = counts.copy()
            loo_counts[labels[node]] -= 1
            vote_count = int(ball_train.size - 1)
            majority = int(np.argmax(loo_counts))
            majority_fraction = float(loo_counts[majority] / max(1, vote_count))
            weight = max(0.0, (majority_fraction - random_baseline) / denominator)
            disagreement = float(labels[node] != majority)
            ball_weights.append(weight)
            ball_disagreements.append(disagreement)

        weights = np.asarray(ball_weights, dtype=np.float64)
        disagreements = np.asarray(ball_disagreements, dtype=np.float64)
        weight_sum = float(weights.sum())
        per_ball.append(
            {
                "ball_id": int(ball_id),
                "train_votes": int(ball_train.size),
                "weighted_estimate": (
                    float(np.dot(weights, disagreements) / weight_sum)
                    if weight_sum > 0.0
                    else float("nan")
                ),
                "unweighted_estimate": float(disagreements.mean()),
                "mean_weight": float(weights.mean()),
                "weight_sum": weight_sum,
            }
        )
        all_weights.extend(ball_weights)
        all_disagreements.extend(ball_disagreements)

    weights = np.asarray(all_weights, dtype=np.float64)
    disagreements = np.asarray(all_disagreements, dtype=np.float64)
    eligible_count = int(disagreements.size)
    weight_sum = float(weights.sum()) if weights.size else 0.0
    weighted_estimate = (
        float(np.dot(weights, disagreements) / weight_sum)
        if weight_sum > 0.0
        else float("nan")
    )
    unweighted_estimate = (
        float(disagreements.mean()) if disagreements.size else float("nan")
    )
    weight_square_sum = float(np.square(weights).sum()) if weights.size else 0.0
    effective_sample_size = (
        float(weight_sum * weight_sum / weight_square_sum)
        if weight_square_sum > 0.0
        else 0.0
    )
    train_count = int(train_idx.size)
    return BallNoiseRateEstimate(
        weighted_estimate=weighted_estimate,
        unweighted_estimate=unweighted_estimate,
        coverage=float(eligible_count / max(1, train_count)),
        eligible_count=eligible_count,
        train_count=train_count,
        effective_sample_size=effective_sample_size,
        weight_sum=weight_sum,
        per_ball=per_ball,
    )
