from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from utils.ordinary_gb import build_random_size_matched_assignment


_EPS = 1e-12


@dataclass(frozen=True)
class TeacherReliability:
    mean_posterior: np.ndarray
    entropy_confidence: np.ndarray
    stability_confidence: np.ndarray
    reliability: np.ndarray


@dataclass(frozen=True)
class BallTeacherAggregate:
    posterior: np.ndarray
    support_count: np.ndarray
    weight_mass: np.ndarray
    node_reliability: np.ndarray


@dataclass(frozen=True)
class TeacherVerifiedTargetResult:
    soft_targets: np.ndarray
    local_targets: np.ndarray
    base_alpha: np.ndarray
    verified_alpha: np.ndarray
    verification_scale: np.ndarray
    teacher_ball_posterior: np.ndarray
    diagnostics: dict


def _validate_probabilities(probabilities, *, ndim: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != ndim:
        raise ValueError(f"probabilities must be {ndim}-D")
    if values.shape[-1] < 2:
        raise ValueError("probabilities must contain at least two classes")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    row_sums = values.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass")
    values = values / row_sums
    return values


def normalized_entropy(probabilities) -> np.ndarray:
    values = _validate_probabilities(probabilities, ndim=np.asarray(probabilities).ndim)
    n_classes = values.shape[-1]
    entropy = -np.sum(values * np.log(np.clip(values, _EPS, 1.0)), axis=-1)
    return np.clip(entropy / math.log(n_classes), 0.0, 1.0)


def js_divergence(left, right) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("JSD inputs must have the same shape")
    left_values = _validate_probabilities(left_values, ndim=left_values.ndim)
    right_values = _validate_probabilities(right_values, ndim=right_values.ndim)
    midpoint = 0.5 * (left_values + right_values)
    left_kl = np.sum(
        left_values
        * (np.log(np.clip(left_values, _EPS, 1.0)) - np.log(np.clip(midpoint, _EPS, 1.0))),
        axis=-1,
    )
    right_kl = np.sum(
        right_values
        * (np.log(np.clip(right_values, _EPS, 1.0)) - np.log(np.clip(midpoint, _EPS, 1.0))),
        axis=-1,
    )
    return np.clip(0.5 * (left_kl + right_kl), 0.0, math.log(2.0))


def teacher_reliability(posterior_samples) -> TeacherReliability:
    samples = _validate_probabilities(posterior_samples, ndim=3)
    mean_posterior = samples.mean(axis=0)
    entropy_confidence = 1.0 - normalized_entropy(mean_posterior)
    expanded_mean = np.broadcast_to(mean_posterior, samples.shape)
    mean_jsd = js_divergence(samples, expanded_mean).mean(axis=0)
    stability_confidence = np.clip(
        1.0 - mean_jsd / math.log(2.0),
        0.0,
        1.0,
    )
    reliability = np.clip(entropy_confidence * stability_confidence, 0.0, 1.0)
    return TeacherReliability(
        mean_posterior=mean_posterior,
        entropy_confidence=entropy_confidence,
        stability_confidence=stability_confidence,
        reliability=reliability,
    )


def aggregate_nontrain_teacher_posteriors(
    assignment,
    train_indices,
    posterior_samples,
) -> BallTeacherAggregate:
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or np.any(assignment < 0):
        raise ValueError("assignment must be a nonnegative 1-D array")
    reliability = teacher_reliability(posterior_samples)
    if reliability.mean_posterior.shape[0] != assignment.shape[0]:
        raise ValueError("assignment and teacher posteriors must share node count")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.ndim != 1:
        raise ValueError("train_indices must be 1-D")
    train_mask = np.zeros(assignment.shape[0], dtype=bool)
    train_mask[train_indices] = True
    n_balls = int(assignment.max()) + 1
    n_classes = reliability.mean_posterior.shape[1]
    posterior = np.zeros((n_balls, n_classes), dtype=np.float64)
    support_count = np.zeros(n_balls, dtype=np.int64)
    weight_mass = np.zeros(n_balls, dtype=np.float64)
    for ball_id in np.unique(assignment):
        nodes = np.flatnonzero((assignment == ball_id) & ~train_mask)
        support_count[ball_id] = nodes.size
        if nodes.size == 0:
            continue
        weights = reliability.reliability[nodes]
        mass = float(weights.sum())
        weight_mass[ball_id] = mass
        if mass <= _EPS:
            continue
        posterior[ball_id] = np.sum(
            reliability.mean_posterior[nodes] * weights[:, None],
            axis=0,
        ) / mass
    return BallTeacherAggregate(
        posterior=posterior,
        support_count=support_count,
        weight_mass=weight_mass,
        node_reliability=reliability.reliability,
    )


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def _local_target_from_counts(counts: np.ndarray, target_beta: float) -> tuple[np.ndarray, float]:
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("counts must contain positive mass")
    distribution = counts.astype(np.float64) / total
    n_classes = counts.shape[0]
    margin = float(distribution.max())
    baseline = 1.0 / n_classes
    alpha = float(np.clip((margin - baseline) / (1.0 - baseline), 0.0, 1.0))
    maxima = np.flatnonzero(counts == counts.max())
    if maxima.size > 1:
        return distribution, alpha
    majority = np.zeros(n_classes, dtype=np.float64)
    majority[int(maxima[0])] = 1.0
    return (1.0 - target_beta) * distribution + target_beta * majority, alpha


def build_teacher_verified_loo_targets(
    noisy_labels,
    assignment,
    train_indices,
    n_classes: int,
    teacher_posterior_samples,
    *,
    target_beta: float = 0.5,
    min_train_per_ball: int = 2,
    verification_scale_override=None,
) -> TeacherVerifiedTargetResult:
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    assignment = np.asarray(assignment, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    n_classes = int(n_classes)
    if noisy_labels.ndim != 1 or assignment.shape != noisy_labels.shape:
        raise ValueError("noisy_labels and assignment must be aligned 1-D arrays")
    if np.any(noisy_labels < 0) or np.any(noisy_labels >= n_classes):
        raise ValueError("noisy_labels contain an invalid class")
    if not 0.0 <= float(target_beta) <= 1.0:
        raise ValueError("target_beta must be in [0, 1]")

    teacher = aggregate_nontrain_teacher_posteriors(
        assignment,
        train_indices,
        teacher_posterior_samples,
    )
    noisy_one_hot = _one_hot(noisy_labels, n_classes)
    soft_targets = noisy_one_hot.copy()
    local_targets = noisy_one_hot.copy()
    base_alpha = np.zeros(noisy_labels.shape[0], dtype=np.float64)
    verified_alpha = np.zeros_like(base_alpha)
    verification_scale = np.zeros_like(base_alpha)
    override = None
    if verification_scale_override is not None:
        override = np.asarray(verification_scale_override, dtype=np.float64)
        if override.shape != noisy_labels.shape:
            raise ValueError("verification_scale_override must match node count")
        if np.any((override < 0.0) | (override > 1.0)):
            raise ValueError("verification_scale_override must be in [0, 1]")

    required_support = int(min_train_per_ball) + 1
    for ball_id in sorted(np.unique(assignment).tolist()):
        voters = train_indices[assignment[train_indices] == ball_id]
        if voters.size < required_support:
            continue
        counts = np.bincount(noisy_labels[voters], minlength=n_classes)
        for node in voters:
            node_counts = counts.copy()
            node_counts[noisy_labels[node]] -= 1
            local_target, node_alpha = _local_target_from_counts(
                node_counts,
                float(target_beta),
            )
            local_targets[node] = local_target
            base_alpha[node] = node_alpha
            if override is not None:
                scale = float(override[node])
            elif teacher.weight_mass[ball_id] <= _EPS:
                scale = 0.0
            else:
                ball_posterior = teacher.posterior[ball_id]
                agreement = 1.0 - float(
                    js_divergence(local_target[None, :], ball_posterior[None, :])[0]
                    / math.log(2.0)
                )
                ball_confidence = 1.0 - float(normalized_entropy(ball_posterior[None, :])[0])
                scale = float(np.clip(agreement * ball_confidence, 0.0, 1.0))
            verification_scale[node] = scale
            verified_alpha[node] = node_alpha * scale
            soft_targets[node] = (
                (1.0 - verified_alpha[node]) * noisy_one_hot[node]
                + verified_alpha[node] * local_target
            )

    train_verified = verified_alpha[train_indices] > 0.0
    train_base = base_alpha[train_indices] > 0.0
    reliable_ball = teacher.weight_mass > _EPS
    diagnostics = {
        "base_correction_coverage": float(np.mean(train_base)),
        "verified_correction_coverage": float(np.mean(train_verified)),
        "mean_base_alpha": float(np.mean(base_alpha[train_indices])),
        "mean_verified_alpha": float(np.mean(verified_alpha[train_indices])),
        "mean_verification_scale": float(np.mean(verification_scale[train_indices])),
        "teacher_ball_coverage": float(np.mean(reliable_ball)),
        "mean_teacher_node_reliability": float(np.mean(teacher.node_reliability)),
    }
    return TeacherVerifiedTargetResult(
        soft_targets=soft_targets,
        local_targets=local_targets,
        base_alpha=base_alpha,
        verified_alpha=verified_alpha,
        verification_scale=verification_scale,
        teacher_ball_posterior=teacher.posterior,
        diagnostics=diagnostics,
    )


def size_matched_shuffled_assignment(assignment, random_state=None) -> np.ndarray:
    return build_random_size_matched_assignment(
        assignment,
        random_state=random_state,
    )

