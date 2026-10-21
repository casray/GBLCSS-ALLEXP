from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.frozen_grace_targets import apply_ball_soft_targets
from utils.teacher_verified_gb import (
    aggregate_nontrain_teacher_posteriors,
    build_teacher_verified_loo_targets,
    size_matched_shuffled_assignment,
    teacher_reliability,
)


def test_teacher_reliability_penalizes_entropy_and_instability() -> None:
    stable_confident = np.asarray(
        [
            [[0.98, 0.02]],
            [[0.97, 0.03]],
            [[0.99, 0.01]],
        ]
    )
    unstable = np.asarray(
        [
            [[0.98, 0.02]],
            [[0.02, 0.98]],
            [[0.50, 0.50]],
        ]
    )
    uniform = np.full((3, 1, 2), 0.5)

    stable = teacher_reliability(stable_confident)
    changing = teacher_reliability(unstable)
    uncertain = teacher_reliability(uniform)

    assert stable.reliability[0] > changing.reliability[0]
    assert changing.reliability[0] >= uncertain.reliability[0]
    assert uncertain.reliability[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all((stable.reliability >= 0.0) & (stable.reliability <= 1.0))


def test_ball_teacher_aggregation_uses_only_nontraining_nodes() -> None:
    assignment = np.asarray([0, 0, 0, 0], dtype=np.int64)
    train = np.asarray([0, 1], dtype=np.int64)
    samples = np.asarray(
        [
            [[0.01, 0.99], [0.01, 0.99], [0.95, 0.05], [0.90, 0.10]],
            [[0.02, 0.98], [0.02, 0.98], [0.94, 0.06], [0.91, 0.09]],
        ],
        dtype=np.float64,
    )

    aggregate = aggregate_nontrain_teacher_posteriors(
        assignment,
        train,
        samples,
    )

    assert aggregate.support_count[0] == 2
    assert aggregate.posterior[0, 0] > 0.90
    assert aggregate.posterior[0, 1] < 0.10


def test_verifier_one_recovers_formal_tie_safe_loo_targets() -> None:
    noisy = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assignment = np.zeros(6, dtype=np.int64)
    train = np.arange(6, dtype=np.int64)
    teacher_samples = np.full((2, 6, 3), 1.0 / 3.0)

    formal, formal_alpha, _ = apply_ball_soft_targets(
        noisy_label=torch.as_tensor(noisy),
        assignment=assignment,
        train_mask=train,
        n_classes=3,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )
    candidate = build_teacher_verified_loo_targets(
        noisy,
        assignment,
        train,
        3,
        teacher_samples,
        verification_scale_override=np.ones(noisy.shape[0]),
    )

    np.testing.assert_allclose(candidate.soft_targets, formal.numpy(), atol=1e-7)
    np.testing.assert_allclose(candidate.base_alpha, formal_alpha.numpy(), atol=1e-7)
    np.testing.assert_allclose(candidate.verified_alpha, formal_alpha.numpy(), atol=1e-7)


def test_no_nontraining_teacher_evidence_disables_correction() -> None:
    noisy = np.asarray([0, 0, 1], dtype=np.int64)
    assignment = np.zeros(3, dtype=np.int64)
    train = np.arange(3, dtype=np.int64)
    teacher_samples = np.asarray(
        [
            [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        ]
    )

    result = build_teacher_verified_loo_targets(
        noisy,
        assignment,
        train,
        2,
        teacher_samples,
    )
    expected = np.eye(2)[noisy]

    np.testing.assert_allclose(result.soft_targets, expected)
    assert np.all(result.verified_alpha[train] == 0.0)
    assert result.diagnostics["verified_correction_coverage"] == pytest.approx(0.0)


def test_tie_uses_distribution_instead_of_minimum_class() -> None:
    noisy = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    assignment = np.zeros(5, dtype=np.int64)
    train = np.arange(5, dtype=np.int64)
    teacher_samples = np.full((2, 5, 3), 1.0 / 3.0)

    result = build_teacher_verified_loo_targets(
        noisy,
        assignment,
        train,
        3,
        teacher_samples,
        verification_scale_override=np.ones(5),
    )

    # Removing node 4 leaves a 2:2 tie. distribution_only must not add a
    # minimum-index one-hot component.
    np.testing.assert_allclose(result.local_targets[4], [0.5, 0.5, 0.0])


def test_size_matched_shuffle_preserves_ball_sizes() -> None:
    assignment = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    shuffled = size_matched_shuffled_assignment(assignment, random_state=7)

    original_sizes = sorted(np.bincount(assignment).tolist())
    shuffled_sizes = sorted(np.bincount(shuffled).tolist())
    assert original_sizes == shuffled_sizes
    assert not np.array_equal(assignment, shuffled)
