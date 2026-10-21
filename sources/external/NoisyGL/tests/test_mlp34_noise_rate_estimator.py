import math

import numpy as np
import torch

from utils.lcss_noise_rate import estimate_ball_noise_rate_loo
from scripts.run_mlp34_noise_rate_diagnostic import evaluate_rhohat_gate


def test_pure_ball_has_zero_estimated_noise():
    result = estimate_ball_noise_rate_loo(
        assignment=np.array([0, 0, 0, 0]),
        noisy_label=torch.tensor([1, 1, 1, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        n_classes=2,
        min_train_votes=3,
    )

    assert result.weighted_estimate == 0.0
    assert result.unweighted_estimate == 0.0
    assert result.coverage == 1.0
    assert result.eligible_count == 4
    assert result.effective_sample_size == 4.0


def test_leave_one_out_excludes_each_nodes_own_vote():
    result = estimate_ball_noise_rate_loo(
        assignment=np.array([0, 0, 0]),
        noisy_label=torch.tensor([0, 0, 1]),
        train_mask=torch.tensor([True, True, True]),
        n_classes=2,
        min_train_votes=3,
    )

    assert result.weighted_estimate == 1.0
    assert result.unweighted_estimate == 1.0 / 3.0
    assert result.effective_sample_size == 1.0
    assert result.per_ball[0]["weighted_estimate"] == 1.0


def test_confidence_weighted_estimate_matches_registered_formula():
    result = estimate_ball_noise_rate_loo(
        assignment=np.array([0, 0, 0, 0]),
        noisy_label=torch.tensor([0, 0, 0, 1]),
        train_mask=torch.tensor([True, True, True, True]),
        n_classes=2,
        min_train_votes=3,
    )

    assert math.isclose(result.weighted_estimate, 0.5)
    assert math.isclose(result.unweighted_estimate, 0.25)
    assert math.isclose(result.weight_sum, 2.0)


def test_low_vote_balls_are_excluded():
    result = estimate_ball_noise_rate_loo(
        assignment=np.array([0, 0, 1, 1, 1]),
        noisy_label=torch.tensor([0, 1, 1, 1, 1]),
        train_mask=torch.tensor([True, True, True, True, True]),
        n_classes=2,
        min_train_votes=3,
    )

    assert result.eligible_count == 3
    assert result.coverage == 3.0 / 5.0
    assert [row["ball_id"] for row in result.per_ball] == [1]


def test_non_train_label_changes_do_not_change_estimate():
    assignment = np.array([0, 0, 0, 1, 1])
    train_mask = torch.tensor([True, True, True, False, False])
    first = estimate_ball_noise_rate_loo(
        assignment=assignment,
        noisy_label=torch.tensor([0, 0, 1, 0, 0]),
        train_mask=train_mask,
        n_classes=2,
        min_train_votes=3,
    )
    second = estimate_ball_noise_rate_loo(
        assignment=assignment,
        noisy_label=torch.tensor([0, 0, 1, 1, 1]),
        train_mask=train_mask,
        n_classes=2,
        min_train_votes=3,
    )

    assert first.weighted_estimate == second.weighted_estimate
    assert first.unweighted_estimate == second.unweighted_estimate
    assert first.per_ball == second.per_ball


def test_rhohat_gate_uses_dataset_means_and_coverage():
    rows = []
    for dataset, error, coverage in [
        ("cora", 0.08, 0.8),
        ("citeseer", 0.09, 0.7),
        ("corafull", 0.12, 0.6),
    ]:
        for seed in range(5):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "weighted_abs_error": error,
                    "coverage": coverage,
                    "status": "ok",
                }
            )

    gate = evaluate_rhohat_gate(rows)

    assert gate["pass"] is True
    assert gate["datasets_with_error_at_most_0p10"] == 2
    assert gate["minimum_mean_coverage"] == 0.6


def test_rhohat_gate_fails_when_any_dataset_error_exceeds_0p15():
    rows = []
    for dataset, error in [("cora", 0.05), ("citeseer", 0.06), ("corafull", 0.16)]:
        for seed in range(5):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "weighted_abs_error": error,
                    "coverage": 0.9,
                    "status": "ok",
                }
            )

    assert evaluate_rhohat_gate(rows)["pass"] is False
