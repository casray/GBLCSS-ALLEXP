from __future__ import annotations

import numpy as np
import torch

from scripts.run_mlp50_loo_min_vote_gate import (
    EXPECTED_METHODS,
    evaluate_phase_a_gate,
    support_diagnostics,
)
from utils.frozen_grace_targets import apply_ball_soft_targets


def _row(dataset, rate, method, *, brier, accuracy, coverage=0.9):
    return {
        "dataset": dataset,
        "noise_rate": rate,
        "method": method,
        "brier": brier,
        "target_accuracy": accuracy,
        "alpha_coverage": coverage,
        "status": "ok",
        "modified_val_count": 0,
        "modified_test_count": 0,
    }


def test_support_diagnostics_distinguishes_empty_and_low_vote_balls():
    assignment = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    train = np.asarray([0, 3, 4, 5, 6], dtype=np.int64)

    summary, bins = support_diagnostics(assignment, train)

    assert summary["num_balls"] == 3
    assert summary["zero_vote_ball_count"] == 1
    assert summary["zero_vote_node_count"] == 2
    assert summary["train_nodes_with_support_lt2"] == 1
    assert summary["train_nodes_with_support_lt5"] == 5
    assert sum(row["train_node_count"] for row in bins) == train.size


def test_loo_smin_counts_other_votes_not_total_votes():
    noisy = torch.tensor([0, 0, 0, 0, 0, 0])
    assignment = np.zeros(6, dtype=np.int64)
    train = np.arange(6)

    active, alpha, _ = apply_ball_soft_targets(
        noisy_label=noisy,
        assignment=assignment,
        train_mask=train,
        n_classes=2,
        min_train_per_ball=5,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )
    inactive, inactive_alpha, _ = apply_ball_soft_targets(
        noisy_label=noisy[:5],
        assignment=assignment[:5],
        train_mask=train[:5],
        n_classes=2,
        min_train_per_ball=5,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )

    assert torch.all(alpha > 0)
    assert torch.all(inactive_alpha == 0)
    assert active.shape[0] == 6
    assert inactive.shape[0] == 5


def test_phase_a_gate_passes_only_when_all_frozen_conditions_hold():
    rows = []
    datasets = ("cora", "citeseer", "corafull")
    for dataset in datasets:
        for rate in (0.4, 0.6):
            rows.extend(
                [
                    _row(dataset, rate, "real_s2", brier=0.30, accuracy=0.60),
                    _row(dataset, rate, "real_s5", brier=0.29, accuracy=0.61),
                    _row(dataset, rate, "real_s10", brier=0.31, accuracy=0.59),
                    _row(dataset, rate, "shuffled_s2", brier=0.40, accuracy=0.50),
                    _row(dataset, rate, "shuffled_s5", brier=0.39, accuracy=0.51),
                    _row(dataset, rate, "shuffled_s10", brier=0.41, accuracy=0.49),
                ]
            )

    decision = evaluate_phase_a_gate(rows)

    assert set(EXPECTED_METHODS) == {
        "real_s2",
        "real_s5",
        "real_s10",
        "shuffled_s2",
        "shuffled_s5",
        "shuffled_s10",
    }
    assert decision["pass"] is True
    assert decision["reason"] == "pass"


def test_phase_a_gate_rejects_incomplete_or_weak_candidates():
    incomplete = evaluate_phase_a_gate([])
    assert incomplete["pass"] is False
    assert incomplete["reason"] == "incomplete"

    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for rate in (0.4, 0.6):
            for method in EXPECTED_METHODS:
                rows.append(
                    _row(
                        dataset,
                        rate,
                        method,
                        brier=0.30,
                        accuracy=0.60,
                    )
                )
    weak = evaluate_phase_a_gate(rows)
    assert weak["pass"] is False
    assert weak["reason"] == "brier_failed"
