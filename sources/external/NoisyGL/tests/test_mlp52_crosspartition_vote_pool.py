from __future__ import annotations

import numpy as np

from scripts.run_mlp52_crosspartition_vote_pool import (
    evaluate_phase_a_gate,
    pooled_ball_loo_counts,
)


def test_pooled_ball_counts_exclude_self_in_every_partition():
    noisy = np.asarray([0, 1, 1, 0], dtype=np.int64)
    train = np.asarray([0, 1, 2, 3], dtype=np.int64)
    assignments = [
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.asarray([0, 1, 1, 0], dtype=np.int64),
    ]

    counts = pooled_ball_loo_counts(assignments, noisy, train, n_classes=2)

    # Node 0 sees labels 1, 1, and 0 from nodes 1, 2, and 3; never its own label.
    np.testing.assert_array_equal(counts[0], np.asarray([1.0, 2.0]))
    assert np.all(counts.sum(axis=1) == 3.0)


def _row(dataset, rate, method, *, brier, accuracy):
    return {
        "dataset": dataset,
        "noise_rate": rate,
        "method": method,
        "brier": brier,
        "target_accuracy": accuracy,
        "status": "ok",
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}",
        "noise_hash": f"noise-{dataset}-{rate}",
        "target_simplex_max_error": 0.0,
    }


def test_phase_a_gate_requires_pareto_wins_and_accuracy_recovery():
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for rate in (0.4, 0.6):
            rows.extend(
                [
                    _row(dataset, rate, "single_h0_gb", brier=0.32, accuracy=0.60),
                    _row(dataset, rate, "mean_target_gb_3h", brier=0.30, accuracy=0.59),
                    _row(dataset, rate, "pooled_gb_3h", brier=0.28, accuracy=0.61),
                    _row(dataset, rate, "pooled_knn_3h", brier=0.29, accuracy=0.61),
                    _row(dataset, rate, "pooled_gb_3h_shuffled", brier=0.40, accuracy=0.50),
                ]
            )

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is True
    assert decision["knn_pareto_wins"] == 6
    assert decision["accuracy_recovery_cells"] == 6


def test_phase_a_gate_rejects_incomplete_grid():
    decision = evaluate_phase_a_gate([])
    assert decision["pass"] is False
    assert decision["reason"] == "incomplete"

