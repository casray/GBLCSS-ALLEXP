from __future__ import annotations

import numpy as np
import pytest

from scripts.run_mlp51_multipartition_gb import (
    EXPECTED_METHODS,
    average_soft_targets,
    evaluate_phase_a_gate,
)


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
        "h_seed_count": 3 if "mp_" in method else 1,
        "target_simplex_max_error": 0.0,
    }


def _passing_rows():
    rows = []
    cells = [
        (dataset, rate)
        for dataset in ("cora", "citeseer", "corafull")
        for rate in (0.4, 0.6)
    ]
    for cell_id, (dataset, rate) in enumerate(cells):
        gb_brier = 0.30
        mp_brier = 0.28
        knn_brier = 0.29 if cell_id < 4 else 0.27
        rows.extend(
            [
                _row(dataset, rate, "single_h0_gb", brier=gb_brier, accuracy=0.60),
                _row(dataset, rate, "mp_gb_3h", brier=mp_brier, accuracy=0.60),
                _row(dataset, rate, "single_h0_knn", brier=0.31, accuracy=0.60),
                _row(dataset, rate, "mp_knn_3h", brier=knn_brier, accuracy=0.60),
                _row(dataset, rate, "mp_gb_3h_shuffled", brier=0.40, accuracy=0.50),
            ]
        )
    return rows


def test_average_soft_targets_is_exact_simplex_mean():
    first = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    second = np.asarray([[0.6, 0.4], [0.3, 0.7]], dtype=np.float64)
    third = np.asarray([[0.4, 0.6], [0.5, 0.5]], dtype=np.float64)

    actual = average_soft_targets([first, second, third])

    np.testing.assert_allclose(actual, (first + second + third) / 3.0)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


def test_average_soft_targets_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one"):
        average_soft_targets([])
    with pytest.raises(ValueError, match="same shape"):
        average_soft_targets([np.ones((2, 2)) / 2, np.ones((3, 2)) / 2])
    with pytest.raises(ValueError, match="simplex"):
        average_soft_targets([np.asarray([[0.8, 0.8]])])


def test_phase_a_gate_rejects_incomplete_grid():
    decision = evaluate_phase_a_gate([])

    assert decision["pass"] is False
    assert decision["reason"] == "incomplete"


def test_phase_a_gate_passes_at_frozen_count_thresholds():
    decision = evaluate_phase_a_gate(_passing_rows())

    assert set(EXPECTED_METHODS) == {
        "single_h0_gb",
        "mp_gb_3h",
        "single_h0_knn",
        "mp_knn_3h",
        "mp_gb_3h_shuffled",
    }
    assert decision["pass"] is True
    assert decision["reason"] == "pass"
    assert decision["brier_wins_vs_single_gb"] == 6
    assert decision["brier_wins_vs_mp_knn"] == 4
    assert decision["brier_wins_vs_shuffled"] == 6


def test_phase_a_gate_rejects_single_accuracy_drop_above_half_point():
    rows = _passing_rows()
    for row in rows:
        if (
            row["dataset"] == "cora"
            and row["noise_rate"] == 0.4
            and row["method"] == "mp_gb_3h"
        ):
            row["target_accuracy"] = 0.5949

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "accuracy_regression"

