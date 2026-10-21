from __future__ import annotations

import numpy as np

from scripts.run_mlp46_cf_nagb import (
    evaluate_phase_a_gate,
    target_quality_metrics,
)


def test_target_quality_metrics_match_project_ncr_cdr_direction():
    noisy = np.array([1, 1, 0, 1], dtype=np.int64)
    clean = np.array([0, 1, 0, 0], dtype=np.int64)
    train = np.arange(4, dtype=np.int64)
    targets = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )

    metrics = target_quality_metrics(targets, noisy, clean, train)

    assert metrics["target_accuracy"] == 0.75
    assert metrics["ncr"] == 1.0
    assert metrics["cdr"] == 0.5
    assert metrics["brier"] >= 0.0


def _row(dataset, method, accuracy, brier, ncr, cdr):
    return {
        "dataset": dataset,
        "method": method,
        "target_accuracy": accuracy,
        "brier": brier,
        "ncr": ncr,
        "cdr": cdr,
        "geometry_status": "ok",
    }


def test_phase_a_gate_requires_noisy_guidance_and_oracle_headroom():
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        rows.extend(
            [
                _row(dataset, "crossfit_formal_gb", 0.50, 0.70, 0.30, 0.20),
                _row(dataset, "cf_nagb_shuffled_guidance", 0.50, 0.70, 0.30, 0.20),
                _row(dataset, "cf_nagb", 0.52, 0.69, 0.31, 0.20),
                _row(dataset, "cf_nagb_clean_oracle", 0.55, 0.65, 0.35, 0.18),
            ]
        )

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is True
    assert decision["noisy_vs_formal_pass_count"] == 3
    assert decision["noisy_vs_shuffled_pass_count"] == 3
    assert decision["oracle_headroom_pass_count"] == 3


def test_phase_a_gate_stops_when_only_clean_oracle_improves():
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        rows.extend(
            [
                _row(dataset, "crossfit_formal_gb", 0.50, 0.70, 0.30, 0.20),
                _row(dataset, "cf_nagb_shuffled_guidance", 0.50, 0.70, 0.30, 0.20),
                _row(dataset, "cf_nagb", 0.50, 0.70, 0.30, 0.20),
                _row(dataset, "cf_nagb_clean_oracle", 0.55, 0.65, 0.35, 0.18),
            ]
        )

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "noisy_guidance_failed"

