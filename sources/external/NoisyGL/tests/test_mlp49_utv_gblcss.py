from __future__ import annotations

import importlib


def _row(dataset: str, noise_rate: float, method: str, accuracy: float, brier: float) -> dict:
    return {
        "dataset": dataset,
        "noise_rate": noise_rate,
        "method": method,
        "target_accuracy": accuracy,
        "brier": brier,
        "ncr": 0.1,
        "cdr": 0.01,
        "verified_correction_coverage": 0.5,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "status": "ok",
    }


def test_phase_a_gate_passes_only_complete_frozen_grid() -> None:
    module = importlib.import_module("scripts.run_mlp49_utv_gblcss")
    rows = []
    for dataset in module.EXPECTED_DATASETS:
        for rate in module.NOISE_RATES:
            rows.extend(
                [
                    _row(dataset, rate, "formal_loo_gb", 0.60, 0.50),
                    _row(dataset, rate, "utv_gb", 0.61, 0.45),
                    _row(dataset, rate, "utv_gb_shuffled", 0.59, 0.52),
                    _row(dataset, rate, "nrgnn_teacher_target", 0.65, 0.40),
                    _row(dataset, rate, "teacher_weighted_noisy", 0.60, 0.80),
                    _row(dataset, rate, "clean_oracle_verifier", 0.61, 0.46),
                ]
            )

    decision = module.evaluate_phase_a_gate(rows)

    assert decision["pass"] is True
    assert decision["brier_both_pass_count"] == 6
    assert decision["oracle_headroom_pass_count"] == 6
    assert decision["mean_accuracy_delta_vs_formal"] > 0.005


def test_phase_a_gate_rejects_missing_cell() -> None:
    module = importlib.import_module("scripts.run_mlp49_utv_gblcss")
    rows = [
        _row("cora", 0.4, method, 0.6, 0.5)
        for method in module.PHASE_A_METHODS
    ]

    decision = module.evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "incomplete"


def test_teacher_cache_metadata_must_match_protocol() -> None:
    module = importlib.import_module("scripts.run_mlp49_utv_gblcss")
    expected = {
        "dataset": "cora",
        "noise_rate": 0.4,
        "seed": 0,
        "split_hash": "split-a",
        "noise_hash": "noise-a",
        "mc_samples": 8,
    }

    module.validate_teacher_cache_metadata(expected, dict(expected))
    wrong = dict(expected)
    wrong["noise_hash"] = "noise-b"
    try:
        module.validate_teacher_cache_metadata(wrong, expected)
    except ValueError as exc:
        assert "noise_hash" in str(exc)
    else:
        raise AssertionError("mismatched teacher cache was accepted")

