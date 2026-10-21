from __future__ import annotations

from scripts.run_mlp48_cf_blagb import PHASE_A_METHODS, evaluate_phase_a_gate


DATASETS = ("cora", "citeseer", "corafull")


def _row(dataset: str, method: str, accuracy: float, brier: float, exact=True):
    return {
        "dataset": dataset,
        "method": method,
        "target_accuracy": accuracy,
        "brier": brier,
        "ncr": 0.2,
        "cdr": 0.05,
        "exact_budget": exact,
        "target_num_balls": 4,
        "min_num_balls": 4,
        "max_num_balls": 4,
        "status": "ok",
    }


def _passing_rows():
    rows = []
    for dataset in DATASETS:
        values = {
            "formal_loo_gb": (0.59, 0.52),
            "crossfit_formal_gb": (0.60, 0.50),
            "fixed_budget_geometry_only": (0.60, 0.49),
            "cf_blagb_shuffled_guidance": (0.60, 0.51),
            "cf_blagb": (0.61, 0.45),
            "cf_blagb_clean_oracle": (0.62, 0.44),
            "crossfit_adaptive_knn": (0.61, 0.46),
        }
        rows.extend(
            _row(dataset, method, *values[method]) for method in PHASE_A_METHODS
        )
    return rows


def _find(rows, dataset: str, method: str):
    return next(
        row
        for row in rows
        if row["dataset"] == dataset and row["method"] == method
    )


def test_phase_a_gate_passes_only_when_all_preregistered_conditions_hold():
    decision = evaluate_phase_a_gate(_passing_rows())

    assert decision["pass"] is True
    assert decision["reason"] == "pass"
    assert decision["brier_both_pass_count"] == 3
    assert decision["oracle_headroom_pass_count"] == 3
    assert decision["exact_budget_dataset_count"] == 3


def test_phase_a_gate_fails_on_one_large_accuracy_drop():
    rows = _passing_rows()
    _find(rows, "cora", "cf_blagb")["target_accuracy"] = 0.589

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "large_accuracy_drop"


def test_phase_a_gate_fails_when_real_does_not_beat_both_brier_controls():
    rows = _passing_rows()
    for dataset in ("cora", "citeseer"):
        _find(rows, dataset, "cf_blagb")["brier"] = 0.505

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "brier_controls_failed"


def test_phase_a_gate_rejects_missing_dataset_or_method():
    rows = _passing_rows()
    rows.remove(_find(rows, "corafull", "cf_blagb_clean_oracle"))

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "incomplete"


def test_phase_a_gate_rejects_inexact_budget():
    rows = _passing_rows()
    _find(rows, "citeseer", "cf_blagb")["exact_budget"] = False

    decision = evaluate_phase_a_gate(rows)

    assert decision["pass"] is False
    assert decision["reason"] == "budget_failed"

