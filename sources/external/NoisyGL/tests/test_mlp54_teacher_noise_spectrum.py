from __future__ import annotations

import importlib

import pytest


def _raw_row(summary, dataset, noise_type, rate, method, seed):
    row = {
        "dataset": dataset,
        "method": method,
        "seed": str(seed),
        "noise_type": noise_type,
        "noise_rate": str(rate),
        "noise_scope": "train",
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{dataset}-{noise_type}-{rate}-{seed}",
        "modified_count": "10",
        "modified_train_count": "10",
        "modified_val_count": "0",
        "modified_test_count": "0",
        "status": "ok",
        "test": "0.6",
        "valid": "0.5",
        "teacher_best_test": "0.7",
        "teacher_best_val": "0.6",
        "wall_time_seconds": "1.0",
        "knn_exclude_self": "1" if method == summary.KNN else "0",
    }
    if method == summary.NRGNN:
        row["student_inference_uses_graph"] = "0"
        row["teacher_probability_row_sum_max_error"] = "1e-7"
    return row


def test_expected_methods_are_registered() -> None:
    summary = importlib.import_module("scripts.summarize_mlp54_teacher_noise_spectrum")
    runner = importlib.import_module("scripts.run_standard_downstream")
    for method in summary.METHODS:
        assert method in runner.PREDICTOR_MODULES


def test_complete_hash_matched_grid_is_required() -> None:
    summary = importlib.import_module("scripts.summarize_mlp54_teacher_noise_spectrum")
    rows = []
    for noise_type, rate in summary.SETTINGS:
        for dataset in summary.DATASETS:
            for method in summary.METHODS:
                for seed in summary.SEEDS:
                    raw = _raw_row(summary, dataset, noise_type, rate, method, seed)
                    rows.append(summary.normalize(raw, source=summary.Path("test.csv")))

    completeness, audits = summary.audit_grid(rows)
    assert len(rows) == 225
    assert len(completeness) == 3 * 3 * 5
    assert len(audits) == 3 * 3 * 5
    assert all(row["complete"] == 1 for row in completeness)
    assert all(row["passed"] == 1 for row in audits)
    assert len(summary.paired_deltas(rows)) == 3 * 3 * 4 * 5

    rows[-1]["noise_hash"] = "mismatch"
    with pytest.raises(ValueError, match="protocol mismatch"):
        summary.audit_grid(rows)


def test_validation_rejects_leakage_and_self_including_knn() -> None:
    summary = importlib.import_module("scripts.summarize_mlp54_teacher_noise_spectrum")

    row = _raw_row(summary, "cora", "pair", 0.4, summary.BALL, 0)
    row["modified_val_count"] = "1"
    with pytest.raises(ValueError, match="validation or test"):
        summary.validate_row(row)

    row = _raw_row(summary, "cora", "pair", 0.4, summary.KNN, 0)
    row["knn_exclude_self"] = "0"
    with pytest.raises(ValueError, match="exclude"):
        summary.validate_row(row)


def test_nrgnn_student_must_be_x_only() -> None:
    summary = importlib.import_module("scripts.summarize_mlp54_teacher_noise_spectrum")
    row = _raw_row(summary, "cora", "instance", 0.4, summary.NRGNN, 0)
    row["student_inference_uses_graph"] = "1"
    with pytest.raises(ValueError, match="X-only"):
        summary.validate_row(row)


def test_full_spectrum_rejects_xh_and_audits_all_five_settings() -> None:
    summary = importlib.import_module("scripts.summarize_mlp54_teacher_noise_spectrum")
    assert summary.FULL_SETTINGS == (
        ("uniform", 0.2),
        ("uniform", 0.4),
        ("uniform", 0.6),
        ("pair", 0.4),
        ("instance", 0.4),
    )

    reference = _raw_row(summary, "cora", "uniform", 0.4, summary.GLNN, 0)
    normalized = summary.normalize_reference(reference, summary.Path("reference.csv"))
    assert normalized["student_inference_input"] == "X-only"

    reference["method"] = "mlp_distill_gcn_xh"
    with pytest.raises(ValueError, match="unexpected reference method"):
        summary.normalize_reference(reference, summary.Path("reference.csv"))

    nrgnn = {
        "dataset": "cora",
        "noise_type": "uniform",
        "noise_rate": "0.4",
        "noise_scope": "train",
        "seed": "0",
        "method": summary.NRGNN,
        "student_test": "0.7",
        "student_valid": "0.6",
        "teacher_test": "0.72",
        "teacher_valid": "0.65",
        "split_hash": "split-cora-0",
        "noise_hash": "noise-cora-uniform-0.4-0",
        "student_inference_uses_graph": "0",
    }
    assert summary.normalize_nrgnn_reference(
        nrgnn, summary.Path("nrgnn.csv")
    )["method"] == summary.NRGNN
    nrgnn["method"] = "mlp_distill_nrgnn_xh"
    with pytest.raises(ValueError, match="unexpected NRGNN reference method"):
        summary.normalize_nrgnn_reference(nrgnn, summary.Path("nrgnn.csv"))
