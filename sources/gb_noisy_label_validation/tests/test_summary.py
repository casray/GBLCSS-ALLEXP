import csv
import tempfile
from pathlib import Path

from noisy_label_gb_grace.evaluation.summary import summarize_gate_results


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_excludes_old_smoke_rows_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        formal = {
            "dataset": "Cora",
            "noise_type": "symmetric",
            "noise_rate": 0.2,
            "method": "gb",
            "alpha_max": 0.3,
            "split_seed": 0,
            "pretrain_seed": 0,
            "noise_seed": 0,
            "classifier_seed": 0,
            "best_val_acc": 0.7,
            "test_acc_at_best_val": 0.8,
        }
        smoke = {
            **formal,
            "method": "topk",
            "pretrain_seed": 39788,
            "test_acc_at_best_val": 0.1,
        }
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "run" / "result_rows.csv", [formal, smoke])

        summary_rows, paired_rows = summarize_gate_results(root)

    assert len(summary_rows) == 1
    assert summary_rows[0]["method"] == "gb"
    assert summary_rows[0]["n"] == 1
    assert summary_rows[0]["test_at_best_val_mean"] == 0.8
    assert paired_rows == []


def test_summary_keeps_split_protocols_separate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        planetoid = {
            "dataset": "Cora",
            "split_mode": "planetoid",
            "split_name": "split0",
            "split_seed": 0,
            "pretrain_seed": 0,
            "noise_seed": 0,
            "classifier_seed": 0,
            "noise_type": "symmetric",
            "noise_rate": 0.2,
            "method": "gb",
            "alpha_max": 0.3,
            "best_val_acc": 0.7,
            "test_acc_at_best_val": 0.7,
        }
        ratio = {
            **planetoid,
            "split_mode": "ratio",
            "split_name": "ratio_train0p1_val0p1_split0",
            "train_ratio": 0.1,
            "val_ratio": 0.1,
            "test_acc_at_best_val": 0.9,
        }
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "split0" / "result_rows.csv", [planetoid])
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "ratio" / "result_rows.csv", [ratio])

        summary_rows, _ = summarize_gate_results(root)

    assert len(summary_rows) == 2
    means = {row["split_name"]: row["test_at_best_val_mean"] for row in summary_rows}
    assert means["split0"] == 0.7
    assert means["ratio_train0p1_val0p1_split0"] == 0.9


def test_summary_includes_weighted_methods_and_paired_controls():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {
            "dataset": "Cora",
            "split_mode": "ratio",
            "split_name": "ratio_train0p2_val0p1_split0",
            "train_ratio": 0.2,
            "val_ratio": 0.1,
            "split_seed": 0,
            "pretrain_seed": 0,
            "noise_seed": 0,
            "classifier_seed": 0,
            "noise_type": "symmetric",
            "noise_rate": 0.4,
            "alpha_max": 0.0,
            "best_val_acc": 0.7,
        }
        rows = [
            {**base, "method": "ce", "test_acc_at_best_val": 0.70},
            {**base, "method": "gb_weighted_ce", "test_acc_at_best_val": 0.74, "weight_mean": 1.0},
            {**base, "method": "kmeans_weighted_ce", "test_acc_at_best_val": 0.72, "weight_mean": 1.0},
            {**base, "method": "topk_matched_weighted_ce", "test_acc_at_best_val": 0.71, "weight_mean": 1.0},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p4" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert "gb_weighted_ce" in methods
    assert "kmeans_weighted_ce" in methods
    weighted = [
        row for row in paired_rows
        if row["method"] == "gb_weighted_ce"
    ]
    baselines = {row["baseline"] for row in weighted}
    assert {"ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"} <= baselines
