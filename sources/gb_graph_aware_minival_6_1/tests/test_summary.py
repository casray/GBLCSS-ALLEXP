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


def test_summary_includes_gagb_fair_controls_and_pairings():
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
            "noise_rate": 0.2,
            "alpha_max": 0.3,
            "best_val_acc": 0.7,
        }
        rows = [
            {**base, "method": "gagb", "test_acc_at_best_val": 0.80},
            {**base, "method": "gb", "test_acc_at_best_val": 0.79},
            {**base, "method": "emb_only_gagb_local", "test_acc_at_best_val": 0.78},
            {**base, "method": "emb_only_gagb_global", "test_acc_at_best_val": 0.77},
            {**base, "method": "khop_only", "test_acc_at_best_val": 0.76},
            {**base, "method": "kmeans_gagb_matched", "test_acc_at_best_val": 0.75},
            {**base, "method": "topk_gagb_matched", "test_acc_at_best_val": 0.74},
            {**base, "method": "gagb_shuffled_alpha_matched", "test_acc_at_best_val": 0.73},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert {
        "emb_only_gagb_local",
        "emb_only_gagb_global",
        "kmeans_gagb_matched",
        "topk_gagb_matched",
        "gagb_shuffled_alpha_matched",
    } <= methods
    gagb_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "gagb"
    }
    assert {
        "gb",
        "emb_only_gagb_local",
        "emb_only_gagb_global",
        "khop_only",
        "kmeans_gagb_matched",
        "topk_gagb_matched",
        "gagb_shuffled_alpha_matched",
    } <= gagb_pairs


def test_summary_includes_v2_target_methods_and_pairings():
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
            "noise_rate": 0.2,
            "alpha_max": 0.3,
            "best_val_acc": 0.7,
            "noise_correction_rate": 0.1,
            "clean_damage_rate": 0.02,
            "target_changed_mask_mismatch_rate": 0.0,
        }
        rows = [
            {**base, "method": "gagb_qb_conflict_gate", "test_acc_at_best_val": 0.80},
            {**base, "method": "gagb_qb_pred_mix", "test_acc_at_best_val": 0.79},
            {**base, "method": "gagb_qb_teacher_prior", "test_acc_at_best_val": 0.78},
            {**base, "method": "gb", "test_acc_at_best_val": 0.77},
            {**base, "method": "gagb", "test_acc_at_best_val": 0.76},
            {**base, "method": "khop_only", "test_acc_at_best_val": 0.75},
            {**base, "method": "emb_only_gagb_local", "test_acc_at_best_val": 0.74},
            {**base, "method": "emb_only_gagb_global", "test_acc_at_best_val": 0.73},
            {**base, "method": "kmeans_gagb_matched", "test_acc_at_best_val": 0.72},
            {**base, "method": "topk_gagb_train_matched", "test_acc_at_best_val": 0.71},
            {**base, "method": "topk_gagb_allnode_matched", "test_acc_at_best_val": 0.70},
            {**base, "method": "gagb_shuffled_alpha_matched", "test_acc_at_best_val": 0.69},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert {
        "gagb_qb_pred_mix",
        "gagb_qb_teacher_prior",
        "gagb_qb_conflict_gate",
        "topk_gagb_train_matched",
        "topk_gagb_allnode_matched",
    } <= methods
    conflict_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "gagb_qb_conflict_gate"
    }
    assert {
        "gb",
        "gagb",
        "khop_only",
        "emb_only_gagb_local",
        "emb_only_gagb_global",
        "kmeans_gagb_matched",
        "topk_gagb_train_matched",
        "topk_gagb_allnode_matched",
        "gagb_shuffled_alpha_matched",
    } <= conflict_pairs


def test_summary_includes_target_repair_methods_and_alpha03_pairings():
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
            "noise_rate": 0.2,
            "best_val_acc": 0.7,
            "noise_correction_rate": 0.1,
            "clean_damage_rate": 0.02,
            "possible_correction_before_alpha": 0.8,
            "lost_by_alpha_rate": 0.0,
            "target_argmax_change_rate": 0.2,
        }
        rows = [
            {**base, "method": "gb", "alpha_max": 0.3, "test_acc_at_best_val": 0.77},
            {**base, "method": "gagb", "alpha_max": 0.3, "test_acc_at_best_val": 0.76},
            {**base, "method": "khop_only", "alpha_max": 0.3, "test_acc_at_best_val": 0.75},
            {**base, "method": "gagb_qb_conflict_gate_alpha03", "alpha_max": 0.3, "test_acc_at_best_val": 0.78},
            {**base, "method": "gagb_qb_conflict_gate_alpha07", "alpha_max": 0.7, "test_acc_at_best_val": 0.80},
            {**base, "method": "gagb_qb_conflict_gate_replace_qmix", "alpha_max": 1.0, "test_acc_at_best_val": 0.79},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert {
        "gagb_qb_conflict_gate_alpha03",
        "gagb_qb_conflict_gate_alpha07",
        "gagb_qb_conflict_gate_replace_qmix",
    } <= methods
    repair_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "gagb_qb_conflict_gate_alpha07"
    }
    assert {"gagb_qb_conflict_gate_alpha03", "gb", "gagb", "khop_only"} <= repair_pairs


def test_summary_pairs_v21_repair_controls_against_alpha03():
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
            "noise_rate": 0.2,
            "best_val_acc": 0.7,
            "noise_correction_rate": 0.1,
            "clean_damage_rate": 0.02,
        }
        rows = [
            {**base, "method": "gagb_qb_conflict_gate_alpha03", "alpha_max": 0.3, "test_acc_at_best_val": 0.78},
            {**base, "method": "gagb_shuffled_alpha09_matched", "alpha_max": 0.9, "test_acc_at_best_val": 0.79},
            {**base, "method": "kmeans_gagb_matched_replace", "alpha_max": 1.0, "test_acc_at_best_val": 0.80},
            {**base, "method": "topk_gagb_train_matched_alpha09", "alpha_max": 0.9, "test_acc_at_best_val": 0.81},
            {**base, "method": "khop_only_replace", "alpha_max": 1.0, "test_acc_at_best_val": 0.82},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p2" / "ratio" / "result_rows.csv", rows)

        _, paired_rows = summarize_gate_results(root)

    alpha03_paired_methods = {
        row["method"]
        for row in paired_rows
        if row["baseline"] == "gagb_qb_conflict_gate_alpha03"
    }
    assert {
        "gagb_shuffled_alpha09_matched",
        "kmeans_gagb_matched_replace",
        "topk_gagb_train_matched_alpha09",
        "khop_only_replace",
    } <= alpha03_paired_methods


def test_summary_includes_grace_first_gagb_methods_and_pairings():
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
            "best_val_acc": 0.7,
            "clean_purity": 0.8,
            "noise_detection_auroc": 0.65,
        }
        rows = [
            {**base, "method": "gb", "alpha_max": 0.3, "test_acc_at_best_val": 0.72},
            {**base, "method": "gagb", "alpha_max": 0.3, "test_acc_at_best_val": 0.73},
            {**base, "method": "khop_only", "alpha_max": 0.3, "test_acc_at_best_val": 0.71},
            {**base, "method": "grace_gagb", "alpha_max": 0.3, "test_acc_at_best_val": 0.76},
            {**base, "method": "grace_gagb_emb_only", "alpha_max": 0.3, "test_acc_at_best_val": 0.74},
            {**base, "method": "grace_gagb_topo_only", "alpha_max": 0.3, "test_acc_at_best_val": 0.70},
            {**base, "method": "kmeans_grace_gagb_matched", "alpha_max": 0.3, "test_acc_at_best_val": 0.69},
            {**base, "method": "topk_grace_gagb_train_matched", "alpha_max": 0.3, "test_acc_at_best_val": 0.68},
            {**base, "method": "grace_gagb_shuffled_alpha_matched", "alpha_max": 0.3, "test_acc_at_best_val": 0.67},
            {**base, "method": "grace_gagb_qb_conflict_gate_alpha07", "alpha_max": 0.7, "test_acc_at_best_val": 0.77},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p4" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert {
        "grace_gagb",
        "grace_gagb_emb_only",
        "grace_gagb_topo_only",
        "kmeans_grace_gagb_matched",
        "topk_grace_gagb_train_matched",
        "grace_gagb_shuffled_alpha_matched",
        "grace_gagb_qb_conflict_gate_alpha07",
    } <= methods
    grace_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "grace_gagb"
    }
    assert {
        "gb",
        "gagb",
        "khop_only",
        "grace_gagb_emb_only",
        "grace_gagb_topo_only",
        "kmeans_grace_gagb_matched",
        "topk_grace_gagb_train_matched",
        "grace_gagb_shuffled_alpha_matched",
    } <= grace_pairs
    repair_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "grace_gagb_qb_conflict_gate_alpha07"
    }
    assert {"gb", "gagb", "khop_only", "grace_gagb"} <= repair_pairs


def test_summary_includes_quality_filter_methods_and_fields():
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
            "best_val_acc": 0.7,
            "quality_threshold": 0.5,
            "quality_selected_ball_count": 2,
            "quality_rejected_ball_count": 1,
            "quality_selected_gate_count": 4,
            "quality_rejected_gate_count": 2,
            "quality_selected_mean": 0.8,
            "quality_rejected_mean": 0.3,
            "quality_sem_mean": 0.9,
            "quality_topo_mean": 0.7,
            "quality_train_mean": 1.0,
            "quality_entropy_mean": 0.6,
        }
        rows = [
            {**base, "method": "gb", "alpha_max": 0.3, "test_acc_at_best_val": 0.735},
            {**base, "method": "grace_gagb_qb_conflict_gate_alpha07", "alpha_max": 0.7, "test_acc_at_best_val": 0.76},
            {**base, "method": "grace_gagb_qfilter_alpha07_q50", "alpha_max": 0.7, "test_acc_at_best_val": 0.77},
            {**base, "method": "grace_gagb_qfilter_alpha07_q70", "alpha_max": 0.7, "test_acc_at_best_val": 0.75},
            {**base, "method": "grace_gagb_qweight_alpha07", "alpha_max": 0.7, "test_acc_at_best_val": 0.78},
            {**base, "method": "grace_gagb_qsafe_boundary_alpha09_q50", "alpha_max": 0.9, "test_acc_at_best_val": 0.79},
            {**base, "method": "grace_gagb_qsafe_boundary_alpha09_q70", "alpha_max": 0.9, "test_acc_at_best_val": 0.785},
            {**base, "method": "grace_gagb_qsafe_class_capped_alpha09_q50", "alpha_max": 0.9, "test_acc_at_best_val": 0.775},
            {**base, "method": "gagb_qsafe_boundary_alpha09_q50", "alpha_max": 0.9, "test_acc_at_best_val": 0.765},
            {**base, "method": "khop_only_qsafe_boundary_alpha09_q50", "alpha_max": 0.9, "test_acc_at_best_val": 0.755},
            {**base, "method": "grace_gagb_shuffled_qsafe_boundary_alpha09_q50", "alpha_max": 0.9, "test_acc_at_best_val": 0.745},
            {**base, "method": "gagb_qfilter_alpha07_q50", "alpha_max": 0.7, "test_acc_at_best_val": 0.74},
            {**base, "method": "khop_only_qfilter_alpha07_q50", "alpha_max": 0.7, "test_acc_at_best_val": 0.73},
            {**base, "method": "grace_gagb_shuffled_qfilter_alpha07_q50", "alpha_max": 0.7, "test_acc_at_best_val": 0.72},
        ]
        _write_rows(root / "gate_runs" / "Cora" / "symmetric_0p4" / "ratio" / "result_rows.csv", rows)

        summary_rows, paired_rows = summarize_gate_results(root)

    methods = {row["method"] for row in summary_rows}
    assert {
        "grace_gagb_qfilter_alpha07_q50",
        "grace_gagb_qfilter_alpha07_q70",
        "grace_gagb_qweight_alpha07",
        "grace_gagb_qsafe_boundary_alpha09_q50",
        "grace_gagb_qsafe_boundary_alpha09_q70",
        "grace_gagb_qsafe_class_capped_alpha09_q50",
        "gagb_qsafe_boundary_alpha09_q50",
        "khop_only_qsafe_boundary_alpha09_q50",
        "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
        "gagb_qfilter_alpha07_q50",
        "khop_only_qfilter_alpha07_q50",
        "grace_gagb_shuffled_qfilter_alpha07_q50",
    } <= methods
    qfilter = next(row for row in summary_rows if row["method"] == "grace_gagb_qfilter_alpha07_q50")
    assert qfilter["quality_threshold_mean"] == 0.5
    assert qfilter["quality_selected_ball_count_mean"] == 2.0
    assert qfilter["quality_selected_mean_mean"] == 0.8
    qfilter_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "grace_gagb_qfilter_alpha07_q50"
    }
    assert {
        "grace_gagb_qb_conflict_gate_alpha07",
        "gagb_qfilter_alpha07_q50",
        "khop_only_qfilter_alpha07_q50",
        "grace_gagb_shuffled_qfilter_alpha07_q50",
    } <= qfilter_pairs
    qsafe_pairs = {
        row["baseline"]
        for row in paired_rows
        if row["method"] == "grace_gagb_qsafe_boundary_alpha09_q50"
    }
    assert {
        "gb",
        "grace_gagb_qb_conflict_gate_alpha07",
        "grace_gagb_qfilter_alpha07_q70",
        "gagb_qsafe_boundary_alpha09_q50",
        "khop_only_qsafe_boundary_alpha09_q50",
        "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
    } <= qsafe_pairs
