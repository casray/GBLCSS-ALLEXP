import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.diagnostics.gated_nodes import (
    GATED_DIAGNOSTIC_FIELDS,
    compute_gated_node_diagnostics,
    summarize_rows,
)


def _toy_grouping():
    return {
        "group_ids": torch.tensor([0, 0, 0, 1]),
        "groups": [[0, 1, 2], [3]],
        "radii": torch.tensor([0.0, 0.0]),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }


def test_gated_mask_matches_conflict_gate_condition():
    clean = torch.tensor([1, 1, 1, 0])
    noisy = torch.tensor([0, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )

    row = compute_gated_node_diagnostics(
        clean,
        noisy,
        train_mask,
        _toy_grouping(),
        teacher_probs,
        num_classes=2,
        alpha_max=1.0,
        dataset="Toy",
        noise_rate=0.25,
        seed=0,
        min_train_per_group=2,
        conflict_noisy_support_max=0.35,
        conflict_target_conf_min=0.55,
    )

    assert row["gated_train_count"] == 1
    assert row["gated_noisy_count"] == 1
    assert row["gated_clean_count"] == 0
    assert row["qmix_noisy_support_mean"] < 0.35


def test_recomputed_ncr_matches_target_argmax_definition():
    clean = torch.tensor([1, 1, 1, 0])
    noisy = torch.tensor([0, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )

    row = compute_gated_node_diagnostics(
        clean,
        noisy,
        train_mask,
        _toy_grouping(),
        teacher_probs,
        num_classes=2,
        alpha_max=1.0,
        dataset="Toy",
        noise_rate=0.25,
        seed=0,
        min_train_per_group=2,
    )

    assert row["recomputed_target_argmax_ncr"] == 1.0
    assert row["gated_noisy_target_argmax_eq_clean_rate"] == 1.0
    assert row["current_reported_ncr"] == row["recomputed_target_argmax_ncr"]


def test_lost_by_alpha_detects_qmix_clean_but_target_stays_noisy():
    clean = torch.tensor([1, 1, 1, 0])
    noisy = torch.tensor([0, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )

    row = compute_gated_node_diagnostics(
        clean,
        noisy,
        train_mask,
        _toy_grouping(),
        teacher_probs,
        num_classes=2,
        alpha_max=0.2,
        dataset="Toy",
        noise_rate=0.25,
        seed=0,
        min_train_per_group=2,
    )

    assert row["possible_correction_rate_before_alpha"] == 1.0
    assert row["recomputed_target_argmax_ncr"] == 0.0
    assert row["lost_by_alpha_rate"] == 1.0


def test_required_gated_diagnostic_fields_exist():
    required = {
        "dataset",
        "noise_rate",
        "seed",
        "gated_noisy_ratio",
        "qmix_argmax_eq_clean_rate",
        "gated_noisy_qmix_argmax_eq_clean_rate",
        "target_argmax_change_rate",
        "gated_noisy_target_argmax_eq_clean_rate",
        "possible_correction_rate_before_alpha",
        "lost_by_alpha_rate",
        "teacher_gated_noisy_clean_acc",
        "gated_ball_clean_purity_mean",
        "current_reported_ncr",
        "recomputed_target_argmax_ncr",
    }
    assert required <= set(GATED_DIAGNOSTIC_FIELDS)


def test_summary_reports_mean_and_std_for_numeric_fields():
    rows = [
        {"dataset": "Cora", "noise_rate": 0.2, "gated_train_count": 2, "gated_noisy_ratio": 0.5},
        {"dataset": "Cora", "noise_rate": 0.2, "gated_train_count": 4, "gated_noisy_ratio": 1.0},
    ]

    summary = summarize_rows(rows)

    assert len(summary) == 1
    assert summary[0]["gated_train_count_mean"] == 3.0
    assert math.isclose(summary[0]["gated_train_count_std"], 1.0)
    assert summary[0]["gated_noisy_ratio_mean"] == 0.75
