import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.weighting import build_reliability_weights


def _toy_grouping():
    h = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.2, 0.8],
    ])
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.tensor([0.2, 0.2]),
        "node_distances": torch.tensor([0.01, 0.02, 0.03, 0.01, 0.02, 0.03]),
        "diagnostics": {"ball_count": 2},
    }
    return h, grouping


def test_reliability_weights_have_train_mean_near_one():
    h, grouping = _toy_grouping()
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True, True, True])
    weights, stats = build_reliability_weights(
        "gb_weighted_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        gb_grouping=grouping,
        min_train_per_region=3,
    )
    assert weights.shape == labels.shape
    assert torch.isclose(weights[train_mask].mean(), torch.tensor(1.0), atol=1e-6)
    assert "weight_std" in stats
    assert stats["coverage_gate_mean"] > 0.0


def test_leave_one_out_support_lowers_local_label_conflict():
    h = torch.stack([
        torch.tensor([1.0, float(i) / 100.0])
        for i in range(11)
    ])
    grouping = {
        "group_ids": torch.zeros(11, dtype=torch.long),
        "groups": [list(range(11))],
        "radii": torch.tensor([0.1]),
        "node_distances": torch.zeros(11),
        "diagnostics": {"ball_count": 1},
    }
    labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    train_mask = torch.ones(11, dtype=torch.bool)
    weights, stats = build_reliability_weights(
        "gb_weighted_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        gb_grouping=grouping,
        min_train_per_region=2,
    )
    assert weights[10] < weights[5]
    assert stats["support_gate_mean"] < 1.0


def test_kmeans_weighted_ce_matches_gb_group_count():
    h, grouping = _toy_grouping()
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True, True, True])
    _, stats = build_reliability_weights(
        "kmeans_weighted_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        gb_grouping=grouping,
        seed=0,
    )
    assert stats["ball_count"] == grouping["diagnostics"]["ball_count"]


def test_gagb_weighted_ce_uses_supplied_grouping():
    h, grouping = _toy_grouping()
    grouping["diagnostics"]["grouping_kind"] = "graph_aware"
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True, True, True])
    _, stats = build_reliability_weights(
        "gagb_weighted_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        gb_grouping=grouping,
        seed=0,
    )
    assert stats["grouping_kind"] == "graph_aware"


def test_topk_matched_weighted_ce_uses_matched_group_size():
    h, grouping = _toy_grouping()
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    train_mask = torch.tensor([True, True, True, True, True, True])
    _, stats = build_reliability_weights(
        "topk_matched_weighted_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        gb_grouping=grouping,
        seed=0,
    )
    assert stats["topk"] == 2
    assert stats["topk_match_rule"] == "mean_eligible_gb_train_nodes_minus_1"
