import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls import build_granular_balls
from noisy_label_gb_grace.smoothing import build_method_targets, build_topk_group_size, kmeans_groups


def test_small_ball_fallback_disables_smoothing():
    h = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "groups": [[0, 1], [2, 3], [4, 5]],
        "radii": torch.ones(3),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 3},
    }
    targets, stats = build_method_targets(
        "gb",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=grouping,
        min_train_per_ball=3,
    )
    assert stats["effective_smoothing_coverage"] == 0.0
    assert torch.allclose(targets[0], torch.tensor([1.0, 0.0]))


def test_coverage_aware_ce_uses_low_confidence_for_sparse_groups():
    h = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "groups": [[0, 1], [2, 3], [4, 5]],
        "radii": torch.ones(3),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 3},
    }
    targets, stats = build_method_targets(
        "gb_ca_ce",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.6,
        gb_grouping=grouping,
        min_train_per_ball=3,
    )
    assert stats["local_smoothing_variant"] == "coverage_aware_ce"
    assert 0.0 <= stats["alpha_max_actual"] < 0.6
    assert targets[0, 0] > targets[0, 1]


def test_coverage_aware_global_falls_back_to_global_distribution():
    h = torch.eye(5, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 1])
    train_mask = torch.tensor([True, True, True, True, False])
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1, 1]),
        "groups": [[0, 1], [2, 3, 4]],
        "radii": torch.ones(2),
        "node_distances": torch.zeros(5),
        "diagnostics": {"ball_count": 2},
    }
    targets, stats = build_method_targets(
        "gb_ca_global",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.4,
        gb_grouping=grouping,
        min_train_per_ball=3,
        prior_gamma=1.0,
    )
    assert stats["local_smoothing_variant"] == "coverage_aware_global"
    assert stats["prior_gamma"] == 1.0
    assert stats["effective_smoothing_coverage"] > 0.0
    assert not torch.allclose(targets[0], torch.tensor([1.0, 0.0]))


def test_kmeans_matches_gb_group_count():
    h = torch.randn(20, 4)
    grouping = build_granular_balls(h, min_ball_size=3)
    kmeans = kmeans_groups(h, grouping["diagnostics"]["ball_count"], seed=0)
    assert kmeans["diagnostics"]["ball_count"] == grouping["diagnostics"]["ball_count"]


def test_topk_matches_non_singleton_average_minus_one():
    assert build_topk_group_size([[0, 1, 2], [3], [4, 5]]) == 2
