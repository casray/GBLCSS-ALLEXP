import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls import build_granular_balls
from noisy_label_gb_grace.smoothing import build_method_targets, build_topk_group_size, compute_group_quality_scores, kmeans_groups
from noisy_label_gb_grace.smoothing.teacher import train_noisy_linear_teacher
from noisy_label_gb_grace.utils import one_hot


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


def test_gagb_method_uses_supplied_graph_aware_grouping():
    h = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.tensor([True, True, True, True, True, True])
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=grouping,
        min_train_per_ball=3,
    )

    assert stats["grouping_kind"] == "graph_aware"
    assert stats["effective_smoothing_coverage"] == 1.0
    assert torch.allclose(targets[0], torch.tensor([1.0, 0.0]))


def test_kmeans_gagb_matched_uses_supplied_group_count():
    h = torch.randn(12, 3)
    labels = torch.tensor([0, 1] * 6)
    train_mask = torch.ones(12, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]),
        "groups": [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11]],
        "radii": torch.ones(5),
        "node_distances": torch.zeros(12),
        "diagnostics": {"ball_count": 5, "grouping_kind": "graph_aware"},
    }

    _, stats = build_method_targets(
        "kmeans_gagb_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
        seed=3,
    )

    assert stats["ball_count"] == 5
    assert stats["matched_grouping_kind"] == "graph_aware"
    assert stats["match_source"] == "gagb"


def test_topk_gagb_matched_uses_gagb_train_group_size_rule():
    h = torch.eye(8, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    train_mask = torch.tensor([True, True, True, True, True, False, True, False])
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
        "groups": [[0, 1, 2], [3, 4, 5], [6, 7]],
        "radii": torch.ones(3),
        "node_distances": torch.zeros(8),
        "diagnostics": {"ball_count": 3, "grouping_kind": "graph_aware"},
    }

    _, stats = build_method_targets(
        "topk_gagb_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
    )

    assert stats["topk_match_rule"] == "mean_eligible_gagb_train_nodes_minus_1"
    assert stats["topk"] == 2
    assert stats["match_source"] == "gagb"


def test_alpha_override_preserves_gagb_alpha_for_shuffled_control():
    h = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.ones(6, dtype=torch.bool)
    source = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }
    shuffled = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 3, 4], [1, 2, 5]],
        "radii": torch.ones(2) * 10.0,
        "node_distances": torch.ones(6) * 10.0,
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware_shuffled"},
    }

    _, source_stats = build_method_targets(
        "gagb",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=source,
        min_train_per_ball=3,
    )
    targets, shuffled_stats = build_method_targets(
        "gagb_shuffled_alpha_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=shuffled,
        min_train_per_ball=3,
        alpha_override=source_stats["node_alpha"],
        alpha_match_reference=source_stats,
    )

    assert shuffled_stats["alpha_matched_source"] == "gagb"
    assert shuffled_stats["alpha_match_max_abs_diff"] == 0.0
    assert shuffled_stats["coverage_match_abs_diff"] == 0.0
    assert shuffled_stats["effective_smoothing_coverage"] == source_stats["effective_smoothing_coverage"]
    assert torch.allclose(torch.tensor(shuffled_stats["node_alpha"]), torch.tensor(source_stats["node_alpha"]))
    assert targets[0, 1] > 0.0


def test_noisy_teacher_returns_probabilities_without_clean_label_inputs():
    signature = inspect.signature(train_noisy_linear_teacher)
    assert "clean_labels" not in signature.parameters
    assert "val_mask" not in signature.parameters
    assert "test_mask" not in signature.parameters

    h = torch.eye(6, dtype=torch.float32)
    noisy_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])

    probs, stats = train_noisy_linear_teacher(
        h,
        noisy_labels,
        train_mask,
        num_classes=2,
        seed=0,
        epochs=3,
        learning_rate=0.05,
        weight_decay=0.0,
        temperature=1.0,
    )

    assert probs.shape == (6, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(6), atol=1e-5)
    assert stats["teacher_epochs"] == 3
    assert stats["teacher_train_nodes"] == 4


def test_gagb_qb_pred_mix_uses_noisy_labels_and_teacher_group_mean():
    h = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    train_mask = torch.tensor([True, True, False, False])
    teacher_probs = torch.tensor(
        [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 0]),
        "groups": [[0, 1, 2, 3]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_pred_mix",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=1.0,
        gb_grouping=grouping,
        min_train_per_ball=2,
        teacher_probs=teacher_probs,
        qb_pred_beta=0.5,
    )

    expected_q = torch.tensor([0.75, 0.25])
    assert torch.allclose(targets[0], expected_q, atol=1e-6)
    assert torch.allclose(targets[1], expected_q, atol=1e-6)
    assert stats["teacher_target_variant"] == "pred_mix"
    assert stats["teacher_groups_used"] == 1


def test_gagb_qb_teacher_prior_gamma_zero_matches_noisy_distribution():
    h = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    train_mask = torch.tensor([True, True, False, False])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 0]),
        "groups": [[0, 1, 2, 3]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_teacher_prior",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=1.0,
        gb_grouping=grouping,
        min_train_per_ball=2,
        teacher_probs=teacher_probs,
        qb_teacher_gamma=0.0,
    )

    assert torch.allclose(targets[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(targets[1], torch.tensor([1.0, 0.0]))
    assert stats["teacher_target_variant"] == "teacher_prior"


def test_gagb_qb_conflict_gate_only_smooths_conflicting_train_nodes():
    h = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, False])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.8, 0.2],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "groups": [[0, 1], [2, 3]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_conflict_gate",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=1.0,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=0.5,
        conflict_noisy_support_max=0.35,
        conflict_target_conf_min=0.55,
    )

    assert targets[0, 1] > targets[0, 0]
    assert torch.allclose(targets[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(targets[2], torch.tensor([1.0, 0.0]))
    assert stats["conflict_gate_smoothed_train_nodes"] == 1
    assert stats["effective_train_nodes_smoothed"] == 1


def test_group_quality_scores_rank_compact_supported_low_entropy_ball_higher():
    h = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.97, 0.03],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 0])
    train_mask = torch.ones(6, dtype=torch.bool)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "grace_gagb"},
    }

    quality = compute_group_quality_scores(
        h,
        edge_index,
        labels,
        train_mask,
        grouping,
        num_classes=2,
        min_train_per_ball=3,
    )

    assert "clean_labels" not in inspect.signature(compute_group_quality_scores).parameters
    assert quality["group_quality"][0] > quality["group_quality"][1]
    assert quality["group_semantic"][0] > quality["group_semantic"][1]
    assert quality["group_topology"][0] > quality["group_topology"][1]
    assert quality["group_entropy"][0] > quality["group_entropy"][1]


def test_quality_filter_keeps_high_quality_gated_nodes_and_rejects_low_quality_ball():
    h = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros(6, dtype=torch.long)
    train_mask = torch.ones(6, dtype=torch.bool)
    teacher_probs = torch.tensor([[0.0, 1.0]] * 6, dtype=torch.float32)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0],
        ],
        dtype=torch.long,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "grace_gagb"},
    }

    targets, stats = build_method_targets(
        "grace_gagb_qfilter_alpha07_q50",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        edge_index=edge_index,
    )

    assert torch.allclose(targets[0], torch.tensor([0.3, 0.7]), atol=1e-6)
    assert torch.allclose(targets[3], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert stats["quality_selected_ball_count"] == 1
    assert stats["quality_rejected_ball_count"] == 1
    assert stats["quality_selected_gate_count"] == 3
    assert stats["quality_rejected_gate_count"] == 3
    assert stats["effective_train_nodes_smoothed"] == 3


def test_quality_filter_empty_eligible_balls_does_not_smooth():
    h = torch.eye(4, dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.long)
    train_mask = torch.ones(4, dtype=torch.bool)
    teacher_probs = torch.tensor([[0.0, 1.0]] * 4, dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "groups": [[0, 1], [2, 3]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 2, "grouping_kind": "grace_gagb"},
    }

    targets, stats = build_method_targets(
        "grace_gagb_qfilter_alpha07_q70",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        edge_index=edge_index,
    )

    assert torch.allclose(targets, torch.nn.functional.one_hot(labels, 2).float())
    assert stats["quality_eligible_ball_count"] == 0
    assert stats["quality_selected_ball_count"] == 0
    assert stats["effective_train_nodes_smoothed"] == 0


def test_quality_weight_scales_fixed_alpha_by_group_quality():
    h = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=torch.float32)
    labels = torch.zeros(3, dtype=torch.long)
    train_mask = torch.ones(3, dtype=torch.bool)
    teacher_probs = torch.tensor([[0.0, 1.0]] * 3, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0]),
        "groups": [[0, 1, 2]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(3),
        "diagnostics": {"ball_count": 1, "grouping_kind": "grace_gagb"},
    }
    quality = compute_group_quality_scores(
        h,
        edge_index,
        labels,
        train_mask,
        grouping,
        num_classes=2,
        min_train_per_ball=3,
    )

    targets, stats = build_method_targets(
        "grace_gagb_qweight_alpha07",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        edge_index=edge_index,
    )

    expected_alpha = 0.7 * quality["group_quality"][0]
    assert abs(stats["node_alpha"][0] - expected_alpha) < 1e-6
    assert torch.allclose(targets[0], torch.tensor([1.0 - expected_alpha, expected_alpha]), atol=1e-6)


def test_conflict_gate_alpha07_can_flip_target_argmax_when_qmix_is_strong():
    h = torch.eye(1, dtype=torch.float32)
    labels = torch.tensor([0])
    train_mask = torch.tensor([True])
    teacher_probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0]),
        "groups": [[0]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(1),
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    target03, stats03 = build_method_targets(
        "gagb_qb_conflict_gate_alpha03",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )
    target07, stats07 = build_method_targets(
        "gagb_qb_conflict_gate_alpha07",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert int(target03[0].argmax().item()) == 0
    assert int(target07[0].argmax().item()) == 1
    assert stats03["alpha_max"] == 0.3
    assert stats07["alpha_max"] == 0.7


def test_conflict_gate_alpha07_uses_fixed_gate_alpha_not_geometry_scaled_alpha():
    h = torch.eye(1, dtype=torch.float32)
    labels = torch.tensor([0])
    train_mask = torch.tensor([True])
    teacher_probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0]),
        "groups": [[0]],
        "radii": torch.ones(1) * 10.0,
        "node_distances": torch.ones(1) * 10.0,
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_conflict_gate_alpha07",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert torch.allclose(targets[0], torch.tensor([0.3, 0.7]), atol=1e-6)
    assert abs(stats["alpha_mean_nonzero"] - 0.7) < 1e-6


def test_conflict_gate_alpha08_uses_fixed_alpha():
    h = torch.eye(1, dtype=torch.float32)
    labels = torch.tensor([0])
    train_mask = torch.tensor([True])
    teacher_probs = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0]),
        "groups": [[0]],
        "radii": torch.ones(1) * 10.0,
        "node_distances": torch.ones(1) * 10.0,
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_conflict_gate_alpha08",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert torch.allclose(targets[0], torch.tensor([0.2, 0.8]), atol=1e-6)
    assert stats["alpha_max"] == 0.8
    assert stats["fixed_gate_alpha"] is True


def test_class_capped_alpha09_limits_each_qmix_target_class():
    h = torch.eye(9, dtype=torch.float32)
    labels = torch.zeros(9, dtype=torch.long)
    train_mask = torch.ones(9, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "radii": torch.zeros(3),
        "node_distances": torch.zeros(9),
        "diagnostics": {"ball_count": 3, "grouping_kind": "graph_aware"},
    }
    teacher_probs = torch.zeros(9, 3)
    teacher_probs[:6, 1] = 1.0
    teacher_probs[6:, 2] = 1.0

    _, stats = build_method_targets(
        "gagb_class_capped_alpha09",
        h,
        labels,
        train_mask,
        num_classes=3,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        seed=0,
    )

    selected = torch.tensor(stats["target_update_selected_mask"], dtype=torch.bool)
    candidate_argmax = torch.tensor(stats["target_update_candidate_argmax"], dtype=torch.long)
    assert stats["class_cap_m"] == 3
    for class_id in (1, 2):
        assert int((selected & (candidate_argmax == class_id)).sum().item()) <= 3
    assert stats["calibrated_high_strength_count"] == 6
    assert stats["calibrated_fallback_count"] == 3


def test_class_balanced_alpha09_uses_noisy_train_prior_budget():
    h = torch.eye(12, dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    train_mask = torch.ones(12, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        "groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        "radii": torch.zeros(4),
        "node_distances": torch.zeros(12),
        "diagnostics": {"ball_count": 4, "grouping_kind": "graph_aware"},
    }
    teacher_probs = torch.zeros(12, 3)
    teacher_probs[:6, 1] = 1.0
    teacher_probs[6:, 2] = 1.0

    _, stats = build_method_targets(
        "gagb_class_balanced_alpha09",
        h,
        labels,
        train_mask,
        num_classes=3,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert "clean_labels" not in inspect.signature(build_method_targets).parameters
    assert stats["class_balanced_budget_by_noisy_prior"] == "[2, 3, 2]"
    assert stats["calibrated_high_strength_count"] == 5


def test_boundary_safe_alpha09_uses_teacher_margins_and_confidence():
    h = torch.eye(7, dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0, 1, 1, 1])
    train_mask = torch.ones(7, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5, 6]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(7),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }
    teacher_probs = torch.tensor([
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.55, 0.45],
        [0.55, 0.45],
        [0.55, 0.45],
        [0.55, 0.45],
    ])

    _, stats = build_method_targets(
        "gagb_boundary_safe_alpha09",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=0.2,
    )

    assert stats["qmix_margin_min"] == 0.2
    assert stats["teacher_conf_min"] == 0.6
    assert stats["teacher_margin_min"] == 0.2
    assert stats["calibrated_high_strength_count"] == 1
    assert stats["calibrated_fallback_count"] == 1


def test_qsafe_boundary_filters_low_quality_candidates_without_fallback():
    h = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    labels = torch.zeros(6, dtype=torch.long)
    train_mask = torch.ones(6, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "grace_gagb"},
    }
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])
    teacher_probs = torch.tensor([
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
    ])

    targets, stats = build_method_targets(
        "grace_gagb_qsafe_boundary_alpha09_q50",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        edge_index=edge_index,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    selected = torch.tensor(stats["target_update_selected_mask"], dtype=torch.bool)
    alphas = torch.tensor(stats["node_alpha"], dtype=torch.float32)
    assert selected.tolist() == [True, True, True, False, False, False]
    assert torch.allclose(alphas[:3], torch.full((3,), 0.9))
    assert torch.allclose(alphas[3:], torch.zeros(3))
    assert torch.allclose(targets[3:], one_hot(labels, 2)[3:])
    assert stats["quality_selected_gate_count"] == 3
    assert stats["quality_rejected_gate_count"] == 3
    assert stats["calibrated_candidate_count"] == 3


def test_qsafe_class_capped_only_budgets_high_quality_candidates():
    h = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    labels = torch.zeros(6, dtype=torch.long)
    train_mask = torch.ones(6, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "grace_gagb"},
    }
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ])
    teacher_probs = torch.tensor([
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
        [0.05, 0.95],
    ])

    _, stats = build_method_targets(
        "grace_gagb_qsafe_class_capped_alpha09_q50",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        edge_index=edge_index,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert stats["calibrated_candidate_count"] == 3
    assert stats["class_cap_m"] == 2
    assert stats["calibrated_high_strength_count"] == 2
    assert stats["calibrated_fallback_count"] == 1
    assert stats["quality_rejected_gate_count"] == 3


def test_replay_random_same_count_preserves_source_count_and_seed_determinism():
    h = torch.eye(9, dtype=torch.float32)
    labels = torch.tensor([0, 1, 1, 0, 1, 1, 0, 2, 2])
    train_mask = torch.ones(9, dtype=torch.bool)
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "radii": torch.zeros(3),
        "node_distances": torch.zeros(9),
        "diagnostics": {"ball_count": 3, "grouping_kind": "graph_aware"},
    }
    teacher_probs = torch.zeros(9, 3)
    teacher_probs[:6, 1] = 1.0
    teacher_probs[6:, 2] = 1.0

    first_targets, first_stats = build_method_targets(
        "gagb_replay_random_same_count",
        h,
        labels,
        train_mask,
        num_classes=3,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        seed=7,
    )
    second_targets, second_stats = build_method_targets(
        "gagb_replay_random_same_count",
        h,
        labels,
        train_mask,
        num_classes=3,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=3,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        seed=7,
    )

    assert first_stats["calibrated_candidate_count"] == 3
    assert first_stats["calibrated_high_strength_count"] == 3
    assert first_stats["fallback_alpha"] == 0.0
    assert torch.allclose(first_targets, second_targets)
    assert first_stats["target_update_gate_mask"] == second_stats["target_update_gate_mask"]


def test_conflict_gate_replace_qmix_uses_qmix_as_target_for_gated_nodes():
    h = torch.eye(1, dtype=torch.float32)
    labels = torch.tensor([0])
    train_mask = torch.tensor([True])
    teacher_probs = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0]),
        "groups": [[0]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(1),
        "diagnostics": {"ball_count": 1, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_conflict_gate_replace_qmix",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert torch.allclose(targets[0], torch.tensor([0.2, 0.8]), atol=1e-6)
    assert int(targets[0].argmax().item()) == 1
    assert stats["target_update_mode"] == "replace_qmix"


def test_repair_controls_accept_shuffled_and_khop_groupings():
    h = torch.eye(2, dtype=torch.float32)
    labels = torch.tensor([0, 0])
    train_mask = torch.tensor([True, True])
    teacher_probs = torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0, 0]),
        "groups": [[0, 1]],
        "radii": torch.zeros(1),
        "node_distances": torch.zeros(2),
        "diagnostics": {"ball_count": 1, "grouping_kind": "control_grouping"},
    }

    alpha_targets, alpha_stats = build_method_targets(
        "khop_only_alpha09",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )
    replace_targets, replace_stats = build_method_targets(
        "gagb_shuffled_replace_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert torch.allclose(alpha_targets[0], torch.tensor([0.1, 0.9]), atol=1e-6)
    assert alpha_stats["alpha_max"] == 0.9
    assert alpha_stats["fixed_gate_alpha"] is True
    assert torch.allclose(replace_targets[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert replace_stats["target_update_mode"] == "replace_qmix"


def test_topk_repair_uses_train_matched_topk_rule():
    h = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1])
    train_mask = torch.tensor([True, True, True, True])
    teacher_probs = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1]),
        "groups": [[0, 1], [2, 3]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(4),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }

    _, stats = build_method_targets(
        "topk_gagb_train_matched_alpha09",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
    )

    assert stats["topk_match_rule"] == "mean_eligible_gagb_train_nodes_minus_1"
    assert stats["topk"] == 1
    assert stats["target_update_mode"] == "mix"
    assert stats["alpha_max"] == 0.9


def test_conflict_gate_margin_replace_only_replaces_when_margin_is_large_enough():
    h = torch.eye(2, dtype=torch.float32)
    labels = torch.tensor([0, 0])
    train_mask = torch.tensor([True, True])
    teacher_probs = torch.tensor([[0.3, 0.7], [0.45, 0.55]], dtype=torch.float32)
    grouping = {
        "group_ids": torch.tensor([0, 1]),
        "groups": [[0], [1]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(2),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }

    targets, stats = build_method_targets(
        "gagb_qb_conflict_gate_margin_replace",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=1,
        teacher_probs=teacher_probs,
        qb_pred_beta=1.0,
        conflict_noisy_support_max=0.5,
        conflict_target_conf_min=0.55,
    )

    assert torch.allclose(targets[0], torch.tensor([0.3, 0.7]), atol=1e-6)
    assert torch.allclose(targets[1], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert stats["conflict_gate_smoothed_train_nodes"] == 1
    assert stats["margin_replace_threshold"] == 0.2


def test_target_building_and_teacher_training_do_not_accept_clean_labels():
    target_signature = inspect.signature(build_method_targets)
    teacher_signature = inspect.signature(train_noisy_linear_teacher)

    assert "clean_labels" not in target_signature.parameters
    assert "clean_labels" not in teacher_signature.parameters


def test_topk_gagb_train_alias_and_allnode_match_rules():
    h = torch.eye(10, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, True, True, False, True, False, False, False])
    grouping = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
        "groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]],
        "radii": torch.ones(3),
        "node_distances": torch.zeros(10),
        "diagnostics": {"ball_count": 3, "grouping_kind": "graph_aware"},
    }

    _, train_stats = build_method_targets(
        "topk_gagb_train_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
    )
    _, old_stats = build_method_targets(
        "topk_gagb_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
    )
    _, allnode_stats = build_method_targets(
        "topk_gagb_allnode_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.3,
        gb_grouping=grouping,
        min_train_per_ball=2,
    )

    assert train_stats["topk_match_rule"] == "mean_eligible_gagb_train_nodes_minus_1"
    assert old_stats["topk"] == train_stats["topk"]
    assert allnode_stats["topk_match_rule"] == "gagb_allnode_avg_ball_size_minus_1"
    assert allnode_stats["topk"] == 2


def test_shuffled_alpha_matched_reports_target_level_sanity():
    h = torch.eye(6, dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    train_mask = torch.ones(6, dtype=torch.bool)
    source = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 1, 2], [3, 4, 5]],
        "radii": torch.zeros(2),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware"},
    }
    shuffled = {
        "group_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        "groups": [[0, 3, 4], [1, 2, 5]],
        "radii": torch.ones(2) * 10.0,
        "node_distances": torch.ones(6) * 10.0,
        "diagnostics": {"ball_count": 2, "grouping_kind": "graph_aware_shuffled"},
    }

    _, source_stats = build_method_targets(
        "gagb",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=source,
        min_train_per_ball=3,
    )
    _, shuffled_stats = build_method_targets(
        "gagb_shuffled_alpha_matched",
        h,
        labels,
        train_mask,
        num_classes=2,
        alpha_max=0.5,
        gb_grouping=shuffled,
        min_train_per_ball=3,
        alpha_override=source_stats["node_alpha"],
        alpha_match_reference=source_stats,
    )

    assert shuffled_stats["target_changed_mask_mismatch_rate"] == 0.0
    assert "target_l1_delta_mean_abs_diff" in shuffled_stats
