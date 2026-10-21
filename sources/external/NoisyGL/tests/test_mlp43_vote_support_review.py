from __future__ import annotations

import numpy as np
import torch

from utils.frozen_grace_targets import (
    build_kmeans_soft_targets_from_features,
    build_knn_soft_targets_from_features,
    vote_support_diagnostics,
)


def test_smin_grid_is_frozen_and_deduplicated():
    from scripts.run_mlp43_smin_sensitivity import parse_smins

    assert parse_smins("2,3,5,10,5") == [2, 3, 5, 10]


def test_knn_can_exclude_query_node_from_its_neighbor_votes():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )
    noisy = torch.tensor([0, 1, 1])
    train_mask = torch.tensor([True, True, True])

    result = build_knn_soft_targets_from_features(
        ball_features=features,
        noisy_label=noisy,
        clean_label=noisy,
        train_mask=train_mask,
        n_classes=2,
        max_balls=1,
        min_ball_size=2,
        split_improvement=0.0,
        matched_k=1,
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
        exclude_self=True,
    )

    assert result.pseudo_label[0].item() == 1
    assert result.diagnostics["knn_exclude_self"] == 1
    assert result.diagnostics["matched_k"] == 1


def test_knn_distribution_only_tie_is_class_permutation_equivariant():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ]
    )
    noisy = torch.tensor([2, 0, 1])
    train_mask = torch.tensor([True, True, True])
    permutation = torch.tensor([1, 2, 0])

    result = build_knn_soft_targets_from_features(
        ball_features=features,
        noisy_label=noisy,
        clean_label=noisy,
        train_mask=train_mask,
        n_classes=3,
        max_balls=1,
        min_ball_size=2,
        split_improvement=0.0,
        matched_k=2,
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
        exclude_self=True,
    )
    permuted = build_knn_soft_targets_from_features(
        ball_features=features,
        noisy_label=permutation[noisy],
        clean_label=permutation[noisy],
        train_mask=train_mask,
        n_classes=3,
        max_balls=1,
        min_ball_size=2,
        split_improvement=0.0,
        matched_k=2,
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
        exclude_self=True,
    )

    assert torch.allclose(permuted.soft_target[:, permutation], result.soft_target)
    assert torch.equal(permuted.alpha, result.alpha)
    assert torch.equal(permuted.pseudo_label, permutation[result.pseudo_label])


def test_kmeans_distribution_only_tie_is_class_permutation_equivariant():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    noisy = torch.tensor([0, 0, 1, 1, 2])
    train_mask = torch.tensor([True, True, True, True, True])
    permutation = torch.tensor([2, 0, 1])

    result = build_kmeans_soft_targets_from_features(
        ball_features=features,
        noisy_label=noisy,
        clean_label=noisy,
        train_mask=train_mask,
        n_classes=3,
        max_balls=1,
        min_ball_size=2,
        split_improvement=0.0,
        matched_kmeans_clusters=1,
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
    )
    permuted = build_kmeans_soft_targets_from_features(
        ball_features=features,
        noisy_label=permutation[noisy],
        clean_label=permutation[noisy],
        train_mask=train_mask,
        n_classes=3,
        max_balls=1,
        min_ball_size=2,
        split_improvement=0.0,
        matched_kmeans_clusters=1,
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
    )

    assert torch.allclose(permuted.soft_target[:, permutation], result.soft_target)
    assert torch.equal(permuted.alpha, result.alpha)
    assert torch.equal(permuted.pseudo_label, permutation[result.pseudo_label])


def test_vote_support_diagnostics_reports_requested_bins_and_node_coverage():
    noisy = torch.tensor(
        [
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    clean = noisy.clone()
    assignment = np.concatenate(
        [
            np.full(2, 0),
            np.full(4, 1),
            np.full(8, 2),
            np.full(15, 3),
            np.full(20, 4),
        ]
    )

    rows = vote_support_diagnostics(
        noisy_label=noisy,
        clean_label=clean,
        assignment=assignment,
        train_mask=np.arange(noisy.numel()),
        n_classes=2,
        min_train_per_ball=2,
        tie_mode="distribution_only",
    )
    by_bin = {row["support_bin"]: row for row in rows}

    assert set(by_bin) == {"2", "3-4", "5-9", "10-19", "20+"}
    assert by_bin["2"]["target_node_count"] == 2
    assert by_bin["3-4"]["target_node_count"] == 4
    assert by_bin["5-9"]["target_node_count"] == 8
    assert by_bin["10-19"]["target_node_count"] == 15
    assert by_bin["20+"]["target_node_count"] == 20
    assert sum(row["target_node_count"] for row in rows) == noisy.numel()
    assert abs(sum(row["target_node_coverage"] for row in rows) - 1.0) < 1e-8
