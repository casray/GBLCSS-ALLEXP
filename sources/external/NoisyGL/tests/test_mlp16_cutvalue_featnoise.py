from __future__ import annotations

import torch

from scripts.run_mlp16_cutvalue_featnoise import cut_value_from_probs, perturb_test_features


def test_cut_value_prefers_edge_consistent_predictions():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    consistent = torch.tensor(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.05, 0.95],
            [0.10, 0.90],
        ],
        dtype=torch.float32,
    )
    inconsistent = torch.tensor(
        [
            [0.95, 0.05],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.95, 0.05],
        ],
        dtype=torch.float32,
    )
    assert cut_value_from_probs(consistent, edge_index, 4) > cut_value_from_probs(inconsistent, edge_index, 4)


def test_cut_value_ignores_self_loops_and_duplicate_direction():
    edge_index = torch.tensor([[0, 1, 0, 0, 1], [1, 0, 1, 0, 1]])
    probs = torch.tensor([[0.8, 0.2], [0.7, 0.3]], dtype=torch.float32)
    duplicated = cut_value_from_probs(probs, edge_index, 2)
    simple = cut_value_from_probs(probs, torch.tensor([[0], [1]]), 2)
    assert abs(duplicated - simple) < 1e-6


def test_perturb_test_features_keeps_train_rows_and_alpha_zero_exact():
    features = torch.arange(20, dtype=torch.float32).view(5, 4)
    test_mask = torch.tensor([1, 3])
    same = perturb_test_features(features, test_mask, 0.0, seed=7, dataset="toy", method="m")
    assert same is features

    noisy = perturb_test_features(features, test_mask, 0.5, seed=7, dataset="toy", method="m")
    train_rows = torch.tensor([0, 2, 4])
    assert torch.allclose(noisy[train_rows], features[train_rows])
    assert not torch.allclose(noisy[test_mask], features[test_mask])
