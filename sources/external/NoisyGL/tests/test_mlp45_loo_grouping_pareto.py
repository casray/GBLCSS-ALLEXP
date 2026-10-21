from __future__ import annotations

import importlib

import numpy as np
import torch

from utils.frozen_grace_targets import build_kmeans_soft_targets_from_features


def test_kmeans_builder_supports_leave_one_out_targets():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    noisy = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    train = np.arange(4, dtype=np.int64)

    result = build_kmeans_soft_targets_from_features(
        ball_features=features,
        noisy_label=noisy,
        clean_label=noisy.clone(),
        train_mask=train,
        n_classes=2,
        matched_kmeans_clusters=1,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
        leave_one_out=True,
        random_state=0,
    )

    assert result.diagnostics["leave_one_out"] == 1
    assert torch.allclose(result.soft_target[0], result.soft_target[1])
    assert torch.allclose(result.soft_target[2], result.soft_target[3])
    assert not torch.allclose(result.soft_target[0], result.soft_target[2])


def test_kmeans_loo_predictor_is_registered():
    runner = importlib.import_module("scripts.run_standard_downstream")
    predictor_module = importlib.import_module(
        "predictor.FrozenGraceGB_GCN_Predictor"
    )

    assert "kmeans_mlp_softdist_beta05_loo" in runner.PREDICTOR_MODULES
    predictor = getattr(
        predictor_module,
        "kmeans_mlp_softdist_beta05_loo_Predictor",
    )
    assert predictor.soft_target_builder == "kmeans"
    assert predictor.leave_one_out is True


def test_structure_storage_formula_orders():
    module = importlib.import_module("scripts.benchmark_mlp45_grouping_pareto")
    gb_bytes = module.structure_storage_bytes(
        method="loo_gb",
        n_nodes=100,
        n_train=20,
        dim=8,
        groups=5,
        k=4,
    )
    knn_bytes = module.structure_storage_bytes(
        method="exact_knn",
        n_nodes=100,
        n_train=20,
        dim=8,
        groups=5,
        k=4,
    )
    kmeans_bytes = module.structure_storage_bytes(
        method="loo_kmeans",
        n_nodes=100,
        n_train=20,
        dim=8,
        groups=5,
        k=4,
    )

    assert gb_bytes == kmeans_bytes
    assert knn_bytes == 20 * 4 * np.dtype(np.int64).itemsize
    assert gb_bytes == (
        100 * np.dtype(np.int64).itemsize
        + 5 * 8 * np.dtype(np.float32).itemsize
    )
