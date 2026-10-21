from __future__ import annotations

import numpy as np
import torch

from scripts.run_mlp53_nrgnn_latent_balls import (
    evaluate_phase_a_gate,
    exact_knn_loo_targets,
    extract_best_nrgnn_state,
)


class _FakeModel:
    def __init__(self):
        self.training = True
        self.loaded = None

    def load_state_dict(self, weights):
        self.loaded = weights

    def eval(self):
        self.training = False

    def forward_with_hidden(self, features, adjacency):
        assert self.loaded == {"best": 1}
        assert not self.training
        return features[:, :2], features[:, 1:3]


class _FakePredictor:
    n_nodes = 3
    model = _FakeModel()
    weights = {"best": 1}
    best_model_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    best_graph = torch.ones(3)
    feats = torch.tensor([[1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0]])


def test_extract_best_nrgnn_state_restores_validation_selected_weights():
    logits, hidden = extract_best_nrgnn_state(_FakePredictor())
    assert logits.shape == (3, 2)
    assert hidden.shape == (3, 2)
    assert torch.isfinite(hidden).all()


def test_exact_knn_targets_exclude_query_node():
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]]
    )
    noisy = torch.tensor([0, 1, 1, 0])
    clean = noisy.clone()
    train = np.asarray([0, 1, 2, 3], dtype=np.int64)

    targets, alpha = exact_knn_loo_targets(
        features=features,
        noisy_label=noisy,
        clean_label=clean,
        train_indices=train,
        n_classes=2,
        k=2,
    )

    # Node 0's two nearest other nodes are both class 1. Including itself changes the vote.
    assert targets[0].argmax() == 1
    assert alpha[0] > 0


def _row(dataset, rate, method, brier, accuracy=0.7):
    return {
        "dataset": dataset,
        "noise_rate": rate,
        "method": method,
        "brier": brier,
        "target_accuracy": accuracy,
        "status": "ok",
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}",
        "noise_hash": f"noise-{dataset}-{rate}",
        "target_simplex_max_error": 0.0,
        "representation_finite": 1,
        "representation_hash": f"z-{dataset}-{rate}",
    }


def test_gate_accepts_complete_passing_grid():
    rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for rate in (0.4, 0.6):
            rows.extend(
                [
                    _row(dataset, rate, "formal_grace_h_gb", 0.40),
                    _row(dataset, rate, "nrgnn_direct_teacher", 0.30),
                    _row(dataset, rate, "nrgnn_z_gb", 0.28, 0.71),
                    _row(dataset, rate, "nrgnn_z_knn", 0.29, 0.71),
                    _row(dataset, rate, "nrgnn_z_gb_shuffled", 0.45),
                ]
            )
    decision = evaluate_phase_a_gate(rows)
    assert decision["pass"] is True
    assert decision["z_gb_brier_wins_vs_z_knn"] == 6


def test_gate_rejects_incomplete_grid():
    decision = evaluate_phase_a_gate([])
    assert decision["pass"] is False
    assert decision["reason"] == "incomplete"
