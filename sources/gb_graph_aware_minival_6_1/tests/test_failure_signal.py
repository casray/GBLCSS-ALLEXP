import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.evaluation.failure_signal import (
    boundary_warning_metrics,
    neighborhood_majority_confidence,
    neighborhood_majority_self_disagreement,
    pairflip_neighborhood_consistency,
)


def _toy_embeddings():
    return torch.tensor(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.00, 1.00],
            [0.01, 0.99],
            [0.02, 0.98],
        ],
        dtype=torch.float32,
    )


def test_failure_signal_functions_are_clean_label_free():
    for fn in [
        neighborhood_majority_self_disagreement,
        neighborhood_majority_confidence,
        pairflip_neighborhood_consistency,
        boundary_warning_metrics,
    ]:
        sig = inspect.signature(fn)
        assert "clean_labels" not in sig.parameters
        assert "clean" not in sig.parameters


def test_neighborhood_warning_metrics_are_scalar_and_unit_range():
    emb = _toy_embeddings()
    labels = torch.tensor([0, 0, 1, 1, 1, 0])
    train_mask = torch.ones(6, dtype=torch.bool)
    disagree = neighborhood_majority_self_disagreement(emb, labels, train_mask, k=2)
    confidence = neighborhood_majority_confidence(emb, labels, train_mask, k=2)
    same_noisy = pairflip_neighborhood_consistency(emb, labels, train_mask, k=2, num_classes=2)
    assert 0.0 <= disagree <= 1.0
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= same_noisy <= 1.0


def test_pair_like_neighborhood_has_higher_noisy_confidence_than_symmetric_like():
    emb = _toy_embeddings()
    train_mask = torch.ones(6, dtype=torch.bool)
    pair_like = torch.tensor([1, 1, 1, 0, 0, 0])
    symmetric_like = torch.tensor([1, 0, 1, 0, 1, 0])
    pair_conf = neighborhood_majority_confidence(emb, pair_like, train_mask, k=2)
    sym_conf = neighborhood_majority_confidence(emb, symmetric_like, train_mask, k=2)
    assert pair_conf > sym_conf


def test_boundary_warning_metrics_include_prefixed_fields():
    metrics = boundary_warning_metrics(
        _toy_embeddings(),
        torch.tensor([1, 1, 1, 0, 0, 0]),
        torch.ones(6, dtype=torch.bool),
        k=2,
        num_classes=2,
        prefix="topk_",
    )
    assert set(metrics) == {
        "topk_neighborhood_majority_self_disagreement",
        "topk_neighborhood_majority_confidence",
        "topk_pairflip_neighborhood_consistency",
    }
