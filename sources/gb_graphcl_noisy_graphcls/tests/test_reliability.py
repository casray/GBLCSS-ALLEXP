import torch

from gb_graphcl.reliability import gb_consensus_reliability, kmeans_consensus_reliability, knn_consensus_reliability


def _toy_inputs():
    gen = torch.Generator().manual_seed(1)
    emb = torch.randn((60, 6), generator=gen)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels = torch.tensor([0] * 30 + [1] * 30, dtype=torch.long)
    train_idx = list(range(60))
    return emb, labels, train_idx


def test_knn_reliability_is_unit_range():
    emb, labels, train_idx = _toy_inputs()
    rel = knn_consensus_reliability(emb, labels, train_idx, k=10)
    assert len(rel) == len(train_idx)
    assert float(rel.min().item()) >= 0.0
    assert float(rel.max().item()) <= 1.0


def test_kmeans_reliability_is_unit_range():
    emb, labels, train_idx = _toy_inputs()
    rel = kmeans_consensus_reliability(emb, labels, train_idx, n_clusters=4)
    assert len(rel) == len(train_idx)
    assert float(rel.min().item()) >= 0.0
    assert float(rel.max().item()) <= 1.0


def test_gb_reliability_respects_groups():
    _, labels, train_idx = _toy_inputs()
    group_ids = torch.tensor([0] * 30 + [1] * 30, dtype=torch.long)
    rel = gb_consensus_reliability(group_ids, labels, train_idx)
    assert torch.allclose(rel, torch.ones_like(rel))

