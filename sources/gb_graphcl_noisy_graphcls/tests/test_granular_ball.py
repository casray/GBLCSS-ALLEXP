import torch

from gb_graphcl.granular_ball import build_granular_balls


def _toy_inputs():
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn((80, 8), generator=gen)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels = torch.tensor([0] * 40 + [1] * 40, dtype=torch.long)
    train_idx = list(range(80))
    return emb, labels, train_idx


def test_all_train_samples_assigned():
    emb, labels, train_idx = _toy_inputs()
    group_ids, _, _, _ = build_granular_balls(emb, labels, train_idx)
    assert len(group_ids) == len(train_idx)
    assert int(group_ids.min().item()) >= 0
    assert len(torch.unique(group_ids)) >= 1


def test_ball_sizes_respect_bounds():
    emb, labels, train_idx = _toy_inputs()
    _, _, sizes, _ = build_granular_balls(emb, labels, train_idx, min_samples=5, max_size=100)
    for size in sizes:
        assert size >= 1
    assert max(sizes) <= 100

