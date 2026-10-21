import torch
import torch.nn.functional as F

from gb_graphcl.reliability import compute_soft_targets
from gb_graphcl.selection import train_ce_baseline, train_with_smoothing


def _toy_problem():
    gen = torch.Generator().manual_seed(11)
    embeddings = torch.randn((60, 16), generator=gen)
    labels = (embeddings[:, 0] + embeddings[:, 1] > 0).long()
    noisy = labels.clone()
    noisy[::6] = 1 - noisy[::6]
    train_idx = list(range(40))
    val_idx = list(range(40, 50))
    test_idx = list(range(50, 60))
    group_ids = torch.tensor([i // 5 for i in range(len(train_idx))], dtype=torch.long)
    return embeddings, noisy, labels, train_idx, val_idx, test_idx, group_ids


def test_soft_targets_sum_to_one():
    embeddings, noisy, _, train_idx, _, _, group_ids = _toy_problem()
    targets = compute_soft_targets("gb", embeddings, noisy, train_idx, num_classes=2, group_ids=group_ids)
    assert torch.allclose(targets.sum(dim=1), torch.ones(len(train_idx)), atol=1e-5)


def test_soft_targets_none_is_one_hot():
    embeddings, noisy, _, train_idx, _, _, _ = _toy_problem()
    targets = compute_soft_targets("none", embeddings, noisy, train_idx, num_classes=2)
    expected = F.one_hot(noisy[torch.tensor(train_idx)], num_classes=2).float()
    assert torch.equal(targets, expected)


def test_smoothing_with_one_hot_equals_ce():
    embeddings, noisy, clean, train_idx, val_idx, test_idx, _ = _toy_problem()
    soft_targets = F.one_hot(noisy[torch.tensor(train_idx)], num_classes=2).float()
    kwargs = {
        "embeddings": embeddings,
        "noisy_labels": noisy,
        "clean_labels": clean,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "num_classes": 2,
        "device": torch.device("cpu"),
        "epochs": 25,
        "lr": 0.02,
        "wd": 0.0,
        "patience": 25,
        "seed": 5,
    }
    ce = train_ce_baseline(**kwargs)
    sm = train_with_smoothing(**kwargs, soft_targets=soft_targets, warmup=0, ramp=1)
    assert abs(ce["test_acc_at_best_val"] - sm["test_acc_at_best_val"]) < 1e-4

