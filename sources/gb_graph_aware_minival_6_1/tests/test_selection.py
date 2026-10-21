import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.evaluation import (
    ball_consensus_score,
    ball_density_reliability,
    keep_ratio_at_epoch,
    train_frozen_classifier_with_selection,
)

CPU = torch.device("cpu")


def _toy(seed=0, n_per=20, num_classes=3, dim=5, noise=0.4, flip_seed=1):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_classes).repeat_interleave(n_per)
    n = num_classes * n_per
    centers = torch.randn(num_classes, dim, generator=g) * 3.0
    emb = centers[labels] + 0.4 * torch.randn(n, dim, generator=g)
    perm = torch.randperm(n, generator=g)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    n_tr, n_va = n // 2, n // 4
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr:n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va:]] = True
    gf = torch.Generator().manual_seed(flip_seed)
    noisy = labels.clone()
    tr = train_mask.nonzero(as_tuple=False).view(-1)
    flip = tr[torch.rand(tr.numel(), generator=gf) < noise]
    noisy[flip] = (labels[flip] + 1 + torch.randint(0, num_classes - 1, (flip.numel(),), generator=gf)) % num_classes
    return emb, labels, noisy, train_mask, val_mask, test_mask, num_classes


def _run(emb, clean, noisy, tr, va, te, c, **kw):
    base = dict(
        embeddings=emb, clean_labels=clean, noisy_labels=noisy, num_classes=c,
        train_mask=tr, val_mask=va, test_mask=te, seed=0,
        keep_ratio_final=0.6, warmup_epochs=5, ramp_epochs=5,
        epochs=60, patience=60, device=CPU,
    )
    base.update(kw)
    return train_frozen_classifier_with_selection(**base)


def test_keep_ratio_schedule_is_fixed_warmup_ramp_hold():
    W, ramp, rho = 10, 10, 0.6
    assert keep_ratio_at_epoch(1, W, ramp, rho) == 1.0
    assert keep_ratio_at_epoch(W, W, ramp, rho) == 1.0
    assert abs(keep_ratio_at_epoch(15, W, ramp, rho) - 0.8) < 1e-6  # mid-ramp linear
    assert abs(keep_ratio_at_epoch(W + ramp, W, ramp, rho) - rho) < 1e-6
    assert abs(keep_ratio_at_epoch(100, W, ramp, rho) - rho) < 1e-6


def test_ball_consensus_score_is_unit_range_and_clean_label_free():
    grouping = {"group_ids": torch.tensor([0, 0, 0, 1, 1, 1]), "groups": [[0, 1, 2], [3, 4, 5]]}
    noisy = torch.tensor([1, 1, 1, 0, 2, 2])  # ball0 pure label-1; ball1 = {0,2,2}
    train_mask = torch.ones(6, dtype=torch.bool)
    rel, covered = ball_consensus_score(grouping, noisy, train_mask, num_classes=3)
    assert covered.all()
    assert torch.allclose(rel[:3], torch.ones(3))            # pure ball -> consensus 1
    assert abs(float(rel[3]) - 0.0) < 1e-6                   # label0 alone among {2,2}
    assert abs(float(rel[4]) - 0.5) < 1e-6                   # label2 among {0,2}
    assert float(rel.min()) >= 0.0 and float(rel.max()) <= 1.0
    # signature accepts no clean labels at all (consensus is computed from noisy only)
    import inspect
    assert "clean" not in inspect.signature(ball_consensus_score).parameters


def test_clean_oracle_selection_precision_is_one():
    emb, clean, noisy, tr, va, te, c = _toy()
    res = _run(emb, clean, noisy, tr, va, te, c, use_reliability=False, clean_oracle=True)
    assert abs(res["selection_precision"] - 1.0) < 1e-9
    expected = int((noisy[tr] == clean[tr]).sum())
    assert res["selected_count"] == expected


def test_loss_only_selection_ignores_reliability():
    emb, clean, noisy, tr, va, te, c = _toy()
    rel1 = torch.rand(clean.numel())
    rel2 = torch.rand(clean.numel())
    a = _run(emb, clean, noisy, tr, va, te, c, use_reliability=False, reliability=rel1)
    b = _run(emb, clean, noisy, tr, va, te, c, use_reliability=False, reliability=rel2)
    assert a["test_acc_at_best_val"] == b["test_acc_at_best_val"]
    assert a["selected_count"] == b["selected_count"]


def test_selection_training_is_deterministic():
    emb, clean, noisy, tr, va, te, c = _toy()
    rel, cov = ball_consensus_score(
        {"group_ids": torch.zeros(clean.numel(), dtype=torch.long), "groups": [list(range(clean.numel()))]},
        noisy, tr, c,
    )
    a = _run(emb, clean, noisy, tr, va, te, c, reliability=rel, reliability_covered=cov)
    b = _run(emb, clean, noisy, tr, va, te, c, reliability=rel, reliability_covered=cov)
    assert a["test_acc_at_best_val"] == b["test_acc_at_best_val"]
    assert a["selected_count"] == b["selected_count"]
    assert a["best_epoch"] == b["best_epoch"]


def test_ball_density_reliability_is_unit_range_and_clean_label_free():
    import inspect
    emb = torch.randn(9, 8)
    grouping = {"group_ids": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                "groups": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}
    noisy = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    train_mask = torch.ones(9, dtype=torch.bool)
    rel, cov = ball_density_reliability(grouping, emb, noisy, train_mask, num_classes=3)
    assert cov.all()
    assert float(rel.min()) >= 0.0 and float(rel.max()) <= 1.0
    # ball-level density signal must not depend on clean labels
    assert "clean" not in inspect.signature(ball_density_reliability).parameters
