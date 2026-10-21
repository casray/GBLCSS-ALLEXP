import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.training import ball_coupling_losses, label_free_quality_mask


def _three_pure_balls():
    # 3 well-separated pure-label balls on a hypersphere, 3 train nodes each
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]])
    pts = []
    gid = []
    noisy = []
    for c in range(3):
        for _ in range(3):
            pts.append(centers[c] + 0.1 * torch.randn(3, generator=g))
            gid.append(c)
            noisy.append(c)
    h = torch.stack(pts)
    group_ids = torch.tensor(gid)
    noisy_labels = torch.tensor(noisy)
    train_mask = torch.ones(9, dtype=torch.bool)
    return h, group_ids, noisy_labels, train_mask


def test_losses_are_differentiable_wrt_h():
    h, gid, noisy, tm = _three_pure_balls()
    h = h.clone().requires_grad_(True)
    qmask = label_free_quality_mask(gid, h.detach(), noisy, tm, num_classes=3)
    L_intra, L_inter = ball_coupling_losses(h, gid, qmask, noisy, tm, num_classes=3, margin=0.5)
    (L_intra + L_inter).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_no_clean_label_in_signatures():
    for fn in (ball_coupling_losses, label_free_quality_mask):
        assert "clean" not in inspect.signature(fn).parameters


def test_zero_quality_mask_gives_zero_grad_safe_loss():
    h, gid, noisy, tm = _three_pure_balls()
    h = h.clone().requires_grad_(True)
    n_balls = int(gid.max()) + 1
    qmask = torch.zeros(n_balls, dtype=torch.bool)
    L_intra, L_inter = ball_coupling_losses(h, gid, qmask, noisy, tm, num_classes=3)
    assert float(L_intra) == 0.0 and float(L_inter) == 0.0
    # still grad-connected (no crash on backward)
    (L_intra + L_inter).backward()
    assert h.grad is not None


def test_shuffled_membership_raises_intra_loss():
    # destroying the ball structure (random membership) must make intra-pull WORSE (higher)
    h, gid, noisy, tm = _three_pure_balls()
    q_true = torch.ones(3, dtype=torch.bool)
    L_intra_true, _ = ball_coupling_losses(h, gid, q_true, noisy, tm, num_classes=3)
    g = torch.Generator().manual_seed(1)
    shuffled_gid = gid[torch.randperm(9, generator=g)]
    L_intra_shuf, _ = ball_coupling_losses(h, shuffled_gid, q_true, noisy, tm, num_classes=3)
    assert float(L_intra_shuf) > float(L_intra_true)


def test_quality_mask_is_boolean_per_ball():
    h, gid, noisy, tm = _three_pure_balls()
    qmask = label_free_quality_mask(gid, h, noisy, tm, num_classes=3, min_train=3)
    assert qmask.dtype == torch.bool
    assert qmask.numel() == int(gid.max()) + 1
