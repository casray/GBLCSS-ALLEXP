"""Minival co-teaching baseline on frozen embeddings.

This is intentionally local to the minival protocol. It does not reuse NoisyGL numbers because those
use a different split/noise protocol. Two linear heads exchange small-loss train nodes under the same
keep-ratio schedule used by commit-style selection controls.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.evaluation.classifier import LinearClassifier
from noisy_label_gb_grace.evaluation.selection import keep_ratio_at_epoch
from noisy_label_gb_grace.utils import mask_to_index, set_seed


def _acc(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return 0.0
    return float((logits[idx].argmax(dim=-1) == labels[idx]).float().mean().item())


def _small_loss_mask(per_loss: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n = int(per_loss.numel())
    k = max(1, int(round(float(keep_ratio) * n)))
    selected = torch.topk(-per_loss.detach(), k).indices
    mask = torch.zeros(n, dtype=torch.bool, device=per_loss.device)
    mask[selected] = True
    return mask


def _selection_diagnostics(
    selected: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    train_idx: torch.Tensor,
) -> dict:
    is_clean = noisy[train_idx] == clean[train_idx]
    is_noisy = ~is_clean
    selected_count = int(selected.sum().item())
    if selected_count > 0:
        precision = float(is_clean[selected].float().mean().item())
    else:
        precision = math.nan
    n_noisy = int(is_noisy.sum().item())
    if n_noisy > 0:
        noise_excl = float(((~selected) & is_noisy).float().sum().item() / n_noisy)
    else:
        noise_excl = math.nan
    return {
        "selected_count": selected_count,
        "selected_coverage": float(selected_count / max(1, int(train_idx.numel()))),
        "selection_precision": precision,
        "noise_excluded_recall": noise_excl,
    }


def train_coteaching(
    embeddings: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
    keep_ratio_final: float = 0.6,
    warmup_epochs: int = 20,
    ramp_epochs: int = 20,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: torch.device | None = None,
) -> dict:
    """Train a two-head co-teaching baseline on frozen embeddings.

    Clean labels are used only for validation/test model selection and diagnostics. The per-epoch
    selected set is determined solely by each network's noisy-label loss.
    """
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = embeddings.detach().float().to(device)
    clean = clean_labels.detach().long().to(device)
    noisy = noisy_labels.detach().long().to(device)
    train_mask = train_mask.detach().bool().to(device)
    val_mask = val_mask.detach().bool().to(device)
    test_mask = test_mask.detach().bool().to(device)
    train_idx = mask_to_index(train_mask)
    val_idx = mask_to_index(val_mask)
    test_idx = mask_to_index(test_mask)

    model_a = LinearClassifier(x.size(1), num_classes).to(device)
    model_b = LinearClassifier(x.size(1), num_classes).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=weight_decay)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_test = 0.0
    best_epoch = 0
    stale = 0
    best_a = copy.deepcopy(model_a.state_dict())
    best_b = copy.deepcopy(model_b.state_dict())
    selected_at_best = torch.ones(int(train_idx.numel()), dtype=torch.bool, device=device)

    for epoch in range(1, epochs + 1):
        model_a.train()
        model_b.train()
        opt_a.zero_grad()
        opt_b.zero_grad()
        logits_a = model_a(x)
        logits_b = model_b(x)
        loss_a = F.cross_entropy(logits_a[train_idx], noisy[train_idx], reduction="none")
        loss_b = F.cross_entropy(logits_b[train_idx], noisy[train_idx], reduction="none")

        keep_ratio = keep_ratio_at_epoch(epoch, warmup_epochs, ramp_epochs, keep_ratio_final)
        if epoch <= warmup_epochs:
            select_a = torch.ones_like(loss_a, dtype=torch.bool)
            select_b = torch.ones_like(loss_b, dtype=torch.bool)
        else:
            select_a = _small_loss_mask(loss_a, keep_ratio)
            select_b = _small_loss_mask(loss_b, keep_ratio)

        # Cross update: A learns from B's small-loss set, B learns from A's small-loss set.
        update_loss_a = loss_a[select_b].mean() if select_b.any() else loss_a.mean()
        update_loss_b = loss_b[select_a].mean() if select_a.any() else loss_b.mean()
        (update_loss_a + update_loss_b).backward()
        opt_a.step()
        opt_b.step()

        model_a.eval()
        model_b.eval()
        with torch.no_grad():
            eval_logits = (model_a(x) + model_b(x)) / 2.0
            val_acc = _acc(eval_logits, clean, val_idx)
            test_acc = _acc(eval_logits, clean, test_idx)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            best_a = copy.deepcopy(model_a.state_dict())
            best_b = copy.deepcopy(model_b.state_dict())
            selected_at_best = select_a.detach().clone()
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    model_a.load_state_dict(best_a)
    model_b.load_state_dict(best_b)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        final_logits = (model_a(x) + model_b(x)) / 2.0
        train_acc = _acc(final_logits, clean, train_idx)
        val_acc = _acc(final_logits, clean, val_idx)
        test_acc = _acc(final_logits, clean, test_idx)

    return {
        "best_val_acc": float(best_val),
        "test_acc_at_best_val": float(best_test),
        "best_epoch": int(best_epoch),
        "train_acc_at_best_state": float(train_acc),
        "val_acc_at_best_state": float(val_acc),
        "test_acc_at_best_state": float(test_acc),
        **_selection_diagnostics(selected_at_best, clean, noisy, train_idx),
    }
