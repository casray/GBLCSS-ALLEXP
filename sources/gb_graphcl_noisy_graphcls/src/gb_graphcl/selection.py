from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def keep_ratio_at_epoch(epoch: int, warmup: int = 20, ramp: int = 20, rho: float = 0.6) -> float:
    """Piecewise warmup/ramp keep-ratio schedule."""
    if epoch < warmup:
        return 1.0
    if epoch >= warmup + ramp:
        return float(rho)
    progress = (epoch - warmup) / float(ramp)
    return 1.0 - progress * (1.0 - float(rho))


def _ramp_progress(epoch: int, warmup: int, ramp: int) -> float:
    """Return a linear warmup-to-ramp progress in [0, 1]."""
    if epoch < warmup:
        return 0.0
    if ramp <= 0 or epoch >= warmup + ramp:
        return 1.0
    return (epoch - warmup) / float(ramp)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accuracy(logits, labels) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def _z_score(x):
    x = x.float()
    std = x.std(unbiased=False)
    if float(std.item()) < 1e-12:
        return torch.zeros_like(x)
    return (x - x.mean()) / (std + 1e-12)


def _init_linear(input_dim: int, num_classes: int, seed: int, device):
    _set_seed(seed)
    return nn.Linear(input_dim, num_classes).to(device)


def _result(model, embeddings, clean_labels, train_idx, last_selected, best_test, loss_history):
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=embeddings.device)
    selected_global = train_idx_t[last_selected]
    clean_train = clean_labels[train_idx_t]
    noisy_mask = clean_labels[train_idx_t] != _result.noisy_labels_for_metrics
    if len(selected_global) == 0:
        selection_precision = 0.0
    else:
        selection_precision = float((clean_labels[selected_global] == _result.noisy_labels_for_metrics[last_selected]).float().mean().item())
    if int(noisy_mask.sum().item()) == 0:
        noise_exclusion = 0.0
    else:
        selected_mask = torch.zeros(len(train_idx_t), dtype=torch.bool, device=embeddings.device)
        selected_mask[last_selected] = True
        excluded_noisy = noisy_mask & (~selected_mask)
        noise_exclusion = float(excluded_noisy.float().sum().item() / noisy_mask.float().sum().item())
    return {
        "test_acc_at_best_val": float(best_test),
        "selection_precision": selection_precision,
        "noise_exclusion_recall": noise_exclusion,
        "loss_history": loss_history,
    }


def _train_loop(
    embeddings,
    train_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    num_classes,
    device,
    seed,
    epochs,
    lr,
    wd,
    patience,
    loss_fn,
    selection_fn=None,
    trace=False,
):
    embeddings = embeddings.detach().float().to(device)
    train_labels = train_labels.detach().long().to(device)
    clean_labels = clean_labels.detach().long().to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
    clean_mask_train = (train_labels[train_idx_t] == clean_labels[train_idx_t])
    model = _init_linear(embeddings.size(1), num_classes, seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = -1.0
    best_test = 0.0
    stale = 0
    loss_history = []
    trace_rows = [] if trace else None
    last_selected = torch.arange(len(train_idx_t), device=device)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(embeddings[train_idx_t])
        per_sample = loss_fn(logits, train_labels[train_idx_t])
        if selection_fn is not None:
            selected = selection_fn(epoch, per_sample.detach())
            last_selected = selected
            loss = per_sample[selected].mean()
        else:
            last_selected = torch.arange(len(train_idx_t), device=device)
            loss = per_sample.mean()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(embeddings[val_idx_t]), clean_labels[val_idx_t])
            test_acc = _accuracy(model(embeddings[test_idx_t]), clean_labels[test_idx_t])
            train_acc = _accuracy(model(embeddings[train_idx_t]), clean_labels[train_idx_t])

        if trace_rows is not None:
            sel_clean_rate = float(clean_mask_train[last_selected].float().mean().item()) if len(last_selected) > 0 else 0.0
            trace_rows.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "keep_ratio": len(last_selected) / len(train_idx_t),
                "selected_count": int(len(last_selected)),
                "selected_clean_rate": sel_clean_rate,
            })

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    _result.noisy_labels_for_metrics = train_labels[train_idx_t]
    res = _result(model, embeddings, clean_labels, train_idx, last_selected, best_test, loss_history)
    if trace_rows is not None:
        res["trace"] = trace_rows
        res["stopped_epoch"] = epoch
    return res


def train_with_selection(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    reliability,
    num_classes,
    device,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    warmup: int = 20,
    ramp: int = 20,
    rho: float = 0.6,
    seed: int = 0,
    trace: bool = False,
):
    """Train a linear classifier with commit-style sample selection."""
    reliability = reliability.detach().float().to(device)

    def ce_loss(logits, labels):
        return F.cross_entropy(logits, labels, reduction="none")

    def select(epoch, per_sample):
        keep = keep_ratio_at_epoch(epoch, warmup=warmup, ramp=ramp, rho=rho)
        num_keep = max(1, int(round(len(per_sample) * keep)))
        score = _z_score(-per_sample)
        if float(reliability.abs().sum().item()) > 0.0:
            score = score + _z_score(reliability)
        return torch.topk(score, k=num_keep, largest=True).indices

    return _train_loop(
        embeddings,
        noisy_labels,
        clean_labels,
        train_idx,
        val_idx,
        test_idx,
        num_classes,
        device,
        seed,
        epochs,
        lr,
        wd,
        patience,
        ce_loss,
        selection_fn=select,
        trace=trace,
    )


def train_with_reweight(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    reliability,
    num_classes,
    device,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    warmup: int = 20,
    ramp: int = 20,
    w_min: float = 0.1,
    seed: int = 0,
    trace: bool = False,
):
    """Train a linear classifier with reliability-guided loss reweighting."""
    embeddings = embeddings.detach().float().to(device)
    noisy_labels = noisy_labels.detach().long().to(device)
    clean_labels = clean_labels.detach().long().to(device)
    reliability = reliability.detach().float().to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
    clean_mask_train = noisy_labels[train_idx_t] == clean_labels[train_idx_t]
    noisy_mask_train = ~clean_mask_train

    model = _init_linear(embeddings.size(1), num_classes, seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = -1.0
    best_test = 0.0
    stale = 0
    loss_history = []
    trace_rows = [] if trace else None
    last_weights = torch.ones(len(train_idx_t), dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(embeddings[train_idx_t])
        per_sample = F.cross_entropy(logits, noisy_labels[train_idx_t], reduction="none")
        trust = _z_score(-per_sample.detach())
        if float(reliability.abs().sum().item()) > 0.0:
            trust = trust + _z_score(reliability)
        w_raw = (trust - trust.min()) / (trust.max() - trust.min() + 1e-12)
        weights = float(w_min) + (1.0 - float(w_min)) * w_raw
        progress = _ramp_progress(epoch, warmup, ramp)
        actual_w = (1.0 - progress) * torch.ones_like(weights) + progress * weights
        last_weights = actual_w.detach()
        loss = (actual_w * per_sample).sum() / actual_w.sum().clamp_min(1e-12)
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(embeddings[val_idx_t]), clean_labels[val_idx_t])
            test_acc = _accuracy(model(embeddings[test_idx_t]), clean_labels[test_idx_t])
            train_acc = _accuracy(model(embeddings[train_idx_t]), clean_labels[train_idx_t])

        high_weight = last_weights > 0.5
        selected_clean_rate = float(clean_mask_train[high_weight].float().mean().item()) if int(high_weight.sum().item()) > 0 else 0.0
        if trace_rows is not None:
            trace_rows.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "keep_ratio": 1.0,
                "selected_count": int(len(train_idx_t)),
                "selected_clean_rate": selected_clean_rate,
                "weight_min": float(last_weights.min().item()),
                "weight_mean": float(last_weights.mean().item()),
                "weight_max": float(last_weights.max().item()),
            })

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    high_weight = last_weights > 0.5
    low_weight = last_weights < 0.5
    selection_precision = float(clean_mask_train[high_weight].float().mean().item()) if int(high_weight.sum().item()) > 0 else 0.0
    if int(noisy_mask_train.sum().item()) == 0:
        noise_exclusion = 0.0
    else:
        noise_exclusion = float((noisy_mask_train & low_weight).float().sum().item() / noisy_mask_train.float().sum().item())
    result = {
        "test_acc_at_best_val": float(best_test),
        "selection_precision": selection_precision,
        "noise_exclusion_recall": noise_exclusion,
        "loss_history": loss_history,
    }
    if trace_rows is not None:
        result["trace"] = trace_rows
        result["stopped_epoch"] = epoch
    return result


def train_with_smoothing(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    soft_targets,
    num_classes,
    device,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    warmup: int = 20,
    ramp: int = 20,
    seed: int = 0,
    trace: bool = False,
):
    """Train a linear classifier with neighborhood-vote soft targets."""
    embeddings = embeddings.detach().float().to(device)
    noisy_labels = noisy_labels.detach().long().to(device)
    clean_labels = clean_labels.detach().long().to(device)
    soft_targets = soft_targets.detach().float().to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long, device=device)
    if soft_targets.shape != (len(train_idx_t), num_classes):
        raise ValueError("soft_targets must have shape [num_train, num_classes]")

    model = _init_linear(embeddings.size(1), num_classes, seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    one_hot = F.one_hot(noisy_labels[train_idx_t], num_classes=num_classes).float()
    best_val = -1.0
    best_test = 0.0
    stale = 0
    loss_history = []
    trace_rows = [] if trace else None

    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(embeddings[train_idx_t])
        progress = _ramp_progress(epoch, warmup, ramp)
        target = (1.0 - progress) * one_hot + progress * soft_targets
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        log_prob = F.log_softmax(logits, dim=-1)
        loss = -(target * log_prob).sum(dim=-1).mean()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(embeddings[val_idx_t]), clean_labels[val_idx_t])
            test_acc = _accuracy(model(embeddings[test_idx_t]), clean_labels[test_idx_t])
            train_acc = _accuracy(model(embeddings[train_idx_t]), clean_labels[train_idx_t])
            entropy = -(target * (target + 1e-12).log()).sum(dim=-1).mean().item()

        if trace_rows is not None:
            trace_rows.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "keep_ratio": 1.0,
                "selected_count": int(len(train_idx_t)),
                "selected_clean_rate": 0.0,
                "target_entropy": float(entropy),
            })

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    result = {
        "test_acc_at_best_val": float(best_test),
        "selection_precision": 0.0,
        "noise_exclusion_recall": 0.0,
        "loss_history": loss_history,
    }
    if trace_rows is not None:
        result["trace"] = trace_rows
        result["stopped_epoch"] = epoch
    return result


def train_ce_baseline(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    num_classes,
    device,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    seed: int = 0,
    trace: bool = False,
):
    """Train standard CE on noisy train labels."""

    def ce_loss(logits, labels):
        return F.cross_entropy(logits, labels, reduction="none")

    return _train_loop(embeddings, noisy_labels, clean_labels, train_idx, val_idx, test_idx, num_classes, device, seed, epochs, lr, wd, patience, ce_loss, trace=trace)


def train_sce_baseline(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    num_classes,
    device,
    alpha: float = 1.0,
    beta: float = 1.0,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    seed: int = 0,
):
    """Train symmetric cross entropy baseline."""

    def sce_loss(logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pred = torch.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot = F.one_hot(labels, num_classes=num_classes).float().clamp(min=1e-4, max=1.0)
        rce = -(pred * torch.log(one_hot)).sum(dim=1)
        return alpha * ce + beta * rce

    return _train_loop(embeddings, noisy_labels, clean_labels, train_idx, val_idx, test_idx, num_classes, device, seed, epochs, lr, wd, patience, sce_loss)


def train_clean_oracle(
    embeddings,
    noisy_labels,
    clean_labels,
    train_idx,
    val_idx,
    test_idx,
    num_classes,
    device,
    epochs: int = 200,
    lr: float = 0.01,
    wd: float = 5e-4,
    patience: int = 30,
    seed: int = 0,
):
    """Train CE only on clean train samples for diagnostic headroom."""
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    clean_mask = noisy_labels[train_idx_t] == clean_labels[train_idx_t]
    clean_train_idx = train_idx_t[clean_mask].tolist()
    return train_ce_baseline(embeddings, clean_labels, clean_labels, clean_train_idx, val_idx, test_idx, num_classes, device, epochs, lr, wd, patience, seed)
