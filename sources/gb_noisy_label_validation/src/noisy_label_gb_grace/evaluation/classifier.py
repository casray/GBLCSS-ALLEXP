from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_label_gb_grace.utils import mask_to_index, set_seed, soft_cross_entropy


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    idx = mask_to_index(mask)
    if idx.numel() == 0:
        return 0.0
    pred = logits[idx].argmax(dim=-1)
    return float((pred == labels[idx]).float().mean().item())


def train_frozen_classifier(
    embeddings: torch.Tensor,
    clean_labels: torch.Tensor,
    soft_targets: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    device: torch.device | None = None,
    sample_weights: torch.Tensor | None = None,
) -> dict:
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = embeddings.detach().float().to(device)
    labels = clean_labels.detach().long().to(device)
    targets = soft_targets.detach().float().to(device)
    train_mask = train_mask.detach().bool().to(device)
    val_mask = val_mask.detach().bool().to(device)
    test_mask = test_mask.detach().bool().to(device)
    train_idx = mask_to_index(train_mask)
    weights = sample_weights.detach().float().to(device) if sample_weights is not None else None

    model = LinearClassifier(x.size(1), targets.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_test = 0.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        if weights is None:
            loss = soft_cross_entropy(logits[train_idx], targets[train_idx])
        else:
            log_probs = F.log_softmax(logits[train_idx], dim=-1)
            per_sample = -(targets[train_idx] * log_probs).sum(dim=-1)
            loss = (weights[train_idx] * per_sample).mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            val_acc = _accuracy(logits, labels, val_mask)
            test_acc = _accuracy(logits, labels, test_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x)
        train_acc = _accuracy(final_logits, labels, train_mask)
        val_acc = _accuracy(final_logits, labels, val_mask)
        test_acc = _accuracy(final_logits, labels, test_mask)
    return {
        "best_val_acc": float(best_val),
        "test_acc_at_best_val": float(best_test),
        "best_epoch": int(best_epoch),
        "train_acc_at_best_state": float(train_acc),
        "val_acc_at_best_state": float(val_acc),
        "test_acc_at_best_state": float(test_acc),
    }
