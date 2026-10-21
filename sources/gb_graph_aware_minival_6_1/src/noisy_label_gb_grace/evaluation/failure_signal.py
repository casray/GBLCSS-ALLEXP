"""Label-free diagnostics for local-consistency boundary analysis.

These metrics intentionally use only frozen embeddings, noisy labels, and masks. Clean labels are
reserved for post-hoc evaluation elsewhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.utils import mask_to_index


def _knn_train_neighbor_indices(embeddings: torch.Tensor, train_mask: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    if train_idx.numel() <= 1 or k <= 0:
        return train_idx, torch.empty((int(train_idx.numel()), 0), dtype=torch.long)
    train_emb = emb[train_idx]
    sim = train_emb @ train_emb.t()
    sim.fill_diagonal_(-2.0)
    kk = min(int(k), int(train_idx.numel()) - 1)
    return train_idx, train_idx[sim.topk(kk, dim=1).indices]


def _majority_label(labels: torch.Tensor, idx: torch.Tensor, num_classes: int | None = None) -> tuple[int, float]:
    if idx.numel() == 0:
        return -1, math.nan
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1 if labels.numel() else 0
    counts = torch.bincount(labels[idx], minlength=int(num_classes)).float()
    total = counts.sum().clamp_min(1.0)
    label = int(counts.argmax().item())
    confidence = float((counts.max() / total).item())
    return label, confidence


def neighborhood_majority_self_disagreement(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    k: int,
) -> float:
    """Fraction of train nodes whose kNN noisy-label majority differs from their noisy label."""
    labels = noisy_labels.detach().cpu().long()
    train_idx, neighbors = _knn_train_neighbor_indices(embeddings, train_mask, k)
    if train_idx.numel() == 0 or neighbors.numel() == 0:
        return 0.0
    num_classes = int(labels.max().item()) + 1
    disagree = 0
    valid = 0
    for row, node in enumerate(train_idx.tolist()):
        majority, _ = _majority_label(labels, neighbors[row], num_classes)
        if majority < 0:
            continue
        valid += 1
        disagree += int(majority != int(labels[node].item()))
    return float(disagree / max(1, valid))


def neighborhood_majority_confidence(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    k: int,
) -> float:
    """Mean kNN noisy-label majority fraction over train nodes."""
    labels = noisy_labels.detach().cpu().long()
    train_idx, neighbors = _knn_train_neighbor_indices(embeddings, train_mask, k)
    if train_idx.numel() == 0 or neighbors.numel() == 0:
        return 0.0
    num_classes = int(labels.max().item()) + 1
    confidences = []
    for row in range(int(train_idx.numel())):
        _, confidence = _majority_label(labels, neighbors[row], num_classes)
        if not math.isnan(confidence):
            confidences.append(confidence)
    return float(sum(confidences) / len(confidences)) if confidences else 0.0


def pairflip_neighborhood_consistency(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    k: int,
    num_classes: int,
) -> float:
    """Mean fraction of kNN train neighbors sharing each node's own noisy label."""
    labels = noisy_labels.detach().cpu().long()
    train_idx, neighbors = _knn_train_neighbor_indices(embeddings, train_mask, k)
    if train_idx.numel() == 0 or neighbors.numel() == 0:
        return 0.0
    scores = []
    for row, node in enumerate(train_idx.tolist()):
        nbr = neighbors[row]
        if nbr.numel() == 0:
            continue
        counts = torch.bincount(labels[nbr], minlength=int(num_classes)).float()
        scores.append(float((counts[int(labels[node].item())] / counts.sum().clamp_min(1.0)).item()))
    return float(sum(scores) / len(scores)) if scores else 0.0


def boundary_warning_metrics(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    k: int,
    num_classes: int,
    prefix: str = "",
) -> dict[str, float]:
    """Return label-free warning metrics for one setting/seed."""
    return {
        f"{prefix}neighborhood_majority_self_disagreement": neighborhood_majority_self_disagreement(
            embeddings, noisy_labels, train_mask, k
        ),
        f"{prefix}neighborhood_majority_confidence": neighborhood_majority_confidence(
            embeddings, noisy_labels, train_mask, k
        ),
        f"{prefix}pairflip_neighborhood_consistency": pairflip_neighborhood_consistency(
            embeddings, noisy_labels, train_mask, k, num_classes
        ),
    }
