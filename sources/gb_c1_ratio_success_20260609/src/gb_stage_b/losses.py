from __future__ import annotations

import torch
import torch.nn.functional as F


def _require_same_shape(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(f"expected tensors with the same shape, got {tuple(left.shape)} and {tuple(right.shape)}")
    if left.ndim != 2:
        raise ValueError("expected rank-2 tensors with shape [N, D]")


def _negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _require_same_shape(p, z)
    p = F.normalize(p, dim=-1, eps=1e-8)
    z = F.normalize(z.detach(), dim=-1, eps=1e-8)
    return -(p * z).sum(dim=-1).mean()


def negative_free_loss(p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    _require_same_shape(p1, z2)
    _require_same_shape(p2, z1)
    return 0.5 * (_negative_cosine(p1, z2) + _negative_cosine(p2, z1))


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    _require_same_shape(z1, z2)

    z1 = F.normalize(z1, dim=-1, eps=1e-8)
    z2 = F.normalize(z2, dim=-1, eps=1e-8)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_12 + loss_21)


def variance_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("expected rank-2 tensor with shape [N, D]")
    if target_std <= 0:
        raise ValueError("target_std must be positive")
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def symmetric_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Symmetric cross entropy for noisy-label classification."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if labels.ndim != 1 or labels.numel() != logits.size(0):
        raise ValueError("labels must have shape [N]")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")
    if not 0 < eps < 1:
        raise ValueError("eps must be in (0, 1)")

    ce = F.cross_entropy(logits, labels.long())
    pred = F.softmax(logits, dim=-1).clamp(min=eps, max=1.0)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, labels.long().unsqueeze(1), 1.0)
    one_hot = one_hot.clamp(min=eps, max=1.0)
    rce = -(pred * torch.log(one_hot)).sum(dim=-1).mean()
    return alpha * ce + beta * rce


def ball_majority_vote(
    assignment: torch.Tensor,
    noisy_labels: torch.Tensor,
    reliable_balls: torch.Tensor,
    num_classes: int,
    train_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace each reliable-ball node's label with the ball's majority-vote label.

    Only train_mask members' labels are counted in the vote to prevent val/test label leakage.
    """
    corrected = noisy_labels.clone()
    reliable_balls = reliable_balls.to(dtype=torch.bool, device=assignment.device)
    num_balls = reliable_balls.numel()
    if train_mask is not None:
        train_mask = train_mask.to(dtype=torch.bool, device=assignment.device)

    for b in range(num_balls):
        if not reliable_balls[b]:
            continue
        members = (assignment == b).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        if train_mask is not None:
            vote_members = members[train_mask[members]]
            if vote_members.numel() == 0:
                continue
        else:
            vote_members = members
        vote_labels = noisy_labels[vote_members]
        counts = torch.zeros(num_classes, dtype=torch.long, device=noisy_labels.device)
        counts.scatter_add_(0, vote_labels.long(), torch.ones_like(vote_labels, dtype=torch.long))
        majority = counts.argmax()
        corrected[members] = majority
    return corrected


def ball_label_smoothing(
    assignment: torch.Tensor,
    noisy_labels: torch.Tensor,
    reliable_balls: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.1,
    train_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Produce soft targets blending one-hot with ball-level label distribution.

    Only train_mask members' labels are used to compute the ball distribution
    to prevent val/test label leakage.
    """
    N = noisy_labels.size(0)
    device = noisy_labels.device
    one_hot = torch.zeros(N, num_classes, device=device)
    one_hot.scatter_(1, noisy_labels.long().unsqueeze(1), 1.0)
    soft_targets = one_hot.clone()
    reliable_balls = reliable_balls.to(dtype=torch.bool, device=device)
    num_balls = reliable_balls.numel()
    if train_mask is not None:
        train_mask = train_mask.to(dtype=torch.bool, device=device)

    for b in range(num_balls):
        if not reliable_balls[b]:
            continue
        members = (assignment == b).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        if train_mask is not None:
            vote_members = members[train_mask[members]]
            if vote_members.numel() == 0:
                continue
        else:
            vote_members = members
        vote_labels = noisy_labels[vote_members]
        counts = torch.zeros(num_classes, dtype=torch.float, device=device)
        counts.scatter_add_(0, vote_labels.long(), torch.ones(vote_members.numel(), device=device))
        ball_dist = counts / counts.sum()
        member_one_hot = one_hot[members]
        soft_targets[members] = (1.0 - smoothing) * member_one_hot + smoothing * ball_dist.unsqueeze(0)
    return soft_targets


def gated_ball_soft_correction(
    assignment: torch.Tensor,
    noisy_labels: torch.Tensor,
    reliable_balls: torch.Tensor,
    num_classes: int,
    train_mask: torch.Tensor,
    tau_cov: int = 3,
    tau_conf: float = 0.6,
    alpha: float = 0.3,
) -> torch.Tensor:
    """Blend noisy one-hot targets with train-only ball label distributions behind a gate."""
    if tau_cov < 1:
        raise ValueError("tau_cov must be at least 1")
    if not 0 <= tau_conf <= 1:
        raise ValueError("tau_conf must be in [0, 1]")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    N = noisy_labels.size(0)
    device = noisy_labels.device
    one_hot = torch.zeros(N, num_classes, device=device)
    one_hot.scatter_(1, noisy_labels.long().unsqueeze(1), 1.0)
    soft_targets = one_hot.clone()
    reliable_balls = reliable_balls.to(dtype=torch.bool, device=device)
    train_mask = train_mask.to(dtype=torch.bool, device=device)

    for b in range(reliable_balls.numel()):
        if not reliable_balls[b]:
            continue
        members = (assignment == b).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        vote_members = members[train_mask[members]]
        if vote_members.numel() < tau_cov:
            continue

        vote_labels = noisy_labels[vote_members]
        counts = torch.zeros(num_classes, dtype=torch.float, device=device)
        counts.scatter_add_(0, vote_labels.long(), torch.ones(vote_members.numel(), device=device))
        total = counts.sum()
        confidence = counts.max() / total
        if confidence < tau_conf:
            continue

        ball_dist = counts / total
        soft_targets[members] = (1.0 - alpha) * one_hot[members] + alpha * ball_dist.unsqueeze(0)
    return soft_targets


def ball_reliability_weights(
    assignment: torch.Tensor,
    noisy_labels: torch.Tensor,
    reliable_balls: torch.Tensor,
    train_mask: torch.Tensor,
    weighted_k_ref: int = 3,
    weighted_floor: float = 0.25,
) -> torch.Tensor:
    """Compute per-node ball reliability weights from train-only coverage and vote margin."""
    if weighted_k_ref < 1:
        raise ValueError("weighted_k_ref must be at least 1")
    if not 0 <= weighted_floor <= 1:
        raise ValueError("weighted_floor must be in [0, 1]")

    device = noisy_labels.device
    weights = torch.ones(noisy_labels.size(0), dtype=torch.float, device=device)
    reliable_balls = reliable_balls.to(dtype=torch.bool, device=device)
    train_mask = train_mask.to(dtype=torch.bool, device=device)

    for b in range(reliable_balls.numel()):
        if not reliable_balls[b]:
            continue
        members = (assignment == b).nonzero(as_tuple=True)[0]
        if members.numel() == 0:
            continue
        vote_members = members[train_mask[members]]
        if vote_members.numel() == 0:
            continue

        vote_labels = noisy_labels[vote_members].long()
        counts = torch.bincount(vote_labels, minlength=int(noisy_labels.max().item()) + 1).to(device=device, dtype=torch.float)
        margin = counts.max() / counts.sum()
        coverage = min(1.0, float(vote_members.numel()) / float(weighted_k_ref))
        weight = max(float(weighted_floor), coverage * float(margin.item()))
        weights[members] = weight
    return weights


def weighted_soft_target_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cross entropy for soft targets."""
    if logits.shape != soft_targets.shape:
        raise ValueError("logits and soft_targets must have the same shape")
    if weights.ndim != 1 or weights.numel() != logits.size(0):
        raise ValueError("weights must have shape [N]")
    per_node = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    weights = weights.to(device=logits.device, dtype=per_node.dtype)
    return (per_node * weights).sum() / weights.sum().clamp(min=1e-12)


def inter_ball_separation_loss(
    centers: torch.Tensor,
    reliable_balls: torch.Tensor,
) -> torch.Tensor:
    """Push reliable ball centers apart in angular space."""
    reliable_balls = reliable_balls.to(dtype=torch.bool, device=centers.device)
    reliable_idx = reliable_balls.nonzero(as_tuple=True)[0]
    if reliable_idx.numel() < 2:
        return centers.sum() * 0.0

    sel = F.normalize(centers[reliable_idx], dim=-1, eps=1e-8)
    cos_sim = sel @ sel.t()
    cos_sim.clamp_(-1.0 + 1e-7, 1.0 - 1e-7)
    ang = torch.acos(cos_sim)
    mask = ~torch.eye(sel.size(0), dtype=torch.bool, device=sel.device)
    return -ang[mask].mean()


def knn_majority_vote(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int,
    k: int = 10,
    train_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace each node's label with the majority vote of its k nearest train neighbors.

    Only train_mask nodes are used as candidate neighbors to prevent val/test label leakage.
    """
    emb_norm = F.normalize(embeddings.detach(), dim=-1, eps=1e-8)
    device = noisy_labels.device

    if train_mask is not None:
        train_mask = train_mask.to(dtype=torch.bool, device=device)
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        actual_k = min(k, max(0, train_idx.numel() - 1))
        if actual_k < 1:
            return noisy_labels.clone()
        sim = emb_norm @ emb_norm[train_idx].t()
        for i in range(train_idx.numel()):
            sim[train_idx[i], i] = -float("inf")
        _, topk_local = sim.topk(actual_k, dim=-1)
        topk_idx = train_idx[topk_local]
    else:
        sim = emb_norm @ emb_norm.t()
        sim.fill_diagonal_(-float("inf"))
        _, topk_idx = sim.topk(k, dim=-1)

    corrected = noisy_labels.clone()
    for i in range(embeddings.size(0)):
        neighbor_labels = noisy_labels[topk_idx[i]]
        counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        counts.scatter_add_(0, neighbor_labels.long(), torch.ones_like(neighbor_labels, dtype=torch.long))
        corrected[i] = counts.argmax()
    return corrected


def knn_label_smoothing(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    num_classes: int,
    k: int = 10,
    smoothing: float = 0.1,
    train_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Produce soft targets blending one-hot with kNN neighborhood label distribution.

    Only train_mask nodes are used as candidate neighbors to prevent val/test label leakage.
    """
    N = noisy_labels.size(0)
    device = noisy_labels.device
    emb_norm = F.normalize(embeddings.detach(), dim=-1, eps=1e-8)

    one_hot = torch.zeros(N, num_classes, device=device)
    one_hot.scatter_(1, noisy_labels.long().unsqueeze(1), 1.0)

    if train_mask is not None:
        train_mask = train_mask.to(dtype=torch.bool, device=device)
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        actual_k = min(k, max(0, train_idx.numel() - 1))
        if actual_k < 1:
            return one_hot.clone()
        sim = emb_norm @ emb_norm[train_idx].t()
        for i in range(train_idx.numel()):
            sim[train_idx[i], i] = -float("inf")
        _, topk_local = sim.topk(actual_k, dim=-1)
        topk_idx = train_idx[topk_local]
    else:
        sim = emb_norm @ emb_norm.t()
        sim.fill_diagonal_(-float("inf"))
        actual_k = k
        _, topk_idx = sim.topk(k, dim=-1)

    soft_targets = one_hot.clone()

    for i in range(N):
        neighbor_labels = noisy_labels[topk_idx[i]]
        counts = torch.zeros(num_classes, dtype=torch.float, device=device)
        counts.scatter_add_(0, neighbor_labels.long(), torch.ones(actual_k, device=device))
        knn_dist = counts / counts.sum()
        soft_targets[i] = (1.0 - smoothing) * one_hot[i] + smoothing * knn_dist
    return soft_targets
