from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .targets import TargetResult, _consensus_weight, _indices


def build_self_vote_ablation_targets(
    noisy_labels: torch.Tensor,
    assignment: np.ndarray,
    train_mask: torch.Tensor | np.ndarray,
    n_classes: int,
    *,
    min_train_per_ball: int = 2,
    beta: float = 0.5,
) -> TargetResult:
    """Build the tie-safe self-vote ablation reported in Appendix D.1."""
    labels = noisy_labels.long()
    device = labels.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != labels.shape[0]:
        raise ValueError("assignment must match the number of labels")

    train_index = _indices(train_mask)
    targets = F.one_hot(labels, num_classes=int(n_classes)).float()
    alpha = torch.zeros(labels.shape[0], dtype=torch.float32, device=device)
    pseudo = labels.detach().clone()
    eligible_nodes = 0
    tied_nodes = 0

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_index[assignment[train_index] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        nodes = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(labels[nodes], minlength=int(n_classes))
        total = int(counts.sum().item())
        maximum = int(counts.max().item())
        node_alpha = _consensus_weight(maximum / float(total), int(n_classes))
        if node_alpha <= 0.0:
            continue

        distribution = counts.float() / float(total)
        tied = int((counts == maximum).sum().item()) > 1
        if tied:
            local_target = distribution
            tied_nodes += int(nodes.numel())
        else:
            majority = int(torch.argmax(counts).item())
            majority_target = F.one_hot(
                torch.tensor(majority, device=device), num_classes=int(n_classes)
            ).float()
            local_target = (1.0 - float(beta)) * distribution + float(beta) * majority_target
            pseudo[nodes] = majority

        targets[nodes] = (1.0 - node_alpha) * targets[nodes] + node_alpha * local_target
        alpha[nodes] = node_alpha
        eligible_nodes += int(nodes.numel())

    return TargetResult(
        soft_target=targets,
        alpha=alpha,
        pseudo_label=pseudo,
        diagnostics={
            "leave_one_out": 0,
            "tie_mode": "distribution_only",
            "target_beta": float(beta),
            "eligible_train_nodes": int(eligible_nodes),
            "tied_train_nodes": int(tied_nodes),
            "num_balls": int(np.unique(assignment).size),
        },
    )
