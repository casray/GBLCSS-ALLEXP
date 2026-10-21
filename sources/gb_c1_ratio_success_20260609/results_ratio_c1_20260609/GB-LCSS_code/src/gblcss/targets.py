from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetResult:
    soft_target: torch.Tensor
    alpha: torch.Tensor
    pseudo_label: torch.Tensor
    diagnostics: dict[str, float | int | str]


def _indices(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    value = torch.as_tensor(mask).detach().cpu()
    if value.dtype == torch.bool:
        return torch.nonzero(value, as_tuple=False).flatten().numpy()
    return value.long().flatten().numpy()


def _consensus_weight(max_fraction: float, n_classes: int) -> float:
    baseline = 1.0 / float(n_classes)
    return float(np.clip((max_fraction - baseline) / (1.0 - baseline), 0.0, 1.0))


def build_leave_one_out_targets(
    noisy_labels: torch.Tensor,
    assignment: np.ndarray,
    train_mask: torch.Tensor | np.ndarray,
    n_classes: int,
    *,
    min_train_per_ball: int = 2,
    beta: float = 0.5,
) -> TargetResult:
    """Build the formal leave-one-out, tie-safe GB-LCSS targets.

    On a tied leave-one-out histogram, the target uses the empirical class
    distribution only. No class index is selected to break the tie.
    """
    labels = noisy_labels.long()
    device = labels.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != labels.shape[0]:
        raise ValueError("assignment must match the number of labels")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must lie in [0, 1]")

    train_index = _indices(train_mask)
    targets = F.one_hot(labels, num_classes=int(n_classes)).float()
    alpha = torch.zeros(labels.shape[0], dtype=torch.float32, device=device)
    pseudo = labels.detach().clone()
    eligible_nodes = 0
    tied_nodes = 0

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_index[assignment[train_index] == ball_id]
        # Each node must retain at least min_train_per_ball votes after removal.
        if ball_train.size < int(min_train_per_ball) + 1:
            continue
        ball_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(labels[ball_tensor], minlength=int(n_classes))
        for node in ball_tensor:
            leave_one_out_counts = counts.clone()
            leave_one_out_counts[labels[node]] -= 1
            total = int(leave_one_out_counts.sum().item())
            maximum = int(leave_one_out_counts.max().item())
            node_alpha = _consensus_weight(maximum / float(total), int(n_classes))
            if node_alpha <= 0.0:
                continue

            tied = int((leave_one_out_counts == maximum).sum().item()) > 1
            distribution = leave_one_out_counts.float() / float(total)
            if tied:
                local_target = distribution
                tied_nodes += 1
            else:
                majority = int(torch.argmax(leave_one_out_counts).item())
                majority_target = F.one_hot(
                    torch.tensor(majority, device=device), num_classes=int(n_classes)
                ).float()
                local_target = (1.0 - float(beta)) * distribution + float(beta) * majority_target
                pseudo[node] = majority

            targets[node] = (1.0 - node_alpha) * targets[node] + node_alpha * local_target
            alpha[node] = node_alpha
            eligible_nodes += 1

    return TargetResult(
        soft_target=targets,
        alpha=alpha,
        pseudo_label=pseudo,
        diagnostics={
            "leave_one_out": 1,
            "tie_mode": "distribution_only",
            "target_beta": float(beta),
            "eligible_train_nodes": int(eligible_nodes),
            "tied_train_nodes": int(tied_nodes),
            "num_balls": int(np.unique(assignment).size),
        },
    )
