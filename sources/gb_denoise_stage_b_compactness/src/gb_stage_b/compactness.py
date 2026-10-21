from __future__ import annotations

import torch
import torch.nn.functional as F

from gb_stage_b.balls import angular_distance


def compactness_loss(
    z: torch.Tensor,
    assignment: torch.Tensor,
    centers: torch.Tensor,
    reliable_balls: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pull nodes toward detached reliable ball centers in angular space."""
    if z.ndim != 2:
        raise ValueError("z must have shape [N, D]")
    if centers.ndim != 2 or centers.size(1) != z.size(1):
        raise ValueError("centers must have shape [B, D] matching z")
    if assignment.ndim != 1 or assignment.numel() != z.size(0):
        raise ValueError("assignment must be a 1D tensor matching z rows")
    if reliable_balls.ndim != 1 or reliable_balls.numel() != centers.size(0):
        raise ValueError("reliable_balls must be a 1D tensor matching centers")
    if assignment.numel() and (assignment.min().item() < 0 or assignment.max().item() >= centers.size(0)):
        raise ValueError("assignment contains ball ids outside centers")

    assignment = assignment.to(device=z.device, dtype=torch.long)
    reliable_balls = reliable_balls.to(device=z.device, dtype=torch.bool)
    z_norm = F.normalize(z, dim=-1, eps=1e-8)
    centers_norm = F.normalize(centers.detach().to(device=z.device, dtype=z.dtype), dim=-1, eps=1e-8)
    node_reliable = reliable_balls[assignment]

    covered_nodes = int(node_reliable.sum().item())
    if covered_nodes == 0:
        return z_norm.sum() * 0.0, {
            "covered_nodes": 0.0,
            "covered_ratio": 0.0,
            "mean_distance": 0.0,
        }

    selected_z = z_norm[node_reliable]
    selected_centers = centers_norm[assignment[node_reliable]]
    dist = angular_distance(selected_z, selected_centers)
    loss = dist.mean()
    return loss, {
        "covered_nodes": float(covered_nodes),
        "covered_ratio": float(node_reliable.float().mean().item()),
        "mean_distance": float(dist.detach().mean().item()),
    }
