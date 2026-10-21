from __future__ import annotations

import torch
import torch.nn.functional as F

from gb_stage_b.balls import BallBatch


def _diagnostic_purity(assignment: torch.Tensor, labels: torch.Tensor) -> float:
    purities = []
    for ball_id in assignment.unique().tolist():
        idx = assignment == int(ball_id)
        if idx.sum().item() == 0:
            continue
        counts = torch.bincount(labels[idx].long())
        purities.append(float(counts.max().item() / idx.sum().item()))
    return float(sum(purities) / len(purities)) if purities else 0.0


def ball_diagnostics(balls: BallBatch, labels: torch.Tensor | None = None) -> dict[str, float]:
    out = {
        "num_balls": float(balls.num_balls),
        "mean_ball_size": float(balls.sizes.float().mean().item()),
        "max_ball_size": float(balls.sizes.max().item()),
        "singleton_ratio": float((balls.sizes == 1).float().mean().item()),
        "mean_radius": float(balls.radius.float().mean().item()),
        "mean_average_angular_distance": float(balls.average_angular_distance.float().mean().item()),
    }
    if labels is not None:
        if labels.numel() != balls.assignment.numel():
            raise ValueError("labels must match ball assignment length")
        out["label_purity_diagnostic"] = _diagnostic_purity(balls.assignment.detach().cpu(), labels.detach().cpu())
    return out


def representation_diagnostics(z: torch.Tensor) -> dict[str, float]:
    if z.ndim != 2:
        raise ValueError("z must have shape [N, D]")
    z = F.normalize(z.detach().float(), dim=-1, eps=1e-8)
    if z.numel() == 0:
        raise ValueError("z must be non-empty")

    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    probs = s / s.sum().clamp_min(1e-12)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    effective_rank = torch.exp(entropy)
    sim = z @ z.t()
    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
    uniformity = torch.pdist(z).pow(2).mul(-2).exp().mean().log() if z.size(0) > 1 else torch.tensor(0.0)
    return {
        "effective_rank": float(effective_rank.item()),
        "feature_std_mean": float(z.std(dim=0, unbiased=False).mean().item()),
        "mean_pairwise_cosine": float(upper.mean().item()) if upper.numel() else 0.0,
        "embedding_uniformity": float(uniformity.item()),
    }
