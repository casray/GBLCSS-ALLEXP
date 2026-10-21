from __future__ import annotations

import math

import torch
from sklearn.cluster import KMeans


def _group_center_radius(embeddings: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    points = embeddings[indices]
    center = points.mean(dim=0)
    distances = torch.norm(points - center, dim=1)
    radius = float(distances.max().item()) if distances.numel() else 0.0
    mean_radius = float(distances.mean().item()) if distances.numel() else 0.0
    return center, distances, radius, mean_radius


def _split_group(embeddings: torch.Tensor, indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    points = embeddings[indices]
    if points.size(0) < 2:
        return None
    labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(points.numpy())
    labels = torch.tensor(labels, dtype=torch.long)
    left = indices[labels == 0]
    right = indices[labels == 1]
    if left.numel() == 0 or right.numel() == 0:
        return None
    return left, right


def build_granular_balls(
    embeddings: torch.Tensor,
    min_ball_size: int = 5,
    min_radius_gain: float = 0.01,
    max_depth: int = 20,
    seed: int = 0,
) -> dict:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be a [num_nodes, dim] tensor")
    emb = embeddings.detach().cpu().float()
    all_indices = torch.arange(emb.size(0), dtype=torch.long)
    leaves: list[torch.Tensor] = []
    stack: list[tuple[torch.Tensor, int]] = [(all_indices, 0)]

    while stack:
        indices, depth = stack.pop()
        if indices.numel() < 2 * min_ball_size or depth >= max_depth:
            leaves.append(indices)
            continue
        _, _, _, parent_mean_radius = _group_center_radius(emb, indices)
        split = _split_group(emb, indices, seed=seed + depth + int(indices.numel()))
        if split is None:
            leaves.append(indices)
            continue
        left, right = split
        if left.numel() < min_ball_size or right.numel() < min_ball_size:
            leaves.append(indices)
            continue
        _, _, _, left_mean = _group_center_radius(emb, left)
        _, _, _, right_mean = _group_center_radius(emb, right)
        child_mean = (left.numel() * left_mean + right.numel() * right_mean) / indices.numel()
        if parent_mean_radius <= 0 or child_mean <= parent_mean_radius * (1.0 - min_radius_gain):
            stack.append((right, depth + 1))
            stack.append((left, depth + 1))
        else:
            leaves.append(indices)

    group_ids = torch.full((emb.size(0),), -1, dtype=torch.long)
    centers = []
    radii = []
    node_distances = torch.zeros(emb.size(0), dtype=torch.float32)
    groups = []
    for group_id, indices in enumerate(leaves):
        center, distances, radius, _ = _group_center_radius(emb, indices)
        group_ids[indices] = group_id
        node_distances[indices] = distances.float()
        centers.append(center)
        radii.append(radius)
        groups.append(indices.tolist())

    payload = {
        "group_ids": group_ids,
        "groups": groups,
        "centers": torch.stack(centers) if centers else torch.empty((0, emb.size(1))),
        "radii": torch.tensor(radii, dtype=torch.float32),
        "node_distances": node_distances,
        "diagnostics": groups_to_stats(group_ids, groups, torch.tensor(radii, dtype=torch.float32)),
        "config": {
            "min_ball_size": int(min_ball_size),
            "min_radius_gain": float(min_radius_gain),
            "max_depth": int(max_depth),
            "seed": int(seed),
        },
    }
    return payload


def groups_to_stats(group_ids: torch.Tensor, groups: list[list[int]], radii: torch.Tensor | None = None) -> dict:
    sizes = [len(group) for group in groups]
    if not sizes:
        return {
            "ball_count": 0,
            "avg_ball_size": math.nan,
            "max_ball_size": 0,
            "singleton_ball_ratio": math.nan,
            "radius_mean": math.nan,
            "radius_std": math.nan,
        }
    radii = radii if radii is not None else torch.zeros(len(sizes))
    return {
        "ball_count": int(len(sizes)),
        "num_balls": int(len(sizes)),
        "avg_ball_size": float(sum(sizes) / len(sizes)),
        "avg_ball_size_all_nodes": float(sum(sizes) / len(sizes)),
        "max_ball_size": int(max(sizes)),
        "singleton_ball_ratio": float(sum(1 for size in sizes if size == 1) / len(sizes)),
        "radius_mean": float(radii.float().mean().item()),
        "radius_std": float(radii.float().std(unbiased=False).item()) if len(sizes) > 1 else 0.0,
        "non_singleton_avg_size": float(
            sum(size for size in sizes if size > 1) / max(1, sum(1 for size in sizes if size > 1))
        ),
    }
