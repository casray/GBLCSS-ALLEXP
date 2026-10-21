from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class GranularBallResult:
    assignment: np.ndarray
    balls: list[dict[str, float | int]]
    summary: dict[str, float | int]


def _as_normalized_array(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().float().numpy()
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _measure_ball(z: np.ndarray, indices: np.ndarray, ball_id: int) -> dict[str, float | int]:
    center = z[indices].mean(axis=0)
    distances = np.linalg.norm(z[indices] - center, axis=1)
    return {
        "ball_id": int(ball_id),
        "size": int(indices.size),
        "radius": float(distances.max()) if distances.size else 0.0,
        "avg_distance": float(distances.mean()) if distances.size else 0.0,
    }


def build_granular_balls(
    embeddings: torch.Tensor | np.ndarray,
    *,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    random_state: int = 0,
) -> GranularBallResult:
    """Construct the fixed-budget, label-agnostic granular-ball partition."""
    z = _as_normalized_array(embeddings)
    n_nodes = int(z.shape[0])
    max_balls = max(1, min(int(max_balls), n_nodes))
    min_ball_size = max(1, int(min_ball_size))
    split_improvement = max(0.0, float(split_improvement))
    base_seed = int(random_state)

    balls = [np.arange(n_nodes)]
    iteration = 0
    while len(balls) < max_balls:
        candidates: list[tuple[float, int, dict[str, float | int]]] = []
        for position, indices in enumerate(balls):
            if indices.size < 2 * min_ball_size:
                continue
            stat = _measure_ball(z, indices, position)
            candidates.append((float(stat["avg_distance"]), position, stat))
        if not candidates:
            break

        accepted = False
        for _, position, parent_stat in sorted(candidates, reverse=True):
            indices = balls[position]
            labels = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=base_seed + iteration + position,
            ).fit_predict(z[indices])
            if np.unique(labels).size < 2:
                continue
            left = indices[labels == 0]
            right = indices[labels == 1]
            if left.size < min_ball_size or right.size < min_ball_size:
                continue

            left_stat = _measure_ball(z, left, 0)
            right_stat = _measure_ball(z, right, 1)
            new_average = (
                float(left_stat["avg_distance"]) * left.size
                + float(right_stat["avg_distance"]) * right.size
            ) / indices.size
            old_average = float(parent_stat["avg_distance"])
            if old_average <= 1e-12 or new_average <= old_average * (1.0 - split_improvement):
                balls[position] = left
                balls.append(right)
                accepted = True
                break
        if not accepted:
            break
        iteration += 1

    assignment = np.empty(n_nodes, dtype=np.int64)
    statistics: list[dict[str, float | int]] = []
    for ball_id, indices in enumerate(balls):
        assignment[indices] = ball_id
        statistics.append(_measure_ball(z, indices, ball_id))

    sizes = np.asarray([item["size"] for item in statistics], dtype=np.float32)
    radii = np.asarray([item["radius"] for item in statistics], dtype=np.float32)
    return GranularBallResult(
        assignment=assignment,
        balls=statistics,
        summary={
            "num_balls": int(len(statistics)),
            "mean_ball_size": float(sizes.mean()),
            "max_ball_size": int(sizes.max()),
            "mean_radius": float(radii.mean()),
        },
    )


def random_size_matched_assignment(assignment: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Randomize memberships while preserving every granular-ball size."""
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1:
        raise ValueError("assignment must be one-dimensional")
    rng = np.random.RandomState(int(random_state))
    nodes = np.arange(assignment.shape[0])
    rng.shuffle(nodes)
    shuffled = np.empty_like(assignment)
    offset = 0
    for ball_id in sorted(np.unique(assignment).tolist()):
        size = int(np.sum(assignment == ball_id))
        shuffled[nodes[offset : offset + size]] = ball_id
        offset += size
    return shuffled
