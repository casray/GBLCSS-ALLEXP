from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
import torch


@dataclass(frozen=True)
class AdaptiveGranularBallResult:
    assignment: np.ndarray
    balls: list[dict[str, float | int]]
    summary: dict[str, Any]
    split_trace: list[dict[str, Any]]


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _normalized_array(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().double().numpy()
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("embeddings must be a finite non-empty 2-D array")
    # Scale first so finite, very large inputs cannot overflow their row norms.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = values / np.maximum(scale, np.finfo(np.float64).tiny)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, np.finfo(np.float64).tiny)


def _training_mask(train_index: Any, n_nodes: int) -> np.ndarray | None:
    if train_index is None:
        return None
    if isinstance(train_index, torch.Tensor):
        train_index = train_index.detach().cpu().numpy()
    indices = np.asarray(train_index)
    if indices.ndim != 1:
        raise ValueError("train_index must be a one-dimensional integer index array")
    if indices.size and (indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= n_nodes)):
        raise ValueError("train_index must contain integer indices in range")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_index must contain unique indices")
    mask = np.zeros(n_nodes, dtype=bool)
    mask[indices.astype(np.int64)] = True
    return mask


def _sse(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0)
    return float(np.einsum("ij,ij->", centered, centered))


def _measure(z: np.ndarray, indices: np.ndarray, ball_id: int, train: np.ndarray | None) -> dict[str, float | int]:
    centered = z[indices] - z[indices].mean(axis=0)
    squared = np.einsum("ij,ij->i", centered, centered)
    distances = np.sqrt(squared)
    stat: dict[str, float | int] = {
        "ball_id": ball_id,
        "size": int(indices.size),
        "radius": float(distances.max()),
        "avg_distance": float(distances.mean()),
        "sse": float(squared.sum()),
    }
    if train is not None:
        stat["train_count"] = int(train[indices].sum())
    return stat


def _constraint_reason(left: np.ndarray, right: np.ndarray, min_size: int, train: np.ndarray | None, min_train: int) -> str | None:
    if min(left.size, right.size) < min_size:
        return "child_size_constraint"
    if train is not None and min(int(train[left].sum()), int(train[right].sum())) < min_train:
        return "child_support_constraint"
    return None


def _projection_split(z: np.ndarray, indices: np.ndarray, direction: np.ndarray, min_size: int, train: np.ndarray | None, min_train: int) -> tuple[np.ndarray, np.ndarray] | None:
    values = z[indices]
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        axis = int(np.argmax(np.ptp(values, axis=0)))
        if np.ptp(values[:, axis]) <= 1e-12:
            return None
        direction = np.zeros(values.shape[1])
        direction[axis] = 1.
        norm = 1.
    coordinates = values @ (direction / norm)
    order = np.argsort(coordinates, kind="stable")
    ordered = values[order]
    ordered_indices = indices[order]
    cuts = np.arange(min_size, indices.size - min_size + 1)
    distinct = np.diff(coordinates[order]) > 1e-12
    cuts = cuts[distinct[cuts - 1]]
    if train is not None:
        cumulative_train = np.cumsum(train[ordered_indices], dtype=np.int64)
        counts = cumulative_train[cuts - 1]
        cuts = cuts[(counts >= min_train) & (cumulative_train[-1] - counts >= min_train)]
    if not cuts.size:
        return None
    # Center before prefix sums to reduce cancellation in the SSE cut objective.
    ordered = ordered - ordered.mean(axis=0)
    sums = np.cumsum(ordered, axis=0)
    squared_sums = np.cumsum(np.einsum("ij,ij->i", ordered, ordered))
    left_sums = sums[cuts - 1]
    right_sums = sums[-1] - left_sums
    costs = (
        squared_sums[cuts - 1] - np.einsum("ij,ij->i", left_sums, left_sums) / cuts
        + squared_sums[-1] - squared_sums[cuts - 1]
        - np.einsum("ij,ij->i", right_sums, right_sums) / (indices.size - cuts)
    )
    cut = int(cuts[int(np.argmin(costs))])
    return ordered_indices[:cut], ordered_indices[cut:]


def build_adaptive_granular_balls(
    embeddings: torch.Tensor | np.ndarray,
    *,
    target_balls: int | None = None,
    train_index: Any = None,
    min_train_per_ball: int = 4,
    random_state: int = 0,
) -> AdaptiveGranularBallResult:
    """Build label-free balls by best-first SSE splits of normalized embeddings.

    Training indices constrain support only; no noisy or clean labels are inputs.
    ``feasible_target`` is a count ceiling, not a promise of geometric feasibility.
    """
    z = _normalized_array(embeddings)
    n_nodes = int(z.shape[0])
    requested = math.ceil(math.sqrt(n_nodes)) if target_balls is None else _integer(target_balls, "target_balls")
    min_train = _integer(min_train_per_ball, "min_train_per_ball")
    seed = _integer(random_state, "random_state", minimum=0)
    train = _training_mask(train_index, n_nodes)
    min_size = max(1, math.ceil(n_nodes / (4 * requested)))
    geometry_cap = max(1, n_nodes // min_size)
    support_cap = None if train is None else max(1, int(train.sum()) // min_train)
    feasible = min(requested, geometry_cap, support_cap if support_cap is not None else n_nodes)
    root = np.arange(n_nodes, dtype=np.int64)
    leaves = {0: root}
    heap = [(-_sse(z), 0)]
    next_id = 1
    trace: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    kmeans_constraint_failures: Counter[str] = Counter()
    fallback_attempts = 0

    while len(leaves) < feasible and heap:
        negative_sse, leaf_id = heapq.heappop(heap)
        indices = leaves[leaf_id]
        parent_sse = -negative_sse
        threshold = max(1e-12, 1e-8 * parent_sse)
        row: dict[str, Any] = {"parent_id": leaf_id, "parent_size": int(indices.size), "parent_sse": parent_sse, "accepted": False}
        reason = None
        if parent_sse <= threshold:
            reason = "zero_geometric_gain"
        elif indices.size < 2 * min_size:
            reason = "parent_size_constraint"
        elif train is not None and int(train[indices].sum()) < 2 * min_train:
            reason = "parent_support_constraint"
        if reason is not None:
            row["reason"] = reason
            rejections[reason] += 1
            trace.append(row)
            continue

        model = KMeans(n_clusters=2, n_init=10, random_state=(seed + leaf_id) % (2**32 - 1))
        labels = model.fit_predict(z[indices])
        left, right = indices[labels == 0], indices[labels == 1]
        reason = _constraint_reason(left, right, min_size, train, min_train)
        method = "kmeans"
        if reason is not None:
            kmeans_constraint_failures[reason] += 1
            fallback_attempts += 1
            row["kmeans_rejection"] = reason
            split = _projection_split(z, indices, model.cluster_centers_[1] - model.cluster_centers_[0], min_size, train, min_train)
            if split is None:
                reason = "no_distinct_feasible_projection_boundary"
            else:
                left, right = split
                reason = None
                method = "projection"
        if reason is None:
            child_sse = [_sse(z[left]), _sse(z[right])]
            gain = parent_sse - sum(child_sse)
            if gain <= threshold:
                reason = "zero_geometric_gain"
        row["method"] = method
        if reason is not None:
            row["reason"] = reason
            rejections[reason] += 1
            trace.append(row)
            continue

        child_ids = [next_id, next_id + 1]
        next_id += 2
        leaves.pop(leaf_id)
        for child_id, child, child_cost in zip(child_ids, (left, right), child_sse):
            leaves[child_id] = child
            heapq.heappush(heap, (-child_cost, child_id))
        row.update(accepted=True, child_ids=child_ids, child_sizes=[int(left.size), int(right.size)], child_sse=child_sse, sse_gain=gain)
        if train is not None:
            row["child_train_counts"] = [int(train[left].sum()), int(train[right].sum())]
        trace.append(row)

    assignment = np.empty(n_nodes, dtype=np.int64)
    statistics = []
    for ball_id, leaf_id in enumerate(sorted(leaves)):
        indices = leaves[leaf_id]
        assignment[indices] = ball_id
        stat = _measure(z, indices, ball_id, train)
        stat["leaf_id"] = leaf_id
        statistics.append(stat)
    sizes = np.asarray([stat["size"] for stat in statistics])
    if train is not None and int(train.sum()) < min_train:
        stop = "insufficient_training_support"
    else:
        stop = "target_reached" if len(leaves) == feasible else "no_feasible_split"
    summary: dict[str, Any] = {
        "requested_target": requested,
        "feasible_target": feasible,
        "num_balls": len(leaves),
        "min_leaf_size": min_size,
        "geometry_count_cap": geometry_cap,
        "min_ball_size": int(sizes.min()),
        "max_ball_size": int(sizes.max()),
        "mean_ball_size": float(sizes.mean()),
        "median_ball_size": float(np.median(sizes)),
        "mean_radius": float(np.mean([stat["radius"] for stat in statistics])),
        "total_sse": float(sum(stat["sse"] for stat in statistics)),
        "stop_reason": stop,
        "rejection_counts": dict(sorted(rejections.items())),
        "kmeans_constraint_failures": dict(sorted(kmeans_constraint_failures.items())),
        "fallback_attempts": fallback_attempts,
        "accepted_splits": len(leaves) - 1,
        "rejected_splits": sum(rejections.values()),
        "random_state": seed,
    }
    if train is not None:
        summary.update(num_training_nodes=int(train.sum()), min_train_per_ball=min_train, support_count_cap=support_cap, min_train_count=min(int(stat["train_count"]) for stat in statistics))
    return AdaptiveGranularBallResult(assignment, statistics, summary, trace)
