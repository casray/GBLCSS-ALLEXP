from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch

from .granular_ball_adaptive import (
    _constraint_reason,
    _integer,
    _measure,
    _normalized_array,
    _projection_split,
    _sse,
    _training_mask,
)


@dataclass(frozen=True)
class GranularBallTrajectory:
    snapshots: dict[int, np.ndarray]
    summaries: dict[int, dict[str, Any]]
    split_trace: list[dict[str, Any]]


def build_ball_trajectory(
    embeddings: torch.Tensor | np.ndarray,
    *,
    budgets: Any,
    train_index: Any,
    min_train_per_ball: int = 16,
    min_leaf_size: int = 16,
    random_state: int = 0,
) -> GranularBallTrajectory:
    """Save partitions along one best-first SSE split tree of normalized rows.

    Leaf size and training support are fixed across every requested budget.
    Training indices provide membership support only; class labels are not inputs.
    ``None`` disables the training support constraint. Count caps are upper bounds,
    because positive geometric gain and feasible projection boundaries may run out.
    """
    z = _normalized_array(embeddings)
    try:
        requested = sorted({_integer(value, "budgets") for value in budgets})
    except TypeError as error:
        raise ValueError("budgets must be a non-empty iterable of positive integers") from error
    if not requested:
        raise ValueError("budgets must be a non-empty iterable of positive integers")
    min_train = _integer(min_train_per_ball, "min_train_per_ball")
    min_size = _integer(min_leaf_size, "min_leaf_size")
    seed = _integer(random_state, "random_state", minimum=0)
    n_nodes = int(z.shape[0])
    train = _training_mask(train_index, n_nodes)
    n_train = None if train is None else int(train.sum())
    geometry_cap = max(1, n_nodes // min_size)
    support_cap = None if train is None else max(1, n_train // min_train)
    count_cap = min(geometry_cap, support_cap if support_cap is not None else n_nodes)
    goal = min(requested[-1], count_cap)
    leaves = {0: np.arange(n_nodes, dtype=np.int64)}
    heap = [(-_sse(z), 0)]
    next_id = 1
    trace: list[dict[str, Any]] = []
    snapshots: dict[int, np.ndarray] = {}
    summaries: dict[int, dict[str, Any]] = {}
    rejections: Counter[str] = Counter()
    kmeans_constraint_failures: Counter[str] = Counter()
    fallback_attempts = 0

    def save_snapshot(budget: int) -> None:
        assignment = np.empty(n_nodes, dtype=np.int64)
        statistics = []
        for ball_id, leaf_id in enumerate(sorted(leaves)):
            indices = leaves[leaf_id]
            assignment[indices] = ball_id
            statistics.append(_measure(z, indices, ball_id, train))
        sizes = np.asarray([stat["size"] for stat in statistics])
        if n_train is not None and n_train < min_train:
            stop = "insufficient_training_support"
        elif len(leaves) == budget:
            stop = "target_reached"
        elif len(leaves) < min(budget, count_cap):
            stop = "no_feasible_split"
        elif support_cap is not None and support_cap <= geometry_cap:
            stop = "support_count_cap_reached"
        else:
            stop = "geometry_count_cap_reached"
        snapshots[budget] = assignment
        summaries[budget] = {
            "requested_target": budget,
            "feasible_target": min(budget, count_cap),
            "num_balls": len(leaves),
            "min_leaf_size": min_size,
            "min_train_per_ball": min_train,
            "geometry_count_cap": geometry_cap,
            "support_count_cap": support_cap,
            "num_training_nodes": n_train,
            "min_train_count": None if train is None else min(int(stat["train_count"]) for stat in statistics),
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
            "split_num_threads": 1,
        }

    if 1 in requested:
        save_snapshot(1)
    # Parallel KMeans reductions can break tied SSE choices on symmetric inputs.
    with threadpool_limits(limits=1):
        while len(leaves) < goal and heap:
            negative_sse, leaf_id = heapq.heappop(heap)
            indices = leaves[leaf_id]
            parent_sse = -negative_sse
            threshold = max(1e-12, 1e-8 * parent_sse)
            row: dict[str, Any] = {
                "parent_id": leaf_id,
                "parent_size": int(indices.size),
                "parent_train_count": None if train is None else int(train[indices].sum()),
                "parent_sse": parent_sse,
                "accepted": False,
                "actual_count_after": len(leaves),
            }
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
            for child_id, child, cost in zip(child_ids, (left, right), child_sse):
                leaves[child_id] = child
                heapq.heappush(heap, (-cost, child_id))
            row.update(
                accepted=True,
                child_ids=child_ids,
                child_sizes=[int(left.size), int(right.size)],
                child_train_counts=None if train is None else [int(train[left].sum()), int(train[right].sum())],
                child_sse=child_sse,
                sse_gain=gain,
                actual_count_after=len(leaves),
            )
            trace.append(row)
            if len(leaves) in requested:
                save_snapshot(len(leaves))

    for budget in requested:
        if budget not in snapshots:
            save_snapshot(budget)
    # Reinsert in budget order even when several high budgets share the final root.
    return GranularBallTrajectory(
        {budget: snapshots[budget] for budget in requested},
        {budget: summaries[budget] for budget in requested},
        trace,
    )
