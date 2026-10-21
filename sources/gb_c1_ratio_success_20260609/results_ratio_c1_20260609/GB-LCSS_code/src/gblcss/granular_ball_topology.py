from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from numbers import Integral
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TopologyBallResult:
    assignment: np.ndarray
    balls: list[dict[str, int]]
    summary: dict[str, int]


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _as_array(values: Any, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.layout != torch.strided:
            raise ValueError(f"{name} must be a dense CPU tensor or numpy array")
        values = values.detach().numpy()
    return np.asarray(values)


def _canonical_adjacency(edge_index: Any, n_nodes: int) -> tuple[list[list[int]], int]:
    values = _as_array(edge_index, "edge_index")
    if values.ndim != 2 or values.shape[0] != 2 or values.dtype.kind not in "iu":
        raise ValueError("edge_index must be an integer array of shape (2, E)")
    if np.any(values < 0) or np.any(values >= n_nodes):
        raise ValueError("edge_index must contain node indices in [0, num_nodes)")
    values = values.astype(np.int64, copy=False)
    nonself = values[:, values[0] != values[1]]
    pairs = np.column_stack((np.minimum(nonself[0], nonself[1]),
                             np.maximum(nonself[0], nonself[1])))
    canonical = np.unique(pairs, axis=0)
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for left, right in canonical:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency, int(canonical.shape[0])


def _num_components(adjacency: list[list[int]]) -> int:
    visited = np.zeros(len(adjacency), dtype=bool)
    count = 0
    for node in range(len(adjacency)):
        if visited[node]:
            continue
        count += 1
        visited[node] = True
        stack = [node]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return count


def build_topology_balls(
    edge_index: torch.Tensor | np.ndarray,
    num_nodes: int,
    *,
    max_ball_size: int | None = None,
) -> TopologyBallResult:
    """Partition an undirected graph by deterministic residual-degree BFS.

    Nodes join a ball only when dequeued. A capacity-truncated queue is discarded,
    leaving its pending nodes eligible for the next ball. The degree heap is
    updated only after the whole current ball is removed from the residual graph.
    """
    n_nodes = _integer(num_nodes, "num_nodes")
    capacity = math.isqrt(n_nodes) if max_ball_size is None else _integer(max_ball_size, "max_ball_size")
    adjacency, canonical_edges = _canonical_adjacency(edge_index, n_nodes)
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    heap = [(-int(degree), node) for node, degree in enumerate(degrees)]
    heapq.heapify(heap)
    assignment = np.full(n_nodes, -1, dtype=np.int64)
    # Epoch markers avoid allocating or clearing an N-element visited mask per ball.
    enqueued_at = np.full(n_nodes, -1, dtype=np.int64)
    statistics: list[dict[str, int]] = []
    remaining = n_nodes
    while remaining:
        while heap:
            negative_degree, center = heapq.heappop(heap)
            if assignment[center] == -1 and -negative_degree == int(degrees[center]):
                break
        else:
            raise RuntimeError("degree heap exhausted before complete graph coverage")

        ball_id = len(statistics)
        center_degree = int(degrees[center])
        queue = deque([(center, 0)])
        enqueued_at[center] = ball_id
        members: list[int] = []
        max_depth = 0
        while queue and len(members) < capacity:
            node, depth = queue.popleft()
            assignment[node] = ball_id
            members.append(node)
            max_depth = depth
            if len(members) == capacity:
                break
            for neighbor in adjacency[node]:
                if assignment[neighbor] == -1 and enqueued_at[neighbor] != ball_id:
                    enqueued_at[neighbor] = ball_id
                    queue.append((neighbor, depth + 1))

        changed: set[int] = set()
        for node in members:
            for neighbor in adjacency[node]:
                if assignment[neighbor] == -1:
                    degrees[neighbor] -= 1
                    changed.add(neighbor)
        for node in changed:
            heapq.heappush(heap, (-int(degrees[node]), node))
        remaining -= len(members)
        statistics.append({
            "ball_id": ball_id,
            "size": len(members),
            "center": center,
            "center_residual_degree": center_degree,
            "max_depth": max_depth,
        })

    sizes = [ball["size"] for ball in statistics]
    return TopologyBallResult(
        assignment=assignment,
        balls=statistics,
        summary={
            "num_balls": len(statistics),
            "max_ball_size": capacity,
            "num_nodes": n_nodes,
            "canonical_edges": canonical_edges,
            "num_components": _num_components(adjacency),
            "isolated_nodes": sum(not neighbors for neighbors in adjacency),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "singleton_balls": sum(size == 1 for size in sizes),
        },
    )


def random_support_matched_assignment(
    assignment: np.ndarray,
    train_index: torch.Tensor | np.ndarray,
    random_state: int = 0,
) -> np.ndarray:
    """Randomize within train/nontrain positions, preserving both support counts.

    No label information is accepted. Separate permutations preserve each ball's
    training count and nontraining count, hence its total size, exactly.
    """
    values = _as_array(assignment, "assignment")
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("assignment must be a one-dimensional nonnegative integer array")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError("assignment ball IDs must fit int64")
    values = values.astype(np.int64, copy=False)
    indices = _as_array(train_index, "train_index")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= values.size)):
        raise ValueError("train_index must be a one-dimensional integer index array in range")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_index must contain unique node indices")
    seed = _integer(random_state, "random_state", minimum=0)
    rng = np.random.RandomState(seed)
    train_mask = np.zeros(values.size, dtype=bool)
    train_mask[indices.astype(np.int64)] = True
    randomized = np.empty_like(values)
    for mask in (train_mask, ~train_mask):
        nodes = np.flatnonzero(mask)
        randomized[rng.permutation(nodes)] = values[nodes]
    return randomized
