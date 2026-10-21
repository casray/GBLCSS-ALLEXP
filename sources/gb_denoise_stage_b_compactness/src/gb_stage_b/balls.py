from __future__ import annotations

from dataclasses import dataclass
from typing import overload
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class GranularBall:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    label_hist: dict[int, int]
    purity: float
    majority_label: int


@dataclass(frozen=True)
class BallBatch:
    assignment: torch.Tensor
    centers: torch.Tensor
    sizes: torch.Tensor
    average_angular_distance: torch.Tensor
    radius: torch.Tensor

    @property
    def num_balls(self) -> int:
        return int(self.centers.size(0))


def _as_numpy_2d(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if arr.shape[0] == 0:
        raise ValueError("embeddings must contain at least one sample")
    return arr


def _as_numpy_1d(values: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _make_granular_ball(embeddings: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> GranularBall:
    members = embeddings[indices]
    member_labels = labels[indices].astype(np.int64, copy=False)
    center = members.mean(axis=0)
    distances = np.linalg.norm(members - center, axis=1)
    unique, counts = np.unique(member_labels, return_counts=True)
    best = int(np.argmax(counts))
    label_hist = {int(label): int(count) for label, count in zip(unique, counts)}
    return GranularBall(
        indices=indices.astype(np.int64, copy=False),
        center=center.astype(np.float64, copy=False),
        radius=float(distances.max(initial=0.0)),
        label_hist=label_hist,
        purity=float(counts[best] / indices.size),
        majority_label=int(unique[best]),
    )


def build_balls(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    min_size: int = 3,
    max_size: int = 256,
    random_state: int = 0,
) -> list[GranularBall]:
    """Build diagnostic granular balls with KMeans.

    ``min_size`` and ``max_size`` are used only to choose a target KMeans
    cluster count. KMeans can still return balls outside those sizes,
    especially for duplicate points or skewed distributions; reliable-ball
    filtering should enforce downstream size policy where needed.
    """
    x = _as_numpy_2d(embeddings)
    y = _as_numpy_1d(labels, "labels")
    if y.shape[0] != x.shape[0]:
        raise ValueError("labels length must match embeddings")
    if min_size <= 0:
        raise ValueError("min_size must be positive")
    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")

    n_samples = x.shape[0]
    if n_samples < min_size or n_samples <= max_size:
        return [_make_granular_ball(x, y, np.arange(n_samples, dtype=np.int64))]

    lower_bound = int(np.ceil(n_samples / max_size))
    upper_bound = max(1, int(n_samples // min_size))
    n_clusters = min(max(lower_bound, 1), upper_bound)
    if n_clusters <= 1:
        return [_make_granular_ball(x, y, np.arange(n_samples, dtype=np.int64))]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        assignments = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(x)
    return make_ball_batch(x, y, assignments)


@overload
def make_ball_batch(embeddings: np.ndarray | torch.Tensor, assignment: np.ndarray | torch.Tensor) -> BallBatch:
    ...


@overload
def make_ball_batch(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    assignments: np.ndarray | torch.Tensor,
) -> list[GranularBall]:
    ...


def make_ball_batch(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    assignments: np.ndarray | torch.Tensor | None = None,
) -> BallBatch | list[GranularBall]:
    if assignments is None:
        return _make_torch_ball_batch(embeddings, labels)

    x = _as_numpy_2d(embeddings)
    y = _as_numpy_1d(labels, "labels")
    assignment = _as_numpy_1d(assignments, "assignments").astype(np.int64, copy=False)
    if y.shape[0] != x.shape[0] or assignment.shape[0] != x.shape[0]:
        raise ValueError("labels and assignments must match embeddings")
    if assignment.size == 0:
        raise ValueError("assignments must be non-empty")

    balls = []
    for ball_id in sorted(np.unique(assignment).tolist()):
        indices = np.flatnonzero(assignment == int(ball_id))
        if indices.size == 0:
            continue
        balls.append(_make_granular_ball(x, y, indices))
    if not balls:
        raise ValueError("assignments must define at least one non-empty ball")
    return balls


def ball_metrics(
    balls: list[GranularBall],
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Compute geometry metrics and label diagnostics for the supplied labels."""
    x = _as_numpy_2d(embeddings)
    y = _as_numpy_1d(labels, "labels")
    if y.shape[0] != x.shape[0]:
        raise ValueError("labels length must match embeddings")
    if not balls:
        raise ValueError("balls must be non-empty")

    sizes = np.asarray([ball.indices.size for ball in balls], dtype=np.float64)
    radii = np.asarray([ball.radius for ball in balls], dtype=np.float64)
    purities = np.asarray([_purity_for_indices(y, ball.indices) for ball in balls], dtype=np.float64)
    intra = []
    centers = np.stack([ball.center for ball in balls], axis=0)
    for ball in balls:
        members = x[ball.indices]
        intra.extend(np.linalg.norm(members - ball.center, axis=1).tolist())
    if centers.shape[0] > 1:
        diff = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        upper = distances[np.triu_indices(centers.shape[0], k=1)]
        center_separation = float(upper.mean())
    else:
        center_separation = 0.0
    return {
        "num_balls": float(len(balls)),
        "mean_radius": float(radii.mean()),
        "mean_purity": float(purities.mean()),
        "weighted_purity": float(np.average(purities, weights=sizes)),
        "mean_intra_distance": float(np.mean(intra)) if intra else 0.0,
        "center_separation": center_separation,
        "ball_size_mean": float(sizes.mean()),
        "ball_size_min": float(sizes.min()),
        "ball_size_max": float(sizes.max()),
    }


def _purity_for_indices(labels: np.ndarray, indices: np.ndarray) -> float:
    member_labels = labels[indices]
    if member_labels.size == 0:
        return 0.0
    _, counts = np.unique(member_labels, return_counts=True)
    return float(counts.max() / member_labels.size)


def angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    sim = (a * b).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(sim)


def _make_torch_ball_batch(embeddings: np.ndarray | torch.Tensor, assignment: np.ndarray | torch.Tensor) -> BallBatch:
    x = torch.as_tensor(embeddings).detach()
    assignment = torch.as_tensor(assignment, dtype=torch.long, device=x.device).detach()
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if assignment.ndim != 1 or assignment.numel() != x.size(0):
        raise ValueError("assignment must be a 1D tensor matching embeddings")
    if assignment.numel() == 0:
        raise ValueError("assignment must be non-empty")

    x = F.normalize(x.float(), dim=-1, eps=1e-8)
    unique_ids = sorted(int(v) for v in assignment.unique().tolist())
    centers = []
    sizes = []
    ads = []
    radii = []
    for ball_id in unique_ids:
        idx = torch.where(assignment == ball_id)[0]
        members = x[idx]
        center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).squeeze(0)
        dist = angular_distance(members, center.expand_as(members))
        centers.append(center)
        sizes.append(idx.numel())
        ads.append(dist.mean())
        radii.append(dist.max())

    remap = {old: new for new, old in enumerate(unique_ids)}
    compact_assignment = torch.tensor([remap[int(v)] for v in assignment.tolist()], dtype=torch.long, device=x.device)
    return BallBatch(
        assignment=compact_assignment,
        centers=torch.stack(centers, dim=0),
        sizes=torch.tensor(sizes, dtype=torch.long, device=x.device),
        average_angular_distance=torch.stack(ads),
        radius=torch.stack(radii),
    )


def _balanced_split(idx: torch.Tensor, members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if members.size(1) == 0:
        order = torch.arange(idx.numel(), device=idx.device)
    else:
        order = torch.argsort(members[:, 0], stable=True)
    midpoint = idx.numel() // 2
    return idx[order[:midpoint]], idx[order[midpoint:]]


def _angular_split(idx: torch.Tensor, members: torch.Tensor, min_size: int, force: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
    center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).squeeze(0)
    dist = angular_distance(members, center.expand_as(members))
    farthest = int(torch.argmax(dist).item())
    c1 = members[farthest]
    dist_to_c1 = angular_distance(members, c1.expand_as(members))
    second = int(torch.argmax(dist_to_c1).item())
    c2 = members[second]
    d1 = angular_distance(members, c1.expand_as(members))
    d2 = angular_distance(members, c2.expand_as(members))
    left = idx[d1 <= d2]
    right = idx[d1 > d2]
    if left.numel() >= min_size and right.numel() >= min_size:
        return left, right
    if force and idx.numel() >= 2 * min_size:
        return _balanced_split(idx, members)
    return None


def _gain_split(idx: torch.Tensor, members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    similarity = members @ members.t()
    distance = 1.0 - similarity
    flat_idx = int(torch.argmax(distance).item())
    num_points = int(members.size(0))
    first_idx = flat_idx // num_points
    second_idx = flat_idx % num_points
    pivot_a = members[first_idx]
    pivot_b = members[second_idx]
    dist_a = angular_distance(members, pivot_a.expand_as(members))
    dist_b = angular_distance(members, pivot_b.expand_as(members))
    left = idx[dist_a <= dist_b]
    right = idx[dist_a > dist_b]
    if left.numel() == 0 or right.numel() == 0:
        midpoint = idx.numel() // 2
        return idx[:midpoint], idx[midpoint:]
    return left, right


def _ball_average_distance(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    members = x[idx]
    center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).expand_as(members)
    return angular_distance(members, center).mean()


def generate_gain_split_balls(
    x: torch.Tensor,
    min_ball_size: int = 8,
    max_depth: int = 12,
    min_split_gain: float = 0.001,
) -> BallBatch:
    if x.ndim != 2 or x.size(0) == 0:
        raise ValueError("x must be a non-empty 2D tensor")
    if min_ball_size < 2:
        raise ValueError("min_ball_size must be at least 2")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if min_split_gain < 0:
        raise ValueError("min_split_gain must be non-negative")

    x = F.normalize(x.detach().float(), dim=-1, eps=1e-8)
    queue: list[tuple[torch.Tensor, int]] = [(torch.arange(x.size(0), device=x.device), 0)]
    final: list[torch.Tensor] = []
    while queue:
        idx, depth = queue.pop(0)
        parent_ad = _ball_average_distance(x, idx)
        if idx.numel() < 2 * min_ball_size or depth >= max_depth:
            final.append(idx)
            continue

        left, right = _gain_split(idx, x[idx])
        if left.numel() < min_ball_size or right.numel() < min_ball_size:
            final.append(idx)
            continue

        left_ad = _ball_average_distance(x, left)
        right_ad = _ball_average_distance(x, right)
        weighted_child_ad = (left_ad * left.numel() + right_ad * right.numel()) / idx.numel()
        gain = parent_ad - weighted_child_ad
        if float(gain.item()) >= min_split_gain:
            queue.append((left, depth + 1))
            queue.append((right, depth + 1))
        else:
            final.append(idx)

    assignment = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for ball_id, idx in enumerate(final):
        assignment[idx] = ball_id
    return _make_torch_ball_batch(x, assignment)


def generate_angular_balls(
    x: torch.Tensor,
    min_size: int = 3,
    quality_threshold: float = 0.35,
    max_size: int | None = None,
) -> BallBatch:
    if min_size <= 0:
        raise ValueError("min_size must be positive")
    if max_size is not None and max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size")
    x = F.normalize(x.detach().float(), dim=-1, eps=1e-8)
    if x.ndim != 2 or x.size(0) == 0:
        raise ValueError("x must be a non-empty 2D tensor")

    groups = [torch.arange(x.size(0), device=x.device)]
    final = []
    while groups:
        idx = groups.pop()
        members = x[idx]
        center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1, eps=1e-8).squeeze(0)
        dist = angular_distance(members, center.expand_as(members))
        oversized = max_size is not None and idx.numel() > max_size
        can_split = idx.numel() >= 2 * min_size
        if not can_split or (not oversized and float(dist.mean()) <= quality_threshold):
            final.append(idx)
            continue

        split = _angular_split(idx, members, min_size=min_size, force=oversized)
        if split is None:
            final.append(idx)
        else:
            left, right = split
            groups.extend([left, right])

    assignment = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for ball_id, idx in enumerate(final):
        assignment[idx] = ball_id
    return _make_torch_ball_batch(x, assignment)


def reliable_ball_mask(balls: BallBatch, min_size: int, max_size: int, ad_percentile: float) -> torch.Tensor:
    if not 0 <= ad_percentile <= 100:
        raise ValueError("ad_percentile must be in [0, 100]")
    threshold = torch.quantile(balls.average_angular_distance, ad_percentile / 100.0)
    return (balls.sizes >= min_size) & (balls.sizes <= max_size) & (balls.average_angular_distance <= threshold)


def shuffled_assignments(assignment: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=assignment.device).manual_seed(seed)
    perm = torch.randperm(assignment.numel(), generator=gen, device=assignment.device)
    return assignment[perm]


def size_matched_random_assignments(assignment: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=assignment.device).manual_seed(seed)
    sizes = torch.bincount(assignment).tolist()
    perm = torch.randperm(assignment.numel(), generator=gen, device=assignment.device)
    out = torch.empty_like(assignment)
    cursor = 0
    for ball_id, size in enumerate(sizes):
        out[perm[cursor : cursor + size]] = ball_id
        cursor += size
    return out
