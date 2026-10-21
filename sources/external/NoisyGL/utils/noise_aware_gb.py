from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class GeometryTreeNode:
    node_id: int
    indices: np.ndarray
    depth: int
    avg_distance: float
    geometry_gain: float = 0.0
    left_id: int | None = None
    right_id: int | None = None


@dataclass(frozen=True)
class GeometryCandidateTree:
    nodes: dict[int, GeometryTreeNode]
    root_id: int
    leaf_ids: tuple[int, ...]
    n_nodes: int
    assignment_hash: str
    config: dict


@dataclass(frozen=True)
class PrunedPartition:
    assignment: np.ndarray
    final_node_ids: tuple[int, ...]
    accepted_split_ids: tuple[int, ...]
    rejected_split_ids: tuple[int, ...]
    log_bayes_factors: dict[int, float]

    @property
    def num_balls(self) -> int:
        return len(self.final_node_ids)


@dataclass(frozen=True)
class CrossFitTargetResult:
    soft_targets: np.ndarray
    local_distributions: np.ndarray
    alpha: np.ndarray
    ball_ids: np.ndarray
    fold_num_balls: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class BudgetedPartition:
    assignment: np.ndarray
    final_node_ids: tuple[int, ...]
    selected_split_ids: tuple[int, ...]
    split_scores: dict[int, float]
    normalized_jsd: dict[int, float]
    assignment_hash: str

    @property
    def num_balls(self) -> int:
        return len(self.final_node_ids)


def _as_embeddings(embeddings) -> np.ndarray:
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().float().numpy()
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("embeddings must be a non-empty 2-D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _measure(points: np.ndarray, indices: np.ndarray) -> float:
    selected = points[indices]
    center = selected.mean(axis=0)
    return float(np.linalg.norm(selected - center, axis=1).mean())


def _assignment_hash(assignment: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(assignment, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def build_geometry_candidate_tree(
    embeddings,
    max_leaves: int = 128,
    min_ball_size: int = 32,
    min_geometry_gain: float = 0.01,
    random_state: int = 0,
) -> GeometryCandidateTree:
    """Build a label-agnostic recursive 2-means candidate tree."""
    points = _as_embeddings(embeddings)
    n_nodes = int(points.shape[0])
    max_leaves = max(1, min(int(max_leaves), n_nodes))
    min_ball_size = max(1, int(min_ball_size))
    min_geometry_gain = max(0.0, float(min_geometry_gain))
    random_state = int(random_state)

    root_indices = np.arange(n_nodes, dtype=np.int64)
    nodes: dict[int, GeometryTreeNode] = {
        0: GeometryTreeNode(
            node_id=0,
            indices=root_indices,
            depth=0,
            avg_distance=_measure(points, root_indices),
        )
    }
    active_leaves = [0]
    next_node_id = 1
    split_iteration = 0

    while len(active_leaves) < max_leaves:
        candidates = []
        for node_id in active_leaves:
            node = nodes[node_id]
            if node.indices.size >= 2 * min_ball_size:
                candidates.append((node.avg_distance, node_id))
        if not candidates:
            break

        accepted = False
        for _, node_id in sorted(candidates, reverse=True):
            parent = nodes[node_id]
            labels = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=random_state + split_iteration + node_id,
            ).fit_predict(points[parent.indices])
            if np.unique(labels).size != 2:
                continue
            left_indices = parent.indices[labels == 0]
            right_indices = parent.indices[labels == 1]
            if (
                left_indices.size < min_ball_size
                or right_indices.size < min_ball_size
            ):
                continue

            left_avg = _measure(points, left_indices)
            right_avg = _measure(points, right_indices)
            child_avg = (
                left_avg * left_indices.size + right_avg * right_indices.size
            ) / parent.indices.size
            geometry_gain = (
                0.0
                if parent.avg_distance <= 1e-12
                else 1.0 - child_avg / parent.avg_distance
            )
            if geometry_gain + 1e-12 < min_geometry_gain:
                continue

            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2
            nodes[left_id] = GeometryTreeNode(
                node_id=left_id,
                indices=left_indices,
                depth=parent.depth + 1,
                avg_distance=left_avg,
            )
            nodes[right_id] = GeometryTreeNode(
                node_id=right_id,
                indices=right_indices,
                depth=parent.depth + 1,
                avg_distance=right_avg,
            )
            nodes[node_id] = GeometryTreeNode(
                node_id=parent.node_id,
                indices=parent.indices,
                depth=parent.depth,
                avg_distance=parent.avg_distance,
                geometry_gain=float(geometry_gain),
                left_id=left_id,
                right_id=right_id,
            )
            active_leaves.remove(node_id)
            active_leaves.extend([left_id, right_id])
            split_iteration += 1
            accepted = True
            break
        if not accepted:
            break

    assignment = np.empty(n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(sorted(active_leaves)):
        assignment[nodes[node_id].indices] = ball_id
    return GeometryCandidateTree(
        nodes=nodes,
        root_id=0,
        leaf_ids=tuple(sorted(active_leaves)),
        n_nodes=n_nodes,
        assignment_hash=_assignment_hash(assignment),
        config={
            "max_leaves": max_leaves,
            "min_ball_size": min_ball_size,
            "min_geometry_gain": min_geometry_gain,
            "random_state": random_state,
        },
    )


def build_prefixed_geometry_candidate_tree(
    embeddings,
    expected_prefix_leaves: int,
    prefix_max_leaves: int = 64,
    max_leaves: int = 128,
    min_ball_size: int = 32,
    prefix_split_improvement: float = 0.05,
    random_state: int = 0,
) -> GeometryCandidateTree:
    """Build a label-free tree whose prefix exactly matches the formal GB tree."""
    points = _as_embeddings(embeddings)
    n_nodes = int(points.shape[0])
    expected_prefix_leaves = int(expected_prefix_leaves)
    prefix_max_leaves = max(1, min(int(prefix_max_leaves), n_nodes))
    max_leaves = max(prefix_max_leaves, min(int(max_leaves), n_nodes))
    min_ball_size = max(1, int(min_ball_size))
    prefix_split_improvement = max(0.0, float(prefix_split_improvement))
    random_state = int(random_state)
    if expected_prefix_leaves < 1 or expected_prefix_leaves > prefix_max_leaves:
        raise ValueError(
            "expected_prefix_leaves must be between 1 and prefix_max_leaves"
        )

    root_indices = np.arange(n_nodes, dtype=np.int64)
    nodes: dict[int, GeometryTreeNode] = {
        0: GeometryTreeNode(
            node_id=0,
            indices=root_indices,
            depth=0,
            avg_distance=_measure(points, root_indices),
        )
    }
    active_leaves = [0]
    next_node_id = 1
    split_iteration = 0

    def split_once(minimum_gain: float) -> bool:
        nonlocal next_node_id, split_iteration
        candidates = []
        for position, node_id in enumerate(active_leaves):
            node = nodes[node_id]
            if node.indices.size >= 2 * min_ball_size:
                candidates.append((node.avg_distance, position, node_id))
        for _distance, position, node_id in sorted(candidates, reverse=True):
            parent = nodes[node_id]
            labels = KMeans(
                n_clusters=2,
                n_init=10,
                random_state=random_state + split_iteration + position,
            ).fit_predict(points[parent.indices])
            if np.unique(labels).size != 2:
                continue
            left_indices = parent.indices[labels == 0]
            right_indices = parent.indices[labels == 1]
            if (
                left_indices.size < min_ball_size
                or right_indices.size < min_ball_size
            ):
                continue
            left_avg = _measure(points, left_indices)
            right_avg = _measure(points, right_indices)
            child_avg = (
                left_avg * left_indices.size + right_avg * right_indices.size
            ) / parent.indices.size
            geometry_gain = (
                0.0
                if parent.avg_distance <= 1e-12
                else 1.0 - child_avg / parent.avg_distance
            )
            if (
                parent.avg_distance > 1e-12
                and geometry_gain + 1e-12 < float(minimum_gain)
            ):
                continue

            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2
            nodes[left_id] = GeometryTreeNode(
                node_id=left_id,
                indices=left_indices,
                depth=parent.depth + 1,
                avg_distance=left_avg,
            )
            nodes[right_id] = GeometryTreeNode(
                node_id=right_id,
                indices=right_indices,
                depth=parent.depth + 1,
                avg_distance=right_avg,
            )
            nodes[node_id] = GeometryTreeNode(
                node_id=parent.node_id,
                indices=parent.indices,
                depth=parent.depth,
                avg_distance=parent.avg_distance,
                geometry_gain=float(geometry_gain),
                left_id=left_id,
                right_id=right_id,
            )
            active_leaves[position] = left_id
            active_leaves.append(right_id)
            split_iteration += 1
            return True
        return False

    while len(active_leaves) < prefix_max_leaves:
        if not split_once(prefix_split_improvement):
            break
    if len(active_leaves) != expected_prefix_leaves:
        raise ValueError(
            "formal prefix leaf count mismatch: "
            f"expected={expected_prefix_leaves}, reproduced={len(active_leaves)}"
        )

    prefix_assignment = np.empty(n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(active_leaves):
        prefix_assignment[nodes[node_id].indices] = ball_id
    prefix_assignment_hash = _assignment_hash(prefix_assignment)
    prefix_leaf_ids = tuple(active_leaves)

    while len(active_leaves) < max_leaves:
        if not split_once(0.0):
            break

    final_assignment = np.empty(n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(sorted(active_leaves)):
        final_assignment[nodes[node_id].indices] = ball_id
    return GeometryCandidateTree(
        nodes=nodes,
        root_id=0,
        leaf_ids=tuple(sorted(active_leaves)),
        n_nodes=n_nodes,
        assignment_hash=_assignment_hash(final_assignment),
        config={
            "builder": "formal_prefix_then_zero_gain_extension",
            "expected_prefix_leaves": expected_prefix_leaves,
            "prefix_leaf_count": len(prefix_leaf_ids),
            "prefix_leaf_ids": prefix_leaf_ids,
            "prefix_assignment_hash": prefix_assignment_hash,
            "prefix_max_leaves": prefix_max_leaves,
            "prefix_split_improvement": prefix_split_improvement,
            "max_leaves": max_leaves,
            "min_ball_size": min_ball_size,
            "extension_min_geometry_gain": 0.0,
            "random_state": random_state,
        },
    )


def make_stratified_crossfit_folds(
    train_indices,
    noisy_labels,
    n_splits: int = 5,
    random_state: int = 0,
    n_nodes: int | None = None,
) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    labels = np.asarray(noisy_labels, dtype=np.int64)
    if train_indices.ndim != 1 or train_indices.size < 1:
        raise ValueError("train_indices must be a non-empty 1-D array")
    if n_nodes is None:
        n_nodes = int(labels.shape[0])
    n_nodes = int(n_nodes)
    if labels.shape[0] != n_nodes:
        raise ValueError("noisy_labels length must equal n_nodes")
    n_splits = int(n_splits)
    if n_splits < 2 or train_indices.size < n_splits:
        raise ValueError("n_splits must be between 2 and the train size")

    fold_ids = np.full(n_nodes, -1, dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    for class_id in np.unique(labels[train_indices]):
        class_indices = train_indices[labels[train_indices] == class_id].copy()
        rng.shuffle(class_indices)
        offset = int(rng.integers(0, n_splits))
        for position, node in enumerate(class_indices):
            fold_ids[node] = (offset + position) % n_splits
    if np.any(fold_ids[train_indices] < 0):
        raise RuntimeError("every training node must receive exactly one fold")
    if np.unique(fold_ids[train_indices]).size != n_splits:
        raise ValueError("stratification produced an empty fold")
    return fold_ids


def _aligned_guidance_labels(
    guidance_indices: np.ndarray,
    guidance_labels,
    n_nodes: int,
) -> np.ndarray:
    labels = np.asarray(guidance_labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("guidance_labels must be 1-D")
    if labels.shape[0] == n_nodes:
        return labels[guidance_indices]
    if labels.shape[0] == guidance_indices.shape[0]:
        return labels
    raise ValueError(
        "guidance_labels must be node-aligned or aligned with guidance_indices"
    )


def _counts_in_node(
    node: GeometryTreeNode,
    guidance_indices: np.ndarray,
    aligned_labels: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, int]:
    inside = np.isin(guidance_indices, node.indices, assume_unique=False)
    selected = aligned_labels[inside]
    return np.bincount(selected, minlength=n_classes).astype(np.int64), int(
        selected.size
    )


def _dirichlet_multinomial_log_marginal(
    counts: np.ndarray,
    alpha: np.ndarray,
) -> float:
    total_alpha = float(alpha.sum())
    total_count = int(counts.sum())
    value = math.lgamma(total_alpha) - math.lgamma(total_alpha + total_count)
    for alpha_c, count_c in zip(alpha, counts):
        value += math.lgamma(float(alpha_c + count_c)) - math.lgamma(
            float(alpha_c)
        )
    return float(value)


def prune_candidate_tree(
    tree: GeometryCandidateTree,
    guidance_indices,
    guidance_labels,
    n_classes: int,
    min_guidance_per_child: int = 5,
    log_bf_threshold: float = math.log(3.0),
    prior_strength: float = 2.0,
) -> PrunedPartition:
    guidance_indices = np.asarray(guidance_indices, dtype=np.int64)
    if guidance_indices.ndim != 1 or guidance_indices.size < 1:
        raise ValueError("guidance_indices must be a non-empty 1-D array")
    if np.any(guidance_indices < 0) or np.any(guidance_indices >= tree.n_nodes):
        raise ValueError("guidance_indices contain out-of-range nodes")
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    aligned_labels = _aligned_guidance_labels(
        guidance_indices, guidance_labels, tree.n_nodes
    )
    if np.any(aligned_labels < 0) or np.any(aligned_labels >= n_classes):
        raise ValueError("guidance labels are outside the class range")

    global_counts = np.bincount(aligned_labels, minlength=n_classes).astype(float)
    prior_mean = (global_counts + 1.0) / (global_counts.sum() + n_classes)
    alpha = max(float(prior_strength), 1e-8) * prior_mean
    min_guidance_per_child = max(1, int(min_guidance_per_child))
    log_bf_threshold = float(log_bf_threshold)

    accepted: list[int] = []
    rejected: list[int] = []
    final_nodes: list[int] = []
    log_bfs: dict[int, float] = {}

    def visit(node_id: int) -> None:
        node = tree.nodes[node_id]
        if node.left_id is None or node.right_id is None:
            final_nodes.append(node_id)
            return
        left = tree.nodes[node.left_id]
        right = tree.nodes[node.right_id]
        left_counts, left_total = _counts_in_node(
            left, guidance_indices, aligned_labels, n_classes
        )
        right_counts, right_total = _counts_in_node(
            right, guidance_indices, aligned_labels, n_classes
        )
        parent_counts = left_counts + right_counts
        log_bf = (
            _dirichlet_multinomial_log_marginal(left_counts, alpha)
            + _dirichlet_multinomial_log_marginal(right_counts, alpha)
            - _dirichlet_multinomial_log_marginal(parent_counts, alpha)
        )
        log_bfs[node_id] = float(log_bf)
        split_is_valid = (
            left_total >= min_guidance_per_child
            and right_total >= min_guidance_per_child
            and node.geometry_gain + 1e-12
            >= float(tree.config["min_geometry_gain"])
            and log_bf >= log_bf_threshold
        )
        if not split_is_valid:
            rejected.append(node_id)
            final_nodes.append(node_id)
            return
        accepted.append(node_id)
        visit(node.left_id)
        visit(node.right_id)

    visit(tree.root_id)
    assignment = np.empty(tree.n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(final_nodes):
        assignment[tree.nodes[node_id].indices] = ball_id
    return PrunedPartition(
        assignment=assignment,
        final_node_ids=tuple(final_nodes),
        accepted_split_ids=tuple(accepted),
        rejected_split_ids=tuple(rejected),
        log_bayes_factors=log_bfs,
    )


def shuffle_guidance_within_construction(
    construction_indices,
    labels,
    random_state: int,
) -> np.ndarray:
    construction_indices = np.asarray(construction_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    aligned = _aligned_guidance_labels(
        construction_indices, labels, labels.shape[0]
    ).copy()
    return np.random.default_rng(int(random_state)).permutation(aligned)


def label_assisted_split_score(
    geometry_gain: float,
    left_distribution,
    right_distribution,
) -> tuple[float, float]:
    geometry_gain = max(0.0, float(geometry_gain))
    left = np.asarray(left_distribution, dtype=np.float64)
    right = np.asarray(right_distribution, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("child distributions must be aligned 1-D arrays")
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("child distributions must be non-negative")
    if left.sum() <= 0.0 or right.sum() <= 0.0:
        raise ValueError("child distributions must have positive mass")
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)

    def kl_divergence(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(np.sum(values[positive] * np.log(values[positive] / midpoint[positive])))

    jsd = 0.5 * kl_divergence(left) + 0.5 * kl_divergence(right)
    normalized_jsd = float(np.clip(jsd / math.log(2.0), 0.0, 1.0))
    return geometry_gain * (1.0 + normalized_jsd), normalized_jsd


def cut_candidate_tree_fixed_budget(
    tree: GeometryCandidateTree,
    target_num_balls: int,
    guidance_indices,
    guidance_labels,
    n_classes: int,
    label_assisted: bool = True,
    prior_strength: float = 2.0,
) -> BudgetedPartition:
    target_num_balls = int(target_num_balls)
    if target_num_balls < 1:
        raise ValueError("target_num_balls must be positive")
    if target_num_balls > len(tree.leaf_ids):
        raise ValueError(
            "candidate tree cannot reach target_num_balls: "
            f"requested={target_num_balls}, capacity={len(tree.leaf_ids)}"
        )
    guidance_indices = np.asarray(guidance_indices, dtype=np.int64)
    if guidance_indices.ndim != 1 or guidance_indices.size < 1:
        raise ValueError("guidance_indices must be a non-empty 1-D array")
    if np.any(guidance_indices < 0) or np.any(guidance_indices >= tree.n_nodes):
        raise ValueError("guidance_indices contain out-of-range nodes")
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    aligned_labels = _aligned_guidance_labels(
        guidance_indices, guidance_labels, tree.n_nodes
    )
    if np.any(aligned_labels < 0) or np.any(aligned_labels >= n_classes):
        raise ValueError("guidance labels are outside the class range")
    prior_strength = float(prior_strength)
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive")
    global_counts = np.bincount(aligned_labels, minlength=n_classes).astype(np.float64)
    prior_mean = (global_counts + 1.0) / (global_counts.sum() + n_classes)
    prior = prior_strength * prior_mean

    active = {int(tree.root_id)}
    selected_split_ids: list[int] = []
    selected_scores: dict[int, float] = {}
    selected_jsd: dict[int, float] = {}
    while len(active) < target_num_balls:
        candidates = []
        for node_id in active:
            node = tree.nodes[node_id]
            if node.left_id is None or node.right_id is None:
                continue
            geometry_gain = max(0.0, float(node.geometry_gain))
            normalized_jsd = 0.0
            score = geometry_gain
            if label_assisted:
                left_counts, left_total = _counts_in_node(
                    tree.nodes[node.left_id],
                    guidance_indices,
                    aligned_labels,
                    n_classes,
                )
                right_counts, right_total = _counts_in_node(
                    tree.nodes[node.right_id],
                    guidance_indices,
                    aligned_labels,
                    n_classes,
                )
                left_distribution = (left_counts + prior) / (left_total + prior_strength)
                right_distribution = (right_counts + prior) / (right_total + prior_strength)
                score, normalized_jsd = label_assisted_split_score(
                    geometry_gain,
                    left_distribution,
                    right_distribution,
                )
            candidates.append(
                (float(score), geometry_gain, -int(node_id), int(node_id), normalized_jsd)
            )
        if not candidates:
            raise ValueError(
                "candidate tree cannot reach target_num_balls from active leaves: "
                f"requested={target_num_balls}, reached={len(active)}"
            )
        score, _gain, _negative_id, selected_id, normalized_jsd = max(candidates)
        selected = tree.nodes[selected_id]
        active.remove(selected_id)
        active.add(int(selected.left_id))
        active.add(int(selected.right_id))
        selected_split_ids.append(selected_id)
        selected_scores[selected_id] = float(score)
        selected_jsd[selected_id] = float(normalized_jsd)

    final_node_ids = tuple(sorted(active))
    assignment = np.empty(tree.n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(final_node_ids):
        assignment[tree.nodes[node_id].indices] = ball_id
    return BudgetedPartition(
        assignment=assignment,
        final_node_ids=final_node_ids,
        selected_split_ids=tuple(selected_split_ids),
        split_scores=selected_scores,
        normalized_jsd=selected_jsd,
        assignment_hash=_assignment_hash(assignment),
    )


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    result[np.arange(labels.shape[0]), labels] = 1.0
    return result


def _crossfit_targets_for_assignments(
    assignments_by_fold: dict[int, np.ndarray],
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    fold_ids: np.ndarray,
    n_classes: int,
    target_beta: float,
    min_train_per_target: int,
) -> CrossFitTargetResult:
    soft_targets = _one_hot(noisy_labels, n_classes)
    local_distributions = np.zeros_like(soft_targets)
    alpha = np.zeros(noisy_labels.shape[0], dtype=np.float64)
    ball_ids = np.full(noisy_labels.shape[0], -1, dtype=np.int64)
    fold_num_balls = []

    for fold_id in sorted(assignments_by_fold):
        assignment = np.asarray(assignments_by_fold[fold_id], dtype=np.int64)
        heldout = train_indices[fold_ids[train_indices] == fold_id]
        construction = train_indices[fold_ids[train_indices] != fold_id]
        fold_num_balls.append(int(np.unique(assignment).size))
        for node in heldout:
            ball_id = int(assignment[node])
            ball_ids[node] = ball_id
            voters = construction[assignment[construction] == ball_id]
            if voters.size < min_train_per_target:
                local_distributions[node] = soft_targets[node]
                continue
            counts = np.bincount(
                noisy_labels[voters], minlength=n_classes
            ).astype(np.float64)
            distribution = counts / counts.sum()
            local_distributions[node] = distribution
            margin = float(distribution.max())
            baseline = 1.0 / n_classes
            node_alpha = np.clip(
                (margin - baseline) / (1.0 - baseline), 0.0, 1.0
            )
            alpha[node] = node_alpha
            maxima = np.flatnonzero(counts == counts.max())
            if maxima.size > 1:
                local_target = distribution
            else:
                majority = int(maxima[0])
                majority_one_hot = np.zeros(n_classes, dtype=np.float64)
                majority_one_hot[majority] = 1.0
                local_target = (
                    (1.0 - target_beta) * distribution
                    + target_beta * majority_one_hot
                )
            soft_targets[node] = (
                (1.0 - node_alpha) * soft_targets[node]
                + node_alpha * local_target
            )

    if np.any(ball_ids[train_indices] < 0):
        raise RuntimeError("every training node must receive one cross-fit target")
    return CrossFitTargetResult(
        soft_targets=soft_targets,
        local_distributions=local_distributions,
        alpha=alpha,
        ball_ids=ball_ids,
        fold_num_balls=np.asarray(fold_num_balls, dtype=np.int64),
        diagnostics={
            "alpha_coverage": float(np.mean(alpha[train_indices] > 0.0)),
            "median_num_balls": float(np.median(fold_num_balls)),
            "min_num_balls": int(np.min(fold_num_balls)),
            "max_num_balls": int(np.max(fold_num_balls)),
        },
    )


def build_crossfit_targets(
    tree: GeometryCandidateTree,
    noisy_labels,
    train_indices,
    fold_ids,
    n_classes: int,
    guidance_mode: str = "noisy",
    clean_labels=None,
    target_beta: float = 0.5,
    min_train_per_target: int = 2,
    min_guidance_per_child: int = 5,
    log_bf_threshold: float = math.log(3.0),
    prior_strength: float = 2.0,
    random_state: int = 0,
) -> CrossFitTargetResult:
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if noisy_labels.shape[0] != tree.n_nodes or fold_ids.shape[0] != tree.n_nodes:
        raise ValueError("tree, noisy_labels, and fold_ids must share node count")
    guidance_mode = str(guidance_mode).lower()
    if guidance_mode not in {"noisy", "shuffled", "clean"}:
        raise ValueError("guidance_mode must be noisy, shuffled, or clean")
    if guidance_mode == "clean":
        if clean_labels is None:
            raise ValueError("clean_labels are required for clean guidance")
        clean_labels = np.asarray(clean_labels, dtype=np.int64)
        if clean_labels.shape[0] != tree.n_nodes:
            raise ValueError("clean_labels length must match node count")

    assignments: dict[int, np.ndarray] = {}
    accepted_counts = []
    rejected_counts = []
    for fold_id in sorted(np.unique(fold_ids[train_indices]).tolist()):
        construction = train_indices[fold_ids[train_indices] != fold_id]
        if guidance_mode == "noisy":
            guidance_labels = noisy_labels
        elif guidance_mode == "clean":
            guidance_labels = clean_labels
        else:
            guidance_labels = shuffle_guidance_within_construction(
                construction,
                noisy_labels,
                random_state=int(random_state) + 1009 * (fold_id + 1),
            )
        partition = prune_candidate_tree(
            tree,
            guidance_indices=construction,
            guidance_labels=guidance_labels,
            n_classes=n_classes,
            min_guidance_per_child=min_guidance_per_child,
            log_bf_threshold=log_bf_threshold,
            prior_strength=prior_strength,
        )
        assignments[fold_id] = partition.assignment
        accepted_counts.append(len(partition.accepted_split_ids))
        rejected_counts.append(len(partition.rejected_split_ids))

    result = _crossfit_targets_for_assignments(
        assignments_by_fold=assignments,
        noisy_labels=noisy_labels,
        train_indices=train_indices,
        fold_ids=fold_ids,
        n_classes=int(n_classes),
        target_beta=float(target_beta),
        min_train_per_target=max(1, int(min_train_per_target)),
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "guidance_mode": guidance_mode,
            "candidate_tree_hash": tree.assignment_hash,
            "candidate_leaf_count": len(tree.leaf_ids),
            "mean_accepted_splits": float(np.mean(accepted_counts)),
            "mean_rejected_splits": float(np.mean(rejected_counts)),
            "full_tree_fold_ratio": float(
                np.mean(result.fold_num_balls == len(tree.leaf_ids))
            ),
        }
    )
    return CrossFitTargetResult(
        soft_targets=result.soft_targets,
        local_distributions=result.local_distributions,
        alpha=result.alpha,
        ball_ids=result.ball_ids,
        fold_num_balls=result.fold_num_balls,
        diagnostics=diagnostics,
    )


def build_crossfit_targets_from_assignment(
    assignment,
    noisy_labels,
    train_indices,
    fold_ids,
    n_classes: int,
    target_beta: float = 0.5,
    min_train_per_target: int = 2,
) -> CrossFitTargetResult:
    """Cross-fit target control using a fixed, label-free partition."""
    assignment = np.asarray(assignment, dtype=np.int64)
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if not (
        assignment.shape[0]
        == noisy_labels.shape[0]
        == fold_ids.shape[0]
    ):
        raise ValueError("assignment, noisy_labels, and fold_ids must align")
    assignments = {
        int(fold_id): assignment
        for fold_id in np.unique(fold_ids[train_indices]).tolist()
    }
    return _crossfit_targets_for_assignments(
        assignments_by_fold=assignments,
        noisy_labels=noisy_labels,
        train_indices=train_indices,
        fold_ids=fold_ids,
        n_classes=int(n_classes),
        target_beta=float(target_beta),
        min_train_per_target=max(1, int(min_train_per_target)),
    )


def build_crossfit_budgeted_targets(
    tree: GeometryCandidateTree,
    noisy_labels,
    train_indices,
    fold_ids,
    n_classes: int,
    target_num_balls: int,
    guidance_mode: str = "noisy",
    clean_labels=None,
    target_beta: float = 0.5,
    min_train_per_target: int = 2,
    prior_strength: float = 2.0,
    random_state: int = 0,
) -> CrossFitTargetResult:
    noisy_labels = np.asarray(noisy_labels, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if noisy_labels.shape[0] != tree.n_nodes or fold_ids.shape[0] != tree.n_nodes:
        raise ValueError("tree, noisy_labels, and fold_ids must share node count")
    guidance_mode = str(guidance_mode).strip().lower()
    if guidance_mode not in {"geometry", "noisy", "shuffled", "clean"}:
        raise ValueError(
            "guidance_mode must be geometry, noisy, shuffled, or clean"
        )
    if guidance_mode == "clean":
        if clean_labels is None:
            raise ValueError("clean_labels are required for clean guidance")
        clean_labels = np.asarray(clean_labels, dtype=np.int64)
        if clean_labels.shape[0] != tree.n_nodes:
            raise ValueError("clean_labels length must match node count")

    assignments: dict[int, np.ndarray] = {}
    assignment_hashes: list[str] = []
    selected_jsd_means: list[float] = []
    selected_geometry_means: list[float] = []
    partitions: list[BudgetedPartition] = []
    unique_folds = sorted(np.unique(fold_ids[train_indices]).tolist())
    for fold_id in unique_folds:
        construction = train_indices[fold_ids[train_indices] != fold_id]
        label_assisted = guidance_mode != "geometry"
        if guidance_mode in {"geometry", "noisy"}:
            guidance_labels = noisy_labels
        elif guidance_mode == "clean":
            guidance_labels = clean_labels
        else:
            guidance_labels = shuffle_guidance_within_construction(
                construction,
                noisy_labels,
                random_state=int(random_state) + 1009 * (int(fold_id) + 1),
            )
        partition = cut_candidate_tree_fixed_budget(
            tree=tree,
            target_num_balls=int(target_num_balls),
            guidance_indices=construction,
            guidance_labels=guidance_labels,
            n_classes=int(n_classes),
            label_assisted=label_assisted,
            prior_strength=float(prior_strength),
        )
        assignments[int(fold_id)] = partition.assignment
        partitions.append(partition)
        assignment_hashes.append(partition.assignment_hash)
        selected_jsd_means.append(
            float(np.mean(list(partition.normalized_jsd.values())))
            if partition.normalized_jsd
            else 0.0
        )
        selected_geometry_means.append(
            float(
                np.mean(
                    [tree.nodes[node_id].geometry_gain for node_id in partition.selected_split_ids]
                )
            )
            if partition.selected_split_ids
            else 0.0
        )

    result = _crossfit_targets_for_assignments(
        assignments_by_fold=assignments,
        noisy_labels=noisy_labels,
        train_indices=train_indices,
        fold_ids=fold_ids,
        n_classes=int(n_classes),
        target_beta=float(target_beta),
        min_train_per_target=max(1, int(min_train_per_target)),
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "guidance_mode": guidance_mode,
            "candidate_tree_hash": tree.assignment_hash,
            "candidate_leaf_count": len(tree.leaf_ids),
            "target_num_balls": int(target_num_balls),
            "fold_assignment_hashes": assignment_hashes,
            "mean_selected_jsd": float(np.mean(selected_jsd_means)),
            "mean_selected_geometry_gain": float(np.mean(selected_geometry_means)),
            "exact_budget": bool(
                partitions
                and all(
                    partition.num_balls == int(target_num_balls)
                    for partition in partitions
                )
            ),
        }
    )
    return CrossFitTargetResult(
        soft_targets=result.soft_targets,
        local_distributions=result.local_distributions,
        alpha=result.alpha,
        ball_ids=result.ball_ids,
        fold_num_balls=result.fold_num_balls,
        diagnostics=diagnostics,
    )
