from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class GranularBallResult:
    assignment: np.ndarray
    balls: list
    summary: dict


def _to_numpy_embeddings(embeddings):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().float().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def _l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _measure_ball(z, indices, ball_id):
    center = z[indices].mean(axis=0)
    distances = np.linalg.norm(z[indices] - center, axis=1)
    return {
        "ball_id": int(ball_id),
        "size": int(indices.size),
        "radius": float(distances.max()) if distances.size else 0.0,
        "avg_distance": float(distances.mean()) if distances.size else 0.0,
    }


def build_ordinary_granular_balls(
    embeddings,
    max_balls=64,
    min_ball_size=32,
    split_improvement=0.05,
    random_state=None,
):
    z = _l2_normalize(_to_numpy_embeddings(embeddings))
    n_nodes = z.shape[0]
    if n_nodes == 0:
        raise ValueError("embeddings must contain at least one node")

    max_balls = max(1, min(int(max_balls), n_nodes))
    min_ball_size = max(1, int(min_ball_size))
    split_improvement = max(0.0, float(split_improvement))
    base_seed = 0 if random_state is None else int(random_state)

    balls = [np.arange(n_nodes)]
    iteration = 0
    while len(balls) < max_balls:
        candidates = []
        for pos, indices in enumerate(balls):
            if indices.size < 2 * min_ball_size:
                continue
            stat = _measure_ball(z, indices, pos)
            candidates.append((stat["avg_distance"], pos, stat))

        if not candidates:
            break

        accepted = False
        for _, pos, parent_stat in sorted(candidates, reverse=True):
            indices = balls[pos]
            labels = KMeans(n_clusters=2, n_init=10, random_state=base_seed + iteration + pos).fit_predict(z[indices])
            if np.unique(labels).size < 2:
                continue

            left = indices[labels == 0]
            right = indices[labels == 1]
            if left.size < min_ball_size or right.size < min_ball_size:
                continue

            left_stat = _measure_ball(z, left, 0)
            right_stat = _measure_ball(z, right, 1)
            new_avg = (
                left_stat["avg_distance"] * left.size + right_stat["avg_distance"] * right.size
            ) / indices.size
            old_avg = parent_stat["avg_distance"]
            if old_avg <= 1e-12 or new_avg <= old_avg * (1.0 - split_improvement):
                balls[pos] = left
                balls.append(right)
                accepted = True
                break

        if not accepted:
            break
        iteration += 1

    assignment = np.empty(n_nodes, dtype=np.int64)
    ball_stats = []
    for ball_id, indices in enumerate(balls):
        assignment[indices] = ball_id
        ball_stats.append(_measure_ball(z, indices, ball_id))

    sizes = np.array([ball["size"] for ball in ball_stats], dtype=np.float32)
    radii = np.array([ball["radius"] for ball in ball_stats], dtype=np.float32)
    summary = {
        "num_balls": int(len(ball_stats)),
        "mean_ball_size": float(sizes.mean()),
        "max_ball_size": int(sizes.max()),
        "singleton_ratio": float(np.mean(sizes == 1)),
        "mean_radius": float(radii.mean()),
    }
    return GranularBallResult(assignment=assignment, balls=ball_stats, summary=summary)


def _to_numpy_labels(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().long().numpy()
    return np.asarray(labels, dtype=np.int64)


def _train_mask_to_bool(train_mask, n_nodes):
    if isinstance(train_mask, torch.Tensor):
        train_mask = train_mask.detach().cpu().numpy()
    train_mask = np.asarray(train_mask)
    if train_mask.dtype == np.bool_:
        if train_mask.shape[0] != n_nodes:
            raise ValueError("boolean train_mask length must match node count")
        return train_mask.copy()
    is_train = np.zeros(n_nodes, dtype=bool)
    is_train[train_mask.astype(np.int64)] = True
    return is_train


def build_purity_adaptive_granular_balls(
    embeddings,
    noisy_label,
    train_mask,
    n_classes,
    purity_threshold=0.6,
    min_train_per_child=4,
    min_train_to_split=16,
    min_ball_size=8,
    max_balls=128,
    min_purity_gain=0.05,
    split_attempts=1,
    random_state=None,
):
    """Noisy-supervised adaptive granular balls with gain-based split acceptance.

    Split moves stay geometric (2-means on L2-normalized embeddings, label-blind);
    labels are only used to ACCEPT or REJECT a candidate split. A candidate is
    accepted only when the train-count-weighted noisy purity of the two children
    exceeds the parent's purity by at least `min_purity_gain`, and every child
    keeps at least `min_train_per_child` train nodes and `min_ball_size` members.
    Balls whose noisy purity already reaches `purity_threshold` (the
    noise-consistent ceiling) or whose train count drops below
    `min_train_to_split` stop splitting. Clean regions therefore stay coarse
    (robust voting) and only splits that measurably separate label structure
    are taken.

    Only noisy train labels are used during construction; val/test labels never
    enter. This is noisy-supervised ball construction, not unsupervised GB.
    """
    z = _l2_normalize(_to_numpy_embeddings(embeddings))
    n_nodes = z.shape[0]
    if n_nodes == 0:
        raise ValueError("embeddings must contain at least one node")

    labels = _to_numpy_labels(noisy_label)
    if labels.shape[0] != n_nodes:
        raise ValueError("noisy_label length must match node count")
    is_train = _train_mask_to_bool(train_mask, n_nodes)

    n_classes = int(n_classes)
    purity_threshold = float(purity_threshold)
    min_train_per_child = max(1, int(min_train_per_child))
    min_train_to_split = max(2 * min_train_per_child, int(min_train_to_split))
    min_ball_size = max(1, int(min_ball_size))
    max_balls = max(1, min(int(max_balls), n_nodes))
    min_purity_gain = max(0.0, float(min_purity_gain))
    split_attempts = max(1, int(split_attempts))
    base_seed = 0 if random_state is None else int(random_state)

    def train_purity(indices):
        t = indices[is_train[indices]]
        if t.size == 0:
            return 0, 0.0
        counts = np.bincount(labels[t], minlength=n_classes)
        return int(t.size), float(counts.max() / t.size)

    balls = [np.arange(n_nodes)]
    terminal = set()
    num_splits = 0
    num_rejected = 0
    attempt = 0
    while len(balls) < max_balls:
        candidates = []
        for pos, indices in enumerate(balls):
            if pos in terminal:
                continue
            if indices.size < 2 * min_ball_size:
                terminal.add(pos)
                continue
            t_count, purity = train_purity(indices)
            if t_count < min_train_to_split:
                terminal.add(pos)
                continue
            if purity >= purity_threshold:
                terminal.add(pos)
                continue
            candidates.append((purity, -t_count, pos))

        if not candidates:
            break

        candidates.sort()
        parent_purity, _, pos = candidates[0]
        indices = balls[pos]
        best_split = None
        best_child_purity = -1.0
        for trial in range(split_attempts):
            km = KMeans(n_clusters=2, n_init=10, random_state=base_seed + attempt).fit_predict(z[indices])
            attempt += 1
            if np.unique(km).size < 2:
                continue
            left = indices[km == 0]
            right = indices[km == 1]
            left_train, left_purity = train_purity(left)
            right_train, right_purity = train_purity(right)
            if (
                left.size < min_ball_size
                or right.size < min_ball_size
                or left_train < min_train_per_child
                or right_train < min_train_per_child
            ):
                continue
            child_purity = (
                left_purity * left_train + right_purity * right_train
            ) / max(1, left_train + right_train)
            if child_purity < parent_purity + min_purity_gain:
                continue
            if child_purity > best_child_purity:
                best_child_purity = child_purity
                best_split = (left, right)
        if best_split is None:
            terminal.add(pos)
            num_rejected += 1
            continue
        balls[pos] = best_split[0]
        balls.append(best_split[1])
        num_splits += 1

    assignment = np.empty(n_nodes, dtype=np.int64)
    ball_stats = []
    zero_train_balls = 0
    for ball_id, indices in enumerate(balls):
        assignment[indices] = ball_id
        stat = _measure_ball(z, indices, ball_id)
        t_count, purity = train_purity(indices)
        stat["train_count"] = t_count
        stat["noisy_train_purity"] = purity
        if t_count == 0:
            zero_train_balls += 1
        ball_stats.append(stat)

    sizes = np.array([ball["size"] for ball in ball_stats], dtype=np.float32)
    radii = np.array([ball["radius"] for ball in ball_stats], dtype=np.float32)
    train_counts = np.array([ball["train_count"] for ball in ball_stats], dtype=np.float32)
    summary = {
        "num_balls": int(len(ball_stats)),
        "mean_ball_size": float(sizes.mean()),
        "max_ball_size": int(sizes.max()),
        "singleton_ratio": float(np.mean(sizes == 1)),
        "mean_radius": float(radii.mean()),
        "ball_mode": "adaptive",
        "purity_threshold": purity_threshold,
        "min_purity_gain": min_purity_gain,
        "split_attempts": int(split_attempts),
        "min_train_per_child": int(min_train_per_child),
        "min_train_to_split": int(min_train_to_split),
        "adaptive_num_splits": int(num_splits),
        "adaptive_num_rejected_splits": int(num_rejected),
        "mean_train_per_ball": float(train_counts.mean()),
        "zero_train_balls": int(zero_train_balls),
    }
    return GranularBallResult(assignment=assignment, balls=ball_stats, summary=summary)


def build_random_size_matched_assignment(assignment, random_state=None):
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1:
        raise ValueError("assignment must be a 1-D array")

    rng = np.random.RandomState(random_state)
    nodes = np.arange(assignment.shape[0])
    rng.shuffle(nodes)

    shuffled = np.empty_like(assignment)
    offset = 0
    for ball_id in sorted(np.unique(assignment).tolist()):
        size = int(np.sum(assignment == ball_id))
        shuffled[nodes[offset: offset + size]] = ball_id
        offset += size
    return shuffled
