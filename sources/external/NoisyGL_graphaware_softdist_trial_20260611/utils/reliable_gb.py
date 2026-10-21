"""Reciprocal-kNN reliable granular balls.

This is the construction validated in the frozen-GRACE diagnostics: ordinary feature/embedding
granular balls collapse in purity on noisy/raw spaces, but balls grown from *mutual* (reciprocal)
nearest-neighbour agreement stay pure across representation spaces. Reliability comes from how the
ball is grown (mutual kNN), not from graph topology (which was shown inert on Cora/CiteSeer).

The output mirrors `utils.ordinary_gb.GranularBallResult` so the existing soft-target / reweight
pipeline and diagnostics consume it unchanged. Clean labels are never used here.
"""
from dataclasses import dataclass

import numpy as np
import torch


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


def _knn_sets(cosine, k):
    """For each row i, the set of its top-k columns excluding self."""
    n = cosine.shape[0]
    if n <= 1:
        return [set() for _ in range(n)]
    kk = max(1, min(int(k) + 1, n))
    # argpartition for the top-kk indices per row (unordered is fine for set membership).
    idx = np.argpartition(-cosine, kth=kk - 1, axis=1)[:, :kk]
    out = []
    for i in range(n):
        s = set(int(j) for j in idx[i].tolist())
        s.discard(i)
        out.append(s)
    return out


def build_reliable_granular_balls(
    embeddings,
    k=15,
    min_ball_size=3,
    max_ball_size=128,
    random_state=None,
):
    """Grow reciprocal-kNN balls.

    Anchor = highest local density (mean cosine to its kNN) among unassigned nodes. A node ``j`` joins
    anchor ``c`` iff they are *mutual* kNN (``j in kNN(c)`` AND ``c in kNN(j)``) and still unassigned.
    Every node is covered (an isolated anchor forms a singleton ball). ``max_ball_size`` caps growth.
    """
    z = _l2_normalize(_to_numpy_embeddings(embeddings))
    n_nodes = z.shape[0]
    if n_nodes == 0:
        raise ValueError("embeddings must contain at least one node")

    k = max(1, min(int(k), n_nodes - 1)) if n_nodes > 1 else 1
    min_ball_size = max(1, int(min_ball_size))
    max_ball_size = max(1, int(max_ball_size))
    rng = np.random.RandomState(0 if random_state is None else int(random_state))

    cosine = z @ z.T
    knn = _knn_sets(cosine, k)
    # Local density = mean cosine to own kNN; interior nodes score high → reliable anchors.
    density = np.array([
        float(np.mean([cosine[i, j] for j in knn[i]])) if knn[i] else -1.0
        for i in range(n_nodes)
    ], dtype=np.float32)
    tie_break = rng.rand(n_nodes).astype(np.float32)

    unassigned = set(range(n_nodes))
    balls_idx = []
    while unassigned:
        avail = np.fromiter(unassigned, dtype=np.int64)
        scores = density[avail] + 1e-6 * tie_break[avail]
        center = int(avail[int(np.argmax(scores))])

        members = [center]
        # Reciprocal-kNN candidates of the anchor, ordered by cosine to anchor.
        cands = [j for j in knn[center] if j in unassigned and center in knn[j]]
        cands.sort(key=lambda j: float(cosine[center, j]), reverse=True)
        for j in cands:
            if len(members) >= max_ball_size:
                break
            members.append(int(j))

        balls_idx.append(np.asarray(members, dtype=np.int64))
        for j in members:
            unassigned.discard(int(j))

    assignment = np.empty(n_nodes, dtype=np.int64)
    ball_stats = []
    for ball_id, indices in enumerate(balls_idx):
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
        "reliable_k": int(k),
    }
    return GranularBallResult(assignment=assignment, balls=ball_stats, summary=summary)


# Re-export so callers can build the size-matched random control from one module.
from utils.ordinary_gb import build_random_size_matched_assignment  # noqa: E402,F401
