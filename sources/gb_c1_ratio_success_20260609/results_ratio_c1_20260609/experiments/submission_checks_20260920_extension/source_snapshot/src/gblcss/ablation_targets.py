"""Target, grouping, and representation controls reported in the manuscript."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .ablations import build_self_vote_ablation_targets
from .data import GraphData
from .granular_ball import build_granular_balls
from .targets import TargetResult, _consensus_weight, _indices, build_leave_one_out_targets


ABLATION_METHODS = (
    "gblcss_self_vote", "uniform_smoothing", "prior_smoothing", "knn",
    "kmeans", "raw_features", "propagated_features",
)


def _assignment_hash(assignment: np.ndarray) -> str:
    values = np.ascontiguousarray(assignment, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _prepare_features(features: torch.Tensor | None, n_nodes: int) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise ValueError("representation must be a two-dimensional tensor")
    features = features.detach().cpu().float()
    if features.layout != torch.strided:
        features = features.to_dense()
    if features.ndim != 2 or features.shape[0] != n_nodes or features.shape[1] == 0:
        raise ValueError("representation shape must match the graph node count")
    if not torch.isfinite(features).all():
        raise ValueError("representation must contain only finite values")
    return F.normalize(features, p=2, dim=1, eps=1e-12)


def _representation_source(data: GraphData, method: str) -> torch.Tensor:
    features = data.x.detach().cpu().float()
    if features.layout != torch.strided:
        features = features.to_dense()
    if method == "propagated_features":
        edges = data.edge_index.detach().cpu().long()
        n_nodes = int(features.shape[0])
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= n_nodes):
            raise ValueError("edge_index contains an out-of-range node")
        # Match NoisyGL's raw COO adjacency, including its row/column orientation.
        adjacency = torch.sparse_coo_tensor(
            edges, torch.ones(edges.shape[1]), (n_nodes, n_nodes),
        ).coalesce()
        for _ in range(2):
            features = torch.sparse.mm(adjacency, features)
    return features


def _alpha_diagnostics(alpha: torch.Tensor, train: np.ndarray) -> dict:
    values = alpha[torch.as_tensor(train, dtype=torch.long, device=alpha.device)]
    return {
        "alpha_positive_train_count": int((values > 0).sum().item()),
        "alpha_positive_train_ratio": float((values > 0).float().mean().item()) if values.numel() else 0.0,
        "mean_alpha_train": float(values.mean().item()) if values.numel() else 0.0,
    }


def _smoothing_targets(labels: torch.Tensor, train: np.ndarray, n_classes: int,
                       method: str, epsilon: float) -> tuple[torch.Tensor, dict]:
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("smoothing_epsilon must lie in [0, 1]")
    nodes = torch.as_tensor(train, dtype=torch.long, device=labels.device)
    targets = F.one_hot(labels, n_classes).float()
    if method == "uniform_smoothing" or nodes.numel() == 0:
        reference = torch.full((n_classes,), 1.0 / n_classes, device=labels.device)
    else:
        counts = torch.bincount(labels[nodes], minlength=n_classes).float()
        reference = counts / counts.sum().clamp_min(1.0)
    targets[nodes] = (1.0 - epsilon) * targets[nodes] + epsilon * reference.unsqueeze(0)
    alpha = torch.zeros(labels.shape[0], device=labels.device)
    alpha[nodes] = epsilon
    return targets, {
        "target_builder": method,
        "smoothing_distribution": method.removesuffix("_smoothing"),
        "smoothing_epsilon": epsilon,
        "smoothing_reference": reference.cpu().tolist(),
        **_alpha_diagnostics(alpha, train),
    }


def _knn_targets(features: torch.Tensor, labels: torch.Tensor, train: np.ndarray,
                 n_classes: int, k: int, beta: float) -> TargetResult:
    n_train = int(train.size)
    noisy = labels.cpu().numpy()
    distributions = np.zeros((n_train, n_classes), dtype=np.float64)
    if k == 0:
        distributions[np.arange(n_train), noisy[train]] = 1.0
    else:
        train_features = features[torch.as_tensor(train, dtype=torch.long)]
        similarity = train_features @ train_features.T
        similarity.fill_diagonal_(float("-inf"))
        neighbors = torch.topk(similarity, k=k, dim=1, largest=True).indices.numpy()
        for row, positions in enumerate(neighbors):
            counts = np.bincount(noisy[train[positions]], minlength=n_classes).astype(np.float64)
            distributions[row] = counts / max(1.0, float(counts.sum()))

    # Preserve the original float64 distribution path and its isclose tie rule.
    distributions /= np.clip(distributions.sum(axis=1, keepdims=True), 1e-12, None)
    majority = distributions.argmax(axis=1).astype(np.int64)
    margin = distributions.max(axis=1)
    tied = np.isclose(distributions, margin[:, None]).sum(axis=1) > 1
    weights = np.asarray([_consensus_weight(float(p), n_classes) for p in margin], dtype=np.float64)
    local = (1.0 - beta) * distributions + beta * np.eye(n_classes, dtype=np.float64)[majority]
    local[tied] = distributions[tied]
    nodes = torch.as_tensor(train, dtype=torch.long, device=labels.device)
    a = torch.as_tensor(weights, dtype=torch.float32, device=labels.device)
    targets = F.one_hot(labels, n_classes).float()
    targets[nodes] = (1.0 - a[:, None]) * targets[nodes] + a[:, None] * torch.as_tensor(
        local, dtype=torch.float32, device=labels.device)
    alpha = torch.zeros(labels.shape[0], device=labels.device)
    alpha[nodes] = a
    pseudo = labels.clone()
    untied = torch.as_tensor(train[~tied], dtype=torch.long, device=labels.device)
    pseudo[untied] = torch.as_tensor(majority[~tied], dtype=torch.long, device=labels.device)
    return TargetResult(targets, alpha, pseudo, {
        "tie_mode": "distribution_only", "target_beta": beta,
        "tie_affected_node_count": int(tied.sum()),
        "tied_train_nodes": int((tied & (weights > 0)).sum()),
        "eligible_train_nodes": int((weights > 0).sum()),
        "knn_exclude_self": 1, "matched_k": k,
    })


def _self_vote_targets(labels: torch.Tensor, assignment: np.ndarray, train: np.ndarray,
                       n_classes: int, min_support: int, beta: float) -> TargetResult:
    result = build_self_vote_ablation_targets(
        labels, assignment, train, n_classes, min_train_per_ball=min_support, beta=beta,
    )
    targets = F.one_hot(labels, n_classes).float()
    # The shared helper uses scalar alpha arithmetic; NoisyGL's inclusive branch
    # rounds alpha to float32 BEFORE subtraction. Preserve that rounding here.
    for group in sorted(np.unique(assignment)):
        nodes = torch.as_tensor(train[assignment[train] == group], device=labels.device)
        if nodes.numel() < min_support or not torch.any(result.alpha[nodes] > 0):
            continue
        counts = torch.bincount(labels[nodes], minlength=n_classes)
        distribution = counts.float() / float(counts.sum().item())
        if int((counts == counts.max()).sum().item()) > 1:
            local = distribution
        else:
            majority = F.one_hot(counts.argmax(), n_classes).float()
            local = (1.0 - beta) * distribution + beta * majority
        a = result.alpha[nodes, None]
        targets[nodes] = (1.0 - a) * targets[nodes] + a * local
    return TargetResult(targets, result.alpha, result.pseudo_label, result.diagnostics)


def build_ablation_targets(
    data: GraphData,
    representation: torch.Tensor | None,
    method: str,
    config: dict,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict]:
    """Return detached all-node float32 targets and JSON-compatible diagnostics.

    Only training rows are altered. Ordered ``data.train_index`` takes precedence
    over the mask, as in the original runner (important for kNN distance ties).
    ``config['method']`` uses the main GB-LCSS settings; the optional
    ``smoothing_epsilon`` defaults to the original fixed 0.1, not matched alpha.
    Smoothing and raw/A^2X controls do not require a frozen representation.

    kNN matches round(mean nonzero GB training support), then caps at n_train-1;
    it does not subtract one from that mean or impose the GB support threshold.
    kmeans matches the actual GB count, not the maximum budget. Formal voting is
    distribution_only + LOO, except the explicitly inclusive self-vote control.
    Grouping uses the supplied run seed, not global RNG state or a seed offset.
    """
    if method not in ABLATION_METHODS:
        raise ValueError(f"unsupported ablation method: {method}")
    settings = config["method"]
    labels = (data.noisy_y if data.noisy_y is not None else data.y).detach().long()
    n_nodes, n_classes = data.num_nodes, data.num_classes
    if n_nodes < 1 or labels.ndim != 1 or labels.shape[0] != n_nodes:
        raise ValueError("labels must match a non-empty graph")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
        raise ValueError("labels are outside the class range")
    mask = data.train_index if data.train_index is not None else data.train_mask
    if torch.as_tensor(mask).dtype == torch.bool and torch.as_tensor(mask).numel() != n_nodes:
        raise ValueError("train_mask length must match the graph")
    train = _indices(mask)
    if train.size and (train.min() < 0 or train.max() >= n_nodes or np.unique(train).size != train.size):
        raise ValueError("training indices must be unique and in range")
    diagnostics = {"method": method, "seed": int(seed)}
    if method in {"uniform_smoothing", "prior_smoothing"}:
        targets, detail = _smoothing_targets(labels, train, n_classes, method,
                                             float(settings.get("smoothing_epsilon", 0.1)))
        return targets.detach(), {**diagnostics, **detail}

    if settings.get("tie_mode", "distribution_only") != "distribution_only":
        raise ValueError("current-paper tie_mode must be distribution_only")
    if method != "gblcss_self_vote" and settings.get("leave_one_out", True) is not True:
        raise ValueError("current-paper leave_one_out must be true")
    beta = float(settings.get("beta", 0.5))
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    min_support = int(settings.get("min_train_per_ball", 2))
    if min_support < 1:
        raise ValueError("min_train_per_ball must be positive")

    if method in {"raw_features", "propagated_features"}:
        representation = _representation_source(data, method)
        diagnostics.update({
            "representation_source": "raw_x" if method == "raw_features" else "adj2_x",
            "propagation_steps": 0 if method == "raw_features" else 2,
            "representation_normalization": "final_l2",
        })
        if method == "propagated_features":
            diagnostics.update(adjacency_normalization="none", added_self_loops=0)
    else:
        diagnostics["representation_source"] = "grace_h"
    features = _prepare_features(representation, n_nodes)
    balls = build_granular_balls(
        features, max_balls=settings.get("max_balls", 64),
        min_ball_size=settings.get("min_ball_size", 32),
        split_improvement=settings.get("split_improvement", 0.05), random_state=int(seed),
    )
    assignment = balls.assignment
    diagnostics.update(balls.summary)
    if method in {"knn", "kmeans"}:
        counts = [int(np.sum(assignment[train] == group)) for group in sorted(np.unique(assignment))]
        nonempty = [count for count in counts if count > 0]
        mean_support = float(np.mean(nonempty)) if nonempty else 1.0
        diagnostics.update({
            "matched_num_balls": len(counts), "matched_reference_num_balls": len(counts),
            "matched_mean_train_per_ball": mean_support,
            "matched_zero_train_balls": counts.count(0),
            "reference_assignment_sha256": _assignment_hash(assignment),
        })
        if method == "knn":
            k = max(0, min(int(round(mean_support)), int(train.size) - 1))
            result = _knn_targets(features, labels, train, n_classes, k, beta)
        else:
            n_clusters = len(counts)
            # sklearn's integer seed initializes the original NumPy RandomState.
            assignment = (np.zeros(n_nodes, dtype=np.int64) if n_clusters == 1 else
                          KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed)).fit_predict(features.numpy()))
            diagnostics["kmeans_n_clusters"] = n_clusters
            result = build_leave_one_out_targets(labels, assignment, train, n_classes,
                        min_train_per_ball=min_support, beta=beta)
    elif method == "gblcss_self_vote":
        result = _self_vote_targets(labels, assignment, train, n_classes, min_support, beta)
    else:
        result = build_leave_one_out_targets(labels, assignment, train, n_classes,
                                             min_train_per_ball=min_support, beta=beta)
    if method != "knn":
        diagnostics["assignment_sha256"] = _assignment_hash(assignment)
    return result.soft_target.detach(), {
        **diagnostics, **result.diagnostics, **_alpha_diagnostics(result.alpha, train),
        "target_builder": method, "soft_alpha_baseline": "multiclass", "soft_target_mode": "distribution",
    }
