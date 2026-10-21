from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.module.GNNs import GCN
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings, partition_purity_diagnostics
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


SOFTDIST_METHOD = "frozengrace_gb_gcn_softdist_xh"
DEFAULT_UNDEFINED_SCORE = 0.5


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _dataset_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    aliases = {
        "amazoncomputers": "amazoncom",
        "amazon-computers": "amazoncom",
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
        "coauthor-cs": "coauthorcs",
        "coauthorcs": "coauthorcs",
        "coauthor-cs": "coauthorcs",
    }
    return aliases.get(key, key)


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values) -> float:
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if len(values) <= 1:
        return 0.0 if values else float("nan")
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_auprc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size <= 1 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _zscore(scores):
    scores = np.asarray(scores, dtype=np.float64)
    std = float(np.nanstd(scores))
    if std <= 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - float(np.nanmean(scores))) / std


def _mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.flatnonzero(mask)
    return mask.astype(np.int64)


def _as_noisy_numpy(noisy_label) -> np.ndarray:
    if isinstance(noisy_label, torch.Tensor):
        return noisy_label.detach().cpu().long().numpy()
    return np.asarray(noisy_label, dtype=np.int64)


def _as_numpy_assignment(assignment) -> np.ndarray:
    return np.asarray(assignment, dtype=np.int64)


def _config_path_for(method: str, dataset_key: str) -> Path:
    return REPO_ROOT / "config" / method / f"{method}_{dataset_key}.yaml"


def _conf_get(section, key, default=None):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def load_softdist_xh_gb_config(dataset_key: str, method: str = SOFTDIST_METHOD) -> dict:
    dataset_key = _dataset_key(dataset_key)
    config_path = _config_path_for(method, dataset_key)
    conf = load_conf(None, method, dataset_key)
    gb_conf = getattr(conf, "gb", None)
    if gb_conf is None:
        raise KeyError(f"Missing gb section in {config_path}")
    embedding_path = _conf_get(gb_conf, "embedding_path")
    if embedding_path is None:
        raise KeyError(f"Missing gb.embedding_path in {config_path}")
    embedding_path = Path(str(embedding_path))
    if not embedding_path.is_absolute():
        candidates = [
            (config_path.parent / embedding_path).resolve(),
            (REPO_ROOT / embedding_path).resolve(),
            (Path.cwd() / embedding_path).resolve(),
        ]
        embedding_path = next((path for path in candidates if path.exists()), candidates[0])
    return {
        "config_path": config_path,
        "embedding_path": embedding_path,
        "max_balls": int(_conf_get(gb_conf, "max_balls")),
        "min_ball_size": int(_conf_get(gb_conf, "min_ball_size")),
        "split_improvement": float(_conf_get(gb_conf, "split_improvement")),
        "min_train_per_ball": int(_conf_get(gb_conf, "min_train_per_ball", 2)),
    }


def _jsd_onehot(label: int, dist: np.ndarray, eps: float = 1e-12) -> float:
    dist = np.asarray(dist, dtype=np.float64)
    dist = np.clip(dist, eps, 1.0)
    dist = dist / float(dist.sum())
    onehot = np.full_like(dist, eps, dtype=np.float64)
    onehot[int(label)] = 1.0
    onehot = onehot / float(onehot.sum())
    mix = 0.5 * (onehot + dist)
    kl_one = float(np.sum(onehot * np.log(onehot / mix)))
    kl_dist = float(np.sum(dist * np.log(dist / mix)))
    return 0.5 * (kl_one + kl_dist)


def ball_leave_one_out_scores(
    noisy_label,
    assignment,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
):
    noisy_np = _as_noisy_numpy(noisy_label)
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    jsd_scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    for ball_id in sorted(np.unique(assignment[train_idx]).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball) or ball_train.size <= 1:
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float64)
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            loo_counts = counts.copy()
            label = int(noisy_np[node])
            loo_counts[label] -= 1.0
            total = float(loo_counts.sum())
            if total <= 0:
                continue
            dist = loo_counts / total
            scores[pos] = 1.0 - float(dist[label])
            jsd_scores[pos] = _jsd_onehot(label, dist)
            eligible[pos] = True
    return scores, eligible, jsd_scores


def ball_entropy_scores(
    noisy_label,
    assignment,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
):
    noisy_np = _as_noisy_numpy(noisy_label)
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    max_entropy = math.log(max(2, int(n_classes)))
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    for ball_id in sorted(np.unique(assignment[train_idx]).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball) or ball_train.size <= 1:
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float64)
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            loo_counts = counts.copy()
            loo_counts[int(noisy_np[node])] -= 1.0
            total = float(loo_counts.sum())
            if total <= 0:
                continue
            dist = loo_counts / total
            nz = dist[dist > 0]
            entropy = -float(np.sum(nz * np.log(nz)))
            scores[pos] = entropy / max_entropy
            eligible[pos] = True
    return scores, eligible


def center_distance_scores(
    embeddings,
    assignment,
    train_idx: np.ndarray,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
    eps: float = 1e-12,
):
    h = embeddings.detach().cpu().float().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    for ball_id in sorted(np.unique(assignment).tolist()):
        nodes = np.flatnonzero(assignment == ball_id)
        if nodes.size <= 1:
            continue
        center = h[nodes].mean(axis=0)
        dists = np.linalg.norm(h[nodes] - center, axis=1)
        radius = float(np.mean(dists))
        if radius <= eps:
            continue
        ball_train = [node for node in nodes.tolist() if int(node) in train_pos]
        for node in ball_train:
            pos = train_pos[int(node)]
            dist2 = float(np.sum((h[int(node)] - center) ** 2))
            scores[pos] = 1.0 - math.exp(-dist2 / (radius * radius + eps))
            eligible[pos] = True
    return scores, eligible


def _majority_labels_by_ball(noisy_label, assignment, train_idx, n_classes, min_train_per_ball):
    noisy_np = _as_noisy_numpy(noisy_label)
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    majority = {}
    eligible_balls = set()
    for ball_id in sorted(np.unique(assignment[train_idx]).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float64)
        majority[int(ball_id)] = int(np.argmax(counts))
        eligible_balls.add(int(ball_id))
    return majority, eligible_balls


def _ball_centers(embeddings, assignment):
    h = embeddings.detach().cpu().float().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    assignment = _as_numpy_assignment(assignment)
    centers = {}
    for ball_id in sorted(np.unique(assignment).tolist()):
        nodes = np.flatnonzero(assignment == ball_id)
        if nodes.size:
            centers[int(ball_id)] = h[nodes].mean(axis=0)
    return centers


def _nearest_ball_ids(centers: dict[int, np.ndarray], ball_id: int, k: int):
    if ball_id not in centers:
        return []
    other = [bid for bid in centers.keys() if int(bid) != int(ball_id)]
    if not other:
        return []
    center = centers[int(ball_id)]
    other.sort(key=lambda bid: float(np.linalg.norm(center - centers[int(bid)])))
    return other[: max(0, int(k))]


def ball_neighbor_consistency_scores(
    embeddings,
    noisy_label,
    assignment,
    train_idx: np.ndarray,
    n_classes: int,
    ball_neighbor_k: int,
    min_train_per_ball: int,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
):
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    majority, eligible_balls = _majority_labels_by_ball(noisy_label, assignment, train_idx, n_classes, min_train_per_ball)
    centers = _ball_centers(embeddings, assignment)
    for ball_id in sorted(eligible_balls):
        neighbors = [bid for bid in _nearest_ball_ids(centers, ball_id, ball_neighbor_k) if bid in eligible_balls]
        if not neighbors:
            continue
        same = [int(majority[bid] == majority[ball_id]) for bid in neighbors]
        score = 1.0 - float(np.mean(same))
        ball_train = train_idx[assignment[train_idx] == ball_id]
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            scores[pos] = score
            eligible[pos] = True
    return scores, eligible


def _edge_index_from_graph(edge_index=None, adj=None):
    if edge_index is not None:
        if isinstance(edge_index, torch.Tensor):
            return edge_index.detach().cpu().long().numpy()
        return np.asarray(edge_index, dtype=np.int64)
    if adj is None:
        raise ValueError("Either edge_index or adj must be provided")
    if isinstance(adj, torch.Tensor) and getattr(adj, "is_sparse", False):
        adj = adj.coalesce()
        return adj.indices().detach().cpu().long().numpy()
    if isinstance(adj, torch.Tensor):
        row, col = torch.nonzero(adj.detach().cpu(), as_tuple=True)
        return torch.stack([row, col], dim=0).long().numpy()
    arr = np.asarray(adj)
    row, col = np.nonzero(arr)
    return np.stack([row, col], axis=0).astype(np.int64)


def graph_density_scores(
    embeddings,
    noisy_label,
    assignment,
    train_idx: np.ndarray,
    edge_index=None,
    adj=None,
    n_classes: int = 0,
    ball_neighbor_k: int = 10,
    min_train_per_ball: int = 2,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
):
    assignment = _as_numpy_assignment(assignment)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    majority, eligible_balls = _majority_labels_by_ball(noisy_label, assignment, train_idx, n_classes, min_train_per_ball)
    centers = _ball_centers(embeddings, assignment)
    edges = _edge_index_from_graph(edge_index=edge_index, adj=adj)
    row = edges[0].astype(np.int64)
    col = edges[1].astype(np.int64)
    non_self = row != col
    row = row[non_self]
    col = col[non_self]
    for ball_id in sorted(eligible_balls):
        neighbors = [bid for bid in _nearest_ball_ids(centers, ball_id, ball_neighbor_k) if bid in eligible_balls]
        if not neighbors:
            continue
        neighbor_set = set(int(bid) for bid in neighbors)
        same_label_set = {int(bid) for bid in neighbors if majority[int(bid)] == majority[int(ball_id)]}
        in_ball = assignment[row] == int(ball_id)
        edge_neighbor_balls = assignment[col[in_ball]]
        denom = int(sum(int(bid) in neighbor_set for bid in edge_neighbor_balls.tolist()))
        if denom <= 0:
            continue
        numer = int(sum(int(bid) in same_label_set for bid in edge_neighbor_balls.tolist()))
        score = 1.0 - float(numer / denom)
        ball_train = train_idx[assignment[train_idx] == ball_id]
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            scores[pos] = score
            eligible[pos] = True
    return scores, eligible


def graph_neighbor_scores(noisy_label, train_idx: np.ndarray, edge_index=None, adj=None, undefined_score=DEFAULT_UNDEFINED_SCORE):
    noisy_np = _as_noisy_numpy(noisy_label)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_set = set(int(node) for node in train_idx.tolist())
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    scores = np.full(train_idx.shape[0], float(undefined_score), dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    edges = _edge_index_from_graph(edge_index=edge_index, adj=adj)
    row = edges[0].astype(np.int64)
    col = edges[1].astype(np.int64)
    neighbor_labels = defaultdict(list)
    for src, dst in zip(row.tolist(), col.tolist()):
        if int(src) == int(dst):
            continue
        if int(src) in train_set and int(dst) in train_set:
            neighbor_labels[int(src)].append(int(noisy_np[int(dst)]))
    for node in train_idx.tolist():
        labels = neighbor_labels.get(int(node), [])
        if not labels:
            continue
        agree = float(np.mean(np.asarray(labels, dtype=np.int64) == int(noisy_np[int(node)])))
        scores[train_pos[int(node)]] = 1.0 - agree
        eligible[train_pos[int(node)]] = True
    return scores, eligible


def lp_scores(adj, noisy_label, train_idx: np.ndarray, n_classes: int, steps: int = 2):
    if not isinstance(adj, torch.Tensor):
        adj = torch.as_tensor(adj)
    device = noisy_label.device if isinstance(noisy_label, torch.Tensor) else torch.device("cpu")
    adj = adj.to(device)
    noisy = noisy_label.long().to(device) if isinstance(noisy_label, torch.Tensor) else torch.as_tensor(noisy_label, dtype=torch.long, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    prop = torch.zeros((int(noisy.shape[0]), int(n_classes)), dtype=torch.float32, device=device)
    prop[train_idx_t, noisy[train_idx_t]] = 1.0
    for _ in range(int(steps)):
        prop = torch.sparse.mm(adj, prop) if getattr(adj, "is_sparse", False) else adj @ prop
    scores = (1.0 - prop[train_idx_t, noisy[train_idx_t]]).detach().cpu().numpy()
    return np.clip(scores, 0.0, 1.0), np.ones(len(train_idx), dtype=bool)


def conditional_auc(labels, scores, subset_mask, min_positive: int = 1) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    subset_mask = np.asarray(subset_mask, dtype=bool)
    if int(labels[subset_mask].sum()) < int(min_positive):
        return float("nan")
    return _safe_auc(labels[subset_mask], scores[subset_mask])


def detection_metrics(labels, scores) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    noise_rate = float(np.mean(labels)) if labels.size else float("nan")
    auroc = _safe_auc(labels, scores)
    auprc = _safe_auprc(labels, scores)
    k = int(labels.sum())
    if k <= 0:
        precision = recall = f1 = float("nan")
    else:
        order = np.argsort(-scores, kind="mergesort")
        selected = order[:k]
        tp = int(labels[selected].sum())
        precision = float(tp / k)
        recall = float(tp / k)
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "noise_rate": noise_rate,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": auprc - noise_rate if np.isfinite(auprc) and np.isfinite(noise_rate) else float("nan"),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
    }


def top_fraction_metrics(labels, scores, fractions=(0.2, 0.3, 0.4)):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    rows = {}
    total_noise = max(1, int(labels.sum()))
    for frac in fractions:
        k = max(1, int(round(float(frac) * labels.size)))
        selected = order[:k]
        tp = int(labels[selected].sum())
        precision = float(tp / k)
        recall = float(tp / total_noise)
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        key = f"top{int(round(frac * 100))}"
        rows[f"precision_{key}"] = precision
        rows[f"recall_{key}"] = recall
        rows[f"f1_{key}"] = f1
    return rows


def class_coverage_at_k(labels, scores, noisy_train, n_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    noisy_train = np.asarray(noisy_train, dtype=np.int64)
    k = int(labels.sum())
    if k <= 0:
        return {"coverage_classes_at_k": 0, "coverage_rate_at_k": float("nan")}
    selected = np.argsort(-scores, kind="mergesort")[:k]
    noisy_selected = selected[labels[selected] == 1]
    covered = set(noisy_train[noisy_selected].tolist())
    return {
        "coverage_classes_at_k": int(len(covered)),
        "coverage_rate_at_k": float(len(covered) / max(1, int(n_classes))),
    }


def retention_metrics(labels, scores, keep_rates=(0.5, 0.6, 0.7)):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    rows = {}
    for keep_rate in keep_rates:
        threshold = float(np.quantile(scores, float(keep_rate)))
        keep = scores <= threshold
        dropped = ~keep
        key = f"keep{int(round(float(keep_rate) * 100))}"
        rows[f"retention_{key}"] = float(np.mean(keep))
        rows[f"dropped_clean_{key}"] = float(np.mean((labels == 0) & dropped))
        rows[f"removed_noise_precision_{key}"] = float(labels[dropped].mean()) if bool(dropped.any()) else float("nan")
    return rows


def _prepare_xh_features(features, embeddings, device):
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    features = features.float().to(device)
    embeddings = F.normalize(embeddings.float().to(device), p=2, dim=1)
    return torch.cat([features, embeddings], dim=1)


def _warmup_gcn_probs(data, noisy_label, embeddings, args):
    features = _prepare_xh_features(data.feats, embeddings, data.feats.device)
    conf = load_conf(None, args.method_config, data.name)
    model = GCN(
        in_channels=int(features.shape[1]),
        hidden_channels=conf.model["n_hidden"],
        out_channels=data.n_classes,
        n_layers=conf.model["n_layer"],
        dropout=conf.model["dropout"],
        norm_info=conf.model["norm_info"],
        act=conf.model["act"],
        input_layer=conf.model["input_layer"],
        output_layer=conf.model["output_layer"],
    ).to(data.feats.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf.training["lr"],
        weight_decay=conf.training["weight_decay"],
    )
    train_mask = data.train_masks
    snapshots = []
    for epoch in range(1, int(args.warmup_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, data.adj)
        loss = F.cross_entropy(logits[train_mask], noisy_label[train_mask].long())
        loss.backward()
        optimizer.step()
        if int(args.window_start) <= epoch <= int(args.window_end):
            model.eval()
            with torch.no_grad():
                snapshots.append(F.softmax(model(features, data.adj), dim=1).detach().cpu())
    if not snapshots:
        raise ValueError("warmup window produced no snapshots")
    return torch.stack(snapshots, dim=0).mean(dim=0)


def _gcn_scores(early_probs, train_idx, noisy_train):
    node_idx = torch.as_tensor(train_idx, dtype=torch.long)
    label_idx = torch.as_tensor(noisy_train, dtype=torch.long)
    probs = early_probs[node_idx]
    p_noisy = probs[torch.arange(len(train_idx)), label_idx].clamp_min(1e-12)
    s_conf = (1.0 - p_noisy).cpu().numpy()
    s_smallloss = (-torch.log(p_noisy)).cpu().numpy()
    entropy = (-(probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))).sum(dim=1) / math.log(max(2, probs.shape[1]))).cpu().numpy()
    return s_conf, entropy, s_smallloss


def _knn_suspicion_scores(embeddings, noisy_label, train_idx, k):
    noisy_np = _as_noisy_numpy(noisy_label)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    k = min(int(k), max(1, int(train_idx.size) - 1))
    if k <= 0:
        return np.zeros(train_idx.shape[0], dtype=np.float64)
    h = F.normalize(embeddings.float(), p=2, dim=1)[torch.as_tensor(train_idx, dtype=torch.long)]
    sim = h @ h.t()
    sim.fill_diagonal_(-float("inf"))
    nn_idx = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()
    train_labels = noisy_np[train_idx]
    scores = np.zeros(train_idx.shape[0], dtype=np.float64)
    for pos in range(train_idx.size):
        neighbor_labels = train_labels[nn_idx[pos]]
        agree = float(np.mean(neighbor_labels == train_labels[pos]))
        scores[pos] = 1.0 - agree
    return scores


def _ball_suspicion_scores(noisy_label, assignment, train_idx, n_classes, min_train_per_ball):
    noisy_np = _as_noisy_numpy(noisy_label)
    assignment = _as_numpy_assignment(assignment)
    scores = np.full(train_idx.shape[0], DEFAULT_UNDEFINED_SCORE, dtype=np.float64)
    eligible = np.zeros(train_idx.shape[0], dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}
    for ball_id in sorted(np.unique(assignment[train_idx]).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float64)
        dist = counts / float(max(1.0, counts.sum()))
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            scores[pos] = 1.0 - float(dist[int(noisy_np[node])])
            eligible[pos] = True
    return scores, eligible


def _shuffled_assignment(assignment, seed: int):
    assignment = _as_numpy_assignment(assignment)
    rng = np.random.default_rng(int(seed) + 7919)
    return rng.permutation(assignment)


def build_detector_scores(data, noisy_label, embeddings, assignment, train_idx, args, seed: int):
    noisy_train = _as_noisy_numpy(noisy_label)[train_idx]
    min_train_per_ball = int(args.min_train_per_ball)
    s_ball, e_ball = _ball_suspicion_scores(noisy_label, assignment, train_idx, data.n_classes, min_train_per_ball)
    s_ball_loo, e_ball_loo, s_jsd_loo = ball_leave_one_out_scores(
        noisy_label, assignment, train_idx, data.n_classes, min_train_per_ball
    )
    s_entropy, e_entropy = ball_entropy_scores(noisy_label, assignment, train_idx, data.n_classes, min_train_per_ball)
    s_center, e_center = center_distance_scores(embeddings, assignment, train_idx)
    s_ballnbr, e_ballnbr = ball_neighbor_consistency_scores(
        embeddings, noisy_label, assignment, train_idx, data.n_classes, args.ball_neighbor_k, min_train_per_ball
    )
    s_graphdens, e_graphdens = graph_density_scores(
        embeddings=embeddings,
        noisy_label=noisy_label,
        assignment=assignment,
        train_idx=train_idx,
        adj=data.adj,
        n_classes=data.n_classes,
        ball_neighbor_k=args.ball_neighbor_k,
        min_train_per_ball=min_train_per_ball,
    )
    shuffled = _shuffled_assignment(assignment, seed)
    s_shuffled, e_shuffled, _ = ball_leave_one_out_scores(
        noisy_label, shuffled, train_idx, data.n_classes, min_train_per_ball
    )
    s_graphdens_shuffled, e_graphdens_shuffled = graph_density_scores(
        embeddings=embeddings,
        noisy_label=noisy_label,
        assignment=shuffled,
        train_idx=train_idx,
        adj=data.adj,
        n_classes=data.n_classes,
        ball_neighbor_k=args.ball_neighbor_k,
        min_train_per_ball=min_train_per_ball,
    )
    s_knn = _knn_suspicion_scores(embeddings, noisy_label, train_idx, args.knn_k)
    s_graphnbr, e_graphnbr = graph_neighbor_scores(noisy_label, train_idx, adj=data.adj)
    s_lp, e_lp = lp_scores(data.adj, noisy_label, train_idx, data.n_classes, args.lp_steps)
    early_probs = _warmup_gcn_probs(data, noisy_label, embeddings, args)
    s_gcn_conf, s_gcn_entropy, s_gcn_smallloss = _gcn_scores(early_probs, train_idx, noisy_train)
    s_combo_h = np.mean(np.vstack([_zscore(s_ball_loo), _zscore(s_center), _zscore(s_entropy)]), axis=0)
    s_combo_hg = np.mean(np.vstack([_zscore(s_ball_loo), _zscore(s_center), _zscore(s_entropy), _zscore(s_graphdens)]), axis=0)
    return {
        "s_ball": (s_ball, e_ball, "reference"),
        "s_ball_loo": (s_ball_loo, e_ball_loo, "primary"),
        "s_jsd_loo": (s_jsd_loo, e_ball_loo, "exploratory"),
        "s_center": (s_center, e_center, "exploratory"),
        "s_entropy": (s_entropy, e_entropy, "exploratory"),
        "s_ballnbr": (s_ballnbr, e_ballnbr, "exploratory"),
        "s_graphdens": (s_graphdens, e_graphdens, "primary"),
        "s_combo_H": (s_combo_h, np.ones_like(e_ball_loo, dtype=bool), "exploratory"),
        "s_combo_HG": (s_combo_hg, np.ones_like(e_ball_loo, dtype=bool), "exploratory"),
        "random": (np.random.default_rng(int(seed)).random(len(train_idx)), np.ones(len(train_idx), dtype=bool), "baseline"),
        "s_knn": (s_knn, np.ones(len(train_idx), dtype=bool), "baseline"),
        "s_gcn_conf": (s_gcn_conf, np.ones(len(train_idx), dtype=bool), "baseline"),
        "s_gcn_entropy": (s_gcn_entropy, np.ones(len(train_idx), dtype=bool), "baseline"),
        "s_gcn_smallloss": (s_gcn_smallloss, np.ones(len(train_idx), dtype=bool), "baseline"),
        "s_graphnbr": (s_graphnbr, e_graphnbr, "baseline"),
        "s_lp": (s_lp, e_lp, "baseline"),
        "s_shuffled": (s_shuffled, e_shuffled, "shuffled"),
        "s_graphdens_shuffled": (s_graphdens_shuffled, e_graphdens_shuffled, "shuffled"),
    }


def _quantile_clean_mask(scores, clean_fraction=0.5):
    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(scores, float(clean_fraction)))
    return scores <= threshold


def _detector_eval_rows(dataset, seed, noisy_flag, noisy_train, n_classes, score_map):
    rows = []
    for name, (scores, eligible, role) in score_map.items():
        scores = np.asarray(scores, dtype=np.float64)
        eligible = np.asarray(eligible, dtype=bool)
        metrics = detection_metrics(noisy_flag.astype(np.int64), scores)
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "detector": name,
            "role": role,
            "eligible_rate": float(np.mean(eligible)),
        }
        row.update(metrics)
        row.update(top_fraction_metrics(noisy_flag.astype(np.int64), scores))
        row.update(class_coverage_at_k(noisy_flag.astype(np.int64), scores, noisy_train, n_classes))
        row.update(retention_metrics(noisy_flag.astype(np.int64), scores))
        if bool(eligible.any()) and np.unique(noisy_flag[eligible]).size >= 2:
            eligible_metrics = detection_metrics(noisy_flag[eligible].astype(np.int64), scores[eligible])
            row["eligible_auroc"] = eligible_metrics["auroc"]
            row["eligible_auprc_lift"] = eligible_metrics["auprc_lift"]
        else:
            row["eligible_auroc"] = float("nan")
            row["eligible_auprc_lift"] = float("nan")
        rows.append(row)
    return rows


def _orthogonality_rows(dataset, seed, noisy_flag, score_map, pass_auc: float, min_intersection_noisy: int):
    rows = []
    s_knn = score_map["s_knn"][0]
    s_gcn = score_map["s_gcn_conf"][0]
    knn_clean50 = _quantile_clean_mask(s_knn, 0.5)
    gcn_clean50 = _quantile_clean_mask(s_gcn, 0.5)
    intersection = knn_clean50 & gcn_clean50
    for primary, shuffled in [("s_ball_loo", "s_shuffled"), ("s_graphdens", "s_graphdens_shuffled")]:
        scores = score_map[primary][0]
        shuffled_scores = score_map[shuffled][0]
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "primary": primary,
            "shuffled": shuffled,
            "auc": _safe_auc(noisy_flag, scores),
            "shuffled_auc": _safe_auc(noisy_flag, shuffled_scores),
            "auc_minus_shuffled": _safe_auc(noisy_flag, scores) - _safe_auc(noisy_flag, shuffled_scores),
            "auc_in_knn_clean50": conditional_auc(noisy_flag, scores, knn_clean50),
            "auc_in_gcn_clean50": conditional_auc(noisy_flag, scores, gcn_clean50),
            "auc_in_knn_AND_gcn_clean50": conditional_auc(
                noisy_flag, scores, intersection, min_positive=int(min_intersection_noisy)
            ),
            "intersection_noisy": int(noisy_flag[intersection].sum()),
            "corr_with_knn": _safe_corr(scores, s_knn),
            "corr_with_gcn": _safe_corr(scores, s_gcn),
            "knn_auc_in_primary_clean50": conditional_auc(noisy_flag, s_knn, _quantile_clean_mask(scores, 0.5)),
        }
        row["pass_knn_clean50"] = int(np.isfinite(row["auc_in_knn_clean50"]) and row["auc_in_knn_clean50"] >= float(pass_auc))
        row["pass_gcn_clean50"] = int(np.isfinite(row["auc_in_gcn_clean50"]) and row["auc_in_gcn_clean50"] >= float(pass_auc))
        rows.append(row)
    return rows


def _graph_gain_rows(dataset, seed, noisy_flag, score_map):
    h = score_map["s_combo_H"][0]
    hg = score_map["s_combo_HG"][0]
    graphdens = score_map["s_graphdens"][0]
    graphnbr = score_map["s_graphnbr"][0]
    return [{
        "dataset": dataset,
        "seed": int(seed),
        "combo_H_auroc": _safe_auc(noisy_flag, h),
        "combo_HG_auroc": _safe_auc(noisy_flag, hg),
        "combo_HG_minus_H": _safe_auc(noisy_flag, hg) - _safe_auc(noisy_flag, h),
        "graphdens_auroc": _safe_auc(noisy_flag, graphdens),
        "graphnbr_auroc": _safe_auc(noisy_flag, graphnbr),
        "graphdens_minus_graphnbr": _safe_auc(noisy_flag, graphdens) - _safe_auc(noisy_flag, graphnbr),
    }]


def _run_one(args, dataset_name: str, seed: int):
    dataset_key = _dataset_key(dataset_name)
    setup_seed(seed)
    data = _load_dataset(
        dataset_name=dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    gb_params = load_softdist_xh_gb_config(dataset_key, method=args.softdist_method)
    h_path = Path(gb_params["embedding_path"])
    if not h_path.exists():
        raise FileNotFoundError(f"Missing frozen embedding for {dataset_key}: {h_path}")
    embeddings = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes))
    gb = build_ordinary_granular_balls(
        embeddings=embeddings,
        max_balls=int(gb_params["max_balls"]),
        min_ball_size=int(gb_params["min_ball_size"]),
        split_improvement=float(gb_params["split_improvement"]),
        random_state=seed,
    )
    train_idx = _mask_to_index_array(data.train_masks)
    noisy_train = _as_noisy_numpy(noisy_label)[train_idx]
    clean_train = data.labels.detach().cpu().long().numpy()[train_idx]
    noisy_flag = noisy_train != clean_train
    args.min_train_per_ball = int(gb_params.get("min_train_per_ball", args.min_train_per_ball))
    purity = partition_purity_diagnostics(
        assignment=gb.assignment,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=train_idx,
        n_classes=data.n_classes,
        min_train_per_ball=args.min_train_per_ball,
    )
    score_map = build_detector_scores(data, noisy_label, embeddings, gb.assignment, train_idx, args, seed)
    run_meta = {
        "dataset": dataset_key,
        "seed": int(seed),
        "n_train": int(train_idx.size),
        "train_noise_rate": float(np.mean(noisy_flag)),
        "embedding_path": str(h_path),
        "config_path": str(gb_params["config_path"]),
        "max_balls": int(gb_params["max_balls"]),
        "min_ball_size": int(gb_params["min_ball_size"]),
        "split_improvement": float(gb_params["split_improvement"]),
        "min_train_per_ball": int(args.min_train_per_ball),
        "num_balls": int(gb.summary["num_balls"]),
        "mean_ball_size": float(gb.summary["mean_ball_size"]),
        "true_all_purity_weighted": float(purity.get("true_all_purity_weighted", 0.0)),
        "noisy_train_purity_weighted": float(purity.get("noisy_train_purity_weighted", 0.0)),
        "mean_train_per_ball": float(purity.get("mean_train_per_ball", 0.0)),
    }
    detector_rows = []
    for row in _detector_eval_rows(dataset_key, seed, noisy_flag, noisy_train, data.n_classes, score_map):
        detector_rows.append({**run_meta, **row})
    orth_rows = []
    for row in _orthogonality_rows(dataset_key, seed, noisy_flag, score_map, args.pass_auc, args.min_intersection_noisy):
        orth_rows.append({**run_meta, **row})
    graph_rows = []
    for row in _graph_gain_rows(dataset_key, seed, noisy_flag, score_map):
        graph_rows.append({**run_meta, **row})
    return detector_rows, orth_rows, graph_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        base = {keys[i]: key[i] for i in range(len(keys))}
        metric_names = sorted({k for row in group for k in row.keys() if k not in keys and isinstance(row.get(k), (int, float))})
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            finite_values = [value for value in values if np.isfinite(value)]
            base[f"{metric}_mean"] = _mean(values)
            base[f"{metric}_std"] = _std(values)
            base[f"{metric}_count"] = len(finite_values)
        base["num_seeds"] = len(group)
        out.append(base)
    return out


def write_report(path: Path, detector_summary, orth_summary, graph_summary, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 39: Label Noise Detection Feasibility Diagnostic",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{','.join(args.datasets)}`; seeds: `{','.join(map(str, args.seeds))}`.",
        f"- Noise: `{args.noise_type}` {args.noise_rate}; split: percent {args.train_percent}/{args.val_percent}/{args.test_percent}.",
        f"- Primary detectors: `s_ball_loo`, `s_graphdens`; pass_auc={args.pass_auc}.",
        "- Clean labels are used only for evaluation labels (`noisy_train != clean_train`).",
        "",
        "## Detector Summary",
        "",
        "| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in detector_summary:
        lines.append(
            "| {dataset} | {detector} | {role} | {auroc:.4f} | {lift:.4f} | {pk:.4f} | {eligible:.4f} |".format(
                dataset=row.get("dataset"),
                detector=row.get("detector"),
                role=row.get("role"),
                auroc=float(row.get("auroc_mean", float("nan"))),
                lift=float(row.get("auprc_lift_mean", float("nan"))),
                pk=float(row.get("precision_at_k_mean", float("nan"))),
                eligible=float(row.get("eligible_rate_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Orthogonality Summary",
        "",
        "| Dataset | Primary | AUC | real-shuffled | AUC in kNN-clean50 | AUC in GCN-clean50 | AUC in kNN∩GCN-clean50 | corr kNN | corr GCN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in orth_summary:
        lines.append(
            "| {dataset} | {primary} | {auc:.4f} | {delta:.4f} | {knn:.4f} | {gcn:.4f} | {both:.4f} | {corr_knn:.4f} | {corr_gcn:.4f} |".format(
                dataset=row.get("dataset"),
                primary=row.get("primary"),
                auc=float(row.get("auc_mean", float("nan"))),
                delta=float(row.get("auc_minus_shuffled_mean", float("nan"))),
                knn=float(row.get("auc_in_knn_clean50_mean", float("nan"))),
                gcn=float(row.get("auc_in_gcn_clean50_mean", float("nan"))),
                both=float(row.get("auc_in_knn_AND_gcn_clean50_mean", float("nan"))),
                corr_knn=float(row.get("corr_with_knn_mean", float("nan"))),
                corr_gcn=float(row.get("corr_with_gcn_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Graph Increment Summary",
        "",
        "| Dataset | combo_HG-H | graphdens-graphnbr |",
        "|---|---:|---:|",
    ])
    for row in graph_summary:
        lines.append(
            "| {dataset} | {combo:.4f} | {graph:.4f} |".format(
                dataset=row.get("dataset"),
                combo=float(row.get("combo_HG_minus_H_mean", float("nan"))),
                graph=float(row.get("graphdens_minus_graphnbr_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Gate Notes",
        "",
        "- G1 is not auto-declared by this report; inspect paired direction counts and the raw `orthogonality_raw.csv`.",
        "- `best-of` and combo detectors are exploratory only and must not trigger go/no-go.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 39 label-noise detection feasibility diagnostic.")
    parser.add_argument("--datasets", default="cora,citeseer,coauthorcs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--softdist_method", default=SOFTDIST_METHOD)
    parser.add_argument("--method_config", default="gcn_xh_ce")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--window_start", type=int, default=5)
    parser.add_argument("--window_end", type=int, default=10)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--ball_neighbor_k", type=int, default=10)
    parser.add_argument("--lp_steps", type=int, default=2)
    parser.add_argument("--pass_auc", type=float, default=0.6)
    parser.add_argument("--min_intersection_noisy", type=int, default=8)
    parser.add_argument("--output_dir", default="runs/noise_detection_feasibility_20260624")
    args = parser.parse_args()
    args.datasets = _parse_str_list(args.datasets)
    args.seeds = _parse_int_list(args.seeds)

    detector_rows = []
    orth_rows = []
    graph_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"PHASE39 dataset={dataset} seed={seed}")
            d_rows, o_rows, g_rows = _run_one(args, dataset, seed)
            detector_rows.extend(d_rows)
            orth_rows.extend(o_rows)
            graph_rows.extend(g_rows)

    output_dir = Path(args.output_dir)
    detector_summary = summarize(detector_rows, ["dataset", "detector", "role"])
    orth_summary = summarize(orth_rows, ["dataset", "primary"])
    graph_summary = summarize(graph_rows, ["dataset"])
    write_csv(output_dir / "detector_raw.csv", detector_rows)
    write_csv(output_dir / "detector_summary.csv", detector_summary)
    write_csv(output_dir / "orthogonality_raw.csv", orth_rows)
    write_csv(output_dir / "orthogonality_summary.csv", orth_summary)
    write_csv(output_dir / "graph_increment_raw.csv", graph_rows)
    write_csv(output_dir / "graph_increment_summary.csv", graph_summary)
    write_report(output_dir / "noise_detection_feasibility_report.md", detector_summary, orth_summary, graph_summary, args)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
