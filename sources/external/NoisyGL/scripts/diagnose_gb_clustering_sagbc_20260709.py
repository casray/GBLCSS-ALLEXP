from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_label_noise_detection import load_softdist_xh_gb_config  # noqa: E402
from scripts.run_standard_downstream import _load_dataset  # noqa: E402
from utils.frozen_grace_targets import load_frozen_embeddings  # noqa: E402
from utils.ordinary_gb import build_ordinary_granular_balls, build_random_size_matched_assignment  # noqa: E402
from utils.tools import setup_seed  # noqa: E402


CANONICAL_EMBEDDING_DIR = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "coauthorcs": "CoauthorCS",
    "amazonpho": "AmazonPhoto",
}


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _dataset_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    aliases = {
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
        "coauthor-cs": "coauthorcs",
        "coauthorcs": "coauthorcs",
    }
    return aliases.get(key, key)


def _mean(values) -> float:
    vals = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _std(values) -> float:
    vals = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return 0.0
    mean = _mean(vals)
    return float(math.sqrt(sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)))


def _as_numpy_1d(values, dtype=np.int64) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype).reshape(-1)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def hungarian_acc(y_true, y_pred) -> float:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_pred = _as_numpy_1d(y_pred, dtype=np.int64)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        return float("nan")

    true_values, true_inv = np.unique(y_true, return_inverse=True)
    pred_values, pred_inv = np.unique(y_pred, return_inverse=True)
    matrix = np.zeros((pred_values.size, true_values.size), dtype=np.int64)
    np.add.at(matrix, (pred_inv, true_inv), 1)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return float(matrix[row_ind, col_ind].sum() / y_true.size)


def clustering_metrics(y_true, y_pred) -> dict:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_pred = _as_numpy_1d(y_pred, dtype=np.int64)
    return {
        "acc": hungarian_acc(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "n_clusters": int(np.unique(y_pred).size),
    }


def _resolve_embedding_path(dataset_key: str) -> Path:
    dataset_key = _dataset_key(dataset_key)
    root = REPO_ROOT.parent.parent / "gb_graph_aware_minival_6_1" / "checkpoints" / "embeddings_v2"
    candidates = []
    canonical = CANONICAL_EMBEDDING_DIR.get(dataset_key)
    if canonical:
        candidates.append(root / canonical / "grace" / "H.pt")
    candidates.append(root / dataset_key / "grace" / "H.pt")
    try:
        gb_config = load_softdist_xh_gb_config(dataset_key)
        candidates.append(Path(gb_config["embedding_path"]))
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def ball_params_for_dataset(dataset_key: str, args) -> dict:
    dataset_key = _dataset_key(dataset_key)
    max_balls = int(args.max_balls)
    min_ball_size = int(args.min_ball_size)
    split_improvement = float(args.split_improvement)
    if dataset_key == "pubmed" and bool(args.pubmed_phase41_params):
        max_balls = int(args.pubmed_max_balls)
        split_improvement = float(args.pubmed_split_improvement)
    return {
        "max_balls": max_balls,
        "min_ball_size": min_ball_size,
        "split_improvement": split_improvement,
    }


def ball_centers(features: np.ndarray, assignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(features, dtype=np.float32)
    assignment = _as_numpy_1d(assignment, dtype=np.int64)
    ball_ids = np.array(sorted(np.unique(assignment).tolist()), dtype=np.int64)
    centers = []
    sizes = []
    for ball_id in ball_ids:
        members = np.flatnonzero(assignment == int(ball_id))
        centers.append(z[members].mean(axis=0))
        sizes.append(members.size)
    return l2_normalize(np.vstack(centers)), np.asarray(sizes, dtype=np.int64)


def _sparse_adj_to_edges(adj) -> np.ndarray:
    if adj is None:
        return np.empty((2, 0), dtype=np.int64)
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj = adj.coalesce()
            return adj.indices().detach().cpu().numpy().astype(np.int64)
        arr = adj.detach().cpu().numpy()
        row, col = np.nonzero(arr)
        return np.vstack([row, col]).astype(np.int64)
    try:
        coo = adj.tocoo()
        return np.vstack([coo.row, coo.col]).astype(np.int64)
    except Exception:
        arr = np.asarray(adj)
        row, col = np.nonzero(arr)
        return np.vstack([row, col]).astype(np.int64)


def compute_soft_affiliation(
    features: np.ndarray,
    centers: np.ndarray,
    m: int = 5,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features = l2_normalize(features)
    centers = l2_normalize(centers)
    n_nodes = int(features.shape[0])
    n_balls = int(centers.shape[0])
    if n_nodes == 0 or n_balls == 0:
        raise ValueError("features and centers must be non-empty")
    m = max(1, min(int(m), n_balls))

    cosine = np.clip(features @ centers.T, -1.0, 1.0)
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))
    nearest = np.argpartition(distances, kth=m - 1, axis=1)[:, :m]
    nearest_dist = np.take_along_axis(distances, nearest, axis=1)
    order = np.argsort(nearest_dist, axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_dist = np.take_along_axis(nearest_dist, order, axis=1)

    if sigma is None:
        sigma = float(np.median(nearest_dist[:, -1]))
    sigma = max(float(sigma), 1e-6)
    weights = np.exp(-np.square(nearest_dist) / (2.0 * sigma * sigma)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    affiliation = np.zeros((n_nodes, n_balls), dtype=np.float32)
    row_index = np.arange(n_nodes)[:, None]
    affiliation[row_index, nearest] = weights
    return affiliation, nearest.astype(np.int64), weights.astype(np.float32), sigma


def _normalize_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    max_value = float(matrix.max()) if matrix.size else 0.0
    if max_value > 0.0:
        matrix = matrix / max_value
    return matrix


def soft_graph_edge_density(
    nearest: np.ndarray,
    weights: np.ndarray,
    adj,
    n_balls: int,
) -> np.ndarray:
    edges = _sparse_adj_to_edges(adj)
    density = np.zeros((n_balls, n_balls), dtype=np.float32)
    if edges.size == 0:
        return density
    row, col = edges
    valid = row != col
    row = row[valid]
    col = col[valid]
    for u, v in zip(row.tolist(), col.tolist()):
        for i, wi in zip(nearest[u], weights[u]):
            if wi <= 0.0:
                continue
            for j, wj in zip(nearest[v], weights[v]):
                if wj <= 0.0 or int(i) == int(j):
                    continue
                density[int(i), int(j)] += float(wi * wj)
    density = density + density.T
    return _normalize_similarity(density)


def hard_ball_edge_density(assignment: np.ndarray, adj, n_balls: int) -> np.ndarray:
    edges = _sparse_adj_to_edges(adj)
    weights = np.zeros((n_balls, n_balls), dtype=np.float32)
    if edges.size == 0:
        return weights
    assignment = _as_numpy_1d(assignment, dtype=np.int64)
    row_balls = assignment[edges[0]]
    col_balls = assignment[edges[1]]
    valid = row_balls != col_balls
    np.add.at(weights, (row_balls[valid], col_balls[valid]), 1.0)
    weights = weights + weights.T
    return _normalize_similarity(weights)


def build_similarity_graph(similarity: np.ndarray, graph_k: int = 10) -> nx.Graph:
    similarity = np.asarray(similarity, dtype=np.float32)
    n_balls = int(similarity.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(range(n_balls))
    if n_balls <= 1:
        return graph
    k = max(1, min(int(graph_k), n_balls - 1))
    for i in range(n_balls):
        row = similarity[i].copy()
        row[i] = -np.inf
        neighbors = np.argpartition(-row, kth=k - 1)[:k]
        for j in neighbors:
            weight = float(similarity[i, int(j)])
            if not np.isfinite(weight) or weight <= 0.0 or int(i) == int(j):
                continue
            if graph.has_edge(int(i), int(j)):
                graph[int(i)][int(j)]["weight"] = max(graph[int(i)][int(j)]["weight"], weight)
            else:
                graph.add_edge(int(i), int(j), weight=weight)
    return graph


def modularity_ball_clusters(graph: nx.Graph) -> np.ndarray:
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return np.empty(0, dtype=np.int64)
    if n_nodes == 1 or graph.number_of_edges() == 0:
        return np.arange(n_nodes, dtype=np.int64)
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    labels = np.empty(n_nodes, dtype=np.int64)
    for cluster_id, community in enumerate(communities):
        for node in community:
            labels[int(node)] = int(cluster_id)
    return labels


def node_labels_from_affiliation(affiliation: np.ndarray, ball_cluster: np.ndarray) -> np.ndarray:
    ball_cluster = _as_numpy_1d(ball_cluster, dtype=np.int64)
    n_clusters = int(np.unique(ball_cluster).size)
    remap = {int(label): idx for idx, label in enumerate(sorted(np.unique(ball_cluster).tolist()))}
    mapped = np.asarray([remap[int(label)] for label in ball_cluster], dtype=np.int64)
    onehot = np.eye(n_clusters, dtype=np.float32)[mapped]
    return np.argmax(affiliation @ onehot, axis=1).astype(np.int64)


def cluster_warnings(k_auto: int, num_balls: int) -> str:
    warnings = []
    if int(k_auto) <= 1:
        warnings.append("degenerate_k_auto")
    if int(num_balls) > 0 and int(k_auto) > 0.5 * int(num_balls):
        warnings.append("fragmented_k_auto")
    return ";".join(warnings) if warnings else "ok"


def sagbc_soft_affiliation_clustering(
    features: np.ndarray,
    assignment: np.ndarray,
    adj=None,
    affiliation_m: int = 5,
    graph_k: int = 10,
    center_weight: float = 0.5,
    overlap_weight: float = 0.3,
    edge_weight: float = 0.2,
) -> dict:
    features = l2_normalize(features)
    assignment = _as_numpy_1d(assignment, dtype=np.int64)
    centers, sizes = ball_centers(features, assignment)
    affiliation, nearest, weights, sigma = compute_soft_affiliation(features, centers, m=affiliation_m)

    center_cosine = _normalize_similarity(np.maximum(centers @ centers.T, 0.0))
    overlap = _normalize_similarity(affiliation.T @ affiliation)
    edge_density = soft_graph_edge_density(nearest, weights, adj, int(centers.shape[0]))
    similarity = (
        float(center_weight) * center_cosine
        + float(overlap_weight) * overlap
        + float(edge_weight) * edge_density
    )
    similarity = _normalize_similarity(similarity)
    graph = build_similarity_graph(similarity, graph_k=graph_k)
    ball_cluster = modularity_ball_clusters(graph)
    node_cluster = node_labels_from_affiliation(affiliation, ball_cluster)
    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    return {
        "node_cluster": node_cluster,
        "ball_cluster": ball_cluster.astype(np.int64),
        "assignment": assignment,
        "affiliation": affiliation,
        "ball_sizes": sizes,
        "num_balls": int(centers.shape[0]),
        "k_auto": int(np.unique(ball_cluster).size),
        "cc_k": int(len(components)),
        "graph_edges": int(graph.number_of_edges()),
        "sigma": float(sigma),
        "warnings": cluster_warnings(int(np.unique(ball_cluster).size), int(centers.shape[0])),
    }


def old_gb_modularity_clustering(features: np.ndarray, assignment: np.ndarray, adj=None, graph_k: int = 10) -> dict:
    features = l2_normalize(features)
    assignment = _as_numpy_1d(assignment, dtype=np.int64)
    centers, sizes = ball_centers(features, assignment)
    center_cosine = _normalize_similarity(np.maximum(centers @ centers.T, 0.0))
    graph = build_similarity_graph(center_cosine, graph_k=graph_k)
    ball_cluster = modularity_ball_clusters(graph)
    node_cluster = ball_cluster[assignment]
    return {
        "node_cluster": node_cluster.astype(np.int64),
        "ball_cluster": ball_cluster.astype(np.int64),
        "assignment": assignment,
        "ball_sizes": sizes,
        "num_balls": int(centers.shape[0]),
        "k_auto": int(np.unique(ball_cluster).size),
        "cc_k": int(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0,
        "graph_edges": int(graph.number_of_edges()),
        "warnings": cluster_warnings(int(np.unique(ball_cluster).size), int(centers.shape[0])),
    }


def _kmeans_predict(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n_clusters = max(1, min(int(n_clusters), int(features.shape[0])))
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=int(seed)).fit_predict(features)


def _spectral_predict(features: np.ndarray, n_clusters: int, seed: int, max_nodes: int, n_neighbors: int) -> tuple[np.ndarray | None, str]:
    if features.shape[0] > int(max_nodes):
        return None, "skipped_large"
    try:
        labels = SpectralClustering(
            n_clusters=max(2, int(n_clusters)),
            affinity="nearest_neighbors",
            n_neighbors=max(2, min(int(n_neighbors), int(features.shape[0]) - 1)),
            assign_labels="kmeans",
            random_state=int(seed),
        ).fit_predict(features)
        return labels.astype(np.int64), "ok"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def wrong_k_values(true_k: int) -> list[int]:
    true_k = int(true_k)
    candidates = {
        true_k - 2,
        true_k + 2,
        int(round(true_k * 0.5)),
        int(round(true_k * 1.5)),
    }
    return sorted(k for k in candidates if k >= 2 and k != true_k)


def _metric_row(common: dict, method: str, labels: np.ndarray, pred: np.ndarray, status: str = "ok", uses_true_k: int = 0) -> dict:
    if status != "ok":
        return {
            **common,
            "method": method,
            "uses_true_k": int(uses_true_k),
            "status": status,
            "acc": float("nan"),
            "nmi": float("nan"),
            "ari": float("nan"),
            "n_clusters": float("nan"),
        }
    return {
        **common,
        "method": method,
        "uses_true_k": int(uses_true_k),
        "status": "ok",
        **clustering_metrics(labels, pred),
    }


def evaluate_one(dataset: str, seed: int, args) -> tuple[list[dict], list[dict], dict]:
    dataset_key = _dataset_key(dataset)
    setup_seed(int(seed))
    data = _load_dataset(dataset_key, args.device, "percent", 0.2, 0.1, 0.7)
    labels = _as_numpy_1d(data.labels, dtype=np.int64)
    true_k = int(getattr(data, "n_classes", np.unique(labels).size))
    embedding_path = _resolve_embedding_path(dataset_key)
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(labels.shape[0]))
    features = l2_normalize(embeddings.detach().cpu().float().numpy())

    params = ball_params_for_dataset(dataset_key, args)
    gb = build_ordinary_granular_balls(
        features,
        max_balls=params["max_balls"],
        min_ball_size=params["min_ball_size"],
        split_improvement=params["split_improvement"],
        random_state=int(seed),
    )
    assignment = gb.assignment.astype(np.int64)
    num_balls = int(gb.summary.get("num_balls", np.unique(assignment).size))

    sagbc = sagbc_soft_affiliation_clustering(
        features,
        assignment,
        adj=data.adj,
        affiliation_m=args.affiliation_m,
        graph_k=args.ball_graph_k,
        center_weight=args.center_weight,
        overlap_weight=args.overlap_weight,
        edge_weight=args.edge_weight,
    )
    old = old_gb_modularity_clustering(features, assignment, adj=data.adj, graph_k=args.ball_graph_k)
    shuffled_assignment = build_random_size_matched_assignment(assignment, random_state=int(seed) + 100003)
    shuffled = sagbc_soft_affiliation_clustering(
        features,
        shuffled_assignment,
        adj=data.adj,
        affiliation_m=args.affiliation_m,
        graph_k=args.ball_graph_k,
        center_weight=args.center_weight,
        overlap_weight=args.overlap_weight,
        edge_weight=args.edge_weight,
    )
    k_auto = int(sagbc["k_auto"])

    common = {
        "dataset": dataset_key,
        "seed": int(seed),
        "true_k": true_k,
        "embedding_path": str(embedding_path),
        "max_balls": int(params["max_balls"]),
        "min_ball_size": int(params["min_ball_size"]),
        "split_improvement": float(params["split_improvement"]),
        "num_balls": num_balls,
        "k_auto": k_auto,
        "cc_k": int(sagbc["cc_k"]),
        "ball_graph_edges": int(sagbc["graph_edges"]),
        "affiliation_m": int(args.affiliation_m),
        "affiliation_sigma": float(sagbc["sigma"]),
        "warnings": str(sagbc["warnings"]),
    }

    metric_rows = [
        _metric_row(common, "sagbc_soft_affiliation", labels, sagbc["node_cluster"], uses_true_k=0),
        {
            **_metric_row(common, "old_gb_modularity", labels, old["node_cluster"], uses_true_k=0),
            "k_auto": int(old["k_auto"]),
            "cc_k": int(old["cc_k"]),
            "ball_graph_edges": int(old["graph_edges"]),
            "warnings": str(old["warnings"]),
        },
        {
            **_metric_row(common, "sagbc_shuffled_ball", labels, shuffled["node_cluster"], uses_true_k=0),
            "k_auto": int(shuffled["k_auto"]),
            "cc_k": int(shuffled["cc_k"]),
            "ball_graph_edges": int(shuffled["graph_edges"]),
            "affiliation_sigma": float(shuffled["sigma"]),
            "warnings": str(shuffled["warnings"]),
        },
    ]

    pred_true_k = _kmeans_predict(features, true_k, seed)
    metric_rows.append(_metric_row(common, "kmeans_H_trueK", labels, pred_true_k, uses_true_k=1))

    pred_auto_k = _kmeans_predict(features, k_auto, seed)
    metric_rows.append(_metric_row(common, "kmeans_H_autoK", labels, pred_auto_k, uses_true_k=0))

    spectral_labels, spectral_status = _spectral_predict(
        features,
        true_k,
        seed,
        max_nodes=args.spectral_max_nodes,
        n_neighbors=args.spectral_neighbors,
    )
    metric_rows.append(
        _metric_row(
            common,
            "spectral_knn_H_trueK",
            labels,
            spectral_labels if spectral_labels is not None else np.zeros_like(labels),
            status=spectral_status,
            uses_true_k=1,
        )
    )

    sagbc_metric = clustering_metrics(labels, sagbc["node_cluster"])
    wrong_rows = []
    for wrong_k in wrong_k_values(true_k):
        pred_wrong = _kmeans_predict(features, wrong_k, seed)
        wrong_metric = clustering_metrics(labels, pred_wrong)
        wrong_rows.append(
            {
                **common,
                "wrong_k": int(wrong_k),
                "wrong_k_ratio": float(wrong_k / max(1, true_k)),
                "sagbc_acc": float(sagbc_metric["acc"]),
                "sagbc_nmi": float(sagbc_metric["nmi"]),
                "kmeans_wrong_acc": float(wrong_metric["acc"]),
                "kmeans_wrong_nmi": float(wrong_metric["nmi"]),
                "acc_advantage": float(sagbc_metric["acc"] - wrong_metric["acc"]),
                "nmi_advantage": float(sagbc_metric["nmi"] - wrong_metric["nmi"]),
            }
        )

    meta = {
        **common,
        "sagbc_acc": float(sagbc_metric["acc"]),
        "sagbc_nmi": float(sagbc_metric["nmi"]),
    }
    return metric_rows, wrong_rows, meta


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_metric_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[(row["dataset"], row["method"])].append(row)

    out = []
    for (dataset, method), items in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": len(items),
                "acc_mean": _mean(row["acc"] for row in items),
                "acc_std": _std(row["acc"] for row in items),
                "nmi_mean": _mean(row["nmi"] for row in items),
                "nmi_std": _std(row["nmi"] for row in items),
                "ari_mean": _mean(row["ari"] for row in items),
                "ari_std": _std(row["ari"] for row in items),
                "k_auto_mean": _mean(row["k_auto"] for row in items),
                "num_balls_mean": _mean(row["num_balls"] for row in items),
                "n_clusters_mean": _mean(row["n_clusters"] for row in items),
                "warnings": ";".join(sorted({str(row.get("warnings", "")) for row in items if row.get("warnings")})),
            }
        )
    return out


def aggregate_wrong_k_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    out = []
    for dataset, items in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "rows": len(items),
                "sagbc_acc_mean": _mean(row["sagbc_acc"] for row in items),
                "kmeans_wrong_acc_mean": _mean(row["kmeans_wrong_acc"] for row in items),
                "acc_advantage_mean": _mean(row["acc_advantage"] for row in items),
                "acc_advantage_std": _std(row["acc_advantage"] for row in items),
                "nmi_advantage_mean": _mean(row["nmi_advantage"] for row in items),
            }
        )
    return out


def _method_mean(metric_rows: list[dict], dataset: str, method: str, metric: str) -> float:
    return _mean(row[metric] for row in metric_rows if row.get("dataset") == dataset and row.get("method") == method and row.get("status") == "ok")


def build_gate_rows(metric_rows: list[dict], wrong_summary: list[dict]) -> list[dict]:
    datasets = sorted({row["dataset"] for row in metric_rows if row.get("status") == "ok"})
    wrong_by_dataset = {row["dataset"]: row for row in wrong_summary}
    out = []
    for dataset in datasets:
        sagbc_acc = _method_mean(metric_rows, dataset, "sagbc_soft_affiliation", "acc")
        sagbc_nmi = _method_mean(metric_rows, dataset, "sagbc_soft_affiliation", "nmi")
        old_acc = _method_mean(metric_rows, dataset, "old_gb_modularity", "acc")
        shuffled_acc = _method_mean(metric_rows, dataset, "sagbc_shuffled_ball", "acc")
        kmeans_acc = _method_mean(metric_rows, dataset, "kmeans_H_trueK", "acc")
        kmeans_nmi = _method_mean(metric_rows, dataset, "kmeans_H_trueK", "nmi")
        spectral_acc = _method_mean(metric_rows, dataset, "spectral_knn_H_trueK", "acc")
        wrong_adv = float(wrong_by_dataset.get(dataset, {}).get("acc_advantage_mean", float("nan")))
        g1 = bool(np.isfinite(sagbc_nmi) and np.isfinite(kmeans_nmi) and sagbc_nmi >= 0.9 * kmeans_nmi and (sagbc_acc - kmeans_acc) >= -0.05)
        g2 = bool((sagbc_acc - old_acc) >= 0.03 and (sagbc_acc - shuffled_acc) >= 0.03)
        g3 = bool((np.isfinite(kmeans_acc) and sagbc_acc > kmeans_acc) or (np.isfinite(spectral_acc) and sagbc_acc > spectral_acc))
        g4 = bool(np.isfinite(wrong_adv) and wrong_adv >= 0.05)
        out.append(
            {
                "dataset": dataset,
                "sagbc_acc": sagbc_acc,
                "sagbc_nmi": sagbc_nmi,
                "old_gb_acc": old_acc,
                "shuffled_acc": shuffled_acc,
                "kmeans_trueK_acc": kmeans_acc,
                "kmeans_trueK_nmi": kmeans_nmi,
                "spectral_trueK_acc": spectral_acc,
                "acc_gap_vs_kmeans_trueK": float(sagbc_acc - kmeans_acc),
                "nmi_ratio_vs_kmeans_trueK": float(sagbc_nmi / max(kmeans_nmi, 1e-12)),
                "acc_gain_vs_old": float(sagbc_acc - old_acc),
                "acc_gain_vs_shuffled": float(sagbc_acc - shuffled_acc),
                "wrong_k_acc_advantage": wrong_adv,
                "g1_precision_close": int(g1),
                "g2_independent_gain": int(g2),
                "g3_strong_baseline_competition": int(g3),
                "g4_wrong_k_robust": int(g4),
                "gate": "PASS" if (g1 and g2) else "FAIL",
            }
        )
    return out


def write_report(output_dir: Path, args, aggregate_rows: list[dict], wrong_summary: list[dict], gate_rows: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in gate_rows})
    n_datasets = len(datasets)
    required_3of5 = math.ceil(3 * n_datasets / 5) if n_datasets else 0
    g1_count = sum(int(row["g1_precision_close"]) for row in gate_rows)
    g2_count = sum(int(row["g2_independent_gain"]) for row in gate_rows)
    g3_count = sum(int(row["g3_strong_baseline_competition"]) for row in gate_rows)
    g4_count = sum(int(row["g4_wrong_k_robust"]) for row in gate_rows)

    if n_datasets and g1_count >= required_3of5 and g2_count >= required_3of5:
        decision = "GO"
        decision_note = "G1+G2 passed on the required number of evaluated datasets."
    elif n_datasets and g2_count >= required_3of5:
        decision = "PARTIAL"
        decision_note = "Only G2 passed broadly; this supports an improvement over old hard balls but not a clustering mainline."
    else:
        decision = "NO-GO"
        decision_note = "G1 failed broadly, so GRACE-H granular-ball clustering should not be promoted without a new hypothesis."

    lines = [
        "# Offline Clustering Smoke v2: SAGBC-style Soft Affiliation",
        "",
        f"Decision: **{decision}**",
        "",
        "## Protocol",
        "",
        f"- Evaluated datasets in this report: `{','.join(datasets)}`.",
        f"- Last command datasets: `{args.datasets}`; seeds: `{args.seeds}`.",
        "- Frozen GRACE-H is used as clustering space. Labels are evaluation-only.",
        f"- Ordinary GB anchors: default max_balls={args.max_balls}, min_ball_size={args.min_ball_size}, split_improvement={args.split_improvement}; PubMed override={bool(args.pubmed_phase41_params)}.",
        f"- Soft affiliation: m={args.affiliation_m}; similarity S={args.center_weight}*center_cosine + {args.overlap_weight}*A^T A + {args.edge_weight}*edge_density.",
        "",
        "## Gate Summary",
        "",
        "| Dataset | SAGBC ACC | KMeans trueK ACC | ACC gap | SAGBC NMI | KMeans NMI | NMI ratio | Old GB ACC | Shuffled ACC | Wrong-K adv | G1 | G2 | G3 | G4 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        lines.append(
            "| {dataset} | {sagbc_acc:.4f} | {kmeans_trueK_acc:.4f} | {acc_gap_vs_kmeans_trueK:+.4f} | "
            "{sagbc_nmi:.4f} | {kmeans_trueK_nmi:.4f} | {nmi_ratio_vs_kmeans_trueK:.4f} | "
            "{old_gb_acc:.4f} | {shuffled_acc:.4f} | {wrong_k_acc_advantage:+.4f} | "
            "{g1_precision_close} | {g2_independent_gain} | {g3_strong_baseline_competition} | "
            "{g4_wrong_k_robust} | {gate} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Dataset | Method | ACC | NMI | ARI | K auto | Balls | Warnings |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    method_order = {
        "kmeans_H_trueK": 0,
        "spectral_knn_H_trueK": 1,
        "kmeans_H_autoK": 2,
        "old_gb_modularity": 3,
        "sagbc_soft_affiliation": 4,
        "sagbc_shuffled_ball": 5,
    }
    for row in sorted(aggregate_rows, key=lambda r: (r["dataset"], method_order.get(r["method"], 99), r["method"])):
        lines.append(
            "| {dataset} | {method} | {acc_mean:.4f}±{acc_std:.4f} | {nmi_mean:.4f}±{nmi_std:.4f} | "
            "{ari_mean:.4f}±{ari_std:.4f} | {k_auto_mean:.2f} | {num_balls_mean:.2f} | {warnings} |".format(**row)
        )

    lines.extend(["", "## Wrong-K Robustness", "", "| Dataset | SAGBC ACC | KMeans wrong-K ACC | ACC advantage | NMI advantage |", "|---|---:|---:|---:|---:|"])
    for row in wrong_summary:
        lines.append(
            "| {dataset} | {sagbc_acc_mean:.4f} | {kmeans_wrong_acc_mean:.4f} | "
            "{acc_advantage_mean:+.4f}±{acc_advantage_std:.4f} | {nmi_advantage_mean:+.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            f"- {decision_note}",
            f"- Gate counts: G1={g1_count}/{n_datasets}, G2={g2_count}/{n_datasets}, G3={g3_count}/{n_datasets}, G4={g4_count}/{n_datasets}.",
            "- G1: NMI >= 0.9 * kmeans_H_trueK NMI and ACC gap >= -5pp.",
            "- G2: SAGBC beats both old hard GB modularity and shuffled-ball control by >= +3pp ACC.",
            "- G3: SAGBC beats kmeans_H_trueK or spectral_knn_H_trueK.",
            "- G4: SAGBC beats mis-specified-K k-means by >= +5pp average ACC.",
        ]
    )
    (output_dir / "gb_clustering_sagbc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metric_rows: list[dict] = _read_csv(output_dir / "raw.csv") if args.append else []
    wrong_rows: list[dict] = _read_csv(output_dir / "wrong_k_raw.csv") if args.append else []
    error_rows: list[dict] = _read_csv(output_dir / "error_rows.csv") if args.append else []
    completed = {
        (str(row.get("dataset")), int(row.get("seed", -1)))
        for row in metric_rows
        if row.get("method") == "sagbc_soft_affiliation" and row.get("status") == "ok"
    }

    for dataset in _parse_str_list(args.datasets):
        for seed in _parse_int_list(args.seeds):
            if args.append and (_dataset_key(dataset), int(seed)) in completed:
                print(f"[skip] dataset={_dataset_key(dataset)} seed={seed} already present", flush=True)
                continue
            try:
                rows, wrong, meta = evaluate_one(dataset, seed, args)
                metric_rows.extend(rows)
                wrong_rows.extend(wrong)
                print(
                    f"[ok] dataset={_dataset_key(dataset)} seed={seed} balls={meta['num_balls']} "
                    f"k_auto={meta['k_auto']} acc={meta['sagbc_acc']:.4f} nmi={meta['sagbc_nmi']:.4f} "
                    f"warnings={meta['warnings']}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "dataset": _dataset_key(dataset),
                    "seed": int(seed),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                error_rows.append(row)
                print(f"[error] {row}", flush=True)
                if args.stop_on_error:
                    raise

            _write_csv(output_dir / "raw.csv", metric_rows)
            _write_csv(output_dir / "wrong_k_raw.csv", wrong_rows)
            _write_csv(output_dir / "error_rows.csv", error_rows)

    aggregate_rows = aggregate_metric_rows(metric_rows)
    wrong_summary = aggregate_wrong_k_rows(wrong_rows)
    gate_rows = build_gate_rows(metric_rows, wrong_summary)
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "wrong_k_robustness.csv", wrong_summary)
    _write_csv(output_dir / "gate_summary.csv", gate_rows)
    write_report(output_dir, args, aggregate_rows, wrong_summary, gate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGBC-style soft affiliation clustering smoke on frozen GRACE-H.")
    parser.add_argument("--datasets", default="cora,citeseer,pubmed")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="runs/gb_clustering_sagbc_smoke_20260709")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--max_balls", type=int, default=64)
    parser.add_argument("--min_ball_size", type=int, default=32)
    parser.add_argument("--split_improvement", type=float, default=0.05)
    parser.add_argument("--pubmed_phase41_params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pubmed_max_balls", type=int, default=256)
    parser.add_argument("--pubmed_split_improvement", type=float, default=0.005)
    parser.add_argument("--affiliation_m", type=int, default=5)
    parser.add_argument("--ball_graph_k", type=int, default=10)
    parser.add_argument("--center_weight", type=float, default=0.5)
    parser.add_argument("--overlap_weight", type=float, default=0.3)
    parser.add_argument("--edge_weight", type=float, default=0.2)
    parser.add_argument("--spectral_max_nodes", type=int, default=25000)
    parser.add_argument("--spectral_neighbors", type=int, default=10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
