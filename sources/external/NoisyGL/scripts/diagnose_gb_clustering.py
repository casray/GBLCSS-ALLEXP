from __future__ import annotations

import argparse
import csv
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
from utils.ordinary_gb import build_ordinary_granular_balls  # noqa: E402
from utils.tools import setup_seed  # noqa: E402


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _dataset_key(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    aliases = {
        "citeseer": "citeseer",
        "cora": "cora",
        "pubmed": "pubmed",
        "coauthor-cs": "coauthorcs",
        "coauthorcs": "coauthorcs",
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
        "amazoncomputers": "amazoncom",
        "amazon-computers": "amazoncom",
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
    matched = int(matrix[row_ind, col_ind].sum())
    return float(matched / y_true.size)


def clustering_metrics(y_true, y_pred) -> dict:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_pred = _as_numpy_1d(y_pred, dtype=np.int64)
    return {
        "acc": hungarian_acc(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "n_clusters": int(np.unique(y_pred).size),
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


def _sparse_adj_to_numpy(adj) -> np.ndarray | None:
    if adj is None:
        return None
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj = adj.coalesce()
            idx = adj.indices().detach().cpu().numpy()
            return idx
        dense = adj.detach().cpu().numpy()
        row, col = np.nonzero(dense)
        return np.vstack([row, col]).astype(np.int64)
    try:
        coo = adj.tocoo()
        return np.vstack([coo.row, coo.col]).astype(np.int64)
    except Exception:
        arr = np.asarray(adj)
        row, col = np.nonzero(arr)
        return np.vstack([row, col]).astype(np.int64)


def ball_edge_density_weights(assignment: np.ndarray, adj, n_balls: int) -> np.ndarray:
    edges = _sparse_adj_to_numpy(adj)
    if edges is None or edges.size == 0:
        return np.zeros((n_balls, n_balls), dtype=np.float32)
    assignment = _as_numpy_1d(assignment, dtype=np.int64)
    row_balls = assignment[edges[0]]
    col_balls = assignment[edges[1]]
    weights = np.zeros((n_balls, n_balls), dtype=np.float32)
    np.add.at(weights, (row_balls, col_balls), 1.0)
    weights = weights + weights.T
    np.fill_diagonal(weights, 0.0)
    max_value = float(weights.max())
    if max_value > 0.0:
        weights /= max_value
    return weights


def build_ball_graph(
    centers: np.ndarray,
    assignment: np.ndarray,
    adj=None,
    graph_k: int = 10,
    edge_density_weight: float = 0.0,
) -> nx.Graph:
    centers = l2_normalize(centers)
    n_balls = int(centers.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from(range(n_balls))
    if n_balls <= 1:
        return graph

    similarity = centers @ centers.T
    np.fill_diagonal(similarity, -np.inf)
    similarity = np.maximum(similarity, 0.0)
    if edge_density_weight > 0.0:
        density = ball_edge_density_weights(assignment, adj, n_balls)
        similarity = similarity + float(edge_density_weight) * density

    k = max(1, min(int(graph_k), n_balls - 1))
    for i in range(n_balls):
        neighbors = np.argpartition(-similarity[i], kth=k - 1)[:k]
        for j in neighbors:
            weight = float(similarity[i, j])
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


def granular_ball_clustering(
    embeddings,
    adj=None,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    random_state: int = 0,
    graph_k: int = 10,
    edge_density_weight: float = 0.0,
) -> dict:
    features = l2_normalize(np.asarray(embeddings.detach().cpu().float().numpy() if isinstance(embeddings, torch.Tensor) else embeddings))
    gb = build_ordinary_granular_balls(
        features,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    centers, sizes = ball_centers(features, gb.assignment)
    graph = build_ball_graph(
        centers,
        assignment=gb.assignment,
        adj=adj,
        graph_k=graph_k,
        edge_density_weight=edge_density_weight,
    )
    ball_cluster = modularity_ball_clusters(graph)
    node_cluster = ball_cluster[gb.assignment]
    return {
        "node_cluster": node_cluster.astype(np.int64),
        "ball_cluster": ball_cluster.astype(np.int64),
        "assignment": gb.assignment.astype(np.int64),
        "summary": dict(gb.summary),
        "ball_sizes": sizes,
        "graph_edges": int(graph.number_of_edges()),
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


def evaluate_one(dataset: str, seed: int, split_improvement: float, args) -> tuple[list[dict], list[dict], dict]:
    dataset_key = _dataset_key(dataset)
    setup_seed(int(seed))
    data = _load_dataset(dataset_key, args.device, "percent", 0.2, 0.1, 0.7)
    labels = _as_numpy_1d(data.labels, dtype=np.int64)
    true_k = int(getattr(data, "n_classes", np.unique(labels).size))
    gb_params = load_softdist_xh_gb_config(dataset_key)
    embeddings = load_frozen_embeddings(gb_params["embedding_path"], expected_nodes=int(labels.shape[0]))
    features = l2_normalize(embeddings.detach().cpu().float().numpy())

    gb_cluster = granular_ball_clustering(
        features,
        adj=data.adj,
        max_balls=gb_params["max_balls"],
        min_ball_size=gb_params["min_ball_size"],
        split_improvement=float(split_improvement),
        random_state=int(seed),
        graph_k=args.ball_graph_k,
        edge_density_weight=args.edge_density_weight,
    )
    gb_labels = gb_cluster["node_cluster"]
    k_auto = int(np.unique(gb_labels).size)

    common = {
        "dataset": dataset_key,
        "seed": int(seed),
        "split_improvement": float(split_improvement),
        "true_k": true_k,
        "embedding_path": str(gb_params["embedding_path"]),
        "max_balls": int(gb_params["max_balls"]),
        "min_ball_size": int(gb_params["min_ball_size"]),
        "num_balls": int(gb_cluster["summary"].get("num_balls", 0)),
        "k_auto": k_auto,
        "ball_graph_edges": int(gb_cluster["graph_edges"]),
    }

    metric_rows = []
    gb_metric = clustering_metrics(labels, gb_labels)
    metric_rows.append({**common, "method": "gb_cluster", "uses_true_k": 0, "status": "ok", **gb_metric})

    pred_true_k = _kmeans_predict(features, true_k, seed)
    metric_rows.append(
        {**common, "method": "kmeans_H_trueK", "uses_true_k": 1, "status": "ok", **clustering_metrics(labels, pred_true_k)}
    )

    pred_auto_k = _kmeans_predict(features, k_auto, seed)
    metric_rows.append(
        {**common, "method": "kmeans_H_autoK", "uses_true_k": 0, "status": "ok", **clustering_metrics(labels, pred_auto_k)}
    )

    spectral_labels, spectral_status = _spectral_predict(
        features,
        true_k,
        seed,
        max_nodes=args.spectral_max_nodes,
        n_neighbors=args.spectral_neighbors,
    )
    if spectral_labels is None:
        metric_rows.append(
            {
                **common,
                "method": "spectral_H_trueK",
                "uses_true_k": 1,
                "status": spectral_status,
                "acc": float("nan"),
                "nmi": float("nan"),
                "ari": float("nan"),
                "n_clusters": float("nan"),
            }
        )
    else:
        metric_rows.append(
            {
                **common,
                "method": "spectral_H_trueK",
                "uses_true_k": 1,
                "status": spectral_status,
                **clustering_metrics(labels, spectral_labels),
            }
        )

    wrong_rows = []
    for wrong_k in wrong_k_values(true_k):
        pred_wrong = _kmeans_predict(features, wrong_k, seed)
        wrong_metrics = clustering_metrics(labels, pred_wrong)
        wrong_rows.append(
            {
                **common,
                "wrong_k": int(wrong_k),
                "wrong_k_ratio": float(wrong_k / max(1, true_k)),
                "gb_acc": float(gb_metric["acc"]),
                "gb_nmi": float(gb_metric["nmi"]),
                "kmeans_acc": float(wrong_metrics["acc"]),
                "kmeans_nmi": float(wrong_metrics["nmi"]),
                "acc_advantage": float(gb_metric["acc"] - wrong_metrics["acc"]),
                "nmi_advantage": float(gb_metric["nmi"] - wrong_metrics["nmi"]),
            }
        )

    meta = {
        **common,
        "gb_acc": float(gb_metric["acc"]),
        "gb_nmi": float(gb_metric["nmi"]),
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


def aggregate_metric_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[(row["dataset"], row["split_improvement"], row["method"])].append(row)

    out = []
    for (dataset, split_improvement, method), items in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "split_improvement": split_improvement,
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
            }
        )
    return out


def aggregate_wrong_k_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split_improvement"])].append(row)
    out = []
    for (dataset, split_improvement), items in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "split_improvement": split_improvement,
                "rows": len(items),
                "acc_advantage_mean": _mean(row["acc_advantage"] for row in items),
                "acc_advantage_std": _std(row["acc_advantage"] for row in items),
                "nmi_advantage_mean": _mean(row["nmi_advantage"] for row in items),
                "gb_acc_mean": _mean(row["gb_acc"] for row in items),
                "kmeans_wrong_acc_mean": _mean(row["kmeans_acc"] for row in items),
            }
        )
    return out


def build_gate_rows(metric_rows: list[dict], wrong_rows: list[dict], main_split: float) -> list[dict]:
    by_dataset = defaultdict(dict)
    for row in metric_rows:
        if row.get("status") != "ok" or abs(float(row["split_improvement"]) - float(main_split)) > 1e-12:
            continue
        by_dataset[row["dataset"]].setdefault(row["method"], []).append(row)

    wrong_by_dataset = defaultdict(list)
    for row in wrong_rows:
        if abs(float(row["split_improvement"]) - float(main_split)) <= 1e-12:
            wrong_by_dataset[row["dataset"]].append(row)

    gate_rows = []
    for dataset, methods in sorted(by_dataset.items()):
        gb = methods.get("gb_cluster", [])
        true_k = methods.get("kmeans_H_trueK", [])
        auto_k = methods.get("kmeans_H_autoK", [])
        if not gb or not true_k:
            continue
        gb_acc = _mean(row["acc"] for row in gb)
        gb_nmi = _mean(row["nmi"] for row in gb)
        kmeans_acc = _mean(row["acc"] for row in true_k)
        kmeans_nmi = _mean(row["nmi"] for row in true_k)
        auto_acc = _mean(row["acc"] for row in auto_k)
        auto_nmi = _mean(row["nmi"] for row in auto_k)
        true_k_value = int(true_k[0]["true_k"])
        k_auto_ratio = _mean(float(row["k_auto"]) / max(1, true_k_value) for row in gb)
        acc_gap = float(gb_acc - kmeans_acc)
        nmi_ratio = float(gb_nmi / max(kmeans_nmi, 1e-12))
        wrong_adv = _mean(row["acc_advantage"] for row in wrong_by_dataset[dataset])
        g1 = bool(nmi_ratio >= 0.9 and acc_gap >= -0.05)
        g2 = bool(wrong_adv >= 0.05)
        g3 = bool(0.5 <= k_auto_ratio <= 2.0)
        g4 = bool((gb_acc - auto_acc) >= 0.005 or (gb_nmi - auto_nmi) >= 0.005)
        gate_rows.append(
            {
                "dataset": dataset,
                "split_improvement": main_split,
                "true_k": true_k_value,
                "k_auto_mean": _mean(row["k_auto"] for row in gb),
                "num_balls_mean": _mean(row["num_balls"] for row in gb),
                "gb_acc_mean": gb_acc,
                "gb_nmi_mean": gb_nmi,
                "kmeans_trueK_acc_mean": kmeans_acc,
                "kmeans_trueK_nmi_mean": kmeans_nmi,
                "acc_gap_vs_trueK": acc_gap,
                "nmi_ratio_vs_trueK": nmi_ratio,
                "wrong_k_acc_advantage_mean": wrong_adv,
                "gb_minus_kmeans_autoK_acc": float(gb_acc - auto_acc),
                "gb_minus_kmeans_autoK_nmi": float(gb_nmi - auto_nmi),
                "g1_precision_close": int(g1),
                "g2_wrong_k_robust": int(g2),
                "g3_k_auto_reasonable": int(g3),
                "g4_agglomeration_nontrivial": int(g4),
                "gate": "PASS" if (g1 and g2) else "FAIL",
            }
        )
    return gate_rows


def write_report(output_dir: Path, args, aggregate_rows: list[dict], wrong_summary: list[dict], gate_rows: list[dict]) -> None:
    datasets = sorted({row["dataset"] for row in gate_rows})
    g1_count = sum(int(row["g1_precision_close"]) for row in gate_rows)
    g2_count = sum(int(row["g2_wrong_k_robust"]) for row in gate_rows)
    required = math.ceil(2 * len(datasets) / 3) if datasets else 0
    decision = "GO" if datasets and g1_count >= required and g2_count >= required else "NO-GO"

    lines = [
        "# Phase 41: GB Clustering Feasibility",
        "",
        f"Decision: **{decision}**",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{args.datasets}`; seeds: `{args.seeds}`.",
        f"- Split improvements: `{args.split_improvements}`; main split: `{args.main_split_improvement}`.",
        "- No label noise and no train/val/test split is used for clustering decisions; labels are evaluation-only.",
        f"- Ball graph k: `{args.ball_graph_k}`; edge density weight: `{args.edge_density_weight}`.",
        "",
        "## Gate Summary",
        "",
        "| Dataset | K true | K auto | Balls | GB ACC | KMeans trueK ACC | ACC gap | GB NMI | KMeans trueK NMI | NMI ratio | Wrong-K ACC adv | G1 | G2 | G3 | G4 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        lines.append(
            "| {dataset} | {true_k} | {k_auto_mean:.2f} | {num_balls_mean:.2f} | {gb_acc_mean:.4f} | "
            "{kmeans_trueK_acc_mean:.4f} | {acc_gap_vs_trueK:+.4f} | {gb_nmi_mean:.4f} | "
            "{kmeans_trueK_nmi_mean:.4f} | {nmi_ratio_vs_trueK:.4f} | {wrong_k_acc_advantage_mean:+.4f} | "
            "{g1_precision_close} | {g2_wrong_k_robust} | {g3_k_auto_reasonable} | "
            "{g4_agglomeration_nontrivial} | {gate} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Metric Summary",
            "",
            "| Dataset | Split | Method | ACC | NMI | ARI | K auto | Balls |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        if abs(float(row["split_improvement"]) - float(args.main_split_improvement)) > 1e-12:
            continue
        lines.append(
            "| {dataset} | {split_improvement} | {method} | {acc_mean:.4f}±{acc_std:.4f} | "
            "{nmi_mean:.4f}±{nmi_std:.4f} | {ari_mean:.4f}±{ari_std:.4f} | "
            "{k_auto_mean:.2f} | {num_balls_mean:.2f} |".format(**row)
        )

    lines.extend(["", "## Wrong-K Robustness", "", "| Dataset | Split | GB ACC | Wrong-K KMeans ACC | ACC advantage | NMI advantage |", "|---|---:|---:|---:|---:|---:|"])
    for row in wrong_summary:
        if abs(float(row["split_improvement"]) - float(args.main_split_improvement)) > 1e-12:
            continue
        lines.append(
            "| {dataset} | {split_improvement} | {gb_acc_mean:.4f} | {kmeans_wrong_acc_mean:.4f} | "
            "{acc_advantage_mean:+.4f}±{acc_advantage_std:.4f} | {nmi_advantage_mean:+.4f} |".format(**row)
        )

    lines.extend(["", "## Decision Notes", ""])
    if decision == "GO":
        lines.append("- GO: G1 and G2 passed on at least 2/3 evaluated datasets.")
    else:
        lines.append("- NO-GO: G1 and/or G2 did not pass on at least 2/3 evaluated datasets.")
    lines.append("- G1 uses `NMI >= 90% of kmeans_H_trueK` and `ACC gap >= -5pp`.")
    lines.append("- G2 uses mean ACC advantage over mis-specified-K k-means >= +5pp.")
    lines.append("- G3/G4 are diagnostic gates for K_auto reasonableness and nontrivial ball agglomeration.")

    (output_dir / "gb_clustering_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose K-free granular-ball clustering on frozen GRACE embeddings.")
    parser.add_argument("--datasets", default="cora,citeseer,pubmed")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split_improvements", default="0.02,0.05,0.1")
    parser.add_argument("--main_split_improvement", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="runs/gb_clustering_feasibility_20260630")
    parser.add_argument("--ball_graph_k", type=int, default=10)
    parser.add_argument("--edge_density_weight", type=float, default=0.0)
    parser.add_argument("--spectral_max_nodes", type=int, default=6000)
    parser.add_argument("--spectral_neighbors", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _parse_str_list(args.datasets)
    seeds = _parse_int_list(args.seeds)
    split_improvements = _parse_float_list(args.split_improvements)

    metric_rows: list[dict] = []
    wrong_rows: list[dict] = []
    error_rows: list[dict] = []

    for dataset in datasets:
        for seed in seeds:
            for split_improvement in split_improvements:
                try:
                    rows, wrong, meta = evaluate_one(dataset, seed, split_improvement, args)
                    metric_rows.extend(rows)
                    wrong_rows.extend(wrong)
                    print(
                        f"[ok] dataset={_dataset_key(dataset)} seed={seed} split={split_improvement} "
                        f"balls={meta['num_balls']} k_auto={meta['k_auto']} "
                        f"gb_acc={meta['gb_acc']:.4f} gb_nmi={meta['gb_nmi']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    row = {
                        "dataset": _dataset_key(dataset),
                        "seed": int(seed),
                        "split_improvement": float(split_improvement),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    error_rows.append(row)
                    print(f"[error] {row}", flush=True)

                _write_csv(output_dir / "metric_raw.csv", metric_rows)
                _write_csv(output_dir / "wrong_k_raw.csv", wrong_rows)
                _write_csv(output_dir / "error_rows.csv", error_rows)

    aggregate_rows = aggregate_metric_rows(metric_rows)
    wrong_summary = aggregate_wrong_k_rows(wrong_rows)
    gate_rows = build_gate_rows(metric_rows, wrong_rows, args.main_split_improvement)

    _write_csv(output_dir / "metric_summary.csv", aggregate_rows)
    _write_csv(output_dir / "wrong_k_summary.csv", wrong_summary)
    _write_csv(output_dir / "gate_summary.csv", gate_rows)
    write_report(output_dir, args, aggregate_rows, wrong_summary, gate_rows)


if __name__ == "__main__":
    main()
