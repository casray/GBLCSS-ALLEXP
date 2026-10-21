from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings, prepare_ball_features
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


DEFAULT_DATASETS = "pubmed,dblp,amazoncom,amazonpho"
DEFAULT_METHOD = "frozengrace_gb_gcn_softdist_xh"
RAW_FIELDS = [
    "dataset",
    "seed",
    "noise_type",
    "noise_rate",
    "noise_scope",
    "n_nodes",
    "n_classes",
    "train_count",
    "actual_train_noise_rate",
    "max_balls",
    "min_ball_size",
    "split_improvement",
    "root_split_improvement",
    "root_split_left_size",
    "root_split_right_size",
    "num_balls",
    "mean_ball_size",
    "max_ball_size",
    "mean_radius",
    "h_knn10_clean_agree",
    "h_knn10_random_agree",
    "h_knn10_lift_over_random",
    "uniform_noisy_train_purity",
    "uniform_true_train_purity",
    "weighted_noisy_train_purity",
    "weighted_true_train_purity",
    "weighted_noisy_train_purity_delta",
    "weighted_true_train_purity_delta",
    "ellipsoid_anisotropy_mean",
    "ellipsoid_anisotropy_median",
    "ellipsoid_anisotropy_max",
    "ellipsoid_condition_number_mean",
    "ellipsoid_top_eigen_ratio_mean",
    "ellipsoid_effective_support_ratio_mean",
    "ellipsoid_effective_support_ratio_min",
    "ellipsoid_weight_std_mean",
    "ellipsoid_min_train_weight_mean",
    "ellipsoid_fallback_ball_ratio",
    "eligible_balls",
    "eligible_train_nodes",
]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _gb_value(model_conf, key, default):
    gb_conf = getattr(model_conf, "gb", {})
    if isinstance(gb_conf, dict):
        value = gb_conf.get(key, default)
    else:
        value = getattr(gb_conf, key, default)
    return default if value is None else value


def _resolve_embedding_path(path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _as_numpy(values, dtype=None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values)
    return arr.astype(dtype) if dtype is not None else arr


def root_split_diagnostics(features: np.ndarray, min_ball_size: int, random_state: int = 0) -> dict:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty 2-D array")
    if features.shape[0] < 2 * int(min_ball_size):
        return {
            "root_split_improvement": 0.0,
            "root_split_left_size": 0,
            "root_split_right_size": 0,
        }
    center = features.mean(axis=0, keepdims=True)
    old_avg = float(np.linalg.norm(features - center, axis=1).mean())
    labels = KMeans(n_clusters=2, n_init=10, random_state=int(random_state)).fit_predict(features)
    if np.unique(labels).size < 2 or old_avg <= 1e-12:
        return {
            "root_split_improvement": 0.0,
            "root_split_left_size": 0,
            "root_split_right_size": 0,
        }
    sizes = []
    weighted_avg = 0.0
    for cluster_id in (0, 1):
        x = features[labels == cluster_id]
        sizes.append(int(x.shape[0]))
        cluster_center = x.mean(axis=0, keepdims=True)
        weighted_avg += float(np.linalg.norm(x - cluster_center, axis=1).mean()) * int(x.shape[0])
    new_avg = weighted_avg / int(features.shape[0])
    return {
        "root_split_improvement": float((old_avg - new_avg) / old_avg),
        "root_split_left_size": int(sizes[0]),
        "root_split_right_size": int(sizes[1]),
    }


def h_knn_label_diagnostics(features: np.ndarray, labels: np.ndarray, k: int = 10) -> dict:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same node count")
    n_neighbors = min(int(k) + 1, int(features.shape[0]))
    if n_neighbors <= 1:
        return {
            "h_knn10_clean_agree": 0.0,
            "h_knn10_random_agree": 0.0,
            "h_knn10_lift_over_random": 0.0,
        }
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    indices = nbrs.fit(features).kneighbors(features, return_distance=False)[:, 1:]
    clean_agree = float((labels[indices] == labels[:, None]).mean())
    counts = np.bincount(labels)
    probs = counts / max(1, int(counts.sum()))
    random_agree = float((probs * probs).sum())
    return {
        "h_knn10_clean_agree": clean_agree,
        "h_knn10_random_agree": random_agree,
        "h_knn10_lift_over_random": float(clean_agree / random_agree) if random_agree > 0 else 0.0,
    }


def ellipsoid_cluster_diagnostics(
    features: np.ndarray,
    assignment: np.ndarray,
    noisy_label: np.ndarray,
    clean_label: np.ndarray,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int = 2,
    ridge_scale: float = 0.05,
    weight_floor: float = 0.1,
) -> dict:
    features = np.asarray(features, dtype=np.float32)
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    noisy_label = np.asarray(noisy_label, dtype=np.int64).reshape(-1)
    clean_label = np.asarray(clean_label, dtype=np.int64).reshape(-1)
    train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    if features.shape[0] != assignment.shape[0]:
        raise ValueError("features and assignment must have the same node count")

    eligible_balls = 0
    eligible_train_nodes = 0
    fallback_balls = 0
    uniform_noisy_sum = 0.0
    uniform_true_sum = 0.0
    weighted_noisy_sum = 0.0
    weighted_true_sum = 0.0
    anisotropy_values = []
    condition_values = []
    top_ratio_values = []
    support_ratio_values = []
    weight_std_values = []
    min_weight_values = []

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id).astype(np.int64)
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < int(min_train_per_ball):
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        uniform_noisy_counts = np.bincount(noisy_label[ball_train], minlength=int(n_classes)).astype(np.float64)
        uniform_true_counts = np.bincount(clean_label[ball_train], minlength=int(n_classes)).astype(np.float64)
        uniform_noisy_sum += float(uniform_noisy_counts.max())
        uniform_true_sum += float(uniform_true_counts.max())

        weights, used_fallback, shape_diag = _ellipsoid_weights_and_shape(
            features=features,
            members=members,
            ball_train=ball_train,
            ridge_scale=float(ridge_scale),
            weight_floor=float(weight_floor),
        )
        fallback_balls += int(used_fallback)
        weighted_noisy_counts = np.bincount(
            noisy_label[ball_train],
            weights=weights,
            minlength=int(n_classes),
        ).astype(np.float64)
        weighted_true_counts = np.bincount(
            clean_label[ball_train],
            weights=weights,
            minlength=int(n_classes),
        ).astype(np.float64)
        weighted_noisy_sum += float(weighted_noisy_counts.max()) / max(1e-12, float(weights.mean()))
        weighted_true_sum += float(weighted_true_counts.max()) / max(1e-12, float(weights.mean()))
        if shape_diag:
            anisotropy_values.append(shape_diag["anisotropy"])
            condition_values.append(shape_diag["condition_number"])
            top_ratio_values.append(shape_diag["top_eigen_ratio"])
        weight_sum = float(weights.sum())
        weight_sq_sum = float((weights * weights).sum())
        support = (weight_sum * weight_sum / weight_sq_sum) if weight_sq_sum > 0 else 0.0
        support_ratio_values.append(float(support / max(1, int(ball_train.size))))
        weight_std_values.append(float(np.std(weights)))
        min_weight_values.append(float(np.min(weights)))

    denom = max(1, int(eligible_train_nodes))
    uniform_noisy = float(uniform_noisy_sum / denom)
    uniform_true = float(uniform_true_sum / denom)
    weighted_noisy = float(weighted_noisy_sum / denom)
    weighted_true = float(weighted_true_sum / denom)
    return {
        "eligible_balls": int(eligible_balls),
        "eligible_train_nodes": int(eligible_train_nodes),
        "uniform_noisy_train_purity": uniform_noisy,
        "uniform_true_train_purity": uniform_true,
        "weighted_noisy_train_purity": weighted_noisy,
        "weighted_true_train_purity": weighted_true,
        "weighted_noisy_train_purity_delta": float(weighted_noisy - uniform_noisy),
        "weighted_true_train_purity_delta": float(weighted_true - uniform_true),
        "ellipsoid_anisotropy_mean": _safe_mean(anisotropy_values),
        "ellipsoid_anisotropy_median": _safe_median(anisotropy_values),
        "ellipsoid_anisotropy_max": float(np.max(anisotropy_values)) if anisotropy_values else 0.0,
        "ellipsoid_condition_number_mean": _safe_mean(condition_values),
        "ellipsoid_top_eigen_ratio_mean": _safe_mean(top_ratio_values),
        "ellipsoid_effective_support_ratio_mean": _safe_mean(support_ratio_values),
        "ellipsoid_effective_support_ratio_min": float(np.min(support_ratio_values)) if support_ratio_values else 0.0,
        "ellipsoid_weight_std_mean": _safe_mean(weight_std_values),
        "ellipsoid_min_train_weight_mean": _safe_mean(min_weight_values),
        "ellipsoid_fallback_ball_ratio": float(fallback_balls / max(1, eligible_balls)),
    }


def _ellipsoid_weights_and_shape(
    features: np.ndarray,
    members: np.ndarray,
    ball_train: np.ndarray,
    ridge_scale: float,
    weight_floor: float,
) -> tuple[np.ndarray, bool, dict]:
    if members.size < 2 or ball_train.size == 0:
        return np.ones(int(ball_train.size), dtype=np.float64), True, {}
    x = np.asarray(features[members], dtype=np.float64)
    if not np.isfinite(x).all():
        return np.ones(int(ball_train.size), dtype=np.float64), True, {}
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / float(max(1, int(members.size) - 1))
    trace = float(np.trace(cov))
    dim = int(cov.shape[0])
    if dim <= 0 or not np.isfinite(trace) or trace <= 0.0:
        return np.ones(int(ball_train.size), dtype=np.float64), True, {}
    ridge = float(ridge_scale) * trace / float(dim) + 1e-6
    try:
        eigvals = np.linalg.eigvalsh(cov + ridge * np.eye(dim))
        inv_cov = np.linalg.pinv(cov + ridge * np.eye(dim))
    except np.linalg.LinAlgError:
        return np.ones(int(ball_train.size), dtype=np.float64), True, {}
    eigvals = np.clip(eigvals, 1e-12, None)
    top = float(eigvals[-1])
    mean = float(np.mean(eigvals))
    shape_diag = {
        "anisotropy": float(top / max(mean, 1e-12)),
        "condition_number": float(top / max(float(eigvals[0]), 1e-12)),
        "top_eigen_ratio": float(top / max(float(np.sum(eigvals)), 1e-12)),
    }
    projected = centered @ inv_cov
    d2_all = np.maximum((projected * centered).sum(axis=1), 0.0)
    if not np.isfinite(d2_all).all():
        return np.ones(int(ball_train.size), dtype=np.float64), True, shape_diag
    median_d2 = float(np.median(d2_all))
    if not np.isfinite(median_d2) or median_d2 <= 0.0:
        return np.ones(int(ball_train.size), dtype=np.float64), True, shape_diag
    member_pos = {int(node_id): pos for pos, node_id in enumerate(members.tolist())}
    try:
        train_pos = np.asarray([member_pos[int(node_id)] for node_id in ball_train.tolist()], dtype=np.int64)
    except KeyError:
        return np.ones(int(ball_train.size), dtype=np.float64), True, shape_diag
    floor = max(0.0, min(1.0, float(weight_floor)))
    d2_norm = d2_all[train_pos] / max(median_d2, 1e-12)
    weights = floor + (1.0 - floor) * np.exp(-0.5 * d2_norm)
    if not np.isfinite(weights).all():
        return np.ones(int(ball_train.size), dtype=np.float64), True, shape_diag
    return weights.astype(np.float64), False, shape_diag


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def diagnose_one(args, dataset_name: str, seed: int) -> dict:
    setup_seed(seed)
    dataset_key = dataset_name.lower()
    data = _load_dataset(
        dataset_key,
        device=args.device,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    model_conf = load_conf(None, args.method, dataset_key)
    embedding_path = _gb_value(model_conf, "ball_embedding_path", _gb_value(model_conf, "embedding_path", None))
    if embedding_path is None:
        raise ValueError(f"missing gb.embedding_path for method={args.method} dataset={dataset_key}")
    features = prepare_ball_features(
        load_frozen_embeddings(_resolve_embedding_path(embedding_path), expected_nodes=int(data.labels.shape[0]))
    ).detach().cpu().numpy()
    clean_label = _as_numpy(data.labels, dtype=np.int64)
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=data.train_masks if args.noise_scope == "train" else None,
        debug=False,
    )
    noisy_np = _as_numpy(noisy_label, dtype=np.int64)
    train_idx = _as_numpy(data.train_masks, dtype=np.int64).reshape(-1)
    modified = _as_numpy(modified_mask, dtype=np.int64).reshape(-1)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    max_balls = int(_gb_value(model_conf, "max_balls", 64))
    min_ball_size = int(_gb_value(model_conf, "min_ball_size", 32))
    split_improvement = float(_gb_value(model_conf, "split_improvement", 0.05))
    gb_result = build_ordinary_granular_balls(
        features,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=seed,
    )
    return {
        "dataset": dataset_key,
        "seed": int(seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "n_nodes": int(features.shape[0]),
        "n_classes": int(data.n_classes),
        "train_count": int(train_idx.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        "max_balls": max_balls,
        "min_ball_size": min_ball_size,
        "split_improvement": split_improvement,
        **root_split_diagnostics(features, min_ball_size=min_ball_size, random_state=seed),
        **gb_result.summary,
        **h_knn_label_diagnostics(features, clean_label, k=int(args.knn_k)),
        **ellipsoid_cluster_diagnostics(
            features=features,
            assignment=gb_result.assignment,
            noisy_label=noisy_np,
            clean_label=clean_label,
            train_idx=train_idx,
            n_classes=int(data.n_classes),
            min_train_per_ball=int(_gb_value(model_conf, "min_train_per_ball", 2)),
            ridge_scale=float(_gb_value(model_conf, "ellipsoid_ridge_scale", args.ellipsoid_ridge_scale)),
            weight_floor=float(_gb_value(model_conf, "ellipsoid_weight_floor", args.ellipsoid_weight_floor)),
        ),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
    out = []
    skip = {"dataset", "seed", "noise_type", "noise_scope"}
    for dataset, group in sorted(groups.items()):
        item = {"dataset": dataset, "n": len(group)}
        keys = [key for key in RAW_FIELDS if key not in skip and key in group[0]]
        for key in keys:
            values = [float(row[key]) for row in group if row.get(key, "") not in ("", None)]
            if not values:
                continue
            item[f"{key}_mean"] = float(np.mean(values))
            item[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for preferred in RAW_FIELDS:
        if any(preferred in row for row in rows):
            fieldnames.append(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, aggregate: list[dict]) -> None:
    lines = [
        "# Ellipsoid geometry diagnostic report",
        "",
        "| Dataset | Balls | Root split gain | H-kNN agree | Lift | Eff. support | Anisotropy | Weighted true purity delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {dataset} | {balls:.1f} | {root:.4f} | {knn:.4f} | {lift:.2f} | {support:.4f} | {anis:.2f} | {delta:.4f} |".format(
                dataset=row["dataset"],
                balls=float(row.get("num_balls_mean", 0.0)),
                root=float(row.get("root_split_improvement_mean", 0.0)),
                knn=float(row.get("h_knn10_clean_agree_mean", 0.0)),
                lift=float(row.get("h_knn10_lift_over_random_mean", 0.0)),
                support=float(row.get("ellipsoid_effective_support_ratio_mean_mean", 0.0)),
                anis=float(row.get("ellipsoid_anisotropy_mean_mean", 0.0)),
                delta=float(row.get("weighted_true_train_purity_delta_mean", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose H-space ball and ellipsoid geometry quality.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--ellipsoid_ridge_scale", type=float, default=0.05)
    parser.add_argument("--ellipsoid_weight_floor", type=float, default=0.1)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "ellipsoid_geometry_20260626"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_rows = []
    for dataset in _parse_str_list(args.datasets):
        for seed in _parse_int_list(args.seeds):
            print(f"ELLIPSOID_GEOM dataset={dataset} seed={seed}", flush=True)
            row = diagnose_one(args, dataset, seed)
            raw_rows.append(row)
            write_csv(output_dir / "geometry_raw.csv", raw_rows)
    aggregate = aggregate_rows(raw_rows)
    write_csv(output_dir / "geometry_aggregate.csv", aggregate)
    write_report(output_dir / "geometry_report.md", aggregate)
    print(f"ELLIPSOID_GEOM output_dir={output_dir}")


if __name__ == "__main__":
    main()
