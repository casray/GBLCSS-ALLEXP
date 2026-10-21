from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = {
    "noise_rate",
    "alpha_max",
    "split_seed",
    "pretrain_seed",
    "noise_seed",
    "classifier_seed",
    "train_ratio",
    "val_ratio",
    "best_val_acc",
    "test_acc_at_best_val",
    "effective_smoothing_coverage",
    "effective_train_nodes_smoothed",
    "alpha_mean_all_train",
    "alpha_mean_nonzero",
    "alpha_max_actual",
    "avg_train_nodes_per_ball",
    "avg_train_nodes_per_nonempty_ball",
    "eligible_ball_count",
    "eligible_train_node_count",
    "gb_kmeans_assignment_nmi",
    "gb_topk_selected_neighbor_overlap",
    "gb_topk_selected_neighbor_jaccard",
    "weight_mean",
    "weight_std",
    "weight_min",
    "weight_max",
    "clip_low_ratio",
    "clip_high_ratio",
    "low_weight_ratio",
    "raw_weight_mean",
    "coverage_gate_mean",
    "entropy_gate_mean",
    "support_gate_mean",
    "geometry_gate_mean",
}


SUMMARY_DIAGNOSTICS = [
    "effective_smoothing_coverage",
    "effective_train_nodes_smoothed",
    "alpha_mean_all_train",
    "alpha_mean_nonzero",
    "alpha_max_actual",
    "avg_train_nodes_per_ball",
    "avg_train_nodes_per_nonempty_ball",
    "eligible_ball_count",
    "eligible_train_node_count",
    "gb_kmeans_assignment_nmi",
    "gb_topk_selected_neighbor_overlap",
    "gb_topk_selected_neighbor_jaccard",
    "weight_mean",
    "weight_std",
    "weight_min",
    "weight_max",
    "clip_low_ratio",
    "clip_high_ratio",
    "low_weight_ratio",
    "raw_weight_mean",
    "coverage_gate_mean",
    "entropy_gate_mean",
    "support_gate_mean",
    "geometry_gate_mean",
]

FORMAL_DATASETS = {"Cora", "CiteSeer"}
FORMAL_NOISE_TYPE = "symmetric"
FORMAL_NOISE_RATES = {0.0, 0.2, 0.4}
FORMAL_METHODS = {
    "ce",
    "global",
    "kmeans",
    "topk_matched",
    "gb",
    "gb_prior",
    "gb_ca_ce",
    "gb_ca_global",
    "gb_weighted_ce",
    "kmeans_weighted_ce",
    "topk_matched_weighted_ce",
}
FORMAL_ALPHAS = {0.0, 0.3, 0.5}
FORMAL_RATIO_TRAIN_RATIOS = {0.1, 0.2}
FORMAL_RATIO_VAL_RATIO = 0.1
WEIGHTED_METHODS = {"gb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"}


def _float_equal(left, right: float, tol: float = 1e-8) -> bool:
    return abs(float(left) - right) <= tol


def _is_formal_gate_row(row: dict) -> bool:
    if row.get("dataset") not in FORMAL_DATASETS:
        return False
    if row.get("noise_type") != FORMAL_NOISE_TYPE:
        return False
    if not any(_float_equal(row.get("noise_rate"), rate) for rate in FORMAL_NOISE_RATES):
        return False
    if row.get("method") not in FORMAL_METHODS:
        return False
    if not _float_equal(row.get("split_seed"), 0.0):
        return False
    split_mode = row.get("split_mode") or "planetoid"
    if split_mode not in {"planetoid", "ratio"}:
        return False
    if split_mode == "ratio":
        if not any(_float_equal(row.get("train_ratio"), value) for value in FORMAL_RATIO_TRAIN_RATIOS):
            return False
        if not _float_equal(row.get("val_ratio"), FORMAL_RATIO_VAL_RATIO):
            return False
    if int(row.get("pretrain_seed", -1)) != int(row.get("noise_seed", -2)):
        return False
    if int(row.get("pretrain_seed", -1)) != int(row.get("classifier_seed", -3)):
        return False
    alpha = float(row.get("alpha_max", -1.0))
    if row.get("method") in WEIGHTED_METHODS:
        return _float_equal(alpha, 0.0)
    if not any(_float_equal(alpha, value) for value in FORMAL_ALPHAS):
        return False
    if row.get("method") == "ce":
        return _float_equal(alpha, 0.0)
    return not _float_equal(alpha, 0.0)


def _read_rows(results_root: Path) -> list[dict]:
    rows = []
    for path in results_root.glob("gate_runs/**/result_rows.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for field in NUMERIC_FIELDS:
                    if row.get(field) not in (None, ""):
                        row[field] = float(row[field])
                row["split_mode"] = row.get("split_mode") or "planetoid"
                row["split_name"] = row.get("split_name") or f"split{int(float(row.get('split_seed', 0)))}"
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_gate_results(results_root: Path, formal_only: bool = True) -> tuple[list[dict], list[dict]]:
    rows = _read_rows(results_root)
    if formal_only:
        rows = [row for row in rows if _is_formal_gate_row(row)]
    aggregate_groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["split_mode"],
            row["split_name"],
            row.get("train_ratio", ""),
            row.get("val_ratio", ""),
            row["noise_type"],
            row["noise_rate"],
            row["method"],
            row.get("alpha_max", ""),
        )
        aggregate_groups[key].append(row)

    summary_rows = []
    for key, group in sorted(aggregate_groups.items()):
        test_mean, test_std = _mean_std([float(row["test_acc_at_best_val"]) for row in group])
        val_mean, val_std = _mean_std([float(row["best_val_acc"]) for row in group])
        summary = {
            "dataset": key[0],
            "split_mode": key[1],
            "split_name": key[2],
            "train_ratio": key[3],
            "val_ratio": key[4],
            "noise_type": key[5],
            "noise_rate": key[6],
            "method": key[7],
            "alpha_max": key[8],
            "n": len(group),
            "best_val_mean": val_mean,
            "best_val_std": val_std,
            "test_at_best_val_mean": test_mean,
            "test_at_best_val_std": test_std,
        }
        for field in SUMMARY_DIAGNOSTICS:
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in (None, "", "nan")
            ]
            mean, std = _mean_std(values) if values else (np.nan, np.nan)
            output_prefix = "coverage" if field == "effective_smoothing_coverage" else field
            summary[f"{output_prefix}_mean"] = mean
            summary[f"{output_prefix}_std"] = std
        summary_rows.append(summary)

    paired_rows = _paired_deltas(rows)
    return summary_rows, paired_rows


def _paired_deltas(rows: list[dict]) -> list[dict]:
    by_pair = {}
    for row in rows:
        alpha_key = "" if row["method"] == "ce" else row.get("alpha_max", "")
        key = (
            row["dataset"],
            row["split_mode"],
            row["split_name"],
            row.get("train_ratio", ""),
            row.get("val_ratio", ""),
            row["noise_type"],
            row["noise_rate"],
            row["split_seed"],
            row["pretrain_seed"],
            row["noise_seed"],
            row["classifier_seed"],
            alpha_key,
            row["method"],
        )
        by_pair[key] = float(row["test_acc_at_best_val"])

    deltas = defaultdict(list)
    baseline_methods = ["ce", "global", "kmeans", "topk_matched", "kmeans_weighted_ce", "topk_matched_weighted_ce"]
    for row in rows:
        method = row["method"]
        if method == "ce":
            continue
        alpha_key = row.get("alpha_max", "")
        method_key = (
            row["dataset"],
            row["split_mode"],
            row["split_name"],
            row.get("train_ratio", ""),
            row.get("val_ratio", ""),
            row["noise_type"],
            row["noise_rate"],
            row["split_seed"],
            row["pretrain_seed"],
            row["noise_seed"],
            row["classifier_seed"],
            alpha_key,
            method,
        )
        method_value = by_pair.get(method_key)
        if method_value is None:
            continue
        for baseline in baseline_methods:
            if baseline == method:
                continue
            baseline_alpha = "" if baseline == "ce" else alpha_key
            baseline_key = (
                row["dataset"],
                row["split_mode"],
                row["split_name"],
                row.get("train_ratio", ""),
                row.get("val_ratio", ""),
                row["noise_type"],
                row["noise_rate"],
                row["split_seed"],
                row["pretrain_seed"],
                row["noise_seed"],
                row["classifier_seed"],
                baseline_alpha,
                baseline,
            )
            baseline_value = by_pair.get(baseline_key)
            if baseline_value is None:
                continue
            delta_key = (
                row["dataset"],
                row["split_mode"],
                row["split_name"],
                row.get("train_ratio", ""),
                row.get("val_ratio", ""),
                row["noise_type"],
                row["noise_rate"],
                method,
                baseline,
                alpha_key,
            )
            deltas[delta_key].append(method_value - baseline_value)

    paired_rows = []
    for key, values in sorted(deltas.items()):
        mean, std = _mean_std(values)
        paired_rows.append({
            "dataset": key[0],
            "split_mode": key[1],
            "split_name": key[2],
            "train_ratio": key[3],
            "val_ratio": key[4],
            "noise_type": key[5],
            "noise_rate": key[6],
            "method": key[7],
            "baseline": key[8],
            "alpha_max": key[9],
            "n": len(values),
            "paired_delta_mean": mean,
            "paired_delta_std": std,
            "positive_delta_count": int(sum(value > 0 for value in values)),
        })
    return paired_rows
