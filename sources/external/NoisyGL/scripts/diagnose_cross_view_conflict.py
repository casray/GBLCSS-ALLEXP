from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_label_noise_detection import (
    DEFAULT_UNDEFINED_SCORE,
    _as_noisy_numpy,
    _dataset_key,
    _knn_suspicion_scores,
    _mask_to_index_array,
    _mean,
    _parse_int_list,
    _parse_str_list,
    _safe_auc,
    _safe_auprc,
    _safe_corr,
    _std,
    class_coverage_at_k,
    detection_metrics,
    load_softdist_xh_gb_config,
    top_fraction_metrics,
)
from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import setup_seed


PRIMARY_SCORES = ("s_xview", "s_xview_soft")
EPS = 1e-12


def _dense_float_features(features) -> torch.Tensor:
    if isinstance(features, torch.Tensor) and getattr(features, "is_sparse", False):
        features = features.to_dense()
    if isinstance(features, torch.Tensor):
        return features.detach().float().cpu()
    return torch.as_tensor(features, dtype=torch.float32)


def _neutral_scores(n: int, value: float = DEFAULT_UNDEFINED_SCORE) -> np.ndarray:
    return np.full(int(n), float(value), dtype=np.float64)


def _jsd_dist(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / float(p.sum())
    q = q / float(q.sum())
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def leave_one_out_label_views(
    noisy_label,
    assignment,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
):
    noisy_np = _as_noisy_numpy(noisy_label)
    assignment = np.asarray(assignment, dtype=np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    n = int(train_idx.size)
    n_classes = int(n_classes)
    dist = np.zeros((n, n_classes), dtype=np.float64)
    majority = np.full(n, -1, dtype=np.int64)
    confidence = _neutral_scores(n)
    support = np.zeros(n, dtype=np.int64)
    eligible = np.zeros(n, dtype=bool)
    train_pos = {int(node): pos for pos, node in enumerate(train_idx.tolist())}

    for ball_id in sorted(np.unique(assignment[train_idx]).tolist()):
        ball_train = train_idx[assignment[train_idx] == int(ball_id)]
        if ball_train.size < int(min_train_per_ball) or ball_train.size <= 1:
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=n_classes).astype(np.float64)
        for node in ball_train.tolist():
            pos = train_pos[int(node)]
            loo_counts = counts.copy()
            loo_counts[int(noisy_np[int(node)])] -= 1.0
            total = float(loo_counts.sum())
            if total <= 0:
                continue
            loo_dist = loo_counts / total
            dist[pos] = loo_dist
            majority[pos] = int(np.argmax(loo_dist))
            confidence[pos] = float(np.max(loo_dist))
            support[pos] = int(total)
            eligible[pos] = True
    return {
        "dist": dist,
        "majority": majority,
        "confidence": confidence,
        "support": support,
        "eligible": eligible,
    }


def cross_view_conflict_scores(
    noisy_label,
    assignment_x,
    assignment_h,
    train_idx: np.ndarray,
    n_classes: int,
    min_train_per_ball: int,
    undefined_score: float = DEFAULT_UNDEFINED_SCORE,
):
    view_x = leave_one_out_label_views(
        noisy_label=noisy_label,
        assignment=assignment_x,
        train_idx=train_idx,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
    )
    view_h = leave_one_out_label_views(
        noisy_label=noisy_label,
        assignment=assignment_h,
        train_idx=train_idx,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
    )
    n = int(np.asarray(train_idx).size)
    eligible = view_x["eligible"] & view_h["eligible"]
    hard = _neutral_scores(n, undefined_score)
    soft = _neutral_scores(n, undefined_score)
    disagree = np.zeros(n, dtype=np.float64)
    noisy_np = _as_noisy_numpy(noisy_label)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if bool(eligible.any()):
        hard[eligible] = (view_x["majority"][eligible] != view_h["majority"][eligible]).astype(np.float64)
        for pos in np.flatnonzero(eligible):
            soft[pos] = _jsd_dist(view_x["dist"][pos], view_h["dist"][pos])
            if hard[pos] > 0.5:
                label = int(noisy_np[int(train_idx[pos])])
                disagree[pos] = 0.5 * (1.0 - float(view_x["dist"][pos, label])) + 0.5 * (
                    1.0 - float(view_h["dist"][pos, label])
                )
    return {
        "s_xview": hard,
        "s_xview_soft": soft,
        "s_xview_disagree": disagree,
        "eligible": eligible,
        "view_x": view_x,
        "view_h": view_h,
    }


def conditional_auc_with_clean_subset(labels, scores, subset_mask, min_positive: int = 1) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    subset_mask = np.asarray(subset_mask, dtype=bool)
    if int(labels[subset_mask].sum()) < int(min_positive):
        return float("nan")
    if np.unique(labels[subset_mask]).size < 2:
        return float("nan")
    return _safe_auc(labels[subset_mask], scores[subset_mask])


def _quantile_clean_mask(scores, clean_fraction=0.5):
    scores = np.asarray(scores, dtype=np.float64)
    threshold = float(np.quantile(scores, float(clean_fraction)))
    return scores <= threshold


def _combo_scores(s_knn_h, s_knn_x) -> dict[str, np.ndarray]:
    s_knn_h = np.asarray(s_knn_h, dtype=np.float64)
    s_knn_x = np.asarray(s_knn_x, dtype=np.float64)
    return {
        "s_combo_2knn_max": np.maximum(s_knn_h, s_knn_x),
        "s_combo_2knn_mean": 0.5 * (s_knn_h + s_knn_x),
    }


def _detector_rows(dataset, seed, noisy_flag, noisy_train, n_classes, score_map):
    rows = []
    for name, item in score_map.items():
        scores = np.asarray(item["scores"], dtype=np.float64)
        eligible = np.asarray(item["eligible"], dtype=bool)
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "detector": name,
            "role": item["role"],
            "eligible_rate": float(np.mean(eligible)),
        }
        row.update(detection_metrics(noisy_flag.astype(np.int64), scores))
        row.update(top_fraction_metrics(noisy_flag.astype(np.int64), scores))
        row.update(class_coverage_at_k(noisy_flag.astype(np.int64), scores, noisy_train, n_classes))
        if bool(eligible.any()) and np.unique(noisy_flag[eligible]).size >= 2:
            eligible_metrics = detection_metrics(noisy_flag[eligible].astype(np.int64), scores[eligible])
            row["eligible_auroc"] = eligible_metrics["auroc"]
            row["eligible_auprc_lift"] = eligible_metrics["auprc_lift"]
        else:
            row["eligible_auroc"] = float("nan")
            row["eligible_auprc_lift"] = float("nan")
        rows.append(row)
    return rows


def _conditional_rows(dataset, seed, noisy_flag, score_map, min_intersection_noisy: int):
    rows = []
    s_knn_h = score_map["s_knn_H"]["scores"]
    s_knn_x = score_map["s_knn_X"]["scores"]
    h_clean = _quantile_clean_mask(s_knn_h, 0.5)
    x_clean = _quantile_clean_mask(s_knn_x, 0.5)
    both_clean = h_clean & x_clean
    combo = score_map["s_combo_2knn_max"]["scores"]
    for detector in [*PRIMARY_SCORES, "s_xview_disagree", "s_combo_2knn_max", "s_combo_2knn_mean"]:
        scores = score_map[detector]["scores"]
        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "detector": detector,
            "role": score_map[detector]["role"],
            "auc": _safe_auc(noisy_flag, scores),
            "auc_in_knnH_clean50": conditional_auc_with_clean_subset(noisy_flag, scores, h_clean),
            "auc_in_knnX_clean50": conditional_auc_with_clean_subset(noisy_flag, scores, x_clean),
            "auc_in_knnH_AND_knnX_clean50": conditional_auc_with_clean_subset(
                noisy_flag, scores, both_clean, min_positive=int(min_intersection_noisy)
            ),
            "combo_auc_in_knnH_AND_knnX_clean50": conditional_auc_with_clean_subset(
                noisy_flag, combo, both_clean, min_positive=int(min_intersection_noisy)
            ),
            "auc_minus_combo_in_knnH_AND_knnX_clean50": (
                conditional_auc_with_clean_subset(noisy_flag, scores, both_clean, min_positive=int(min_intersection_noisy))
                - conditional_auc_with_clean_subset(noisy_flag, combo, both_clean, min_positive=int(min_intersection_noisy))
            ),
            "intersection_size": int(both_clean.sum()),
            "intersection_noisy": int(noisy_flag[both_clean].sum()),
            "knnH_clean50_size": int(h_clean.sum()),
            "knnH_clean50_noisy": int(noisy_flag[h_clean].sum()),
            "knnX_clean50_size": int(x_clean.sum()),
            "knnX_clean50_noisy": int(noisy_flag[x_clean].sum()),
            "corr_with_knn_H": _safe_corr(scores, s_knn_h),
            "corr_with_knn_X": _safe_corr(scores, s_knn_x),
        })
    return rows


def _conflict_rows(dataset, seed, noisy_flag, scores):
    hard = np.asarray(scores["s_xview"], dtype=np.float64)
    eligible = np.asarray(scores["eligible"], dtype=bool)
    conflict = eligible & (hard > 0.5)
    non_conflict = eligible & (hard <= 0.5)
    return [{
        "dataset": dataset,
        "seed": int(seed),
        "eligible_rate": float(np.mean(eligible)),
        "conflict_rate_all": float(np.mean(hard > 0.5)),
        "conflict_rate_eligible": float(np.mean(conflict[eligible])) if bool(eligible.any()) else float("nan"),
        "conflict_count": int(conflict.sum()),
        "non_conflict_count": int(non_conflict.sum()),
        "conflict_noise_rate": float(np.mean(noisy_flag[conflict])) if bool(conflict.any()) else float("nan"),
        "non_conflict_noise_rate": float(np.mean(noisy_flag[non_conflict])) if bool(non_conflict.any()) else float("nan"),
        "conflict_minus_non_conflict_noise_rate": (
            (float(np.mean(noisy_flag[conflict])) if bool(conflict.any()) else float("nan"))
            - (float(np.mean(noisy_flag[non_conflict])) if bool(non_conflict.any()) else float("nan"))
        ),
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
    train_idx = _mask_to_index_array(data.train_masks)
    eligible_mask = data.train_masks if args.noise_scope == "train" else None
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        eligible_mask=eligible_mask,
        debug=False,
    )
    gb_params = load_softdist_xh_gb_config(dataset_key, method=args.softdist_method)
    h_path = Path(gb_params["embedding_path"])
    if not h_path.exists():
        raise FileNotFoundError(f"Missing frozen embedding for {dataset_key}: {h_path}")
    h = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes)).detach().float().cpu()
    x = _dense_float_features(data.feats)
    gb_h = build_ordinary_granular_balls(
        embeddings=h,
        max_balls=int(gb_params["max_balls"]),
        min_ball_size=int(gb_params["min_ball_size"]),
        split_improvement=float(gb_params["split_improvement"]),
        random_state=seed,
    )
    gb_x = build_ordinary_granular_balls(
        embeddings=x,
        max_balls=int(gb_params["max_balls"]),
        min_ball_size=int(gb_params["min_ball_size"]),
        split_improvement=float(gb_params["split_improvement"]),
        random_state=seed,
    )
    min_train_per_ball = int(gb_params.get("min_train_per_ball", args.min_train_per_ball))
    noisy_np = _as_noisy_numpy(noisy_label)
    noisy_train = noisy_np[train_idx]
    clean_train = data.labels.detach().cpu().long().numpy()[train_idx]
    noisy_flag = noisy_train != clean_train
    xview = cross_view_conflict_scores(
        noisy_label=noisy_label,
        assignment_x=gb_x.assignment,
        assignment_h=gb_h.assignment,
        train_idx=train_idx,
        n_classes=data.n_classes,
        min_train_per_ball=min_train_per_ball,
    )
    s_knn_h = _knn_suspicion_scores(h, noisy_label, train_idx, args.knn_k)
    s_knn_x = _knn_suspicion_scores(x, noisy_label, train_idx, args.knn_k)
    from scripts.diagnose_label_noise_detection import ball_leave_one_out_scores

    s_ball_h, e_ball_h, _ = ball_leave_one_out_scores(
        noisy_label, gb_h.assignment, train_idx, data.n_classes, min_train_per_ball
    )
    s_ball_x, e_ball_x, _ = ball_leave_one_out_scores(
        noisy_label, gb_x.assignment, train_idx, data.n_classes, min_train_per_ball
    )
    combo = _combo_scores(s_knn_h, s_knn_x)
    all_eligible = np.ones(train_idx.size, dtype=bool)
    score_map = {
        "s_xview": {"scores": xview["s_xview"], "eligible": xview["eligible"], "role": "primary"},
        "s_xview_soft": {"scores": xview["s_xview_soft"], "eligible": xview["eligible"], "role": "primary"},
        "s_xview_disagree": {"scores": xview["s_xview_disagree"], "eligible": xview["eligible"], "role": "exploratory"},
        "s_knn_H": {"scores": s_knn_h, "eligible": all_eligible, "role": "baseline"},
        "s_knn_X": {"scores": s_knn_x, "eligible": all_eligible, "role": "baseline"},
        "s_ball_loo_H": {"scores": s_ball_h, "eligible": e_ball_h, "role": "baseline"},
        "s_ball_loo_X": {"scores": s_ball_x, "eligible": e_ball_x, "role": "baseline"},
        "s_combo_2knn_max": {"scores": combo["s_combo_2knn_max"], "eligible": all_eligible, "role": "baseline"},
        "s_combo_2knn_mean": {"scores": combo["s_combo_2knn_mean"], "eligible": all_eligible, "role": "baseline"},
        "random": {
            "scores": np.random.default_rng(int(seed)).random(train_idx.size),
            "eligible": all_eligible,
            "role": "baseline",
        },
    }
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
        "min_train_per_ball": int(min_train_per_ball),
        "num_balls_H": int(gb_h.summary["num_balls"]),
        "mean_ball_size_H": float(gb_h.summary["mean_ball_size"]),
        "num_balls_X": int(gb_x.summary["num_balls"]),
        "mean_ball_size_X": float(gb_x.summary["mean_ball_size"]),
    }
    detector_rows = [{**run_meta, **row} for row in _detector_rows(dataset_key, seed, noisy_flag, noisy_train, data.n_classes, score_map)]
    conditional_rows = [{**run_meta, **row} for row in _conditional_rows(dataset_key, seed, noisy_flag, score_map, args.min_intersection_noisy)]
    conflict_rows = [{**run_meta, **row} for row in _conflict_rows(dataset_key, seed, noisy_flag, xview)]
    return detector_rows, conditional_rows, conflict_rows


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


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
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
        metric_names = sorted({
            k
            for row in group
            for k in row.keys()
            if k not in keys and isinstance(row.get(k), (int, float, np.integer, np.floating))
        })
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            finite_values = [value for value in values if np.isfinite(value)]
            base[f"{metric}_mean"] = _mean(values)
            base[f"{metric}_std"] = _std(values)
            base[f"{metric}_count"] = len(finite_values)
        base["num_seeds"] = len(group)
        out.append(base)
    return out


def gate_rows(conditional_rows: list[dict], conflict_rows: list[dict], pass_auc: float):
    groups = defaultdict(list)
    for row in conditional_rows:
        if row.get("detector") in PRIMARY_SCORES:
            groups[(row["dataset"], row["detector"])].append(row)
    conflict_by_dataset = defaultdict(list)
    for row in conflict_rows:
        conflict_by_dataset[row["dataset"]].append(row)
    out = []
    for (dataset, detector), rows in sorted(groups.items()):
        h_values = [float(row["auc_in_knnH_clean50"]) for row in rows]
        x_values = [float(row["auc_in_knnX_clean50"]) for row in rows]
        both_delta = [float(row["auc_minus_combo_in_knnH_AND_knnX_clean50"]) for row in rows]
        h_pass_count = sum(np.isfinite(v) and v >= float(pass_auc) for v in h_values)
        x_pass_count = sum(np.isfinite(v) and v >= float(pass_auc) for v in x_values)
        both_delta_pos = sum(np.isfinite(v) and v > 0.0 for v in both_delta)
        conflict_delta = [
            float(row.get("conflict_minus_non_conflict_noise_rate", float("nan")))
            for row in conflict_by_dataset.get(dataset, [])
        ]
        out.append({
            "dataset": dataset,
            "detector": detector,
            "knnH_clean_auc_mean": _mean(h_values),
            "knnX_clean_auc_mean": _mean(x_values),
            "both_clean_minus_combo_mean": _mean(both_delta),
            "knnH_pass_seed_count": int(h_pass_count),
            "knnX_pass_seed_count": int(x_pass_count),
            "both_clean_delta_positive_seed_count": int(both_delta_pos),
            "conflict_noise_delta_mean": _mean(conflict_delta),
            "conflict_noise_delta_positive_seed_count": int(sum(np.isfinite(v) and v > 0.0 for v in conflict_delta)),
            "dataset_gate_pass": bool(
                h_pass_count >= 4
                and x_pass_count >= 4
                and _mean(both_delta) >= 0.02
                and sum(np.isfinite(v) and v > 0.0 for v in conflict_delta) >= 3
            ),
        })
    return out


def write_report(path: Path, detector_summary, conditional_summary, conflict_summary, gates, args):
    dataset_passes = defaultdict(int)
    for row in gates:
        if str(row.get("dataset_gate_pass", "False")).lower() == "true" or row.get("dataset_gate_pass") is True:
            dataset_passes[row["detector"]] += 1
    go = any(count >= 2 for count in dataset_passes.values())
    lines = [
        "# Phase 40: Cross-view conflict diagnostic",
        "",
        f"Gate decision: {'GO' if go else 'STOP'}",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{','.join(args.datasets)}`; seeds: `{','.join(map(str, args.seeds))}`.",
        f"- Noise: `{args.noise_type}` {args.noise_rate}; scope `{args.noise_scope}`.",
        f"- Split: `{args.split_type}` {args.train_percent}/{args.val_percent}/{args.test_percent}.",
        f"- kNN k: `{args.knn_k}`; pass_auc: `{args.pass_auc}`; double-clean min noisy: `{args.min_intersection_noisy}`.",
        "- Scores use train noisy labels only. Clean labels are used only for evaluation.",
        "",
        "## Gate Summary",
        "",
        "| Dataset | Detector | H-clean AUC | X-clean AUC | Both-clean minus combo | H pass seeds | X pass seeds | Conflict noise delta | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gates:
        lines.append(
            "| {dataset} | {detector} | {h:.4f} | {x:.4f} | {delta:.4f} | {hp}/5 | {xp}/5 | {cd:.4f} | {gate} |".format(
                dataset=row["dataset"],
                detector=row["detector"],
                h=float(row.get("knnH_clean_auc_mean", float("nan"))),
                x=float(row.get("knnX_clean_auc_mean", float("nan"))),
                delta=float(row.get("both_clean_minus_combo_mean", float("nan"))),
                hp=int(row.get("knnH_pass_seed_count", 0)),
                xp=int(row.get("knnX_pass_seed_count", 0)),
                cd=float(row.get("conflict_noise_delta_mean", float("nan"))),
                gate="PASS" if row.get("dataset_gate_pass") is True else "FAIL",
            )
        )
    lines.extend([
        "",
        "## Detector Summary",
        "",
        "| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in detector_summary:
        lines.append(
            "| {dataset} | {detector} | {role} | {auc:.4f} | {lift:.4f} | {pk:.4f} | {eligible:.4f} |".format(
                dataset=row.get("dataset"),
                detector=row.get("detector"),
                role=row.get("role"),
                auc=float(row.get("auroc_mean", float("nan"))),
                lift=float(row.get("auprc_lift_mean", float("nan"))),
                pk=float(row.get("precision_at_k_mean", float("nan"))),
                eligible=float(row.get("eligible_rate_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Conditional AUC Summary",
        "",
        "| Dataset | Detector | H clean50 | X clean50 | H∩X clean50 | H∩X minus combo | corr H | corr X |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in conditional_summary:
        lines.append(
            "| {dataset} | {detector} | {h:.4f} | {x:.4f} | {both:.4f} | {delta:.4f} | {ch:.4f} | {cx:.4f} |".format(
                dataset=row.get("dataset"),
                detector=row.get("detector"),
                h=float(row.get("auc_in_knnH_clean50_mean", float("nan"))),
                x=float(row.get("auc_in_knnX_clean50_mean", float("nan"))),
                both=float(row.get("auc_in_knnH_AND_knnX_clean50_mean", float("nan"))),
                delta=float(row.get("auc_minus_combo_in_knnH_AND_knnX_clean50_mean", float("nan"))),
                ch=float(row.get("corr_with_knn_H_mean", float("nan"))),
                cx=float(row.get("corr_with_knn_X_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Conflict Summary",
        "",
        "| Dataset | Conflict rate | Conflict noise | Non-conflict noise | Delta | Eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in conflict_summary:
        lines.append(
            "| {dataset} | {rate:.4f} | {cn:.4f} | {nn:.4f} | {delta:.4f} | {eligible:.4f} |".format(
                dataset=row.get("dataset"),
                rate=float(row.get("conflict_rate_eligible_mean", float("nan"))),
                cn=float(row.get("conflict_noise_rate_mean", float("nan"))),
                nn=float(row.get("non_conflict_noise_rate_mean", float("nan"))),
                delta=float(row.get("conflict_minus_non_conflict_noise_rate_mean", float("nan"))),
                eligible=float(row.get("eligible_rate_mean", float("nan"))),
            )
        )
    lines.extend([
        "",
        "## Decision",
        "",
    ])
    if go:
        lines.append("- GO: at least one primary passed G1 on >=2/3 datasets. Plan a separate downstream phase.")
    else:
        lines.append("- STOP: no primary satisfied the preregistered cross-view non-homologous gate on >=2/3 datasets.")
        lines.append("- This supports closing new granular-ball variants for graph-node denoising and using the result as boundary evidence.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 40 cross-view conflict diagnostic.")
    parser.add_argument("--datasets", default="cora,citeseer,coauthorcs")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--softdist_method", default="frozengrace_gb_gcn_softdist_xh")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--pass_auc", type=float, default=0.6)
    parser.add_argument("--min_intersection_noisy", type=int, default=8)
    parser.add_argument("--output_dir", default="runs/cross_view_conflict_20260629")
    args = parser.parse_args()
    args.datasets = _parse_str_list(args.datasets)
    args.seeds = _parse_int_list(args.seeds)

    detector_rows = []
    conditional_rows = []
    conflict_rows = []
    output_dir = Path(args.output_dir)
    for raw_name in ["detector_raw.csv", "conditional_raw.csv", "conflict_raw.csv"]:
        raw_path = output_dir / raw_name
        if raw_path.exists():
            raw_path.unlink()
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"CROSS_VIEW dataset={dataset} seed={seed}", flush=True)
            d_rows, c_rows, f_rows = _run_one(args, dataset, seed)
            detector_rows.extend(d_rows)
            conditional_rows.extend(c_rows)
            conflict_rows.extend(f_rows)
            append_csv(output_dir / "detector_raw.csv", d_rows)
            append_csv(output_dir / "conditional_raw.csv", c_rows)
            append_csv(output_dir / "conflict_raw.csv", f_rows)
            print(f"CROSS_VIEW_DONE dataset={dataset} seed={seed}", flush=True)

    detector_summary = summarize(detector_rows, ["dataset", "detector", "role"])
    conditional_summary = summarize(conditional_rows, ["dataset", "detector", "role"])
    conflict_summary = summarize(conflict_rows, ["dataset"])
    gates = gate_rows(conditional_rows, conflict_rows, args.pass_auc)
    write_csv(output_dir / "detector_summary.csv", detector_summary)
    write_csv(output_dir / "conditional_summary.csv", conditional_summary)
    write_csv(output_dir / "conflict_summary.csv", conflict_summary)
    write_csv(output_dir / "gate_summary.csv", gates)
    write_report(output_dir / "cross_view_conflict_report.md", detector_summary, conditional_summary, conflict_summary, gates, args)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
