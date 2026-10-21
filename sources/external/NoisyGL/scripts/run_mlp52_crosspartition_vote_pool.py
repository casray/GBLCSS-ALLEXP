from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mlp37_grace_seed_variance import load_pretrain_manifest
from scripts.run_mlp46_cf_nagb import _as_numpy, _mask_indices, target_quality_metrics
from scripts.run_mlp51_multipartition_gb import (
    DEFAULT_MANIFEST,
    EXPECTED_DATASETS,
    H_SEEDS,
    NOISE_RATES,
    SEED,
    average_soft_targets,
)
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import (
    _targets_from_node_distributions,
    apply_ball_soft_targets,
    load_frozen_embeddings,
    prepare_ball_features,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.teacher_verified_gb import size_matched_shuffled_assignment
from utils.tools import setup_seed


EXPECTED_METHODS = (
    "single_h0_gb",
    "mean_target_gb_3h",
    "pooled_gb_3h",
    "pooled_knn_3h",
    "pooled_gb_3h_shuffled",
)


def pooled_ball_loo_counts(
    assignments: list[np.ndarray],
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if not assignments:
        raise ValueError("at least one assignment is required")
    noisy = np.asarray(noisy_labels, dtype=np.int64).reshape(-1)
    train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    counts = np.zeros((train.size, int(n_classes)), dtype=np.float64)
    for raw_assignment in assignments:
        assignment = np.asarray(raw_assignment, dtype=np.int64).reshape(-1)
        if assignment.shape[0] != noisy.shape[0]:
            raise ValueError("assignment and label lengths must match")
        train_groups = assignment[train]
        for group in np.unique(train_groups):
            positions = np.flatnonzero(train_groups == group)
            group_labels = noisy[train[positions]]
            total = np.bincount(group_labels, minlength=int(n_classes)).astype(np.float64)
            counts[positions] += total[None, :]
            counts[positions, group_labels] -= 1.0
    if np.any(counts < -1e-12):
        raise RuntimeError("leave-one-out pooling produced negative counts")
    return np.maximum(counts, 0.0)


def pooled_knn_counts(
    feature_views: list[torch.Tensor],
    matched_ks: list[int],
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if len(feature_views) != len(matched_ks) or not feature_views:
        raise ValueError("feature views and matched k values must be non-empty and aligned")
    noisy = np.asarray(noisy_labels, dtype=np.int64).reshape(-1)
    train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    counts = np.zeros((train.size, int(n_classes)), dtype=np.float64)
    train_tensor = torch.as_tensor(train, dtype=torch.long)
    for features, raw_k in zip(feature_views, matched_ks):
        train_features = prepare_ball_features(features).cpu()[train_tensor]
        similarity = train_features @ train_features.T
        similarity.fill_diagonal_(float("-inf"))
        k = max(0, min(int(raw_k), max(0, train.size - 1)))
        if k == 0:
            continue
        neighbors = torch.topk(similarity, k=k, dim=1, largest=True).indices.cpu().numpy()
        for row_id, neighbor_positions in enumerate(neighbors):
            labels = noisy[train[neighbor_positions]]
            counts[row_id] += np.bincount(labels, minlength=int(n_classes)).astype(np.float64)
    return counts


def target_from_pooled_counts(
    *,
    counts: np.ndarray,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_indices: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    support = counts.sum(axis=1)
    distributions = np.zeros_like(counts)
    valid = support >= 2.0
    distributions[valid] = counts[valid] / support[valid, None]
    noisy_np = noisy_label.detach().cpu().long().numpy()
    invalid_rows = np.flatnonzero(~valid)
    if invalid_rows.size:
        distributions[invalid_rows, noisy_np[train[invalid_rows]]] = 1.0
    soft_target, alpha, _, _ = _targets_from_node_distributions(
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_idx=train,
        distributions=distributions,
        n_classes=int(n_classes),
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
    )
    if invalid_rows.size:
        invalid_nodes = torch.as_tensor(train[invalid_rows], dtype=torch.long)
        soft_target[invalid_nodes] = F.one_hot(
            noisy_label[invalid_nodes].long(), num_classes=int(n_classes)
        ).float()
        alpha[invalid_nodes] = 0.0
    return _as_numpy(soft_target, np.float64), _as_numpy(alpha, np.float64)


def _cell_key(dataset: str, rate: float) -> tuple[str, float]:
    return str(dataset).lower(), round(float(rate), 6)


def evaluate_phase_a_gate(rows: list[dict]) -> dict:
    expected_cells = {
        _cell_key(dataset, rate)
        for dataset in EXPECTED_DATASETS
        for rate in NOISE_RATES
    }
    by_cell: dict[tuple[str, float], dict[str, dict]] = {}
    duplicates = []
    for row in rows:
        key = _cell_key(row["dataset"], row["noise_rate"])
        method = str(row["method"])
        if method in by_cell.setdefault(key, {}):
            duplicates.append(f"{key}:{method}")
        by_cell[key][method] = row
    missing = {
        f"{dataset}:{rate}": sorted(set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {})))
        for dataset, rate in sorted(expected_cells)
        if set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {}))
    }
    if set(by_cell) != expected_cells or missing or duplicates:
        return {"pass": False, "reason": "incomplete", "missing": missing, "duplicates": duplicates}

    brier_wins_single = 0
    knn_pareto_wins = 0
    accuracy_recovery = 0
    shuffled_wins = 0
    accuracy_regressions = []
    integrity_failures = []
    per_cell = {}
    for key in sorted(expected_cells):
        methods = by_cell[key]
        split_hashes = {str(row.get("split_hash", "")) for row in methods.values()}
        noise_hashes = {str(row.get("noise_hash", "")) for row in methods.values()}
        if len(split_hashes) != 1 or "" in split_hashes:
            integrity_failures.append(f"{key}:split_hash")
        if len(noise_hashes) != 1 or "" in noise_hashes:
            integrity_failures.append(f"{key}:noise_hash")
        for method, row in methods.items():
            if str(row.get("status", "")) != "ok":
                integrity_failures.append(f"{key}:{method}:status")
            if int(row.get("modified_count", -1)) != int(row.get("modified_train_count", -2)):
                integrity_failures.append(f"{key}:{method}:noise_scope")
            if int(row.get("modified_val_count", -1)) != 0 or int(row.get("modified_test_count", -1)) != 0:
                integrity_failures.append(f"{key}:{method}:label_leakage")
            if float(row.get("target_simplex_max_error", 1.0)) > 1e-6:
                integrity_failures.append(f"{key}:{method}:simplex")
        single = methods["single_h0_gb"]
        mean_target = methods["mean_target_gb_3h"]
        candidate = methods["pooled_gb_3h"]
        knn = methods["pooled_knn_3h"]
        shuffled = methods["pooled_gb_3h_shuffled"]
        brier_single = float(candidate["brier"]) < float(single["brier"]) - 1e-12
        pareto_knn = (
            float(candidate["brier"]) < float(knn["brier"]) - 1e-12
            and float(candidate["target_accuracy"]) >= float(knn["target_accuracy"]) - 0.005 - 1e-12
        )
        recovered = float(candidate["target_accuracy"]) >= float(mean_target["target_accuracy"]) - 1e-12
        real_win = float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        accuracy_delta = float(candidate["target_accuracy"]) - float(single["target_accuracy"])
        brier_wins_single += int(brier_single)
        knn_pareto_wins += int(pareto_knn)
        accuracy_recovery += int(recovered)
        shuffled_wins += int(real_win)
        if accuracy_delta < -0.005 - 1e-12:
            accuracy_regressions.append(f"{key[0]}:{key[1]}")
        per_cell[f"{key[0]}:{key[1]}"] = {
            "brier_delta_vs_single": float(candidate["brier"]) - float(single["brier"]),
            "brier_delta_vs_pooled_knn": float(candidate["brier"]) - float(knn["brier"]),
            "accuracy_delta_vs_single": accuracy_delta,
            "accuracy_delta_vs_mean_target": float(candidate["target_accuracy"]) - float(mean_target["target_accuracy"]),
        }
    if integrity_failures:
        reason = "integrity_failed"
    elif brier_wins_single < 4:
        reason = "single_gb_brier_gate_failed"
    elif knn_pareto_wins < 4:
        reason = "pooled_knn_gate_failed"
    elif accuracy_recovery < 4:
        reason = "accuracy_recovery_failed"
    elif accuracy_regressions:
        reason = "accuracy_regression"
    elif shuffled_wins < 6:
        reason = "shuffled_gate_failed"
    else:
        reason = "pass"
    shortfall = (
        max(0, 4 - brier_wins_single)
        + max(0, 4 - knn_pareto_wins)
        + max(0, 4 - accuracy_recovery)
        + max(0, 6 - shuffled_wins)
    )
    return {
        "pass": reason == "pass",
        "near_pass": bool(reason != "pass" and not integrity_failures and not accuracy_regressions and shortfall == 1),
        "reason": reason,
        "brier_wins_vs_single_gb": brier_wins_single,
        "knn_pareto_wins": knn_pareto_wins,
        "accuracy_recovery_cells": accuracy_recovery,
        "brier_wins_vs_shuffled": shuffled_wins,
        "accuracy_regression_cells": accuracy_regressions,
        "integrity_failures": integrity_failures,
        "per_cell": per_cell,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_row(common, method, targets, alpha, noisy, clean, train):
    return {
        **common,
        "method": method,
        **target_quality_metrics(targets, noisy, clean, train),
        "mean_alpha": float(np.mean(alpha[train])),
        "alpha_coverage": float(np.mean(alpha[train] > 0.0)),
        "target_simplex_max_error": float(np.max(np.abs(targets.sum(axis=1) - 1.0))),
        "status": "ok",
    }


def run_cell(dataset: str, noise_rate: float, device: str, manifest: dict):
    setup_seed(SEED)
    data = _load_dataset(dataset, device=device, split_type="percent", train_percent=0.2, val_percent=0.1, test_percent=0.7)
    noisy_tensor, modified = _build_noisy_labels(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=float(noise_rate),
        seed=SEED,
        noise_scope="train",
        train_mask=data.train_masks,
        val_mask=data.val_masks,
        debug=False,
    )
    clean = _as_numpy(data.labels, np.int64)
    noisy = _as_numpy(noisy_tensor, np.int64)
    train = _mask_indices(data.train_masks)
    val = _mask_indices(data.val_masks)
    test = _mask_indices(data.test_masks)
    modified = _as_numpy(modified, np.int64).reshape(-1)
    if np.any(noisy[val] != clean[val]) or np.any(noisy[test] != clean[test]) or np.any(~np.isin(modified, train)):
        raise RuntimeError("train-only noise integrity failed")
    common = {
        "dataset": dataset,
        "seed": SEED,
        "noise_type": "uniform",
        "noise_rate": float(noise_rate),
        "noise_scope": "train",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
        "modified_count": int(modified.size),
        "modified_train_count": int(np.isin(modified, train).sum()),
        "modified_val_count": int(np.isin(modified, val).sum()),
        "modified_test_count": int(np.isin(modified, test).sum()),
        "tie_mode": "distribution_only",
        "leave_one_out": 1,
        "h_seed_count": 3,
    }
    features_views = []
    assignments = []
    shuffled_assignments = []
    matched_ks = []
    final_targets = []
    final_alphas = []
    per_view = []
    noisy_cpu = noisy_tensor.detach().cpu()
    clean_cpu = data.labels.detach().cpu()
    for h_seed in H_SEEDS:
        item = manifest[(dataset, h_seed)]
        features = prepare_ball_features(
            load_frozen_embeddings(Path(str(item["embedding_path"])), expected_nodes=int(data.n_nodes))
        ).cpu()
        partition = build_ordinary_granular_balls(features, max_balls=64, min_ball_size=32, split_improvement=0.05, random_state=SEED)
        assignment = np.asarray(partition.assignment, dtype=np.int64)
        shuffled = size_matched_shuffled_assignment(assignment, random_state=52000 + 101 * h_seed)
        train_counts = [int(np.sum(assignment[train] == group)) for group in np.unique(assignment)]
        nonzero = [count for count in train_counts if count > 0]
        matched_k = max(1, int(round(float(np.mean(nonzero)))))
        target, alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy_cpu,
            assignment=assignment,
            train_mask=train,
            n_classes=int(data.n_classes),
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            leave_one_out=True,
            tie_mode="distribution_only",
        )
        target_np = _as_numpy(target, np.float64)
        alpha_np = _as_numpy(alpha, np.float64)
        features_views.append(features)
        assignments.append(assignment)
        shuffled_assignments.append(shuffled)
        matched_ks.append(matched_k)
        final_targets.append(target_np)
        final_alphas.append(alpha_np)
        per_view.append({
            **common,
            "h_seed": h_seed,
            "h_path": str(item["embedding_path"]),
            "h_sha256": str(item["embedding_sha256"]),
            "num_balls": int(partition.summary["num_balls"]),
            "matched_k": matched_k,
            **target_quality_metrics(target_np, noisy, clean, train),
        })

    pooled_gb_counts = pooled_ball_loo_counts(assignments, noisy, train, int(data.n_classes))
    pooled_shuffled_counts = pooled_ball_loo_counts(shuffled_assignments, noisy, train, int(data.n_classes))
    pooled_knn_raw = pooled_knn_counts(features_views, matched_ks, noisy, train, int(data.n_classes))
    pooled_gb, pooled_gb_alpha = target_from_pooled_counts(counts=pooled_gb_counts, noisy_label=noisy_cpu, clean_label=clean_cpu, train_indices=train, n_classes=int(data.n_classes))
    pooled_shuffled, pooled_shuffled_alpha = target_from_pooled_counts(counts=pooled_shuffled_counts, noisy_label=noisy_cpu, clean_label=clean_cpu, train_indices=train, n_classes=int(data.n_classes))
    pooled_knn, pooled_knn_alpha = target_from_pooled_counts(counts=pooled_knn_raw, noisy_label=noisy_cpu, clean_label=clean_cpu, train_indices=train, n_classes=int(data.n_classes))
    mean_target = average_soft_targets(final_targets)
    mean_alpha = np.mean(np.stack(final_alphas), axis=0)
    rows = [
        _target_row(common, "single_h0_gb", final_targets[0], final_alphas[0], noisy, clean, train),
        _target_row(common, "mean_target_gb_3h", mean_target, mean_alpha, noisy, clean, train),
        _target_row(common, "pooled_gb_3h", pooled_gb, pooled_gb_alpha, noisy, clean, train),
        _target_row(common, "pooled_knn_3h", pooled_knn, pooled_knn_alpha, noisy, clean, train),
        _target_row(common, "pooled_gb_3h_shuffled", pooled_shuffled, pooled_shuffled_alpha, noisy, clean, train),
    ]
    return rows, per_view


def _report(rows, decision):
    lines = [
        "# MLP-52 Cross-Partition Vote Pooling report",
        "",
        "| Dataset | Noise | Method | Acc | Brier | NCR | CDR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["method"])):
        lines.append(f"| {row['dataset']} | {float(row['noise_rate']):.1f} | {row['method']} | {float(row['target_accuracy']):.4f} | {float(row['brier']):.4f} | {float(row['ncr']):.4f} | {float(row['cdr']):.4f} |")
    lines.extend([
        "",
        "## Preregistered gate",
        "",
        f"- PASS: `{int(bool(decision.get('pass', False)))}`.",
        f"- Near-pass: `{int(bool(decision.get('near_pass', False)))}`.",
        f"- Reason: `{decision.get('reason', 'unknown')}`.",
        f"- Brier wins vs single-H GB: `{decision.get('brier_wins_vs_single_gb', 0)}/6`.",
        f"- Pareto wins vs pooled kNN: `{decision.get('knn_pareto_wins', 0)}/6`.",
        f"- Accuracy recovery vs mean-target averaging: `{decision.get('accuracy_recovery_cells', 0)}/6`.",
        f"- Brier wins vs pooled shuffled GB: `{decision.get('brier_wins_vs_shuffled', 0)}/6`.",
        f"- Accuracy regressions vs single-H GB: `{decision.get('accuracy_regression_cells', [])}`.",
        "",
        "Phase A passed; preregister downstream separately." if decision.get("pass") else "Phase A failed; no downstream training is authorized.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument("--noise_rates", default=",".join(str(rate) for rate in NOISE_RATES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run_dir", type=Path, default=REPO_ROOT / "runs" / "mlp52_crosspartition_vote_pool_20260806")
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    rates = tuple(float(item.strip()) for item in args.noise_rates.split(",") if item.strip())
    if datasets != EXPECTED_DATASETS or rates != NOISE_RATES or args.seed != SEED:
        raise ValueError("formal MLP-52 protocol is frozen")
    manifest = load_pretrain_manifest(args.manifest)
    rows, per_view = [], []
    for dataset in datasets:
        for rate in rates:
            print(f"MLP52 dataset={dataset} noise_rate={rate} seed={SEED}", flush=True)
            cell_rows, cell_views = run_cell(dataset, rate, args.device, manifest)
            rows.extend(cell_rows)
            per_view.extend(cell_views)
    decision = evaluate_phase_a_gate(rows)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.run_dir / "phase_a_raw.csv", rows)
    _write_csv(args.run_dir / "per_view_diagnostics.csv", per_view)
    (args.run_dir / "phase_a_gate.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    (args.run_dir / "mlp52_report.md").write_text(_report(rows, decision), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pass": decision["pass"], "reason": decision["reason"]}))


if __name__ == "__main__":
    main()

