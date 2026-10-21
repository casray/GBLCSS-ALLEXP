from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mlp37_grace_seed_variance import load_pretrain_manifest
from scripts.run_mlp46_cf_nagb import _as_numpy, _mask_indices, target_quality_metrics
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    build_knn_soft_targets_from_features,
    load_frozen_embeddings,
    prepare_ball_features,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.teacher_verified_gb import size_matched_shuffled_assignment
from utils.tools import setup_seed


EXPECTED_DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
H_SEEDS = (0, 1, 2)
SEED = 0
EXPECTED_METHODS = (
    "single_h0_gb",
    "mp_gb_3h",
    "single_h0_knn",
    "mp_knn_3h",
    "mp_gb_3h_shuffled",
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "runs"
    / "mlp37_grace_seed_variance_20260716"
    / "grace_pretrain_manifest.csv"
)


def average_soft_targets(targets: list[np.ndarray]) -> np.ndarray:
    if not targets:
        raise ValueError("at least one target matrix is required")
    arrays = [np.asarray(target, dtype=np.float64) for target in targets]
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError("target matrices must be two-dimensional")
    if any(array.shape != shape for array in arrays):
        raise ValueError("all target matrices must have the same shape")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("target matrices must be finite")
    if any(np.max(np.abs(array.sum(axis=1) - 1.0)) > 1e-6 for array in arrays):
        raise ValueError("each target row must lie on the probability simplex")
    averaged = np.mean(np.stack(arrays, axis=0), axis=0)
    if np.max(np.abs(averaged.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("averaged target rows left the probability simplex")
    return averaged


def _cell_key(dataset: str, noise_rate: float) -> tuple[str, float]:
    return str(dataset).lower(), round(float(noise_rate), 6)


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
            duplicates.append(f"{key[0]}:{key[1]}:{method}")
        by_cell[key][method] = row
    missing_cells = sorted(expected_cells - set(by_cell))
    unexpected_cells = sorted(set(by_cell) - expected_cells)
    missing_methods = {
        f"{dataset}:{rate}": sorted(set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {})))
        for dataset, rate in sorted(expected_cells)
        if set(EXPECTED_METHODS) - set(by_cell.get((dataset, rate), {}))
    }
    if missing_cells or unexpected_cells or missing_methods or duplicates:
        return {
            "pass": False,
            "reason": "incomplete",
            "missing_cells": missing_cells,
            "unexpected_cells": unexpected_cells,
            "missing_methods": missing_methods,
            "duplicates": duplicates,
        }

    wins_single = 0
    wins_knn = 0
    wins_shuffled = 0
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
            if int(row.get("modified_val_count", -1)) != 0:
                integrity_failures.append(f"{key}:{method}:val_noise")
            if int(row.get("modified_test_count", -1)) != 0:
                integrity_failures.append(f"{key}:{method}:test_noise")
            if float(row.get("target_simplex_max_error", 1.0)) > 1e-6:
                integrity_failures.append(f"{key}:{method}:simplex")
            expected_h_count = 3 if method.startswith("mp_") else 1
            if int(row.get("h_seed_count", -1)) != expected_h_count:
                integrity_failures.append(f"{key}:{method}:h_seed_count")

        base = methods["single_h0_gb"]
        candidate = methods["mp_gb_3h"]
        mirror = methods["mp_knn_3h"]
        shuffled = methods["mp_gb_3h_shuffled"]
        delta_single = float(candidate["brier"]) - float(base["brier"])
        delta_knn = float(candidate["brier"]) - float(mirror["brier"])
        delta_shuffled = float(candidate["brier"]) - float(shuffled["brier"])
        accuracy_delta = float(candidate["target_accuracy"]) - float(base["target_accuracy"])
        wins_single += int(delta_single < -1e-12)
        wins_knn += int(delta_knn < -1e-12)
        wins_shuffled += int(delta_shuffled < -1e-12)
        if accuracy_delta < -0.005 - 1e-12:
            accuracy_regressions.append(f"{key[0]}:{key[1]}")
        per_cell[f"{key[0]}:{key[1]}"] = {
            "brier_delta_vs_single_gb": delta_single,
            "brier_delta_vs_mp_knn": delta_knn,
            "brier_delta_vs_shuffled": delta_shuffled,
            "target_accuracy_delta_vs_single_gb": accuracy_delta,
        }

    if integrity_failures:
        reason = "integrity_failed"
    elif wins_single < 4:
        reason = "single_gb_brier_gate_failed"
    elif wins_knn < 4:
        reason = "mp_knn_brier_gate_failed"
    elif accuracy_regressions:
        reason = "accuracy_regression"
    elif wins_shuffled < 6:
        reason = "shuffled_structure_gate_failed"
    else:
        reason = "pass"
    count_shortfall = max(0, 4 - wins_single) + max(0, 4 - wins_knn) + max(0, 6 - wins_shuffled)
    near_pass = bool(
        reason != "pass"
        and not integrity_failures
        and not accuracy_regressions
        and count_shortfall == 1
    )
    return {
        "pass": reason == "pass",
        "near_pass": near_pass,
        "reason": reason,
        "cell_count": len(expected_cells),
        "brier_wins_vs_single_gb": wins_single,
        "brier_wins_vs_mp_knn": wins_knn,
        "brier_wins_vs_shuffled": wins_shuffled,
        "accuracy_regression_cells": accuracy_regressions,
        "integrity_failures": integrity_failures,
        "per_cell": per_cell,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _target_row(
    *,
    common: dict,
    method: str,
    targets: np.ndarray,
    alpha: np.ndarray,
    noisy: np.ndarray,
    clean: np.ndarray,
    train: np.ndarray,
    h_seeds: tuple[int, ...],
    h_sha256: tuple[str, ...],
) -> dict:
    metrics = target_quality_metrics(targets, noisy, clean, train)
    return {
        **common,
        "method": method,
        **metrics,
        "mean_alpha": float(np.mean(alpha[train])),
        "alpha_coverage": float(np.mean(alpha[train] > 0.0)),
        "h_seeds": ",".join(str(seed) for seed in h_seeds),
        "h_seed_count": len(h_seeds),
        "h_sha256": ",".join(h_sha256),
        "target_simplex_max_error": float(np.max(np.abs(targets.sum(axis=1) - 1.0))),
        "status": "ok",
    }


def _embedding(manifest: dict, dataset: str, h_seed: int, n_nodes: int) -> tuple[torch.Tensor, dict]:
    item = manifest[(dataset, h_seed)]
    path = Path(str(item["embedding_path"]))
    features = prepare_ball_features(
        load_frozen_embeddings(path, expected_nodes=n_nodes)
    ).cpu()
    if not torch.isfinite(features).all():
        raise ValueError(f"non-finite H for {(dataset, h_seed)}")
    return features, item


def run_cell(
    *,
    dataset: str,
    noise_rate: float,
    device: str,
    manifest: dict,
) -> tuple[list[dict], list[dict]]:
    setup_seed(SEED)
    data = _load_dataset(
        dataset,
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
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
    if np.any(noisy[val] != clean[val]) or np.any(noisy[test] != clean[test]):
        raise RuntimeError("validation/test labels were modified")
    if np.any(~np.isin(modified, train)):
        raise RuntimeError("noise_scope=train was violated")

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
        "n_nodes": int(data.n_nodes),
        "n_classes": int(data.n_classes),
        "train_count": int(train.size),
        "tie_mode": "distribution_only",
        "leave_one_out": 1,
        "knn_exclude_self": 1,
    }
    gb_targets = []
    gb_alphas = []
    shuffled_targets = []
    shuffled_alphas = []
    knn_targets = []
    knn_alphas = []
    h_hashes = []
    per_view = []
    noisy_cpu = noisy_tensor.detach().cpu()
    clean_cpu = data.labels.detach().cpu()
    for h_seed in H_SEEDS:
        features, manifest_item = _embedding(manifest, dataset, h_seed, int(data.n_nodes))
        h_hash = str(manifest_item["embedding_sha256"])
        h_hashes.append(h_hash)
        partition = build_ordinary_granular_balls(
            features,
            max_balls=64,
            min_ball_size=32,
            split_improvement=0.05,
            random_state=SEED,
        )
        gb_target, gb_alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy_cpu,
            assignment=partition.assignment,
            train_mask=train,
            n_classes=int(data.n_classes),
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            leave_one_out=True,
            tie_mode="distribution_only",
        )
        shuffled_assignment = size_matched_shuffled_assignment(
            partition.assignment,
            random_state=51000 + 101 * h_seed,
        )
        shuffled_target, shuffled_alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy_cpu,
            assignment=shuffled_assignment,
            train_mask=train,
            n_classes=int(data.n_classes),
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            leave_one_out=True,
            tie_mode="distribution_only",
        )
        knn_result = build_knn_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy_cpu,
            clean_label=clean_cpu,
            train_mask=train,
            n_classes=int(data.n_classes),
            max_balls=64,
            min_ball_size=32,
            split_improvement=0.05,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            random_state=SEED,
            matched_k=None,
            tie_mode="distribution_only",
            exclude_self=True,
        )
        gb_np = _as_numpy(gb_target, np.float64)
        gb_alpha_np = _as_numpy(gb_alpha, np.float64)
        shuffled_np = _as_numpy(shuffled_target, np.float64)
        shuffled_alpha_np = _as_numpy(shuffled_alpha, np.float64)
        knn_np = _as_numpy(knn_result.soft_target, np.float64)
        knn_alpha_np = _as_numpy(knn_result.alpha, np.float64)
        gb_targets.append(gb_np)
        gb_alphas.append(gb_alpha_np)
        shuffled_targets.append(shuffled_np)
        shuffled_alphas.append(shuffled_alpha_np)
        knn_targets.append(knn_np)
        knn_alphas.append(knn_alpha_np)
        for builder, targets, alpha, extra in (
            ("gb", gb_np, gb_alpha_np, {"num_balls": int(partition.summary["num_balls"]), "matched_k": ""}),
            ("knn", knn_np, knn_alpha_np, {"num_balls": int(knn_result.diagnostics["matched_num_balls"]), "matched_k": int(knn_result.diagnostics["matched_k"])}),
            ("gb_shuffled", shuffled_np, shuffled_alpha_np, {"num_balls": int(partition.summary["num_balls"]), "matched_k": ""}),
        ):
            per_view.append(
                {
                    **common,
                    "h_seed": h_seed,
                    "h_path": str(manifest_item["embedding_path"]),
                    "h_sha256": h_hash,
                    "builder": builder,
                    **target_quality_metrics(targets, noisy, clean, train),
                    "mean_alpha": float(np.mean(alpha[train])),
                    "target_simplex_max_error": float(np.max(np.abs(targets.sum(axis=1) - 1.0))),
                    **extra,
                    "status": "ok",
                }
            )

    mp_gb = average_soft_targets(gb_targets)
    mp_knn = average_soft_targets(knn_targets)
    mp_shuffled = average_soft_targets(shuffled_targets)
    mp_gb_alpha = np.mean(np.stack(gb_alphas), axis=0)
    mp_knn_alpha = np.mean(np.stack(knn_alphas), axis=0)
    mp_shuffled_alpha = np.mean(np.stack(shuffled_alphas), axis=0)
    rows = [
        _target_row(common=common, method="single_h0_gb", targets=gb_targets[0], alpha=gb_alphas[0], noisy=noisy, clean=clean, train=train, h_seeds=(0,), h_sha256=(h_hashes[0],)),
        _target_row(common=common, method="mp_gb_3h", targets=mp_gb, alpha=mp_gb_alpha, noisy=noisy, clean=clean, train=train, h_seeds=H_SEEDS, h_sha256=tuple(h_hashes)),
        _target_row(common=common, method="single_h0_knn", targets=knn_targets[0], alpha=knn_alphas[0], noisy=noisy, clean=clean, train=train, h_seeds=(0,), h_sha256=(h_hashes[0],)),
        _target_row(common=common, method="mp_knn_3h", targets=mp_knn, alpha=mp_knn_alpha, noisy=noisy, clean=clean, train=train, h_seeds=H_SEEDS, h_sha256=tuple(h_hashes)),
        _target_row(common=common, method="mp_gb_3h_shuffled", targets=mp_shuffled, alpha=mp_shuffled_alpha, noisy=noisy, clean=clean, train=train, h_seeds=H_SEEDS, h_sha256=tuple(h_hashes)),
    ]
    return rows, per_view


def _report(rows: list[dict], decision: dict) -> str:
    lines = [
        "# MLP-51 Multi-Partition Granular-Ball Consensus report",
        "",
        "## Frozen protocol",
        "",
        "- Three archived, independently pretrained GRACE H views; fixed equal target averaging.",
        "- Formal tie-safe LOO targets: `distribution_only`, `beta=0.5`, `s_min=2`.",
        "- Exact kNN excludes self and matches each view's mean train votes per ball.",
        "- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.",
        "",
        "## Target quality",
        "",
        "| Dataset | Noise | Method | Acc | Brier | NCR | CDR |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["method"])):
        lines.append(
            f"| {row['dataset']} | {float(row['noise_rate']):.1f} | {row['method']} | "
            f"{float(row['target_accuracy']):.4f} | {float(row['brier']):.4f} | "
            f"{float(row['ncr']):.4f} | {float(row['cdr']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- PASS: `{int(bool(decision.get('pass', False)))}`.",
            f"- Near-pass: `{int(bool(decision.get('near_pass', False)))}`.",
            f"- Reason: `{decision.get('reason', 'unknown')}`.",
            f"- Brier wins vs single-H GB: `{decision.get('brier_wins_vs_single_gb', 0)}/6`.",
            f"- Brier wins vs 3H kNN mirror: `{decision.get('brier_wins_vs_mp_knn', 0)}/6`.",
            f"- Brier wins vs shuffled 3H GB: `{decision.get('brier_wins_vs_shuffled', 0)}/6`.",
            f"- Accuracy-regression cells: `{decision.get('accuracy_regression_cells', [])}`.",
            "",
            "## Decision",
            "",
            (
                "Phase A passed. A separate downstream preregistration is required before training."
                if decision.get("pass")
                else "Phase A did not pass. No downstream training is authorized by this plan."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP-51 multi-partition target diagnostics.")
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument("--noise_rates", default=",".join(str(rate) for rate in NOISE_RATES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "mlp51_multipartition_gb_20260806",
    )
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    rates = tuple(float(item.strip()) for item in args.noise_rates.split(",") if item.strip())
    if datasets != EXPECTED_DATASETS or rates != NOISE_RATES or args.seed != SEED:
        raise ValueError("formal MLP-51 protocol is frozen; datasets/rates/seed cannot change")
    manifest = load_pretrain_manifest(args.manifest)
    rows = []
    per_view = []
    for dataset in datasets:
        for noise_rate in rates:
            print(f"MLP51 dataset={dataset} noise_rate={noise_rate} seed={SEED}", flush=True)
            cell_rows, cell_views = run_cell(
                dataset=dataset,
                noise_rate=noise_rate,
                device=args.device,
                manifest=manifest,
            )
            rows.extend(cell_rows)
            per_view.extend(cell_views)
    decision = evaluate_phase_a_gate(rows)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.run_dir / "phase_a_raw.csv", rows)
    _write_csv(args.run_dir / "per_view_diagnostics.csv", per_view)
    (args.run_dir / "phase_a_gate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run_dir / "mlp51_report.md").write_text(
        _report(rows, decision), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "pass": decision["pass"], "reason": decision["reason"]}))


if __name__ == "__main__":
    main()

