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

from scripts.run_mlp46_cf_nagb import _as_numpy, _mask_indices, target_quality_metrics
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import apply_ball_soft_targets, load_frozen_embeddings
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.teacher_verified_gb import size_matched_shuffled_assignment
from utils.tools import load_conf, setup_seed


EXPECTED_DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SMINS = (2, 5, 10)
EXPECTED_METHODS = tuple(
    f"{control}_s{smin}"
    for control in ("real", "shuffled")
    for smin in SMINS
)
SEED = 0
GB_CONFIG = "frozengrace_gb_mlp_softdist_beta05_loo"


def _cell_key(dataset: str, noise_rate: float) -> tuple[str, float]:
    return str(dataset).lower(), round(float(noise_rate), 6)


def support_diagnostics(
    assignment: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[dict, list[dict]]:
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    ball_ids, all_counts = np.unique(assignment, return_counts=True)
    train_counts = {
        int(ball_id): int(np.sum(assignment[train_indices] == ball_id))
        for ball_id in ball_ids
    }
    zero_vote_ids = {ball_id for ball_id, count in train_counts.items() if count == 0}
    effective_support = np.asarray(
        [train_counts[int(assignment[node])] - 1 for node in train_indices],
        dtype=np.int64,
    )
    bin_specs = (
        ("0-1", 0, 1),
        ("2-4", 2, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20+", 20, None),
    )
    rows = []
    for name, lower, upper in bin_specs:
        selected = effective_support >= lower
        if upper is not None:
            selected &= effective_support <= upper
        count = int(selected.sum())
        rows.append(
            {
                "support_bin": name,
                "train_node_count": count,
                "train_node_fraction": float(count / max(1, train_indices.size)),
            }
        )
    summary = {
        "num_balls": int(ball_ids.size),
        "zero_vote_ball_count": int(len(zero_vote_ids)),
        "zero_vote_node_count": int(
            sum(
                int(count)
                for ball_id, count in zip(ball_ids.tolist(), all_counts.tolist())
                if int(ball_id) in zero_vote_ids
            )
        ),
        "zero_vote_node_fraction": float(
            sum(
                int(count)
                for ball_id, count in zip(ball_ids.tolist(), all_counts.tolist())
                if int(ball_id) in zero_vote_ids
            )
            / max(1, assignment.size)
        ),
        "train_nodes_with_support_lt2": int(np.sum(effective_support < 2)),
        "train_nodes_with_support_lt5": int(np.sum(effective_support < 5)),
        "train_nodes_with_support_lt10": int(np.sum(effective_support < 10)),
        "mean_effective_support": float(effective_support.mean()) if effective_support.size else 0.0,
        "median_effective_support": float(np.median(effective_support)) if effective_support.size else 0.0,
    }
    return summary, rows


def evaluate_phase_a_gate(rows: list[dict]) -> dict:
    by_cell: dict[tuple[str, float], dict[str, dict]] = {}
    duplicates = []
    for row in rows:
        key = _cell_key(row["dataset"], row["noise_rate"])
        method = str(row["method"])
        if method in by_cell.setdefault(key, {}):
            duplicates.append(f"{key[0]}:{key[1]}:{method}")
        by_cell[key][method] = row

    expected_cells = {
        _cell_key(dataset, rate)
        for dataset in EXPECTED_DATASETS
        for rate in NOISE_RATES
    }
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

    brier_improved = 0
    real_structure_improved = 0
    brier_deltas = []
    accuracy_deltas = []
    large_accuracy_drops = []
    low_coverage = []
    integrity_failures = []
    per_cell = {}
    for key in sorted(expected_cells):
        methods = by_cell[key]
        for method, row in methods.items():
            if str(row.get("status", "")) != "ok":
                integrity_failures.append(f"{key}:{method}:status")
            if int(row.get("modified_val_count", -1)) != 0:
                integrity_failures.append(f"{key}:{method}:val_noise")
            if int(row.get("modified_test_count", -1)) != 0:
                integrity_failures.append(f"{key}:{method}:test_noise")
        base = methods["real_s2"]
        candidate = methods["real_s5"]
        shuffled = methods["shuffled_s5"]
        brier_delta = float(candidate["brier"]) - float(base["brier"])
        accuracy_delta = float(candidate["target_accuracy"]) - float(base["target_accuracy"])
        coverage = float(candidate["alpha_coverage"])
        brier_better = brier_delta < -1e-12
        structure_better = float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        brier_improved += int(brier_better)
        real_structure_improved += int(structure_better)
        brier_deltas.append(brier_delta)
        accuracy_deltas.append(accuracy_delta)
        if accuracy_delta < -0.01 - 1e-12:
            large_accuracy_drops.append(f"{key[0]}:{key[1]}")
        if coverage < 0.70 - 1e-12:
            low_coverage.append(f"{key[0]}:{key[1]}")
        per_cell[f"{key[0]}:{key[1]}"] = {
            "brier_delta_s5_vs_s2": brier_delta,
            "accuracy_delta_s5_vs_s2": accuracy_delta,
            "s5_coverage": coverage,
            "real_s5_brier_better_than_shuffled": bool(structure_better),
        }

    mean_brier_delta = float(np.mean(brier_deltas))
    mean_accuracy_delta = float(np.mean(accuracy_deltas))
    if integrity_failures:
        reason = "integrity_failed"
    elif brier_improved < 4 or mean_brier_delta > -0.005 + 1e-12:
        reason = "brier_failed"
    elif mean_accuracy_delta < 0.005 - 1e-12 or large_accuracy_drops:
        reason = "accuracy_failed"
    elif low_coverage:
        reason = "coverage_failed"
    elif real_structure_improved < 4:
        reason = "real_structure_failed"
    else:
        reason = "pass"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "cell_count": len(expected_cells),
        "brier_improved_cell_count": brier_improved,
        "mean_brier_delta_s5_vs_s2": mean_brier_delta,
        "mean_accuracy_delta_s5_vs_s2": mean_accuracy_delta,
        "real_structure_improved_cell_count": real_structure_improved,
        "large_accuracy_drop_cells": large_accuracy_drops,
        "low_coverage_cells": low_coverage,
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


def _result_row(common: dict, method: str, result, noisy, clean, train) -> dict:
    targets = _as_numpy(result.soft_target, np.float64)
    alpha = _as_numpy(result.alpha, np.float64)
    metrics = target_quality_metrics(targets, noisy, clean, train)
    return {
        **common,
        "method": method,
        **metrics,
        "alpha_coverage": float(np.mean(alpha[train] > 0.0)),
        "mean_alpha": float(np.mean(alpha[train])),
        "leave_one_out": int(result.diagnostics.get("leave_one_out", -1)),
        "tie_mode": result.diagnostics.get("tie_mode", ""),
        "min_train_per_ball": int(result.diagnostics.get("min_train_per_ball", -1)),
        "ball_assignment_hash": result.diagnostics.get("ball_assignment_hash", ""),
        "status": "ok",
    }


def run_cell(dataset: str, noise_rate: float, seed: int, device: str) -> tuple[list[dict], list[dict]]:
    setup_seed(seed)
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
        seed=seed,
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

    conf = load_conf(None, GB_CONFIG, dataset)
    embedding_path = Path(conf.gb["embedding_path"])
    if not embedding_path.is_absolute():
        embedding_path = (REPO_ROOT / embedding_path).resolve()
    embeddings = _as_numpy(
        load_frozen_embeddings(embedding_path, expected_nodes=data.n_nodes),
        np.float32,
    )
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    partition = build_ordinary_granular_balls(
        embeddings,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=seed,
    )
    shuffled_assignment = size_matched_shuffled_assignment(
        partition.assignment,
        random_state=seed + 50001,
    )
    common = {
        "dataset": dataset,
        "seed": int(seed),
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
        "embedding_path": str(embedding_path),
    }
    rows = []
    support_rows = []
    for control, assignment in (
        ("real", partition.assignment),
        ("shuffled", shuffled_assignment),
    ):
        summary, bins = support_diagnostics(assignment, train)
        for item in bins:
            support_rows.append({**common, "control": control, **summary, **item})
        for smin in SMINS:
            targets, alpha, pseudo = apply_ball_soft_targets(
                noisy_label=torch.as_tensor(noisy, dtype=torch.long),
                assignment=assignment,
                train_mask=train,
                n_classes=int(data.n_classes),
                min_train_per_ball=smin,
                alpha_baseline="multiclass",
                target_mode="distribution",
                target_beta=0.5,
                leave_one_out=True,
                tie_mode="distribution_only",
            )
            result = type(
                "TargetResult",
                (),
                {
                    "soft_target": targets,
                    "alpha": alpha,
                    "pseudo_label": pseudo,
                    "diagnostics": {
                        "leave_one_out": 1,
                        "tie_mode": "distribution_only",
                        "min_train_per_ball": smin,
                        "ball_assignment_hash": partition.summary.get("assignment_hash", "")
                        if control == "real"
                        else "size_matched_shuffled",
                    },
                },
            )()
            rows.append(
                _result_row(common, f"{control}_s{smin}", result, noisy, clean, train)
            )
    return rows, support_rows


def _report(rows: list[dict], support_rows: list[dict], decision: dict) -> str:
    lines = [
        "# MLP-50 LOO minimum-vote gate report",
        "",
        "## Protocol",
        "",
        "- Formal `tie_mode=distribution_only`, `leave_one_out=true` targets.",
        "- `s_min` counts other training votes after removing the target node.",
        "- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.",
        "- `s_min=5` is the only candidate; `s_min=10` is diagnostic only.",
        "",
        "## Target quality",
        "",
        "| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Coverage |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["method"])):
        lines.append(
            f"| {row['dataset']} | {float(row['noise_rate']):.1f} | {row['method']} | "
            f"{float(row['target_accuracy']):.4f} | {float(row['brier']):.4f} | "
            f"{float(row['ncr']):.4f} | {float(row['cdr']):.4f} | "
            f"{float(row['alpha_coverage']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Label-availability diagnostic",
            "",
            "| Dataset | Noise | Control | Empty balls | Empty-node fraction | Mean other votes | <5-vote train nodes |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    seen = set()
    for row in sorted(support_rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["control"])):
        key = (row["dataset"], row["noise_rate"], row["control"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {row['dataset']} | {float(row['noise_rate']):.1f} | {row['control']} | "
            f"{int(row['zero_vote_ball_count'])} | {float(row['zero_vote_node_fraction']):.4f} | "
            f"{float(row['mean_effective_support']):.2f} | {int(row['train_nodes_with_support_lt5'])} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- PASS: `{int(bool(decision.get('pass', False)))}`.",
            f"- Reason: `{decision.get('reason', 'unknown')}`.",
            f"- Brier-improved cells: `{decision.get('brier_improved_cell_count', 0)}/6`.",
            f"- Mean Brier delta, s5-s2: `{float(decision.get('mean_brier_delta_s5_vs_s2', 0.0)):+.6f}`.",
            f"- Mean target-accuracy delta, s5-s2: `{100 * float(decision.get('mean_accuracy_delta_s5_vs_s2', 0.0)):+.3f}pp`.",
            f"- Real-structure Brier wins: `{decision.get('real_structure_improved_cell_count', 0)}/6`.",
            "",
            "## Decision",
            "",
            (
                "Phase A passed. Run the preregistered seed-0 downstream comparison."
                if decision.get("pass")
                else "Phase A failed. Keep formal s_min=2 and do not run downstream Phase B."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP-50 LOO minimum-vote target gate.")
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument("--noise_rates", default=",".join(str(value) for value in NOISE_RATES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run_dir",
        default=str(REPO_ROOT / "runs" / "mlp50_loo_min_vote_gate_20260805"),
    )
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    rates = tuple(float(item.strip()) for item in args.noise_rates.split(",") if item.strip())
    if datasets != EXPECTED_DATASETS or rates != NOISE_RATES or args.seed != SEED:
        raise ValueError("formal MLP-50 protocol is frozen; datasets/rates/seed cannot change")

    rows = []
    support_rows = []
    for dataset in datasets:
        for rate in rates:
            print(f"MLP50 dataset={dataset} noise_rate={rate} seed={args.seed}", flush=True)
            cell_rows, cell_support = run_cell(dataset, rate, args.seed, args.device)
            rows.extend(cell_rows)
            support_rows.extend(cell_support)

    decision = evaluate_phase_a_gate(rows)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "phase_a_raw.csv", rows)
    _write_csv(run_dir / "support_diagnostics.csv", support_rows)
    (run_dir / "phase_a_gate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "mlp50_report.md").write_text(
        _report(rows, support_rows, decision), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "pass": decision["pass"], "reason": decision["reason"]}))


if __name__ == "__main__":
    main()
