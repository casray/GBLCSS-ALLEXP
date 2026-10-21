from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mlp46_cf_nagb import (
    _as_numpy,
    _crossfit_knn_targets,
    _mask_indices,
    target_quality_metrics,
)
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    load_frozen_embeddings,
)
from utils.noise_aware_gb import (
    CrossFitTargetResult,
    build_crossfit_budgeted_targets,
    build_crossfit_targets_from_assignment,
    build_prefixed_geometry_candidate_tree,
    make_stratified_crossfit_folds,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


METHOD_CONFIG = "frozengrace_gb_mlp_softdist_beta05_loo"
EXPECTED_DATASETS = ("cora", "citeseer", "corafull")
PHASE_A_METHODS = (
    "formal_loo_gb",
    "crossfit_formal_gb",
    "fixed_budget_geometry_only",
    "cf_blagb_shuffled_guidance",
    "cf_blagb",
    "cf_blagb_clean_oracle",
    "crossfit_adaptive_knn",
)
BUDGETED_METHODS = (
    "fixed_budget_geometry_only",
    "cf_blagb_shuffled_guidance",
    "cf_blagb",
    "cf_blagb_clean_oracle",
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def evaluate_phase_a_gate(rows: list[dict]) -> dict:
    by_dataset: dict[str, dict[str, dict]] = {}
    duplicate_keys = []
    for row in rows:
        dataset = str(row["dataset"]).lower()
        method = str(row["method"])
        if method in by_dataset.setdefault(dataset, {}):
            duplicate_keys.append(f"{dataset}:{method}")
        by_dataset[dataset][method] = row

    missing = {}
    for dataset in EXPECTED_DATASETS:
        absent = sorted(set(PHASE_A_METHODS) - set(by_dataset.get(dataset, {})))
        if absent:
            missing[dataset] = absent
    unexpected_datasets = sorted(set(by_dataset) - set(EXPECTED_DATASETS))
    if missing or unexpected_datasets or duplicate_keys:
        return {
            "pass": False,
            "reason": "incomplete",
            "missing": missing,
            "unexpected_datasets": unexpected_datasets,
            "duplicate_keys": duplicate_keys,
        }

    exact_budget_datasets = 0
    brier_both_pass_count = 0
    oracle_headroom_pass_count = 0
    accuracy_deltas = []
    large_accuracy_drops = []
    per_dataset = {}
    for dataset in EXPECTED_DATASETS:
        methods = by_dataset[dataset]
        budget_ok = True
        for method in BUDGETED_METHODS:
            row = methods[method]
            target_balls = int(row["target_num_balls"])
            budget_ok = budget_ok and _as_bool(row["exact_budget"])
            budget_ok = budget_ok and int(row["min_num_balls"]) == target_balls
            budget_ok = budget_ok and int(row["max_num_balls"]) == target_balls
            budget_ok = budget_ok and str(row.get("status", "ok")) == "ok"
        exact_budget_datasets += int(budget_ok)

        candidate = methods["cf_blagb"]
        formal = methods["crossfit_formal_gb"]
        geometry = methods["fixed_budget_geometry_only"]
        shuffled = methods["cf_blagb_shuffled_guidance"]
        oracle = methods["cf_blagb_clean_oracle"]
        brier_both = bool(
            float(candidate["brier"]) < float(geometry["brier"]) - 1e-12
            and float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        )
        brier_both_pass_count += int(brier_both)
        oracle_headroom = bool(
            float(oracle["target_accuracy"])
            > float(formal["target_accuracy"]) + 1e-12
        )
        oracle_headroom_pass_count += int(oracle_headroom)
        accuracy_delta = float(candidate["target_accuracy"]) - float(
            formal["target_accuracy"]
        )
        accuracy_deltas.append(accuracy_delta)
        if accuracy_delta < -0.01 - 1e-12:
            large_accuracy_drops.append(dataset)
        per_dataset[dataset] = {
            "exact_budget": budget_ok,
            "brier_both_controls": brier_both,
            "oracle_accuracy_headroom": oracle_headroom,
            "accuracy_delta_vs_crossfit_formal": accuracy_delta,
            "brier_delta_vs_geometry": float(candidate["brier"])
            - float(geometry["brier"]),
            "brier_delta_vs_shuffled": float(candidate["brier"])
            - float(shuffled["brier"]),
        }

    needed = 2
    mean_accuracy_delta = float(np.mean(accuracy_deltas))
    if exact_budget_datasets != len(EXPECTED_DATASETS):
        reason = "budget_failed"
    elif large_accuracy_drops:
        reason = "large_accuracy_drop"
    elif oracle_headroom_pass_count < needed:
        reason = "oracle_no_headroom"
    elif brier_both_pass_count < needed:
        reason = "brier_controls_failed"
    elif mean_accuracy_delta < -1e-12:
        reason = "mean_accuracy_failed"
    else:
        reason = "pass"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "dataset_count": len(EXPECTED_DATASETS),
        "required_pass_count": needed,
        "exact_budget_dataset_count": exact_budget_datasets,
        "brier_both_pass_count": brier_both_pass_count,
        "oracle_headroom_pass_count": oracle_headroom_pass_count,
        "mean_accuracy_delta_vs_crossfit_formal": mean_accuracy_delta,
        "large_accuracy_drop_datasets": large_accuracy_drops,
        "per_dataset": per_dataset,
    }


def _result_row(
    dataset: str,
    method: str,
    soft_targets,
    alpha,
    noisy: np.ndarray,
    clean: np.ndarray,
    train: np.ndarray,
    common: dict,
    diagnostics: dict | None = None,
) -> dict:
    targets = _as_numpy(soft_targets, np.float64)
    alpha_values = _as_numpy(alpha, np.float64)
    metrics = target_quality_metrics(targets, noisy, clean, train)
    diagnostics = dict(diagnostics or {})
    return {
        **common,
        "dataset": dataset,
        "method": method,
        **metrics,
        "alpha_coverage": float(np.mean(alpha_values[train] > 0.0)),
        "mean_alpha": float(np.mean(alpha_values[train])),
        "target_num_balls": diagnostics.get("target_num_balls", ""),
        "median_num_balls": diagnostics.get("median_num_balls", ""),
        "min_num_balls": diagnostics.get("min_num_balls", ""),
        "max_num_balls": diagnostics.get("max_num_balls", ""),
        "exact_budget": diagnostics.get("exact_budget", ""),
        "guidance_mode": diagnostics.get("guidance_mode", ""),
        "candidate_leaf_count": diagnostics.get("candidate_leaf_count", ""),
        "candidate_tree_hash": diagnostics.get("candidate_tree_hash", ""),
        "mean_selected_jsd": diagnostics.get("mean_selected_jsd", ""),
        "mean_selected_geometry_gain": diagnostics.get(
            "mean_selected_geometry_gain", ""
        ),
        "status": "ok",
        "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
    }


def _budgeted_rows(
    dataset_name: str,
    tree,
    noisy: np.ndarray,
    clean: np.ndarray,
    train: np.ndarray,
    folds: np.ndarray,
    n_classes: int,
    target_num_balls: int,
    seed: int,
    common: dict,
) -> tuple[list[dict], list[dict]]:
    rows = []
    partition_rows = []
    modes = (
        ("geometry", "fixed_budget_geometry_only"),
        ("shuffled", "cf_blagb_shuffled_guidance"),
        ("noisy", "cf_blagb"),
        ("clean", "cf_blagb_clean_oracle"),
    )
    for guidance_mode, method in modes:
        result = build_crossfit_budgeted_targets(
            tree=tree,
            noisy_labels=noisy,
            clean_labels=clean if guidance_mode == "clean" else None,
            train_indices=train,
            fold_ids=folds,
            n_classes=n_classes,
            target_num_balls=target_num_balls,
            guidance_mode=guidance_mode,
            target_beta=0.5,
            min_train_per_target=2,
            prior_strength=2.0,
            random_state=seed,
        )
        rows.append(
            _result_row(
                dataset_name,
                method,
                result.soft_targets,
                result.alpha,
                noisy,
                clean,
                train,
                common,
                result.diagnostics,
            )
        )
        hashes = result.diagnostics["fold_assignment_hashes"]
        for fold_position, num_balls in enumerate(result.fold_num_balls):
            partition_rows.append(
                {
                    **common,
                    "dataset": dataset_name,
                    "method": method,
                    "guidance_mode": guidance_mode,
                    "fold": fold_position,
                    "num_balls": int(num_balls),
                    "target_num_balls": target_num_balls,
                    "exact_budget": int(int(num_balls) == target_num_balls),
                    "assignment_hash": hashes[fold_position],
                    "candidate_tree_hash": tree.assignment_hash,
                }
            )
    return rows, partition_rows


def run_phase_a_dataset(
    dataset_name: str,
    seed: int,
    device: str,
    noise_type: str,
    noise_rate: float,
) -> tuple[list[dict], dict, list[dict]]:
    setup_seed(seed)
    data = _load_dataset(
        dataset_name,
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
        noise_type=noise_type,
        noise_rate=noise_rate,
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
    modified_indices = _as_numpy(modified, np.int64).reshape(-1)
    if np.any(noisy[val] != clean[val]) or np.any(noisy[test] != clean[test]):
        raise RuntimeError("validation/test labels were modified")
    if np.any(~np.isin(modified_indices, train)):
        raise RuntimeError("noise_scope=train was violated")

    conf = load_conf(None, METHOD_CONFIG, dataset_name)
    embedding_path = Path(conf.gb["embedding_path"])
    if not embedding_path.is_absolute():
        embedding_path = (REPO_ROOT / embedding_path).resolve()
    embeddings_tensor = load_frozen_embeddings(
        embedding_path, expected_nodes=int(clean.shape[0])
    )
    embeddings = _as_numpy(embeddings_tensor, np.float32)
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    folds = make_stratified_crossfit_folds(
        train,
        noisy,
        n_splits=5,
        random_state=seed,
        n_nodes=clean.shape[0],
    )

    formal = build_ordinary_granular_balls(
        embeddings,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=seed,
    )
    target_num_balls = int(formal.summary["num_balls"])
    candidate_started = time.perf_counter()
    tree = build_prefixed_geometry_candidate_tree(
        embeddings,
        expected_prefix_leaves=target_num_balls,
        prefix_max_leaves=64,
        max_leaves=128,
        min_ball_size=32,
        prefix_split_improvement=0.05,
        random_state=seed,
    )
    candidate_seconds = time.perf_counter() - candidate_started
    formal_assignment_hash = hashlib.sha256(
        np.ascontiguousarray(formal.assignment, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    prefix_assignment_hash = str(tree.config["prefix_assignment_hash"])
    if prefix_assignment_hash != formal_assignment_hash:
        raise RuntimeError(
            "candidate tree prefix does not match the formal assignment"
        )
    if len(tree.leaf_ids) < target_num_balls:
        raise RuntimeError(
            "candidate tree cannot reproduce formal ball budget: "
            f"capacity={len(tree.leaf_ids)}, K_geo={target_num_balls}"
        )

    common = {
        "seed": seed,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "split_hash": _split_fingerprint(
            data.train_masks, data.val_masks, data.test_masks
        ),
        "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
        "modified_count": int(modified_indices.size),
        "modified_train_count": int(np.isin(modified_indices, train).sum()),
        "modified_val_count": int(np.isin(modified_indices, val).sum()),
        "modified_test_count": int(np.isin(modified_indices, test).sum()),
        "train_count": int(train.size),
        "n_nodes": int(clean.shape[0]),
        "n_classes": int(data.n_classes),
        "embedding_path": str(embedding_path),
        "formal_num_balls": target_num_balls,
    }

    rows = []
    formal_soft, formal_alpha, _ = apply_ball_soft_targets(
        noisy_label=torch.as_tensor(noisy, dtype=torch.long),
        assignment=formal.assignment,
        train_mask=train,
        n_classes=int(data.n_classes),
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        leave_one_out=True,
        tie_mode="distribution_only",
    )
    rows.append(
        _result_row(
            dataset_name,
            "formal_loo_gb",
            formal_soft,
            formal_alpha,
            noisy,
            clean,
            train,
            common,
            {
                "target_num_balls": target_num_balls,
                "median_num_balls": target_num_balls,
                "min_num_balls": target_num_balls,
                "max_num_balls": target_num_balls,
            },
        )
    )
    crossfit_formal = build_crossfit_targets_from_assignment(
        formal.assignment,
        noisy,
        train,
        folds,
        int(data.n_classes),
        target_beta=0.5,
        min_train_per_target=2,
    )
    rows.append(
        _result_row(
            dataset_name,
            "crossfit_formal_gb",
            crossfit_formal.soft_targets,
            crossfit_formal.alpha,
            noisy,
            clean,
            train,
            common,
            {
                **crossfit_formal.diagnostics,
                "target_num_balls": target_num_balls,
            },
        )
    )

    budgeted, partition_rows = _budgeted_rows(
        dataset_name,
        tree,
        noisy,
        clean,
        train,
        folds,
        int(data.n_classes),
        target_num_balls,
        seed,
        common,
    )
    rows.extend(budgeted)
    knn_targets, knn_alpha, knn_diagnostics = _crossfit_knn_targets(
        embeddings,
        noisy,
        train,
        folds,
        int(data.n_classes),
        adaptive=True,
    )
    rows.append(
        _result_row(
            dataset_name,
            "crossfit_adaptive_knn",
            knn_targets,
            knn_alpha,
            noisy,
            clean,
            train,
            common,
            knn_diagnostics,
        )
    )

    candidate_row = {
        **common,
        "dataset": dataset_name,
        "candidate_leaf_count": len(tree.leaf_ids),
        "candidate_node_count": len(tree.nodes),
        "candidate_tree_hash": tree.assignment_hash,
        "prefix_assignment_hash": prefix_assignment_hash,
        "formal_assignment_hash": formal_assignment_hash,
        "prefix_assignment_match": 1,
        "candidate_build_seconds": candidate_seconds,
        "target_num_balls": target_num_balls,
        "candidate_capacity_ok": int(len(tree.leaf_ids) >= target_num_balls),
    }
    return rows, candidate_row, partition_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict], gate: dict) -> str:
    lines = [
        "# MLP-48 CF-BLAGB Phase A report",
        "",
        "## Protocol",
        "",
        "- Cora, CiteSeer, CoraFull; uniform 0.4; seed 0; train-only noise.",
        "- Candidate tree is label-free; final budget equals the formal K_geo.",
        "- Five-fold cross-fitting excludes each held-out fold from guidance and votes.",
        "- Clean guidance is diagnostic only and cannot pass the method by itself.",
        "",
        "## Target quality",
        "",
        "| Dataset | Method | Acc | Brier | NCR | CDR | Balls | Exact | JSD |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], item["method"])):
        balls = row.get("median_num_balls", "")
        jsd = row.get("mean_selected_jsd", "")
        lines.append(
            "| {dataset} | {method} | {target_accuracy:.4f} | {brier:.4f} | "
            "{ncr:.4f} | {cdr:.4f} | {balls} | {exact} | {jsd} |".format(
                **row,
                balls=balls,
                exact=row.get("exact_budget", ""),
                jsd=(f"{float(jsd):.4f}" if jsd != "" else ""),
            )
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"- Pass: **{gate.get('pass', False)}**",
            f"- Reason: `{gate.get('reason', 'unknown')}`",
            f"- Exact-budget datasets: {gate.get('exact_budget_dataset_count', 0)}/3",
            f"- Brier better than both controls: {gate.get('brier_both_pass_count', 0)}/3",
            f"- Clean-oracle accuracy headroom: {gate.get('oracle_headroom_pass_count', 0)}/3",
            f"- Mean accuracy delta vs cross-fit formal: {gate.get('mean_accuracy_delta_vs_crossfit_formal', float('nan')):+.4f}",
            "",
            "Phase B is allowed only when every preregistered gate passes. No score, prior, fold count, or budget may be changed after this report.",
        ]
    )
    by_dataset = {
        dataset: {
            row["method"]: row for row in rows if row["dataset"] == dataset
        }
        for dataset in EXPECTED_DATASETS
    }
    lines.extend(
        [
            "",
            "## Detailed conclusion",
            "",
            "### Hypothesis and granular-ball role",
            "",
            "CF-BLAGB tested whether cross-fitted noisy labels could improve where a fixed granular-ball budget is spent. Labels only re-ranked label-free geometric splits; they could neither veto a split nor change the final K_geo. All other LOO-GB-LCSS target and student settings remained frozen.",
            "",
            "### What the diagnostics show",
            "",
        ]
    )
    for dataset in EXPECTED_DATASETS:
        methods = by_dataset[dataset]
        candidate = methods["cf_blagb"]
        geometry = methods["fixed_budget_geometry_only"]
        shuffled = methods["cf_blagb_shuffled_guidance"]
        oracle = methods["cf_blagb_clean_oracle"]
        knn = methods["crossfit_adaptive_knn"]
        capacity = int(candidate["candidate_leaf_count"])
        budget = int(candidate["target_num_balls"])
        lines.append(
            f"- **{dataset}:** candidate capacity {capacity}, K_geo {budget}. "
            f"CF-BLAGB changed target accuracy by "
            f"{float(candidate['target_accuracy']) - float(geometry['target_accuracy']):+.4f} "
            f"and Brier by {float(candidate['brier']) - float(geometry['brier']):+.4f} "
            f"versus geometry-only; its Brier delta versus shuffled was "
            f"{float(candidate['brier']) - float(shuffled['brier']):+.4f}. "
            f"The clean oracle changed Brier by "
            f"{float(oracle['brier']) - float(geometry['brier']):+.4f}, while "
            f"cross-fit adaptive-k changed it by "
            f"{float(knn['brier']) - float(geometry['brier']):+.4f}."
        )
    lines.extend(
        [
            "",
            "Cora offered no alternative cut because candidate capacity equalled K_geo, so every fixed-budget arm reproduced the same partition. CiteSeer offered only one additional leaf and noisy guidance selected the geometry-only cut. CoraFull offered substantial re-ranking capacity, but real noisy and shuffled guidance produced the same target accuracy and Brier; only clean guidance selected a measurably better cut. Thus fixed budgeting repaired CF-NAGB's collapse without creating a reliable noisy-label advantage.",
            "",
            "### Supported conclusion",
            "",
            "The preregistered Phase A gate failed: exact budgeting passed on 3/3 datasets, but CF-BLAGB improved Brier over both geometry-only and shuffled controls on 0/3 datasets, and clean-oracle accuracy headroom appeared on only 1/3. Phase B is not triggered. Directly using the available noisy labels to alter granular-ball construction is closed under this frozen formulation.",
            "",
            "### Interpretation boundary",
            "",
            "The experiment does not show that labels can never help partitioning. It shows that, with 20% noisy-label coverage, a bounded JSD re-ranking signal adds no stable information beyond the frozen geometry under the current fixed budget. Teacher-denoised guidance would change the information source and belongs to a separate method rather than a second CF-BLAGB tuning round.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLP-48 CF-BLAGB experiments")
    parser.add_argument("--phase", choices=("diagnostics",), default="diagnostics")
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir", default="runs/mlp48_cf_blagb_20260805"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = tuple(
        item.strip().lower() for item in args.datasets.split(",") if item.strip()
    )
    if datasets != EXPECTED_DATASETS:
        raise ValueError(
            f"Phase A datasets are frozen as {EXPECTED_DATASETS}, received {datasets}"
        )
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    all_rows = []
    candidate_rows = []
    partition_rows = []
    for dataset in datasets:
        rows, candidate, partitions = run_phase_a_dataset(
            dataset,
            seed=args.seed,
            device=args.device,
            noise_type=args.noise_type,
            noise_rate=args.noise_rate,
        )
        all_rows.extend(rows)
        candidate_rows.append(candidate)
        partition_rows.extend(partitions)
        print(f"[MLP-48 Phase A] completed {dataset}: {len(rows)} methods", flush=True)

    gate = evaluate_phase_a_gate(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "target_quality.csv", all_rows)
    _write_csv(output_dir / "candidate_tree_diagnostics.csv", candidate_rows)
    _write_csv(output_dir / "crossfit_partition_diagnostics.csv", partition_rows)
    _write_csv(
        output_dir / "protocol_audit.csv",
        [
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "split_hash": row["split_hash"],
                "noise_hash": row["noise_hash"],
                "modified_count": row["modified_count"],
                "modified_train_count": row["modified_train_count"],
                "modified_val_count": row["modified_val_count"],
                "modified_test_count": row["modified_test_count"],
                "noise_scope": row["noise_scope"],
                "status": "ok",
            }
            for row in candidate_rows
        ],
    )
    _write_csv(
        output_dir / "completeness.csv",
        [
            {
                "dataset": dataset,
                "expected_methods": len(PHASE_A_METHODS),
                "observed_methods": len(
                    {row["method"] for row in all_rows if row["dataset"] == dataset}
                ),
                "expected_budgeted_folds": 5 * len(BUDGETED_METHODS),
                "observed_budgeted_folds": len(
                    [row for row in partition_rows if row["dataset"] == dataset]
                ),
                "status": (
                    "complete"
                    if len(
                        {row["method"] for row in all_rows if row["dataset"] == dataset}
                    )
                    == len(PHASE_A_METHODS)
                    and len(
                        [row for row in partition_rows if row["dataset"] == dataset]
                    )
                    == 5 * len(BUDGETED_METHODS)
                    else "incomplete"
                ),
            }
            for dataset in datasets
        ],
    )
    (output_dir / "command_manifest.json").write_text(
        json.dumps(
            {
                "phase": args.phase,
                "datasets": datasets,
                "seed": args.seed,
                "noise_type": args.noise_type,
                "noise_rate": args.noise_rate,
                "noise_scope": "train",
                "split": [0.2, 0.1, 0.7],
                "device": args.device,
                "python": sys.executable,
                "candidate_tree": {
                    "builder": "formal_prefix_then_zero_gain_extension",
                    "prefix_max_leaves": 64,
                    "prefix_split_improvement": 0.05,
                    "max_leaves": 128,
                    "min_ball_size": 32,
                    "extension_min_geometry_gain": 0.0,
                },
                "crossfit": {"folds": 5, "prior_strength": 2.0},
                "label_score": "G * (1 + JSD/log(2))",
                "target": {
                    "beta": 0.5,
                    "leave_one_out": True,
                    "tie_mode": "distribution_only",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "phase_a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "mlp48_report.md").write_text(
        _report(all_rows, gate), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
