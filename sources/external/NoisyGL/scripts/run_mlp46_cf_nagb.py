from __future__ import annotations

import argparse
import csv
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
    build_crossfit_targets,
    build_crossfit_targets_from_assignment,
    build_geometry_candidate_tree,
    make_stratified_crossfit_folds,
)
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


METHOD_CONFIG = "frozengrace_gb_mlp_softdist_beta05_loo"
PHASE_A_METHODS = (
    "formal_loo_gb",
    "crossfit_formal_gb",
    "exact_knn20_loo",
    "crossfit_knn20",
    "crossfit_adaptive_knn",
    "fine_tree_crossfit",
    "cf_nagb_shuffled_guidance",
    "cf_nagb",
    "cf_nagb_clean_oracle",
)


def _as_numpy(values, dtype=None) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def _mask_indices(mask) -> np.ndarray:
    values = _as_numpy(mask)
    if values.dtype == np.bool_:
        return np.flatnonzero(values).astype(np.int64)
    return values.astype(np.int64).reshape(-1)


def target_quality_metrics(
    soft_targets,
    noisy_labels,
    clean_labels,
    train_indices,
) -> dict:
    targets = np.asarray(soft_targets, dtype=np.float64)
    noisy = np.asarray(noisy_labels, dtype=np.int64)
    clean = np.asarray(clean_labels, dtype=np.int64)
    train = np.asarray(train_indices, dtype=np.int64)
    if targets.ndim != 2 or targets.shape[0] != noisy.shape[0]:
        raise ValueError("soft_targets must align with labels")
    predicted = targets.argmax(axis=1)
    n_classes = int(targets.shape[1])
    clean_one_hot = np.zeros((train.size, n_classes), dtype=np.float64)
    clean_one_hot[np.arange(train.size), clean[train]] = 1.0
    brier = float(np.mean(np.sum((targets[train] - clean_one_hot) ** 2, axis=1)))
    noisy_flag = noisy[train] != clean[train]
    clean_flag = ~noisy_flag
    changed = predicted[train] != noisy[train]
    ncr = (
        float(np.mean(changed[noisy_flag] & (predicted[train][noisy_flag] == clean[train][noisy_flag])))
        if np.any(noisy_flag)
        else 0.0
    )
    cdr = (
        float(np.mean(changed[clean_flag] & (predicted[train][clean_flag] != clean[train][clean_flag])))
        if np.any(clean_flag)
        else 0.0
    )
    return {
        "target_accuracy": float(np.mean(predicted[train] == clean[train])),
        "brier": brier,
        "ncr": ncr,
        "cdr": cdr,
    }


def _is_improved(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["target_accuracy"] > baseline["target_accuracy"] + 1e-12
        or candidate["ncr"] > baseline["ncr"] + 1e-12
        or candidate["brier"] < baseline["brier"] - 1e-12
        or candidate["cdr"] < baseline["cdr"] - 1e-12
    )


def _has_large_adverse_change(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["target_accuracy"] < baseline["target_accuracy"] - 0.01
        or candidate["ncr"] < baseline["ncr"] - 0.01
        or candidate["brier"] > baseline["brier"] + 0.01
        or candidate["cdr"] > baseline["cdr"] + 0.01
    )


def evaluate_phase_a_gate(rows: list[dict]) -> dict:
    by_dataset: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), {})[str(row["method"])] = row
    required = {
        "crossfit_formal_gb",
        "cf_nagb_shuffled_guidance",
        "cf_nagb",
        "cf_nagb_clean_oracle",
    }
    missing = {
        dataset: sorted(required - set(methods))
        for dataset, methods in by_dataset.items()
        if required - set(methods)
    }
    if missing:
        return {"pass": False, "reason": "incomplete", "missing": missing}

    noisy_vs_formal = 0
    noisy_vs_shuffled = 0
    oracle_headroom = 0
    geometry_ok = 0
    mean_deltas = {metric: [] for metric in ("target_accuracy", "brier", "ncr", "cdr")}
    for methods in by_dataset.values():
        noisy = methods["cf_nagb"]
        formal = methods["crossfit_formal_gb"]
        shuffled = methods["cf_nagb_shuffled_guidance"]
        oracle = methods["cf_nagb_clean_oracle"]
        if _is_improved(noisy, formal) and not _has_large_adverse_change(noisy, formal):
            noisy_vs_formal += 1
        if _is_improved(noisy, shuffled) and not _has_large_adverse_change(noisy, shuffled):
            noisy_vs_shuffled += 1
        if _is_improved(oracle, formal) and not _has_large_adverse_change(oracle, formal):
            oracle_headroom += 1
        if str(noisy.get("geometry_status", "")) == "ok":
            geometry_ok += 1
        for metric in mean_deltas:
            mean_deltas[metric].append(float(noisy[metric]) - float(formal[metric]))

    dataset_count = len(by_dataset)
    needed = math.ceil(2 * dataset_count / 3)
    mean_direction_ok = bool(
        np.mean(mean_deltas["target_accuracy"]) >= -1e-12
        and np.mean(mean_deltas["ncr"]) >= -1e-12
        and np.mean(mean_deltas["brier"]) <= 1e-12
        and np.mean(mean_deltas["cdr"]) <= 1e-12
    )
    if geometry_ok != dataset_count:
        reason = "geometry_failed"
    elif oracle_headroom < needed:
        reason = "oracle_no_headroom"
    elif noisy_vs_formal < needed or noisy_vs_shuffled < needed:
        reason = "noisy_guidance_failed"
    elif not mean_direction_ok:
        reason = "mean_direction_failed"
    else:
        reason = "pass"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "dataset_count": dataset_count,
        "required_pass_count": needed,
        "geometry_ok_count": geometry_ok,
        "noisy_vs_formal_pass_count": noisy_vs_formal,
        "noisy_vs_shuffled_pass_count": noisy_vs_shuffled,
        "oracle_headroom_pass_count": oracle_headroom,
        "mean_direction_ok": mean_direction_ok,
        **{f"mean_delta_{key}": float(np.mean(value)) for key, value in mean_deltas.items()},
    }


def _blend_distributions(
    distributions: np.ndarray,
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    n_classes: int,
    target_beta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.eye(n_classes, dtype=np.float64)[noisy_labels]
    alpha = np.zeros(noisy_labels.shape[0], dtype=np.float64)
    for row, node in enumerate(train_indices):
        distribution = distributions[row]
        if distribution.sum() <= 0:
            continue
        distribution = distribution / distribution.sum()
        margin = float(distribution.max())
        baseline = 1.0 / n_classes
        node_alpha = float(np.clip((margin - baseline) / (1.0 - baseline), 0.0, 1.0))
        alpha[node] = node_alpha
        maxima = np.flatnonzero(distribution == distribution.max())
        if maxima.size > 1:
            local_target = distribution
        else:
            one_hot = np.zeros(n_classes, dtype=np.float64)
            one_hot[int(maxima[0])] = 1.0
            local_target = (1.0 - target_beta) * distribution + target_beta * one_hot
        targets[node] = (1.0 - node_alpha) * targets[node] + node_alpha * local_target
    return targets, alpha


def _exact_knn_targets(
    embeddings: np.ndarray,
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    n_classes: int,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_features = embeddings[train_indices]
    similarity = train_features @ train_features.T
    np.fill_diagonal(similarity, -np.inf)
    actual_k = min(int(k), max(0, train_indices.size - 1))
    distributions = np.zeros((train_indices.size, n_classes), dtype=np.float64)
    if actual_k:
        neighbors = np.argpartition(-similarity, kth=actual_k - 1, axis=1)[:, :actual_k]
        for row, positions in enumerate(neighbors):
            counts = np.bincount(noisy_labels[train_indices[positions]], minlength=n_classes)
            distributions[row] = counts / counts.sum()
    targets, alpha = _blend_distributions(
        distributions, noisy_labels, train_indices, n_classes
    )
    return targets, alpha, {"matched_k": actual_k, "exclude_self": 1}


def _crossfit_knn_targets(
    embeddings: np.ndarray,
    noisy_labels: np.ndarray,
    train_indices: np.ndarray,
    fold_ids: np.ndarray,
    n_classes: int,
    adaptive: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    candidate_ks = (5, 10, 20, 40) if adaptive else (20,)
    distributions = np.zeros((train_indices.size, n_classes), dtype=np.float64)
    row_for_node = {int(node): row for row, node in enumerate(train_indices)}
    selected_ks = []
    for fold_id in sorted(np.unique(fold_ids[train_indices]).tolist()):
        heldout = train_indices[fold_ids[train_indices] == fold_id]
        construction = train_indices[fold_ids[train_indices] != fold_id]
        similarity = embeddings[heldout] @ embeddings[construction].T
        order = np.argsort(-similarity, axis=1)
        for heldout_row, node in enumerate(heldout):
            best_score = -np.inf
            best_k = 0
            best_distribution = None
            for requested_k in candidate_ks:
                actual_k = min(int(requested_k), construction.size)
                positions = order[heldout_row, :actual_k]
                counts = np.bincount(
                    noisy_labels[construction[positions]], minlength=n_classes
                ).astype(np.float64)
                distribution = counts / counts.sum()
                q_max = float(distribution.max())
                score = q_max - math.sqrt(q_max * (1.0 - q_max) / (actual_k + 2.0))
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12 and actual_k > best_k
                ):
                    best_score = score
                    best_k = actual_k
                    best_distribution = distribution
            distributions[row_for_node[int(node)]] = best_distribution
            selected_ks.append(best_k)
    targets, alpha = _blend_distributions(
        distributions, noisy_labels, train_indices, n_classes
    )
    return targets, alpha, {
        "adaptive_k": int(adaptive),
        "mean_selected_k": float(np.mean(selected_ks)),
        "min_selected_k": int(np.min(selected_ks)),
        "max_selected_k": int(np.max(selected_ks)),
    }


def _candidate_leaf_assignment(tree) -> np.ndarray:
    assignment = np.empty(tree.n_nodes, dtype=np.int64)
    for ball_id, node_id in enumerate(tree.leaf_ids):
        assignment[tree.nodes[node_id].indices] = ball_id
    return assignment


def _result_row(
    dataset: str,
    method: str,
    result: CrossFitTargetResult | None,
    soft_targets: np.ndarray,
    alpha: np.ndarray,
    noisy: np.ndarray,
    clean: np.ndarray,
    train: np.ndarray,
    common: dict,
    diagnostics: dict | None = None,
) -> dict:
    metrics = target_quality_metrics(soft_targets, noisy, clean, train)
    combined = dict(diagnostics or {})
    if result is not None:
        combined.update(result.diagnostics)
    median_balls = combined.get("median_num_balls", "")
    alpha_coverage = float(np.mean(alpha[train] > 0.0))
    full_tree_ratio = float(combined.get("full_tree_fold_ratio", 0.0) or 0.0)
    if method == "cf_nagb":
        if median_balls == "" or float(median_balls) < 8:
            geometry_status = "collapsed"
        elif float(median_balls) > 96:
            geometry_status = "too_many_balls"
        elif full_tree_ratio >= 1.0:
            geometry_status = "full_tree_unpruned"
        elif alpha_coverage < 0.8:
            geometry_status = "low_alpha_coverage"
        else:
            geometry_status = "ok"
    else:
        geometry_status = "not_applicable"
    return {
        **common,
        "dataset": dataset,
        "method": method,
        **metrics,
        "alpha_coverage": alpha_coverage,
        "mean_alpha": float(np.mean(alpha[train])),
        "median_num_balls": median_balls,
        "min_num_balls": combined.get("min_num_balls", ""),
        "max_num_balls": combined.get("max_num_balls", ""),
        "candidate_leaf_count": combined.get("candidate_leaf_count", ""),
        "candidate_tree_hash": combined.get("candidate_tree_hash", ""),
        "mean_accepted_splits": combined.get("mean_accepted_splits", ""),
        "mean_rejected_splits": combined.get("mean_rejected_splits", ""),
        "full_tree_fold_ratio": combined.get("full_tree_fold_ratio", ""),
        "geometry_status": geometry_status,
        "diagnostics_json": json.dumps(combined, sort_keys=True),
    }


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
    if np.any(noisy[val] != clean[val]) or np.any(noisy[test] != clean[test]):
        raise RuntimeError("validation/test labels were modified")

    conf = load_conf(None, METHOD_CONFIG, dataset_name)
    embedding_path = Path(conf.gb["embedding_path"])
    if not embedding_path.is_absolute():
        embedding_path = (REPO_ROOT / embedding_path).resolve()
    embeddings_tensor = load_frozen_embeddings(
        embedding_path, expected_nodes=int(clean.shape[0])
    )
    embeddings = _as_numpy(embeddings_tensor, np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    folds = make_stratified_crossfit_folds(
        train,
        noisy,
        n_splits=5,
        random_state=seed,
        n_nodes=clean.shape[0],
    )

    candidate_started = time.perf_counter()
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=128,
        min_ball_size=32,
        min_geometry_gain=0.01,
        random_state=seed,
    )
    candidate_seconds = time.perf_counter() - candidate_started
    formal = build_ordinary_granular_balls(
        embeddings,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=seed,
    )

    common = {
        "seed": seed,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "train_percent": 0.2,
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
        "modified_count": int(_as_numpy(modified).size),
        "train_count": int(train.size),
        "n_nodes": int(clean.shape[0]),
        "n_classes": int(data.n_classes),
        "embedding_path": str(embedding_path),
    }

    rows: list[dict] = []
    partition_rows: list[dict] = []

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
            None,
            _as_numpy(formal_soft, np.float64),
            _as_numpy(formal_alpha, np.float64),
            noisy,
            clean,
            train,
            common,
            diagnostics={
                "median_num_balls": int(formal.summary["num_balls"]),
                "min_num_balls": int(formal.summary["num_balls"]),
                "max_num_balls": int(formal.summary["num_balls"]),
            },
        )
    )

    crossfit_formal = build_crossfit_targets_from_assignment(
        formal.assignment, noisy, train, folds, int(data.n_classes)
    )
    rows.append(
        _result_row(
            dataset_name,
            "crossfit_formal_gb",
            crossfit_formal,
            crossfit_formal.soft_targets,
            crossfit_formal.alpha,
            noisy,
            clean,
            train,
            common,
        )
    )

    exact_targets, exact_alpha, exact_diag = _exact_knn_targets(
        embeddings, noisy, train, int(data.n_classes), k=20
    )
    rows.append(
        _result_row(
            dataset_name,
            "exact_knn20_loo",
            None,
            exact_targets,
            exact_alpha,
            noisy,
            clean,
            train,
            common,
            diagnostics=exact_diag,
        )
    )
    for adaptive, method in (
        (False, "crossfit_knn20"),
        (True, "crossfit_adaptive_knn"),
    ):
        targets, method_alpha, method_diag = _crossfit_knn_targets(
            embeddings, noisy, train, folds, int(data.n_classes), adaptive
        )
        rows.append(
            _result_row(
                dataset_name,
                method,
                None,
                targets,
                method_alpha,
                noisy,
                clean,
                train,
                common,
                diagnostics=method_diag,
            )
        )

    fine_tree = build_crossfit_targets_from_assignment(
        _candidate_leaf_assignment(tree), noisy, train, folds, int(data.n_classes)
    )
    rows.append(
        _result_row(
            dataset_name,
            "fine_tree_crossfit",
            fine_tree,
            fine_tree.soft_targets,
            fine_tree.alpha,
            noisy,
            clean,
            train,
            common,
            diagnostics={
                **fine_tree.diagnostics,
                "candidate_leaf_count": len(tree.leaf_ids),
                "candidate_tree_hash": tree.assignment_hash,
            },
        )
    )

    for guidance_mode, method in (
        ("shuffled", "cf_nagb_shuffled_guidance"),
        ("noisy", "cf_nagb"),
        ("clean", "cf_nagb_clean_oracle"),
    ):
        result = build_crossfit_targets(
            tree=tree,
            noisy_labels=noisy,
            clean_labels=clean if guidance_mode == "clean" else None,
            train_indices=train,
            fold_ids=folds,
            n_classes=int(data.n_classes),
            guidance_mode=guidance_mode,
            random_state=seed,
        )
        rows.append(
            _result_row(
                dataset_name,
                method,
                result,
                result.soft_targets,
                result.alpha,
                noisy,
                clean,
                train,
                common,
            )
        )
        for fold_position, num_balls in enumerate(result.fold_num_balls):
            partition_rows.append(
                {
                    **common,
                    "dataset": dataset_name,
                    "method": method,
                    "fold": fold_position,
                    "num_balls": int(num_balls),
                    "candidate_leaf_count": len(tree.leaf_ids),
                    "candidate_tree_hash": tree.assignment_hash,
                }
            )

    candidate_row = {
        **common,
        "dataset": dataset_name,
        "candidate_leaf_count": len(tree.leaf_ids),
        "candidate_node_count": len(tree.nodes),
        "candidate_tree_hash": tree.assignment_hash,
        "candidate_build_seconds": candidate_seconds,
        "formal_num_balls": int(formal.summary["num_balls"]),
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: list[dict], gate: dict) -> str:
    lines = [
        "# MLP-46 CF-NAGB Phase A report",
        "",
        "## Protocol",
        "",
        "- Cora, CiteSeer, CoraFull; uniform 0.4; seed 0; train-only noise.",
        "- Candidate tree: 128 leaves, min ball size 32, geometry gain 0.01.",
        "- Five-fold cross-fitting; held-out labels never prune or vote for their own local target.",
        "- Clean labels are used only by diagnostic metrics and the preregistered oracle.",
        "",
        "## Target quality",
        "",
        "| Dataset | Method | Acc | Brier | NCR | CDR | Alpha coverage | Median balls | Geometry |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        median = row.get("median_num_balls", "")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {float(row['target_accuracy']):.4f} | "
            f"{float(row['brier']):.4f} | {float(row['ncr']):.4f} | {float(row['cdr']):.4f} | "
            f"{float(row['alpha_coverage']):.4f} | {median} | {row['geometry_status']} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"- Pass: **{gate.get('pass', False)}**",
            f"- Reason: `{gate.get('reason', 'unknown')}`",
            f"- Noisy guidance vs formal: {gate.get('noisy_vs_formal_pass_count', 0)}/{gate.get('dataset_count', 0)}",
            f"- Noisy guidance vs shuffled: {gate.get('noisy_vs_shuffled_pass_count', 0)}/{gate.get('dataset_count', 0)}",
            f"- Clean oracle headroom: {gate.get('oracle_headroom_pass_count', 0)}/{gate.get('dataset_count', 0)}",
            "",
            "Phase B may run only when all Phase A gates pass. No thresholds were changed after observing these diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MLP-46 CF-NAGB experiments")
    parser.add_argument("--phase", choices=("diagnostics",), default="diagnostics")
    parser.add_argument("--datasets", default="cora,citeseer,corafull")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default="runs/mlp46_cf_nagb_20260805",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
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
        print(f"[MLP-46 Phase A] completed {dataset}: {len(rows)} target builders")

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
                "train_count": row["train_count"],
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
                "phase": "A",
                "expected_methods": len(PHASE_A_METHODS),
                "observed_methods": len(
                    {row["method"] for row in all_rows if row["dataset"] == dataset}
                ),
                "status": (
                    "complete"
                    if len({row["method"] for row in all_rows if row["dataset"] == dataset})
                    == len(PHASE_A_METHODS)
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
                    "max_leaves": 128,
                    "min_ball_size": 32,
                    "min_geometry_gain": 0.01,
                },
                "crossfit": {
                    "folds": 5,
                    "min_guidance_per_child": 5,
                    "log_bf_threshold": math.log(3.0),
                    "prior_strength": 2.0,
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
    (output_dir / "mlp46_report.md").write_text(
        _report(all_rows, gate), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
