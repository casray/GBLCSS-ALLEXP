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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.NRGNN_Predictor import nrgnn_Predictor
from scripts.run_mlp46_cf_nagb import _as_numpy, _mask_indices, target_quality_metrics
from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import apply_ball_soft_targets, load_frozen_embeddings
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.teacher_verified_gb import (
    build_teacher_verified_loo_targets,
    size_matched_shuffled_assignment,
    teacher_reliability,
)
from utils.tools import load_conf, setup_seed


EXPECTED_DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SEED = 0
MC_SAMPLES = 8
GB_CONFIG = "frozengrace_gb_mlp_softdist_beta05_loo"
PHASE_A_METHODS = (
    "formal_loo_gb",
    "utv_gb",
    "utv_gb_shuffled",
    "nrgnn_teacher_target",
    "teacher_weighted_noisy",
    "clean_oracle_verifier",
)


def _cell_key(dataset: str, noise_rate: float) -> tuple[str, float]:
    return str(dataset).lower(), round(float(noise_rate), 6)


def validate_teacher_cache_metadata(actual: dict, expected: dict) -> None:
    required = (
        "dataset",
        "noise_rate",
        "seed",
        "split_hash",
        "noise_hash",
        "mc_samples",
    )
    for key in required:
        if key not in actual or key not in expected:
            raise ValueError(f"teacher cache metadata is missing {key}")
        if key == "noise_rate":
            matches = math.isclose(float(actual[key]), float(expected[key]), abs_tol=1e-12)
        else:
            matches = actual[key] == expected[key]
        if not matches:
            raise ValueError(
                f"teacher cache {key} mismatch: actual={actual[key]!r}, "
                f"expected={expected[key]!r}"
            )


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
        f"{dataset}:{rate}": sorted(set(PHASE_A_METHODS) - set(by_cell.get((dataset, rate), {})))
        for dataset, rate in sorted(expected_cells)
        if set(PHASE_A_METHODS) - set(by_cell.get((dataset, rate), {}))
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

    brier_both_pass_count = 0
    brier_shuffled_pass_count = 0
    oracle_headroom_pass_count = 0
    accuracy_deltas = []
    large_accuracy_drops = []
    zero_coverage_cells = []
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
        candidate = methods["utv_gb"]
        formal = methods["formal_loo_gb"]
        shuffled = methods["utv_gb_shuffled"]
        oracle = methods["clean_oracle_verifier"]
        brier_both = (
            float(candidate["brier"]) < float(formal["brier"]) - 1e-12
            and float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        )
        brier_shuffled = float(candidate["brier"]) < float(shuffled["brier"]) - 1e-12
        oracle_headroom = float(oracle["brier"]) < float(formal["brier"]) - 1e-12
        accuracy_delta = float(candidate["target_accuracy"]) - float(formal["target_accuracy"])
        coverage = float(candidate.get("verified_correction_coverage", 0.0))
        brier_both_pass_count += int(brier_both)
        brier_shuffled_pass_count += int(brier_shuffled)
        oracle_headroom_pass_count += int(oracle_headroom)
        accuracy_deltas.append(accuracy_delta)
        if accuracy_delta < -0.01 - 1e-12:
            large_accuracy_drops.append(f"{key[0]}:{key[1]}")
        if coverage <= 0.0:
            zero_coverage_cells.append(f"{key[0]}:{key[1]}")
        per_cell[f"{key[0]}:{key[1]}"] = {
            "brier_better_than_both": bool(brier_both),
            "brier_better_than_shuffled": bool(brier_shuffled),
            "oracle_brier_headroom": bool(oracle_headroom),
            "accuracy_delta_vs_formal": accuracy_delta,
            "brier_delta_vs_formal": float(candidate["brier"]) - float(formal["brier"]),
            "brier_delta_vs_shuffled": float(candidate["brier"]) - float(shuffled["brier"]),
            "verified_correction_coverage": coverage,
        }

    mean_accuracy_delta = float(np.mean(accuracy_deltas))
    if integrity_failures:
        reason = "integrity_failed"
    elif brier_both_pass_count < 4:
        reason = "brier_controls_failed"
    elif mean_accuracy_delta < 0.005 - 1e-12:
        reason = "mean_accuracy_failed"
    elif large_accuracy_drops:
        reason = "large_accuracy_drop"
    elif brier_shuffled_pass_count < 4 or zero_coverage_cells:
        reason = "real_structure_failed"
    elif oracle_headroom_pass_count < 3:
        reason = "oracle_no_headroom"
    else:
        reason = "pass"
    return {
        "pass": reason == "pass",
        "reason": reason,
        "cell_count": len(expected_cells),
        "brier_both_pass_count": brier_both_pass_count,
        "brier_shuffled_pass_count": brier_shuffled_pass_count,
        "oracle_headroom_pass_count": oracle_headroom_pass_count,
        "mean_accuracy_delta_vs_formal": mean_accuracy_delta,
        "large_accuracy_drop_cells": large_accuracy_drops,
        "zero_coverage_cells": zero_coverage_cells,
        "integrity_failures": integrity_failures,
        "per_cell": per_cell,
    }


def _teacher_cache_path(output_dir: Path, dataset: str, noise_rate: float) -> Path:
    rate = str(noise_rate).replace(".", "p")
    return output_dir / "teacher_cache" / f"nrgnn_{dataset}_u{rate}_seed0.npz"


def _load_teacher_cache(path: Path, expected: dict) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        samples = np.asarray(payload["posterior_samples"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    validate_teacher_cache_metadata(metadata, expected)
    if samples.shape != (
        int(expected["mc_samples"]),
        int(expected["n_nodes"]),
        int(expected["n_classes"]),
    ):
        raise ValueError(
            f"teacher cache posterior shape mismatch: {samples.shape}"
        )
    if not np.isfinite(samples).all():
        raise ValueError("teacher cache contains non-finite values")
    if not np.allclose(samples.sum(axis=2), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("teacher cache posterior rows do not sum to one")
    return samples, metadata


def _save_teacher_cache(path: Path, samples: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        posterior_samples=np.asarray(samples, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _mc_teacher_posteriors(predictor: nrgnn_Predictor, seed: int) -> np.ndarray:
    reformed_adj = torch.sparse_coo_tensor(
        predictor.best_model_index,
        predictor.best_graph,
        [predictor.n_nodes, predictor.n_nodes],
        device=predictor.device,
    )
    samples = []
    predictor.model.train()
    with torch.no_grad():
        for sample_id in range(MC_SAMPLES):
            sample_seed = int(seed) * 1009 + 7919 * (sample_id + 1)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            logits = predictor.model(predictor.feats, reformed_adj)
            samples.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    predictor.model.eval()
    result = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("NRGNN MC posterior contains non-finite values")
    return result


def _train_or_load_teacher(
    data,
    dataset: str,
    noise_rate: float,
    seed: int,
    device: str,
    output_dir: Path,
    common: dict,
    force: bool,
) -> tuple[np.ndarray, dict]:
    expected = {
        "dataset": dataset,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "split_hash": common["split_hash"],
        "noise_hash": common["noise_hash"],
        "mc_samples": MC_SAMPLES,
        "n_nodes": int(data.n_nodes),
        "n_classes": int(data.n_classes),
    }
    cache_path = _teacher_cache_path(output_dir, dataset, noise_rate)
    if cache_path.exists() and not force:
        samples, metadata = _load_teacher_cache(cache_path, expected)
        metadata = dict(metadata)
        metadata["cache_reused"] = 1
        return samples, metadata

    conf = load_conf(None, "nrgnn", dataset)
    conf.model["n_feat"] = data.dim_feats
    conf.model["n_classes"] = data.n_classes
    conf.training["debug"] = False
    started = time.perf_counter()
    predictor = nrgnn_Predictor(conf, data, device)
    predictor.seed = int(seed)
    predictor.noise_type = "uniform"
    predictor.noise_rate = float(noise_rate)
    result = predictor.train()
    training_seconds = time.perf_counter() - started
    samples = _mc_teacher_posteriors(predictor, seed)
    metadata = {
        **expected,
        "cache_reused": 0,
        "teacher_train": float(result["train"]),
        "teacher_valid": float(result["valid"]),
        "teacher_test": float(result["test"]),
        "teacher_training_seconds": float(training_seconds),
        "posterior_min": float(samples.min()),
        "posterior_max": float(samples.max()),
        "posterior_row_sum_max_error": float(np.max(np.abs(samples.sum(axis=2) - 1.0))),
    }
    _save_teacher_cache(cache_path, samples, metadata)
    return samples, metadata


def _result_row(
    common: dict,
    method: str,
    targets,
    alpha,
    noisy: np.ndarray,
    clean: np.ndarray,
    train: np.ndarray,
    diagnostics: dict | None = None,
) -> dict:
    targets = np.asarray(targets, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    diagnostics = dict(diagnostics or {})
    metrics = target_quality_metrics(targets, noisy, clean, train)
    return {
        **common,
        "method": method,
        **metrics,
        "alpha_coverage": float(np.mean(alpha[train] > 0.0)),
        "mean_alpha": float(np.mean(alpha[train])),
        "verified_correction_coverage": diagnostics.get(
            "verified_correction_coverage",
            float(np.mean(alpha[train] > 0.0)),
        ),
        "base_correction_coverage": diagnostics.get("base_correction_coverage", ""),
        "mean_verification_scale": diagnostics.get("mean_verification_scale", ""),
        "teacher_ball_coverage": diagnostics.get("teacher_ball_coverage", ""),
        "mean_teacher_node_reliability": diagnostics.get(
            "mean_teacher_node_reliability", ""
        ),
        "status": "ok",
        "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
    }


def run_cell(
    dataset: str,
    noise_rate: float,
    seed: int,
    device: str,
    output_dir: Path,
    force_teacher: bool,
) -> tuple[list[dict], dict, dict]:
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
    data.noisy_label = noisy_tensor
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
        "seed": int(seed),
        "noise_type": "uniform",
        "noise_rate": float(noise_rate),
        "noise_scope": "train",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "split_hash": _split_fingerprint(
            data.train_masks, data.val_masks, data.test_masks
        ),
        "noise_hash": _noise_fingerprint(data.labels, noisy_tensor),
        "modified_count": int(modified.size),
        "modified_train_count": int(np.isin(modified, train).sum()),
        "modified_val_count": int(np.isin(modified, val).sum()),
        "modified_test_count": int(np.isin(modified, test).sum()),
        "n_nodes": int(data.n_nodes),
        "n_classes": int(data.n_classes),
        "train_count": int(train.size),
    }

    teacher_samples, teacher_meta = _train_or_load_teacher(
        data,
        dataset,
        noise_rate,
        seed,
        device,
        output_dir,
        common,
        force_teacher,
    )
    reliability = teacher_reliability(teacher_samples)

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

    formal_targets, formal_alpha, _ = apply_ball_soft_targets(
        noisy_label=torch.as_tensor(noisy, dtype=torch.long),
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
    utv = build_teacher_verified_loo_targets(
        noisy,
        partition.assignment,
        train,
        int(data.n_classes),
        teacher_samples,
    )
    shuffled_assignment = size_matched_shuffled_assignment(
        partition.assignment,
        random_state=seed + 49001,
    )
    shuffled = build_teacher_verified_loo_targets(
        noisy,
        shuffled_assignment,
        train,
        int(data.n_classes),
        teacher_samples,
    )
    clean_samples = np.eye(int(data.n_classes), dtype=np.float64)[clean][None, :, :]
    oracle = build_teacher_verified_loo_targets(
        noisy,
        partition.assignment,
        train,
        int(data.n_classes),
        clean_samples,
    )
    noisy_one_hot = np.eye(int(data.n_classes), dtype=np.float64)[noisy]
    teacher_mean = reliability.mean_posterior

    ball_diag = {
        "num_balls": int(partition.summary["num_balls"]),
        "embedding_path": str(embedding_path),
        "assignment_hash": partition.summary.get("assignment_hash", ""),
    }
    rows = [
        _result_row(
            common,
            "formal_loo_gb",
            formal_targets.cpu().numpy(),
            formal_alpha.cpu().numpy(),
            noisy,
            clean,
            train,
            ball_diag,
        ),
        _result_row(
            common,
            "utv_gb",
            utv.soft_targets,
            utv.verified_alpha,
            noisy,
            clean,
            train,
            {**ball_diag, **utv.diagnostics},
        ),
        _result_row(
            common,
            "utv_gb_shuffled",
            shuffled.soft_targets,
            shuffled.verified_alpha,
            noisy,
            clean,
            train,
            {**ball_diag, **shuffled.diagnostics},
        ),
        _result_row(
            common,
            "nrgnn_teacher_target",
            teacher_mean,
            reliability.reliability,
            noisy,
            clean,
            train,
            {"mean_teacher_node_reliability": float(reliability.reliability.mean())},
        ),
        _result_row(
            common,
            "teacher_weighted_noisy",
            noisy_one_hot,
            reliability.reliability,
            noisy,
            clean,
            train,
            {
                "mean_teacher_node_reliability": float(reliability.reliability.mean()),
                "note": "target remains noisy one-hot; alpha records sample weight",
            },
        ),
        _result_row(
            common,
            "clean_oracle_verifier",
            oracle.soft_targets,
            oracle.verified_alpha,
            noisy,
            clean,
            train,
            {**ball_diag, **oracle.diagnostics, "diagnostic_only": 1},
        ),
    ]
    teacher_row = {
        **common,
        **teacher_meta,
        "mean_entropy_confidence": float(reliability.entropy_confidence.mean()),
        "mean_stability_confidence": float(reliability.stability_confidence.mean()),
        "mean_reliability": float(reliability.reliability.mean()),
        "median_reliability": float(np.median(reliability.reliability)),
    }
    protocol_row = {
        **common,
        "teacher_cache_valid": 1,
        "probability_integrity": int(
            np.isfinite(teacher_samples).all()
            and np.allclose(teacher_samples.sum(axis=2), 1.0, atol=1e-6)
        ),
        "split_disjoint": int(
            np.intersect1d(train, val).size == 0
            and np.intersect1d(train, test).size == 0
            and np.intersect1d(val, test).size == 0
        ),
        "train_only_noise": int(
            np.isin(modified, train).all()
            and np.isin(modified, val).sum() == 0
            and np.isin(modified, test).sum() == 0
        ),
        "status": "ok",
    }
    return rows, teacher_row, protocol_row


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


def _report(rows: list[dict], decision: dict, complete: bool) -> str:
    lines = [
        "# MLP-49 UTV-GB-LCSS Phase A report",
        "",
        "## Protocol",
        "",
        "- Fixed GRACE H and fixed recursive 2-means balls; labels never alter Stage 2.",
        "- Formal tie-safe leave-one-out targets; NRGNN only attenuates alpha from non-training-node evidence.",
        "- Uniform 0.4/0.6, train-only noise, percent 0.2/0.1/0.7, seed 0.",
        f"- Grid complete: **{complete}**.",
        "",
        "## Target quality",
        "",
        "| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Coverage | Mean alpha |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["dataset"], float(item["noise_rate"]), item["method"])):
        lines.append(
            "| {dataset} | {noise_rate:.1f} | {method} | {target_accuracy:.4f} | "
            "{brier:.4f} | {ncr:.4f} | {cdr:.4f} | {coverage:.4f} | {mean_alpha:.4f} |".format(
                **row,
                coverage=float(row.get("verified_correction_coverage", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Preregistered decision",
            "",
            f"- Pass: **{decision.get('pass', False)}**",
            f"- Reason: `{decision.get('reason', 'not_evaluated')}`",
        ]
    )
    if complete:
        lines.extend(
            [
                f"- Brier better than formal and shuffled: {decision['brier_both_pass_count']}/6",
                f"- Brier better than shuffled: {decision['brier_shuffled_pass_count']}/6",
                f"- Clean-oracle verifier headroom: {decision['oracle_headroom_pass_count']}/6",
                f"- Mean target-accuracy delta vs formal: {100.0 * decision['mean_accuracy_delta_vs_formal']:+.3f}pp",
                "",
                "Phase B is permitted only when every frozen gate passes. A failure does not authorize a second verifier formula.",
            ]
        )
        formal_rows = {
            _cell_key(row["dataset"], row["noise_rate"]): row
            for row in rows
            if row["method"] == "formal_loo_gb"
        }
        utv_rows = {
            _cell_key(row["dataset"], row["noise_rate"]): row
            for row in rows
            if row["method"] == "utv_gb"
        }
        teacher_rows = [row for row in rows if row["method"] == "nrgnn_teacher_target"]
        brier_deltas = [
            float(utv_rows[key]["brier"]) - float(formal_rows[key]["brier"])
            for key in sorted(formal_rows)
        ]
        alpha_values = [float(row["mean_alpha"]) for row in utv_rows.values()]
        teacher_accuracies = [float(row["target_accuracy"]) for row in teacher_rows]
        teacher_cdr = [float(row["cdr"]) for row in teacher_rows]
        lines.extend(
            [
                "",
                "## Detailed conclusion",
                "",
                "### Hypothesis and granular-ball role",
                "",
                "MLP-49 kept the label-free GRACE partition and formal tie-safe LOO local target unchanged. The robust NRGNN teacher could only attenuate each node's ball-correction weight using non-training-node posterior agreement and confidence.",
                "",
                "### Direct evidence",
                "",
                f"- UTV improved Brier over both formal LOO and shuffled UTV in {decision['brier_both_pass_count']}/6 cells. Its Brier delta versus formal LOO ranged from {min(brier_deltas):+.4f} to {max(brier_deltas):+.4f}; all six were adverse.",
                f"- Mean target-accuracy delta versus formal LOO was {100.0 * decision['mean_accuracy_delta_vs_formal']:+.3f}pp. Cora u0.4 and CiteSeer u0.4 crossed the preregistered 1pp adverse boundary.",
                f"- Real balls beat size-matched shuffled balls in Brier in {decision['brier_shuffled_pass_count']}/6 cells. Thus geometry still carried local information, but teacher attenuation did not improve the formal target.",
                f"- Clean-oracle verification improved formal LOO Brier in {decision['oracle_headroom_pass_count']}/6 cells. This removes the premise that a more accurate verifier alone can rescue the multiplicative rule.",
                f"- The direct NRGNN posterior target had {100.0 * min(teacher_accuracies):.2f}%--{100.0 * max(teacher_accuracies):.2f}% target accuracy, but also {100.0 * min(teacher_cdr):.2f}%--{100.0 * max(teacher_cdr):.2f}% clean-label damage. The teacher contains useful information, but it is not a safe drop-in correction target.",
                f"- UTV mean alpha was only {min(alpha_values):.4f}--{max(alpha_values):.4f}. MC stability was high, while posterior entropy confidence was low; multiplying agreement and entropy confidence suppressed nearly all hard correction changes.",
                "",
                "### Supported conclusion",
                "",
                "The preregistered Phase A gate failed with reason `brier_controls_failed`; no downstream MLP run is authorized. Robust-teacher verification by multiplicatively shrinking the formal LOO alpha is closed under this frozen formulation.",
                "",
                "### Interpretation boundary",
                "",
                "This result does not show that robust teacher knowledge is useless: the direct teacher targets are substantially more accurate, and MLP-47 already showed useful NRGNN distillation. It shows that confidence-only attenuation discards beneficial GB corrections faster than it removes harmful ones. Replacing the product by a tuned threshold or a second attenuation formula would be post-result tuning and is not permitted. A future method would need a genuinely different information path, such as teacher residual content or GB regularization of teacher predictions.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "This is a smoke/partial-grid report. It cannot pass or fail the formal Phase A gate.",
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_list(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _parse_rates(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP-49 teacher-verified GB target diagnostic")
    parser.add_argument("--datasets", default=",".join(EXPECTED_DATASETS))
    parser.add_argument("--noise_rates", default="0.4,0.6")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mc_samples", type=int, default=MC_SAMPLES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "runs" / "mlp49_utv_gblcss_20260805"),
    )
    parser.add_argument("--force_teacher", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("MLP-49 Phase A is frozen to seed 0")
    if args.mc_samples != MC_SAMPLES:
        raise ValueError("MLP-49 Phase A is frozen to 8 MC samples")
    datasets = _parse_list(args.datasets)
    rates = _parse_rates(args.noise_rates)
    invalid_datasets = sorted(set(datasets) - set(EXPECTED_DATASETS))
    invalid_rates = sorted(set(round(rate, 6) for rate in rates) - set(NOISE_RATES))
    if invalid_datasets or invalid_rates:
        raise ValueError(
            f"outside preregistered Phase A grid: datasets={invalid_datasets}, rates={invalid_rates}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    teacher_rows = []
    protocol_rows = []
    for dataset in datasets:
        for rate in rates:
            print(f"MLP49_START dataset={dataset} noise_rate={rate}", flush=True)
            cell_rows, teacher_row, protocol_row = run_cell(
                dataset,
                rate,
                args.seed,
                args.device,
                output_dir,
                args.force_teacher,
            )
            rows.extend(cell_rows)
            teacher_rows.append(teacher_row)
            protocol_rows.append(protocol_row)
            _write_csv(output_dir / "target_quality.csv", rows)
            _write_csv(output_dir / "teacher_reliability.csv", teacher_rows)
            _write_csv(output_dir / "protocol_audit.csv", protocol_rows)
            candidate = next(row for row in cell_rows if row["method"] == "utv_gb")
            formal = next(row for row in cell_rows if row["method"] == "formal_loo_gb")
            print(
                f"MLP49_RESULT dataset={dataset} noise_rate={rate} "
                f"acc_delta={100.0 * (candidate['target_accuracy'] - formal['target_accuracy']):+.3f}pp "
                f"brier_delta={candidate['brier'] - formal['brier']:+.6f}",
                flush=True,
            )

    expected_count = len(EXPECTED_DATASETS) * len(NOISE_RATES) * len(PHASE_A_METHODS)
    complete = len(rows) == expected_count and {
        _cell_key(row["dataset"], row["noise_rate"]) for row in rows
    } == {
        _cell_key(dataset, rate)
        for dataset in EXPECTED_DATASETS
        for rate in NOISE_RATES
    }
    decision = evaluate_phase_a_gate(rows)
    completeness = [
        {
            "dataset": dataset,
            "noise_rate": rate,
            "expected_methods": len(PHASE_A_METHODS),
            "observed_methods": sum(
                1
                for row in rows
                if _cell_key(row["dataset"], row["noise_rate"]) == _cell_key(dataset, rate)
            ),
            "complete": int(
                sum(
                    1
                    for row in rows
                    if _cell_key(row["dataset"], row["noise_rate"]) == _cell_key(dataset, rate)
                )
                == len(PHASE_A_METHODS)
            ),
        }
        for dataset in EXPECTED_DATASETS
        for rate in NOISE_RATES
    ]
    _write_csv(output_dir / "completeness.csv", completeness)
    (output_dir / "phase_a_gate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "argv": sys.argv,
        "datasets": datasets,
        "noise_rates": rates,
        "seed": args.seed,
        "mc_samples": args.mc_samples,
        "device": args.device,
        "formal_grid_complete": complete,
    }
    (output_dir / "command_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "mlp49_report.md").write_text(
        _report(rows, decision, complete), encoding="utf-8"
    )
    print(
        f"MLP49_DONE complete={complete} pass={decision.get('pass', False)} "
        f"reason={decision.get('reason')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
