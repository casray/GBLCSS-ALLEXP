from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _load_dataset  # noqa: E402
from utils.frozen_grace_targets import (  # noqa: E402
    apply_ball_majority_correction,
    diagnostics_for_correction,
    load_frozen_embeddings,
    partition_purity_diagnostics,
)
from utils.labelnoise import label_process  # noqa: E402
from utils.ordinary_gb import build_random_size_matched_assignment, build_size_adaptive_ellipsoid_balls  # noqa: E402
from utils.tools import load_conf, setup_seed  # noqa: E402


DEFAULT_DATASETS = "dblp,pubmed,amazonpho,amazoncom"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_CONFIDENCE_THRESHOLDS = "0.0,0.6,0.7"
REAL_METHOD = "real"
SHUFFLED_METHOD = "shuffled"


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row.get("status", "ok")).lower() == "ok"]


def _mean_for(rows: list[dict], metric: str, default: float = 0.0) -> float:
    values = [_as_float(row, metric, default) for row in rows]
    return float(np.mean(values)) if values else default


def aggregate_geaf_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for row in _completed_rows(rows):
        key = (str(row["dataset"]), str(row["method"]), _as_float(row, "majority_confidence"))
        groups.setdefault(key, []).append(row)
    out = []
    metrics = [
        "num_balls",
        "target_ball_size",
        "mean_ball_size",
        "max_ball_size",
        "mean_train_per_ball",
        "zero_train_balls",
        "zero_train_ball_ratio",
        "changed_train_ratio",
        "ncr",
        "cdr",
        "hard_net",
        "geaf_gmean",
        "geaf_f1",
        "true_train_purity_weighted",
        "noisy_train_purity_weighted",
    ]
    for (dataset, method, confidence), group in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "method": method,
            "majority_confidence": float(confidence),
            "n": int(len(group)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = _mean_for(group, metric)
        out.append(row)
    return out


def paired_geaf_delta_rows(rows: list[dict], reference_method: str, candidate_method: str) -> list[dict]:
    completed = _completed_rows(rows)
    by_key = {
        (
            str(row["dataset"]),
            _as_int(row, "seed"),
            _as_float(row, "majority_confidence"),
            str(row["method"]),
        ): row
        for row in completed
    }
    out = []
    seen_settings = sorted({(str(row["dataset"]), _as_int(row, "seed"), _as_float(row, "majority_confidence")) for row in completed})
    for dataset, seed, confidence in seen_settings:
        ref = by_key.get((dataset, seed, confidence, reference_method))
        cand = by_key.get((dataset, seed, confidence, candidate_method))
        if ref is None or cand is None:
            continue
        row = {
            "dataset": dataset,
            "seed": int(seed),
            "majority_confidence": float(confidence),
            "reference_method": reference_method,
            "candidate_method": candidate_method,
        }
        for metric in ["hard_net", "ncr", "cdr", "changed_train_ratio", "geaf_gmean", "geaf_f1"]:
            row[f"{metric}_delta"] = _as_float(cand, metric) - _as_float(ref, metric)
        out.append(row)
    return out


def evaluate_geaf_gates(
    rows: list[dict],
    reference_method: str = SHUFFLED_METHOD,
    candidate_method: str = REAL_METHOD,
    min_hard_net: float = 0.02,
    min_ncr: float = 0.05,
    max_cdr: float = 0.03,
    min_positive_seeds: int = 4,
    max_zero_train_ball_ratio: float = 0.25,
) -> list[dict]:
    completed = _completed_rows(rows)
    deltas = paired_geaf_delta_rows(completed, reference_method, candidate_method)
    delta_groups: dict[tuple[str, float], list[dict]] = {}
    for row in deltas:
        delta_groups.setdefault((str(row["dataset"]), _as_float(row, "majority_confidence")), []).append(row)
    candidate_groups: dict[tuple[str, float], list[dict]] = {}
    for row in completed:
        if str(row["method"]) == candidate_method:
            candidate_groups.setdefault((str(row["dataset"]), _as_float(row, "majority_confidence")), []).append(row)

    decisions = []
    for key, group in sorted(candidate_groups.items()):
        dataset, confidence = key
        setting_deltas = delta_groups.get(key, [])
        positive_seed_count = sum(1 for row in setting_deltas if _as_float(row, "hard_net_delta") > 0.0)
        hard_net_mean = _mean_for(group, "hard_net")
        ncr_mean = _mean_for(group, "ncr")
        cdr_mean = _mean_for(group, "cdr")
        zero_ratio_mean = _mean_for(group, "zero_train_ball_ratio")
        hard_net_delta_mean = _mean_for(setting_deltas, "hard_net_delta")
        passed = (
            hard_net_mean >= float(min_hard_net)
            and ncr_mean >= float(min_ncr)
            and cdr_mean <= float(max_cdr)
            and positive_seed_count >= int(min_positive_seeds)
            and zero_ratio_mean <= float(max_zero_train_ball_ratio)
            and hard_net_delta_mean > 0.0
        )
        reasons = []
        if hard_net_mean < float(min_hard_net):
            reasons.append("hard_net below threshold")
        if ncr_mean < float(min_ncr):
            reasons.append("ncr below threshold")
        if cdr_mean > float(max_cdr):
            reasons.append("cdr above threshold")
        if positive_seed_count < int(min_positive_seeds):
            reasons.append("not consistently better than shuffled")
        if zero_ratio_mean > float(max_zero_train_ball_ratio):
            reasons.append("zero-train balls dominate")
        if hard_net_delta_mean <= 0.0:
            reasons.append("mean hard_net not above shuffled")
        decisions.append(
            {
                "dataset": dataset,
                "majority_confidence": float(confidence),
                "passed": bool(passed),
                "reason": "passed" if passed else "; ".join(reasons),
                "n": int(len(group)),
                "positive_seed_count": int(positive_seed_count),
                "hard_net_mean": float(hard_net_mean),
                "hard_net_delta_mean": float(hard_net_delta_mean),
                "ncr_mean": float(ncr_mean),
                "cdr_mean": float(cdr_mean),
                "zero_train_ball_ratio_mean": float(zero_ratio_mean),
            }
        )
    return decisions


def _resolve_embedding_path(path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.flatnonzero(mask)
    return mask.astype(np.int64).reshape(-1)


def _train_support_summary(assignment: np.ndarray, train_mask) -> dict:
    train_idx = _index_array(train_mask)
    train_counts = []
    for ball_id in sorted(np.unique(assignment).tolist()):
        train_counts.append(int(np.sum(assignment[train_idx] == ball_id)))
    train_counts_array = np.asarray(train_counts, dtype=np.float32)
    zero = int(np.sum(train_counts_array == 0))
    return {
        "mean_train_per_ball": float(train_counts_array.mean()) if train_counts_array.size else 0.0,
        "zero_train_balls": zero,
        "zero_train_ball_ratio": float(zero / max(1, int(train_counts_array.size))),
    }


def _geaf_scores(diag: dict) -> dict:
    ncr = float(diag.get("ncr", 0.0))
    cdr = float(diag.get("cdr", 0.0))
    clean_keep = max(0.0, 1.0 - cdr)
    denom = ncr + clean_keep
    return {
        "hard_net": float(ncr - cdr),
        "geaf_gmean": float(math.sqrt(max(0.0, ncr * clean_keep))),
        "geaf_f1": float((2.0 * ncr * clean_keep / denom) if denom > 0 else 0.0),
    }


def diagnose_one(args, dataset_name: str, seed: int, confidence: float) -> list[dict]:
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
    model_conf = load_conf(None, "gcn_xh_ce", dataset_key)
    gb_conf = getattr(model_conf, "gb", {})
    embedding_path = gb_conf.get("ball_embedding_path", gb_conf.get("embedding_path")) if isinstance(gb_conf, dict) else getattr(gb_conf, "embedding_path")
    embeddings = load_frozen_embeddings(_resolve_embedding_path(embedding_path), expected_nodes=int(noisy_label.shape[0]))
    gb_result = build_size_adaptive_ellipsoid_balls(
        embeddings,
        n_classes=data.n_classes,
        random_state=int(seed),
    )
    train_idx = _index_array(data.train_masks)
    modified_idx = _index_array(modified_mask)
    base = {
        "dataset": dataset_key,
        "seed": int(seed),
        "majority_confidence": float(confidence),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": args.noise_scope,
        "modified_count": int(modified_idx.size),
        "modified_train_count": int(np.intersect1d(train_idx, modified_idx, assume_unique=False).size),
        "actual_train_noise_rate": float(np.intersect1d(train_idx, modified_idx, assume_unique=False).size / max(1, train_idx.size)),
        "status": "ok",
        "error": "",
    }
    rows = []
    for method, assignment in [
        (REAL_METHOD, gb_result.assignment),
        (SHUFFLED_METHOD, build_random_size_matched_assignment(gb_result.assignment, random_state=int(seed) + 17)),
    ]:
        corrected, changed = apply_ball_majority_correction(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            min_train_per_ball=int(args.min_train_per_ball),
            min_majority_confidence=float(confidence),
        )
        correction_diag = diagnostics_for_correction(
            noisy_label=noisy_label,
            clean_label=data.labels,
            corrected_label=corrected,
            changed=changed,
            train_mask=data.train_masks,
        )
        row = {
            **base,
            "method": method,
            **gb_result.summary,
            **_train_support_summary(assignment, data.train_masks),
            **correction_diag,
            **_geaf_scores(correction_diag),
            **partition_purity_diagnostics(
                noisy_label=noisy_label,
                clean_label=data.labels,
                assignment=assignment,
                train_mask=data.train_masks,
                n_classes=data.n_classes,
                min_train_per_ball=int(args.min_train_per_ball),
            ),
        }
        rows.append(row)
    return rows


def _error_rows(args, dataset: str, seed: int, confidence: float, exc: Exception) -> list[dict]:
    return [
        {
            "dataset": str(dataset).lower(),
            "seed": int(seed),
            "majority_confidence": float(confidence),
            "method": method,
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "noise_scope": args.noise_scope,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        for method in [REAL_METHOD, SHUFFLED_METHOD]
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_report(path: Path, raw_rows: list[dict], aggregate: list[dict], decisions: list[dict]) -> None:
    passed = [row for row in decisions if row.get("passed") in {True, "True", "true", "1", 1}]
    lines = [
        "# GEAF bridge diagnostic report",
        "",
        "Scope: offline hard-correction diagnostic. Clean labels are diagnostic only.",
        "",
        f"Raw rows: {len(raw_rows)}",
        f"Gate passed settings: {len(passed)}",
        "",
        "## Gate decisions",
        "",
        "| dataset | confidence | gate | hard_net | ncr | cdr | hard_net_delta | positive seeds | reason |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in decisions:
        lines.append(
            "| {dataset} | {confidence:.2f} | {gate} | {hard_net:.4f} | {ncr:.4f} | {cdr:.4f} | {delta:.4f} | {pos} | {reason} |".format(
                dataset=row["dataset"],
                confidence=_as_float(row, "majority_confidence"),
                gate="PASS" if row.get("passed") else "FAIL",
                hard_net=_as_float(row, "hard_net_mean"),
                ncr=_as_float(row, "ncr_mean"),
                cdr=_as_float(row, "cdr_mean"),
                delta=_as_float(row, "hard_net_delta_mean"),
                pos=_as_int(row, "positive_seed_count"),
                reason=str(row.get("reason", "")),
            )
        )
    lines.extend(["", "## Aggregate snapshot", ""])
    for row in aggregate:
        if str(row.get("method")) != REAL_METHOD:
            continue
        lines.append(
            "- {dataset} confidence={confidence:.2f}: hard_net={hard_net:.4f}, ncr={ncr:.4f}, cdr={cdr:.4f}, balls={balls:.1f}, target_size={target:.1f}".format(
                dataset=row["dataset"],
                confidence=_as_float(row, "majority_confidence"),
                hard_net=_as_float(row, "hard_net_mean"),
                ncr=_as_float(row, "ncr_mean"),
                cdr=_as_float(row, "cdr_mean"),
                balls=_as_float(row, "num_balls_mean"),
                target=_as_float(row, "target_ball_size_mean"),
            )
        )
    if not passed:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "当前 H 空间粒椭球过滤无法提供可用训练标签修正；按预注册规则停止 downstream。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Run GEAF-style hard-correction bridge diagnostics.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--confidence_thresholds", default=DEFAULT_CONFIDENCE_THRESHOLDS)
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--noise_scope", default="train", choices=["train", "all"])
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "geaf_bridge_20260629"))
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_rows = []
    for dataset in _parse_str_list(args.datasets):
        for seed in _parse_int_list(args.seeds):
            for confidence in _parse_float_list(args.confidence_thresholds):
                print(f"GEAF_BRIDGE dataset={dataset} seed={seed} confidence={confidence}", flush=True)
                try:
                    raw_rows.extend(diagnose_one(args, dataset, seed, confidence))
                except Exception as exc:
                    raw_rows.extend(_error_rows(args, dataset, seed, confidence, exc))
                write_csv(output_dir / "geaf_bridge_raw.csv", raw_rows)
    aggregate = aggregate_geaf_rows(raw_rows)
    paired = paired_geaf_delta_rows(raw_rows, SHUFFLED_METHOD, REAL_METHOD)
    decisions = evaluate_geaf_gates(raw_rows)
    write_csv(output_dir / "geaf_bridge_aggregate.csv", aggregate)
    write_csv(output_dir / "geaf_bridge_paired_deltas.csv", paired)
    write_csv(output_dir / "geaf_bridge_gate.csv", decisions)
    write_report(output_dir / "geaf_bridge_report.md", raw_rows, aggregate, decisions)
    print(f"GEAF_BRIDGE passed_settings={sum(1 for row in decisions if row.get('passed'))}", flush=True)


if __name__ == "__main__":
    main()
