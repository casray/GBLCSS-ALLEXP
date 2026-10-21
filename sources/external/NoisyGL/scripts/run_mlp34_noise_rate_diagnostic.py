from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset
from scripts.run_standard_downstream import (
    _noise_fingerprint,
    _noise_stats,
    _split_fingerprint,
)
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.lcss_noise_rate import estimate_ball_noise_rate_loo
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


DEFAULT_METHOD = "frozengrace_gb_mlp_softdist_beta05_small_loss"


def _parse_list(value, cast=str):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _mask_to_indices(mask) -> np.ndarray:
    values = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    if values.dtype == np.bool_:
        return np.flatnonzero(values).astype(np.int64)
    return values.astype(np.int64, copy=False).reshape(-1)


def _gb_value(conf, key, default=None):
    gb = getattr(conf, "gb", {})
    return gb.get(key, default) if isinstance(gb, dict) else getattr(gb, key, default)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_rhohat_gate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    grouped = defaultdict(list)
    for row in valid:
        grouped[str(row["dataset"])].append(row)
    dataset_rows = []
    for dataset, members in sorted(grouped.items()):
        mean_error = float(np.mean([float(row["weighted_abs_error"]) for row in members]))
        mean_coverage = float(np.mean([float(row["coverage"]) for row in members]))
        dataset_rows.append(
            {
                "dataset": dataset,
                "n_seeds": len({int(row["seed"]) for row in members}),
                "mean_weighted_abs_error": mean_error,
                "mean_coverage": mean_coverage,
            }
        )
    good_error = sum(row["mean_weighted_abs_error"] <= 0.10 for row in dataset_rows)
    maximum_error = max(
        (row["mean_weighted_abs_error"] for row in dataset_rows),
        default=float("inf"),
    )
    minimum_coverage = min(
        (row["mean_coverage"] for row in dataset_rows),
        default=0.0,
    )
    complete = bool(dataset_rows) and all(row["n_seeds"] == 5 for row in dataset_rows)
    passed = bool(
        complete
        and good_error >= 2
        and maximum_error <= 0.15
        and minimum_coverage >= 0.50
    )
    return {
        "pass": passed,
        "datasets_with_error_at_most_0p10": int(good_error),
        "maximum_mean_abs_error": float(maximum_error),
        "minimum_mean_coverage": float(minimum_coverage),
        "complete_five_seed": complete,
        "datasets": dataset_rows,
    }


def diagnose_one(args, dataset_name: str, seed: int) -> tuple[dict, list[dict]]:
    setup_seed(seed)
    data = _load_dataset(
        dataset_name,
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
        eligible_mask=data.train_masks,
        debug=False,
    )
    conf = load_conf(None, args.method, dataset_name)
    embedding_path = _gb_value(conf, "embedding_path")
    if not embedding_path:
        raise ValueError(f"missing gb.embedding_path for {args.method}/{dataset_name}")
    embeddings = load_frozen_embeddings(
        embedding_path,
        expected_nodes=int(data.n_nodes),
    )
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=int(_gb_value(conf, "max_balls", 64)),
        min_ball_size=int(_gb_value(conf, "min_ball_size", 32)),
        split_improvement=float(_gb_value(conf, "split_improvement", 0.05)),
        random_state=int(seed),
    )
    estimate = estimate_ball_noise_rate_loo(
        assignment=gb_result.assignment,
        noisy_label=noisy_label,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_votes=args.min_train_votes,
    )
    stats = _noise_stats(
        data.labels,
        noisy_label,
        modified_mask,
        data.train_masks,
        data.val_masks,
        data.test_masks,
    )
    actual = float(stats["actual_train_noise_rate"])
    row = {
        "dataset": dataset_name,
        "seed": int(seed),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "noise_scope": "train",
        "split_type": args.split_type,
        "train_percent": float(args.train_percent),
        "val_percent": float(args.val_percent),
        "test_percent": float(args.test_percent),
        "split_hash": _split_fingerprint(data.train_masks, data.val_masks, data.test_masks),
        "noise_hash": _noise_fingerprint(data.labels, noisy_label),
        "embedding_path": str(embedding_path),
        "max_balls": int(_gb_value(conf, "max_balls", 64)),
        "min_ball_size": int(_gb_value(conf, "min_ball_size", 32)),
        "split_improvement": float(_gb_value(conf, "split_improvement", 0.05)),
        "min_train_votes": int(args.min_train_votes),
        "num_balls": int(gb_result.summary["num_balls"]),
        "weighted_estimate": float(estimate.weighted_estimate),
        "unweighted_estimate": float(estimate.unweighted_estimate),
        "weighted_abs_error": abs(float(estimate.weighted_estimate) - actual),
        "unweighted_abs_error": abs(float(estimate.unweighted_estimate) - actual),
        "coverage": float(estimate.coverage),
        "eligible_count": int(estimate.eligible_count),
        "train_count": int(estimate.train_count),
        "effective_sample_size": float(estimate.effective_sample_size),
        "weight_sum": float(estimate.weight_sum),
        **stats,
        "status": "ok",
        "error": "",
    }
    per_ball = [
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "noise_type": args.noise_type,
            "noise_rate": float(args.noise_rate),
            "split_hash": row["split_hash"],
            "noise_hash": row["noise_hash"],
            **ball_row,
        }
        for ball_row in estimate.per_ball
    ]
    return row, per_ball


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLP-34 noisy-only ball LOO noise-rate diagnostic")
    parser.add_argument("--datasets", default="cora,citeseer,corafull")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--min_train_votes", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = []
    per_ball_rows = []
    for dataset in _parse_list(args.datasets):
        for seed in _parse_list(args.seeds, int):
            try:
                row, ball_rows = diagnose_one(args, dataset, seed)
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "noise_type": args.noise_type,
                    "noise_rate": float(args.noise_rate),
                    "noise_scope": "train",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                ball_rows = []
            rows.append(row)
            per_ball_rows.extend(ball_rows)
            print(
                f"RHOHAT dataset={dataset} seed={seed} status={row['status']} "
                f"estimate={row.get('weighted_estimate', 'NA')} "
                f"coverage={row.get('coverage', 'NA')}"
            )

    output = Path(args.output)
    _write_csv(output, rows)
    _write_csv(output.with_name(f"{output.stem}_per_ball.csv"), per_ball_rows)
    gate = evaluate_rhohat_gate(rows)
    gate_path = output.with_name(f"{output.stem}_gate.json")
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"WROTE {output}")
    print(f"B1_GATE pass={int(gate['pass'])} details={gate}")


if __name__ == "__main__":
    main()
