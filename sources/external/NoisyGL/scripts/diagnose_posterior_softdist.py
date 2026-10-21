from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import (
    load_frozen_embeddings,
    support_bin_diagnostics,
)
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import load_conf, setup_seed


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate_support_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["support_bin"]))].append(row)

    output = []
    for (dataset, support_bin), group in sorted(grouped.items()):
        aggregated = {
            "dataset": dataset,
            "support_bin": support_bin,
            "n": len(group),
            "seeds": ",".join(str(seed) for seed in sorted({int(row["seed"]) for row in group})),
        }
        numeric_keys = sorted(
            {
                key
                for row in group
                for key, value in row.items()
                if key not in {"seed"} and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in group if key in row and math.isfinite(float(row[key]))]
            aggregated[f"{key}_mean"] = _mean(values)
            aggregated[f"{key}_std"] = _std(values)
        output.append(aggregated)
    return output


def evaluate_support_gate(
    rows: list[dict],
    cdr_gap_threshold: float = 0.05,
    mass_gap_threshold: float = 0.02,
    min_positive_seeds: int = 4,
    min_passing_datasets: int = 2,
) -> tuple[bool, list[dict]]:
    by_dataset_seed = defaultdict(dict)
    for row in rows:
        by_dataset_seed[(str(row["dataset"]), int(row["seed"]))][str(row["support_bin"])] = row

    datasets = sorted({dataset for dataset, _ in by_dataset_seed})
    decisions = []
    for dataset in datasets:
        seed_rows = {
            seed: bins
            for (row_dataset, seed), bins in by_dataset_seed.items()
            if row_dataset == dataset
        }
        comparison_bin = "8+" if any("8+" in bins for bins in seed_rows.values()) else "4-7"
        paired = [
            (seed, bins["2-3"], bins[comparison_bin])
            for seed, bins in sorted(seed_rows.items())
            if "2-3" in bins and comparison_bin in bins
        ]
        cdr_gaps = [float(low["cdr"]) - float(high["cdr"]) for _, low, high in paired]
        mass_gaps = [
            float(high["clean_mass_delta_per_node"]) - float(low["clean_mass_delta_per_node"])
            for _, low, high in paired
        ]
        cdr_positive = sum(gap > 0.0 for gap in cdr_gaps)
        mass_positive = sum(gap > 0.0 for gap in mass_gaps)
        cdr_gate = _mean(cdr_gaps) >= float(cdr_gap_threshold) and cdr_positive >= int(min_positive_seeds)
        mass_gate = (
            _mean(mass_gaps) >= float(mass_gap_threshold)
            and mass_positive >= int(min_positive_seeds)
        )
        decisions.append(
            {
                "dataset": dataset,
                "comparison_bin": comparison_bin,
                "paired_seed_count": len(paired),
                "cdr_gap_mean": _mean(cdr_gaps),
                "cdr_positive_seed_count": cdr_positive,
                "mass_gap_mean": _mean(mass_gaps),
                "mass_positive_seed_count": mass_positive,
                "cdr_gate": cdr_gate,
                "mass_gate": mass_gate,
                "passed": bool(cdr_gate or mass_gate),
            }
        )
    return sum(bool(row["passed"]) for row in decisions) >= int(min_passing_datasets), decisions


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, args, gate_passed: bool, decisions: list[dict]) -> None:
    lines = [
        "# Posterior-SoftDist support diagnostic",
        "",
        f"- Decision: **{'PROCEED' if gate_passed else 'STOP'}**",
        f"- Datasets: {', '.join(args.datasets)}",
        f"- Seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- Noise: {args.noise_type} {args.noise_rate}",
        (
            f"- Split: {args.split_type}, train/val/test="
            f"{args.train_percent}/{args.val_percent}/{args.test_percent}"
        ),
        "- Support comparison: 2-3 vs 8+; use 4-7 only when 8+ is absent.",
        (
            f"- Gate: mean CDR gap >= {args.cdr_gap_threshold} or mean clean-mass gap "
            f">= {args.mass_gap_threshold}, with >= {args.min_positive_seeds} positive paired seeds, "
            f"on >= {args.min_passing_datasets} datasets."
        ),
        "",
        "| Dataset | Comparator | Paired seeds | CDR gap | CDR +seeds | Mass gap | Mass +seeds | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['dataset']} | {row['comparison_bin']} | {row['paired_seed_count']} "
            f"| {row['cdr_gap_mean']:.4f} | {row['cdr_positive_seed_count']} "
            f"| {row['mass_gap_mean']:.4f} | {row['mass_positive_seed_count']} "
            f"| {'yes' if row['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            (
                "结论：低票数球的统计可靠性缺口达到预注册标准，进入 Posterior-SoftDist。"
                if gate_passed
                else "结论：低票数球的统计可靠性缺口未达到预注册标准，停止主方法改动。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args, dataset_name: str, seed: int) -> list[dict]:
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
    noisy_label, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    conf = load_conf(None, "frozengrace_gb_gcn_softdist_xh", dataset_key)
    gb_conf = conf.gb
    embeddings = load_frozen_embeddings(
        gb_conf["embedding_path"],
        expected_nodes=int(data.labels.shape[0]),
    )
    balls = build_ordinary_granular_balls(
        embeddings,
        max_balls=gb_conf["max_balls"],
        min_ball_size=gb_conf["min_ball_size"],
        split_improvement=gb_conf["split_improvement"],
        random_state=seed,
    )
    rows = support_bin_diagnostics(
        noisy_label=noisy_label,
        clean_label=data.labels,
        assignment=balls.assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=gb_conf["min_train_per_ball"],
    )
    for row in rows:
        row.update(
            {
                "dataset": dataset_key,
                "seed": int(seed),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "split_type": args.split_type,
                "train_percent": float(args.train_percent),
                "val_percent": float(args.val_percent),
                "test_percent": float(args.test_percent),
                "embedding_path": gb_conf["embedding_path"],
                "max_balls": int(gb_conf["max_balls"]),
                "min_ball_size": int(gb_conf["min_ball_size"]),
                "split_improvement": float(gb_conf["split_improvement"]),
                "num_balls": int(balls.summary["num_balls"]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether low-support granular balls justify posterior calibration."
    )
    parser.add_argument("--datasets", type=_parse_str_list, default=["cora", "citeseer", "coauthorcs"])
    parser.add_argument("--seeds", type=_parse_int_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--noise-type", default="uniform")
    parser.add_argument("--noise-rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-type", default="percent")
    parser.add_argument("--train-percent", type=float, default=0.2)
    parser.add_argument("--val-percent", type=float, default=0.1)
    parser.add_argument("--test-percent", type=float, default=0.7)
    parser.add_argument("--cdr-gap-threshold", type=float, default=0.05)
    parser.add_argument("--mass-gap-threshold", type=float, default=0.02)
    parser.add_argument("--min-positive-seeds", type=int, default=4)
    parser.add_argument("--min-passing-datasets", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "posterior_softdist_20260622",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    raw_path = output_dir / "support_bin_raw.csv"
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"[support-diagnostic] dataset={dataset} seed={seed}", flush=True)
            rows.extend(run_one(args, dataset, seed))
            _write_csv(raw_path, rows)

    aggregate = aggregate_support_rows(rows)
    gate_passed, decisions = evaluate_support_gate(
        rows,
        cdr_gap_threshold=args.cdr_gap_threshold,
        mass_gap_threshold=args.mass_gap_threshold,
        min_positive_seeds=args.min_positive_seeds,
        min_passing_datasets=args.min_passing_datasets,
    )
    _write_csv(output_dir / "support_bin_aggregate.csv", aggregate)
    _write_csv(output_dir / "support_bin_gate.csv", decisions)
    _write_report(output_dir / "support_bin_report.md", args, gate_passed, decisions)
    print(f"[support-diagnostic] decision={'PROCEED' if gate_passed else 'STOP'}")
    print(f"[support-diagnostic] report={output_dir / 'support_bin_report.md'}")


if __name__ == "__main__":
    main()
