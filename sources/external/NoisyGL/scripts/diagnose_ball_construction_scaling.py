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

from scripts.linear_probe_embeddings import embedding_path_for
from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import (
    load_frozen_embeddings,
    partition_purity_diagnostics,
)
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import setup_seed


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / max(1, len(values)))


def _std(values) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def _mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().numpy().astype(np.int64)
        return mask.detach().cpu().numpy().astype(np.int64)
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr).astype(np.int64)
    return arr.astype(np.int64)


def _train_coverage(assignment: np.ndarray, train_mask, min_train_per_ball: int) -> dict:
    train_idx = _mask_to_index_array(train_mask)
    train_counts = []
    eligible_counts = []
    for ball_id in sorted(np.unique(assignment).tolist()):
        count = int(np.sum(assignment[train_idx] == ball_id))
        train_counts.append(count)
        if count >= int(min_train_per_ball):
            eligible_counts.append(count)
    return {
        "mean_train_per_ball": float(np.mean(train_counts)) if train_counts else 0.0,
        "min_train_per_ball_observed": int(min(train_counts)) if train_counts else 0,
        "eligible_balls": int(len(eligible_counts)),
        "eligible_train_nodes": int(sum(eligible_counts)),
        "eligible_mean_train_per_ball": float(np.mean(eligible_counts)) if eligible_counts else 0.0,
    }


def diagnose_one_grid_point(args, dataset_name, seed, embeddings, data, max_balls, min_ball_size, split_improvement) -> dict:
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=seed,
    )
    purity = partition_purity_diagnostics(
        noisy_label=noisy_label,
        clean_label=data.labels,
        assignment=gb_result.assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=args.min_train_per_ball,
    )
    modified = np.asarray(modified_mask, dtype=np.int64)
    train_idx = _mask_to_index_array(data.train_masks)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    return {
        "dataset": str(dataset_name).lower(),
        "seed": int(seed),
        "max_balls": int(max_balls),
        "min_ball_size": int(min_ball_size),
        "split_improvement": float(split_improvement),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "actual_noise_rate": float(modified.size / max(1, int(data.labels.shape[0]))),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
        **gb_result.summary,
        **_train_coverage(gb_result.assignment, data.train_masks, args.min_train_per_ball),
        **purity,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(
            row["dataset"],
            int(row["max_balls"]),
            int(row["min_ball_size"]),
            float(row["split_improvement"]),
        )].append(row)
    metrics = [
        "num_balls",
        "mean_ball_size",
        "singleton_ratio",
        "mean_radius",
        "mean_train_per_ball",
        "eligible_mean_train_per_ball",
        "true_all_purity_weighted",
        "true_train_purity_weighted",
        "noisy_train_purity_weighted",
    ]
    summary = []
    for (dataset, max_balls, min_ball_size, split_improvement), group in sorted(groups.items()):
        out = {
            "dataset": dataset,
            "max_balls": int(max_balls),
            "min_ball_size": int(min_ball_size),
            "split_improvement": float(split_improvement),
            "n": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric, "") not in {"", None}]
            if values:
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_std"] = _std(values)
        summary.append(out)
    return summary


def _row_key(row):
    return (
        float(row.get("true_all_purity_weighted_mean", 0.0)),
        float(row.get("num_balls_mean", 0.0)),
        -float(row.get("mean_ball_size_mean", 0.0)),
    )


def best_noncollapsed(summary: list[dict], dataset: str, min_balls: int) -> dict | None:
    candidates = [
        row for row in summary
        if row["dataset"] == dataset and float(row.get("num_balls_mean", 0.0)) >= float(min_balls)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_row_key, reverse=True)[0]


def default_row(summary: list[dict], dataset: str) -> dict | None:
    for row in summary:
        if (
            row["dataset"] == dataset
            and int(row["max_balls"]) == 64
            and int(row["min_ball_size"]) == 32
            and abs(float(row["split_improvement"]) - 0.05) < 1e-12
        ):
            return row
    return None


def min_noncollapse(summary: list[dict], dataset: str, min_balls: int) -> str:
    candidates = [
        row for row in summary
        if row["dataset"] == dataset and float(row.get("num_balls_mean", 0.0)) >= float(min_balls)
    ]
    if not candidates:
        return "not reached"
    row = sorted(
        candidates,
        key=lambda item: (int(item["min_ball_size"]), float(item["split_improvement"]), int(item["max_balls"])),
    )[0]
    return f"min_ball_size={row['min_ball_size']}, split_improvement={row['split_improvement']}, max_balls={row['max_balls']}"


def decision_for_dataset(best: dict | None, reference_purity: float, min_balls: int) -> str:
    if best is None:
        return "A: collapsed for all scanned parameters"
    purity = float(best.get("true_all_purity_weighted_mean", 0.0))
    if purity >= reference_purity - 0.05:
        return "B: non-collapsed and near small-graph purity; downstream validation warranted"
    return "A/partial: non-collapsed but purity remains below small-graph reference"


def write_report(path: Path, summary: list[dict], args) -> None:
    default_cora = default_row(summary, "cora")
    default_citeseer = default_row(summary, "citeseer")
    reference_values = [
        float(row.get("true_all_purity_weighted_mean", 0.0))
        for row in (default_cora, default_citeseer)
        if row is not None
    ]
    reference_purity = min(reference_values) if reference_values else 0.70

    lines = [
        "# Phase 32a: Ball Construction Scaling Diagnostic",
        "",
        "## Protocol",
        "",
        f"- Datasets: `{args.datasets}`",
        f"- Seeds: `{args.seeds}`",
        f"- max_balls: `{args.max_balls}`",
        f"- min_ball_size: `{args.min_ball_sizes}`",
        f"- split_improvement: `{args.split_improvements}`",
        f"- Noise: `{args.noise_type}` `{args.noise_rate}`",
        "- Split: percent 0.2 / 0.1 / 0.7",
        "- Training: none. This is a geometry-only diagnostic on frozen H.",
        "- Anchor configs are not modified.",
        "",
        "## Small-Graph Reference",
        "",
        "| Dataset | Params | num_balls | true_all_purity_weighted | mean_train_per_ball |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, row in [("cora", default_cora), ("citeseer", default_citeseer)]:
        if row is None:
            continue
        lines.append(
            "| {dataset} | 64 / 32 / 0.05 | {balls:.2f} | {purity:.4f} | {train:.2f} |".format(
                dataset=dataset,
                balls=row.get("num_balls_mean", 0.0),
                purity=row.get("true_all_purity_weighted_mean", 0.0),
                train=row.get("mean_train_per_ball_mean", 0.0),
            )
        )
    lines.extend([
        "",
        f"Reference threshold for 'near small-graph purity': `{reference_purity - 0.05:.4f}` (min default Cora/CiteSeer purity minus 0.05).",
        "",
        "## Best Non-Collapsed Points",
        "",
        "| Dataset | Best params | num_balls | true_all_purity_weighted | mean_train_per_ball | Min non-collapse setting | Decision |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    for dataset in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        best = best_noncollapsed(summary, dataset, args.noncollapse_balls)
        if best is None:
            lines.append(
                f"| {dataset} | none | 0.00 | 0.0000 | 0.00 | {min_noncollapse(summary, dataset, args.noncollapse_balls)} | {decision_for_dataset(best, reference_purity, args.noncollapse_balls)} |"
            )
            continue
        params = f"{best['max_balls']} / {best['min_ball_size']} / {best['split_improvement']}"
        lines.append(
            "| {dataset} | {params} | {balls:.2f} | {purity:.4f} | {train:.2f} | {min_setting} | {decision} |".format(
                dataset=dataset,
                params=params,
                balls=best.get("num_balls_mean", 0.0),
                purity=best.get("true_all_purity_weighted_mean", 0.0),
                train=best.get("mean_train_per_ball_mean", 0.0),
                min_setting=min_noncollapse(summary, dataset, args.noncollapse_balls),
                decision=decision_for_dataset(best, reference_purity, args.noncollapse_balls),
            )
        )

    lines.extend([
        "",
        "## Full Summary",
        "",
        "| Dataset | max_balls | min_ball_size | split_improvement | num_balls | true_all_purity | noisy_train_purity | mean_train_per_ball |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            "| {dataset} | {max_balls} | {min_ball_size} | {split_improvement} | {balls:.2f} | {purity:.4f} | {noisy:.4f} | {train:.2f} |".format(
                dataset=row["dataset"],
                max_balls=row["max_balls"],
                min_ball_size=row["min_ball_size"],
                split_improvement=row["split_improvement"],
                balls=row.get("num_balls_mean", 0.0),
                purity=row.get("true_all_purity_weighted_mean", 0.0),
                noisy=row.get("noisy_train_purity_weighted_mean", 0.0),
                train=row.get("mean_train_per_ball_mean", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether frozen-H granular-ball construction needs scale-aware parameters.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--datasets", default="cora,citeseer,dblp,pubmed,amazonpho")
    parser.add_argument("--ssl_method", default="grace")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max_balls", default="64,128,256")
    parser.add_argument("--min_ball_sizes", default="32,64,128,256,512")
    parser.add_argument("--split_improvements", default="0.05,0.01,0.005,0.001")
    parser.add_argument("--min_train_per_ball", type=int, default=2)
    parser.add_argument("--noncollapse_balls", type=int, default=8)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "ball_scaling_diag_20260616"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "ball_scaling_raw.csv"
    summary_path = output_dir / "ball_scaling_summary.csv"
    report_path = output_dir / "ball_scaling_report.md"

    datasets = _parse_str_list(args.datasets)
    seeds = _parse_int_list(args.seeds)
    max_balls_values = _parse_int_list(args.max_balls)
    min_ball_size_values = _parse_int_list(args.min_ball_sizes)
    split_improvement_values = _parse_float_list(args.split_improvements)

    rows = []
    for dataset_name in datasets:
        dataset_key = str(dataset_name).lower()
        embedding_path = embedding_path_for(args.embedding_root, dataset_name, args.ssl_method)
        if not embedding_path.exists():
            raise FileNotFoundError(f"missing H for dataset={dataset_name}: {embedding_path}")
        for seed in seeds:
            setup_seed(seed)
            data = _load_dataset(
                dataset_key,
                device=args.device,
                split_type=args.split_type,
                train_percent=args.train_percent,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
            )
            embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(data.labels.shape[0]))
            print(f"BALL_SCALE dataset={dataset_key} seed={seed} H={tuple(embeddings.shape)}")
            for max_balls in max_balls_values:
                for min_ball_size in min_ball_size_values:
                    for split_improvement in split_improvement_values:
                        row = diagnose_one_grid_point(
                            args,
                            dataset_key,
                            seed,
                            embeddings,
                            data,
                            max_balls=max_balls,
                            min_ball_size=min_ball_size,
                            split_improvement=split_improvement,
                        )
                        rows.append(row)
                        write_csv(raw_path, rows)
                        print(
                            "RESULT "
                            f"dataset={dataset_key} seed={seed} max={max_balls} min={min_ball_size} "
                            f"imp={split_improvement:g} balls={row['num_balls']} "
                            f"purity={row['true_all_purity_weighted']:.4f}"
                        )

    summary = summarize_rows(rows)
    write_csv(summary_path, summary)
    write_report(report_path, summary, args)
    print(f"WROTE {raw_path}")
    print(f"WROTE {summary_path}")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
