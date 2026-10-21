from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


PREDICTOR_MODULE = "predictor.FrozenGraceGB_GCN_Predictor"


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


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


def predictor_class_for(method: str):
    module = importlib.import_module(PREDICTOR_MODULE)
    return getattr(module, f"{method}_Predictor")


def _set_gb_value(model_conf, key: str, value) -> None:
    gb_conf = getattr(model_conf, "gb", None)
    if gb_conf is None:
        model_conf.gb = {key: value}
    elif isinstance(gb_conf, dict):
        gb_conf[key] = value
    else:
        setattr(gb_conf, key, value)


def _noise_stats(labels, noisy_label, modified_mask, train_mask) -> dict:
    n_nodes = int(labels.shape[0])
    train_idx = np.asarray(train_mask, dtype=np.int64)
    modified = np.asarray(modified_mask, dtype=np.int64)
    modified_train = np.intersect1d(train_idx, modified, assume_unique=False)
    return {
        "modified_count": int(modified.size),
        "actual_noise_rate": float(modified.size / max(1, n_nodes)),
        "modified_train_count": int(modified_train.size),
        "actual_train_noise_rate": float(modified_train.size / max(1, train_idx.size)),
    }


def run_one(args, dataset_name: str, method: str, seed: int, max_balls: int) -> dict:
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
    data.noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=seed,
        debug=False,
    )
    model_conf = load_conf(None, method, dataset_key)
    _set_gb_value(model_conf, "max_balls", int(max_balls))
    _set_gb_value(model_conf, "min_ball_size", int(args.min_ball_size))
    model_conf.model["n_feat"] = data.dim_feats
    model_conf.model["n_classes"] = data.n_classes
    model_conf.training["debug"] = bool(args.debug)
    predictor_cls = predictor_class_for(method)
    predictor = predictor_cls(model_conf, data, args.device)
    predictor.modified_mask = modified_mask
    predictor.method_name = f"{method}_maxballs{int(max_balls)}"
    predictor.noise_type = args.noise_type
    predictor.noise_rate = args.noise_rate
    predictor.seed = seed
    result = predictor.train()
    diagnostics = getattr(predictor, "gb_diagnostics", {}) or {}
    selected_diag = {
        key: diagnostics.get(key, "")
        for key in [
            "num_balls",
            "mean_ball_size",
            "eligible_balls",
            "eligible_train_nodes",
            "mean_train_per_ball",
            "soft_pseudo_ncr",
            "soft_pseudo_cdr",
            "mean_alpha_train",
            "true_all_purity_weighted",
            "noisy_train_purity_weighted",
            "true_train_purity_weighted",
        ]
    }
    if selected_diag.get("mean_train_per_ball", "") in {"", None}:
        eligible_balls = float(selected_diag.get("eligible_balls", 0.0) or 0.0)
        eligible_train_nodes = float(selected_diag.get("eligible_train_nodes", 0.0) or 0.0)
        selected_diag["mean_train_per_ball"] = float(eligible_train_nodes / eligible_balls) if eligible_balls > 0 else 0.0
    return {
        "dataset": dataset_key,
        "method": method,
        "seed": int(seed),
        "max_balls": int(max_balls),
        "min_ball_size": int(args.min_ball_size),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        **_noise_stats(data.labels, data.noisy_label, modified_mask, data.train_masks),
        "status": "ok",
        "error": "",
        "train": float(result.get("train", 0.0)),
        "valid": float(result.get("valid", 0.0)),
        "test": float(result.get("test", 0.0)),
        **selected_diag,
    }


def error_row(dataset_name: str, method: str, seed: int, max_balls: int, args, exc: Exception) -> dict:
    return {
        "dataset": dataset_name.lower(),
        "method": method,
        "seed": int(seed),
        "max_balls": int(max_balls),
        "min_ball_size": int(args.min_ball_size),
        "noise_type": args.noise_type,
        "noise_rate": float(args.noise_rate),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "train": "",
        "valid": "",
        "test": "",
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
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
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        groups[(str(row["dataset"]).lower(), str(row["method"]), int(row["max_balls"]))].append(row)
    metrics = [
        "test",
        "valid",
        "num_balls",
        "mean_train_per_ball",
        "soft_pseudo_ncr",
        "soft_pseudo_cdr",
        "mean_alpha_train",
        "true_all_purity_weighted",
    ]
    summary = []
    for (dataset, method, max_balls), group in sorted(groups.items()):
        out = {
            "dataset": dataset,
            "method": method,
            "max_balls": int(max_balls),
            "n": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric, "") not in {"", None}]
            if values:
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_std"] = _std(values)
        summary.append(out)
    return summary


def plot_summary(summary: list[dict], output_prefix: str | Path) -> None:
    datasets = sorted({row["dataset"] for row in summary})
    methods = sorted({row["method"] for row in summary})
    if not datasets or not methods:
        raise ValueError("no successful rows to plot")
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        for method in methods:
            rows = [row for row in summary if row["dataset"] == dataset and row["method"] == method]
            rows.sort(key=lambda row: row["max_balls"])
            if not rows:
                continue
            x = [row["max_balls"] for row in rows]
            y = [row.get("test_mean", float("nan")) for row in rows]
            err = [row.get("test_std", 0.0) for row in rows]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=method.replace("frozengrace_gb_gcn_", ""))
        ax.set_title(dataset)
        ax.set_xlabel("max_balls")
        ax.set_xticks(sorted({row["max_balls"] for row in summary if row["dataset"] == dataset}))
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Test accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_report(path: str | Path, summary: list[dict], args) -> None:
    lines = [
        "# Phase 21 Ball Count Sensitivity",
        "",
        "## Protocol",
        "",
        f"- Methods: `{args.methods}`",
        f"- Datasets: `{args.datasets}`",
        f"- Seeds: `{args.seeds}`",
        f"- max_balls: `{args.max_balls}`",
        f"- min_ball_size: fixed `{args.min_ball_size}`",
        f"- Noise: `{args.noise_type}` `{args.noise_rate}`",
        "- Split: percent 0.2 / 0.1 / 0.7",
        "- Mechanism: SoftDist [X||H] anchor unchanged; only `gb.max_balls` is overridden in memory.",
        "",
        "## Summary",
        "",
        "| Dataset | Method | max_balls | Test mean | Test std | Actual balls | Mean train/ball | NCR | CDR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {method} | {max_balls} | {test:.4f} | {test_std:.4f} | {balls:.2f} | {train_per_ball:.2f} | {ncr:.4f} | {cdr:.4f} |".format(
                dataset=row["dataset"],
                method=row["method"],
                max_balls=row["max_balls"],
                test=row.get("test_mean", 0.0),
                test_std=row.get("test_std", 0.0),
                balls=row.get("num_balls_mean", 0.0),
                train_per_ball=row.get("mean_train_per_ball_mean", 0.0),
                ncr=row.get("soft_pseudo_ncr_mean", 0.0),
                cdr=row.get("soft_pseudo_cdr_mean", 0.0),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SoftDist [X||H] ball-count sensitivity without changing anchor configs.")
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--methods", default="frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max_balls", default="16,32,64,128")
    parser.add_argument("--min_ball_size", type=int, default=32)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent", choices=["default", "percent", "samples_per_class"])
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "ball_count_sensitivity_20260615"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "ball_count_sensitivity_raw.csv"
    summary_path = output_dir / "ball_count_sensitivity_summary.csv"
    rows = []
    for dataset in _parse_str_list(args.datasets):
        for method in _parse_str_list(args.methods):
            for max_balls in _parse_int_list(args.max_balls):
                for seed in _parse_int_list(args.seeds):
                    print(f"BALL_COUNT dataset={dataset} method={method} max_balls={max_balls} seed={seed}")
                    try:
                        row = run_one(args, dataset, method, seed, max_balls)
                        print(f"RESULT dataset={row['dataset']} method={method} max_balls={max_balls} seed={seed} test={row['test']:.4f}")
                    except Exception as exc:
                        if args.stop_on_error:
                            raise
                        row = error_row(dataset, method, seed, max_balls, args, exc)
                        print(f"FAILED dataset={dataset} method={method} max_balls={max_balls} seed={seed} error={row['error']}")
                    rows.append(row)
                    write_csv(raw_path, rows)
    summary = summarize_rows(rows)
    write_csv(summary_path, summary)
    plot_summary(summary, output_dir / "ball_count_sensitivity")
    write_report(output_dir / "ball_count_sensitivity_report.md", summary, args)
    print(f"WROTE {summary_path}")


if __name__ == "__main__":
    main()
