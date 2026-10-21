from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset
from utils.tools import setup_seed


RUN_DIR = REPO_ROOT / "runs" / "mlp_labelrate_20260701"

RATIO_SPECS = (
    ("0p05", 0.05, 0.10, 0.85),
    ("0p10", 0.10, 0.10, 0.80),
    ("0p20", 0.20, 0.10, 0.70),
    ("0p35", 0.35, 0.10, 0.55),
    ("0p50", 0.50, 0.10, 0.40),
    ("0p65", 0.65, 0.10, 0.25),
)
ALIAS_TO_RATIO = {alias: (train, val, test) for alias, train, val, test in RATIO_SPECS}

METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_jocor",
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "MLP+small-loss",
    "mlp_jocor": "MLP+JoCoR",
    "frozengrace_gb_mlp_softdist_beta05": "SoftDist-MLP beta=0.5",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "SoftDist beta=0.5 shuffled",
}
SOFTDIST = "frozengrace_gb_mlp_softdist_beta05"
COMPARISONS = (
    ("beta05_vs_ce", "mlp_ce"),
    ("beta05_vs_small_loss", "mlp_small_loss"),
    ("beta05_vs_jocor", "mlp_jocor"),
    ("beta05_vs_shuffled", "frozengrace_gb_mlp_softdist_beta05_shuffled"),
)


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def source_ratio(path: Path) -> tuple[float, float, float] | None:
    name = path.stem.lower()
    for alias, train, val, test in RATIO_SPECS:
        if f"train{alias}" in name:
            return train, val, test
    return None


def normalize_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in METHODS:
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    if (row.get("noise_type") or "").strip().lower() != "uniform":
        return None
    if not math.isclose(float(row.get("noise_rate", "nan")), 0.4, abs_tol=1e-12):
        return None
    ratio = source_ratio(source)
    if ratio is None:
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    train_percent, val_percent, test_percent = ratio
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": "uniform",
        "noise_rate": 0.4,
        "train_percent": train_percent,
        "val_percent": val_percent,
        "test_percent": test_percent,
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_train_noise_rate": _optional_float(row.get("actual_train_noise_rate")),
        "source": str(source),
    }


def _optional_float(value) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (row["dataset"], row["train_percent"], row["method"], row["seed"])
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} from "
                    f"{previous.get('source', '')} vs {row['test_acc']} from {row.get('source', '')}"
                )
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda item: (
            str(item["dataset"]),
            float(item["train_percent"]),
            METHODS.index(str(item["method"])),
            int(item["seed"]),
        ),
    )


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp_labelrate_"):
            continue
        for row in read_csv(path):
            normalized = normalize_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    labels = {}
    for row in rows:
        key = (
            row["dataset"],
            row["train_percent"],
            row["val_percent"],
            row["test_percent"],
            row["method"],
        )
        groups[key].append(float(row["test_acc"]))
        labels[str(row["method"])] = str(row["method_label"])
    output = []
    for (dataset, train_percent, val_percent, test_percent, method), values in groups.items():
        output.append(
            {
                "dataset": dataset,
                "train_percent": train_percent,
                "val_percent": val_percent,
                "test_percent": test_percent,
                "method": method,
                "method_label": labels[str(method)],
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda item: (str(item["dataset"]), float(item["train_percent"]), METHODS.index(str(item["method"]))))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], float(row["train_percent"]), row["method"], int(row["seed"])): float(row["test_acc"])
        for row in rows
    }
    regimes = sorted({(row["dataset"], float(row["train_percent"])) for row in rows})
    output = []
    for dataset, train_percent in regimes:
        seeds = sorted({seed for d, tr, _, seed in values if d == dataset and math.isclose(tr, train_percent)})
        for comparison, baseline in COMPARISONS:
            deltas = []
            used_seeds = []
            for seed in seeds:
                left = (dataset, train_percent, SOFTDIST, seed)
                right = (dataset, train_percent, baseline, seed)
                if left in values and right in values:
                    deltas.append(values[left] - values[right])
                    used_seeds.append(seed)
            if not deltas:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "train_percent": train_percent,
                    "comparison": comparison,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(str(seed) for seed in used_seeds),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                }
            )
    return output


def trend_rows(deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = defaultdict(list)
    for row in deltas:
        by_key[(row["dataset"], row["comparison"])].append(row)
    output = []
    for (dataset, comparison), rows in sorted(by_key.items()):
        low = [float(row["mean_delta"]) for row in rows if float(row["train_percent"]) in {0.05, 0.10}]
        mid = [float(row["mean_delta"]) for row in rows if float(row["train_percent"]) in {0.20, 0.35}]
        high = [float(row["mean_delta"]) for row in rows if float(row["train_percent"]) in {0.50, 0.65}]
        low_mean = statistics.mean(low) if low else math.nan
        mid_mean = statistics.mean(mid) if mid else math.nan
        high_mean = statistics.mean(high) if high else math.nan
        low_minus_high = low_mean - high_mean if low and high else math.nan
        if math.isnan(low_minus_high):
            direction = "incomplete"
        elif low_minus_high > 0.01:
            direction = "low_label_advantage"
        elif low_minus_high < -0.01:
            direction = "high_label_advantage"
        else:
            direction = "flat"
        output.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "low_mean_delta": low_mean,
                "mid_mean_delta": mid_mean,
                "high_mean_delta": high_mean,
                "low_minus_high": low_minus_high,
                "trend_direction": direction,
            }
        )
    return output


def split_audit(datasets: list[str], seeds: list[int]) -> list[dict[str, object]]:
    output = []
    for dataset in datasets:
        for _, train_percent, val_percent, test_percent in RATIO_SPECS:
            for seed in seeds:
                setup_seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    data = _load_dataset(
                        dataset,
                        device="cpu",
                        split_type="percent",
                        train_percent=train_percent,
                        val_percent=val_percent,
                        test_percent=test_percent,
                    )
                labels = data.labels.detach().cpu().long()
                train_idx = torch.as_tensor(data.train_masks, dtype=torch.long)
                counts = torch.bincount(labels[train_idx], minlength=int(data.n_classes))
                min_count = int(counts.min().item()) if counts.numel() else 0
                missing = int((counts == 0).sum().item()) if counts.numel() else int(data.n_classes)
                output.append(
                    {
                        "dataset": dataset,
                        "train_percent": train_percent,
                        "val_percent": val_percent,
                        "test_percent": test_percent,
                        "seed": seed,
                        "train_count": int(train_idx.numel()),
                        "n_classes": int(data.n_classes),
                        "min_class_train_count": min_count,
                        "missing_class_count": missing,
                        "class_counts": " ".join(str(int(value)) for value in counts.tolist()),
                        "warning": "low_class_count" if min_count < 3 else "",
                    }
                )
    return output


def completeness(rows: list[dict[str, object]], datasets: list[str], seeds: list[int]) -> list[dict[str, object]]:
    observed = {(row["dataset"], float(row["train_percent"]), row["method"], int(row["seed"])) for row in rows}
    output = []
    for dataset in datasets:
        for _, train_percent, _, _ in RATIO_SPECS:
            for method in METHODS:
                for seed in seeds:
                    present = (dataset, train_percent, method, seed) in observed
                    output.append(
                        {
                            "dataset": dataset,
                            "train_percent": train_percent,
                            "method": method,
                            "seed": seed,
                            "status": "ok" if present else "missing",
                        }
                    )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_curves(deltas: list[dict[str, object]], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generated = []
    datasets = sorted({str(row["dataset"]) for row in deltas})
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for comparison, _baseline in COMPARISONS:
            rows = sorted(
                [row for row in deltas if row["dataset"] == dataset and row["comparison"] == comparison],
                key=lambda item: float(item["train_percent"]),
            )
            if not rows:
                continue
            xs = [float(row["train_percent"]) for row in rows]
            ys = [float(row["mean_delta"]) for row in rows]
            ax.plot(xs, ys, marker="o", label=comparison.replace("beta05_vs_", "vs "))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("train_percent")
        ax.set_ylabel("test_acc delta")
        ax.set_title(f"SoftDist beta=0.5 label-rate gains: {dataset}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"mlp_labelrate_delta_curves_{dataset}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))
    return generated


def _fmt(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def render_report(
    aggregate_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    trends: list[dict[str, object]],
    audit: list[dict[str, object]],
    completeness_rows: list[dict[str, object]],
    plots: list[str],
    output_dir: Path,
) -> str:
    lines = [
        "# MLP label-rate sensitivity report",
        "",
        "Scope: Cora/CiteSeer/PubMed, uniform noise 0.4, train-only noise, weak X-only MLP, seed0 smoke unless n>1.",
        "",
        "## Accuracy",
        "",
        "| dataset | train | method | n | mean | std |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['train_percent'], 2)} | {row['method_label']} | "
            f"{row['n']} | {_fmt(row['mean'])} | {_fmt(row['std'])} |"
        )
    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "| dataset | train | comparison | n | mean_delta | positive | seed_deltas |",
        "|---|---:|---|---:|---:|---:|---|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['train_percent'], 2)} | {row['comparison']} | "
            f"{row['n']} | {_fmt(row['mean_delta'])} | {row['positive_count']}/{row['n']} | {row['seed_deltas']} |"
        )
    lines.extend([
        "",
        "## Trend Summary",
        "",
        "| dataset | comparison | low mean | mid mean | high mean | low-high | direction |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in trends:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {_fmt(row['low_mean_delta'])} | "
            f"{_fmt(row['mid_mean_delta'])} | {_fmt(row['high_mean_delta'])} | "
            f"{_fmt(row['low_minus_high'])} | {row['trend_direction']} |"
        )

    missing = [row for row in completeness_rows if row["status"] != "ok"]
    warnings = [row for row in audit if row.get("warning")]
    lines.extend([
        "",
        "## Split Audit",
        "",
        f"- Missing runs: {len(missing)}",
        f"- Low class-count warnings (min class train count < 3): {len(warnings)}",
    ])
    if warnings:
        lines.extend(["", "| dataset | train | seed | min class count | missing classes | class counts |", "|---|---:|---:|---:|---:|---|"])
        for row in warnings:
            lines.append(
                f"| {row['dataset']} | {_fmt(row['train_percent'], 2)} | {row['seed']} | "
                f"{row['min_class_train_count']} | {row['missing_class_count']} | {row['class_counts']} |"
            )

    lines.extend(["", "## Plots", ""])
    if plots:
        for path in plots:
            lines.append(f"- `{path}`")
    else:
        lines.append("- Plot generation skipped because matplotlib was unavailable.")

    # Seed0 trend interpretation, deliberately conservative.
    anti_noise = [row for row in trends if row["comparison"] in {"beta05_vs_small_loss", "beta05_vs_jocor"}]
    low_adv = sum(row["trend_direction"] == "low_label_advantage" for row in anti_noise)
    high_adv = sum(row["trend_direction"] == "high_label_advantage" for row in anti_noise)
    flat = sum(row["trend_direction"] == "flat" for row in anti_noise)
    lines.extend([
        "",
        "## Conservative Seed0 Interpretation",
        "",
        f"- Anti-noise trend counts across dataset-comparison pairs: low={low_adv}, high={high_adv}, flat={flat}.",
        "- Because this is seed0 only, any clear pattern is a 5-seed trigger, not a final claim.",
        f"- Output directory: `{output_dir}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP label-rate sensitivity runs.")
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--datasets", default="cora,citeseer,pubmed")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    seeds = parse_seed_list(args.seeds)
    rows = load_rows(args.run_dir)
    aggregate_rows = aggregate(rows)
    delta_rows = paired_deltas(rows)
    trends = trend_rows(delta_rows)
    audit = split_audit(datasets, seeds)
    completeness_rows = completeness(rows, datasets, seeds)
    if args.require_complete:
        missing = [row for row in completeness_rows if row["status"] != "ok"]
        if missing:
            raise SystemExit(f"missing required runs: {missing[:10]} total={len(missing)}")
    plots = maybe_plot_curves(delta_rows, args.run_dir)

    write_csv(args.run_dir / "mlp_labelrate_combined_raw.csv", rows)
    write_csv(args.run_dir / "mlp_labelrate_aggregate.csv", aggregate_rows)
    write_csv(args.run_dir / "mlp_labelrate_paired_deltas.csv", delta_rows)
    write_csv(args.run_dir / "mlp_labelrate_trends.csv", trends)
    write_csv(args.run_dir / "mlp_labelrate_split_audit.csv", audit)
    write_csv(args.run_dir / "mlp_labelrate_completeness.csv", completeness_rows)
    report = render_report(aggregate_rows, delta_rows, trends, audit, completeness_rows, plots, args.run_dir)
    (args.run_dir / "mlp_labelrate_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
