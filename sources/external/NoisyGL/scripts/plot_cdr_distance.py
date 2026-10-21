from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / max(1, len(values)))


def _std(values) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)))


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def summarize_distance_bins(rows: list[dict], variant: str = "baseline") -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if str(row.get("variant", "")) != str(variant):
            continue
        bin_value = str(row.get("bin", ""))
        if bin_value == "all":
            continue
        groups[(str(row["dataset"]).lower(), int(float(bin_value)))].append(row)

    summary = []
    for (dataset, bin_id), group in sorted(groups.items()):
        cdr_values = [float(row["cdr"]) for row in group]
        ncr_values = [float(row["ncr"]) for row in group]
        summary.append(
            {
                "dataset": dataset,
                "bin": int(bin_id),
                "n": len(group),
                "cdr_mean": _mean(cdr_values),
                "cdr_std": _std(cdr_values),
                "ncr_mean": _mean(ncr_values),
                "ncr_std": _std(ncr_values),
            }
        )
    return summary


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict], output_prefix: str | Path) -> None:
    datasets = sorted({row["dataset"] for row in summary})
    if not datasets:
        raise ValueError("no rows to plot")
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        rows = [row for row in summary if row["dataset"] == dataset]
        rows.sort(key=lambda row: row["bin"])
        x = [row["bin"] for row in rows]
        cdr = [row["cdr_mean"] for row in rows]
        ncr = [row["ncr_mean"] for row in rows]
        ax.plot(x, cdr, marker="o", label="CDR")
        ax.plot(x, ncr, marker="s", label="NCR")
        ax.set_title(dataset)
        ax.set_xlabel("Distance-to-center bin")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def write_report(path: str | Path, summary: list[dict], input_path: str | Path, variant: str) -> None:
    lines = [
        "# CDR/NCR Distance-Bin Figure Report",
        "",
        f"- Input: `{input_path}`",
        f"- Variant: `{variant}`",
        "- Clean labels are used only in the pre-existing diagnostic CSV; this script only plots aggregated diagnostics.",
        "",
        "| Dataset | Bin | CDR mean | NCR mean | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['bin']} | {row['cdr_mean']:.4f} | {row['ncr_mean']:.4f} | {row['n']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDR/NCR by distance-to-ball-center bins.")
    parser.add_argument("--input", default="runs/nodewise_diag_20260612/distance_bins.csv")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--output_dir", default="runs/figures_20260615")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    summary = summarize_distance_bins(rows, variant=args.variant)
    write_csv(output_dir / "cdr_distance_bins_summary.csv", summary)
    plot_summary(summary, output_dir / "cdr_distance_bins")
    write_report(output_dir / "cdr_distance_bins_report.md", summary, args.input, args.variant)
    print(f"WROTE {output_dir / 'cdr_distance_bins.png'}")


if __name__ == "__main__":
    main()
