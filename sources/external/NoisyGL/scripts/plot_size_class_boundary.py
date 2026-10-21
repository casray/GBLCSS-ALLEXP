from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DATASET_META = {
    "cora": {"label": "Cora", "nodes": 2708, "classes": 7, "status": "success"},
    "citeseer": {"label": "CiteSeer", "nodes": 3327, "classes": 6, "status": "success"},
    "coauthorcs": {"label": "Coauthor-CS", "nodes": 18333, "classes": 15, "status": "success"},
    "amazonpho": {"label": "Amazon-Photo", "nodes": 7650, "classes": 8, "status": "fail"},
    "amazoncom": {"label": "Amazon-Computers", "nodes": 13752, "classes": 10, "status": "partial"},
    "dblp": {"label": "DBLP", "nodes": 17716, "classes": 4, "status": "fail"},
    "pubmed": {"label": "PubMed", "nodes": 19717, "classes": 3, "status": "fail"},
}

STATUS_STYLE = {
    "success": {"color": "#2ca25f", "marker": "o", "text": "success"},
    "partial": {"color": "#f0ad4e", "marker": "s", "text": "partial"},
    "fail": {"color": "#de2d26", "marker": "X", "text": "fail"},
}

METHOD_LABELS = {
    "gcn": "GCN+CE",
    "gcn_xh_ce": "GCN+CE [X||H]",
    "nrgnn": "NRGNN",
    "nrgnn_xh": "NRGNN [X||H]",
    "frozengrace_gb_gcn_softdist_xh": "SoftDist [X||H]",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "SoftDist [X||H] shuffled",
}

PAIRS = [
    ("frozengrace_gb_gcn_softdist_xh", "gcn_xh_ce"),
    ("frozengrace_gb_gcn_softdist_xh", "frozengrace_gb_gcn_softdist_xh_shuffled"),
    ("frozengrace_gb_gcn_softdist_xh", "nrgnn_xh"),
    ("nrgnn_xh", "nrgnn"),
    ("gcn_xh_ce", "gcn"),
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _method_from_row(row: dict[str, str]) -> str:
    return row.get("method") or row.get("base_method") or ""


def _dataset_from_row(row: dict[str, str]) -> str:
    return (row.get("dataset") or "").lower()


def _seed_from_row(row: dict[str, str]) -> int | None:
    value = row.get("seed")
    if value in (None, ""):
        return None
    return int(value)


def _test_from_row(row: dict[str, str]) -> float | None:
    value = row.get("test")
    if value in (None, ""):
        return None
    return float(value)


def collect_results(paths: list[Path]) -> dict[tuple[str, str], dict[int, float]]:
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for path in paths:
        for row in _read_rows(path):
            dataset = _dataset_from_row(row)
            method = _method_from_row(row)
            seed = _seed_from_row(row)
            test = _test_from_row(row)
            if not dataset or not method or seed is None or test is None:
                continue
            values[(dataset, method)][seed] = test
    return values


def mean_std(items: list[float]) -> tuple[float, float]:
    if not items:
        return math.nan, math.nan
    mean = statistics.fmean(items)
    if len(items) == 1:
        return mean, 0.0
    return mean, statistics.stdev(items)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(values: dict[tuple[str, str], dict[int, float]]) -> list[dict[str, object]]:
    rows = []
    for (dataset, method), by_seed in sorted(values.items()):
        tests = [by_seed[seed] for seed in sorted(by_seed)]
        mean, std = mean_std(tests)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n": len(tests),
                "mean": mean,
                "std": std,
                "seeds": " ".join(f"{value:.4f}" for value in tests),
            }
        )
    return rows


def paired_delta_rows(values: dict[tuple[str, str], dict[int, float]]) -> list[dict[str, object]]:
    rows = []
    datasets = sorted({dataset for dataset, _ in values})
    for dataset in datasets:
        for left, right in PAIRS:
            left_values = values.get((dataset, left), {})
            right_values = values.get((dataset, right), {})
            seeds = sorted(set(left_values) & set(right_values))
            if not seeds:
                continue
            deltas = [left_values[seed] - right_values[seed] for seed in seeds]
            mean, std = mean_std(deltas)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{left}_minus_{right}",
                    "left_label": METHOD_LABELS.get(left, left),
                    "right_label": METHOD_LABELS.get(right, right),
                    "n": len(deltas),
                    "mean_delta": mean,
                    "std_delta": std,
                    "positive_seed_count": sum(delta > 0 for delta in deltas),
                    "deltas": " ".join(f"{delta:.4f}" for delta in deltas),
                }
            )
    return rows


def _format_mean_std(rows: list[dict[str, object]], dataset: str, method: str) -> str:
    for row in rows:
        if row["dataset"] == dataset and row["method"] == method:
            return f"{float(row['mean']):.4f} ± {float(row['std']):.4f}"
    return "NA"


def _format_delta(rows: list[dict[str, object]], dataset: str, comparison: str) -> str:
    for row in rows:
        if row["dataset"] == dataset and row["comparison"] == comparison:
            return (
                f"{float(row['mean_delta']) * 100:+.2f}pp "
                f"({int(row['positive_seed_count'])}/{int(row['n'])})"
            )
    return "NA"


def plot_boundary(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=180)
    for key, meta in DATASET_META.items():
        style = STATUS_STYLE[meta["status"]]
        ax.scatter(
            meta["nodes"],
            meta["classes"],
            s=130,
            c=style["color"],
            marker=style["marker"],
            edgecolors="#222222",
            linewidths=0.7,
            zorder=3,
        )
        offset = (-78, 5) if meta["nodes"] > 15000 and meta["classes"] > 10 else (6, 5)
        ax.annotate(
            meta["label"],
            (meta["nodes"], meta["classes"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    for status, style in STATUS_STYLE.items():
        ax.scatter([], [], s=95, c=style["color"], marker=style["marker"], label=style["text"])
    ax.set_xscale("log")
    ax.set_xlim(2400, 24500)
    ax.set_ylim(2.4, 15.6)
    ax.set_xlabel("Number of nodes (log scale)")
    ax.set_ylabel("Number of classes")
    ax.set_title("Size x class-count boundary of ball gains")
    ax.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def boundary_rows() -> list[dict[str, object]]:
    note = {
        "cora": "SoftDist > GCN[X||H] and shuffled.",
        "citeseer": "SoftDist > GCN[X||H] and shuffled.",
        "coauthorcs": "Large multi-class success.",
        "amazonpho": "Healthy balls but real <= shuffled / GCN[X||H].",
        "amazoncom": "Real > shuffled, but absorbed by GCN[X||H].",
        "dblp": "Large low-class graph; soft target not better than shuffled.",
        "pubmed": "Low-class strong-baseline regime; denoising methods do not beat GCN.",
    }
    rows = []
    for key, meta in DATASET_META.items():
        rows.append(
            {
                "dataset": key,
                "label": meta["label"],
                "nodes": meta["nodes"],
                "classes": meta["classes"],
                "status": meta["status"],
                "note": note[key],
            }
        )
    return rows


def write_report(path: Path, aggregate: list[dict[str, object]], deltas: list[dict[str, object]]) -> None:
    datasets = ["cora", "citeseer", "coauthorcs", "amazoncom"]
    methods = [
        "gcn",
        "gcn_xh_ce",
        "nrgnn",
        "nrgnn_xh",
        "frozengrace_gb_gcn_softdist_xh",
        "frozengrace_gb_gcn_softdist_xh_shuffled",
    ]
    lines = [
        "# Phase 36 Coauthor-CS main-table baseline report",
        "",
        "Protocol: percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4.",
        "",
        "## Aggregate",
        "",
        "| Dataset | Method | Test acc |",
        "|---|---|---:|",
    ]
    for dataset in datasets:
        for method in methods:
            value = _format_mean_std(aggregate, dataset, method)
            if value != "NA":
                lines.append(f"| {dataset} | {METHOD_LABELS.get(method, method)} | {value} |")
    lines.extend(["", "## Paired deltas", "", "| Dataset | Comparison | Delta |", "|---|---|---:|"])
    for dataset in datasets:
        for left, right in PAIRS:
            comparison = f"{left}_minus_{right}"
            value = _format_delta(deltas, dataset, comparison)
            if value != "NA":
                lines.append(
                    f"| {dataset} | {METHOD_LABELS.get(left, left)} - {METHOD_LABELS.get(right, right)} | {value} |"
                )
    lines.extend(
        [
            "",
            "## Boundary interpretation",
            "",
            "- Coauthor-CS tests whether large graphs necessarily fail. A positive ball gain here rules out graph size as the sole failure variable.",
            "- Amazon-Computers is retained as a partial boundary point: real balls beat shuffled balls, but the full-sample GCN [X||H] absorbs the gain.",
            "- The resulting boundary is size x class count x graph regime, not a one-dimensional small-graph-only story.",
            "",
            "Figure: `size_class_boundary.png`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_inputs() -> list[Path]:
    return [
        REPO_ROOT / "runs" / "phase24_nrgnn_xh_20260616" / "phase24_fairness_combined_raw.csv",
        REPO_ROOT / "runs" / "coauthorcs_20260616" / "softdist_xh_5seed.csv",
        REPO_ROOT / "runs" / "coauthorcs_baseline_20260617" / "coauthorcs_baselines_5seed.csv",
        REPO_ROOT / "runs" / "amazoncom_20260616" / "softdist_xh_5seed.csv",
        REPO_ROOT / "runs" / "amazoncom_baseline_20260617" / "amazoncom_baselines_5seed.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 36 and plot size x class-count boundary.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "figures_20260617"))
    parser.add_argument("--inputs", nargs="*", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    inputs = [Path(path) for path in args.inputs] if args.inputs else default_inputs()
    values = collect_results(inputs)
    aggregate = aggregate_rows(values)
    deltas = paired_delta_rows(values)
    write_csv(output_dir / "phase36_aggregate.csv", aggregate)
    write_csv(output_dir / "phase36_paired_deltas.csv", deltas)
    write_csv(output_dir / "size_class_boundary_points.csv", boundary_rows())
    plot_boundary(output_dir / "size_class_boundary.png")
    write_report(output_dir / "phase36_coauthorcs_maintable_report.md", aggregate, deltas)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
