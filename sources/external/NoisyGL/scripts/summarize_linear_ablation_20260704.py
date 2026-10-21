from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


NEW_METHODS = (
    "linear_ce",
    "frozengrace_gb_linear_softdist_beta05",
    "frozengrace_gb_linear_softdist_beta05_shuffled",
)
REFERENCE_METHODS = (
    "mlp_ce",
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
)
METHOD_LABELS = {
    "linear_ce": "Linear+CE",
    "frozengrace_gb_linear_softdist_beta05": "SoftDist-Linear beta=0.5",
    "frozengrace_gb_linear_softdist_beta05_shuffled": "SoftDist-Linear shuffled",
    "mlp_ce": "MLP+CE",
    "frozengrace_gb_mlp_softdist_beta05": "SoftDist-MLP beta=0.5",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "SoftDist-MLP shuffled",
}
COMPARISONS = (
    ("softdist_linear_vs_linear_ce", "frozengrace_gb_linear_softdist_beta05", "linear_ce"),
    ("softdist_linear_vs_linear_shuffled", "frozengrace_gb_linear_softdist_beta05", "frozengrace_gb_linear_softdist_beta05_shuffled"),
    ("softdist_mlp_vs_softdist_linear", "frozengrace_gb_mlp_softdist_beta05", "frozengrace_gb_linear_softdist_beta05"),
    ("softdist_linear_vs_mlp_ce", "frozengrace_gb_linear_softdist_beta05", "mlp_ce"),
    ("softdist_mlp_vs_mlp_ce", "frozengrace_gb_mlp_softdist_beta05", "mlp_ce"),
    ("softdist_mlp_vs_mlp_shuffled", "frozengrace_gb_mlp_softdist_beta05", "frozengrace_gb_mlp_softdist_beta05_shuffled"),
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_value(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _seed_value(value) -> int:
    return int(float(value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _format_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "NA"
    return f"{mean:.4f}±{std:.4f}"


def _load_new_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("linear_ablation_uniform0p4_seeds_*.csv")):
        for row in _read_csv(path):
            if row.get("status", "ok") != "ok":
                continue
            if row.get("method") not in NEW_METHODS:
                continue
            if row.get("noise_type") != "uniform":
                continue
            if abs(_float_value(row.get("noise_rate")) - 0.4) > 1e-9:
                continue
            rows.append({
                "dataset": row.get("dataset", "").lower(),
                "method": row["method"],
                "seed": _seed_value(row["seed"]),
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "test_acc": _float_value(row.get("test")),
                "source": str(path),
            })
    return rows


def _load_reference_rows(maintable_path: Path) -> list[dict]:
    rows: list[dict] = []
    for row in _read_csv(maintable_path):
        if row.get("table") != "main":
            continue
        if abs(_float_value(row.get("train_percent")) - 0.2) > 1e-9:
            continue
        if row.get("method") not in REFERENCE_METHODS:
            continue
        if row.get("noise_type") != "uniform":
            continue
        if abs(_float_value(row.get("noise_rate")) - 0.4) > 1e-9:
            continue
        rows.append({
            "dataset": row.get("dataset", "").lower(),
            "method": row["method"],
            "seed": _seed_value(row["seed"]),
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "test_acc": _float_value(row.get("test_acc")),
            "source": str(maintable_path),
        })
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(float(row["test_acc"]))
    output = []
    for (dataset, method), values in sorted(grouped.items()):
        mean = _mean(values)
        std = _std(values)
        output.append({
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n": len(values),
            "mean": mean,
            "std": std,
            "mean_std": _format_mean_std(mean, std),
        })
    return output


def _paired_deltas(rows: list[dict]) -> list[dict]:
    lookup = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test_acc"])
        for row in rows
    }
    output = []
    for dataset, seed in sorted({(row["dataset"], int(row["seed"])) for row in rows}):
        for comparison, left, right in COMPARISONS:
            left_key = (dataset, left, seed)
            right_key = (dataset, right, seed)
            if left_key not in lookup or right_key not in lookup:
                continue
            output.append({
                "dataset": dataset,
                "seed": seed,
                "comparison": comparison,
                "left_method": left,
                "right_method": right,
                "left_acc": lookup[left_key],
                "right_acc": lookup[right_key],
                "delta": lookup[left_key] - lookup[right_key],
            })
    return output


def _delta_summary(delta_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in delta_rows:
        grouped[(row["dataset"], row["comparison"])].append(float(row["delta"]))
    output = []
    for (dataset, comparison), values in sorted(grouped.items()):
        mean = _mean(values)
        std = _std(values)
        output.append({
            "dataset": dataset,
            "comparison": comparison,
            "n": len(values),
            "mean_delta": mean,
            "std_delta": std,
            "positive_seeds": sum(1 for value in values if value > 0),
            "mean_delta_pp": mean * 100.0,
        })
    return output


def _completeness(rows: list[dict], delta_rows: list[dict]) -> list[dict]:
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({*NEW_METHODS, *REFERENCE_METHODS})
    seeds = sorted({int(row["seed"]) for row in rows})
    observed = {(row["dataset"], row["method"], int(row["seed"])) for row in rows}
    expected = {(dataset, method, seed) for dataset in datasets for method in methods for seed in seeds}
    return [{
        "datasets": ",".join(datasets),
        "methods": ",".join(methods),
        "seeds": ",".join(str(seed) for seed in seeds),
        "observed_rows": len(observed),
        "expected_rows": len(expected),
        "missing_rows": len(expected - observed),
        "paired_delta_rows": len(delta_rows),
    }]


def _build_report(aggregate_rows: list[dict], delta_summary_rows: list[dict], completeness_rows: list[dict]) -> str:
    lines = [
        "# SoftDist-Linear Ablation Report",
        "",
        "Protocol: Cora/CiteSeer/PubMed, uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, seeds 0-4.",
        "",
        "## Mean ± Std",
        "",
        "| Dataset | Method | Test Acc | n |",
        "|---|---|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(f"| {row['dataset']} | {row['method_label']} | {row['mean_std']} | {row['n']} |")
    lines.extend([
        "",
        "## Paired Delta Summary",
        "",
        "| Dataset | Comparison | Mean Delta (pp) | Positive Seeds | n |",
        "|---|---|---:|---:|---:|",
    ])
    for row in delta_summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | "
            f"{float(row['mean_delta_pp']):+.2f} | {row['positive_seeds']} | {row['n']} |"
        )
    completeness = completeness_rows[0]
    lines.extend([
        "",
        "## Completeness",
        "",
        f"- Observed rows: {completeness['observed_rows']} / expected {completeness['expected_rows']}",
        f"- Missing rows: {completeness['missing_rows']}",
        f"- Paired delta rows: {completeness['paired_delta_rows']}",
        "",
        "## Interpretation Guardrails",
        "",
        "- If SoftDist-Linear beats Linear+CE and shuffled, the soft target mechanism works without nonlinear classifier capacity.",
        "- If SoftDist-MLP beats SoftDist-Linear, report that the X-only student needs mild nonlinear capacity to absorb GRACE-derived soft supervision.",
        "- Keep GRACE-H direct classifiers in the separate H-access scope table; this table is an X-only classifier-capacity ablation.",
    ])
    return "\n".join(lines) + "\n"


def summarize(run_dir: Path, maintable_path: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    maintable_path = Path(maintable_path)
    rows = _load_new_rows(run_dir) + _load_reference_rows(maintable_path)
    aggregate_rows = _aggregate(rows)
    delta_rows = _paired_deltas(rows)
    delta_summary_rows = _delta_summary(delta_rows)
    completeness_rows = _completeness(rows, delta_rows)

    aggregate_path = run_dir / "linear_ablation_aggregate.csv"
    delta_path = run_dir / "linear_ablation_paired_deltas.csv"
    delta_summary_path = run_dir / "linear_ablation_delta_summary.csv"
    completeness_path = run_dir / "linear_ablation_completeness.csv"
    report_path = run_dir / "linear_ablation_report.md"

    _write_csv(aggregate_path, aggregate_rows)
    _write_csv(delta_path, delta_rows)
    _write_csv(delta_summary_path, delta_summary_rows)
    _write_csv(completeness_path, completeness_rows)
    report_path.write_text(_build_report(aggregate_rows, delta_summary_rows, completeness_rows), encoding="utf-8")
    return {
        "aggregate": aggregate_path,
        "paired_deltas": delta_path,
        "delta_summary": delta_summary_path,
        "completeness": completeness_path,
        "report": report_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize SoftDist-Linear ablation.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--maintable", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name, path in summarize(args.run_dir, args.maintable).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
