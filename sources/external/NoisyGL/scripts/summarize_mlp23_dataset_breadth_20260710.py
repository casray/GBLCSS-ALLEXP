from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


RUN_DIR = Path("runs/mlp23_dataset_breadth_20260710")
DATASETS = ("coraml", "attribwiki", "citeseerfull", "coauthorph")
SEEDS = tuple(range(5))
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
BASELINES = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_ct",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_logitclip",
    "mlp_anl",
    "mlp_fine",
    "mlp_dividemix_lite",
)
METHODS = (OURS, SHUFFLED, *BASELINES)


def _float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def protocol_matches(row: dict[str, str]) -> bool:
    return (
        (row.get("split_type") or "percent").strip().lower() == "percent"
        and math.isclose(_float(row.get("train_percent"), 0.2), 0.2)
        and math.isclose(_float(row.get("val_percent"), 0.1), 0.1)
        and math.isclose(_float(row.get("test_percent"), 0.7), 0.7)
        and (row.get("noise_scope") or "train").strip().lower() == "train"
        and (row.get("noise_type") or "uniform").strip().lower() == "uniform"
        and math.isclose(_float(row.get("noise_rate"), 0.4), 0.4)
    )


def load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        if not path.exists() or path.name.startswith("mlp23_"):
            continue
        with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
            for raw in csv.DictReader(stream):
                dataset = (raw.get("dataset") or "").strip().lower()
                method = (raw.get("method") or "").strip()
                seed = _int(raw.get("seed"), -1)
                if dataset not in DATASETS or method not in METHODS or seed not in SEEDS:
                    continue
                if (raw.get("status") or "ok").strip().lower() not in {"", "ok"}:
                    continue
                if not protocol_matches(raw):
                    continue
                key = (dataset, method, seed)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "test": _float(raw.get("test") or raw.get("test_acc")),
                        "modified_count": _int(raw.get("modified_count")),
                        "modified_train_count": _int(raw.get("modified_train_count")),
                        "modified_val_count": _int(raw.get("modified_val_count")),
                        "modified_test_count": _int(raw.get("modified_test_count")),
                        "source": str(path),
                    }
                )
    return sorted(rows, key=lambda row: (row["dataset"], row["method"], row["seed"]))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(float(row["test"]))
    return [
        {
            "dataset": dataset,
            "method": method,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for (dataset, method), values in sorted(groups.items())
    ]


def _paired(rows: list[dict[str, object]], dataset: str, left: str, right: str) -> list[float]:
    lookup = {
        (str(row["dataset"]), str(row["method"]), int(row["seed"])): float(row["test"])
        for row in rows
    }
    return [
        lookup[(dataset, left, seed)] - lookup[(dataset, right, seed)]
        for seed in SEEDS
        if (dataset, left, seed) in lookup and (dataset, right, seed) in lookup
    ]


def decide_gate(rows: list[dict[str, object]], dataset: str) -> dict[str, object]:
    means: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["dataset"] == dataset:
            means[str(row["method"])].append(float(row["test"]))
    available = [method for method in BASELINES if len(means[method]) == 5]
    best = max(available, key=lambda method: statistics.mean(means[method])) if available else ""
    vs_shuffled = _paired(rows, dataset, OURS, SHUFFLED)
    vs_best = _paired(rows, dataset, OURS, best) if best else []
    structure_pass = len(vs_shuffled) == 5 and statistics.mean(vs_shuffled) >= 0.01 and sum(x > 0 for x in vs_shuffled) >= 4
    baseline_pass = len(vs_best) == 5 and statistics.mean(vs_best) >= 0.01 and sum(x > 0 for x in vs_best) >= 4
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    audit_pass = bool(dataset_rows) and all(
        int(row.get("modified_val_count", 0)) == 0
        and int(row.get("modified_test_count", 0)) == 0
        and int(row.get("modified_count", 0)) == int(row.get("modified_train_count", 0))
        and math.isfinite(float(row["test"]))
        for row in dataset_rows
    )
    complete = len(dataset_rows) == len(METHODS) * len(SEEDS)
    if complete and audit_pass and structure_pass and baseline_pass:
        decision = "pass"
    elif complete and audit_pass and structure_pass:
        decision = "boundary"
    elif not dataset_rows:
        decision = "missing"
    else:
        decision = "fail"
    return {
        "dataset": dataset,
        "decision": decision,
        "best_baseline": best,
        "ours_mean": statistics.mean(means[OURS]) if means[OURS] else math.nan,
        "best_baseline_mean": statistics.mean(means[best]) if best else math.nan,
        "vs_shuffled_mean_delta": statistics.mean(vs_shuffled) if vs_shuffled else math.nan,
        "vs_shuffled_positive": sum(x > 0 for x in vs_shuffled),
        "vs_best_mean_delta": statistics.mean(vs_best) if vs_best else math.nan,
        "vs_best_positive": sum(x > 0 for x in vs_best),
        "complete": int(complete),
        "audit_pass": int(audit_pass),
    }


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for dataset in DATASETS:
        for baseline in (SHUFFLED, *BASELINES):
            values = _paired(rows, dataset, OURS, baseline)
            if values:
                out.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "n": len(values),
                        "mean_delta": statistics.mean(values),
                        "std_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "positive_count": sum(value > 0 for value in values),
                        "seed_deltas": " ".join(f"{value:.6f}" for value in values),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, decisions: list[dict[str, object]], agg: list[dict[str, object]]) -> None:
    lines = [
        "# MLP-23 Dataset Breadth Report",
        "",
        "Protocol: percent 0.2/0.1/0.7, train-only uniform 0.4, seeds 0-4.",
        "",
        "## Gate Decisions",
        "",
        "| Dataset | Decision | Ours | Best baseline | Ours-best | Positive | Ours-shuffled | Positive | Complete/Audit |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        best = row["best_baseline"] or "-"
        best_value = row["best_baseline_mean"]
        best_text = f"{best} {100 * best_value:.2f}" if math.isfinite(best_value) else best
        lines.append(
            f"| {row['dataset']} | {row['decision']} | {100 * row['ours_mean']:.2f} | {best_text} | "
            f"{100 * row['vs_best_mean_delta']:+.2f}pp | {row['vs_best_positive']}/5 | "
            f"{100 * row['vs_shuffled_mean_delta']:+.2f}pp | {row['vs_shuffled_positive']}/5 | "
            f"{row['complete']}/{row['audit_pass']} |"
        )
    lines.extend(["", "## Uniform 0.4 Aggregate", "", "| Dataset | Method | n | Mean±Std |", "|---|---|---:|---:|"])
    for row in sorted(agg, key=lambda item: (item["dataset"], -float(item["mean"]))):
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {100 * row['mean']:.2f}±{100 * row['std']:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.run_dir.glob("*.csv"))
    rows = load_rows(inputs)
    agg = aggregate(rows)
    decisions = [decide_gate(rows, dataset) for dataset in DATASETS]
    deltas = paired_deltas(rows)
    write_csv(args.run_dir / "mlp23_raw.csv", rows)
    write_csv(args.run_dir / "mlp23_aggregate.csv", agg)
    write_csv(args.run_dir / "mlp23_gate_decisions.csv", decisions)
    write_csv(args.run_dir / "mlp23_paired_deltas.csv", deltas)
    write_report(args.run_dir / "mlp23_report.md", decisions, agg)


if __name__ == "__main__":
    main()
