from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp_weighting_revalidate_20260701"

BASE = "frozengrace_gb_mlp_softdist_beta05"
ELLIPSOID = "frozengrace_gb_mlp_softdist_ellipsoid_beta05"
GMM_RESP = "frozengrace_gb_mlp_softdist_gmmresp_beta05"
ELLIPSOID_SHUFFLED = "frozengrace_gb_mlp_softdist_ellipsoid_beta05_shuffled"

METHODS = (BASE, ELLIPSOID, GMM_RESP, ELLIPSOID_SHUFFLED)
METHOD_LABELS = {
    BASE: "beta=0.5 baseline",
    ELLIPSOID: "ellipsoid + beta=0.5",
    GMM_RESP: "gmm_resp + beta=0.5",
    ELLIPSOID_SHUFFLED: "ellipsoid shuffled + beta=0.5",
}


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


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
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": "uniform",
        "noise_rate": 0.4,
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "source": str(source),
    }


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (row["dataset"], row["method"], row["seed"])
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} from "
                    f"{previous.get('source', '')} vs {row['test_acc']} from {row.get('source', '')}"
                )
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda item: (str(item["dataset"]), str(item["method"]), int(item["seed"])))


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith(("mlp_weighting_revalidate_", "paired_", "decision_")):
            continue
        for row in read_csv(path):
            normalized = normalize_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(float(row["test_acc"]))
    output = []
    for (dataset, method), values in groups.items():
        output.append(
            {
                "dataset": dataset,
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "method": method,
                "method_label": METHOD_LABELS[str(method)],
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda item: (str(item["dataset"]), str(item["method"])))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test_acc"])
        for row in rows
    }
    datasets = sorted({row["dataset"] for row in rows})
    comparisons = (
        ("ellipsoid_minus_beta05", ELLIPSOID, BASE),
        ("gmmresp_minus_beta05", GMM_RESP, BASE),
        ("ellipsoid_minus_shuffled", ELLIPSOID, ELLIPSOID_SHUFFLED),
    )
    output = []
    for dataset in datasets:
        seeds = sorted({seed for d, _, seed in values if d == dataset})
        for name, left_method, right_method in comparisons:
            deltas = []
            used_seeds = []
            for seed in seeds:
                left = (dataset, left_method, seed)
                right = (dataset, right_method, seed)
                if left in values and right in values:
                    deltas.append(values[left] - values[right])
                    used_seeds.append(seed)
            if not deltas:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "noise_type": "uniform",
                    "noise_rate": 0.4,
                    "comparison": name,
                    "left_method": left_method,
                    "right_method": right_method,
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(str(seed) for seed in used_seeds),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
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


def decide(deltas: list[dict[str, object]], expected_datasets: list[str]) -> list[dict[str, object]]:
    by_key = {(row["dataset"], row["comparison"]): row for row in deltas}
    rows = []
    for method_name, comparison in (
        ("ellipsoid+beta05", "ellipsoid_minus_beta05"),
        ("gmm_resp+beta05", "gmmresp_minus_beta05"),
    ):
        complete = [dataset for dataset in expected_datasets if (dataset, comparison) in by_key]
        positives = [
            dataset
            for dataset in complete
            if float(by_key[(dataset, comparison)]["mean_delta"]) > 0.0
        ]
        rows.append(
            {
                "method_family": method_name,
                "comparison": comparison,
                "complete_datasets": " ".join(complete),
                "positive_datasets": " ".join(positives),
                "positive_count": len(positives),
                "total_count": len(complete),
                "seed0_signal": int(len(complete) > 0 and len(positives) == len(complete)),
            }
        )
    return rows


def render_report(
    aggregate_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    run_dir: Path,
) -> str:
    lines = [
        "# MLP weighting revalidation report",
        "",
        "Scope: Cora/CiteSeer, uniform noise 0.4, percent split 0.2/0.1/0.7, seed0 smoke unless n>1.",
        "",
        "## Accuracy",
        "",
        "| dataset | method | n | mean | std |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | {row['n']} | "
            f"{float(row['mean']):.4f} | {float(row['std']):.4f} |"
        )
    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "| dataset | comparison | n | mean_delta | positive | seed_deltas |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['n']} | "
            f"{float(row['mean_delta']):+.4f} | {row['positive_count']}/{row['n']} | {row['seed_deltas']} |"
        )
    lines.extend([
        "",
        "## Seed0 Decision",
        "",
        "| family | complete datasets | positive datasets | seed0 signal |",
        "|---|---|---|---:|",
    ])
    for row in decision_rows:
        lines.append(
            f"| {row['method_family']} | {row['complete_datasets']} | "
            f"{row['positive_datasets']} | {row['seed0_signal']} |"
        )

    if any(int(row["seed0_signal"]) == 1 for row in decision_rows):
        conclusion = "At least one weighting family is positive on all completed seed0 datasets; run 5-seed confirmation before changing the main method."
    else:
        conclusion = "No weighting family beats beta=0.5 on all completed seed0 datasets; keep beta=0.5 as main unless a 5-seed rerun is explicitly requested."
    lines.extend([
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"Source directory: `{run_dir}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP weighting revalidation runs.")
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    expected_datasets = parse_list(args.datasets)
    expected_seeds = parse_seed_list(args.seeds)
    rows = load_rows(args.run_dir)
    if args.require_complete:
        observed = {(row["dataset"], row["method"], int(row["seed"])) for row in rows}
        missing = [
            (dataset, method, seed)
            for dataset in expected_datasets
            for method in METHODS
            for seed in expected_seeds
            if (dataset, method, seed) not in observed
        ]
        if missing:
            raise SystemExit(f"missing required runs: {missing[:10]} total={len(missing)}")

    aggregate_rows = aggregate(rows)
    delta_rows = paired_deltas(rows)
    decision_rows = decide(delta_rows, expected_datasets)

    write_csv(args.run_dir / "mlp_weighting_revalidate_aggregate.csv", aggregate_rows)
    write_csv(args.run_dir / "mlp_weighting_revalidate_paired_deltas.csv", delta_rows)
    write_csv(args.run_dir / "mlp_weighting_revalidate_decision.csv", decision_rows)
    report = render_report(aggregate_rows, delta_rows, decision_rows, args.run_dir)
    (args.run_dir / "mlp_weighting_revalidate_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
