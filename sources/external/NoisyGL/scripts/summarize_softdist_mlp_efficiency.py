from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = (
    "mlp_ce",
    "softdist_mlp",
    "gcn",
    "softdist_gcn",
    "gcn_xh_ce",
    "softdist_xh",
    "nrgnn",
)
METRICS = (
    "init_time_sec",
    "ball_target_time_sec",
    "classifier_train_time_sec",
    "downstream_total_time_sec",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "test",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def normalize_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for path in paths:
        for raw in read_csv(path):
            if raw.get("status") != "ok" or raw.get("method") not in METHOD_ORDER:
                continue
            key = (raw["dataset"], raw["method"], int(raw["seed"]))
            if key in seen:
                raise ValueError(f"duplicate efficiency row: {key}")
            seen.add(key)
            row = {
                "dataset": raw["dataset"].lower(),
                "method": raw["method"],
                "seed": int(raw["seed"]),
                "trainable_params": int(raw["trainable_params"]),
            }
            for metric in METRICS:
                row[metric] = float(raw[metric])
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            METHOD_ORDER.index(row["method"]),
            row["seed"],
        ),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)
    output = []
    for (dataset, method), values in groups.items():
        result = {
            "dataset": dataset,
            "method": method,
            "n": len(values),
            "trainable_params": int(values[0]["trainable_params"]),
        }
        for metric in METRICS:
            metric_values = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.mean(metric_values)
            result[f"{metric}_std"] = (
                statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            )
        output.append(result)
    return sorted(
        output,
        key=lambda row: (row["dataset"], METHOD_ORDER.index(row["method"])),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(aggregate: list[dict[str, object]], missing: set[tuple[str, str, int]]) -> str:
    lines = [
        "# SoftDist-MLP Efficiency Report",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        f"- Status: {'INCOMPLETE' if missing else 'COMPLETED'}",
        "- Protocol: Cora/CiteSeer, uniform 0.4, seeds 0-4, ROCm `cuda:0`",
        "- GRACE pretraining is excluded and must be reported as reusable preprocessing.",
        "",
        "| Dataset | Method | Params | Init(s) | Ball target(s) | Classifier train(s) | Downstream total(s) | Peak alloc(MiB) | Test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['trainable_params']} | "
            f"{row['init_time_sec_mean']:.3f}+/-{row['init_time_sec_std']:.3f} | "
            f"{row['ball_target_time_sec_mean']:.3f}+/-{row['ball_target_time_sec_std']:.3f} | "
            f"{row['classifier_train_time_sec_mean']:.3f}+/-{row['classifier_train_time_sec_std']:.3f} | "
            f"{row['downstream_total_time_sec_mean']:.3f}+/-{row['downstream_total_time_sec_std']:.3f} | "
            f"{row['peak_allocated_bytes_mean'] / 2**20:.1f} | "
            f"{100 * row['test_mean']:.2f}+/-{100 * row['test_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- This table supports downstream parameter/time/memory comparisons only.",
            "- It does not support an end-to-end complexity win because GRACE pretraining is excluded.",
            "- SoftDist-MLP may be called lightweight downstream only when its measured classifier cost is below the GCN and NRGNN controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SoftDist-MLP efficiency runs.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "softdist_mlp_efficiency_20260618",
    )
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    rows = normalize_rows(sorted(args.input_dir.glob("*.csv")))
    expected = {
        (dataset, method, seed)
        for dataset in ("cora", "citeseer")
        for method in METHOD_ORDER
        for seed in range(5)
    }
    present = {(row["dataset"], row["method"], row["seed"]) for row in rows}
    missing = expected - present
    aggregate = aggregate_rows(rows)
    write_csv(args.input_dir / "softdist_mlp_efficiency_raw.csv", rows)
    write_csv(args.input_dir / "softdist_mlp_efficiency_aggregate.csv", aggregate)
    (args.input_dir / "softdist_mlp_efficiency_report.md").write_text(
        build_report(aggregate, missing), encoding="utf-8"
    )
    print(f"rows={len(rows)} aggregate={len(aggregate)} missing={len(missing)}")
    if args.require_complete and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
