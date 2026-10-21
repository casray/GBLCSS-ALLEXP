from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_METHODS = (
    "gcn",
    "nrgnn",
    "gcn_xh_ce",
    "frozengrace_gb_gcn_softdist_xh",
    "frozengrace_gb_gcn_softdist_xh_shuffled",
)
METHOD_LABELS = {
    "gcn": "GCN+CE",
    "nrgnn": "NRGNN",
    "gcn_xh_ce": "GCN+CE [X||H]",
    "frozengrace_gb_gcn_softdist_xh": "SoftDist [X||H]",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "SoftDist [X||H] shuffled",
}
COMPARISONS = (
    ("softdist_vs_gcn_xh", "frozengrace_gb_gcn_softdist_xh", "gcn_xh_ce"),
    (
        "softdist_vs_shuffled",
        "frozengrace_gb_gcn_softdist_xh",
        "frozengrace_gb_gcn_softdist_xh_shuffled",
    ),
    ("softdist_vs_nrgnn", "frozengrace_gb_gcn_softdist_xh", "nrgnn"),
    ("gcn_xh_vs_gcn", "gcn_xh_ce", "gcn"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def normalize_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in TARGET_METHODS:
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_noise_rate": _optional_float(
            row.get("actual_train_noise_rate") or row.get("actual_noise_rate") or row.get("actual_noise")
        ),
        "source": str(source.relative_to(REPO_ROOT)),
    }


def _optional_float(value: str | None) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["noise_type"],
            row["noise_rate"],
            row["seed"],
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} from {previous['source']} "
                    f"vs {row['test_acc']} from {row['source']}"
                )
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda item: (
            str(item["dataset"]),
            str(item["noise_type"]),
            float(item["noise_rate"]),
            TARGET_METHODS.index(str(item["method"])),
            int(item["seed"]),
        ),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["noise_type"], row["noise_rate"], row["method"], row["method_label"])
        groups[key].append(float(row["test_acc"]))
    output = []
    for (dataset, noise_type, noise_rate, method, method_label), values in groups.items():
        output.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "method": method,
                "method_label": method_label,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(
        output,
        key=lambda item: (
            str(item["dataset"]),
            str(item["noise_type"]),
            float(item["noise_rate"]),
            TARGET_METHODS.index(str(item["method"])),
        ),
    )


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["noise_type"], row["noise_rate"], row["method"], row["seed"]): float(
            row["test_acc"]
        )
        for row in rows
    }
    regimes = sorted({(row["dataset"], row["noise_type"], row["noise_rate"]) for row in rows})
    output = []
    for dataset, noise_type, noise_rate in regimes:
        for comparison, left, right in COMPARISONS:
            deltas = []
            seeds = []
            for seed in range(5):
                left_key = (dataset, noise_type, noise_rate, left, seed)
                right_key = (dataset, noise_type, noise_rate, right, seed)
                if left_key in values and right_key in values:
                    seeds.append(seed)
                    deltas.append(values[left_key] - values[right_key])
            if not deltas:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "comparison": comparison,
                    "left": left,
                    "right": right,
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(map(str, seeds)),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                }
            )
    return output


def requested_new_keys() -> set[tuple[str, str, float, str, int]]:
    keys = set()
    for dataset in ("cora", "citeseer"):
        for method in TARGET_METHODS:
            for seed in range(5):
                keys.add((dataset, "uniform", 0.5, method, seed))
    for noise_type, noise_rate in (("uniform", 0.2), ("uniform", 0.6), ("pair", 0.4), ("instance", 0.4)):
        for method in TARGET_METHODS:
            for seed in range(5):
                keys.add(("coauthorcs", noise_type, noise_rate, method, seed))
    return keys


def filter_table(
    rows: list[dict[str, object]],
    *,
    datasets: set[str],
    noise_types: set[str],
    rates_by_dataset: dict[str, set[float]],
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if str(row["dataset"]) in datasets
        and str(row["noise_type"]) in noise_types
        and float(row["noise_rate"]) in rates_by_dataset[str(row["dataset"])]
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(new_dir: Path) -> list[dict[str, object]]:
    sources = [
        REPO_ROOT / "runs" / "noise_grid_rocm_20260611" / "noise_grid_raw_summary_20260611.csv",
        REPO_ROOT / "runs" / "instance_noise_20260612" / "instance_standard_rate0p4_summary.csv",
        REPO_ROOT / "runs" / "instance_noise_20260612" / "instance_xh_rate0p4_summary.csv",
        REPO_ROOT / "runs" / "coauthorcs_baseline_20260617" / "coauthorcs_baselines_5seed.csv",
        REPO_ROOT / "runs" / "coauthorcs_20260616" / "softdist_xh_5seed.csv",
    ]
    sources.extend(sorted(new_dir.glob("*.csv")))
    normalized = []
    for source in sources:
        for row in read_csv(source):
            item = normalize_row(row, source)
            if item is not None:
                normalized.append(item)
    return deduplicate(normalized)


def _paired_lookup(deltas: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
    return {
        (row["dataset"], row["noise_type"], row["noise_rate"], row["comparison"]): row for row in deltas
    }


def build_report(
    aggregate: list[dict[str, object]],
    deltas: list[dict[str, object]],
    missing: set[tuple[str, str, float, str, int]],
) -> str:
    delta_lookup = _paired_lookup(deltas)
    lines = [
        "# Noise Robustness Completion Report (2026-06-18)",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        f"- Status: {'INCOMPLETE' if missing else 'COMPLETED'}",
        "- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, ROCm `cuda:0`",
        "- Verification status: ANALYZED",
        "",
        "## Requested-run completeness",
        "",
        f"- Expected new per-seed rows: {len(requested_new_keys())}",
        f"- Missing new per-seed rows: {len(missing)}",
    ]
    if missing:
        for key in sorted(missing)[:20]:
            lines.append(f"- Missing: `{key}`")
    lines.extend(
        [
            "",
            "## Regime decisions",
            "",
            "The decision column evaluates the controlled granular-ball contribution only; it does not claim an overall win over NRGNN.",
            "",
            "| Dataset | Noise | Rate | SoftDist-GCN[XH] | SoftDist-shuffled | SoftDist-NRGNN | Ball contribution |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    regimes = sorted({(row["dataset"], row["noise_type"], row["noise_rate"]) for row in aggregate})
    for dataset, noise_type, noise_rate in regimes:
        first = delta_lookup.get((dataset, noise_type, noise_rate, "softdist_vs_gcn_xh"))
        second = delta_lookup.get((dataset, noise_type, noise_rate, "softdist_vs_shuffled"))
        competitor = delta_lookup.get((dataset, noise_type, noise_rate, "softdist_vs_nrgnn"))
        if not first or not second or int(first["n"]) < 5 or int(second["n"]) < 5:
            decision = "incomplete"
        elif (
            float(first["mean_delta"]) > 0
            and int(first["positive_count"]) >= 4
            and float(second["mean_delta"]) > 0
            and int(second["positive_count"]) >= 4
        ):
            decision = "retain increment"
        else:
            decision = "no stable increment"
        first_text = "NA" if not first else f"{100*float(first['mean_delta']):+.2f}pp ({first['positive_count']}/{first['n']})"
        second_text = "NA" if not second else f"{100*float(second['mean_delta']):+.2f}pp ({second['positive_count']}/{second['n']})"
        competitor_text = (
            "NA"
            if not competitor
            else f"{100*float(competitor['mean_delta']):+.2f}pp ({competitor['positive_count']}/{competitor['n']})"
        )
        lines.append(
            f"| {dataset} | {noise_type} | {float(noise_rate):.1f} | {first_text} | {second_text} | {competitor_text} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Uniform 0.5 is a predeclared transition diagnostic, not a new tuning point. Pair 0.6 is not treated as a generic high-noise setting because the designated wrong class becomes the population majority above rate 0.5. High-noise failures are retained as applicability evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the 2026-06-18 noise robustness completion runs.")
    parser.add_argument(
        "--new_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "noise_breadth_20260618",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.new_dir

    rows = collect_rows(args.new_dir)
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    present = {
        (str(row["dataset"]), str(row["noise_type"]), float(row["noise_rate"]), str(row["method"]), int(row["seed"]))
        for row in rows
    }
    missing = requested_new_keys() - present

    type_table = filter_table(
        aggregate,
        datasets={"cora", "citeseer", "coauthorcs"},
        noise_types={"uniform", "pair", "instance"},
        rates_by_dataset={"cora": {0.4}, "citeseer": {0.4}, "coauthorcs": {0.4}},
    )
    rate_table = filter_table(
        aggregate,
        datasets={"cora", "citeseer", "coauthorcs"},
        noise_types={"uniform"},
        rates_by_dataset={
            "cora": {0.2, 0.4, 0.5, 0.6},
            "citeseer": {0.2, 0.4, 0.5, 0.6},
            "coauthorcs": {0.2, 0.4, 0.6},
        },
    )

    write_csv(output_dir / "noise_robustness_raw.csv", rows)
    write_csv(output_dir / "noise_robustness_aggregate.csv", aggregate)
    write_csv(output_dir / "noise_type_robustness_rate0p4.csv", type_table)
    write_csv(output_dir / "uniform_rate_sensitivity.csv", rate_table)
    write_csv(output_dir / "noise_robustness_paired_deltas.csv", deltas)
    (output_dir / "noise_robustness_report.md").write_text(
        build_report(aggregate, deltas, missing), encoding="utf-8"
    )
    print(f"rows={len(rows)} aggregate={len(aggregate)} missing_new={len(missing)}")
    if args.require_complete and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
