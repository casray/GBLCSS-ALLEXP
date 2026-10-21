from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "mlp_ce",
    "frozengrace_gb_mlp_softdist",
    "frozengrace_gb_mlp_softdist_shuffled",
)
METHOD_LABELS = {
    "mlp_ce": "MLP(X)+CE",
    "frozengrace_gb_mlp_softdist": "SoftDist-MLP(X)",
    "frozengrace_gb_mlp_softdist_shuffled": "SoftDist-MLP(X) shuffled",
}
COMPARISONS = (
    ("softdist_vs_ce", "frozengrace_gb_mlp_softdist", "mlp_ce"),
    (
        "softdist_vs_shuffled",
        "frozengrace_gb_mlp_softdist",
        "frozengrace_gb_mlp_softdist_shuffled",
    ),
)
NEW_REGIMES = (
    ("clean", 0.0),
    ("uniform", 0.2),
    ("uniform", 0.5),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def normalize_row(row: dict[str, str], source: str | Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in METHODS:
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
            row.get("actual_train_noise_rate") or row.get("actual_noise_rate")
        ),
        "source": str(source),
    }


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(
                float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12
            ):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} "
                    f"from {previous.get('source', '')} vs {row['test_acc']} "
                    f"from {row.get('source', '')}"
                )
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda item: (
            str(item["dataset"]),
            str(item["noise_type"]),
            float(item["noise_rate"]),
            METHODS.index(str(item["method"])),
            int(item["seed"]),
        ),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            row.get("method_label", METHOD_LABELS[str(row["method"])]),
        )
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
            METHODS.index(str(item["method"])),
        ),
    )


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in rows
    }
    regimes = sorted(
        {
            (row["dataset"], row["noise_type"], float(row["noise_rate"]))
            for row in rows
        }
    )
    output = []
    for dataset, noise_type, noise_rate in regimes:
        for comparison, left, right in COMPARISONS:
            seeds = []
            deltas = []
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
                    "within_minus_1pp_count": sum(delta >= -0.01 for delta in deltas),
                    "seeds": " ".join(map(str, seeds)),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                }
            )
    return output


def evaluate_regimes(deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["comparison"],
        ): row
        for row in deltas
    }
    regimes = sorted({key[:3] for key in lookup})
    output = []
    for dataset, noise_type, noise_rate in regimes:
        vs_ce = lookup.get((dataset, noise_type, noise_rate, "softdist_vs_ce"))
        vs_shuffled = lookup.get(
            (dataset, noise_type, noise_rate, "softdist_vs_shuffled")
        )
        complete = int(
            bool(vs_ce)
            and bool(vs_shuffled)
            and int(vs_ce["n"]) == 5
            and int(vs_shuffled["n"]) == 5
        )
        stable = int(
            complete
            and float(vs_ce["mean_delta"]) > 0
            and int(vs_ce["positive_count"]) >= 4
            and float(vs_shuffled["mean_delta"]) > 0
            and int(vs_shuffled["positive_count"]) >= 4
        )
        clean_safe = ""
        if noise_type == "clean" and vs_ce:
            clean_safe = int(
                float(vs_ce["mean_delta"]) >= -0.01
                and int(vs_ce["within_minus_1pp_count"]) >= 3
            )
        output.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "complete": complete,
                "stable_increment": stable,
                "clean_safe": clean_safe,
                "vs_ce_mean_delta": "" if not vs_ce else vs_ce["mean_delta"],
                "vs_ce_positive_count": "" if not vs_ce else vs_ce["positive_count"],
                "vs_shuffled_mean_delta": (
                    "" if not vs_shuffled else vs_shuffled["mean_delta"]
                ),
                "vs_shuffled_positive_count": (
                    "" if not vs_shuffled else vs_shuffled["positive_count"]
                ),
            }
        )
    return output


def expected_new_keys() -> set[tuple[str, str, float, str, int]]:
    return {
        (dataset, noise_type, noise_rate, method, seed)
        for dataset in ("cora", "citeseer")
        for noise_type, noise_rate in NEW_REGIMES
        for method in METHODS
        for seed in range(5)
    }


def cross_noise_robust(
    decisions: list[dict[str, object]],
) -> tuple[bool, dict[str, int]]:
    counts = {"cora": 0, "citeseer": 0}
    for row in decisions:
        dataset = str(row["dataset"])
        if dataset in counts and str(row["noise_type"]) != "clean":
            counts[dataset] += int(row["stable_increment"])
    return all(counts[dataset] >= 4 for dataset in counts), counts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(new_dir: Path, anchor_path: Path) -> list[dict[str, object]]:
    sources = [anchor_path, *sorted(new_dir.glob("*.csv"))]
    normalized = []
    for source in sources:
        for row in read_csv(source):
            item = normalize_row(row, source)
            if item is not None:
                normalized.append(item)
    return deduplicate(normalized)


def build_report(
    aggregate: list[dict[str, object]],
    decisions: list[dict[str, object]],
    missing: set[tuple[str, str, float, str, int]],
) -> str:
    aggregates = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
        ): row
        for row in aggregate
    }
    robust, stable_counts = cross_noise_robust(decisions)
    lines = [
        "# SoftDist-MLP Noise Spectrum Report",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        f"- Status: {'INCOMPLETE' if missing else 'COMPLETED'}",
        "- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, ROCm `cuda:0`",
        "- Verification status: ANALYZED",
        "",
        "## Completeness",
        "",
        f"- Expected new rows: {len(expected_new_keys())}",
        f"- Missing new rows: {len(missing)}",
        "- Existing uniform 0.4 rows are reused from the preregistered MLP diagnostic.",
        "",
        "## Results",
        "",
        "| Dataset | Noise | Rate | MLP+CE | SoftDist-MLP | Shuffled | vs CE | vs shuffled | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for decision in decisions:
        dataset = str(decision["dataset"])
        noise_type = str(decision["noise_type"])
        noise_rate = float(decision["noise_rate"])
        values = []
        for method in METHODS:
            row = aggregates.get((dataset, noise_type, noise_rate, method))
            values.append(
                "NA"
                if row is None
                else f"{100 * float(row['mean']):.2f}+/-{100 * float(row['std']):.2f}"
            )
        if not int(decision["complete"]):
            label = "incomplete"
        elif noise_type == "clean":
            if not int(decision["clean_safe"]):
                label = "clean harm"
            elif int(decision["stable_increment"]):
                label = "stable increment; clean safe"
            else:
                label = "clean safe"
        elif int(decision["stable_increment"]):
            label = "stable increment"
        else:
            label = "no stable increment"
        lines.append(
            f"| {dataset} | {noise_type} | {noise_rate:.1f} | {values[0]} | "
            f"{values[1]} | {values[2]} | "
            f"{100 * float(decision['vs_ce_mean_delta']):+.2f}pp "
            f"({decision['vs_ce_positive_count']}/5) | "
            f"{100 * float(decision['vs_shuffled_mean_delta']):+.2f}pp "
            f"({decision['vs_shuffled_positive_count']}/5) | {label} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Cora stable noisy regimes: {stable_counts['cora']}/6.",
            f"- CiteSeer stable noisy regimes: {stable_counts['citeseer']}/6.",
            (
                "- Clean safety: "
                + "; ".join(
                    f"{row['dataset']}={'pass' if int(row['clean_safe']) else 'fail'}"
                    for row in decisions
                    if row["noise_type"] == "clean"
                )
                + "."
            ),
            (
                "- Claim: cross-noise robustness is supported for transfer to an X-only MLP."
                if robust and not missing
                else "- Claim: effectiveness on an X-only MLP is conditional on the noise regime."
            ),
            "- Boundary: this does not establish classifier-independent or model-independent robustness.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the SoftDist-MLP noise spectrum.")
    parser.add_argument(
        "--new_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "softdist_mlp_noise_spectrum_20260618",
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "softdist_mlp_5seed_20260618"
            / "softdist_mlp_5seed_summary.csv"
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.new_dir
    rows = collect_rows(args.new_dir, args.anchor)
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    decisions = evaluate_regimes(deltas)
    present = {
        (
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            str(row["method"]),
            int(row["seed"]),
        )
        for row in rows
    }
    missing = expected_new_keys() - present

    write_csv(output_dir / "softdist_mlp_noise_raw.csv", rows)
    write_csv(output_dir / "softdist_mlp_noise_aggregate.csv", aggregate)
    write_csv(output_dir / "softdist_mlp_noise_paired_deltas.csv", deltas)
    write_csv(output_dir / "softdist_mlp_noise_decisions.csv", decisions)
    (output_dir / "softdist_mlp_noise_report.md").write_text(
        build_report(aggregate, decisions, missing), encoding="utf-8"
    )
    print(
        f"rows={len(rows)} aggregate={len(aggregate)} "
        f"decisions={len(decisions)} missing_new={len(missing)}"
    )
    if args.require_complete and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
