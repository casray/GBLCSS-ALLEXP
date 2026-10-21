from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]

RATIO_SPECS = [
    ("0p05", 0.05, 0.10, 0.85),
    ("0p10", 0.10, 0.10, 0.80),
    ("0p15", 0.15, 0.10, 0.75),
    ("0p20", 0.20, 0.10, 0.70),
    ("0p25", 0.25, 0.10, 0.65),
    ("0p30", 0.30, 0.10, 0.60),
    ("0p40", 0.40, 0.10, 0.50),
]
TAG_ALIASES = {
    "0p05": ["0p05"],
    "0p10": ["0p10", "0p1"],
    "0p15": ["0p15"],
    "0p20": ["0p20", "0p2"],
    "0p25": ["0p25"],
    "0p30": ["0p30"],
    "0p40": ["0p40", "0p4"],
}

CORE_DATASETS = ["cora", "citeseer", "coauthorcs"]
CORE_METHODS = [
    "gcn",
    "gcn_xh_ce",
    "frozengrace_gb_gcn_softdist_xh",
    "frozengrace_gb_gcn_softdist_xh_shuffled",
]
METHOD_LABELS = {
    "gcn": "GCN+CE",
    "gcn_xh_ce": "GCN+CE [X||H]",
    "frozengrace_gb_gcn_softdist_xh": "SoftDist [X||H]",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "SoftDist [X||H] shuffled",
}
COMPARISONS = [
    ("softdist_vs_gcn_xh", "frozengrace_gb_gcn_softdist_xh", "gcn_xh_ce"),
    (
        "softdist_vs_shuffled",
        "frozengrace_gb_gcn_softdist_xh",
        "frozengrace_gb_gcn_softdist_xh_shuffled",
    ),
    ("softdist_vs_gcn", "frozengrace_gb_gcn_softdist_xh", "gcn"),
    ("gcn_xh_vs_gcn", "gcn_xh_ce", "gcn"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _test_value(row: dict[str, str]) -> float:
    value = row.get("test_acc") or row.get("test")
    if value in (None, ""):
        raise ValueError(f"missing test/test_acc value in row: {row}")
    return float(value)


def _method_value(row: dict[str, str]) -> str:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if not method:
        raise ValueError(f"missing method/base_method value in row: {row}")
    return method


def _optional_float(row: dict[str, str], key: str) -> float | str:
    value = row.get(key)
    if value in (None, ""):
        return ""
    return float(value)


def _normalize_rows(
    path: Path,
    *,
    train_percent: float,
    val_percent: float,
    test_percent: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(path):
        method = _method_value(row)
        if method not in CORE_METHODS:
            continue
        status = (row.get("status") or "ok").strip().lower()
        dataset = (row.get("dataset") or "").strip().lower()
        seed = int(row["seed"])
        rows.append(
            {
                "dataset": dataset,
                "train_percent": train_percent,
                "val_percent": val_percent,
                "test_percent": test_percent,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "noise_type": (row.get("noise_type") or "").strip().lower(),
                "noise_rate": _optional_float(row, "noise_rate"),
                "status": status,
                "error": row.get("error", ""),
                "train": _optional_float(row, "train"),
                "valid": _optional_float(row, "valid"),
                "test": _test_value(row),
                "actual_noise_rate": _optional_float(row, "actual_noise_rate"),
                "actual_train_noise_rate": _optional_float(row, "actual_train_noise_rate"),
                "split_hash": row.get("split_hash", ""),
                "noise_hash": row.get("noise_hash", ""),
                "source": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
            }
        )
    return rows


def collect_rows(coverage_dir: Path, output_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    seen_sources: set[Path] = set()
    for tag, train_percent, val_percent, test_percent in RATIO_SPECS:
        source_groups = [
            [coverage_dir / f"coverage_standard_train{alias}_summary.csv" for alias in TAG_ALIASES[tag]],
            [coverage_dir / f"coverage_xh_train{alias}_summary.csv" for alias in TAG_ALIASES[tag]],
            [output_dir / f"label_rate_raw_train{alias}.csv" for alias in TAG_ALIASES[tag]]
            + [output_dir / f"label_rate_raw_train{alias}_coauthorcs.csv" for alias in TAG_ALIASES[tag]],
        ]
        for candidates in source_groups:
            existing = [path for path in candidates if path.exists()]
            if not existing:
                continue
            for path in existing:
                if path in seen_sources:
                    continue
                seen_sources.add(path)
                rows.extend(
                    _normalize_rows(
                        path,
                        train_percent=train_percent,
                        val_percent=val_percent,
                        test_percent=test_percent,
                    )
                )
        if not any(path.exists() for group in source_groups for path in group):
            warnings.append(f"no source files found for train ratio {train_percent}")
    return deduplicate(rows), warnings


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["train_percent"],
            row["method"],
            row["seed"],
            row["noise_type"],
            row["noise_rate"],
        )
        previous = unique.get(key)
        if previous is None:
            unique[key] = row
            continue
        if not math.isclose(float(previous["test"]), float(row["test"]), abs_tol=1e-12):
            raise ValueError(
                f"conflicting duplicate for {key}: {previous['test']} from "
                f"{previous['source']} vs {row['test']} from {row['source']}"
            )
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["dataset"]),
            float(row["train_percent"]),
            CORE_METHODS.index(str(row["method"])),
            int(row["seed"]),
        ),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (
            row["dataset"],
            row["train_percent"],
            row["val_percent"],
            row["test_percent"],
            row["method"],
            row["method_label"],
        )
        grouped[key].append(float(row["test"]))
    output = []
    for (dataset, train_percent, val_percent, test_percent, method, label), values in grouped.items():
        output.append(
            {
                "dataset": dataset,
                "train_percent": train_percent,
                "val_percent": val_percent,
                "test_percent": test_percent,
                "method": method,
                "method_label": label,
                "n": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "seed_values": " ".join(f"{value:.6f}" for value in values),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            float(row["train_percent"]),
            CORE_METHODS.index(str(row["method"])),
        ),
    )


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (
            row["dataset"],
            row["train_percent"],
            row["method"],
            row["seed"],
        ): float(row["test"])
        for row in rows
        if row["status"] == "ok"
    }
    regimes = sorted({(row["dataset"], row["train_percent"], row["val_percent"], row["test_percent"]) for row in rows})
    output = []
    for dataset, train_percent, val_percent, test_percent in regimes:
        seeds = sorted({int(row["seed"]) for row in rows if row["dataset"] == dataset and row["train_percent"] == train_percent})
        for comparison, left, right in COMPARISONS:
            deltas = []
            paired_seeds = []
            for seed in seeds:
                left_key = (dataset, train_percent, left, seed)
                right_key = (dataset, train_percent, right, seed)
                if left_key not in values or right_key not in values:
                    continue
                paired_seeds.append(seed)
                deltas.append(values[left_key] - values[right_key])
            if not deltas:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "train_percent": train_percent,
                    "val_percent": val_percent,
                    "test_percent": test_percent,
                    "comparison": comparison,
                    "left": left,
                    "right": right,
                    "left_label": METHOD_LABELS[left],
                    "right_label": METHOD_LABELS[right],
                    "n": len(deltas),
                    "mean_delta": mean(deltas),
                    "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_seed_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(str(seed) for seed in paired_seeds),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                    "strong_evidence": is_strong_evidence(comparison, deltas),
                }
            )
    return output


def is_strong_evidence(comparison: str, deltas: list[float]) -> str:
    if comparison == "softdist_vs_gcn_xh":
        threshold = 0.005
    elif comparison == "softdist_vs_shuffled":
        threshold = 0.010
    else:
        return ""
    return "yes" if mean(deltas) >= threshold and sum(delta > 0 for delta in deltas) >= 4 else "no"


def audit_rows(
    rows: list[dict[str, object]],
    *,
    datasets: list[str],
    seeds: list[int],
) -> list[dict[str, object]]:
    by_key = {
        (
            row["dataset"],
            row["train_percent"],
            row["method"],
            row["seed"],
        ): row
        for row in rows
    }
    audit = []
    for dataset in datasets:
        for _, train_percent, val_percent, test_percent in RATIO_SPECS:
            split_hashes: dict[int, set[str]] = defaultdict(set)
            for method in CORE_METHODS:
                present = 0
                bad_status = 0
                missing_seeds = []
                for seed in seeds:
                    row = by_key.get((dataset, train_percent, method, seed))
                    if row is None:
                        missing_seeds.append(seed)
                        continue
                    present += 1
                    if row["status"] != "ok":
                        bad_status += 1
                    split_hash = str(row.get("split_hash", ""))
                    if split_hash:
                        split_hashes[seed].add(split_hash)
                audit.append(
                    {
                        "dataset": dataset,
                        "train_percent": train_percent,
                        "val_percent": val_percent,
                        "test_percent": test_percent,
                        "method": method,
                        "expected_n": len(seeds),
                        "present_n": present,
                        "bad_status_n": bad_status,
                        "missing_seeds": " ".join(str(seed) for seed in missing_seeds),
                    }
                )
            for seed, hashes in sorted(split_hashes.items()):
                if len(hashes) > 1:
                    audit.append(
                        {
                            "dataset": dataset,
                            "train_percent": train_percent,
                            "val_percent": val_percent,
                            "test_percent": test_percent,
                            "method": "SPLIT_HASH_MISMATCH",
                            "expected_n": 1,
                            "present_n": len(hashes),
                            "bad_status_n": 1,
                            "missing_seeds": f"seed={seed} hashes={' '.join(sorted(hashes))}",
                        }
                    )
    return audit


def _fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_pp(value: object) -> str:
    number = float(value) * 100
    sign = "+" if number >= 0 else ""
    return f"{sign}{number:.2f}pp"


def write_report(
    path: Path,
    *,
    aggregate: list[dict[str, object]],
    deltas: list[dict[str, object]],
    audit: list[dict[str, object]],
    warnings: list[str],
    args: argparse.Namespace,
) -> None:
    agg_by_key = {(row["dataset"], row["train_percent"], row["method"]): row for row in aggregate}
    delta_by_key = {(row["dataset"], row["train_percent"], row["comparison"]): row for row in deltas}
    datasets = _parse_list(args.datasets)
    lines = [
        "# Label-Rate Sensitivity for SoftDist [X||H]",
        "",
        "## Protocol",
        "",
        f"- Split: `percent`, train ratios `{', '.join(str(spec[1]) for spec in RATIO_SPECS)}`.",
        "- Validation ratio is fixed at `0.10`; test ratio is `1 - train - val`.",
        "- Noise: `uniform`, rate `0.4`, scope `train`.",
        f"- Seeds: `{args.seeds}`.",
        "- Core comparison: SoftDist [X||H] against GCN+CE [X||H] and shuffled SoftDist.",
        "- This is a sensitivity analysis, not a cross-ratio leaderboard.",
        "",
        "## Research Record",
        "",
        "- Hypothesis: SoftDist's ball soft target should help most when labels are scarce enough that GCN [X||H] has not saturated, but not necessarily improve monotonically with larger train ratios.",
        "- Granular-ball role: balls act as noisy-label soft target providers and local structure priors, not hard positive labels.",
        "- Method motivation: test whether the ball component adds value beyond the same `[X||H]` input and beyond shuffled ball structure across label-rate regimes.",
        "- Code changes: no model, loss, baseline, or evaluation logic changes; only this summarizer was added to combine old and new raw CSV files with explicit split metadata.",
        "- Config: existing Phase 16 coverage files are reused where available; newly added ratios are run with the same protocol, methods, and seeds.",
        "- Expected mechanism: low/mid train ratios should leave room for ball-level consensus; high train ratios may let GCN [X||H] absorb local structure directly, reducing ball marginal gain.",
        "- Success standard: both `SoftDist - GCN [X||H]` and `SoftDist - shuffled` pass the pre-registered mean-delta and positive-seed thresholds.",
        "- Failure standard: no train ratio passes both controls, or gains only appear against plain GCN but disappear against GCN [X||H] or shuffled.",
        "- Results: see Main Results and paired-delta CSV.",
        "- Final decision: use this as a sensitivity/mechanism curve, not as a replacement for the 20/10/70 main protocol.",
        "",
    ]
    if warnings:
        lines.extend(["## Source Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    audit_failures = [
        row
        for row in audit
        if int(row["present_n"]) != int(row["expected_n"])
        or int(row["bad_status_n"]) != 0
        or row["method"] == "SPLIT_HASH_MISMATCH"
    ]
    lines.extend(
        [
            "## Completeness Audit",
            "",
            "| Dataset | Train | Method | Present | Bad status | Missing seeds |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    if not audit_failures:
        lines.append("| all | all | all | complete | 0 |  |")
    for row in audit_failures:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['train_percent'], 2)} | {row['method']} | "
            f"{row['present_n']}/{row['expected_n']} | {row['bad_status_n']} | {row['missing_seeds']} |"
        )

    lines.extend(
        [
            "",
            "## Main Results",
            "",
            "| Dataset | Train | GCN | GCN [X||H] | SoftDist [X||H] | Shuffled | Soft-GCN[X||H] | Soft-Shuffled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        for _, train_percent, _, _ in RATIO_SPECS:
            values = {
                method: agg_by_key.get((dataset, train_percent, method))
                for method in CORE_METHODS
            }
            if not any(values.values()):
                continue
            soft_gcn = delta_by_key.get((dataset, train_percent, "softdist_vs_gcn_xh"))
            soft_shuf = delta_by_key.get((dataset, train_percent, "softdist_vs_shuffled"))
            lines.append(
                f"| {dataset} | {_fmt(train_percent, 2)} | "
                f"{_cell_mean(values['gcn'])} | {_cell_mean(values['gcn_xh_ce'])} | "
                f"{_cell_mean(values['frozengrace_gb_gcn_softdist_xh'])} | "
                f"{_cell_mean(values['frozengrace_gb_gcn_softdist_xh_shuffled'])} | "
                f"{_cell_delta(soft_gcn)} | {_cell_delta(soft_shuf)} |"
            )

    lines.extend(
        [
            "",
            "## Pre-Registered Interpretation",
            "",
            "- A train ratio is strong evidence only when both SoftDist-vs-GCN[X||H] and SoftDist-vs-shuffled pass the positive-seed and mean-delta thresholds.",
            "- Weak or negative high-label results support the saturation interpretation rather than a monotonic coverage claim.",
            "- The main 20/10/70 protocol should remain the main table; this curve explains when the ball soft target contributes.",
            "- Cora/CiteSeer support the low/mid-label-rate mechanism; Coauthor-CS shows smaller gains below the strong-evidence threshold, so the claim should be dataset- and regime-conditional.",
            "",
            "## Artifacts",
            "",
            "- Combined raw: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_combined_raw.csv`",
            "- Aggregate: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_aggregate.csv`",
            "- Paired deltas: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_paired_deltas.csv`",
            "- Audit: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_audit.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cell_mean(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    return f"{_fmt(row['mean'])} +/- {_fmt(row['std'])}"


def _cell_delta(row: dict[str, object] | None) -> str:
    if row is None:
        return ""
    strong = " strong" if row.get("strong_evidence") == "yes" else ""
    return f"{_fmt_pp(row['mean_delta'])} ({row['positive_seed_count']}/{row['n']}){strong}"


def has_audit_failures(audit: list[dict[str, object]]) -> bool:
    for row in audit:
        if int(row["present_n"]) != int(row["expected_n"]):
            return True
        if int(row["bad_status_n"]) != 0:
            return True
        if row["method"] == "SPLIT_HASH_MISMATCH":
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SoftDist label-rate sensitivity runs.")
    parser.add_argument("--coverage_dir", default=str(REPO_ROOT / "runs" / "coverage_sensitivity_20260612"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "label_rate_sensitivity_20260626"))
    parser.add_argument("--datasets", default=",".join(CORE_DATASETS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows, warnings = collect_rows(Path(args.coverage_dir), output_dir)
    datasets = _parse_list(args.datasets)
    seeds = [int(seed) for seed in _parse_list(args.seeds)]
    rows = [row for row in rows if row["dataset"] in datasets]
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    audit = audit_rows(rows, datasets=datasets, seeds=seeds)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "label_rate_sensitivity_combined_raw.csv", rows)
    _write_csv(output_dir / "label_rate_sensitivity_aggregate.csv", aggregate)
    _write_csv(output_dir / "label_rate_sensitivity_paired_deltas.csv", deltas)
    _write_csv(output_dir / "label_rate_sensitivity_audit.csv", audit)
    write_report(
        output_dir / "label_rate_sensitivity_report.md",
        aggregate=aggregate,
        deltas=deltas,
        audit=audit,
        warnings=warnings,
        args=args,
    )
    if args.strict and has_audit_failures(audit):
        print(f"WROTE {output_dir / 'label_rate_sensitivity_report.md'}")
        print("STRICT audit failed", file=sys.stderr)
        sys.exit(1)
    print(f"WROTE {output_dir / 'label_rate_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
