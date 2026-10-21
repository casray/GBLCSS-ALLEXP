from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp_maintable_20260701"
SOFTDIST = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MAIN_METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_fine",
    "mlp_dividemix_lite",
    SOFTDIST,
    SHUFFLED,
)
LABELRATE_METHODS = (
    "mlp_small_loss",
    "mlp_jocor",
    "mlp_fine",
    SOFTDIST,
    SHUFFLED,
)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "MLP+small-loss",
    "mlp_coteaching": "MLP+co-teaching",
    "mlp_jocor": "MLP+JoCoR",
    "mlp_elr": "MLP+ELR",
    "mlp_sce": "MLP+SCE",
    "mlp_fine": "MLP-FINE",
    "mlp_dividemix_lite": "DivideMix-lite",
    SOFTDIST: "SoftDist-MLP beta=0.5",
    SHUFFLED: "SoftDist beta=0.5 shuffled",
}
MAIN_DATASETS = ("cora", "citeseer", "pubmed")
MAIN_REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
LABELRATE_DATASETS = ("cora", "citeseer")
LABELRATE_TRAINS = (0.05, 0.65)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def table_and_train_from_name(path: Path) -> tuple[str, float] | None:
    name = path.name
    if name.startswith("main_"):
        return "main", 0.2
    match = re.match(r"labelrate_train(\d+p\d+)_seeds_", name)
    if match:
        return "labelrate", float(match.group(1).replace("p", "."))
    return None


def normalize_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    parsed = table_and_train_from_name(source)
    if parsed is None:
        return None
    table, train_percent = parsed
    method = (row.get("method") or row.get("base_method") or "").strip()
    allowed = MAIN_METHODS if table == "main" else LABELRATE_METHODS
    if method not in allowed:
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "table": table,
        "train_percent": float(train_percent),
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_train_noise_rate": _optional_float(row.get("actual_train_noise_rate")),
        "source": str(source),
    }


def _optional_float(value) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp_maintable_"):
            continue
        for row in read_csv(path):
            normalized = normalize_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows)


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(f"conflicting duplicate for {key}")
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["table"]),
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            float(row["train_percent"]),
            str(row["method"]),
            int(row["seed"]),
        ),
    )


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            row["method_label"],
        )
        groups[key].append(float(row["test_acc"]))
    output = []
    for (table, dataset, noise_type, noise_rate, train_percent, method, label), values in groups.items():
        output.append(
            {
                "table": table,
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "train_percent": train_percent,
                "method": method,
                "method_label": label,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda row: (row["table"], row["dataset"], row["noise_type"], row["noise_rate"], row["train_percent"], row["method"]))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in rows
    }
    settings = sorted({key[:5] for key in values})
    output = []
    for table, dataset, noise_type, noise_rate, train_percent in settings:
        methods = MAIN_METHODS if table == "main" else LABELRATE_METHODS
        for baseline in methods:
            if baseline == SOFTDIST:
                continue
            seeds = []
            deltas = []
            for seed in range(5):
                left = (table, dataset, noise_type, noise_rate, train_percent, SOFTDIST, seed)
                right = (table, dataset, noise_type, noise_rate, train_percent, baseline, seed)
                if left in values and right in values:
                    seeds.append(seed)
                    deltas.append(values[left] - values[right])
            if not deltas:
                continue
            output.append(
                {
                    "table": table,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "train_percent": train_percent,
                    "comparison": f"beta05_vs_{baseline}",
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(map(str, seeds)),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                }
            )
    return output


def expected_keys() -> set[tuple[object, ...]]:
    keys = {
        ("main", dataset, noise_type, noise_rate, 0.2, method, seed)
        for dataset in MAIN_DATASETS
        for noise_type, noise_rate in MAIN_REGIMES
        for method in MAIN_METHODS
        for seed in range(5)
    }
    keys.update(
        {
            ("labelrate", dataset, "uniform", 0.4, train_percent, method, seed)
            for dataset in LABELRATE_DATASETS
            for train_percent in LABELRATE_TRAINS
            for method in LABELRATE_METHODS
            for seed in range(5)
        }
    )
    return keys


def missing_keys(rows: list[dict[str, object]]) -> set[tuple[object, ...]]:
    observed = {
        (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        )
        for row in rows
    }
    return expected_keys() - observed


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    return f"{100.0 * float(value):.2f}"


def _lookup(rows):
    return {
        (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
        ): row
        for row in rows
    }


def build_report(aggregate, deltas, missing) -> str:
    agg = _lookup(aggregate)
    delta_lookup = {
        (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["baseline"],
        ): row
        for row in deltas
    }
    lines = [
        "# MLP beta=0.5 Main-Table Report",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        "- Protocol: train-only label noise, explicit percent split, ROCm `cuda:0`",
        "- Backbone: X-only 2-layer MLP for all compared noise-robust strategies",
        "- Main method: SoftDist-MLP beta=0.5",
        "- FINE/DivideMix note: MLP-adapted strategy baselines, not original image-domain reproduction",
        "- Verification status: ANALYZED",
        "",
        "## Completeness",
        "",
        f"- Expected rows: {len(expected_keys())}",
        f"- Observed aggregate cells: {len(aggregate)}",
        f"- Missing rows: {len(missing)}",
        "",
        "## Main Label-Noise Table",
        "",
        "| Dataset | Noise | Rate | CE | Small-loss | Co-teaching | JoCoR | ELR | SCE | FINE | DivideMix-lite | beta=0.5 | Shuffled |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in MAIN_DATASETS:
        for noise_type, noise_rate in MAIN_REGIMES:
            values = []
            for method in MAIN_METHODS:
                row = agg.get(("main", dataset, noise_type, noise_rate, 0.2, method))
                values.append("NA" if row is None else f"{_fmt(row['mean'])}+/-{_fmt(row['std'])}")
            lines.append(
                "| " + " | ".join([dataset, noise_type, f"{noise_rate:.1f}", *values]) + " |"
            )
    lines += [
        "",
        "## Paired Position of beta=0.5",
        "",
        "| Table | Dataset | Noise | Rate | Train | Baseline | Delta | Wins |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["table"]),
                    str(row["dataset"]),
                    str(row["noise_type"]),
                    f"{float(row['noise_rate']):.1f}",
                    f"{float(row['train_percent']):.2f}",
                    str(row["baseline_label"]),
                    f"{100.0 * float(row['mean_delta']):+.2f}pp",
                    f"{row['positive_count']}/{row['n']}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Label-Rate Endpoint Table",
        "",
        "| Dataset | Train | Small-loss | JoCoR | FINE | beta=0.5 | Shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in LABELRATE_DATASETS:
        for train_percent in LABELRATE_TRAINS:
            values = []
            for method in LABELRATE_METHODS:
                row = agg.get(("labelrate", dataset, "uniform", 0.4, train_percent, method))
                values.append("NA" if row is None else f"{_fmt(row['mean'])}+/-{_fmt(row['std'])}")
            lines.append("| " + " | ".join([dataset, f"{train_percent:.2f}", *values]) + " |")
    lines += [
        "",
        "## Compact Verdict Aids",
        "",
    ]
    baselines = ("mlp_small_loss", "mlp_coteaching", "mlp_jocor", "mlp_elr", "mlp_sce", "mlp_fine", "mlp_dividemix_lite", SHUFFLED)
    for baseline in baselines:
        rows = [row for row in deltas if row["table"] == "main" and row["baseline"] == baseline and int(row["n"]) == 5]
        wins = sum(float(row["mean_delta"]) > 0 and int(row["positive_count"]) >= 3 for row in rows)
        lines.append(f"- beta=0.5 vs {METHOD_LABELS[baseline]}: {wins}/{len(rows)} main settings mean-positive with >=3/5 seeds.")
    if missing:
        lines += ["", "## Missing Keys", ""]
        for key in sorted(missing):
            lines.append(f"- {key}")
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize MLP beta=0.5 main-table experiments.")
    parser.add_argument("--run_dir", default=str(RUN_DIR))
    parser.add_argument("--require_complete", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir)
    missing = missing_keys(rows)
    if args.require_complete and missing:
        raise SystemExit(f"missing {len(missing)} expected rows")
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    write_csv(run_dir / "mlp_maintable_combined_raw.csv", rows)
    write_csv(run_dir / "mlp_maintable_aggregate.csv", aggregate)
    write_csv(run_dir / "mlp_maintable_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp_maintable_completeness.csv", [{"missing": len(missing), "expected": len(expected_keys()), "observed": len(rows)}])
    (run_dir / "mlp_maintable_report.md").write_text(
        build_report(aggregate, deltas, missing),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} aggregate={len(aggregate)} deltas={len(deltas)} missing={len(missing)}")


if __name__ == "__main__":
    main()
