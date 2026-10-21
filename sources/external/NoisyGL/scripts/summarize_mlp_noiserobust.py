from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp_noiserobust_20260701"
METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "frozengrace_gb_mlp_softdist",
    "frozengrace_gb_mlp_softdist_shuffled",
)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "MLP+small-loss",
    "mlp_coteaching": "MLP+co-teaching",
    "mlp_jocor": "MLP+JoCoR",
    "mlp_elr": "MLP+ELR",
    "mlp_sce": "MLP+SCE",
    "frozengrace_gb_mlp_softdist": "SoftDist-MLP",
    "frozengrace_gb_mlp_softdist_shuffled": "SoftDist-MLP shuffled",
}
SOFTDIST = "frozengrace_gb_mlp_softdist"
REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
)
G1_COMPARISONS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_elr",
    "frozengrace_gb_mlp_softdist_shuffled",
)
STRICT_EXTRA = ("mlp_jocor", "mlp_sce")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


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
        if path.name.startswith("mlp_noiserobust_"):
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
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
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
            row["method_label"],
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
        for baseline in METHODS:
            if baseline == SOFTDIST:
                continue
            seeds = []
            deltas = []
            for seed in range(5):
                left_key = (dataset, noise_type, noise_rate, SOFTDIST, seed)
                right_key = (dataset, noise_type, noise_rate, baseline, seed)
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
                    "comparison": f"softdist_vs_{baseline}",
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


def evaluate_gates(deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["baseline"],
        ): row
        for row in deltas
    }
    regimes = sorted({key[:3] for key in lookup})
    output = []
    for dataset, noise_type, noise_rate in regimes:
        g1_parts = []
        strict_parts = []
        for baseline in G1_COMPARISONS + STRICT_EXTRA:
            row = lookup.get((dataset, noise_type, noise_rate, baseline))
            passed = bool(
                row
                and int(row["n"]) == 5
                and float(row["mean_delta"]) > 0
                and int(row["positive_count"]) >= 3
            )
            if baseline in G1_COMPARISONS:
                g1_parts.append(passed)
            strict_parts.append(passed)
        output.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "g1_pass": int(all(g1_parts) and len(g1_parts) == len(G1_COMPARISONS)),
                "strict_pass": int(all(strict_parts) and len(strict_parts) == len(G1_COMPARISONS) + len(STRICT_EXTRA)),
                "g1_baselines": ",".join(G1_COMPARISONS),
                "strict_extra": ",".join(STRICT_EXTRA),
            }
        )
    return output


def expected_keys() -> set[tuple[str, str, float, str, int]]:
    return {
        (dataset, noise_type, noise_rate, method, seed)
        for dataset in ("cora", "citeseer")
        for noise_type, noise_rate in REGIMES
        for method in METHODS
        for seed in range(5)
    }


def missing_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, float, str, int]]:
    observed = {
        (
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            str(row["method"]),
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


def _fmt_pct(value):
    return f"{100.0 * float(value):.2f}"


def build_report(aggregate, deltas, gates, missing) -> str:
    agg = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"]): row
        for row in aggregate
    }
    delta_lookup = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["baseline"]): row
        for row in deltas
    }
    gate_lookup = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"])): row
        for row in gates
    }
    lines = [
        "# MLP Noise-Robust Baseline Report",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        "- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, train-only label noise, ROCm `cuda:0`",
        "- Backbone: X-only MLP for every compared noise-robust strategy",
        "- Verification status: ANALYZED",
        "",
        "## Completeness",
        "",
        f"- Expected rows: {len(expected_keys())}",
        f"- Missing rows: {len(missing)}",
        "",
        "## Main Table",
        "",
        "| Dataset | Noise | Rate | CE | Small-loss | Co-teaching | JoCoR | ELR | SCE | SoftDist | Shuffled | G1 | Strict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for dataset, noise_type, noise_rate in sorted({key[:3] for key in agg}):
        row_values = []
        for method in METHODS:
            row = agg.get((dataset, noise_type, noise_rate, method))
            if not row:
                row_values.append("NA")
            else:
                row_values.append(f"{_fmt_pct(row['mean'])}+/-{_fmt_pct(row['std'])}")
        gate = gate_lookup.get((dataset, noise_type, noise_rate), {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dataset),
                    str(noise_type),
                    f"{noise_rate:.1f}",
                    *row_values,
                    "pass" if gate.get("g1_pass") else "fail",
                    "pass" if gate.get("strict_pass") else "fail",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Paired Deltas Against SoftDist",
        "",
        "| Dataset | Noise | Rate | Baseline | Delta | Wins |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["noise_type"]),
                    f"{float(row['noise_rate']):.1f}",
                    str(row["baseline_label"]),
                    f"{100.0 * float(row['mean_delta']):+.2f}pp",
                    f"{row['positive_count']}/{row['n']}",
                ]
            )
            + " |"
        )
    uniform_summary = defaultdict(int)
    for gate in gates:
        if gate["noise_type"] == "uniform":
            uniform_summary[gate["dataset"]] += int(gate["g1_pass"])
    pair_summary = {
        gate["dataset"]: int(gate["g1_pass"])
        for gate in gates
        if gate["noise_type"] == "pair" and math.isclose(float(gate["noise_rate"]), 0.4)
    }
    overall_g1 = all(uniform_summary.get(dataset, 0) >= 2 for dataset in ("cora", "citeseer"))
    lines += [
        "",
        "## Gate Decision",
        "",
        f"- Uniform-rate G1 counts: cora={uniform_summary.get('cora', 0)}/3, citeseer={uniform_summary.get('citeseer', 0)}/3.",
        f"- Pair-0.4 G1: cora={pair_summary.get('cora', 0)}, citeseer={pair_summary.get('citeseer', 0)}.",
        f"- G1 overall under preregistered weak-MLP criterion: {'GO' if overall_g1 else 'NO-GO'}",
        "- G1 counts CE, small-loss, co-teaching, ELR, and shuffled. Strict additionally counts JoCoR and SCE.",
    ]
    if missing:
        lines += ["", "## Missing Keys", ""]
        for key in sorted(missing):
            lines.append(f"- {key}")
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize weak-MLP noise-robust baselines.")
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
    gates = evaluate_gates(deltas)
    write_csv(run_dir / "mlp_noiserobust_raw.csv", rows)
    write_csv(run_dir / "mlp_noiserobust_aggregate.csv", aggregate)
    write_csv(run_dir / "mlp_noiserobust_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp_noiserobust_gates.csv", gates)
    (run_dir / "mlp_noiserobust_report.md").write_text(
        build_report(aggregate, deltas, gates, missing),
        encoding="utf-8",
    )
    print(f"rows={len(rows)} aggregate={len(aggregate)} missing={len(missing)}")


if __name__ == "__main__":
    main()
