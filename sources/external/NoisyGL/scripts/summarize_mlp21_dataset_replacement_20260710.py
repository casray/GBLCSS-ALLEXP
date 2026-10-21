from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp21_dataset_replacement_20260710"

DATASETS = ("amazoncom", "corafull")
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
METHODS = (
    OURS,
    SHUFFLED,
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
BASELINES = tuple(method for method in METHODS if method not in {OURS, SHUFFLED})
GRAPHFREE_METHODS = (
    "graphmlp",
    "gsdn",
    "mlp_distill_gcn",
    "ffg2m_adapted",
    "krd_adapted",
    "vqgraph_adapted",
    "mlp_distill_softdist_xh",
)
ALL_METHODS = METHODS + GRAPHFREE_METHODS
SEEDS = tuple(range(5))
GATE_REGIME = ("uniform", 0.4)
CORAFULL_REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)

GRAPHFREE_RESOURCES = {
    "graphmlp": ("yes", "no", "adapted"),
    "gsdn": ("yes", "no", "adapted"),
    "mlp_distill_gcn": ("yes", "no", "GLNN-style"),
    "ffg2m_adapted": ("yes", "no", "adapted"),
    "krd_adapted": ("yes", "no", "adapted"),
    "vqgraph_adapted": ("yes", "no", "adapted"),
    "mlp_distill_softdist_xh": ("yes", "no", "LCSS teacher"),
}


def _as_float(value, default=None):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _as_int(value, default=None):
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def _split_protocol_matches(row: dict[str, str]) -> bool:
    if (row.get("noise_scope") or "train").strip().lower() != "train":
        return False
    optional_expected = {
        "split_type": ("percent", str),
        "train_percent": (0.2, float),
        "val_percent": (0.1, float),
        "test_percent": (0.7, float),
    }
    for key, (expected, caster) in optional_expected.items():
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        value = caster(raw)
        if isinstance(expected, float):
            if not math.isclose(value, expected, abs_tol=1e-12):
                return False
        elif value != expected:
            return False
    return True


def _regime_matches(row: dict[str, str], regimes: set[tuple[str, float]] | None) -> bool:
    noise_type = (row.get("noise_type") or "").strip().lower()
    noise_rate = _as_float(row.get("noise_rate"), -1.0)
    if regimes is None:
        return True
    return any(noise_type == expected_type and math.isclose(noise_rate, expected_rate, abs_tol=1e-12)
               for expected_type, expected_rate in regimes)


def _protocol_matches(row: dict[str, str]) -> bool:
    return _split_protocol_matches(row) and _regime_matches(row, {GATE_REGIME})


def _row_from_raw(raw: dict[str, str], source: Path) -> dict[str, object]:
    return {
        "dataset": (raw.get("dataset") or "").strip().lower(),
        "method": (raw.get("method") or raw.get("base_method") or "").strip(),
        "seed": _as_int(raw.get("seed")),
        "noise_type": (raw.get("noise_type") or "").strip().lower(),
        "noise_rate": _as_float(raw.get("noise_rate")),
        "noise_scope": (raw.get("noise_scope") or "train").strip().lower(),
        "test": float(raw.get("test") or raw.get("test_acc")),
        "status": (raw.get("status") or "ok").strip().lower(),
        "modified_count": _as_int(raw.get("modified_count")),
        "modified_train_count": _as_int(raw.get("modified_train_count")),
        "modified_val_count": _as_int(raw.get("modified_val_count"), 0),
        "modified_test_count": _as_int(raw.get("modified_test_count"), 0),
        "num_balls": _as_float(raw.get("num_balls")),
        "mean_train_per_ball": _as_float(raw.get("mean_train_per_ball")),
        "wall_time_seconds": _as_float(raw.get("wall_time_seconds")),
        "parameter_count": _as_int(raw.get("parameter_count")),
        "source": str(source),
    }


def load_rows(
    paths: list[Path],
    allowed_methods: tuple[str, ...] | set[str] | None = None,
    regimes: set[tuple[str, float]] | None = None,
    datasets: tuple[str, ...] | set[str] | None = DATASETS,
) -> list[dict[str, object]]:
    allowed = set(allowed_methods) if allowed_methods is not None else None
    dataset_set = set(datasets) if datasets is not None else None
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, str, float]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
            for raw in csv.DictReader(stream):
                dataset = (raw.get("dataset") or "").strip().lower()
                method = (raw.get("method") or raw.get("base_method") or "").strip()
                status = (raw.get("status") or "ok").strip().lower()
                test = raw.get("test") or raw.get("test_acc") or ""
                if dataset_set is not None and dataset not in dataset_set:
                    continue
                if allowed is not None and method not in allowed:
                    continue
                if status not in {"", "ok"} or test == "":
                    continue
                if not _split_protocol_matches(raw) or not _regime_matches(raw, regimes):
                    continue
                seed = _as_int(raw.get("seed"))
                noise_type = (raw.get("noise_type") or "").strip().lower()
                noise_rate = _as_float(raw.get("noise_rate"))
                key = (dataset, method, seed, noise_type, noise_rate)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(_row_from_raw(raw, path))
    return sorted(rows, key=lambda row: (
        row["dataset"], row["noise_type"], row["noise_rate"], row["method"], row["seed"],
    ))


def load_protocol_rows(paths: list[Path]) -> list[dict[str, object]]:
    return load_rows(paths, allowed_methods=METHODS, regimes={GATE_REGIME}, datasets=DATASETS)


def _paired(
    rows: list[dict[str, object]],
    dataset: str,
    left: str,
    right: str,
    noise_type: str = "uniform",
    noise_rate: float = 0.4,
) -> list[float]:
    lookup = {
        (
            row["dataset"],
            row["method"],
            int(row["seed"]),
            row.get("noise_type", "uniform"),
            float(row.get("noise_rate", 0.4)),
        ): float(row["test"])
        for row in rows
    }
    return [
        lookup[(dataset, left, seed, noise_type, noise_rate)] - lookup[(dataset, right, seed, noise_type, noise_rate)]
        for seed in SEEDS
        if (dataset, left, seed, noise_type, noise_rate) in lookup
        and (dataset, right, seed, noise_type, noise_rate) in lookup
    ]


def decide_gate(rows: list[dict[str, object]], dataset: str) -> dict[str, object]:
    means = defaultdict(list)
    for row in rows:
        if (
            row["dataset"] == dataset
            and row.get("noise_type", "uniform") == "uniform"
            and math.isclose(float(row.get("noise_rate", 0.4)), 0.4)
        ):
            means[row["method"]].append(float(row["test"]))
    available_baselines = [method for method in BASELINES if len(means[method]) == 5]
    best = max(available_baselines, key=lambda method: statistics.mean(means[method])) if available_baselines else ""
    vs_shuffled = _paired(rows, dataset, OURS, SHUFFLED)
    vs_best = _paired(rows, dataset, OURS, best) if best else []
    structure_pass = len(vs_shuffled) == 5 and statistics.mean(vs_shuffled) >= 0.01 and sum(x > 0 for x in vs_shuffled) >= 4
    best_pass = len(vs_best) == 5 and statistics.mean(vs_best) >= 0.01 and sum(x > 0 for x in vs_best) >= 4
    audit_rows = [
        row for row in rows
        if row["dataset"] == dataset
        and row.get("noise_type", "uniform") == "uniform"
        and math.isclose(float(row.get("noise_rate", 0.4)), 0.4)
    ]
    audit_pass = bool(audit_rows) and all(
        int(row.get("modified_val_count") or 0) == 0
        and int(row.get("modified_test_count") or 0) == 0
        and (
            row.get("modified_count") is None
            or row.get("modified_train_count") is None
            or int(row["modified_count"]) == int(row["modified_train_count"])
        )
        and math.isfinite(float(row["test"]))
        for row in audit_rows
    )
    if structure_pass and best_pass and audit_pass:
        decision = "pass"
    elif structure_pass and audit_pass:
        decision = "boundary"
    else:
        decision = "fail"
    return {
        "dataset": dataset,
        "decision": decision,
        "best_baseline": best,
        "vs_shuffled_mean_delta": statistics.mean(vs_shuffled) if vs_shuffled else math.nan,
        "vs_shuffled_positive": sum(x > 0 for x in vs_shuffled),
        "vs_best_baseline_mean_delta": statistics.mean(vs_best) if vs_best else math.nan,
        "vs_best_baseline_positive": sum(x > 0 for x in vs_best),
        "audit_pass": int(audit_pass),
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"])].append(float(row["test"]))
    return [
        {
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for (dataset, noise_type, noise_rate, method), values in sorted(grouped.items())
    ]


def paired_deltas(rows: list[dict[str, object]], datasets: tuple[str, ...], regimes: tuple[tuple[str, float], ...]) -> list[dict[str, object]]:
    out = []
    for dataset in datasets:
        for noise_type, noise_rate in regimes:
            for baseline in METHODS:
                if baseline == OURS:
                    continue
                values = _paired(rows, dataset, OURS, baseline, noise_type, noise_rate)
                if values:
                    out.append(
                        {
                            "dataset": dataset,
                            "noise_type": noise_type,
                            "noise_rate": noise_rate,
                            "baseline": baseline,
                            "n": len(values),
                            "mean_delta": statistics.mean(values),
                            "std_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
                            "positive_count": sum(x > 0 for x in values),
                            "seed_deltas": " ".join(f"{value:.6f}" for value in values),
                        }
                    )
    return out


def completeness_for(
    rows: list[dict[str, object]],
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    regimes: tuple[tuple[str, float], ...],
    kind: str,
) -> list[dict[str, object]]:
    observed = {
        (row["dataset"], row["method"], int(row["seed"]), row["noise_type"], float(row["noise_rate"]))
        for row in rows
    }
    out = []
    for dataset in datasets:
        for noise_type, noise_rate in regimes:
            for method in methods:
                found = [
                    seed for seed in SEEDS
                    if (dataset, method, seed, noise_type, noise_rate) in observed
                ]
                out.append(
                    {
                        "kind": kind,
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": noise_rate,
                        "method": method,
                        "expected": 5,
                        "observed": len(found),
                        "missing_seeds": " ".join(str(seed) for seed in SEEDS if seed not in found),
                    }
                )
    return out


def noise_spectrum_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    agg_lookup = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"]): row
        for row in aggregate(rows)
    }
    summary = []
    for noise_type, noise_rate in CORAFULL_REGIMES:
        ours = agg_lookup.get(("corafull", noise_type, noise_rate, OURS))
        shuffled = agg_lookup.get(("corafull", noise_type, noise_rate, SHUFFLED))
        baseline_rows = [
            agg_lookup[("corafull", noise_type, noise_rate, method)]
            for method in BASELINES
            if ("corafull", noise_type, noise_rate, method) in agg_lookup
        ]
        if not ours or not shuffled or not baseline_rows:
            continue
        best = max(baseline_rows, key=lambda row: float(row["mean"]))
        summary.append(
            {
                "dataset": "corafull",
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "ours_mean": ours["mean"],
                "ours_std": ours["std"],
                "shuffled_mean": shuffled["mean"],
                "best_baseline": best["method"],
                "best_baseline_mean": best["mean"],
                "best_baseline_std": best["std"],
                "ours_minus_shuffled": float(ours["mean"]) - float(shuffled["mean"]),
                "ours_minus_best": float(ours["mean"]) - float(best["mean"]),
            }
        )
    return summary


def graphfree_aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in aggregate(rows):
        if row["dataset"] != "corafull" or row["method"] not in GRAPHFREE_METHODS:
            continue
        train_graph, inference_graph, note = GRAPHFREE_RESOURCES[row["method"]]
        out.append({**row, "train_uses_graph": train_graph, "inference_uses_graph": inference_graph, "note": note})
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{100 * mean:.2f}+-{100 * std:.2f}"


def write_report(path: Path, gate_agg, noise_summary, graphfree_rows, delta_rows, decisions, complete_rows) -> None:
    missing = [row for row in complete_rows if row["observed"] != row["expected"]]
    lines = [
        "# MLP-21 Dataset Replacement Report",
        "",
        "Protocol: percent 0.2/0.1/0.7, train-only noise, seeds 0-4.",
        "",
        "## Gate Decisions",
        "| Dataset | Decision | Best baseline | Ours-shuffled | Positive | Ours-best | Positive | Audit |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['dataset']} | {row['decision']} | {row['best_baseline']} | "
            f"{100 * row['vs_shuffled_mean_delta']:+.2f}pp | {row['vs_shuffled_positive']}/5 | "
            f"{100 * row['vs_best_baseline_mean_delta']:+.2f}pp | {row['vs_best_baseline_positive']}/5 | {row['audit_pass']} |"
        )
    lines.extend(["", "## Gate Aggregate", "| Dataset | Method | mean+-std |", "|---|---|---:|"])
    for row in gate_agg:
        if row["noise_type"] == "uniform" and math.isclose(float(row["noise_rate"]), 0.4) and row["method"] in METHODS:
            lines.append(f"| {row['dataset']} | {row['method']} | {_fmt_mean_std(row['mean'], row['std'])} |")

    lines.extend([
        "",
        "## CoraFull Noise Spectrum",
        "| Noise | Best baseline | Ours | Shuffled | Ours-best | Ours-shuffled |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in noise_summary:
        noise = f"{row['noise_type']}{row['noise_rate']}"
        best = f"{row['best_baseline']} {_fmt_mean_std(row['best_baseline_mean'], row['best_baseline_std'])}"
        lines.append(
            f"| {noise} | {best} | {_fmt_mean_std(row['ours_mean'], row['ours_std'])} | "
            f"{100 * row['shuffled_mean']:.2f} | {100 * row['ours_minus_best']:+.2f}pp | "
            f"{100 * row['ours_minus_shuffled']:+.2f}pp |"
        )

    lines.extend([
        "",
        "## CoraFull Graph-Free Table B",
        "| Method | Train graph | Inference graph | Note | mean+-std |",
        "|---|---|---|---|---:|",
    ])
    for row in graphfree_rows:
        lines.append(
            f"| {row['method']} | {row['train_uses_graph']} | {row['inference_uses_graph']} | "
            f"{row['note']} | {_fmt_mean_std(row['mean'], row['std'])} |"
        )

    lines.extend(["", "## Completeness", f"- Missing groups: {len(missing)}"])
    for row in missing:
        lines.append(
            f"- {row['kind']} / {row['dataset']} / {row['noise_type']}{row['noise_rate']} / "
            f"{row['method']}: missing seeds {row['missing_seeds']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    inputs = [path for path in sorted(args.run_dir.glob("*.csv")) if not path.name.startswith("mlp21_")]
    all_rows = load_rows(inputs, allowed_methods=ALL_METHODS, regimes=None, datasets=DATASETS)
    main_rows = [row for row in all_rows if row["method"] in METHODS]
    graph_rows = [row for row in all_rows if row["method"] in GRAPHFREE_METHODS]
    gate_rows = load_protocol_rows(inputs)

    agg_rows = aggregate(main_rows)
    delta_rows = paired_deltas(main_rows, DATASETS, CORAFULL_REGIMES)
    decisions = [decide_gate(gate_rows, dataset) for dataset in DATASETS]
    noise_summary = noise_spectrum_summary(main_rows)
    graphfree_rows = graphfree_aggregate(graph_rows)
    complete_rows = []
    complete_rows.extend(completeness_for(main_rows, ("amazoncom",), METHODS, (GATE_REGIME,), "gate"))
    complete_rows.extend(completeness_for(main_rows, ("corafull",), METHODS, CORAFULL_REGIMES, "noise_spectrum"))
    complete_rows.extend(completeness_for(graph_rows, ("corafull",), GRAPHFREE_METHODS, (GATE_REGIME,), "graphfree_tableb"))

    write_csv(args.run_dir / "mlp21_raw.csv", all_rows)
    write_csv(args.run_dir / "mlp21_aggregate.csv", agg_rows)
    write_csv(args.run_dir / "mlp21_paired_deltas.csv", delta_rows)
    write_csv(args.run_dir / "mlp21_gate_decisions.csv", decisions)
    write_csv(args.run_dir / "mlp21_noise_spectrum_aggregate.csv", noise_summary)
    write_csv(args.run_dir / "mlp21_tableb_aggregate.csv", graphfree_rows)
    write_csv(args.run_dir / "mlp21_completeness.csv", complete_rows)
    write_report(
        args.run_dir / "mlp21_report.md",
        agg_rows,
        noise_summary,
        graphfree_rows,
        delta_rows,
        decisions,
        complete_rows,
    )


if __name__ == "__main__":
    main()
