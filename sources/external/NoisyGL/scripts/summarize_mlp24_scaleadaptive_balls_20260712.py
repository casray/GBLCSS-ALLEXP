from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp24_scaleadaptive_balls_20260712"

ALL_DATASETS = (
    "cora",
    "citeseer",
    "corafull",
    "coauthorcs",
    "amazonpho",
    "citeseerfull",
    "coauthorph",
    "pubmed",
    "amazoncom",
    "coraml",
    "attribwiki",
)
AFFECTED_DATASETS = (
    "citeseerfull",
    "corafull",
    "coauthorcs",
    "amazonpho",
    "pubmed",
    "amazoncom",
    "coauthorph",
)
CORE_AFFECTED = ("corafull", "coauthorcs", "amazonpho")
REAL_METHOD = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED_METHOD = "frozengrace_gb_mlp_softdist_beta05_shuffled"


def _float(value, default=0.0):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _int(value, default=0):
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protocol_row_ok(row: dict, noise_type: str, noise_rate: float) -> bool:
    return (
        str(row.get("status", "ok")).strip().lower() in {"", "ok"}
        and str(row.get("noise_scope", "train")).strip().lower() == "train"
        and str(row.get("noise_type", "")).strip().lower() == str(noise_type).lower()
        and math.isclose(_float(row.get("noise_rate")), float(noise_rate), abs_tol=1e-12)
        and math.isclose(_float(row.get("train_percent"), 0.2), 0.2, abs_tol=1e-12)
        and math.isclose(_float(row.get("val_percent"), 0.1), 0.1, abs_tol=1e-12)
        and math.isclose(_float(row.get("test_percent"), 0.7), 0.7, abs_tol=1e-12)
        and _int(row.get("modified_val_count")) == 0
        and _int(row.get("modified_test_count")) == 0
        and _int(row.get("modified_count")) == _int(row.get("modified_train_count"))
    )


def variant_for(row: dict) -> str:
    explicit = str(row.get("variant", "")).strip().lower()
    if explicit:
        return explicit
    method = str(row.get("method") or row.get("base_method") or "").strip()
    policy = str(row.get("max_balls_policy") or row.get("gb_max_balls_policy") or "fixed").strip().lower()
    if method == SHUFFLED_METHOD:
        return f"{policy}_shuffled"
    if method == REAL_METHOD:
        return policy
    return method


def normalize_downstream(row: dict, source: Path | str = "") -> dict[str, object]:
    normalized = dict(row)
    normalized.update(
        {
            "dataset": str(row.get("dataset", "")).strip().lower(),
            "seed": _int(row.get("seed")),
            "noise_type": str(row.get("noise_type", "")).strip().lower(),
            "noise_rate": _float(row.get("noise_rate")),
            "test": _float(row.get("test") or row.get("test_acc")),
            "variant": variant_for(row),
            "source": str(source),
        }
    )
    return normalized


def paired_deltas(rows: list[dict], left_variant: str, right_variant: str) -> list[dict[str, object]]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for raw in rows:
        row = normalize_downstream(raw)
        key = (
            row["dataset"],
            int(row["seed"]),
            row.get("noise_type", ""),
            float(row.get("noise_rate", 0.0)),
        )
        grouped[key][str(row["variant"])] = row

    out = []
    for key, variants in sorted(grouped.items()):
        if left_variant not in variants or right_variant not in variants:
            continue
        left = variants[left_variant]
        right = variants[right_variant]
        if left.get("split_hash", "") != right.get("split_hash", "") or left.get("noise_hash", "") != right.get("noise_hash", ""):
            raise ValueError(f"hash mismatch for {key}: {left_variant} vs {right_variant}")
        out.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "noise_type": key[2],
                "noise_rate": key[3],
                "left_variant": left_variant,
                "right_variant": right_variant,
                "left_test": float(left["test"]),
                "right_test": float(right["test"]),
                "delta": float(left["test"]) - float(right["test"]),
                "split_hash": left.get("split_hash", ""),
                "noise_hash": left.get("noise_hash", ""),
            }
        )
    return out


def geometry_gate(rows: list[dict], require_all: bool = True) -> dict[str, object]:
    reasons: list[str] = []
    by_key = {
        (str(row.get("dataset", "")).lower(), str(row.get("variant", "")).lower()): row
        for row in rows
    }
    fixed = by_key.get(("coauthorph", "fixed"))
    adaptive = by_key.get(("coauthorph", "sqrt_nodes"))
    if fixed is None or adaptive is None:
        reasons.append("missing_coauthorph_geometry")
    else:
        if str(fixed.get("status", "ok")).lower() not in {"", "ok"} or str(adaptive.get("status", "ok")).lower() not in {"", "ok"}:
            reasons.append("physics_geometry_failed")
        fixed_size = _float(fixed.get("mean_ball_size"))
        adaptive_size = _float(adaptive.get("mean_ball_size"))
        if _int(adaptive.get("num_balls")) <= 64 or fixed_size <= 0 or adaptive_size > 0.75 * fixed_size:
            reasons.append("physics_partition_not_exercised")
        fixed_time = _float(fixed.get("ball_build_time_seconds"))
        adaptive_time = _float(adaptive.get("ball_build_time_seconds"))
        if adaptive_time > 600 or (fixed_time > 0 and adaptive_time > 10 * fixed_time):
            reasons.append("physics_build_too_slow")

    if require_all:
        for dataset in ALL_DATASETS:
            for variant in ("fixed", "sqrt_nodes", "sqrt_nodes_shuffled"):
                if (dataset, variant) not in by_key:
                    reasons.append(f"missing_geometry:{dataset}:{variant}")
        for dataset in ("cora", "citeseer", "coraml", "attribwiki"):
            adaptive_row = by_key.get((dataset, "sqrt_nodes"))
            if adaptive_row is not None and not all(
                _int(adaptive_row.get(field)) == 1
                for field in ("assignment_equal_fixed", "soft_target_equal_fixed", "alpha_equal_fixed")
            ):
                reasons.append(f"small_graph_not_identical:{dataset}")
        for dataset in ALL_DATASETS:
            shuffled_row = by_key.get((dataset, "sqrt_nodes_shuffled"))
            if shuffled_row is not None:
                size_match = shuffled_row.get("size_multiset_equal_source")
                if size_match is None or str(size_match).strip() == "":
                    size_match = shuffled_row.get("size_multiset_equal_adaptive")
                if _int(size_match) != 1:
                    reasons.append(f"shuffled_size_mismatch:{dataset}")

    for row in rows:
        if str(row.get("status", "ok")).lower() not in {"", "ok"}:
            reasons.append(f"failed_geometry:{row.get('dataset')}:{row.get('variant')}")
        if _int(row.get("num_balls")) <= 1:
            reasons.append(f"collapsed:{row.get('dataset')}:{row.get('variant')}")
        if _float(row.get("mean_train_per_ball")) < 2:
            reasons.append(f"low_train_votes:{row.get('dataset')}:{row.get('variant')}")
        if _int(row.get("alpha_positive_train_count")) <= 0:
            reasons.append(f"zero_target_coverage:{row.get('dataset')}:{row.get('variant')}")
    return {"stage": "geometry", "pass": int(not reasons), "reasons": ";".join(sorted(set(reasons)))}


def _variant_rows(rows: list[dict]) -> dict[tuple[str, int, str], dict[str, object]]:
    indexed: dict[tuple[str, int, str], dict[str, object]] = {}
    for raw in rows:
        row = normalize_downstream(raw)
        indexed[(str(row["dataset"]), int(row["seed"]), str(row["variant"]))] = row
    return indexed


def _paired_dataset_stats(
    rows: list[dict],
    left_variant: str,
    right_variant: str,
    datasets: tuple[str, ...] | list[str],
    expected_seeds: tuple[int, ...] | list[int],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    indexed = _variant_rows(rows)
    stats: dict[str, dict[str, object]] = {}
    reasons: list[str] = []
    for dataset in datasets:
        deltas: list[float] = []
        for seed in expected_seeds:
            left = indexed.get((dataset, seed, left_variant))
            right = indexed.get((dataset, seed, right_variant))
            if left is None or right is None:
                reasons.append(f"missing_pair:{dataset}:{seed}:{left_variant}:{right_variant}")
                continue
            if not protocol_row_ok(left, "uniform", 0.4) or not protocol_row_ok(right, "uniform", 0.4):
                reasons.append(f"invalid_protocol:{dataset}:{seed}:{left_variant}:{right_variant}")
                continue
            if left.get("split_hash", "") != right.get("split_hash", "") or left.get("noise_hash", "") != right.get("noise_hash", ""):
                reasons.append(f"hash_mismatch:{dataset}:{seed}:{left_variant}:{right_variant}")
                continue
            deltas.append(float(left["test"]) - float(right["test"]))
        stats[dataset] = {
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas) if deltas else float("nan"),
            "positive_seeds": sum(delta > 0 for delta in deltas),
            "nonnegative_seeds": sum(delta >= 0 for delta in deltas),
        }
    return stats, reasons


def smoke_gate(rows: list[dict]) -> dict[str, object]:
    reasons: list[str] = []
    adaptive_fixed, pair_reasons = _paired_dataset_stats(
        rows, "sqrt_nodes", "fixed", list(AFFECTED_DATASETS), [0]
    )
    reasons.extend(pair_reasons)
    adaptive_shuffled, pair_reasons = _paired_dataset_stats(
        rows, "sqrt_nodes", "sqrt_nodes_shuffled", ["coauthorph"], [0]
    )
    reasons.extend(pair_reasons)

    if not reasons:
        physics_delta = float(adaptive_fixed["coauthorph"]["mean_delta"])
        if physics_delta < 0.005:
            reasons.append("physics_adaptive_vs_fixed_below_0p5pp")
        affected_mean = statistics.mean(
            float(adaptive_fixed[dataset]["mean_delta"]) for dataset in AFFECTED_DATASETS
        )
        if affected_mean < 0.0025:
            reasons.append("affected_mean_adaptive_vs_fixed_below_0p25pp")
        for dataset in CORE_AFFECTED:
            if float(adaptive_fixed[dataset]["mean_delta"]) < -0.01:
                reasons.append(f"core_drop_over_1pp:{dataset}")
        if float(adaptive_shuffled["coauthorph"]["mean_delta"]) < 0.01:
            reasons.append("physics_adaptive_vs_shuffled_below_1pp")

    return {
        "stage": "seed0_smoke",
        "pass": int(not reasons),
        "reasons": ";".join(sorted(set(reasons))),
    }


def five_seed_gate(rows: list[dict]) -> dict[str, object]:
    reasons: list[str] = []
    seeds = list(range(5))
    adaptive_fixed, pair_reasons = _paired_dataset_stats(
        rows, "sqrt_nodes", "fixed", list(AFFECTED_DATASETS), seeds
    )
    reasons.extend(pair_reasons)
    adaptive_shuffled, pair_reasons = _paired_dataset_stats(
        rows, "sqrt_nodes", "sqrt_nodes_shuffled", list(AFFECTED_DATASETS), seeds
    )
    reasons.extend(pair_reasons)
    adaptive_jocor, pair_reasons = _paired_dataset_stats(
        rows, "sqrt_nodes", "mlp_jocor", ["coauthorph"], seeds
    )
    reasons.extend(pair_reasons)

    if not reasons:
        physics = adaptive_fixed["coauthorph"]
        if float(physics["mean_delta"]) < 0.01 or int(physics["positive_seeds"]) < 4:
            reasons.append("physics_adaptive_vs_fixed_gate_failed")
        physics_jocor = adaptive_jocor["coauthorph"]
        if float(physics_jocor["mean_delta"]) < 0.01 or int(physics_jocor["positive_seeds"]) < 4:
            reasons.append("physics_adaptive_vs_jocor_gate_failed")

        affected_mean = statistics.mean(
            float(adaptive_fixed[dataset]["mean_delta"]) for dataset in AFFECTED_DATASETS
        )
        if affected_mean < 0.005:
            reasons.append("affected_mean_adaptive_vs_fixed_below_0p5pp")
        if sum(float(adaptive_fixed[dataset]["mean_delta"]) >= 0 for dataset in AFFECTED_DATASETS) < 5:
            reasons.append("fewer_than_5_of_7_affected_nonnegative")
        for dataset in CORE_AFFECTED:
            if float(adaptive_fixed[dataset]["mean_delta"]) < -0.005:
                reasons.append(f"core_drop_over_0p5pp:{dataset}")

        shuffled_passes = sum(
            float(adaptive_shuffled[dataset]["mean_delta"]) >= 0.01
            and int(adaptive_shuffled[dataset]["positive_seeds"]) >= 4
            for dataset in AFFECTED_DATASETS
        )
        if shuffled_passes < 6:
            reasons.append("fewer_than_6_of_7_beat_shuffled")

    return {
        "stage": "uniform0p4_5seed",
        "pass": int(not reasons),
        "reasons": ";".join(sorted(set(reasons))),
    }


def aggregate(rows: list[dict]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for raw in rows:
        row = normalize_downstream(raw)
        grouped[(row["dataset"], row["noise_type"], row["noise_rate"], row["variant"])].append(float(row["test"]))
    return [
        {
            "dataset": key[0],
            "noise_type": key[1],
            "noise_rate": key[2],
            "variant": key[3],
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in sorted(grouped.items())
    ]


def _deduplicate(rows: list[dict]) -> list[dict[str, object]]:
    indexed: dict[tuple, dict[str, object]] = {}
    for raw in rows:
        row = normalize_downstream(raw, raw.get("source", ""))
        key = (
            row["dataset"],
            int(row["seed"]),
            row["noise_type"],
            float(row["noise_rate"]),
            row["variant"],
        )
        indexed[key] = row
    return [indexed[key] for key in sorted(indexed)]


def _collect(run_dir: Path, patterns: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            rows.extend(normalize_downstream(row, path) for row in read_csv(path))
    return _deduplicate(rows)


def completeness_rows(
    rows: list[dict],
    stage: str,
    datasets: tuple[str, ...] | list[str],
    variants: tuple[str, ...] | list[str],
    seeds: tuple[int, ...] | list[int],
) -> list[dict[str, object]]:
    indexed = _variant_rows(rows)
    out: list[dict[str, object]] = []
    for dataset in datasets:
        for variant in variants:
            present = sum((dataset, seed, variant) in indexed for seed in seeds)
            out.append(
                {
                    "stage": stage,
                    "dataset": dataset,
                    "variant": variant,
                    "expected": len(seeds),
                    "present": present,
                    "complete": int(present == len(seeds)),
                }
            )
    return out


def _format_pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def write_report(
    path: Path,
    geometry_rows: list[dict],
    downstream_rows: list[dict],
    gates: list[dict[str, object]],
) -> None:
    lines = [
        "# MLP-24 Scale-Adaptive Ball Report",
        "",
        "Candidate rule: `max(64, ceil(sqrt(|V|)))`. Only the ball-count cap changes.",
        "",
        "## Gate decisions",
        "",
        "| Stage | Pass | Reasons |",
        "|---|---:|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate['stage']} | {gate['pass']} | {gate.get('reasons', '') or '-'} |")
    if geometry_rows:
        lines.extend(
            [
                "",
                "## Geometry",
                "",
                "| Dataset | Variant | Balls | Mean size | Mean train votes | Build seconds |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in geometry_rows:
            lines.append(
                f"| {row.get('dataset')} | {row.get('variant')} | {row.get('num_balls')} | "
                f"{_float(row.get('mean_ball_size')):.2f} | {_float(row.get('mean_train_per_ball')):.2f} | "
                f"{_float(row.get('ball_build_time_seconds')):.2f} |"
            )
    if downstream_rows:
        lines.extend(
            [
                "",
                "## Downstream aggregate",
                "",
                "| Dataset | Noise | Variant | n | Accuracy |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in aggregate(downstream_rows):
            lines.append(
                f"| {row['dataset']} | {row['noise_type']} {row['noise_rate']} | {row['variant']} | "
                f"{row['n']} | {100 * float(row['mean']):.2f} +/- {100 * float(row['std']):.2f} |"
            )
        adaptive_fixed, _ = _paired_dataset_stats(
            downstream_rows, "sqrt_nodes", "fixed", list(AFFECTED_DATASETS), [0]
        )
        adaptive_shuffled, _ = _paired_dataset_stats(
            downstream_rows, "sqrt_nodes", "sqrt_nodes_shuffled", list(AFFECTED_DATASETS), [0]
        )
        lines.extend(
            [
                "",
                "## Seed-0 paired effects",
                "",
                "| Dataset | Adaptive - fixed | Adaptive - shuffled |",
                "|---|---:|---:|",
            ]
        )
        for dataset in AFFECTED_DATASETS:
            lines.append(
                f"| {dataset} | {_format_pp(float(adaptive_fixed[dataset]['mean_delta']))} | "
                f"{_format_pp(float(adaptive_shuffled[dataset]['mean_delta']))} |"
            )
        affected_mean = statistics.mean(
            float(adaptive_fixed[dataset]["mean_delta"]) for dataset in AFFECTED_DATASETS
        )
        lines.extend(
            [
                "",
                "## Final decision",
                "",
                f"The seven affected datasets improve by {_format_pp(affected_mean)} on average at seed 0, "
                f"but Coauthor-Physics improves by only {_format_pp(float(adaptive_fixed['coauthorph']['mean_delta']))} "
                "against the preregistered +0.50pp gate.",
                "",
                "**Decision: STOP after Phase B.** Keep fixed `max_balls=64`; do not run Phase C or Phase D, "
                "do not change the paper method, and record the scale-adaptive rule as a negative/boundary result.",
                "",
                "Interpretation: increasing the cap materially changes large-graph geometry and preserves a strong "
                "real-vs-shuffled signal, but the finer partition adds little downstream information beyond the "
                "existing fixed-64 consensus targets.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP-24 scale-adaptive ball experiments.")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = read_csv(args.run_dir / "mlp24_geometry.csv")
    smoke_rows = _collect(args.run_dir, ("smoke_*.csv",))
    five_new_rows = _collect(args.run_dir, ("five_*.csv",))
    five_rows = _deduplicate(smoke_rows + five_new_rows) if five_new_rows else []
    full_rows = _collect(args.run_dir, ("full_*.csv",))
    if smoke_rows:
        write_csv(args.run_dir / "mlp24_smoke_seed0.csv", smoke_rows)
    if five_rows:
        write_csv(args.run_dir / "mlp24_uniform0p4_5seed.csv", five_rows)
    if full_rows:
        write_csv(args.run_dir / "mlp24_full_spectrum.csv", full_rows)

    downstream_rows = full_rows or five_rows or smoke_rows
    gates = [geometry_gate(geometry_rows)] if geometry_rows else []
    if smoke_rows:
        gates.append(smoke_gate(smoke_rows))
    if five_rows and all(
        row["present"] == 5
        for row in completeness_rows(
            five_rows,
            "uniform0p4_5seed",
            list(AFFECTED_DATASETS),
            ("fixed", "sqrt_nodes", "sqrt_nodes_shuffled", "mlp_jocor"),
            range(5),
        )
    ):
        gates.append(five_seed_gate(five_rows))
    paired = []
    if downstream_rows:
        paired.extend(paired_deltas(downstream_rows, "sqrt_nodes", "fixed"))
        paired.extend(paired_deltas(downstream_rows, "sqrt_nodes", "sqrt_nodes_shuffled"))
    write_csv(args.run_dir / "mlp24_aggregate.csv", aggregate(downstream_rows))
    write_csv(args.run_dir / "mlp24_paired_deltas.csv", paired)
    write_csv(args.run_dir / "mlp24_gate_decisions.csv", gates)
    completeness: list[dict[str, object]] = []
    if smoke_rows:
        completeness.extend(
            completeness_rows(
                smoke_rows,
                "seed0_smoke",
                list(ALL_DATASETS),
                ("fixed", "sqrt_nodes", "sqrt_nodes_shuffled", "mlp_jocor"),
                [0],
            )
        )
    if five_rows:
        completeness.extend(
            completeness_rows(
                five_rows,
                "uniform0p4_5seed",
                list(ALL_DATASETS),
                ("fixed", "sqrt_nodes", "sqrt_nodes_shuffled", "mlp_jocor"),
                range(5),
            )
        )
    write_csv(args.run_dir / "mlp24_completeness.csv", completeness)
    write_report(args.run_dir / "mlp24_report.md", geometry_rows, downstream_rows, gates)
    print(args.run_dir / "mlp24_gate_decisions.csv")


if __name__ == "__main__":
    main()
