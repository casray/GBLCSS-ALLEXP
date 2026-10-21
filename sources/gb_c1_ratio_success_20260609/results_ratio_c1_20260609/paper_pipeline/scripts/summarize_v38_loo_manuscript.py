from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats


PAPER_ROOT = Path(__file__).resolve().parents[2]
NOISYGL_ROOT = PAPER_ROOT.parents[1] / "external" / "NoisyGL"
OUTPUT = PAPER_ROOT / "paper_pipeline" / "00_evidence" / "v38_loo_summary"

LOO_GRID = (
    NOISYGL_ROOT
    / "runs"
    / "mlp44_loo_tiesafe_fullspectrum_20260730"
    / "mlp44_combined_raw.csv"
)
BASELINE_GRID = (
    NOISYGL_ROOT
    / "runs"
    / "v31_baseline_refresh_20260728"
    / "baselines_combined_raw.csv"
)
NEAREST_GRID = (
    NOISYGL_ROOT
    / "runs"
    / "nearest_work_full_grid_20260728"
    / "recent_work_raw_combined.csv"
)
CLEAN_GRID = (
    NOISYGL_ROOT
    / "runs"
    / "mlp46_loo_clean_control_20260730"
    / "clean_uniform0p0.csv"
)

DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

LOO = "frozengrace_gb_mlp_softdist_beta05_loo"
LOO_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"
MAIN_BASELINES = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_sce",
    "mlp_elr",
    "mlp_logitclip",
)
NEAREST = (
    "postel_mlp_noipl_rm",
    "gsdn_faithful_core",
    "als_mlp_rm",
    "sals_mlp_adapted",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, float, int]:
    return (
        row["dataset"].lower(),
        row["noise_type"].lower(),
        float(row["noise_rate"]),
        int(float(row["seed"])),
    )


def validate_row(row: dict[str, str]) -> None:
    if row.get("status", "").lower() != "ok":
        raise ValueError(f"failed row: {row}")
    if row.get("noise_scope", "").lower() != "train":
        raise ValueError(f"non-train noise scope: {row}")
    if int(float(row["modified_val_count"])) != 0:
        raise ValueError(f"validation contamination: {row}")
    if int(float(row["modified_test_count"])) != 0:
        raise ValueError(f"test contamination: {row}")
    if int(float(row["modified_count"])) != int(
        float(row["modified_train_count"])
    ):
        raise ValueError(f"non-train modification: {row}")


def select(
    rows: list[dict[str, str]], methods: tuple[str, ...]
) -> dict[tuple[str, str, float, int, str], dict[str, str]]:
    result: dict[tuple[str, str, float, int, str], dict[str, str]] = {}
    for row in rows:
        method = row["method"]
        row_key = key(row)
        if (
            method not in methods
            or row_key[0] not in DATASETS
            or (row_key[1], row_key[2]) not in SETTINGS
            or row_key[3] not in SEEDS
        ):
            continue
        validate_row(row)
        full_key = (*row_key, method)
        if full_key in result:
            raise ValueError(f"duplicate row: {full_key}")
        result[full_key] = row
    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def paired_stats(deltas: list[float]) -> dict[str, object]:
    n = len(deltas)
    mean = statistics.mean(deltas)
    std = statistics.stdev(deltas)
    sem = std / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    test = stats.ttest_1samp(deltas, popmean=0.0)
    return {
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
        "t_statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
        "positive_seeds": sum(value > 0 for value in deltas),
        "negative_seeds": sum(value < 0 for value in deltas),
    }


def build() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    loo_source = read_csv(LOO_GRID)
    loo_rows = select(loo_source, (LOO, LOO_SHUFFLED, "mlp_ce", "mlp_jocor"))
    baseline_rows = select(read_csv(BASELINE_GRID), MAIN_BASELINES)
    nearest_rows = select(read_csv(NEAREST_GRID), NEAREST)
    all_rows = {**baseline_rows, **loo_rows}

    formal_methods = (LOO, LOO_SHUFFLED, *MAIN_BASELINES)
    expected = len(DATASETS) * len(SETTINGS) * len(SEEDS) * len(formal_methods)
    formal: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            for seed in SEEDS:
                hashes: set[tuple[str, str]] = set()
                for method in formal_methods:
                    row = all_rows[(dataset, noise_type, noise_rate, seed, method)]
                    hashes.add((row["split_hash"], row["noise_hash"]))
                    formal.append(dict(row))
                if len(hashes) != 1:
                    raise ValueError(
                        f"formal hash mismatch: {dataset}, "
                        f"{noise_type}{noise_rate}, seed {seed}"
                    )
    if len(formal) != expected:
        raise ValueError(f"expected {expected} formal rows, found {len(formal)}")

    aggregate: list[dict[str, object]] = []
    grouped: defaultdict[
        tuple[str, str, float, str], list[float]
    ] = defaultdict(list)
    for row in formal:
        grouped[
            (
                str(row["dataset"]),
                str(row["noise_type"]),
                float(row["noise_rate"]),
                str(row["method"]),
            )
        ].append(float(row["test"]))
    for group_key, values in sorted(grouped.items()):
        aggregate.append(
            {
                "dataset": group_key[0],
                "noise_type": group_key[1],
                "noise_rate": group_key[2],
                "method": group_key[3],
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
        )

    comparisons = (LOO_SHUFFLED, "mlp_ce", "mlp_jocor", *NEAREST)
    paired: list[dict[str, object]] = []
    for comparator in comparisons:
        family_indices: list[int] = []
        for dataset in DATASETS:
            for noise_type, noise_rate in SETTINGS:
                deltas = []
                for seed in SEEDS:
                    left = loo_rows[(dataset, noise_type, noise_rate, seed, LOO)]
                    if comparator in NEAREST:
                        right = nearest_rows[
                            (dataset, noise_type, noise_rate, seed, comparator)
                        ]
                    else:
                        right = loo_rows[
                            (dataset, noise_type, noise_rate, seed, comparator)
                        ]
                    if (
                        left["split_hash"],
                        left["noise_hash"],
                    ) != (
                        right["split_hash"],
                        right["noise_hash"],
                    ):
                        raise ValueError(
                            f"paired hash mismatch: {dataset}, "
                            f"{noise_type}{noise_rate}, seed {seed}, "
                            f"{comparator}"
                        )
                    deltas.append(float(left["test"]) - float(right["test"]))
                row = {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "comparator": comparator,
                    **paired_stats(deltas),
                }
                family_indices.append(len(paired))
                paired.append(row)
        adjusted = holm_adjust(
            [float(paired[index]["p_value_two_sided"]) for index in family_indices]
        )
        for index, adjusted_p in zip(family_indices, adjusted):
            paired[index]["holm_adjusted_p"] = adjusted_p
            paired[index]["holm_significant_0p05"] = int(adjusted_p < 0.05)

    headline: list[dict[str, object]] = []
    aggregate_lookup = {
        (
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            str(row["method"]),
        ): float(row["mean"])
        for row in aggregate
    }
    best_count = 0
    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            ours = aggregate_lookup[(dataset, noise_type, noise_rate, LOO)]
            baseline_means = [
                aggregate_lookup[(dataset, noise_type, noise_rate, method)]
                for method in MAIN_BASELINES
            ]
            best_count += int(ours >= max(baseline_means))
    headline.append(
        {
            "metric": "settings_best_or_tied_among_seven_main_baselines",
            "value": best_count,
            "denominator": 25,
        }
    )
    for comparator in comparisons:
        family = [row for row in paired if row["comparator"] == comparator]
        headline.extend(
            [
                {
                    "metric": f"mean_delta_pp_vs_{comparator}",
                    "value": 100
                    * statistics.mean(
                        float(row["mean_delta"]) for row in family
                    ),
                    "denominator": 25,
                },
                {
                    "metric": f"positive_settings_vs_{comparator}",
                    "value": sum(float(row["mean_delta"]) > 0 for row in family),
                    "denominator": 25,
                },
                {
                    "metric": f"holm_significant_settings_vs_{comparator}",
                    "value": sum(
                        int(row["holm_significant_0p05"]) for row in family
                    ),
                    "denominator": 25,
                },
            ]
        )
    return formal, aggregate, paired, headline


def main() -> None:
    formal, aggregate, paired, headline = build()
    write_csv(OUTPUT / "formal_raw.csv", formal)
    write_csv(OUTPUT / "formal_aggregate.csv", aggregate)
    write_csv(OUTPUT / "paired_tests.csv", paired)
    write_csv(OUTPUT / "headline_metrics.csv", headline)

    lookup = {str(row["metric"]): row for row in headline}
    lines = [
        "# v38 Formal LOO Manuscript Summary",
        "",
        "Formal method identity: `tie_mode=distribution_only`, "
        "`leave_one_out=true`.",
        "",
        f"- Best or tied among seven main baselines: "
        f"{int(lookup['settings_best_or_tied_among_seven_main_baselines']['value'])}/25.",
    ]
    for comparator in (LOO_SHUFFLED, "mlp_ce", "mlp_jocor", *NEAREST):
        mean = lookup[f"mean_delta_pp_vs_{comparator}"]["value"]
        positive = lookup[f"positive_settings_vs_{comparator}"]["value"]
        significant = lookup[
            f"holm_significant_settings_vs_{comparator}"
        ]["value"]
        lines.append(
            f"- Versus `{comparator}`: mean {mean:+.2f}pp, "
            f"positive {int(positive)}/25, Holm-significant "
            f"{int(significant)}/25."
        )
    lines.extend(
        [
            "",
            "The clean-label control is reported separately and is not "
            "included in these noisy-label headline counts.",
        ]
    )
    (OUTPUT / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(OUTPUT / "report.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

