from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def _aggregate(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if 'method' not in raw and 'representation_source' in raw:
        raw = raw.rename(columns={'representation_source': 'method'})
    required = {"dataset", "method", "seed", "test_accuracy"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    group = [column for column in ["dataset", "noise_type", "noise_rate", "validation_protocol", "method"] if column in raw]
    aggregate = (
        raw.groupby(group, as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
        )
        .sort_values(group)
    )
    aggregate["mean_test_accuracy_pct"] = 100.0 * aggregate["mean_test_accuracy"]
    aggregate["std_test_accuracy_pct"] = 100.0 * aggregate["std_test_accuracy"]
    return aggregate


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _paired_statistics() -> pd.DataFrame:
    main = pd.read_csv(ROOT / "results" / "main_tables" / "table2_per_seed.csv")
    random = pd.read_csv(ROOT / "results" / "ablations" / "random_grouping_per_seed.csv")
    proposed = main[main.method == "GB-LCSS"]
    comparators = {
        "Random size-matched grouping": random,
        "MLP+CE": main[main.method == "MLP+CE"],
        "JoCoR": main[main.method == "JoCoR"],
    }
    rows: list[dict[str, object]] = []
    keys = ["dataset", "noise_type", "noise_rate", "seed"]
    for comparator, frame in comparators.items():
        paired = proposed.merge(frame, on=keys, suffixes=("_gblcss", "_comparator"))
        family_rows = []
        for setting, group in paired.groupby(keys[:-1], sort=True):
            delta = group.test_accuracy_gblcss - group.test_accuracy_comparator
            mean = float(delta.mean())
            std = float(delta.std(ddof=1))
            sem = std / len(delta) ** 0.5
            margin = float(stats.t.ppf(0.975, len(delta) - 1) * sem)
            test = stats.ttest_1samp(delta, popmean=0.0)
            family_rows.append(
                {
                    "dataset": setting[0],
                    "noise_type": setting[1],
                    "noise_rate": setting[2],
                    "comparison": f"GB-LCSS - {comparator}",
                    "mean_difference": mean,
                    "mean_difference_pp": 100.0 * mean,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "ci95_low_pp": 100.0 * (mean - margin),
                    "ci95_high_pp": 100.0 * (mean + margin),
                    "positive_seeds": int((delta > 0).sum()),
                    "p_value_two_sided": float(test.pvalue),
                }
            )
        adjusted = _holm_adjust([float(row["p_value_two_sided"]) for row in family_rows])
        for row, value in zip(family_rows, adjusted):
            row["holm_adjusted_p"] = value
        rows.extend(family_rows)
    return pd.DataFrame(rows).sort_values(["dataset", "noise_type", "noise_rate", "comparison"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate aggregate tables from per-seed results.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "reproduced"))
    parser.add_argument("--input", type=Path, default=None, help="Aggregate a CSV from a new training run.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.input is not None:
        output = output_dir / f"{args.input.stem}_aggregate.csv"
        _aggregate(args.input).to_csv(output, index=False)
        print(output)
        return

    inputs = {
        "table3": ROOT / "results" / "main_tables" / "table2_per_seed.csv",
        "table4": ROOT / "results" / "main_tables" / "table3_per_seed.csv",
        "table8": ROOT / "results" / "distillation" / "table4_per_seed.csv",
        "appendix_c": ROOT / "results" / "ablations" / "validation_protocol_per_seed.csv",
        "appendix_e": ROOT / "results" / "main_tables" / "structure_soft_targets_full_spectrum_per_seed.csv",
        "table5_smoothing": ROOT / "results/ablations/explicit_smoothing_per_seed.csv",
        "table5_self_vote": ROOT / "results/ablations/self_vote_ablation_per_seed.csv",
        "random_grouping": ROOT / "results/ablations/random_grouping_per_seed.csv",
        "table6": ROOT / "results/ablations/grouping_interfaces_per_seed.csv",
        "table7": ROOT / "results/ablations/representation_source_per_seed.csv",
        "appendix_g2": ROOT / "results/ablations/supplementary_uniform0p4_per_seed.csv",
        "appendix_g3": ROOT / "results/ablations/citeseerfull_full_spectrum_per_seed.csv",
    }
    for name, path in inputs.items():
        aggregate = _aggregate(path)
        output = output_dir / f"{name}_aggregate.csv"
        aggregate.to_csv(output, index=False)
        print(f"{name}: {len(aggregate)} aggregate rows -> {output}")

    paired = _paired_statistics()
    paired_output = output_dir / "table_b1_paired_statistics.csv"
    paired.to_csv(paired_output, index=False)
    print(f"table_b1: {len(paired)} paired rows -> {paired_output}")


if __name__ == "__main__":
    main()
