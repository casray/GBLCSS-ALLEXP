import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


LCSS = "frozengrace_gb_mlp_softdist_beta05"
SMALL_ONLY = "mlp_small_loss"
SMALL_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SMALL_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"
METHODS = (LCSS, SMALL_ONLY, SMALL_COMBO, SMALL_SHUFFLED)
SEEDS = tuple(range(5))
DATASETS_BY_RATE = {
    0.4: ("cora", "citeseer", "corafull", "amazonpho", "pubmed", "coauthorph"),
    0.6: ("cora", "citeseer", "citeseerfull", "corafull", "pubmed", "coauthorph"),
}


def _paired_t(values):
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values)
    if sample_std == 0:
        return (math.copysign(math.inf, mean), 0.0) if mean else (0.0, 1.0)
    t_value = mean / (sample_std / math.sqrt(len(values)))
    try:
        from scipy.stats import t as student_t

        p_value = float(2 * student_t.sf(abs(t_value), len(values) - 1))
    except ImportError:
        p_value = math.nan
    return t_value, p_value


def build_summary(rows):
    keyed = {}
    selected = []
    for row in rows:
        try:
            rate = float(row.get("noise_rate", "nan"))
        except ValueError:
            continue
        dataset = row.get("dataset")
        method = row.get("method")
        if rate not in DATASETS_BY_RATE or dataset not in DATASETS_BY_RATE[rate] or method not in METHODS:
            continue
        seed = int(row["seed"])
        key = (dataset, rate, method, seed)
        if key in keyed:
            raise ValueError(f"duplicate grid cell: {key}")
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed: {key}")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {key}")
        if row.get("noise_type") != "uniform" or row.get("noise_scope") != "train":
            raise ValueError(f"wrong noise protocol: {key}")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError(f"validation/test labels were modified: {key}")
        if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
            raise ValueError(f"modified labels are not restricted to train: {key}")
        normalized = dict(row)
        normalized["noise_rate"] = rate
        normalized["seed"] = seed
        normalized["test"] = float(row["test"])
        keyed[key] = normalized
        selected.append(normalized)

    for rate, datasets in DATASETS_BY_RATE.items():
        for dataset in datasets:
            for seed in SEEDS:
                hashes = set()
                for method in METHODS:
                    key = (dataset, rate, method, seed)
                    if key not in keyed:
                        raise ValueError(f"missing grid cell: {key}")
                    hashes.add((keyed[key]["split_hash"], keyed[key]["noise_hash"]))
                if len(hashes) != 1:
                    raise ValueError(f"protocol hash mismatch: dataset={dataset} rate={rate} seed={seed}")

    selected.sort(
        key=lambda row: (
            float(row["noise_rate"]),
            DATASETS_BY_RATE[float(row["noise_rate"])].index(row["dataset"]),
            int(row["seed"]),
            METHODS.index(row["method"]),
        )
    )

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["dataset"], row["noise_rate"], row["method"])].append(row["test"])
    aggregate = []
    for rate, datasets in DATASETS_BY_RATE.items():
        for dataset in datasets:
            for method in METHODS:
                values = grouped[(dataset, rate, method)]
                aggregate.append(
                    {
                        "dataset": dataset,
                        "noise_rate": rate,
                        "method": method,
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.pstdev(values),
                    }
                )

    seed_deltas = []
    for rate, datasets in DATASETS_BY_RATE.items():
        for dataset in datasets:
            for seed in SEEDS:
                value = lambda method: keyed[(dataset, rate, method, seed)]["test"]
                lcss = value(LCSS)
                small = value(SMALL_ONLY)
                combo = value(SMALL_COMBO)
                shuffled = value(SMALL_SHUFFLED)
                seed_deltas.append(
                    {
                        "dataset": dataset,
                        "noise_rate": rate,
                        "seed": seed,
                        "lcss": lcss,
                        "small_only": small,
                        "combo": combo,
                        "shuffled_combo": shuffled,
                        "combo_minus_lcss": combo - lcss,
                        "combo_minus_small_only": combo - small,
                        "synergy_delta": combo - max(lcss, small),
                        "structure_delta": combo - shuffled,
                    }
                )

    paired = []
    for rate, datasets in DATASETS_BY_RATE.items():
        for dataset in datasets:
            current = [
                row for row in seed_deltas if row["noise_rate"] == rate and row["dataset"] == dataset
            ]
            synergy = [row["synergy_delta"] for row in current]
            structure = [row["structure_delta"] for row in current]
            combo_lcss = [row["combo_minus_lcss"] for row in current]
            combo_small = [row["combo_minus_small_only"] for row in current]
            synergy_t, synergy_p = _paired_t(synergy)
            structure_t, structure_p = _paired_t(structure)
            paired.append(
                {
                    "dataset": dataset,
                    "noise_rate": rate,
                    "combo_minus_lcss": statistics.mean(combo_lcss),
                    "combo_minus_small_only": statistics.mean(combo_small),
                    "synergy_delta": statistics.mean(synergy),
                    "synergy_std": statistics.pstdev(synergy),
                    "synergy_positive_seeds": sum(value > 0 for value in synergy),
                    "synergy_t": synergy_t,
                    "synergy_p": synergy_p,
                    "structure_delta": statistics.mean(structure),
                    "structure_std": statistics.pstdev(structure),
                    "structure_positive_seeds": sum(value > 0 for value in structure),
                    "structure_t": structure_t,
                    "structure_p": structure_p,
                }
            )

    gates = []
    for rate, datasets in DATASETS_BY_RATE.items():
        current = [row for row in paired if row["noise_rate"] == rate]
        synergy_passes = sum(
            row["synergy_delta"] > 0 and row["synergy_positive_seeds"] >= 4 for row in current
        )
        structure_passes = sum(
            row["structure_delta"] >= 0.01 and row["structure_positive_seeds"] >= 4 for row in current
        )
        mean_synergy = statistics.mean(row["synergy_delta"] for row in current)
        minimum_synergy = min(row["synergy_delta"] for row in current)
        reasons = []
        if synergy_passes < 4:
            reasons.append("fewer_than_4_of_6_paired_synergy_passes")
        if mean_synergy < 0.01:
            reasons.append("mean_synergy_below_1pp")
        if minimum_synergy < -0.005:
            reasons.append("dataset_regression_over_0p5pp")
        if structure_passes < 4:
            reasons.append("fewer_than_4_of_6_structure_passes")
        gates.append(
            {
                "noise_rate": rate,
                "pass": int(not reasons),
                "qualifying_synergy_datasets": synergy_passes,
                "mean_synergy_delta": mean_synergy,
                "minimum_synergy_delta": minimum_synergy,
                "qualifying_structure_datasets": structure_passes,
                "reasons": ";".join(reasons),
            }
        )
    return selected, aggregate, seed_deltas, paired, gates


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(path, aggregate, paired, gates):
    lookup = {(row["dataset"], row["noise_rate"], row["method"]): row for row in aggregate}
    lines = ["# MLP-31 Representative Five-Seed Report", ""]
    for rate, datasets in DATASETS_BY_RATE.items():
        lines.extend(
            [
                f"## Uniform {rate:.1f}",
                "",
                "| Dataset | LCSS | Small-loss | Combination | Shuffled | Synergy | Positive | t / p | Real-shuffled | Positive |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in paired if item["noise_rate"] == rate]:
            dataset = row["dataset"]
            fmt = lambda method: (
                f"{100*lookup[(dataset, rate, method)]['mean']:.2f}+/-"
                f"{100*lookup[(dataset, rate, method)]['std']:.2f}"
            )
            lines.append(
                f"| {dataset} | {fmt(LCSS)} | {fmt(SMALL_ONLY)} | {fmt(SMALL_COMBO)} | "
                f"{fmt(SMALL_SHUFFLED)} | {100*row['synergy_delta']:+.2f}pp | "
                f"{row['synergy_positive_seeds']}/5 | {row['synergy_t']:.2f} / {row['synergy_p']:.4f} | "
                f"{100*row['structure_delta']:+.2f}pp | {row['structure_positive_seeds']}/5 |"
            )
        gate = next(item for item in gates if item["noise_rate"] == rate)
        lines.extend(
            [
                "",
                f"Gate: pass={gate['pass']}; paired synergy={gate['qualifying_synergy_datasets']}/6; "
                f"mean synergy={100*gate['mean_synergy_delta']:+.2f}pp; "
                f"minimum={100*gate['minimum_synergy_delta']:+.2f}pp; "
                f"structure={gate['qualifying_structure_datasets']}/6; reasons={gate['reasons'] or 'pass'}.",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(_read(path))
    combined, aggregate, seed_deltas, paired, gates = build_summary(rows)
    output = Path(args.output_dir)
    _write(output / "mlp31_combined_raw.csv", combined)
    _write(output / "mlp31_aggregate.csv", aggregate)
    _write(output / "mlp31_seed_deltas.csv", seed_deltas)
    _write(output / "mlp31_paired_deltas.csv", paired)
    _write(output / "mlp31_rate_gates.csv", gates)
    _report(output / "mlp31_report.md", aggregate, paired, gates)


if __name__ == "__main__":
    main()
