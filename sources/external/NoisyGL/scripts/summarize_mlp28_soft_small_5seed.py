import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


LCSS = "frozengrace_gb_mlp_softdist_beta05"
SMALL_ONLY = "mlp_small_loss"
SMALL_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SMALL_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"
DATASETS = ("cora", "citeseer", "citeseerfull")
METHODS = (LCSS, SMALL_ONLY, SMALL_COMBO, SMALL_SHUFFLED)
SEEDS = tuple(range(5))


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


def build_summary(rows):
    selected = [row for row in rows if row.get("dataset") in DATASETS and row.get("method") in METHODS]
    by_key = {}
    for row in selected:
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {row}")
        if row.get("noise_type") != "uniform" or abs(float(row["noise_rate"]) - 0.6) > 1e-12:
            raise ValueError("five-seed confirmation requires uniform 0.6")
        if row.get("noise_scope") != "train":
            raise ValueError("five-seed confirmation requires train-only noise")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError("validation/test labels were modified")
        by_key[(row["dataset"], row["method"], int(row["seed"]))] = row

    for dataset in DATASETS:
        for seed in SEEDS:
            hashes = set()
            for method in METHODS:
                key = (dataset, method, seed)
                if key not in by_key:
                    raise ValueError(f"missing row: {key}")
                hashes.add((by_key[key]["split_hash"], by_key[key]["noise_hash"]))
            if len(hashes) != 1:
                raise ValueError(f"protocol hash mismatch: {dataset} seed={seed}")

    aggregate = []
    grouped = defaultdict(list)
    for (dataset, method, _seed), row in by_key.items():
        grouped[(dataset, method)].append(float(row["test"]))
    for (dataset, method), values in sorted(grouped.items()):
        aggregate.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
            }
        )

    deltas = []
    for dataset in DATASETS:
        synergy = []
        structure = []
        combo_minus_lcss = []
        combo_minus_small = []
        for seed in SEEDS:
            value = lambda method: float(by_key[(dataset, method, seed)]["test"])
            combo = value(SMALL_COMBO)
            combo_minus_lcss.append(combo - value(LCSS))
            combo_minus_small.append(combo - value(SMALL_ONLY))
            synergy.append(combo - max(value(LCSS), value(SMALL_ONLY)))
            structure.append(combo - value(SMALL_SHUFFLED))
        deltas.append(
            {
                "dataset": dataset,
                "combo_minus_lcss": statistics.mean(combo_minus_lcss),
                "combo_minus_small_only": statistics.mean(combo_minus_small),
                "synergy_delta": statistics.mean(synergy),
                "synergy_positive_seeds": sum(value > 0 for value in synergy),
                "structure_delta": statistics.mean(structure),
                "structure_positive_seeds": sum(value > 0 for value in structure),
            }
        )

    qualifying_synergy = sum(
        row["synergy_delta"] > 0 and row["synergy_positive_seeds"] >= 4 for row in deltas
    )
    qualifying_structure = sum(
        row["structure_delta"] >= 0.01 and row["structure_positive_seeds"] >= 4 for row in deltas
    )
    mean_synergy = statistics.mean(row["synergy_delta"] for row in deltas)
    minimum_synergy = min(row["synergy_delta"] for row in deltas)
    reasons = []
    if qualifying_synergy < 2:
        reasons.append("fewer_than_2_of_3_paired_synergy_passes")
    if mean_synergy < 0.01:
        reasons.append("mean_synergy_below_1pp")
    if minimum_synergy < -0.005:
        reasons.append("dataset_regression_over_0p5pp")
    if qualifying_structure < 2:
        reasons.append("fewer_than_2_of_3_structure_passes")
    gate = {
        "pass": int(not reasons),
        "qualifying_synergy_datasets": qualifying_synergy,
        "mean_synergy_delta": mean_synergy,
        "minimum_synergy_delta": minimum_synergy,
        "qualifying_structure_datasets": qualifying_structure,
        "reasons": ";".join(reasons),
    }
    return aggregate, deltas, gate


def _report(path, aggregate, deltas, gate):
    lookup = {(row["dataset"], row["method"]): row for row in aggregate}
    lines = [
        "# MLP-28 LCSS + Soft-Small-Loss Five-Seed Report",
        "",
        "| Dataset | LCSS | Small-loss-only | Combination | Shuffled combination | Synergy | Synergy positive | Real-shuffled | Structure positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        dataset = row["dataset"]
        fmt = lambda method: f"{100*lookup[(dataset, method)]['mean']:.2f}+/-{100*lookup[(dataset, method)]['std']:.2f}"
        lines.append(
            f"| {dataset} | {fmt(LCSS)} | {fmt(SMALL_ONLY)} | {fmt(SMALL_COMBO)} | {fmt(SMALL_SHUFFLED)} | "
            f"{100*row['synergy_delta']:+.2f}pp | {row['synergy_positive_seeds']}/5 | "
            f"{100*row['structure_delta']:+.2f}pp | {row['structure_positive_seeds']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Pass: {gate['pass']}",
            f"- Qualifying synergy datasets: {gate['qualifying_synergy_datasets']}/3",
            f"- Mean synergy: {100*gate['mean_synergy_delta']:+.2f}pp",
            f"- Minimum dataset synergy: {100*gate['minimum_synergy_delta']:+.2f}pp",
            f"- Qualifying structure datasets: {gate['qualifying_structure_datasets']}/3",
            f"- Reasons: {gate['reasons'] or 'pass'}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0_input", required=True)
    parser.add_argument("--additional_input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = _read(args.seed0_input) + _read(args.additional_input)
    aggregate, deltas, gate = build_summary(rows)
    selected = [row for row in rows if row.get("dataset") in DATASETS and row.get("method") in METHODS]
    output = Path(args.output_dir)
    _write(output / "mlp28_soft_small_5seed_raw.csv", selected)
    _write(output / "mlp28_soft_small_5seed_aggregate.csv", aggregate)
    _write(output / "mlp28_soft_small_5seed_paired_deltas.csv", deltas)
    _write(output / "mlp28_soft_small_5seed_gate.csv", [gate])
    _report(output / "mlp28_soft_small_5seed_report.md", aggregate, deltas, gate)


if __name__ == "__main__":
    main()
