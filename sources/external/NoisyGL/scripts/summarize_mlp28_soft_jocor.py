import argparse
import csv
import statistics
from pathlib import Path


LCSS = "frozengrace_gb_mlp_softdist_beta05"
JOCOR_ONLY = "mlp_jocor"
SOFT_SMALL = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SOFT_JOCOR = "frozengrace_gb_mlp_softdist_beta05_soft_jocor"
SOFT_JOCOR_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_soft_jocor_shuffled"
DATASETS = ("cora", "citeseer", "citeseerfull")
METHODS = (LCSS, JOCOR_ONLY, SOFT_SMALL, SOFT_JOCOR, SOFT_JOCOR_SHUFFLED)


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
        if row.get("status") not in {None, "", "ok"}:
            raise ValueError(f"non-ok row: {row}")
        if row.get("noise_type") and row["noise_type"] != "uniform":
            raise ValueError("soft-JoCoR smoke requires uniform noise")
        if row.get("noise_rate") and abs(float(row["noise_rate"]) - 0.6) > 1e-12:
            raise ValueError("soft-JoCoR smoke requires noise_rate=0.6")
        if row.get("noise_scope") and row["noise_scope"] != "train":
            raise ValueError("soft-JoCoR smoke requires train-only noise")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError("validation/test labels were modified")
        key = (row["dataset"], row["method"], int(row["seed"]))
        by_key[key] = row

    for dataset in DATASETS:
        for method in METHODS:
            if (dataset, method, 0) not in by_key:
                raise ValueError(f"missing seed0 row: {dataset} {method}")
        hashes = {
            (by_key[(dataset, method, 0)].get("split_hash"), by_key[(dataset, method, 0)].get("noise_hash"))
            for method in METHODS
            if by_key[(dataset, method, 0)].get("split_hash")
        }
        if len(hashes) > 1:
            raise ValueError(f"protocol hash mismatch for {dataset}")

    aggregate = []
    deltas = []
    for dataset in DATASETS:
        values = {method: float(by_key[(dataset, method, 0)]["test"]) for method in METHODS}
        for method, value in values.items():
            aggregate.append({"dataset": dataset, "method": method, "seed": 0, "test": value})
        deltas.append(
            {
                "dataset": dataset,
                "lcss": values[LCSS],
                "jocor_only": values[JOCOR_ONLY],
                "soft_small": values[SOFT_SMALL],
                "soft_jocor": values[SOFT_JOCOR],
                "soft_jocor_shuffled": values[SOFT_JOCOR_SHUFFLED],
                "soft_jocor_minus_soft_small": values[SOFT_JOCOR] - values[SOFT_SMALL],
                "soft_jocor_minus_jocor_only": values[SOFT_JOCOR] - values[JOCOR_ONLY],
                "structure_delta": values[SOFT_JOCOR] - values[SOFT_JOCOR_SHUFFLED],
            }
        )

    gains = [row["soft_jocor_minus_soft_small"] for row in deltas]
    structures = [row["structure_delta"] for row in deltas]
    reasons = []
    if sum(value > 0 for value in gains) < 2:
        reasons.append("fewer_than_2_of_3_above_soft_small")
    if statistics.mean(gains) < 0.01:
        reasons.append("mean_gain_below_1pp")
    if min(gains) < -0.005:
        reasons.append("regression_over_0p5pp")
    if sum(value >= 0.01 for value in structures) < 2:
        reasons.append("fewer_than_2_of_3_structure_gain_1pp")
    gate = {
        "pass": int(not reasons),
        "mean_gain_vs_soft_small": statistics.mean(gains),
        "min_gain_vs_soft_small": min(gains),
        "positive_datasets": sum(value > 0 for value in gains),
        "structure_gain_1pp_datasets": sum(value >= 0.01 for value in structures),
        "reasons": ";".join(reasons),
    }
    return aggregate, deltas, gate


def _report(path, deltas, gate):
    lines = [
        "# MLP-28 Soft-JoCoR Kill Smoke",
        "",
        "| Dataset | LCSS | JoCoR-only | Soft-small-loss | Soft-JoCoR | Shuffled Soft-JoCoR | vs soft-small | Real-shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {100*row['lcss']:.2f} | {100*row['jocor_only']:.2f} | "
            f"{100*row['soft_small']:.2f} | {100*row['soft_jocor']:.2f} | "
            f"{100*row['soft_jocor_shuffled']:.2f} | {100*row['soft_jocor_minus_soft_small']:+.2f}pp | "
            f"{100*row['structure_delta']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Pass: {gate['pass']}",
            f"- Mean gain vs soft-small-loss: {100*gate['mean_gain_vs_soft_small']:+.2f}pp",
            f"- Minimum gain: {100*gate['min_gain_vs_soft_small']:+.2f}pp",
            f"- Positive datasets: {gate['positive_datasets']}/3",
            f"- Real-shuffled >=1pp datasets: {gate['structure_gain_1pp_datasets']}/3",
            f"- Reasons: {gate['reasons'] or 'pass'}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input", required=True)
    parser.add_argument("--jocor_input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = _read(args.base_input) + _read(args.jocor_input)
    aggregate, deltas, gate = build_summary(rows)
    output = Path(args.output_dir)
    _write(output / "mlp28_soft_jocor_aggregate.csv", aggregate)
    _write(output / "mlp28_soft_jocor_paired_deltas.csv", deltas)
    _write(output / "mlp28_soft_jocor_gate.csv", [gate])
    _report(output / "mlp28_soft_jocor_report.md", deltas, gate)


if __name__ == "__main__":
    main()
