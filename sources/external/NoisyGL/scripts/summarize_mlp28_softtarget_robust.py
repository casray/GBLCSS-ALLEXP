import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


LCSS = "frozengrace_gb_mlp_softdist_beta05"
LOGIT_ONLY = "mlp_logitclip"
LOGIT_COMBO = "frozengrace_gb_mlp_softdist_beta05_logitclip"
LOGIT_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled"
SMALL_ONLY = "mlp_small_loss"
SMALL_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SMALL_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"

EXPECTED_DATASETS = ("cora", "citeseer", "citeseerfull")
EXPECTED_METHODS = (
    LCSS,
    LOGIT_ONLY,
    LOGIT_COMBO,
    LOGIT_SHUFFLED,
    SMALL_ONLY,
    SMALL_COMBO,
    SMALL_SHUFFLED,
)
STRATEGIES = {
    "logitclip": (LOGIT_ONLY, LOGIT_COMBO, LOGIT_SHUFFLED),
    "soft_small_loss": (SMALL_ONLY, SMALL_COMBO, SMALL_SHUFFLED),
}


def _float(row, key):
    return float(row[key])


def _validate_rows(rows):
    selected = []
    for row in rows:
        if row.get("method") not in EXPECTED_METHODS or row.get("dataset") not in EXPECTED_DATASETS:
            continue
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {row.get('dataset')} {row.get('method')} seed={row.get('seed')}")
        if row.get("noise_type") != "uniform" or abs(_float(row, "noise_rate") - 0.6) > 1e-12:
            raise ValueError("MLP-28 accepts only uniform 0.6 rows")
        if row.get("noise_scope") != "train":
            raise ValueError("MLP-28 accepts only train-only noise rows")
        if int(row.get("modified_val_count", 0)) != 0 or int(row.get("modified_test_count", 0)) != 0:
            raise ValueError("validation/test labels were modified")
        selected.append(row)

    keyed = {(row["dataset"], row["method"], int(row["seed"])) for row in selected}
    seeds = sorted({seed for _, _, seed in keyed})
    for dataset in EXPECTED_DATASETS:
        for method in EXPECTED_METHODS:
            for seed in seeds:
                if (dataset, method, seed) not in keyed:
                    raise ValueError(f"missing row: dataset={dataset} method={method} seed={seed}")
    if not seeds:
        raise ValueError("no MLP-28 rows found")
    return selected, seeds


def build_summary(rows):
    selected, seeds = _validate_rows(rows)
    grouped = defaultdict(list)
    by_seed = {}
    for row in selected:
        dataset = row["dataset"]
        method = row["method"]
        seed = int(row["seed"])
        test = _float(row, "test")
        grouped[(dataset, method)].append(test)
        by_seed[(dataset, method, seed)] = test

    aggregate = []
    for (dataset, method), values in sorted(grouped.items()):
        aggregate.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )

    deltas = []
    for strategy, (robust_only, combo, shuffled) in STRATEGIES.items():
        for dataset in EXPECTED_DATASETS:
            lcss_values = [by_seed[(dataset, LCSS, seed)] for seed in seeds]
            robust_values = [by_seed[(dataset, robust_only, seed)] for seed in seeds]
            combo_values = [by_seed[(dataset, combo, seed)] for seed in seeds]
            shuffled_values = [by_seed[(dataset, shuffled, seed)] for seed in seeds]
            synergy = [
                combo_value - max(lcss_value, robust_value)
                for combo_value, lcss_value, robust_value in zip(combo_values, lcss_values, robust_values)
            ]
            structure = [
                combo_value - shuffled_value
                for combo_value, shuffled_value in zip(combo_values, shuffled_values)
            ]
            deltas.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    "n": len(seeds),
                    "lcss_mean": statistics.mean(lcss_values),
                    "robust_only_mean": statistics.mean(robust_values),
                    "combo_mean": statistics.mean(combo_values),
                    "shuffled_combo_mean": statistics.mean(shuffled_values),
                    "combo_minus_lcss": statistics.mean(combo_values) - statistics.mean(lcss_values),
                    "combo_minus_robust_only": statistics.mean(combo_values) - statistics.mean(robust_values),
                    "synergy_delta": statistics.mean(synergy),
                    "synergy_positive_seeds": sum(value > 0 for value in synergy),
                    "structure_delta": statistics.mean(structure),
                    "structure_positive_seeds": sum(value > 0 for value in structure),
                }
            )

    gates = []
    for strategy in STRATEGIES:
        current = [row for row in deltas if row["strategy"] == strategy]
        positive_datasets = sum(row["synergy_delta"] > 0 for row in current)
        structure_datasets = sum(row["structure_delta"] >= 0.01 for row in current)
        average_synergy = statistics.mean(row["synergy_delta"] for row in current)
        minimum_synergy = min(row["synergy_delta"] for row in current)
        reasons = []
        if positive_datasets < 2:
            reasons.append("fewer_than_2_of_3_synergy_positive")
        if average_synergy < 0.005:
            reasons.append("mean_synergy_below_0p5pp")
        if minimum_synergy < -0.005:
            reasons.append("regression_over_0p5pp")
        if structure_datasets < 2:
            reasons.append("fewer_than_2_of_3_structure_gain_1pp")
        gates.append(
            {
                "strategy": strategy,
                "pass": int(not reasons),
                "synergy_positive_datasets": positive_datasets,
                "mean_synergy_delta": average_synergy,
                "min_synergy_delta": minimum_synergy,
                "structure_gain_1pp_datasets": structure_datasets,
                "reasons": ";".join(reasons),
            }
        )
    return aggregate, deltas, gates


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value):
    return f"{100.0 * float(value):+.2f}pp"


def _write_report(path, aggregate, deltas, gates):
    lines = [
        "# MLP-28 Soft-Target Robust Optimization Smoke Report",
        "",
        "Protocol: Cora/CiteSeer/CiteSeerFull, uniform 0.6, train-only noise, percent 0.2/0.1/0.7.",
        "",
        "## Paired attribution",
        "",
        "| Strategy | Dataset | LCSS | Robust-only | Combination | Shuffled combination | Synergy | Real-shuffled |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['strategy']} | {row['dataset']} | {100*row['lcss_mean']:.2f} | "
            f"{100*row['robust_only_mean']:.2f} | {100*row['combo_mean']:.2f} | "
            f"{100*row['shuffled_combo_mean']:.2f} | {_pct(row['synergy_delta'])} | "
            f"{_pct(row['structure_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            "| Strategy | Pass | Mean synergy | Minimum synergy | Positive datasets | Structure >=1pp datasets | Reasons |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in gates:
        lines.append(
            f"| {row['strategy']} | {row['pass']} | {_pct(row['mean_synergy_delta'])} | "
            f"{_pct(row['min_synergy_delta'])} | {row['synergy_positive_datasets']}/3 | "
            f"{row['structure_gain_1pp_datasets']}/3 | {row['reasons'] or 'pass'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "A passing strategy is only a candidate for paired five-seed confirmation. It does not modify the frozen LCSS paper mainline.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    aggregate, deltas, gates = build_summary(_read_csv(args.input))
    _write_csv(output_dir / "mlp28_aggregate.csv", aggregate)
    _write_csv(output_dir / "mlp28_paired_deltas.csv", deltas)
    _write_csv(output_dir / "mlp28_gate_decisions.csv", gates)
    _write_report(output_dir / "mlp28_report.md", aggregate, deltas, gates)


if __name__ == "__main__":
    main()
