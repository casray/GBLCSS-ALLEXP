import argparse
import csv
import statistics
from pathlib import Path


DATASETS = (
    "cora",
    "citeseer",
    "citeseerfull",
    "corafull",
    "coauthorcs",
    "amazonpho",
    "pubmed",
    "amazoncom",
    "coauthorph",
)
RATES = (0.2, 0.4, 0.6)

LCSS = "frozengrace_gb_mlp_softdist_beta05"
SMALL_ONLY = "mlp_small_loss"
SMALL_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SMALL_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"
METHODS = (LCSS, SMALL_ONLY, SMALL_COMBO, SMALL_SHUFFLED)


def build_summary(rows):
    selected = []
    keyed = {}
    for row in rows:
        if row.get("dataset") not in DATASETS or row.get("method") not in METHODS:
            continue
        rate = float(row["noise_rate"])
        if rate not in RATES:
            continue
        key = (row["dataset"], rate, row["method"], int(row["seed"]))
        if key in keyed:
            raise ValueError(f"duplicate grid cell: {key}")
        if int(row["seed"]) != 0:
            raise ValueError("MLP-30 accepts seed0 only")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {key}")
        if row.get("noise_type") != "uniform" or row.get("noise_scope") != "train":
            raise ValueError("MLP-30 requires uniform train-only noise")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError("validation/test labels were modified")
        if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
            raise ValueError("modified labels are not restricted to train")
        normalized = dict(row)
        normalized["noise_rate"] = rate
        normalized["seed"] = 0
        normalized["test"] = float(row["test"])
        keyed[key] = normalized
        selected.append(normalized)

    for dataset in DATASETS:
        for rate in RATES:
            hashes = set()
            for method in METHODS:
                key = (dataset, rate, method, 0)
                if key not in keyed:
                    raise ValueError(f"missing grid cell: {key}")
                hashes.add((keyed[key]["split_hash"], keyed[key]["noise_hash"]))
            if len(hashes) != 1:
                raise ValueError(f"protocol hash mismatch: dataset={dataset} rate={rate}")

    selected.sort(key=lambda row: (DATASETS.index(row["dataset"]), float(row["noise_rate"]), METHODS.index(row["method"])))
    deltas = []
    for dataset in DATASETS:
        for rate in RATES:
            value = lambda method: keyed[(dataset, rate, method, 0)]["test"]
            lcss = value(LCSS)
            small = value(SMALL_ONLY)
            combo = value(SMALL_COMBO)
            shuffled = value(SMALL_SHUFFLED)
            deltas.append(
                {
                    "dataset": dataset,
                    "noise_rate": rate,
                    "lcss": lcss,
                    "small_only": small,
                    "combo": combo,
                    "shuffled_combo": shuffled,
                    "combo_minus_lcss": combo - lcss,
                    "combo_minus_small_only": combo - small,
                    "synergy_delta": combo - max(lcss, small),
                    "structure_delta": combo - shuffled,
                    "best_single": LCSS if lcss >= small else SMALL_ONLY,
                }
            )

    gates = []
    for rate in RATES:
        current = [row for row in deltas if row["noise_rate"] == rate]
        synergy = [row["synergy_delta"] for row in current]
        structure = [row["structure_delta"] for row in current]
        reasons = []
        positive = sum(value > 0 for value in synergy)
        regressions = sum(value < -0.01 for value in synergy)
        structure_pass = sum(value >= 0.01 for value in structure)
        mean_synergy = statistics.mean(synergy)
        if positive < 6:
            reasons.append("fewer_than_6_of_9_synergy_positive")
        if mean_synergy < 0.01:
            reasons.append("mean_synergy_below_1pp")
        if regressions > 2:
            reasons.append("more_than_2_regressions_over_1pp")
        if structure_pass < 6:
            reasons.append("fewer_than_6_of_9_structure_gain_1pp")
        gates.append(
            {
                "noise_rate": rate,
                "pass": int(not reasons),
                "synergy_positive_datasets": positive,
                "mean_synergy_delta": mean_synergy,
                "regressions_over_1pp": regressions,
                "structure_gain_1pp_datasets": structure_pass,
                "reasons": ";".join(reasons),
            }
        )

    trends = []
    for dataset in DATASETS:
        current = {row["noise_rate"]: row for row in deltas if row["dataset"] == dataset}
        trends.append(
            {
                "dataset": dataset,
                "synergy_0p2": current[0.2]["synergy_delta"],
                "synergy_0p4": current[0.4]["synergy_delta"],
                "synergy_0p6": current[0.6]["synergy_delta"],
                "structure_0p2": current[0.2]["structure_delta"],
                "structure_0p4": current[0.4]["structure_delta"],
                "structure_0p6": current[0.6]["structure_delta"],
                "monotonic_synergy": int(
                    current[0.2]["synergy_delta"]
                    <= current[0.4]["synergy_delta"]
                    <= current[0.6]["synergy_delta"]
                ),
            }
        )
    return selected, deltas, gates, trends


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


def _report(path, deltas, gates, trends):
    lines = [
        "# MLP-30 LCSS + Soft-Small-Loss Noise-Rate Grid Seed0",
        "",
        "## Per-setting results",
        "",
        "| Dataset | Rate | LCSS | Small-loss | Combination | Shuffled | Synergy | Combo-LCSS | Combo-Small | Real-shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']:.1f} | {100*row['lcss']:.2f} | "
            f"{100*row['small_only']:.2f} | {100*row['combo']:.2f} | {100*row['shuffled_combo']:.2f} | "
            f"{100*row['synergy_delta']:+.2f}pp | {100*row['combo_minus_lcss']:+.2f}pp | "
            f"{100*row['combo_minus_small_only']:+.2f}pp | {100*row['structure_delta']:+.2f}pp |"
        )
    lines.extend(["", "## Rate gates", ""])
    for row in gates:
        lines.append(
            f"- uniform {row['noise_rate']:.1f}: pass={row['pass']}, synergy-positive="
            f"{row['synergy_positive_datasets']}/9, mean={100*row['mean_synergy_delta']:+.2f}pp, "
            f"regressions>1pp={row['regressions_over_1pp']}, structure>=1pp="
            f"{row['structure_gain_1pp_datasets']}/9, reasons={row['reasons'] or 'pass'}"
        )
    lines.extend(
        [
            "",
            "## Synergy trends",
            "",
            "| Dataset | u0.2 | u0.4 | u0.6 | Monotonic |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in trends:
        lines.append(
            f"| {row['dataset']} | {100*row['synergy_0p2']:+.2f}pp | "
            f"{100*row['synergy_0p4']:+.2f}pp | {100*row['synergy_0p6']:+.2f}pp | "
            f"{row['monotonic_synergy']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform0p2", required=True)
    parser.add_argument("--uniform0p4", required=True)
    parser.add_argument("--uniform0p6_small", required=True)
    parser.add_argument("--uniform0p6_large", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for path in (args.uniform0p2, args.uniform0p4, args.uniform0p6_small, args.uniform0p6_large):
        rows.extend(_read(path))
    combined, deltas, gates, trends = build_summary(rows)
    output = Path(args.output_dir)
    _write(output / "mlp30_combined_raw.csv", combined)
    _write(output / "mlp30_setting_deltas.csv", deltas)
    _write(output / "mlp30_rate_gates.csv", gates)
    _write(output / "mlp30_trends.csv", trends)
    _report(output / "mlp30_report.md", deltas, gates, trends)


if __name__ == "__main__":
    main()
