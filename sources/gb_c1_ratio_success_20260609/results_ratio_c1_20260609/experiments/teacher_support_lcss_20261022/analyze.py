from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def read_rows():
    with (ROOT / "results.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["status"] == "complete"]
    for row in rows:
        for key in ("seed", "test_accuracy", "test_macro_f1", "best_valid_accuracy", "best_epoch"):
            row[key] = float(row[key])
    expected = len(CONFIG["datasets"]) * len(CONFIG["seeds"]) * len(CONFIG["conditions"]) * len(CONFIG["arms"])
    keys = {(r["dataset"], int(r["seed"]), r["condition"], r["arm"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f"Expected {expected} unique complete fits; found {len(rows)}")
    return rows


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    rows = read_rows()
    lookup = {(r["dataset"], int(r["seed"]), r["condition"], r["arm"]): r for r in rows}
    arm_rows = []
    for dataset in CONFIG["datasets"]:
        for condition in CONFIG["conditions"]:
            for arm in CONFIG["arms"]:
                values = [lookup[(dataset, seed, condition, arm)] for seed in CONFIG["seeds"]]
                acc_mean, acc_sd = mean_sd([r["test_accuracy"] * 100 for r in values])
                f1_mean, f1_sd = mean_sd([r["test_macro_f1"] * 100 for r in values])
                arm_rows.append({"dataset": dataset, "condition": condition, "arm": arm, "n": len(values),
                                 "test_accuracy_mean_pct": acc_mean, "test_accuracy_sd_pp": acc_sd,
                                 "test_macro_f1_mean_pct": f1_mean, "test_macro_f1_sd_pp": f1_sd,
                                 "mean_best_epoch": statistics.mean(r["best_epoch"] for r in values)})
    write_csv(ROOT / "arm_summary.csv", arm_rows)

    contrasts = [("SUPPORT_LCSS", "OLD_LCSS"), ("OLD_LCSS", "CE"), ("SUPPORT_LCSS", "CE")]
    paired = []
    cell_diffs = {}
    for first, second in contrasts:
        for dataset in CONFIG["datasets"]:
            for condition in CONFIG["conditions"]:
                diffs = [(lookup[(dataset, seed, condition, first)]["test_accuracy"] -
                          lookup[(dataset, seed, condition, second)]["test_accuracy"]) * 100
                         for seed in CONFIG["seeds"]]
                mean, sd = mean_sd(diffs)
                paired.append({"contrast": f"{first}-{second}", "dataset": dataset,
                               "condition": condition, "n_seeds": len(diffs),
                               "mean_gain_pp": mean, "sd_paired_gain_pp": sd,
                               "positive_seeds": sum(d > 0 for d in diffs)})
                cell_diffs[(first, second, dataset, condition)] = diffs
        all_diffs = [d for (f, s, _, _), values in cell_diffs.items() if f == first and s == second for d in values]
        mean, sd = mean_sd(all_diffs)
        paired.append({"contrast": f"{first}-{second}", "dataset": "ALL_NOISY_CELLS",
                       "condition": "all", "n_seeds": len(all_diffs),
                       "mean_gain_pp": mean, "sd_paired_gain_pp": sd,
                       "positive_seeds": sum(d > 0 for d in all_diffs)})
        for seed in CONFIG["seeds"]:
            by_seed = [cell_diffs[(first, second, dataset, condition)][seed]
                       for dataset in CONFIG["datasets"] for condition in CONFIG["conditions"]]
            paired.append({"contrast": f"{first}-{second}", "dataset": f"SEED_{seed}_MACRO",
                           "condition": "all", "n_seeds": len(by_seed),
                           "mean_gain_pp": statistics.mean(by_seed), "sd_paired_gain_pp": "",
                           "positive_seeds": int(statistics.mean(by_seed) > 0)})
        for dataset in CONFIG["datasets"]:
            by_dataset = [d for condition in CONFIG["conditions"] for d in cell_diffs[(first, second, dataset, condition)]]
            paired.append({"contrast": f"{first}-{second}", "dataset": f"{dataset}_MACRO",
                           "condition": "all", "n_seeds": len(by_dataset),
                           "mean_gain_pp": statistics.mean(by_dataset), "sd_paired_gain_pp": "",
                           "positive_seeds": int(statistics.mean(by_dataset) > 0)})
    write_csv(ROOT / "paired_summary.csv", paired)

    primary = cell_diffs[("SUPPORT_LCSS", "OLD_LCSS", CONFIG["datasets"][0], next(iter(CONFIG["conditions"]))) ]
    all_primary = [d for (f, s, _, _), vals in cell_diffs.items() if f == "SUPPORT_LCSS" and s == "OLD_LCSS" for d in vals]
    seed_means = [statistics.mean(cell_diffs[("SUPPORT_LCSS", "OLD_LCSS", dataset, condition)][seed]
                                  for dataset in CONFIG["datasets"] for condition in CONFIG["conditions"])
                  for seed in CONFIG["seeds"]]
    dataset_means = [statistics.mean(d for condition in CONFIG["conditions"]
                                     for d in cell_diffs[("SUPPORT_LCSS", "OLD_LCSS", dataset, condition)])
                     for dataset in CONFIG["datasets"]]
    overall_gain = statistics.mean(all_primary)
    gate = CONFIG["success_gate"]
    passed = (overall_gain >= gate["min_mean_support_gain_pp"] and
              sum(v > 0 for v in seed_means) >= gate["positive_seed_means_required"] and
              sum(v > 0 for v in dataset_means) >= gate["positive_dataset_means_required"])
    decision = {
        "primary_contrast": "SUPPORT_LCSS minus OLD_LCSS test accuracy",
        "mean_paired_gain_pp": overall_gain,
        "seed_level_means_pp": dict(zip(map(str, CONFIG["seeds"]), seed_means)),
        "dataset_level_means_pp": dict(zip(CONFIG["datasets"], dataset_means)),
        "gate": gate, "passed": passed,
        "interpretation": ("Predeclared practical gate passed; proceed to a paired student-distillation test."
                           if passed else "Predeclared practical gate failed; do not claim a material SUPPORT teacher-side increment under this protocol."),
        "scope": "Exploratory evidence on five datasets, three protocol seeds, and three train-only noise conditions; not a formal equivalence or significance test."
    }
    (ROOT / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Teacher SUPPORT-LCSS result", "", f"Primary paired gain: **{overall_gain:+.3f} pp** (SUPPORT-LCSS minus historical LCSS).",
             f"Decision gate: **{'PASS' if passed else 'FAIL'}**.", "", "Seed-level mean gains:"]
    lines.extend(f"- Seed {seed}: {value:+.3f} pp" for seed, value in zip(CONFIG["seeds"], seed_means))
    lines.extend(["", "Dataset-level mean gains:"])
    lines.extend(f"- {dataset}: {value:+.3f} pp" for dataset, value in zip(CONFIG["datasets"], dataset_means))
    lines.extend(["", decision["interpretation"], "", decision["scope"], "",
                  "See `arm_summary.csv` for per-dataset/noise metrics and `paired_summary.csv` for paired contrasts. Three seeds are exploratory; report means and sample SDs without claiming statistical significance."])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((ROOT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    analyze()
