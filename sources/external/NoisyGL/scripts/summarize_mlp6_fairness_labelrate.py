from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp6_fairness_labelrate_20260701"
MLP5_RAW = REPO_ROOT / "runs" / "mlp_maintable_20260701" / "mlp_maintable_combined_raw.csv"

SOFTDIST = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
X_METHODS = ("mlp_ce", "mlp_small_loss", "mlp_jocor")
XH_METHODS = ("mlp_xh_ce", "mlp_small_loss_xh", "mlp_jocor_xh")
FAIRNESS_METHODS = (SOFTDIST, *X_METHODS, *XH_METHODS)
FAIRNESS_LOW_METHODS = (SOFTDIST, "mlp_small_loss_xh", "mlp_jocor_xh")
LABELRATE_METHODS = (SOFTDIST, "mlp_small_loss", "mlp_jocor", "mlp_fine", SHUFFLED)
LABELRATE_TRAINS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65)

METHOD_LABELS = {
    "mlp_ce": "MLP+CE (X)",
    "mlp_small_loss": "MLP+small-loss (X)",
    "mlp_jocor": "MLP+JoCoR (X)",
    "mlp_fine": "MLP-FINE (X)",
    "mlp_xh_ce": "MLP+CE [X||H]",
    "mlp_small_loss_xh": "MLP+small-loss [X||H]",
    "mlp_jocor_xh": "MLP+JoCoR [X||H]",
    SOFTDIST: "SoftDist-MLP beta=0.5",
    SHUFFLED: "SoftDist beta=0.5 shuffled",
}


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


def parse_current_table(path: Path) -> tuple[str, float] | None:
    name = path.name
    if name.startswith("fairness_uniform0p4_train0p20"):
        return "fairness", 0.20
    if name.startswith("fairness_uniform0p4_train0p05"):
        return "fairness_low", 0.05
    match = re.match(r"labelrate_train(\d+p\d+)_seeds_", name)
    if match:
        return "labelrate", float(match.group(1).replace("p", "."))
    return None


def optional_float(value) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def normalize_current_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    parsed = parse_current_table(source)
    if parsed is None:
        return None
    table, train_percent = parsed
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    method = (row.get("method") or row.get("base_method") or "").strip()
    if table == "fairness" and method not in XH_METHODS:
        return None
    if table == "fairness_low" and method not in {"mlp_small_loss_xh", "mlp_jocor_xh"}:
        return None
    if table == "labelrate" and method not in LABELRATE_METHODS:
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "table": table,
        "train_percent": float(train_percent),
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_train_noise_rate": optional_float(row.get("actual_train_noise_rate")),
        "source_group": "mlp6_new",
        "source": str(source),
    }


def normalize_mlp5_row(row: dict[str, str]) -> list[dict[str, object]]:
    method = row["method"].strip()
    dataset = row["dataset"].strip().lower()
    noise_type = row["noise_type"].strip().lower()
    noise_rate = float(row["noise_rate"])
    train_percent = float(row["train_percent"])
    seed = int(row["seed"])
    test_acc = float(row["test_acc"])
    source = row.get("source", str(MLP5_RAW))
    common = {
        "train_percent": train_percent,
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS.get(method, row.get("method_label", method)),
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "seed": seed,
        "test_acc": test_acc,
        "actual_train_noise_rate": optional_float(row.get("actual_train_noise_rate")),
        "source_group": "mlp5_reused",
        "source": source,
    }
    output = []
    if (
        row["table"] == "main"
        and math.isclose(train_percent, 0.20)
        and dataset in {"cora", "citeseer", "pubmed"}
        and noise_type == "uniform"
        and math.isclose(noise_rate, 0.4)
        and method in (SOFTDIST, *X_METHODS)
    ):
        output.append({"table": "fairness", **common})
    if (
        row["table"] == "labelrate"
        and math.isclose(train_percent, 0.05)
        and dataset in {"cora", "citeseer"}
        and noise_type == "uniform"
        and math.isclose(noise_rate, 0.4)
        and method == SOFTDIST
    ):
        output.append({"table": "fairness_low", **common})
    if (
        row["table"] == "labelrate"
        and train_percent in {0.05, 0.65}
        and dataset in {"cora", "citeseer"}
        and noise_type == "uniform"
        and math.isclose(noise_rate, 0.4)
        and method in LABELRATE_METHODS
    ):
        output.append({"table": "labelrate", **common})
    return output


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(f"conflicting duplicate for {key}: {previous['test_acc']} vs {row['test_acc']}")
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (
        row["table"],
        row["dataset"],
        row["noise_type"],
        float(row["noise_rate"]),
        float(row["train_percent"]),
        row["method"],
        int(row["seed"]),
    ))


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp6_"):
            continue
        for row in read_csv(path):
            normalized = normalize_current_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    for row in read_csv(MLP5_RAW):
        rows.extend(normalize_mlp5_row(row))
    return deduplicate(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            row["method_label"],
        )
        groups[key].append(float(row["test_acc"]))
    output = []
    for (table, dataset, noise_type, noise_rate, train_percent, method, label), values in groups.items():
        output.append({
            "table": table,
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "train_percent": train_percent,
            "method": method,
            "method_label": label,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return sorted(output, key=lambda row: (
        row["table"],
        row["dataset"],
        float(row["train_percent"]),
        row["method"],
    ))


def values_by_key(rows: list[dict[str, object]]) -> dict[tuple[object, ...], float]:
    return {
        (
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in rows
    }


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = values_by_key(rows)
    output = []
    comparisons = []
    for dataset in ("cora", "citeseer", "pubmed"):
        for baseline in (*X_METHODS, *XH_METHODS):
            comparisons.append(("fairness", dataset, 0.20, baseline))
    for dataset in ("cora", "citeseer"):
        for baseline in ("mlp_small_loss_xh", "mlp_jocor_xh"):
            comparisons.append(("fairness_low", dataset, 0.05, baseline))
    for dataset in ("cora", "citeseer"):
        for train_percent in LABELRATE_TRAINS:
            for baseline in ("mlp_small_loss", "mlp_jocor", "mlp_fine", SHUFFLED):
                comparisons.append(("labelrate", dataset, train_percent, baseline))

    for table, dataset, train_percent, baseline in comparisons:
        seeds = []
        deltas = []
        for seed in range(5):
            left = (table, dataset, "uniform", 0.4, float(train_percent), SOFTDIST, seed)
            right = (table, dataset, "uniform", 0.4, float(train_percent), baseline, seed)
            if left in values and right in values:
                seeds.append(seed)
                deltas.append(values[left] - values[right])
        if not deltas:
            continue
        output.append({
            "table": table,
            "dataset": dataset,
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "train_percent": float(train_percent),
            "comparison": f"beta05_vs_{baseline}",
            "baseline": baseline,
            "baseline_label": METHOD_LABELS[baseline],
            "n": len(deltas),
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "positive_count": sum(delta > 0 for delta in deltas),
            "seeds": " ".join(map(str, seeds)),
            "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
        })
    return output


def expected_new_keys() -> set[tuple[object, ...]]:
    keys = {
        ("fairness", dataset, "uniform", 0.4, 0.20, method, seed)
        for dataset in ("cora", "citeseer", "pubmed")
        for method in XH_METHODS
        for seed in range(5)
    }
    keys.update({
        ("fairness_low", dataset, "uniform", 0.4, 0.05, method, seed)
        for dataset in ("cora", "citeseer")
        for method in ("mlp_small_loss_xh", "mlp_jocor_xh")
        for seed in range(5)
    })
    keys.update({
        ("labelrate", dataset, "uniform", 0.4, train_percent, method, seed)
        for dataset in ("cora", "citeseer")
        for train_percent in (0.10, 0.20, 0.35, 0.50)
        for method in LABELRATE_METHODS
        for seed in range(5)
    })
    return keys


def expected_combined_keys() -> set[tuple[object, ...]]:
    keys = {
        ("fairness", dataset, "uniform", 0.4, 0.20, method, seed)
        for dataset in ("cora", "citeseer", "pubmed")
        for method in FAIRNESS_METHODS
        for seed in range(5)
    }
    keys.update({
        ("fairness_low", dataset, "uniform", 0.4, 0.05, method, seed)
        for dataset in ("cora", "citeseer")
        for method in FAIRNESS_LOW_METHODS
        for seed in range(5)
    })
    keys.update({
        ("labelrate", dataset, "uniform", 0.4, train_percent, method, seed)
        for dataset in ("cora", "citeseer")
        for train_percent in LABELRATE_TRAINS
        for method in LABELRATE_METHODS
        for seed in range(5)
    })
    return keys


def observed_keys(rows: list[dict[str, object]], source_group: str | None = None) -> set[tuple[object, ...]]:
    output = set()
    for row in rows:
        if source_group is not None and row["source_group"] != source_group:
            continue
        output.add((
            row["table"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            float(row["train_percent"]),
            row["method"],
            int(row["seed"]),
        ))
    return output


def completeness_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    new_expected = expected_new_keys()
    new_observed = observed_keys(rows, source_group="mlp6_new")
    combined_expected = expected_combined_keys()
    combined_observed = observed_keys(rows)
    output = [
        {
            "scope": "mlp6_new",
            "expected": len(new_expected),
            "observed": len(new_expected & new_observed),
            "missing": len(new_expected - new_observed),
        },
        {
            "scope": "combined_with_mlp5_reuse",
            "expected": len(combined_expected),
            "observed": len(combined_expected & combined_observed),
            "missing": len(combined_expected - combined_observed),
        },
    ]
    for key in sorted(new_expected - new_observed)[:50]:
        output.append({"scope": "missing_mlp6_new", "expected": "", "observed": "", "missing": key})
    for key in sorted(combined_expected - combined_observed)[:50]:
        output.append({"scope": "missing_combined", "expected": "", "observed": "", "missing": key})
    return output


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def find_aggregate(aggregates, table, dataset, train_percent, method):
    for row in aggregates:
        if (
            row["table"] == table
            and row["dataset"] == dataset
            and math.isclose(float(row["train_percent"]), float(train_percent))
            and row["method"] == method
        ):
            return row
    return None


def fairness_verdict(deltas: list[dict[str, object]]) -> tuple[str, list[str]]:
    notes = []
    dataset_states = []
    for dataset in ("cora", "citeseer"):
        candidate = [
            row for row in deltas
            if row["table"] == "fairness"
            and row["dataset"] == dataset
            and row["baseline"] in XH_METHODS
            and int(row["n"]) == 5
        ]
        if not candidate:
            dataset_states.append("missing")
            notes.append(f"{dataset}: missing H-input paired deltas")
            continue
        worst = min(candidate, key=lambda row: float(row["mean_delta"]))
        mean_delta = float(worst["mean_delta"])
        positive = int(worst["positive_count"])
        if mean_delta >= 0.0 and positive >= 3:
            dataset_states.append("locked")
        elif mean_delta >= -0.005:
            dataset_states.append("tie")
        else:
            dataset_states.append("reframe")
        notes.append(
            f"{dataset}: weakest beta05-vs-H delta is {mean_delta:+.4f} "
            f"against {worst['baseline']} ({positive}/5 positive)"
        )
    if all(state == "locked" for state in dataset_states):
        verdict = "fairness_locked"
    elif all(state in {"locked", "tie"} for state in dataset_states):
        verdict = "claim_narrow_to_tie"
    elif "missing" in dataset_states:
        verdict = "incomplete"
    else:
        verdict = "reframe_needed"
    return verdict, notes


def write_report(run_dir: Path, rows, aggregates, deltas, completeness) -> None:
    verdict, verdict_notes = fairness_verdict(deltas)
    lines = [
        "# MLP-6 Fairness + Label-Rate Report",
        "",
        "## Data Completeness",
    ]
    for row in completeness[:2]:
        lines.append(
            f"- {row['scope']}: observed {row['observed']}/{row['expected']}, missing={row['missing']}"
        )
    lines.extend(["", "## Part A: H-input Fairness Control", ""])
    lines.append(f"Verdict: **{verdict}**")
    for note in verdict_notes:
        lines.append(f"- {note}")
    lines.extend(["", "### Aggregates", ""])
    for dataset in ("cora", "citeseer", "pubmed"):
        lines.append(f"#### {dataset}")
        for method in FAIRNESS_METHODS:
            row = find_aggregate(aggregates, "fairness", dataset, 0.20, method)
            if row is None:
                continue
            lines.append(
                f"- {METHOD_LABELS[method]}: {fmt_mean_std(float(row['mean']), float(row['std']))} (n={row['n']})"
            )
    lines.extend(["", "### Paired Deltas: beta05 - H-input baselines", ""])
    for row in deltas:
        if row["table"] != "fairness" or row["baseline"] not in XH_METHODS:
            continue
        lines.append(
            f"- {row['dataset']} vs {row['baseline_label']}: "
            f"{float(row['mean_delta']):+.4f} +/- {float(row['std_delta']):.4f}, "
            f"{row['positive_count']}/{row['n']} positive"
        )
    lines.extend(["", "### Low-label Fairness (train=0.05)", ""])
    for row in deltas:
        if row["table"] != "fairness_low":
            continue
        lines.append(
            f"- {row['dataset']} vs {row['baseline_label']}: "
            f"{float(row['mean_delta']):+.4f} +/- {float(row['std_delta']):.4f}, "
            f"{row['positive_count']}/{row['n']} positive"
        )

    lines.extend(["", "## Part B: 5-seed Label-rate Curve", ""])
    for dataset in ("cora", "citeseer"):
        lines.append(f"### {dataset}")
        for train_percent in LABELRATE_TRAINS:
            cells = []
            for method in LABELRATE_METHODS:
                row = find_aggregate(aggregates, "labelrate", dataset, train_percent, method)
                if row is None:
                    continue
                cells.append(f"{METHOD_LABELS[method]}={fmt_mean_std(float(row['mean']), float(row['std']))}")
            lines.append(f"- train={train_percent:.2f}: " + "; ".join(cells))
        lines.append("")
    lines.append("### Label-rate Paired Deltas")
    for row in deltas:
        if row["table"] != "labelrate":
            continue
        lines.append(
            f"- {row['dataset']} train={float(row['train_percent']):.2f} vs {row['baseline_label']}: "
            f"{float(row['mean_delta']):+.4f}, {row['positive_count']}/{row['n']} positive"
        )

    lines.extend([
        "",
        "## Interpretation Guardrails",
        "- Part A is an H-access fairness control, not a replacement for the shuffled-ball control.",
        "- Part B reuses MLP-5 endpoints train=0.05/0.65 and newly fills train=0.10/0.20/0.35/0.50.",
        "- PubMed is reported as a boundary dataset; Cora/CiteSeer determine the main fairness claim.",
        "",
    ])
    (run_dir / "mlp6_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP-6 H-input fairness and label-rate experiments.")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(run_dir)
    aggregates = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    completeness = completeness_rows(rows)

    write_csv(run_dir / "mlp6_combined_raw.csv", rows)
    write_csv(run_dir / "mlp6_aggregate.csv", aggregates)
    write_csv(run_dir / "mlp6_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp6_completeness.csv", completeness)
    write_report(run_dir, rows, aggregates, deltas, completeness)

    missing_new = completeness[0]["missing"] if completeness else "?"
    missing_combined = completeness[1]["missing"] if len(completeness) > 1 else "?"
    print(f"WROTE {run_dir / 'mlp6_report.md'}")
    print(f"MLP6_NEW_MISSING={missing_new}")
    print(f"MLP6_COMBINED_MISSING={missing_combined}")


if __name__ == "__main__":
    main()
