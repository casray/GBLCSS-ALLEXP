from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp7_baselines_20260701"
MLP5_RAW = REPO_ROOT / "runs" / "mlp_maintable_20260701" / "mlp_maintable_combined_raw.csv"

SOFTDIST = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
CT = "mlp_ct"
MAIN_DATASETS = ("cora", "citeseer", "pubmed")
MAIN_REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
MLP5_METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_fine",
    "mlp_dividemix_lite",
    SOFTDIST,
    SHUFFLED,
)
MAIN_METHODS = (*MLP5_METHODS, CT)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "MLP+small-loss",
    "mlp_coteaching": "MLP+co-teaching",
    "mlp_jocor": "MLP+JoCoR",
    "mlp_elr": "MLP+ELR",
    "mlp_sce": "MLP+SCE",
    "mlp_fine": "MLP-FINE",
    "mlp_dividemix_lite": "DivideMix-lite",
    CT: "CT-MLP adapted",
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


def regime_from_current_name(path: Path) -> tuple[str, float] | None:
    match = re.match(r"main_ct_([a-z]+)(\d+p\d+)_seeds_", path.name)
    if not match:
        return None
    return match.group(1), float(match.group(2).replace("p", "."))


def optional_float(value) -> float | str:
    if value is None or not str(value).strip():
        return ""
    return float(value)


def normalize_current_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    parsed = regime_from_current_name(source)
    if parsed is None:
        return None
    noise_type, noise_rate = parsed
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method != CT:
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "table": "main",
        "train_percent": 0.2,
        "dataset": row["dataset"].strip().lower(),
        "method": CT,
        "method_label": METHOD_LABELS[CT],
        "noise_type": noise_type,
        "noise_rate": float(noise_rate),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_train_noise_rate": optional_float(row.get("actual_train_noise_rate")),
        "source_group": "mlp7_new",
        "source": str(source),
    }


def normalize_mlp5_row(row: dict[str, str]) -> dict[str, object] | None:
    if row.get("table") != "main":
        return None
    method = row["method"].strip()
    dataset = row["dataset"].strip().lower()
    noise_type = row["noise_type"].strip().lower()
    noise_rate = float(row["noise_rate"])
    if method not in MLP5_METHODS or dataset not in MAIN_DATASETS or (noise_type, noise_rate) not in MAIN_REGIMES:
        return None
    return {
        "table": "main",
        "train_percent": 0.2,
        "dataset": dataset,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "seed": int(row["seed"]),
        "test_acc": float(row["test_acc"]),
        "actual_train_noise_rate": optional_float(row.get("actual_train_noise_rate")),
        "source_group": "mlp5_reused",
        "source": row.get("source", str(MLP5_RAW)),
    }


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
                raise ValueError(f"conflicting duplicate for {key}")
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (
        row["dataset"],
        row["noise_type"],
        float(row["noise_rate"]),
        row["method"],
        int(row["seed"]),
    ))


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp7_") or path.name.startswith("grace_encoder"):
            continue
        for row in read_csv(path):
            normalized = normalize_current_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    for row in read_csv(MLP5_RAW):
        normalized = normalize_mlp5_row(row)
        if normalized is not None:
            rows.append(normalized)
    return deduplicate(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            row["method_label"],
        )
        groups[key].append(float(row["test_acc"]))
    output = []
    for (dataset, noise_type, noise_rate, method, label), values in groups.items():
        output.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "train_percent": 0.2,
            "method": method,
            "method_label": label,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return sorted(output, key=lambda row: (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"]))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in rows
    }
    output = []
    for dataset in MAIN_DATASETS:
        for noise_type, noise_rate in MAIN_REGIMES:
            seeds = []
            deltas = []
            for seed in range(5):
                left = (dataset, noise_type, noise_rate, SOFTDIST, seed)
                right = (dataset, noise_type, noise_rate, CT, seed)
                if left in values and right in values:
                    seeds.append(seed)
                    deltas.append(values[left] - values[right])
            if not deltas:
                continue
            output.append({
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "comparison": "beta05_vs_mlp_ct",
                "baseline": CT,
                "baseline_label": METHOD_LABELS[CT],
                "n": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_count": sum(delta > 0 for delta in deltas),
                "seeds": " ".join(map(str, seeds)),
                "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
            })
    return output


def expected_new_keys() -> set[tuple[object, ...]]:
    return {
        (dataset, noise_type, noise_rate, CT, seed)
        for dataset in MAIN_DATASETS
        for noise_type, noise_rate in MAIN_REGIMES
        for seed in range(5)
    }


def expected_combined_keys() -> set[tuple[object, ...]]:
    return {
        (dataset, noise_type, noise_rate, method, seed)
        for dataset in MAIN_DATASETS
        for noise_type, noise_rate in MAIN_REGIMES
        for method in MAIN_METHODS
        for seed in range(5)
    }


def observed_keys(rows: list[dict[str, object]], source_group: str | None = None) -> set[tuple[object, ...]]:
    keys = set()
    for row in rows:
        if source_group is not None and row["source_group"] != source_group:
            continue
        keys.add((row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])))
    return keys


def completeness_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    new_expected = expected_new_keys()
    new_observed = observed_keys(rows, "mlp7_new")
    combined_expected = expected_combined_keys()
    combined_observed = observed_keys(rows)
    return [
        {
            "scope": "mlp7_ct_new",
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


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def aggregate_lookup(aggregates):
    return {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"]): row
        for row in aggregates
    }


def checkpoint_summary(run_dir: Path) -> tuple[int, list[str]]:
    path = run_dir / "grace_encoder_checkpoint_scan.csv"
    rows = read_csv(path)
    candidates = [row for row in rows if row.get("status") == "candidate"]
    return len(candidates), [row["path"] for row in candidates[:10]]


def load_hinput_structure_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(run_dir / "hinput_structure_seed0.csv"):
        if row.get("status") != "ok":
            continue
        rows.append({
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "structure_mode": row["structure_mode"],
            "structure_p": float(row["structure_p"]),
            "h_recomputed": int(row.get("h_recomputed", 0) or 0),
            "test": float(row["test"]),
            "clean_reference_test": float(row["clean_reference_test"]),
            "delta_from_clean": float(row["delta_from_clean"]),
        })
    return rows


def hinput_structure_verdict(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "not_run"
    xh_rows = [
        row for row in rows
        if row["method"] in {"mlp_small_loss_xh", "mlp_jocor_xh"} and row["structure_mode"] != "clean"
    ]
    if not xh_rows:
        return "missing_hinput_rows"
    worst_drop = min(float(row["delta_from_clean"]) for row in xh_rows)
    if worst_drop <= -0.05:
        return "strong_structure_tradeoff_signal_seed0"
    if worst_drop <= -0.01:
        return "weak_structure_tradeoff_signal_seed0"
    return "hinput_baselines_stable_seed0"


def verdict_from_deltas(deltas: list[dict[str, object]]) -> str:
    core = [
        row for row in deltas
        if row["dataset"] in {"cora", "citeseer"} and int(row["n"]) == 5
    ]
    positive_settings = sum(float(row["mean_delta"]) > 0.0 for row in core)
    if len(core) < 10:
        return "incomplete"
    if positive_settings >= 8:
        return "beta05_stronger_than_ct_on_core"
    if positive_settings >= 5:
        return "mixed_core_result"
    return "ct_challenges_beta05_on_core"


def write_report(run_dir: Path, aggregates, deltas, completeness) -> None:
    lookup = aggregate_lookup(aggregates)
    verdict = verdict_from_deltas(deltas)
    candidate_count, candidate_paths = checkpoint_summary(run_dir)
    hinput_rows = load_hinput_structure_rows(run_dir)
    hinput_verdict = hinput_structure_verdict(hinput_rows)
    lines = [
        "# MLP-7 Baselines Report",
        "",
        "## Data Completeness",
    ]
    for row in completeness:
        lines.append(f"- {row['scope']}: observed {row['observed']}/{row['expected']}, missing={row['missing']}")
    lines.extend([
        "",
        "## 7a: CT-MLP Adapted Baseline",
        f"Verdict: **{verdict}**",
        "",
        "DSS was pre-registered as optional and was not implemented in this run; CT is the only new latest-style MLP-adapted baseline added here.",
        "",
    ])
    for dataset in MAIN_DATASETS:
        lines.append(f"### {dataset}")
        for noise_type, noise_rate in MAIN_REGIMES:
            beta = lookup.get((dataset, noise_type, noise_rate, SOFTDIST))
            ct = lookup.get((dataset, noise_type, noise_rate, CT))
            delta = next(
                (
                    row for row in deltas
                    if row["dataset"] == dataset
                    and row["noise_type"] == noise_type
                    and math.isclose(float(row["noise_rate"]), noise_rate)
                ),
                None,
            )
            if beta is None or ct is None or delta is None:
                lines.append(f"- {noise_type}{noise_rate}: missing")
                continue
            lines.append(
                f"- {noise_type}{noise_rate}: beta05={fmt(float(beta['mean']), float(beta['std']))}; "
                f"CT={fmt(float(ct['mean']), float(ct['std']))}; "
                f"delta={float(delta['mean_delta']):+.4f}, {delta['positive_count']}/{delta['n']} positive"
            )
        lines.append("")
    lines.extend([
        "## 7c-step0: GRACE Encoder Checkpoint Scan",
        f"- Candidate encoder/model checkpoint files found: {candidate_count}",
    ])
    if candidate_paths:
        lines.append("- First candidates:")
        for path in candidate_paths:
            lines.append(f"  - `{path}`")
    else:
        lines.append("- No reusable encoder checkpoint candidate was found by the conservative scan; 7c should not proceed unless a checkpoint is supplied or GRACE is retrained.")
    lines.extend([
        "",
        "## 7c: H-input Structure Robustness (seed0)",
        f"Verdict: **{hinput_verdict}**",
    ])
    if not hinput_rows:
        lines.append("- Not run in this summary.")
    else:
        for dataset in MAIN_DATASETS:
            dataset_rows = [row for row in hinput_rows if row["dataset"] == dataset]
            if not dataset_rows:
                continue
            lines.append(f"### {dataset}")
            for method in ("frozengrace_gb_mlp_softdist_beta05", "mlp_small_loss_xh", "mlp_jocor_xh"):
                method_rows = [row for row in dataset_rows if row["method"] == method]
                if not method_rows:
                    continue
                clean = next((row for row in method_rows if row["structure_mode"] == "clean"), None)
                clean_text = "NA" if clean is None else f"{float(clean['test']):.4f}"
                lines.append(f"- {method}: clean={clean_text}")
                for row in sorted(
                    [item for item in method_rows if item["structure_mode"] != "clean"],
                    key=lambda item: (item["structure_mode"], float(item["structure_p"])),
                ):
                    lines.append(
                        f"  - {row['structure_mode']} p={float(row['structure_p']):.2f}: "
                        f"test={float(row['test']):.4f}, delta={float(row['delta_from_clean']):+.4f}, "
                        f"H_recomputed={row['h_recomputed']}"
                    )
    lines.extend([
        "",
        "## Claim Boundary",
        "- CT is an MLP-adapted strategy, not an official reproduction of the cited CT method.",
        "- Graph-native 7b methods were not run in this phase; they remain related-work or optional positioning-table work.",
        "- MLP-6 reframe remains active: do not claim superiority under H-input robust baselines.",
        "",
    ])
    (run_dir / "mlp7_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP-7 CT baseline experiments.")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(run_dir)
    aggregates = aggregate_rows(rows)
    deltas = paired_deltas(rows)
    completeness = completeness_rows(rows)
    write_csv(run_dir / "mlp7_combined_raw.csv", rows)
    write_csv(run_dir / "mlp7_aggregate.csv", aggregates)
    write_csv(run_dir / "mlp7_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp7_completeness.csv", completeness)
    write_report(run_dir, aggregates, deltas, completeness)
    print(f"WROTE {run_dir / 'mlp7_report.md'}")
    print(f"MLP7_CT_NEW_MISSING={completeness[0]['missing']}")
    print(f"MLP7_COMBINED_MISSING={completeness[1]['missing']}")


if __name__ == "__main__":
    main()
