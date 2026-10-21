from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path(__file__).resolve().parents[1]
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
FULL_SPECTRUM_BASELINES = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_logitclip",
)
FAITHFUL_UNIFORM04_BASELINES = (
    "mlp_anl_ce",
    "mlp_gmm_ct",
    "mlp_fine_official",
)
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "Small-loss",
    "mlp_coteaching": "Co-teaching",
    "mlp_jocor": "JoCoR",
    "mlp_elr": "ELR",
    "mlp_sce": "SCE",
    "mlp_logitclip": "LogitClip",
    "mlp_anl_ce": "ANL-CE (MLP-adapted)",
    "mlp_gmm_ct": "GMM+CT (MLP-adapted)",
    "mlp_fine_official": "FINE (MLP-adapted)",
    OURS: "LCSS",
    SHUFFLED: "Shuffled",
}


def formal_baselines_for_regime(noise_type: str, noise_rate: float) -> tuple[str, ...]:
    methods = list(FULL_SPECTRUM_BASELINES)
    if noise_type == "uniform" and math.isclose(float(noise_rate), 0.4):
        methods.extend(FAITHFUL_UNIFORM04_BASELINES)
    return tuple(methods)


def _normalized_row(row: dict[str, str], source: Path) -> dict | None:
    if row.get("status", "ok") not in {"", "ok"}:
        return None
    test_value = row.get("test", row.get("test_acc", ""))
    if test_value in {None, ""}:
        return None
    return {
        "dataset": row["dataset"].lower(),
        "noise_type": row["noise_type"].lower(),
        "noise_rate": float(row["noise_rate"]),
        "method": row["method"],
        "seed": int(row["seed"]),
        "test": float(test_value),
        "source": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
    }


def load_rows(paths: list[Path]) -> list[dict]:
    selected = set(FULL_SPECTRUM_BASELINES) | set(FAITHFUL_UNIFORM04_BASELINES) | {OURS, SHUFFLED}
    rows_by_key: dict[tuple, dict] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                row = _normalized_row(raw, path)
                if row is None or row["dataset"] not in DATASETS or row["method"] not in selected:
                    continue
                if (row["noise_type"], row["noise_rate"]) not in REGIMES:
                    continue
                key = (row["dataset"], row["noise_type"], row["noise_rate"], row["method"], row["seed"])
                if key in rows_by_key:
                    previous = rows_by_key[key]
                    if not math.isclose(previous["test"], row["test"], abs_tol=1e-12):
                        raise ValueError(f"conflicting duplicate row {key}: {previous['test']} vs {row['test']}")
                    continue
                rows_by_key[key] = row
    return list(rows_by_key.values())


def summarize_setting(
    rows: list[dict],
    ours_method: str,
    baseline_methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        values[row["method"]][int(row["seed"])] = float(row["test"])
    expected = set(expected_seeds)
    if set(values.get(ours_method, {})) != expected:
        raise ValueError(f"incomplete ours rows: {sorted(values.get(ours_method, {}))}")
    complete_baselines = [method for method in baseline_methods if set(values.get(method, {})) == expected]
    if not complete_baselines:
        raise ValueError("no complete formal baseline")
    strongest = max(complete_baselines, key=lambda method: mean(values[method].values()))
    ours_values = [values[ours_method][seed] for seed in expected_seeds]
    baseline_values = [values[strongest][seed] for seed in expected_seeds]
    deltas = [ours - baseline for ours, baseline in zip(ours_values, baseline_values)]
    first = rows[0]
    summary = {
        "dataset": first["dataset"],
        "noise_type": first["noise_type"],
        "noise_rate": float(first["noise_rate"]),
        "ours_mean": mean(ours_values),
        "ours_std": stdev(ours_values),
        "strongest_method": strongest,
        "strongest_mean": mean(baseline_values),
        "strongest_std": stdev(baseline_values),
        "mean_delta": mean(deltas),
        "positive_seeds": sum(delta > 0 for delta in deltas),
        "n_pairs": len(deltas),
        "numeric_win": int(mean(deltas) > 0),
        "stable_direction": int(sum(delta > 0 for delta in deltas) >= 4),
        "complete_formal_baselines": len(complete_baselines),
    }
    paired = [
        {
            "dataset": first["dataset"],
            "noise_type": first["noise_type"],
            "noise_rate": float(first["noise_rate"]),
            "strongest_method": strongest,
            "seed": seed,
            "ours_test": ours,
            "baseline_test": baseline,
            "delta": delta,
        }
        for seed, ours, baseline, delta in zip(expected_seeds, ours_values, baseline_values, deltas)
    ]
    return summary, paired


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(summaries: list[dict]) -> str:
    wins = sum(row["numeric_win"] for row in summaries)
    stable = sum(row["numeric_win"] and row["stable_direction"] for row in summaries)
    lines = [
        "# Phase 4 faithful-claim audit",
        "",
        "## Scope",
        "",
        "- Full 25-setting ranking excludes historical ANL/CT/FINE and DivideMix-lite proxies.",
        "- Seven identity-preserving baselines are ranked across all settings.",
        "- Faithful ANL-CE, GMM+CT, and FINE reruns are additionally ranked for uniform 0.4 only.",
        "- Every strongest-baseline comparison requires five paired seeds.",
        "",
        f"Numeric wins: **{wins}/25**; numeric wins with at least 4/5 positive paired seeds: **{stable}/25**.",
        "",
        "| Dataset | Noise | Strongest formal baseline | LCSS | Baseline | Delta | Direction |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        noise = f"{row['noise_type']}{row['noise_rate']:.1f}"
        lines.append(
            f"| {row['dataset']} | {noise} | {METHOD_LABELS.get(row['strongest_method'], row['strongest_method'])} "
            f"| {100*row['ours_mean']:.2f}+-{100*row['ours_std']:.2f} "
            f"| {100*row['strongest_mean']:.2f}+-{100*row['strongest_std']:.2f} "
            f"| {100*row['mean_delta']:+.2f}pp | {row['positive_seeds']}/5 |"
        )
    lines.extend([
        "",
        "## Claim decision",
        "",
        "The manuscript may report the numeric-win count above only with the explicit formal-baseline scope. "
        "Historical proxy rows remain useful as mechanism explorations but must not enter the published-method count.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="runs/phase4_faithful_claims_20260712")
    args = parser.parse_args()
    paths = [
        REPO_ROOT / "runs/mlp10_baseline_refresh_20260702/mlp10_combined_raw.csv",
        REPO_ROOT / "runs/mlp12_amazonphoto_20260702/mlp12_combined_raw.csv",
        REPO_ROOT / "runs/mlp21_dataset_replacement_20260710/mlp21_raw.csv",
        REPO_ROOT / "runs/mlp24_faithful_anl_ct_20260712/mlp24_combined_raw.csv",
        REPO_ROOT / "runs/mlp25_fine_fidelity_20260712/mlp25_combined_raw.csv",
    ]
    rows = load_rows(paths)
    summaries, paired_rows = [], []
    for dataset in DATASETS:
        for noise_type, noise_rate in REGIMES:
            setting_rows = [
                row for row in rows
                if row["dataset"] == dataset
                and row["noise_type"] == noise_type
                and math.isclose(row["noise_rate"], noise_rate)
            ]
            summary, paired = summarize_setting(
                setting_rows,
                ours_method=OURS,
                baseline_methods=formal_baselines_for_regime(noise_type, noise_rate),
                expected_seeds=EXPECTED_SEEDS,
            )
            summaries.append(summary)
            paired_rows.extend(paired)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    _write_csv(output_dir / "phase4_selected_raw.csv", sorted(rows, key=lambda r: (r["dataset"], r["noise_type"], r["noise_rate"], r["method"], r["seed"])))
    _write_csv(output_dir / "phase4_strongest_formal_baselines.csv", summaries)
    _write_csv(output_dir / "phase4_paired_deltas.csv", paired_rows)
    (output_dir / "phase4_report.md").write_text(_report(summaries), encoding="utf-8")
    print(_report(summaries))


if __name__ == "__main__":
    main()
