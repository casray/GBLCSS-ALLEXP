from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp44_loo_tiesafe_fullspectrum_20260730"
MAIN_GRID = (
    REPO_ROOT / "runs" / "tie_safe_full_grid_20260727_v2_immutable"
    / "raw_combined.csv"
)
LOO_U04 = (
    REPO_ROOT / "runs" / "v31_tiesafe_affected_20260728"
    / "loo_uniform0p4.csv"
)

DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

BASE_METHOD = "frozengrace_gb_mlp_softdist_beta05"
BASE_SHUFFLED_METHOD = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MLP_CE_METHOD = "mlp_ce"
JOCOR_METHOD = "mlp_jocor"
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"
LOO_SHUFFLED_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"
ANCHOR_METHODS = (
    BASE_METHOD,
    BASE_SHUFFLED_METHOD,
    MLP_CE_METHOD,
    JOCOR_METHOD,
)
METHODS = (*ANCHOR_METHODS, LOO_METHOD, LOO_SHUFFLED_METHOD)


@dataclass(frozen=True)
class SummaryResult:
    combined: list[dict[str, object]]
    aggregate: list[dict[str, object]]
    seed_deltas: list[dict[str, object]]
    setting_deltas: list[dict[str, object]]
    dataset_deltas: list[dict[str, object]]
    gate: dict[str, object]


def read_csv(path: Path) -> list[dict[str, str]]:
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


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing numeric field {key!r}: {row}")
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    return int(_float(row, key))


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if value == "" else float(value)


def _setting(row: dict[str, str]) -> tuple[str, float]:
    return row.get("noise_type", "").strip().lower(), _float(row, "noise_rate")


def _is_expected_setting(setting: tuple[str, float]) -> bool:
    return any(
        setting[0] == noise_type and abs(setting[1] - noise_rate) <= 1e-12
        for noise_type, noise_rate in SETTINGS
    )


def _validate_protocol(row: dict[str, str], method: str) -> None:
    if row.get("status", "").strip().lower() != "ok":
        raise ValueError(f"non-success row: {row}")
    if row.get("noise_scope", "").strip().lower() != "train":
        raise ValueError(f"noise scope is not train-only: {row}")
    if _int(row, "modified_val_count") != 0:
        raise ValueError(f"validation labels were modified: {row}")
    if _int(row, "modified_test_count") != 0:
        raise ValueError(f"test labels were modified: {row}")
    if _int(row, "modified_count") != _int(row, "modified_train_count"):
        raise ValueError(f"modified labels are not restricted to train: {row}")
    if method in {LOO_METHOD, LOO_SHUFFLED_METHOD}:
        if row.get("tie_mode", "").strip() != "distribution_only":
            raise ValueError(f"LOO row is not tie-safe v2: {row}")
        if row.get("leave_one_out", "").strip().lower() not in {
            "1",
            "true",
        }:
            raise ValueError(f"LOO row is not leave-one-out: {row}")
    if method in {BASE_METHOD, BASE_SHUFFLED_METHOD}:
        if row.get("tie_mode", "").strip() != "distribution_only":
            raise ValueError(f"base anchor is not tie-safe v2: {row}")


def build_summary(rows: list[dict[str, str]]) -> SummaryResult:
    keyed: dict[tuple[str, str, float, int, str], dict[str, object]] = {}
    for raw in rows:
        dataset = raw.get("dataset", "").strip().lower()
        method = raw.get("method", "").strip()
        if dataset not in DATASETS or method not in METHODS:
            continue
        setting = _setting(raw)
        if not _is_expected_setting(setting):
            continue
        seed = _int(raw, "seed")
        if seed not in SEEDS:
            continue
        _validate_protocol(raw, method)
        key = (dataset, setting[0], setting[1], seed, method)
        if key in keyed:
            raise ValueError(f"duplicate grid cell: {key}")
        normalized: dict[str, object] = dict(raw)
        normalized.update(
            {
                "dataset": dataset,
                "method": method,
                "noise_type": setting[0],
                "noise_rate": setting[1],
                "seed": seed,
                "test": _float(raw, "test"),
            }
        )
        keyed[key] = normalized

    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            for seed in SEEDS:
                hashes: set[tuple[str, str]] = set()
                for method in METHODS:
                    key = (dataset, noise_type, noise_rate, seed, method)
                    if key not in keyed:
                        raise ValueError(f"missing grid cell: {key}")
                    row = keyed[key]
                    hashes.add(
                        (
                            str(row.get("split_hash", "")),
                            str(row.get("noise_hash", "")),
                        )
                    )
                if len(hashes) != 1:
                    raise ValueError(
                        "protocol hash mismatch: "
                        f"dataset={dataset} setting={noise_type}{noise_rate} "
                        f"seed={seed}"
                    )

    combined = [keyed[key] for key in sorted(keyed)]
    expected = len(DATASETS) * len(SETTINGS) * len(SEEDS) * len(METHODS)
    if len(combined) != expected:
        raise ValueError(f"expected {expected} rows, found {len(combined)}")

    grouped: dict[tuple[str, str, float, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in combined:
        grouped[
            (
                str(row["dataset"]),
                str(row["noise_type"]),
                float(row["noise_rate"]),
                str(row["method"]),
            )
        ].append(row)

    aggregate: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            for method in METHODS:
                current = grouped[(dataset, noise_type, noise_rate, method)]
                values = [float(row["test"]) for row in current]
                aggregate_row: dict[str, object] = {
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "method": method,
                    "n": len(values),
                    "test_mean": statistics.mean(values),
                    "test_std": statistics.stdev(values),
                }
                for field in (
                    "alpha_positive_train_ratio",
                    "soft_pseudo_ncr",
                    "soft_pseudo_cdr",
                ):
                    diagnostics = [
                        value
                        for row in current
                        if (value := _optional_float(row, field)) is not None
                    ]
                    aggregate_row[f"{field}_mean"] = (
                        statistics.mean(diagnostics) if diagnostics else ""
                    )
                aggregate.append(aggregate_row)

    seed_deltas: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            for seed in SEEDS:
                def value(method: str) -> float:
                    return float(
                        keyed[
                            (dataset, noise_type, noise_rate, seed, method)
                        ]["test"]
                    )

                loo = value(LOO_METHOD)
                seed_deltas.append(
                    {
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": noise_rate,
                        "seed": seed,
                        "loo": loo,
                        "base": value(BASE_METHOD),
                        "loo_shuffled": value(LOO_SHUFFLED_METHOD),
                        "base_shuffled": value(BASE_SHUFFLED_METHOD),
                        "mlp_ce": value(MLP_CE_METHOD),
                        "jocor": value(JOCOR_METHOD),
                        "loo_minus_base": loo - value(BASE_METHOD),
                        "loo_minus_loo_shuffled": loo
                        - value(LOO_SHUFFLED_METHOD),
                        "loo_minus_mlp_ce": loo - value(MLP_CE_METHOD),
                        "loo_minus_jocor": loo - value(JOCOR_METHOD),
                    }
                )

    setting_deltas: list[dict[str, object]] = []
    for dataset in DATASETS:
        for noise_type, noise_rate in SETTINGS:
            current = [
                row
                for row in seed_deltas
                if row["dataset"] == dataset
                and row["noise_type"] == noise_type
                and abs(float(row["noise_rate"]) - noise_rate) <= 1e-12
            ]
            row: dict[str, object] = {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "n": len(current),
            }
            for delta_name in (
                "loo_minus_base",
                "loo_minus_loo_shuffled",
                "loo_minus_mlp_ce",
                "loo_minus_jocor",
            ):
                values = [float(item[delta_name]) for item in current]
                row[f"{delta_name}_mean"] = statistics.mean(values)
                row[f"{delta_name}_std"] = statistics.stdev(values)
                row[f"{delta_name}_positive_seeds"] = sum(
                    value > 0 for value in values
                )
            setting_deltas.append(row)

    dataset_deltas: list[dict[str, object]] = []
    for dataset in DATASETS:
        current = [
            row for row in setting_deltas if row["dataset"] == dataset
        ]
        values = [float(row["loo_minus_base_mean"]) for row in current]
        dataset_deltas.append(
            {
                "dataset": dataset,
                "settings": len(current),
                "loo_minus_base_mean": statistics.mean(values),
                "loo_minus_base_min": min(values),
                "positive_setting_means": sum(value > 0 for value in values),
                "nonnegative_setting_means": sum(value >= 0 for value in values),
            }
        )

    base_values = [
        float(row["loo_minus_base_mean"]) for row in setting_deltas
    ]
    positive_structure = sum(
        float(row["loo_minus_loo_shuffled_mean"]) > 0
        for row in setting_deltas
    )
    stable_structure = sum(
        int(row["loo_minus_loo_shuffled_positive_seeds"]) >= 4
        for row in setting_deltas
    )
    datasets_nonnegative = sum(
        float(row["loo_minus_base_mean"]) >= 0 for row in dataset_deltas
    )
    reasons = []
    if statistics.mean(base_values) <= 0:
        reasons.append("global_mean_loo_minus_base_not_positive")
    if datasets_nonnegative < 4:
        reasons.append("fewer_than_4_datasets_nonnegative")
    if min(base_values) < -0.01:
        reasons.append("setting_regression_below_minus_1pp")
    if positive_structure < 24:
        reasons.append("fewer_than_24_positive_structure_settings")
    if stable_structure < 20:
        reasons.append("fewer_than_20_stable_structure_settings")
    gate: dict[str, object] = {
        "pass": int(not reasons),
        "global_loo_minus_base_mean": statistics.mean(base_values),
        "minimum_setting_loo_minus_base": min(base_values),
        "datasets_nonnegative_vs_base": datasets_nonnegative,
        "positive_structure_settings": positive_structure,
        "stable_structure_settings": stable_structure,
        "reasons": ";".join(reasons),
    }
    return SummaryResult(
        combined=combined,
        aggregate=aggregate,
        seed_deltas=seed_deltas,
        setting_deltas=setting_deltas,
        dataset_deltas=dataset_deltas,
        gate=gate,
    )


def write_report(path: Path, result: SummaryResult) -> None:
    aggregate = {
        (
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            str(row["method"]),
        ): row
        for row in result.aggregate
    }
    lines = [
        "# MLP-44 Tie-Safe LOO-GB-LCSS Full-Spectrum Report",
        "",
        "Formal lineage: `tie_mode=distribution_only`, "
        "`leave_one_out=1` for both real and shuffled LOO targets. "
        "Uniform 0.4 is reused only from the audited v2 artifact.",
        "",
        "| Dataset | Noise | Base GB | LOO-GB | LOO-base | "
        "LOO shuffled | Real-shuffled | Seeds vs base | Seeds vs shuffled |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.setting_deltas:
        dataset = str(row["dataset"])
        noise_type = str(row["noise_type"])
        noise_rate = float(row["noise_rate"])
        base = aggregate[(dataset, noise_type, noise_rate, BASE_METHOD)]
        loo = aggregate[(dataset, noise_type, noise_rate, LOO_METHOD)]
        shuffled = aggregate[
            (dataset, noise_type, noise_rate, LOO_SHUFFLED_METHOD)
        ]
        lines.append(
            "| {dataset} | {noise}{rate:.1f} | {base:.2f}+/-{base_std:.2f} | "
            "{loo:.2f}+/-{loo_std:.2f} | {delta:+.2f} | "
            "{shuf:.2f}+/-{shuf_std:.2f} | {structure:+.2f} | "
            "{base_pos}/5 | {structure_pos}/5 |".format(
                dataset=dataset,
                noise=noise_type,
                rate=noise_rate,
                base=100 * float(base["test_mean"]),
                base_std=100 * float(base["test_std"]),
                loo=100 * float(loo["test_mean"]),
                loo_std=100 * float(loo["test_std"]),
                delta=100 * float(row["loo_minus_base_mean"]),
                shuf=100 * float(shuffled["test_mean"]),
                shuf_std=100 * float(shuffled["test_std"]),
                structure=100
                * float(row["loo_minus_loo_shuffled_mean"]),
                base_pos=int(row["loo_minus_base_positive_seeds"]),
                structure_pos=int(
                    row["loo_minus_loo_shuffled_positive_seeds"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Dataset-level LOO minus base",
            "",
            "| Dataset | Mean across five settings | Minimum setting | "
            "Non-negative settings |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in result.dataset_deltas:
        lines.append(
            f"| {row['dataset']} | "
            f"{100*float(row['loo_minus_base_mean']):+.2f}pp | "
            f"{100*float(row['loo_minus_base_min']):+.2f}pp | "
            f"{row['nonnegative_setting_means']}/5 |"
        )
    gate = result.gate
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- PASS: `{gate['pass']}`.",
            "- Global LOO-base: "
            f"`{100*float(gate['global_loo_minus_base_mean']):+.2f}pp`.",
            "- Minimum setting LOO-base: "
            f"`{100*float(gate['minimum_setting_loo_minus_base']):+.2f}pp`.",
            "- Datasets with non-negative cross-setting LOO-base: "
            f"`{gate['datasets_nonnegative_vs_base']}/5`.",
            "- Positive real-shuffled setting means: "
            f"`{gate['positive_structure_settings']}/25`.",
            "- Real-shuffled settings with at least 4/5 positive seeds: "
            f"`{gate['stable_structure_settings']}/25`.",
            f"- Reasons: `{gate['reasons'] or 'pass'}`.",
            "",
            "## Decision",
            "",
            (
                "Tie-safe LOO-GB-LCSS satisfies all preregistered gates and "
                "may replace the self-inclusive target."
                if int(gate["pass"])
                else "Tie-safe LOO-GB-LCSS does not satisfy all "
                "preregistered gates; retain the self-inclusive target."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_grid", type=Path, default=MAIN_GRID)
    parser.add_argument(
        "--loo_inputs",
        nargs="+",
        type=Path,
        default=[
            LOO_U04,
            RUN_DIR / "loo_uniform0p2.csv",
            RUN_DIR / "loo_uniform0p6.csv",
            RUN_DIR / "loo_pair0p4.csv",
            RUN_DIR / "loo_instance0p4.csv",
        ],
    )
    parser.add_argument("--run_dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.main_grid)
    for path in args.loo_inputs:
        rows.extend(read_csv(path))
    result = build_summary(rows)
    write_csv(args.run_dir / "mlp44_combined_raw.csv", result.combined)
    write_csv(args.run_dir / "mlp44_aggregate.csv", result.aggregate)
    write_csv(args.run_dir / "mlp44_seed_deltas.csv", result.seed_deltas)
    write_csv(args.run_dir / "mlp44_setting_deltas.csv", result.setting_deltas)
    write_csv(args.run_dir / "mlp44_dataset_deltas.csv", result.dataset_deltas)
    write_csv(args.run_dir / "mlp44_gate_decision.csv", [result.gate])
    write_report(args.run_dir / "mlp44_report.md", result)
    print(
        f"wrote {len(result.combined)} rows; "
        f"gate_pass={result.gate['pass']}; "
        f"reasons={result.gate['reasons'] or 'pass'}"
    )


if __name__ == "__main__":
    main()
