from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull")
RATES = (0.4, 0.6)
SEEDS = tuple(range(5))
REAL = "mlp_distill_formal_lcss_x"
SHUFFLED = "mlp_distill_formal_lcss_x_shuffled"
GCN = "mlp_distill_gcn"
BETA0 = "mlp_distill_softdist_x"

COMPARISONS = {
    "vs_shuffled": SHUFFLED,
    "vs_gcn": GCN,
    "vs_beta0": BETA0,
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="raise")


def validate_new_runs(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "method",
        "seed",
        "noise_type",
        "noise_rate",
        "noise_scope",
        "status",
        "test",
        "teacher_use_xh",
        "target_beta",
        "leave_one_out",
        "tie_mode",
        "random_control",
        "modified_val_count",
        "modified_test_count",
        "split_hash",
        "noise_hash",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"MLP-58 rows are missing columns: {sorted(missing)}")

    work = raw.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["noise_rate"] = _numeric(work, "noise_rate")
    work["seed"] = _numeric(work, "seed").astype(int)
    work["test"] = _numeric(work, "test")

    expected = {
        (dataset, rate, seed, method)
        for dataset in DATASETS
        for rate in RATES
        for seed in SEEDS
        for method in (REAL, SHUFFLED)
    }
    observed = set(
        zip(work.dataset, work.noise_rate, work.seed, work.method, strict=False)
    )
    if len(work) != len(expected) or observed != expected:
        missing_rows = sorted(expected - observed)[:5]
        extra_rows = sorted(observed - expected)[:5]
        raise ValueError(
            "incomplete or unexpected MLP-58 grid: "
            f"rows={len(work)}, missing={missing_rows}, extra={extra_rows}"
        )
    if work.duplicated(["dataset", "noise_rate", "seed", "method"]).any():
        raise ValueError("duplicate MLP-58 experiment rows")

    global_checks = {
        "status_ok": (work["status"] == "ok").all(),
        "uniform_noise": (work["noise_type"] == "uniform").all(),
        "train_only_noise": (work["noise_scope"] == "train").all(),
        "teacher_x_only": (_numeric(work, "teacher_use_xh") == 0).all(),
        "formal_beta": np.allclose(_numeric(work, "target_beta"), 0.5),
        "leave_one_out": (_numeric(work, "leave_one_out") == 1).all(),
        "tie_safe": (work["tie_mode"] == "distribution_only").all(),
        "validation_clean": (_numeric(work, "modified_val_count") == 0).all(),
        "test_clean": (_numeric(work, "modified_test_count") == 0).all(),
        "finite_test": np.isfinite(work["test"]).all(),
        "nonempty_hashes": (
            work["split_hash"].astype(str).str.len().gt(0)
            & work["noise_hash"].astype(str).str.len().gt(0)
        ).all(),
    }
    if {"modified_count", "modified_train_count"}.issubset(work.columns):
        global_checks["noise_count_consistent"] = (
            _numeric(work, "modified_count")
            == _numeric(work, "modified_train_count")
        ).all()
    failed = [name for name, value in global_checks.items() if not bool(value)]
    if failed:
        raise ValueError(f"MLP-58 protocol audit failed: {failed}")

    audit_rows = []
    for (rate, dataset, seed), group in work.groupby(
        ["noise_rate", "dataset", "seed"], sort=True
    ):
        method_set = set(group["method"])
        control_map = dict(zip(group["method"], _numeric(group, "random_control")))
        checks = {
            "method_pair": method_set == {REAL, SHUFFLED},
            "random_control_identity": (
                control_map.get(REAL) == 0 and control_map.get(SHUFFLED) == 1
            ),
            "split_hash_paired": group["split_hash"].nunique() == 1,
            "noise_hash_paired": group["noise_hash"].nunique() == 1,
        }
        audit_rows.append(
            {
                "noise_rate": float(rate),
                "dataset": dataset,
                "seed": int(seed),
                "method_count": len(group),
                "split_hash": group["split_hash"].iloc[0],
                "noise_hash": group["noise_hash"].iloc[0],
                "passed": bool(all(checks.values())),
                "failed_checks": ",".join(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["passed"].all():
        raise ValueError("real/shuffled protocol hashes are not paired")
    return audit


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["test"] = _numeric(work, "test")
    return (
        work.groupby(["noise_rate", "dataset", "method"], sort=True)["test"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def paired_comparisons(
    raw: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    new = raw[["noise_rate", "dataset", "seed", "method", "test"]].copy()
    old = controls[
        controls["method"].isin((GCN, BETA0))
    ][["noise_rate", "dataset", "seed", "method", "test"]].copy()
    combined = pd.concat([new, old], ignore_index=True)
    combined["noise_rate"] = _numeric(combined, "noise_rate")
    combined["seed"] = _numeric(combined, "seed").astype(int)
    combined["test"] = _numeric(combined, "test")
    selected = combined[
        combined["dataset"].isin(DATASETS)
        & combined["noise_rate"].isin(RATES)
        & combined["seed"].isin(SEEDS)
    ]
    if selected.duplicated(["noise_rate", "dataset", "seed", "method"]).any():
        raise ValueError("duplicate rows in MLP-58 paired inputs")

    wide = selected.pivot(
        index=["noise_rate", "dataset", "seed"],
        columns="method",
        values="test",
    ).reset_index()
    required_methods = {REAL, SHUFFLED, GCN, BETA0}
    if not required_methods.issubset(wide.columns) or len(wide) != 30:
        raise ValueError("incomplete controls for MLP-58 paired comparisons")
    if wide[list(required_methods)].isna().any().any():
        raise ValueError("missing paired test value in MLP-58 controls")

    paired_frames = []
    for comparison, right_method in COMPARISONS.items():
        frame = wide[["noise_rate", "dataset", "seed"]].copy()
        frame["comparison"] = comparison
        frame["left_method"] = REAL
        frame["right_method"] = right_method
        frame["left_test"] = wide[REAL]
        frame["right_test"] = wide[right_method]
        frame["delta"] = frame["left_test"] - frame["right_test"]
        frame["delta_pp"] = frame["delta"] * 100.0
        paired_frames.append(frame)
    paired = pd.concat(paired_frames, ignore_index=True)
    summary = (
        paired.groupby(["noise_rate", "dataset", "comparison"], sort=True)["delta"]
        .agg(
            mean_delta="mean",
            std_delta="std",
            positive_seeds=lambda values: int((values > 0).sum()),
            count="count",
        )
        .reset_index()
    )
    summary["mean_delta_pp"] = summary["mean_delta"] * 100.0
    summary["std_delta_pp"] = summary["std_delta"] * 100.0
    return paired, summary


def validate_control_pairing(
    raw: pd.DataFrame, controls: pd.DataFrame
) -> pd.DataFrame:
    required = {"dataset", "noise_rate", "seed", "method", "split_hash", "noise_hash"}
    missing = required.difference(controls.columns)
    if missing:
        raise ValueError(f"control rows are missing hash columns: {sorted(missing)}")
    real = raw[raw["method"] == REAL][
        ["dataset", "noise_rate", "seed", "split_hash", "noise_hash"]
    ].copy()
    rows = []
    for method in (GCN, BETA0):
        selected = controls[controls["method"] == method][
            ["dataset", "noise_rate", "seed", "split_hash", "noise_hash"]
        ].copy()
        merged = real.merge(
            selected,
            on=["dataset", "noise_rate", "seed"],
            how="outer",
            suffixes=("_new", "_control"),
            indicator=True,
        )
        for row in merged.to_dict(orient="records"):
            complete = row["_merge"] == "both"
            split_paired = complete and row["split_hash_new"] == row["split_hash_control"]
            noise_paired = complete and row["noise_hash_new"] == row["noise_hash_control"]
            rows.append(
                {
                    "dataset": row["dataset"],
                    "noise_rate": row["noise_rate"],
                    "seed": row["seed"],
                    "control_method": method,
                    "complete": bool(complete),
                    "split_hash_paired": bool(split_paired),
                    "noise_hash_paired": bool(noise_paired),
                    "passed": bool(complete and split_paired and noise_paired),
                }
            )
    audit = pd.DataFrame(rows)
    if len(audit) != 60 or not audit["passed"].all():
        raise ValueError("MLP-58 controls are not fully hash-paired")
    return audit


def evaluate_gates(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate in RATES:
        rate_rows = summary[np.isclose(summary["noise_rate"], rate)]
        for gate in ("vs_gcn", "vs_beta0", "vs_shuffled"):
            selected = rate_rows[rate_rows["comparison"] == gate].copy()
            if set(selected["dataset"]) != set(DATASETS):
                raise ValueError(f"incomplete {gate} summary at noise_rate={rate}")
            average_delta_pp = float(selected["mean_delta_pp"].mean())
            if gate in ("vs_gcn", "vs_shuffled"):
                passing = selected[
                    (selected["mean_delta_pp"] >= 1.0)
                    & (selected["positive_seeds"] >= 4)
                ]
                passed = len(passing) >= 2
                criterion = ">=2/3 datasets: mean delta >=1pp and >=4/5 positive seeds"
            else:
                passing = selected[selected["mean_delta_pp"] >= -1.0]
                passed = len(passing) == len(DATASETS) and average_delta_pp >= -1.0
                criterion = "no dataset mean regression >1pp and rate-average delta >=-1pp"
            rows.append(
                {
                    "noise_rate": rate,
                    "gate": gate,
                    "passing_datasets": int(len(passing)),
                    "average_delta_pp": average_delta_pp,
                    "worst_dataset_delta_pp": float(selected["mean_delta_pp"].min()),
                    "criterion": criterion,
                    "passed": bool(passed),
                }
            )
    return pd.DataFrame(rows)


def _format_rate(rate: float) -> str:
    return f"u{rate:.1f}"


def write_report(
    path: Path,
    aggregate: pd.DataFrame,
    paired_summary: pd.DataFrame,
    gates: pd.DataFrame,
    audit: pd.DataFrame,
    control_audit: pd.DataFrame,
) -> None:
    lines = [
        "# MLP-58 Formal LCSS Target Teacher Report",
        "",
        "The GCN teacher consumes X and A. Frozen GRACE H is used only to build granular balls.",
        "Teacher supervision is the formal GB-LCSS target: beta=0.5, leave-one-out voting, and distribution-only tie handling.",
        "The student is the same X-only MLP used by the existing distillation experiments.",
        "",
        "## Protocol audit",
        "",
        f"- New rows: {int(aggregate['count'].sum())}/60.",
        f"- Paired dataset/rate/seed groups: {len(audit)}/30; all passed={bool(audit['passed'].all())}.",
        f"- Hash-paired control comparisons: {len(control_audit)}/60; all passed={bool(control_audit['passed'].all())}.",
        "- Noise scope: training labels only; validation and test labels remain clean.",
        "",
        "## Aggregate accuracy",
        "",
        "| noise | dataset | method | mean+/-std (%) | n |",
        "|---|---|---|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {_format_rate(row.noise_rate)} | {row.dataset} | {row.method} | "
            f"{100*row.mean:.2f}+/-{100*row.std:.2f} | {row.count} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "| noise | dataset | comparison | mean delta (pp) | positive seeds |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in paired_summary.itertuples(index=False):
        lines.append(
            f"| {_format_rate(row.noise_rate)} | {row.dataset} | {row.comparison} | "
            f"{row.mean_delta_pp:+.2f} | {row.positive_seeds}/{row.count} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered gates",
            "",
            "| noise | gate | passing datasets | average delta (pp) | worst delta (pp) | passed |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in gates.itertuples(index=False):
        lines.append(
            f"| {_format_rate(row.noise_rate)} | {row.gate} | "
            f"{row.passing_datasets}/3 | {row.average_delta_pp:+.2f} | "
            f"{row.worst_dataset_delta_pp:+.2f} | {bool(row.passed)} |"
        )
    all_passed = bool(gates["passed"].all())
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"All pre-registered gates passed: {all_passed}.",
            "If all gates pass, the formal target becomes the preferred granular-ball teacher for the manuscript.",
            "A failed gate would only reject target transfer to the teacher route; it would not invalidate the base GB-LCSS method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    run_dir: Path,
    controls_path: Path,
    current_gcn_paths: list[Path] | None = None,
) -> None:
    formal_paths = sorted(run_dir.glob("formal_u0p*_seeds0_4.csv"))
    if len(formal_paths) != 2:
        raise ValueError(f"expected two formal result files, found {formal_paths}")
    raw = pd.concat([pd.read_csv(path) for path in formal_paths], ignore_index=True)
    audit = validate_new_runs(raw)
    aggregate = aggregate_results(raw)
    controls = pd.read_csv(controls_path)
    if current_gcn_paths:
        current_gcn = pd.concat(
            [pd.read_csv(path) for path in current_gcn_paths], ignore_index=True
        )
        current_gcn = current_gcn[current_gcn["method"] == GCN].copy()
        controls = pd.concat(
            [controls[controls["method"] != GCN], current_gcn],
            ignore_index=True,
            sort=False,
        )
    control_audit = validate_control_pairing(raw, controls)
    paired, paired_summary = paired_comparisons(raw, controls)
    gates = evaluate_gates(paired_summary)

    raw.to_csv(run_dir / "mlp58_raw.csv", index=False)
    aggregate.to_csv(run_dir / "mlp58_aggregate.csv", index=False)
    paired.to_csv(run_dir / "mlp58_paired_deltas.csv", index=False)
    paired_summary.to_csv(run_dir / "mlp58_paired_summary.csv", index=False)
    audit.to_csv(run_dir / "mlp58_protocol_audit.csv", index=False)
    control_audit.to_csv(run_dir / "mlp58_control_hash_audit.csv", index=False)
    controls[
        controls["method"].isin((GCN, BETA0))
        & controls["dataset"].isin(DATASETS)
        & controls["noise_rate"].isin(RATES)
        & controls["seed"].isin(SEEDS)
    ].to_csv(run_dir / "mlp58_controls.csv", index=False)
    gates.to_csv(run_dir / "mlp58_gate_decisions.csv", index=False)
    write_report(
        run_dir / "mlp58_report.md",
        aggregate,
        paired_summary,
        gates,
        audit,
        control_audit,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--current-gcn", nargs="*", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.run_dir, args.controls, args.current_gcn)


if __name__ == "__main__":
    main()
