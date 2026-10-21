from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull")
RATES = (0.4, 0.6)
SEEDS = tuple(range(5))
X_TEACHER = "mlp_distill_softdist_x"
CONTROLS = (
    "mlp_distill_gcn",
    "mlp_distill_gcn_xh",
    "mlp_distill_softdist_xh",
)
COMPARISON_LABELS = {
    "mlp_distill_gcn": "x_ball_vs_gcn_x_ce",
    "mlp_distill_gcn_xh": "x_ball_vs_gcn_xh_ce",
    "mlp_distill_softdist_xh": "x_ball_vs_xh_ball",
}


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["delta"] = pd.to_numeric(work["delta"])
    group_columns = ["noise_rate", "dataset"]
    if "comparison" in work.columns:
        group_columns.append("comparison")
    output = (
        work.groupby(group_columns, sort=True)["delta"]
        .agg(
            mean_delta="mean",
            std_delta="std",
            positive_seeds=lambda values: int((values > 0).sum()),
            n="count",
        )
        .reset_index()
    )
    output["mean_delta_pp"] = output["mean_delta"] * 100.0
    output["std_delta_pp"] = output["std_delta"] * 100.0
    return output


def pre_registered_gate(summary: pd.DataFrame, noise_rate: float) -> dict[str, object]:
    selected = summary[np.isclose(summary["noise_rate"], noise_rate)].copy()
    if "comparison" in selected.columns:
        selected = selected[selected["comparison"] == "x_ball_vs_gcn_x_ce"]
    if set(selected["dataset"]) != set(DATASETS):
        raise ValueError(f"incomplete gate summary for noise_rate={noise_rate}")
    passing = selected[
        (selected["mean_delta_pp"] >= 1.0) & (selected["positive_seeds"] >= 4)
    ]
    average_gain_pp = float(selected["mean_delta_pp"].mean())
    passed = len(passing) >= 2 and average_gain_pp >= 1.0
    return {
        "noise_rate": float(noise_rate),
        "passing_datasets": int(len(passing)),
        "average_gain_pp": average_gain_pp,
        "passed": bool(passed),
    }


def _normalize_new(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
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
        "teacher_input_dim",
        "student_input_dim",
        "target_beta",
        "leave_one_out",
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"new X-teacher file is missing columns: {sorted(missing)}")
    selected = frame[
        (frame["method"] == X_TEACHER)
        & frame["dataset"].isin(DATASETS)
        & frame["seed"].isin(SEEDS)
        & frame["noise_rate"].isin(RATES)
    ].copy()
    observed_rates = tuple(sorted(pd.to_numeric(selected["noise_rate"]).unique()))
    if len(observed_rates) != 1 or observed_rates[0] not in RATES:
        raise ValueError(f"expected one registered noise rate per file, found {observed_rates}")
    expected = {
        (dataset, rate, seed)
        for dataset in DATASETS
        for rate in observed_rates
        for seed in SEEDS
    }
    actual = set(zip(selected.dataset, selected.noise_rate, selected.seed))
    if len(selected) != 15 or actual != expected:
        raise ValueError(f"incomplete X-teacher grid: rows={len(selected)}")
    checks = {
        "status": (selected.status == "ok").all(),
        "uniform": (selected.noise_type == "uniform").all(),
        "train_only": (selected.noise_scope == "train").all(),
        "teacher_x_only": (pd.to_numeric(selected.teacher_use_xh) == 0).all(),
        "input_dims_match": (
            pd.to_numeric(selected.teacher_input_dim)
            == pd.to_numeric(selected.student_input_dim)
        ).all(),
        "beta_zero": np.allclose(pd.to_numeric(selected.target_beta), 0.0),
        "non_loo_teacher_target": (pd.to_numeric(selected.leave_one_out) == 0).all(),
        "val_clean": (pd.to_numeric(selected.modified_val_count) == 0).all(),
        "test_clean": (pd.to_numeric(selected.modified_test_count) == 0).all(),
        "noise_count_consistent": (
            pd.to_numeric(selected.modified_count)
            == pd.to_numeric(selected.modified_train_count)
        ).all(),
    }
    failed = [name for name, value in checks.items() if not bool(value)]
    if failed:
        raise ValueError(f"X-teacher protocol audit failed: {failed}")
    selected["source"] = "mlp57_xteacher"
    return selected


def _normalize_controls(mlp9_path: Path, corafull_path: Path) -> pd.DataFrame:
    legacy = pd.read_csv(mlp9_path)
    legacy = legacy[
        legacy.dataset.isin(("cora", "citeseer"))
        & legacy.method.isin(CONTROLS)
        & legacy.noise_rate.isin(RATES)
        & legacy.seed.isin(SEEDS)
    ][["dataset", "method", "seed", "noise_type", "noise_rate", "student_test"]].copy()
    legacy = legacy.rename(columns={"student_test": "test"})
    legacy["noise_scope"] = "train"
    legacy["status"] = "ok"
    legacy["source"] = "mlp9_audited_legacy"

    corafull = pd.read_csv(corafull_path)
    corafull = corafull[
        (corafull.dataset == "corafull")
        & corafull.method.isin(CONTROLS)
        & corafull.noise_rate.isin(RATES)
        & corafull.seed.isin(SEEDS)
    ][
        [
            "dataset",
            "method",
            "seed",
            "noise_type",
            "noise_rate",
            "noise_scope",
            "status",
            "test",
            "split_hash",
            "noise_hash",
        ]
    ].copy()
    corafull["source"] = "mlp43_audited_control"
    controls = pd.concat([legacy, corafull], ignore_index=True, sort=False)
    expected = {
        (dataset, rate, seed, method)
        for dataset in DATASETS
        for rate in RATES
        for seed in SEEDS
        for method in CONTROLS
    }
    actual = set(zip(controls.dataset, controls.noise_rate, controls.seed, controls.method))
    if len(controls) != 90 or actual != expected:
        raise ValueError(f"incomplete teacher controls: rows={len(controls)}")
    return controls


def summarize(
    run_dir: Path,
    mlp9_path: Path,
    corafull_path: Path,
) -> None:
    new = pd.concat(
        [
            _normalize_new(run_dir / "xteacher_uniform0p4_seeds0_4.csv"),
            _normalize_new(run_dir / "xteacher_uniform0p6_seeds0_4.csv"),
        ],
        ignore_index=True,
    )
    new = new.drop_duplicates(["dataset", "noise_rate", "seed", "method"])
    if len(new) != 30:
        raise ValueError(f"expected 30 unique X-teacher rows, found {len(new)}")
    controls = _normalize_controls(mlp9_path, corafull_path)
    raw = pd.concat([new, controls], ignore_index=True, sort=False)
    raw.to_csv(run_dir / "xteacher_raw.csv", index=False)

    aggregate = (
        raw.groupby(["noise_rate", "dataset", "method"], sort=True)["test"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.to_csv(run_dir / "xteacher_aggregate.csv", index=False)

    wide = raw.pivot(
        index=["noise_rate", "dataset", "seed"], columns="method", values="test"
    ).reset_index()
    paired_frames = []
    for control, label in COMPARISON_LABELS.items():
        rows = wide[["noise_rate", "dataset", "seed"]].copy()
        rows["comparison"] = label
        rows["delta"] = wide[X_TEACHER] - wide[control]
        paired_frames.append(rows)
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(run_dir / "xteacher_paired_deltas.csv", index=False)
    summary = paired_summary(paired)
    summary.to_csv(run_dir / "xteacher_paired_summary.csv", index=False)

    decisions = pd.DataFrame(
        [pre_registered_gate(summary, noise_rate=rate) for rate in RATES]
    )
    decisions.to_csv(run_dir / "xteacher_gate_decisions.csv", index=False)
    completeness = pd.DataFrame(
        [
            {
                "artifact": "xteacher_new",
                "expected_rows": 30,
                "observed_rows": len(new),
                "complete": len(new) == 30,
            },
            {
                "artifact": "teacher_controls",
                "expected_rows": 90,
                "observed_rows": len(controls),
                "complete": len(controls) == 90,
            },
        ]
    )
    completeness.to_csv(run_dir / "xteacher_completeness.csv", index=False)

    lines = [
        "# MLP-57 X-only ball-teacher report",
        "",
        "The teacher and student both consume X. Frozen GRACE H is used only to build ball targets.",
        "Teacher targets use beta=0 and the non-LOO distribution target, matching the existing teacher ablation.",
        "",
        "## Aggregate",
        "",
        "| noise | dataset | method | mean+/-std (%) |",
        "|---:|---|---|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.noise_rate:.1f} | {row.dataset} | {row.method} | "
            f"{100*row.mean:.2f}+/-{100*row.std:.2f} |"
        )
    lines.extend(["", "## Paired deltas", ""])
    for row in summary.itertuples(index=False):
        lines.append(
            f"- u{row.noise_rate:.1f} {row.dataset} {row.comparison}: "
            f"{row.mean_delta_pp:+.2f}pp, {row.positive_seeds}/{row.n} positive seeds."
        )
    lines.extend(["", "## Pre-registered adoption gate", ""])
    for row in decisions.itertuples(index=False):
        lines.append(
            f"- u{row.noise_rate:.1f}: passed={bool(row.passed)}, "
            f"passing_datasets={row.passing_datasets}/3, "
            f"average gain over GCN(X)+CE={row.average_gain_pp:+.2f}pp."
        )
    lines.extend(
        [
            "",
            "Controls for Cora/CiteSeer come from the audited MLP-9 table lineage, which predates hash logging.",
            "CoraFull controls come from the MLP-43 audited table lineage and retain split/noise hashes.",
        ]
    )
    (run_dir / "xteacher_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mlp9", type=Path, required=True)
    parser.add_argument("--corafull-controls", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.run_dir, args.mlp9, args.corafull_controls)


if __name__ == "__main__":
    main()
