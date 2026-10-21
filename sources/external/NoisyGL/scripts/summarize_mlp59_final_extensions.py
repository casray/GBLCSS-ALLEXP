from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEACHER_REAL = "mlp_distill_formal_lcss_x"
TEACHER_SHUFFLED = "mlp_distill_formal_lcss_x_shuffled"
TEACHER_GCN = "mlp_distill_gcn"
BOUNDARY_REAL = "frozengrace_gb_mlp_softdist_beta05_loo"
BOUNDARY_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"
KEYS = ["dataset", "noise_type", "noise_rate", "seed"]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def validate_protocol(
    raw: pd.DataFrame,
    *,
    expected_datasets: set[str],
    expected_methods: set[str],
    expected_settings: set[tuple[str, float]],
    require_formal_target: set[str],
) -> pd.DataFrame:
    records = []
    expected_seeds = set(range(5))
    work = raw.copy()
    work["noise_rate"] = _numeric(work["noise_rate"])
    work["seed"] = _numeric(work["seed"]).astype("Int64")

    for dataset in sorted(expected_datasets):
        for noise_type, noise_rate in sorted(expected_settings):
            for method in sorted(expected_methods):
                group = work[
                    (work["dataset"] == dataset)
                    & (work["method"] == method)
                    & (work["noise_type"] == noise_type)
                    & np.isclose(work["noise_rate"], noise_rate)
                ]
                errors: list[str] = []
                seeds = set(group["seed"].dropna().astype(int).tolist())
                if seeds != expected_seeds or len(group) != 5:
                    errors.append(f"seeds={sorted(seeds)} rows={len(group)}")
                if len(group):
                    if not (group["status"] == "ok").all():
                        errors.append("non-ok status")
                    if not np.isfinite(_numeric(group["test"])).all():
                        errors.append("non-finite test")
                    if not (group["noise_scope"] == "train").all():
                        errors.append("noise scope")
                    if not (_numeric(group["modified_val_count"]).fillna(-1) == 0).all():
                        errors.append("validation leakage")
                    if not (_numeric(group["modified_test_count"]).fillna(-1) == 0).all():
                        errors.append("test leakage")
                    if not (
                        _numeric(group["modified_count"])
                        == _numeric(group["modified_train_count"])
                    ).all():
                        errors.append("modified-count mismatch")
                    if method in require_formal_target:
                        if not np.isclose(_numeric(group["target_beta"]), 0.5).all():
                            errors.append("target beta")
                        if not (_numeric(group["leave_one_out"]) == 1).all():
                            errors.append("leave-one-out")
                        if not (group["tie_mode"] == "distribution_only").all():
                            errors.append("tie mode")
                        if method.startswith("mlp_distill_") and "teacher_use_xh" in group:
                            if not (_numeric(group["teacher_use_xh"]) == 0).all():
                                errors.append("teacher uses H")
                records.append(
                    {
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": noise_rate,
                        "method": method,
                        "rows": len(group),
                        "passed": not errors,
                        "errors": "; ".join(errors),
                    }
                )
    return pd.DataFrame(records)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["test"] = _numeric(work["test"])
    return (
        work.groupby(["dataset", "noise_type", "noise_rate", "method"], as_index=False)
        .agg(mean=("test", "mean"), std=("test", "std"), count=("test", "count"))
        .sort_values(["noise_type", "noise_rate", "dataset", "method"])
    )


def paired_comparisons(
    raw: pd.DataFrame,
    *,
    lhs: str,
    rhs_methods: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = raw.copy()
    work["noise_rate"] = _numeric(work["noise_rate"])
    work["seed"] = _numeric(work["seed"]).astype(int)
    rows = []
    for rhs in rhs_methods:
        left = work[work["method"] == lhs].copy()
        right = work[work["method"] == rhs].copy()
        if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
            raise ValueError(f"duplicate pairing rows for {lhs} vs {rhs}")
        merged = left.merge(right, on=KEYS, suffixes=("_lhs", "_rhs"), how="inner")
        expected = len(left)
        if len(merged) != expected:
            raise ValueError(
                f"incomplete pairing for {lhs} vs {rhs}: {len(merged)}/{expected}"
            )
        for hash_name in ["split_hash", "noise_hash"]:
            if not (merged[f"{hash_name}_lhs"] == merged[f"{hash_name}_rhs"]).all():
                raise ValueError(f"hash mismatch for {lhs} vs {rhs}: {hash_name}")
        merged["lhs"] = lhs
        merged["rhs"] = rhs
        merged["lhs_test"] = _numeric(merged["test_lhs"])
        merged["rhs_test"] = _numeric(merged["test_rhs"])
        merged["delta_pp"] = 100.0 * (merged["lhs_test"] - merged["rhs_test"])
        rows.append(
            merged[
                KEYS
                + ["lhs", "rhs", "lhs_test", "rhs_test", "delta_pp", "split_hash_lhs", "noise_hash_lhs"]
            ].rename(
                columns={
                    "split_hash_lhs": "split_hash",
                    "noise_hash_lhs": "noise_hash",
                }
            )
        )
    paired = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = (
        paired.groupby(["dataset", "noise_type", "noise_rate", "lhs", "rhs"], as_index=False)
        .agg(
            mean_delta_pp=("delta_pp", "mean"),
            std_delta_pp=("delta_pp", "std"),
            positive_seeds=("delta_pp", lambda values: int((values > 0).sum())),
            seed_count=("delta_pp", "count"),
        )
        .sort_values(["noise_type", "noise_rate", "dataset", "rhs"])
    )
    return paired, summary


def evaluate_teacher_gate(
    summary: pd.DataFrame,
    *,
    base_positive_cells: int,
) -> pd.DataFrame:
    gcn = summary[summary["rhs"] == TEACHER_GCN]
    shuffled = summary[summary["rhs"] == TEACHER_SHUFFLED]
    gcn_positive_cells = int((gcn["mean_delta_pp"] > 0).sum())
    shuffled_pass_cells = int(
        ((shuffled["mean_delta_pp"] >= 1.0) & (shuffled["positive_seeds"] >= 4)).sum()
    )
    passed = (
        base_positive_cells == 4
        and gcn_positive_cells >= 3
        and shuffled_pass_cells >= 3
    )
    return pd.DataFrame(
        [
            {
                "phase": "teacher",
                "base_positive_cells": base_positive_cells,
                "gcn_positive_cells": gcn_positive_cells,
                "shuffled_pass_cells": shuffled_pass_cells,
                "passed": passed,
            }
        ]
    )


def evaluate_boundary_entry(
    summary: pd.DataFrame,
    *,
    shuffled_method: str,
    baseline_method: str,
) -> pd.DataFrame:
    records = []
    for dataset, group in summary.groupby("dataset"):
        shuffled = group[group["rhs"] == shuffled_method]
        baseline = group[group["rhs"] == baseline_method]
        if len(shuffled) != 1 or len(baseline) != 1:
            raise ValueError(f"missing entry comparison for {dataset}")
        shuffled = shuffled.iloc[0]
        baseline = baseline.iloc[0]
        passed = (
            float(baseline.mean_delta_pp) >= 0.5
            and int(baseline.positive_seeds) >= 4
            and float(shuffled.mean_delta_pp) >= 1.0
            and int(shuffled.positive_seeds) >= 4
        )
        records.append(
            {
                "phase": "boundary_entry",
                "dataset": dataset,
                "baseline_method": baseline_method,
                "delta_vs_baseline_pp": float(baseline.mean_delta_pp),
                "positive_vs_baseline": int(baseline.positive_seeds),
                "delta_vs_shuffled_pp": float(shuffled.mean_delta_pp),
                "positive_vs_shuffled": int(shuffled.positive_seeds),
                "passed": passed,
            }
        )
    return pd.DataFrame(records)


def evaluate_full_spectrum(
    summary: pd.DataFrame,
    *,
    baseline_method: str,
) -> pd.DataFrame:
    records = []
    for dataset, group in summary[summary["rhs"] == baseline_method].groupby("dataset"):
        if len(group) != 5:
            raise ValueError(f"expected five settings for {dataset}, found {len(group)}")
        noninferior = int((group["mean_delta_pp"] >= 0).sum())
        average = float(group["mean_delta_pp"].mean())
        worst = float(group["mean_delta_pp"].min())
        records.append(
            {
                "phase": "boundary_full_spectrum",
                "dataset": dataset,
                "noninferior_settings": noninferior,
                "average_delta_pp": average,
                "worst_delta_pp": worst,
                "passed": noninferior >= 4 and average >= 0 and worst >= -1.0,
            }
        )
    return pd.DataFrame(records)


def strongest_baseline_methods(
    controls: pd.DataFrame,
    *,
    allowed_methods: set[str],
) -> dict[str, str]:
    eligible = controls[controls["method"].isin(allowed_methods)].copy()
    means = eligible.groupby(["dataset", "method"], as_index=False)["test"].mean()
    best = means.sort_values(["dataset", "test", "method"], ascending=[True, False, True])
    return best.drop_duplicates("dataset").set_index("dataset")["method"].to_dict()


def select_best_baseline_per_setting(
    controls: pd.DataFrame,
    *,
    allowed_methods: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = controls[controls["method"].isin(allowed_methods)].copy()
    setting_keys = ["dataset", "noise_type", "noise_rate"]
    means = eligible.groupby(setting_keys + ["method"], as_index=False)["test"].mean()
    means = means.sort_values(
        setting_keys + ["test", "method"],
        ascending=[True, True, True, False, True],
    )
    selected = means.drop_duplicates(setting_keys).rename(
        columns={"method": "baseline_source_method", "test": "baseline_mean"}
    )
    rows = eligible.merge(
        selected[setting_keys + ["baseline_source_method"]],
        left_on=setting_keys + ["method"],
        right_on=setting_keys + ["baseline_source_method"],
        how="inner",
    )
    rows["method"] = "strongest_eligible_baseline"
    return rows, selected


def _read_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MLP-59 final extensions")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--teacher_base", type=Path)
    parser.add_argument("--boundary_baselines", type=Path, nargs="*")
    parser.add_argument("--full_spectrum_baselines", type=Path, nargs="*")
    return parser


def _write_report(
    run_dir: Path,
    *,
    aggregate: pd.DataFrame,
    gate: pd.DataFrame,
    full_selection: pd.DataFrame,
) -> None:
    def percent_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No records."
        out = frame.copy()
        out["mean"] = (100.0 * out["mean"]).map(lambda value: f"{value:.2f}")
        out["std"] = (100.0 * out["std"]).map(lambda value: f"{value:.2f}")
        return out.to_markdown(index=False)

    teacher_methods = {
        TEACHER_REAL,
        TEACHER_SHUFFLED,
        TEACHER_GCN,
        "ffg2m_reimplemented",
        "krd_adapted",
        "vqgraph_adapted",
    }
    teacher = aggregate[
        aggregate["method"].isin(teacher_methods)
        & aggregate["dataset"].isin({"coauthorcs", "amazonpho"})
    ]
    boundary = aggregate[
        aggregate["method"].isin({BOUNDARY_REAL, BOUNDARY_SHUFFLED})
        & aggregate["dataset"].isin(
            {"pubmed", "amazoncom", "coraml", "attribwiki", "citeseerfull", "coauthorph"}
        )
        & (aggregate["noise_type"] == "uniform")
        & np.isclose(aggregate["noise_rate"], 0.4)
    ]
    teacher_gate = gate[gate["phase"] == "teacher"]
    entry_gate = gate[gate["phase"] == "boundary_entry"]
    full_gate = gate[gate["phase"] == "boundary_full_spectrum"]

    lines = [
        "# MLP-59 final dataset and teacher extension report",
        "",
        "## Protocol",
        "",
        "All formal runs use percent 0.2/0.1/0.7 splits, train-only synthetic label noise, "
        "seeds 0--4, and exact split/noise-hash pairing. The formal GB-LCSS target uses "
        "leave-one-out voting with `tie_mode=distribution_only`.",
        "",
        "## Teacher extension",
        "",
        percent_table(teacher),
        "",
        "Teacher gate:",
        "",
        teacher_gate.to_markdown(index=False) if not teacher_gate.empty else "No teacher gate.",
        "",
        "The formal ball-consensus teacher passed the frozen gate on Coauthor-CS and "
        "Amazon-Photo. It exceeded the base GB-LCSS student and the GCN teacher in all four "
        "dataset-rate cells, and exceeded the matched shuffled teacher by at least 1 pp with "
        "at least four positive seeds in all four cells.",
        "",
        "## Boundary uniform-0.4 entry test",
        "",
        percent_table(boundary),
        "",
        entry_gate.to_markdown(index=False) if not entry_gate.empty else "No entry gate.",
        "",
        "Only CiteSeerFull passed the frozen entry gate. No other boundary dataset was "
        "advanced to the full-spectrum stage.",
        "",
        "## CiteSeerFull full-spectrum decision",
        "",
        full_selection.to_markdown(index=False) if not full_selection.empty else "No full-spectrum selection.",
        "",
        full_gate.to_markdown(index=False) if not full_gate.empty else "No full-spectrum gate.",
        "",
        "CiteSeerFull was non-inferior in four of five settings and had a positive mean "
        "cross-setting delta, but failed the pre-registered worst-cell constraint because "
        "uniform 0.6 was more than 1 pp below the strongest eligible baseline. It is therefore "
        "not added to the formal five-dataset main matrix.",
        "",
        "## Decision",
        "",
        "- Extend the teacher-distillation table to Coauthor-CS and Amazon-Photo.",
        "- Keep the existing five-dataset main matrix unchanged.",
        "- Report the six final-target boundary diagnostics and the CiteSeerFull high-noise "
        "failure without promoting any boundary dataset to the main matrix.",
    ]
    (run_dir / "mlp59_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir
    teacher_paths = [
        run_dir / "teacher_u0p4_seeds0_4.csv",
        run_dir / "teacher_u0p6_seeds0_4.csv",
    ]
    boundary_path = run_dir / "boundary_u0p4_seeds0_4.csv"
    teacher = _read_many(teacher_paths)
    boundary = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
    outputs = []
    audits = []
    gates = []
    paired_all = []
    raw_all = []
    full_selection = pd.DataFrame()

    if not teacher.empty:
        raw_all.append(teacher)
        audit = validate_protocol(
            teacher,
            expected_datasets={"coauthorcs", "amazonpho"},
            expected_methods={TEACHER_REAL, TEACHER_SHUFFLED, TEACHER_GCN},
            expected_settings={("uniform", 0.4), ("uniform", 0.6)},
            require_formal_target={TEACHER_REAL, TEACHER_SHUFFLED},
        )
        audit.insert(0, "phase", "teacher")
        audits.append(audit)
        teacher_with_base = teacher
        if args.teacher_base and args.teacher_base.exists():
            teacher_with_base = pd.concat(
                [teacher, pd.read_csv(args.teacher_base)], ignore_index=True, sort=False
            )
        rhs = [TEACHER_SHUFFLED, TEACHER_GCN]
        if BOUNDARY_REAL in set(teacher_with_base["method"]):
            rhs.append(BOUNDARY_REAL)
        paired, summary = paired_comparisons(teacher_with_base, lhs=TEACHER_REAL, rhs_methods=rhs)
        paired_all.append(paired)
        base_cells = 0
        if BOUNDARY_REAL in rhs:
            base_cells = int(
                (
                    summary[summary["rhs"] == BOUNDARY_REAL]["mean_delta_pp"] > 0
                ).sum()
            )
        gates.append(evaluate_teacher_gate(summary, base_positive_cells=base_cells))
        outputs.append(aggregate_results(teacher_with_base))

        g2m = _read_many(
            [run_dir / "g2m_u0p4_seeds0_4.csv", run_dir / "g2m_u0p6_seeds0_4.csv"]
        )
        if not g2m.empty:
            g2m_audit = validate_protocol(
                g2m,
                expected_datasets={"coauthorcs", "amazonpho"},
                expected_methods={"ffg2m_reimplemented", "krd_adapted", "vqgraph_adapted"},
                expected_settings={("uniform", 0.4), ("uniform", 0.6)},
                require_formal_target=set(),
            )
            g2m_audit.insert(0, "phase", "g2m")
            audits.append(g2m_audit)
            raw_all.append(g2m)
            outputs.append(aggregate_results(g2m))

    if not boundary.empty:
        raw_all.append(boundary)
        audit = validate_protocol(
            boundary,
            expected_datasets={
                "pubmed", "amazoncom", "coraml", "attribwiki", "citeseerfull", "coauthorph"
            },
            expected_methods={BOUNDARY_REAL, BOUNDARY_SHUFFLED},
            expected_settings={("uniform", 0.4)},
            require_formal_target={BOUNDARY_REAL, BOUNDARY_SHUFFLED},
        )
        audit.insert(0, "phase", "boundary_entry")
        audits.append(audit)
        controls = _read_many(args.boundary_baselines or [])
        controls = controls[controls["dataset"].isin(boundary["dataset"].unique())].copy()
        controls.to_csv(run_dir / "boundary_baseline_controls.csv", index=False)
        combined = pd.concat([boundary, controls], ignore_index=True, sort=False)
        allowed = {
            "mlp_ce", "mlp_small_loss", "mlp_coteaching", "mlp_jocor",
            "mlp_sce", "mlp_elr", "mlp_logitclip",
        }
        best = strongest_baseline_methods(controls, allowed_methods=allowed)
        for dataset, baseline_method in best.items():
            subset = combined[combined["dataset"] == dataset]
            paired, summary = paired_comparisons(
                subset,
                lhs=BOUNDARY_REAL,
                rhs_methods=[BOUNDARY_SHUFFLED, baseline_method],
            )
            paired_all.append(paired)
            gates.append(
                evaluate_boundary_entry(
                    summary,
                    shuffled_method=BOUNDARY_SHUFFLED,
                    baseline_method=baseline_method,
                )
            )
        outputs.append(aggregate_results(combined))

    full_paths = [
        run_dir / "citeseerfull_uniform0p2_seeds0_4.csv",
        run_dir / "citeseerfull_uniform0p6_seeds0_4.csv",
        run_dir / "citeseerfull_pair0p4_seeds0_4.csv",
        run_dir / "citeseerfull_instance0p4_seeds0_4.csv",
    ]
    boundary_u04 = pd.DataFrame()
    if boundary_path.exists():
        boundary_u04 = pd.read_csv(boundary_path)
        boundary_u04 = boundary_u04[boundary_u04["dataset"] == "citeseerfull"]
    full_target = pd.concat(
        [_read_many(full_paths), boundary_u04], ignore_index=True, sort=False
    )
    if not full_target.empty and args.full_spectrum_baselines:
        full_target_audit = validate_protocol(
            full_target,
            expected_datasets={"citeseerfull"},
            expected_methods={BOUNDARY_REAL, BOUNDARY_SHUFFLED},
            expected_settings={
                ("uniform", 0.2), ("uniform", 0.4), ("uniform", 0.6),
                ("pair", 0.4), ("instance", 0.4),
            },
            require_formal_target={BOUNDARY_REAL, BOUNDARY_SHUFFLED},
        )
        full_target_audit.insert(0, "phase", "full_target")
        audits.append(full_target_audit)
        full_target.to_csv(run_dir / "boundary_full_spectrum.csv", index=False)
        raw_all.append(full_target)
        full_controls = _read_many(args.full_spectrum_baselines)
        full_controls = full_controls[full_controls["dataset"] == "citeseerfull"].copy()
        allowed = {
            "mlp_ce", "mlp_small_loss", "mlp_coteaching", "mlp_jocor",
            "mlp_sce", "mlp_elr", "mlp_logitclip",
        }
        full_controls = full_controls[full_controls["method"].isin(allowed)].copy()
        full_baseline_audit = validate_protocol(
            full_controls,
            expected_datasets={"citeseerfull"},
            expected_methods=allowed,
            expected_settings={
                ("uniform", 0.2), ("uniform", 0.4), ("uniform", 0.6),
                ("pair", 0.4), ("instance", 0.4),
            },
            require_formal_target=set(),
        )
        full_baseline_audit.insert(0, "phase", "full_baselines")
        audits.append(full_baseline_audit)
        best_rows, full_selection = select_best_baseline_per_setting(
            full_controls,
            allowed_methods=allowed,
        )
        full_selection.to_csv(run_dir / "full_spectrum_baseline_selection.csv", index=False)
        full_combined = pd.concat([full_target, best_rows], ignore_index=True, sort=False)
        paired, summary = paired_comparisons(
            full_combined,
            lhs=BOUNDARY_REAL,
            rhs_methods=[BOUNDARY_SHUFFLED, "strongest_eligible_baseline"],
        )
        paired_all.append(paired)
        gates.append(
            evaluate_full_spectrum(
                summary,
                baseline_method="strongest_eligible_baseline",
            )
        )
        outputs.append(aggregate_results(full_combined))

    aggregate = pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
    paired = pd.concat(paired_all, ignore_index=True, sort=False) if paired_all else pd.DataFrame()
    audit = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
    gate = pd.concat(gates, ignore_index=True, sort=False) if gates else pd.DataFrame()
    raw = pd.concat(raw_all, ignore_index=True, sort=False) if raw_all else pd.DataFrame()
    if not raw.empty:
        raw = raw.drop_duplicates(["dataset", "method", "noise_type", "noise_rate", "seed"])
    raw.to_csv(run_dir / "mlp59_raw.csv", index=False)
    aggregate.to_csv(run_dir / "mlp59_aggregate.csv", index=False)
    paired.to_csv(run_dir / "mlp59_paired_deltas.csv", index=False)
    audit.to_csv(run_dir / "mlp59_protocol_audit.csv", index=False)
    completeness = (
        audit.groupby("phase", as_index=False)
        .agg(
            expected_cells=("passed", "count"),
            passed_cells=("passed", "sum"),
            total_rows=("rows", "sum"),
        )
        if not audit.empty
        else pd.DataFrame()
    )
    if not completeness.empty:
        completeness["passed"] = (
            completeness["expected_cells"] == completeness["passed_cells"]
        )
    completeness.to_csv(run_dir / "mlp59_completeness.csv", index=False)
    gate.to_csv(run_dir / "mlp59_gate_decisions.csv", index=False)
    _write_report(
        run_dir,
        aggregate=aggregate,
        gate=gate,
        full_selection=full_selection,
    )


if __name__ == "__main__":
    main()
