from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORMAL_DATASETS = ("cora", "citeseer", "coauthorcs", "amazonpho")
RATES = (0.4, 0.6)
SEEDS = tuple(range(5))
METHOD = "vqgraph_sampled_faithful"
BALL_METHOD = "mlp_distill_formal_lcss_x"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="raise")


def validate_formal_rows(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
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
        "vqgraph_active_codes",
        "vqgraph_teacher_parameter_device",
        "vqgraph_student_parameter_device",
        "g2m_train_uses_graph",
        "g2m_inference_uses_graph",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"MLP-62 rows are missing columns: {sorted(missing)}")

    work = raw.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["seed"] = _numeric(work, "seed").astype(int)
    work["noise_rate"] = _numeric(work, "noise_rate")
    work["test"] = _numeric(work, "test")
    work["vqgraph_active_codes"] = _numeric(work, "vqgraph_active_codes")

    expected = {
        (dataset, rate, seed)
        for dataset in FORMAL_DATASETS
        for rate in RATES
        for seed in SEEDS
    }
    observed = set(zip(work.dataset, work.noise_rate, work.seed, strict=False))
    if len(work) != len(expected) or observed != expected:
        raise ValueError(
            "incomplete or unexpected MLP-62 formal grid: "
            f"rows={len(work)}, missing={sorted(expected-observed)[:5]}, "
            f"unexpected={sorted(observed-expected)[:5]}"
        )
    if work.duplicated(["dataset", "noise_rate", "seed"]).any():
        raise ValueError("duplicate MLP-62 formal rows")

    checks = pd.DataFrame(
        {
            "dataset": work["dataset"],
            "noise_rate": work["noise_rate"],
            "seed": work["seed"],
            "status_ok": work["status"].eq("ok"),
            "method_ok": work["method"].eq(METHOD),
            "uniform_noise": work["noise_type"].eq("uniform"),
            "train_only_noise": work["noise_scope"].eq("train"),
            "finite_test": np.isfinite(work["test"]),
            "active_code": work["vqgraph_active_codes"].gt(0),
            "teacher_gpu": work["vqgraph_teacher_parameter_device"].eq("cuda:0"),
            "student_gpu": work["vqgraph_student_parameter_device"].eq("cuda:0"),
            "training_graph": _numeric(work, "g2m_train_uses_graph").eq(1),
            "inference_graph_free": _numeric(work, "g2m_inference_uses_graph").eq(0),
            "nonempty_hashes": (
                work["split_hash"].astype(str).str.len().gt(0)
                & work["noise_hash"].astype(str).str.len().gt(0)
            ),
        }
    )
    check_columns = list(checks.columns[3:])
    checks["passed"] = checks[check_columns].all(axis=1)
    checks["failed_checks"] = checks.apply(
        lambda row: ",".join(name for name in check_columns if not bool(row[name])),
        axis=1,
    )
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], ["dataset", "noise_rate", "seed", "failed_checks"]]
        raise ValueError(f"MLP-62 protocol audit failed:\n{failed.to_string(index=False)}")
    return checks


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["noise_rate"] = _numeric(work, "noise_rate")
    work["test"] = _numeric(work, "test")
    return (
        work.groupby(["noise_rate", "dataset", "method"], sort=True)["test"]
        .agg(mean="mean", std="std", count="count", minimum="min", maximum="max")
        .reset_index()
    )


def pair_with_ball_teacher(raw: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "seed", "noise_rate", "method", "test", "split_hash", "noise_hash"}
    missing = required.difference(controls.columns)
    if missing:
        raise ValueError(f"ball-teacher rows are missing columns: {sorted(missing)}")

    left = raw[["dataset", "seed", "noise_rate", "test", "split_hash", "noise_hash"]].copy()
    left.columns = ["dataset", "seed", "noise_rate", "vqgraph_test", "split_hash_vq", "noise_hash_vq"]
    right = controls.loc[
        controls["method"].eq(BALL_METHOD),
        ["dataset", "seed", "noise_rate", "test", "split_hash", "noise_hash"],
    ].copy()
    right.columns = ["dataset", "seed", "noise_rate", "ball_test", "split_hash_ball", "noise_hash_ball"]
    for frame in (left, right):
        frame["dataset"] = frame["dataset"].astype(str).str.lower()
        frame["seed"] = _numeric(frame, "seed").astype(int)
        frame["noise_rate"] = _numeric(frame, "noise_rate")
    left["vqgraph_test"] = _numeric(left, "vqgraph_test")
    right["ball_test"] = _numeric(right, "ball_test")

    paired = left.merge(right, on=["dataset", "seed", "noise_rate"], how="left", validate="one_to_one")
    if paired["ball_test"].isna().any():
        raise ValueError("missing paired ball-teacher rows")
    paired["hashes_paired"] = (
        paired["split_hash_vq"].eq(paired["split_hash_ball"])
        & paired["noise_hash_vq"].eq(paired["noise_hash_ball"])
    )
    if not paired["hashes_paired"].all():
        raise ValueError("VQGraph and ball-teacher split/noise hash mismatch")
    paired["delta_pp"] = 100.0 * (paired["vqgraph_test"] - paired["ball_test"])
    return paired.sort_values(["noise_rate", "dataset", "seed"]).reset_index(drop=True)


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["noise_rate", "dataset"], sort=True)
        .agg(
            vqgraph_mean=("vqgraph_test", "mean"),
            vqgraph_std=("vqgraph_test", "std"),
            ball_mean=("ball_test", "mean"),
            ball_std=("ball_test", "std"),
            mean_delta_pp=("delta_pp", "mean"),
            std_delta_pp=("delta_pp", "std"),
            positive_seeds=("delta_pp", lambda values: int((values > 0).sum())),
            count=("seed", "count"),
        )
        .reset_index()
    )


def load_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def write_report(
    path: Path,
    aggregate: pd.DataFrame,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# MLP-62 VQGraph neighbor-sampled ROCm report",
        "",
        "## Research question",
        "",
        "Can the official VQGraph GraphSAGE execution semantics be transferred to the current AMD ROCm host, removing the full-batch graph-reconstruction bottleneck and supporting the manuscript noise protocol?",
        "",
        "## Method and protocol",
        "",
        "- VQGraph role: graph-teacher comparison; granular balls are not used by this baseline.",
        "- Source: official VQGraph commit `60aa1c99782d542c316a37f0394eba934bcf97b0`.",
        "- Identity: faithful PyTorch/ROCm protocol reimplementation, not an official DGL run.",
        "- Teacher: sampled local GraphConv, official VectorQuantize, feature reconstruction, local adjacency reconstruction, and the existing VQGraph soft-label/soft-token student path.",
        "- Sampling: CPU inbound-neighbor sampling with fanouts [5,5]; all trainable operations run on ROCm `cuda:0`.",
        "- Evaluation: percent 0.2/0.1/0.7 splits, train-only uniform noise at 0.4 and 0.6, seeds 0-4, clean-validation checkpoint selection.",
        "- Hyperparameters were frozen before the runs. No post-result tuning was performed.",
        "",
        "## Code and configuration changes",
        "",
        "- Added a deterministic CPU inbound-neighbor sampler and a PyTorch implementation of DGL GraphConv with `norm='both'`.",
        "- Added a separate sampled VQGraph predictor; the existing full-batch predictor was not changed.",
        "- Reused the vendored official VectorQuantize implementation and the existing VQGraph student objective.",
        "- Added five sampled-path configurations. Published settings were used where available; CoraFull and Coauthor-CS used the pre-registered nearest-node-count mapping.",
        "",
        "## Expected mechanism and decision criteria",
        "",
        "Neighbor sampling was expected to replace the global dense graph-reconstruction tensor with bounded local tensors while preserving the tokenizer and distillation objectives. Feasibility required finite outputs, at least one active code, ROCm teacher/student parameters, paired split/noise hashes, and no global N-by-N reconstruction. Method ranking was not a feasibility criterion. If the smoke checks passed, every retained dataset-rate-seed row had to be reported without post-result tuning.",
        "",
        "## Formal four-dataset results",
        "",
        "| noise | dataset | VQGraph (%) | ball teacher (%) | delta (pp) | positive seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.noise_rate:.1f} | {row.dataset} | "
            f"{100*row.vqgraph_mean:.2f}+/-{100*row.vqgraph_std:.2f} | "
            f"{100*row.ball_mean:.2f}+/-{100*row.ball_std:.2f} | "
            f"{row.mean_delta_pp:+.2f} | {row.positive_seeds}/{row.count} |"
        )
    lines.extend(
        [
            "",
            "## CoraFull diagnostic",
            "",
            "CoraFull was stopped by author decision after two completed uniform-0.4 seeds. These rows are diagnostic only and are excluded from every formal aggregate and win/loss count.",
            "",
            "| seed | test (%) | teacher val (%) | active codes | wall time (min) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | {100*float(row.test):.2f} | "
            f"{100*float(row.teacher_best_val):.2f} | {int(row.vqgraph_active_codes)} | "
            f"{float(row.wall_time_seconds)/60:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol audit",
            "",
            f"- Formal rows: {int(aggregate['count'].sum())}/40.",
            "- All formal rows have finite test metrics, positive active-code counts, ROCm teacher/student parameters, training-time graph use, graph-free student inference, and paired split/noise hashes.",
            "- No CPU model training or inference fallback was used.",
            "- CoraFull and Coauthor-CS are protocol extensions because they are not official VQGraph datasets.",
            "",
            "## Interpretation and decision",
            "",
            "The sampled port removes the global N-by-N reconstruction allocation and runs end to end on the AMD GPU. It does not make VQGraph a strong noisy-label baseline under this protocol. The four-dataset results are below the paired granular-ball teacher in every aggregate cell and show large seed variance, especially on Amazon-Photo. The two completed CoraFull runs use only one and three active codes and yield near-random accuracy.",
            "",
            "The complete five-dataset row is rejected because CoraFull was stopped. The four-dataset results may be retained as a supplementary protocol-reimplementation diagnostic, but they must not be presented as an official reproduction or as a complete Table 4 row.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ball-u04-small", type=Path, required=True)
    parser.add_argument("--ball-u06-small", type=Path, required=True)
    parser.add_argument("--ball-u04-large", type=Path, required=True)
    parser.add_argument("--ball-u06-large", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    formal_paths = [
        run_dir / "cora_u04_seed0_smoke_multigraph.csv",
        run_dir / "formal_u04_missing_seed0.csv",
        run_dir / "coauthorcs_u04_seed0_smoke_multigraph.csv",
        run_dir / "formal_u04_seeds1_4.csv",
        run_dir / "formal_u04_coauthor_amazon_seeds1_4.csv",
        run_dir / "formal_u04_coauthor_seed4.csv",
        run_dir / "formal_u04_amazon_seeds1_4.csv",
        run_dir / "formal_u06_small_medium_seeds0_4.csv",
        run_dir / "formal_u06_coauthor_seeds0_4.csv",
    ]
    raw = load_csvs(formal_paths)
    raw["dataset"] = raw["dataset"].astype(str).str.lower()
    raw["seed"] = _numeric(raw, "seed").astype(int)
    raw["noise_rate"] = _numeric(raw, "noise_rate")
    formal = raw.loc[
        raw["dataset"].isin(FORMAL_DATASETS)
        & raw["noise_rate"].isin(RATES)
        & raw["seed"].isin(SEEDS)
    ].drop_duplicates(["dataset", "noise_rate", "seed"], keep="last")
    formal = formal.sort_values(["noise_rate", "dataset", "seed"]).reset_index(drop=True)
    audit = validate_formal_rows(formal)

    controls = load_csvs(
        [args.ball_u04_small, args.ball_u06_small, args.ball_u04_large, args.ball_u06_large]
    )
    controls["dataset"] = controls["dataset"].astype(str).str.lower()
    paired = pair_with_ball_teacher(formal, controls)
    aggregate = aggregate_results(formal)
    comparison = paired_summary(paired)

    diagnostic_paths = [
        run_dir / "corafull_u04_seed0_smoke_multigraph.csv",
        run_dir / "formal_u04_seeds1_4.csv",
    ]
    diagnostic_raw = load_csvs(diagnostic_paths)
    diagnostics = diagnostic_raw.loc[
        diagnostic_raw["dataset"].astype(str).str.lower().eq("corafull")
        & _numeric(diagnostic_raw, "seed").isin((0, 1))
    ].drop_duplicates(["dataset", "noise_rate", "seed"], keep="last")
    if len(diagnostics) != 2:
        raise ValueError(f"expected two completed CoraFull diagnostic rows, got {len(diagnostics)}")

    formal.to_csv(run_dir / "mlp62_formal_four_dataset_raw.csv", index=False)
    diagnostics.to_csv(run_dir / "mlp62_corafull_diagnostic_raw.csv", index=False)
    aggregate.to_csv(run_dir / "mlp62_aggregate.csv", index=False)
    paired.to_csv(run_dir / "mlp62_paired_to_ball_teacher.csv", index=False)
    comparison.to_csv(run_dir / "mlp62_paired_summary.csv", index=False)
    audit.to_csv(run_dir / "mlp62_protocol_audit.csv", index=False)
    write_report(run_dir / "mlp62_report.md", aggregate, comparison, diagnostics)


if __name__ == "__main__":
    main()
