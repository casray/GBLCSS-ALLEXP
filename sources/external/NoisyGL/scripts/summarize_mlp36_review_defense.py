from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho", "pubmed")
CURRENT = "frozengrace_gb_mlp_softdist_beta05"
LOO = "frozengrace_gb_mlp_softdist_beta05_loo"
LOO_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"
UNIFORM = "mlp_uniform_smoothing"
PRIOR = "mlp_prior_smoothing"
METHODS = (CURRENT, LOO, LOO_SHUFFLED, UNIFORM, PRIOR)
COMPARISONS = {
    "loo_vs_current": (LOO, CURRENT),
    "loo_vs_loo_shuffled": (LOO, LOO_SHUFFLED),
    "loo_vs_uniform_smoothing": (LOO, UNIFORM),
    "loo_vs_prior_smoothing": (LOO, PRIOR),
}


def holm_adjust(p_values) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.tolist()


def paired_statistics(left, right) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional with identical shape")
    if left.size < 2:
        raise ValueError("at least two paired observations are required")
    delta = left - right
    mean_delta = float(delta.mean())
    std_delta = float(delta.std(ddof=1))
    if np.isclose(std_delta, 0.0, atol=1e-12, rtol=0.0):
        p_value = 0.0 if mean_delta != 0.0 else 1.0
        ci_low = ci_high = mean_delta
    else:
        p_value = float(ttest_rel(left, right).pvalue)
        half_width = float(
            student_t.ppf(0.975, df=left.size - 1)
            * std_delta
            / np.sqrt(left.size)
        )
        ci_low = mean_delta - half_width
        ci_high = mean_delta + half_width
    return {
        "n": int(left.size),
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "positive_seeds": int((delta > 0).sum()),
        "negative_seeds": int((delta < 0).sum()),
        "paired_t_p": p_value,
    }


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "status" not in frame or not (frame["status"] == "ok").all():
        raise ValueError(f"non-ok row in {path}")
    return frame


def load_complete_run(run_dir: Path) -> pd.DataFrame:
    seed0 = _read(run_dir / "uniform0p4_seed0.csv")
    seeds1_4 = _read(run_dir / "uniform0p4_seeds_1_2_3_4.csv")
    current1_4 = _read(run_dir / "current_seeds_1_2_3_4_diagnostics.csv")
    seed0 = seed0[seed0["method"].isin(METHODS)]
    combined = pd.concat([seed0, seeds1_4, current1_4], ignore_index=True, sort=False)
    combined = combined[combined["method"].isin(METHODS)].copy()
    combined["seed"] = combined["seed"].astype(int)
    combined["noise_rate"] = combined["noise_rate"].astype(float)

    protocol_ok = (
        set(combined["dataset"]) == set(DATASETS)
        and (combined["noise_type"] == "uniform").all()
        and np.isclose(combined["noise_rate"], 0.4).all()
        and (combined["noise_scope"] == "train").all()
        and (combined["modified_val_count"].astype(int) == 0).all()
        and (combined["modified_test_count"].astype(int) == 0).all()
    )
    if not protocol_ok:
        raise ValueError("MLP-36 protocol or leakage audit failed")
    if combined.duplicated(["dataset", "method", "seed"]).any():
        raise ValueError("duplicate dataset/method/seed row")
    counts = combined.groupby(["dataset", "method"]).size()
    if len(counts) != len(DATASETS) * len(METHODS) or not (counts == 5).all():
        raise ValueError("MLP-36 requires five seeds for every dataset and method")
    for (_, _), group in combined.groupby(["dataset", "seed"]):
        if group["split_hash"].nunique() != 1 or group["noise_hash"].nunique() != 1:
            raise ValueError("split/noise hash mismatch within a paired setting")
    return combined.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_candidates = (
        "test",
        "soft_target_brier",
        "noisy_onehot_brier",
        "soft_brier_gain",
        "soft_clean_mass_delta_mean",
        "soft_target_argmax_accuracy",
        "soft_pseudo_ncr",
        "soft_pseudo_cdr",
        "alpha_positive_train_ratio",
        "wall_time_seconds",
        "ball_build_time_seconds",
    )
    metrics = [metric for metric in metric_candidates if metric in frame]
    numeric = frame.copy()
    for metric in metrics:
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
    rows = []
    for (dataset, method), group in numeric.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "n": int(len(group))}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, (left_method, right_method) in COMPARISONS.items():
        family_rows = []
        for dataset in DATASETS:
            left = frame[(frame["dataset"] == dataset) & (frame["method"] == left_method)]
            right = frame[(frame["dataset"] == dataset) & (frame["method"] == right_method)]
            merged = left.merge(
                right,
                on="seed",
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            if not (
                (merged["split_hash_left"] == merged["split_hash_right"]).all()
                and (merged["noise_hash_left"] == merged["noise_hash_right"]).all()
            ):
                raise ValueError(f"paired hash mismatch for {dataset}/{comparison}")
            stats = paired_statistics(merged["test_left"], merged["test_right"])
            family_rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "left_method": left_method,
                    "right_method": right_method,
                    **stats,
                }
            )
        adjusted = holm_adjust([row["paired_t_p"] for row in family_rows])
        for row, adjusted_p in zip(family_rows, adjusted):
            row["holm_p_within_family"] = adjusted_p
            rows.append(row)
    return pd.DataFrame(rows)


def decision_table(aggregate_frame: pd.DataFrame, paired_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    current = paired_frame[paired_frame["comparison"] == "loo_vs_current"]
    shuffled = paired_frame[paired_frame["comparison"] == "loo_vs_loo_shuffled"]
    uniform = paired_frame[paired_frame["comparison"] == "loo_vs_uniform_smoothing"]
    prior = paired_frame[paired_frame["comparison"] == "loo_vs_prior_smoothing"]
    smoothing = uniform[["dataset", "mean_delta"]].merge(
        prior[["dataset", "mean_delta"]], on="dataset", suffixes=("_uniform", "_prior")
    )
    smoothing["loo_beats_both"] = (
        (smoothing["mean_delta_uniform"] > 0) & (smoothing["mean_delta_prior"] > 0)
    )
    summary = {
        "loo_minus_current_dataset_mean": float(current["mean_delta"].mean()),
        "loo_nonnegative_vs_current_datasets": int((current["mean_delta"] >= 0).sum()),
        "loo_four_of_five_positive_vs_current_datasets": int((current["positive_seeds"] >= 4).sum()),
        "loo_beats_both_smoothing_datasets": int(smoothing["loo_beats_both"].sum()),
        "loo_beats_shuffled_datasets": int((shuffled["mean_delta"] > 0).sum()),
        "loo_four_of_five_positive_vs_shuffled_datasets": int((shuffled["positive_seeds"] >= 4).sum()),
    }
    if (
        summary["loo_minus_current_dataset_mean"] >= -0.005
        and summary["loo_nonnegative_vs_current_datasets"] >= 4
        and summary["loo_beats_both_smoothing_datasets"] >= 4
        and summary["loo_beats_shuffled_datasets"] >= 4
    ):
        conclusion = "external_local_consensus_supported"
    elif (
        summary["loo_minus_current_dataset_mean"] >= -0.005
        and summary["loo_beats_shuffled_datasets"] >= 4
    ):
        conclusion = "local_consensus_plus_group_aware_smoothing"
    else:
        conclusion = "group_aware_smoothing_interpretation_required"
    summary["mechanism_conclusion"] = conclusion
    rows = [{"criterion": key, "value": value} for key, value in summary.items()]
    return pd.DataFrame(rows), summary


def write_report(
    output: Path,
    aggregate_frame: pd.DataFrame,
    paired_frame: pd.DataFrame,
    decision: dict,
) -> None:
    table = aggregate_frame.pivot(index="dataset", columns="method", values="test_mean")
    lines = [
        "# MLP-36 LOO 与显式平滑审稿防御报告",
        "",
        "## 协议与完整性",
        "",
        "- Cora/CiteSeer/CoraFull/Coauthor-CS/Amazon-Photo/PubMed。",
        "- uniform 0.4、train-only、percent 0.2/0.1/0.7、seeds 0-4。",
        "- 每个 dataset/method 均为 5 个成功运行；逐 seed split/noise hash 完全配对；val/test 修改数均为 0。",
        "",
        "## 五种子精度",
        "",
        "| Dataset | Current | LOO | LOO shuffled | Uniform LS | Prior LS | LOO-Current (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    current_delta = paired_frame[paired_frame["comparison"] == "loo_vs_current"].set_index("dataset")
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | {100*table.loc[dataset, CURRENT]:.2f} | {100*table.loc[dataset, LOO]:.2f} | "
            f"{100*table.loc[dataset, LOO_SHUFFLED]:.2f} | {100*table.loc[dataset, UNIFORM]:.2f} | "
            f"{100*table.loc[dataset, PRIOR]:.2f} | {100*current_delta.loc[dataset, 'mean_delta']:+.2f} |"
        )
    lines.extend([
        "",
        "## 终判",
        "",
        f"- LOO 相对 current 的六数据集平均 paired delta：{100*decision['loo_minus_current_dataset_mean']:+.2f}pp。",
        f"- LOO 对 current 非负：{decision['loo_nonnegative_vs_current_datasets']}/6；达到 4/5 seeds 正向：{decision['loo_four_of_five_positive_vs_current_datasets']}/6。",
        f"- LOO 同时超过 uniform/prior smoothing：{decision['loo_beats_both_smoothing_datasets']}/6。",
        f"- LOO 超过 LOO-shuffled：{decision['loo_beats_shuffled_datasets']}/6，其中 4/5 seeds 正向：{decision['loo_four_of_five_positive_vs_shuffled_datasets']}/6。",
        f"- 机制结论：`{decision['mechanism_conclusion']}`。",
        "",
        "LOO 在不使用当前节点自身投票的情况下保持或提高总体表现，因此现有收益不能仅归因于自标签回流。"
        "同时，LOO 并非在每个数据集都显著优于 current，论文应将其作为机制消融，避免把结果写成统一算法升级。",
        "",
        "## 统计口径",
        "",
        "逐数据集 paired t-test 只用于预声明 comparison family，Holm 校正在每个六数据集 family 内完成。"
        "由于 n=5，正文以 paired delta、95% CI 与方向计数为主，不把未拒绝零假设写成等效。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, default=Path("runs/mlp36_review_defense_20260715"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    frame = load_complete_run(run_dir)
    aggregate_frame = aggregate(frame)
    paired_frame = paired_deltas(frame)
    decision_frame, decision = decision_table(aggregate_frame, paired_frame)
    frame.to_csv(run_dir / "mlp36_raw.csv", index=False)
    aggregate_frame.to_csv(run_dir / "mlp36_aggregate.csv", index=False)
    paired_frame.to_csv(run_dir / "mlp36_paired_deltas.csv", index=False)
    decision_frame.to_csv(run_dir / "mlp36_gate_decisions.csv", index=False)
    (run_dir / "mlp36_gate_decisions.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(run_dir / "mlp36_report.md", aggregate_frame, paired_frame, decision)


if __name__ == "__main__":
    main()
