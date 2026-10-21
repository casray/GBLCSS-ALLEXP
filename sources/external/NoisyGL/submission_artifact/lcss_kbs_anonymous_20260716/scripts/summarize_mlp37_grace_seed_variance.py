from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


DATASETS = ("cora", "citeseer", "corafull")
H_SEEDS = (0, 1, 2)
DOWNSTREAM_SEEDS = (0, 1, 2)
LCSS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
CE = "mlp_ce"
JOCOR = "mlp_jocor"
METHODS = (LCSS, SHUFFLED, CE, JOCOR)


def assignment_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("assignment arrays must be one-dimensional and have equal shapes")
    return float(adjusted_rand_score(first, second))


def _numeric_h_seed(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["h_seed"], errors="coerce")


def validate_raw(frame: pd.DataFrame) -> None:
    required = {
        "dataset",
        "method",
        "seed",
        "h_seed",
        "test",
        "status",
        "noise_scope",
        "modified_val_count",
        "modified_test_count",
        "split_hash",
        "noise_hash",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"incomplete MLP-37 raw data: missing columns {sorted(missing)}")

    formal = frame[frame["dataset"].isin(DATASETS) & frame["method"].isin(METHODS)].copy()
    if len(formal) != 72:
        raise ValueError(f"incomplete MLP-37 design: expected 72 rows, found {len(formal)}")
    if not (formal["status"] == "ok").all():
        raise ValueError("incomplete MLP-37 design: non-ok run present")
    if not (formal["noise_scope"] == "train").all():
        raise ValueError("incomplete MLP-37 design: non-train noise scope present")
    if not (pd.to_numeric(formal["modified_val_count"]) == 0).all():
        raise ValueError("incomplete MLP-37 design: validation labels were modified")
    if not (pd.to_numeric(formal["modified_test_count"]) == 0).all():
        raise ValueError("incomplete MLP-37 design: test labels were modified")

    formal["seed"] = pd.to_numeric(formal["seed"], errors="raise").astype(int)
    formal["h_seed_num"] = _numeric_h_seed(formal)
    for dataset in DATASETS:
        subset = formal[formal["dataset"] == dataset]
        for method in (LCSS, SHUFFLED):
            method_rows = subset[subset["method"] == method]
            cells = set(zip(method_rows["h_seed_num"].astype(int), method_rows["seed"]))
            expected = {(h_seed, seed) for h_seed in H_SEEDS for seed in DOWNSTREAM_SEEDS}
            if cells != expected or len(method_rows) != 9:
                raise ValueError(f"incomplete {dataset}/{method} 3x3 grid")
        for method in (CE, JOCOR):
            method_rows = subset[subset["method"] == method]
            if set(method_rows["seed"]) != set(DOWNSTREAM_SEEDS) or len(method_rows) != 3:
                raise ValueError(f"incomplete {dataset}/{method} reference grid")

        for seed in DOWNSTREAM_SEEDS:
            paired = subset[subset["seed"] == seed]
            if paired["split_hash"].nunique() != 1 or paired["noise_hash"].nunique() != 1:
                raise ValueError(f"hash mismatch for {dataset} downstream seed {seed}")


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def summarize_variance_and_gates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_raw(frame)
    work = frame.copy()
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    work["h_seed_num"] = _numeric_h_seed(work)
    work["test"] = pd.to_numeric(work["test"])

    variance_rows = []
    gate_rows = []
    for dataset in DATASETS:
        subset = work[work["dataset"] == dataset]
        ce_by_seed = subset[subset["method"] == CE].set_index("seed")["test"]
        jocor_by_seed = subset[subset["method"] == JOCOR].set_index("seed")["test"]
        h_means = []
        within_stds = []
        delta_shuffled = []
        delta_ce = []
        delta_jocor = []

        for h_seed in H_SEEDS:
            lcss = (
                subset[(subset["method"] == LCSS) & (subset["h_seed_num"] == h_seed)]
                .set_index("seed")["test"]
                .sort_index()
            )
            shuffled = (
                subset[(subset["method"] == SHUFFLED) & (subset["h_seed_num"] == h_seed)]
                .set_index("seed")["test"]
                .sort_index()
            )
            h_means.append(float(lcss.mean()))
            within_stds.append(float(lcss.std(ddof=1)))
            delta_shuffled.append(float((lcss - shuffled).mean()))
            delta_ce.append(float((lcss - ce_by_seed).mean()))
            delta_jocor.append(float((lcss - jocor_by_seed).mean()))

        h_std = float(np.std(h_means, ddof=1))
        downstream_rms_std = _rms(within_stds)
        dispersion_limit = max(0.0025, 2.0 * downstream_rms_std)
        shuffled_pass = all(value >= 0.01 for value in delta_shuffled)
        ce_pass = all(value > 0.0 for value in delta_ce)
        dispersion_pass = h_std <= dispersion_limit
        strong_pass = sum(value > 0.0 for value in delta_jocor) >= 2 and min(delta_jocor) >= -0.005
        primary_pass = shuffled_pass and ce_pass and dispersion_pass

        variance_rows.append(
            {
                "dataset": dataset,
                "h_seed_mean_0": h_means[0],
                "h_seed_mean_1": h_means[1],
                "h_seed_mean_2": h_means[2],
                "h_seed_std": h_std,
                "downstream_rms_std": downstream_rms_std,
                "dispersion_limit": dispersion_limit,
                "delta_shuffled_min": min(delta_shuffled),
                "delta_ce_min": min(delta_ce),
                "delta_jocor_min": min(delta_jocor),
                "delta_jocor_positive_hseeds": sum(value > 0.0 for value in delta_jocor),
            }
        )
        gate_rows.append(
            {
                "dataset": dataset,
                "shuffled_gate": shuffled_pass,
                "ce_gate": ce_pass,
                "dispersion_gate": dispersion_pass,
                "primary_pass": primary_pass,
                "strong_baseline_pass": strong_pass,
            }
        )

    gates = pd.DataFrame(gate_rows)
    gates["all_datasets_primary_pass"] = bool(gates["primary_pass"].all())
    return pd.DataFrame(variance_rows), gates


def aggregate_h_seed_results(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["test"] = pd.to_numeric(work["test"])
    work["h_seed_num"] = _numeric_h_seed(work)
    rows = []
    for dataset in DATASETS:
        subset = work[work["dataset"] == dataset]
        for method in (LCSS, SHUFFLED):
            for h_seed in H_SEEDS:
                values = subset[(subset["method"] == method) & (subset["h_seed_num"] == h_seed)]["test"]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "h_seed": h_seed,
                        "n": int(len(values)),
                        "test_mean": float(values.mean()),
                        "test_std": float(values.std(ddof=1)),
                    }
                )
        for method in (CE, JOCOR):
            values = subset[subset["method"] == method]["test"]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "h_seed": "",
                    "n": int(len(values)),
                    "test_mean": float(values.mean()),
                    "test_std": float(values.std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["test"] = pd.to_numeric(work["test"])
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    work["h_seed_num"] = _numeric_h_seed(work)
    rows = []
    comparators = ((SHUFFLED, "lcss_vs_shuffled"), (CE, "lcss_vs_ce"), (JOCOR, "lcss_vs_jocor"))
    for dataset in DATASETS:
        subset = work[work["dataset"] == dataset]
        for h_seed in H_SEEDS:
            lcss = (
                subset[(subset["method"] == LCSS) & (subset["h_seed_num"] == h_seed)]
                .set_index("seed")["test"]
                .sort_index()
            )
            for method, name in comparators:
                if method in (CE, JOCOR):
                    other = subset[subset["method"] == method].set_index("seed")["test"].sort_index()
                else:
                    other = (
                        subset[(subset["method"] == method) & (subset["h_seed_num"] == h_seed)]
                        .set_index("seed")["test"]
                        .sort_index()
                    )
                delta = (lcss - other).to_numpy(dtype=float)
                mean = float(delta.mean())
                std = float(delta.std(ddof=1))
                half_width = 4.302652729911275 * std / math.sqrt(len(delta))
                rows.append(
                    {
                        "dataset": dataset,
                        "h_seed": h_seed,
                        "comparison": name,
                        "n": int(len(delta)),
                        "mean_delta": mean,
                        "std_delta": std,
                        "ci95_low": mean - half_width,
                        "ci95_high": mean + half_width,
                        "positive_seeds": int((delta > 0).sum()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_assignment_agreement(geometry: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "seed", "h_seed", "assignment_path"}
    missing = required.difference(geometry.columns)
    if missing:
        raise ValueError(f"incomplete geometry data: missing columns {sorted(missing)}")
    work = geometry.copy()
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    work["h_seed"] = pd.to_numeric(work["h_seed"]).astype(int)
    rows = []
    for dataset in DATASETS:
        for seed in DOWNSTREAM_SEEDS:
            subset = work[(work["dataset"] == dataset) & (work["seed"] == seed)]
            if set(subset["h_seed"]) != set(H_SEEDS) or len(subset) != 3:
                raise ValueError(f"incomplete geometry 3-seed cell for {dataset}/dseed{seed}")
            assignments = {
                int(row.h_seed): np.load(Path(row.assignment_path))
                for row in subset.itertuples(index=False)
            }
            for first, second in ((0, 1), (0, 2), (1, 2)):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "h_seed_pair": f"{first}-{second}",
                        "adjusted_rand_index": assignment_agreement(assignments[first], assignments[second]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pretrain_costs(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "h_seed",
        "wall_time_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "embedding_size_bytes",
        "checkpoint_size_bytes",
        "embedding_sha256",
        "checkpoint_sha256",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"incomplete pretrain cost manifest: missing columns {sorted(missing)}")
    result = manifest.copy()
    result["wall_time_minutes"] = pd.to_numeric(result["wall_time_seconds"]) / 60.0
    result["peak_memory_allocated_gib"] = pd.to_numeric(result["peak_memory_allocated_bytes"]) / (1024 ** 3)
    result["peak_memory_reserved_gib"] = pd.to_numeric(result["peak_memory_reserved_bytes"]) / (1024 ** 3)
    result["embedding_size_mib"] = pd.to_numeric(result["embedding_size_bytes"]) / (1024 ** 2)
    result["checkpoint_size_mib"] = pd.to_numeric(result["checkpoint_size_bytes"]) / (1024 ** 2)
    return result


def build_report(
    raw: pd.DataFrame,
    variance: pd.DataFrame,
    gates: pd.DataFrame,
    agreements: pd.DataFrame,
    costs: pd.DataFrame,
) -> str:
    all_primary = bool(gates["primary_pass"].all())
    verdict = "PASS" if all_primary else "FAIL"
    lines = [
        "# MLP-37 GRACE 预训练种子分层方差报告",
        "",
        f"## 预注册终判：{verdict}",
        "",
        "本实验属于参数冻结后的条件评测：LCSS、beta、球构造参数、噪声协议与学生模型均未调整。",
        "",
        "| 数据集 | H-seed SD | 下游 RMS SD | LCSS-shuffled 最小增量 | LCSS-CE 最小增量 | 主门 | 强对照门 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    merged = variance.merge(gates, on="dataset", how="left")
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {100 * row.h_seed_std:.2f}pp | {100 * row.downstream_rms_std:.2f}pp | "
            f"{100 * row.delta_shuffled_min:.2f}pp | {100 * row.delta_ce_min:.2f}pp | "
            f"{'PASS' if row.primary_pass else 'FAIL'} | "
            f"{'PASS' if row.strong_baseline_pass else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 主门通过时，只能写核心增量在三个预训练种子下方向稳定；不能把 n=3 的结果表述为完整刻画了所有预训练不确定性。",
            "- 主门未通过时，不调参、不删数据；摘要和 Highlights 必须把 fixed-H 条件提升为醒目限制。",
            "- 球数、target quality 与跨 H 分区 ARI 是描述量，不参与通过判定。",
        ]
    )
    if not agreements.empty:
        lines.extend(
            [
                "",
                "## 分区稳定性（描述量）",
                "",
                f"跨 H-seed 分区 ARI 的均值为 {agreements['adjusted_rand_index'].mean():.3f}，"
                f"范围为 {agreements['adjusted_rand_index'].min():.3f}--{agreements['adjusted_rand_index'].max():.3f}。",
            ]
        )
    if not costs.empty:
        lines.extend(
            [
                "",
                "## GRACE 预训练成本",
                "",
                "| 数据集 | seed 数 | 平均分钟 | 峰值 allocated GiB | checkpoint MiB |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset, group in costs.groupby(costs["dataset"].astype(str).str.lower()):
            lines.append(
                f"| {dataset} | {len(group)} | {group['wall_time_minutes'].mean():.2f} | "
                f"{group['peak_memory_allocated_gib'].max():.2f} | {group['checkpoint_size_mib'].mean():.2f} |"
            )
    lines.extend(
        [
            "",
            "## 完整性",
            "",
            f"- 下游正式行：{len(raw)}（预期 72）。",
            f"- 全数据集主门：{'通过' if all_primary else '未通过'}。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the preregistered MLP-37 3x3 GRACE seed experiment.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--pretrain-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.raw)
    geometry = pd.read_csv(args.geometry)
    manifest = pd.read_csv(args.pretrain_manifest)
    validate_raw(raw)
    variance, gates = summarize_variance_and_gates(raw)
    aggregate = aggregate_h_seed_results(raw)
    deltas = paired_deltas(raw)
    agreements = summarize_assignment_agreement(geometry)
    costs = summarize_pretrain_costs(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "mlp37_hseed_aggregate.csv", index=False)
    deltas.to_csv(args.output_dir / "mlp37_paired_deltas.csv", index=False)
    variance.to_csv(args.output_dir / "mlp37_variance.csv", index=False)
    agreements.to_csv(args.output_dir / "mlp37_assignment_agreement.csv", index=False)
    costs.to_csv(args.output_dir / "mlp37_grace_costs.csv", index=False)
    gates.to_csv(args.output_dir / "mlp37_gate_decisions.csv", index=False)
    report_path = args.output_dir / "mlp37_report.md"
    report_path.write_text(build_report(raw, variance, gates, agreements, costs), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
