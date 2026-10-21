#!/usr/bin/env python3
"""Recompute one Holm family across the 75 frozen paired-test p-values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path


EXPECTED_HOLM_COLUMNS = [
    "dataset",
    "noise_type",
    "noise_rate",
    "comparison",
    "n",
    "mean_delta",
    "std_delta",
    "ci95_low",
    "ci95_high",
    "positive_seeds",
    "negative_seeds",
    "paired_t_p",
    "holm_p",
]
EXPECTED_SEED_COLUMNS = [
    "dataset",
    "noise_type",
    "noise_rate",
    "seed",
    "method",
    "test",
]
TEST_KEY_COLUMNS = ("dataset", "noise_type", "noise_rate", "comparison")
SEED_KEY_COLUMNS = ("dataset", "noise_type", "noise_rate", "seed", "method")
COMPARISONS = ("lcss_vs_ce", "lcss_vs_jocor", "lcss_vs_shuffled")
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply one Holm step-down adjustment to 75 frozen p-values."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--seed-input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if columns != expected_columns:
            raise ValueError(
                f"Unexpected columns in {path}: {columns}; expected {expected_columns}"
            )
        return list(reader)


def holm_adjust(
    rows: list[dict[str, str]], p_column: str
) -> dict[tuple[str, ...], dict[str, float | int]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[p_column]),
            *(row[column] for column in TEST_KEY_COLUMNS),
        ),
    )
    total = len(ordered)
    running_max = 0.0
    result: dict[tuple[str, ...], dict[str, float | int]] = {}
    for zero_based_rank, row in enumerate(ordered):
        raw_p = float(row[p_column])
        factor = total - zero_based_rank
        scaled = min(1.0, factor * raw_p)
        running_max = max(running_max, scaled)
        key = tuple(row[column] for column in TEST_KEY_COLUMNS)
        result[key] = {
            "rank": zero_based_rank + 1,
            "factor": factor,
            "scaled": scaled,
            "adjusted": min(1.0, running_max),
        }
    return result


def validate_seed_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    if len(rows) != 500:
        raise ValueError(f"Expected 500 seed-level rows, found {len(rows)}")
    keys = [tuple(row[column] for column in SEED_KEY_COLUMNS) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate seed-level key detected")
    settings = {
        (row["dataset"], row["noise_type"], row["noise_rate"]) for row in rows
    }
    if len(settings) != 25:
        raise ValueError(f"Expected 25 settings, found {len(settings)}")
    if {row["method"] for row in rows} != {"ce", "jocor", "lcss", "shuffled"}:
        raise ValueError("Unexpected method set in seed-level input")
    if {row["seed"] for row in rows} != {"0", "1", "2", "3", "4"}:
        raise ValueError("Unexpected seed set in seed-level input")
    setting_counts = Counter(
        (row["dataset"], row["noise_type"], row["noise_rate"]) for row in rows
    )
    if set(setting_counts.values()) != {20}:
        raise ValueError("Every setting must contain 4 methods x 5 seeds")
    return {"rows": len(rows), "settings": len(settings), "unique_keys": len(keys)}


def validate_holm_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    if len(rows) != 75:
        raise ValueError(f"Expected 75 tests, found {len(rows)}")
    keys = [tuple(row[column] for column in TEST_KEY_COLUMNS) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate four-column test key detected")
    counts = Counter(row["comparison"] for row in rows)
    if counts != Counter({comparison: 25 for comparison in COMPARISONS}):
        raise ValueError(f"Expected three 25-test comparison families, found {counts}")
    if {row["n"] for row in rows} != {"5"}:
        raise ValueError("Expected n=5 for every paired test")

    for row in rows:
        for column in ("paired_t_p", "holm_p"):
            value = float(row[column])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"Invalid probability in {column}: {row[column]}")

    max_family_difference = 0.0
    for comparison in COMPARISONS:
        family = [row for row in rows if row["comparison"] == comparison]
        recalculated = holm_adjust(family, "paired_t_p")
        for row in family:
            key = tuple(row[column] for column in TEST_KEY_COLUMNS)
            difference = abs(float(row["holm_p"]) - float(recalculated[key]["adjusted"]))
            max_family_difference = max(max_family_difference, difference)
    if max_family_difference > 1e-12:
        raise ValueError(
            "Existing 25-test Holm values do not match step-down recalculation; "
            f"max absolute difference={max_family_difference}"
        )
    return {
        "rows": len(rows),
        "unique_keys": len(keys),
        "comparison_counts": dict(sorted(counts.items())),
        "max_family_difference": max_family_difference,
    }


def format_float(value: float) -> str:
    return format(value, ".17g")


def write_output(
    rows: list[dict[str, str]],
    adjustments: dict[tuple[str, ...], dict[str, float | int]],
    output_path: Path,
) -> list[dict[str, str]]:
    output_columns = [
        *EXPECTED_HOLM_COLUMNS[:-1],
        "holm_p_25_family",
        "holm_rank_75",
        "holm_factor_75",
        "holm_scaled_p_75",
        "holm_p_75_family",
        "significant_25_family_alpha_0_05",
        "significant_75_family_alpha_0_05",
        "significance_changed_25_to_75",
    ]
    output_rows = []
    for source_row in rows:
        key = tuple(source_row[column] for column in TEST_KEY_COLUMNS)
        adjustment = adjustments[key]
        old_significant = float(source_row["holm_p"]) < ALPHA
        new_significant = float(adjustment["adjusted"]) < ALPHA
        output_row = {
            column: source_row[column] for column in EXPECTED_HOLM_COLUMNS[:-1]
        }
        output_row.update(
            {
                "holm_p_25_family": source_row["holm_p"],
                "holm_rank_75": str(adjustment["rank"]),
                "holm_factor_75": str(adjustment["factor"]),
                "holm_scaled_p_75": format_float(float(adjustment["scaled"])),
                "holm_p_75_family": format_float(float(adjustment["adjusted"])),
                "significant_25_family_alpha_0_05": str(int(old_significant)),
                "significant_75_family_alpha_0_05": str(int(new_significant)),
                "significance_changed_25_to_75": str(
                    int(old_significant != new_significant)
                ),
            }
        )
        output_rows.append(output_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def write_report(
    source_path: Path,
    seed_path: Path | None,
    output_path: Path,
    report_path: Path,
    source_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
    validation: dict[str, object],
    seed_validation: dict[str, int] | None,
) -> None:
    old_counts = Counter(
        row["comparison"]
        for row in output_rows
        if row["significant_25_family_alpha_0_05"] == "1"
    )
    new_counts = Counter(
        row["comparison"]
        for row in output_rows
        if row["significant_75_family_alpha_0_05"] == "1"
    )
    changed = [
        row for row in output_rows if row["significance_changed_25_to_75"] == "1"
    ]
    ordered = sorted(output_rows, key=lambda row: int(row["holm_rank_75"]))
    monotonic_violations = sum(
        float(ordered[index]["holm_p_75_family"])
        < float(ordered[index - 1]["holm_p_75_family"])
        for index in range(1, len(ordered))
    )
    adjusted_below_raw = sum(
        float(row["holm_p_75_family"]) < float(row["paired_t_p"])
        for row in output_rows
    )
    global_below_family = sum(
        float(row["holm_p_75_family"]) < float(row["holm_p_25_family"]) - 1e-15
        for row in output_rows
    )

    comparison_table = "\n".join(
        (
            f"| `{comparison}` | {old_counts[comparison]} | "
            f"{new_counts[comparison]} | "
            f"{old_counts[comparison] - new_counts[comparison]} |"
        )
        for comparison in COMPARISONS
    )
    changed_table = "\n".join(
        (
            f"| `{row['dataset']}` | `{row['noise_type']}` | "
            f"`{row['noise_rate']}` | `{row['comparison']}` | "
            f"{float(row['paired_t_p']):.9g} | "
            f"{float(row['holm_p_25_family']):.9g} | "
            f"{float(row['holm_p_75_family']):.9g} |"
        )
        for row in sorted(
            changed,
            key=lambda row: (
                float(row["paired_t_p"]),
                *(row[column] for column in TEST_KEY_COLUMNS),
            ),
        )
    )
    if not changed_table:
        changed_table = "| 无 | 无 | 无 | 无 | 无 | 无 | 无 |"

    seed_section = (
        f"- seed-level 输入：`{seed_path}`；SHA-256 "
        f"`{sha256(seed_path)}`；{seed_validation['rows']} 行、"
        f"{seed_validation['settings']} 个 setting、"
        f"{seed_validation['unique_keys']} 个唯一五列键。\n"
        if seed_path is not None and seed_validation is not None
        else "- seed-level 输入：未提供；本次只校验 75 行统计表。\n"
    )
    report = rf"""# 75 个 p 值的单一 Holm 敏感性重算报告

## 范围与输入

本分析严格限于已有配对检验的统计敏感性重算：未训练模型，未改变任何 seed 结果、效应量、置信区间或论文正文，也未修改原始统计表。

- 75-test 输入：`{source_path}`；SHA-256 `{sha256(source_path)}`。
{seed_section}- 输出 CSV：`{output_path}`；SHA-256 `{sha256(output_path)}`。

## 实际列与唯一键

- `primary_statistics_raw.csv` 的实际列为 `dataset, noise_type, noise_rate, seed, method, test`。它提供 25 settings × 5 seeds × 4 methods 的 500 个 seed-level 观测；其唯一键为 `(dataset, noise_type, noise_rate, seed, method)`。该表不含 p 值列。
- `primary_statistics_holm.csv` 的实际列为 `dataset, noise_type, noise_rate, comparison, n, mean_delta, std_delta, ci95_low, ci95_high, positive_seeds, negative_seeds, paired_t_p, holm_p`。
- 本次 75 个检验的唯一键是 `(dataset, noise_type, noise_rate, comparison)`；校验得到 {validation['unique_keys']}/{validation['rows']} 个唯一键，无重复。
- 原始 p 值只从现有列 `paired_t_p` 读取；现有列 `holm_p` 映射为各 comparator 内 25-test Holm 校正值 `holm_p_25_family`。没有从列名、行顺序或效应量猜测 p 值。
- 三个 comparison 分别为 `lcss_vs_ce`、`lcss_vs_jocor`、`lcss_vs_shuffled`，每组各 25 项。

## Holm step-down 公式

令全部检验数 \(m=75\)，将原始 p 值升序排列为
\(p_{{(1)}}\le\cdots\le p_{{(m)}}\)。第 \(i\) 项的未单调化候选值为

\[
s_{{(i)}}=\min\{{1,(m-i+1)p_{{(i)}}\}}.
\]

Holm 调整值采用累计最大值保证单调性：

\[
\tilde p_{{(i)}}=\max_{{1\le j\le i}}s_{{(j)}}.
\]

最后按四列唯一键映射回原行。CSV 中 `holm_rank_75` 是升序位置，
`holm_factor_75=75-i+1`，`holm_scaled_p_75=s_{{(i)}}`，
`holm_p_75_family=\tilde p_{{(i)}}`。并列 p 值仅以四列键作确定性排序，
不改变并列项的最终调整值。

## 数量与校验

- 输入检验数：75；原始 `paired_t_p < 0.05`：{sum(float(row['paired_t_p']) < ALPHA for row in source_rows)}。
- 最小原始 p 值：{min(float(row['paired_t_p']) for row in source_rows):.17g}。
- 现有三个 25-test 家族的 Holm 独立复算最大绝对差：
  {float(validation['max_family_difference']):.3g}。
- 按 `holm_rank_75` 检查的调整值单调性违反数：{monotonic_violations}。
- `holm_p_75_family < paired_t_p` 的违反数：{adjusted_below_raw}。
- `holm_p_75_family < holm_p_25_family` 的违反数：{global_below_family}。

| comparison | 25-test 家族显著数 | 单一 75-test 家族显著数 | 减少 |
|---|---:|---:|---:|
{comparison_table}
| **合计** | **{sum(old_counts.values())}** | **{sum(new_counts.values())}** | **{sum(old_counts.values()) - sum(new_counts.values())}** |

在 \(\alpha=0.05\) 下，单一 75-test 家族保留 {sum(new_counts.values())}/75 项，
而三个分开的 25-test 家族合计为 {sum(old_counts.values())}/75 项；有
{len(changed)} 项从“族内显著”变为“75-test 不显著”，没有反向变化。

## 显著性状态改变的项目

| dataset | noise_type | noise_rate | comparison | raw p | Holm-25 p | Holm-75 p |
|---|---|---:|---|---:|---:|---:|
{changed_table}

## 结论边界

### 已被本次重算支持

1. 对这 75 个冻结的 `paired_t_p` 一次性控制 Holm FWER 后，显著项数为
   {sum(new_counts.values())}/75；按 CE、JoCoR、shuffled 分别为
   {new_counts['lcss_vs_ce']}/{new_counts['lcss_vs_jocor']}/{new_counts['lcss_vs_shuffled']}。
2. 现有三个作者定义的 25-test 家族算术可复算，显著项数为
   {old_counts['lcss_vs_ce']}/{old_counts['lcss_vs_jocor']}/{old_counts['lcss_vs_shuffled']}；
   单一 75-test 敏感性分析更保守，减少 {len(changed)} 项。
3. 本次调整不改变 `mean_delta`、`std_delta`、95% CI、正/负方向 seed 计数或
   任一原始 p 值。

### 仅为合理解释

显著项减少来自检验族从 25 扩大到 75 后阈值变严；它说明统计结论对 family
定义敏感，但不说明效应量本身变小，也不构成方法失效的证据。

### 尚未由本次分析验证

本次没有检验 n=5 配对差的正态性、异常值稳健性，也没有建立比较族在观察结果前
已外部预注册的时间证据。75-test Holm 只回答给定 75 个 p 值下的 FWER
敏感性，不替代效应量、95% CI、方向计数或稳健统计分析。

### 已被敏感性结果削弱的表述

不能把三个分别校正的 25-test 家族结果表述为“全论文 75 个确认性检验的统一
FWER 保证”，也不能用 24/8/25 概括单一 75-test 家族；对应数字应明确标注
family 定义。

## 最终决策

保留三个 comparator-specific 25-test 家族作为作者定义、非外部预注册的原报告，
同时增加本文件的单一 75-test Holm 作为敏感性分析。论文层面的显著性措辞应限定到
具体 family 与 setting；本次不产生新的模型性能 claim。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    source_rows = read_csv(args.input, EXPECTED_HOLM_COLUMNS)
    validation = validate_holm_rows(source_rows)
    seed_validation = None
    if args.seed_input is not None:
        seed_rows = read_csv(args.seed_input, EXPECTED_SEED_COLUMNS)
        seed_validation = validate_seed_rows(seed_rows)
    adjustments = holm_adjust(source_rows, "paired_t_p")
    output_rows = write_output(source_rows, adjustments, args.output)
    write_report(
        args.input,
        args.seed_input,
        args.output,
        args.report,
        source_rows,
        output_rows,
        validation,
        seed_validation,
    )
    print(
        f"Wrote {len(output_rows)} rows; "
        f"Holm-75 significant={sum(row['significant_75_family_alpha_0_05'] == '1' for row in output_rows)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
