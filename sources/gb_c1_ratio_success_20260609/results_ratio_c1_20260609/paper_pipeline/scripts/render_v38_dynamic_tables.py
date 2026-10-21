from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper_final_zh_v38_loo_main_20260730.md"
SUMMARY = ROOT / "paper_pipeline" / "00_evidence" / "v38_loo_summary"
NOISYGL = ROOT.parents[1] / "external" / "NoisyGL"

DATASETS = (
    ("cora", "Cora"),
    ("citeseer", "CiteSeer"),
    ("corafull", "CoraFull"),
    ("coauthorcs", "Coauthor-CS"),
    ("amazonpho", "Amazon-Photo"),
)
SETTINGS = (
    ("uniform", 0.2, "Uniform 0.2"),
    ("uniform", 0.4, "Uniform 0.4"),
    ("uniform", 0.6, "Uniform 0.6"),
    ("pair", 0.4, "Pair 0.4"),
    ("instance", 0.4, "Instance 0.4"),
)
LOO = "frozengrace_gb_mlp_softdist_beta05_loo"
COMPARATORS = (
    (
        "frozengrace_gb_mlp_softdist_beta05_loo_shuffled",
        "GB-LCSS−留一随机分组",
    ),
    ("mlp_ce", "GB-LCSS−MLP+CE"),
    ("mlp_jocor", "GB-LCSS−JoCoR"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def replace_section(
    text: str, heading: str, next_heading: str, content: str
) -> str:
    pattern = re.compile(
        rf"{re.escape(heading)}\n.*?(?=\n{re.escape(next_heading)}\n)",
        re.DOTALL,
    )
    updated, count = pattern.subn(content.rstrip() + "\n", text)
    if count != 1:
        raise ValueError(f"expected one section {heading!r}, replaced {count}")
    return updated


def render_appendix_b() -> str:
    rows = read_csv(SUMMARY / "paired_tests.csv")
    lookup = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["comparator"],
        ): row
        for row in rows
    }
    expected = len(DATASETS) * len(SETTINGS) * len(COMPARATORS)
    if len(
        [
            key
            for key in lookup
            if key[3] in {comparator for comparator, _ in COMPARATORS}
        ]
    ) != expected:
        raise ValueError("primary paired table is incomplete")

    lines = [
        "## 附录 B 配对统计",
        "",
        "均值差与 95% 置信区间以百分点表示。Holm 校正在三个独立的 "
        "25 设置比较族内进行。",
        "",
        "| 数据集 | 噪声 | 比较 | 均值差 | 95% CI | 正向 seeds | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, dataset_label in DATASETS:
        for noise_type, noise_rate, setting_label in SETTINGS:
            for comparator, comparison_label in COMPARATORS:
                row = lookup[(dataset, noise_type, noise_rate, comparator)]
                lines.append(
                    f"| {dataset_label} | {setting_label} | "
                    f"{comparison_label} | "
                    f"{100*float(row['mean_delta']):+.2f} | "
                    f"[{100*float(row['ci95_low']):+.2f}, "
                    f"{100*float(row['ci95_high']):+.2f}] | "
                    f"{int(float(row['positive_seeds']))}/5 | "
                    f"{float(row['holm_adjusted_p']):.4g} |"
                )
    return "\n".join(lines) + "\n"


def render_appendix_h() -> str:
    loo_rows = read_csv(SUMMARY / "formal_aggregate.csv")
    loo_lookup = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
        ): row
        for row in loo_rows
    }
    recent = read_csv(
        NOISYGL
        / "runs"
        / "nearest_work_full_grid_20260728"
        / "five_method_aggregate.csv"
    )
    recent_lookup = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
        ): row
        for row in recent
    }
    recent_methods = (
        ("postel_mlp_noipl_rm", "PosteL"),
        ("als_mlp_rm", "ALS"),
        ("sals_mlp_adapted", "SALS"),
    )
    lines = [
        "## 附录 H 结构辅助软目标方法的完整噪声谱",
        "",
        "表 H.1 给出正式留一 GB-LCSS、PosteL、ALS 和 SALS 在五个主"
        "数据集、五种噪声设置下的五种子均值与样本标准差。四种方法共享"
        "数据划分、训练标签噪声实例和 X-only 推理约束。",
        "",
        "| 数据集 | 噪声设置 | GB-LCSS | PosteL | ALS | SALS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, dataset_label in DATASETS:
        for noise_type, noise_rate, setting_label in SETTINGS:
            ours = loo_lookup[(dataset, noise_type, noise_rate, LOO)]
            values = [
                f"{100*float(ours['mean']):.2f}±{100*float(ours['std']):.2f}"
            ]
            for method, _ in recent_methods:
                row = recent_lookup[(dataset, noise_type, noise_rate, method)]
                mean_key = "test_mean" if "test_mean" in row else "mean"
                std_key = (
                    "test_sample_std"
                    if "test_sample_std" in row
                    else ("test_std" if "test_std" in row else "std")
                )
                values.append(
                    f"{100*float(row[mean_key]):.2f}±"
                    f"{100*float(row[std_key]):.2f}"
                )
            lines.append(
                f"| {dataset_label} | {setting_label} | "
                + " | ".join(values)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    text = replace_section(
        text,
        "## 附录 B 配对统计",
        "## 附录 C 验证标签噪声敏感性",
        render_appendix_b(),
    )
    text = replace_section(
        text,
        "## 附录 H 结构辅助软目标方法的完整噪声谱",
        "## 附录 I 关键实现参数",
        render_appendix_h(),
    )
    MANUSCRIPT.write_text(text, encoding="utf-8")
    print(MANUSCRIPT)


if __name__ == "__main__":
    main()
