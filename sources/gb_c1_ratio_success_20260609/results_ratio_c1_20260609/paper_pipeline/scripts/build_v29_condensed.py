from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper_final_zh_v29_human_style_20260727.md"
OUTPUT = ROOT / "paper_final_zh_v29-condensed.md"
ASSETS = (
    ROOT.parents[1]
    / "external"
    / "NoisyGL"
    / "paper_assets_20260706"
)
RUNS = ROOT.parents[1] / "external" / "NoisyGL" / "runs"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    start_pos = text.index(start)
    end_pos = text.index(end, start_pos)
    return text[:start_pos] + replacement.rstrip() + "\n\n" + text[end_pos:]


def filter_table_lines(text: str, forbidden: tuple[str, ...]) -> str:
    kept = []
    for line in text.splitlines():
        lowered = line.lower()
        if line.startswith("|") and any(token.lower() in lowered for token in forbidden):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def appendix_content() -> str:
    tables = ASSETS / "tables"

    appendix_a = (tables / "appendix_A_complete_main_table.md").read_text(
        encoding="utf-8"
    )
    appendix_a = appendix_a.replace(
        "# Appendix A. Complete main table",
        "## 附录 A 完整主结果表",
    )
    appendix_a = appendix_a.replace(
        "Values are 5-seed test accuracy mean±std (%). Bold marks the best value in each dataset/noise column. `*` marks LCSS vs the strongest non-LCSS non-shuffled baseline with paired t-test p<0.05.",
        "结果为 5 个配对种子的测试精度均值±样本标准差（%）。粗体表示每个数据集与噪声设置下的最高均值，配对统计检验见附录 B。",
    )
    # Appendix B contains the multiplicity-corrected inference. Remove the old
    # uncorrected significance markers from Appendix A to avoid mixed criteria.
    appendix_a = appendix_a.replace("*** |", "** |")
    appendix_a = appendix_a.replace("| LCSS |", "| GB-LCSS |")
    appendix_a = appendix_a.replace("## Cora\n", "### A.1 Cora\n")
    appendix_a = appendix_a.replace("## CiteSeer\n", "### A.2 CiteSeer\n")
    appendix_a = appendix_a.replace("## PubMed\n", "### A.3 PubMed\n")
    appendix_a = appendix_a.replace("## Coauthor-CS\n", "### A.4 Coauthor-CS\n")
    appendix_a = appendix_a.replace("## Amazon-Photo\n", "### A.5 Amazon-Photo\n")
    corafull = """### A.3 CoraFull

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
|---|---:|---:|---:|---:|---:|
| MLP+CE | 41.11±0.46 | 31.98±0.98 | 21.16±0.81 | 28.14±1.17 | 31.33±0.85 |
| Small-loss | 41.58±0.94 | 33.04±0.50 | 21.51±0.98 | 30.15±1.11 | 32.90±0.87 |
| Co-teaching | 41.70±1.05 | 34.39±0.24 | 22.84±1.38 | 30.28±0.52 | 33.39±0.90 |
| GMM+CT | 42.05±0.74 | 34.11±0.24 | 24.01±0.89 | 30.83±1.23 | 33.91±0.59 |
| JoCoR | 43.33±0.70 | 35.70±0.32 | 24.20±0.98 | 31.45±0.80 | 35.17±0.51 |
| ELR | 41.08±0.53 | 32.14±0.67 | 21.32±0.98 | 28.91±1.44 | 31.52±0.83 |
| SCE | 37.93±0.72 | 29.29±0.45 | 17.79±1.33 | 26.98±0.51 | 29.68±1.06 |
| LogitClip | 41.24±0.78 | 34.69±1.17 | 24.62±0.72 | 31.23±1.84 | 34.19±0.17 |
| ANL-CE | 39.45±0.52 | 28.12±0.61 | 16.81±1.28 | 28.55±0.72 | 28.84±1.09 |
| FINE | 40.86±0.73 | 32.00±0.88 | 21.12±0.89 | 29.26±0.79 | 30.80±0.50 |
| DivideMix-lite | 42.02±0.87 | 32.01±0.80 | 21.16±0.81 | 30.00±0.72 | 31.51±0.86 |
| 随机化分组 | 40.84±1.20 | 32.27±0.81 | 21.40±0.69 | 28.43±1.41 | 31.19±1.21 |
| GB-LCSS | **43.51±0.41** | **37.33±0.81** | **27.32±0.75** | **34.41±1.30** | **37.08±0.69** |
"""
    appendix_a = replace_between(
        appendix_a,
        "### A.3 PubMed",
        "### A.4 Coauthor-CS",
        corafull,
    )

    stats_path = RUNS / "mlp36_review_defense_20260715" / "primary_statistics_holm.csv"
    import csv

    with stats_path.open("r", encoding="utf-8-sig", newline="") as handle:
        stat_rows = list(csv.DictReader(handle))
    dataset_order = {
        "cora": 0,
        "citeseer": 1,
        "corafull": 2,
        "coauthorcs": 3,
        "amazonpho": 4,
    }
    noise_order = {
        ("uniform", "0.2"): 0,
        ("uniform", "0.4"): 1,
        ("uniform", "0.6"): 2,
        ("pair", "0.4"): 3,
        ("instance", "0.4"): 4,
    }
    comparison_order = {
        "lcss_vs_ce": 0,
        "lcss_vs_shuffled": 1,
        "lcss_vs_jocor": 2,
    }
    stat_rows.sort(
        key=lambda row: (
            dataset_order[row["dataset"]],
            noise_order[(row["noise_type"], row["noise_rate"])],
            comparison_order[row["comparison"]],
        )
    )
    dataset_labels = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "corafull": "CoraFull",
        "coauthorcs": "Coauthor-CS",
        "amazonpho": "Amazon-Photo",
    }
    comparison_labels = {
        "lcss_vs_ce": "GB-LCSS−MLP+CE",
        "lcss_vs_shuffled": "GB-LCSS−随机化分组",
        "lcss_vs_jocor": "GB-LCSS−JoCoR",
    }
    stat_lines = [
        "## 附录 B 配对统计检验",
        "",
        "本表给出预声明的三个 25-setting 比较族。均值差和 95% 置信区间以百分点表示，Holm 校正在每个比较族的 25 个设置内进行。",
        "",
        "| 数据集 | 噪声设置 | 比较 | 均值差 | 95% CI | 正向种子 | 配对 t 检验 p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stat_rows:
        noise = (
            f"{row['noise_type']} {float(row['noise_rate']):.1f}"
        )
        mean_delta = 100 * float(row["mean_delta"])
        ci_low = 100 * float(row["ci95_low"])
        ci_high = 100 * float(row["ci95_high"])
        stat_lines.append(
            "| {dataset} | {noise} | {comparison} | {delta:+.2f} | "
            "[{low:+.2f}, {high:+.2f}] | {positive}/5 | {p:.4g} | {holm:.4g} |".format(
                dataset=dataset_labels[row["dataset"]],
                noise=noise,
                comparison=comparison_labels[row["comparison"]],
                delta=mean_delta,
                low=ci_low,
                high=ci_high,
                positive=row["positive_seeds"],
                p=float(row["paired_t_p"]),
                holm=float(row["holm_p"]),
            )
        )
    appendix_b = "\n".join(stat_lines)

    appendix_c = """## 附录 C 图依赖参考方法

本附录列出统一 train-only 标签噪声协议下的图依赖参考结果。此类方法在推理阶段仍需要图结构，因而用于衡量 X-only 学生与图模型之间的精度距离。

| 方法 | 推理需图 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|:---:|---:|---:|---:|---:|---:|
| GCN(X) | 是 | 78.97±1.63 | 64.53±1.54 | 61.28±0.52 | 92.06±0.34 | 90.77±0.43 |
| NRGNN | 是 | **81.76±1.28** | **71.03±1.01** | **62.70±0.63** | 93.56±0.11 | 89.02±1.08 |
| TFR-GCN（统一协议适配） | 是 | 78.75±0.11 | 68.92±1.77 | 42.12±1.14 | 78.36±2.39 | 35.36±9.43 |
| ProCon（统一协议适配） | 是 | 77.19±2.17 | 61.07±3.32 | 36.42±3.48 | 86.08±6.85 | 20.74±4.82 |
| GD²（统一协议适配） | 是 | 76.89±3.51 | 63.07±1.58 | 59.88±0.79 | 91.57±0.18 | **91.04±1.00** |
| GB-LCSS 蒸馏扩展（学生） | 否 | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 | **93.75±0.22** | 90.49±1.05 |

TFR-GCN、ProCon 与 GD² 为统一协议下的本地适配或重实现结果。
"""

    appendix_d1 = (tables / "appendix_D1_parameter_sensitivity.md").read_text(
        encoding="utf-8"
    )
    appendix_d1 = appendix_d1.replace(
        "# Appendix D1. Construction-parameter sensitivity",
        "### 附录 D.1 构造参数敏感性",
    )
    appendix_d1 = appendix_d1.replace(
        "## Stage A seed0 grid",
        "#### 阶段 A：seed 0 候选网格",
    )
    appendix_d1 = appendix_d1.replace(
        "## Stage B 5-seed selected configurations",
        "#### 阶段 B：候选配置的 5 种子复核",
    )
    appendix_d2 = (tables / "appendix_D2_beta_audit.md").read_text(
        encoding="utf-8"
    )
    appendix_d2 = appendix_d2.replace(
        "# Appendix D2. Beta validation audit",
        "### 附录 D.2 β 的验证集选择审计",
    )
    appendix_d2 = appendix_d2.replace(
        "## Stage A seed0 grid",
        "#### 阶段 A：seed 0 候选网格",
    )
    appendix_d2 = appendix_d2.replace(
        "## Stage B 5-seed selected configurations",
        "#### 阶段 B：候选配置的 5 种子复核",
    )
    appendix_d = "## 附录 D 参数敏感性与 β 选择\n\n" + appendix_d1 + "\n\n" + appendix_d2

    appendix_e = """## 附录 E 测试特征噪声曲线

图 E.1 给出 5 种方法在 5 个数据集和 6 个扰动水平下的完整测试特征噪声曲线。零扰动点逐种子锚定正文主表结果。

![图 E.1 测试特征噪声完整曲线](../../external/NoisyGL/paper_assets_20260706/figures/appendix_E_feature_noise_curves.png)
"""

    appendix_f = (tables / "appendix_F_cut_value.md").read_text(encoding="utf-8")
    appendix_f = filter_table_lines(appendix_f, ("knn",))
    appendix_f = appendix_f.replace(
        "# Appendix F. Cut Value table",
        "## 附录 F Cut Value 结构一致性",
    )
    appendix_f = appendix_f.replace(
        "Footnote: high CV for CE/shuffled can be contaminated by prediction collapse under noisy-label protocols. CV is therefore not used as a clean structure metric here; the distill-vs-train-only rows are the only mechanism evidence quoted in the manuscript.",
        "说明：在含噪标签协议下，CE 或随机化分组的高 CV 可能受到预测坍缩影响。因此，本文不将 CV 作为独立的结构质量指标，仅使用蒸馏学生与基础 GB-LCSS 学生的比较作为辅助机制证据。",
    )
    appendix_f = appendix_f.replace(
        "| dataset | method | method_label | n | mean | std |",
        "| 数据集 | 方法标识 | 方法 | n | 均值 | 标准差 |",
    )
    appendix_f = appendix_f.replace(
        "Ours beta=0.5 train-only",
        "GB-LCSS",
    )
    appendix_f = appendix_f.replace(
        "Shuffled local consensus",
        "随机化分组",
    )
    appendix_f = appendix_f.replace(
        "GLNN-style T1",
        "GLNN 式蒸馏 T1",
    )
    appendix_f = appendix_f.replace(
        "Ours-distill T3",
        "GB-LCSS 蒸馏 T3",
    )
    appendix_f = appendix_f.replace(
        "Raw per-seed rows are preserved in `mlp16_cut_value_raw.csv`.",
        "逐种子原始记录保存在 `mlp16_cut_value_raw.csv` 中。",
    )
    appendix_f = appendix_f.replace(
        "mlp_distill_softdist_xh", "gb_lcss_consensus_distill"
    )

    appendix_g = """## 附录 G 表征空间与粒球机制可视化

### 附录 G.1 表征空间三着色图

![图 G.1 真实类别、含噪标签与粒球归属的 t-SNE 可视化](../../external/NoisyGL/paper_assets_20260706/figures/appendix_G_tsne_three_coloring.png)

### 附录 G.2 冻结 GRACE 表征中的粒球覆盖

![图 G.2 冻结 GRACE 表征中的粒球覆盖](../../external/NoisyGL/paper_assets_20260706/figures/appendix_G2_grace_ball_cover.png)

### 附录 G.3 噪声标签的软修正示意

![图 G.3 粒球局部共识对噪声标签的软修正](../../external/NoisyGL/paper_assets_20260706/figures/appendix_G3_noise_soft_correction.png)

上述图用于说明粒球划分与软监督构造过程，定量结论以正文和附录中的配对实验为准。
"""

    appendix_h_table = (
        tables / "appendix_H_labelrate_monotonicity.md"
    ).read_text(encoding="utf-8")
    appendix_h_table = appendix_h_table.replace(
        "# H2 monotonicity check",
        "### 附录 H.1 趋势单调性核验",
    )
    appendix_h_table = appendix_h_table.replace(
        "H2 strict monotonic decrease toward high-label end: cora=True, citeseer=False",
        "严格单调性核验结果：Cora 通过，CiteSeer 未通过。",
    )
    appendix_h = f"""## 附录 H 低标签率实验

图 H.1 给出 Cora 与 CiteSeer 上 6 个标签比例和 5 种方法的完整结果。

![图 H.1 低标签率完整曲线](../../external/NoisyGL/paper_assets_20260706/figures/appendix_H_labelrate_curves.png)

{appendix_h_table}
"""

    graphfree = """### 附录 I.1 无图推理方法族完整表（uniform 0.4）

| 方法 | 训练资源 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---|---:|---:|---:|---:|---:|
| MLP+CE | 含噪标签 | 44.24±2.62 | 43.50±3.02 | 31.98±0.98 | 88.42±0.62 | 73.90±1.80 |
| Graph-MLP（protocol port） | +多跳邻接 NContrast | 64.91±2.04 | 59.87±1.73 | 44.51±0.55 | 89.97±0.39 | 80.44±0.65 |
| GSDN-inspired（mechanism adaptation） | +邻域自蒸馏 | 55.97±4.29 | 50.36±3.39 | 35.80±0.66 | 86.55±0.78 | 73.31±2.00 |
| GB-LCSS | +图（经冻结表征） | 55.59±1.50 | 53.83±2.75 | 37.33±0.81 | 90.62±0.57 | 81.64±0.93 |
| GLNN 式蒸馏 | +GNN 教师+无标签节点 | 77.95±2.62 | 66.09±1.38 | **58.74±0.82** | 93.20±0.09 | 89.86±0.77 |
| FF-G2M（common-backbone reimplementation） | +GNN 教师+LFD/HFD | 77.65±2.97 | 65.80±0.49 | 56.58±0.93 | 91.58±0.24 | 84.82±0.84 |
| KRD-inspired reliability proxy | +GNN 教师+全局可靠性筛选 | 71.77±2.20 | 64.96±1.03 | 47.12±0.95 | 91.39±0.49 | 85.17±0.94 |
| VQGraph-inspired token proxy | +GNN 教师+软码本代理 | 78.89±2.34 | 65.64±1.72 | 58.52±0.77 | 93.25±0.09 | **90.54±0.85** |
| GB-LCSS 蒸馏扩展（球式共识教师） | +球式去噪共识教师+无标签节点 | **80.60±0.69** | **69.40±1.08** | 58.22±0.69 | **93.75±0.22** | 90.49±1.05 |
"""

    distill = (tables / "extension_denoised_teacher_distill.md").read_text(
        encoding="utf-8"
    )
    distill_lines = []
    for line in distill.splitlines():
        lowered = line.lower()
        if line.startswith("|") and (
            "t2 teacher" in lowered
            or "t4 teacher" in lowered
            or "d1_h_input" in lowered
            or "d2_knn" in lowered
        ):
            continue
        line = line.replace(
            "T3 teacher=SoftDist-GCN[X\\|\\|H]",
            "T3 teacher=GB-LCSS consensus GCN",
        )
        line = line.replace(
            "T5 teacher=shuffled-SoftDist-GCN[X\\|\\|H]",
            "T5 teacher=shuffled GB-LCSS consensus GCN",
        )
        distill_lines.append(line)
    distill = "\n".join(distill_lines).strip()
    distill = distill.replace(
        "# Extension table. Denoised-teacher distillation",
        "### 附录 I.2 共识教师蒸馏补充结果（Cora、CiteSeer 与 PubMed）",
    )
    distill = distill.replace(
        "| Dataset | Noise | Arm | Student | Teacher | n |",
        "| 数据集 | 噪声设置 | 实验臂 | 学生 | 教师 | n |",
    )
    distill = distill.replace(
        "T1 GLNN teacher=GCN(X)+CE",
        "T1 GLNN 式教师=GCN(X)+CE",
    )
    distill = distill.replace(
        "T3 teacher=GB-LCSS consensus GCN",
        "T3 粒球共识 GCN 教师",
    )
    distill = distill.replace(
        "T5 teacher=shuffled GB-LCSS consensus GCN",
        "T5 随机分组共识 GCN 教师",
    )
    if "## Decomposition deltas" in distill:
        distill = distill.split("## Decomposition deltas", 1)[0].rstrip()

    appendix_i = "## 附录 I 无图推理与共识教师蒸馏补充结果\n\n" + graphfree + "\n\n" + distill

    return "\n\n---\n\n".join(
        (
            appendix_a,
            appendix_b,
            appendix_c,
            appendix_d,
            appendix_e,
            appendix_f,
            appendix_g,
            appendix_h,
            appendix_i,
        )
    )


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    text = text.replace(
        "在分类任务中，GBkNN 在粒球集合上依据查询点到球的距离执行近邻分类，粒球采样则利用球的纯度和边界结构选择代表性样本 [34,37]。",
        "在分类任务中，粒球分类器利用球中心、半径和类别信息完成预测，粒球采样则依据球的纯度和边界结构选择代表性样本 [34,37]。",
    )

    text = text.replace(
        "教师与冻结表征均在训练结束后移除。实际墙钟时间、显存和吞吐量在第 5.5 节单独报告。",
        "教师与冻结表征均在训练结束后移除。",
    )

    text = text.replace(
        "表 5 将进一步分解 H 输入、全节点覆盖和局部共识目标的贡献。",
        "第 6 节进一步比较标准教师、随机分组教师与粒球共识教师。",
    )
    text = text.replace(
        "](figures/fig1_lcss_pipeline_20260716.png)",
        "](../../external/NoisyGL/docs/figures/fig1_lcss_pipeline_20260716.png)",
    )
    text = text.replace(
        "完整表与 uniform 0.6 结果见附录 I。",
        "补充结果见附录 I。",
    )
    text = text.replace(
        "ANL-CE、GMM+CT 和 FINE 的补充评测未改变最强对照，具体结果见附录 B。",
        "ANL-CE、GMM+CT 和 FINE 的补充评测未改变最强对照，具体结果见附录 A。",
    )
    text = text.replace(
        "在 uniform 0.4 下，本文还评测了 ANL-CE [21]、GMM+CT [25] 和 FINE [24] 的 MLP-adapted 版本。适配过程保留各方法的核心目标或样本选择机制，同时统一骨干、优化器和节点分类协议。",
        "本文还评测了 ANL-CE [21]、GMM+CT [25] 和 FINE [24] 的 MLP-adapted 版本。正文仅讨论 uniform 0.4 结果，完整噪声谱列于附录 A。适配过程保留各方法的核心目标或样本选择机制，同时统一骨干、优化器和节点分类协议。",
    )
    text = text.replace(
        "默认方法允许节点自身的含噪标签参与所属球的统计。第 5.3 节使用 leave-one-out 变体单独分析自票影响，该变体不改变基础算法的定义。",
        "默认方法允许节点自身的含噪标签参与所属球的统计。",
    )
    text = text.replace(
        "局部共识在线性学生上仍保持稳定增益。线性选择方法均为统一协议下的适配实现，详细结果见附录。",
        "局部共识在线性学生上仍保持稳定增益。线性选择方法均为统一协议下的适配实现。",
    )

    ablation = """#### 5.3.1 局部共识与标签平滑

为检验增益是否仅来自目标软化，本文比较 GB-LCSS、随机化分组、均匀标签平滑和含噪训练先验平滑。四种方法使用相同的 X-only 学生和优化配置，其中随机化分组仅打乱粒球成员归属，两种标签平滑方法不使用局部几何结构。

| 数据集 | GB-LCSS | 随机化分组 | 均匀平滑 | 先验平滑 |
|---|---:|---:|---:|---:|
| Cora | **55.59±1.50** | 43.63±1.62 | 44.69±2.90 | 44.89±2.38 |
| CiteSeer | **53.83±2.75** | 42.35±2.74 | 43.69±3.31 | 43.43±2.99 |
| CoraFull | **37.33±0.81** | 32.27±0.81 | 32.59±1.12 | 32.63±0.84 |
| Coauthor-CS | **90.62±0.57** | 87.60±1.03 | 88.25±0.80 | 88.04±0.73 |
| Amazon-Photo | **81.64±0.93** | 72.06±2.97 | 74.18±1.72 | 74.72±1.44 |
| PubMed | **71.90±0.88** | 62.72±4.49 | 63.67±5.08 | 64.25±4.52 |

GB-LCSS 在六个数据集上均高于随机化分组和两种显式标签平滑。该结果说明，性能增益依赖冻结表征中的真实粒球结构，而不是一般性的目标平滑。
"""
    text = replace_between(
        text,
        "#### 5.3.1 局部共识与标签平滑",
        "#### 5.3.3 共识空间",
        ablation,
    )
    text = text.replace(
        "#### 5.3.3 共识空间",
        "#### 5.3.2 共识空间",
    )

    text = text.replace(
        "#### 5.3.4 学生容量与 PubMed",
        "#### 5.3.3 学生容量与 PubMed",
    )
    text = replace_between(
        text,
        "#### 5.3.5 H-access 参照",
        "#### 5.3.6 标签率与输入扰动",
        "",
    )
    text = text.replace(
        "#### 5.3.6 标签率与输入扰动",
        "#### 5.3.4 标签率与输入扰动",
    )

    text = replace_between(
        text,
        "### 5.5 计算与部署开销",
        "## 6 扩展：以共识教师蒸馏弥合精度差距",
        "## 6 扩展：以共识教师蒸馏弥合精度差距",
    )

    extension = r"""## 6 扩展：以共识教师蒸馏弥合精度差距

基础 GB-LCSS 将图结构压缩为训练节点上的软目标，未标记节点不参与学生的直接监督。共识教师蒸馏通过图教师为全部节点生成逐节点软预测，扩大监督覆盖，并向 X-only 学生传递更细粒度的结构信息。

BO-NNC 与 SAMTD 通过多教师组合和权重学习提高鲁棒性 [45,58]。本文采用两层 GCN 作为图教师，并以第 3.5 节定义的粒球共识目标训练教师。表 4 比较 GLNN 式教师、随机分组共识教师和粒球共识教师对应的完整训练配置。学生均为同一 X-only MLP，粒球教师使用完整球内分布目标（\(\beta_T=0\)）。

**表 4** 共识教师蒸馏（测试精度 %，5 种子均值±标准差）。粗体为每列最高均值。

| 教师 | Cora u0.4 | CiteSeer u0.4 | CoraFull u0.4 | Cora u0.6 | CiteSeer u0.6 | CoraFull u0.6 |
|---|---:|---:|---:|---:|---:|---:|
| GCN(X)+CE（GLNN 式） | 77.95±2.62 | 66.09±1.38 | 58.74±0.82 | 65.50±3.28 | 53.30±4.42 | 51.63±2.02 |
| 随机分组共识教师 | 78.81±1.63 | 68.60±1.14 | **58.98±0.41** | 68.51±1.82 | 61.07±2.76 | **52.20±0.72** |
| 粒球共识教师（本文） | **80.60±0.69** | **69.40±1.08** | 58.22±0.69 | **71.27±2.46** | **62.28±2.30** | 51.82±0.86 |

粒球共识教师在 Cora 和 CiteSeer 上持续优于 GLNN 式教师。uniform 0.4 下，学生精度分别提高 2.65 和 3.31 个百分点。uniform 0.6 下，增量扩大至 5.77 和 8.98 个百分点。CoraFull 上，粒球教师未超过随机分组教师，说明该数据集的共识目标贡献有限。

附录 F 中，蒸馏学生的 cut value 比基础 GB-LCSS 学生高 0.14 至 0.25，且五个种子方向一致。该现象与软标签蒸馏向纯特征学生传递图偏置的理论分析 [32] 相符。

表 5 将蒸馏学生与推理阶段仍使用图结构的 GCN、NRGNN [9]、TFR [40]、ProCon [41] 和 GD² [42] 置于同一噪声协议下比较。两类方法的训练资源与推理输入不同，因此该表用于衡量 X-only 学生与图模型之间的精度距离。其他图方法的运行与协议记录见附录 C。

**表 5** 图领域专用去噪方法参照（uniform 0.4，测试精度 %，5 种子均值±标准差）。粗体为每列最高均值。

| 方法 | 推理需图 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|:---:|---:|---:|---:|---:|---:|
| GCN(X) | 是 | 78.97±1.63 | 64.53±1.54 | 61.28±0.52 | 92.06±0.34 | 90.77±0.43 |
| NRGNN | 是 | **81.76±1.28** | **71.03±1.01** | **62.70±0.63** | 93.56±0.11 | 89.02±1.08 |
| TFR-GCN | 是 | 78.75±0.11 | 68.92±1.77 | 42.12±1.14 | 78.36±2.39 | 35.36±9.43 |
| ProCon | 是 | 77.19±2.17 | 61.07±3.32 | 36.42±3.48 | 86.08±6.85 | 20.74±4.82 |
| GD² | 是 | 76.89±3.51 | 63.07±1.58 | 59.88±0.79 | 91.57±0.18 | **91.04±1.00** |
| **GB-LCSS 蒸馏扩展（学生）** | **否** | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 | **93.75±0.22** | 90.49±1.05 |

注：TFR-GCN、ProCon 与 GD² 为统一协议下的本地适配或重实现结果。

蒸馏后的 X-only 学生在 Cora、CiteSeer 和 CoraFull 上分别比 NRGNN 低 1.16、1.63 和 4.48 个百分点，在 Coauthor-CS 上达到该表最高均值，在 Amazon-Photo 上与最强图方法相差约 0.55 个百分点。共识教师蒸馏在不改变学生推理输入的条件下，缩小了基础 GB-LCSS 与图模型之间的精度差距。
"""
    text = replace_between(
        text,
        "## 6 扩展：以共识教师蒸馏弥合精度差距",
        "## 7 结论",
        extension,
    )

    conclusion = """## 7 结论

本文面向带噪训练标签下的仅属性节点分类，提出粒球局部共识软监督方法 GB-LCSS。该方法在冻结图对比表征上构造粒球，将球内含噪标签聚合为软监督目标，并训练推理阶段只接收节点属性的分类器。在七个正式同骨干对照中，GB-LCSS 在参数冻结后完成完整噪声谱评测的三个数据集上，于 15 个设置中的 14 个取得最高平均精度。纳入两个开发数据集后，该计数为 24/25。

随机化分组和标签平滑对照表明，真实粒球结构是局部共识增益的重要来源。共识空间、学生容量和预训练随机性实验进一步显示，方法收益取决于冻结表征能否形成可用的局部几何，以及局部共识目标与学生训练过程之间的匹配。共识教师蒸馏将训练期图结构转化为全节点软监督，使 X-only 学生在多个数据集上达到具有竞争力的 GCN 精度。

GB-LCSS 依赖能够形成稳定局部几何的冻结图表征，目前采用 transductive 预训练，并在完全干净标签和部分数据集上表现出有限收益。未来工作将面向真实标注噪声、inductive 表征学习和更大规模图，进一步研究可部署的粒球共识估计与不确定性建模。
"""
    text = replace_between(text, "## 7 结论", "## 8 数据、代码与结果资产", conclusion)

    appendix_start = "## 附录索引（投稿时编排为 Supplementary Material）"
    text = text[: text.index(appendix_start)].rstrip()
    # The dataset statistics table was numbered 0 in the working draft.
    # Normalize the body to conventional one-based table numbering.
    for old, new in ((5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)):
        text = text.replace(f"表 {old}", f"表 {new}")
    if text.endswith("---"):
        text = text[:-3].rstrip()
    text += "\n\n---\n\n# 补充材料\n\n"
    text += appendix_content().strip()
    text += "\n"

    OUTPUT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build()
