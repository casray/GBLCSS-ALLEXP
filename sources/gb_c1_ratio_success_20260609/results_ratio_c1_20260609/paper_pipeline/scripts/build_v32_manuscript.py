#!/usr/bin/env python3
"""Build the v32 Chinese submission candidate from v29 prose and v32 evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parents[1]
NOISYGL_ROOT = EXPERIMENT_ROOT / "external" / "NoisyGL"

V29_PATH = PROJECT_ROOT / "paper_final_zh_v29_human_style_20260727.md"
TABLE_DIR = PROJECT_ROOT / "paper_pipeline" / "03_tables"
TIE_SAFE_DIR = (
    NOISYGL_ROOT
    / "runs"
    / "tie_safe_full_grid_20260727_v2_immutable"
)
BASELINE_DIR = (
    NOISYGL_ROOT / "runs" / "v31_baseline_refresh_20260728"
)
VALIDATION_DIR = (
    NOISYGL_ROOT / "runs" / "validation_scope_sensitivity_20260729"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "paper_final_zh_v32_submission_candidate_20260729.md"
)

DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}
REGIMES = (
    ("uniform0.2", "Uniform 0.2"),
    ("uniform0.4", "Uniform 0.4"),
    ("uniform0.6", "Uniform 0.6"),
    ("pair0.4", "Pair 0.4"),
    ("instance0.4", "Instance 0.4"),
)
GB = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = f"{GB}_shuffled"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def section(text: str, start: str, end: str | None) -> str:
    start_index = text.index(start)
    end_index = len(text) if end is None else text.index(end, start_index)
    return text[start_index:end_index].rstrip()


def replace_subsection(
    text: str,
    start: str,
    end: str,
    replacement: str,
) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]


def table_fragment(name: str) -> str:
    text = (TABLE_DIR / name).read_text(encoding="utf-8").strip()
    first_break = text.find("\n\n")
    return text[first_break + 2 :] if first_break >= 0 else text


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_score(value: float, std: float) -> str:
    return f"{100 * value:.2f}±{100 * std:.2f}"


def build_front() -> str:
    return r"""# 粒球局部共识软监督：带噪标签下的无图推理节点分类

**Granular-Ball Local-Consensus Soft Supervision for Noisy-Label Node Classification with Graph-Free Inference**

---

## Highlights

- GB-LCSS 面向训练标签噪声与仅属性推理并存的节点分类。
- 标签无关粒球将冻结图表征中的局部共识转化为软监督。
- 真实粒球分组在 25 个噪声设置上均优于随机等规模分组。
- GB-LCSS 在统一协议下整体优于 PosteL、ALS 和 SALS。
- 共识教师蒸馏在不改变学生推理输入的情况下提高绝对精度。

---

## 摘要

训练阶段可访问图结构而推理阶段只能使用节点属性，是关系缺失、隐私约束和低时延需求下常见的部署场景。现有图标签去噪方法通常在训练或推理时依赖邻接信息，GNN-to-MLP 方法则主要在干净监督下研究结构知识迁移。本文提出粒球局部共识软监督（Granular-Ball Local-Consensus Soft Supervision，GB-LCSS）。该方法在冻结的无监督图对比表征中构造标签无关粒球，聚合球内含噪训练标签形成并列安全的软监督，再训练仅以节点属性为输入的 MLP。实验采用转导式 20%/10%/70% 划分，默认只污染训练标签，并使用干净验证标签选择模型。在五个数据集和五种噪声设置中，真实粒球分组在 25/25 个设置上优于随机等规模分组；GB-LCSS 分别在 24/25 和 23/25 个设置上高于 MLP+CE 与 JoCoR，并在统一协议下整体优于 PosteL、ALS 和 SALS。验证标签同步受噪时，主要分组增量仍保持。共识教师蒸馏进一步缩小了仅属性学生与图模型之间的精度差距。结果表明，GB-LCSS 为带噪标签下的无图推理节点分类提供了一种简单、可审计的监督构造方式。

**关键词**：标签噪声，节点分类，无图推理，图对比学习，粒球计算，知识蒸馏

---"""


def build_intro_related(v29: str) -> str:
    intro = section(v29, "## 1 引言", "## 2 相关工作")
    intro = intro.replace(
        "整个过程不依赖干净标签或模型预测反馈，训练后的分类器仅使用节点属性。",
        "软目标构造不读取干净训练标签，也不依赖模型预测反馈。干净验证标签只用于模型选择，训练后的分类器仅使用节点属性。",
    )
    old_result = (
        "在 11 个数据集的统一评测和其中 5 个数据集的完整噪声谱实验中，"
        "GB-LCSS 在多数设置下优于采用相同 X-only MLP 骨干的通用抗噪方法。"
    )
    intro = intro.replace(
        old_result,
        "实验覆盖五个完整噪声谱数据集和五个补充数据集。当前 tie-safe 主网格中，真实粒球分组在 25 个设置上均高于随机等规模分组，GB-LCSS 相对 MLP+CE 与 JoCoR 的均值差分别在 24 和 23 个设置上为正。",
    )
    intro = intro.replace(
        "该扩展仍只使用原有的含噪训练标签，不额外使用干净标签，学生在测试时也仍然只使用节点属性。",
        "该扩展不增加人工标注，学生在测试时仍然只使用节点属性。",
    )
    intro = intro.replace(
        "2. 设计粒球划分与置信度软目标机制。粒球划分不使用标签，球内含噪训练标签用于形成组级目标。共识置信度决定组级目标与节点原始标签的混合比例，整个过程无需干净标签或模型预测反馈。",
        "2. 设计标签无关的粒球划分与并列安全软目标。球内含噪标签形成组级分布，归一化共识权重控制组级目标与节点原始标签的混合，避免多数类并列时依赖类别索引。",
    )
    intro = intro.replace(
        "3. 在 11 个图数据集和多种噪声协议下验证 GB-LCSS 的有效性与适用范围，并通过随机分组对照分析局部几何结构的贡献。共识教师蒸馏进一步将覆盖全部节点的教师预测传递给学生，在保持 X-only 推理的同时提高分类精度。",
        "3. 在多数据集和多噪声协议下比较同资源抗噪方法与结构辅助 X-only 方法，并通过随机分组、LOO、分组替代和验证标签噪声实验界定增益来源。共识教师蒸馏进一步向全部节点传递教师预测，同时保持 X-only 推理。",
    )
    related = section(v29, "## 2 相关工作", "## 3 方法")
    return intro + "\n\n" + related


def build_method(v29: str) -> str:
    method = section(v29, "## 3 方法", "## 4 实验设置")
    method = method.replace(
        "| \\(p_j,m_j,r_j\\) | 球内含噪标签分布、多数类及多数类票占比 |",
        "| \\(p_j,M_j,r_j\\) | 球内含噪标签分布、最大票类别集合及最大票占比 |",
    ).replace(
        "| \\(q_j\\) | 球级半硬共识目标 |",
        "| \\(q_j\\) | 球级并列安全共识目标 |",
    ).replace(
        "| \\(\\alpha_j\\) | 球级共识权重 |",
        "| \\(\\alpha_j\\) | 归一化共识权重 |",
    )

    tie_safe = r"""#### 3.4.1 球级并列安全共识目标

对于粒球 \(B_j\)，记其中训练节点为 \(T_j=B_j\cap\mathcal V_L\)。当 \(|T_j|\ge s_{\min}\) 时，统计含噪标签分布

\[
p_{jc}=
\frac{1}{|T_j|}
\sum_{v\in T_j}\mathbb I(\tilde y_v=c),
\qquad c=1,\ldots,C.
\tag{7}
\]

令

\[
M_j=\{c:p_{jc}=r_j\},
\qquad
r_j=\max_c p_{jc}.
\tag{8}
\]

\(M_j\) 是获得最大票数的类别集合。并列判断直接使用整数票数。多数类唯一时，将其记为 \(m_j\)，并混合经验分布与多数类 one-hot。出现并列时只保留经验分布：

\[
q_j=
\begin{cases}
(1-\beta)p_j+\beta e_{m_j}, & |M_j|=1,\\
p_j, & |M_j|>1.
\end{cases}
\tag{9}
\]

其中 \(e_c\) 为类别 \(c\) 的 one-hot 向量，基础方法固定 \(\beta=0.5\)。该分支在实现中记为 `distribution_only`。它避免了常规 `argmax` 在并列时选择最小类别索引，从而使目标在类别重编号下保持置换等变。
"""
    alpha = r"""#### 3.4.2 归一化共识权重与节点目标

本文以均匀类别概率 \(1/C\) 为下界，对最大票占比作线性归一化：

\[
\alpha_j=
\max\left(
0,
\frac{r_j-1/C}{1-1/C}
\right).
\tag{10}
\]

\(\alpha_j\) 是启发式混合权重，不是经过校准的正确概率。对于 \(v\in T_j\)，最终训练目标为

\[
t_v=(1-\alpha_j)e_{\tilde y_v}+\alpha_j q_j.
\tag{11}
\]

当球内投票完全均匀时，\(\alpha_j=0\)，目标退回原始含噪 one-hot。当前实现取 \(s_{\min}=2\)。票数不足的球同样保留原始标签。默认方法允许节点自身的含噪标签参与所属球统计，LOO 变体在机制分析中单独移除该自票。

粒球成员关系的生成不读取标签。式 (7) 至式 (11) 只读取训练节点的观测标签，不读取验证标签、测试标签或模型预测。干净验证标签用于 checkpoint 选择，测试标签只用于最终评估。
"""
    method = replace_subsection(
        method,
        "#### 3.4.1 球级半硬目标",
        "#### 3.4.2 置信度自适应的节点目标",
        tie_safe,
    )
    method = replace_subsection(
        method,
        "#### 3.4.2 置信度自适应的节点目标",
        "#### 3.4.3 仅属性学生训练",
        alpha,
    )
    method = method.replace(
        "按式 (7)–(10)计算 p_j, m_j, r_j, q_j 和 α_j",
        "按式 (7)–(10)计算 p_j, M_j, r_j, q_j 和 α_j",
    )
    return method


def build_experiment_setup() -> str:
    return r"""## 4 实验设置

### 4.1 数据集与评测范围

正文完整噪声谱覆盖 Cora、CiteSeer、CoraFull、Coauthor-CS 和 Amazon-Photo。另以 Amazon-Computers、Cora_ML、Attributed-Wiki、CiteSeerFull 和 Coauthor-Physics 考察跨数据集表现。一个额外边界数据集及其全部结果统一置于附录 E，不参与正文核心结论计数。

| 数据集 | 节点 | 无向边 | 特征维数 | 类别 | 训练节点 | 边同质性 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 2,708 | 5,278 | 1,433 | 7 | 541 | 0.810 |
| CiteSeer | 3,327 | 4,552 | 3,703 | 6 | 665 | 0.736 |
| CoraFull | 19,793 | 63,421 | 8,710 | 70 | 3,958 | 0.567 |
| Coauthor-CS | 18,333 | 81,894 | 6,805 | 15 | 3,666 | 0.808 |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | 1,530 | 0.827 |
| Amazon-Computers | 13,752 | 245,861 | 767 | 10 | 2,750 | 0.777 |
| Cora_ML | 2,995 | 8,158 | 2,879 | 7 | 599 | 0.789 |
| Attributed-Wiki | 2,405 | 8,990 | 4,973 | 17 | 481 | 0.669 |
| CiteSeerFull | 4,230 | 5,337 | 602 | 6 | 846 | 0.949 |
| Coauthor-Physics | 34,493 | 247,962 | 8,415 | 5 | 6,898 | 0.931 |

边同质性仅描述数据，不参与训练、目标构造或模型选择。

### 4.2 数据划分与标签噪声

本文采用转导式（transductive）节点分类。每个数据集按 20%/10%/70% 划分训练、验证和测试节点。默认只污染训练标签，验证标签保持干净并用于 checkpoint 选择，测试标签只用于最终评估。该 percent 划分是本文的固定协议，并非 NoisyGL 的全部默认设置。

uniform 噪声满足 \(P_{ii}=1-\rho\) 与 \(P_{ij}=\rho/(C-1)\)，其中 \(\rho\in\{0.2,0.4,0.6\}\)。pair 噪声使用 \(\rho=0.4\) 和 \(i\rightarrow(i-1)\bmod C\)。instance-dependent 噪声同样取平均翻转率 0.4，并根据节点属性与类别相关随机映射分配非真类概率。噪声生成形式参考 NoisyGL [13]，所有方法在同一数据集和种子下共享划分与训练噪声实例。

### 4.3 比较方法与资源分层

表 1 比较只由含噪训练标签构造监督的同骨干方法，包括 MLP+CE、Small-loss、Co-teaching、JoCoR、SCE、ELR 和 LogitClip。表 2 单独比较训练期使用图信息、推理期仍为 X-only 的方法。PosteL、ALS 和 SALS 使用当前统一协议结果。Graph-MLP 与旧 GSDN 为项目适配实现，其中旧 GSDN 仅作临时定位参照，待新的实现完成后再形成最终强弱结论。表 4 的蒸馏扩展使用更强的 GCN 教师和全节点软预测，因此不与基础方法混为同一资源等级。

全部基础方法采用相同的两层 MLP，隐层宽度为 64，dropout 为 0.5。模型使用 Adam，学习率为 0.01，权重衰减为 \(5\times10^{-4}\)，最多训练 200 个 epoch。Small-loss、Co-teaching 和 JoCoR 按已知合成噪声率设置 forget-rate schedule，其他基础方法不使用该信息。

### 4.4 统计与证据谱系

正式结果报告 5 个配对种子的均值和样本标准差，并结合逐种子方向、配对均值差、95% 置信区间和 Holm 校正解释。当前核心结论只来自 tie-safe v2 不可变网格和 PosteL/ALS/SALS 正式网格。LOO、kNN、H-access 与教师蒸馏属于较早的机制或扩展批次，均明确标为历史证据，不并入当前主网格计数。附录 G 提供源 CSV、SHA256、方法身份和协议登记。
"""


def build_results() -> str:
    table_a = table_fragment("v32_table_a_target_only.md")
    table_b = table_fragment("v32_table_b_graph_assisted_xonly.md")
    return rf"""## 5 实验结果与分析

### 5.1 同资源抗噪方法

**表 1** 同资源方法在 uniform 0.4 下的测试精度。

{table_a}

更新后的 tie-safe v2 网格覆盖五个数据集和五种噪声设置。GB-LCSS 相对随机等规模分组的平均增量为 7.84 个百分点，25/25 个设置均为正。相对 MLP+CE 和 JoCoR 的平均增量分别为 7.25 和 2.96 个百分点，正向设置数分别为 24/25 和 23/25。完整噪声谱和配对统计见附录 A、B。上述计数只描述预先固定的三个比较族，不解释为对所有方法的无条件胜率。

真实分组相对随机等规模分组的稳定差异是当前最直接的机制证据。随机化保留球数、球规模分布、目标公式和学生训练过程，只破坏节点与局部单元的对应关系。因此，25/25 的正向结果说明冻结表征中的实际成员关系包含随机分组未保留的信息。

### 5.2 结构辅助 X-only 方法

**表 2** 结构辅助 X-only 方法在 uniform 0.4 下的测试精度。

{table_b}

在完整 25-setting 统一协议中，GB-LCSS 相对 PosteL、ALS 和 SALS 的平均配对增量分别为 5.20、7.22 和 4.20 个百分点，正向设置数分别为 24/25、25/25 和 24/25。它表明冻结表征中的分组共识没有被一般后验软标签或邻域平滑完全吸收。

旧 GSDN 行使用项目早期的临时适配实现。该实现参与验证标签敏感性测试，但不承担对 GSDN 的最终排名结论。新的、更贴合实验协议的实现将在独立版本中替换，正文因此不报告“GB-LCSS 全面优于或弱于 GSDN”的结论。

### 5.3 消融与机制分析

#### 5.3.1 自票、显式平滑与随机分组

历史 LOO 批次在计算节点目标时移除节点自身投票。Cora 和 CiteSeer 上，LOO 相对当时默认目标分别提高 1.56 和 1.37 个百分点；CoraFull 与 Coauthor-CS 的差值接近零，Amazon-Photo 提高 0.47 个百分点。LOO 在这些数据集上均高于均匀平滑、训练先验平滑和 LOO 随机分组。与此同时，LOO 会同时提高噪声纠正率与干净标签误伤率，且单数据集增量未通过族内 Holm 校正。因此，LOO 用于说明其他球内节点确实提供额外监督信息，不替换当前默认目标。完整表见附录 D。

#### 5.3.2 分组形式与有效票数

历史 kNN 和 k-means 消融表明，局部共识的收益并非粒球划分独有。在所测四个数据集上，事后观察到的 \(k=20\) kNN 高于早期粒球约 0.3 至 2.6 个百分点。该 \(k\) 来自测试曲线，不构成经过验证集选择的正式基线。调大粒球后，有效训练票数由约 10 增至约 24，部分差距随之缩小。现有证据支持“稳定多数统计需要足够票数”，但没有证明粒球优于 kNN 或 k-means。本文保留粒球，是因为它提供球数上限受控的一次性分区接口。

#### 5.3.3 H-access 参照

直接使用冻结表征 \(H\) 会改变推理资源。uniform 0.4 的历史结果如下。

| 分类路线 | 推理使用 \(H\) | Cora | CiteSeer |
|---|:---:|---:|---:|
| GB-LCSS（X-only，历史批次） | 否 | 55.59 | 53.83 |
| H-only 线性分类头 | 是 | 78.61 | 68.87 |
| H-only MLP 分类头 | 是 | 73.09 | 68.69 |
| MLP[X‖H]+CE | 是 | 48.54 | 51.66 |
| MLP[X‖H]+Small-loss | 是 | 57.70 | 56.30 |
| MLP[X‖H]+JoCoR | 是 | 60.13 | 58.60 |

\(H\) 携带强结构信息，但其收益取决于训练策略。该表将“直接携带图派生表征”与“只在训练期把结构信息转化为监督”区分开来。第 6 节进一步考察如何通过教师蒸馏把更多结构知识转移给 X-only 学生。

#### 5.3.4 并列安全分支

当前规则在多数类唯一时混合球内分布与 one-hot，在并列时仅使用分布。125 个 GB-LCSS seed-level 运行中有 123 行出现有效并列，共检测到 911 个并列球。该分支涉及 12,788/259,000 个节点，占 4.94%。修订首先解决类别置换正确性，而不是为了提高精度。由于没有同轮、仅改变 tie 规则的完整因果消融，本文不把当前精度变化归因于该分支。

#### 5.3.5 表征与预训练随机性

冻结 GRACE 表征在当前递归划分下形成了稳定的局部单元。部分其他自监督表征虽然线性探针更高，却在相同构造参数下退化为单球，说明线性可分性与局部可聚类性并不相同。早期 \(3\) 个 GRACE seed \(\times3\) 个下游 seed 诊断中，Cora、CiteSeer 和 CoraFull 的跨表征种子标准差分别为 0.53、0.07 和 0.44 个百分点。该诊断使用早期目标规则，只说明方向在已测试表征种子上保持，不替代当前 tie-safe 网格。

### 5.4 验证标签噪声敏感性

默认协议使用干净验证标签选择模型。为检验这一条件是否决定主要结论，本文保持训练噪声实例、数据划分和测试标签不变，仅以独立随机流向验证标签注入 uniform 0.4 噪声。实验覆盖 Cora、CiteSeer 和 CoraFull，六种方法共 90 个新增运行。

GB-LCSS 相对随机分组在三个数据集上分别领先 12.45、11.45 和 4.96 个百分点，均为 5/5 seeds 正向；相对 MLP+CE 分别领先 12.64、11.41 和 5.05 个百分点。六方法的 clean-validation 与 noisy-validation 排序 Kendall \(\tau\) 均为 0.867。预注册的主要机制门槛全部通过。该结果说明真实分组和基础 CE 之间的相对差异不依赖验证标签保持干净。旧 GSDN 仍只作临时参照，完整协议交互见附录 C。

### 5.5 计算与部署开销

GB-LCSS 与 MLP+CE 使用相同学生，因此参数量、推理输入和前向复杂度相同。当前 tie-safe seed-0 记录中，Cora、CiteSeer 和 CoraFull 的构球时间分别为 1.59、2.51 和 6.08 秒。两者推理均不访问邻接矩阵。一次性 GRACE 预训练时间分别为 0.07、0.13 和 74.48 分钟，峰值显存为 0.40、0.67 和 12.73 GiB。同一份表征可供同一数据集的全部噪声设置和下游种子复用。当前数据只支持指定硬件和 full-batch 协议内的比较。
"""


def build_extension() -> str:
    table_c = table_fragment("v32_table_c_distillation.md")
    return rf"""## 6 扩展：共识教师蒸馏

基础 GB-LCSS 只在训练节点上生成目标，图结构被压缩为少量节点上的组级监督。共识教师蒸馏用相同目标训练读取图和 \(X\Vert H\) 的 GCN 教师，再将教师对全部节点的逐节点预测蒸馏给 X-only 学生。该扩展不增加人工标注，也不改变学生的推理输入。

**表 3** 共识教师蒸馏的历史五臂比较。

{table_c}

球式教师对应学生在 Cora 和 CiteSeer uniform 0.6 上分别达到 71.27 和 62.28，较 GLNN 式学生提高 5.77 和 8.98 个百分点。CoraFull 上，直接使用 \(X\Vert H\) 训练教师更强。票数匹配 kNN 教师与球式教师的六项差值均小于 1 个百分点，现有 \(n=5\) 结果只能说明二者数值接近，不能证明统计等效或粒球不可替代。

该表属于早期扩展批次，同时改变教师输入、图传播和全节点监督覆盖。它回答的是完整蒸馏路线能否缩小 X-only 学生与图模型的精度差距，而不是各因素的独立因果贡献。与 BO-NNC 和 SAMTD 的多教师加权不同，本文固定教师架构，只改变教师监督目标。完整教师与学生结果见附录 D。
"""


def build_conclusion() -> str:
    return r"""## 7 结论

本文研究训练标签含噪、训练阶段可访问图、推理阶段只使用节点属性的转导节点分类。GB-LCSS 在冻结图对比表征中构造标签无关粒球，再将球内含噪标签聚合为并列安全的软目标。当前 tie-safe 网格中，真实分组在 25/25 个设置上高于随机等规模分组；GB-LCSS 相对 MLP+CE 和 JoCoR 分别在 24/25 和 23/25 个设置上取得正增量。统一协议结果还显示，GB-LCSS 整体高于 PosteL、ALS 和 SALS。验证标签同步受噪后，主要分组增量仍保持。

这些结果支持粒球作为固定预算局部共识接口的有效性，但不证明它优于所有邻域构造。历史 kNN 和 k-means 消融获得了相近结果，旧 GSDN 也只完成临时适配，尚不承担最终排名结论。共识教师蒸馏在若干数据集上进一步提高 X-only 学生精度，同时引入更强的训练资源。

本文证据限于合成标签噪声、干净验证主协议、同图转导评测和固定冻结表征。对 uniform 0.4 目标的事后分箱诊断显示，目标正确率在五个数据集中的四个随 \(\alpha\) 单调提高，但合并结果并非严格单调。因此，\(\alpha\) 仅被解释为启发式共识权重，而不是经过校准的正确概率。附录同时报告验证标签噪声、历史负结果和边界案例。后续工作将面向真实标注噪声、归纳式新节点、预训练种子扩展，以及更严格的结构辅助方法复现。
"""


def build_data_section(v29: str) -> str:
    data = section(v29, "## 8 数据、代码与结果资产", "## 参考文献")
    data = data.replace(
        "## 8 数据、代码与结果资产",
        "## 8 数据与代码可用性",
    )
    return data


def _aggregate_index() -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["dataset"], row["regime"], row["method"]): row
        for row in read_csv(TIE_SAFE_DIR / "aggregate.csv")
    }


def build_appendix_a() -> str:
    indexed = _aggregate_index()
    methods = (
        ("mlp_ce", "MLP+CE"),
        ("mlp_jocor", "JoCoR"),
        (SHUFFLED, "随机等规模分组"),
        (GB, "GB-LCSS"),
    )
    chunks = ["## 附录 A 当前 tie-safe v2 完整主网格"]
    for dataset in DATASETS:
        rows = []
        for method, label in methods:
            scores = []
            for regime, _ in REGIMES:
                row = indexed[(dataset, regime, method)]
                scores.append(
                    format_score(
                        float(row["test_mean"]),
                        float(row["test_sample_std"]),
                    )
                )
            rows.append((label, *scores))
        chunks.append(
            f"### A.{len(chunks)} {DATASET_LABELS[dataset]}\n\n"
            + markdown_table(
                ("方法", *(label for _, label in REGIMES)),
                rows,
            )
        )
    return "\n\n".join(chunks)


def build_appendix_b() -> str:
    rows = []
    comparator_labels = {
        SHUFFLED: "随机分组",
        "mlp_ce": "MLP+CE",
        "mlp_jocor": "JoCoR",
    }
    for row in read_csv(TIE_SAFE_DIR / "paired_tests.csv"):
        rows.append(
            (
                DATASET_LABELS[row["dataset"]],
                dict(REGIMES)[row["regime"]],
                f"GB-LCSS−{comparator_labels[row['comparator']]}",
                f"{100 * float(row['paired_delta_mean']):+.2f}",
                (
                    f"[{100 * float(row['paired_delta_ci95_low']):+.2f}, "
                    f"{100 * float(row['paired_delta_ci95_high']):+.2f}]"
                ),
                f"{row['positive_seed_count']}/5",
                f"{float(row['holm_adjusted_p']):.4g}",
            )
        )
    return (
        "## 附录 B 当前配对统计\n\n"
        "均值差与 95% 置信区间以百分点表示。Holm 校正在三个独立的 "
        "25-setting 比较族内进行。\n\n"
        + markdown_table(
            (
                "数据集",
                "噪声",
                "比较",
                "均值差",
                "95% CI",
                "正向 seeds",
                "Holm p",
            ),
            rows,
        )
    )


def build_appendix_c() -> str:
    rows = []
    labels = {
        GB: "GB-LCSS",
        SHUFFLED: "随机分组",
        "mlp_ce": "MLP+CE",
        "mlp_jocor": "JoCoR",
        "postel_mlp_noipl_rm": "PosteL",
        "gsdn": "GSDN（临时适配）",
    }
    for row in read_csv(VALIDATION_DIR / "aggregate.csv"):
        rows.append(
            (
                row["protocol"],
                DATASET_LABELS[row["dataset"]],
                labels[row["method"]],
                format_score(
                    float(row["test_mean"]),
                    float(row["test_sample_std"]),
                ),
            )
        )
    return (
        "## 附录 C 验证标签噪声敏感性\n\n"
        "训练噪声哈希在两种验证协议之间保持一致，测试标签始终不受污染。"
        "旧 GSDN 行仅作临时定位。\n\n"
        + markdown_table(("验证协议", "数据集", "方法", "测试精度"), rows)
    )


def build_appendix_d() -> str:
    return r"""## 附录 D 历史机制与扩展证据

### D.1 LOO 与显式平滑

| 数据集 | 历史 GB-LCSS | LOO-GB-LCSS | LOO-随机分组 | 均匀平滑 | 先验平滑 | LOO−历史基线 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 55.59±1.50 | 57.15±0.84 | 42.79±1.11 | 44.69±2.90 | 44.89±2.38 | +1.56 |
| CiteSeer | 53.83±2.75 | 55.21±2.34 | 41.62±2.80 | 43.69±3.31 | 43.43±2.99 | +1.37 |
| CoraFull | 37.33±0.81 | 37.18±0.48 | 32.18±0.73 | 32.59±1.12 | 32.63±0.84 | −0.15 |
| Coauthor-CS | 90.62±0.57 | 90.61±0.62 | 87.46±1.08 | 88.25±0.80 | 88.04±0.73 | −0.01 |
| Amazon-Photo | 81.59±0.91 | 82.07±1.15 | 72.83±2.67 | 74.18±1.72 | 74.72±1.44 | +0.47 |

该表来自早期目标规则，只用于机制解释，不并入当前核心结论。

### D.2 分组替代与 H-access

历史 kNN/k-means、原始属性空间、邻接传播空间和监督 GCN 表征实验的完整原始记录保存在证据登记表所列目录。它们共同支持两个边界：实际局部成员关系优于随机分组，但粒球不是已被证明不可替代的唯一分组；直接在推理时使用 \(H\) 可以获得更高精度，但属于更强输入资源。

### D.3 教师蒸馏

正文表 3 的教师与学生逐种子结果、票数匹配规则和 cut value 记录属于历史扩展证据，不参与当前基础方法的 tie-safe 计数。
"""


def build_appendix_e() -> str:
    return (
        "## 附录 E PubMed 边界案例\n\n"
        + table_fragment("v32_boundary_datasets.md")
    )


def build_appendix_f() -> str:
    return r"""## 附录 F 对称噪声下的局部排序保持

**命题 1（期望排序保持）。** 设某粒球内真实类别分布为 \(\pi\)，标签以 \(C\) 类对称噪声独立翻转，噪声率为 \(\rho\)。观测类别概率为

\[
\tilde{\pi}_c
=(1-\rho)\pi_c+
\frac{\rho}{C-1}(1-\pi_c).
\tag{20}
\]

任意两类 \(a,b\) 满足

\[
\tilde{\pi}_a-\tilde{\pi}_b
=
\left(1-\frac{C\rho}{C-1}\right)
(\pi_a-\pi_b).
\tag{21}
\]

因此，当 \(\rho<(C-1)/C\) 时，期望类别排序保持。若真实多数类 \(a\) 相对每个竞争类的最小 margin 为

\[
\Delta_{\min}=\min_{b\ne a}(\pi_a-\pi_b)>0,
\]

并记 \(\gamma=1-C\rho/(C-1)>0\)，对球内 \(n\) 个独立训练票使用 Hoeffding 不等式与 union bound 可得经验多数类错误概率

\[
\Pr(\hat a\ne a)
\le
(C-1)\exp\left(
-\frac{n\gamma^2\Delta_{\min}^2}{2}
\right).
\tag{22}
\]

该命题只解释票数、类别 margin 与对称噪声率之间的关系。它假设标签噪声独立、粒球成员关系不由标签生成，并要求球内真实分布存在正 margin。结论不覆盖 pair 或 instance-dependent 噪声，也不证明 GRACE 空间中的任意粒球都具有语义纯度。图节点标签存在相关性时，式 (22) 只能作为理想化说明，不能视为完整泛化界。
"""


def build_appendix_g() -> str:
    metrics = json.loads(
        (TABLE_DIR / "v32_headline_metrics.json").read_text(encoding="utf-8")
    )
    return f"""## 附录 G 证据谱系与复现边界

当前 tie-safe 原始网格含 500/500 条成功记录，PosteL/ALS/SALS 最近工作网格含 625 条记录。协议敏感性新增 90/90 条 noisy-validation 记录。当前三项核心结论均值差为：

- GB-LCSS−随机分组：{metrics['vs_shuffled']['mean_delta_pp']:.4f} 个百分点；
- GB-LCSS−MLP+CE：{metrics['vs_ce']['mean_delta_pp']:.4f} 个百分点；
- GB-LCSS−JoCoR：{metrics['vs_jocor']['mean_delta_pp']:.4f} 个百分点。

证据登记表记录每个方法的源 CSV、SHA256、tie mode、协议、实现身份和核心结论使用权限。旧 LOO、kNN、H-access 与蒸馏结果均标记为 historical。旧 GSDN 标记为 provisional adapted implementation。`gsdn_faithful_core` 不进入 v32 结论。
"""


def build_supplement() -> str:
    return "\n\n---\n\n".join(
        (
            "# 补充材料",
            build_appendix_a(),
            build_appendix_b(),
            build_appendix_c(),
            build_appendix_d(),
            build_appendix_e(),
            build_appendix_f(),
            build_appendix_g(),
        )
    )


def build_manuscript(*, output_path: Path = DEFAULT_OUTPUT) -> None:
    v29 = V29_PATH.read_text(encoding="utf-8")
    references = section(v29, "## 参考文献", "## 附录索引")
    parts = (
        build_front(),
        build_intro_related(v29),
        build_method(v29),
        build_experiment_setup(),
        build_results(),
        build_extension(),
        build_conclusion(),
        build_data_section(v29),
        references,
        build_supplement(),
    )
    manuscript = "\n\n---\n\n".join(part.strip() for part in parts) + "\n"
    manuscript = manuscript.replace(
        "待用户完成新实现后再替换。",
        "正式适配实现将在后续独立实验中更新。",
    )
    manuscript = manuscript.replace("；", "。")
    while "\n\n---\n\n---\n\n" in manuscript:
        manuscript = manuscript.replace("\n\n---\n\n---\n\n", "\n\n---\n\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manuscript, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manuscript(output_path=args.output)


if __name__ == "__main__":
    main()
