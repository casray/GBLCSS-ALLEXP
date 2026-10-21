# 最近工作共同协议完整网格实验报告

## 1. 实验结论

本轮完成了 GB-LCSS 与 PosteL、GSDN、ALS 和 SALS 的同资源共同协议比较。
全部方法在训练阶段使用图结构，测试阶段只接收节点属性。正式结果由 5 个
数据集、5 种噪声设置和 5 个配对种子构成。

- 25/25 个命令块完成并生成不可变原始 CSV。
- 原生最近工作结果为 500/500 行，失败、缺失和重复均为 0。
- 与冻结 GB-LCSS v2 合并后的五方法证据为 625/625 行，每种方法 125 行。
- 125/125 个 dataset--noise--seed 组通过 split、noise、污染计数和
  train-only 标签噪声审计。
- GB-LCSS 相对 PosteL、ALS 和 SALS 的 25-setting 平均配对增量分别为
  +5.202、+7.215 和 +4.202 个百分点。
- GB-LCSS 相对 GSDN 的平均配对差为 -0.698 个百分点，10/25 个设置均值
  更高，15/25 个设置均值更低。

因此，正式结论不是“GB-LCSS 优于全部最近工作”。可支持的结论是：
GB-LCSS 稳定优于三种结构辅助软目标基线 PosteL、ALS 和 SALS；与 GSDN
相比，优势随数据集变化。GSDN 在 CoraFull 和 Amazon-Photo 上更强，
GB-LCSS 在 CiteSeer 和 Coauthor-CS 上更强。该结果要求论文把性能定位
从普遍领先收紧为一种具有明确机制区别和适用范围的局部共识监督接口。

最终决策：**继续投稿收口，但收紧 claim，不再把 GB-LCSS 描述为同资源
X-only 最近工作中的整体最优方法。**

## 2. 研究假设

在相同数据划分、训练集标签噪声实例、种子、MLP 推理资源和评估流程下，
冻结自监督表征中的标签无关粒球共识可能提供不同于邻域自蒸馏、标签传播、
邻域平滑和后验标签估计的监督信号。本实验检验该信号是否在完整噪声谱中
稳定优于四个直接邻近方法。

## 3. 粒球扮演的角色

粒球仍是冻结图对比表征中的标签无关局部关系单元。它限定含噪训练标签的
聚合范围，并提供组级分布和启发式共识权重。粒球不作为硬正样本标签、
负样本过滤器或最终分类器，也不在学生训练期间更新。

## 4. 方法动机

历史表只比较了通用标签噪声策略或较宽资源层级的 graph-free 方法，无法
回答两个直接问题：

1. GB-LCSS 是否优于同样把训练期图结构转化为软监督的 PosteL、ALS 和
   SALS。
2. GB-LCSS 是否优于已经在 X-only 框架中考察标签噪声的 GSDN。

因此，本轮冻结 GB-LCSS v2，不修改目标、粒球参数、数据集或噪声协议，
只补齐四个直接方法的共同协议网格。

## 5. 代码改动

- `external/NoisyGL/predictor/GSDNFaithful_Predictor.py`
  - 按 GSDN Eq. (3)、Eq. (7) 和 Eq. (8) 实现共享主干与双预测头。
  - 将项目中的成对有向弧规范为唯一无向关系，并显式保留两个方向的损失。
- `external/NoisyGL/scripts/run_standard_downstream.py`
  - 注册 PosteL、GSDN、ALS 和 SALS 的共同协议能力与结果诊断。
- `paper_pipeline/scripts/run_nearest_work_full_grid.py`
  - 实现 25 命令 fail-fast 编排、断点精确恢复、不可变原始结果、协议审计、
    配对统计和跨目录冻结 GB-LCSS 对齐。
  - 网格完成后补充五方法联合汇总输出，避免论文阶段手工拼接遗漏 GSDN。
- `paper_pipeline/tests/test_nearest_work_full_grid.py`
  - 覆盖首次运行、精确恢复、不可变原始结果、五方法 625 行联合证据和
    125 行聚合表。

## 6. 配置改动

本轮没有修改 GB-LCSS 或四个对照方法的运行中超参数。固定协议为：

- 数据集：Cora、CiteSeer、CoraFull、Coauthor-CS、Amazon-Photo。
- 噪声：uniform 0.2、uniform 0.4、uniform 0.6、pair 0.4、
  instance-dependent 0.4。
- 噪声范围：仅训练节点。
- 划分：20% 训练、10% 验证、70% 测试。
- 种子：0、1、2、3、4。
- 推理资源：全部方法测试时只接收节点属性。
- 设备：AMD Radeon RX 7700 XT，PyTorch 2.9.1+rocm7.2.1，
  `cuda:0`。

实现身份固定为：

- PosteL-noIPL：官方代码忠实目标加项目 MLP。
- GSDN：方程驱动的共同协议适配，独立实现，不是官方代码复现。
- ALS：机制忠实的项目 MLP 适配。
- SALS：仅使用有标注训练邻居的 MLP 适配。

## 7. 预期机制

- PosteL 依据邻域标签共现估计后验软标签。
- ALS 结合标签传播、可学习类别关系、KL 正则和 pacing 在线细化目标。
- SALS 由有标注训练邻居的类别分布构造结构感知平滑目标。
- GSDN 通过图邻域自蒸馏约束双预测头。
- GB-LCSS 先在冻结无监督表征中形成标签无关分组，再聚合组内含噪训练
  标签并一次性生成并列安全目标。

若 GB-LCSS 的优势只来自“用图生成软标签”，PosteL、ALS 或 SALS 应能
吸收其收益。若冻结语义分组提供独立作用，GB-LCSS 应在这些对照上保持
正向差异。GSDN 使用逐节点在线自蒸馏，可能在部分数据集上保留更多分辨率，
因此不预设 GB-LCSS 必须胜过 GSDN。

## 8. 成功标准

1. 25 个命令块全部精确完成，每块 20 行，并具有不可变原始副本。
2. 原生最近工作网格为 500/500 行，五方法联合证据为 625/625 行。
3. 125/125 个协议组通过，验证和测试标签污染计数均为 0。
4. 每个方法、数据集和噪声设置均包含 seeds 0--4。
5. 跨方法配对只使用相同 split/noise hash。
6. 第二次运行不启动训练子进程，并把 25 个块全部标记为
   `skipped_exact_complete`。
7. 性能结论同时报告效应量、方向计数、95% 置信区间和 25 项族内 Holm
   结果。

## 9. 失败标准

- 任一缺失、重复、失败、NaN 或无穷结果。
- 任一方法修改验证或测试标签。
- 同一配对组的 split、noise、污染计数或比例不同。
- 用历史 GSDN-inspired 数值替代新的 GSDN 共同协议适配。
- 把项目适配写成官方精确复现。
- 根据结果修改数据集、方法参数或噪声设置。

本轮没有触发上述失败条件。

## 10. 实验结果

### 10.1 完整性、协议和谱系

| 检查项 | 结果 |
|---|---:|
| 正式命令块 | 25/25 |
| 不可变原始 CSV | 25/25 |
| immutable SHA-256 不一致 | 0 |
| 原生最近工作行 | 500/500 |
| 五方法联合行 | 625/625 |
| 五方法联合聚合行 | 125/125 |
| 协议组 | 125/125 PASS |
| 独立证据完整性 | 500/500 OK |
| 跨方法配对差 | 500/500 |
| 跨方法配对检验 | 100/100 |
| 非空 stderr | 0 |
| 第二次运行启动训练子进程 | 0 |

五方法各有 125 行。全部行满足 `noise_scope=train`，
`modified_count=modified_train_count`，且
`modified_val_count=modified_test_count=0`。原生四个最近工作方法均记录
为 X-only inference。

### 10.2 Uniform 0.4 主表

| 方法 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---:|---:|---:|---:|---:|
| PosteL-noIPL | 47.09±1.81 | 45.84±2.48 | 36.51±1.01 | 84.74±1.51 | 80.59±0.94 |
| GSDN 共同协议适配 | **57.45±3.82** | 50.68±3.31 | **41.64±1.67** | 89.41±0.22 | **84.55±0.78** |
| ALS MLP 适配 | 45.20±1.24 | 43.77±2.76 | 31.91±1.05 | 88.55±0.86 | 73.41±2.75 |
| SALS MLP 适配 | 48.27±2.50 | 47.20±1.69 | 37.06±0.78 | 88.85±0.82 | 77.86±0.79 |
| GB-LCSS | 56.28±1.54 | **53.52±2.71** | 37.30±0.57 | **90.62±0.47** | 81.33±0.70 |

GB-LCSS 在 CiteSeer 和 Coauthor-CS 上最高；GSDN 在 Cora、CoraFull 和
Amazon-Photo 上最高。PosteL、ALS 和 SALS 均未在该子表中取得列最高。

### 10.3 完整 25-setting 配对比较

| 对照 | GB-LCSS 平均差 (pp) | 正/负 setting | 至少 4/5 seeds 正向 | 25 项族内 Holm |
|---|---:|---:|---:|---:|
| PosteL-noIPL | +5.202 | 24/1 | 24/25 | 18/25 |
| ALS | +7.215 | 25/0 | 25/25 | 24/25 |
| SALS | +4.202 | 24/1 | 23/25 | 13/25 |
| GSDN | -0.698 | 10/15 | 8/25 | 2/25 |

GB-LCSS 相对 PosteL 的唯一负向单元为 CoraFull uniform 0.2
（-1.324 pp，0/5 seeds 正向，Holm \(p=0.0219\)）。相对 SALS 的唯一
负向单元也是 CoraFull uniform 0.2（-1.217 pp，0/5 seeds 正向，Holm
\(p=0.1549\)）。相对 ALS 的 25 个设置全部为正。

相对 GSDN，GB-LCSS 的数据集级五设置平均差为：

| 数据集 | GB-LCSS-GSDN 平均差 (pp) | GB-LCSS 正向设置 |
|---|---:|---:|
| Cora | -0.96 | 1/5 |
| CiteSeer | +2.69 | 5/5 |
| CoraFull | -5.07 | 0/5 |
| Coauthor-CS | +2.17 | 4/5 |
| Amazon-Photo | -2.33 | 0/5 |

GSDN 比 GB-LCSS 更强的差异主要集中于 CoraFull 与 Amazon-Photo。
25 项 GSDN 比较族中只有两个设置经 Holm 校正后拒绝零假设：
CoraFull uniform 0.2 支持 GSDN（GB-LCSS-GSDN=-6.526 pp），
Coauthor-CS uniform 0.6 支持 GB-LCSS（+3.438 pp）。其余差异应按效应量
和方向解释，不能写成显著胜负。

### 10.4 新发现

1. **直接最近工作改变了排名结论。** 在五方法联合表中，GB-LCSS 只在
   10/25 个设置取得最高均值；另外 15 个设置由 GSDN 取得最高均值。
2. **局部软目标路线仍有稳定价值。** GB-LCSS 对 ALS 为 25/25 正向，
   对 PosteL 和 SALS 均为 24/25 正向，说明其信号不能被一般邻域传播、
   平滑或后验统计完全吸收。
3. **逐节点在线约束与组级离线共识具有不同适用域。** GSDN 在
   CoraFull 和 Amazon-Photo 上稳定更强，GB-LCSS 在 CiteSeer 和
   Coauthor-CS 上更强。这与组级目标的分辨率限制相符，但当前实验不能
   单独识别因果机制。
4. **方差不能替代均值结论。** GSDN 在 Cora 等数据集上的标准差较大，
   但其在 15 个设置上均值更高。论文可以报告稳定性差异，不能据此把
   15/25 的均值劣势改写为总体优越。

## 11. 论文处置

- 表 1 的 23/25 结论保留，但必须明确限定为相对 JoCoR 这一通用同骨干
  抗噪基线，不得扩展到全部 X-only 最近工作。
- 正文表 3 替换为本报告第 10.2 节的五方法共同协议表。
- 历史 Graph-MLP、GLNN、FF-G2M、KRD 和 VQGraph 表降为补充材料中的
  资源分层诊断。
- 摘要、Highlights、引言、贡献和结论加入 GSDN 校准结果。
- 不再使用“整体最优”“优于全部同资源方法”或类似表述。
- 保留方法机制区别：标签无关冻结分组、仅聚合 noisy train labels、
  并列安全目标和固定预算的粒球实例。

## 12. 证据文件

结果目录：

`external/NoisyGL/runs/nearest_work_full_grid_20260728/`

关键文件：

- `raw_immutable/*.csv`
- `immutable_raw_manifest.csv`
- `recent_work_raw_combined.csv`
- `raw_combined.csv`
- `five_method_raw_combined.csv`
- `five_method_aggregate.csv`
- `completeness.csv`
- `protocol_audit.csv`
- `cross_vs_lcss_paired_deltas.csv`
- `cross_vs_lcss_paired_tests.csv`
- `command_manifest.csv`
- `report.md`

冻结 GB-LCSS v2 来源：

`external/NoisyGL/runs/tie_safe_full_grid_20260727_v2_immutable/raw_combined.csv`
