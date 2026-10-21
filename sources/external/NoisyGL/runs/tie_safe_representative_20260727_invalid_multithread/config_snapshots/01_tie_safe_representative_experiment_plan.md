# Tie-safe 代表性对照实验计划（运行前冻结）

## 1. 状态与触发证据

- 计划状态：`FROZEN_BEFORE_TRAINING`
- 冻结日期：2026-07-27（Asia/Shanghai）
- 目标期刊：Information Sciences
- 触发证据：
  - `paper_pipeline/00_evidence/tie_safety_audit.csv`
  - `paper_pipeline/00_evidence/tie_safety_audit_report.md`
- 审计事实：125/125 个历史运行可重建；124/125 存在会进入软目标的多数类并列；受影响训练节点占 4.762934%，受影响 `Σα` 占 2.238636%；108/125 个运行超过预先规定的 1% material 阈值。
- 审计边界：上述结果证明目标构造对类别编号敏感，但没有证明测试准确率偏差的方向或幅度。

## 2. 研究假设

`H1`：将多数类并列时的最小类别索引 one-hot 分量替换为类别置换等变的处理后，目标张量在类别置换及逆置换下保持一致。

`H2`：估计并报告预先固定的 tie-safe 处理在并列暴露较高设置中的性能变化及其不确定性；本实验不作等价性或非劣效性声明。

本实验是实现安全性和结果敏感性检查，不是新方法搜索，不用于增加主论文的性能胜场。

## 3. 粒球扮演的角色

粒球仍仅作为局部语义关系单元和软置信度来源。它提供球内训练标签分布 `p_j` 与混合强度 `α_j`；并列处理不得把球成员关系改成 hard positive label，也不得修改构球、数据划分或评估协议。

## 4. 方法动机与三个对照

当前实现：

`q_j = (1 - β) p_j + β onehot(argmax_c p_jc)`。

当多个类别达到相同最大计数时，`argmax` 选择最小类别索引。对于 `C > 2`，此时 `α_j` 仍可能大于 0，故 one-hot 分量使目标不具类别置换等变性。

固定比较三个模式：

1. `min_index`：历史实现；并列时保留最小索引 one-hot。它是性能控制，不是推荐修复。
2. `alpha_zero`：并列球内令 `α_j = 0`，即这些节点保持原始 noisy-label one-hot。
3. `distribution_only`：并列球内令有效 `β_j = 0`，即 `q_j = p_j`，保留原 `α_j`。

除上述并列分支外，三个模式必须逐元素一致。非并列球必须与历史实现逐元素一致。

主修复在看到任何新训练结果前固定为 `distribution_only`。理由是它删除依赖类别编号的 one-hot 分量，同时保留可置换的球内经验分布 `p_j` 和原混合强度 `α_j`。`alpha_zero` 仅作为“完全放弃并列球干预”的机制消融；测试集结果不得用于把主修复切换为 `alpha_zero` 或退回 `min_index`。

并列谓词固定为：对球内训练标签的**整数类别计数**，若等于最大整数计数的类别数不少于 2，则该球为 tie；不对浮点分布使用 `isclose` 或任何容差。

## 5. 代码改动

计划中的最小改动：

- `external/NoisyGL/utils/frozen_grace_targets.py`
  - 为 uniform ball soft-target 路径增加显式 `tie_mode`；
  - 默认值必须是 `min_index`，保证历史配置向后兼容；
  - 输出 tie 数量、受影响节点和受影响 `Σα` 等诊断。
- `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`
  - 只负责从 `gb.tie_mode` 透传到目标构造器。
- 十个既有配置（两个方法 × 五个主数据集）
  - 方法：`frozengrace_gb_mlp_softdist_beta05`、`frozengrace_gb_mlp_softdist_beta05_shuffled`；
  - 数据集：Cora、CiteSeer、Coauthor-CS、CoraFull、AmazonPhoto；
  - 增加默认 `tie_mode: min_index`，使命令行 `config_overrides` 可显式覆盖。
- 测试
  - 先增加失败测试，再实现；
  - 覆盖默认向后兼容、两种 safe 模式、非并列不变性、类别置换等变性和非法模式拒绝。
- 新增一个独立运行/汇总脚本，保存精确命令、配置、日志、原始 CSV、配对表、环境指纹和报告。

不允许改动：GRACE embedding、构球 assignment、split、noise realization、MLP 结构、优化器、早停、validation/test mask 或指标实现。

## 6. 配置与运行范围

### 6.1 代表性设置

- 数据集：Cora、CoraFull
- 噪声：uniform，rate = 0.6，仅污染训练标签
- seeds：0, 1, 2, 3, 4
- 方法骨架：`frozengrace_gb_mlp_softdist_beta05`
- 三个 `tie_mode`：`min_index`、`alpha_zero`、`distribution_only`
- 总运行数：2 × 5 × 3 = 30

选择理由：tie-safety 审计中，Cora uniform0.6 seed 2 的受影响节点占比最高（32.163%）；CoraFull 提供类别数更多的大图压力测试。数据集选择完全基于训练前的 tie 暴露与类别规模，不参考任何 safe-mode 性能结果。

### 6.2 必须固定的协议

- SSL：GRACE；使用同一冻结 embedding 路径
- split：percent = 20% train / 10% validation / 70% test
- noise scope：train
- 同一 dataset/seed 下三个模式必须具有相同：
  - split fingerprint；
  - noise fingerprint；
  - modified label count；
  - embedding path/hash；
  - ball assignment hash；
  - 模型、优化器、epoch、patience 和 checkpoint selection。
- 每个模式均重跑，`min_index` 不只复用历史均值；同时将其与历史冻结行核对，以发现环境或代码漂移。
- 每个模式必须在独立进程中重新设置 seed；保存 Python、PyTorch、ROCm/CUDA、GPU、代码 commit、dirty diff hash 与确定性相关环境信息。

## 7. 预期机制

- `alpha_zero` 应完全删除并列球对目标的干预，因此受影响节点的最终目标等于原始 noisy-label one-hot。
- `distribution_only` 应只删除有编号偏置的 majority one-hot 分量，同时保留由球内分布产生的软监督。
- 两个 safe 模式在类别置换后再逆置换，目标张量和 `α` 都应恢复到原值（浮点容差 `atol = 1e-7`, `rtol = 0`）。
- 非并列球和 `min_index` 默认路径必须与历史行为逐元素一致。

## 8. 主要统计量

对每个 dataset × safe mode，以同 seed 的 `min_index` 为配对控制：

- 测试准确率 mean ± sample std；
- 五个 seed 的配对差 `Δ_s = Acc_safe,s - Acc_min_index,s`；
- 配对差 mean、sample std、未作多重比较校正的 nominal 95% t 区间（df = 4，`t* = 2.7764451051977987`）；
- 正/负/零方向 seed 数；
- validation accuracy、selected epoch 与训练耗时作为审计字段，不用于选择最优测试结果；
- 目标机制量：tie balls、affected nodes、affected-node ratio（分母为训练节点数）、affected `Σα` ratio（分母为训练节点 `Σα`）、目标 L1 差（先对每个受影响训练节点求类别维 L1，再取节点均值）。

配对差的正/负/零计数使用比例尺度上的绝对容差 `1e-12`；小于或等于该容差记为零。test set 只用于报告预先固定方法与消融的外部结果，不参与方法、阈值、数据集或扩展范围选择。

## 9. 成功、失败与扩展判据

### 9.1 数学成功标准

两个 safe 模式都必须通过：

- 类别置换等变性测试；
- 非并列目标与历史实现一致；
- 默认 `min_index` 与改动前测试向后兼容；
- 每个实验行协议 fingerprint 一致。

任一失败即停止训练或判为实现失败，不解释性能。

### 9.2 性能敏感性描述

对任一 dataset × safe mode，分别报告以下两个描述性信号：

1. `statistical detectability`：未校正的双侧 nominal 95% t 区间是否包含 0；
2. `practical materiality`：`|mean Δ| ≥ 1.00` 个百分点，且至少 4/5 seeds 同方向。

两者均为审计触发的探索性敏感性描述，不用于选择主修复，也不构成确认性显著性、等价性或非劣效性证据。

### 9.3 固定的方法与消融角色

1. `distribution_only`：训练前固定的唯一主修复。
2. `alpha_zero`：固定的机制消融，不参加方法选择。
3. `min_index`：历史实现对照，不是候选主修复。
4. 任一 safe 模式未通过数学测试时，停止全部性能训练并修复实现；不得只排除失败模式后继续。
5. 无论性能变化方向如何，都不得利用本实验 test accuracy 改写上述角色。

### 9.4 完整 25-setting 网格

- 扩展决定在新训练前已经固定：只要第 9.1 节数学测试、代表性运行的协议 fingerprint 与运行完整性检查通过，就无条件执行完整网格。
- 数据集固定为：Cora、CiteSeer、Coauthor-CS、CoraFull、AmazonPhoto。
- 每个数据集固定五个噪声设置：
  1. uniform 0.2；
  2. uniform 0.4；
  3. uniform 0.6；
  4. pair 0.4；
  5. instance 0.4。
- 每个 setting 固定 seeds 0、1、2、3、4，共 25 settings × 5 seeds。
- 完整网格不复用 fingerprint 不完整的历史 baseline。为形成同轮可比的新主表，每个 setting × seed 固定共同运行四个方法：
  1. `frozengrace_gb_mlp_softdist_beta05` + `tie_mode=distribution_only`；
  2. `frozengrace_gb_mlp_softdist_beta05_shuffled` + `tie_mode=distribution_only`；
  3. `mlp_ce`；
  4. `mlp_jocor`。
- 因此完整网格的计划规模为 25 × 5 × 4 = 500 个 seed-level rows。四个方法必须共享相同 split/noise realization；两个 SoftDist 方法还必须记录 embedding 与 assignment fingerprint。
- 网格定义也由 `paper_pipeline/00_evidence/protocol_ledger.csv` 中筛选条件 `method=="frozengrace_gb_mlp_softdist_beta05" AND in_primary_statistics=="TRUE"` 的 125 行交叉验证。运行前参考文件 SHA-256 为 `73CEC921C092EC398FC13EC4E1C4A85A43BC3792298F702930222E1524ADCB90`；若当前文件哈希不同，必须停止并重新审计，不得静默采用新网格。
- 代表性 test accuracy 的方向、均值、CI 或 materiality 不参与该决定。
- 在完整网格结束前，历史主表必须继续明确标记为 `min_index` 实现；不得把它冒充为 `distribution_only` 结果。
- 完整网格完成后，论文主表只能引用新 `distribution_only` 行；历史 `min_index` 数值降为审计对照。
- 每个新行必须闭合 split hash、noise hash、modified train/validation/test counts、noise scope、数据比例与 config。任一方法 fingerprint 不一致时，该 setting 的四个方法全部不得进入主表。
- 新 `distribution_only` 与新 shuffled 行必须闭合 embedding path/hash；真实球与 shuffled 球的 assignment hash 按定义可以不同，但各自必须跨可比重复可追溯。若 assignment 生成不能确定性重建，该行标记为 GAP，不用于 GB-specific claim。
- 若完整网格中任一方法失败或缺行，必须保留失败行与原始日志；禁止用历史值补洞后当作同轮结果。

## 10. 失败标准

- split/noise/embedding/assignment fingerprint 在配对模式间不一致；
- 当前 `min_index` 的 split/noise/assignment hash 必须与可用历史证据完全一致；soft target 的回归测试必须逐元素一致。每 seed 测试准确率与冻结历史行的绝对误差容差为 `1e-6`（比例尺度，即 `0.0001` 个百分点），selected epoch 若历史行可用则必须完全一致。超限时停止论文数值替换，定位环境或代码漂移；代表性同轮配对结果可保留为诊断，但不得闭合历史复现；
- 有缺失 seed、NaN/Inf、异常退出或自动回退到 CPU/不同设备而未标注；
- 测试准确率被用于事后改变本计划的候选、阈值或数据集；
- 只报告均值，不保留 raw log、seed row、mean/std 和失败行。

## 11. 必须保存的产物

- 本计划及其 SHA-256；
- 代码 diff 与测试的 RED/GREEN 记录；
- 每个模式的完整 config snapshot；
- 完整命令、stdout/stderr raw log；
- 代表性阶段 30 行原始结果表及协议 fingerprint；
- 配对差表、mean/std/CI 汇总；
- 机制诊断表；
- 成功/失败报告；
- 完整网格 500 个计划行的 completeness 表，包含成功、失败和未运行状态；
- 完整网格每个 seed 的 raw row、config snapshot、stdout/stderr、split/noise/modified-count/embedding/assignment fingerprint；
- 完整网格 dataset × setting × method 的 mean、sample std、seed 列表与配对差；
- 新论文表格每个数值单元到 raw rows、聚合公式、源文件字节数和 SHA-256 的 lineage 映射；
- 任一缺失或失败行的显式清单，禁止静默丢弃；
- 最终决策：`继续`、`修改`、`暂缓` 或 `放弃`；
- 与论文 claim ledger、issue traceability 和 v30 文本的映射。

## 12. Claim 边界

- 已支持：历史 tie-break 非类别置换等变，且目标暴露规模达到 material。
- 待本实验验证：两种 safe 处理是否在代表性设置下保持性能。
- 不得声称：历史准确率因最小索引而系统性高估或低估；代表性两数据集足以证明全部 25 settings 等价；未重跑主表对应 tie-safe 实现；nominal CI 包含 0 证明等价或非劣。
- 未触发 detectability 或 materiality 仅表示这 10 个配对运行中未观察到预定义信号，不构成等价性或非劣效性证明。

## 13. 实验结果

`NOT_RUN`

运行前不得填写结果。运行后只在本节追加带时间戳、产物路径和哈希的事实，不改动第 1–12 节的冻结设计。

## 14. 最终决策

`PENDING`

允许的最终状态：`继续`、`修改`、`暂缓` 或 `放弃`。决定必须同时引用数学测试、协议完整性、代表性结果与完整 25-setting 网格（若已进入该阶段）。
