# 第二篇论文候选方向统一评分与排序

> 日期：2026-07-14
> 决策假设：用户尚未指定“新颖性优先”或“成功率优先”，本轮采用可撤销的**平衡策略**——主线优先选择重复风险较低、可快速证伪的方向；备选保留已有正信号，但不把高重复方向包装成主创新。
> 输入证据：`research_asset_audit.md`、`literature_matrix.md`、`复盘_open-intent方向收口_20260630.md`、`next_paper_research_agenda_20260712.md` 及对应原始运行目录。

## 1. 评分协议

所有候选使用相同的 1–5 分量表，5 为最好。总分为加权平均：

| 维度 | 权重 | 5 分含义 | 1 分含义 |
|---|---:|---|---|
| N：新颖性与文献空白 | 20% | 未发现近似问题/机制，差异清楚 | 已有直接同构论文 |
| G：粒球不可替代性 | 20% | 研究问题天然需要自适应多粒度 GB | kNN/KMeans/GMM/单原型可直接平替 |
| E：本地证据 | 15% | 多数据集、多 seed、强对照已有正证据 | 已有多数据集稳定负证据 |
| F：实现与数据可行性 | 15% | 现有代码/特征/数据可直接复用 | 需新数据、大规模重构或新领域管线 |
| T：可证伪性 | 10% | 一次小 Pilot 即可明确 Go/No-Go | 需长训练才能解释失败 |
| P：论文故事完整度 | 10% | 问题、方法、对照、分析均可形成闭环 | 只能成为小技巧/附录分析 |
| C：成本适配 | 10% | CPU/单卡、数小时内完成 Pilot | 多卡、长周期或复杂外部依赖 |

计算式：

`Score = 0.20N + 0.20G + 0.15E + 0.15F + 0.10T + 0.10P + 0.10C`

除总分外设置三条**否决门**：

1. **重复门**：若核心方法与 2022–2026 正式论文高度同构，不能作为主方向，即使实验分高。
2. **平替门**：若 size-matched KMeans/kNN/GMM 在同一表示上稳定不弱于真 GB，停止精度型 GB claim。
3. **负证据门**：已有两个数据集或五个 paired seeds 的一致反证时，除非研究问题发生实质变化，否则不得通过换分数/换超参复活。

## 2. 候选方向定义

### D1：资源约束的粒球 Coreset 深度 kNN-OOD（GB-Coreset）

**研究问题**：在冻结深表示上，能否用自适应多粒度粒球把 IND memory bank 压缩到 1%–20%，在给定 AUROC/FPR95/known-accuracy 容差下，比随机、KMeans、k-center 和类原型保留更多 OOD 检测质量并降低检索成本？

**粒球角色**：不是伪标签、正样本或“更准的边界”，而是带半径/密度/边界壳样本的**自适应 coreset 单元**。

**最小方法**：

- 每个球保留 center/medoid、radius、size 和少量 boundary-shell representatives；
- 查询分数由最近代表距离与球半径归一化距离构成；
- 以固定 memory budget 比较相同代表数的 random、class prototype、KMeans medoid、greedy k-center 和 GB-Coreset；
- 不训练 encoder，不改变原 split，不用 test OOD 调阈值。

**最近竞品/重复边界**：Deep kNN 是性能基线；PatchCore 已用 greedy coreset 压缩 nominal patch memory；cDNP 已比较 random、greedy coreset 和 per-class greedy coreset 的 dense NN-OOD；SupLID 已提出几何 coreset + 实时 post-hoc OOD；GBOC/GBDO 又覆盖粒球原型和局部密度。因此“budget-constrained deep OOD coreset frontier”本身不是空白，仅把 coreset selector 换成 GB 不足以形成主创新。

**本地证据**：现有 open-intent Pilot 证明 GB 不具独立精度优势，但尚未比较 memory/延迟/质量 Pareto 曲线。负结果反而支持将问题改为压缩，而不是继续争精度。

**MVP**：复用 BANKING77、StackOverflow 落盘 `features_logits_seed*.npz`，先做 seed0×5 budgets；只有至少一个数据集上 GB-Coreset 在同 budget 下超过最强非 GB coreset ≥1 AUROC point，且不恶化 known accuracy >0.5pp，才扩多 seed。

**停止条件**：两个数据集上均被 KMeans/k-center 支配；或仅在比对方保留更多代表时获益；或速度收益小于 2× 且 AUROC 损失超过 1pp。

### D2：多视图稳定性校准的粒球可靠性（Stable-GB Reliability）

**研究问题**：粒球划分跨 GRACE view/seed/embedding perturbation 的持久性，能否作为标签无关的 node reliability，改善 noisy-label 图分类中的 target mixing 或置信校准，并提供 kNN/loss-history/model-confidence 之外的条件信息？

**粒球角色**：只估计局部结构的**稳定性/可靠性**，不直接生成 hard positive、hard label 或负样本过滤。

**最小方法**：对多个冻结表示视图构球，计算 node co-assignment stability、boundary-margin stability、ball-size stability；用固定单调映射调节 LCSS soft target 权重，保留 no-GB、single-view GB、kNN stability、shuffled controls。

**最近竞品/重复边界**：BNO-GB 的球级 outlier + 样本级 consistency、ProCon 的 peer consistency、RELGCL 的 relative relation 都是强近邻。必须证明“跨视图球稳定性”是独立变量，而非局部一致性换名。

**本地证据**：LCSS 有强下游信号，但动态 alpha/gate、ball consistency、kNN 平替已多次失败；稳定性诊断本身尚未被严格做成 out-of-fold conditional-gain test。

**MVP**：Cora/CiteSeer uniform0.6 seed0，冻结 3 个 GRACE views；先只做 reliability AUC/Brier/partial correlation 和 tiny calibrator。若控制 loss、kNN agreement、teacher confidence 后的条件增益 <0.01，立即停止，不进入训练。

**停止条件**：两个数据集条件增益均不过线；或 shuffled/view-randomized stability 等效；或自适应 mixing 不超过 fixed LCSS 且增加明显复杂度。

### D3：LCSS soft target + soft-small-loss（现有强结果保底）

**研究问题**：局部软监督修正与小损失选择在中高噪声下是否存在稳定协同？

**粒球角色**：从冻结 GRACE 空间提取局部 soft target；小损失选择抑制训练期记忆。

**最近竞品/重复边界**：MgBIF、MgCNL 已做粒球局部标签融合/修正 + 高置信样本选择；GBmooer 做 contrastive hypersphere + GB local label smoothing；ProCon 在图上做 peer consistency + GMM/prototype feedback。

**本地证据**：uniform0.6 六数据集 5/5 paired seeds 正向；uniform0.4 六数据集平均 synergy +2.84pp；pair/instance0.4 也有多 seed 支持；clean-val 选择接近 oracle。

**定位限制**：它可以成为可靠的工程结果或第一篇的扩展实验，但在没有新的科学问题和直接近邻复现对比前，不能作为第二篇主创新。

**MVP**：已基本完成；剩余任务是 novelty audit、ProCon/MgBIF/MgCNL 功能对照和成本/假设披露，而非继续调公式。

**停止条件**：直接论文写作前若方法差异无法用一句可检验陈述区分于 MgBIF/MgCNL/ProCon，则降级为 backup/附录。

### D4：噪声转移感知的局部软共识（Transition-aware LCSS）

**研究问题**：局部观察到的是含噪分布 `q_B = T^T p_B`；能否用全局/类条件转移矩阵恢复潜在干净分布，再做保守 soft correction，从而专门改善 pair/instance noise？

**粒球角色**：提供多个局部观测分布 `q_B`，不直接决定 clean label。

**最近竞品/重复边界**：大量 noisy-label transition-matrix 方法已存在；粒球部分又接近局部 label smoothing。真正差异只能是“多局部观测使 T/p_B 联合估计更可识别”。

**本地证据**：pair/instance0.4 的 LCSS+small-loss 已有正信号，但 noisy-only LOO 球投票的噪声率估计误差 0.120–0.233，说明从球统计估噪并不可靠。

**MVP**：只在现有 synthetic transition 已知时做 oracle-T 上限；若 oracle-T 相比原 LCSS <1pp，则不再研究 T estimation。

**停止条件**：oracle-T 上限不足；或 kNN-local transition 同等；或需要 clean validation/test 标签才能稳定求解。

### D5：粒球条件残差证据的开放集校准（GB Conditional Calibration）

**研究问题**：给定 MSP/Energy/kNN/prototype/Mahalanobis 等强分数后，球半径、密度、边界重叠和 purity proxy 是否仍提供可重复的条件 OOD 信息？

**粒球角色**：作为可解释的 residual features，不单独充当拒识器。

**最近竞品/重复边界**：BNO-GB、GBDO、MOGB；另外 OOD 分数分解研究警示不同 score 可能只是相同核心量重参数化。

**本地证据**：BANKING77 的 cover/density/purity/overlap AUROC≈0.5，组合低于 `s_dist`；7 组超参中 real GB 对 dist+margin 和 ALL_safe 均 0/7 胜 KMeans；StackOverflow real 与 KMeans 逐位相等。

**MVP**：现有 `offline_probe_generic.py` 已实质完成关键检验。

**停止条件**：已满足——两个数据集被平替/退化。该项保留为**已完成的优先验证方向和 No-Go 结论**，不进入新实现。

### D6：持续/流式开放集中的增量粒球记忆（Streaming GB Memory）

**研究问题**：在 known intents 持续加入、数据分布漂移且不能保存全部样本时，增量 split/merge 粒球能否同时控制 memory、forgetting 与 unknown detection？

**粒球角色**：可更新的局部结构记忆与漂移摘要。

**最近竞品/重复边界**：FedOC-GB 已覆盖分布式粒球知识库；GBOC 覆盖 local prototypes；持续 OOD/continual intent 还需单独系统检索。

**本地证据**：无现成流式协议和数据流水线；只能复用部分表示/球构造代码。

**MVP**：需先构建 stream protocol，超出当前最低成本 Pilot。

**停止条件**：若没有明确 replay/memory budget 场景或需大规模新基线复现，则本周期不启动。

## 3. 统一评分

| 排名 | 方向 | N | G | E | F | T | P | C | 总分 / 5 | 否决门 | 决策 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | D3 LCSS + soft-small-loss | 2.0 | 3.0 | 5.0 | 5.0 | 5.0 | 3.0 | 5.0 | **3.80** | **触发重复门** | 保底工程结果，不作主创新 |
| 2 | D2 Stable-GB Reliability | 4.0 | 3.5 | 3.5 | 3.5 | 4.0 | 4.0 | 3.5 | **3.70** | 未触发；平替风险高 | **主研究方向（待设计审批）** |
| 3 | D4 Transition-aware LCSS | 3.5 | 3.0 | 3.0 | 3.5 | 4.0 | 4.0 | 3.5 | **3.43** | 估噪负信号；重复风险中高 | **备选研究方向：只做 oracle 上限** |
| 4 | D1 GB-Coreset 深度 kNN-OOD | 2.5 | 3.0 | 2.5 | 4.5 | 5.0 | 3.0 | 4.5 | **3.40** | **触发重复门**：PatchCore/cDNP/SupLID | 效率型工程分析，不作主创新 |
| 5 | D5 GB Conditional Calibration | 3.5 | 2.0 | 1.0 | 5.0 | 5.0 | 2.5 | 5.0 | **3.25** | **触发平替门与负证据门** | **已完成 No-Go** |
| 6 | D6 Streaming GB Memory | 4.5 | 4.0 | 1.0 | 2.0 | 3.5 | 4.0 | 2.0 | **3.10** | 当前成本/证据不足 | 长期储备，不在本周期启动 |

## 4. 主方向与备选方向

### 4.1 主方向：Stable-GB Reliability for Noisy Graph Learning

选择理由：

1. 它直接遵守项目当前偏好：粒球只做置信度/soft structural prior，不做 hard label、hard positive 或强 prototype pulling。
2. 核心问题不是再次证明“球内共识有效”，而是检验**跨无标签表示视图的球划分持久性**是否能预测局部修正何时可靠，并在控制 kNN agreement、loss history、teacher confidence 后仍有条件信息。
3. 第一阶段是 diagnosis-only：只用冻结 GRACE views 与既有结果做 out-of-fold reliability audit；条件增益不过线就停止，不写训练插件，成本可控。
4. 它仍有 BNO-GB/ProCon/RELGCL 与本地 dynamic gate 失败的高风险，因此主方向只是“最值得先证伪”，不是预设成功。

主张上限：即使 diagnosis 成功，也只能先写“cross-view GB persistence provides conditional reliability information beyond non-GB local/confidence signals”；只有预注册训练实验进一步稳定改善 calibration/accuracy，才允许写成方法贡献。

### 4.2 备选研究方向：Transition-aware LCSS（oracle-first）

选择理由：pair/instance noise 下观察球分布与潜在干净分布之间存在明确的 transition 问题，可先用已知 synthetic `T` 测 oracle 上限。若 oracle-T 相比原 LCSS 仍不足 1pp，立即停止；只有上限足够才讨论估计 `T`，避免先陷入不稳定噪声率/转移矩阵估计。

### 4.3 保底成果：LCSS + soft-small-loss

该方向拥有最强本地效果证据，但文献重复门已触发。它适合：

- 作为第一篇论文扩展或独立技术报告；
- 作为主/备选方向失败时的可交付实验结果；
- 用于说明 target correction 与 robust selection 的条件性协同。

在没有直接对照 MgBIF/MgCNL/ProCon 并形成明确任务级差异前，不应把它确定为第二篇主线。

## 5. 设计审批与执行边界

按照 brainstorming 工作流，应先向用户展示 D2 的完整设计，包括组件、数据流、公平协议、统计检验、错误处理和测试；`method_design.md` 已完成该步骤，并于 2026-07-14 获用户确认。该确认只批准设计边界，不等于 Stable-GB 已有实验支持；任何 A0/A1 实现仍必须使用独立计划、目录和原始输出。

若用户否决 D2：优先执行 D4 的 oracle-only 上限检验，而不是回到已判负的 D5、已被 coreset 文献占据的 D1，或直接把 D3 包装成新论文。
