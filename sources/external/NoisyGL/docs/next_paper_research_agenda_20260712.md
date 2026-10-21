# 下一篇论文研究议程：LCSS 后续精进方向

## 1. 定位与纪律

当前 LCSS 论文的方法、固定粒球构造和主实验矩阵保持冻结。本文件记录投稿后或 rebuttal 明确要求时才启动的研究方向，不作为继续修改当前论文主算法的依据。

后续方向必须继续遵守以下边界：

- 不再通过数据集级调参追求粒球必须超过 kNN；
- 不复活动态 alpha、二值 gate、ball consistency、无标签球监督和模型预测反馈循环；
- 新方法必须包含同资源 robust-only、LCSS-only 和 shuffled-local-consensus 对照；
- 单 seed 只用于止损，正式结论必须使用同 split/noise hash 的 5-seed paired comparison；
- 必须区分“局部共识有效”和“粒球分组独有”，不得把可被 kNN/kmeans 平替的收益归因给粒球本体。

## 2. 已关闭的方向

| 方向 | 主要证据 | 决策 |
|---|---|---|
| 粒球变细、变粗、按规模或票数自适应 | MLP-13/24/25，holdout 不泛化 | 关闭粒度轴 |
| 动态 alpha、置信 gate、early-learning 保护 | 多轮从过度校正跳到几乎不校正，跨数据集不稳 | 关闭 |
| Ball consistency、无标签节点扩展 | 球级目标分辨率不足，新增损失无稳定收益 | 关闭 |
| GBSelect/样本筛选 | 没有稳定超过模型单视角选择 | 关闭 |
| 椭球、GMM、距离与模糊隶属 | 被 beta=0.5 或邻域对照吸收 | 关闭 |
| 让球式共识稳定超过 kNN | 蒸馏 G3 在 u0.4 下 15/15 seeds 未赢 kNN | 不再作为论文目标 |
| 用硬最小票数门忽略少票球 | MLP-50 正式 LOO 诊断：`s_min=5` 的 Brier 在 0/6 单元改善，平均变差 0.003886；零票球为 0 | 关闭硬门控，不再搜索阈值 |

## 3. 候选方向与优先级

### P1：LCSS soft target 与鲁棒学生优化

**研究问题**：局部共识负责修复监督信号，鲁棒损失负责减缓训练期记忆；两者是否存在超过任一单独机制的协同增益？

**当前空格**：项目已经比较了 `LCSS + soft CE` 和 `hard noisy label + robust loss`，但尚未比较 `LCSS soft target + robust optimization`。

**推荐顺序**：

1. LCSS + LogitClip：只改变 logits 的范数，soft-target 定义清楚；
2. LCSS + soft-small-loss：按逐节点 soft cross-entropy 选择小损失样本；
3. 仅当前两项出现稳定协同时，再研究双网络 soft-JoCoR。

**关键风险**：鲁棒损失和 target correction 可能处理同一批样本而产生冗余；在 uniform 0.6 下，有限票数形成的共识本身也可能偏置，鲁棒损失不能保证修复错误 target。

**最低证据要求**：组合同时超过 LCSS-only 和对应 robust-only，并稳定超过 shuffled-combination。

**MLP-28 更新（2026-07-12）**：`LCSS+LogitClip` 因 CiteSeer 回退 0.77pp 未过预注册门。`LCSS+soft-small-loss` 已完成 5-seed：在 Cora/CiteSeer/CiteSeerFull 上相对 `max(LCSS, Small-loss-only)` 提高 4.75/3.63/6.24pp，三项均 5/5 paired seeds 正向；real-shuffled 为 11.39/9.75/14.24pp，同样均 5/5 正向。因此 P1 已从“未测空格”升级为下一篇论文的稳定候选。Soft-JoCoR seed0 又比 soft-small-loss 高 0.53/3.48/1.89pp，但参数量约翻倍，仍需 5-seed 后才能判断是否值得替代更简单版本。

**MLP-29 广度更新**：在 CoraFull/Coauthor-CS/Amazon-Photo/PubMed/Amazon-Computers/Coauthor-Physics 的 uniform 0.6 seed0 上，direct soft-small-loss synergy 为 `+5.10/+2.65/+5.38/+6.50/+2.72/+2.66pp`，6/6 real-shuffled 均超过 1pp。它提供了跨规模/领域的强候选信号，但这些新增数据仍需多种子确认。

**MLP-30 噪声率适用域更新**：九数据集 seed0 网格显示，平均 synergy 随 uniform 噪声率从 `+0.46pp (0.2)` 增至 `+2.57pp (0.4)` 和 `+4.85pp (0.6)`；通过预注册门的数据集数为 `5/9、6/9、9/9`，8/9 数据集呈单调增强。结论应定位为“中高噪声下 target correction 与 robust selection 互补”，不能声称低噪声下普适改进。后续多种子优先补 0.4；0.6 已有三个数据集 5-seed，新增广度数据按代表性选择，不全量扩张；0.2 保留为边界证据。

**MLP-31 五种子终判**：uniform 0.4 在六个代表性数据集上平均 synergy `+2.84pp`，Cora/CiteSeer/CoraFull/PubMed 为 5/5 正向，Amazon-Photo 与 Coauthor-Physics 仅 3/5；预注册总体门通过，但必须写成条件性中噪声收益。uniform 0.6 在 Cora/CiteSeer/CiteSeerFull/CoraFull/PubMed/Coauthor-Physics 上 synergy 为 `+4.75/+3.63/+6.24/+5.02/+5.19/+3.27pp`，全部 5/5 正向；real-shuffled 为 `+6.15` 至 `+14.24pp`，同样全部 5/5。P1 因此升级为具有跨规模 paired 证据的高噪声主线候选，但仍须披露 `forget_rate=auto` 使用已知合成噪声率。

**MLP-32 蒸馏与图原生对比更新**：把 LCSS+soft-small-loss 叠加到标准 `GCN[X||H]` teacher 的 student 端后，seed0 只在 1/6 数据集超过 plain distillation，平均 `-0.50pp`，real-shuffled 仅 2/6 达到 +1pp；该等权融合关闭。完整五种子资源表显示 plain X-only distillation 相对 NRGNN 在 Amazon-Photo/PubMed 为 `+1.49/+4.11pp`，在 Cora/CiteSeer/CoraFull/Coauthor-Physics 为 `-1.53/-2.23/-3.25/-0.40pp`。因此蒸馏作为 accuracy extension 保留，但 claim 必须是 graph-free competitive trade-off，不是全面超越图原生方法。

**MLP-33 噪声类型与高噪声边界更新**：uniform 0.6 seed0 的最佳冻结 X-only 路线仅在 PubMed 胜过最佳图原生方法 `+0.78pp`；Cora/CiteSeer/CoraFull/Amazon-Photo 分别落后 `6.33/8.89/5.83/3.75pp`，仅 1/5 进入预注册 2pp 窗口，因此不扩图原生五种子。Pair 0.4 五种子平均 synergy 为 `+1.49pp`（4/5 数据集满足方向门），Instance 0.4 为 `+3.04pp`（5/5）；两种噪声的 real-shuffled 均为 5/5 数据集稳定通过。真实 u0.6 下将 forget rate 低估 0.1 的影响约为 `-1.03` 至 `+0.43pp`，低估 0.2 在 Cora/CoraFull 超过 1pp。P1 因此获得跨噪声类型支持，但仍是 graph-free 高噪声鲁棒方法，不是图原生 SOTA；已知噪声率依赖必须披露。

**MLP-34 强基线与机制更新**：PubMed 上组合相对 JoCoR 在 Pair/Instance 0.4 分别为 `+3.78/+4.05pp`，均 5/5 正向；uniform 0.6 仅 `+1.79pp、3/5`，因此只能写 Pair/Instance 的条件性边界消除。选择 trace 否定了“大量硬翻标签后保留被修复节点”的原假设：LCSS 的主要作用是移动 clean-class 概率质量、改善 Brier score 并重排样本损失，使 selected-target accuracy 在 Pair/Instance 平均提高 `3.60/6.60pp`，同时 final mask turnover 从 `0.280/0.243` 降至 `0.105/0.083`。noisy-only LOO 球投票噪声率估计在 Cora/CiteSeer/CoraFull 的绝对误差为 `0.120/0.132/0.233`，未过门并关闭，不再调第二个估计公式。

**MLP-35 clean-val 与重复性更新**：在 Cora/CiteSeer/CoraFull/Amazon-Photo/PubMed 的 uniform 0.2/0.4/0.6、seeds 0-4 上，clean-val 从 `{0,0.2,0.4,0.6}` 选择 forget rate。15/15 个 setting 的五种子均值都在 oracle 1pp 内，全局 selected-oracle 为 `+0.11pp`，最差 `-0.46pp`；相对固定 0.5 有 12/15 非负，全局平均 `+1.26pp`。该变体可以解决“训练时必须知道精确注入率”的实验依赖，但仍需要干净验证集，且只能称为训练强度选择，不能称噪声率估计或完全 noise-rate-free。三组同 seed 三重复审计的最大方法内 range 为 `0.10pp`，paired delta 均 3/3 同向，因此 MLP-34 的 2.76pp trace 差异属于局部/偶发现象，不应当作全局 ROCm 方差带。

### P2：噪声鲁棒的 graph-free teacher distillation

**研究问题**：如何在 noisy-label 条件下训练可靠 GNN teacher，并把其知识迁移给 X-only MLP student？

**已有信号**：MLP-9 在 uniform 0.6 下，denoised teacher 相对 bare teacher 在 Cora/CiteSeer/PubMed 均有 5-seed 增益；但球式 teacher 没有稳定超过 kNN teacher。

**论文归属**：该方向更适合定位为 noisy-label graph-to-MLP distillation，而不是粒球专属方法。局部共识只能作为 teacher 去噪候选之一，必须保留 no-consensus 和 kNN 对照。

**MLP-29 负结果边界**：把 direct soft-small-loss 原样用于 beta0.5 GCN[X||H] teacher 后，5/6 数据集数值提高，但 PubMed 相对 plain teacher 回退 `15.94pp`，整体平均为 `-1.02pp`，且只有 3/6 real-shuffled >=1pp。该普适插件方案关闭，不扩 5-seed。后续若研究 P2，必须重新处理 teacher 图传播下的样本覆盖/类别平衡，而不是继续调同一 forget schedule。

### P3：噪声转移感知的局部共识

**研究问题**：当前组内投票得到的是含噪分布 `q_B`，能否显式建模 `q_B = T^T p_B`，估计噪声转移矩阵 `T` 后恢复潜在干净分布 `p_B`？

**动机**：该方向直接针对 pair/instance noise 下系统性类别翻转，而不是继续调球数、margin 或 alpha。

**风险**：转移矩阵估计本身可能不稳定；同样存在 kNN 局部统计的镜像实现。若实施，贡献应表述为 transition-aware local consensus，而非粒球优越性。

### P4：面向局部可聚类性的 label-free 表征学习

**研究问题**：哪些自监督目标能够同时维持线性可分性和局部可聚类性？

**历史约束**：BGRL、GraphMAE、CCA-SSG、learnable temperature 与 GRACE+uniformity 已有负结果；probe 高不代表球几何有效。任何新表征必须先过 label-free geometry gate，再进入下游。

**判断**：创新潜力高、风险和成本也最高，适合作为独立论文，不是当前 LCSS 的廉价增强。

### P5：共识 teacher 的适用条件

**研究问题**：为什么共识 teacher 在高噪声小图有效，却在 CoraFull 等设置被 bare/kNN teacher 吸收？

**建议指标**：teacher entropy、teacher calibration、ball-GCN disagreement、局部可聚类性、共识覆盖率和原始 teacher 错误相关性。

**用途**：优先作为机制分析；只有出现不依赖 clean/test 标签的可靠触发信号后，才考虑自适应 teacher 路由。

## 4. 推荐路线图

1. 当前论文：冻结 LCSS，完成忠实 baseline、引用、事实句和统计审计。
2. P1 已完成跨噪声率、跨噪声类型、强基线、机制 trace 和 clean-val 选择验证；进入第二篇写作，不再增加同类公式调试。
3. clean-val tuned 版与 known-rate 版都需保留，分别报告部署假设与额外训练开销。
4. P2 的 direct robust-teacher/plugin 已判负；plain distillation 只作为独立 accuracy extension。
5. P3/P4 只在有独立研究周期时启动，不与第二篇写作串行。

## 5. Claim 上限

即使某个组合提高精度，也只能在完成 paired 5-seed、robust-only、LCSS-only 和 shuffled-combination 四类对照后写“二者具有协同作用”。单 seed 或只超过 CE 的结果只能写成候选信号。
