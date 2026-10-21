# 子图空间与图 SSL 方向：历史去重与机制审计

审计日期：2026-10-13。范围是本地原实验历史、当前重设计记忆和提交稿快照；没有训练、改代码或改评测。

路径缩写：`P = C:/Users/Admin/Desktop/实验8去噪5.19-`；`W = P/gb_c1_ratio_success_20260609/results_ratio_c1_20260609`。

## 结论

**本地历史没有找到“以节点为根的 ego-subgraph，在两种图增强视图之间做无监督稳定性学习，同时保留根节点表征、只把粒球当软区域上下文”的实验。** 这应记为“本次受限历史检索未命中 / 尚无已做证据”，不能写成已证明从未做过，也不能据此声称文献首创。

最接近、必须去重的已有工作分四类：

1. **GB 直接改图增强：已试且在 Cora 上失败。** 按同球/跨球关系控制边保留与扰动，试过层级采样、后期衰减、冻结 hierarchy、anchor/explore；这是若新方案通过保留社区内边、删除社区间边来制造稳定子图，最直接的重复风险。
2. **GB 直接提供正关系：已试过但证据等级有限。** prototype pulling、ball consistency、same-ball soft positives 在 Cora/CiteSeer 做过 seed-0 sanity，未达验证集 stop/go；这些不是“根节点 ego-context 跨增强一致性”，但如果新损失实际把同球根节点/邻居都拉近，机制会重回旧失败。
3. **双空间 / 表示几何：部分已试，任务和梯度路径不同。** X/H 粒球冲突是离线噪声检测并判 STOP；GRACE+显式 uniformity 在 Cora/CiteSeer 几何门上失败；CCA-SSG、GraphMAE 等 encoder swap 的球几何也出现单球退化。它们没有测试 rooted-subgraph 稳定性。
4. **社区与子图：没有社区保持增强实验命中。** Phase 41 的 `communities` 是 GB 球图上的模块度凝聚，用于免 K 聚类，三数据集 NO-GO；唯一检索到的 `subgraph` 是 VQGraph neighbor-sampling 的局部邻接重建工程实现，既不是 GB，也不是图 SSL。

## 历史去重台账

| 机制 | 本地证据与状态 | 与新候选的边界 |
|---|---|---|
| GB-view / edge augmentation | **已做，负结果。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons.md` L36–145：按 same-ball/cross-ball 控制边保留、层级采样、edge perturbation；Cora 的长训 GB-GCL 0.7656±0.0109，GRACE 0.7746±0.0113；late-recovery 0.7482，freeze 0.7460，anchor/explore 0.7354±0.0146。文档判 abandon，并把视图分布漂移列为解释。 | 候选若重写 edge_index、按球/社区决定保留率，属于老轴的再实现。若只用固定增强，从每个根节点的邻居中池化一个 context 并加小权重一致性项，梯度目标不同，但仍须证明不是“把 GB-view 从输入端搬到 loss 端”。 |
| Direct GB positive / prototype / consistency | **做过，sanity 负向，不是正式多 seed 结论。** 同一历史 L317–425：Cora/CiteSeer seed-0；prototype、consistency、top-1/top-2 shared-ball positive 均无 validation gain，Cora test 多数下降。记录明确说只够 stop/go，不能断言所有关系目标失败。 | 根节点 context 是同一 root 跨视图对齐，不等于同球节点互为正样本；但若区域项造成球内节点相互拉近，机制实质重叠。必须单独报告 root 保持与节点表示保真。 |
| 非 InfoNCE / BGRL、GBT | **GB-view + 换目标做过，不能外推算法家族。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons 2.md` L509–574：Cora 五 seed，GB-GBT 0.7304±0.0138，GB-BGRL 0.4164±0.0456；同时换了目标、视图和 teacher/student，归因混杂。 | 新候选应保留 GRACE 主目标，最多加一个 context 辅项；不要第一轮再换 BGRL/GraphMAE/MVGRL 或增强器。 |
| CCA-SSG / GraphMAE / BGRL encoder swap 与显式几何目标 | **相邻但非 rooted-subgraph。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons 11.md` L231–249：CCA-SSG、GraphMAE 的 H 构球时退化成一个球；Phase 20 的 GRACE+uniformity 也出现一球。`... 12.md` L4–10：λ={0.1,0.5,1.0} 的 Cora/CiteSeer 几何 gate 全失败，Phase 20b 按预注册规则未做下游。 | 这否定的是这些具体替换/正则和当前分球器下的候选，不是否定所有 label-free 表示学习；新方法不能只凭 probe 或 alignment/uniformity 指标过门，必须实际检查 H′ 上的球数、成员稳定性及下游 GB-LCSS。 |
| X/H 双空间冲突 | **已做，离线诊断判 STOP。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons 18.md` L1–15、L87–124：X 与 frozen H 各自构球后比较局部投票冲突；primary 条件 AUC 接近随机，且冲突节点噪声率反而更低。 | 这是“两个空间的冲突能否做噪声分数”，不是联合优化两个几何空间，也不是子图跨增强表示学习。不能把它写成双空间 SSL 已被否定。 |
| 社区 / community | **聚类用过，增强未命中。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons 19.md` L33–44、L105–130：NetworkX greedy modularity communities 凝聚球图；Cora/CiteSeer/PubMed 五 seed，GB 聚类未胜 true-K K-means，也未胜 auto-K K-means，Phase 41 NO-GO。43 个历史文件中 `community-preserving/community-aware augmentation` 搜索为 0 命中，`communities` 只在此处命中。 | 凝聚球到社区是离线分簇，不是训练增强。社区保持增强仍属未命中；但若操作就是“保内边、扰跨边”，与 DIR-01 的 same-ball/cross-ball edge keep 已高度重叠。 |
| 子图级 | **唯一命中与本题无关。** `P/experiment_history_and_lessons/02_experiment_history_and_lessons 38.md` L1–21：VQGraph 在 neighbor-sampled 子图上做局部 dense-adjacency reconstruction；L9 明确 GB 不属于 VQGraph。 | 是采样执行/内存路径，不是 rooted-subgraph SSL，也不是跨增强稳定目标。 |
| 当前 GB-LCSS 的固定 H 局部票 | **已做，是当前主方法流程。** `W/research/algorithm_redesign_20261013/99_研究记忆与接续.md` L14–18：无标签 GRACE → 冻结预投影 H → 无标签递归分球 → 训练节点观测标签留一投票 → 置信混合软目标 → 下游 X-only MLP；扩展有 GCN teacher/distillation。 | 球投票改变的是下游监督 target；H 不从该票损失接收梯度。候选需保留此下游流程不变，单独检验前置 SSL 是否改善 H′。 |
| 静态替代表征 / propagation | **A²X 已有负对照，不能遗漏。** 提交稿 `W/research/algorithm_redesign_20261013/paper_snapshot/GB-LCSS_PR_submission_FINAL_20260930/main.tex` L494–500、`table_6.tex` L11–19：比较 GRACE、X、A²X 的固定表示分球与下游准确率；X 在五数据集都只有一球，A²X 仅在 Coauthor-CS/Amazon-Photo 形成多球，GRACE 在全部数据集形成多球且平均准确率最高。 | 这是静态传播替换分球空间，不是一个可训练的几何目标；说明“换空间”本身已有对照，也提示新空间要和球形成/下游用途一起评价。 |

### 当前稿对“空间效应”的关键补充

不能把所有问题归因于 GRACE 表示几何。提交稿默认 `B_max=64`（`main.tex` L416），CoraFull 是 70 类图（L392）；附录 I.1 将 `B_max` 提到 128 后，CoraFull 构出 128 个球，五个 seed 都提升，配对增益 `1.53±0.40` 个百分点（`supplementary.tex` L190–200；`table_I_1.tex` L25–26）。因此，至少在 CoraFull，球预算/分辨率是一个已证实的贡献来源，不能只说表示空间不合适。候选空间比较必须交叉控制 `B_max={64,128,256}`，并在每个预算下与同球数预算的 K-means 对照；256 是建议的新对照，历史中未见已跑证据。

## 与固定 H 后处理的机制区别

| 做法 | 作用位置 | 梯度是否回到 GRACE encoder | 主要监督/信号 |
|---|---|---|---|
| 当前 GB-LCSS 局部留一投票 | frozen H 上分球，使用其他训练节点观测标签构造软 target | 否 | 含噪标签的局部共识，给下游分类器训练 |
| `[X\|\|H]` | 历史 SoftDist 路线将 frozen H 拼接到下游 GCN 输入（`P/experiment_history_and_lessons/02_experiment_history_and_lessons 11.md` L147–160） | 否；更新的是下游分类器 | 分类损失 |
| A²X / X / GRACE 换分球空间 | 冻结表示后做分球和 target refinement 对照 | 否 | 静态表示差异 |
| Phase 40 X/H 冲突 | frozen X/H 双空间离线噪声诊断 | 否 | 局部投票冲突分数 |
| 候选 rooted ego-context SSL | 在 GB-LCSS 之前的 label-free 预训练阶段，对相同 root 的增强子图上下文加稳定性项；再冻结 H′，接原 GB-LCSS | **是**，但只由无标签 SSL 更新 | 相同 root 跨增强的上下文不变性；球只作低权重区域 pooling prior |

所以候选与固定 H smoothing、concat、local vote 的**因果位置不同**：它能改变 H′，不只是对固定 H 做后处理。这是实验问题改变，不等于方法创新；已有图子图 SSL、MGBCC 球级跨视图对应等工作需要另行逐篇比对。

## 检索边界与文献邻接

本轮只搜索 43 个 `P/experiment_history_and_lessons/02_experiment_history_and_lessons*.md`、指定旧 session、当前 redesign 文件和提交稿快照，没有重扫磁盘。对 43 份历史的关键词检查：`rooted`、`ego-subgraph`、`PPR`、`MVGRL`、`GraphCL`、learned/learnable augmentation、community-preserving/aware augmentation 均 0 命中；`subgraph` 只在 VQGraph history38 命中；`communities` 只在 history19 命中。GCA 有多处，但作为外部 baseline / 比较方法出现；BGRL/GBT 的 GB-view 替换已做；CCA-SSG/GraphMAE encoder swap 已做。关键词检索可能漏掉不同命名的代码或未入历史文档的实验，因此“未命中”不等于全盘证明未做。

文献侧有真实邻接，创新不能靠重命名：`P/01_literature_and_research_map.md` L366–397 将 MGBCC 记为 instance/cluster 间的 ball-level multi-view contrast，但也明确它是多视图聚类而非单图节点表示学习；L2011–2058 记 PGBSF 用粒球采样图子图并 progressive/incremental 训练，同时说明它不是 contrastive pretraining。当前 `W/research/algorithm_redesign_20261013/02_文献核查与机制比较.md` L48–50 和 `03_算法重设计建议.md` L90 已指出 MGBCC 球级跨视图成员关联及软化方案的有限差异。它们是文献/方案邻接，不是本地复现实验。

> **版本边界：** 本节评估的是讨论期备选：GRACE + rooted context consistency + stop-gradient 的软粒球区域 pooling。它不是最新 `04_粒球所需空间与子图增强方案.md` 的算法定义。最新 `04` 的首版是 PiNGDA 节点基座 + C-only VICReg 上下文分支 + H/C 组合构球空间，不含粒球加权 pooling 或 stop-gradient 粒球目标；当前方案以 `04` 为准，不要把两版拼成多个最终推荐。

## 候选机制审查：必须拆成三个可证伪问题

候选表达为：固定 GRACE 两视图增强；对每个 root i 分别取不含 root 本人的 1-hop context；用同一 root 的 context across views 做一个小权重、stop-gradient 的一致性项；root node embedding 保持原有 GRACE instance objective；粒球只给 context pooling 提供静态、软、detach 的区域权重；无标签训练完成后冻结 H′，再跑未改动的 GB-LCSS。

1. **是否有一般子图 SSL 收益？** 比较 GRACE 与无 GB 的 root-context consistency，预算、视图、encoder、seed 对齐。
2. **是否有粒球独立收益？** 比较普通均匀邻居 pooling、soft-GB pooling、匹配 K-means pooling、打乱/随机区域权重；若只有前两者都涨，结论是子图 SSL 有用，不能归功 GB。
3. **是否最终帮助原任务？** 对每种预训练表示固定跑原 GB-LCSS 和 no-ball 下游对照；另外交叉 `B_max={64,128,256}` 与同预算 K-means。若只提高 probe、球数或 purity 而不提高原下游，机制不通过。

> **版本边界：** 以下五项风险原本针对上述讨论期 GRACE/软粒球 pooling 备选。它们只在最新 `04` 仍包含同一机制时适用；`04` 特有的 VICReg、组合空间和 PiNGDA 风险应按 `04` 本身审查。

## 5 个实际风险与最小安全做法

1. **重复 DIR-01（边保持机制换名）。** 如果按球/社区保护内部边、丢弃跨区边，实质是原 same-ball/cross-ball view generation。最小安全做法：增强分布保持 GRACE 原样；球不改 edge_index，只作用于 context pooling 权重。
2. **与节点 GNN 表示冗余。** 根节点 H 已经通过消息传递汇总 ego 邻居；context consistency 可能只是重复 GRACE 的 root alignment。最小安全做法：context 用不含 root 的单独向量；加无 GB 的相同 context 对照、去掉 context 对照，并报告 context/root 表征差异与下游增量。
3. **稳定性变成过平滑或错误不变性。** 两个 view 间邻居变化可能是有效结构扰动；强迫 context 一致会抹掉异配邻居、边界节点或稀有局部差异。最小安全做法：只对同一 root 的 context 施加一个小的预注册 λ；不把同球不同 root 当 positive，不改 InfoNCE denominator，先用 label-free 几何 gate 判单球/碎片化和成员稳定性。
4. **球与 encoder 的反馈回路。** 若边训练边从 H′ 重构球，球成员和 target 会跟着 encoder 一起漂移，容易自我确认或振荡；若整个球塌成一个区域，区域项可能退化为普通上下文一致性。最小安全做法：从固定 GRACE H 生成一次 detach 的软区域先验，首轮不在线重分球；记录有效球数、大小分布、跨 seed 成员 ARI/稳定性和每个 root 的 context 覆盖。
5. **提升来源与球预算混杂，且子图成本会放大。** CoraFull 已显示 Bmax64→128 可带来约 +1.53pp；新 H′若球数、球大小或分裂阈值也变了，就无法归因空间目标。最小安全做法：对 H 与 H′统一跑 Bmax64/128/256，并在每个 Bmax 匹配 K-means；所有分组用同一 downstream seeds、split、噪声和训练轮数。按同一采样预算算 context，报告时延/显存，不能全图逐节点重复抽 ego 子图后再说开销低。

## 建议给主代理的判定

- **复现风险：中等。** “按社区保护增强”高风险重复 DIR-01；“root-indexed context consistency + frozen GB-LCSS 下游”在已索引本地历史里未命中，属于可检验的新实验轴。
- **机制风险：中高。** root context 与 GNN 聚合高度冗余，球权重可带来过平滑、分区反馈和 group-budget 混杂；必须有无 GB、均匀邻居、匹配 K-means 与随机区域对照。
- **论文新颖性：未定。** 不能因本地未做或把 MGBCC 软化，就称新颖。需要补查图级/节点级 subgraph contrastive learning 和 rooted-subgraph invariance 的近邻文献。
- **最小可执行决策：先只做无标签预训练筛查；不接 noisy CE 联训，不改 GB-LCSS target/分类器，不改图增强。先过表示几何与预算匹配门，再决定是否做原 noisy-label downstream。
- **潜在论文价值：条件性，不能从“新拼接”推出。** 若 root-context 目标先证明了节点目标未提供的结构信息，而且 soft-GB 区域权重优于无 GB、匹配 K-means 和随机分区，它可能把研究从固定 H 后的局部票推进到“局部结构如何改变预训练表示”；否则只是通用 subgraph SSL 的工程迁移。
- **成功可能性：目前不可估计，先验应保守。** 根节点 context 与 GNN 聚合冗余、GB-view 和多种显式几何改动已有负结果，会压低性能提升预期；但 rooted-subgraph 目标本身未被本地实验覆盖，不能据此判死。CoraFull 的 Bmax 结果说明分辨率混杂真实存在，先排除预算效应才有可解释的胜负。

## 主要来源登记

- `P/experiment_history_and_lessons/02_experiment_history_and_lessons.md`：DIR-01 L36–145；DIR-03 L317–425。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 2.md`：GB-GBT/GB-BGRL DIR-04 L509–574。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 11.md`：历史 `[X\|\|H]` 下游 GCN 路线 L147–160、受控增量 L264–266；SSL encoder swaps L231–249。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 12.md`：GRACE+uniformity gate、Phase 20b 未执行 L4–10、L20–70。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 18.md`：X/H 离线冲突 L1–15、结果 L87–124。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 19.md`：modularity communities L33–44、聚类 NO-GO L105–130。
- `P/experiment_history_and_lessons/02_experiment_history_and_lessons 38.md`：VQGraph sampled subgraph L1–21。
- `P/gb_graph_aware_minival_6_1/session_e5f46d7e_对话记录.md`：ball consistency 实现建议 L2860–2879；冻结 H / 下游分辨率问题 L3428–3449。
- `P/01_literature_and_research_map.md`：MGBCC L366–397；SGBGC coarsening L1322–1361；PGBSF subgraph sampling L2011–2058。
- `W/research/algorithm_redesign_20261013/99_研究记忆与接续.md`：当前 GB-LCSS 流程 L14–18；空间增量边界 L30–43；候选仍未验证 L55。
- `W/research/algorithm_redesign_20261013/01_项目与旧实验复盘.md`：当前空间/流程细节 L26–36；尚未验证在线空间目标 L112–114。
- `W/research/algorithm_redesign_20261013/02_文献核查与机制比较.md`：MGBCC/GFSAF 邻接 L48–50。
- `W/research/algorithm_redesign_20261013/03_算法重设计建议.md`：软集合对应与重叠判定 L88–100；匹配分组对照 L151–164。
- `W/research/algorithm_redesign_20261013/paper_snapshot/GB-LCSS_PR_submission_FINAL_20260930/main.tex`：CoraFull 70 类 L392；默认 Bmax=64 L416；表示空间及 A²X 对照 L494–500。
- 同一 snapshot 的 `supplementary.tex` L190–202、`table_I_1.tex` L25–26：Bmax=128 与 CoraFull 全五 seed +1.53±0.40pp；`table_6.tex` L11–19：GRACE/X/A²X 静态表示对照。
