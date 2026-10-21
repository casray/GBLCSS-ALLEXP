# 跨实验方向级总复盘

来源文件：

- `02_experiment_history_and_lessons.md`
- `02_experiment_history_and_lessons 2.md`
- `02_experiment_history_and_lessons 3.md`
- `02_experiment_history_and_lessons 4.md`
- `02_experiment_history_and_lessons 5.md`
- `02_experiment_history_and_lessons 6.md`
- `02_experiment_history_and_lessons 7 .md`
- `02_experiment_history_and_lessons 10.md`（实验 B：Stage B compactness 反向约束）
- `02_experiment_history_and_lessons 11.md`（实验 C/D：噪声标签节点分类主线收官，2026-06-09 至 06-11）

本复盘按实验方向合并，不按项目目录合并。结论只来自上述历史文件；凡是只在单个文件中出现的结果，均不提升为跨项目规律。

> 2026-06-11 更新：文件 11 记录的窗口使原"Inconclusive: frozen GRACE + GB local label smoothing"方向升级为**已确认主线（SoftDist [X||H]）**，并新增 11 个判死方向。本文件相应章节已更新；文件 11 的第 4 节"大总结判定表"是当前最权威的方向判定来源。

## 1. 总判断

- Confirmed: GB 深度介入 view generation / augmentation control 在多个项目中反复低于健康 GRACE，适合作为负结果和动机，不适合作为下一轮主线。
- Confirmed: hard / full / ratio-style same-ball 语义过粗，直接当 positive、negative mask、prototype 或 edge rule 都反复不稳。
- Confirmed（2026-06-11 升级）: **噪声标签下游方向已收敛为确认主线**——SoftDist [X||H]（frozen GRACE 几何球 + multiclass-alpha 球分布 soft target + H 拼接端到端 GCN）。5-seed Cora 0.8137±0.0067 ≥ NRGNN 0.8107±0.0230（std 小 3.4 倍），CiteSeer 0.6891±0.0159（差 NRGNN 1.36pp）。归因链完整：H +1.9/+2.3pp，ball-on-H +1.1/+1.3pp，real>shuffled 双数据集 5/5。这取代了旧的"实验7 弱信号"定位。
- Confirmed（2026-06-11）: GRACE encoder 在 noisy CE 下微调是反模式（linear 0.804 → classifier 0.570）；frozen 使用 + 独立端到端分类器是唯一有效框架形态。
- Confirmed（2026-06-11）: 在 SoftDist 之上的七个增强方向全部判负（动态 alpha、ball consistency、NRGNN 杂交、自适应粒度、无标签扩展、early-learning 裁决×2），每个都有预注册门槛和机理解释，构成机制边界负结果集。
- Confirmed: 复杂公式追加通常没有超过 fixed topk / 固定几何形态，包括 similarity-adaptive、mass-preserve、cross-view/cross-only、alpha/tau 小范围继续扫参等；文件 11 的 E1（标签驱动自适应粒度）以票密度不足再次印证。
- Likely: GB 更适合作为温和辅助先验、候选范围、诊断工具和（已确认的）标签预处理 teacher，而不是视图生成主体、硬语义标签或目标函数替代。
- Likely: 非 GB 的 x-space / percentile / degree-protect edge denoise 是 pretraining 侧最强鲁棒性正信号，但不能归因给 granular ball。
- Inconclusive: RDCL 在 PubMed 上有真实正信号，但机制更像 local relation regularization / group-size constraint，尚未证明来自高纯度语义粒球。
- Risk: Cora 单 seed、小于 std 的均值差、缺少 no-ball/shuffled/kmeans/topk 控制，是最容易导致论文 overclaim 的风险；文件 11 窗口内 seed0 两次误导（gated hard、ball-on-H 符号翻转）再次确认。

## 2. 按方向合并后的索引

| 合并方向 | 覆盖文件 | 总趋势 | 证据等级 | 决策 |
|---|---:|---|---|---|
| GB-guided view generation / augmentation control | 1-5 | 长训/formal 反复低于 GRACE | 多项目支持的负结论 | abandon |
| GB-BGRL / GB-GBT / non-InfoNCE objective replacement | 1-5 | 未修复 view-target mismatch，低于 GRACE | 多项目支持的负结论 | abandon |
| GB-FN-GRACE topk-soft false-negative suppression | 1-5 | Cora/CiteSeer 小正，PubMed 持平，std overlap，no-ball 接近 | 多项目支持的弱正/不充分 | modify / inconclusive |
| Hard/broad same-ball positive、prototype、direct relation regularization | 1-5，6 有 PCL 单项目 | 硬语义/宽范围关系约束反复失败；PCL 只有 Cora 单 seed 正信号 | 跨项目负结论 + 单项目现象 | abandon / inconclusive |
| GB membership same/cross-ball edge denoise | 1-5 | 噪声命中弱、clean collateral damage、formal 不足 | 多项目支持的负结论 | abandon |
| GB diagnostics / candidate precision / topology sensitivity | 1-6 | 有助于解释失败和防止 overclaim | 多项目支持的正向工具 | diagnostic-only |
| x-space percentile / degree-protect edge denoise | 3-5 | noisy graph 多设置正向，但不是 GB 贡献 | 多文件正信号，需限定范围 | continue as non-GB robustness |
| Risk-aware augmentation / CLSA weak-strong relation | 3 | 混合结果，弱于 S2 full 或数据集不稳 | 单项目现象 | ablation only / modify |
| GNNGuard-lite control | 5 | quick val 不过，权重近似全 1 | 单项目失败 | abandon as current control |
| RDCL local relation consistency | 6 | PubMed 3-seed +0.0030 / +0.0025，但 shuffled/random 控制削弱语义解释 | 单项目中等正信号 | modify |
| GB-PCL positive expansion | 6 | Cora 单 seed best F1Mi 0.8358，缺 raw log/多 seed/非 Cora | 单项目现象 | inconclusive |
| Noisy-label GB-GRACE frozen local smoothing | 7 | Cora train 20% / 40% noise 有局部正向；CiteSeer 不稳定；Planetoid coverage 低 | 被文件 11 取代 | superseded → 见 DIR-12 |
| Stage B GB compactness 反向约束 encoder | 10 | Cora seed0 正信号但跨 seed 不稳；B2 刷新无效 | 单项目负结论 | abandon |
| **DIR-12: SoftDist [X\|\|H] 噪声标签主线** | 11 | 5-seed 双数据集确认；Cora ≥ NRGNN 且 std 小 3.4 倍；归因链完整 | **已确认主线** | **adopt as paper mainline** |
| DIR-13: SoftDist 之上的七个增强 | 11 | 动态 alpha / BC / BallAux / E1 自适应粒度 / 无标签扩展 / early 裁决×2 全判负 | 多机制一致负结论 | abandon（见文件 11 §4.2 判死表） |
| DIR-14: noise regime 适用边界 | 11 | Cora 4/5 regime tie NRGNN；CiteSeer 全 loss；uniform 0.6 管线崩溃；球增量峰值在 0.4 | 边界分析完成 | 写入论文 limitation |

## 3. 跨项目反复失败的方向

1. GB 作为直接 view generator。
   - 覆盖文件 1-5。
   - 反复现象：active balls 不等于下游表示变好；长训、decay、freeze、anchor/explore、stronger recovery 后仍低于 GRACE。
   - 最合理解释：GB 视图改变了 GRACE 原本 augmentation distribution，减少有效增强多样性或造成 view drift。

2. GB-BGRL / GB-GBT 目标替换。
   - 覆盖文件 1-5。
   - 反复现象：换成 bootstrap / redundancy objective 没有解决 GB view mismatch，反而引入更多机制变量。
   - 结论：不要再用“换 objective”修 GB view generation，除非先有健康 no-GB BGRL/GBT baseline 和 view-gap 诊断。

3. Hard 或 broad same-ball 语义。
   - 覆盖文件 1-5。
   - 反复现象：hard deletion、full-soft、ratio-wide、prototype、same-ball positive、direct consistency 都不如 conservative topk-soft 或没有 validation gain。
   - 结论：same-ball membership 只能当弱结构先验，不能当可靠语义标签。

4. GB-FN 公式购物。
   - 覆盖文件 1-5。
   - 反复现象：similarity-adaptive、mass-preserve、cross-view ranking、cross-only、stable/fuzzy 变体都没有稳定超过 v2 fixed topk。
   - 结论：停止发明新公式；下一步必须是机制证明和 no-ball 控制。

5. 纯 same/cross-ball edge denoise。
   - 覆盖文件 1-5。
   - 反复现象：噪声边命中弱、clean edge 被误伤；文件 4 记录 downweight precision 0.5596、recall 0.2272，quick val 低于 GRACE。
   - 结论：GB membership 不应作为 edge-weight rule，只能作为 edge scorer 的一个可选 feature。

6. 只看均值差、不看 std / no-ball / shuffled 控制。
   - 覆盖文件 1-7。
   - 反复现象：GB-FN Cora +0.0014、CiteSeer +0.0010 这类提升小于 std；RDCL shuffled/random 可复现大部分 PubMed gain；实验7只在 Cora 局部设置正向。
   - 结论：任何 main claim 都必须配 paired seed、std、no-ball/shuffled/fixed/kmeans/topk 控制。

## 4. 只是单项目现象，不能过度推广

- RDCL local relation consistency：只在文件 6 有系统记录。PubMed 3-seed 正信号较强，但 DBLP 单 seed 退化，且 random / shuffled 控制削弱“语义粒球必要”的解释。
- GB-PCL positive expansion：只在文件 6 出现，Cora 单 seed 有正结果，但缺多 seed、非 Cora、完整 raw log。
- Risk-aware augmentation control / CLSA：主要在文件 3 出现，结果混合，且多为非 GB 或弱 GB role。
- GNNGuard-lite control：只在文件 5 出现，当前实现失败不能推广为 GNNGuard 类方法失败。
- old gb_v2 collapse / bounded gb_v2：文件 6 的构球诊断现象，能作为实现约束，但不能单独推出所有 GB 构球都会 collapse。
- Noisy-label GB-GRACE frozen local smoothing：只在文件 7 出现。Cora medium-label train 20% / 40% noise 有弱正向；CiteSeer 不支持强 claim。它说明 post-pretraining GB 用法值得保留为弱信号，但不能推广为 GB noisy-label 主方法成功。

## 5. 证据分类

### 已被多个实验支持的结论

- GB 直接改 view generation 不可靠。
- hard / full / ratio same-ball 关系过粗。
- topk-soft 比 broad suppression 更安全。
- GB-FN 是目前最安全的 GB 使用位置，但主效果弱。
- no-ball similarity topk 是必须对照；当前 GB 候选相对 no-ball 没有稳定打开 gap。
- 纯 GB membership edge denoise 不够强。
- 诊断先行能有效阻止低价值 formal run。
- 下游 noisy-label 方向必须报告 effective smoothing coverage；coverage 低会阻塞机制解释，但 coverage 提高本身不等于 GB-specific value。

### 只在单个项目中出现的现象

- RDCL PubMed 正信号。
- GB-PCL Cora 单 seed 正信号。
- risk-aware augmentation / CLSA 混合结果。
- GNNGuard-lite 当前实现失败。
- bounded gb_v2 可作为 kmeans 替代的可能性。
- frozen GRACE + GB local smoothing 在 Cora noisy-label medium-label 设置有局部正向，但 CiteSeer 不稳。

### 证据不足的解释

- “GB provides robust local structure prior”目前最多 weak。
- “高纯度语义粒球导致 RDCL gain”被 shuffled/random 控制削弱，not ready。
- “GB-guided augmentation preserves local semantics”被 view generation 负结果削弱，not ready。
- “GB denoising improves robustness under graph noise”被纯 GB edge denoise 负结果削弱，not ready。
- “GB confidence reduces noisy relation damage”缺少稳定 confidence 指标和下游收益，weak / not ready。
- “GB local smoothing improves noisy-label node classification after GRACE pretraining”目前只由文件 7 弱支持，CiteSeer 削弱该 claim，weak / not ready。

### 被负结果削弱的方向

- GB view generation as main method。
- GB-BGRL / GB-GBT as repair。
- hard same-ball positive / prototype pulling / direct relation regularization。
- full-ball / ratio-wide / hard negative suppression。
- pure same/cross-ball edge denoise。
- topology-only reliable-ball filtering without assignment movement。
- Cora-only single-seed positive expansion as main method。
- strict GB local label smoothing as noisy-label main method。

## 6. 最重要的失败模式

| 失败模式 | 为什么重要 | 后续约束 |
|---|---|---|
| 把 GB membership 当语义真值 | 误伤有效负样本、clean edge 和正常 augmentation | 只允许 soft、topk、confidence-gated，并配 no-ball 控制 |
| 改 view 比改 loss 更容易破坏 GRACE | 多项目 long/formal 低于 GRACE | GB 不再默认参与 view generation |
| 候选信号被 similarity-only 解释掉 | no-ball topk 接近 ball-topk | 所有 GB-FN claim 必须包含 no-ball |
| 效果量小于 seed 方差 | clean citation graph 增益多为 tiny delta | 小于 std 一律标 inconclusive |
| 复杂度增长快于证据 | 多个公式变体没有收益 | 不做新公式，先做机制诊断 |
| 诊断纯度不等于下游收益 | RDCL shuffled/random 控制接近主方法 | downstream control 必须和 purity 一起报告 |
| clean/noisy 边界不清 | denoise 方法会伤 clean graph | robustness claim 必须报告 clean cost 和 gate |
| coverage 不等于 GB-specific value | 实验7 medium-label 提高 coverage 后 CiteSeer 仍不稳 | noisy-label GB 必须同时过 coverage、kmeans/topk、paired delta 三道门 |

## 7. 仍值得保留的正向信号

- GB-FN v2 topk-soft：作为低侵入 loss-side prior 保留，但不是强胜利；适合作为“GB 的安全用法”或 ablation。
- Top-k fine suppression：比 hard/full/ratio 更稳，是未来所有 false-negative correction 的最小安全形式。
- x-space percentile / degree-protect edge denoise：如果论文转向 noisy graph robustness，这是最强方法信号；必须明确不是 GB 贡献。
- RDCL local relation regularization：PubMed 3-seed 正信号值得保留，但 claim 应写成 local relation regularizer，而不是语义粒球必要性。
- Diagnostics pack：candidate precision、no-ball topk、edge precision/recall、ball stats、assignment change、loss scale，是下一轮实验入口条件。
- bounded gb_v2：可能作为构球实现替代，但需要证明超过 shuffled/random/fixed controls。
- Post-pretraining GB local smoothing：比直接改 GRACE pretraining 更安全，实验7保留了 noisy-label downstream 的弱信号；但当前只能作为 motivation / ablation。

## 8. 最可能形成论文 claim 的方向

### 主线 claim（2026-06-11 确认，取代以下全部旧候选）

**SoftDist [X||H]: a simple non-iterative granular-ball label-correction pipeline on frozen graph-contrastive representations matches NRGNN on Cora with 3× lower variance.**

证据等级：strong within mainline protocol（percent split 0.2/0.1/0.7, uniform 0.4, 5 seeds, full ablation chain + shuffled controls）。配套 claim：
- 表示阶段对标签噪声构造性免疫（全 regime 复用同一 label-free H）。
- coverage sensitivity：球机制依赖 local labeled coverage（C1 阶段数据）。
- 适用边界：CiteSeer 弱于 NRGNN 1.4pp；uniform 0.6 不适用；低噪声下球冗余。
- 机制边界负结果集：六个增强方向的预注册判负（文件 11 §4.2），作为方法论严谨度展示。

### 旧候选的处置

- "Hard / broad same-ball relation is not reliable; GB safer as conservative auxiliary prior" —— 仍然成立，作为 pretraining 侧背景动机引用。
- "GB-FN topk-soft as lightweight false-negative prior" —— 保留为 pretraining 侧 weak ablation，不进主线。
- "x-space percentile edge weighting robustness" —— 非 GB 贡献，如转向 robustness 论文再启用。
- "RDCL local relation regularizer" —— 保留 PubMed 信号，claim 降级为 local regularizer form。
- "Coverage-aware GB local smoothing may help"（文件 7 弱信号）—— 已被 DIR-12 强化取代，不再以弱形式表述。

## 9. 最不建议继续投入的方向

（2026-06-11 更新：前半为 pretraining 侧旧清单，后半为文件 11 噪声标签窗口新增判死项）

- 继续调 GB view generation 的 schedule、freeze、anchor/explore。
- 用 GB-BGRL / GB-GBT 修复 GB view generation。
- hard same-ball positive、prototype pulling、整球 suppression、ratio-wide suppression。
- GB-FN 的新公式购物，包括 adaptive/mass/cross-view 小改。
- pure same/cross-ball membership edge denoise。
- 只做 Cora single-seed 的 GB-PCL 扫参。
- 只调 RDCL alpha/tau，而不做 fixed/random/shuffled/size-matched 机制隔离。
- topology-only filtering 在没有 assignment/candidate movement 前直接 formal。
- 用 `gb_prior` 结果包装 strict GB claim。
- （新增）GRACE encoder 在 noisy CE 下微调——frozen + 独立分类器是唯一有效形态。
- （新增）二值 gate / 可靠性加权 / 动态 alpha / ball consistency / NRGNN 杂交 / 标签驱动自适应粒度 / 球级无标签伪监督 / early-learning 裁决——全部在文件 11 判死，证据与机理见其 §4.2，不要换皮复活。
- （新增）任何依赖"模型自身预测作为球信任反馈"的机制——自强化已两次实证。

## 10. 下一轮最小实验建议

（2026-06-11 更新：NEXT-6 已完成并超额达成——演化为 SoftDist [X||H] 主线，见文件 11。原 NEXT-1 至 NEXT-5 仍属 pretraining 侧，保留原文备查但当前优先级让位于论文写作。）

### 当前最高优先级：论文写作（数据已齐）

- 主表：noise grid 5 regime × 5 方法 × 2 数据集 × 5 seeds（`noise_grid_rocm_20260611/`）。
- 消融链：H 增量 / ball-on-H 增量 / real-vs-shuffled（`xh_5seed_rocm_20260611/`）。
- coverage sensitivity：C1 阶段 ratio 数据（`gb_denoise_stage_b_compactness/reports/`）。
- 负结果集：文件 11 §4.2 判死表 + 各报告。
- future work：CiteSeer 的 GRACE 预训练质量（H 增量 +2.3pp 是杠杆）、online 机制应对 uniform 0.6、PubMed/instance noise 扩展。

## 附录：旧版下一轮实验建议（2026-06-11 前，备查）

### NEXT-1: GB-FN 独立贡献证明

- Change only: 固定 v2 topk-soft，不再改公式。
- Baselines: GRACE、no-ball topk、random topk、full/ratio 作为 ablation。
- Datasets/seeds: Cora/CiteSeer 5 seeds，PubMed 至少 3 seeds。
- Diagnostics: candidate precision、ball/no-ball overlap、masked ratio、loss scale、paired seed delta。
- Success: GB-topk 在至少两个数据集稳定超过 no-ball topk，且 paired delta 不被 std 淹没。
- Failure: no-ball 持平或更强，则 GB-FN 只能作为 weak ablation / limitation。

### NEXT-2: RDCL 机制隔离

- Change only: 加 size-matched fixed/random/shuffled partition controls。
- Keep fixed: encoder、epochs、batch、eval、alpha/tau 主配置。
- Datasets/seeds: PubMed 3-5 seeds，加一个非负数据集。
- Diagnostics: protected_precision、num_groups、max_group_size、group-size distribution、edge_ratio。
- Success: true GB/kmeans groups 稳定超过 fixed/random/shuffled。
- Failure: 控制组接近主方法，则 claim 改为“local regularizer form works”，不能 claim semantic GB。

### NEXT-3: 非 GB noisy edge robustness 主线确认

- Change only: x_pct / S2 percentile edge weighting + 无监督 noise gate。
- Controls: always-on、gate-off、shuffled-score、GRACE、DropEdge。
- Datasets/seeds: Cora/CiteSeer/PubMed clean + edge rewire noise，多 seed。
- Success: noisy setting 稳定正向，clean 不退化或 gate 自动关闭。
- Failure: clean cost 无法控制，则只能写 robustness-only，不写 general graph claim。

### NEXT-4: GB as edge-scorer feature，而不是 edge rule

- Change only: 在 endpoint similarity scorer 中加入 GB feature，比较 no-GB scorer。
- Metrics: edge noise AUC/AP、downweight precision/recall、downstream paired delta。
- Success: GB feature 明确提升 scorer 且转化为 downstream gain。
- Failure: no-GB scorer 持平，则停止 GB edge denoise。

### NEXT-5: 统一 diagnostic gate

每个新方向进入 formal 前必须先通过：

- baseline comparable；
- no label leakage；
- 多 seed 或明确 exploratory；
- no-ball / no-GB / shuffled 控制；
- candidate 或 edge 机制诊断；
- ball stats，包括 num balls、max group size、singleton ratio、assignment change；
- 增益大于 std 或有 paired seed 一致性。
- 对 noisy-label smoothing 方向，还必须报告 effective smoothing coverage、coverage-by-class，并比较 CE/global/kmeans/topk_matched。

### NEXT-6: Coverage-aware noisy-label GB local smoothing（✅ 已完成并超额达成）

> 2026-06-11：此建议已由文件 11 的窗口执行完毕。成功标准达成方式与原设想不同——不是 coverage-aware smoothing 变体胜出，而是经由"框架迁移（frozen GRACE + 端到端 GCN）→ SoftDist target → [X||H]"路径演化为确认主线。原文保留如下备查。

- Change only: 不再扩展 strict ball-only smoothing；把方法重定义为 coverage-aware 或 prior-corrected local smoothing。
- Baselines: CE、global、kmeans、topk_matched、GB-strict、GB-prior。
- Datasets/seeds: Cora + CiteSeer，ratio train 10%/20%，5 seeds，先做 symmetric 20%/40%。
- Diagnostics: effective smoothing coverage、coverage-by-class、GB-kmeans NMI、GB-topk overlap、ball train-label count distribution、paired delta。
- Success: 两个数据集都稳定超过 CE/global/kmeans/topk_matched，并解释 coverage bins 为什么有效。
- Failure: CiteSeer 继续不稳或 kmeans/topk 持平，则 noisy-label GB 只能作为弱动机/负结果，不进入主线。
