# GB-GCL Experiment History and Lessons - Experiment A

本文档只整理实验 A 方向的证据。实验 A 对应 `gb_denoise_stage_a_negfree_vs_infonce`：在不改变同一套粒球生成算法的前提下，只改变表示学习方式，比较 raw features、InfoNCE 和 negative-free 表示空间上生成粒球的质量。它是对既有 `02_experiment_history_and_lessons.md` 的独立补充，不覆盖旧方向。

# 1. Current Overall Judgment

- Confirmed: 实验 A 是 diagnostic-only 方向。它主要回答“负样本排斥是否会影响粒球局部紧凑性”，而不是直接提出一个可投稿主方法。
- Confirmed: 在 Cora、CiteSeer、PubMed 的 5 seed 结果中，negative-free 表示通常给出更小的粒球半径和平均角距离，但 linear accuracy 和 label purity 明显低于 InfoNCE。
- Confirmed: `negfree_z` 尤其容易出现极端压缩，表现为球数很少、平均球规模巨大、AD 很小，但分类准确率和纯度很差。
- Confirmed: 后续稳定性消融显示，这个现象不是单一增强强度或单一粒球参数造成的。raw negative-free 在弱、中、强增强下都倾向于紧凑但语义可分性差。
- Confirmed: BN 是关键干扰因素。加入 BN 后，negative-free 的 accuracy、purity 和 effective rank 可以接近 InfoNCE，但小半径、小 AD 优势基本消失。
- Likely: 普通 InfoNCE 的负样本排斥确实会降低局部几何紧凑性，但它也帮助维持全局语义可分性。
- Risk: 如果只看 radius 或 AD，会把表示塌缩误判为粒球质量提升。
- Decision: diagnostic-only。A 组支持“局部紧凑和全局可分存在张力”这个论文动机，但不支持“直接用 negative-free 替代 InfoNCE”。
- Next priority: 后续方法应在稳定 encoder 上引入 soft / confidence GB 约束，而不是追求 raw negative-free 的极小半径。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-09 | Stage A representation-space diagnostic: negative-free vs InfoNCE | 粒球作为表示空间诊断器，不参与训练 | raw_features, grace_infonce_h, grace_infonce_z, negfree_h, negfree_z; augmentation / stabilizer / ball-parameter ablations | Cora, CiteSeer, PubMed; stability ablation on Cora/CiteSeer | negative-free 产生更小 radius/AD；BN negative-free 可恢复语义质量 | 小 radius/AD 主要来自过度压缩，不等于语义粒球更好 | diagnostic-only | motivation / diagnostic |

# 3. Direction-Level Detailed Record

## DIR-09: Stage A Representation-Space Diagnostic

### 3.1 Direction Summary

- Direction ID: DIR-09
- Direction name: Stage A representation-space diagnostic: negative-free vs InfoNCE
- Time period: 2026-05-19 to 2026-05-20
- Target problem: 验证普通对比学习中的负样本排斥是否会和粒球局部聚合目标冲突。
- Core hypothesis: 如果负样本排斥把样本推远，那么 InfoNCE 表示空间上的粒球可能更松散；negative-free 表示空间可能更适合生成紧凑粒球。
- Baseline: `raw_features` 和 GRACE/InfoNCE。
- Main related folder: `gb_denoise_stage_a_negfree_vs_infonce`
- Follow-up stability folder: `gb_denoise_stage_b_compactness`
- Result files: `gb_denoise_stage_a_negfree_vs_infonce/results/stage_a_metrics.csv`, `gb_denoise_stage_a_negfree_vs_infonce/results/stage_a_summary.md`
- Stability report: `gb_denoise_stage_b_compactness/reports/2026-05-20_stability_ablation_formal_report.md`
- Config snapshots: `gb_denoise_stage_a_negfree_vs_infonce/results/*/*/config_snapshot.json`
- Raw embeddings: `gb_denoise_stage_a_negfree_vs_infonce/results/*/*/embedding.pt`
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: InfoNCE 通过负样本排斥保持全局分离，但这种排斥可能让局部样本不够紧凑。
- Why GB might help: 粒球生成依赖“距离约等于语义相似度”的表示空间。如果表示空间局部更紧，粒球可能更小、更稳定。
- Expected improvement: negative-free 表示应生成半径更小、平均角距离更小、语义纯度不下降的粒球。
- Literature or previous experiment inspiration: 2603.27734v1 使用 negative sample free 方式；CGBC 类思想强调利用粒球结构约束表示空间。
- Expected paper claim: 负样本排斥和粒球局部聚合存在冲突，因此需要重新设计 GB-GCL 目标。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset / Seeds | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| A0 | raw_features，不训练 encoder，直接建球 | Cora, CiteSeer, PubMed; seeds 0-4 作为重复诊断 | acc 中等，radius/AD 最大 | 原始特征参照 | 原始特征不代表可学习表示，但提供最低参照 |
| A1 | GRACE/InfoNCE，观察 encoder output `h` | Cora, CiteSeer, PubMed; seeds 0-4 | radius/AD 较大，purity 和 acc 较高 | 普通对比学习表示空间 | InfoNCE 保留更好的全局语义可分性 |
| A2 | GRACE/InfoNCE，观察 projector output `z` | Cora, CiteSeer, PubMed; seeds 0-4 | z 空间 rank 更低，但 acc/purity 仍明显优于 raw negative-free | projector 空间是否适合建球 | z 可作为诊断，但不应默认优于 h |
| A3 | negative-free，观察 encoder output `h` | Cora, CiteSeer, PubMed; seeds 0-4 | radius/AD 小很多，但 purity 和 acc 大幅下降 | 负样本消除是否改善粒球 | 只改善几何紧凑，不改善语义质量 |
| A4 | negative-free，观察 projector output `z` | Cora, CiteSeer, PubMed; seeds 0-4 | 极少球、超大球、极低 AD，acc 最差 | projector 是否更紧凑 | `z` 空间更容易塌缩，不适合作为主建球空间 |
| A5 | 增强强度消融 | Cora, CiteSeer; seeds 0-2 | negative-free 在弱、中、强增强下都更紧但更差 | 是否由增强参数造成 | 不是单一增强强度假象 |
| A6 | BN / variance / EMA stabilizer 消融 | Cora, CiteSeer; seeds 0-2 | BN 恢复 acc/rank/purity，但 radius/AD 变大 | negative-free 是否本身必然失败 | 不是 negative-free 必然失败，而是 raw negative-free 缺稳定机制 |
| A7 | 粒球参数消融 | Cora, CiteSeer; seeds 0-2 embeddings | 改建球参数会改 purity/radius，但不改 embedding acc/rank | 是否由粒球参数造成 | 建球参数不能修复弱表示空间 |

### 3.4 Role of Granular Ball

- GB role: 只作为 label-free 表示空间诊断器。
- Where GB is generated: 保存后的 node embedding 上。
- Representation space: raw features、encoder output `h`、projector output `z`。
- Affects loss: no。
- Affects augmentation: no。
- Affects graph structure: no。
- Affects sampling: no。
- Affects evaluation: no。标签只用于 `label_purity_diagnostic` 和 linear evaluation。

### 3.5 Experimental Setup Summary

- Datasets: Cora, CiteSeer, PubMed。
- Seeds: Stage A 主结果使用 seeds 0-4；稳定性消融使用 Cora/CiteSeer seeds 0-2。
- Methods: `raw_features`, `grace_infonce_h`, `grace_infonce_z`, `negfree_h`, `negfree_z`。
- Ball generation: embedding L2 normalize 后用角距离递归切分。
- Evaluation protocol: linear evaluation、label purity diagnostic、radius、mean average angular distance、effective rank、embedding std / uniformity。
- Label usage: 标签不进入训练和建球，只用于后验诊断。

### 3.6 Result Summary

#### Stage A 5-seed primary summary

| Dataset | Method | Acc mean | Purity mean | Effective rank mean | Num balls mean | Radius mean | AD mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | grace_infonce_h | 0.7336 | 0.7952 | 64.8646 | 150.0 | 0.4756 | 0.2839 |
| Cora | negfree_h | 0.2534 | 0.3047 | 68.1271 | 2.2 | 0.3313 | 0.0709 |
| Cora | negfree_z | 0.1480 | 0.3026 | 16.6362 | 4.2 | 0.0742 | 0.0054 |
| CiteSeer | grace_infonce_h | 0.6752 | 0.7211 | 60.7406 | 207.2 | 0.3918 | 0.2419 |
| CiteSeer | negfree_h | 0.2056 | 0.2377 | 36.1097 | 9.4 | 0.1185 | 0.0317 |
| CiteSeer | negfree_z | 0.1386 | 0.2169 | 12.8685 | 2.6 | 0.0464 | 0.0039 |
| PubMed | grace_infonce_h | 0.6942 | 0.7944 | 40.9990 | 713.6 | 0.2950 | 0.1826 |
| PubMed | negfree_h | 0.4232 | 0.5585 | 53.7560 | 204.4 | 0.1035 | 0.0661 |
| PubMed | negfree_z | 0.3572 | 0.4057 | 10.6909 | 8.6 | 0.0596 | 0.0060 |

直接结论：negative-free 的 radius 和 AD 确实更小，但同时伴随 accuracy 和 purity 大幅下降；尤其 `negfree_z` 的“紧凑”更接近表示压扁，而不是语义粒球质量提升。

#### Stabilizer ablation summary

| Dataset | Objective / Variant | Radius | AD | Purity | Acc | Effective rank |
|---|---|---:|---:|---:|---:|---:|
| Cora | InfoNCE plain | 0.8983 | 0.5858 | 0.7064 | 0.7880 | 135.4512 |
| Cora | negfree plain | 0.0639 | 0.0140 | 0.5613 | 0.3557 | 2.2150 |
| Cora | negfree BN | 1.0917 | 0.6581 | 0.7224 | 0.7910 | 131.8897 |
| CiteSeer | InfoNCE plain | 0.9398 | 0.6340 | 0.6054 | 0.6213 | 145.8881 |
| CiteSeer | negfree plain | 0.2452 | 0.0371 | 0.2808 | 0.4033 | 6.8008 |
| CiteSeer | negfree BN | 1.1034 | 0.6789 | 0.6176 | 0.5857 | 136.4544 |

BN 结果很关键：它说明 negative-free 不是必然差，但当 negative-free 变得语义可分时，小 radius / AD 优势也消失了。

### 3.7 What Is Directly Supported

- negative-free 表示空间能生成更紧的粒球，但这种紧凑性不能直接解释为“粒球质量更好”。
- InfoNCE 的负样本排斥会让局部球更松，但它保留了更好的 linear accuracy 和 label purity。
- projector output `z` 不能默认用于生成粒球。`negfree_z` 的几何指标很好看，但语义诊断很差。
- BN 可以显著缓解 raw negative-free 的过度压缩，但会削弱或消除小半径优势。
- 粒球参数消融不能改变底层 embedding 的线性分类和 rank，因此不能把表示失败归因于某个建球参数。

### 3.8 Reasonable Interpretation

- 解释 1: 普通 InfoNCE 的负样本排斥和粒球局部聚合目标确实存在张力。它不利于最小化球半径，但有利于全局类别或语义区域分离。
- 解释 2: negative-free 缺少额外稳定机制时，容易把“视图一致性”变成全局压缩。粒球生成器会把这种压缩看成小半径和小 AD。
- 解释 3: 合理的 GB-GCL 方向不是删除负样本，而是在保持全局可分的基础上加入温和的局部结构约束。

### 3.9 What Cannot Be Concluded

- 不能结论说 negative-free 一定比 InfoNCE 差。BN negative-free 已经给出反例。
- 不能结论说 InfoNCE 一定不适合粒球。InfoNCE 的粒球更大，但 purity 和分类更好。
- 不能结论说小 radius / AD 就是好粒球。A 组恰好证明这是危险指标。
- 不能结论说 2603.27734v1 使用 negative sample free 的原因一定是避免粒球冲突。A 组只能支持“这种解释合理”，不能证明论文作者动机。

### 3.10 Failure Analysis

- Main failure reason: raw negative-free 的局部紧凑收益主要来自表示过度压缩，而不是语义结构增强。
- Secondary failure reason: 只用 radius / AD 评价粒球，会奖励 collapsed representation。
- Evidence: `negfree_h` 和 `negfree_z` 在三数据集上 radius/AD 更小，但 linear accuracy 和 label purity 明显低于 `grace_infonce_h`；BN negative-free 恢复语义质量时 radius/AD 变大。
- Alternative explanation: 部分结果可能受训练超参数影响，但增强强度和球参数消融已经削弱了“单一参数导致”的解释。
- Minimum check needed: 任何后续方法只要声称粒球更好，都必须同时报告 accuracy、purity、effective rank、feature std、radius 和 AD。

### 3.11 Lessons Learned

- Lesson: radius / AD 只能表示几何紧凑，不能单独表示语义粒球质量。Evidence: `negfree_z` 的 AD 极小但 acc 最差。Future constraint: 报告中必须把 compactness 和 semantic separability 分开。
- Lesson: 负样本排斥不是纯负面机制。Evidence: InfoNCE 粒球较松，但 purity 和 acc 更好。Future constraint: 不要简单用 negative-free 替代 InfoNCE。
- Lesson: 稳定器会改变结论。Evidence: BN negative-free 接近 InfoNCE，但小球优势消失。Future constraint: 比较 negative-free 和 InfoNCE 时必须说明是否使用 BN、EMA、variance regularizer。
- Lesson: projector 空间更容易产生误导性紧凑。Evidence: `negfree_z` 球数极少、AD 极低、accuracy 极差。Future constraint: 默认在 encoder output `H` 上生成粒球，`Z` 只做诊断。

### 3.12 What Is Still Useful

- A 组提供了论文动机：局部紧凑和全局语义可分之间确实存在可测张力。
- A 组提供了诊断指标组合：radius / AD 必须配合 purity、linear accuracy、effective rank、feature std。
- A 组支持后续 C 组设计：在稳定 InfoNCE 或 BN-stabilized encoder 上加入 soft GB 约束，而不是直接依赖 raw negative-free。

### 3.13 Decision

- Decision: diagnostic-only。
- Reason: A 组没有产生可直接作为主方法的改进，但清楚解释了为什么“只追求小粒球”会失败。
- What should not be tried again: 不要把 raw negative-free 的小 radius / AD 当作成功，不要把 `negfree_z` 当作主建球空间。
- Variant still worth trying: 稳定 encoder 上的 soft local prior、confidence weighting、weak-view relation teacher。
- Evidence required to revive as main method: negative-free 变体必须在三数据集多 seed 上同时保持小 radius/AD、高 purity、健康 effective rank 和不低于 InfoNCE 的 linear accuracy。

# 4. Cross-Direction Failed Pattern Ledger

## FP-A1: 小半径误判为高质量

### Status

confirmed

### Related Directions

DIR-09, DIR-10, DIR-08

### Observation

negative-free 和随机 compactness 都可以显著降低 radius/AD，但分类准确率、purity 或 effective rank 会下降。

### Possible Reason

几何紧凑可以来自真实局部语义，也可以来自整体塌缩、过平滑或错误聚合。

### Evidence

`negfree_z` 在 Cora/CiteSeer/PubMed 上 AD 极低但 acc 最差；B 组 B3/B4 也能降低 radius/AD 但准确率崩溃。

### Future Constraint

任何 GB 方法不能只用 radius/AD 证明有效，必须同时看 semantic 和 anti-collapse 指标。

## FP-A2: raw negative-free 容易制造虚假的粒球优势

### Status

likely

### Related Directions

DIR-09, DIR-10

### Observation

raw negative-free 很容易产生紧凑空间，后续再加 compactness loss 会进一步放大局部拉近风险。

### Possible Reason

缺少负样本排斥或足够强的 variance / BN 约束后，视图一致性目标可能退化为低维压缩。

### Evidence

Stabilizer ablation 中 `negfree plain` 的 effective rank 和 feature std 很低；BN 恢复后语义质量接近 InfoNCE。

### Future Constraint

raw negative-free 不应作为后续主 baseline，除非加入稳定器并通过 anti-collapse gate。

# 5. Cross-Direction Positive Signal Ledger

## PS-A1: 局部紧凑和全局可分的张力是真问题

### Status

confirmed

### Related Directions

DIR-09, DIR-10, DIR-08

### Observation

InfoNCE 更可分但粒球更松；negative-free 更紧但语义更差；BN 恢复可分后紧凑优势消失。

### Why It May Matter

这个张力可以成为论文方法动机：GB 不能作为 hard pulling 目标，而应作为 soft local structure prior。

### Required Verification

后续主方法必须证明能在不牺牲 rank/accuracy 的情况下利用局部结构。

### Possible Paper Use

motivation / diagnostic

## PS-A2: BN-stabilized negative-free 是有用对照

### Status

possible

### Related Directions

DIR-09

### Observation

BN negative-free 在 Cora 上 acc 0.7910、rank 131.8897，接近 InfoNCE；CiteSeer 上也明显优于 raw negative-free。

### Why It May Matter

它说明后续不能把 negative-free 简化成“必然失败”，而要区分 objective 和 stabilizer。

### Required Verification

如果继续使用 negative-free，需要把 BN 版本作为正式 baseline。

### Possible Paper Use

ablation / diagnostic

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| 只看 radius/AD 评价粒球 | 会奖励表示塌缩 | `negfree_z` AD 极低但 acc/purity 极差 | 同时报告 acc、purity、rank、feature std |
| raw negative-free 直接作为主方法 | 语义可分性不稳定 | Cora/CiteSeer/PubMed 上 acc 明显低于 InfoNCE | BN/variance/EMA 后多数据集多 seed 不低于 InfoNCE |
| 在 projector `Z` 上默认建球 | `Z` 更容易被投影头压缩 | `negfree_z` 球数极少且 acc 最差 | 证明 `Z` 空间在 purity、rank、acc 上优于 `H` |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| 用 negative-free 小球半径证明方法有效 | 局部紧凑确实可被表示学习改变 | 在稳定 encoder 上用 soft GB relation，而不是追求极小半径 |
| 删除负样本排斥来适配粒球 | 负样本排斥和局部聚合有张力 | 保留 InfoNCE 全局分离，再加弱局部先验 |
| 在 `Z` 空间生成粒球 | projector 可暴露 collapse 现象 | `Z` 只做诊断，主方法使用 `H` |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: 必须同时比较 InfoNCE、raw negative-free、稳定版 negative-free。
- Multi-seed evidence: 至少 seeds 0/1/2，最好 5 seeds。
- Multi-dataset evidence: 至少 Cora、CiteSeer，主结论不能只靠 Cora。
- Ablation removing GB signal: 必须有无 GB、随机 GB 或 shuffled assignment 对照。
- Mechanism diagnostic: 必须报告 compactness 和 anti-collapse 指标。
- No label leakage: 标签只用于后验诊断。

# 7. Direction-Level Lessons for Future Experiments

- 不要把“球更小”写成“语义更好”。
- 不要把 negative-free 当成天然适合粒球的 encoder。
- 不要把 InfoNCE 的负样本排斥简单视为噪声，它也是全局分离机制。
- 新方法必须明确是在解决局部紧凑、全局分离、防塌缩三者之间的平衡。
- 所有 GB-GCL 实验必须报告 radius/AD、purity、linear accuracy、effective rank 和 feature std。
- 如果 BN 或其他稳定器改变结论，必须把它当成正式变量，而不是训练细节。

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| Experiment A / DIR-09 | 比较 raw、InfoNCE、negative-free 表示空间上的粒球质量 | negative-free 更紧但语义更差，BN 可恢复语义但失去小球优势 | 小半径主要来自过度压缩，不是语义结构提升 | 是，作为动机和诊断 | diagnostic-only |

# 9. Recommended Next Directions

## NEXT-DIR-A1: Stable encoder + soft GB relation

### Why This Direction Is Recommended

A 组说明局部紧凑不能脱离全局可分。更合理的方向是在稳定 InfoNCE 或 BN encoder 上加入温和的 GB relation。

### Previous Failure It Addresses

避免 raw negative-free collapse 和 hard pulling 过强。

### Core Hypothesis

粒球可以提供局部关系先验，但必须以 soft distribution 或 confidence 权重进入 loss。

### GB Role

weak-view relation teacher / soft local prior。

### Minimal Implementation

保持 InfoNCE baseline 不变，只添加小权重 soft relation loss，并加入 shuffled / size-matched random 控制。

### Minimal Datasets and Seeds

Cora、CiteSeer，seeds 0/1/2。

### Success Criteria

accuracy 不低于 baseline 的 pair 至少 4/6，effective rank 不明显下降，真实球优于随机球。

### Failure Criteria

随机球复现同等收益，或收益只体现在 radius/AD。

### Paper Claim If Successful

GB 可以作为稳定 encoder 上的 soft local structure prior。

### Reason to Stop Early

如果真实球和随机球无法区分，不进入调参。

# 10. Paper Claim Implications

## Claim-A1

### Claim

负样本排斥和粒球局部紧凑目标之间存在可测张力。

### Supported By Which Directions

DIR-09。

### Weakened By Which Directions

None, but BN negative-free shows the statement must be phrased carefully。

### Current Evidence

InfoNCE radius/AD 较大但 acc/purity 较高；negative-free radius/AD 较小但 acc/purity 较低；BN negative-free 恢复语义质量时 radius/AD 变大。

### Missing Evidence

更多数据集和更严格的统计显著性检验。

### Required Experiments

主方法阶段继续使用随机 GB 控制和 anti-collapse 诊断。

### Risk

不能把该 claim 写成“negative-free 优于 InfoNCE”。

### Current Readiness

moderate as motivation, not ready as main method claim

## Claim-A2

### Claim

小粒球半径不是粒球语义质量的充分证据。

### Supported By Which Directions

DIR-09, DIR-10。

### Weakened By Which Directions

None。

### Current Evidence

`negfree_z` 和 B3/B4 都展示了 radius/AD 改善但语义指标恶化。

### Missing Evidence

可视化和更多 collapse 指标。

### Required Experiments

在所有后续实验中保留 radius/AD 与 rank/acc/purity 的成对报告。

### Risk

如果论文中不强调该点，容易被质疑“只优化了几何指标”。

### Current Readiness

strong as diagnostic claim

# 11. TODO List

## Missing raw logs

- A 组主要证据有 config snapshots、metrics.csv 和 embeddings；如果要写论文附录，还需要整理每个 run 的 stdout/stderr 原始训练日志。

## Missing baseline checks

- 需要在论文前确认 A 组和后续 B/C 组是否使用完全一致的 Planetoid split、linear evaluation 次数和训练协议。

## Missing seeds/std

- A 组主结果有 5 seeds；稳定性消融有 3 seeds。后续如果引用 BN 结论，最好补 5 seeds 或至少说明只做 3 seeds。

## Missing datasets

- 稳定性消融没有 PubMed。不能把 BN / augmentation ablation 的结论直接推广到 PubMed。

## Missing ablations

- 如果继续 negative-free，BN、variance、EMA 需要作为正式 baseline，而不是只作为调试记录。

## Missing GB diagnostics

- 建议补充 ball size distribution、assignment stability、singleton ratio 的可视化。

## Missing paper-claim evidence

- A 组可支撑 motivation，不足以支撑主方法有效性。
