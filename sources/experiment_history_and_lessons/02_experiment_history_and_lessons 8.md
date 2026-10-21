# GB-GCL Experiment History and Lessons - Experiment C

本文档只整理实验 C 方向的证据。实验 C 对应 `gb_denoise_stage_b_compactness` 中的 Stage C：在 InfoNCE encoder 上测试粒球约束能否以更温和的方式反向影响表示空间，尤其是 C4 soft relation alignment。它是对既有 `02_experiment_history_and_lessons.md` 和 `02_experiment_history_and_lessons 7 .md` 的独立补充，不覆盖旧方向。

# 1. Current Overall Judgment

- Confirmed: 实验 C 是一个新方向：粒球不再主要用于改 view、过滤负样本或 noisy-label classifier，而是作为训练期 encoder 表示空间上的辅助约束。
- Confirmed: C1/C2/C3 的 hard compactness 在稳定 InfoNCE encoder 上没有复现 B1 式崩溃，C1 在 6/6 个 Cora/CiteSeer seed pair 上 accuracy 不低于 C0。
- Confirmed: hard compactness 虽然 accuracy 稳定，但会带来 effective rank 小幅下降。C1 平均 `delta_effective_rank=-0.386`，C2/C3 平均 `-0.297`。
- Confirmed: C2 的 label-free confidence weighting 确实激活，但没有明显优于 C1。C2 相对 C1 只有 3/6 个 pair 不低于。
- Confirmed: C3 的 anti-collapse gate 在当前阈值下完全没有触发，`total_gate_off_epochs=0`，因此本轮 C3 实际等价于 C2。
- Confirmed: C4 soft relation alignment 没有明显提升 accuracy，但与 C1/C2 不同，它让 effective rank 平均上升。C4_current 平均 `delta_effective_rank=+0.115`。
- Confirmed: C4 Round1 控制实验削弱了“真实粒球结构导致 rank 改善”的解释。`C4_shuffled` 和 `C4_size_random` 也能得到近似 rank 增益，分别为 `+0.113` 和 `+0.106`。
- Decision: modify。当前 C4 不能继续作为论文主方法调参；它应降级为机制诊断结果，后续若继续必须重设计 relation teacher，使信号真正依赖可靠粒球结构。
- Next priority: 不进入 C4 Round2 的 `lambda/topk/tau` 调参；下一步应测试更严格的真实粒球依赖机制，例如跨 epoch 稳定球、ball-to-ball relation、或噪声结构场景中的真实球 vs 随机球差异。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-08 | Stage C GB constraint for encoder representation | 粒球作为 hard compactness、confidence-weighted compactness、防塌缩门控、soft node-to-ball relation teacher | C0, C1, C2, C3, C4; C4_no_conf, C4_shuffled, C4_size_random, C4_refresh100 | Cora, CiteSeer | C1 accuracy 稳定不低于 C0；C4 rank 不降反升 | C4 rank 信号可被随机球复现；C2/C3 没有证明额外机制价值 | modify | negative / diagnostic / future soft-relation motivation |

# 3. Direction-Level Detailed Record

## DIR-08: Stage C GB Constraint for Encoder Representation

### 3.1 Direction Summary

- Direction ID: DIR-08
- Direction name: Stage C GB Constraint for Encoder Representation
- Time period: 2026-05-21 to 2026-05-25
- Target problem: 验证粒球结构能否作为训练期局部结构先验，反向约束 encoder 表示空间，同时避免 B1 中局部紧凑与全局语义可分性的冲突。
- Core hypothesis: 如果 encoder baseline 更稳定，并且粒球约束从 hard pulling 改为 confidence-weighted 或 soft relation teacher，则粒球可以保留局部结构而不破坏全局表示。
- Baseline: C0，即 InfoNCE encoder，不加粒球训练约束。
- Main related worktree: `C:/Users/Admin/Desktop/实验8去噪5.19-_stagec_implementation/gb_denoise_stage_b_compactness`
- Main code files: `src/gb_stage_b/train.py`, `src/gb_stage_b/compactness.py`, `src/gb_stage_b/confidence.py`, `src/gb_stage_b/gates.py`, `src/gb_stage_b/relations.py`
- Config files: `configs/stage_c/*.yaml`, `results_stage_c/_configs/*.yaml`, `results_c4_next/round1_controls/_configs/*.yaml`
- Result files: `results_stage_c/stage_c_summary.csv`, `results_stage_c/stage_c_method_summary.csv`, `results_stage_c/stage_c_paired_deltas_full.csv`, `results_c4_next/round1_controls/c4_round1_setting_summary.csv`, `results_c4_next/round1_controls/c4_round1_paired_deltas_vs_c0.csv`
- Reports: `results_stage_c/2026-05-21_stage_c_formal_results.md`, `results_c4_next/round1_controls/2026-05-25_c4_round1_controls.md`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 普通对比学习需要全局分离，粒球局部聚合目标可能与之冲突；B1/B2 已显示 hard compactness 在 negative-free encoder 上不稳定。
- Why GB might help: 粒球可以作为局部语义结构单元，提供中心、半径、置信度和节点到球的软关系。
- Expected improvement: 在 InfoNCE 稳定表示空间上，粒球约束应减少 B1 式崩溃；soft relation 应比 hard pulling 更不伤全局语义可分性。
- Literature or previous experiment inspiration: CGBC、GBmooer、CLSA 弱视图教师思想；项目偏好中也明确将 GB 视为 soft structural constraint / weak view relation teacher。
- Expected paper claim: 粒球可作为 weak-view local relation teacher，温和约束 encoder 表示空间。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset / Seeds | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| C0 | InfoNCE baseline，不加粒球约束 | Cora, CiteSeer; seeds 0/1/2 | Cora acc 0.7790 +/- 0.0010；CiteSeer acc 0.6357 +/- 0.0188 | 建立 C 组 paired baseline | baseline 健康且可配对 |
| C1 | C0 + 原始粒球 compactness | Cora, CiteSeer; seeds 0/1/2 | 6/6 pair accuracy 不低于 C0；mean delta acc +0.0030；mean delta rank -0.386 | 稳定 InfoNCE encoder 上 hard compactness 是否仍崩 | 不崩，但压低 rank |
| C2 | C1 + label-free confidence weighting | Cora, CiteSeer; seeds 0/1/2 | 5/6 pair 不低于 C0；mean delta acc +0.0023；mean confidence 0.316；mean delta rank -0.297 | 置信度是否减少错误球伤害 | 有激活，但不明显优于 C1 |
| C3 | C2 + anti-collapse gate | Cora, CiteSeer; seeds 0/1/2 | 与 C2 完全一致；total gate_off_epochs=0 | rank/std 门控是否保护表示空间 | 当前阈值无实际干预 |
| C4_current | node-to-ball soft relation alignment | Cora, CiteSeer; seeds 0/1/2 | 3/6 pair 不低于 C0；mean delta acc +0.0008；mean delta rank +0.115 | soft relation 是否比 hard compactness 更安全 | rank 更安全，但 accuracy 未明显提升 |
| C4_no_conf | C4 关闭 confidence weight | Cora, CiteSeer; seeds 0/1/2 | 3/6 pair 不低于 C0；mean delta acc -0.0003；mean delta rank +0.093 | confidence weight 是否必要 | rank 信号不依赖 confidence weight |
| C4_shuffled | C4 使用 shuffled assignment | Cora, CiteSeer; seeds 0/1/2 | 4/6 pair 不低于 C0；mean delta acc +0.0007；mean delta rank +0.113 | 排除随便球关系也有效 | 随机 assignment 也能复现 rank 增益 |
| C4_size_random | C4 使用 size-matched random assignment | Cora, CiteSeer; seeds 0/1/2 | 4/6 pair 不低于 C0；mean delta acc +0.0008；mean delta rank +0.106 | 排除仅球大小分布有效 | 保留大小分布的随机球也接近 current |
| C4_refresh100 | C4 每 100 epoch 刷新粒球 | Cora, CiteSeer; seeds 0/1/2 | 4/6 pair 不低于 C0；mean delta acc +0.0010；mean delta rank +0.105 | 固定一次球是否限制 C4 | 动态刷新未明显优于 current/random controls |

### 3.4 Role of Granular Ball

- GB role in C1: hard compactness target，直接把节点表示拉向可靠球中心。
- GB role in C2: confidence-weighted compactness target，用 size / AD / radius 的无标签置信度降低低可信球影响。
- GB role in C3: C2 加 rank/std gate，理论上在表示空间健康度下降时关闭粒球损失。
- GB role in C4: weak view node-to-ball distribution teacher，用 KL 对齐 strong view 的 node-to-ball distribution。
- Where GB is generated: warmup 后在 encoder output `H` 上生成 angular balls；C4_refresh100 每 100 epoch 刷新。
- Representation space: `ball_space=H`；不是 projector `Z`。
- Affects loss: yes；C1-C3 影响 compactness loss，C4 影响 relation loss。
- Affects augmentation: no；仍使用 GRACE/InfoNCE 的原始图增强。
- Affects graph structure: no。
- Affects evaluation: no；linear eval protocol 保持一致。

### 3.5 Experimental Setup Summary

- Datasets: Cora, CiteSeer。
- Seeds: 0, 1, 2。
- Encoder objective: InfoNCE。
- Epochs: 1000。
- Warmup: 300 epochs。
- Hidden/projector dim: 256 / 256。
- Edge drop / feature mask: 0.2 / 0.2。
- Ball generation space: encoder output `H`。
- Reliable ball size range: min 3, max 256。
- Reliable ball AD percentile: 50.0。
- C1-C3 loss weight: `lambda_gb=0.0005`。
- C4 relation weight: `lambda_rel=0.001`。
- C4 relation settings: `topk_centers=8`, `tau_ball=0.2`, `teacher_view=weak`, `student_view=strong`。
- C4 Round1 controls: `use_confidence_weight=false`, `assignment_control=shuffled`, `assignment_control=size_matched_random`, `ball_refresh_interval=100`。

### 3.6 Result Summary

#### Stage C formal summary

| Method | Pairs | Acc >= C0 | Mean delta acc | Std delta acc | Mean delta rank | Mean delta AD | Mean delta purity | Mechanism active |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C1 | 6 | 6 | +0.0030 | 0.0033 | -0.3859 | +0.0014 | +0.0112 | compact=700 epochs |
| C2 | 6 | 5 | +0.0023 | 0.0044 | -0.2967 | +0.0031 | +0.0038 | compact=700 epochs, confidence=0.316 |
| C3 | 6 | 5 | +0.0023 | 0.0044 | -0.2967 | +0.0031 | +0.0038 | compact=700 epochs, confidence=0.316, gate_off=0 |
| C4 | 6 | 3 | +0.0008 | 0.0038 | +0.1148 | +0.0037 | -0.0002 | relation=700 epochs, confidence=0.316 |

#### C4 Round1 control summary

| Setting | Pairs | Acc >= C0 | Mean delta acc | Std delta acc | Mean delta rank | Mean delta AD | Mean delta purity | Relation active |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C4_current | 6 | 3 | +0.0008 | 0.0038 | +0.1148 | +0.0037 | -0.0002 | 700 epochs |
| C4_no_conf | 6 | 3 | -0.0003 | 0.0033 | +0.0926 | +0.0005 | +0.0033 | 700 epochs |
| C4_refresh100 | 6 | 4 | +0.0010 | 0.0047 | +0.1054 | +0.0046 | -0.0037 | 700 epochs |
| C4_shuffled | 6 | 4 | +0.0007 | 0.0036 | +0.1134 | +0.0112 | -0.0032 | 700 epochs |
| C4_size_random | 6 | 4 | +0.0008 | 0.0047 | +0.1065 | +0.0076 | -0.0016 | 700 epochs |

### 3.7 What Is Directly Supported

- C1 说明 hard compactness 在稳定 InfoNCE encoder 上没有必然崩溃，和 B1 在 negative-free encoder 上的不稳定不同。
- C1/C2/C3 也说明 hard compactness 会压缩表示空间，accuracy 不崩不等于全局表示完全健康。
- C2 说明当前无标签置信度 proxy 可以工作，但不足以带来稳定优于 C1 的效果。
- C3 说明当前 rank/std 门控实现和阈值没有实际保护作用。
- C4 说明 soft relation alignment 比 hard compactness 更不容易压低 effective rank。
- C4 Round1 说明 C4 的 rank 增益不能归因于真实粒球结构，因为随机控制组也能复现。

### 3.8 Reasonable Interpretation

解释：C4 当前更像一种 node-to-center distribution consistency regularization，而不是一个明确利用语义粒球结构的 teacher。只要存在一组中心，teacher/student 对齐就可能带来表示平滑或 rank 变化；真实 granular-ball assignment 并不是该信号的必要条件。

另一个解释是：当前 C4 的 teacher 太弱，主要由节点到中心的 soft distribution 决定；而中心集合本身来自真实球、shuffled assignment 或 size-matched random assignment 时，对 loss 的影响差异不够大。因此 C4 可以作为“软一致性正则”的诊断，但不能作为“粒球结构有效”的主证据。

### 3.9 What Cannot Be Concluded

- 不能声称 C4 已验证“真实粒球结构”能提升 GRACE。
- 不能声称 C4 是当前可投稿主方法。
- 不能用 C4 的 effective rank 上升直接证明局部语义结构被保留。
- 不能继续只调 `lambda_rel`、`topk_centers` 或 `tau_ball` 来寻找论文主结果，除非先解决随机控制组接近的问题。
- 不能把 C1/C2 的小幅 accuracy 正向写成显著提升；delta 小，且没有统计显著性检验。
- 不能声称 confidence weighting 有明确价值；C2 不优于 C1，C4_no_conf 也没有明显输给 C4_current。

### 3.10 Failure Analysis

- Main failure reason: C4 的机制归因失败。真实粒球结构没有明显优于 shuffled 或 size-matched random 控制组。
- Secondary failure reason: C2 置信度 proxy 太粗，只让约束更保守，没有稳定筛掉错误球。
- Secondary failure reason: C3 gate 参考阈值过宽或诊断不敏感，导致从未关闭。
- Evidence: `results_c4_next/round1_controls/c4_round1_setting_summary.csv` 中 C4_shuffled `mean_delta_rank=+0.113`，C4_size_random `+0.106`，接近 C4_current `+0.115`。
- Alternative explanation: clean graph setting 可能不容易体现真实粒球结构优势；在 noisy graph 或更强 augmentation 下，真实球可能才与随机球拉开差距。
- Minimum check needed: 若复活 C4，必须先证明真实球 relation 明显优于 shuffled/size_random，而不是只比较 C4 vs C0。

### 3.11 Baseline Comparability Check

- Same dataset: yes，Cora 和 CiteSeer。
- Same split: yes，Planetoid split 未改。
- Same seeds: yes，seeds 0/1/2。
- Same encoder/projector: yes，hidden/proj dim 256。
- Same training protocol: yes，InfoNCE, epochs 1000, warmup 300。
- Same evaluation: yes，linear evaluation metrics 来自同一 `gb_stage_b` pipeline。
- Baseline rerun or copied: C0 来自同一 Stage C clean worktree 结果，作为 paired baseline 复用。
- Comparable conclusion: yes within Stage C and C4 Round1。

### 3.12 Label Leakage Check

- Labels used in GB generation: no evidence；balls are generated from encoder representations。
- Labels used in pretraining loss: no。
- Label purity used in training: no；only diagnostic metric。
- Validation/test labels used in training: no evidence。
- Leakage risk: low。

### 3.13 Granular-Ball Diagnostics

- Mechanism activation:
  - C1/C2/C3: `compact_nonzero_epochs=700`。
  - C4 and all C4 controls: `relation_nonzero_epochs=700`。
- Confidence:
  - C2/C3/C4_current mean confidence about 0.316。
  - C4_no_conf still records confidence but does not use it in relation weighting。
  - C4_shuffled and C4_size_random confidence slightly higher around 0.321, because reassignment changes ball statistics。
- Gate:
  - C3 `total_gate_off_epochs=0`。
- Missing diagnostics:
  - C4 teacher/student distribution entropy by setting。
  - Real vs random ball assignment NMI or overlap。
  - Center distribution and top-k center stability。
  - Cross-epoch ball stability for refresh100。
  - Runtime summary by setting。

### 3.14 What Is Still Useful

- C1/C2/C3 provide evidence that InfoNCE encoder is a safer base than negative-free encoder for GB constraints。
- C4 provides a useful rank-preserving soft relation implementation, even if not yet GB-specific。
- C4 controls provide an important anti-overclaim lesson: every soft relation method must include random structure controls。
- The vectorized node-to-ball distribution implementation is reusable for future ball-to-ball or stability-aware variants。
- `stage_c_method_summary.csv` and `c4_round1_setting_summary.csv` are clean paired-result artifacts for future reports。

### 3.15 Decision

- Decision: modify。
- Reason: current direction has useful signals but fails GB-specific mechanism attribution。
- What should not be tried again: do not directly tune current C4 `lambda_rel/topk/tau` as Round 2 before proving real-ball advantage。
- Variant still worth trying:
  - stability-filtered C4: only use balls stable across refreshes / seeds；
  - ball-to-ball relation teacher instead of node-to-all-centers teacher；
  - noisy graph or stronger augmentation setting where real local structure may matter more；
  - confidence based on cross-epoch assignment consistency rather than only size/AD/radius。
- Evidence required to revive as paper main line: C4-like method must beat shuffled and size-matched random controls on Cora/CiteSeer, preferably with 4/6 or more pair nonnegative accuracy, positive rank, and no purity collapse。

# 4. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| Direct C4 hyperparameter sweep before mechanism proof | Random controls already match rank signal; tuning may optimize soft consistency rather than GB structure | `c4_round1_setting_summary.csv`: shuffled +0.113 rank, size_random +0.106 rank | Real C4 must outperform shuffled/size_random on paired deltas |
| Claim C4 rank gain proves true GB relation | Rank gain appears even under randomized assignments | C4_current +0.115 rank; C4_shuffled +0.113; C4_size_random +0.106 | Assignment-overlap or stability diagnostic showing true balls carry unique signal |
| Treat confidence weighting as solved | C2 not better than C1; C4_no_conf not clearly worse than C4_current | Stage C formal summary and Round1 no_conf summary | Confidence ablation must show consistent benefit over no_conf |
| Use C3 gate as protection in current form | Gate never triggered | C3 `total_gate_off_epochs=0` | More sensitive gate or C0 epoch-reference curve that actually triggers on risky seeds |
| Evaluate GB soft relation without random controls | Random controls can look as good as real relation | C4 Round1 controls | Every future relation method must include shuffled and size-matched random controls |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| Hard compactness pulling | InfoNCE base can tolerate weak GB constraint | Use only as diagnostic or very weak auxiliary loss |
| Size/AD/radius confidence weighting | Confidence can reduce effective strength but is not enough | Add cross-epoch stability and relation consistency confidence |
| Current C4 node-to-center soft distribution | Rank-preserving soft relation is promising | Make teacher depend on stable true balls or ball-to-ball relations |
| C3 rank/std binary gate | Representation-health monitoring is useful | Use calibrated continuous gate with baseline reference trajectory |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: C0/GRACE same split, seed, encoder, epoch, eval。
- Multi-seed evidence: at least seeds 0/1/2; preferably 5 seeds if promoted。
- Multi-dataset evidence: Cora and CiteSeer minimum; PubMed only after mechanism passes。
- Ablation removing GB signal: shuffled assignment and size-matched random assignment are mandatory。
- Mechanism diagnostic: relation entropy, ball stability, true-vs-random assignment difference。
- No label leakage: no label purity in training。
- Effect threshold: must improve or preserve accuracy while not reducing rank/purity; small gains below std should be marked weak/inconclusive。

# 5. Direction-Level Lessons for Future Experiments

- Do not interpret effective rank improvement as GB-specific unless random structure controls fail to reproduce it。
- Hard compactness can be safe on InfoNCE but still compresses representation; report rank together with accuracy。
- Confidence weighting must be proven against no-confidence ablation, not assumed。
- Dynamic refresh is not automatically better; refresh must be evaluated against static current and random controls。
- Soft relation teacher should be redesigned to depend more directly on reliable GB structure。
- Every new GB relation loss must include at least one mechanism diagnostic beyond downstream accuracy。
- Do not enter PubMed or heavy hyperparameter sweeps before Cora/CiteSeer mechanism attribution is clear。

# 6. Current Multi-Direction Summary Addendum

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-08 Stage C encoder constraint | Use GB compactness / confidence / gate / soft relation to shape encoder | Accuracy mostly stable; C4 rank positive; but GB-specific mechanism not proven | Random center/assignment controls reproduce C4 rank signal | Yes, as diagnostic and redesign basis | modify |

# 7. Recommended Next Directions

## NEXT-DIR-C1: Stability-Filtered C4

### Why This Direction Is Recommended

当前 C4 的主要问题不是 loss 无效，而是真实粒球与随机控制差异不够。跨 epoch 稳定的 ball assignment 可能比单次 warmup ball 更能代表真实局部结构。

### Previous Failure It Addresses

解决 C4_current 与 shuffled/size_random 接近的问题。

### Core Hypothesis

只有稳定出现的粒球关系才应该作为 teacher；不稳定球应降权或排除。

### GB Role

cross-epoch stable local relation teacher。

### What Must Be Different From Failed Directions

不能只用 size/AD/radius confidence；必须加入 assignment stability 或 center stability。

### Minimal Implementation

在 warmup 后每隔固定 epoch 构球，计算节点 assignment consistency；只对高稳定球或高稳定 node-to-ball relation 施加 C4 loss。

### Minimal Datasets and Seeds

Cora, CiteSeer; seeds 0/1/2。

### Success Criteria

真实稳定球 C4 至少 4/6 accuracy 不低于 C0，平均 rank 为正，并且优于 shuffled/size_random。

### Failure Criteria

随机控制仍能复现 rank/accuracy，或稳定筛选导致 coverage 太低。

### Paper Claim If Successful

Stable granular-ball relations provide a reliable local structure prior for graph contrastive learning。

### Reason to Stop Early

如果 stable coverage 低于 30% 或 loss 大量为 0，停止。

## NEXT-DIR-C2: Ball-to-Ball Relation Teacher

### Why This Direction Is Recommended

当前 node-to-all-centers distribution 可能太像普通中心一致性。ball-to-ball relation 可能更直接表达粒球结构。

### Previous Failure It Addresses

解决 C4 主要依赖中心集合而非真实 assignment 的风险。

### Core Hypothesis

真实 ball graph 的局部关系比随机 ball graph 更能提供结构约束。

### GB Role

soft ball relation graph / local structure teacher。

### What Must Be Different From Failed Directions

必须比较 real ball graph、shuffled ball graph、size-matched random graph。

### Minimal Implementation

构建 ball centers 之间的 top-k relation，将 node relation 聚合到 ball-level，再用 weak/strong view 对齐 ball relation distribution。

### Minimal Datasets and Seeds

Cora, CiteSeer; seeds 0/1/2。

### Success Criteria

real ball relation 明显优于 random controls，且不降低 effective rank。

### Failure Criteria

ball-level relation 太粗导致 accuracy 下降，或 random controls 接近。

### Paper Claim If Successful

Granular-ball relation graph can serve as a multi-granularity structural teacher。

### Reason to Stop Early

如果 num_balls 太少或 relation density 不稳定，先做诊断不进训练。

## NEXT-DIR-C3: Noisy-Structure C4 Diagnostic

### Why This Direction Is Recommended

clean graph 上真实粒球与随机控制差距小。噪声边或强增强场景可能更能体现真实局部结构的鲁棒价值。

### Previous Failure It Addresses

解决 clean Cora/CiteSeer 上 accuracy delta 太小、机制差异不显著的问题。

### Core Hypothesis

真实粒球结构在 noisy graph 或强 augmentation 下比随机中心更能保持局部语义。

### GB Role

robust local structure prior。

### What Must Be Different From Failed Directions

必须先固定 C4 variant，不边跑边调；必须保留 clean/noisy 两套结果。

### Minimal Implementation

在已有 C4_current、C4_shuffled、C4_size_random 中加入 controlled edge noise 或 stronger edge_drop/feat_mask。

### Minimal Datasets and Seeds

Cora, CiteSeer; seeds 0/1/2；noise level 至少 1 个。

### Success Criteria

noisy setting 下 real C4 相对 random controls 明显更稳。

### Failure Criteria

noisy setting 中 random controls 仍接近或更好。

### Paper Claim If Successful

Granular-ball relation teacher improves robustness under graph perturbation。

### Reason to Stop Early

clean 和 noisy 都没有 real-vs-random gap。

# 8. Paper Claim Implications

## Claim-C1

### Claim

GB can directly constrain encoder representation through compactness.

### Supported By Which Directions

C1 accuracy 6/6 不低于 C0。

### Weakened By Which Directions

C1/C2/C3 均降低 effective rank；B1/B2 在 negative-free encoder 下不稳定。

### Current Evidence

Weak / diagnostic only。

### Missing Evidence

更强 rank/purity 保护和跨更多数据集结果。

### Required Experiments

除非有新机制，否则不建议继续。

### Risk

会被解释为过平滑或表示压缩。

### Current Readiness

weak。

## Claim-C2

### Claim

GB soft relation teacher is safer than hard compactness.

### Supported By Which Directions

C4_current rank +0.115；C1/C2/C3 rank 为负。

### Weakened By Which Directions

C4_shuffled 和 C4_size_random 也产生近似 rank 增益。

### Current Evidence

Moderate for “soft relation is safer”，weak for “GB-specific soft relation is useful”。

### Missing Evidence

真实球必须优于 random controls。

### Required Experiments

Stability-filtered C4 or ball-to-ball relation with random controls。

### Risk

方法实际只是 generic consistency regularization。

### Current Readiness

weak to moderate, depending on claim wording。

## Claim-C3

### Claim

GB confidence weighting reduces noisy relation damage.

### Supported By Which Directions

C2/C3 mean rank drop smaller than C1；confidence mechanism activated。

### Weakened By Which Directions

C2 不明显优于 C1；C4_no_conf 不明显差于 C4_current。

### Current Evidence

weak。

### Missing Evidence

confidence vs no-confidence 的稳定收益。

### Required Experiments

stability-based confidence, confidence ablation, and per-ball error diagnostics。

### Risk

当前 confidence 只是缩小 loss 强度，不是识别可靠语义球。

### Current Readiness

not ready。

# 9. TODO List

## Missing raw logs

- TODO: 如果要写论文附录，需要整理 `results_stage_c/run_logs/` 和 `results_c4_next/round1_controls/run_logs/` 的运行耗时表。

## Missing baseline checks

- TODO: 若未来扩展到 PubMed，必须先确认 C0 PubMed baseline 健康。

## Missing seeds/std

- TODO: 若某个 C4 改进方向进入主线，需扩展到 5 seeds 或做 paired significance test。

## Missing datasets

- TODO: 当前只有 Cora/CiteSeer；PubMed 暂缓是合理的，因为机制未通过。

## Missing ablations

- TODO: future C4 必须继续保留 shuffled 和 size-matched random controls。

## Missing GB diagnostics

- TODO: relation entropy、assignment stability、real-vs-random overlap、center stability、ball count/radius by seed。

## Missing visualizations

- TODO: 可视化 C4_current vs shuffled 的 node-to-ball distribution entropy 和 top-k center overlap。

## Missing explanation for failed directions

- TODO: 进一步解释为什么 random assignment 也能提升 rank，可能需要 loss geometry 分析。

## Missing paper-claim evidence

- TODO: 目前没有足够证据支持 “true granular-ball relation improves GCL”；只能支持 “hard compactness has compression risk; soft relation is safer but not yet GB-specific”。

