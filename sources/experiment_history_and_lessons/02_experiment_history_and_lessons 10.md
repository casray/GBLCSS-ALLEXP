# GB-GCL Experiment History and Lessons - Experiment B

本文档只整理实验 B 方向的证据。实验 B 对应 `gb_denoise_stage_b_compactness`：验证“粒球损失反向约束 encoder 是否能提升粒球生成质量”。它延续实验 A 的诊断结论，但从离线诊断进入训练期约束。本文档是独立补充，不覆盖旧的 `02_experiment_history_and_lessons.md`。

# 1. Current Overall Judgment

- Confirmed: 实验 B 正式测试了粒球 compactness loss 反向约束 encoder，而不只是离线观察表示空间。
- Confirmed: 第一版 Cora seed0 gate 没有真正测试 B1，因为粒球生成成一个超大球后被 size gate 全部过滤，`covered_nodes=0`，compactness 没有激活。
- Confirmed: size-aware split 修复后，B1 在 Cora seed0 上相对 B0 提升 acc、purity、radius/AD 和 effective rank；B3/B4 随机 assignment 虽然让球更小，但 accuracy 和 rank 崩溃。
- Confirmed: 多 seed 后，B1 不稳定。Cora 平均只小幅高于 B0，且 1/3 seed 提升；CiteSeer 平均低于 B0，seed2 大幅下降。
- Confirmed: 单纯调小 `lambda_gb` 不能解决稳定性。`0.0005` 最平衡但 mean delta acc 约为 0；`0.001` 有正均值但方差大；`0.0001` 太弱且平均下降。
- Confirmed: B2 动态刷新粒球不能解决 B1 的根本不稳定。最佳设置也只有 3/6 个 dataset-seed pair 不低于 B0，`lambda_gb=0.001` 动态刷新后 0/6 不低于 B0。
- Likely: 问题不只是粒球是否过时，而是 raw negative-free 表示空间上 hard compactness pulling 本身容易和全局语义可分性冲突。
- Risk: 继续在 raw negative-free + hard center pulling 上调 lambda、刷新间隔或单个 gate，收益会很低。
- Decision: abandon current form。放弃 B 当前形式，即“raw negative-free encoder + hard compactness 拉向粒球中心”作为主线；保留真实球 vs 随机球控制、activation diagnostics 和 size-aware split 作为后续工具。
- Next priority: 转向稳定 encoder 上的 soft / confidence / relation teacher 形式，而不是继续硬拉近。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-10 | Stage B GB compactness reverse-constraint encoder | 粒球作为训练期 compactness 约束，拉近同球节点到球中心 | B0, B1, B3, B4; size-aware split; lambda sweep; B2 dynamic refresh | Cora, CiteSeer; seeds 0/1/2 | Cora seed0 B1 有明显正信号；B3/B4 证明随机 compactness 有害 | B1/B2 跨 seed 不稳定，accuracy/rank 不能稳定优于 B0 | abandon | negative result / mechanism diagnostic |

# 3. Direction-Level Detailed Record

## DIR-10: Stage B GB Compactness Reverse-Constraint Encoder

### 3.1 Direction Summary

- Direction ID: DIR-10
- Direction name: Stage B GB compactness reverse-constraint encoder
- Time period: 2026-05-19 to 2026-05-21
- Target problem: 验证粒球结构是否能通过可优化的 compactness loss 反向塑造 encoder 表示空间，提高粒球质量和下游分类。
- Core hypothesis: 如果 warmup 后的粒球代表可靠局部语义结构，那么把节点表示拉向真实球中心应提升局部紧凑性，并在不破坏全局语义的情况下提高 accuracy / purity。
- Baseline: B0，negative-free SSL baseline，不加粒球约束。
- Main related folder: `gb_denoise_stage_b_compactness`
- Clean worktree: `C:/Users/Admin/Desktop/实验8去噪5.19-_stageb_fc62447_runs/gb_denoise_stage_b_compactness`
- Fixed commit: `fc62447`
- Main config: `gb_denoise_stage_b_compactness/configs/stage_b.yaml`
- Main code files: `gb_denoise_stage_b_compactness/src/gb_stage_b/train.py`, `gb_denoise_stage_b_compactness/src/gb_stage_b/balls.py`, `gb_denoise_stage_b_compactness/src/gb_stage_b/losses.py`, `gb_denoise_stage_b_compactness/src/gb_stage_b/report.py`
- Reports: `gb_denoise_stage_b_compactness/reports/2026-05-19_cora_seed0_formal_gate.md`, `gb_denoise_stage_b_compactness/reports/2026-05-20_cora_seed0_size_aware_gate.md`, `gb_denoise_stage_b_compactness/reports/2026-05-20_cora_citeseer_seed012_stability.md`, `gb_denoise_stage_b_compactness/reports/2026-05-20_lambda_gb_stability_sweep.md`, `gb_denoise_stage_b_compactness/reports/2026-05-21_b2_refresh_stability_sweep.md`
- Result files: `results/stage_b_summary.csv`, `results/stage_b_paired_deltas.csv`, `results_lambda_sweep/stage_b_summary.csv`, `results_lambda_sweep/stage_b_paired_deltas.csv`, `results_b2_refresh_sweep/stage_b_summary.csv`, `results_b2_refresh_sweep/stage_b_paired_deltas_vs_b0.csv`, `results_b2_refresh_sweep/stage_b_paired_deltas_vs_b1.csv`
- Current decision: abandon current form

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 仅靠自监督对比学习得到的表示空间不一定适合粒球划分，局部结构可能不够紧凑或不够稳定。
- Why GB might help: 粒球具有中心、半径、覆盖节点、球内平均距离等结构，可转化为可优化约束。
- Expected improvement: B1 应相对 B0 提升 radius/AD、label purity、linear accuracy，同时不降低 effective rank。
- Literature or previous experiment inspiration: CGBC / GBmooer 等方向都利用粒球反向影响 encoder 或表征；实验 A 也提示局部紧凑和全局可分需要平衡。
- Expected paper claim: 粒球可作为 label-free local structural constraint，帮助 encoder 学到更适合粒球划分和下游分类的表示空间。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset / Seeds | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| B0 | negative-free baseline，无粒球约束 | Cora, CiteSeer; seeds 0/1/2 | 作为 paired baseline | 无粒球约束时的表现 | baseline 本身 seed 方差较大 |
| B1-inactive gate | warmup 后固定生成粒球，但 size gate 拒绝超大球 | Cora seed0 | B0/B1/B3/B4 完全相同，compact=0 | 第一版是否激活机制 | 没有测试到 B1，需要修正建球 |
| B1-size-aware | 强制 oversized ball 继续 split，B1 真实激活 | Cora seed0 | B1 acc +0.076；B3/B4 acc -0.214 | 真实球 compactness 是否可能有效 | seed0 有正信号，随机 compactness 明显有害 |
| B1-stability | B0/B1/B3/B4 多 seed | Cora, CiteSeer; seeds 0/1/2 | B1 混合；B3/B4 全面崩 | B1 是否稳定 | 真球优于随机球，但 B1 不稳 |
| B1-lambda sweep | `lambda_gb=0.001/0.0005/0.0001` | Cora, CiteSeer; seeds 0/1/2 | 小 lambda 缓解但不稳定 | 是否只是原权重太强 | 单纯调小 lambda 不够 |
| B2-refresh | `ball_refresh_interval=50/100`，`lambda_gb=0.001/0.0005` | Cora, CiteSeer; seeds 0/1/2 | 最好 3/6 不低于 B0；0.001 为 0/6 | 固定 assignment 是否主要问题 | 动态刷新不能解决根本冲突 |

### 3.4 Role of Granular Ball

- GB role in B1: warmup 后在表示空间生成真实粒球，将节点拉向 detached ball center。
- GB role in B3: 使用 shuffled ball assignment，排除“随便压紧也有效”。
- GB role in B4: 使用 size-matched random assignment，排除“只是球大小分布有效”。
- GB role in B2: 训练中定期刷新真实球 assignment。
- Where GB is generated: encoder output `H`。
- Representation space: `ball_space=H`。
- Affects loss: yes，增加 compactness loss。
- Affects augmentation: no。
- Affects graph structure: no。
- Affects sampling: no。
- Affects evaluation: no。标签只用于后验 purity 和 linear evaluation。

### 3.5 Experimental Setup Summary

- Datasets: Cora, CiteSeer。
- Seeds: 0, 1, 2。
- Epochs: 1000。
- Warmup epochs: 300。
- SSL objective: negative-free。
- Ball space: H。
- Reliable ball min size: 3。
- Reliable ball max size: 256。
- Reliable ball AD percentile: 50。
- B1 lambda sweep: 0.001, 0.0005, 0.0001。
- B2 refresh interval: 50, 100。
- Baseline comparability: paired comparison under same clean detached worktree commit `fc62447`。

### 3.6 Result Summary

#### First Cora seed0 formal gate: mechanism inactive

| Method | Acc | Num balls | Radius | AD | Effective rank | Purity | compact_nonzero_epochs | covered_nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.344 | 1 | 0.2064 | 0.0290 | 2.4701 | 0.3021 | 0 | 0 |
| B1 | 0.344 | 1 | 0.2064 | 0.0290 | 2.4701 | 0.3021 | 0 | 0 |
| B3 | 0.344 | 1 | 0.2064 | 0.0290 | 2.4701 | 0.3021 | 0 | 0 |
| B4 | 0.344 | 1 | 0.2064 | 0.0290 | 2.4701 | 0.3021 | 0 | 0 |

这个结果不能作为 B1 负结果，因为 compactness 完全没有激活。

#### Size-aware Cora seed0 gate

| Method | Acc | Num balls | Radius | AD | Effective rank | Purity | compact_nonzero_epochs | covered_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.344 | 24 | 0.0389 | 0.0207 | 2.4701 | 0.4384 | 0 | 0 |
| B1 | 0.420 | 26 | 0.0293 | 0.0160 | 2.6149 | 0.5764 | 700 | 1661 |
| B3 | 0.130 | 37 | 0.0101 | 0.0065 | 1.6555 | 0.3947 | 700 | 1249 |
| B4 | 0.130 | 33 | 0.0122 | 0.0067 | 1.7120 | 0.3893 | 700 | 1059 |

seed0 正信号存在：真实 B1 比 B0 好；随机 assignment 虽然让球更小，但 accuracy 和 rank 崩溃。

#### Multi-seed stability gate

| Dataset | Method | Acc mean | Acc std |
|---|---|---:|---:|
| Cora | B0 | 0.4273 | 0.1146 |
| Cora | B1 | 0.4360 | 0.0812 |
| Cora | B3 | 0.1300 | 0.0000 |
| Cora | B4 | 0.1300 | 0.0000 |
| CiteSeer | B0 | 0.4503 | 0.1610 |
| CiteSeer | B1 | 0.3877 | 0.0448 |
| CiteSeer | B3 | 0.0770 | 0.0000 |
| CiteSeer | B4 | 0.0770 | 0.0000 |

B1 paired accuracy delta:

| Dataset | seed0 | seed1 | seed2 |
|---|---:|---:|---:|
| Cora | +0.076 | -0.016 | -0.034 |
| CiteSeer | +0.033 | +0.069 | -0.290 |

结果削弱了“B1 稳定有效”的 claim。B3/B4 全面崩溃支持“任意 compactness 有害”，但不能证明当前 B1 稳定有效。

#### Lambda sweep

| lambda_gb | Mean delta acc | Std delta acc | Mean delta effective rank | Mean delta AD | Mean delta purity |
|---:|---:|---:|---:|---:|---:|
| 0.001 | +0.0142 | 0.1454 | -0.9038 | -0.0084 | -0.0230 |
| 0.0005 | -0.0015 | 0.0974 | +0.1315 | -0.0041 | +0.0026 |
| 0.0001 | -0.0427 | 0.0988 | +0.6936 | -0.0004 | -0.0224 |

所有 B1 lambda 设置都激活 compactness，`compact_nonzero_epochs=700`，`covered_nodes_max>0`。因此失败不是“loss 没有生效”，而是生效后没有稳定收益。

#### B2 dynamic refresh

| lambda_gb | refresh_interval | Mean delta acc vs B0 | Std | Acc >= B0 | Mean delta rank | Mean delta AD | Mean delta purity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 50 | -0.0353 | 0.1294 | 3/6 | +1.7359 | +0.0086 | -0.0030 |
| 0.0005 | 100 | -0.0443 | 0.1046 | 2/6 | -0.4800 | -0.0048 | -0.0251 |
| 0.001 | 50 | -0.1400 | 0.1180 | 0/6 | -0.6675 | -0.0072 | -0.0466 |
| 0.001 | 100 | -0.1602 | 0.1235 | 0/6 | -0.6715 | -0.0062 | -0.0419 |

B2 compactness 也确实激活：B2 新 run 数 24，`compact_nonzero_epochs` 最小值 700，`covered_nodes_max` 最小值 1838。但动态刷新没有整体优于 B0，也没有整体优于静态 B1。

### 3.7 What Is Directly Supported

- size-aware splitting 是必要修正。没有它，B1 可能因为 `covered_nodes=0` 完全不生效。
- Cora seed0 显示真实粒球 compactness 有过正信号，但这个信号不能跨 seed 稳定复现。
- B3/B4 证明“随便把节点分组并压紧”是有害的。随机 assignment 可以得到更小 radius/AD，却严重损害 accuracy 和 rank。
- `lambda_gb` 太大时风险高，但调小 lambda 不足以让 B1 稳定。
- 动态刷新 assignment 不是 B1 不稳定的充分解决方案。
- 当前 B1/B2 结果继续支持：局部 compactness 目标和全局语义可分性存在冲突。

### 3.8 Reasonable Interpretation

- 解释 1: B1 的失败不是单一超参数问题，而是机制问题。raw negative-free 表示本来就偏压缩，再用粒球中心拉近，会进一步牺牲全局分离。
- 解释 2: fixed assignment 不是唯一问题。B2 刷新后仍然不稳，可能说明动态刷新会把 encoder 的短期波动反馈给粒球 assignment。
- 解释 3: 真实粒球结构不是完全无用。它至少比 B3/B4 随机 assignment 安全得多，但当前 hard compactness 形式太强。
- 解释 4: B 组应该被视为“失败形式，不是失败思想”。它推动后续 C 组改用稳定 InfoNCE encoder 和 soft relation。

### 3.9 What Cannot Be Concluded

- 不能说粒球反向约束 encoder 一定无效。B 组只否定了当前 raw negative-free + hard compactness 的形式。
- 不能说动态刷新永远无效。B2 没有测试 assignment churn、EMA assignment、稳定球筛选或 confidence refresh。
- 不能说 B1 的 seed0 提升可作为论文主 claim。多 seed 已经削弱它。
- 不能说 radius/AD 改善等于粒球质量改善。B3/B4 和 B2 都说明这点不成立。

### 3.10 Failure Analysis

- Main failure reason: hard compactness pulling 在 raw negative-free 表示空间上过强，容易破坏全局语义可分性。
- Secondary failure reason: warmup 后的粒球 assignment 对 seed 敏感，动态刷新也没有可靠性控制。
- Evidence: B1 multi-seed 中 Cora 只有 1/3 seed 改善，CiteSeer seed2 大幅下降；lambda sweep 和 B2 均未达到多数 pair 不低于 B0。
- Alternative explanation: B0 baseline 本身方差较大，导致 paired delta 受 seed 影响。但 B3/B4 全面崩溃和 B2 失败说明问题不只是 baseline 方差。
- Minimum check needed: 若想复活该思想，必须换稳定 encoder，并加入 confidence / anti-collapse / soft relation 控制。

### 3.11 Lessons Learned

- Lesson: 机制是否激活必须先检查。Evidence: 第一版 B1 compactness 为 0，所有方法完全相同。Future constraint: 每个 run 必须报告 `compact_nonzero_epochs` 和 `covered_nodes_max`。
- Lesson: 真实 assignment 和随机 assignment 的差异很重要。Evidence: B1 seed0 改善，B3/B4 崩溃。Future constraint: 每个 GB loss 都要有 shuffled 和 size-matched random 控制。
- Lesson: 小 lambda 不是万能修复。Evidence: lambda sweep 中 0.0005 平均接近 0，0.0001 平均下降。Future constraint: 不要继续只调 lambda。
- Lesson: 动态刷新不是自然稳定器。Evidence: B2 最好只 3/6 不低于 B0，0.001 全部低于 B0。Future constraint: 刷新必须配合 assignment stability / churn 诊断。
- Lesson: hard compactness 对 raw negative-free 风险过高。Evidence: 多 seed 不稳定，B2 不解决。Future constraint: 后续优先 soft relation、confidence weighting、stable encoder。

### 3.12 What Is Still Useful

- size-aware ball split 代码和测试仍有用，因为它保证粒球不会被超大球吞掉。
- `compact_nonzero_epochs`、`covered_nodes_max`、`covered_nodes_last` 是必要 activation diagnostics。
- B3/B4 控制组设计非常有价值，后续 C 组和其他 GB loss 都应保留。
- B 组负结果为论文提供边界：hard center pulling 不是安全主线，GB 更适合作为 soft prior。

### 3.13 Decision

- Decision: abandon current form。
- Reason: B1/B2 多 seed、多 lambda、动态刷新后仍无法稳定优于 B0，继续在该形式上微调收益低。
- What should not be tried again: 不要继续 raw negative-free + hard ball center pulling 的小范围 lambda/refresh 调参；不要扩展 PubMed 来“赌”结果。
- Variant still worth trying: 稳定 InfoNCE 或 BN negative-free encoder 上的 soft relation、confidence-weighted relation、assignment stability filtered balls。
- Evidence required to revive this direction: 真实球必须在 Cora/CiteSeer 至少 4/6 pair 不低于 baseline，且优于 shuffled/size-random 控制，同时 effective rank 不下降。

# 4. Cross-Direction Failed Pattern Ledger

## FP-B1: Hard compactness pulling is unstable

### Status

confirmed for raw negative-free encoder

### Related Directions

DIR-10, DIR-09, DIR-08

### Observation

B1/B2 在 raw negative-free encoder 上不能稳定提升 accuracy；C 组也显示 hard compactness 会压低 rank。

### Possible Reason

局部拉近目标会和全局分离目标冲突；如果 encoder 已经偏压缩，compactness loss 会放大这个问题。

### Evidence

B1 多 seed mixed，lambda sweep 不稳定，B2 dynamic refresh 失败。

### Future Constraint

不要把 hard center pulling 作为主方法，除非有稳定 encoder、soft weight 和 anti-collapse gate。

## FP-B2: Dynamic refresh without reliability control can fail

### Status

confirmed in B2 setting

### Related Directions

DIR-10

### Observation

B2 每 50/100 epoch 刷新粒球后没有整体优于静态 B1 或 B0。

### Possible Reason

刷新可能让不稳定 assignment 随 encoder 波动传播，而不是修正旧球。

### Evidence

`lambda_gb=0.0005, refresh=50` 只有 3/6 不低于 B0；`lambda_gb=0.001` 为 0/6。

### Future Constraint

任何动态刷新必须记录 assignment churn，并用稳定球或 EMA 筛选。

## FP-B3: Arbitrary compactness is actively harmful

### Status

confirmed

### Related Directions

DIR-10

### Observation

B3/B4 随机 assignment 全面低于 B0。

### Possible Reason

错误分组会把不同语义节点强行压到一起，直接破坏表示空间。

### Evidence

Cora B3/B4 acc 均值 0.130；CiteSeer B3/B4 acc 均值 0.077；所有 seed 均明显低于 B0。

### Future Constraint

每个 GB loss 必须有随机 assignment 控制；如果真实球不能优于随机球，不能声称 GB 结构有效。

# 5. Cross-Direction Positive Signal Ledger

## PS-B1: True GB assignment is safer than random assignment

### Status

confirmed

### Related Directions

DIR-10

### Observation

B1 虽不稳定，但明显优于 B3/B4；随机 compactness 总是有害。

### Why It May Matter

说明粒球结构不是完全无意义，问题主要在 hard loss 形式和 encoder 稳定性。

### Required Verification

后续真实球信号必须在 soft relation 中仍优于 shuffled / size-matched random。

### Possible Paper Use

negative result / ablation

## PS-B2: Activation diagnostics are mandatory and useful

### Status

confirmed

### Related Directions

DIR-10

### Observation

第一版 B1 因 `covered_nodes=0` 失效，如果没有 activation diagnostics，会被误判为方法无效。

### Why It May Matter

GB loss 很容易被 size gate、coverage、reliable mask 静默关闭。

### Required Verification

所有后续训练日志必须保留 `compact_nonzero_epochs`、`covered_nodes_max`、loss scale。

### Possible Paper Use

appendix / experimental rigor

## PS-B3: Size-aware splitting is a reusable implementation improvement

### Status

confirmed

### Related Directions

DIR-10, DIR-08

### Observation

强制 oversized ball 继续 split 后，compactness 才真正激活。

### Why It May Matter

它避免粒球生成退化成一个超大球，保证后续 GB 机制有可用覆盖。

### Required Verification

在 PubMed 或更大图上需要检查球规模分布和 coverage。

### Possible Paper Use

method detail / appendix

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| raw negative-free + hard compactness center pulling | 多 seed 不稳定，B2 不解决 | B1 stability、lambda sweep、B2 refresh | 换稳定 encoder，且真实球 4/6 pair 优于 baseline |
| 只继续调小 `lambda_gb` | 0.0005 约等于无收益，0.0001 变弱且下降 | lambda sweep | 新增可靠性机制，而非单调权重 |
| 只加动态刷新 | refresh 50/100 没有稳定提升 | B2 report | assignment churn + stable ball filter + EMA |
| 用 radius/AD 作为成功标准 | B3/B4 radius/AD 更好但 acc/rank 崩 | size-aware gate and stability gate | success 必须同时看 acc、rank、purity |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| hard ball-center pulling | 真实球 assignment 比随机 assignment 更可靠 | soft relation alignment / confidence-weighted relation |
| fixed warmup assignment | 粒球结构需要随表示变化 | 只刷新稳定球，记录 churn，不直接全量替换 |
| raw negative-free baseline | negative-free 可产生局部紧凑倾向 | BN-stabilized negative-free 或 InfoNCE baseline |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: 同 dataset / split / seed / epoch / evaluation。
- Multi-seed evidence: 至少 Cora、CiteSeer seeds 0/1/2。
- Multi-dataset evidence: 不接受只在 Cora seed0 提升。
- Ablation removing GB signal: B3/B4 或等价 shuffled / size-matched random。
- Mechanism diagnostic: compactness/relation 必须真实激活。
- No label leakage: label purity 只能作为后验诊断。

# 7. Direction-Level Lessons for Future Experiments

- 实验开始前必须先确认 loss 是否激活，不能只看最终 accuracy。
- 如果一个 GB 约束会让随机 assignment 也表现好，那么它不是可靠 GB 机制。
- 如果一个 GB 约束让随机 assignment 崩溃，但真实 assignment 也不稳定，说明真实球有信号但当前 loss 太强。
- 动态刷新要有 churn 诊断，否则无法判断它是在修正旧结构还是引入新噪声。
- 不要在 B 当前形式上直接扩展 PubMed。小图多 seed 不稳时，大图只会增加成本。
- 后续优先从 hard compactness 转向 soft relation、confidence 和 denoising。

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| Experiment B / DIR-10 | 用真实粒球 compactness loss 反向约束 raw negative-free encoder | seed0 有正信号，但多 seed、lambda sweep、B2 refresh 均不稳定 | hard compactness 与全局可分冲突，且 assignment seed-sensitive | 是，作为负结果、控制组和诊断工具 | abandon current form |

# 9. Recommended Next Directions

## NEXT-DIR-B1: Stable encoder + confidence-weighted soft relation

### Why This Direction Is Recommended

B 组说明 hard compactness 过强，A 组说明 raw negative-free 容易压缩。下一步应使用稳定 encoder，并把 GB 约束变成软关系。

### Previous Failure It Addresses

避免 hard pulling、raw negative-free collapse 和动态刷新无诊断。

### Core Hypothesis

真实粒球可作为 soft local relation prior，但不能直接把节点拉到球中心。

### GB Role

confidence-weighted soft relation teacher。

### What Must Be Different From Failed Directions

不使用 hard center pulling；必须保留 shuffled / size-random；必须报告 rank/feature std。

### Minimal Implementation

在 InfoNCE baseline 上加入弱权重 relation alignment，只对高置信稳定球生效。

### Minimal Datasets and Seeds

Cora、CiteSeer，seeds 0/1/2。

### Success Criteria

accuracy >= baseline 至少 4/6，mean delta rank >= 0 或轻微下降可解释，真实球优于随机球。

### Failure Criteria

随机球复现收益，或 accuracy/purity 下降但 radius/AD 改善。

### Paper Claim If Successful

粒球可作为 soft local structure prior，而不是 hard compactness target。

### Reason to Stop Early

如果 controls 与真实球接近，停止调参并重设机制。

## NEXT-DIR-B2: Assignment stability filtered GB

### Why This Direction Is Recommended

B2 失败可能与刷新时 assignment 波动有关。只使用跨 epoch 稳定的球，可能比全量刷新更安全。

### Previous Failure It Addresses

固定球过时和动态刷新噪声两种问题。

### Core Hypothesis

只有跨 epoch 稳定的粒球才适合作为训练约束。

### GB Role

stable local prior / reliability filter。

### Minimal Implementation

记录相邻 refresh 的节点球归属变化，筛选低 churn balls，再用于 soft relation。

### Minimal Datasets and Seeds

Cora、CiteSeer，seeds 0/1/2。

### Success Criteria

相比普通 refresh，accuracy/rank 更稳定，并且真实稳定球优于随机稳定球。

### Failure Criteria

覆盖率过低或稳定球与随机球无差异。

### Paper Claim If Successful

粒球稳定性可作为 GB-GCL 可靠性信号。

### Reason to Stop Early

如果 stable balls 覆盖节点太少，不能作为主方法。

# 10. Paper Claim Implications

## Claim-B1

### Claim

粒球 hard compactness loss 可以直接提升 encoder 表示空间。

### Supported By Which Directions

Cora seed0 size-aware gate only。

### Weakened By Which Directions

B1 multi-seed stability、lambda sweep、B2 dynamic refresh。

### Current Evidence

单 seed 正信号存在，但跨 seed 和跨设置不稳定。

### Missing Evidence

稳定多 seed、多数据集正收益。

### Required Experiments

若要复活，必须换稳定 encoder 和 soft/weighted loss。

### Risk

当前证据不足，直接写成主 claim 会被负结果反驳。

### Current Readiness

not ready

## Claim-B2

### Claim

任意 compactness 不是有效机制，真实粒球 assignment 必须优于随机 assignment。

### Supported By Which Directions

DIR-10。

### Weakened By Which Directions

None in B。

### Current Evidence

B3/B4 全 seed 崩溃，B1 至少明显优于随机 compactness。

### Missing Evidence

在 soft relation 方向中重复该控制。

### Required Experiments

C 组或后续方向中继续保留 shuffled / size-matched random。

### Risk

如果后续 soft relation 中随机球也有效，真实 GB 机制 claim 会被削弱。

### Current Readiness

moderate as ablation rule

## Claim-B3

### Claim

动态刷新粒球能解决 warmup 后固定 assignment 的不稳定。

### Supported By Which Directions

None。

### Weakened By Which Directions

B2 dynamic refresh。

### Current Evidence

refresh 50/100 均未稳定优于 B0 或静态 B1。

### Missing Evidence

assignment churn、EMA refresh、stable ball filtering。

### Required Experiments

如果继续做刷新，必须加入稳定性诊断，而不是只换 interval。

### Risk

当前简单刷新方向已被负结果明显削弱。

### Current Readiness

weak

# 11. TODO List

## Missing raw logs

- B 组有 reports 和 CSV；如果要写论文附录，需要整理每个 run 的 stdout/stderr 原始日志。

## Missing baseline checks

- B0 复用 baseline 是 paired 可比的，但论文前需要确认 `fc62447` 的 evaluation protocol 与 C 组完全一致。

## Missing seeds/std

- Cora/CiteSeer 有 seeds 0/1/2。若将 B 组作为负结果附录，3 seeds 足够；若复活为主方法，需要更多 seeds。

## Missing datasets

- PubMed 暂缓是合理的。B 当前形式不应补 PubMed，除非机制先在小图稳定。

## Missing ablations

- 缺 BN-stabilized negative-free + B1/B2 对照；不过该实验更像新方向，不属于当前 B 硬拉近主线。

## Missing GB diagnostics

- B2 缺 assignment churn，无法判断刷新是在修正旧球还是引入噪声。

## Missing visualizations

- 可补 t-SNE / UMAP 或 ball-size distribution，帮助解释过度压缩。

## Missing paper-claim evidence

- B 当前形式不能支持“GB loss 直接提升 encoder”主 claim，只能支持“hard compactness 风险”和“随机 compactness 有害”的负结果。
