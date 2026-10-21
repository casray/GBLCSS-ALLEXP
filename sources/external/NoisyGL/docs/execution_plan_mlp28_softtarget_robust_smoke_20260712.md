# MLP-28 Soft-Target Robust Optimization Smoke

## Research hypothesis

LCSS 的局部共识 soft target 先修复监督信号，LogitClip 或 small-loss 再抑制训练期对残余错误监督的记忆。若两个机制正交，组合应同时超过 LCSS-only 和对应 robust-only，而不是只超过裸 MLP+CE。

## Granular-ball role

粒球保持为 frozen GRACE H 上 label-blind 的局部共识分组。球构造、beta=0.5 target、覆盖范围和固定参数完全不变；本实验只修改学生训练损失。

## Methods and controls

每个数据集运行以下方法：

1. `frozengrace_gb_mlp_softdist_beta05`：LCSS + soft CE anchor。
2. `mlp_logitclip`：hard noisy label + LogitClip anchor。
3. `frozengrace_gb_mlp_softdist_beta05_logitclip`：LCSS + LogitClip。
4. `frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled`：分组随机化的组合对照。
5. `mlp_small_loss`：hard noisy label + Small-loss anchor。
6. `frozengrace_gb_mlp_softdist_beta05_small_loss`：LCSS + soft-small-loss。
7. `frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled`：分组随机化的组合对照。

LogitClip 沿用现有 `tau=1.0`。Soft-small-loss 对每个 train node 计算 soft cross-entropy，按现有 Small-loss 的 `forget_rate=auto, num_gradual=10, exponent=1.0` 保留小损失样本。

## Protocol

- Datasets: Cora, CiteSeer, CiteSeerFull。
- Noise: uniform 0.6, train-only。
- Split: percent 0.2/0.1/0.7。
- Seed: 0 smoke。
- Student: 2-layer X-only MLP, hidden=64, dropout=0.5。
- GRACE H 与 LCSS config: 复用现有 frozen anchor，不重新预训练或调参。
- Device: ROCm `cuda:0`。

选择 CiteSeerFull 是因为其 u0.6 相对 JoCoR 落后约 4.71pp；Cora/CiteSeer 用于判断组合是否只修复一个边界数据集，还是具有跨数据集信号。

## Expected mechanism

- LogitClip 限制过大 logits，降低错误 target 被高置信记忆的程度；它不改变样本排序，因而是最干净的第一种组合。
- Soft-small-loss 按 LCSS target 的拟合损失选择样本，检验“先修 target、再选择剩余可靠样本”是否优于任一单独机制。

## Seed-0 advancement gate

LogitClip 和 soft-small-loss 分别独立判定。某个组合只有同时满足以下条件才进入 5-seed：

1. 至少 2/3 数据集上，组合严格高于 `max(LCSS-only, robust-only)`；
2. 三数据集平均 synergy delta 至少 +0.5pp；
3. 任一数据集的 synergy delta 不低于 -0.5pp；
4. 至少 2/3 数据集上 real combination - shuffled combination >= +1.0pp；
5. 无 NaN、OOM、预测单类坍缩或标签泄漏；`modified_val_count=modified_test_count=0`。

若两种简单组合都失败，关闭“直接 loss stacking”方向，不实现 soft-JoCoR。若任一通过，只记录为 5-seed 候选，不修改当前论文主方法。

## Code changes

- 新增 `predictor/LCSSRobust_Predictor.py`：soft CE、LogitClip 和 soft-small-loss 组合。
- 在 `scripts/run_standard_downstream.py` 注册四个新方法。
- 新增四组方法的 Cora/CiteSeer/CiteSeerFull config。
- 新增 `scripts/summarize_mlp28_softtarget_robust.py`。
- 新增 `tests/test_mlp28_softtarget_robust.py`。

## Required outputs

输出目录：`runs/mlp28_softtarget_robust_20260712/`

- `smoke_seed0.csv`
- `mlp28_aggregate.csv`
- `mlp28_paired_deltas.csv`
- `mlp28_gate_decisions.csv`
- `mlp28_report.md`
- stdout/stderr log

## Results

Seed0 smoke 已完成，共 21/21 个 run 成功。三个数据集内所有方法的 split hash 与 noise hash 完全一致；`modified_val_count=modified_test_count=0`，且 `modified_count=modified_train_count`。通过测试集多数类精度 `29.97/21.39/18.84%` 复核，两个组合的测试精度均明显高于单类预测上限，未出现单类坍缩。

| Strategy | Dataset | LCSS | Robust-only | Combination | Shuffled combination | Synergy | Real-shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| LogitClip | Cora | 38.94 | 32.66 | 39.42 | 34.20 | +0.47pp | +5.22pp |
| LogitClip | CiteSeer | 38.62 | 32.13 | 37.84 | 32.30 | -0.77pp | +5.54pp |
| LogitClip | CiteSeerFull | 39.68 | 36.27 | 42.08 | 33.54 | +2.40pp | +8.54pp |
| soft-small-loss | Cora | 38.94 | 37.68 | 44.96 | 34.56 | +6.02pp | +10.40pp |
| soft-small-loss | CiteSeer | 38.62 | 36.94 | 41.80 | 31.92 | +3.18pp | +9.88pp |
| soft-small-loss | CiteSeerFull | 39.68 | 38.26 | 49.11 | 30.56 | +9.42pp | +18.54pp |

- LogitClip：平均 synergy `+0.70pp`，但 CiteSeer 回退 `-0.77pp`，触发“任一数据集不得低于 -0.5pp”，判定不通过。
- soft-small-loss：三个数据集 synergy 全正，平均 `+6.21pp`、最小 `+3.18pp`；三个 real-shuffled 差值均超过 1pp，判定通过 seed0 advancement gate。
- 以上结果仍是单 seed 候选信号，不能作为论文结论或修改当前 LCSS anchor。

## Final decision

- 关闭 `LCSS+LogitClip` 的直接组合，不调 tau。
- `LCSS+soft-small-loss` 进入 paired 5-seed 候选池；下一步必须补 seeds 1-4 后重新判断幅度和方向一致性。
- 暂不实现 soft-JoCoR。只有 soft-small-loss 通过 5-seed，且仍有明确未覆盖误差时，才允许评估双网络版本。
- 当前论文主方法、主表与固定 LCSS 配置不变；该结果归档为下一篇论文/投稿后扩展方向。

## Phase 2: Soft-JoCoR kill smoke

在 soft-small-loss 已通过 seed0 后，只增加一次高门槛复杂候选检验。Soft-JoCoR 保持两个独立 X-only MLP，二者共享同一份 LCSS soft target。逐节点联合损失严格沿用现有 JoCoR 权重形式：

\[
\ell_i=(1-\lambda_{co})(\ell_{soft}^{(1)}+\ell_{soft}^{(2)})+\lambda_{co}\left[KL(p_i^{(1)}\Vert p_i^{(2)})+KL(p_i^{(2)}\Vert p_i^{(1)})\right].
\]

按联合损失执行与现有 JoCoR/soft-small-loss 相同的 forget-rate schedule；验证和测试使用两个模型平均 logits。固定 `co_lambda=0.1`，不搜索。

Seed0 方法臂：LCSS、JoCoR-only、LCSS+soft-small-loss、LCSS+soft-JoCoR、shuffled-LCSS+soft-JoCoR。已有且 hash 一致的行直接复用。

Soft-JoCoR 仅在以下条件全部满足时保留：

1. 至少 2/3 数据集超过 LCSS+soft-small-loss；
2. 三数据集平均领先至少 +1pp；
3. 任一数据集不得下降超过 0.5pp；
4. real-shuffled 至少 2/3 数据集领先 1pp；
5. 无标签泄漏、双网络预测坍缩或协议 hash 不匹配。

未通过即关闭 soft-JoCoR，不调 `co_lambda` 或 schedule。该 smoke 不阻塞 soft-small-loss 的多种子确认。

## Phase 3: Soft-small-loss five-seed confirmation

复用 seed0，为以下四臂补 seeds 1–4：LCSS、Small-loss-only、LCSS+soft-small-loss、shuffled-LCSS+soft-small-loss。若已有 anchor 的 split/noise/config hash 无法完全匹配，则重跑 anchor。

五种子通过标准：

- 至少 2/3 数据集上 `combination > max(LCSS, Small-loss-only)`，且至少 4/5 paired seeds 正向；
- 三数据集平均 synergy 至少 +1pp；
- 任一数据集平均 synergy 不得低于 -0.5pp；
- 至少 2/3 数据集 real-shuffled 平均差 >=1pp 且至少 4/5 seeds 正向；
- 5-seed 结果只用于下一篇论文候选，不回写当前冻结论文。

## Phase 2 results: Soft-JoCoR seed0

9/9 run 成功，协议 hash 与对应 seed0 anchors 一致，val/test 标签零污染。Soft-JoCoR 参数量为单 MLP 的约 2 倍。

| Dataset | LCSS | JoCoR-only | Soft-small-loss | Soft-JoCoR | Shuffled Soft-JoCoR | Soft-JoCoR - soft-small | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 38.94 | 36.68 | 44.96 | 45.49 | 33.09 | +0.53pp | +12.40pp |
| CiteSeer | 38.62 | 37.80 | 41.80 | 45.27 | 33.81 | +3.48pp | +11.47pp |
| CiteSeerFull | 39.68 | 41.30 | 49.11 | 51.00 | 36.00 | +1.89pp | +14.99pp |

Soft-JoCoR 三数据集均超过 soft-small-loss，平均 `+1.97pp`，最小增益 `+0.53pp`，且 real-shuffled 三项均超过 1pp，因此通过复杂候选 seed0 门。该结论仅表示保留候选；在 5-seed 前不能判断双网络一致性增益稳定。

## Phase 3 results: Soft-small-loss five-seed

Seed0 复用，seeds 1–4 新增 48/48 run 成功；每个 dataset/seed 的四臂 split/noise hash 一致，val/test 标签零污染。

| Dataset | LCSS | Small-loss-only | Combination | Shuffled combination | Synergy | Positive seeds | Real-shuffled | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 41.04+/-2.12 | 36.72+/-0.97 | 45.79+/-1.55 | 34.41+/-3.02 | +4.75pp | 5/5 | +11.39pp | 5/5 |
| CiteSeer | 40.17+/-2.46 | 37.73+/-2.56 | 43.80+/-3.13 | 34.05+/-2.48 | +3.63pp | 5/5 | +9.75pp | 5/5 |
| CiteSeerFull | 41.93+/-2.79 | 45.18+/-3.61 | 51.71+/-2.23 | 37.47+/-3.74 | +6.24pp | 5/5 | +14.24pp | 5/5 |

五种子门全部通过：3/3 数据集 synergy 为正且 5/5 paired seeds 正向，平均 synergy `+4.87pp`、最小 `+3.63pp`；3/3 real-shuffled 均超过 1pp 且 5/5 正向。

**阶段决策**：`LCSS+soft-small-loss` 升级为下一篇论文的已确认候选；`LCSS+soft-JoCoR` 保留为更复杂但仅 seed0 的候选，后续若要比较最终算法，必须单独补 5-seed。当前冻结论文不更新。
