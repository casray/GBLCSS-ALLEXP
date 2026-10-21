# C1_weighted Seed0 Report

## 研究假设

保留 C1 的 hard majority repair，同时用球内 train coverage 和 vote margin 调节 train loss 权重，可以在不改变半监督协议的前提下降低低可靠球的影响。

## 粒球扮演的角色

粒球仍作为局部 target repair 单元；新增的 coverage/confidence 只作为 loss weight，不作为 hard gate，也不让 val/test 伪标签进入 supervised loss。

## 方法动机

Phase A 中 gated soft correction 弱于 C1，说明削弱 hard repair 会伤害主机制。本轮改为保留 C1 target 构造，仅对 train 节点 CE 权重做 coverage-aware 调节。

## 代码改动

- 新增 `C1_weighted`、`C1_weighted_shuffled`、`C1_weighted_noball`。
- 新增 `weighted_k_ref=3`、`weighted_floor=0.25`。
- 新增 `ball_reliability_weights`：只用 `train_mask` 节点统计 `n_train(B)` 和 `margin(B)`。
- 新增 `weighted_soft_target_loss`：`sum(w_i * CE_i) / sum(w_i)`。

## 配置改动

基础设置：Cora, ratio split, train_ratio=0.2, val_ratio=0.1, symmetric noise_rate=0.4, seed=0, Phase1=1000 epochs, Phase2=200 epochs。

权重公式：

```text
w_B = max(weighted_floor, min(1, n_train(B) / weighted_k_ref) * margin(B))
```

## 预期机制

高覆盖、高一致性投票的球保留接近 C1 的强监督；低覆盖或投票不一致的球仍执行 hard repair，但降低其 train loss 贡献。

## 成功标准

- 必须：`C1_weighted >= C1 - 1pp`
- 理想：`C1_weighted >= C1`
- 结构贡献：`C1_weighted > C1_weighted_shuffled + 5pp`
- 粒球增量：`C1_weighted > C1_weighted_noball + 10pp`

## 实验结果

| method | classifier_test_acc | linear_eval_acc |
|---|---:|---:|
| C1 baseline | 0.7443 | 0.8366 |
| C1_weighted | 0.7433 | 0.8392 |
| C1_weighted_shuffled | 0.6199 | 0.8350 |
| C1_weighted_noball | 0.5704 | 0.8039 |

Gate 判定：

| Gate | 结果 | 依据 |
|---|---|---|
| 不明显伤害 C1 | PASS | 0.7433 vs 0.7443，低 0.11pp |
| 超过 C1 | FAIL | 低 0.11pp，未超过 |
| 结构贡献 | PASS | 高于 shuffled 12.34pp |
| 粒球增量 | PASS | 高于 noball 17.29pp |

## 最终决策

`C1_weighted` 达到“不伤害 C1”的最低标准，并保留真实球相对 shuffled/noball 的大幅优势；但 seed0 未超过 C1。当前应记录为 coverage-aware weighting 稳健但非提升。若继续，建议优先做 Cora/CiteSeer 5-seed 验证其是否保持 C1 水平，而不是继续叠加 graph coherence。

