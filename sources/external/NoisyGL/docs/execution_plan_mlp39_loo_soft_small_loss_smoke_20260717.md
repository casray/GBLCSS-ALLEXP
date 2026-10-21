# MLP-39：LOO-LCSS + Soft-Small-Loss 最小 Smoke

## 研究假设

默认 LCSS target 包含节点自身 noisy label，可能让错误标签节点的 soft loss 人为偏低，从而躲过 small-loss 筛选。若去除自票能改善可靠样本排序，则 LOO-LCSS+soft-small-loss 应超过默认 LCSS+soft-small-loss。

## 粒球角色

粒球继续作为 frozen GRACE H 上 label-blind 的局部共识单元。标签只在 train vote 阶段使用；LOO target 对每个节点排除自身 noisy vote。

## 方法动机

MLP-36 已确认 LOO-LCSS 在 Cora/CiteSeer 上有约 1.4--1.6pp 增量，MLP-30 已确认默认 LCSS+soft-small-loss 在 uniform0.4 有明显协同，但二者尚未组合。MLP-38 只判负了 LOO teacher 蒸馏，不能替代本次直接 robust optimization 检验。

## 代码改动

- 新增 LOO real/shuffled 两个 soft-small-loss predictor。
- 新增 Cora/CiteSeer/CoraFull 共六份配置。
- 新增 exact-hash 汇总脚本和测试。

## 配置改动

只在现有 soft-small-loss 配置中增加 `gb.leave_one_out: true`。`beta=0.5`、`max_balls=64`、`min_ball_size=32`、`split_improvement=0.05`、`forget_rate=auto`、`num_gradual=10`、`exponent=1.0` 均保持冻结。

## 预期机制

LOO target 减少自标签回流，使与外部局部共识冲突的 noisy 节点获得更高 soft CE，从而更可能被 small-loss 排除。潜在风险是干净少数类节点也被提高损失并过度删除。

## 成功标准

1. 至少 2/3 数据集 LOO-combo 超过 default-combo。
2. 三数据集平均增量至少 +0.5pp。
3. 任一数据集不下降超过 0.5pp。
4. 至少 2/3 数据集 LOO real-shuffled 至少 +1pp。
5. 6/6 运行成功、协议 hash 匹配、val/test 零污染、无坍缩。

## 失败标准

任一成功门未满足即不扩五种子；不调整 beta、forget rate、球参数或 LOO fallback。

## 实验结果

正式 seed0 smoke 共 6/6 个新运行成功。每个数据集内新方法与 MLP-30/36 锚点的 split/noise hash 完全一致，`modified_val_count=modified_test_count=0`，无 NaN、OOM 或预测坍缩。

| 数据集 | LOO combo | 默认 combo | LOO-LCSS | Small-loss | LOO shuffled | LOO−默认 | Real−shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 59.37 | 60.16 | 56.99 | 49.50 | 45.91 | −0.79pp | +13.46pp |
| CiteSeer | 60.22 | 59.84 | 56.36 | 49.05 | 43.99 | +0.39pp | +16.24pp |
| CoraFull | 40.45 | 41.27 | 37.64 | 33.05 | 33.97 | −0.82pp | +6.49pp |

- 仅 1/3 数据集超过默认 combo，未达到 2/3。
- 三数据集平均条件增量为 `−0.41pp`，未达到 `+0.5pp`。
- 最低增量为 CoraFull `−0.82pp`，触发不得低于 `−0.5pp` 的回退门。
- real−shuffled 为 3/3 大于 1pp，说明真实局部结构仍有效，但 LOO 没有给现有 soft-small-loss 带来额外收益。
- LOO 与默认版本保持相同的约 60% 最终保留比例。MLP-36 的 target 诊断显示 LOO 虽改善 Brier，但通常同步提高 CDR；small-loss 已利用默认 target 完成损失排序后，更激进的 LOO target 未形成稳定互补。

## 最终决策

**关闭该组合，不扩 5-seed。** 不调整 beta、forget rate、球参数、fallback 或第二版 LOO 公式。LOO 仍作为第一篇论文中支持外部局部共识的机制消融，但不升级为 soft-small-loss 的默认 target。第一篇论文保持冻结。
