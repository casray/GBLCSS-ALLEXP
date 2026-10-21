# MLP-38 LOO-LCSS 蒸馏学生 Seed0 Smoke 设计

## 目标

检验现有球式共识教师蒸馏中，排除每个训练节点自身的含噪标签投票，是否能改善最终 X-only MLP student。实验只回答 LOO 的单变量作用，不改变主算法或论文主表。

## 单变量设计

现有 `mlp_distill_softdist_xh` 使用 frozen GRACE H 构造球式局部共识 target，再训练 `GCN[X||H]` teacher，最后以 teacher 的全节点概率蒸馏 X-only MLP student。新方法只增加：

```yaml
leave_one_out: true
```

`target_beta=0.0`、球参数、teacher、student、split、noise 和 early stopping 全部与现有蒸馏线一致。这样比较不会把 LOO 与 beta 变化混淆。它是现有蒸馏 T3 的 LOO 对应版本；不是把 train-only LOO-LCSS 的 beta=0.5 配置整体搬入 teacher。

## 实验臂

1. `mlp_distill_gcn_xh`：无局部共识的 H-access teacher。
2. `mlp_distill_softdist_xh`：现有默认球式共识 teacher。
3. `mlp_distill_softdist_xh_loo`：新 LOO 球式共识 teacher。
4. `mlp_distill_softdist_xh_loo_shuffled`：LOO size-matched shuffled control。

## 协议

- 数据集：Cora、CiteSeer、CoraFull。
- Noise：uniform 0.4，仅污染 train。
- Split：percent 0.2/0.1/0.7。
- Seed：0。
- Device：ROCm `cuda:0`。
- Frozen H：复用现有蒸馏配置，不重新预训练 GRACE。

## 判定

Seed0 只作 smoke，不作显著性结论。进入 5-seed 的预注册门为：

1. LOO−默认 teacher 至少 2/3 数据集为正；
2. 三数据集平均 LOO−默认 teacher >= +0.5pp；
3. 任一数据集不得下降超过 0.5pp；
4. LOO real−LOO shuffled 至少 2/3 数据集 >= +1pp；
5. 无 NaN、OOM、标签泄漏或预测坍缩。

未通过则关闭 LOO 蒸馏线，不调整 beta、球参数或 teacher 超参数救结果。

## 输出

- `runs/mlp38_loo_distill_smoke_20260717/seed0.csv`
- `runs/mlp38_loo_distill_smoke_20260717/mlp38_summary.csv`
- `runs/mlp38_loo_distill_smoke_20260717/mlp38_report.md`

