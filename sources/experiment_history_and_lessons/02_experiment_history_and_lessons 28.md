# MLP-38：LOO-LCSS 蒸馏学生 Seed0 Smoke

## 研究假设

若 LOO 在 train-target LCSS 中减少自标签回流的收益能够迁移到 GCN[X||H] teacher，则使用 LOO-LCSS target 训练 teacher 后蒸馏到 X-only MLP，可能超过默认 LCSS teacher。

## 粒球扮演的角色

粒球仍是在 frozen GRACE H 上无标签构造的局部共识单元。标签只在 train vote 阶段使用；为节点构造 LOO target 时，从所在球的计数中移除该节点自身 noisy label。

## 方法动机

此前只验证了 train-only LOO-LCSS，尚未验证 LOO target-quality 改善是否能穿过“teacher 训练 + 全节点蒸馏”链路。本实验只改变 `leave_one_out`，不同时改变 beta、球参数、teacher 或 student。

## 代码与配置改动

- 新增 `mlp_distill_softdist_xh_loo` 与 `mlp_distill_softdist_xh_loo_shuffled`。
- 新增 Cora/CiteSeer/CoraFull 配置，继承默认 T3 口径并保持 `target_beta=0.0`，仅打开 `leave_one_out=true`。
- 新增测试与预注册门汇总脚本。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- Split：percent 0.2/0.1/0.7。
- Noise：uniform0.4，仅污染 train。
- Seed：0。
- Device：ROCm `cuda:0`。

## 预期机制

LOO 可减少 teacher target 对节点自身 noisy label 的依赖；但 GCN 图传播与全节点蒸馏也可能吸收或稀释该微小差异。

## 成功标准

LOO−默认在至少 2/3 数据集正向、三数据集平均至少 `+0.5pp`、无数据集回退超过 `0.5pp`，且 LOO real−LOO shuffled 在至少 2/3 数据集达到 `+1pp`。全部通过才扩 5-seed。

## 实验结果

| 数据集 | GCN[X\|H]+CE | 默认 LCSS teacher | LOO-LCSS teacher | LOO shuffled | LOO−默认 | LOO−shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 79.42 | 80.90 | 81.37 | 76.89 | +0.47pp | +4.49pp |
| CiteSeer | 69.46 | 69.12 | 69.72 | 69.54 | +0.60pp | +0.17pp |
| CoraFull | 58.61 | 57.81 | 57.90 | 58.29 | +0.09pp | -0.39pp |

12/12 单元完成且协议一致。LOO−默认 3/3 数值正向，平均 `+0.39pp`；但真实结构归因门仅 1/3 通过。

## 最终决策

**关闭，不扩 5-seed。** LOO 在蒸馏线路中只有微弱 seed0 增量，且 CiteSeer/CoraFull 缺乏 real-over-shuffled 支撑。该结果只能描述为“LOO 没有稳定迁移到蒸馏学生”，不能写成性能改进。默认蒸馏 anchor 保持不变。

## 证据路径

- 计划：`external/NoisyGL/docs/execution_plan_mlp38_loo_distill_smoke_20260717.md`
- 原始结果：`external/NoisyGL/runs/mlp38_loo_distill_smoke_20260717/seed0.csv`
- 报告：`external/NoisyGL/runs/mlp38_loo_distill_smoke_20260717/mlp38_report.md`
