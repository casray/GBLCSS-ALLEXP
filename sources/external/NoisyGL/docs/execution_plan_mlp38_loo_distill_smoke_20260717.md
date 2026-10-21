# MLP-38 LOO-LCSS 蒸馏学生 Seed0 Smoke

## 研究假设

默认蒸馏 teacher 的球内投票包含节点自身噪声标签。若 train-only LOO-LCSS 的 target-quality 改善能迁移到 teacher 训练，则 LOO teacher 可能提高 X-only student 精度。

## 粒球角色

粒球是 frozen GRACE H 上 label-blind 的局部共识单元；标签只在组内 train vote 阶段使用。LOO 对节点 `v` 构造 target 时从计数中移除 `v` 自身标签。

## 方法动机与预期机制

LOO 消除自标签回流，使 teacher 的监督更依赖外部局部共识。但 teacher 的图传播和全节点蒸馏可能吸收该小幅 target 改善，因此预期增量有限，也可能因 target 更激进而下降。

## 代码与配置改动

- 新增 `mlp_distill_softdist_xh_loo` 与 shuffled 对照 predictor。
- 新增 Cora/CiteSeer/CoraFull 配置，仅相对现有 T3 增加 `leave_one_out: true`。
- 注册到标准 runner。
- 新增 MLP-38 汇总脚本与单元测试。

## 成功与失败标准

成功门与设计文档一致：2/3 正向、平均至少 +0.5pp、无超过 0.5pp 回退、real−shuffled 至少 2/3 达 +1pp。任一核心门失败即停止，不补 5-seed。

## 实验结果

- 协议：Cora/CiteSeer/CoraFull，uniform0.4，train-only noise，percent 0.2/0.1/0.7，seed0，ROCm `cuda:0`。
- 完整性：12/12 单元 `status=ok`；无 NaN、OOM、预测坍缩或 val/test 标签污染。

| 数据集 | GCN[X\|H]+CE teacher | 默认 LCSS teacher | LOO-LCSS teacher | LOO shuffled | LOO−默认 | LOO−shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 79.42 | 80.90 | 81.37 | 76.89 | +0.47pp | +4.49pp |
| CiteSeer | 69.46 | 69.12 | 69.72 | 69.54 | +0.60pp | +0.17pp |
| CoraFull | 58.61 | 57.81 | 57.90 | 58.29 | +0.09pp | -0.39pp |

LOO 相对默认 teacher 在 3/3 数据集数值正向，但平均仅 `+0.39pp`，低于 `+0.5pp` 门槛。LOO real 相对 LOO shuffled 仅 Cora 达到 `+1pp`，为 1/3，未达到 2/3 门槛。

## 最终决策

**停止，不扩 5-seed。** LOO 在蒸馏线上存在微弱 seed0 正向信号，但没有形成足够的真实局部结构归因，不能称为稳定改进。保留默认蒸馏方法；不得继续调整 beta、球参数或 teacher 结构来追逐该信号。

证据路径：

- 原始结果：`runs/mlp38_loo_distill_smoke_20260717/seed0.csv`
- 汇总：`runs/mlp38_loo_distill_smoke_20260717/mlp38_summary.csv`
- 报告：`runs/mlp38_loo_distill_smoke_20260717/mlp38_report.md`
