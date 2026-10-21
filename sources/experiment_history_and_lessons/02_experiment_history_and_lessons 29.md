# MLP-39：LOO-LCSS + Soft-Small-Loss Seed0 Smoke

## 研究假设

默认 LCSS target 包含节点自身 noisy label，可能让错误标签节点的 soft loss 偏低并躲过 small-loss。若 LOO 能改善样本排序，则 LOO-LCSS+soft-small-loss 应稳定超过默认 LCSS+soft-small-loss。

## 粒球扮演的角色

粒球是在 frozen GRACE H 上无标签构造的局部共识单元。标签仅在 train vote 阶段使用；LOO 对每个节点排除自身 noisy vote。分组、beta 和 alpha 公式均不变。

## 方法动机

MLP-36 中 LOO-LCSS+soft CE 在 Cora/CiteSeer 分别提高约 1.56/1.37pp；MLP-30/31 已确认默认 LCSS+soft-small-loss 的协同。两项正信号此前未直接组合。MLP-38 只测了 LOO teacher 蒸馏，不能回答直接 robust optimization 的问题。

## 代码改动

- 在 `predictor/LCSSRobust_Predictor.py` 新增 LOO real/shuffled 两个 soft-small-loss predictor。
- 在 `scripts/run_standard_downstream.py` 注册两个方法。
- 新增 Cora/CiteSeer/CoraFull 六份配置，唯一机制变化为 `gb.leave_one_out: true`。
- 新增 `scripts/summarize_mlp39_loo_soft_small_loss.py` 与 `tests/test_mlp39_loo_soft_small_loss.py`。

## 配置改动

无参数调优。固定 `beta=0.5`、`max_balls=64`、`min_ball_size=32`、`split_improvement=0.05`、`forget_rate=auto`、`num_gradual=10`、`exponent=1.0`。

## 预期机制

排除自身 noisy vote 后，与外部共识冲突的错误节点应获得更高 soft CE，更容易被 small-loss 排除。风险是 LOO target 更激进，也会提高干净少数类或边界节点的损失。

## 成功与失败标准

进入五种子要求：至少 2/3 数据集超过默认 combo；平均增量至少 +0.5pp；无数据集低于 −0.5pp；至少 2/3 real−shuffled 达 +1pp；协议完整。任一失败则关闭，不调整第二套参数。

## 实验结果

- 6/6 新运行成功，split/noise hash 与 MLP-30/36 锚点完全配对，val/test 标签修改数均为 0。
- Cora：LOO combo `59.37`，默认 combo `60.16`，差 `−0.79pp`；real−shuffled `+13.46pp`。
- CiteSeer：`60.22` vs `59.84`，差 `+0.39pp`；real−shuffled `+16.24pp`。
- CoraFull：`40.45` vs `41.27`，差 `−0.82pp`；real−shuffled `+6.49pp`。
- 只有 1/3 数据集正向，平均 `−0.41pp`，最低 `−0.82pp`。方向门、平均增量门和最大回退门均失败；结构归因门 3/3 通过。
- LOO 与默认 small-loss 的最终保留比例相同（约 60%），差异来自 target 与排序，不是保留数量。

## 解释边界

LOO 改善 direct soft-CE LCSS 的事实仍成立，但该增量没有叠加到 soft-small-loss。与 MLP-38 一起看，LOO 的收益会被更强的后续训练机制吸收或抵消。当前数据支持“默认 target 与 small-loss 更匹配”，但 seed0 不能证明所有 LOO robust optimization 必然失败。

## 最终决策

**NO-GO，不扩 5-seed。** 不调 beta、forget rate、球参数、fallback 或新 LOO 公式。第一篇论文保持冻结；下一篇继续使用默认 LCSS+soft-small-loss，而不是 LOO 版本。

## 证据路径

- 计划：`external/NoisyGL/docs/execution_plan_mlp39_loo_soft_small_loss_smoke_20260717.md`
- 原始结果：`external/NoisyGL/runs/mlp39_loo_soft_small_loss_smoke_20260717/seed0.csv`
- 汇总：`external/NoisyGL/runs/mlp39_loo_soft_small_loss_smoke_20260717/mlp39_report.md`
