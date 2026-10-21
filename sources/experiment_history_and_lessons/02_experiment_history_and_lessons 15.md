# Phase42：SoftDist 标签率敏感性加密实验（2026-06-28）

## 1. 研究问题

本轮回答一个论文表述风险：SoftDist [X‖H] 之前主表使用 `train/val/test = 0.20/0.10/0.70`。需要确认 `20%` 是否只是偶然好点，以及是否能声称 train ratio 越大 SoftDist 越强。

最终目标不是替换主表，而是形成可审稿的标签率敏感性证据：在不同 `train_percent` 下，固定噪声、种子、模型协议，比较 SoftDist [X‖H] 相对 `GCN+CE [X‖H]` 和 shuffled SoftDist 的配对增量。

## 2. 研究假设

SoftDist 的球软目标应在标签较稀缺、GCN [X‖H] baseline 尚未饱和时更有价值；随着 train ratio 升高，纯 CE 分类器可能直接吸收更多监督信号，导致球软目标边际收益减弱。因此不预设单调提升。

## 3. 粒球角色

粒球只作为 noisy-label soft target provider 和局部结构先验，不作为 hard positive label，也不参与修改 baseline evaluation。真实球必须同时超过同输入的 `GCN+CE [X‖H]` 和 shuffled ball 才能算作粒球结构增量。

## 4. 配置

- 数据集：Cora、CiteSeer、Coauthor-CS。
- 划分：`split_type=percent`；`train_percent = 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40`；`val_percent=0.10`；test 为剩余比例。
- 噪声：`noise_type=uniform`，`noise_rate=0.4`，`noise_scope=train`。
- seeds：`0,1,2,3,4`。
- 方法：`gcn`、`gcn_xh_ce`、`frozengrace_gb_gcn_softdist_xh`、`frozengrace_gb_gcn_softdist_xh_shuffled`。
- 核心比较：`SoftDist [X‖H] - GCN+CE [X‖H]` 与 `SoftDist [X‖H] - shuffled SoftDist`。

## 5. 代码和产物

本轮不改模型机制、不改 loss、不改 baseline evaluation、不改训练协议。新增/使用的汇总脚本为：

- `external/NoisyGL/scripts/summarize_label_rate_sensitivity.py`

实验输出目录：

- `external/NoisyGL/runs/label_rate_sensitivity_20260626/`

关键产物：

- `label_rate_raw_train0p15.csv`
- `label_rate_raw_train0p25.csv`
- `label_rate_raw_train0p30.csv`
- `label_rate_sensitivity_combined_raw.csv`
- `label_rate_sensitivity_aggregate.csv`
- `label_rate_sensitivity_paired_deltas.csv`
- `label_rate_sensitivity_audit.csv`
- `label_rate_sensitivity_report.md`

完整性：`combined_raw=420` rows，`aggregate=84` rows，`paired_deltas=84` rows，audit failures `0`。证据见 `label_rate_sensitivity_report.md` 与 `label_rate_sensitivity_audit.csv`。

## 6. 成功标准

某个划分称为强证据，当且仅当：

- `SoftDist - GCN [X‖H]` 为正，至少 `4/5` seeds 为正，平均增量约 `>= +0.5pp`。
- `SoftDist - shuffled` 为正，至少 `4/5` seeds 为正，平均增量约 `>= +1.0pp`。
- 两类标准必须同时满足。

## 7. 结果摘要

### Cora

- `0.05`：`+1.48pp` vs GCN[X‖H]，`+2.28pp` vs shuffled，双强。
- `0.10`：`+1.32pp`，`+1.76pp`，双强。
- `0.15`：`+0.61pp` 但 only `3/5` vs GCN[X‖H]，`+1.29pp` vs shuffled 强；不算双强。
- `0.20`：`+1.10pp`，`+2.16pp`，双强，是主协议强证据点。
- `0.25`：`+0.39pp`，`+1.00pp`，不算双强。
- `0.30`：`+0.95pp`，`+0.57pp`，不算双强。
- `0.40`：`-0.99pp` vs GCN[X‖H]，`+0.65pp` vs shuffled；高标签率下被 GCN[X‖H] 吸收或反超。

### CiteSeer

- `0.05`：`+1.83pp`，`+2.20pp`，双强。
- `0.10`：`+1.59pp`，`+2.39pp`，双强。
- `0.15`：`+1.39pp`，`+1.87pp`，双强。
- `0.20`：`+1.30pp`，`+1.95pp`，双强。
- `0.25`：`+1.49pp`，`+1.26pp`，双强。
- `0.30`：`+0.57pp` vs GCN[X‖H] 强，但 `+0.71pp` vs shuffled 不过 `+1.0pp` 门槛。
- `0.40`：`+0.59pp` vs GCN[X‖H] 强，但 `+0.49pp` vs shuffled 且 only `3/5`，不算双强。

### Coauthor-CS

七个比例均为正，但幅度低于强证据门槛：

- vs GCN[X‖H]：约 `+0.13pp` 到 `+0.44pp`。
- vs shuffled：约 `+0.16pp` 到 `+0.63pp`。

这说明 Coauthor-CS 支持“真实球结构有小幅正增量”，但不支持“标签率曲线强证据”或“增量很大”。

## 8. 结论分级

### 已被实验支持

- Cora/CiteSeer 在低/中标签率下更容易出现 SoftDist 的双重受控增量。
- `20/10/70` 主协议仍是合理主表点：Cora/CiteSeer 均为双强，Coauthor-CS 在主协议下仍为小幅正增量。
- Coauthor-CS 的标签率曲线完整补齐，但增量整体小于预注册强证据阈值。
- train ratio 变大不会保证 SoftDist 增量变大。

### 合理猜测

- 高标签率下 GCN [X‖H] 能利用更多监督信号，球软目标的边际价值被吸收。
- CiteSeer 的低/中标签率曲线更稳，可能因为 baseline 未像 Coauthor-CS 那样快速饱和。

### 尚未验证

- 这条标签率规律是否适用于 pair/instance 噪声、其他噪声率或更多数据集。
- 若把 NRGNN[X‖H] 加入同一标签率网格，SoftDist 的相对位置是否保持当前论文定位。
- 标签率曲线是否需要显著性检验或置信区间图才能支撑最终论文图表。

### 已被负结果削弱

- 不能声称 `train_percent` 越大 SoftDist 越好。
- 不能把 `0.40` 写成 SoftDist 更稳的证据；Cora `0.40` 对 GCN[X‖H] 为负。
- 不能把 Coauthor-CS 曲线写成强增量；它是稳定小正，不是强证据。

## 9. 最终决策

主表继续保留 `20/10/70`。标签率敏感性作为机制补充：展示 `SoftDist-GCN[X‖H]` 与 `SoftDist-shuffled` 的配对增量曲线，论文表述为“低/中标签率更有效，baseline 饱和后边际变弱”，不作为跨比例 leaderboard。

下一步若继续写论文，应优先生成标签率曲线图并更新论文初稿中的旧 coverage 表；不建议继续盲目增加更多 train ratio。
