# MLP-30 LCSS + Soft-Small-Loss Noise-Rate Grid Seed0

## Research hypothesis

MLP-28/29 已确认 `LCSS+soft-small-loss` 在 uniform 0.6 上具有强协同。MLP-30 检验该协同是高噪声专用、跨噪声率普适，还是在低噪声下因错误筛除监督而产生代价。

## Frozen protocol

- Datasets: Cora, CiteSeer, CiteSeerFull, CoraFull, Coauthor-CS, Amazon-Photo, PubMed, Amazon-Computers, Coauthor-Physics。
- Noise: uniform 0.2/0.4/0.6, train-only。
- Split: percent 0.2/0.1/0.7。
- Seed: 0 only。
- Four arms: LCSS, Small-loss-only, LCSS+soft-small-loss, shuffled-LCSS+soft-small-loss。
- LCSS: frozen GRACE H, `64/32/0.05`, beta=0.5。
- Small-loss: `forget_rate=auto`，即使用当前合成噪声率；`num_gradual=10, exponent=1.0`。
- uniform 0.6 复用 MLP-28/29；只新跑 uniform 0.2/0.4，共 72 run。

## Metrics

对每个 dataset/rate 报告：

\[
\Delta_{syn}=Acc_{combo}-\max(Acc_{LCSS},Acc_{small-loss}),
\]

以及 `combo-LCSS`、`combo-Small-loss`、`combo-shuffled`。同时输出每个数据集的 `Delta_syn(0.2/0.4/0.6)` 趋势。

## Seed0 gate per noise rate

每个噪声率独立判定：

1. 至少 6/9 数据集 synergy 为正；
2. 九数据集平均 synergy >=+1pp；
3. 回退超过 1pp 的数据集不超过 2 个；
4. 至少 6/9 数据集 real-shuffled >=+1pp。

通过只表示该噪声率进入代表性数据集 5-seed 候选，不修改当前论文或升级稳定 claim。

## Integrity rules

- 最终矩阵必须恰好包含 `9*3*4=108` 个唯一 dataset/rate/method/seed 格。
- 每个 dataset/rate 的四臂 split/noise hash 必须一致。
- `modified_val_count=modified_test_count=0` 且 `modified_count=modified_train_count`。
- 拒绝错误噪声类型、非 train-only、非 seed0、缺格和重复格。

## Outputs

目录：`runs/mlp30_noise_rate_grid_seed0_20260712/`

- `uniform0p2_seed0.csv`
- `uniform0p4_seed0.csv`
- `mlp30_combined_raw.csv`
- `mlp30_setting_deltas.csv`
- `mlp30_rate_gates.csv`
- `mlp30_trends.csv`
- `mlp30_report.md`
- stdout/stderr logs

## Results and decision

所有新增数据均为 seed0；uniform 0.6 复用 MLP-28/29。最终矩阵为 108/108 个唯一单元格，所有运行均为 `status=ok`，四臂 split/noise hash 一致，且 `modified_val_count=modified_test_count=0`、`modified_count=modified_train_count`。

### 按噪声率汇总

| Noise rate | Synergy positive | Mean synergy | Regression >1pp | Real-shuffled >=1pp | Gate |
|---:|---:|---:|---:|---:|---|
| 0.2 | 5/9 | +0.46pp | 1 | 8/9 | Fail |
| 0.4 | 6/9 | +2.57pp | 0 | 9/9 | Pass |
| 0.6 | 9/9 | +4.85pp | 0 | 9/9 | Pass |

### 机制趋势

- 8/9 数据集的 synergy 从 0.2 到 0.6 单调增强；唯一非单调项为 CiteSeer，但其三档均为正。
- Cora、CiteSeer、CiteSeerFull、CoraFull、PubMed 在三档噪声率下均为正协同。
- Coauthor-CS、Amazon-Photo、Amazon-Computers、Coauthor-Physics 在低噪声下接近零或为负，但在 0.6 下全部转正。
- real-shuffled 在 0.4/0.6 为 9/9 >=1pp；0.2 为 8/9 >=1pp，说明真实局部结构贡献广泛存在，但低噪声下 robust selection 未必能在强单方法之上继续增益。

### Final decision

该组合判定为**中高噪声协同方法**，不是跨所有噪声率普适改进：uniform 0.4 和 0.6 进入代表性数据集 paired 5-seed 候选，uniform 0.2 不扩全量 5-seed。当前论文继续冻结；MLP-30 作为下一篇论文的适用域证据，不回填当前主表。
