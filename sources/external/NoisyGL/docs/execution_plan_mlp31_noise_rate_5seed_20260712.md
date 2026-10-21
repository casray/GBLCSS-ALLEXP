# MLP-31 LCSS + Soft-Small-Loss Representative Five-Seed Plan

## Research hypothesis

MLP-30 的 seed0 网格显示 `LCSS+soft-small-loss` 的协同随噪声率增强。本轮检验 uniform 0.4 的中噪声信号与 uniform 0.6 的跨数据集广度信号能否在 paired 5-seed 下成立。

## Frozen protocol

- uniform 0.4 datasets: Cora, CiteSeer, CoraFull, Amazon-Photo, PubMed, Coauthor-Physics。
- uniform 0.6 datasets: Cora, CiteSeer, CiteSeerFull, CoraFull, PubMed, Coauthor-Physics。
- Split: percent 0.2/0.1/0.7；noise scope: train；seeds: 0-4。
- Four arms: LCSS, Small-loss-only, LCSS+soft-small-loss, shuffled-LCSS+soft-small-loss。
- LCSS 固定 `64/32/0.05, beta=0.5`；Small-loss 固定 `forget_rate=auto, num_gradual=10, exponent=1.0`。
- 不运行 uniform 0.2，不运行 Soft-JoCoR，不修改当前论文或 predictor。

## Reuse and new runs

- u0.4 seed0 复用 MLP-30；只补六数据集 seeds 1-4，共 96 run。
- u0.6 的 Cora/CiteSeer/CiteSeerFull 复用 MLP-28 完整五种子。
- u0.6 的 CoraFull/PubMed/Coauthor-Physics seed0 复用 MLP-29；补 seeds 1-4，共 48 run。
- 总新增 144 个 GPU MLP run；最终汇总矩阵 240 个唯一单元格。

## Paired metrics

每个 dataset/rate/seed 计算：

\[
\Delta_{syn}=Acc_{combo}-\max(Acc_{LCSS},Acc_{small-loss}),
\]

以及 `combo-LCSS`、`combo-Small-loss`、`combo-shuffled`。每个数据集报告 mean+/-population std、正向 seed 数，以及对零的 paired one-sample t 检验；t/p 只作统计说明，不代替预注册方向门。

## Five-seed gate per rate

每个噪声率独立通过需同时满足：

1. 至少 4/6 数据集 mean synergy >0 且至少 4/5 seeds 正向；
2. 六数据集平均 synergy >=+1pp；
3. 任一数据集 mean synergy 不得低于 -0.5pp；
4. 至少 4/6 数据集 real-shuffled >=+1pp 且至少 4/5 seeds 正向。

通过表示该噪声率可支撑下一篇论文的 paired claim；未通过则保留为 seed0 候选或边界，不调整门槛和方法。

## Integrity

- 最终必须恰好包含 `2 rates * 6 datasets * 4 methods * 5 seeds = 240` 个唯一格。
- 每个 dataset/rate/seed 的四臂 split/noise hash 必须一致。
- 所有行必须为 uniform、train-only、`status=ok`。
- `modified_val_count=modified_test_count=0` 且 `modified_count=modified_train_count`。

## Outputs

目录：`runs/mlp31_noise_rate_5seed_20260712/`

- `uniform0p4_seeds_1_2_3_4.csv`
- `uniform0p6_seeds_1_2_3_4.csv`
- `mlp31_combined_raw.csv`
- `mlp31_aggregate.csv`
- `mlp31_seed_deltas.csv`
- `mlp31_paired_deltas.csv`
- `mlp31_rate_gates.csv`
- `mlp31_report.md`

## Results and decision

最终矩阵为 240/240 个唯一单元格；144 个新增 GPU 运行全部 `status=ok`，无重复、无 val/test 标签污染，四臂在每个 dataset/rate/seed 上的 split/noise hash 一致。

### Uniform 0.4

| Dataset | Synergy | Positive seeds | Real-shuffled | Positive seeds |
|---|---:|---:|---:|---:|
| Cora | +3.91pp | 5/5 | +13.69pp | 5/5 |
| CiteSeer | +4.45pp | 5/5 | +13.35pp | 5/5 |
| CoraFull | +3.30pp | 5/5 | +6.81pp | 5/5 |
| Amazon-Photo | +1.13pp | 3/5 | +5.76pp | 5/5 |
| PubMed | +4.17pp | 5/5 | +7.59pp | 5/5 |
| Coauthor-Physics | +0.10pp | 3/5 | +2.03pp | 5/5 |

Gate: **PASS**。4/6 数据集达到 paired synergy 门；平均 synergy `+2.84pp`，最小 `+0.10pp`；real-shuffled 6/6 达标。Amazon-Photo 和 Coauthor-Physics 的真实局部结构贡献稳定，但组合相对最强单方法的额外协同只为 3/5 seeds，属于中噪声边界。

### Uniform 0.6

| Dataset | Synergy | Positive seeds | Real-shuffled | Positive seeds |
|---|---:|---:|---:|---:|
| Cora | +4.75pp | 5/5 | +11.39pp | 5/5 |
| CiteSeer | +3.63pp | 5/5 | +9.75pp | 5/5 |
| CiteSeerFull | +6.24pp | 5/5 | +14.24pp | 5/5 |
| CoraFull | +5.02pp | 5/5 | +9.49pp | 5/5 |
| PubMed | +5.19pp | 5/5 | +9.88pp | 5/5 |
| Coauthor-Physics | +3.27pp | 5/5 | +6.15pp | 5/5 |

Gate: **PASS**。6/6 数据集均为 5/5 paired synergy 正向；平均 synergy `+4.68pp`，最小 `+3.27pp`；real-shuffled 6/6 均为 5/5 正向。

### Final decision

`LCSS+soft-small-loss` 的稳定 claim 限定为：在 uniform 0.6 高噪声下，target correction 与 robust sample selection 在六个跨规模数据集上形成稳定协同；uniform 0.4 为多数数据集有效但存在数据集边界；uniform 0.2 未过 seed0 门，不属于适用主场。该结果进入下一篇论文资产，当前冻结论文不修改。
