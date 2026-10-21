# MLP-29 Robust Local-Consensus Breadth and Distillation Seed0

## Research hypothesis

MLP-28 已在 Cora/CiteSeer/CiteSeerFull 的 uniform 0.6 上确认 LCSS soft target 与 small-loss selection 存在 5-seed 协同。MLP-29 检验该协同是否扩展到不同规模和领域的数据集，以及把同一去噪策略用于 GCN[X||H] teacher 后，是否能进一步改善 X-only MLP student。

## Frozen scope

- Noise: uniform 0.6, train-only。
- Split: percent 0.2/0.1/0.7。
- Seed: 0 only。
- Datasets: CoraFull, Coauthor-CS, Amazon-Photo, PubMed, Amazon-Computers, Coauthor-Physics。
- LCSS: frozen GRACE H, `64/32/0.05`, beta=0.5。
- Small-loss: `forget_rate=auto, num_gradual=10, exponent=1.0`。
- 不调球、beta、forget schedule、teacher 或 student 超参数。

## Phase A: direct X-only MLP breadth

四臂：

1. LCSS + soft CE。
2. Small-loss-only + hard noisy labels。
3. LCSS + soft-small-loss。
4. Shuffled LCSS + soft-small-loss。

Seed0 候选门：至少 4/6 数据集组合高于 `max(LCSS, Small-loss-only)`；六数据集平均 synergy >=1pp；任一数据集不得回退超过 1pp；至少 4/6 数据集 real-shuffled >=1pp。未过门不否定 MLP-28，只说明跨数据集适用域有限。

## Phase B: robust teacher distillation

Teacher 训练阶段可访问图和 `[X||H]`，student 始终是 X-only 两层 MLP，蒸馏使用 teacher 对所有节点的 soft predictions。五臂：

1. `mlp_distill_gcn_xh`: bare CE teacher。
2. `mlp_distill_softdist_beta05_xh`: beta0.5 ball SoftDist teacher + soft CE。
3. `mlp_distill_knn_beta05_xh`: beta0.5 kNN consensus teacher + soft CE。
4. `mlp_distill_softdist_beta05_small_loss_xh`: beta0.5 ball SoftDist teacher + soft-small-loss。
5. `mlp_distill_softdist_beta05_small_loss_xh_shuffled`: size-matched shuffled ball control。

旧 `mlp_distill_softdist_xh` 和 `mlp_distill_knn_xh` 使用 beta=0.0，不作为本轮公平归因臂。

Seed0 候选门：robust teacher student 至少 4/6 数据集高于 beta0.5 plain SoftDist teacher；平均增益 >=0.5pp；任一数据集不得回退超过 1pp；至少 4/6 数据集 real-shuffled >=1pp。与 kNN teacher 的差值只报告，不要求球必须胜出。

## Leakage and fairness

- 球/kNN target 只统计 train noisy labels。
- clean labels 只用于 val/test evaluation 和离线诊断，不进入 target 或 selection。
- 每个 dataset/seed 五臂必须共享 split/noise hash。
- teacher small-loss 的每节点损失基于 LCSS soft target；forget schedule 与 direct soft-small-loss 一致。
- student 只接收 X 和 teacher probabilities，推理不访问图/H。

## Code changes

- 扩展 direct robust config 到六个数据集。
- 新增 `predictor/LCSSRobustDistill_Predictor.py`。
- 新增四种 beta0.5 distill 方法配置并注册。
- 新增 `scripts/summarize_mlp29_breadth_distill.py` 和测试。

## Required outputs

目录：`runs/mlp29_breadth_distill_seed0_20260712/`

- `direct_seed0.csv`
- `distill_seed0.csv`
- `mlp29_direct_deltas.csv`
- `mlp29_distill_deltas.csv`
- `mlp29_gate_decisions.csv`
- `mlp29_report.md`
- stdout/stderr logs

## Results and decision

### Phase A results

24/24 run 成功；每个数据集四臂 split/noise hash 一致，val/test 标签零污染。

| Dataset | LCSS | Small-loss | Combination | Shuffled | Synergy | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|
| CoraFull | 28.13 | 22.40 | 33.23 | 23.83 | +5.10pp | +9.40pp |
| Coauthor-CS | 86.22 | 81.33 | 88.87 | 80.36 | +2.65pp | +8.52pp |
| Amazon-Photo | 66.63 | 55.20 | 72.01 | 57.74 | +5.38pp | +14.27pp |
| PubMed | 49.75 | 58.82 | 65.32 | 51.48 | +6.50pp | +13.84pp |
| Amazon-Computers | 68.92 | 64.74 | 71.64 | 62.78 | +2.72pp | +8.86pp |
| Coauthor-Physics | 86.90 | 89.15 | 91.81 | 85.02 | +2.66pp | +6.80pp |

Phase A 通过：6/6 synergy 正，平均 `+4.17pp`、最小 `+2.65pp`；6/6 real-shuffled >=1pp。该结果把 direct X-only 候选从三个小/中型引文数据集扩展到大规模多类图、合著图、共购图和此前边界数据集，但仍只有 seed0。

### Phase B results

30/30 run 成功；每个数据集五臂 split/noise hash 一致，val/test 标签零污染。Robust teacher 在 schedule 末保留约 40% train nodes。

| Dataset | Bare | Beta0.5 SoftDist | Beta0.5 kNN | Robust SoftDist | Robust shuffled | Robust-plain | Robust-kNN | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoraFull | 52.61 | 51.65 | 52.85 | 53.78 | 54.02 | +2.13pp | +0.92pp | -0.24pp |
| Coauthor-CS | 89.04 | 92.60 | 93.05 | 93.99 | 92.36 | +1.39pp | +0.94pp | +1.64pp |
| Amazon-Photo | 84.65 | 87.56 | 87.84 | 90.27 | 74.77 | +2.71pp | +2.43pp | +15.50pp |
| PubMed | 76.38 | 76.90 | 77.67 | 60.96 | 61.05 | -15.94pp | -16.71pp | -0.09pp |
| Amazon-Computers | 78.69 | 75.89 | 75.10 | 78.97 | 71.44 | +3.09pp | +3.87pp | +7.53pp |
| Coauthor-Physics | 92.96 | 94.50 | 94.71 | 94.97 | 94.26 | +0.48pp | +0.27pp | +0.71pp |

Phase B 不通过：虽然 robust teacher 在 5/6 数据集高于 plain beta0.5 teacher，但 PubMed 回退 `15.94pp`，使平均增益为 `-1.02pp`；只有 3/6 数据集 real-shuffled >=1pp。PubMed real 与 shuffled 同时约 61%，表明主要失败来自 teacher 端强 small-loss selection 与图传播/蒸馏的交互，而非真实球结构本身。该机理解释仍是假设，当前证据只支持关闭“把 direct soft-small-loss 原样搬到 GCN teacher”的普适方案。

### Final decision

- Direct `LCSS+soft-small-loss`：扩展 seed0 通过，保留为下一篇论文主候选；后续 5-seed 应优先选择代表性数据集，而不是一次扩全矩阵。
- Robust teacher distillation：全局判负，不扩 5-seed，不做数据集排除后重算，不调 teacher forget schedule。
- 普通 beta0.5 SoftDist/kNN teacher 和已有 MLP-9 蒸馏结论不受影响。
- 所有结果均不修改当前冻结论文。
