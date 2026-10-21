# MLP-35：Clean-Val Forget-Rate 选择与 ROCm 重复性审计

## 1. 研究假设

MLP-34 证明 noisy-only 球内分歧不能准确估计合成噪声率，因此当前
`LCSS + soft-small-loss` 仍依赖已知 `forget_rate`。本实验检验一个更标准、
但假设更明确的替代方案：当协议提供干净验证集时，能否仅根据验证准确率从
预注册网格中选择 `forget_rate`，在不读取测试标签和真实噪声率的情况下接近
oracle true-rate 版本。

本实验同时量化同一 seed、同一协议在 ROCm 上的重复运行波动。MLP-34 的
`2.76pp` 是最大单次差值，不是方差估计；本轮用重复运行给出可解释的
within-seed spread。

## 2. 粒球角色与方法动机

- 粒球角色保持不变：frozen GRACE H 上的 label-blind 局部分组，用 train noisy
  vote 构造 beta=0.5 soft target。
- 不修改 LCSS、GRACE H、`64/32/0.05`、MLP、soft-small-loss 公式或训练 epoch。
- clean-val 网格只选择 robust optimization 的筛除强度，不重新设计粒球。
- 该扩展只能支持“存在干净验证集时不需要 oracle 噪声率”，不能写成完全
  noise-rate-free，也不能推广到验证标签同样受污染的真实场景。

## 3. 代码改动

- 新增 `scripts/run_mlp35_cleanval_grid.py`：可恢复地运行候选网格，通过现有
  `run_standard_downstream.run_one` 和 `model.forget_rate` override 调用冻结 predictor。
- 新增 `scripts/summarize_mlp35_cleanval_grid.py`：协议审计、clean-val 选参、
  oracle/fixed/LCSS 对比和预注册判定。
- 新增 `scripts/run_mlp35_repeatability_audit.py`：三个代表设置、同 seed 三次重跑。
- 新增 `scripts/summarize_mlp35_repeatability_audit.py`：报告方法内 spread 和 paired-delta spread。
- 新增 `tests/test_mlp35_cleanval_repeatability.py`。
- 不修改 YAML、predictor 或历史结果。

## 4. 固定协议

- 数据集：Cora、CiteSeer、CoraFull、Amazon-Photo、PubMed。
- 噪声：uniform `{0.2, 0.4, 0.6}`，仅污染 train。
- split：percent `0.2/0.1/0.7`。
- 主方法：`frozengrace_gb_mlp_softdist_beta05_small_loss`。
- clean-val 候选网格：`{0.0, 0.2, 0.4, 0.6}`。
- 固定默认对照：`0.5`，不参与选择。
- tie-break：验证准确率完全相同时选择更小的 forget rate。
- GPU：ROCm Python，`cuda:0`。

每个 dataset/noise/seed 的候选必须共享 split/noise hash，且
`modified_val_count=modified_test_count=0`、
`modified_count=modified_train_count`。

## 5. Phase A：Seed0 Clean-Val 网格

运行量：`5 datasets x 3 noise rates x 5 rates = 75 runs`。

对每个 dataset/noise setting 输出：

- clean-val 选择的 rate、valid、test；
- oracle true-rate test；
- fixed-0.5 test；
- MLP-30 中相同 hash 的原始 LCSS test；
- selected-oracle、selected-fixed0.5、selected-LCSS；
- 所有候选的 validation 排名。

进入五种子的门槛：

1. 15 个 setting 的 selected-oracle 平均不低于 `-1pp`；
2. 至少 `12/15` setting 的 selected-oracle 不低于 `-1pp`；
3. 任一 setting 不得低于 oracle 超过 `2pp`；
4. 所有完整性和泄漏审计通过。

`u0.2` 是否选择 rate 0 只作机制观察，不作为过门条件。

## 6. Phase B：五种子终判

Phase A 过门后补 seeds 1-4，共新增 300 runs。每个 seed 独立按 clean val 选 rate，
禁止跨 seed 共享选择结果。

“可替代 oracle”门：

1. 15 个 dataset/rate setting 的五种子 paired mean 中，至少 `12/15` 的
   selected-oracle 不低于 `-1pp`；
2. 所有 setting 的总体 paired mean不低于 `-1pp`；
3. 任一 setting 的 paired mean 不得低于 `-2pp`。

“相对固定默认有实际价值”单独判定：

1. 至少 `9/15` setting 的 selected-fixed0.5 paired mean 非负；
2. 全局 selected-fixed0.5 平均至少 `+0.5pp`。

只通过第一门时，可写“有干净验证集时无需准确的 oracle 噪声率即可基本保持性能”；
两门都过时，才可写“clean-val 选择优于固定默认”。两门均不过则保持 known-rate
限制，不更换第二套网格。

## 7. Phase C：同 Seed 重复性审计

固定 seed 0，每臂独立重复三次：

| 设置 | 方法 1 | 方法 2 | 目的 |
|---|---|---|---|
| Cora pair 0.4 | Small-loss | 组合 | 检查约 1pp 的边际 Pair 增量 |
| Amazon-Photo pair 0.4 | Small-loss | 组合 | 检查 Pair 边界/回退格 |
| PubMed pair 0.4 | JoCoR | 组合 | 检查 +3.78pp 强锚点 |

本阶段只报告：

- 每个方法三次 test 的 min/max/range/std；
- 每次 paired delta 及其符号；
- split/noise/config hash 一致性。

若方法内最大 range `<=1pp` 且 paired delta `3/3` 同向，记为重复性支持；否则
如实降低相关幅度 claim。不得把全局最大 range 当作硬件方差，也不得择优覆盖正式结果。

## 8. 成功、失败与论文边界

- **成功**：Phase B 的 oracle 替代门通过。方法可增加 clean-val tuned 变体。
- **部分成功**：接近 oracle，但不优于 fixed 0.5。只写免精确 rate，不写性能提升。
- **失败**：seed0 或五种子门未过。继续披露 known-rate 限制，关闭本方向。
- **重复性风险**：若弱 Pair 增量与 within-seed spread 同量级，措辞降为
  “方向性/条件性增益”；PubMed Pair/Instance 强锚点单独判断。
- 不重启 GBSelect、几何过滤、球数、动态 alpha 或 noisy-only 第二估计器。
- 不补 NRGNN-PubMed u0.6；正文若没有五种子证据则删除 seed0 `+0.78pp`。

## 9. 输出

目录：`runs/mlp35_cleanval_repeatability_20260714/`

- `cleanval_seed0_raw.csv`
- `cleanval_fiveseed_raw.csv`
- `mlp35_cleanval_candidates.csv`
- `mlp35_cleanval_selected.csv`
- `mlp35_cleanval_aggregate.csv`
- `mlp35_cleanval_gate.json`
- `repeatability_raw.csv`
- `mlp35_repeatability_summary.csv`
- `mlp35_repeatability_report.md`
- `mlp35_report.md`

## 10. 最终决策字段

实验结束后必须明确记录：`continue`、`modify`、`hold` 或 `abandon`，并分别标注
五种子性能证据、seed0 机制观察和重复性诊断。

## 11. 实验结果（2026-07-14）

### 11.1 完整性

- Phase A：75/75 个 seed0 候选运行成功。
- Phase B：新增 300/300 个 seeds 1-4 运行成功；合并后为
  `5 datasets x 3 rates x 5 seeds x 5 candidate/comparator rates = 375` 行。
- Phase C：18/18 个重复性运行成功。
- 同一 dataset/noise/seed 内 split/noise hash 一致；所有正式运行均为
  train-only noise，且 val/test modified count 为 0。
- 本实验未修改 predictor、LCSS target、GRACE H 或历史 YAML anchor。

### 11.2 Clean-val 选择

Seed0 预注册门通过：15/15 个 setting 均在 oracle 的 1pp 范围内，
selected-oracle 平均为 `+0.23pp`，最差为 `-0.51pp`。

五种子终判两道门均通过：

| 判定 | 结果 |
|---|---:|
| setting mean 在 oracle 1pp 内 | 15/15 |
| 全局 selected-oracle | +0.11pp |
| 最差 setting mean selected-oracle | -0.46pp |
| selected-fixed0.5 非负 | 12/15 |
| 全局 selected-fixed0.5 | +1.26pp |

逐设置五种子结果如下，单位为百分比：

| Dataset | Noise | Clean-val selected | Oracle true-rate | Fixed 0.5 | Sel.-Oracle | Sel.-Fixed |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.2 | 65.06+/-0.94 | 65.00+/-1.34 | 63.88+/-1.60 | +0.05 | +1.17 |
| Cora | 0.4 | 60.59+/-0.46 | 59.49+/-1.02 | 60.05+/-1.26 | +1.10 | +0.54 |
| Cora | 0.6 | 45.87+/-1.59 | 45.79+/-1.55 | 45.57+/-1.66 | +0.07 | +0.30 |
| CiteSeer | 0.2 | 64.18+/-0.56 | 63.90+/-0.87 | 64.21+/-1.75 | +0.28 | -0.03 |
| CiteSeer | 0.4 | 57.82+/-4.40 | 58.28+/-2.39 | 58.51+/-3.06 | -0.46 | -0.70 |
| CiteSeer | 0.6 | 43.75+/-3.19 | 43.80+/-3.13 | 42.91+/-2.66 | -0.05 | +0.83 |
| CoraFull | 0.2 | 44.71+/-0.27 | 44.69+/-0.24 | 43.83+/-0.30 | +0.03 | +0.88 |
| CoraFull | 0.4 | 40.61+/-0.17 | 40.63+/-0.50 | 40.55+/-0.29 | -0.02 | +0.06 |
| CoraFull | 0.6 | 32.34+/-0.61 | 32.34+/-0.61 | 32.12+/-1.09 | +0.00 | +0.22 |
| Amazon-Photo | 0.2 | 85.99+/-1.38 | 85.17+/-0.82 | 78.09+/-0.67 | +0.82 | +7.90 |
| Amazon-Photo | 0.4 | 82.53+/-0.73 | 82.56+/-0.79 | 81.95+/-0.61 | -0.03 | +0.59 |
| Amazon-Photo | 0.6 | 75.62+/-2.07 | 75.62+/-2.07 | 73.02+/-2.28 | +0.00 | +2.60 |
| PubMed | 0.2 | 83.94+/-0.49 | 84.00+/-0.40 | 81.61+/-1.19 | -0.06 | +2.33 |
| PubMed | 0.4 | 81.49+/-0.28 | 81.49+/-0.28 | 81.78+/-0.68 | +0.00 | -0.29 |
| PubMed | 0.6 | 60.19+/-6.35 | 60.19+/-6.35 | 57.62+/-5.57 | +0.00 | +2.57 |

Clean-val 只在 45/75 个 dataset/noise/seed 单元中选择了真实注入率。因此它应解释为
**训练强度选择器**，不是噪声率估计器。u0.2 仅 4/25 次选择 rate 0，不能声称低噪声下
总会自动退化为 LCSS。最差单 seed 是 CiteSeer u0.4 seed4，选择 rate 0 后相对 oracle
低 4.38pp；这说明小验证集仍可能产生单 seed 选择误差，五种子均值不能掩盖该风险。

### 11.3 同 seed 重复性

| Setting | Paired delta | 方法内最大 range | 方向 |
|---|---:|---:|---:|
| Cora pair0.4：combo-Small-loss | +2.53pp | 0.00pp | 3/3 正向 |
| Amazon-Photo pair0.4：combo-Small-loss | +5.28pp | 0.00pp | 3/3 正向 |
| PubMed pair0.4：combo-JoCoR | +3.07pp | 0.10pp | 3/3 正向 |

重复性门通过。MLP-34 trace 中观察到的最大 2.76pp 差异不能当作全局 ROCm 方差带；
在标准运行器与本次审计设置下，结果为完全或近似可重复。Pair 的全局协同幅度仍较小，
论文应继续使用五种子、条件性措辞，但无需仅因 2.76pp trace 差异整体降级。

## 12. 最终决策与 Claim 边界

**最终决策：`continue`，保留 clean-val tuned 变体；原始冻结方法不被静默替换。**

实验支持：

> 在已有干净验证集的协议下，预声明的 forget-rate 网格选择不需要知道精确的
> 合成噪声率，即可保持 oracle-rate 水平，并在平均意义上优于固定 0.5 默认值。

实验不支持：

- 不得写“noise-rate-free”或“估计了噪声率”；方法仍依赖干净验证集。
- 不得写低噪声下必然自动关闭 selection。
- 不得把单 seed clean-val 选择视为无风险。
- `forget_rate=0` 是同机制的零筛选臂；与历史 LCSS anchor 的 15 个可比单元中
  12 个 test 完全一致，PubMed 最大差异 1.12pp，因此不得宣称 bitwise 等价。

结果目录：`runs/mlp35_cleanval_repeatability_20260714/`。
