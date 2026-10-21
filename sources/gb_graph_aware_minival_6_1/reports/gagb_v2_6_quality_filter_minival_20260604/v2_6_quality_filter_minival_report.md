# GAGB v2.6 Quality Filter Minival Report

## 1. 研究假设

在 `grace_gagb` 的 frozen-GRACE grouping 上，问题不一定是 `qmix` 完全错误，也可能是：

- 一部分球有机制信号，但球质量混杂；
- 低质量球参与 target repair 后，把 AUROC 信号带成 classifier damage；
- 因此只让高质量球参与修正，或按 `Q(B)` 缩放修正强度，可能降低 `CDR`，并保留部分 `NCR`。

## 2. 粒球角色

本轮粒球不是 hard label，也不是 view controller，而是：

- local repair unit
- confidence carrier
- weak structural prior

具体做法是对已有 `grace_gagb` 球计算 clean-label-free 的 `Q(B)`，再决定是否允许该球参与 target repair。

## 3. 本轮代码与实验边界

- 本轮没有新增算法代码；`v2.6` 的 `Q(B)` / `qfilter` / `qweight` 实现已经存在于工程中。
- 本轮工作是运行 full minival、生成汇总，并写出可追溯结论。
- 不改 GRACE embedding，不改 grouping builder，不改 teacher，不改 split，不改 baseline evaluation。

## 4. 运行配置

- 数据集：`Cora`, `CiteSeer`
- 噪声：symmetric `0.2`, `0.4`
- split：`ratio_train0p2_val0p1_split0`
- seeds：`0 1 2 3 4`
- 分类器：frozen embedding classifier
- 主方法：
  - `grace_gagb_qb_conflict_gate_alpha07`
  - `grace_gagb_qfilter_alpha07_q50`
  - `grace_gagb_qfilter_alpha07_q70`
  - `grace_gagb_qweight_alpha07`
- 对照：
  - `gb`, `gagb`, `khop_only`, `grace_gagb`
  - `gagb_qfilter_alpha07_q50`
  - `khop_only_qfilter_alpha07_q50`
  - `grace_gagb_shuffled_qfilter_alpha07_q50`

主命令：

```powershell
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce gb gagb khop_only grace_gagb grace_gagb_qb_conflict_gate_alpha07 grace_gagb_qfilter_alpha07_q50 grace_gagb_qfilter_alpha07_q70 grace_gagb_qweight_alpha07 gagb_qfilter_alpha07_q50 khop_only_qfilter_alpha07_q50 grace_gagb_shuffled_qfilter_alpha07_q50 --save-diagnostics
py -3 scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_v2_6_quality_filter_minival_20260604
```

原始批量日志目录：

- `logs/gate/20260604_165429/`

汇总文件：

- `reports/gagb_v2_6_quality_filter_minival_20260604/gate_summary.csv`
- `reports/gagb_v2_6_quality_filter_minival_20260604/gate_paired_deltas.csv`

## 5. 预期机制

- `qfilter_q50/q70`：优先筛掉低质量球，目标是 `CDR` 下降，`NCR` 保留一部分。
- `qweight`：不做硬过滤，而是用 `alpha = 0.7 * Q(B)` 软化修正。
- 如果 `Q(B)` 真有独立价值，应看到：
  - selected balls 的 purity 高于 rejected balls；
  - `qfilter` 相对 unfiltered alpha07 的 `CDR` 下降；
  - 且不能退化成仅仅“把干预全部关掉”的弱化版本。

## 6. 成功标准

- 至少在大多数 setting 中，`qfilter` 相对 `grace_gagb_qb_conflict_gate_alpha07` 降低 `CDR`；
- `NCR` 不归零；
- classifier accuracy 至少不稳定变差，最好在 `Cora 0.4` 这种 failure setting 上明显改善；
- 不能被 `shuffled_qfilter` 或 `khop_only_qfilter` 轻易追平。

## 7. 失败标准

- `qfilter` 只降低了干预强度，但没有稳定改善 accuracy；
- `qweight` 把 `NCR` 压到接近 0；
- `Cora 0.4` 仍然低于 `gb` / `gagb` / `grace_gagb`；
- `qfilter` 与 unfiltered alpha07 在关键 setting 上几乎等价，说明过滤没有真正改变有效干预集合。

## 8. 结果

### 8.1 四个主 setting

| dataset | noise | method | test mean ± std | NCR | CDR | AUROC |
|---|---:|---|---:|---:|---:|---:|
| CiteSeer | 0.2 | `gb` | 0.7009 ± 0.0127 | 0.0000 | 0.0000 | 0.6061 |
| CiteSeer | 0.2 | `grace_gagb_qb_conflict_gate_alpha07` | 0.7085 ± 0.0056 | 0.2120 | 0.0462 | 0.6184 |
| CiteSeer | 0.2 | `grace_gagb_qfilter_alpha07_q50` | 0.7080 ± 0.0041 | 0.1805 | 0.0360 | 0.6014 |
| CiteSeer | 0.2 | `grace_gagb_qfilter_alpha07_q70` | 0.7094 ± 0.0040 | 0.1579 | 0.0285 | 0.5819 |
| CiteSeer | 0.2 | `grace_gagb_qweight_alpha07` | 0.7060 ± 0.0080 | 0.0000 | 0.0000 | 0.6175 |
| Cora | 0.2 | `gb` | 0.8118 ± 0.0051 | 0.0000 | 0.0000 | 0.6824 |
| Cora | 0.2 | `grace_gagb_qb_conflict_gate_alpha07` | 0.8143 ± 0.0039 | 0.3333 | 0.0207 | 0.6677 |
| Cora | 0.2 | `grace_gagb_qfilter_alpha07_q50` | 0.8118 ± 0.0067 | 0.2741 | 0.0124 | 0.6409 |
| Cora | 0.2 | `grace_gagb_qfilter_alpha07_q70` | 0.8122 ± 0.0038 | 0.2296 | 0.0078 | 0.6165 |
| Cora | 0.2 | `grace_gagb_qweight_alpha07` | 0.8133 ± 0.0051 | 0.0000 | 0.0000 | 0.6679 |
| CiteSeer | 0.4 | `gb` | 0.6976 ± 0.0097 | 0.0000 | 0.0000 | 0.5822 |
| CiteSeer | 0.4 | `grace_gagb_qb_conflict_gate_alpha07` | 0.6979 ± 0.0075 | 0.0556 | 0.0050 | 0.5347 |
| CiteSeer | 0.4 | `grace_gagb_qfilter_alpha07_q50` | 0.6979 ± 0.0075 | 0.0556 | 0.0050 | 0.5347 |
| CiteSeer | 0.4 | `grace_gagb_qfilter_alpha07_q70` | 0.6988 ± 0.0062 | 0.0526 | 0.0040 | 0.5322 |
| CiteSeer | 0.4 | `grace_gagb_qweight_alpha07` | 0.6966 ± 0.0077 | 0.0000 | 0.0000 | 0.5347 |
| Cora | 0.4 | `gb` | 0.7974 ± 0.0054 | 0.0000 | 0.0000 | 0.6450 |
| Cora | 0.4 | `grace_gagb_qb_conflict_gate_alpha07` | 0.7931 ± 0.0076 | 0.1198 | 0.0055 | 0.5582 |
| Cora | 0.4 | `grace_gagb_qfilter_alpha07_q50` | 0.7940 ± 0.0059 | 0.1032 | 0.0031 | 0.5506 |
| Cora | 0.4 | `grace_gagb_qfilter_alpha07_q70` | 0.7940 ± 0.0059 | 0.1032 | 0.0031 | 0.5506 |
| Cora | 0.4 | `grace_gagb_qweight_alpha07` | 0.7937 ± 0.0071 | 0.0000 | 0.0000 | 0.5581 |

### 8.2 `Q(B)` 是否真的筛到了更好球？

有。`qfilter` 选中的球比 rejected balls 更纯，噪声熵更低：

- `CiteSeer 0.2 / q50`：
  - selected purity `0.7413` vs rejected `0.5270`
  - selected entropy `0.5813` vs rejected `0.8134`
- `Cora 0.2 / q70`：
  - selected purity `0.9618` vs rejected `0.5977`
  - selected entropy `0.4037` vs rejected `0.6803`
- `Cora 0.4 / q50`：
  - selected purity `0.8382` vs rejected `0.5742`
  - selected entropy `0.6690` vs rejected `0.8218`

这说明 `Q(B)` 作为“球质量排序器”是有效的。

### 8.3 `Q(B)` 是否真的改变了有效干预？

分 setting 看：

- `CiteSeer 0.2`：有明显变化。`q50` 仅保留 `57.4` 个 selected-gated nodes，reject `13.6`；`q70` 更保守，只保留 `44.4`。
- `Cora 0.2`：也有变化。`q50` 选中 `39.4`，reject `10.2`；`q70` 选中 `30.0`，reject `19.6`。
- `CiteSeer 0.4`：`q50` 没有真正改变干预，selected gate `23.4`、rejected gate `0`，所以它和 alpha07 完全等价。
- `Cora 0.4`：有轻度变化，selected gate `23.6`、rejected gate `4.6`，但强度不够扭转整体 accuracy。

### 8.4 paired delta

重点看 `train_ratio=0.2`：

- `CiteSeer 0.2`
  - `q70 - gb = +0.0085`，5 seeds 里 `4` 个为正
  - `q70 - alpha07 = +0.0009`
- `Cora 0.2`
  - `alpha07 - gb = +0.0024`，5 seeds 里 `4` 个为正
  - `q50 - alpha07 = -0.0024`
  - `q70 - alpha07 = -0.0020`
- `CiteSeer 0.4`
  - `q70 - gb = +0.0012`
  - `q50 - alpha07 = 0.0`
- `Cora 0.4`
  - `q50 - alpha07 = +0.0008`
  - `q50 - gb = -0.0034`
  - `q70 - gb = -0.0034`

## 9. 结论

### 已被结果支持

1. `Q(B)` 能把更纯、更低熵的球排到前面，作为球质量诊断信号成立。  
2. `qfilter` 基本都能降低 `CDR`，同时保留非零 `NCR`。  
3. `qweight` 过于保守，几乎把 argmax-level correction 压成 `0`，不适合作为主方法。  

### 仅为合理解释

1. `Q(B)` 更像一个“安全阀”，而不是足够强的主收益来源。  
2. 现在的 `Q(B)` 过滤只能筛掉明显差球，但还不能解决 `qmix` 内容方向和 repair strength 在 classifier 上的稳定收益问题。  

### 尚未被验证

1. 更强的球质量建模（例如引入更好的 topology support、teacher-ball consistency、ball reliability prior）是否能把诊断信号转成稳定 accuracy。  
2. `Q(B)` 与 class-safe calibration、target strength calibration 联合后是否能改善 `Cora 0.4`。  

### 已被负结果削弱

1. “只靠质量过滤就能把 `grace_gagb` 修成稳定优于 `gb` 的方法” 这一说法不成立。  
2. `grace_gagb_qweight_alpha07` 作为主路径被削弱，因为它通过把干预缩到接近无效来换低 `CDR`。  

## 10. 最终决策

**决策：继续，但不再单独沿 `Q(B)` 过滤继续扫阈值。**

原因：

- `Q(B)` 作为 diagnosis/filtering signal 是正结果；
- 但它单独不足以形成主方法收益；
- 继续只扫 `q50/q70` 或继续压 alpha，得到的大概率只是更弱干预，不是更强方法。

更合理的下一步是：

1. 把 `Q(B)` 保留为 guard / reliability weight；
2. 下一轮改 `target content` 或 `repair entry condition`，而不是只改过滤阈值；
3. 优先考虑 `Q(B)` 与已有 class-safe / conflict-gate 机制联合，而不是把 `Q(B)` 当唯一主创新点。
