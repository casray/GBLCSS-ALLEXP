# MLP-34：强基线、噪声率估计与选择机制诊断

## 1. 研究假设

1. PubMed 上 `LCSS+soft-small-loss` 是否真正越过 JoCoR 强基线，需要在 Pair/Instance 0.4 下补齐五种子配对证据。
2. frozen GRACE-H 粒球内的 noisy train label 分歧，是否可通过 leave-one-out 投票估计未知噪声率。
3. LCSS soft target 与 Small-loss 的协同，是否来自更高质量的样本选择，而不是 Small-loss 单独贡献或 shuffled grouping。

## 2. 粒球角色与方法动机

粒球保持为 label-blind 的局部共识单元。标签只在球生成完成后用于 train-only soft target、LOO noisy vote 和事后机制诊断。粒球不读取 val/test 标签，不改变 frozen GRACE、`64/32/0.05`、beta=0.5 或 MLP。

## 3. 代码改动

- `utils/lcss_noise_rate.py`：noisy-only LOO 噪声率估计器。
- `utils/selection_trace.py`：把局部 selected index 映射为全局节点 mask，并保存 NPZ。
- `predictor/MLP34SelectionTrace_Predictor.py`：三个只读 trace predictor。
- `predictor/LCSSRobust_Predictor.py`、`predictor/MLPNoiseBaselines_Predictor.py`：仅增加无状态 selection hook；未启用 trace 时行为不变。
- `scripts/run_mlp34_noise_rate_diagnostic.py`：B1 诊断与闸门。
- `scripts/run_mlp34_selection_trace.py`：45 个 seed0 trace 运行。
- `scripts/summarize_mlp34_pubmed_rhohat_mechanism.py`：严格哈希审计、机制指标、LODO 相关性和报告。
- 测试：`tests/test_mlp34_noise_rate_estimator.py`、`tests/test_mlp34_selection_trace.py`、`tests/test_mlp34_summary.py`。

## 4. 固定协议与预注册门槛

- Split：percent 0.2/0.1/0.7。
- Noise scope：train-only。
- GPU：ROCm `cuda:0`。
- Phase A：PubMed JoCoR，Pair/Instance 0.4，seeds 0-4。
- Phase B：Cora/CiteSeer/CoraFull，uniform 0.6，seeds 0-4。至少 2/3 数据集 mean abs error <=0.10、任一不超过 0.15、coverage >=50%，才进入 B2。
- Phase C：五数据集乘三噪声设置乘三方法，seed0；epoch 1/10/50/100/200 保存 mask。

## 5. 实验结果

### 5.1 Phase A：PubMed JoCoR

| 噪声 | 组合 | JoCoR | 配对差 | 正向 seed | 1pp+4/5 门 |
|---|---:|---:|---:|---:|---:|
| Pair 0.4 | 77.71+-0.73 | 73.93+-1.32 | +3.78pp | 5/5 | 通过 |
| Instance 0.4 | 80.94+-0.93 | 76.89+-0.72 | +4.05pp | 5/5 | 通过 |
| Uniform 0.6 | 59.91+-6.21 | 58.12+-5.22 | +1.79pp | 3/5 | 未过 |

结论：只允许写“在 Pair/Instance 结构化噪声下消除 PubMed 的 JoCoR 边界”或“条件性边界消除”。禁止写“三种噪声均稳定越过 JoCoR”。

### 5.2 Phase B：noisy-only LOO 噪声率估计

| 数据集 | weighted rho-hat | mean abs error | coverage | effective n |
|---|---:|---:|---:|---:|
| Cora | 0.724 | 0.120 | 1.000 | 425.2 |
| CiteSeer | 0.734 | 0.132 | 0.998 | 526.5 |
| CoraFull | 0.838 | 0.233 | 1.000 | 3227.1 |

B1 未过门，B2 按预注册跳过。失败不是票数 coverage 不足，而是球内分歧同时包含语义边界和噪声，尤其在 70 类 CoraFull 上产生严重上偏。当前方法仍需已知/外部估计的 forget rate；不调第二个估计公式。

### 5.3 Phase C：选择掩码

完整性：45/45 行成功，45 个 NPZ 均含 5 个指定 epoch，val/test modified count 为 0，每个设置三臂 split/noise hash 完全一致。与历史 anchor 比较时 40/45 精度逐位相同，45/45 的协议哈希和选择数量一致；其余 5 行最大偏差 2.76pp。对 PubMed real trace 原样复跑后回到历史值，确认这是 ROCm 运行级非确定性而非 trace hook 改变 loss。正式 trace 不因复跑结果而覆盖。

在 epoch 200：

- Pair 0.4：组合 selected-target accuracy 平均 74.21%，Small-loss 为 70.62%，提高 3.60pp。
- Instance 0.4：组合为 84.02%，Small-loss 为 77.41%，提高 6.60pp。
- 7/10 Pair/Instance 设置提高至少 3pp；real 对 shuffled 为 10/10 正向。
- 但“正确 hard-repaired noisy node 的保留率提高 5pp”仅 1/10，通过联合机制门的也仅 1/10，因此原先的 hard-repair retention 假设被否定。
- final mask turnover：Pair 中组合 0.105、Small-loss 0.280；Instance 中组合 0.083、Small-loss 0.243。组合选择更稳定，但该结果为 seed0 描述性证据。

目标诊断进一步显示：Pair/Instance 的 soft-target Brier gain 分别平均 0.236/0.286，但 argmax NCR 只有 0.056/0.066。说明主要机制不是把大量 noisy label 硬翻成 clean argmax，而是用更合理的 soft probability mass 重排逐节点损失，进而选择更干净、且更稳定的训练子集。

### 5.4 饱和假设

`Combination-LCSS` 与独立 target 指标的 Spearman：Brier gain 0.039、NCR -0.509、CDR -0.528、margin 0.636。虽然后三项绝对值超过 0.5 且 LODO 方向一致，只有 NCR 的符号符合“target 越充分修复，额外选择收益越弱”；CDR 和 margin 方向相反。故不能写一般性的饱和规律，Amazon-Photo 仍只作为边界案例。

## 6. 最终决策

- **继续**：第二篇主线保留“LCSS soft target + soft-small-loss 在中高噪声下协同”；Pair/Instance 的 PubMed 强基线证据得到补强。
- **关闭**：当前 noisy-only LOO rho estimator；不执行 B2，不调第二公式。
- **修正机制**：从“保留被硬纠正的噪声节点”改为“soft target 改善损失排序与所选监督质量”。
- **Claim 边界**：Phase A/B 为五种子性能/估计证据；Phase C 是 seed0、clean-dependent 事后解释，不能包装成五种子性能结论。

## 7. 可追溯资产

- 输出目录：`runs/mlp34_pubmed_rhohat_mechanism_20260714/`
- 主报告：`mlp34_report.md`
- Phase A：`mlp34_phase_a_pubmed_jocor*.csv`
- Phase B：`rhohat_uniform0p6_seeds_0_1_2_3_4*.csv`、`mlp34_rhohat_gate.json`
- Phase C：`selection_trace_seed0_raw.csv`、`traces/*.npz`、`mlp34_selection_*.csv`、`mlp34_target_diagnostics.csv`
- 饱和分析：`mlp34_saturation_rows.csv`、`mlp34_saturation_correlations.csv`
