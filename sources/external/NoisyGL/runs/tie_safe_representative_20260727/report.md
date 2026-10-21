# Tie-safe 代表性实验报告

- 生成时间（UTC）：2026-07-27T11:05:14.614033+00:00
- 状态：代表性 30 行已完成并通过跨模式协议一致性检查。
- 固定主修复：`distribution_only`；`alpha_zero` 仅为机制消融。
- 统计区间：n=5 配对差、df=4、未校正 nominal 95% t 区间。
- Claim 边界：CI 包含 0 不证明等价或非劣。

## 汇总

| dataset | mode | role | test mean±std | paired Δ mean [95% CI] | detectable | material |
|---|---|---|---:|---:|---:|---:|
| cora | min_index | historical_control | 0.4104±0.0237 | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| cora | alpha_zero | mechanism_ablation | 0.3902±0.0266 | -0.0203 [-0.0410, +0.0004] | 0 | 1 |
| cora | distribution_only | fixed_primary_repair | 0.4021±0.0294 | -0.0083 [-0.0234, +0.0067] | 0 | 0 |
| corafull | min_index | historical_control | 0.2716±0.0096 | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| corafull | alpha_zero | mechanism_ablation | 0.2657±0.0076 | -0.0059 [-0.0117, -0.0001] | 1 | 0 |
| corafull | distribution_only | fixed_primary_repair | 0.2735±0.0085 | +0.0020 [-0.0044, +0.0083] | 0 | 0 |

## 历史 min_index 复现

- 对照行数：10。
- PASS：5。
- FAIL：5。
- GAP：0。

## 固定后续

- 只要数学测试与本次协议完整性通过，完整 25-setting × 5-seed × 4-method 网格无条件执行。
- 代表性 test accuracy 不改变主修复、消融角色或扩展决定。

## 产物

- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\raw_combined.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\paired_deltas.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\summary.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\historical_reproduction.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\command_manifest.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\input_snapshot_manifest.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\runs\tie_safe_representative_20260727\environment_fingerprint.json`
