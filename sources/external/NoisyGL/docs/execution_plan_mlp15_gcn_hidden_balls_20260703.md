# MLP-15 GCN-Hidden Balls Diagnostic

日期：2026-07-03
定位：why-GRACE 消融。检查“用 noisy-label GCN 训练出的 hidden embedding 替代 GRACE H 来建球，然后仍训练 SoftDist-MLP”是否可行。

## 研究假设

如果普通 supervised GCN hidden embedding 已经能提供足够好的局部语义结构，那么 frozen GRACE 不是必要选择；反之，如果 GCN hidden 的球结构只优于 shuffled 但不能超过 GRACE anchor，则 frozen label-free GRACE 仍是更稳的构球表征。

## 粒球角色

粒球仍是 SoftDist-MLP 的局部共识 soft-target builder。唯一变化是构球表征：

- Anchor：frozen GRACE H
- Candidate：noisy-label GCN hidden Z

下游 student 仍是 X-only MLP，不把 GCN hidden 拼进特征，不改变主训练损失。

## 协议

- Datasets：Cora, CiteSeer
- Seed：0
- Noise：train-only uniform 0.4
- Split：percent split，`val_percent=0.1`
- Train ratios：`0.05, 0.10, 0.20`
- Arms：
  - frozen-GRACE anchor + SoftDist-MLP
  - GCN-hidden real + SoftDist-MLP
  - GCN-hidden shuffled + SoftDist-MLP
  - MLP+CE
  - GCN+CE reference

## 成功门

该方向只有在 seed0 同时满足以下条件时才考虑扩 5-seed：

1. GCN-hidden real 在至少 `4/6` 设置中超过 GRACE anchor `>0.5pp`。
2. GCN-hidden real 在至少 `5/6` 设置中超过 GCN-hidden shuffled。

## 执行结果

输出目录：`runs/mlp15_gcn_hidden_balls_20260703/`

### 主结果

| Dataset | train% | GRACE anchor | GCN-hidden real | GCN-hidden shuffled | MLP+CE | GCN+CE ref |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 5 | 40.98 | 41.24 | 34.72 | 33.12 | 62.06 |
| Cora | 10 | 47.51 | 45.89 | 39.80 | 37.95 | 67.91 |
| Cora | 20 | 55.46 | 55.04 | 44.01 | 41.74 | 79.26 |
| CiteSeer | 5 | 44.43 | 40.08 | 35.27 | 35.02 | 49.59 |
| CiteSeer | 10 | 50.36 | 50.43 | 41.75 | 40.40 | 58.29 |
| CiteSeer | 20 | 55.33 | 56.19 | 43.60 | 46.09 | 64.86 |

### Paired deltas

| Dataset | train% | GCN-hidden real - GRACE | GCN-hidden real - shuffled | GCN-hidden real - MLP+CE |
|---|---:|---:|---:|---:|
| Cora | 5 | +0.26pp | +6.52pp | +8.13pp |
| Cora | 10 | -1.62pp | +6.09pp | +7.94pp |
| Cora | 20 | -0.42pp | +11.03pp | +13.30pp |
| CiteSeer | 5 | -4.35pp | +4.81pp | +5.06pp |
| CiteSeer | 10 | +0.08pp | +8.68pp | +10.03pp |
| CiteSeer | 20 | +0.86pp | +12.59pp | +10.09pp |

### Gate

- GCN-hidden real > GRACE anchor by `>0.5pp`：`1/6`，不过门。
- GCN-hidden real > shuffled：`6/6`，通过。

## 结论

GCN-hidden 不是无效表征：它在所有 6 个设置中都显著超过 shuffled，说明 supervised GCN hidden 中确实存在可用的局部结构。但它不能作为 frozen GRACE 的替代 anchor：只有 CiteSeer 20% 一个设置超过 GRACE anchor 超过 0.5pp，其余设置要么持平、要么落后，CiteSeer 5% 甚至落后 `-4.35pp`。

最终决策：**不扩 5-seed，不改主线。** 该结果应作为 why-GRACE 消融写入分析：普通 noisy-label supervised GCN hidden 可以产生真实局部结构，但其结构质量和稳定性不足以替代 label-free frozen GRACE。

输出文件：

- `runs/mlp15_gcn_hidden_balls_20260703/gcn_hidden_pretrain.csv`
- `runs/mlp15_gcn_hidden_balls_20260703/mlp15_raw.csv`
- `runs/mlp15_gcn_hidden_balls_20260703/mlp15_summary.csv`
- `runs/mlp15_gcn_hidden_balls_20260703/mlp15_paired_deltas.csv`
- `runs/mlp15_gcn_hidden_balls_20260703/mlp15_report.md`
