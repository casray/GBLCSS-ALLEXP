# MLP-8 P2a CE-0.3 Comparison and Complete SoftDist-MLP Rerun

日期：2026-07-03
定位：回应“CE=0.3 是否可能超过 CE=0.2，以及是否跑过完整 SoftDist-MLP”的最后补充实验。该实验只做 Cora seed0，不产生主表 claim。

## 研究问题

CE=0.2 已经能避免 CE=0.1 的单球塌缩，但没有稳定超过 frozen-GRACE anchor。进一步提高到 CE=0.3 可能提高类别可分性，也可能更强地拟合 noisy labels。因此先比较 CE=0.2 与 CE=0.3 的 default/loose SoftDist-MLP 下游，再选最高设置做一次独立命名的完整 SoftDist-MLP 标准复跑。

## 协议

- Dataset：Cora
- Seed：0
- Noise：train-only uniform 0.4
- Split：percent 0.2/0.1/0.7
- Candidate pretraining：
  - GRACE + `0.2 * CE(noisy train labels)`
  - GRACE + `0.3 * CE(noisy train labels)`
- SoftDist-MLP：
  - real：`frozengrace_gb_mlp_softdist_beta05`
  - shuffled：`frozengrace_gb_mlp_softdist_beta05_shuffled`
  - default params：`64/32/0.05`
  - loose params：`64/16/0.0`

## 结果

| Lambda | Params | Real test | Shuffled test | Real - shuffled | Balls | True train purity |
|---:|---|---:|---:|---:|---:|---:|
| 0.2 | default 64/32/0.05 | 55.46 | 42.64 | +12.82pp | 39 | 0.7967 |
| 0.2 | loose 64/16/0.0 | 53.67 | 42.16 | +11.50pp | 64 | 0.8219 |
| 0.3 | default 64/32/0.05 | 54.99 | 44.06 | +10.92pp | 50 | 0.8299 |
| 0.3 | loose 64/16/0.0 | 53.09 | 42.37 | +10.71pp | 64 | 0.8430 |

Anchor 对照：

| Arm | Test |
|---|---:|
| frozen-GRACE default | 55.46 |
| frozen-GRACE loose | 53.61 |

## 完整 SoftDist-MLP 复跑

按最高设置 `CE=0.2, default 64/32/0.05`，用标准 `run_ssl_swap_downstream.py` 独立复跑完整 SoftDist-MLP 流程，输出 `runs/mlp8_p2a_ce03_rescue_20260703/selected_complete_ce02_default.csv`。

复跑结果：

| Method | Test | Balls | True train purity |
|---|---:|---:|---:|
| CE-0.2 default real | 55.46 | 39 | 0.7967 |
| CE-0.2 default shuffled | 42.64 | 39 | 0.3530 |

## 决策

CE=0.3 没有超过 CE=0.2。最高设置仍是 CE=0.2 default，但它与 frozen-GRACE default anchor 持平，未达到“超过 anchor”的门槛。CE=0.2/0.3 都显示 real local structure 明显强于 shuffled，说明有监督辅助表征中仍存在有效局部结构；但它们不能提供 frozen label-free GRACE 之外的稳定下游收益。

最终结论：**P2a 全部关闭，不扩 5-seed。论文/导师沟通中可写：noisy CE 辅助预训练能改善构球几何、避免塌缩，但不会超过 frozen-GRACE anchor；这反而支持继续使用 label-free frozen GRACE 作为主方法表征源。**
