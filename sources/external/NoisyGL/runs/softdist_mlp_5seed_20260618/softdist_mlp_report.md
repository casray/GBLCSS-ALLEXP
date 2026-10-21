# SoftDist-MLP Diagnostic Report

## Question

当冻结 GRACE 表示 H 不再拼接到下游输入、图卷积被两层 MLP 替代后，粒球 SoftDist 是否仍能提供独立且可归因的抗噪增量？

## Protocol

- Datasets: Cora, CiteSeer
- Split: train/validation/test = 20%/10%/70%
- Noise: uniform, rate 0.4
- Seeds: 0, 1, 2, 3, 4
- Classifier input: original X only
- Ball construction: frozen GRACE H
- Shared classifier: 2-layer MLP, hidden 64, dropout 0.5
- Controls: matched `MLP(X)+CE` and shuffled ball membership
- Raw evidence: `softdist_mlp_5seed_summary.csv`, `softdist_mlp_5seed_raw.log`

## Results

Test accuracy is reported as mean +/- sample standard deviation.

| Dataset | MLP(X)+CE | SoftDist-MLP(X) | Shuffled control |
|---|---:|---:|---:|
| Cora | 45.88 +/- 3.44 | **54.63 +/- 1.46** | 46.51 +/- 1.92 |
| CiteSeer | 44.53 +/- 4.11 | **52.30 +/- 2.84** | 44.93 +/- 3.46 |

Paired gains of real SoftDist:

| Dataset | vs CE | vs shuffled |
|---|---:|---:|
| Cora | +8.75 +/- 3.37 points, 5/5 wins | +8.12 +/- 2.27 points, 5/5 wins |
| CiteSeer | +7.77 +/- 2.27 points, 5/5 wins | +7.37 +/- 2.79 points, 5/5 wins |

## Decision

**Continue.** 当前结果支持粒球 SoftDist 的贡献可以脱离 `[X||H]` 下游输入和 GCN 消息传递而存在。shuffled control 与 CE 接近，说明收益主要来自有语义的粒球成员关系，而不是任意分组或一般性软标签平滑。

结论边界必须保留：该方法仍使用 H 建球，因此不是端到端“无 H”；目前只验证了两个引文图、uniform 0.4 和一个 split 协议，也尚未测量运行时间/显存优势。下一步应优先补噪声率、噪声类型和复杂度证据。
