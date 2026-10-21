# MLP-41：表 4 固定 k=5 的 kNN 共识教师

## 研究假设

表 4 当前 kNN teacher 按每个 seed 的球内平均 train 票数匹配 k，实际为 Cora 10、
CiteSeer 10/11、CoraFull 约 62。用户要求用更局部的固定 k=5 重新评估，以检验球式
teacher 相对小邻域 kNN teacher 的表现。

## 协议边界

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：train-only uniform 0.4/0.6。
- Split：percent 0.2/0.1/0.7。
- Seeds：0--4。
- Teacher：GCN[X||H]，用固定 k=5 的 H 空间 train-neighbor target 训练。
- Student：与表 4 相同的 X-only 两层 MLP，蒸馏温度与优化参数不变。
- 原 T4 的 `target_beta=0.0` 保持不变；唯一实验变量是 `matched_k=5`。

## 实现与审计

- 新方法名 `mlp_distill_knn_k5_xh`，不得覆盖 `mlp_distill_knn_xh` 历史 anchor。
- CSV 必须记录 `matched_k=5`、teacher/student test、split/noise hash、标签污染范围。
- 每个 dataset/rate/seed 必须 1 行，共 30 行；val/test 修改数必须为 0。

## 执行顺序

1. seed0 smoke：3×2=6 runs。
2. smoke 全部成功且 `matched_k=5` 后，补 seeds1--4，共 24 runs。
3. 汇总五种子均值±标准差并替换 v15 表 4 的 kNN 行，行名必须注明 `k=5`。

## 结果解释

- 本实验只回答固定小邻域 kNN teacher 与球式 teacher 的比较。
- 不删除或改写原 matched-k 结果；原结果继续保留在 MLP-9 历史报告中。
- 若球式 teacher 更高，只能写“相对固定 k=5 的局部 kNN teacher 更高”，不能据此声称
  粒球普遍优于调优或尺度匹配的 kNN。

## 实验结果

30/30 个正式运行均成功，`matched_k=5`，且 validation/test 标签修改数均为 0。

| 数据集 | 噪声率 | 固定 k=5 kNN teacher | 球式 teacher | 原票数匹配 kNN teacher | k=5−球式 | k=5−票数匹配 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.4 | 81.23±0.89 | 80.60±0.69 | 80.99±1.02 | +0.63pp（4/5） | +0.24pp（4/5） |
| CiteSeer | 0.4 | 68.82±1.76 | 69.40±1.08 | 69.57±1.46 | −0.58pp（3/5） | −0.75pp（2/5） |
| CoraFull | 0.4 | 59.44±0.54 | 58.22±0.69 | 58.95±0.56 | +1.21pp（5/5） | +0.49pp（4/5） |
| Cora | 0.6 | 71.13±2.55 | 71.27±2.46 | 71.07±2.29 | −0.14pp（2/5） | +0.06pp（2/5） |
| CiteSeer | 0.6 | 62.41±2.30 | 62.28±2.30 | 61.96±2.98 | +0.13pp（2/5） | +0.45pp（3/5） |
| CoraFull | 0.6 | 53.25±0.65 | 51.82±0.86 | 52.73±0.95 | +1.42pp（4/5） | +0.52pp（4/5） |

## 最终决策

- 初次按用户要求将正文表 4 替换为固定 `k=5`；随后经公平性复核，正文恢复票数匹配 kNN，固定 `k=5` 降为附录敏感性结果。
- 固定 `k=5` 相对球式在 4/6 个设置均值更高；球式仅在 CiteSeer u0.4 与 Cora u0.6 更高。因此不能声称“小 k 会使粒球稳定胜过 kNN”。
- 票数匹配是正文更公平的主对照，因为它控制两种局部共识方法的有效训练票数；固定 `k=5` 继续完整披露，不删除负结果。
- 本轮不改变 LCSS 主算法、教师结构或其他论文表格。

证据：`runs/mlp41_knn_k5_teacher_20260717/`。

## 事后机制诊断

使用 clean label 的事后诊断只用于解释，不参与训练或方法选择。关键发现如下：

- kNN 相似度 top-1 在至少 97.44% 的训练节点上是节点自身，说明当前 kNN target 包含自票。
- 外部邻居 clean-label agreement（k=5 vs 票数匹配）为：Cora `0.786/0.759`、CiteSeer `0.633/0.612`、CoraFull `0.485/0.317`。CoraFull 的差距最大。
- CoraFull 上，k=5 相对 k=62 将 soft-target Brier 从 `0.619→0.602`（u0.4）和 `1.013→0.954`（u0.6），pseudo CDR 从 `0.475→0.353` 和 `0.480→0.422`。这与 k=5 学生在 CoraFull 的 +0.49/+0.52pp 增量方向一致。
- Cora/CiteSeer 的 Brier/CDR 不呈现统一优势，且学生差值仅 `+0.24/−0.75/+0.06/+0.45pp`，多数设置的逐 seed 方向不稳定。因此不能外推“小 k 普遍更好”。
- MLP-11 的直接 target 训练中，k=5 明显低于 k=20；小 k 在 teacher 蒸馏中变得可竞争，可能与 GCN[X||H] teacher 的图平滑、干净验证早停和全节点蒸馏吸收局部方差有关，但由于两轮 target_beta 不同，该解释不是严格因果证明。

诊断文件：`runs/mlp41_knn_k5_teacher_20260717/mlp41_k5_target_diagnostics.csv` 与 `mlp41_k5_target_diagnostics_aggregate.csv`。
