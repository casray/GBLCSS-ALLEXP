# MLP-41：固定 k=5 的 kNN 共识教师

## 研究假设

检验表 4 中球式共识教师是否会在更局部的固定 `k=5` kNN 对照下表现更好。历史表 4 的 kNN 对照按每个 seed 的平均球内训练票数匹配 k，不能直接回答固定小邻域问题。

## 粒球角色与方法动机

粒球和 kNN 都在 frozen GRACE H 空间产生训练节点局部共识 target，再训练同一 GCN[X||H] teacher，最后蒸馏到同一 X-only MLP student。粒球提供自适应分组；本轮 kNN 刻意冻结为 `k=5`，只比较局部尺度，不改蒸馏流程。

## 代码与配置改动

- 新方法：`mlp_distill_knn_k5_xh`，继承原 kNN teacher predictor，只把 `matched_k` 固定为 5。
- 新配置：`external/NoisyGL/config/mlp_distill_knn_k5_xh/`。
- 汇总：`external/NoisyGL/scripts/summarize_mlp41_knn_k5_teacher.py`。
- 测试：`external/NoisyGL/tests/test_mlp41_knn_k5_teacher.py`。

## 固定协议

- 数据集：Cora、CiteSeer、CoraFull。
- 噪声：train-only uniform 0.4/0.6。
- Split：percent 0.2/0.1/0.7。
- Seeds：0--4。
- Teacher/student、H、损失、早停与原表 4 完全相同；唯一变量为 `k=5`。

## 成功与失败标准

本轮是表格替换和尺度归因实验，不以“球必须胜 kNN”为运行通过条件。完整性要求为 30/30 行、`matched_k=5`、相同 split/noise hash、无 val/test 标签污染。解释上禁止用固定 `k=5` 结果覆盖原票数匹配 kNN 证据。

## 结果

| 数据集 | 噪声率 | k=5 kNN | 球式 | 原票数匹配 kNN | k=5−球式 |
|---|---:|---:|---:|---:|---:|
| Cora | 0.4 | 81.23±0.89 | 80.60±0.69 | 80.99±1.02 | +0.63pp（4/5） |
| CiteSeer | 0.4 | 68.82±1.76 | 69.40±1.08 | 69.57±1.46 | −0.58pp（3/5） |
| CoraFull | 0.4 | 59.44±0.54 | 58.22±0.69 | 58.95±0.56 | +1.21pp（5/5） |
| Cora | 0.6 | 71.13±2.55 | 71.27±2.46 | 71.07±2.29 | −0.14pp（2/5） |
| CiteSeer | 0.6 | 62.41±2.30 | 62.28±2.30 | 61.96±2.98 | +0.13pp（2/5） |
| CoraFull | 0.6 | 53.25±0.65 | 51.82±0.86 | 52.73±0.95 | +1.42pp（4/5） |

完整性审计：30/30 行成功，所有 k 均为 5，validation/test 修改数为 0，且三臂 split/noise hash 配对一致。

## 结论与最终决策

固定 `k=5` 在 4/6 个设置均值高于球式，球式只在 CiteSeer u0.4 和 Cora u0.6 更高。该结果否定“把 k 调小即可让粒球稳定胜过 kNN”的猜测。初次按用户要求替换正文后，经公平性复核，正文恢复票数匹配 kNN：它控制了两种局部共识方法的有效训练票数，更适合作为主对照；固定 `k=5` 降为附录敏感性结果并完整保留。论文只能写两种局部共识实现随数据集和噪声率互有胜负，不能写粒球普遍优于 kNN。

证据目录：`external/NoisyGL/runs/mlp41_knn_k5_teacher_20260717/`。

## 事后机制诊断补充

- 票数匹配 k 实际为 Cora 10、CiteSeer 10/11、CoraFull 62。此前计划中“CoraFull 约 24”是记录错误，已修正。
- 当前 kNN top-k 包含节点自身；至少 97.44% 的训练节点把自身放在 top-1。由于外层 target 还会与 noisy one-hot 混合，k 的变化同时改变局部性、自票比例、margin 和 alpha，不能只理解为“票数变少”。
- k=5 的外部邻居 clean agreement 相对 matched-k：Cora `0.786 vs 0.759`，CiteSeer `0.633 vs 0.612`，CoraFull `0.485 vs 0.317`。CoraFull 的 k=62 明显超出高纯度局部邻域。
- CoraFull 的 k=5 target Brier 在 u0.4/u0.6 分别改善 `0.017/0.059`，pseudo CDR 分别下降 `12.2/5.8pp`，与学生 +0.49/+0.52pp 同向。这是目前最可信的解释线索。
- Cora/CiteSeer 没有统一 target-quality 改善；它们的 k5−matched 学生差距很小且 seed 方向不稳定。不能声称小 k 普遍优越。
