# MLP-26 FF-G2M 忠实度补跑

## 研究假设

旧 `ffg2m_adapted` 使用边相似度/距离 MSE，不能代表 FF-G2M。新 `ffg2m_reimplemented` 保留官方核心 LFD、HFD 及其与训练标签 CE 的组合，可作为 common-backbone paper reimplementation 进入 graph-free 扩展表。

## 固定协议

- uniform 0.4、train-only noise、percent 0.2/0.1/0.7。
- GCN teacher 与 X-only MLP student 均为 2 层、hidden 64、200 epochs。
- `lambda=0.4`，`tau=1.5`，所有数据集固定，不按结果调参。
- Teacher checkpoint 只看 validation；恢复后 test 只评一次。
- Student loss：`0.4*CE_noisy + 0.6*(LFD+HFD)`。

## 成功标准

- seed0 Cora/CiteSeer 均 finite、无 OOM；LFD/HFD/label loss diagnostics 均有限。
- 推理阶段只给 X，不访问邻接矩阵。
- 通过后扩核心五数据集、seeds 0--4。

## 身份边界

使用官方公式的语义保持重实现，但骨干和协议统一为本文设置，称为 `FF-G2M (common-backbone reimplementation)`，不称 official tuned reproduction。

## 五种子结果与决策

| 数据集 | FF-G2M reimplemented |
|---|---:|
| Cora | 77.65+-2.97 |
| CiteSeer | 65.80+-0.49 |
| CoraFull | 56.58+-0.93 |
| Coauthor-CS | 91.58+-0.24 |
| Amazon-Photo | 84.82+-0.84 |

25/25 行全部成功。相对旧 frequency proxy，忠实结果下降 0.37--5.19pp；相对 LCSS 蒸馏扩展，五数据集分别低 2.95、3.60、1.64、2.17、5.67pp。正式 Table B 用该行替换 proxy。
