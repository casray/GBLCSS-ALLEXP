# MLP-25 忠实 FINE 基线补跑

## 研究假设

旧 `mlp_fine` 仅使用中心化特征方向和固定 top-k，不能代表 FINE。新方法保留官方核心：逐 noisy class 对未中心化隐藏特征做 SVD、绝对归一化投影评分、逐类双高斯选择高均值分量。

## 固定协议

- Cora / CiteSeer seed0 smoke；通过后扩核心五数据集、seeds 0--4。
- uniform 0.4、train-only noise、percent 0.2/0.1/0.7。
- 2-layer X-only MLP，hidden 64，dropout 0.5。
- 10 epoch CE warmup；其后每轮动态重算 FINE；GMM posterior threshold=0.5。
- 不读取 clean label，不按数据集调阈值。

## 成功与失败标准

- 成功：输出有限、每类至少一个训练样本被保留、选择比例非零、无 OOM。
- 失败：选择坍缩、SVD/GMM 系统性失败或大图单 run 超过 30 分钟。

## 方法忠实度边界

本实现保留 FINE detector 的算法身份，但使用本文统一 MLP/Adam 图节点协议，因此标注为 `FINE (MLP-adapted)`，不声称复现原图像数据训练栈。

## 五种子结果与决策

| 数据集 | FINE (MLP-adapted) |
|---|---:|
| Cora | 44.83+-2.19 |
| CiteSeer | 45.27+-2.82 |
| CoraFull | 32.01+-0.96 |
| Coauthor-CS | 88.51+-0.66 |
| Amazon-Photo | 72.69+-3.15 |

25/25 行均成功，所有类别均被保留。FINE 相对 CE 只在 CiteSeer 有约 +1.77pp，Cora 约 +0.59pp，CoraFull/Coauthor-CS 基本相同，Amazon-Photo 低约 1.21pp。正式论文改用该行；旧 `mlp_fine` 仅保留为 eigenvector-alignment proxy，不参与已发表方法排名。
