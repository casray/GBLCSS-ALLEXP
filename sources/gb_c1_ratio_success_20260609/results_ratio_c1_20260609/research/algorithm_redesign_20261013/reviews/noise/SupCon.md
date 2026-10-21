# SupCon：Supervised Contrastive Learning

## 基本信息

- **作者**：Prannay Khosla et al.（九位作者）。
- **出处**：NeurIPS 2020；[会议页](https://papers.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)，[PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf)。

## 核心机制与目标

SupCon 把一个 batch 中同标签的其他视图/样本都作为 anchor 的 positives，其他样本作为 negatives；对每个正对使用温度缩放的 softmax 对比项并在正样本上平均。与单正样本的自监督对比相比，它利用标签形成类级簇结构，推动同类聚集、异类分离。原文比较了两种 SupCon 写法，并选择表现更好的形式。

以简式表示核心关系（具体索引/增强视图按原文）：\(L_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}\log\frac{\exp(z_i\cdot z_p/\tau)}{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}\)。正对由观测标签给出，所以若标签错，可能制造语义错误的正对；SupCon 是重要基线，不是本身具有标签噪声保证的算法。

## 实验与本项目适用性

原文在 ImageNet 等图像分类任务显示相较 CE 的表征/分类优势，并报告自然腐蚀鲁棒性；自然腐蚀鲁棒性不等于标签噪声鲁棒性。迁移到图节点任务时，batch 构造、节点增强、采样标签密度和正负集合定义必须一致比较。凡以“球同属/球异属”替换 P(i)，都需要额外论证球成员置信度与错误正对率。**阅读状态**：已核官方会议页及核心 SupCon 定义/公式。**相关度**：高（监督对比基线）；标签噪声风险为项目机制定位参照，而非 SupCon 的抗噪保证。


