# SymNCE：An Inclusive Theoretical Framework of Robust Supervised Contrastive Loss against Label Noise

## 基本信息

- **作者与状态**：Jingyi Cui, Yi-Ge Zhang, Hengyu Liu, Yisen Wang；arXiv:2501.01130，2025-01-02 版本。当前按预印本记录，未核到正式会议/期刊出版信息。
- **原始来源**：[arXiv 页面](https://arxiv.org/abs/2501.01130)、[arXiv HTML 全文](https://arxiv.org/html/2501.01130v1)。
- **阅读状态**：已核 arXiv 全文的理论条件、SymNCE 构造与实验概览。**相关度**：高（噪声 SupCon 的风险条件与稳健对比损失；图外理论参照）。

## 理论与方法

文章在 pairwise contrastive paradigm 下推导一般的 label-noise robustness condition，分析 InfoNCE 的 noisy-risk 额外项并据此提出 SymNCE。其构造为

\[
\mathcal L_{SymNCE}=\mathcal L_{InfoNCE}+\beta\,\mathcal L_{RevNCE},
\]

其中 RevNCE 是反向项，经验形式用正样本集合和包含增强样本的候选集合计算；β 控制稳健项与原 InfoNCE 的权衡。论文理论中有明确假设，关键结果在 class-balanced 与对称标签噪声等条件下给出；不能将其表述成对任意噪声转移、任意类不均衡图数据都自动鲁棒。

## 实验、局限与迁移

全文报告 CIFAR-10/100、TinyImageNet 等合成噪声 benchmark，包含对称噪声以及若干非对称噪声测试，并含真实噪声数据实验（如 Clothing1M）；实验考察 β 对性能的影响。迁移到图节点任务时，理论中 iid/类别平衡及标签噪声模型的适用性需重新推导，不能只把损失公式放进 GNN 就沿用定理。

**本项目用途**：提供 robust contrastive loss 的理论基线与风险条件；与球关系置信度是不同层次的问题。若提出球级 SupCon，应先明确噪声怎样污染正对/负对，再验证是否满足可比的风险条件。
