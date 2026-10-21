# NR-GNN：A Noise-Resistant Graph Neural Network by Semi-Supervised Contrastive Learning

## 基本信息

- **作者与出处**：Zhengyu Lu, Junbo Ma, Zongqian Wu, Bo Zhou, Xiaofeng Zhu；*Information Sciences* 658 (2024), article 120001，DOI [10.1016/j.ins.2023.120001](https://doi.org/10.1016/j.ins.2023.120001)。
- **原始来源**：[ScienceDirect 官方页面](https://www.sciencedirect.com/science/article/pii/S0020025523015864)。
- **阅读状态**：已核出版社题录、摘要和方法摘要；未取得/未逐页审读全文。**相关度**：高（直接图节点标签噪声 + 半监督对比约束）。
- **名称消歧**：本卡用 NR-GNN 指 Lu et al. (2024) 的 *A noise-resistant graph neural network by semi-supervised contrastive learning*；不要与 Dai, Aggarwal & Wang 的 NRGNN（KDD 2021，*Learning a Label Noise Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs*）混为一谈。后者以特征相似连接未标注节点和可信标注节点、生成伪标签，不是本卡所述的半监督对比排斥方法。核对基线时按标题、年份、DOI/代码仓库区分。参考：[KDD 2021 官方录用页](https://www.kdd.org/kdd2021/accepted-papers/)、[NRGNN 作者实现](https://github.com/EnyanDai/NRGNN)。

## 问题与机制

论文指出：有噪标签节点的表示可能靠近未标注节点，使带噪监督影响由 GNN 传播。提出的半监督对比约束将被判为噪声的已标注节点与其未标注邻居推远，减小这些噪声节点对未标注节点表示及分类器的影响，并把该约束加入 GNN 总目标。

**目标变量**是图节点嵌入和节点分类器；核心关系是“噪声已标注节点—未标注邻居”的排斥约束。出版社摘要未提供足以准确重建正负集合和温度项的损失表达式，故不在此写公式。应注意它的设计对象不是所有同球节点间隔，也不是一般性 supervised InfoNCE。

## 实验、局限与迁移

出版社摘要报告在真实图数据上，以不同噪声比例进行分类比较，并称该方法优于对比基线；当前未核对具体数据集、噪声协议、方差和显著性。其机制要求能识别/估计噪声节点，且“噪声节点应远离未标注邻居”可能在同类邻接占优时惩罚正确的局部同类关系；这是迁移时应验证的风险，不是摘要已证明的缺陷。

**本项目用途**：它已经构成图上的 contrastive noisy-label 先例，因此不能将“首次把图对比学习用于噪声标签”作为新颖性主张。粒球关系置信度如何缓和误排斥、如何不同于按噪声标记直接推远节点，是可继续比较的问题。


