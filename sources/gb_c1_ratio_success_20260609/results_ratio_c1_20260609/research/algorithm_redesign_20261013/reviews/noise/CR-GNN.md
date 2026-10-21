# CR-GNN：Contrastive Learning of Graphs under Label Noise

## 基本信息

- **作者与出处**：Xianxian Li, Qiyu Li, De Li, Haodong Qian, Jinyan Wang；*Neural Networks* 172 (2024), 106113，DOI [10.1016/j.neunet.2024.106113](https://doi.org/10.1016/j.neunet.2024.106113)。
- **原始来源**：[ScienceDirect 官方页面](https://www.sciencedirect.com/science/article/pii/S0893608024000273)、[PubMed 题录与摘要](https://pubmed.ncbi.nlm.nih.gov/38232430/)。
- **阅读状态**：已核出版社页面、PubMed 摘要及公开方法/实验描述；未取得完整可读论文正文。**相关度**：最高（直接 noisy-node-label graph contrastive learning）。

## 问题与机制

CR-GNN 面向标签稀疏且有噪声的半监督节点分类。它使用两种图增强视图（特征掩码、边丢弃）上的无监督对比损失，并依据图同质性加入邻居对比损失，以利用局部结构。考虑到增强视图可能导致预测不一致，动态交叉熵项选择预测一致的节点作为较可信监督样本；cross-space consistency 项缩小对比表示空间与分类空间的语义差距。

**目标变量**是噪声图上的节点表征及分类精度；对比项以增强视图和邻居关系作结构监督，另用动态筛选保护分类监督。可核来源未提供完整损失索引定义，因此不重写具体公式。

## 实验、局限与迁移

摘要报告 Cora、CiteSeer、PubMed、DBLP 四个常用节点分类数据集，研究标签稀疏/带噪情形。具体噪声矩阵、比例和完整统计量未在本卡复核。作者明确以 homophily 为邻居对比的结构依据；因此在异配图、边噪声重或假邻居较多时，邻居对比假设需要额外验证。特征掩码与边丢弃的增强也可能产生表示/预测不一致，这正是该文加入 consistency 机制的动机。

**本项目用途**：它是直接先例，足以否定“此前图对比学习没有处理标签噪声”的主张。可比较的问题应是：粒球证据是否优于单纯同质性邻居关系，以及粒球软置信度能否降低假正例/假负例，而非仅将普通 GCL 搬到图噪声场景。
