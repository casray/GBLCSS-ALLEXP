# CAL

## 元数据

Zhe Chen, Fanhui Meng, Tianyang Xu, Xiao-Jun Wu. “Cluster-aware Anchor Learning for Multi-View Clustering.” CVPR 2026, pp. 17714–17723. [CVPR 官方PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Cluster-aware_Anchor_Learning_for_Multi-View_Clustering_CVPR_2026_paper.pdf)。

## 问题与任务

无监督多视图聚类。传统锚点方法预先固定锚点数，难适应簇之间规模、结构和信息量的差异。CAL 不是对比学习方法，也不把图标签或标签噪声作为任务。

## 创新与流程

学习共识锚点关系矩阵，并将锚点列按簇组织成组；对每簇施加列稀疏的 ℓ2,1 正则，自动压缩冗余锚点、决定各簇保留的锚点数；另以簇间正则使组间关系更可区分。最终学习表示矩阵并以 K-means 得到聚类结果。

## 目标及结构更新

迭代更新视图锚点关系、表示矩阵、簇组稀疏矩阵和乘子；损失由重构/共识表示、稀疏锚点和簇间分离等项构成。CAL 的“自适应”体现在每簇锚点预算，而不是噪声置信度或标签可信度。

## 实验

八个多视图基准：MSRC、Dermatology、100Leaves、HW、OutdoorScene、CCV、ALOI、YTF20；与十种多视图聚类方法比较，报告 ACC、NMI、Purity 等指标，并有参数/消融分析。官方论文报告多数数据集上的优势；这不代表所有数据集都显著更好，论文缺少足以支持普遍统计显著性的统一误差/检验依据。

## 对本项目的启发

粒球数和粒球内代表中心数也可能因局部异质性不同而变化。CAL 可作为“每个簇/局部区域需要多少代表单元”的锚点基线，但粒球的额外主张须证明中心、成员、半径带来锚点稀疏模型没有的收益。

## 风险与边界

簇组结构以待优化表示和分配为依据，簇错分会影响锚点选择。方法没有对比损失和噪声标签实验，不能当可信关系学习或抗噪分类证据。

## 阅读深度与证据

核对 CVPR 官方 PDF 索引中的方法优化、Algorithm 1、实验设置和八数据集列表；CVF 文件端点直接访问受限，记录为官方PDF可检索正文片段级，未声称逐页视觉核对。[官方PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Cluster-aware_Anchor_Learning_for_Multi-View_Clustering_CVPR_2026_paper.pdf)。
