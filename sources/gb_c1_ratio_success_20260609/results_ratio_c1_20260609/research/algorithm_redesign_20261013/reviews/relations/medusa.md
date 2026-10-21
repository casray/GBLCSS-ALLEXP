# Medusa

## 元数据

Liang Chen, Zhe Xue, Yawen Li, Meiyu Liang, Yan Wang, Anton van den Hengel, Yuankai Qi. “Medusa: A Multi-Scale High-order Contrastive Dual-Diffusion Approach for Multi-View Clustering.” CVPR 2025, pp. 10295–10304. [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Medusa_A_Multi-Scale_High-order_Contrastive_Dual-Diffusion_Approach_for_Multi-View_Clustering_CVPR_2025_paper.html)；[官方PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Medusa_A_Multi-Scale_High-order_Contrastive_Dual-Diffusion_Approach_for_Multi-View_Clustering_CVPR_2025_paper.pdf)。

## 问题与任务

无监督多视图聚类，目标是共同利用视图内高阶关系与视图间互补性；不是节点标签分类，也不以训练标签污染为研究变量。

## 创新与流程

三部分：共识图引导的双图扩散卷积，把跨视图信息传播到各视图的节点/边表示；超图对比损失捕捉单视图内多个样本共同形成的高阶关系；样本级注意力融合根据每个样本学习各视图权重。先预训练学习表示和共识结构，再用聚类目标微调。

## 目标及结构更新

论文联合图扩散表征、超图对比学习和自适应融合；微调阶段加入目标分布与预测分布的聚类一致性，以及共识图重构。图/超图关系从多视图样本特征和共识图构造，属于样本图，不是已知标签图上的结构监督。

## 实验

CVPR 论文在八个多视图聚类基准上比较，包括 Citeseer、MSRC-v1、Yale，以及图像/多模态集合；使用 ACC、NMI、Purity，并进行三组数据集的模块消融。官方 PDF 的主表报告若干指标的均值和波动量；消融对比双扩散、超图关系权重及样本级自适应融合。摘要称整体优于多个基线，但这仍是无监督聚类结果。

## 对本项目的启发

可借鉴多元关系损失和样本级可信视图融合；若引入粒球超边，应验证一个球/超边表示的成员关系是否比普通K近邻和样本超图更有效，并检查异质球对节点标签的潜在错误拉近。

## 风险与边界

超图高阶关系有表达力，也会把错分成员的监督扩散给更多点；论文并未检验噪声标签抗性。模型模块较多，迁移时要逐项做消融，不能把整体提升归因于粒球式结构。

## 阅读深度与证据

已核对 CVPR 官方论文页及可检索 PDF 中摘要、方法、主实验表和消融描述；证据：[CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Medusa_A_Multi-Scale_High-order_Contrastive_Dual-Diffusion_Approach_for_Multi-View_Clustering_CVPR_2025_paper.html)；[官方PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Medusa_A_Multi-Scale_High-order_Contrastive_Dual-Diffusion_Approach_for_Multi-View_Clustering_CVPR_2025_paper.pdf)。
