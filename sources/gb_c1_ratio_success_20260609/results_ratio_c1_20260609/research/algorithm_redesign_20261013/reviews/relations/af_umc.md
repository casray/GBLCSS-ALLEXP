# AF-UMC

## 元数据

Bohang Sun, Yuena Lin, Tao Yang, Zhen Zhu, Zhen Yang, Gengyu Lyu. “AF-UMC: An Alignment-Free Fusion Framework for Unaligned Multi-View Clustering.” NeurIPS 2025. [NeurIPS 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/965484d5b2b2624ba17295612a3ba7e8-Abstract-Conference.html)；[官方PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/965484d5b2b2624ba17295612a3ba7e8-Paper-Conference.pdf)。

## 问题与任务

未对齐多视图聚类：跨视图样本特征没有完整对应，传统逐样本对齐再融合可能产生不可靠匹配。该任务是无监督多视图聚类，不是“没有图节点标签的图分类”，也没有节点标签噪声设置。

## 创新与流程

先以跨视图重构和结构清晰度正则学习共识基空间；各视图自编码器将视图特异输入投影至该基空间并提取一致表示；再用实例到全局中心的对比融合，将各视图表示拉向全局统计结构，最后用融合表示做 K-means。作者刻意绕开不可靠的逐实例匹配。

## 目标及结构更新

联合重构损失、基空间 Structural Clarity Regularization 和 Instance Global Contrastive Fusion。Fusion 模块是跨视图全局中心级对齐；不是通过已知节点标签把同类节点拉近。

## 实验

十个多视图数据集：Caltech7-5、Handwritten、Scene、Caltech102-5、Hdigit、ALOI、NUSWIDEOBJ、NoisyMNIST、CIFAR-10、YoutubeFace。与未对齐聚类方法比较小规模集合，并在大规模集合加入可扩展的多视图聚类基线，指标含 ACC、NMI、PUR、ARI；在 Caltech7-5 与 NoisyMNIST 上做消融。论文 checklist 明确未报告误差棒/显著性检验，理由是大多数基线没有报告显著性。

## 对本项目的启发

在图任务中，结构视图与属性视图可能存在关系错位；全局/簇级信号可提供比逐节点硬匹配更稳的桥梁。但节点级监督任务需要保留逐节点可预测性，不能只优化跨视图全局对齐。若无图推理作为独立问题，AF-UMC 的一致表示思路可作启发，但它不验证图结构蒸馏或标签去噪。

## 风险与边界

它关注跨视图样本未对齐而非图边污染、观测噪声或训练标签错误。全局中心对比可能掩盖样本级差异；没有统计显著性报告。

## 阅读深度与证据

已读 NeurIPS 全文第 3 节基空间、结构清晰度和全局对比融合，第 4 节十数据集、比较/消融及 checklist（未报告显著性）。证据：[官方PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/965484d5b2b2624ba17295612a3ba7e8-Paper-Conference.pdf)，尤其第 4–8 页和第 15 页 checklist。
