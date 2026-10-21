# GBS：Granular Ball Sampling for Noisy Label Classification or Imbalanced Classification

## 基本信息

- **作者**：Shuyin Xia, Shaoyuan Zheng, Guoyin Wang, Xinbo Gao, Binggui Wang。
- **出处**：*IEEE Transactions on Neural Networks and Learning Systems* 34(4), 2144–2155 (2023；DOI 10.1109/TNNLS.2021.3105984)。[IEEE Xplore](https://ieeexplore.ieee.org/document/9525034/)，[PubMed 题录与摘要](https://pubmed.ncbi.nlm.nih.gov/34460405/)。

## 问题、方法与目标

提出与具体分类器相对独立的采样/压缩方法：用自适应生成的超球覆盖数据空间，再从球边界/球表示中选取代表样本组成更小的数据集。作者主张该过程既降低训练量，也改善 noisy-label classification 的样本质量，并可用于不平衡分类欠采样；其对象是**训练数据选择与压缩**，不是可学习对比损失或标签概率估计器。

文章摘要未提供一条统一可写的损失公式；关键变量是球划分和采样点集合。不能把“保留边界、分类精度接近原数据”当作错误标签逐一识别/修正的证据。

## 实验与迁移

摘要称在含噪标签分类中较随机采样获得更好准确率、且时间复杂度接近 \(O(N)\)，同时用于类别不平衡欠采样；详细噪声类型、数据集、重复次数和统计量需回全文确认。对本项目可作粒球数据压缩先例，但核心目标与在线训练中每个节点关系置信度不同。若考虑采样，需固定训练/测试划分并与不采样 baseline 对齐，避免采样改变评估协议。**阅读状态**：已核 PubMed 题录/摘要与方法概述，未逐页核全文实验表。**相关度**：中（粒球抗噪采样先例）。


