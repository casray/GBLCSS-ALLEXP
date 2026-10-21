# VPGB：A Granular-Ball Based Model for Attribute Reduction and Classification with Label Noise

## 基本信息与阅读状态

- **作者**：Xiaoli Peng, Ping Wang, Shuyin Xia, Cheng Wang, Weiqi Chen。
- **出处**：*Information Sciences* 611 (2022), 504–521；DOI [10.1016/j.ins.2022.08.066](https://doi.org/10.1016/j.ins.2022.08.066)。[出版社摘要页](https://www.sciencedirect.com/science/article/pii/S0020025522009641)。
- **阅读状态**：已核出版社题录与摘要；未取得可核验全文。**相关度**：中（粒球抗标签噪声与分类；非神经图对比学习）。

## 问题与方法

论文认为传统邻域粗糙集定义在标签噪声下过于严格，且按单样本学习的分类器难兼顾属性约简与抗噪分类。VPGB（variable-parameter granular-ball model）分两路：属性约简用同尺寸、固定中心/半径的 FGB 表达邻域；分类阶段用不同尺度、自适应中心与半径的 AGB，再按测试样本到 AGB 的距离分类。它隐式用球聚合减轻单点错误标签影响，不是深度学习或对比学习方法。

摘要未提供可安全转录的损失公式；目标变量是属性约简及分类结果。FGB 与 AGB 用途不同，不能合并概括成统一的“粒球置信度权重”。

## 实验、限制与迁移

出版社摘要报告公开数据集上的属性约简和 noisy-label classification 比较，并称性能/鲁棒性有改善；具体数据集、噪声比例、随机种子与结果未核全文。该方法依赖邻域与球尺度设置，且与神经网络端到端表示空间不同。迁移启示仅是粗粒度聚合或能减弱单样本标签的影响；图节点上若球内混入异类节点，硬化球标签可能扩大错误。
