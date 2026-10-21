# MGBCC：Multi-view Granular-ball Contrastive Clustering

## 文献信息与阅读层级
Peng Su, Shudong Huang, Weihong Ma, Deng Xiong, Jiancheng Lv。AAAI-25 39(19):20637–20645，DOI 10.1609/aaai.v39i19.34274，2025-04-11。阅读层级：官方全文级（AAAI PDF 9 页）。[AAAI记录](https://ojs.aaai.org/index.php/AAAI/article/view/34274)；[官方PDF](https://ojs.aaai.org/index.php/AAAI/article/download/34274/36429)；[代码](https://github.com/Duo-laimi/mgbcc)。

## 任务、公式与实验
多视图无监督聚类，不是节点分类或图对比学习。数据集：BBCSport、Caltech101-20、Cora、Scene-15、MNIST-USPS、ALOI-100、NoisyMNIST。此处 Cora 是多视图聚类基准，不等于标准节点级图 GCL。每个视图通过自编码器编码，再进行 K-means 粒化；球中心、半径可微。官方 PDF Eq. (2)–(14)按几何重叠和跨视图成员交集构造关联球对，中心级余弦损失将关联球作为正对、不关联球作负对；总目标 Eq. (14) 为 L=L_con+λL_rec。跨视图关联按共同成员比例 |I_both|/min(t_i,t_j) 与阈值 τ 比较，并最终变成离散正负掩码，不是连续关系置信度。作者报告 τ=0.1 常用、p=2 效果较好，球过大时关联质量下降。Table 2 主结果给点估计，未见该主表误差条/标准差。

## 迁移判断与限制
可借鉴“让局部结构单元参与跨视图/增强表示关系”的问题设定，可将球关系作为候选软结构信号研究。不能把多视图聚类球掩码等同图邻接语义、标签置信度或噪声检测。硬阈值球关联若出错会传递错误监督；异配图、节点级噪声标签鲁棒性均未验证。论文讨论不完整多视图数据作为后续方向。

## 追溯
公式见官方 PDF Eq. (2)–(14)，数据见 Table 1，主结果见 Table 2。未逐格复抄数值；具体效能引用需回查表格并核对种子/方差。论文页：[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34274)。

