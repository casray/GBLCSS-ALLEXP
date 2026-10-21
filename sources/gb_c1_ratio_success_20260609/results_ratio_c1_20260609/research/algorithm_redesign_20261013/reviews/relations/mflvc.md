# MFLVC

## 元数据

Jie Xu, Huayi Tang, Yazhou Ren, Liang Peng, Xiaofeng Zhu, Lifang He. “Multi-Level Feature Learning for Contrastive Multi-View Clustering.” CVPR 2022, pp. 16051–16060. [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.html)；[作者代码库](https://github.com/SubmissionsIn/MFLVC)。

## 问题与任务

无监督多视图聚类。作者指出，把重构和跨视图一致性都压在同一特征空间，会让视图私有信息与跨视图共同语义相冲突。它不是图节点分类，也没有研究训练标签污染。

## 创新与流程

将表示分成三个层级：低层特征承担输入重构；高层特征承担跨视图对比一致性；语义标签/簇分配承担聚类级一致性。不同目标在不同表示层优化，避免直接融合时无关的视图私有信息压过共性语义。框架称为 fusion-free，核心不是再造一个统一表示后施加所有损失，而是分层使用各自信号。

## 目标及结构更新

联合目标由三部分组成：低层表示重构、高层表示跨视图对比、簇分配语义对比。高层特征和标签预测器分别由共享映射从低层视图表示得到；簇分配也从低层表示预测，减少特征对比头与聚类头互相牵制。流程先优化重构和多层一致性，再匹配各视图的聚类编号并微调标签预测器。论文给出完整损失和算法 1，本文不把伪簇分配称为真实标签监督。

## 实验

五个基准数据集为 MNIST-USPS、BDGP、CCV、Fashion 和 Caltech；Caltech 还按 2 至 5 个视图分别评估。主比较表覆盖前四个数据集，指标为 ACC、NMI、PUR；另有损失组件消融、收敛和参数敏感性分析。论文报告 MFLVC 在这些无监督聚类设置中表现有竞争力，但表格未提供能支持普遍显著性结论的重复运行误差统计。

## 对本项目的启发

可借鉴“把信息保留目标与语义关系目标分层”的设计：图/属性重构、局部结构保持、簇或标签关系不必挤在同一表示上。但其多视图对应关系与聚类伪标签假设，不能直接等同于单图上的节点噪声标签分类。若迁移，必须说明关系从何产生、哪些分支接收硬标签、以及噪声标签如何影响簇级对比。

## 风险与边界

论文只支持无监督多视图聚类场景下的分层表示学习；不提供粒球、图结构鲁棒性或监督标签噪声的直接证据。它是早期范式参考，不是当前“关系可信度/抗噪节点分类”的直接基线。

## 阅读深度与证据

已读作者公开稿全文的第 3 节方法、算法 1、第 4 节主结果与第 5 节消融/参数分析；发表信息据 CVF 官方论文页核对。全文来源为 [arXiv HTML](https://arxiv.org/html/2106.11193)，发表版本入口为 [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Multi-Level_Feature_Learning_for_Contrastive_Multi-View_Clustering_CVPR_2022_paper.html)，实现见 [作者代码库](https://github.com/SubmissionsIn/MFLVC)。证据级别：作者稿全文核读，CVPR 发表版本 PDF 未作逐页视觉比对。
