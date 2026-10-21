# MGNR

## 元数据

Jiang Xie, Xuexin Xiang, Shuyin Xia, Lian Jiang, Guoyin Wang, Xinbo Gao. “MGNR: A Multi-Granularity Neighbor Relationship and Its Application in KNN Classification and Clustering Methods.” IEEE TPAMI 46(12), 2024, pp. 7956–7972. DOI: 10.1109/TPAMI.2024.3400281. [PubMed 题录/摘要](https://pubmed.ncbi.nlm.nih.gov/38739513/)；[作者代码](https://github.com/xjnine/MGNR)。

## 问题与任务

为 KNN 分类和密度聚类构造自适应多粒度邻居关系，避免对全局固定 K、距离阈值或密度参数的依赖。它是少数明确包含分类实验的粒球邻域论文，但不是对比学习、图神经网络或标签噪声研究。

## 创新与流程

先自顶向下划分粒球，使用 distribution measure（球内点到中心的平均距离）控制球细化，再按球重叠形成粒球组。分类端为每个球组估计多粒度 K 值：根据测试点最近训练点所在球组取得该组 K，再按这些近邻多数票分类。聚类端以球中心构造多粒度自然邻居，通过样本分布组得到 NARD（Natural-Adaptive Relative Density），再替换 DPeak、DBSCAN、DADC、HCDC 等方法中的局部密度/邻域设置。

## 目标及结构更新

球通过 DM 加权划分逐轮细化直至不再增加；相邻重叠球合并成球组；自然邻居迭代决定各组邻接尺度。MGKNN 的最终预测仍是监督 KNN，多粒度关系负责自适应选择 K。NARD 是聚类前端关系/密度估计，不是端到端可学习对比损失。

## 实验

论文评估两条应用线：真实分类数据上的 MGKNN，报告 ACC 与运行时间（数据集规模约 60 至 70,000）；合成及真实数据上的 NARD，并比较多种密度聚类算法的 ACC/NMI、运行时间。作者报告大型数据上运行效率明显改善，分类多数数据集表现较好；同时明确指出 crowdsourced 等数据的 NMI 仍不理想，某些情形的精度可能退化，需权衡原方法效果和提速。细节见原文第 11–15 页表 IV–VIII、实验结论及局限段。

## 对本项目的启发

可作为“粒球局部结构如何提供样本邻域/多尺度范围”的算法先例，也提供分类迁移证据，但分类器使用带标签训练集做 KNN，任务协议与图上的对比表征学习不同。可考虑把粒球组内多粒度邻域转为软关系，再研究其在节点表示训练中的贡献。

## 风险与边界

球划分/合并规则仍会受异常值、密度不均和高维距离影响；固定邻域不合适的问题可能只是转为 DM 和球组构造质量问题。论文中监督分类任务不是鲁棒标签学习，不能据其分类 ACC 推断抗噪声分类能力。

## 阅读深度与证据

已阅读本地文献库中提取的 TPAMI 原文全文，核对第 III 节方法、MGKNN/NARD 流程、第 IV 节实验表和作者讨论的失败边界；元数据由 PubMed 与 DOI 核实。[PubMed](https://pubmed.ncbi.nlm.nih.gov/38739513/)；[IEEE DOI](https://doi.org/10.1109/TPAMI.2024.3400281)；[作者实现](https://github.com/xjnine/MGNR)。
