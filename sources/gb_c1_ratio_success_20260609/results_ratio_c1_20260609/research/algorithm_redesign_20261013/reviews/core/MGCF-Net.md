# MGCF-Net：A multi-view granular-ball contrastive fusion network for spatial transcriptomics clustering

## 文献信息与阅读层级
作者 Li Zhu, Haiyao Dong, Yuji Zhang, Yong Zhang, Xinyi Huang, Huishan Wang。Frontiers in Physiology 17:1967219，DOI 10.3389/fphys.2026.1967219，2026-10-07；开放获取 CC-BY。**阅读层级：全文级**，已读开放全文方法、实验、消融与讨论。[全文](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1967219/full)。

## 任务与方法
任务为空间转录组 spatial domain clustering。构建 spatial graph（物理邻近）、feature graph（转录组相似）和融合/多视图表示；GCN 编码多视图。MVRR 目标保留跨视图 spot 对应并抑制不同 spot 的虚假相似；spot 自适应融合；PhCR 保持物理空间邻近结构；GBCL 在各视图分别以 K-means 形成球，按成员交叠建立跨视图球关联，再进行球级对比，降低实例级对比假负例。球成员/关联阈值为离散构造，在反向传播中固定；中心可学习。整体由 MVRR、PhCR、GBCL 组成。此方法不需要生成式 decoder。

## 数据、结果与负结果
DLPFC 八个切片：151507、151508、151669–151674。方法只使用簇数 K（来自人工注释）；spot 标签用于评测 ARI/NMI，不作为节点噪声训练标签。没有注入标签噪声实验。论文报告平均 ARI 约 0.609，相比最佳基线 stMMR 约 0.56；NMI 约 0.653，接近比较方法而非全面领先。消融中去掉 GBCL 后 ARI 0.5884、NMI 0.6519，说明 GBCL 有贡献但增量较小；去掉 PhCR 的 ARI 损失更明显。运行时间约每片 10^3 秒，为对比方法中较慢者。作者提出可微粒球构造、扩展性、3D/multi-omics 为后续方向。

## 与图噪声节点分类的关系
这篇是最直接的“多视图图表示 + 粒球级对比”先例之一。它支持局部结构单元可以减少 instance-level false negatives，但**不能**支持对噪声标签有效：人工注释只确定聚类数并用于评测，未作为噪声标签进行学习。新方案应清楚区别于它的多视图跨球成员重叠目标，例如研究噪声标签节点分类下球关系的置信度与退化边界；不可宣称粒球级图对比本身是新颖。

## 证据追溯与限制
方法与实验以 Frontiers 开放全文为主。需直接查阅全文公式/表格时使用网页原文；本笔记只记录已核对数值。数据集聚类标签不等于标签噪声设置，两者不可混称。
