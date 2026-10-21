# GCFAggMVC

## 元数据

Weiqing Yan, Yuanyang Zhang, Chenlei Lv, Chang Tang, Guanghui Yue, Liang Liao, Weisi Lin. “GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering.” CVPR 2023, pp. 19863–19872. DOI: 10.1109/CVPR52729.2023.01902. [CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.html)；[CVF 官方 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.pdf)；[作者代码](https://github.com/zhangyuanyang21/2023-CVPR-GCFAggMVC)。

## 问题与任务

完整及不完整多视图数据上的无监督聚类。作者关注仅逐视图融合忽略跨样本的全局关系，以及常规对比把不同样本一概推开可能破坏同簇关系。方法中的关系图由多视图样本特征学习得到；它不是图节点分类输入，也没有噪声标签协议。

## 创新与机制

各视图先由自编码器获得重构表示。Global and Cross-view Feature Aggregation 模块使用 Transformer 式跨视图特征变换学习样本间全局关系矩阵，并以高关系样本的跨样本、跨视图特征聚合增强共识表示。Structure-guided Contrastive Learning 将同一实例的共识表示与视图表示作为正关系，再以学习到的结构关系指导其他样本对的对比贡献，避免不区分关系地把所有不同实例当作负对。最后在共识表示上用 K-means 聚类。

## 实验证据

论文报告 13 个公开多视图数据集，指标为 ACC、NMI、PUR；比较完整和不完整聚类设置，并有消融与参数分析。不完整设置对若干基准改变视图缺失率。结果说明结构引导聚合在作者所设 MVC/IMVC 聚类协议中有效，但不等于验证了输入特征噪声或训练标签噪声下的鲁棒分类。

## 对本项目的启发与迁移边界

可参考将粒球的局部成员统计作为关系图/全局结构的压缩输入，或用关系权重调节不同球之间的对比贡献。要验证球表示相对样本级相似图是否保留有用语义、降低了关系估计错误或计算成本；不能将“有关系”直接视作类标签。

GCFAggMVC 的共识结构是从表示中学习出的关系，不是外部可靠拓扑；伪关系可能自我强化。其完整/不完整多视图聚类结果不能替代带噪节点分类实验。论文没有证明粒球本身能提高关系可靠性。

## 阅读深度与证据

已读作者公开稿全文的摘要、方法（含聚合和结构引导对比）、13 数据集实验设置、缺失视图实验与消融；CVF 官方页面用于核对正式题录，CVF PDF 端点本轮不能正常打开，故没有声称逐页核验发表版。[arXiv 作者稿 HTML](https://arxiv.org/html/2305.06799)；[CVF 官方论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Yan_GCFAgg_Global_and_Cross-View_Feature_Aggregation_for_Multi-View_Clustering_CVPR_2023_paper.html)；[作者代码](https://github.com/zhangyuanyang21/2023-CVPR-GCFAggMVC)。阅读级别：作者稿全文方法与实验核读，发表版元数据核对。
