# RAV

## 元数据

Shuangmei Peng, Zhe Chen, Tianyang Xu, Xiao-Jun Wu. “Relationship Alignment for View-aware Multi-view Clustering.” ICLR 2026. [ICLR 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee63888edc6155083c87a8df83581047-Abstract-Conference.html)；[官方PDF](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee63888edc6155083c87a8df83581047-Paper-Conference.pdf)；[作者代码](https://github.com/chenzhe207/RAV)。

## 问题与任务

无监督多视图聚类，关注跨视图样本关系不一致和不适合强制对齐的视图对。Cora 在该文被整理为四个特征视图（内容、入向、出向、引用特征），仍是聚类评估，不是使用节点类别标签训练的分类任务。论文未设计节点标签翻转实验；结论部分将噪声数据和不完整视图列为未来方向。

## 创新与算法流程

各视图编码后构造视图内样本关系矩阵，并形成全局关系矩阵；用关系对齐约束促进全局与局部关系一致。再根据视图关系差异自适应调整视图对的簇标签对比强度：关系接近的视图对齐更强，差异大的视图减少对齐。由各视图簇分配的融合结果得到最终聚类预测。

## 目标及结构更新

方法包括表示重构、关系矩阵对齐和簇分配级对比/一致性目标。视图权重依视图关系距离动态计算，降低差异视图被硬拉齐的程度。它动态调的是聚类视图之间的对齐权重；不能简化为“自动识别并去除噪声视图”，因为实验并未以注入观测噪声作为主设置。

## 实验

论文在 10 个多视图数据集上评估，包含 Cora 四视图设置；指标包括 ACC、NMI、PUR。采用分阶段训练并报告主结果、消融和视图权重分析。论文展示不同数据上的增益并非每个数据都同样明显（如 Digit-Product 未显示同等优势），不应把“动态加权”外推为稳定抗噪结论。具体实验见官方 PDF 实验与消融部分（约第 6–8 页）；论文末尾把 noisy data / incomplete views 留作未来研究。

## 对本项目的启发

关系冲突本身可以作为“不要强制对齐”的信号。粒球可提供局部结构关系，但迁移到单图节点任务时应区分视图（例如结构视图与特征视图）和不同节点关系；用伪标签验证关系质量只能作为诊断，不可当训练标签。

## 风险与边界

RAV 是多视图聚类，Cora 的数据形式是图派生视图但目标依旧是聚类。无监督聚类指标使用数据集真类标签事后评估，不代表真标签进入训练。它没有证明 GRACE 对节点标签噪声失败，也未证明关系加权可处理标注错误。

## 阅读深度与证据

核对官方论文全文的方法、实验和结论：第 3 节方法、第 4 节实验及结论（未来工作）；元数据与标题据 ICLR proceedings。证据：[ICLR 官方论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee63888edc6155083c87a8df83581047-Abstract-Conference.html)；[官方PDF](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ee63888edc6155083c87a8df83581047-Paper-Conference.pdf)。
