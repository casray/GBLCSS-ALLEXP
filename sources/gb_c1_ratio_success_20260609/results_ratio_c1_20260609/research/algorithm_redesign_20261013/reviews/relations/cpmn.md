# CPMN

## 元数据

Yiming Wang, Qun Li, Dongxia Chang, Jie Wen, Hua Dai, Fu Xiao, Yao Zhao. “Learning from Disjoint Views: A Contrastive Prototype Matching Network for Fully Incomplete Multi-View Clustering.” NeurIPS 2025. [NeurIPS 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a7a22152cd21f0ca3c0f8139bb32905-Abstract-Conference.html)；[官方PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/1a7a22152cd21f0ca3c0f8139bb32905-Paper-Conference.pdf)。

## 问题与任务

完全不完整多视图聚类（FIMC）：不同视图没有任何样本对应关系。它与仅仅随机缺少某些视图不同，也不是图节点任务或标签噪声方法。

## 创新与流程

先用不依赖对应关系的跨视图图对比学习，以互近邻（MNN）关系形成初始结构；各视图分别做 K-means 得到原型；将原型间相似度写成二部匹配，使用线性分配求跨视图簇对应；最后依据原型映射进行原型级对比学习，并以软聚类分配和 KL 聚类目标精炼簇表示。

## 目标及结构更新

阶段一结合重构与视图内/跨视图图对比；阶段二在独立初始化的原型间做显式匹配；阶段三把匹配后的原型关系用于样本到原型对比，并联合簇分配更新表示。依赖的是原型/结构对齐，不是粒球成员重叠，也没有真实标签监督。

## 实验

Caltech-101、CUB、NoisyMNIST、MNIST-USPS、YouTubeFaces 五组数据。评测完全视图不对齐和完全缺失等设置，也报告部分不对齐/部分缺失及消融，聚类指标为 ACC、NMI、ARI。NoisyMNIST 是数据集名称；论文主变量是视图对应缺失/不完整比例，不能据名称把实验说成了标签噪声测试。

## 对本项目的启发

当跨视图或跨图没有逐节点配对，局部集合/原型匹配可作为语义级桥梁；粒球中心和成员分布可构造候选球级匹配。但需比较普通 K-means 原型与粒球匹配，并处理跨图簇编号置换、不同簇数与原型错误匹配。

## 风险与边界

初始图关系和 K-means 原型都可能错误，后续匹配可能扩大确认偏差。论文测试无对应视图聚类，不证明图上无监督节点分类或节点标签噪声场景可直接使用。

## 阅读深度与证据

已读 NeurIPS 原文第 3 节对应关系无关对比、原型图匹配、原型对比，以及第 4 节五数据集实验设置、缺失/未对齐协议、主结果和消融。实验明确随机抽取样本删除视图或移除其跨视图对应（第 7–8 页）。[官方PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/1a7a22152cd21f0ca3c0f8139bb32905-Paper-Conference.pdf)。
