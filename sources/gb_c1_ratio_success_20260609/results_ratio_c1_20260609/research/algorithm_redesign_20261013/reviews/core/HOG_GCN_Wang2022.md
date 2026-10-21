# HOG-GCN：GBGCL的Wang等人[64]

核读：2026-10-14。Tao Wang、Di Jin、Rui Wang、Dongxiao He、Yuxiao Huang；Powerful Graph Convolutional Networks with Adaptive Propagation Mechanism for Homophily and Heterophily；AAAI 2022，4210–4218。

来源：[官方全文](https://ojs.aaai.org/index.php/AAAI/article/download/20340/20099)；本地PDF/提取文本../../sources/HOG_GCN_AAAI2022*，哈希见../../sources/OFA_reference_sources.json。

## 支持什么

Table2的Texas：MLP81.08±4.83、GCN54.05±4.36、GAT57.30±3.38、HOG-GCN85.17±4.40。它支持部分异配任务中原属性MLP强于传统GCN/GAT，也显示合理结构模型可以进一步改善。论文用十次随机划分、48/32/20；不可与本项目直接排名。

HOG-GCN从监督MLP预测B构造属性关系BBᵀ，再结合可学习标签传播和图卷积。它并非仅属性方法，也未证明噪声标签下这些关系可靠。

## 对本项目的判断

可作为保留属性分类器的动机；不能写“异配图上MLP普遍优于GNN”。若借其关系矩阵，需重新处理噪声与预测反馈，并区分标签无关几何和监督关系；不能沿用无标签权重的独立噪声证明。

状态：方法、Table2与协议已核；未做官方复现。
