# SAGHL：Structure-aware Granular-ball Hypergraph Learning

## 文献信息与阅读层级
Jinyuan Ni, Shuyin Xia, Guan Wang, Meng Yang, Long Chen, Yi Liu, Yi Wang。Engineering Applications of Artificial Intelligence, 181(Part 2), 115428，2026-10-01，DOI 10.1016/j.engappai.2026.115428。**阅读层级：官方出版商长摘要/引言预览级**；ScienceDirect 记录可检索长预览，但打开正文返回 403，尚无全文核读。作者公开代码入口：[SAGHL代码](https://anonymous.4open.science/r/SAGHL-4266)；本轮未能读取该代码页面内容。[ScienceDirect官方记录与长预览](https://www.sciencedirect.com/science/article/pii/S0952197626017124)；[DOI](https://doi.org/10.1016/j.engappai.2026.115428)。

## 当前能由预览支持的方法与任务主张
官方摘要明确宣称对 benchmark citation/social media datasets 的实验表明其有鲁棒性并能缓解 label noise；因此它不能被描述成只研究干净标签分类。当前官方预览还支持以下结构：面向**homogeneous, static, undirected graphs**；按 betweenness centrality 选 hub，利用 modularity-density 引导自适应层级 granular-ball 分解，形成多尺度超边；三并行残差通道融合局部/全局尺度特征，配合 gated attention 和 skip connection；对比采样由超边结构引导，动态温度随超边尺度和密度调整；联合优化分类与对比目标，在有限监督下提升泛化。后续异构/时序图是扩展方向，故它自身不是 heterogeneous graph 方法。

## 与当前噪声节点分类提案的重叠
此文已经同时占据“粒球分解 + 图/超图结构推理 + 对比采样 + 声称缓解标签噪声”的组合，属于高度相关工作。任何新方案都不能宣称首次把粒球结构关系用于带噪图分类或首次融合球结构与对比目标。可区分方向需精确落在机制上，例如不依赖有标签分类损失的自监督节点 GCL、把球关系当软置信先验而非强制正负、或 graph-free 训练；最终新颖性判断必须以正文为准。

## 当前证据边界与待核实
出版商长预览还概述平均约 +3 个百分点 Accuracy、+5 个百分点 Macro-F1，大数据集最高约 +7 个百分点 Accuracy；这些是作者/出版商摘要性结果陈述，全文表格尚未核对，暂不能当作已复核数值。尚未核实具体数据集名称、标签噪声是合成还是自然噪声、噪声率/注入协议、公式、基线、消融、seed/std。取得正文后首要核对其 label-noise protocol、球构建是否用标签、动态温度公式、弱监督比例和表格误差。

