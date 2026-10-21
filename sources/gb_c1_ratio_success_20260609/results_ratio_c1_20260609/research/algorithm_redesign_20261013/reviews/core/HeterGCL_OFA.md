# HeterGCL 与 Original Feature Analysis

核读：2026-10-14；方法、理论来源与分类消融。作者 Chenhao Wang、Yong Liu、Yan Yang、Wei Li；IJCAI 2024，2397–2405，DOI 10.24963/ijcai.2024/265。

来源：[官方全文](https://www.ijcai.org/proceedings/2024/0265.pdf)；本地PDF/提取文本在../../sources/HeterGCL_IJCAI2024*，哈希见../../sources/OFA_reference_sources.json。

## 机制与证据

这是GBGCL的[42]，OFA为其模块。共享MLP编码原X与特征掩蔽X1（式4/8）；式9包含表示一致性与各视图Gram去相关，无节点负样本。另有ANA结构对比及潜在软聚类关系，整个HeterGCL仍使用图。

Table2删除OFA：Cornell75.5→66.1、Texas74.7→69.8、Wisconsin75.6→65.6；Actor收益弱。论文采用10/10/80、10次运行、无人工噪声标签实验。均值差不能代替配对检验。直接输入属性不等于验证了噪声鲁棒性。

## 对本项目的判断

契合：属性编码与图关系监督分开；保留节点自身属性。不同：GB-LCSS冻结图GRACE后投票，而OFA直接训练属性表示。优先将OFA设为独立、标签无关的分球表示候选，固定下游机制检验。不能把组合现有OFA与粒球本身称为创新；需有噪声条件下的机制增量。

状态：原文方法和消融已核；未做HeterGCL官方代码复现，未运行本项目OFA消融。
