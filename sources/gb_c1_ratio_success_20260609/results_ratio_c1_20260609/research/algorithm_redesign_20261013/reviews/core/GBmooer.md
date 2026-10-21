# GBmooer：Granular-ball Text Label Smoothing with Hypersphere Contrastive Learning

## 文献信息与阅读层级
- 作者：Jintao Zhai, Zeli Wang, Shuyin Xia, Longlong Lin。
- SSRN 6022343，DOI 10.2139/ssrn.6022343；2026-01-05 发布，页面注明尚未同行评审。
- 阅读层级：全文级（本地提取：文献文件\_literature_review_work\extracted_text\粒球+对比学习\gbMOOER.txt）。
- 原文：[SSRN记录与全文入口](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6022343)。

## 任务与方法
这是有标签文本分类的噪声鲁棒学习。数据集包括 TREC、AGNEWS、IMDB、Chnsenticorp；以 BERT 为主，另有 LSTM、TextCNN、GRU 迁移实验。噪声设定含 symmetric、asymmetric、instance-dependent，具体噪声比例依数据集/实验而异。方法结合归一化角度/超球表征的对比预训练、自适应迭代粒化、局部软标签和分类/对称 KL 一致性目标。球内噪声标签分布生成软标签，因此目标仍依赖观测标签；并非无监督真标签辨识。

## 证据与迁移
论文比较不同噪声模式/比例和骨干并报告改善；本笔记未重抄全部数值，不引用精确百分点。可迁移的候选想法是局部角度几何质量与动态粒度用于置信度诊断。不可直接迁移的是监督局部标签分布：在 label-free 图对比预训练里会改变学习信号；若错标签局部成簇，软化可能扩散系统性错误。图拓扑、跨视图增强和节点关系都未验证。

## 风险与待核实
标签平滑不是独立标签纠错证据；类不平衡、稀有类及错误标签成簇的影响需另测。各数据集精确噪声率、主表数字和方差尚未逐格核录。引用时标注 SSRN 预印本状态。来源优先 SSRN 原文及公式/实验表；本地提取只作检索。
