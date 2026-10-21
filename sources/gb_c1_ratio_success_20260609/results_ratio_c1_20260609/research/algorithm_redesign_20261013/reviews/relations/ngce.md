# NGCE

## 元数据

Yazhou Ren, Junlong Ke, Zichen Wen, Tianyi Wu, Yang Yang, Xiaorong Pu, Lifang He. “Multi-View Graph Clustering via Node-Guided Contrastive Encoding.” ICML 2025, PMLR 267, pp. 51421–51435. [PMLR 官方论文页与摘要](https://proceedings.mlr.press/v267/ren25a.html)；[官方PDF入口](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ren25a/ren25a.pdf)；[作者代码](https://github.com/Rirayh/NGCE)。

## 问题与任务

多视图图聚类（MVGC）。作者关注节点特征与图结构怎样一起编码，以及 GNN 过滤器在同配/异配图上的适配问题。目标是图上的无监督聚类，不是节点标签分类；官方摘要没有报告训练标签翻转实验。

## 创新与流程

NGCE 全称 Node-Guided Contrastive Encoding。摘要确认其以节点特征引导嵌入生成，设计与 GNN 图滤波兼容的表示过程，融合适用于同配与异配的图信息，并加强多视图之间的对比学习。实现仓库可用于追踪代码细节，但本卡不补写无法从可读官方正文独立核对的具体算子/公式。

## 目标及结构更新

高层目标是以节点特征和多视图图结构学习图表示，并进行跨视图对比及聚类。官方摘要未暴露足够正文信息核验聚类头、结构更新节奏和具体对比正负样本定义，因此不作推断。

## 实验

ICML 官方摘要确认六个同配与异配多视图图基准，并称取得稳健结果；本次可访问的一手来源未呈现完整实验表，故不复述具体数据集、指标和数字。摘要也未证明标签噪声鲁棒性。

## 对本项目的启发

它提示图对比的结构偏差不只有邻居同类性：异配图中结构邻居不一定同语义。可把粒球当作局部结构单元时，需要同时看属性/语义证据，不应把球间连接直接当正样本。可用于有图节点研究，但监督目标仍需从聚类改造成分类任务后重新验证。

## 风险与边界

同配/异配鲁棒表示不等于标签噪声鲁棒性；多视图图聚类结果不能直接佐证单图节点分类。原文机制的具体细节需读完整 PDF 后再引用。

## 阅读深度与证据

已核对 PMLR 官方元数据/摘要和 [作者实现仓库](https://github.com/Rirayh/NGCE)。本轮检查本地抽取文本，没有找到 NGCE 正文；PMLR 页给出的 raw-GitHub PDF 端点返回内部错误，OpenReview 备用全文入口则进入浏览器验证页。因此仍按摘要/题录级记录，不从摘要或代码补写论文正文结论。可核实的信息仅包括六个同配/异配多视图图聚类基准及节点特征引导、兼容 GNN filtering、跨视图对比的总体目标。来源：[PMLR 官方论文页](https://proceedings.mlr.press/v267/ren25a.html)；[OpenReview 全文入口](https://openreview.net/pdf/bfbbe589ca3650e32fccc9566946441bbd4cb1a1.pdf)（本轮无法读取）。
