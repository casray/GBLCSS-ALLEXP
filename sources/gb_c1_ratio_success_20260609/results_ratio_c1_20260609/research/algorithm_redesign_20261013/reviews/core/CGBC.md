# CGBC：Robust Smart Contract Vulnerability Detection via Contrastive Learning-Enhanced Granular-ball Training

## 文献信息与阅读层级
- 作者：Zeli Wang, Qingxuan Yang, Shuyin Xia, Yueming Wu, Bo Liu, Longlong Lin。缩写 CGBC 指 Contrastive learning-enhanced Granular-Ball smart Contracts training。
- arXiv:2603.27734，2026-03-29 提交的预印本；同行评审发表状态未核实。
- 阅读层级：全文级（本地原文提取：文献文件\_literature_review_work\extracted_text\粒球+对比学习\CGBC.txt；另参照文献图）。
- 原文：[arXiv](https://arxiv.org/abs/2603.27734)。

## 任务与方法
面向静态分析工具产生噪声标签的智能合约漏洞检测。流程是语义一致的对比预训练，继而以粒球监督层和 SCE 损失微调。属于有标签漏洞分类，不是图节点分类或无标签节点级 GCL。粒球纯度、中心标签依赖带噪训练标签；式 (11)–(13)表达同标签球中心紧凑、异标签球中心按 margin 分离，而非不区分类别地任意球球分离。摘要和正文的 intra/inter loss 命名可能不一致，故按公式行为解释，不沿用冲突命名。

## 任务数据与证据
本地全文记录预训练数据 Ethereum-SC Large，微调约 40k 合约；评测 ReentrancyBenchmark、SolidiFiBenchmark，涵盖 Reentrancy、Timestamp Dependency、Integer Overflow 等漏洞。本地研究图记录结果采用 5 次运行。本笔记未重抄表格数值；引用效能数字前应回查原 PDF 对应表、协议和方差。

## 对图对比学习噪声节点分类的迁移
可借鉴的仅是“先学表征，再谨慎引入局部结构单元”的研究思路。标签纯度、多数类球标签和异类球中心分离均为监督机制；直接放进 label-free GRACE 会引入标签信息、且错误标签能污染球纯度并导致过强 prototype pulling。该论文不支持其方法能提升无监督图对比或识别节点噪声。

## 局限与待核实
- 任务和模态迁移跨度大；没有图节点或 GCL 证据。
- 具体结果表、数据拆分、运行方差尚未逐项复核；代码仓库与同行评审状态未核实。
- 来源追溯：优先原始 arXiv 论文；方法重点复核 Eq. (11)–(13) 及数据/结果表。本地提取文本只辅助检索，公式以 PDF 为准。

