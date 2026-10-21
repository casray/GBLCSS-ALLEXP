# Literature and Research Map

## 1. Current Research Goal

本项目目标是在节点级图对比学习框架中探索 Granular Ball Computing 的有效作用方式，重点关注 GRACE 风格方法、增强一致性、局部语义关系、置信度、噪声鲁棒性和多粒度结构建模。

### 1.1 Current Working Thesis

当前最稳妥的研究主线不是把粒球当作新的硬标签，而是把粒球作为节点级 GCL 的辅助结构信号：

- `GRACE` 保持为 baseline 与评估协议中心。
- 粒球优先进入 `augmentation control`、`weak-view relation teacher`、`soft ball-to-ball relation`、`node-to-ball confidence`、`graph denoising / local structure prior`。
- 任何使用 label purity、class-wise boundary、supervised coarsening 或 supervised sampling 的文献，只能提供机制启发或诊断指标，不能直接进入无监督预训练 loss。
- 论文 claim 应优先围绕“粒球提供鲁棒局部结构先验 / 置信度加权 / 结构感知增强”，而不是“same-ball 节点天然是正样本”。

### 1.2 Evidence Level Used in This Document

- `Direct evidence`：文献方法与本项目设置接近，例如节点级 GCL、无标签图构建、无监督 ball relation、GNN 表征学习。
- `Transferable evidence`：文献任务不同，但粒球接口可转化为 soft relation、confidence、denoising、augmentation control。
- `Diagnostic evidence`：文献依赖标签、监督 purity 或任务特定目标，只能用于离线诊断、可视化或风险判断。
- `Warning evidence`：文献提示某类迁移有风险，例如 hard membership、固定粒度、过强 prototype、改变 evaluation。

后续实验设计应优先使用 `Direct evidence` 和 `Transferable evidence`；`Diagnostic evidence` 不得被写成训练依据；`Warning evidence` 必须进入失败标准和 ablation 设计。

当前文档整理已读取的四类材料：

- 对比学习文献：`文献文件/对比学习/CLSA.pdf`、`文献文件/对比学习/GRACE.pdf`、`文献文件/对比学习/Simplified_Graph_Contrastive_Learning_Model_Without_Augmentation.pdf`
- 粒球 + 对比学习文献：`文献文件/粒球+对比学习/Multi-view Granular-ball Contrastive Clustering.pdf`、`文献文件/粒球+对比学习/Views Attention Fusion of Granular-ball Fuzzy Representations Split for Improved Multi-view Clustering(1).pdf`、`C:/Users/Admin/Downloads/2603.27734v1 (1).pdf`、`C:/Users/Admin/Downloads/ssrn-6022343 (1).pdf`
- 粒球文献：`文献文件/粒球` 下 7 篇基础粒球 / 粒球生成 / 粒球聚类 / 粒椭球论文
- 深度学习 + 粒球文献：`文献文件/深度学习+粒球` 下 13 篇粒球进入 GNN、GAE、RVFL、图像增强、开放意图识别、GNN 解释、采样/粗化/拓扑构建等流程的论文。GAdaBoost、EvoSampling、MGNR 属于非神经但相关的粒球 + 学习方法，按实际方法内容记录，不强行套 GCL 模板。

### 1.3 Fast Decision Snapshot for the Next Experiment

| 问题 | 当前建议 | 依据 | 主要风险 |
|---|---|---|---|
| 第一优先级做什么 | weak-view / boundary-distance / confidence-weighted soft relation | GRACE、CLSA、GBCM-GCN、AW-GBGAE、GFSAF | relation 不稳定或权重太强 |
| 粒球生成空间 | 先比较 raw feature、encoder output、weak-view encoder output；projector output 仅做诊断 | GRACE、SPARROW、GBG++、adaptive GB generation | 训练早期 embedding 波动 |
| 是否使用 same-ball positive | 只能 soft / confidence-weighted，不能 hard binary label | MGBCC、GFSAF、MOGB、GBRIP | 错误正样本、过平滑 |
| 是否使用 negative filtering | 不能单独作为主线，可作为 denoising ablation | GFSAF、GAdaBoost、EvoSampling | 信号太弱，只减少负样本 |
| 是否使用 graph coarsening | 暂缓做主线，只做无标签结构诊断或效率实验 | SGBGC、PGBSF | label leakage、节点级协议改变 |
| 是否使用 prototype / center pulling | 暂缓强拉近；只用 center distance 做 confidence | GB-RVFL、GBRIP、MOGB | prototype pulling 过强 |
| 成功判断 | 跨 seed、跨数据集、mean/std 可比较，最好有噪声场景收益 | AGENTS.md、GRACE baseline | 单数据集小涨点不可写 claim |

## 2. High Level Literature Map

| 类别 | 核心关注问题 | 常见方法结构 | 与本项目的关系 | 对下一步实验的启发 | 局限性 |
|---|---|---|---|---|---|
| 粒球和对比学习结合文献 | 用粒球缓解实例级 false negative、簇级过粗、局部结构缺失、高维表示聚球不稳和噪声标签鲁棒训练等问题 | 多视图/代码/文本编码器生成表示，构造粒球，定义粒球间正负关系、fuzzy relation、majority-vote label 或局部标签分布，再做对比损失、局部平滑或鲁棒训练 | 直接提供“粒球作为中间粒度关系单元”的方法参考；CGBC 和 GBmooer 都提示 CL warm-up 可改善后续粒球质量 | 优先尝试 soft ball relation、weak relation teacher、confidence weighting、contrastive warm-up 后的粒球稳定性诊断、hypersphere / angular ball diagnostic，而不是硬 same-ball positive | 现有文献主要是多视图聚类、有监督智能合约检测或文本噪声分类，不是节点级 GRACE；粒球正负关系、监督 purity、majority-vote label 和 label-distribution smoothing 迁移到 GRACE 需谨慎 |
| 粒球文献 | 粒球生成、粒度调节、纯度/质量、鲁棒分类、粗粒度聚类、密度峰值加速、非球形粒椭球建模 | 用球或椭球替代样本点，定义中心、半径、纯度、质量、邻域、密度或边界，再用于分类/聚类/采样 | 为本项目提供稳定粒球生成、置信度、局部结构先验和形状风险依据 | 优先使用稳定/自适应生成方法；把粒球质量、半径、纯度、大小、重叠度作为 confidence，而不是只用 membership | 多数是传统机器学习分类/聚类，不是深度 GCL；部分方法依赖标签纯度，不能直接用于无监督 GRACE |
| 深度学习 + 粒球文献 | 粒球如何进入神经模型或深度学习流程，包括图粗化、子图采样、图构建、图自编码器、视觉 mask、开放集边界、GNN 解释、RVFL 输入压缩等 | 先生成粒球或粒球图，再作为 super node、subgraph、weighted graph、mask primitive、decision boundary、GB center 或解释结构接入模型 | 直接回答“粒球在深度表征学习中能扮演什么接口角色”，尤其是 graph coarsening、topology construction、weak structural prior、confidence / boundary signal | 优先迁移 label-free topology construction、confidence weighting、boundary-aware relation、structure-aware augmentation；监督 purity / class-wise boundary 只能做诊断，避免 label leakage | 很多方法不是无监督节点级 GCL；部分依赖标签、purity、任务特定 evaluation 或图级/图像任务，不能直接当 GRACE 论文 claim |
| 对比学习文献 | 正负样本构造、增强视图一致性、增强破坏语义、无增强对比学习、encoder/projector 表示关系 | 双视图增强 + encoder/projector + NT-Xent/InfoNCE，或弱/强视图分布一致性，或 encoder-output 与 projector-output 对比 | 提供 GRACE baseline、增强风险、projector 前后表征差异和 weak teacher 思路 | 粒球更适合作为 soft relation、confidence 或 view relation teacher；不要随意改 evaluation/split | 大多数对比学习论文没有粒球机制，不能直接证明 same-ball membership 是语义正样本 |

## 3. How Granular Ball Can Connect to Contrastive Learning

| 结合位置 | 粒球角色 | 可能解决的问题 | 主要风险 | 是否适合继续 | 优先级 |
|---|---|---|---|---|---|
| 数据增强阶段 | 局部结构保护器或增强强度控制信号 | 降低 edge dropping / feature masking 对局部语义的破坏 | 过度依赖粒球会削弱增强多样性 | 适合，但需先做机制诊断 | 高 |
| weak view / teacher view | 从弱视图生成粒球关系，指导强视图 | 受 CLSA 启发，避免强增强直接当作同一表示硬拉近 | teacher view 若本身不稳定会传递错误关系 | 适合 | 高 |
| encoder representation 阶段 | 在 encoder output 上生成粒球 | encoder 表示更接近下游节点语义 | 训练早期表示不稳定，粒球划分波动大 | 适合，需 warm-up 或 EMA | 高 |
| projector 前后阶段 | 比较 encoder H 与 projector Z 的粒球关系 | 受 SPARROW 启发，检查 projector 是否形成可用替代视图 | projector 空间可能只服务 loss，不一定稳定 | 可尝试诊断实验 | 中 |
| positive construction | soft positive relation，而不是二值 same-ball label | 利用局部语义近邻，缓解单节点正样本过窄 | hard same-ball 会引入错误正样本 | 只适合 soft/confidence 版本 | 中 |
| negative filtering | 过滤高风险 false negative | 避免把潜在相似节点推远 | 信号可能太弱，只减少负样本但不提供新监督 | 暂缓作为主线 | 低 |
| prototype / cluster | 粒球中心作为局部原型 | 提供多粒度结构摘要 | 原型拉近过强可能过平滑或塌缩 | 谨慎尝试 | 中 |
| node-to-ball relation | 节点到粒球的软隶属度 | 提供节点级 confidence 和局部语义分布 | 隶属度定义不稳会污染 loss | 适合 | 高 |
| ball-to-ball relation | 粒球间 overlap / intersection / distance relation | 建模局部结构之间关系，可能比原图邻接更鲁棒 | 多视图聚类中的 ball relation 不一定适合单图节点 | 适合小步实验 | 高 |
| confidence weighting | 用粒球纯度、半径、密度或稳定性加权 | 降低低置信粒球对训练的伤害 | confidence 无依据会变成新超参噪声 | 适合 | 高 |
| local structure preservation | soft structural constraint | 保留局部语义结构，避免增强破坏 | 约束过强会限制对比学习表达 | 适合 | 高 |
| graph denoising | 边或邻域的噪声过滤信号 | 面向 noisy graph 更容易形成 claim | clean graph 上收益可能不明显 | 适合噪声场景 | 中 |
| robust GCL | 鲁棒局部结构先验 | 连接粒球鲁棒性与 GCL 表征学习 | 需要跨数据集、跨 seed 证明稳定 | 适合 | 高 |

## 4. Paper Notes

### 4.0 How to Read These Notes

单篇论文笔记用于支持实验设计，不用于直接拼接成论文综述。后续 Codex 读取本节时，应先判断论文属于哪类证据，再决定是否能转成实验：

| 文献类别 | 可直接借鉴什么 | 不能直接借鉴什么 |
|---|---|---|
| GRACE / GCL baseline | baseline、view、encoder/projector、evaluation protocol、augmentation risk | 粒球正负样本结论，因为原文没有粒球 |
| 粒球 + 对比学习 | soft relation、fuzzy relation、false negative / noise stripping 动机 | 多视图聚类中的 hard ball positive / negative 不能原样迁移到单图节点级 GRACE |
| 纯粒球文献 | 生成稳定性、半径/密度/边界/质量、无监督 confidence proxy | 监督 purity、hard membership、传统分类器规则 |
| 深度学习 + 粒球 | 粒球进入深度模型的接口，如 graph construction、sampling、masking、center confidence、boundary distance | 监督图粗化、class-wise boundary、任务特定 loss 不能直接作为无监督 GCL loss |

本节中 `Unknown` 和 `TODO` 是刻意保留的不确定信息。后续实验不得用这些字段支撑 claim；只能把它们作为需要补读或补实验的检查项。

## Paper: Deep Graph Contrastive Representation Learning

### 4.1 Basic Information

- 年份：2020，具体正式发表信息 TODO
- 领域：图自监督学习 / 节点级图对比学习
- 任务：transductive node classification、inductive node classification
- 数据集：Cora、Citeseer、Pubmed、DBLP、Reddit、PPI
- 是否节点级：是
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/CRIPAC-DIG/GRACE`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

GRACE 解决无监督图表示学习中如何在节点级构造对比目标的问题。它通过结构和属性扰动生成两个 graph views，并最大化同一节点在两视图中的表示一致性。

### 4.3 Method Structure

- 输入：带属性图 `(X, A)`
- 编码器：transductive citation graph 使用 GCN；inductive large graph 使用 GraphSAGE-GCN 变体
- 对比对象：两个增强图视图中同一节点的表示
- 正样本定义：同一节点在两个 views 中的表示
- 负样本定义：其他节点表示
- 增强：edge removing + feature masking
- loss：节点级对称 contrastive loss，形式接近 NT-Xent / InfoNCE
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 view 和 mask；不使用粒球、prototype、cluster、confidence

### 4.4 Granular Ball Related Points

- 局部结构建模：通过 GNN 和图增强间接建模
- 聚类或原型：无
- 样本可信度：无
- 邻域选择：无显式置信邻域，只使用图结构和增强
- 噪声过滤：无显式机制
- 多粒度表示：无
- Deep learning integration point：GNN encoder + projector
- Representation learning role：节点表征学习 baseline
- Robust training role：通过双视图增强提升鲁棒性，但没有粒球鲁棒先验

### 4.5 Inspiration for This Project

GRACE 是本项目最重要 baseline。粒球不应首先替代 GRACE 的 evaluation 或 split，而应作为附加的局部关系、置信度或弱结构约束进入训练。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

GRACE 没有粒球机制，也没有证明 same-ball membership 可以作为正样本。直接把粒球作为 hard positive label 会改变原始节点级一一对应正样本假设。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：TODO，需要读取 `02_experiment_history_and_lessons.md`
- 支持继续某个方向：支持以 GRACE 为 baseline 做局部结构增强
- 提示某个方向应该放弃：不支持随意修改 baseline evaluation 或 train/validation/test split

### 4.8 Experimental Hypotheses

- H1：在 GRACE 中加入 confidence-weighted ball relation，比 hard same-ball positive 更稳定。
- H2：粒球关系应先作为 auxiliary loss，而不是替代原始 contrastive loss。
- H3：粒球生成应优先在 encoder 表示或 weak view 表示上诊断，而不是直接在 projector loss 空间强拉近。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是

### 4.10 Important Formula or Pseudocode

```text
view1 = corrupt_graph(X, A, edge_drop, feature_mask)
view2 = corrupt_graph(X, A, edge_drop, feature_mask)
z1 = projector(encoder(view1))
z2 = projector(encoder(view2))
loss = node_level_contrast(z1, z2)
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它定义了本项目的节点级 GCL 基线、评估协议和最小可比较方法结构。

## Paper: Simplified Graph Contrastive Learning Model Without Augmentation

### 4.1 Basic Information

- 年份：2025，PDF 显示 TKDE 2025
- 领域：图自监督学习 / augmentation-free GCL
- 任务：节点分类、节点聚类、图分类
- 数据集：Cora、CiteSeer、Pubmed、DBLP、Amazon Photo、Amazon Computers、Ogbn-arxiv、Ogbn-Mag、Ogbn-Products，以及 NCI1、PROTEINS、MUTAG、DD、COLLAB、REDDIT-BINARY、IMDB-BINARY 等图级数据集
- 是否节点级：是
- 是否图级：是
- 是否有代码：Unknown，已读文本未确认 GitHub 链接
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

该文认为标准 GCL 的 GraphDA 可能破坏图语义，且多分支 encoder/projector 带来计算和超参成本，因此提出无增强、单流的 SPARROW。

### 4.3 Method Structure

- 输入：原始图 `(X, A)`
- 编码器：单个 GNN encoder，文中使用 GCN 形式说明
- 对比对象：GNN encoder 输出 `H` 与 projection head 输出 `Z`
- 正样本定义：同一节点的 `h_i` 与 `z_i`
- 负样本定义：其他节点在 `H` 或 `Z` 中的表示
- 增强：无 GraphDA
- loss：NT-Xent，文中从 alignment / uniformity 和 mutual information lower bound 角度解释
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：不使用粒球、prototype、cluster、mask、confidence；使用 encoder/projector 形成隐式双表示

### 4.4 Granular Ball Related Points

- 局部结构建模：依赖 GNN message passing
- 聚类或原型：节点聚类作为实验任务，但方法本身不是 prototype 方法
- 样本可信度：无
- 邻域选择：无
- 噪声过滤：无显式机制
- 多粒度表示：无
- Deep learning integration point：encoder output 与 projector output 的对比
- Representation learning role：说明 projector 后表示可作为训练视图
- Robust training role：避免 GraphDA 破坏语义

### 4.5 Inspiration for This Project

粒球可以先用于诊断 encoder 表示和 projector 表示之间的局部结构是否一致。若 projector 空间粒球不稳定，粒球生成应优先放在 encoder output。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

SPARROW 的核心是取消增强并用 `H-Z` 对比，不包含粒球。若直接套用，会偏离 GRACE 风格双视图增强框架；更适合做辅助诊断或 ablation。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：TODO
- 支持继续某个方向：支持“不要盲目依赖强 GraphDA”的方向
- 提示某个方向应该放弃：提示如果粒球只增加复杂度但没有稳定机制，可能不值得作为主线

### 4.8 Experimental Hypotheses

- H1：encoder-output 粒球关系比 projector-output 粒球关系更稳定。
- H2：无增强或弱增强设置下的粒球 soft constraint 可减少增强语义破坏。
- H3：`H-Z` 表示一致性可作为粒球稳定性诊断指标。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：中
- 是否适合下一轮实验：暂缓，先作为诊断参考

### 4.10 Important Formula or Pseudocode

```text
H = encoder(X, A)
Z = projector(H)
positive = (H[i], Z[i])
negative = (H[i], H[j]), (H[i], Z[j]), j != i
loss = NT-Xent(H, Z)
```

### 4.11 Final Value Judgment

- 价值：中
- 理由：它对“增强可能破坏图语义”和“projector 可形成训练视图”有启发，但不是粒球方法，也不是 GRACE 原始设置。

## Paper: Contrastive Learning with Stronger Augmentations

### 4.1 Basic Information

- 年份：PDF 显示 2021，arXiv v2 为 2022；正式年份 TODO
- 领域：图像自监督对比学习
- 任务：视觉表示学习、ImageNet 线性/微调评估及下游任务
- 数据集：ImageNet，downstream datasets 具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/maple-research-lab/CLSA`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

强增强能暴露新模式，但直接把强增强视图作为普通正样本可能破坏结构并导致检索困难。CLSA 用弱增强视图在 representation bank 上的分布来监督强增强视图。

### 4.3 Method Structure

- 输入：图像
- 编码器：ResNet-50，配合 projection head；基于 MoCo 风格 queue / representation bank
- 对比对象：弱增强和强增强图像表示
- 正样本定义：同一实例的弱/强增强视图，但强增强不直接简单拉近，而由弱视图分布监督
- 负样本定义：representation bank / queue 中其他实例
- 增强：弱增强 + RandAugment 风格强增强
- loss：原有 contrastive loss + distributional divergence minimization
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 weak/strong view 和 distribution teacher；不使用粒球

### 4.4 Granular Ball Related Points

- 局部结构建模：无图结构，只有 representation bank 分布
- 聚类或原型：无显式 prototype
- 样本可信度：弱视图分布相当于更可信 teacher signal
- 邻域选择：通过 representation bank 上的相似分布间接表达邻域关系
- 噪声过滤：避免强增强直接主导 contrastive loss
- 多粒度表示：无
- Deep learning integration point：weak/strong augmentation consistency
- Representation learning role：分布级监督强视图
- Robust training role：降低强增强破坏 identity 的风险

### 4.5 Inspiration for This Project

这篇对本项目最重要的启发是“弱视图关系 teacher”。粒球可以先在 weak view 或较稳定的 encoder 表示上生成关系分布，再用 soft distribution 或 confidence 去约束 strong view，而不是直接把 strong view 的 same-ball 当正样本。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

CLSA 是图像实例级方法，不处理图节点、邻接结构或 GNN message passing。它的 representation bank 机制也不能直接替代图上的节点关系。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：TODO
- 支持继续某个方向：强烈支持 weak view relation teacher 和 soft relation
- 提示某个方向应该放弃：警示不要把被强扰动破坏的视图直接硬拉近

### 4.8 Experimental Hypotheses

- H1：weak-view 粒球关系比 strong-view 粒球关系更适合作 teacher。
- H2：用 ball relation distribution 对齐，比 same-ball hard positive 更稳。
- H3：粒球 confidence 可用于降低强增强节点表示的错误监督权重。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是

### 4.10 Important Formula or Pseudocode

```text
weak_relation = relation_distribution(weak_view_repr)
strong_relation = relation_distribution(strong_view_repr)
loss_aux = divergence(stopgrad(weak_relation), strong_relation)
loss = grace_loss + lambda * loss_aux
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：虽然不是图论文，但它直接支持“weak teacher + soft distribution”路线，正好规避 hard same-ball positive 的风险。

## Paper: Multi-view Granular-ball Contrastive Clustering

### 4.1 Basic Information

- 年份：2025，PDF 显示 AAAI-25
- 领域：多视图聚类 / 粒球对比学习
- 任务：multi-view clustering
- 数据集：BBCSport、Caltech101-20、Cora、Scene-15、MNIST-USPS、ALOI-100、NoisyMNIST
- 是否节点级：否；Cora 在该文中作为多视图聚类数据，不是标准节点级 GCL
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/Duo-laimi/mgbcc`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

实例级多视图对比学习容易引入 false negatives，cluster-level 方法又过于宏观并忽略局部结构。MGBCC 使用位于 instance 与 cluster 之间的 granular ball 做多粒度对比。

### 4.3 Method Structure

- 输入：多视图样本
- 编码器：deep multi-view clustering encoder，具体架构需进一步查表 TODO
- 对比对象：不同视图或同视图中的 granular balls
- 正样本定义：通过 intra-view overlap 与 cross-view intersection size 建立连接的 granular balls
- 负样本定义：未连接或 dissimilar granular balls
- 增强：多视图天然作为 views
- loss：granular-ball level contrastive loss
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 granular balls、multi-view、ball relation；未确认显式 confidence

### 4.4 Granular Ball Related Points

- 局部结构建模：是，粒球被声明为保留 sample set 的 local topological structure
- 聚类或原型：粒球中心 / 粗粒度单元
- 样本可信度：Unknown
- 邻域选择：通过 overlap 与 intersection 建立 ball connection
- 噪声过滤：通过避免实例级 false negative 间接处理
- 多粒度表示：是，instance 与 cluster 之间的 granular-ball level
- Deep learning integration point：deep multi-view clustering
- Representation learning role：共享 latent space 中拉近关联粒球
- Robust training role：缓解 false negative 和局部结构缺失

### 4.5 Inspiration for This Project

粒球最自然的角色不是 hard node label，而是“介于节点与全局簇之间的局部关系单元”。对 GRACE 而言，可借鉴 ball-to-ball relation，但应改成 soft relation 或 confidence-weighted relation。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

MGBCC 是多视图聚类，不是单图节点表示学习。它把 connected granular balls 拉近的设计如果直接迁移到 GRACE，可能变成过强 prototype pulling 或 hard same-ball positive。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：TODO
- 支持继续某个方向：支持 ball-to-ball relation、local structure preservation
- 提示某个方向应该放弃：不支持简单 same-ball hard label；需要 soft/置信度控制

### 4.8 Experimental Hypotheses

- H1：ball-to-ball soft relation 比 node same-ball hard positive 更稳定。
- H2：基于 overlap / shared nodes 的跨视图 ball relation 可作为 weak relation teacher。
- H3：粒球数量和粒度参数对结果高度敏感，需要先做稳定性诊断。

### 4.9 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是，但只迁移 soft relation 思路

### 4.10 Important Formula or Pseudocode

```text
balls_v1 = build_granular_balls(Z_view1)
balls_v2 = build_granular_balls(Z_view2)
M = ball_relation_by_overlap_or_intersection(balls_v1, balls_v2)
positive_balls = {(i, j) | M[i, j] > 0}
negative_balls = {(i, j) | M[i, j] == 0}
loss_ball = contrast(ball_centers, positive_balls, negative_balls)
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：这是当前目录中最直接的“粒球 + 对比学习”论文，支持粒球作为中间粒度关系单元，但迁移时必须软化。

## Paper: Views Attention Fusion of Granular-ball Fuzzy Representations Split for Improved Multi-View Clustering

### 4.1 Basic Information

- 年份：PDF 显示 AAAI 2026
- 领域：多视图聚类 / 粒球 fuzzy contrastive learning
- 任务：multi-view clustering
- 数据集：WebKB、Multi-COIL-10、Multi-COIL-20、Caltech101-7、Prokaryotic、NUSWIDE、Reuters、DHA、UCI-Digits
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/Lsy235/GFSAF`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

多视图融合中不同 view 的内在质量差异会导致 Representation Degradation。GFSAF 通过 granular-ball fuzzy contrastive learning 提取 mutual representation，用 Noise Stripping Loss 提取 complementary representation，再用 Cross Views Attention 融合。

### 4.3 Method Structure

- 输入：多视图数据
- 编码器：AE 或 DAE backbone
- 对比对象：多视图 granular-ball representations
- 正样本定义：PGB，包括跨视图同一样本相关 granular balls；同一视图内有样本重叠的 granular balls 也归入 PGB 以考虑 false negatives
- 负样本定义：dissimilar granular balls，记为 NGB
- 增强：多视图天然作为 views
- loss：Granular-ball Fuzzy Contrastive loss、Noise Stripping Loss、Views-Fusion Stage loss
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 granular ball、view、attention、noise stripping；confidence 未确认

### 4.4 Granular Ball Related Points

- 局部结构建模：是
- 聚类或原型：粒球表示 point set
- 样本可信度：fuzzy split 相关，但是否显式 confidence 未确认
- 邻域选择：通过 overlapping granular balls 处理潜在 false negative
- 噪声过滤：Noise Stripping Loss
- 多粒度表示：是
- Deep learning integration point：AE/DAE + GFCL + attention fusion
- Representation learning role：分离 mutual / complementary representation
- Robust training role：降低 view quality discrepancy 和噪声影响

### 4.5 Inspiration for This Project

GFSAF 比 MGBCC 更强调 fuzzy 和噪声剥离。对本项目而言，粒球可作为“弱关系 + 置信度 + 去噪”的组合信号，而不是只做正负样本二值划分。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

GFSAF 是多视图聚类和 AE/DAE 框架，不是 GNN 节点级学习。它的 PGB/NGB 定义若直接移植到 GRACE，仍可能变成 hard positive / hard negative。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：TODO
- 支持继续某个方向：支持 fuzzy relation、noise stripping、view quality control
- 提示某个方向应该放弃：不建议只做简单 negative filtering

### 4.8 Experimental Hypotheses

- H1：same-ball relation 应转化为 fuzzy weight，而非二值标签。
- H2：低质量或不稳定粒球应降低 loss 权重。
- H3：将粒球信号用于 denoising / confidence，比用于 hard pulling 更适合形成论文 claim。

### 4.9 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是，但应简化为 confidence / fuzzy auxiliary loss

### 4.10 Important Formula or Pseudocode

```text
PGB = positive granular-ball pairs
NGB = negative granular-ball pairs
L_GFCL = average InfoNCE-like loss over view pairs using PGB and NGB

for GRACE adaptation:
    weight_ij = fuzzy_ball_relation(i, j)
    loss_aux = weighted_relation_alignment(z_view1, z_view2, weight_ij)
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它直接提示粒球关系应 fuzzy 化，并把噪声处理纳入方法动机，适合本项目当前偏好的 soft structural constraint。

## Paper: Robust Smart Contract Vulnerability Detection via Contrastive Learning-Enhanced Granular-ball Training

### 4.1 Basic Information

- 年份：2026，arXiv v1；PDF 元数据 DOI 为 `https://doi.org/10.48550/arXiv.2603.27734`
- 领域：智能合约漏洞检测 / 鲁棒深度学习 / 对比学习 + 粒球训练
- 任务：contract-level smart contract vulnerability detection under label noise
- 数据集：预训练使用 Ethereum-SC Large，PDF 写明原始约 113,000 个合约，预处理后 78,211 个；微调用 [10] 的约 40,000 个真实智能合约；评估聚合 ReentrancyBenchmark 和 SolidiFiBenchmark，漏洞类型包括 Reentrancy、Timestamp Dependency、Integer Overflow
- 是否节点级：否
- 是否图级：否；更接近代码/合约级二分类或多任务漏洞检测
- 是否有代码：PDF 未发现 GitHub 或代码链接
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

智能合约漏洞检测数据常由开源工具或静态分析器标注，标签噪声会削弱 DNN 检测模型。CGBC 的核心问题是：如何先用无监督对比学习得到更可聚类的合约表示，再用粒球层生成粗粒度表示和 majority-vote 标签，以降低噪声标签对微调训练的影响。

### 4.3 Method Structure

- 输入：原始智能合约 `C` 及语义保持增强合约 `C~`；微调阶段输入带标签合约 embedding 和 noisy label
- 编码器：多层 Transformer contract encoder；对比预训练阶段使用共享权重双分支 encoder 和 predictor MLP，威胁有效性部分说明实现参考 SimSiam
- 对比对象：原始合约和语义保持增强合约的表示
- 正样本定义：同一合约的原始版本与增强版本
- 负样本定义：无显式负样本；stop-gradient + cosine symmetric loss
- 增强：生成不改变功能语义的智能合约代码片段并插入函数；用 IR core operation sequence 的 SHA256 digest 验证语义一致性；PDF 报告 1,000 对样本语义相似率 99.60%，Jaccard diversity 在阈值 0.9 时为 49.49%，阈值 0.95 时超过 74.03%
- loss：预训练使用 symmetric cosine contrastive loss；微调使用 `L = L_clustering + L_SCE`，其中 `L_clustering` 包含 intra-GB compactness loss 和 inter-GB looseness loss，`L_SCE` 为 symmetric cross entropy
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 augmented view、GB center、GB majority-vote label、supervised purity、coarse-grained representation；不使用图 subgraph；confidence 主要体现为 label purity / GB quality，但依赖标签

### 4.4 Granular Ball Related Points

- 局部结构建模：在合约 embedding 空间中把相似合约聚成 GB
- 聚类或原型：GB center 作为一组样本的粗粒度表示
- 样本可信度：使用 supervised `Purity(X)`，即 GB 内 majority label 占比
- 邻域选择：通过 cosine distance 找最远锚点并递归二分
- 噪声过滤：用 GB center 和 majority-vote label 替代组内样本表示/标签，降低少数 noisy labels 的影响
- 多粒度表示：是，粒球数量和大小自适应，不预设固定簇数
- Deep learning integration point：GBC layer 位于 encoder layer 和 classifier layer 之间
- Representation learning role：contrastive pretraining 让相似合约表示更接近，从而辅助后续粒球聚类
- Robust training role：SCE + GB 粗粒度标签校正 + intra/inter GB clustering loss 面向 noisy label robustness

### 4.5 Inspiration for This Project

CGBC 对本项目最有价值的启发不是照搬 supervised GB label correction，而是提示“先用对比学习 warm-up 表示，再生成粒球”的顺序可能比训练早期直接构造粒球更稳定。它还支持把粒球作为 robust local structure prior 和 noisy-label / noisy-relation 缓冲层，而不是直接把每个节点的 noisy relation 放进主 loss。

对 GRACE + 粒球而言，可迁移为三类低风险设计：第一，先在 GRACE warm-up 后的 encoder embedding 中离线诊断粒球质量；第二，用 label-free confidence proxy 替代 supervised purity；第三，优先在噪声边、强增强或伪标签不确定场景检验粒球鲁棒性，而不是只追 clean graph 小幅涨点。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

CGBC 是有监督智能合约漏洞检测，不是无监督节点级图对比学习。它的关键 GB 质量指标 `Purity(X)` 直接使用标签，GB label 也来自 majority vote；这些机制如果进入 GRACE 预训练会造成 label leakage。其 inter-GB loss 还按 label 拉近同类 GB center、推远异类 GB center，直接迁移会变成强监督 prototype pulling，与本项目已知失败模式冲突。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：支持“训练早期 embedding 不稳定会影响粒球质量”的担忧，也解释为什么直接 GB view generation 或 direct relation regularization 容易失效。
- 支持继续某个方向：支持 contrastive warm-up 后再构造粒球、robust/noisy setting、confidence-weighted soft relation 和 label-free purity proxy。
- 提示某个方向应该放弃：不支持把 supervised purity、majority-vote GB label、same-label GB center pulling 原样用于无监督 GRACE。

### 4.8 Experimental Hypotheses

- H1：在 GRACE warm-up 后生成粒球，比从训练初期动态生成粒球更稳定。
- H2：label-free confidence proxy 如果能近似 CGBC 中的 supervised purity 作用，可降低错误 same-ball relation 对 loss 的污染。
- H3：粒球信号在噪声边、强增强或伪标签不确定场景中的收益，应比 clean graph 标准设置更容易稳定出现。

### 4.9 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中到高；原文使用四张 Tesla V100-SXM2-32GB，且所有结果为 5 次独立运行平均
- 复现风险：高；PDF 未提供代码链接，且任务、数据和模型栈与 GRACE 差异大
- 是否适合下一轮实验：暂缓直接复现；适合作为 warm-up + 粒球稳定性诊断 + 噪声鲁棒性实验的设计依据

### 4.10 Important Formula or Pseudocode

```text
# contrastive pretraining
z = F1(C)
z_tilde = F2(augment_semantic_preserving(C))
p = P(z)
p_tilde = P(z_tilde)
L_CL = 0.5 * (L_cos(p, stopgrad(z_tilde)) + L_cos(p_tilde, stopgrad(z)))

# supervised GB training in CGBC
GBs = adaptive_split_by_cosine_distance(embeddings, labels, purity_threshold)
center_GB = mean(embeddings_in_GB)
label_GB = majority_vote(labels_in_GB)
L = L_intra_GB_compactness + L_inter_GB_looseness + L_SCE

# safer GRACE adaptation idea
h = encoder_after_warmup(view)
GBs = build_label_free_balls(h)
confidence = f(radius, density, size, center_distance, epoch_stability)
loss = grace_loss + lambda * confidence_weighted_soft_relation_loss
```

### 4.11 Final Value Judgment

- 价值：中到高
- 理由：它是明确的“contrastive learning + granular-ball robust training”论文，证明 CL warm-up 与 GB 粗粒度鲁棒训练可以互补；但其核心 purity、majority label 和 inter-class separation 都依赖标签，只能提供机制启发，不能直接作为无监督 GRACE 方法依据。

## Paper: GBmooer: Granular-ball Text Label Smoothing with Hypersphere Contrastive Learning

### 4.1 Basic Information

- 年份：2026 SSRN preprint；PDF metadata 显示创建时间 2025-12，修改时间 2026-01；页脚标注 `Preprint not peer reviewed`
- 领域：粒球 + 对比学习 + NLP 噪声标签鲁棒训练
- 任务：text classification under label noise
- 数据集：TREC、AGNEWS、IMDB、Chnsenticorp
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown；论文声明 data available on request
- 本地来源：`C:/Users/Admin/Downloads/ssrn-6022343 (1).pdf`

### 4.2 Core Problem

文本标签噪声会使深度模型过拟合错误 feature-label mapping。传统全局 label smoothing 依赖预训练模型的一次性全局平滑，可能累积错误。GBmooer 的核心问题是：先让高维文本表示更适合聚成局部一致的粒球，再把 label smoothing 限制在局部粒球内，降低噪声扩散范围。

### 4.3 Method Structure

- 输入：带噪声标签的文本分类训练集
- 编码器：BERT 作为主要 encoder；迁移实验还包括 LSTM、TextCNN、GRU
- 对比对象：原始文本样本、文本增强得到的 positive samples、batch 内其他样本
- 正样本定义：同一样本经 synonym replacement、word order shuffling、random word deletion 等增强得到的视图
- 负样本定义：batch 内其他样本；具体 hard negative 策略 Unknown
- 增强：文本 token 级增强，包括同义词替换、词序打乱、随机删词
- loss：hypersphere angular contrastive loss、KL label smoothing loss、symmetric KL prediction consistency loss
- 是否使用 prototype / cluster / subgraph / view / mask / confidence：使用 hypersphere projection、granular-ball clustering、GB center/radius/quality、局部 label distribution；未发现显式 node/subgraph 机制

### 4.4 Granular Ball Related Points

- 局部结构建模：在对比预训练后的 hypersphere 表示空间中生成局部高密度粒球
- 聚类或原型：粒球作为局部 normal prototype，包含 center 和 radius
- 样本可信度：粒球质量由 average angular distance 描述，可转成无标签 confidence proxy；论文中 smoothing 仍依赖球内标签分布
- 邻域选择：通过角距离和二分 split 形成局部样本集合
- 噪声过滤：不是直接删除噪声，而是在每个粒球内部做局部 soft label smoothing
- 多粒度表示：是，粒球通过迭代 split 形成多粒度局部区域
- Deep learning integration point：contrastive pretraining 后生成粒球，再用粒球局部标签分布指导 fine-tuning
- Representation learning role：contrastive learning 先优化表示，使同类更近、异类更分离，从而提高后续 GBC 质量
- Robust training role：把噪声影响限制在局部粒球，避免全局 label smoothing 的错误传播

### 4.5 Inspiration for This Project

GBmooer 对本项目最有价值的启发不是“球内样本可直接共享标签”，而是两点：第一，先用 contrastive warm-up / hypersphere normalization 改善表示空间，再评估粒球稳定性；第二，把粒球质量、角半径、中心距离和球内分布差异作为 soft confidence，而不是只用 same-ball membership。对 GRACE 可迁移为 encoder embedding 或 weak-view embedding 上的 angle-space ball diagnostic。

### 4.6 Why It Cannot Be Directly Copied to GRACE plus Granular Ball

GBmooer 是有标签文本分类和噪声标签学习，不是无监督节点级 GCL。它的 label smoothing 明确使用球内标签分布，如果直接迁移到 GRACE 会造成 label leakage。文本增强、BERT fine-tuning 和 KL classification objective 也不等价于图增强、InfoNCE 和节点级 linear evaluation。因此只能迁移“表示预热后建球、角距离质量、局部软化/降权”的思想，不能迁移监督标签平滑本身。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：支持“直接 same-ball hard positive / prototype pulling 风险高”的判断；GBmooer 自身也没有把 majority label 作为硬替换，而是使用 soft label distribution
- 支持继续某个方向：支持 contrastive warm-up 后的粒球稳定性诊断、node-to-ball confidence、local denoising / confidence weighting
- 提示某个方向应该放弃：不支持把球内多数类、球内标签一致性或球内 membership 原样迁移为 GRACE 训练标签

### 4.8 Experimental Hypotheses

- H1：在 GRACE warm-up encoder embedding 上建 hypersphere / angle-distance 粒球，比 raw feature 或 projector output 更稳定。
- H2：average angular distance、radius margin、center distance 可作为无标签 confidence，比 hard same-ball membership 更安全。
- H3：如果粒球只在噪声边/强增强场景中提供稳定收益，应把论文 claim 限定为 robust local structure prior，而不是 clean graph 通用提升。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中；需要额外 warm-up 表示抽取、angle-space clustering 和跨 epoch/seed 稳定性诊断
- 复现风险：中到高；PDF 是 SSRN preprint，代码 Unknown，且任务不匹配
- 是否适合下一轮实验：是，但只适合先做诊断或小权重 confidence，不适合直接做监督 label smoothing

### 4.10 Important Formula or Pseudocode

```text
h_i = normalize(x_i W)
Dis_a(a, b) = arccos(a^T b)

AD_k = mean_{x_i in GB_k} Dis_a(x_i, center_k)
confidence(GB_k) = f(-AD_k, radius_margin, size, stability)

for GRACE adaptation:
    z = stopgrad(encoder_warmup(data))
    balls = angular_granular_ball_clustering(normalize(z))
    use confidence(GB_k) only as soft loss weight or diagnostic
```

### 4.11 Final Value Judgment

- 价值：中到高
- 理由：它不是图或节点级 GCL 文献，不能直接支撑 GRACE 方法 claim；但它强烈支持“对比学习先塑造适合粒球的表示空间”和“局部粒球质量可作为鲁棒训练信号”这两个可迁移机制。

## Paper: Granular Ball Computing Classifiers for Efficient, Scalable and Robust Learning

### 4.1 Basic Information

- 年份：2019，Information Sciences
- 领域：粒球计算 / 传统分类器
- 任务：分类，重点是 GBSVM 和 GBkNN
- 数据集：PDF 提到 public and artificial datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

该文提出用 granular ball 替代样本点参与分类器建模，使传统分类器在效率、可扩展性和鲁棒性上受益。

### 4.3 Granular Ball Mechanism

- Granular object：granular ball，用球覆盖局部样本集合。
- Shape assumption：ball，球形局部结构。
- Supervision setting：supervised classification。
- Depends on label purity：是，使用 majority label 和 purity degree。
- Center definition：球内样本均值。
- Radius / shape / density definition：半径为球内样本到中心的平均距离，避免最大距离受 outlier 支配。
- Ball quality / confidence signal：purity degree、半径、球大小、ball distance。
- Local structure represented：局部样本覆盖、粗粒度边界、全局-局部混合结构。

### 4.4 Generation / Splitting / Stopping

- Generation method：从全数据集开始，使用 k-means 风格划分生成粒球。
- Split / merge rule：低 purity 粒球继续划分。
- Stopping criterion：粒球 purity 达到阈值；具体阈值依实验设置。
- Granularity control：由 purity threshold 和划分过程控制。
- Overlap handling：半径为平均距离时可能产生 overlap；更多粒球会降低 overlap。
- Outlier / noise handling：平均半径降低 outlier 影响；粗粒度覆盖可过滤类噪声。
- Complexity / scalability claim：强调 efficient、scalable；GBkNN 可替代点级 kNN 加速。
- Stability claim：Unknown，论文更强调鲁棒和可扩展。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：center、radius、ball size、node-to-ball distance、ball-to-ball distance、purity proxy。
- How to convert：用半径/大小/距离/跨 epoch 稳定性构造 label-free confidence；不使用真实标签 purity 进训练。
- Suitable representation space：encoder output 或 weak-view representation。
- Expected benefit：让粒球成为软局部结构先验，而不是二值 membership 标签。

### 4.6 Risks if Used Directly

- Supervised information leakage：purity 和 majority label 依赖标签，不能用于无监督预训练 loss。
- Hard membership risk：只使用 membership 会丢掉 radius、distance、quality 等关键信息。
- Shape assumption risk：球形假设不一定适合图表示空间。
- Computational risk：频繁在线生成粒球可能增加训练开销。
- Mismatch with node-level GCL：该文是传统分类器，不处理双视图增强或节点级对比。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：若只用 same-ball membership，会丢掉该文强调的半径、距离、纯度等质量信息。
- 支持继续某个方向：支持 confidence-weighted local structure。
- 提示某个方向应该放弃：不支持仅用 hard membership 作为正样本标签。

### 4.8 Experimental Hypotheses

- H1：粒球半径和大小可作为局部关系可信度，降低噪声球影响。
- H2：node-to-ball distance 比 binary membership 更适合作 soft relation。
- H3：无监督场景需要用 label-free purity proxy，例如密度、稳定性、半径分布，而非真实标签纯度。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：中
- 是否适合下一轮实验：是

### 4.10 Important Formula or Pseudocode

```text
center(GB) = mean(points in GB)
radius(GB) = average_distance(points, center)
purity(GB) = majority_class_count / ball_size  # supervised only
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它定义了粒球在本项目中最基础的可迁移信号：中心、半径、距离、纯度/质量、鲁棒局部覆盖。

## Paper: GBG++: A Fast and Stable Granular Ball Generation Method for Classification

### 4.1 Basic Information

- 年份：2024，IEEE Transactions on Emerging Topics in Computational Intelligence
- 领域：粒球生成 / 分类
- 任务：GB generation、GB kNN++ 分类
- 数据集：24 public benchmark datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

现有 GBG 方法依赖 k-means 或 k-division，稳定性和效率不足；GB-based classifiers 也忽略了 GB quality。GBG++ 提出更快、更稳定的粒球生成和带质量考虑的 GB kNN++。

### 4.3 Granular Ball Mechanism

- Granular object：granular ball。
- Shape assumption：ball。
- Supervision setting：supervised classification。
- Depends on label purity：是，分类任务下仍依赖标签质量判断。
- Center definition：data-driven center，避免随机选中心。
- Radius / shape / density definition：使用球形覆盖；具体半径公式需进一步查算法细节 TODO。
- Ball quality / confidence signal：GB quality、sample size、local outlier。
- Local structure represented：稳定局部覆盖和边界附近粒球质量。

### 4.4 Generation / Splitting / Stopping

- Generation method：attention-inspired / data-driven splitting。
- Split / merge rule：每次 split 只计算 data-driven center 到未划分样本的距离。
- Stopping criterion：Unknown，需进一步查具体算法。
- Granularity control：由 GBG++ 划分策略与 GB quality 控制。
- Overlap handling：Unknown。
- Outlier / noise handling：引入 local outlier detection。
- Complexity / scalability claim：减少距离计算，提升效率。
- Stability claim：论文明确强调 fast and stable，且称结果 absolutely stable。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：data-driven center、outlier flag、GB quality、ball size。
- How to convert：低质量球、异常点、边界球在 auxiliary relation loss 中降权。
- Suitable representation space：warm-up 后 encoder output；不建议训练早期直接在线强约束。
- Expected benefit：减少粒球划分随机性，提升 relation teacher 稳定性。

### 4.6 Risks if Used Directly

- Supervised information leakage：分类任务质量判断可能使用标签。
- Hard membership risk：稳定生成不等于 membership 可作为 hard positive。
- Shape assumption risk：仍是球形假设。
- Computational risk：若每个 epoch 在线重建，可能开销较大。
- Mismatch with node-level GCL：GB kNN++ 分类规则不能直接迁移到 contrastive loss。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：若训练早期粒球波动大，可能源于生成方法对初始中心或划分方式敏感。
- 支持继续某个方向：支持 stable GB generation、quality/confidence weighting。
- 提示某个方向应该放弃：不应在未诊断稳定性的情况下把粒球关系加入主 loss。

### 4.8 Experimental Hypotheses

- H1：稳定粒球生成比简单 k-means ball 对 GCL 更重要。
- H2：local outlier 节点应降低 relation loss 权重。
- H3：ball quality 可作为 confidence，优于均匀对待所有球。

### 4.9 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：暂缓，先做稳定性诊断

### 4.10 Important Formula or Pseudocode

```text
while ball needs split:
    choose data-driven center
    split by distances from center to undivided samples
    detect local outliers
    compute ball quality / size-aware confidence
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它直接支撑“粒球生成稳定性”和“质量加权”两条关键实验约束。

## Paper: An Efficient and Adaptive Granular-Ball Generation Method in Classification Problem

### 4.1 Basic Information

- 年份：2023，IEEE Transactions on Neural Networks and Learning Systems
- 领域：粒球生成 / 分类
- 任务：加速和自适应 GB generation
- 数据集：some real datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `http://www.cquptshuyinxia.com/GBC.html` 和 `https://github.com/syxiaa/gbc`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

原始 GB generation 依赖 k-means，效率和参数自适应不足。该文提出用 division 替代 k-means 加速生成，并提出考虑 GB overlap 等因素的 adaptive GB generation，使生成过程 parameter-free。

### 4.3 Granular Ball Mechanism

- Granular object：granular ball covering。
- Shape assumption：ball。
- Supervision setting：supervised classification。
- Depends on label purity：是，分类问题中的 GB generation 通常使用标签/类噪声信息。
- Center definition：由 division / GB covering 过程决定；具体公式 TODO。
- Radius / shape / density definition：球形覆盖；半径细节 TODO。
- Ball quality / confidence signal：overlap、covering quality、adaptive stopping。
- Local structure represented：用多个粒球覆盖数据分布，减少重叠并控制粒度。

### 4.4 Generation / Splitting / Stopping

- Generation method：用 division 替代 k-means 加速 GB generation。
- Split / merge rule：根据 overlap elimination 和覆盖质量自适应划分。
- Stopping criterion：parameter-free adaptive stopping。
- Granularity control：由自适应机制决定，减少手工 ball number。
- Overlap handling：显式考虑 elimination of GB overlap。
- Outlier / noise handling：涉及 class noise / label noise，具体细节 TODO。
- Complexity / scalability claim：提升 GB generation 效率。
- Stability claim：自适应生成降低参数敏感性。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：overlap ratio、covering quality、adaptive ball number。
- How to convert：高 overlap 的 ball relation 标记为 uncertain；自适应粒球数量作为默认生成策略候选。
- Suitable representation space：固定 embedding、warm-up embedding、weak view embedding。
- Expected benefit：减少随意 ball number 带来的实验噪声。

### 4.6 Risks if Used Directly

- Supervised information leakage：标签驱动停止规则不能进入无监督训练。
- Hard membership risk：overlap 低也不等于同球节点一定语义相同。
- Shape assumption risk：球形覆盖可能不适合所有局部结构。
- Computational risk：自适应生成在线化后可能增加训练成本。
- Mismatch with node-level GCL：原文不涉及图增强、双视图或 contrastive objective。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：粒球数量过少、重叠过大、划分不稳都可能导致 same-ball 信号噪声。
- 支持继续某个方向：支持自适应粒度和 overlap-aware confidence。
- 提示某个方向应该放弃：不应固定一个随意 ball number 后直接跑主实验。

### 4.8 Experimental Hypotheses

- H1：自适应粒球数量比固定粒球数量更稳定。
- H2：高 overlap 粒球关系应被降权或作为 uncertain relation。
- H3：无监督 proxy 的 overlap elimination 可提升 ball relation 的可靠性。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是，优先做诊断和无监督改写

### 4.10 Important Formula or Pseudocode

```text
replace k-means split with division split
repeat:
    split low-quality / overlapping balls
    update GB covering
until overlap and quality criteria are satisfied
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它直接回应本项目“粒球数量和生成稳定性”问题，是设计无监督 GB generation 的重要依据。

## Paper: Ball k-Means: Fast Adaptive Clustering With No Bounds

### 4.1 Basic Information

- 年份：2022，IEEE Transactions on Pattern Analysis and Machine Intelligence
- 领域：快速 k-means / 聚类
- 任务：accelerated exact k-means
- 数据集：多组聚类 benchmark，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

标准 k-means 需要大量 point-centroid distance computation。Ball k-means 用 ball 描述 cluster，并利用邻居簇、stable area、active area、annular area 减少距离计算。

### 4.3 Granular Ball Mechanism

- Granular object：ball cluster。
- Shape assumption：ball around k-means centroid。
- Supervision setting：unsupervised clustering。
- Depends on label purity：否。
- Center definition：cluster centroid。
- Radius / shape / density definition：cluster ball and annular regions；具体半径细节 TODO。
- Ball quality / confidence signal：stable area、active area、annular area、distance to neighbor clusters。
- Local structure represented：簇内稳定区域与边界活动区域。

### 4.4 Generation / Splitting / Stopping

- Generation method：k-means 迭代中用 ball 描述每个 cluster。
- Split / merge rule：无显式 split/merge；主要是 assignment 加速。
- Stopping criterion：k-means 收敛。
- Granularity control：k 值控制。
- Overlap handling：通过 neighbor clusters 和区域划分减少不必要比较。
- Outlier / noise handling：无显式机制。
- Complexity / scalability claim：减少 point-centroid distance computation。
- Stability claim：stable area assignment 不变，active area 只在少数邻近簇中调整。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：stable/active area、center distance、neighbor-cluster relation。
- How to convert：稳定区域节点赋高 confidence，active/boundary 节点降权。
- Suitable representation space：encoder output 的离线诊断或 warm-up 后在线更新。
- Expected benefit：避免边界节点被 hard same-ball relation 错误拉近。

### 4.6 Risks if Used Directly

- Supervised information leakage：无。
- Hard membership risk：k-means membership 仍不等价于语义正样本。
- Shape assumption risk：簇被球形近似。
- Computational risk：k 和迭代设置会引入额外超参。
- Mismatch with node-level GCL：该文解决聚类加速，不解决 contrastive 目标设计。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：边界节点被 hard same-ball 拉近可能引入错误正样本。
- 支持继续某个方向：支持 distance-to-center / boundary confidence。
- 提示某个方向应该放弃：不支持把同簇所有点等权拉近。

### 4.8 Experimental Hypotheses

- H1：靠近球心的节点关系更可信，靠近边界的节点关系应降权。
- H2：active area 节点更可能导致 false positive，需要 confidence 控制。
- H3：ball-kmeans 可作为快速初始粒球构造器，但不应直接形成 hard label。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：中
- 是否适合下一轮实验：是，适合作 confidence 设计参考

### 4.10 Important Formula or Pseudocode

```text
for each ball cluster:
    find neighbor clusters
    keep stable-area assignments unchanged
    only adjust active / annular area points among neighbor clusters
```

### 4.11 Final Value Judgment

- 价值：中
- 理由：它不是 GCL 方法，但对“中心高置信、边界低置信”的粒球信号很有启发。

## Paper: A Fast Granular-Ball-Based Density Peaks Clustering Algorithm for Large-Scale Data

### 4.1 Basic Information

- 年份：Unknown，PDF 显示 IEEE Transactions on Neural Networks and Learning Systems；具体年份 TODO
- 领域：粒球聚类 / density peaks clustering
- 任务：large-scale clustering
- 数据集：synthetic and real datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

Density Peaks clustering 需要对象级距离矩阵，时间复杂度为 `O(n^2)`，难以处理大规模数据。GB-DP 用 granular balls 作为 coarse-grained representation 来定义 GB density 和距离。

### 4.3 Granular Ball Mechanism

- Granular object：granular ball as coarse-grained data representation。
- Shape assumption：ball。
- Supervision setting：unsupervised clustering。
- Depends on label purity：否。
- Center definition：GB center。
- Radius / shape / density definition：GB density 根据 center、radius、成员到中心距离等定义；具体公式 TODO。
- Ball quality / confidence signal：GB density、radius、center distance。
- Local structure represented：对象与局部邻居分布相似且高概率同类。

### 4.4 Generation / Splitting / Stopping

- Generation method：unsupervised partitioning 先生成 GBs。
- Split / merge rule：具体规则 TODO。
- Stopping criterion：具体规则 TODO。
- Granularity control：由 unsupervised partitioning 控制。
- Overlap handling：Unknown。
- Outlier / noise handling：粗粒度 GB 表示降低点级噪声影响。
- Complexity / scalability claim：避免对象级 `O(n^2)` 距离矩阵，适合 large-scale data。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：density、radius、center distance、GB-level neighborhood。
- How to convert：用 density/radius 构造 label-free confidence；低密度或大半径球降低 relation 权重。
- Suitable representation space：无标签 encoder embedding 或 weak-view embedding。
- Expected benefit：为无监督 GRACE 提供比标签 purity 更合规的 confidence proxy。

### 4.6 Risks if Used Directly

- Supervised information leakage：无。
- Hard membership risk：密度峰值簇不能直接当 hard positive。
- Shape assumption risk：GB 仍是球形粗粒度表示。
- Computational risk：密度和中心距离计算需要额外缓存。
- Mismatch with node-level GCL：聚类目标不等价于节点级对比目标。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：只用 membership 忽略 density，会把低密度边界区域也当高置信。
- 支持继续某个方向：支持 density-weighted relation 和 noisy graph robustness。
- 提示某个方向应该放弃：不建议直接把密度峰值簇当硬正样本。

### 4.8 Experimental Hypotheses

- H1：GB density 可作为无监督 confidence，优于标签 purity。
- H2：低密度或大半径粒球应降低 relation loss 权重。
- H3：GB-level density relation 对噪声边/噪声特征更鲁棒。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是

### 4.10 Important Formula or Pseudocode

```text
GBs = unsupervised_partition(X)
density(GB) = f(center, radius, member_center_distances)
delta(GB) = distance_to_nearest_higher_density_GB
```

### 4.11 Final Value Judgment

- 价值：高
- 理由：它提供了无监督密度型 confidence 的依据，比标签 purity 更适合 GRACE 预训练。

## Paper: GEC: A Novel and Efficient Classifier Based on Granular-Ellipsoid Model

### 4.1 Basic Information

- 年份：2025，Information Sciences
- 领域：粒椭球计算 / 分类
- 任务：granular-ellipsoid classifier
- 数据集：multiple datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

Granular-ball 对球形簇更合适，但对 elongated 或 irregular clusters 表达不足。GEC 用 granular ellipsoid 替代 granular ball，提高非球形数据表示能力。

### 4.3 Granular Ball Mechanism

- Granular object：granular ellipsoid。
- Shape assumption：ellipsoid，ball 是特殊情况。
- Supervision setting：supervised classification。
- Depends on label purity：是，GE generation / classifier 使用类别纯度。
- Center definition：ellipsoid center。
- Radius / shape / density definition：shape matrix `H` 和 ellipsoid boundary。
- Ball quality / confidence signal：purity、shape fit、ellipsoid coverage。
- Local structure represented：非球形、拉长或不规则局部结构。

### 4.4 Generation / Splitting / Stopping

- Generation method：minimum volume enclosing ellipsoid 或 covariance-based GE，具体以论文算法为准。
- Split / merge rule：低 purity GE 继续划分。
- Stopping criterion：purity 达到阈值；样本数不足维度时需额外处理。
- Granularity control：purity threshold 和 GE 划分过程。
- Overlap handling：Unknown。
- Outlier / noise handling：Unknown。
- Complexity / scalability claim：训练可在线性复杂度内完成；预测复杂度与 GE 数、维度相关。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：shape assumption diagnostic、local covariance、anisotropic distance。
- How to convert：先检测 ball radius 是否难以覆盖 elongated local structure，再考虑 shape-aware confidence。
- Suitable representation space：离线诊断阶段的 encoder output。
- Expected benefit：避免把球形近似失败误判为粒球机制失败。

### 4.6 Risks if Used Directly

- Supervised information leakage：purity 依赖标签。
- Hard membership risk：椭球 membership 也不等价于语义正样本。
- Shape assumption risk：椭球更灵活但更复杂，可能过拟合局部噪声。
- Computational risk：shape matrix / covariance / inverse 计算成本高。
- Mismatch with node-level GCL：分类器规则不可直接变成 GCL loss。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：如果同球关系不稳定，可能是球形假设不匹配局部结构。
- 支持继续某个方向：支持 shape-aware confidence。
- 提示某个方向应该放弃：不建议一开始就做复杂椭球 GCL 主线。

### 4.8 Experimental Hypotheses

- H1：球形假设不成立的数据集上，ball relation 更容易失败。
- H2：各向异性距离或局部协方差可改善 confidence，但成本较高。
- H3：先做球形诊断，再决定是否引入椭球。

### 4.9 Implementation Difficulty

- 代码改动成本：高
- 计算成本：高
- 复现风险：中到高
- 是否适合下一轮实验：暂缓

### 4.10 Important Formula or Pseudocode

```text
GE = {x | (x - c)^T H^{-1} (x - c) <= rho^2}
```

### 4.11 Final Value Judgment

- 价值：中
- 理由：它提示粒球的球形假设有边界，但实现成本高，适合作风险分析和后续扩展。

## Paper: Synthetic Minority Oversampling Technique Based on Granular Ellipsoid Model for Imbalanced Data Classification

### 4.1 Basic Information

- 年份：Unknown
- 领域：粒椭球 / 不平衡分类 / SMOTE
- 任务：minority oversampling
- 数据集：多组 imbalanced datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

SMOTE 可能过拟合并生成不真实样本。该文用 granular ellipsoid 改进少数类采样，提出 GEB-SMOTE 和 GED-SMOTE。

### 4.3 Granular Ball Mechanism

- Granular object：granular ellipsoid。
- Shape assumption：ellipsoid。
- Supervision setting：supervised imbalanced classification。
- Depends on label purity：是，少数类/多数类和边界依赖标签。
- Center definition：GE center。
- Radius / shape / density definition：ellipsoid shape / boundary / distribution。
- Ball quality / confidence signal：boundary position、distribution consistency、non-overlap between classes。
- Local structure represented：少数类局部分布、边界区域、采样可信区域。

### 4.4 Generation / Splitting / Stopping

- Generation method：先生成 granular ellipsoids，再按边界或分布采样。
- Split / merge rule：基于 k-division 等 GE 生成方法；具体细节 TODO。
- Stopping criterion：purity / GE quality 达标。
- Granularity control：由 GE 划分与少数类粒度控制。
- Overlap handling：强调不同类别 GE 不重叠以避免噪声样本。
- Outlier / noise handling：缓解 SMOTE 的 overfitting 和 unrealistic sample。
- Complexity / scalability claim：实验基于 Python 3.10；复杂度细节已读到但需进一步整理 TODO。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

- Usable signal：boundary confidence、distribution preservation、class-overlap warning。
- How to convert：在 GCL 中不要生成/强化不真实正关系；边界或分布不一致节点降权。
- Suitable representation space：仅适合离线风险分析或后续有监督/半监督扩展。
- Expected benefit：提醒增强和关系扩展必须保持局部分布真实性。

### 4.6 Risks if Used Directly

- Supervised information leakage：少数类、多数类、类别边界都依赖标签。
- Hard membership risk：椭球边界不能直接定义无监督正样本。
- Shape assumption risk：ellipsoid 比 ball 更复杂，可能增加不稳定性。
- Computational risk：采样和 shape 估计增加额外成本。
- Mismatch with node-level GCL：oversampling 目标与 contrastive representation learning 不一致。

### 4.7 Relation to Existing Failed Experiments

- 解释已有失败实验：过强增强或硬拉近可能类似 SMOTE 的 unrealistic sample 风险。
- 支持继续某个方向：支持 boundary-aware confidence 和 distribution-preserving relation。
- 提示某个方向应该放弃：不建议随意基于粒球生成新边或新正样本。

### 4.8 Experimental Hypotheses

- H1：边界节点关系应降权，避免生成不真实正关系。
- H2：relation augmentation 应保持局部分布，而不是只追求更多正样本。
- H3：distribution-preserving ball constraint 比 hard sample generation 更适合 GCL。

### 4.9 Implementation Difficulty

- 代码改动成本：高
- 计算成本：中到高
- 复现风险：高
- 是否适合下一轮实验：暂缓

### 4.10 Important Formula or Pseudocode

```text
GEB-SMOTE: sample near granular-ellipsoid boundary
GED-SMOTE: sample to preserve granular-ellipsoid distribution
```

### 4.11 Final Value Judgment

- 价值：低到中
- 理由：它不直接服务 GRACE，但对“增强/采样不能制造不真实结构”有警示意义。

## Paper: Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training

### 4.1 Basic Information

- 年份：2025，PDF 显示 AAAI copyright；正式卷期 TODO
- 领域：GNN scalability / graph coarsening / node classification
- 任务：用监督粒球粗化图，再训练 GNN 做节点分类
- 数据集：Cora、Citeseer、Pubmed、Co-CS、Co-Phy
- 是否节点级：是
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/mxjun17/Granular-Ball-Node-Classifction`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

GNN 在大图上训练成本高，已有 graph coarsening 方法常需要预设 coarsening ratio。SGBGC 用粒球把原图压缩成 coarsened graph，并让每个粒球作为 super node 训练 GNN。

### 4.3 Deep Model Interface

- Deep model family：GCN / GAT / APPNP 等 GNN。
- Backbone / encoder / decoder / classifier：下游 GNN classifier。
- Input data：原图节点特征、邻接和标签。
- Representation space used by the model：coarsened graph 表示。
- Downstream task：节点分类。
- Training setting：supervised。
- End-to-end training：否，先粒球粗化，再训练 GNN。

### 4.4 Granular Ball Integration

- Granular object：节点集合，对应 coarsened graph 的 super node。
- Integration position：graph coarsening / super node / coarsened adjacency。
- Granular-ball generation space：原始图结构 + 节点标签。
- Update schedule：预处理式迭代 split；可 adaptive，也可按 coarsening ratio 非自适应运行。
- Uses labels or purity：是，purity 明确依赖节点标签。
- Node-to-ball relation：projection matrix `P`。
- Ball-to-ball relation：若原图中跨两个粒球存在边，则构造 super edge。
- Ball quality / confidence / weight：purity threshold。
- Graph or topology preservation：super node feature 用原节点特征均值，super edge 保留跨球连接。
- Local / global / multi-granularity role：用较粗粒度保留主要图结构并降低训练规模。

### 4.5 Training and Objective

先用 SGBGC 得到 `Ggb`，再在 `Ggb` 上训练 GNN。super node 标签由 `P^T Y` 和 majority / argmax 确定，loss 为 cross-entropy。

### 4.6 Mechanism Claimed by the Paper

论文报告可显著压缩图规模并提升训练效率；噪声注入实验显示一定鲁棒性。依据来自 PDF 摘要、方法和 conclusion。复现稳定性：Unknown。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 coarsened graph、node-to-ball projection、ball-to-ball relation 和 quality diagnostic。迁移到 GRACE 时只能使用无标签版本；监督 purity 和 super node label 不能进入无监督预训练。

### 4.8 Risks if Used Directly

最大风险是监督信息泄漏；其次是 graph coarsening 会丢失节点级细节，可能破坏 GRACE 的节点级 instance discrimination。super node 也容易诱发过强 prototype pulling。

### 4.9 Relation to Existing Failed Experiments

TODO: `02_experiment_history_and_lessons.md` 缺失，无法确认是否解释已有失败实验。该文支持“粒球作为结构摘要/图去噪信号”，不支持“same-ball hard positive”。

### 4.10 Experimental Hypotheses

- H1：无标签 node-to-ball projection 可以作为 GRACE 的结构诊断信号。
- H2：ball-to-ball coarsened adjacency 在噪声图上可能比原始边更鲁棒。
- H3：如果 coarsening ratio 过高，节点级对比学习性能会下降。

### 4.11 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：暂缓，先做无标签粗化诊断

### 4.12 Important Formula or Pseudocode

```text
P: node-to-ball projection
X_gb = P^T X
Y_gb = majority(P^T Y)   # only for supervised diagnosis
A_gb[i,j] = 1 if any original edge connects GB_i and GB_j
train GNN(A_gb, X_gb) with CE loss
```

### 4.13 Final Value Judgment

- 价值：中到高
- 理由：对 graph coarsening 和 ball-to-ball relation 很有启发，但监督 purity 与无监督 GRACE 冲突明显。

## Paper: GBRIP: Granular Ball Representation for Imbalanced Partial Label Learning

### 4.1 Basic Information

- 年份：PDF 未确认正式发表年份，TODO
- 领域：partial label learning / imbalanced learning
- 任务：imbalanced partial label classification
- 数据集：CIFAR10-LT、CIFAR100-LT、Lost、Bird Song、Soccer Player、Yahoo! News、SUN397 等
- 是否节点级：否
- 是否图级：否
- 是否有代码：PDF 写明 code will be available in supplemental materials；公开链接 Unknown
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

partial label learning 中候选标签不唯一，long-tail / imbalance 会影响 label confidence。GBRIP 用 coarse-grained granular ball representation 和 multi-center loss 缓解类间与类内不平衡。

### 4.3 Deep Model Interface

- Deep model family：神经分类器 + label disambiguation。
- Backbone / encoder / decoder / classifier：classifier，具体 backbone TODO。
- Input data：带候选标签的图像/特征数据。
- Representation space used by the model：GB-based feature space。
- Downstream task：partial label classification。
- Training setting：weakly supervised。
- End-to-end training：Unknown。

### 4.4 Granular Ball Integration

- Granular object：feature space 中的 coarse-grained sub-GB。
- Integration position：label confidence construction、multi-center loss、feature-space rebalancing。
- Granular-ball generation space：特征空间。
- Update schedule：基于 unsupervised 2NN clustering criterion 生成近似均衡 sub-GB。
- Uses labels or purity：候选标签用于 label confidence；粒球生成描述为 unsupervised 2NN。
- Node-to-ball relation：样本到 sub-GB / center 的关系。
- Ball-to-ball relation：GB graph representation。
- Ball quality / confidence / weight：label confidence matrix、GB graph weight、center relation。
- Graph or topology preservation：通过 GB graph 捕获类内/类间不平衡。
- Local / global / multi-granularity role：用粗粒度空间降低 outlier 和 confusing feature 影响。

### 4.5 Training and Objective

GBRIP 先用粒球构造更平衡的 feature space，再估计 label confidence，并用 cross-entropy + multi-center loss 强化样本和球中心关系。

### 4.6 Mechanism Claimed by the Paper

论文称粒球表示能缓解 confusing features、outliers 和类内/类间不平衡，MCL 让表示更贴近对应中心。依据来自摘要、方法和 conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 sample-to-center confidence、multi-center relation、imbalance-aware ball graph。适合转成 node-to-ball soft assignment 或 confidence weighting；不适合直接迁移 label confidence matrix。

### 4.8 Risks if Used Directly

任务是 partial label learning，不是无监督 GCL；label disambiguation 目标和 GRACE instance contrast 不同。multi-center loss 若过强，可能变成 prototype pulling。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“粒球中心可提供 soft relation / confidence”，但提醒必须避免过强 center loss。

### 4.10 Experimental Hypotheses

- H1：node-to-ball center distance 可作为 same-ball relation 的置信度。
- H2：多中心关系比单 prototype 更适合避免过平滑。
- H3：在类别不平衡或低同质图上，confidence weighting 比 hard membership 更稳。

### 4.11 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：修改后可尝试，重点是 confidence 而不是 center pulling

### 4.12 Important Formula or Pseudocode

```text
GBs = unsupervised_2NN_cluster(features)
P_label = estimate_label_confidence(GB graph, candidate_labels)
loss = CE(pred, P_label) + lambda * multi_center_loss(features, GB_centers)
```

### 4.13 Final Value Judgment

- 价值：中
- 理由：对 node-to-ball confidence 有启发，但弱监督 partial-label 目标不能直接转为 GRACE 训练目标。

## Paper: Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary

### 4.1 Basic Information

- 年份：2024 arXiv / 2025 AAAI copyright，具体正式信息 TODO
- 领域：NLP / open intent classification
- 任务：已知意图分类 + 未知意图检测
- 数据集：StackOverflow、SNIPS、BANKING
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/Liyanhuaa/MOGB`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

已有 open intent boundary 方法假设每个已知类近似单个紧致球形区域，但真实意图分布可能非球形且类内存在 open space risk。MOGB 用多粒度粒球边界刻画类内子结构。

### 4.3 Deep Model Interface

- Deep model family：BERT feature extractor + nearest sub-centroid classifier。
- Backbone / encoder / decoder / classifier：BERT + nearest sub-centroid classifier。
- Input data：文本意图样本。
- Representation space used by the model：BERT intent representation。
- Downstream task：open intent classification。
- Training setting：supervised。
- End-to-end training：partial / Unknown。

### 4.4 Granular Ball Integration

- Granular object：每个已知意图类中的多粒度 sub-part。
- Integration position：adaptive GB clustering + multi-granularity decision boundary。
- Granular-ball generation space：class-wise intent embedding。
- Update schedule：hierarchical representation learning 中交替 adaptive GB clustering 与 nearest sub-centroid classification。
- Uses labels or purity：是，按已知 intent class 内生成边界。
- Node-to-ball relation：样本到最近 sub-centroid / ball 的归属。
- Ball-to-ball relation：未作为主机制。
- Ball quality / confidence / weight：centroid、radius、边界内外判定。
- Graph or topology preservation：不涉及图拓扑。
- Local / global / multi-granularity role：多个 class-wise ball 表示复杂类内语义结构。

### 4.5 Training and Objective

先用 BERT 得到特征，再在已知类内生成多个粒球；nearest sub-centroid classifier 学细粒度表示，推理时用多个 center/radius 决定 known / unknown。

### 4.6 Mechanism Claimed by the Paper

论文称多粒度边界能同时降低 empirical risk 与 open space risk，比单球边界更适合非高斯/非球形意图分布。依据来自摘要、introduction 和 conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 boundary-aware confidence、半径内外置信度、多子中心而非单 prototype。适合做 GRACE 的 evaluation-only diagnosis 或 confidence weighting；不能使用 class-wise 已知标签生成预训练信号。

### 4.8 Risks if Used Directly

监督 class-wise boundary 与无监督节点级 GCL 不匹配；hard boundary 容易把边界节点错误处理为负例或低置信样本。多中心边界也可能引入大量超参。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“边界节点降权”和“多中心比单 prototype 更合理”，不支持 hard same-ball positive。

### 4.10 Experimental Hypotheses

- H1：边界附近节点对的 same-ball relation 应降权。
- H2：多个局部 center 比单一 ball prototype 更适合复杂类内结构。
- H3：GRACE 的粒球 confidence 可由 center distance / radius margin 构成。

### 4.11 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：暂缓，先转化为 boundary-aware diagnostic

### 4.12 Important Formula or Pseudocode

```text
h = BERT(text)
GBs_c = adaptive_GB_cluster(h where label=c)
train nearest_sub_centroid_classifier(h, GB_centers)
known if distance(h, nearest_center) <= radius(center)
```

### 4.13 Final Value Judgment

- 价值：中
- 理由：对 boundary confidence 很有价值，但监督开放集任务与 GRACE 主任务距离较远。

## Paper: Graph-based Representation for Image based on Granular-ball

### 4.1 Basic Information

- 年份：PDF 未确认正式年份，TODO
- 领域：computer vision / graph representation / GNN
- 任务：把图像表示为粒球图，再做图像分类
- 数据集：MNIST、CIFAR-10、ImageNet 子集
- 是否节点级：图像内部 structural block 级，不是图数据节点分类
- 是否图级：是，每张图像变成一个 graph
- 是否有代码：是，PDF 标注 `https://github.com/ddw2AIGROUP2CQUPT/GRIG`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

像素级图像处理效率、鲁棒性和可解释性不足。该文用改进粒球方法把图像重组为结构块图，每个粒球/矩形区域作为节点，邻接区域构成边。

### 4.3 Deep Model Interface

- Deep model family：GAT / GNN for image classification。
- Backbone / encoder / decoder / classifier：GAT；对比 ResNet18、DenseNet-121 等。
- Input data：图像。
- Representation space used by the model：image graph representation。
- Downstream task：图像分类。
- Training setting：supervised。
- End-to-end training：否，粒球图构建是预处理。

### 4.4 Granular Ball Integration

- Granular object：图像中的矩形结构块。
- Integration position：graph construction / image-to-graph representation。
- Granular-ball generation space：原始像素/灰度梯度。
- Update schedule：每张图像自适应区域扩张。
- Uses labels or purity：使用像素区域内部 purity，不是类别标签 purity。
- Node-to-ball relation：像素归入结构块。
- Ball-to-ball relation：共享区域/邻接结构形成边。
- Ball quality / confidence / weight：purity、variance、mean feature。
- Graph or topology preservation：图像结构块邻接图。
- Local / global / multi-granularity role：用结构块替代像素，减少冗余。

### 4.5 Training and Objective

先把图像转为 graph，再训练 GAT。论文在 MNIST、CIFAR-10、ImageNet 子集上验证，不以 SOTA 为主要目标。

### 4.6 Mechanism Claimed by the Paper

粒球图能减少像素级冗余，并保留结构块关系；论文强调效率、鲁棒性和 understandability。具体 claim 仅作为该任务证据，不作为 GRACE 结论。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是结构块图、局部区域 purity / variance、边界邻接。对 GRACE 的启发是 structure-aware augmentation，但不能替换已有图结构和评估协议。

### 4.8 Risks if Used Directly

这是 image-to-graph 任务，不是已有图上的节点级 GCL；粒球生成依赖像素拓扑，不能直接迁移到 citation graph。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“增强应保留结构块”，不支持改变 baseline evaluation。

### 4.10 Experimental Hypotheses

- H1：对节点特征 mask 时，可借鉴 structure-aware 思路，避免破坏高置信局部结构。
- H2：节点局部邻域也可用 variance / purity-like 指标估计稳定性。
- H3：将原图完全替换为粒球图会损失节点级评估细节。

### 4.11 Implementation Difficulty

- 代码改动成本：高
- 计算成本：中
- 复现风险：高
- 是否适合下一轮实验：否，作为增强启发

### 4.12 Important Formula or Pseudocode

```text
image -> adaptive rectangular regions
region node = (center, size, purity, variance, mean feature)
edge = adjacent/shared-boundary regions
train GAT on image graph
```

### 4.13 Final Value Judgment

- 价值：低到中
- 理由：说明粒球可做结构化输入，但与节点级 GRACE 距离较远。

## Paper: EvoSampling: A Granular Ball-based Evolutionary Hybrid Sampling with Knowledge Transfer for Imbalanced Learning

### 4.1 Basic Information

- 年份：arXiv 编号 `2412.10461v1`，正式发表信息 Unknown
- 领域：imbalanced learning / evolutionary computation / sampling
- 任务：混合过采样与欠采样，提升不平衡分类
- 数据集：20 个 KEEL imbalanced datasets
- 是否节点级：否
- 是否图级：否
- 是否有代码：PDF 可读部分未确认公开代码
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

不平衡分类中，过采样可能生成噪声样本，欠采样可能误删重要多数类样本。EvoSampling 用 GP + multi-task learning 生成少数类样本，再用粒球多粒度欠采样去除低质量实例。

### 4.3 Granular Ball Mechanism

- Granular object：oversampled training data 上的多粒度 ball。
- Shape assumption：ball。
- Supervision setting：supervised imbalanced classification。
- Depends on label purity：Unknown。
- Center definition：GB center。
- Radius / shape / density definition：TODO。
- Ball quality / confidence signal：低质量实例删除，具体质量函数需进一步精读。
- Local structure represented：多数类/少数类局部分布与噪声区域。

### 4.4 Generation / Splitting / Stopping

- Generation method：GB initialization + GB generation + GB-based undersampling。
- Split / merge rule：TODO。
- Stopping criterion：TODO。
- Granularity control：多粒度 undersampling。
- Overlap handling：Unknown。
- Outlier / noise handling：去除噪声或低质量实例。
- Complexity / scalability claim：基于 20 个 KEEL 数据集验证，复杂度细节 TODO。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

可迁移的是“低质量实例/边界实例应降权或剔除”的思想。对 GRACE 更适合作为 confidence / denoising 机制，而不是构造新正样本。

### 4.6 Risks if Used Directly

这是 tabular imbalanced sampling，不是图对比学习；过采样思想直接迁移到图节点会改变数据分布，风险高。

### 4.7 Relation to Existing Failed Experiments

TODO。该文支持“粒球更适合作为噪声/低质量结构识别器”，不支持简单复制采样流程。

### 4.8 Experimental Hypotheses

- H1：低置信粒球中的节点对不应强制拉近。
- H2：粒球多粒度欠采样思想可转化为边/节点对 loss 降权。
- H3：图数据不应直接过采样节点，除非任务是 imbalanced node classification。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：否，作为 denoising 启发

### 4.10 Important Formula or Pseudocode

```text
X_balanced = GP_multitask_oversampling(X_imbalanced)
GBs = generate_GBs(X_balanced)
X_clean = GB_based_undersampling(GBs)
train classifier on X_clean
```

### 4.11 Final Value Judgment

- 价值：低到中
- 理由：非深度、非图 GCL，但强化了“粒球用于低质量样本识别”的证据链。

## Paper: Granular Ball Guided Masking: Structure-aware Data Augmentation

### 4.1 Basic Information

- 年份：PDF 标注 2025，正式发表信息 Unknown
- 领域：computer vision / data augmentation / CNN / Vision Transformer
- 任务：图像分类、masked image reconstruction、image tampering detection
- 数据集：CIFAR-10、CIFAR-100、ImageNet-1K、ImageNet-100、ADCD tampering detection protocol
- 是否节点级：否
- 是否图级：否
- 是否有代码：PDF 可读部分未确认
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

随机或启发式 masking 可能丢弃关键语义区域。GBGM 用粒球引导 coarse-to-fine hierarchical masking，保留结构重要区域，抑制冗余背景。

### 4.3 Deep Model Interface

- Deep model family：CNN、Vision Transformer、MAE、tampering detector。
- Backbone / encoder / decoder / classifier：EfficientNet、Swin-T、ViT-Base MAE、ADCDNet 等。
- Input data：图像。
- Representation space used by the model：输入 patch / region。
- Downstream task：classification、masked reconstruction、tampering detection。
- Training setting：supervised / self-supervised reconstruction。
- End-to-end training：mask 作为 augmentation 策略接入，模型本身不因 GB 改结构。

### 4.4 Granular Ball Integration

- Granular object：图像局部区域/patch。
- Integration position：structure-aware masking / data augmentation。
- Granular-ball generation space：图像区域，coarse-to-fine 分层。
- Update schedule：每个输入生成 mask，训练时使用；推理用原图。
- Uses labels or purity：mask 由区域 purity / saliency-like score 引导，未见类别标签参与。
- Node-to-ball relation：patch / region 到粒球。
- Ball-to-ball relation：层级区域关系。
- Ball quality / confidence / weight：高结构重要区域保留，冗余区域 mask。
- Graph or topology preservation：保留目标区域和边界结构。
- Local / global / multi-granularity role：coarse-to-fine mask generation。

### 4.5 Training and Objective

GBGM 不替换网络结构，而是作为模型无关的 masking 策略接入 CNN/ViT/MAE/检测器训练。下游 loss 保持原任务定义。

### 4.6 Mechanism Claimed by the Paper

结构感知 mask 比 random erasing / GridMask 更少破坏语义；coarse-to-fine 过程保留目标区域和边界细节。依据来自摘要、实验和 conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 structure-aware augmentation、mask confidence、coarse-to-fine preservation。适合用于 GRACE 的 feature masking / edge dropping 强度控制。

### 4.8 Risks if Used Directly

图像 patch 的空间连续性不同于图节点邻接；若直接按粒球保护所有 same-ball 边，可能降低 GCL 增强多样性。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“粒球控制增强强度”，与 AGENTS 中 soft structural constraint 偏好一致。

### 4.10 Experimental Hypotheses

- H1：粒球可用于 edge drop / feature mask 的结构感知采样。
- H2：弱视图保留高置信粒球，强视图扰动低置信区域，可能比随机增强稳。
- H3：若保护过强，GRACE loss 的 view diversity 会下降。

### 4.11 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：中
- 是否适合下一轮实验：是，作为 augmentation control 小实验

### 4.12 Important Formula or Pseudocode

```text
GBs = coarse_to_fine_regions(input)
score = region_importance_or_purity(GBs)
mask low-score redundant regions
train model on masked input; inference uses original input
```

### 4.13 Final Value Judgment

- 价值：高
- 理由：它直接支持“粒球作为增强保护/扰动控制信号”，比 hard positive 更贴合本项目偏好。

## Paper: AW-GBGAE: An Adaptive Weighted Graph Autoencoder Based on Granular-Balls for General Data Clustering

### 4.1 Basic Information

- 年份：2025，PDF 显示 TPAMI Vol. 47 No. 12
- 领域：deep clustering / graph autoencoder / graph construction
- 任务：无标签高维一般数据聚类
- 数据集：多种 clustering benchmark，具体清单 TODO
- 是否节点级：一般数据点作为 graph node
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/xjnine/AWGBGAE`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

许多无标签高维数据没有天然边结构且存在无关特征。AW-GBGAE 用 feature weighting 和 weighted granular-ball 构建图，再用 GAE 学表示并聚类。

### 4.3 Deep Model Interface

- Deep model family：GCN-based graph autoencoder。
- Backbone / encoder / decoder / classifier：GAE + K-means。
- Input data：无标签高维数据。
- Representation space used by the model：weighted graph 上的 GAE embedding。
- Downstream task：clustering。
- Training setting：unsupervised。
- End-to-end training：partial，WGB 图构建 + GAE 动态更新。

### 4.4 Granular Ball Integration

- Granular object：weighted granular-ball set。
- Integration position：weighted graph construction / edge generation。
- Granular-ball generation space：feature-weighted raw data。
- Update schedule：weighted splitting + intra/inter WGB edge construction + reconstruction-based dynamic update。
- Uses labels or purity：无监督，未见类别标签参与图构建。
- Node-to-ball relation：样本属于 WGB。
- Ball-to-ball relation：inter-WGB connections。
- Ball quality / confidence / weight：feature weights、weighted radius / volume、动态邻居数 `knei`。
- Graph or topology preservation：WGB 构造 directed/undirected weighted graph，作为 GAE reconstruction target。
- Local / global / multi-granularity role：intra-WGB 捕获局部，inter-WGB 捕获全局结构。

### 4.5 Training and Objective

先构造 WGBS，再用 intra-WGB 与 inter-WGB 连接生成 weighted graph；GAE 重构该图，动态调整邻居数，最终对 embedding `Z` 做 K-means。

### 4.6 Mechanism Claimed by the Paper

论文称 WGB 同时捕获局部细节和全局结构，feature weighting 缓解 irrelevant features，动态更新减少预训练依赖。依据来自摘要、contribution 和方法。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是无标签 weighted graph construction、intra/inter ball relation、feature-weighted confidence。适合做 ball-to-ball relation alignment、graph denoising、local structure preservation。

### 4.8 Risks if Used Directly

目标是 clustering + reconstruction，不是 contrastive；如果把 reconstructed graph 当真值边，可能引入错误结构。GAE 动态更新成本也较高。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“粒球做 graph construction / soft topology prior”，不支持 hard same-ball positive。

### 4.10 Experimental Hypotheses

- H1：WGB intra/inter relation 可转成 GRACE 的 soft structural constraint。
- H2：feature-weighted ball quality 比纯 membership 更稳。
- H3：动态更新粒球图可能比固定初始粒球更稳定，但计算成本更高。

### 4.11 Implementation Difficulty

- 代码改动成本：高
- 计算成本：中到高
- 复现风险：中
- 是否适合下一轮实验：是，但先做离线 relation diagnostic

### 4.12 Important Formula or Pseudocode

```text
WGBS = weighted_GB_split(X, feature_weights)
W = intra_WGB_edges(WGBS) + inter_WGB_edges(WGBS)
Z = GAE(X, W).embedding
update W by reconstruction / neighbor schedule
clusters = KMeans(Z)
```

### 4.13 Final Value Judgment

- 价值：高
- 理由：提供无监督粒球图构建和 intra/inter ball relation，最接近本项目的 soft topology 方向。

## Paper: Capturing Local and Global Information: Multi-view Graph Convolutional Network via Granular-ball Computing and Collaborative Matrix

### 4.1 Basic Information

- 年份：2026，PDF 显示 Expert Systems With Applications 296 (2026) 129057
- 领域：multi-view learning / GCN / semi-supervised classification
- 任务：多视图半监督分类
- 数据集：MSRC-v1、MNIST、BBCnews、BBCsports、NUS-WIDE、OutScene、UCI
- 是否节点级：样本作为节点的半监督分类
- 是否图级：否
- 是否有代码：PDF 可读部分未确认
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

多视图 GCN 需要高质量拓扑，kNN 固定 k 会带来噪声；简单融合多视图特征/拓扑也忽略卷积过程中的局部信息。GBCM-GCN 用粒球边界距离构造每个视图拓扑，并用 collaborative matrix 融合局部/全局信息。

### 4.3 Deep Model Interface

- Deep model family：multi-view GCN。
- Backbone / encoder / decoder / classifier：GCN + collaborative matrix。
- Input data：多视图样本特征。
- Representation space used by the model：各视图 GCN embedding。
- Downstream task：semi-supervised classification。
- Training setting：semi-supervised。
- End-to-end training：partial，拓扑构建不是普通端到端学习。

### 4.4 Granular Ball Integration

- Granular object：每个视图中无监督生成的 GB。
- Integration position：topology construction。
- Granular-ball generation space：每个 view 的样本特征。
- Update schedule：预处理式无监督 GB generation。
- Uses labels or purity：拓扑构建使用无监督 GB generation。
- Node-to-ball relation：样本到 view-specific GB。
- Ball-to-ball relation：minimum boundary distance。
- Ball quality / confidence / weight：boundary distance；负值表示球内连接，正值表示跨球连接。
- Graph or topology preservation：用 boundary distance 构造每个 view 的 topology。
- Local / global / multi-granularity role：负边界距离捕获局部，正边界距离捕获全局。

### 4.5 Training and Objective

先用 GB boundary distance 为每个 view 构图，再在 GCN convolution 后用不同维度卷积核提取 local/global 信息并学习 collaborative matrix，跨视图共享以增强一致性。loss 为 cross-entropy。

### 4.6 Mechanism Claimed by the Paper

论文称无监督 GB topology 可减少 k-value noise，并让 GCN 同时利用 local/global 信息。实验表明在多视图半监督数据集上优于 baselines。依据来自摘要、方法、conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 boundary-distance topology、local/global relation、view-shared relation consistency。适合 GRACE 的 weak/strong view relation alignment、ball-to-ball relation、graph denoising。

### 4.8 Risks if Used Directly

多视图半监督分类不是单图双增强 GCL；collaborative matrix 会显著改变模型结构，不宜先作为最小实验。跨视图共享矩阵不能直接等价于 GRACE 双 view。

### 4.9 Relation to Existing Failed Experiments

TODO。该文强烈支持 ball-to-ball soft topology，但不支持 hard same-ball positive。

### 4.10 Experimental Hypotheses

- H1：GB boundary distance 可作为 ball-to-ball confidence。
- H2：weak view 和 strong view 的 ball-to-ball relation 对齐比 node hard relation 稳。
- H3：无监督 GB topology 在噪声边场景下比 kNN 或原始 adjacency 更鲁棒。

### 4.11 Implementation Difficulty

- 代码改动成本：中到高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是，先实现 boundary-distance relation diagnostic

### 4.12 Important Formula or Pseudocode

```text
GBs_v = unsupervised_GB_generation(X_view_v)
topology_v = connect_by_min_boundary_distance(GBs_v)
H_v = GCN(X_v, topology_v)
T = collaborative_matrix(local_global_kernels(H_main_view))
H_v = T * H_v
```

### 4.13 Final Value Judgment

- 价值：高
- 理由：无监督 boundary distance 和 local/global relation 对本项目的 ball-to-ball soft relation 非常直接。

## Paper: Enhancing Graph Convolutional Networks with Progressive Granular Ball Sampling Fusion

### 4.1 Basic Information

- 年份：2024，PDF 显示 Information Sciences 676 (2024)
- 领域：GCN training / graph sampling / semi-supervised node classification
- 任务：用 progressive GB sampling 提高 GCN 训练效率和准确率
- 数据集：Cora、Citeseer、Pubmed、Chameleon、Texas、Cornell、Squirrel、Acm
- 是否节点级：是
- 是否图级：否
- 是否有代码：PDF 可读部分未确认
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

全图 GCN 训练慢且内存开销高，普通采样方法可能忽略图结构。PGBSF 用粒球采样把原图分为一组子图，并用 progressive / incremental training 与参数共享提升收敛速度。

### 4.3 Deep Model Interface

- Deep model family：GCN，可嵌入多种 GNN。
- Backbone / encoder / decoder / classifier：GCN-based learner。
- Input data：邻接矩阵 `A` 和节点属性矩阵 `X`。
- Representation space used by the model：GCN hidden representation。
- Downstream task：semi-supervised node classification。
- Training setting：semi-supervised。
- End-to-end training：GB sampling 后分阶段训练。

### 4.4 Granular Ball Integration

- Granular object：节点子集 / subgraph。
- Integration position：subgraph sampling / progressive minibatch graph construction。
- Granular-ball generation space：图节点集合及特征，使用 k-division / k-means 思路和 purity 质量控制。
- Update schedule：progressive，子图规模逐渐增加；后阶段继承前阶段权重。
- Uses labels or purity：使用 purity threshold，半监督设置下有标签风险。
- Node-to-ball relation：节点属于最近且半径覆盖的 GB。
- Ball-to-ball relation：通过 progressive subgraph union 间接体现。
- Ball quality / confidence / weight：purity、GB 数量、半径。
- Graph or topology preservation：子图保留原图部分邻接。
- Local / global / multi-granularity role：从小粒度子图逐步扩大训练范围。

### 4.5 Training and Objective

先生成多个 GB/subgraph，训练小规模子图得到权重，再把权重作为下一阶段更大子图训练初始化，实现 incremental GCN training。loss 为 cross-entropy。

### 4.6 Mechanism Claimed by the Paper

论文称 PGBSF 可降低时间复杂度、加快收敛、减小 loss variance，并在多个 label rate 下提升 GCN 表现。依据来自摘要、方法和结论。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 progressive granular subgraph、GB sampling、purity/quantity sensitivity。适合做训练效率诊断或 multi-scale view construction；若 purity 来自标签，不能进入无监督对比预训练。

### 4.8 Risks if Used Directly

PGBSF 是监督/半监督 GCN 训练框架，不是 contrastive pretraining。progressive 子图可能改变 GRACE 的负样本分布和 view consistency。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“粒球作为 sampling unit”，但不支持把 sampled same-ball 节点当 hard positives。

### 4.10 Experimental Hypotheses

- H1：粒球子图可作为 GRACE 的 multi-scale diagnostic view。
- H2：粒球数量和 purity/quality 对结果敏感，必须先做 sensitivity。
- H3：如果 subgraph sampling 改变负样本池，结果不可与原 GRACE 直接比较。

### 4.11 Implementation Difficulty

- 代码改动成本：高
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：暂缓，先做采样诊断

### 4.12 Important Formula or Pseudocode

```text
GBs = granular_ball_sampling(V, X, purity_threshold)
subgraphs = progressive_union(GBs)
for g_t in subgraphs:
    W_t = train_GCN(g_t, init=W_{t-1}, loss=CE)
```

### 4.13 Final Value Judgment

- 价值：中
- 理由：支持粒球作为训练/采样单元，但监督 purity 和训练协议变更使其不适合直接做 GRACE 主线。

## Paper: GAdaBoost: An Efficient and Robust AdaBoost Algorithm Based on Granular-ball Structure

### 4.1 Basic Information

- 年份：2025/2026，PDF 显示 Knowledge-Based Systems，具体卷期 TODO
- 领域：ensemble learning / robust boosting
- 任务：带标签噪声的多分类 boosting
- 数据集：some noisy datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：PDF 可读部分未确认
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

AdaBoost 对标签噪声和 outliers 敏感，且逐样本训练成本高。GAdaBoost 先做 data granulation，再让 boosting 阶段关注 information granules / GB，而不是单个样本。

### 4.3 Granular Ball Mechanism

- Granular object：information granules / granular balls。
- Shape assumption：ball。
- Supervision setting：supervised classification。
- Depends on label purity：Unknown，面向 noisy label classification。
- Center definition：TODO。
- Radius / shape / density definition：TODO。
- Ball quality / confidence signal：通过粒球压缩、边界样本关注、噪声样本抑制体现。
- Local structure represented：边界难样本、冗余样本和噪声样本。

### 4.4 Generation / Splitting / Stopping

- Generation method：GAdaBoost 两阶段框架；GAdaBoost.SA 将 GBG 与 GB-based SAMME 结合。
- Split / merge rule：TODO。
- Stopping criterion：TODO。
- Granularity control：通过 GBG 压缩数据。
- Overlap handling：Unknown。
- Outlier / noise handling：抑制 label noise 和 outliers。
- Complexity / scalability claim：减少冗余数据使用，提升效率。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

可迁移的是“粒球可抑制 noisy/outlier 样本权重，同时保留边界难样本”的思想。对 GRACE 可转成 loss weighting 或 low-confidence relation filtering。

### 4.6 Risks if Used Directly

AdaBoost 不是深度 GCL；其样本权重迭代机制和 contrastive loss 不同。若照搬“关注难样本”，可能强化噪声负样本。

### 4.7 Relation to Existing Failed Experiments

TODO。该文支持 robust weighting，不支持 negative filtering 作为唯一机制。

### 4.8 Experimental Hypotheses

- H1：低质量粒球中的对比项应降权，而不是完全删除。
- H2：边界难节点可保留但降低 hard pull/push 强度。
- H3：粒球权重机制应与原 GRACE loss 并行，不应替代 baseline evaluation。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：中
- 是否适合下一轮实验：否，作为 weighting 启发

### 4.10 Important Formula or Pseudocode

```text
GBs = granularize(training_data)
for boosting_round:
    train base learner on GBs
    update GB/sample weights with noise-aware rule
```

### 4.11 Final Value Judgment

- 价值：低到中
- 理由：非深度方法，但对 robust weighting 和 noisy label 风险有参考价值。

## Paper: GB-RVFL: Fusion of Randomized Neural Network and Granular Ball Computing

### 4.1 Basic Information

- 年份：2025，PDF 显示 Pattern Recognition 159 (2025)
- 领域：randomized neural network / robust classification
- 任务：分类、噪声鲁棒、可扩展训练
- 数据集：KEEL、UCI、NDC、ADNI、BreakHis breast cancer 等
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/mtanveer1/GB-RVFL`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

RVFL 需要对整个训练矩阵求逆，扩展性受限；同时逐样本等权处理容易受噪声和 outlier 影响。GB-RVFL 用粒球中心替代训练样本，GE-GB-RVFL 再加入 graph embedding 保留 GB 拓扑。

### 4.3 Deep Model Interface

- Deep model family：Random Vector Functional Link，浅层随机神经网络。
- Backbone / encoder / decoder / classifier：RVFL / GE-GB-RVFL。
- Input data：GB centers，而非全部样本。
- Representation space used by the model：GB center matrix / RVFL hidden space。
- Downstream task：classification。
- Training setting：supervised。
- End-to-end training：否，GB 生成 + closed-form solver。

### 4.4 Granular Ball Integration

- Granular object：训练样本集合的 GB，输入为中心。
- Integration position：input compression / GB center / topology graph。
- Granular-ball generation space：原始特征空间。
- Update schedule：预处理。
- Uses labels or purity：监督分类，粒球标签参与训练；具体 purity 依赖 TODO。
- Node-to-ball relation：样本到 GB。
- Ball-to-ball relation：GE-GB-RVFL 用 graph embedding 保留 GB centers 拓扑。
- Ball quality / confidence / weight：粗粒度降低噪声/outlier 影响。
- Graph or topology preservation：GE framework 保留 GB centers 的拓扑结构。
- Local / global / multi-granularity role：用 GB center 替代全样本，提高可扩展性。

### 4.5 Training and Objective

用 GB centers 构造 RVFL 输入矩阵，减少矩阵逆规模；GE-GB-RVFL 用 intrinsic / penalty graph embedding 保留拓扑结构。优化为输出层闭式解。

### 4.6 Mechanism Claimed by the Paper

论文称 GB centers 替代样本可提升 scalability，粗粒度提高对 noise/outlier 鲁棒性，GE 保留拓扑并增强解释性。依据来自摘要、方法和 conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 GB center compression、topology-preserving ball graph、coarse-granularity robustness。适合做 ball center diagnostic、ball graph regularization。

### 4.8 Risks if Used Directly

RVFL 是分类器，不是 GNN/GCL；GB center 替代样本会丢失节点级 instance discrimination。过度依赖中心可能导致 prototype pulling。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持“中心可做摘要”，但警示不能让 center 直接主导 contrastive objective。

### 4.10 Experimental Hypotheses

- H1：GB center 适合做局部结构摘要，不适合直接替代节点表示。
- H2：ball graph topology 可作为辅助 relation prior。
- H3：中心拉近强度必须小于原 GRACE node-level contrastive loss。

### 4.11 Implementation Difficulty

- 代码改动成本：中
- 计算成本：低到中
- 复现风险：低到中
- 是否适合下一轮实验：暂缓，作为 center/graph diagnostic

### 4.12 Important Formula or Pseudocode

```text
GBs = generate_GBs(X_train)
O = centers(GBs)
H = random_features(O)
beta = closed_form_solve([O, H], labels_of_GBs)
GE_beta = solve_with_graph_embedding(GB_center_graph)
```

### 4.13 Final Value Judgment

- 价值：中
- 理由：对 GB center 和 topology-preserving ball graph 有启发，但分类器目标与 GRACE 差距较大。

## Paper: GBExplainer: Generalizing Adaptive-granularity Explanations with Granular Balls for Graph Neural Networks

### 4.1 Basic Information

- 年份：reviewer PDF，正式发表信息 Unknown
- 领域：GNN explainability / graph classification
- 任务：为 GNN prediction 生成 adaptive-granularity explanation
- 数据集：MUTAG、Mutagenicity、NCI1、IMDB-BINARY、ENZYMES
- 是否节点级：解释内部包含节点/边，但任务是图分类解释
- 是否图级：是
- 是否有代码：PDF 标注匿名链接 `https://anonymous.4open.science/r/GBExplainer-32E6/`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

已有 GNN explainer 常用固定节点数或固定比例生成解释子图，难以适应不同图的最优解释规模。GBExplainer 用粒球做 global-to-local subgraph splitting，结合结构级和边级解释。

### 4.3 Deep Model Interface

- Deep model family：model-independent GNN explainer。
- Backbone / encoder / decoder / classifier：已训练 GNN；解释器后处理。
- Input data：已训练 GNN、图结构、节点特征。
- Representation space used by the model：图结构重要性 / granular-ball graph。
- Downstream task：graph classification explanation。
- Training setting：post-hoc explanation。
- End-to-end training：否。

### 4.4 Granular Ball Integration

- Granular object：图中的多粒度子结构。
- Integration position：explanation subgraph generation。
- Granular-ball generation space：原图结构 + importance representation。
- Update schedule：global-to-local splitting。
- Uses labels or purity：解释依赖训练好模型预测，非无监督 GCL 训练信号。
- Node-to-ball relation：节点/边归入解释粒球。
- Ball-to-ball relation：granular-ball graph。
- Ball quality / confidence / weight：information gain、coarse structural importance、fine edge importance。
- Graph or topology preservation：保留解释性子结构。
- Local / global / multi-granularity role：先结构级粗解释，再球内边级细解释。

### 4.5 Training and Objective

先拟合图分布生成多粒度粒球图，再构造结构级解释，最后在粒球内部给出边级细粒度解释。评价使用 fidelity / stability 等解释指标。

### 4.6 Mechanism Claimed by the Paper

论文声称无需固定解释子图规模，能生成更稳定的多粒度解释，并在多个图分类数据集上优于 baselines。依据来自摘要、方法和 conclusion。

### 4.7 Transferable Signals for GRACE + Granular Ball

可迁移信号是 information-gain splitting、coarse-to-fine structural importance、edge-level confidence。适合作训练后诊断 / 可视化，检查 ball relation 是否捕获真实结构。

### 4.8 Risks if Used Directly

解释方法依赖已训练模型预测，若用于训练过程会形成循环依赖；图分类解释和节点级对比目标不一致。

### 4.9 Relation to Existing Failed Experiments

TODO。该文支持把粒球用于机制诊断和可视化，尤其检查 ball relation 是否对应重要结构。

### 4.10 Experimental Hypotheses

- H1：粒球 relation 可用 GNN explanation 指标做后验诊断。
- H2：高 information-gain ball 内的节点关系更适合作为 soft constraint。
- H3：训练时使用 explainer 代价太高，适合 evaluation-only。

### 4.11 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中到高
- 复现风险：高
- 是否适合下一轮实验：暂缓，作为可视化/诊断工具

### 4.12 Important Formula or Pseudocode

```text
importance = measure_GNN_prediction_change(G)
GB_graph = global_to_local_split(G, information_gain)
explanation = coarse_structure(GB_graph) + fine_edges(within_GBs)
```

### 4.13 Final Value Judgment

- 价值：中
- 理由：对“粒球作为解释和诊断工具”有价值，但不宜作为训练主线。

## Paper: MGNR: A Multi-Granularity Neighbor Relationship and Its Application in KNN Classification and Clustering Methods

### 4.1 Basic Information

- 年份：PDF 显示 TPAMI，具体年份 TODO
- 领域：multi-granularity neighbor / KNN / clustering
- 任务：改进 KNN 分类和 density-based clustering
- 数据集：synthetic and real datasets，具体清单 TODO
- 是否节点级：否
- 是否图级：否
- 是否有代码：是，PDF 标注 `https://github.com/xjnine/MGNR`
- 代码是否可直接复现：Unknown

### 4.2 Core Problem

KNN 和密度聚类常依赖单一手动 `k` 或固定邻域，面对多密度、多粒度数据会欠拟合或密度估计偏差。MGNR 用粒球构建多粒度邻居关系。

### 4.3 Granular Ball Mechanism

- Granular object：不同 granularity layer 上的数据表示。
- Shape assumption：ball。
- Supervision setting：KNN 分类为 supervised，聚类为 unsupervised。
- Depends on label purity：分类部分可能依赖标签；聚类部分不依赖。
- Center definition：GB center。
- Radius / shape / density definition：density level、NARD、MGK / MGN。
- Ball quality / confidence signal：density-aware multi-granularity neighbor。
- Local structure represented：不同密度区域的自适应邻居尺度。

### 4.4 Generation / Splitting / Stopping

- Generation method：用 GBC 构建 multi-granularity dataset。
- Split / merge rule：TODO。
- Stopping criterion：TODO。
- Granularity control：多个 granularity layers。
- Overlap handling：Unknown。
- Outlier / noise handling：通过 density-aware neighbor 降低固定 k 的密度偏差。
- Complexity / scalability claim：论文称提升 KNN/聚类效率，尤其大数据集。
- Stability claim：Unknown。

### 4.5 Transferable Signals for GRACE + Granular Ball

MGNR 对本项目最重要的是“不同节点/区域不应共享同一个固定邻域尺度”。可转化为 adaptive node-to-ball / ball-to-ball relation，以及 density-aware confidence。

### 4.6 Risks if Used Directly

这是 KNN/聚类算法，不是深度 GCL；邻居关系可作为诊断或先验，但不能直接替代 GRACE 的 contrastive pair 定义。

### 4.7 Relation to Existing Failed Experiments

TODO。该文支持避免固定 ball number / fixed k，并提醒 density bias 风险。

### 4.8 Experimental Hypotheses

- H1：不同密度区域应使用不同粒球半径或邻居阈值。
- H2：density-aware ball relation 比固定 same-ball membership 更稳。
- H3：固定粒球数量可能掩盖数据集间密度差异。

### 4.9 Implementation Difficulty

- 代码改动成本：中
- 计算成本：中
- 复现风险：中
- 是否适合下一轮实验：是，作为 density-aware confidence diagnostic

### 4.10 Important Formula or Pseudocode

```text
DG = {D_l: l in granularity_layers}
MGK = adaptive_neighbors_by_density_and_granularity(DG)
NARD = relative_density(MGN)
```

### 4.11 Final Value Judgment

- 价值：中到高
- 理由：虽然非深度，但它直接支持 adaptive multi-granularity relation 和 density-aware confidence。

## 5. Literature to Experiment Mapping

| 可实验方向 | 支持文献 | 可迁移点 | 当前判断 |
|---|---|---|---|
| weak-view ball relation teacher | CLSA、GRACE | 弱视图生成更可信关系，指导强视图 | 高优先级 |
| soft ball-to-ball relation | MGBCC、GFSAF | 用粒球连接关系表达局部结构，不直接用 hard node label | 高优先级 |
| confidence-weighted local structure | GFSAF、CLSA | 低质量强增强或不稳定粒球降低权重 | 高优先级 |
| graph augmentation semantic guard | GRACE、SPARROW、CLSA | 增强可能破坏语义，需要局部结构约束或弱 teacher | 高优先级 |
| projector / encoder space diagnostic | SPARROW、CLSA | projector 前后表示可形成关系诊断 | 中优先级 |
| graph denoising | GFSAF、MGBCC | 从 noise stripping / false negative 缓解借鉴动机 | 中优先级 |
| hypersphere / angle-space ball diagnostic | GBmooer、SPARROW | 对表示做归一化或角距离诊断，检查哪一层表示更适合稳定建球 | 高优先级，先做诊断 |
| contrastive warm-up before GB generation | CGBC、GBmooer、GRACE、SPARROW | 先让 encoder 表示具备基本语义结构，再生成粒球，降低训练早期划分波动 | 高优先级，先做诊断 |
| noisy-label / noisy-relation robust GB training | CGBC、GBmooer、GFSAF、GAdaBoost、EvoSampling | 粒球可作为低质量标签、低质量关系或 noisy sample 的缓冲层和降权依据 | 中高优先级，优先噪声场景 |
| prototype / cluster pulling | MGBCC | 粒球中心可作为局部原型 | 中低优先级，需防止过强 |
| hard same-ball positives | MGBCC、GFSAF 提供粒球正负关系，但不直接支持 GRACE hard migration | 迁移风险高 | 低优先级 / 警示 |
| supervised purity / majority-vote GB label | CGBC | 在有监督 noisy-label 检测中有效，但依赖标签 | 仅诊断 / 警示，不可进入无监督预训练 |
| local smoothing / denoising as soft weight | GBmooer、GFSAF、GAdaBoost、EvoSampling | 把噪声影响限制在局部粒球；迁移到 GRACE 时只能做无标签 confidence 或 loss 降权 | 中优先级，禁止使用标签分布 |
| 简单 negative filtering | MGBCC、GFSAF 都不是只做过滤 | 单独过滤监督信号弱 | 低优先级 |
| stable GB generation | GBG++、Efficient and Adaptive GB Generation | 粒球生成稳定性、outlier detection、adaptive split、overlap control | 高优先级，先做诊断 |
| label-free confidence proxy | GB classifiers、GB-DP、Ball k-means | 半径、密度、大小、中心距离、stable/active area 可转成无监督 confidence | 高优先级 |
| boundary-aware relation weighting | Ball k-means、GEB/GED-SMOTE、GEC | 球边界/active area/非球形区域更容易产生错误关系，应降权 | 高优先级 |
| density-based robust local prior | GB-DP | GB density 和 center distance 可作为无标签局部结构强度 | 中高优先级 |
| shape-aware granular modeling | GEC、GEB/GED-SMOTE | 球形假设可能不适配 elongated / irregular local structure | 暂缓，先诊断 |
| deep GB graph construction | AW-GBGAE、GBCM-GCN | 用 WGB intra/inter connection 或 boundary distance 构造 soft topology | 高优先级，适合转成 ball-to-ball relation diagnostic |
| structure-aware graph augmentation | GBGM、Graph-based Representation for Image based on Granular-ball | 粒球控制 masking / structure preservation，避免增强破坏语义 | 高优先级，适合 GRACE edge/feature augmentation control |
| graph coarsening / subgraph sampling | SGBGC、PGBSF | 粒球可作为 super node 或 progressive subgraph，但常依赖 purity / labels | 中优先级，只做无标签诊断或效率 ablation |
| node-to-ball confidence / center relation | GBRIP、GB-RVFL、MOGB | center distance、multi-center relation、radius margin 可转为 soft confidence | 中高优先级，需防止 prototype pulling |
| adaptive multi-granularity neighbor | MGNR、GBCM-GCN、AW-GBGAE | 不同密度区域应使用不同粒度和邻域尺度 | 高优先级，适合 density-aware relation weighting |
| deep GB explanation / diagnosis | GBExplainer | information gain、coarse-to-fine structure 可用于后验诊断和可视化 | 中优先级，不作为训练主 loss |
| robust weighting / low-quality filtering | GAdaBoost、EvoSampling、GB-RVFL | 粒球可识别 noisy/outlier/low-quality 样本，适合 loss 降权 | 中优先级，避免只做负样本过滤 |

### 5.1 Experiment Design Rules Derived from the Mapping

下一次实验应按以下顺序推进，不应直接实现复杂完整模型：

1. 先做粒球生成诊断：比较表示空间、粒球数量、半径、密度、跨 seed/epoch 稳定性。
2. 再做无训练协议破坏的最小辅助信号：只加小权重 soft relation / confidence / augmentation control，不改 split、evaluation、baseline encoder。
3. 优先测试噪声或强增强场景：如果 clean graph 只小幅涨点但噪声场景稳定，论文 claim 更容易成立。
4. 每个方向必须有 hard / soft / confidence ablation：证明收益来自粒球机制，而不是超参或增强强度变化。
5. 任何依赖标签 purity 的文献机制，只能转换为 label-free proxy，或仅在诊断中使用标签。
6. 受 CGBC 和 GBmooer 启发，生成粒球前应比较 `no warm-up`、`GRACE warm-up`、`EMA encoder`、`hypersphere normalization`、`angular distance` 等表示与距离设置；不要默认训练早期 embedding 可直接用于构造稳定粒球。

## 6. Recommended Research Directions

### 6.0 Recommended Execution Order

| 阶段 | 目标 | 推荐方向 | 进入下一阶段条件 |
|---|---|---|---|
| Stage 0 | 确认粒球信号是否稳定 | 方向 1、方向 5 | 找到稳定表示空间和无标签 confidence proxy |
| Stage 1 | 最小侵入训练验证 | 方向 2、方向 7、方向 8 | 不改 baseline protocol，多 seed 不退化，至少一个方向有稳定收益 |
| Stage 2 | 机制增强与 ablation | 方向 3、方向 4、方向 9 | soft / confidence 版本明显优于 hard membership |
| Stage 3 | 鲁棒性 claim | 方向 6 + 噪声边/特征实验 | 噪声场景稳定优于 GRACE，clean graph 不明显下降 |

如果 Stage 0 失败，即粒球划分在表示空间中不稳定，则不应继续实现复杂 loss；应回到粒球生成方法、warm-up、EMA 或表示空间选择。

### 方向 1：Stable / Adaptive / Angle-space Granular-ball Generation Diagnostic

- 核心假设：粒球关系是否有效，首先取决于粒球生成是否稳定；GBG++ 和自适应 GB generation 提示生成稳定性、overlap、outlier 和 ball quality 必须先诊断；CGBC 和 GBmooer 进一步提示对比 warm-up 后的表示、hypersphere normalization 和 angular distance 可能更适合构造粒球。
- 粒球角色：稳定局部结构单元、confidence source。
- 适合验证的数据集或场景：Cora、Citeseer、Pubmed；先在固定 encoder embedding 或 GRACE warm-up embedding 上做离线诊断。
- 最小实验设计：比较 k-means ball、division ball、adaptive ball、angular ball 的粒球数量、半径分布、average angular distance、节点覆盖、跨 seed/epoch 稳定性、与标签只用于诊断的一致性；同时比较 raw feature、no-warm-up encoder、GRACE warm-up encoder、EMA encoder 表示空间。
- 成功标准：找到稳定且无监督可用的粒球生成配置；跨 seed/epoch relation matrix 变化可控。
- 失败标准：粒球数量、成员和关系高度波动；无法形成稳定辅助信号。

### 方向 2：Weak-view Granular-ball Relation Teacher

- 核心假设：weak view 上生成的粒球关系比 strong view 更稳定，可指导 strong view 保持局部语义。
- 粒球角色：weak view relation teacher、soft structural constraint。
- 适合验证的数据集或场景：Cora、Citeseer、Pubmed；后续加入噪声边或强增强设置。
- 最小实验设计：保持 GRACE loss 不变，增加 `KL/CE/MSE relation alignment` 辅助项；weak relation stop-gradient。
- 成功标准：多数据集 mean 提升且超过 std 重叠范围，或在强增强/噪声场景下稳定改善。
- 失败标准：只在单 seed 或单数据集提升；std 重叠；主 loss 被破坏。

### 方向 3：Confidence-weighted Same-ball Soft Relation

- 核心假设：same-ball membership 只有在高置信粒球内才可作为弱正关系。
- 粒球角色：confidence signal、soft positive weight。
- 适合验证的数据集或场景：同质性较强和较弱的 citation graph 对比。
- 最小实验设计：用粒球半径、密度、大小、中心距离、边界 active-area 或跨 epoch 稳定性生成 confidence，对节点对关系加权。
- 成功标准：比 hard same-ball positive 更稳定；ablation 显示 confidence 有贡献。
- 失败标准：去掉 confidence 后效果相同或更好；粒球划分波动导致 loss 噪声。

### 方向 4：Ball-to-ball Local Structure Preservation

- 核心假设：ball-to-ball relation 比 node-to-node hard relation 更能表达鲁棒局部结构。
- 粒球角色：局部语义关系单元、robust local structure prior。
- 适合验证的数据集或场景：含噪图、边扰动较强的 GRACE setting。
- 最小实验设计：在两个 views 中构造粒球，计算 ball overlap / center similarity / shared-node relation，并对齐 relation matrix。
- 成功标准：噪声增强下优于 baseline；干净图上不显著退化。
- 失败标准：计算成本过高；ball relation 不稳定；提升小于 std。

### 方向 5：Encoder vs Projector Granular-ball Diagnostic

- 核心假设：encoder output 更适合生成语义粒球，projector output 更适合优化 contrastive loss。
- 粒球角色：诊断工具，不先作为主损失。
- 适合验证的数据集或场景：Cora、Citeseer、Pubmed。
- 最小实验设计：分别在 encoder output 和 projector output 上生成粒球，比较纯度、稳定性、与标签/邻接的一致性。
- 成功标准：明确找到更稳定的粒球生成空间。
- 失败标准：两者都不稳定或与下游表现无关。

### 方向 6：Granular-ball Denoising Auxiliary Loss

- 核心假设：粒球可识别低置信局部结构，帮助降低噪声边或噪声特征影响。
- 粒球角色：graph denoising signal、confidence signal。
- 适合验证的数据集或场景：人工加边/删边/特征噪声。
- 最小实验设计：基于粒球一致性为边或节点对赋权，不直接删除 evaluation 中的边。
- 成功标准：噪声场景下稳健提升，clean graph 不显著下降。
- 失败标准：只在手工噪声强度下有效，不能泛化。

### 方向 7：Boundary-distance Ball-to-ball Topology Prior

- 核心假设：GBCM-GCN 和 AW-GBGAE 提示无标签粒球边界距离、intra/inter ball connection 可以比固定 kNN 或原始边更稳，适合转为 GRACE 的 soft topology prior。
- 粒球角色：ball-to-ball relation、local/global structure prior。
- 适合验证的数据集或场景：Cora、Citeseer、Pubmed；人工加噪边场景优先。
- 最小实验设计：不改 evaluation，离线生成 ball-to-ball boundary-distance relation，只加入小权重 relation consistency 或作为诊断矩阵。
- 成功标准：噪声边场景稳定优于 GRACE；ablation 显示 boundary distance 比 same-ball hard label 更稳。
- 失败标准：relation matrix 跨 seed/epoch 不稳定；clean graph 上明显下降；计算成本超过收益。

### 方向 8：Granular-ball Guided Augmentation Control

- 核心假设：GBGM 支持 structure-aware masking，粒球更适合控制增强强度，而不是直接定义正样本。
- 粒球角色：augmentation guard、confidence signal、soft structural constraint。
- 适合验证的数据集或场景：GRACE 的 edge drop / feature mask 设置；强增强和弱增强对比。
- 最小实验设计：高置信粒球内边/特征降低 drop 概率，低置信区域保持或提高扰动；与随机增强保持相同总体 drop rate。
- 成功标准：相同 drop rate 下 mean/std 稳定优于随机增强，且 view diversity 不明显降低。
- 失败标准：只因降低增强强度涨点；强约束导致 representation collapse 或 baseline 不可比。

### 方向 9：Density-aware Node-to-ball Confidence

- 核心假设：MGNR、GBRIP、MOGB 和 GB-RVFL 都提示节点到中心的距离、半径 margin、密度层级和多中心关系比 hard membership 更可靠。
- 粒球角色：node-to-ball confidence、boundary-aware soft weight。
- 适合验证的数据集或场景：同质/异质 citation graph 对比；低同质或密度不均数据集更关键。
- 最小实验设计：构造 `confidence(i, ball)`，由 center distance、radius margin、ball density、ball size 和跨 epoch 稳定性组合；仅用于 loss weight。
- 成功标准：confidence ablation 明确优于 hard same-ball 和无 confidence 版本。
- 失败标准：confidence 指标与结果无关，或不同数据集需要完全不同规则。

## 7. Directions to Avoid or Deprioritize

- 不优先做 hard same-ball positive pulling：MGBCC/GFSAF 的粒球正负关系来自多视图聚类，不能直接证明 GRACE 节点 same-ball 就是语义正样本。
- 不优先只做 negative filtering：文献中粒球主要提供多粒度关系或 fuzzy relation，单独过滤负样本信号太弱。
- 不优先做强 prototype pulling：粒球中心可用，但直接拉向 prototype 有过平滑和塌缩风险。
- 不优先固定随意粒球数量：GBG++ 和 adaptive GB generation 表明粒球生成稳定性、overlap 和质量会显著影响后续学习。
- 不在无监督预训练中使用标签 purity：粒球分类论文中的 purity 是监督信号，只能作为诊断，不可进入 GRACE 预训练 loss。
- 不直接照搬 CGBC 的 supervised purity、majority-vote GB label 或 same-label GB center pulling：这些设计服务于有监督 noisy-label 智能合约检测，进入 GRACE 会造成 label leakage 或过强 prototype pulling。
- 不直接照搬 GBmooer 的球内 label distribution smoothing：该机制依赖训练标签，服务于文本噪声分类；在 GRACE 中只能转成无标签 confidence、局部降权或诊断指标。
- 不默认局部结构一定是球形：GEC 和 granular ellipsoid 文献提示 elongated / irregular local structure 会削弱球形粒球表达。
- 不直接照搬 SGBGC / PGBSF 中的监督 purity：它们对图粗化或子图采样有价值，但进入无监督 GRACE 会造成 label leakage。
- 不直接把 coarsened graph 替代原始图做主结果：这会改变节点级任务、负样本池和 baseline 协议，除非明确做效率/诊断实验。
- 不把 GB-RVFL / GBRIP 的 center loss 原样搬到 GRACE：中心可作为 confidence 或摘要，但过强 center pulling 会削弱 instance discrimination。
- 不把 GBExplainer 的后验解释信号当训练真值：解释器依赖已训练模型预测，更适合诊断和可视化。
- 不忽略多粒度密度差异：MGNR 提醒固定 k、固定 ball number 或固定半径可能在多密度图上产生系统偏差。
- 不优先重写 baseline evaluation：GRACE 的价值在于可比较节点级协议，不能为了涨点改变 split 或评估。
- 不把单数据集小幅提升写成 claim：必须有 mean/std、跨 seed、跨数据集或噪声场景支持。

## 8. Open Questions

- 粒球在 raw feature、encoder output、projector output、weak view output 哪个空间生成最稳定？
- weak view 上的粒球关系是否能稳定指导 strong view？
- same-ball membership 是否真的对应节点语义相似，还是只对应当前表示空间的局部聚集？
- ball-to-ball relation 是否比原图邻接更抗噪？
- 粒球信号更适合 clean graph，还是 noisy graph / strong augmentation graph？
- 粒球 confidence 应该来自半径、密度、大小、纯度、跨 epoch 稳定性，还是多指标组合？
- 无监督场景下，能否用 density、radius、overlap、center distance、stable/active area 替代监督 purity？
- 粒球生成方法之间的稳定性差异是否大于具体 loss 设计差异？
- GRACE contrastive warm-up 是否能显著提升粒球划分稳定性，还是粒球质量主要由原始特征/图结构决定？
- 能否构造 label-free purity proxy，替代 CGBC 中依赖标签的 `Purity(X)` 和 majority-vote GB label？
- GBmooer 使用的 hypersphere normalization、angular distance 和 average angular distance 能否在 GRACE warm-up embedding 上形成比欧氏距离更稳定的粒球 confidence？
- 球内 label distribution smoothing 能否在 GRACE 中改写成完全无标签的 local uncertainty / loss weight，而不引入 label leakage？
- 图节点表示空间中的局部结构是否满足球形假设，还是需要各向异性/椭球诊断？
- 粒球关系应对齐 node-level similarity matrix、ball center matrix，还是 node-to-ball assignment distribution？
- MGBCC/GFSAF 的多视图粒球机制迁移到单图双增强视图时，哪些定义需要软化？
- SGBGC / PGBSF 的 graph coarsening 或 subgraph sampling 能否改成无标签版本，并且不破坏 GRACE 的节点级负样本分布？
- GBCM-GCN 的 boundary distance 是否能在 GRACE 双增强视图之间形成稳定 ball-to-ball teacher？
- AW-GBGAE 的 weighted intra/inter ball graph 是否比原始 adjacency 更适合作为噪声图中的 auxiliary topology？
- GBGM 式 structure-aware masking 在图上应控制 edge drop、feature mask，还是二者同时控制？
- MGNR 的 density-aware neighbor 是否能解决固定粒球数量和固定半径在不同图数据集上的不稳定？
- GBExplainer 能否用于后验验证“粒球关系是否覆盖 GNN 认为重要的结构”，而不进入训练？

### 8.1 Paper Claim Boundary Matrix

| 类型 | 当前可以怎么写 | 不能怎么写 | 还需要什么证据 |
|---|---|---|---|
| 已有文献直接支持 | GRACE 是节点级 GCL baseline；粒球可作为局部结构单元、图构建单元、采样单元、mask 控制信号或 confidence source；CGBC 和 GBmooer 支持 CL warm-up 与 GB robust training / local smoothing 可互补 | 粒球已经被证明能提升 GRACE | 在本项目代码和相同 evaluation protocol 下复现实验 |
| 合理但尚未验证 | weak-view ball relation、boundary-distance topology、confidence-weighted soft relation、contrastive warm-up 后构造粒球、angular ball confidence 可能比 hard membership 更稳 | 这些机制已经有效 | 多 seed、多数据集、ablation、噪声场景 |
| 已被文献警示 | hard same-ball positive、监督 purity、强 prototype pulling、固定粒度和单数据集 claim 风险高 | 只要 same-ball 就是语义正样本 | hard / soft / confidence 对照和稳定性诊断 |
| 仅适合诊断 | SGBGC 的监督 coarsening、PGBSF 的 purity sampling、MOGB 的 class-wise boundary、GBExplainer 的后验解释 | 直接作为无监督 GRACE loss | label-free proxy 或 evaluation-only 使用方式 |
| 可形成论文主线的方向 | 粒球作为 robust local structure prior、soft relation teacher、structure-aware augmentation guard、confidence weighting | 粒球作为新的硬监督标签 | 稳定超过 std 的结果，最好有 noisy graph / strong augmentation 支持 |

## 9. Key Takeaways for Future Experiments

### 9.1 最重要的 5 条文献启发

1. GRACE 应作为节点级 GCL baseline，粒球应先作为辅助局部结构信号，而不是替代原始 evaluation、split 或负样本协议。
2. CLSA 支持 weak-view teacher；GBCM-GCN 和 AW-GBGAE 进一步支持用无标签 ball-to-ball / intra-inter ball relation 做 soft topology teacher。
3. MGBCC、GFSAF、CGBC、GBmooer、GBRIP 和 GB-RVFL 都提示粒球更适合作为 instance 与 prototype/cluster 之间的中间粒度关系、confidence、局部平滑或鲁棒训练缓冲层，而不是 hard label。
4. GBGM、Graph-based Image Representation 和 SPARROW 共同提示增强可能破坏语义，粒球优先适合作为 structure-aware augmentation guard。
5. GBG++、adaptive GB generation、CGBC、GBmooer、MGNR、MOGB 和 GEC 提示粒球信号不应只有 membership，还应包含 warm-up stability、average angular distance、boundary distance、radius margin、density、shape assumption 等 confidence 信息。

### 9.2 最应该避免的 5 个误区

1. 不要把 same-ball membership 直接当 hard positive label，也不要把 ball center loss 做成过强 prototype pulling。
2. 不要只做 negative filtering；GAdaBoost、EvoSampling、GFSAF 更支持 confidence weighting / denoising，而不是单独过滤。
3. 不要把多视图聚类、CGBC supervised purity / majority-vote GB label、GBmooer label-distribution smoothing、监督图粗化、PGBSF 采样或 MOGB class-wise boundary 原样搬到 GRACE；其中很多机制依赖标签、purity 或任务特定协议。
4. 不要在未检查粒球生成稳定性、boundary distance、粒球数量、密度层级和边界节点置信度前，把粒球关系加入主 loss。
5. 不要使用监督 purity、改变 train/validation/test split、替换 baseline evaluation，或用单数据集、单 seed、小于 std 波动的提升写成论文 claim。

## 10. 2026-05-14 CGBC / GBmooer 复读补充

本次重新阅读的 PDF：

- `C:/Users/Admin/Downloads/2603.27734v1 (2).pdf`: Robust Smart Contract Vulnerability Detection via Contrastive Learning-Enhanced Granular-ball Training.
- `C:/Users/Admin/Downloads/ssrn-6022343 (2).pdf`: GBmooer: Granular-ball Text Label Smoothing with Hypersphere Contrastive Learning.

### 10.1 对当前 GB-GCL 项目的新增判断

1. 两篇论文都采用“先用 contrastive learning 改善表示，再让 GB 在较稳定表示空间中工作”的顺序。这直接支持本项目停止在训练早期不稳定 `z` 上强行构球，优先做 warm-up / EMA / hypersphere / angular-distance 的构球稳定性诊断。
2. 两篇论文中的 GB 主要承担鲁棒训练缓冲层：CGBC 用 GB center / majority label 缓解智能合约 noisy label，GBmooer 用球内 label distribution 做局部 smoothing。迁移到 GRACE 时，不能使用标签，只能转成 label-free confidence、局部降权、关系一致性或噪声场景诊断。
3. GBmooer 的 angle distance、unit hypersphere、average angular distance 对本项目很有启发：下一轮应比较 Euclidean / cosine / angular ball，在 encoder warm-up embedding 上记录球数、半径、平均角距离、跨 seed/epoch 成员稳定性和 relation matrix 稳定性。
4. CGBC 的 supervised purity、majority-vote GB label、same-label inter-GB center pulling 是有监督噪声分类机制，不能进入无监督 GRACE 预训练。它只能启发 label-free purity proxy，例如半径、密度、中心距离、边界距离、跨 epoch 稳定性、同视图/跨视图关系一致性。
5. 两篇论文更支持“noisy-label / noisy-relation robustness”叙事，而不是 clean graph 小幅涨点叙事。后续 RDCL 或 GB-FN 如果继续推进，应优先测试 noisy graph、强增强、边扰动或 label-free noisy-relation setting。

### 10.2 对实验 6 RDCL 的直接启发

- RDCL 目前 PubMed 正信号不能直接解释为“语义粒球有效”，因为 random / shuffled controls 已经复现大部分收益。
- 结合 GBmooer，RDCL 的更合理改法不是继续扫 `alpha/tau`，而是把 group relation loss 做成 confidence-weighted：高稳定、低半径、低平均角距离的球权重大；边界球、低稳定球权重小。
- 结合 CGBC，RDCL 可以转向“鲁棒局部关系正则”：在 noisy edge / strong augmentation 下测试真实分组是否优于 fixed / size-matched random / shuffled 分组。
- 如果 fixed 或 size-matched random partition 仍然等价于 kmeans / gb_v2，则论文 claim 应降级为 generic local relation regularizer，不能写成 GB semantic prior。

### 10.3 最小后续实验建议

| 优先级 | 实验 | 只改变什么 | 成功标准 | 失败标准 |
|---|---|---|---|---|
| P0 | Warm-up angular ball diagnostic | 不训练新方法，只比较 no-warmup / GRACE-warmup / EMA，cosine vs angular distance | 找到跨 seed/epoch 更稳定的球和 relation matrix | 所有构球方式都高度不稳定 |
| P1 | RDCL size-matched mechanism control | 固定 PubMed RDCL 主配置，只加 fixed / size-matched random / shuffled controls | true kmeans/gb_v2 比 controls 高至少约 +0.0015 mean F1Mi | controls 与 true groups 持平 |
| P2 | Confidence-weighted RDCL | 只把 RDCL group loss 按 radius / angular distance / stability 加权 | 加权版优于无加权和 shuffled control | 去掉 confidence 后一样好或更好 |
| P3 | Noisy graph RDCL | 固定 RDCL 机制，在 edge noise / strong drop setting 下比较 | true groups 在噪声下稳定优于 GRACE 和 shuffled control | clean/noisy 都无优势或只靠降低增强强度 |

### 10.4 本轮复读后的强约束

- 不要照搬 CGBC 的 `Purity(X)`、majority-vote label、same-label GB center pulling。
- 不要照搬 GBmooer 的球内 label distribution smoothing。
- 不要把 GB center 当强 prototype 拉近节点。
- 不要再只看 protected_precision；必须同时看 downstream control。
- 不要在没有 fixed / random / shuffled / size-matched controls 前，把 RDCL 写成 GB 语义机制。

## 11. 2026-05-19 CGBC / GBmooer skill 执行更新

本节是对 `C:/Users/Admin/Downloads/2603.27734v1 (2).pdf` 和 `C:/Users/Admin/Downloads/ssrn-6022343 (2).pdf` 的项目内更新版整理，重点服务后续实验决策，而不是重复通用综述。

### 11.1 本轮新增核心判断

这两篇论文共同支持一个更强的判断：

> 直接在原始特征 `X` 或普通 GRACE 当前表示 `z` 上构粒球，风险很高；更合理的路线是先通过 contrastive / hypersphere / warm-up / EMA 学出更适合构球的表示空间，再让粒球以 confidence、local consistency、weak prototype 或 compactness regularizer 的形式进入训练目标。

对已有实验的解释是：

- 之前的粒球不是完全“一次生成就不管”，因为很多配置会 periodic refresh。
- 但粒球结构本身没有被 loss 直接优化；它更多是外部规则生成器。
- 因此粒球会随 `z` 间接变化，但没有被明确要求变得更紧凑、更分离、更稳定或更适合去噪。
- 这能解释为什么 same-ball / cross-ball 规则在 view、negative、edge weight 上效果弱甚至伤害 clean graph。

### 11.2 Paper Note: GBmooer

**论文定位**：文本噪声标签分类中的 Granular-ball Text Label Smoothing with Hypersphere Contrastive Learning。

**核心机制**：

- 先用 hypersphere contrastive learning 整理文本表示空间。
- 将 embedding 投影到 unit hypersphere，用 angular distance 衡量局部关系。
- 再在该空间中做 unsupervised granular-ball clustering。
- 球内 label distribution 用于 local label smoothing，避免全局 smoothing 扩散噪声。
- 损失包括 angular contrastive loss、KL smoothing loss 和 symmetric KL consistency。

**对本项目最有价值的启发**：

- 不要直接相信 raw embedding 或 early `z` 的欧氏距离。
- 可以比较 `X`、early `z`、late `z`、EMA `z`、normalized/angular `z` 的构球质量。
- `average angular distance`、`radius`、`temporal stability` 可以作为无标签 confidence。
- 球内 label distribution smoothing 不能直接搬到 GRACE，因为会引入标签信息；只能改写成无标签的 local uncertainty、loss weight 或 relation consistency。

**对失败实验的解释**：

- hard same-ball positive 风险高，因为 same-ball 不等于 same-label。
- full-ball suppression 太粗，因为球内可能混入语义不一致节点。
- 如果普通 GRACE `z` 没有被训练成适合构球的空间，那么基于它的 hard rule 很容易变成噪声规则。

### 11.3 Paper Note: CGBC

**论文定位**：噪声标签下的智能合约漏洞检测，方法是 Contrastive Learning-Enhanced Granular-ball Training。

**核心机制**：

- 先做 unsupervised contrastive representation initialization。
- 再在 encoder 与 classifier 之间加入 GBC layer。
- 用标签 purity 构造 eligible GB，并生成 GB center / majority label。
- 训练中使用 intra-GB compactness、inter-GB looseness / separation 和 symmetric cross entropy。

**对本项目最有价值的启发**：

- GB 不只是外部规则，而是训练目标的一部分。
- 训练目标明确要求球内更紧、球间更分离，粒球结构才会随训练朝“更好”方向变化。
- 迁移到 GRACE 时可以保留 compactness / weak prototype 思路，但必须去监督化。
- GB center 只能作为 `stopgrad` weak prototype，小权重、late enable、高置信球限定，不应替代节点表示。

**不能直接迁移的部分**：

- `Purity(X)`。
- majority-vote GB label。
- same-label inter-GB separation。
- 任何依赖训练标签的球质量判断。

这些机制在 CGBC 中合理，因为它是有监督噪声标签分类；在无监督 GRACE 预训练中会造成 label leakage。

### 11.4 与当前实验方向的映射

| 当前方向 | 两篇论文给出的判断 | 决策 |
|---|---|---|
| GB view generation | 两篇论文都没有支持 GB 直接主导图增强；已有实验也显示 view drift | 不作为主线 |
| hard same-ball positive | 两篇论文的“同球可靠”都有表示学习或标签/purity 前提 | 放弃 hard 版 |
| GB-FN topk | 可保留，但必须证明超过 no-ball similarity topk | 修改后继续 |
| same/cross-ball edge denoise | 规则太粗，文献支持弱，已有 hit-rate 弱 | 不作为主线 |
| angular / hypersphere GB diagnostic | GBmooer 明确支持 | 高优先级 |
| GB-in-loss compactness | CGBC 明确支持，但需无标签化 | 高优先级小实验 |
| detached weak prototype | CGBC 的 GB center 可弱迁移 | 中高优先级 |
| pct/S2 noisy graph | 强结果不是 GB 贡献 | 可作为非 GB robustness 线 |

### 11.5 推荐的最小后续实验

| 优先级 | 实验 | 只改变什么 | 成功标准 | 停止标准 |
|---|---|---|---|---|
| P0 | Warmup / EMA / angular GB diagnostic | 不训练新方法，只比较构球空间 | angular/EMA balls 的成员稳定性、same-label consistency 或 candidate precision 明显优于普通 `z` | 不优于 no-ball similarity 或跨 seed 极不稳定 |
| P1 | Confidence-weighted GB-FN | 保持 GRACE loss 主体，只把 GB 候选降权改成 confidence 权重 | 至少 2 个数据集超过 no-ball topk，或 std 明显下降 | 与 no-ball topk 持平 |
| P2 | GB-in-loss local compactness | late enable、小 `lambda`、high-confidence balls only | noisy graph 下稳定优于 GRACE 和 no-GB control | clean/noisy 都下降或强依赖超参 |
| P3 | Detached weak prototype | `stopgrad(mean(z in ball))` 作为弱 teacher | 降低方差或提升 noisy robustness | 出现过平滑，低于 GB-FN/no-ball |

### 11.6 本轮更新后的 claim 边界

当前可以写：

- CGBC 和 GBmooer 支持“contrastive representation first, granular-ball robust correction second”。
- 这两篇论文支持把粒球放进训练目标或局部鲁棒机制，而不是只作为一次性外部规则。
- 对 GRACE 来说，最安全的迁移是无标签 confidence、angular diagnostic、late-enable weak prototype 和小权重 compactness regularizer。

当前不能写：

- 粒球已经被证明能提升节点级 GRACE。
- same-ball 节点天然是语义正样本。
- CGBC / GBmooer 的标签相关机制可以直接迁移到无监督图对比学习。
- pct/S2 的 noisy graph 强结果来自粒球。

### 11.7 5 条最重要启发

1. **GBmooer**：先学习适合构球的 hypersphere / angular 表示空间，再构粒球。
2. **GBmooer**：average angular distance、radius、temporal stability 比 hard membership 更适合作 confidence。
3. **CGBC**：GB 进入 loss 后，粒球结构才有机会被训练目标持续约束。
4. **CGBC**：GB center 可启发 weak prototype，但必须 stopgrad、小权重、late enable。
5. **两篇共同启发**：GB 主叙事应优先放在 robustness，而不是 clean graph 小幅提升。

### 11.8 5 个最重要避坑

1. 不要直接在 `X` 或普通 early `z` 上构球后写 hard rule。
2. 不要把 same-ball 当 hard positive 或同类标签。
3. 不要把 supervised purity、majority label、label distribution smoothing 搬进 GRACE。
4. 不要做复杂 GB loss 却缺少 no-ball、random、shuffled、size-matched controls。
5. 不要把非 GB 的 pct/S2 结果归因给 GB；GB claim 必须有独立增量证据。

### 11.9 2026-05-19 周报理解核验

本节来自 `C:/Users/Admin/Downloads/NAUGRIN研究生周报-曹思磊26.5.15.docx` 中对 CGBC 的深度思考。周报中的实验 1-6 复盘此处不重复，只保留对论文机制和后续方向有价值的判断。

**判断基本正确的部分**：

- CGBC 的整体流程可以概括为：前期用无监督 contrastive learning 做 representation initialization，后期用有监督 GBC layer 和分类损失解决 noisy-label vulnerability detection。
- CGBC 使用 negative-sample-free 对比学习，这一点与 SimSiam 式 stop-gradient 机制一致；它避免显式把其他样本当负样本。
- CGBC 的构球使用带标签数据、purity threshold、majority vote label，并用 cosine distance 辅助分裂，这说明它不是普通无监督构球，迁移到 GRACE 时必须去监督化。
- CGBC 的 GB loss 确实不是“只生成球后就不管”：它用 intra-GB compactness 约束球内样本靠近中心，用 inter-GB looseness / separation 约束不同标签粒球的中心关系。
- `L_SCE` 由 CE 与 RCE 组合，目标是提高噪声标签下的鲁棒性。
- “之前实验里的粒球参与了训练流程，但没有成为训练目标的一部分”这个判断是当前项目非常重要的复盘结论。已有 GB 主要生成规则去改 view / negative / edge weight，而不是通过 loss 明确优化球内紧凑、球间分离、球质量或球稳定性。
- “虽然粒球生成本身不可微，但可以利用球中心、半径、质量、稳定性等结构产生可优化约束，反过来塑造 encoder 表示空间”这个理解是正确的，也是下一轮 GB-in-loss 方向的核心依据。
- “好的增强不是扰动越强越好，而是在不破坏任务语义的前提下制造足够视图差异”适合作为后续 augmentation 诊断原则。
- “不把粒球用作硬划分，而是作为 soft prior”与当前文献和已有失败实验一致。

**需要更谨慎表述的部分**：

- “CGBC 选择 negative-sample-free 是因为普通对比学习会导致样本不够紧凑、降低粒球质量”是一个合理解释，但论文没有直接证明这是唯一原因。更稳妥的写法是：negative-sample-free 可能减少 false negative 和负样本排斥对局部聚合的干扰，值得在本项目中作为假设验证。
- “普通 InfoNCE 的负样本排斥力可能与粒球局部聚合目标冲突”是有研究价值的机制假设，但目前项目还缺少直接实验。后续应比较 InfoNCE、negative-free、CLSA/weak-teacher 在构球稳定性、球内一致性、candidate precision 上的差异。
- “后续使用 negative sample free 改进 edge denoise”不能直接作为结论。negative-free 可能改善表示空间和构球质量，但 same/cross-ball 边规则本身仍然太粗，必须先做构球诊断和 hit-rate 对照。

### 11.10 两篇文献的优先级调整

从 2026-05-19 起，CGBC 和 GBmooer 在本项目文献图谱中的优先级上调为 **P0 核心启发文献**，原因是它们是真正把 **granular ball** 与 **contrastive learning** 放在同一训练框架中讨论的文章，而不是只提供泛化的粒球分类、图粗化或普通 GCL 背景。

| 文献 | 新优先级 | 为什么上调 | 对当前项目的直接价值 | 迁移边界 |
|---|---|---|---|---|
| GBmooer | P0 | 明确提出 hypersphere contrastive learning + granular-ball local smoothing | 支持“先学习适合构球的表示空间，再构球”；提供 angular distance、average angular distance、radius 等 confidence 线索 | 球内 label distribution smoothing 不能直接进入无监督 GRACE |
| CGBC | P0 | 明确提出 contrastive representation initialization + GBC layer robust training | 支持“GB 进入 loss / training objective”；启发 compactness、weak prototype、GB quality regularizer | purity、majority label、same-label separation 不能迁移到无监督 GRACE |

因此，后续所有 GB-GCL 新实验在立项前应优先回答两个问题：

1. 是否学习或选择了一个更适合构球的表示空间，而不是直接依赖 raw `X` 或 early `z`？
2. 粒球是否以可优化的 soft objective / confidence / weak teacher 进入训练，而不是只作为一次性外部规则？

如果一个新实验不能回答这两个问题，默认不应作为 GB 主线，只能作为诊断或负结果。
