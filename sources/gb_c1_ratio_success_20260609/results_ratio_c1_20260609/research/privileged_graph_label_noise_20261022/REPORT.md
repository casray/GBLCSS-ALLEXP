# 带噪标签下的图特权信息学习：原文核实、机制边界与研究空间

核查日期：2026-10-22。用途：研究方向判断，不是投稿稿件的修改。本轮未训练模型、未改实验代码、未修改论文。

## 0. 决策摘要

1. **记忆中的两篇论文及会议年份均正确。** TRAM 为 ICML 2022，Pi-DUAL 为 ICML 2024。它们的原文实验均不涉及图节点分类。
2. **“图只在训练时可用”不是新问题，“表示和预测两种通道”也不是空白。** GLNN、Graph-MLP、NOSMOG、VQGraph、SimMLP 等已覆盖不同实现。GSSC 还明确测过标签噪声，不能声称首次实现图辅助、MLP 部署的抗噪节点分类。
3. **“多通道用于图标签噪声”也已有直接近邻。** SAMTD 的公开原始材料明确包含表示结构迁移、教师预测一致性加权和标签增强。其严格的 X-only 部署条件与完整消融本轮未核实，不能把它直接当成同部署条件方法，也不能忽略这个机制重叠。
4. **图上噪声建模已有先例。** NoisyGL 将噪声适配层和损失校正放到 GCN 节点分类上评估；NAEM 明确将邻域结构先验与噪声转移矩阵结合。不能声称首次在图上建模标签噪声。
5. **更窄的交集尚未核实到完全重合的工作：** 图条件化的噪声分支，仅在训练期使用，部署只保留原始属性分类器，并系统比较它与表示迁移、预测迁移的关系。这里的“尚未核实”不是“确认无人做过”。
6. **建议：作为返修解释框架可以采用；作为新论文问题可以继续，但不能只换一个 LUPI 名字或照搬 Pi-DUAL。** 优先验证不同通道的可实现互补，并区分类别信息、噪声信息、训练覆盖和优化预算。

## 1. 研究范围与判定口径

这是按用户指定题名及学科锚词开展的定向核查，并通过原文链接追踪相关论文，不是穷尽所有数据库的系统综述。正式结论优先使用会议论文、作者原文、出版社原始介绍、官方代码。搜索结果中的第三方自动摘要不作为结论依据。

逐篇字段见同目录 `evidence_matrix.csv`。全文可访问者检查方法及实验部分；无法取得全文者明确标记“原始摘要/预览”，不称为完成全文精读。代码链接核实不等于运行或数值复现。

### 1.1 必须区分的概念

- **训练期特权信息：** 本任务约束下，训练可用而最终分类器推理不接收的信息。对于 X-only 学生，图及其派生表示可如此理解；这并不要求原作者使用 LUPI 术语。
- **类别证据：** 图帮助判断节点是什么类。
- **噪声证据：** 信息帮助判断标注过程如何出错、谁更容易出错、错误转向哪一类。
- **噪声筛选与纠正：** 找疑似错标或重建标签，不等于显式估计一个噪声生成头。
- **噪声建模：** 显式参数化观测标签如何由真实类别和其他因素产生，例如转移矩阵、残差 logits 或混合噪声分支。

“标签、表示、预测、噪声”是分析维度，而不是互不相交的四个信息源。LCSS 软目标与教师软预测最终都可能作为学生的目标向量；两者的区别主要在来源、监督覆盖与生成机制。多路输入也不等于独立多视图。

### 1.2 三种不同的部署约束

1. 不做在线消息传递；
2. 不访问邻接矩阵，但允许缓存位置编码、图表示或节点预测；
3. 只输入原始属性 X，不查询节点专属图缓存。

这三者不等价。NOSMOG 的位置特征、Cold Brew 的潜在邻居机制等，不能不加说明地当成原始 X-only 两层 MLP 的同条件对照。GLNN、SimMLP 等仍是必须考虑的邻近方法。

## 2. A：TRAM 与 Pi-DUAL

### A1. Transfer and Marginalize: Explaining Away Label Noise with Privileged Information

- **作者/出处：** Mark Collier、Rodolphe Jenatton、Efi Kokiopoulou、Jesse Berent；ICML 2022，PMLR 162，4219-4237。
- **特权信息：** 标注者身份、经验、反应时间，以及机器标注者相关元数据。核心是标注过程信息，不是图拓扑。
- **机制：** 一个共享特征提取器、一个接收普通特征与 PI 的条件预测头，以及一个只接收普通特征的部署头。共享特征主要由有 PI 的头更新，部署头采用停止梯度的共享表示；另有两阶段版本。
- **目标：** 借助 PI 学习条件观测标签分布，再通过共享表示和近似边缘化构造不需要 PI 的预测器。它不是给每个节点估计噪声转移矩阵的图模型。
- **多机制：** 原文 Section 3.3 有 Distilled-TRAM；Table 1/Section 5 比较了知识蒸馏、共享参数与其他 PI 使用方法。因此“先表示学习再预测迁移”这种组合思想本身不能算空白。
- **实验：** CIFAR-10H、重新标注的 ImageNet、Civil Comments 等，非图节点分类。
- **边界：** 近似边缘化的目标是观测分布，不自动等于真实干净标签分布；不能引用其理论为任意图抗噪改进背书。

阅读定位：Section 3，尤其 3.2/3.3；Section 5；Appendix D。原文：[会议页](https://proceedings.mlr.press/v162/collier22a.html)、[PDF](https://proceedings.mlr.press/v162/collier22a/collier22a.pdf)。

**代码：** 已核实 Google `uncertainty-baselines` 的 [CIFAR TRAM](https://github.com/google/uncertainty-baselines/blob/main/baselines/privileged_information/cifar_pi/tram.py) 和 [ImageNet TRAM](https://github.com/google/uncertainty-baselines/blob/main/baselines/privileged_information/imagenet_pi/tram.py)。未运行。

### A2. Pi-DUAL: Using Privileged Information to Distinguish Clean from Noisy Labels

- **作者/出处：** Ke Wang 等；ICML 2024，PMLR 235，51214-51236。
- **PI：** 人类/机器标注者的元数据；另为每个训练样本加入固定随机标识向量，以吸收未观测的噪声因素。
- **机制：** 普通特征进入类别预测网络；PI 进入噪声网络和门控。训练时在 logits 层混合两者，推理只保留类别预测网络。
- **不是标准转移矩阵：** 噪声分支输出的是拟合错误标注的 logits，门控决定两条学习路径的权重，并非 C×C 的类别翻转矩阵。
- **原式结构：** `h(x,a) = [1-gamma(a)] f(x) + gamma(a) epsilon(a)`。损失作用于混合输出，不额外强迫部署分支单独拟合全部噪声标签。
- **实验：** CIFAR-10H、CIFAR-10N、CIFAR-100N、ImageNet-PI。无图节点分类实验。
- **比较：** 有去门控、去噪声头、不同门控位置、噪声信号输入等消融；不是图上的四通道统一比较。
- **移植风险：** 若把能强烈预测类别的 GRACE 表示送进所谓噪声头，它可能接管干净类别预测，而非专门吸收噪声。

阅读定位：Section 3.1-3.3、5.3；Appendix B.1、C.6-C.9。[会议页](https://proceedings.mlr.press/v235/wang24bb.html)、[作者全文](https://arxiv.org/html/2310.06600v2)。

**代码状态：** 原文说明实验基于 `uncertainty_baselines`；本轮没有核实到独立公开、明确对应 Pi-DUAL 完整实现的代码入口。不能把 TRAM 目录当成 Pi-DUAL 官方实现。未发现入口不等于作者从未开源。

## 3. B：基础与必须补读的限制性文献

### B1. A new learning paradigm: Learning using privileged information

Vladimir Vapnik、Akshay Vashist，Neural Networks 22，2009，544-557，DOI `10.1016/j.neunet.2009.06.042`。训练给出 `(x,x*,y)`，部署只保留 x；SVM+ 利用 PI 控制样本的松弛/难度。它是训练与推理信息不对称的基础，不是图标签噪声方法，也不保证增加 PI 总会提高精度。出处可由[原文](https://dl.icdst.org/pdfs/files/3e25a143988b3a8f7005c8cb756b4113.pdf)与后续 JMLR 原文交叉核实。

### B2. Learning Using Privileged Information: Similarity Control and Knowledge Transfer

Vapnik、Rauf Izmailov，JMLR 16(61)，2015，2023-2049。它已经区分相似性控制与直接师生知识转移。把“多种使用 PI 的途径”作为概念框架有依据，但不能声称这一本体划分由本项目首次提出。[JMLR 原始页](https://www.jmlr.org/beta/papers/v16/vapnik15b.html)。

### B3. Unifying distillation and privileged information

David Lopez-Paz、Leon Bottou、Bernhard Scholkopf、Vladimir Vapnik，ICLR 2016。教师利用 PI 产生软目标，学生结合普通输入、标签与教师输出训练，统一为 generalized distillation。原文 Section 5 的第一个数值例子就涉及受污染标签与更干净的 PI；所以“PI 抵抗标签噪声”不应被写成 2022 年才出现。非图任务；比较 SVM+ 与输出迁移，不是当前图通道设计的完整因子实验。[原文](https://arxiv.org/html/1511.03643v3)。

### A3. When does Privileged Information Explain Away Label Noise?

Guillermo Ortiz-Jimenez 等，ICML 2023，PMLR 202，26646-26669。必须纳入，而非只读 TRAM/Pi-DUAL。论文比较多种 PI 方法，操控 PI 的性质，发现能够区分错标、并提供噪声拟合捷径的 PI 更有帮助；单纯高度预测类别的 PI 未必有益。非图任务。对本项目的启发是：**“图预测类别很好”不能直接推出“图适合作为噪声吸收分支的输入”。**[会议页](https://proceedings.mlr.press/v202/ortiz-jimenez23a.html)、[PDF](https://proceedings.mlr.press/v202/ortiz-jimenez23a/ortiz-jimenez23a.pdf)。代码见 [Google PI 目录](https://github.com/google/uncertainty-baselines/tree/main/baselines/privileged_information)。

### A4. Rethinking Knowledge Transfer in Learning Using Privileged Information

Danil Provodin 等；作者单位记录为 TMLR 2025。它重新审视 generalized distillation 和 TRAM，认为某些收益可能来自训练不足、架构差异或数据条件，而不是 PI 知识迁移本身。其负结果不等于反证所有 LUPI；也没有检验本项目的图机制。它要求我们做匹配预算、零/打乱 PI、同容量等对照。[作者预印本](https://arxiv.org/html/2408.14319v1)、[作者单位出版记录](https://research.tue.nl/en/publications/rethinking-knowledge-transfer-in-learning-using-privileged-inform/)、[作者代码](https://github.com/danilprov/rethinking_lupi)。本轮详细方法阅读来自预印本；正式 TMLR 全文访问受限，未逐字比对版本。

## 4. C：图信息向属性学生迁移，哪些已经做过

| ID | 工作与出处 | 已经覆盖的内容 | 与当前问题的边界 |
|---|---|---|---|
| C1 | GLNN，ICLR 2022 | GNN 预测迁移到 X-only MLP；转导/归纳 | 原文的噪声检查为节点特征噪声，不是系统标签翻转实验 |
| C2 | Graph-MLP，2021 预印本 | 图邻域进入对比损失，部署不需边 | 结构先验不等于噪声生成头；不能称首次把图放入监督 |
| C3 | NOSMOG，ICLR 2023 正式题名见下 | 教师软预测 + 表示相似性迁移 + 位置特征 | 已组合表示与预测；其 noise-robust 主要针对特征噪声，且用图派生位置输入 |
| C4 | GSSC，TNNLS，2024 online/2025 卷期 | 图稀疏化、自对比、双层优化、纯 MLP 部署 | **明确测标签噪声**；直接否定“图辅助 X-only 抗噪尚无人做”的宽主张 |
| C5 | SimMLP，WSDM 2025 | 标签无关的 GNN/MLP 表示对齐 | 原文鲁棒实验为特征噪声、边噪声、标签稀缺；不是系统标签翻转实验 |
| C6 | KRD，ICML 2023 | 评估教师知识可靠性，抽取额外可靠监督 | 教师预测不可靠不等于训练标签被污染；原文没有据此完成图噪声头设计 |
| C7 | VQGraph，ICLR 2024 | 类别预测迁移 + 结构码本分配迁移 | **已有表示相关目标与预测目标的联合及替换消融**；噪声实验针对特征 |
| C8 | DA-MLP，AAAI 2026 | 分析标签对齐与表示对齐，双流互补图滤波 | **不能再声称首次分析两类迁移**；未见系统标签噪声实验 |
| C9 | SAMTD，IP&M 2026 | 多教师表示结构迁移 + 预测一致性加权 + 标签增强 | **图 + 标签噪声 + 多机制的直接重叠**；原始预览可核实，X-only 部署与完整消融未核实 |
| C10 | KACG，出版社原始预览 | 融合结构知识与逐节点知识，置信度指导及动态更新 | 与“可靠性/互补”主题重叠；标签翻转实验、严格部署约束和全文细节未核实 |

### 4.1 容易弄错的题名与噪声类型

**NOSMOG：** arXiv 题名是 *NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs*；ICLR 2023 正式题名是 *Learning MLPs on Graphs: A Unified View of Effectiveness, Robustness, and Efficiency*。是同一方法的版本，不是两篇独立先例。正文方法包含 soft-label KL 和 representational similarity loss；原文 Section 5.4 的鲁棒实验是 noisy features。[正式会议 PDF](https://openreview.net/pdf?id=Cs3r5KLdoj)、[作者原文](https://arxiv.org/html/2208.10010v2)、[代码](https://github.com/meettyj/NOSMOG)。

**GLNN：** Section 6 明确向 X 加独立高斯噪声，不能因为出现 noisy 就把它记为标签噪声实验。它已经提供 GNN 教师到无图依赖 MLP 的基础路线。[原文](https://arxiv.org/html/2110.08727v2)、[官方代码](https://github.com/snap-research/graphless-neural-networks)。

**SimMLP：** Section 5.6 是特征扰动、边扰动与标签稀缺；不能拿本项目给 SimMLP 加噪复现的结果，倒写成原论文已经做过标签噪声。[原文](https://arxiv.org/html/2412.03864v1)、[代码](https://github.com/Zehong-Wang/SimMLP)。

### 4.2 多通道不是空白，精确重叠在哪里

**VQGraph：** 原文 Eq. (7) 同时包含分类目标、类别输出迁移、结构代码迁移，Table 6 比较 class-based、AE+class-based、Only-VQ 和完整方法。这已是“不同迁移内容的对照”，虽不是含噪标签下四通道的统一识别实验。[原文](https://arxiv.org/html/2308.02117v3)、[代码](https://github.com/YangLing0818/VQGraph)。

**DA-MLP：** 原文分析输出对齐如何改变学生的训练动态，并讨论细粒度层间对齐的局限，再引入双流结构。它直接限制了“首次从机制上比较表示与输出迁移”的论断；不等于已经解决带噪标签的问题。[正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/40025)、[PDF](https://ojs.aaai.org/index.php/AAAI/article/download/40025/43986)。未从所读 PDF 核实到代码入口。

**SAMTD：** 原始预览明确给出多种自监督方法构建教师、表示结构一致性迁移、预测一致性权重和标签增强，并在五图、五噪声条件下评估。它是新方向必须逐项比较的近邻，不可仅列成泛泛相关工作。[出版社原始材料](https://www.sciencedirect.com/science/article/abs/pii/S0306457326002682)。作者给出 `https://anonymous.4open.science/r/SAMTD`，本轮访问失败；未取得全文或验证学生是否严格仅用原始 X。卷期标为 November 2026，属于当前可检索的 2026 原始出版材料，不能仅因卷期未来日期排除其在线材料。

**KACG：** *From GNN to MLP: Enhancing learning with knowledge-augmented distillation and confidence guidance* 的原始预览已讨论低同质性区域的结构知识局限、逐节点知识互补、置信度引导与动态更新。能确认机制邻近；不能从“noisy supervision”字样推断有训练标签翻转实验。[出版社原始材料](https://www.sciencedirect.com/science/article/pii/S1566253525005901)。全文获取受限，具体实验与代码不作确定判断。

### 4.3 “graph as privileged information”的严谨表述

可以说：**在我们的部署约束下，图构成训练期特权信息，既有 graph-free / GNN-to-MLP 方法可按使用方式归入这一视角。**

不应说：上述每篇原文都把方法叫作 LUPI；或者没有检索到题名含该短语就证明没有图 PI 研究。功能等价的工作通常沿用 graph-free、knowledge distillation、structural supervision 等术语。严格限定题名的检索本身也会降低新题名发现率。

## 5. D：图上噪声识别、转移矩阵与噪声头

### D1. NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise

NeurIPS 2024 Datasets and Benchmarks。它在 GCN 节点分类上比较 S-model、Forward、Backward 等噪声建模/校正方法。Section 4 及 Appendix B.1 明确分析所学噪声转移模式。**这足以证明“给节点分类加训练期噪声层”不新，但不能证明其中的转移矩阵由图上下文逐节点条件化。** 原文还报告这些矩阵可能退化、学不到真实转移，不能预设噪声头容易训练。[会议页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html)、[作者全文](https://arxiv.org/html/2406.04299v2)、[代码](https://github.com/eaglelab-zju/NoisyGL)。

### D2. Graph Label Denoising via Neighborhood Agreement-Guided Expectation Maximization

NAEM，Dezhi Liu 等，IJCAI 2026，2852-2860。**最接近“结构信息帮助噪声建模”的关键先例。** 官方摘要明确：干净标签为隐变量；E-step 使用邻域一致性结构先验，M-step 显式建模噪声转移矩阵，分离语义表示与噪声动态。

因此，图结构与噪声建模的结合已经有人做。**但官方摘要不足以确认矩阵是否为逐节点 `T_v(A,X)`，也没有支持“图只训练用、最终 X-only MLP”的结论。** PDF 在本轮多次访问超时，代码入口未核实；这里是官方摘要级核实，不冒充全文精读。[官方页](https://www.ijcai.org/proceedings/2026/317)、[官方 PDF 链接](https://www.ijcai.org/proceedings/2026/0317.pdf)。

### D3. Noise-robust Graph Learning by Estimating and Leveraging Pairwise Interactions

PI-GNN，TMLR 2023，预印本始于 2021。这里 PI 是 **pairwise interactions**，不是 privileged information。先估计节点对是否同类的软关系，再用于分类正则，存在解耦训练。它测图标签噪声，但不是图条件化的标注噪声生成头，也不是最终 X-only 学生。[原文](https://arxiv.org/html/2106.07451v2)、[代码](https://github.com/TianBian95/pi-gnn)。

### D4. Unified Robust Training for Graph Neural Networks Against Label Noise

UnionNET，PAKDD 2021。聚合图上标签证据，估计节点类别分布，用于重加权和纠正。原文明确不要求显式噪声转移矩阵。它是“图辅助可信度/标签纠正”的先例，不应被标成显式噪声头。[正式页](https://link.springer.com/chapter/10.1007/978-3-030-75762-5_42)、[作者单位全文](https://opus.lib.uts.edu.au/bitstream/10453/157517/2/2103.03414.pdf)。代码可从 NoisyGL 的适配实现考察，原作者仓库本轮未单独核实。

### D5. Robust Training of Graph Neural Networks via Noise Governance

RTGNN，WSDM 2023。图增强、双网络一致性、区分可靠/疑似噪声节点、修正和伪标注。原文的 governance 不是一个训练后丢弃的 PI 残差噪声头。它测图标签噪声；最终仍是图学习路线。[原文](https://arxiv.org/abs/2211.06614)、[作者全文](https://zhoujingbo.github.io/paper/2023robustWSDM.pdf)。原作者代码本轮未单独核实，NoisyGL 有适配。

### D6. DREAM: Dual-Standard Semantic Homogeneity with Dynamic Optimization for Graph Learning with Label Noise

ICML 2026。由属性接近与图拓扑选锚点，用语义同质性动态评估标签可靠性并指导训练。这说明“图提供噪声可靠性线索”也不是空白；它不是据此已经实现 Pi-DUAL 式 X-only 部署。[正式会议页](https://proceedings.mlr.press/v306/zhao26bg.html)、[作者预印本](https://arxiv.org/html/2601.17449v1)。代码入口本轮未核实。

### 5.1 给问题③分层作答

| 问题强度 | 本轮回答 | 证据 |
|---|---|---|
| 图帮助发现不可靠标签 | 已有大量工作 | UnionNET、RTGNN、DREAM 等 |
| 图节点分类中使用噪声适配层/转移矩阵 | 已有 | NoisyGL 对应实验；NAEM |
| 结构先验与潜在干净标签/噪声动态联合建模 | 已有 | NAEM 官方材料 |
| 图上下文逐节点预测转移或噪声残差，且只用于训练 | 本轮未核实到完整重合先例 | 不能由 global T 或可靠性分数替代这个证明 |
| 上述分支部署被移除，保留 X-only 学生，并公平比较多通道 | 更窄交集仍待核查与验证 | 不能据当前检索宣称“首次” |

## 6. 对当前实验最关键的理论限制：图知道类别，不等于知道随机噪声

以下是针对本项目噪声协议的推导，不是对某篇论文的转述。

设 `Y` 为真实类，`Z` 为观测类，`F = I(Z != Y)` 为本次是否被翻转。若每个节点均以相同 eta 独立翻转，噪声随机数独立于 X、图 A 和 Y，则生成模型满足：

`P(F=1 | X,A,Y) = eta`

这意味着图不能预先知道独立随机抽签的结果。但在观察 Z 后，图提供的类别证据可使 `P(F=1 | X,A,Z)` 更可辨识。因此：

- 可以研究“用图语义判断当前标签是否不协调”；
- 不能将该收益写成“图包含了随机翻转的产生原因”；
- class-conditional pair noise 的错误目标与 Y 有关，必须与错误发生的随机开关区分；
- 若要研究结构条件化的错标机制，应另设噪声概率依赖结构/社区或标注来源的协议，并与原始独立翻转协议分开报告；
- 不排除样本 ID 或高容量噪声头通过记忆分担错误拟合，但这种优化收益不证明图携带额外噪声生成信息。

对 Pi-DUAL 尤其要警惕：它让 PI 分支解释错误标注。图表示却常能很好地预测干净类别，这可能让所谓噪声分支接管正常语义预测，部署时移除后反而丢失能力。必须检查门控在干净/错标节点上的行为，不能只看混合训练损失下降。

## 7. 剩余空间应怎样定义

### 7.1 可以保留的研究问题

> 标签受到污染时，训练期图信息应当用于修正监督、约束表示、提供教师预测，还是吸收标注噪声？在给定 X-only 部署和训练预算下，哪些通道提供互补信息，哪些只是重复同一份类别证据？

这比“把图叫作 PI”具体，但仍需新增算法或强机制证据，不能只靠分类框架构成方法论文。

### 7.2 不应作为创新主张的内容

- 首次把图只用在训练阶段；
- 首次用图辅助 MLP 抵抗标签噪声；
- 首次联合表示迁移与预测迁移；
- 首次在图上建模噪声转移；
- 将 R2 初始化加普通预测蒸馏直接称为新理论或新学习范式；
- 将 PI 方法的收益自动解释为新增信息被传给了学生。

### 7.3 可以检验的新增贡献

1. **同资源下识别通道交互。** 固定数据、噪声、图表示、教师、学生和预算，分别开关表示迁移与输出迁移；补同预算继续训练对照。若组合只比训练不足的基线好，不成立。
2. **区分噪声可靠性与语义可学习性。** 不仅检测错标，还测哪些教师目标属性学生能学到，哪些只是共享错误；避免重走已经失败的粒球错标检测路线。
3. **防止噪声分支吞掉干净类别。** 若研究 graph-PI 噪声头，需要比直接 Pi-DUAL 移植多一个明确机制和对应消融，并与 NAEM 式噪声建模及普通置信度控制比较。
4. **把图条件噪声设为独立任务条件。** 只有噪声源与图有关时，才有理由强调图提供噪声过程信息；独立随机翻转仍是必要控制，不能只报告为方法量身构造的噪声。

这些是研究候选，不是本轮已被训练结果支持的算法。

## 8. 建议的最小验证顺序

### 第一优先：先确认通道互补，不增加噪声头

- 随机初始化学生 + 固定教师预测迁移；
- R2 初始化学生 + 同一教师预测迁移；
- R2 初始化 + 同预算原监督继续训练；
- 表示、教师和标签通道的数据范围要明确，训练层、轮数和选模规则匹配；
- 检查 clean、两种噪声及 Accuracy/macro-F1；不得用测试结果挑选融合系数。

这能判断后续是否值得研究选择性迁移。R2 与蒸馏学生的不同错误集合只提供潜在互补，不保证可训练组合必然提升。

### 第二优先：若要测噪声吸收分支，先做识别性对照

- 普通 X 分类器、同容量双头但无图 PI、真实图 PI、节点打乱 PI、随机 ID PI；
- 真实图与保留度分布/打乱语义的图控制，不可让对照仅因参数更少或训练更短而失败；
- 用交叉拟合或标签隔离生成诊断信号，防止当前节点标签经图编码器间接回流；
- 真实标签只在训练和选模结束后评估门控、误伤与噪声头行为；
- 结构条件噪声与独立标签翻转分别报告，不能合并成一组均值掩盖机制差异。

尚不建议直接开始完整四通道组合与支持量全网格扫描。该检索没有证明更复杂组合会有效。

## 9. 对用户四个问题的明确回答

**① TRAM/Pi-DUAL 是否做过图？** 原论文没有图节点分类实验。Pi-DUAL 的 X、A 矩阵理论符号也不能被误解成属性矩阵和邻接矩阵；其 A 表示 PI 数据。后续是否存在图扩展，本轮没有穷尽证明。

**② 是否有人比较多个 PI 通道？** 有。TRAM 与 ICML 2023 的 PI 研究比较多种 PI 使用方式；NOSMOG、VQGraph、DA-MLP 在图上组合/比较表示与预测相关迁移；SAMTD 在图标签噪声下组合表示结构、预测一致性和标签增强。尚未核实到与本项目拟定约束完全一致的四通道同资源因子比较，但不能声称“多通道比较无人做”。

**③ 图作为预测噪声的 PI 是否有人做过？** 宽义有充分先例，特别是 NAEM、NoisyGL 与图可靠性方法。严格意义的“图条件噪声分支仅训练使用、部署丢弃、最终原始 X-only MLP”本轮未核实到完整对应论文。这是有条件的候选交集，不是已确认的空白。

**④ 返修框架、新论文还是放弃？**

- **返修框架：可以用。** 有助于解释 LCSS、R2 与教师蒸馏的区别和适用范围，但不应把已投稿文章扩大成尚未验证的统一算法。
- **新论文：保留候选，缩小问题。** 围绕噪声下通道的可靠性、互补和可迁移性开展有识别力的实验；若只是 R2+蒸馏、或把 Pi-DUAL 的 PI 换成 G，证据与差异都不足。
- **应放弃的不是整个方向，而是宽泛的首创叙事与无依据的机制解释。** 尤其不要在独立随机翻转协议中默认图编码了噪声生成原因。

## 10. 未完成与不能据此推断的事

- Pi-DUAL 独立官方实现入口未核实；TRAM 实现已找到但未运行。
- NAEM 官方 PDF 超时，不能确定其噪声矩阵是否逐节点条件化，不能宣称完成全文精读。
- SAMTD、KACG 全文/代码访问受限，严格部署输入和全部实验细节未核实。
- 未对全文受限论文伪造结果、参数、代码路径或“没有该实验”的确定结论。
- Rethinking LUPI 的详细阅读来自预印本，出版状态由作者单位记录核实；两版本未逐字对照。
- 不把少量题名查询无结果当成全领域不存在的证据；不能据此在论文中使用 first/首次。
