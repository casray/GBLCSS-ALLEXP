# 检索与访问记录

日期：2026-10-22。定向文献核查；不冒充穷尽检索或 PRISMA 系统综述。

## 1. 检索约束与范围

- 主检索使用论文完整题名与 `graph neural network`、`node classification` 或 `label noise` 锚词。
- 题名中的原有 `distillation` 保留不改；题名外不以单独的 `distillation` 作宽泛检索。
- 未使用 `denoise + synthetic` 类型组合。NAEM 原题含 `Denoising`，仍按完整原题检索，没有扩展到数据生成或扩散模型方向。
- 从已核实论文、会议页、作者页和代码 README 追踪链接，不把第三方自动综述当正式证据。
- 检索规则执行记录：A 的代码定位有两条完整题名后额外加 `code` 的查询；“Graph as Privileged Information”短语和两个待核实图 PI 题名曾作定位探测。它们均不用于支持“某工作不存在”的结论。后续正式检索保留完整题名与学科锚词。

## 2. 已使用的题名检索族

以下为实际检索的题名与锚词组合；不同锚词重复命中同一论文不算新文献。

| 完整题名/定位对象 | 使用过的锚词或限定 | 处理 |
|---|---|---|
| Transfer and Marginalize: Explaining Away Label Noise with Privileged Information | label noise；graph neural network；代码定位另加 code | ICML 2022 原始页、19页 PDF、Google TRAM 实现 |
| Pi-DUAL: Using Privileged Information to Distinguish Clean from Noisy Labels | label noise；graph neural network；代码定位另加 code | ICML 2024 原始页、arXiv v2 全文；独立实现未核实 |
| A new learning paradigm: Learning using privileged information | label noise | 2009 原文与 JMLR 后续交叉核实 |
| Unifying distillation and privileged information | node classification | ICLR 2016 原文、corrupt-label 数值例子 |
| Learning Using Privileged Information: Similarity Control and Knowledge Transfer | graph neural network | JMLR 原始论文 |
| Graph-less Neural Networks: Teaching Old MLPs New Tricks via Distillation | label noise；graph neural network 与 privileged information 定位 | 官方代码、全文方法和 Section 6 |
| NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs | label noise | 原文；发现正式会议题名别名 |
| Learning MLPs on Graphs: A Unified View of Effectiveness, Robustness, and Efficiency | label noise | 正式 ICLR 2023 PDF |
| Learning to Model Graph Structural Information on MLPs via Graph Structure Self-Contrasting | label noise；graph neural network | 全文 Table IV 与官方 README 明确标签噪声 |
| Graph-MLP: Node Classification without Message Passing in Graph | label noise | 原始摘要和方法定位 |
| Training MLPs on Graphs without Supervision | label noise | WSDM 2025 全文，区分标签稀缺与标签噪声 |
| Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs | label noise | PMLR、全文、当前 KRD 代码 |
| VQGraph: Rethinking Graph Representation Space for Bridging GNNs and MLPs | label noise | arXiv v3，Eq.7、Table6 与附录 |
| Demystifying GNN-to-MLP Knowledge Transfer: Theoretical Grounding and Dual-Stream Distillation Method | label noise | AAAI 2026 正式页与 PDF |
| Structure-aware multi-teacher distillation for noise node classification | graph neural network | 出版社原始预览，全文/匿名代码访问受限 |
| From GNN to MLP: Enhancing learning with knowledge-augmented distillation and confidence guidance | label noise | 出版社原始预览，实验细节未确认 |
| NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise | node classification | 会议页、作者全文、官方库 |
| Learning on Graphs under Label Noise | node classification | CGNN，筛选/纠正而非噪声头；背景追溯 |
| Learning Graph Neural Networks with Noisy Labels | node classification | 2019 D-GNN 原始摘要实际以 graph classification 为任务，避免误当节点 PI 噪声头 |
| Noise-robust Graph Learning by Estimating and Leveraging Pairwise Interactions | node classification | 作者全文；PI 不是 privileged information |
| Unified Robust Training for Graph Neural Networks Against Label Noise | node classification | UnionNET 正式页与作者单位原文 |
| Robust Training of Graph Neural Networks via Noise Governance | node classification | RTGNN 原文 |
| Mitigating Label Noise on Graph via Topological Sample Selection | node classification | 样本选择背景；不冒充噪声生成头 |
| Graph Label Denoising via Neighborhood Agreement-Guided Expectation Maximization | node classification；graph neural network；label noise | 官方摘要已核实；PDF 超时 |
| When does Privileged Information Explain Away Label Noise? | graph neural network | ICML 2023 原文与代码目录 |
| Rethinking Knowledge Transfer in Learning Using Privileged Information | label noise | 预印本方法、作者单位 TMLR 2025 出版记录 |
| Training deep neural-networks using a noise adaptation layer | node classification | 2017 适配层背景；图上的实际评估以 NoisyGL 为证据 |
| Edge-free but Structure-aware: Prototype-Guided Knowledge Distillation from GNNs to MLPs | label noise | COLING 原始 PDF；区分 feature noise |
| Extracting Low-/High-Frequency Knowledge from Graph Neural Networks and Injecting It into MLPs: An Effective GNN-to-MLP Distillation Framework | label noise | AAAI 原始页/PDF；频带知识迁移背景 |
| Cold Brew: Distilling Graph Node Representations with Incomplete or Missing Neighborhoods | label noise | 原始摘要，邻居缺失不等于标签翻转 |
| LR-SVM+: Learning Using Privileged Information with Noisy Labels | graph neural network | 非图 LUPI 抗噪背景，未作为图 PI 先例 |
| RGPTN: Learning on graphs, hypergraphs, and heterogeneous graphs with noisy labels | node classification | 原始预览；非本次核心训练期 PI 噪声头 |
| Graph topology adaptive judgment against node label noise | node classification | 调边背景，不当作噪声生成头 |
| Noise-resistant graph neural networks with manifold consistency and label consistency | node classification | 原始摘要表征/标签一致性互补背景 |

定位性而非已确认题名的探测：`Graph as Privileged Information`、`Learning Graph Neural Networks Using Privileged Information`、`Graph Neural Networks with Privileged Information`。未得到可据此确定的原始对应论文，不把该结果写成不存在证明。

## 3. 访问限制与修正

- Pi-DUAL 当前作者文本为 arXiv `2310.06600v2`，不是曾尝试的 v3。
- NOSMOG 当前所读为 `2208.10010v2`；VQGraph 为 `2308.02117v3`；PI-GNN 为 `2106.07451v2`。定位中若干不匹配版本 URL 访问失败后，回到摘要页点击实际版本，不把失败猜测作为证据。
- SimMLP 使用 `2412.03864` 的 WSDM 2025 版本，不使用已被新记录替代的旧 `2402.08918` 作为最终出处。
- GLNN 官方库为 `snap-research/graphless-neural-networks`，不是未能访问的 `snap-stanford` 路径。
- KRD 原文曾写 `LirongWu/RKD`；本轮核实当前库为 `LirongWu/KRD`。
- 多个 OpenReview 页面触发浏览器验证；能读作者全文时改用作者全文。正式出版元数据单独核实。
- TRAM 原 PDF 没找到代码链接不等于无实现；已实际定位 Google PI 目录中的 `tram.py`。
- Pi-DUAL 原文提到实验基于 uncertainty_baselines，但不能因此把该库的 TRAM 实现标成 Pi-DUAL 源码。
- NAEM 官方 PDF 多次超时；保留官方摘要级结论，不推断未读公式、局部矩阵形式与部署输入。
- SAMTD 和 KACG 的出版社全文获取受限；搜索索引可见的出版社原始摘要/引言只支持相应层面的判断。
- SAMTD 匿名源码入口不可访问；未确认其模型架构和训练预算。
- Rethinking LUPI 的 TMLR 出版记录指向 `dg1tqNIWg3`，旧搜索结果的其他 OpenReview 记录不混用为正式版本。详细阅读仅为预印本，未逐字比对正式版本。

## 4. 结论强度控制

- “原文未做图”依据 TRAM/Pi-DUAL 的实验及数据集，不只是题名或关键词搜索。
- “已有多通道工作”依据 TRAM 比较表、NOSMOG 联合损失、VQGraph Eq.7/Table6、DA-MLP 分析与 SAMTD 原始描述。
- “图上噪声建模已有”依据 NoisyGL 实验与 NAEM 官方材料，不把所有邻域一致性方法统称为 noise head。
- “严格交集尚未核实到重合”不等于“全领域没有”；全文受限、术语差异和定向题名检索都限制召回率。
- 本轮没有验证任何论文公开成绩，也未重现源码；不输出跨论文精度排行榜。
