# 相关工作三路审阅与修订审计

## 1. 审阅对象

- 原稿：`paper_final_zh_v23_graph_free_inference_title_20260722.md`
- 修订稿：`paper_final_zh_v24_related_work_revised_20260723.md`
- 改动范围：第 2 节“相关工作”及新增参考文献 [46]--[49]
- 未改内容：方法、实验设置、结果、表格、摘要、引言和结论

## 2. 三路审阅分工

### 路线 A：学术写作规范

该路线检索了高校写作中心和学术出版机构的相关工作写作建议。共同原则如下。

1. 相关工作应由研究问题和技术主题组织，而不是按作者或年份逐篇罗列。
2. 每段应先给出主题判断，再用多篇文献共同支撑。
3. 综述不仅要概括已有方法，还要比较其信息来源、假设和适用范围。
4. 实验实现、复现边界和本文结果应放在实验部分，不应混入相关工作。
5. 与本文的差异应建立在技术维度上，而不是反复使用“现有方法不足，因此本文提出”的模板。

参考写作资源：

- University of Toronto, *The Literature Review: A Few Tips*: https://advice.writing.utoronto.ca/types-of-writing/literature-review/
- George Mason University Writing Center, *Writing a Literature Review*: https://writingcenter.gmu.edu/writing-resources/research-based-writing/writing-a-literature-review
- Columbia University, *Writing Technical Articles*: https://www.cs.columbia.edu/~hgs/etc/writing-style.html
- Elsevier Researcher Academy, *Writing for Research*: https://researcheracademy.elsevier.com/writing-research

### 路线 B：本地 Elsevier 范文

本路线的子代理未完成全文抽取。主线程随后直接读取本地提取文本，补充审阅了以下 15 篇 Elsevier 期刊论文。前 14 篇具有独立的 Related Work 或同等功能章节。原始 GBC 论文没有独立的相关工作章节，用作早期紧凑写法的对照。

1. *G-GBC: A Gaussian Mixture-Based Granular Ball Computing Method for Robust CNN Classification under Noisy Labels*, Knowledge-Based Systems, 2026.
2. *Beneficial Noise Learning for Open Intent Classification via Granular-Ball Representation*, Pattern Recognition, 2026.
3. *Like Draws to Like: A Multi-Granularity-Ball Intra-Fusion Approach*, Advanced Engineering Informatics, 2024.
4. *Structure-Aware Granular Ball Clustering*, Information Sciences, 2026.
5. *A Structure-Aware Multi-Subspace Granular-Ball Clustering Framework*, Information Sciences, 2026.
6. *Granular-Ball Computing-Based Random Walk for Anomaly Detection*, Pattern Recognition, 2025.
7. *Fusing Multi-Granular-Ball Fuzzy Information to Detect Outliers*, Applied Soft Computing, 2025.
8. *An Adaptive Density Clustering Approach with Multi-Granularity Fusion*, Information Fusion, 2024.
9. *GB-RVFL: Fusion of Randomized Neural Network and Granular Ball Computing*, Pattern Recognition.
10. *GAdaBoost: An Efficient and Robust AdaBoost Algorithm Based on Granular-Ball Structure*, Knowledge-Based Systems.
11. *Capturing Local and Global Information: Multi-View Graph Convolutional Network via Granular-Ball Computing and Collaborative Matrix*, Expert Systems with Applications.
12. *Enhancing Graph Convolutional Networks with Progressive Granular Ball Sampling Fusion*, Information Sciences, 2024.
13. *Granular-Ball Computing-Based Manifold Clustering Algorithms for Ultra-Scalable Data*, Expert Systems with Applications, 2024.
14. *GBMOD: A Granular-Ball Mean-Shift Outlier Detector*, Pattern Recognition.
15. *Granular Ball Computing Classifiers for Efficient, Scalable and Robust Learning*, Information Sciences, 2019.

这些范文的共同结构不是完全一致，但存在稳定模式。

- 先用一至两句说明本节覆盖的研究方向。
- 按技术问题或机制设置二至四个小节。
- 小节内部先概括方法族，再引入代表工作。
- 与本文最接近的工作会获得更具体的机制描述。
- 小节结尾指出仍未连接的技术维度，或说明本文与已有工作的具体区别。
- 很少在相关工作中报告本文精度、消融结果或实现细节。
- 粒球论文尤其重视说明粒球承担的角色，例如分类单元、采样单元、噪声筛选单元或聚类表示。

### 路线 C：方法事实与竞争关系

该路线逐句核验了图标签噪声、GNN-to-MLP 和粒球噪声学习文献。主要发现如下。

1. v23 遗漏了 UnionNET、TSS，并未在相关工作正文讨论已有参考文献中的 TFR、ProCon 和 GD²。
2. ProCon 同样利用语义 peer 的标签一致性识别噪声，是机制上必须正面讨论的竞争者。
3. SA-MLP 的资源边界不能简单写成“推理必须使用邻接行”，因为其在线变体会估计潜在结构。
4. GSDN 在高噪声下传播错误监督是合理解释，但不是原论文已经证明的结论。
5. GNN-CFGD 的机制不应概括为单纯的“置信度重构图”。
6. MgCNL 和 MgBIF 不只是分离噪声样本。前者使用球级标签置信度筛选可靠样本，后者还执行组内伪标签融合、标签修正和高置信样本选择。
7. G-GBC 和 BNO-GB 是 2026 年直接相关的粒球噪声学习工作，必须纳入。

## 3. 对 v23 的主要问题判断

| 问题 | 严重性 | 处理 |
|---|---:|---|
| 相关工作混入 adapted、proxy 和表格位置等实验说明 | 高 | 已移至其原有实验章节承担，相关工作中删除 |
| 图标签噪声近期方法覆盖不足 | 高 | 补入 UnionNET、TSS、TFR、ProCon 和 GD² |
| ProCon 与 GB-LCSS 的局部一致性关系未讨论 | 高 | 在 2.2 中说明其 peer 构造和标签一致性机制 |
| SA-MLP、GSDN、GNN-CFGD 表述不够准确 | 高 | 已按原论文机制和资源边界改写 |
| GNN-to-MLP 小节接近文献清单 | 中 | 重组为直接 MLP 路线、教师蒸馏路线和监督假设差异 |
| 粒球小节对 MgCNL、MgBIF 的概括过粗 | 高 | 补充可靠样本选择、伪标签融合和标签修正 |
| 遗漏 G-GBC 与 BNO-GB | 高 | 新增参考文献 [48]、[49] 并正面切割 |
| 用本文消融结果定义相关工作定位 | 中 | 删除 kNN、kmeans 的实验结论，保留在实验章节 |
| gap 容易滑向“此前无人研究” | 高 | 改为“很少同时作为中心问题”，不使用 first claim |

## 4. 修订后的结构

### 2.1 通用标签噪声学习

按照样本选择、鲁棒损失、标签修正与半监督训练组织。结尾说明这些方法不利用训练期图结构构造监督，因此是抗噪机制参照，而不是完整的同设定先行工作。

### 2.2 图上标签噪声学习

先综述标签聚合、图增强、标签修正与鲁棒伪标签，再综述拓扑样本选择、结构重构、peer 一致性、双视图差异和多教师蒸馏。结尾说明这些方法处理相同任务，但通常使用不同的推理资源。

### 2.3 GNN-to-MLP 蒸馏与无图推理

先讨论直接训练 MLP 的结构知识迁移，再讨论 GNN 教师蒸馏，最后比较其监督假设和推理资源边界。基础 GB-LCSS 与蒸馏扩展被明确区分。

### 2.4 粒球计算与噪声学习

按照粒球分类与采样、标签噪声处理、深度模型联合训练组织。结尾说明 GB-LCSS 中粒球只承担离线监督构造角色，粒球划分不读取标签，但软目标构造仍使用含噪训练标签。

## 5. 事实核验表

| 修订内容 | 核验结论 |
|---|---|
| UnionNET 聚合结构邻近标签并用于重加权和标签修正 | 与 PAKDD 2021 原文一致 |
| TSS 将拓扑信息用于小损失样本选择 | 与 ICML 2024 原文一致 |
| TFR 使用拓扑特征重构并筛选伪标签 | 与论文及官方实现说明一致 |
| ProCon 结合特征相似性、邻接和 peer 标签一致性 | 与 IJCAI 2025 原文一致 |
| GD² 使用节点视图与结构视图的预测差异 | 与 NeurIPS 2025 原文一致 |
| GNN-CFGD 使用记忆效应、混合模型、图连接和置信度细分 | 与 Neural Networks 2025 原文一致 |
| SAMTD 使用多种自监督图编码器和一致性加权 | 与 Information Processing & Management 2026 原文一致 |
| KRD 的可靠性来自特征扰动下的熵稳定性 | 与 ICML 2023 原文一致，不再误写为标签正确性 |
| GSDN 具有非对称标签噪声实验，但未显式识别和修正噪声标签 | 与原文范围一致 |
| SA-MLP 主要结构分支使用邻接行，在线变体估计潜在结构 | 已避免绝对化 |
| MgCNL 使用球级标签置信度筛选可靠样本 | 与全文一致 |
| MgBIF 融合组内伪标签与原标签并筛选高置信样本 | 与全文一致 |
| G-GBC 使用 CNN、GMM、Mahalanobis 距离、球纯度与权重 | 与 KBS 2026 原文一致 |
| BNO-GB 使用标签无关粒球聚类和球级、样本级噪声指标 | 与 Pattern Recognition 2026 原文一致 |
| GB-LCSS 的粒球划分不读取标签 | 与代码和方法节一致 |
| GB-LCSS 的软目标构造使用含噪训练标签 | 已显式披露，未误写为整个管线 label-free |

## 6. 最终审计结论

修订后的第 2 节在内容上合理，主要竞争者和直接粒球先行工作已覆盖。当前 gap 被限定为“含噪节点标签与严格 X-only 推理很少被同时作为中心问题”，该表述与现有文献证据相符，也避免了无法穷尽检索的绝对 first claim。

三路终审中，写作结构与 Elsevier 风格两路均直接通过。事实审计发现原修订稿使用“它们”概括无图推理方法，可能误将 Graph-MLP、GSDN 和 SimMLP 排除在 X-only 资源层之外。终稿已改为分别说明 NOSMOG、SA-MLP 的资源边界，并确认 Graph-MLP、GSDN 和 SimMLP 与本文属于相同的 X-only 推理资源层。

仍需注意三点。

1. 相关工作中的 X-only 是推理资源条件，不代表训练阶段不使用图。
2. 基础 GB-LCSS 不是标准 GNN-to-MLP 蒸馏，只有教师扩展属于蒸馏路线。
3. 粒球划分是标签无关的，但局部标签分布和软目标不是标签无关的。
