# 图对比增强扩展：从 GRACE 到粒球/上下文表示

日期：2026-10-13  
用途：给“粒球 + 图对比 + 噪声节点分类”筛选增强基座、近邻工作和可证伪方案。本文只做文献与设计审查；没有训练、改代码或复跑论文结果。  
阅读范围：优先核对会议/出版社论文、正式论文 PDF 和作者代码。每篇标注阅读深度；论文已有结果不等于本项目协议下有效。

## 1. 结论先行

**最适合先作为 GRACE 增强替换基线的是 PiNGDA（ICML 2025）；最应作为粒球/社区表示直接近邻的是 CL-GCL（ICML 2026）。** 这两个判断不同：PiNGDA 更接近“保留节点级 InfoNCE、替换随机增强”的工程问题；CL-GCL 更接近“把节点对比信号聚合到社区/局部几何表示”的研究问题。当前没有证据足以断言哪一个在本项目含噪标签、原始划分或 AMD 12GB 显卡上更好。

如果论文问题仍是“粒球给节点噪声去除带来什么独立增量”，建议先把 PiNGDA 作为增强对照/可替换基座，同时把 CL-GCL 纳入直接基线和新颖性审查；不要把 PiNGDA、rooted-context VICReg、联合构球空间与 LCSS 一次拼成已定算法。**若最终主张重点转到球级/社区表示，CL-GCL 的优先级应与 PiNGDA 相当，甚至应先复现 CL-GCL 的节点分支，再谈新方法。**

必须区分以下几件事：

- PiNGDA 学的是对比预训练任务有益的图扰动，不是“识别训练标签错标”的去噪器。它的单图节点实验不能直接证明标签噪声鲁棒。
- 同根跨视图上下文的 VICReg 一致性，能要求表示稳定，不能单独证明上下文语义正确，更不能证明粒球成员应不变。
- CTAug、CI-GCL 都已研究结构凝聚/社区信息保持；“用粒球保持社区”不能单独作为首次性或创新性主张。
- CL-GCL 在单图节点表示与社区锚点方面和粒球高度接近；即使它使用社区而不是粒球，换称谓也不能绕开机制重叠。
- AFECL 等增强自由方法说明，性能变化未必来自更聪明的视图增强；遮蔽自编码也提供了非实例对比的替代训练目标。

## 2. 先把经典方法任务和增强机制分开

| 方法 | 原论文任务与正负关系 | 增强/优化到底做什么 | 对本项目的边界 |
|---|---|---|---|
| GRACE，ICML Workshop 2020 | 单个大图上的节点级对比；相同节点的两视图表示为正例，其他节点跨视图及视图内构成负例；下游为节点分类 | 对图结构和特征做随机扰动，典型实现是随机删边与特征 masking，再用共享 GNN 和 projection head 做节点 InfoNCE | 是本任务合适的起点。不要将其称为“第一个同时随机删边和 mask 特征的工作”；更早图对比已有边/特征扰动。GRACE 代码 README 标注 ICML Graph Representation Learning and Beyond workshop，不应误写为 ICML 主会论文 |
| GraphCL，NeurIPS 2020 | 多图数据集的图级对比；同一张图经两个增强后的 pooled graph embedding 为正例 | 系统比较 node dropping、edge perturbation、attribute masking、subgraph sampling 等图级操作 | GraphCL 中的 subgraph sampling 是对图样本做增强，再比较整图表示；不等于“每个节点拥有一个根子图表示并做节点分类” |
| JOAO，ICML 2021 | 继承 GraphCL 的图级任务和增强候选 | 对离散增强对及其采样分布做双层优化，自动选择预定义增强组合；还用 augmentation-aware projection heads | 不是连续可微、由监督标签驱动的节点增强生成器，也不是单图节点级基线。它更像“自动选增强菜单” |
| GCA，WWW 2021 | 单图节点级对比，GRACE 风格任务 | 按节点中心性和特征重要性为节点/边/特征设置非均匀随机扰动概率 | 是重要节点级增强先例，但其重要性加权随机策略不等于联合学习一个端到端生成器；论文比较时应分开写“自适应概率”和“学习增强器” |

GRACE 的“随机 edge drop + feature mask”是常用机制概括，不是优先权断言。本文只描述其任务与实现差异，不宣称来源首创。

## 3. 2024–2026 最值得追的四篇

下表只保留四个对当前方案最有决策价值的条目；它们分别覆盖“可学习节点增强”“凝聚结构保持”“社区保持可学习增强”和“社区级表示/对比”。SOLA-GCL 等图级子图方法不塞进节点任务的候选清单。

| 方法 | 论文元数据与节点任务证据 | 方法与项目交集 | 可复用性判断 |
|---|---|---|---|
| PiNGDA | Siqi Huang 等，ICML 2025，PMLR 267:25987–26001。正式论文与作者代码已核；全文方法、节点实验和实现已读。单图节点任务含 Cora、CiteSeer、PubMed、Wiki-CS、Amazon Photo、Coauthor Physics、ogbn-arxiv；另有独立图分类实验。 | 可学习拓扑与属性扰动生成器，和节点 InfoNCE 一起训练；作者做了 GRACE + PiNGDA 的增量比较。不是对抗 min-max，也不使用干净/含噪类别标签选择扰动。 | 四者里最适合先作为“改 GRACE 增强”的基座。作者代码有单图节点训练入口。收益来自原文自监督/线性评估，不能直接推出含噪标签任务收益。 |
| CTAug | Yucheng Wu 等，WWW 2024；论文、官方作者仓库已核。单图节点扩展明确包含 CTAug-GRACE/CTAug-GCA，在 Coauthor-CS、Coauthor-Physics、Amazon-Computers 上评估。 | 用 k-core/k-truss 等凝聚子图结构改变删点/删边概率；还可选 O-GSN 子结构计数学习模块。它直接覆盖“在 GRACE/GCA 上保护 cohesive subgraph”的节点级方向。 | 必须作为结构保持增强先例。先只看拓扑增强模块；O-GSN 会同时改节点特征和学习目标，增加算力与因果归属混淆。公开代码依赖较旧 PyTorch/PyG/PyGCL，复用有移植成本。 |
| CI-GCL | Shiyin Tan 等，ICML 2024，PMLR 235:47579–47606。官方 camera-ready PDF §5.3/Table 8 有单图节点分类：Cora、CiteSeer、PubMed、Amazon Computers、Amazon Photo；报告的划分为 10/10/80。正式论文和作者仓库已核。 | 学习拓扑与特征增强，并通过谱/社区约束尽量在增强中保持社区信息。它有直接节点任务证据；不能说“没有节点实验”。 | 结构思想相关，但比 PiNGDA 更重：谱变化/社区保持计算会带来开销；作者 README 当前公开命令主要展示图分类等任务，未确认一条可直接复跑 Table 8 单图节点实验的入口。方法可作强参考，工程复用成本需核代码。 |
| CL-GCL | Jianqing Liang、Xinkai Wei、Zhiqiang Li，ICML 2026，PMLR 306:73611–73630。官方元数据/摘要和作者仓库已核；由于当前抓取通道未能读取官方全文，以下方法细节以作者公开实现为依据，不能冒充全文精读。 | 社区锚点与节点表示之间的加权局部关系用于对比，代码含 Planetoid/ClusterData 单图节点路径、node-to-community membership/weight 和 community-level negative relation。它显式保留对比负关系，不是 raw negative-free。 | 与“粒球作为局部表示单元、节点对社区/球软关系、局部几何对比”最接近，是本项目最关键的 2026 近邻和基线。仓库较轻，但 README、依赖和复现说明有限；论文/代码结果不能直接替代本项目噪声标签协议。 |

### 3.1 PiNGDA：为什么推荐作增强基座，但不推荐直接当新方法结论

PiNGDA 的核心是学习“对当前对比预训练目标有益”的噪声。拓扑生成器在现有边上产生可学习的删边扰动，属性生成器产生条件特征扰动，训练时与节点 InfoNCE 协同。节点实验中作者把自监督预训练与线性分类评估分开；论文不训练一个识别错标标签的节点分类头。它的 pi-noise 理论目标从作者定义的 GCL loss 构造，不是目标类别标签熵。因此：

1. 将 PiNGDA 迁移到本项目，首先能检验“GRACE 的固定随机增强是否限制表征”；不能检验“增强器已学会纠正错标”。
2. PiNGDA 学出来的边扰动可能利用当前 InfoNCE 的正负关系来降低训练损失；这不自动等价于保护粒球语义或提高噪声监督精度。
3. 不要把 PiNGDA 误称成对抗式增强。论文将 AD-GCL 作为已有对抗增强路线讨论；PiNGDA 的目标是学习 beneficial noise，不是 augmenter 与 encoder 围绕同一性能指标做 adversarial min-max。
4. 原文 Table 5 的 GRACE 对照在多个节点数据集有提升，但量级并不统一；例如 Photo 的差距很小。无论总体平均如何，这些结果都不是本项目指定 split、噪声比例和下游 LCSS 的配对结果。
5. 对 12GB AMD GPU：从算法看，单图节点生成器主要沿已有边和节点特征运算，Cora/CiteSeer 级 pilot 有机会放入 12GB；但作者实验主要使用 24GB NVIDIA 4090，较大任务还用 40GB A100，原文并未验证 AMD/ROCm。全图节点对比的相似度矩阵也可能成为显存瓶颈。结论只能是“值得先做小图资源试验”，不能说“12GB AMD 已确认可跑”。还需先核对当前 ROCm、PyTorch/PyG 和项目专用代码路径。
6. 一旦要用 PiNGDA 输出给 LCSS，固定预训练、原图推理和下游冻结策略，和 GRACE 基线对齐；只替换增强器。不要同时更换 contrastive loss、负例策略、球构造或去噪投票，再把整体差值归因到“新增强”。

### 3.2 CTAug 与 CI-GCL：结构保持是已有方向，不代表粒球等于社区

CTAug 的节点分类实现不仅是图分类论文中的类比；作者 README 直接列出 GRACE/GCA 和对应 CTAug-GRACE/CTAug-GCA 节点版本。它针对的是凝聚拓扑结构（如 k-core、k-truss）的保留倾向。它构成一个很具体的先例：仅提出“让局部凝聚结构在两个视图中少被破坏”，不足以支撑创新性。可追问的差异应是：粒球由何种信号定义、它是否代表结构凝聚、它如何给出可校准的弱关系、以及它比原作者的拓扑保留概率增加了什么。

CI-GCL 的 camera-ready 明确补有单图节点结果。早期 arXiv v1 HTML 与最终会议 PDF 的内容版本不完全一致；做节点任务判断应以 PMLR 正式论文链接的最终 PDF 为准，而非只看 v1 HTML。其“community-invariant”约束的是图谱/结构社区变化，不是“同一标签类别一定不变”，也没有证明用户构造的 granular balls 是社区真值。让粒球直接指导 CI-GCL 式保持，会把“球是可靠语义组”这个未经验证的前提写进增强器。应先测球成员、社区结构与标签纯度的对应关系，再决定是否把球用作增强先验。

### 3.3 CL-GCL：近期方法表里不能遗漏的直接近邻

CL-GCL (Comprehensive and Lightweight Graph Contrastive Learning) 是本轮筛选出的最重要 2026 近邻。作者代码公开了 Planetoid/ClusterData 的节点任务路径，并在损失实现中把社区/聚类锚点与节点表示关联，同时保留社区间负关系。它与当前“球级局部表示、软成员关系、节点判别和对比关系并存”的设想有明显构件交叉。

含义是双向的：

- CL-GCL 是对照对象：要验证粒球是不是比其社区锚点和局部成员加权关系更适合噪声标签下的 GB-LCSS。
- 它也可能是更合适的新基座：如果本项目重点是把局部社区/粒球表示加入节点级 GCL，应先确认 CL-GCL 的单图实现、损失和节点任务协议，再决定 PiNGDA 是否仍是唯一首选。
- 它不是 rawnegfree 的替代说法。项目历史中的 rawnegfree 塌缩，不足以推断 CL-GCL 同样“无负例”；应按其真实负关系项实现评估。反过来，CL-GCL 有社区级负关系，也不能据此保证对本项目噪声标签安全。
- 当前抓取核验局限：正式 PMLR 元数据/摘要和作者仓库可核，未完成官方正文 PDF 的方法级全文核读。因此上面不对未核的定理、表格精度或具体消融下结论；应在实施或论文 claim 前继续核全文与代码版本。

## 4. 近邻与反证：修改增强是否真是最值得做的事

### 4.1 图级子图增强不能当单图节点证据

SOLA-GCL（AAAI 2025）通过子图划分与学习式增强处理图对比，但官方任务是图分类。它可作为“子图增强如何在多个图样本中工作”的设计参考，不能写成已验证的单图节点增强结果，也不能说作者研究了每个根节点的局部子图节点对比。

相同原则也适用于 GraphCL/JOAO：一个方法在图分类中使用 subgraph sampling，不意味着它产生每个节点的子图表示。本文把任务差异作为筛选条件，不把子图一词当迁移证据。

### 4.2 学习型和对抗型增强不是一个概念

PiNGDA 是可学习扰动生成器；AD-GCL 是显式 adversarial graph augmentation 代表。若主张需要“对抗性地逼出鲁棒表示”，应引用并比较 AD-GCL 机制，而不是将所有 trainable augmenter 都称为 adversarial。GCA 是由重要性规则调节删改概率，JOAO 是自动选预定义增强对，PiNGDA 是学习可微噪声，三者的优化对象不同。

### 4.3 增强自由和遮蔽重构反证“必须再设计两视图”

AFECL（AAAI 2025，Edge Contrastive Learning: An Augmentation-Free Graph Contrastive Learning Model）用边级对比和节点对身份关系做节点表示，不以传统随机双视图增强作为核心。它改变了对比对象与目标，所以不能直接替换 GRACE 做同协议比较；但它提醒我们，改进也可能来自对比结构本身而不是增强器。

GraphMAE（KDD 2022）和 GraphMAE2（WWW 2023）以特征 masking 和重构/潜变量预测为自监督任务，不是 GRACE 式正负样本对比。它们可回答“单图节点预训练是否必须使用实例对比”的反问题，却不能被错误归为新的随机图增强算法。若资源预算只够一个反证对照，应挑一个与现有项目最可比的预训练目标，固定编码器和下游；不能把完整 MAE 管线的成绩当作只改增强的消融。

### 4.4 社区保持假设可能伤害粒球本身

粒球可能跨越结构社区但共享属性语义，也可能属于同一结构社区却涵盖不同标签；边界节点、异配边和少数类更可能违背“组内不变”。所以：

- 不先将球内边保留、球间边删除写成规则。那会把待验证机制伪装为增强先验。
- 先报告球与结构社区、标签类别、边界位置的匹配度；所有标签分析只用于冻结模型后的离线诊断。
- 若球的结构凝聚度/标签纯度低，不应继续用球语义约束 CI-GCL、CTAug 或 PiNGDA。
- 对社区/子图保持机制，最合适的角色可能是弱可靠性权重或诊断信号，而不是硬不变性教师。

## 5. 对 PiNGDA + rooted-context VICReg + LCSS 的审查

这个方案可作为有边界的候选，但目前不应把“更有希望”写成已获证据。它有一个合理的因果顺序：先做无含噪 CE 的自监督预训练，再冻结空间跑原 LCSS，从而不把监督分类、增强、表示和粒球去噪同时改掉。其不足如下。

1. **PiNGDA 的目标与标签噪声目标有距离。** PiNGDA 优化 InfoNCE 驱动的扰动收益，不读有噪类别标签。即使它在干净线性探针上提高，也可能没有帮助标签错标修正；必须以项目的 label-noise + 原始 LCSS 结果判断。
2. **rooted-context VICReg 不是增强器的球感知版本。** 如果 context loss 只约束同一根节点跨视图的上下文向量一致，而生成器仍只由原 PiNGDA-InfoNCE 更新，球没有参与删边概率或特征扰动的学习。它是独立的表示正则/辅助任务，不能包装成“粒球引导学习增强”。
3. **平均邻居上下文不等于独立子图编码。** 当每个 h_j 已由全图消息传递得到，邻居表示可能已经含根节点及更大感受野信息；根节点集合也高度重叠。应如主文 04 所列，比较固定邻域均值、更深/额外传播、打乱上下文和真正的 LS-GCL 局部子图编码，才能判断增量来自“局部上下文学习”。
4. **小权重不能保证 H 保持判别。** H、C 共享编码器时 VICReg 梯度仍会改变 θ；所谓 H 分支仍有语义，仅是结构上保留输出，不等于表征不受影响。要看 λ=0、λ>0 的 H-only 探针、分类/对比损失、梯度范数或梯度夹角，并检查类别边界/少数类退化。
5. **VICReg 防坍塌不等于防语义过平滑。** 方差和协方差项能抑制常数表示，但所有节点的上下文可能依旧过度相似，或丢掉标签判别；小权重也不是证据。用表示有效秩、类间间隔、邻域语义混合和 downstream accuracy 一起看。
6. **R 的新增信息要和分球作用分开。** R 若只是冻结 H 的确定性邻居池化，可能是冗余平滑；若是独立训练分支，才有新增学习变量。需要 H-only vs C-only vs [H,C]，再比较同一 H/C/R 的 GB、kNN、k-means，不然模型表征和构球器两种收益混在一起。
7. **历史证据提高了这一候选的举证门槛。** 2026-10-12 冻结 H 的多尺度/增强稳定性诊断没有独立增量（AUROC 平均变化约 −0.000935；GB Full 修正准确率约比 kNN 低 2.75pp）。这不等于否定训练期上下文学习，但说明“再加一个稳定性/多尺度信号”必须证明确有新机制，不能沿用已失败的简单稳定性解释。
8. **这不是 rawnegfree 的同一方法，也不能借其历史失败直接判死刑。** PiNGDA 保留 InfoNCE 节点负关系；VICReg 是额外辅助项；LCSS 在冻结空间后处理。应单独复现与对照，避免把项目记忆中的 rawnegfree 塌缩泛化到新目标。
9. **PiNGDA 与 CL-GCL 的基座选择需按研究重心决定。** 目标若为增强替换/GRACE 延续，先 PiNGDA；目标若是粒球和社区表示的对比建模，先 CL-GCL 对照，之后再看 PiNGDA 是否带来独立互补收益。PiNGDA+CL-GCL+VICReg 不应作为第一轮组合。

## 6. 建议的最小、可归因筛选顺序

保持项目原 split、噪声生成、seed、干净验证选模、指标和 LCSS 参数；用低成本 pilot 先查可运行性，正式比较前登记成功门槛。建议顺序不是要求本轮启动实验，而是后续如获用户要求可据此实施：

| 阶段 | 对照 | 要隔离的因果问题 | 先停条件 |
|---|---|---|---|
| A | 原 GRACE + 原 LCSS；PiNGDA + 同一 LCSS | 只替换增强，PiNGDA 是否在项目含噪标签协议下改善最终学生/教师 | PiNGDA 不能稳定复现作者节点管线，或无实用下游增益则暂不加上下文支路 |
| B | 同一 PiNGDA 编码器 + H-only LCSS；再加 context VICReg、相同 LCSS | 训练期上下文正则是否给 H/R 提供新增信息 | 相对 λ=0 无可靠下游收益，或类边界/少数类明显退化则停止 |
| C | 同一已冻结 H/R 分别用 GB、kNN、匹配 k-means/随机分组 | 粒球是否比普通局部聚合独特有用 | R 下 GB 不胜可比替代，拒绝粒球特有收益；仍可报告“上下文表示改善” |
| D | 将 CL-GCL 原节点方案作为独立基座，不与 PiNGDA 混合 | 社区锚点/成员权重型节点 GCL 是否比随机增强路径更贴合粒球任务 | 按代码/数据/损失版本核齐，否则只作定性近邻，不搬论文准确率 |

先做作者 PiNGDA 代码路径版本与项目图、特征格式的兼容检查，再在 Cora/CiteSeer 级小图量显存、步时和显存峰值；AMD ROCm/图算子可用性必须实测。12GB 不保证跑通 full-batch 训练、超大图或所有重复种子。若 PiNGDA 内部负相似度矩阵是瓶颈，应先确认分块是否保持作者损失和梯度，不可静默换采样估计后仍称复现。

不把“CL-GCL 摘要/作者仓库”包装成已复现论文成绩。以作者版本跑通后，固定同一任务协议并和 PiNGDA/GRACE 配对重跑，才能作量化排序。如果 CL-GCL 文本公式和代码有不一致，按版本逐项记下，不能只挑对本方法有利的实现解释。

## 7. 对旧失败的继承边界

本笔记延续项目中已确认的失败教训：

- raw negative-free 训练曾塌缩；BNnegfree 是另一个设置，不能混为一谈。PiNGDA/CL-GCL 保留的目标与负关系必须按各自代码核实。
- 多轮 GB-view 增强替换、负样本过滤没有建立稳定独立增益；不能因为这次方法更“可学习”或“社区感知”就忽略旧失败。
- 固定 H 的多尺度/增强稳定性诊断没带来独立信号；新的根上下文训练目标需要证明信号来源不同。
- 历史 C1 相对其自身 no-ball 有改善；当前稿 GB-LCSS 是不同的冻结表示/目标修正管线，不能将两者写成同一个方法。当前稿粒球尚未稳定胜过有竞争力的 kNN/其他局部规则；不能把“某个旧方案有效”扩大成“粒球普遍优于近邻”。
- 项目旧 C1 现存第二阶段配置是 negfree/negfree_bn/phase2 CE 相关分支，没有证明“GRACE InfoNCE 与含噪 CE 同时训练必然失败”。所以本候选第一阶段不加 noisy CE 是为了变量控制，不得声称已被旧实验验证为最优。
- 2026-10-12 冻结 H 多尺度诊断中，AUROC 增量约 −0.000935、GB Full 修正精度约低 kNN 2.75pp；具体条件见项目研究记忆及对应实验日志。它限制稳定性信号和该协议，不证明所有训练期上下文都无效。

## 8. 证据状态与阅读记录

- **全文/方法核读**：GRACE、GraphCL、JOAO、GCA 的任务和增强机制按正式论文/作者实现核对；PiNGDA 的正式 ICML 论文与 arXiv 方法、节点实验、作者代码均核对；CTAug 论文与作者 README 的节点分支核对；CI-GCL 以 PMLR camera-ready PDF（特别是 §5.3/Table 8）和作者仓库核对。本文没有据摘要推断其节点任务。
- **部分阅读/仓库审阅**：CL-GCL 为 PMLR 元数据/摘要加作者代码核验；尚未完成官方全文的方法级审读，性能数字、公式细节和论文完整消融不作本文结论。
- **任务边界核对**：SOLA-GCL 官方 PDF 属图分类，不作为单图节点方法证据。AFECL 是增强自由、边级对比；GraphMAE/GraphMAE2 是 mask 重构/潜变量预测路线，作为反证，不冒充 GRACE 增强算法。
- **项目历史**：用户给定的 99 研究记忆和 03 重设计建议已读；另外审阅了主文 04，提出的精确修订点通过主代理回报，没有改动该文件。

## 9. 主要一手来源与作者代码

### 经典节点/图级 GCL

- GRACE, “Deep Graph Contrastive Representation Learning,” arXiv:2006.04131 (2020): [论文](https://arxiv.org/abs/2006.04131) · [作者代码](https://github.com/CRIPAC-DIG/GRACE)
- GraphCL, “Graph Contrastive Learning with Augmentations,” NeurIPS 2020: [NeurIPS 论文页](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) · [作者代码](https://github.com/Shen-Lab/GraphCL)
- JOAO, “Graph Contrastive Learning Automated,” ICML 2021, PMLR 139:12121–12132: [论文](https://proceedings.mlr.press/v139/you21a.html) · [作者代码](https://github.com/Shen-Lab/GraphCL_Automated)
- GCA, “Graph Contrastive Learning with Adaptive Augmentation,” WWW 2021: [作者论文 PDF](https://sxkdz.github.io/files/publications/WWW/GCA/GCA.pdf) · [DOI](https://doi.org/10.1145/3442381.3449802) · [作者代码](https://github.com/CRIPAC-DIG/GCA)

### 2024–2026 节点增强、社区/粒球近邻

- PiNGDA, “Learn Beneficial Noise as Graph Augmentation,” ICML 2025, PMLR 267:25987–26001: [正式论文与元数据](https://proceedings.mlr.press/v267/huang25ao.html) · [全文 HTML](https://arxiv.org/html/2505.19024) · [作者代码](https://github.com/h-4777/PiNGDA)
- CTAug, “Graph Contrastive Learning with Cohesive Subgraph Awareness,” WWW 2024: [论文 HTML](https://arxiv.org/html/2401.17580) · [作者代码（含节点 GRACE/GCA 分支说明）](https://github.com/wuyucheng2002/CTAug)
- CI-GCL, “Community-Invariant Graph Contrastive Learning,” ICML 2024, PMLR 235:47579–47606: [PMLR 元数据与正式 PDF](https://proceedings.mlr.press/v235/tan24b.html) · [camera-ready PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/tan24b/tan24b.pdf) · [作者代码](https://github.com/ShiyinTan/CI-GCL)
- CL-GCL, “Comprehensive and Lightweight Graph Contrastive Learning,” ICML 2026, PMLR 306:73611–73630: [正式论文与元数据](https://proceedings.mlr.press/v306/liang26y.html) · [作者代码](https://github.com/a-hou/CL-GCL)
- SOLA-GCL, “Subgraph-Oriented Learnable Augmentation Method for Graph Contrastive Learning,” AAAI 2025: [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/34189) · [官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/34189/36344)

### 反证：增强自由与掩码重构

- AFECL, “Edge Contrastive Learning: An Augmentation-Free Graph Contrastive Learning Model,” AAAI 2025: [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/34044) · [官方 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/34044/36199) · [作者代码](https://github.com/YujunLi361/AFECL)
- GraphMAE, “Self-Supervised Masked Graph Autoencoders,” KDD 2022: [论文](https://arxiv.org/abs/2205.10803) · [作者代码](https://github.com/THUDM/GraphMAE)
- GraphMAE2, “GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner,” WWW 2023: [论文 PDF](https://keg.cs.tsinghua.edu.cn/jietang/publications/WWW23-Hou-GraphMAE2.pdf) · [作者代码](https://github.com/THUDM/GraphMAE2)
- AD-GCL, “Adversarial Graph Augmentation to Improve Graph Contrastive Learning,” NeurIPS 2021: [NeurIPS 论文页](https://papers.neurips.cc/paper_files/paper/2021/hash/854f1fb6f65734d9e49f708d6cd84ad6-Abstract.html) · [作者代码](https://github.com/susheels/adgcl)


