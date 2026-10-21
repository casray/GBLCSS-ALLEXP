# 论文正文初稿 v1（中文审阅版）

> 供导师审阅。英文投稿版将按本稿翻译并压缩。所有数字均为 5 个随机种子的均值±标准差，出处见文末数据来源表。
> **命名约定（全文统一）**：方法主名 **LCSS**（Local-Consensus Soft Supervision，局部共识软监督）；其分组步骤采用粒球计算的递归二分实现，正文统一称"局部共识分组（球式实现）"；历史工作代号 SoftDist-MLP 不在正文出现。v1.1（2026-07-06）：按内部审查修正 13 处——表 B 蒸馏行数字勘误（曾误用 kNN teacher 数字）、"全部/稳定/显著"措辞收窄、H 上界拆分为线性/MLP 两行、α 公式与 fallback 规则补全、协议资源边界显式化。v1.2（2026-07-06）：附录 A-H 产出物生成完毕（`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706`，23 项对拍全 pass）。**v1.3（2026-07-06）：附录内容全部内嵌——主表五个数据集块、显著性汇总计数、graph-dependent 参考表（Phase 24 数字核验）、敏感性/β 审计结论、特征噪声与 t-SNE 与低标签率的图说明文字、CV 坍缩脚注全文——附录不再是文件指针，而是可直接审阅的实体章节。**

---

# 标题

**基于冻结对比表征局部共识软监督的无图推理标签噪声鲁棒节点分类**

Local-Consensus Soft Supervision on Frozen Contrastive Representations for Label-Noise-Robust Graph-Free Node Classification

---

# 摘要

图节点分类在实际部署中面临两个相互叠加的困难：训练标签含有噪声，以及推理阶段往往无法使用图结构（延迟受限的在线服务、拓扑隐私、图数据不可携带等）。现有的图上标签噪声方法（如 NRGNN、DND-NET）其去噪机制与推理过程均依赖邻接矩阵；而 graph-free 推理方向的 GNN-to-MLP 方法（如 GLNN、VQGraph）以逼近干净标签下的 GNN 精度为目标，其鲁棒性研究集中于特征噪声、特征缺失或标签稀缺——训练标签噪声在这条文献线中是缺失的。

本文提出局部共识软监督（LCSS）：先用图对比学习（GRACE）在完整图上无监督地预训练节点表征并冻结；在冻结表征空间中用递归二分法将节点划分为若干局部组（粒球式聚合，所有数据集共用一套固定构造参数）；每组内用含噪训练标签投票，生成半硬软标签（β=0.5：组内标签分布与多数类 one-hot 各占一半）；最后仅用节点特征训练一个两层 MLP。推理时不接触图、不接触表征编码器。目标构造不使用任何干净标签、不依赖模型自身预测，整个流程非迭代、一遍完成。

在 5 个数据集（涵盖引文、合著、共购三类图）× 5 种噪声设置（uniform 0.2/0.4/0.6、pair 0.4、instance 0.4）上，与 11 个移植到同一 MLP 骨干的标签噪声学习策略（2018–2025）对比：LCSS 在 Cora、CiteSeer、Amazon-Photo 的全部设置中最优，Coauthor-CS 4/5 设置最优，PubMed 是如实报告的边界。将局部组随机打乱后增益全部消失（25 个设置全部逆转，平均下降约 9 个百分点），证明增益来自真实局部结构。我们进一步给出一条可选蒸馏扩展，使纯特征输入的学生网络达到并超过同协议 GCN 的绝对精度。

我们同时主动报告三项诚实消融：分组形式可被同表征空间的 kNN/kmeans 共识替代（贡献在配方组合而非分组算法本身）；若允许推理期直接使用对比表征，其线性分类器远强于一切纯特征方法（本文贡献限定于 graph-free 推理部署，蒸馏扩展可在该限定下追平此上界）；共识目标在完全干净标签下可有约 1 个百分点的代价。

**关键词**：标签噪声；节点分类；graph-free 推理；图对比学习；局部共识；知识蒸馏

---

# 1 引言

图神经网络（GNN）通过消息传递聚合邻域信息，在节点分类任务上表现出色。但在工业部署中，GNN 的推理需要取回目标节点的多跳邻居，在延迟敏感的在线服务中代价高昂；某些场景下推理端甚至根本拿不到图（拓扑涉及隐私、图数据库与服务分离等）。因此"训练期利用图、推理期只用节点特征"的 graph-free 推理范式近年形成了一条成熟的文献线：GLNN 用 GNN teacher 蒸馏 MLP，NOSMOG、VQGraph、KRD 等工作沿蒸馏信号与结构编码方向持续改进。

与此并行，真实图数据的标注常来自自动抽取或众包，训练标签的错误率不可忽视。图上标签噪声学习（graph LNL）同样是活跃方向：NRGNN、RTGNN、DND-NET 等方法利用图结构识别与修正噪声标签。

**问题在于这两条线的交集。** 图 LNL 方法的去噪机制与推理都依赖邻接矩阵，无法在 graph-free 部署下使用；而 GNN-to-MLP 文献虽然把"鲁棒性"作为常规评测维度，但我们对该家族 13 篇代表工作的逐篇核验显示：其鲁棒性研究集中于**特征**高斯噪声（NOSMOG、PGKD、VQGraph 使用同一扰动协议）、特征缺失与标签稀缺（AdaGMLP）、或特征扰动下的教师预测稳定性（KRD）——**训练标签噪声从未被作为中心问题、用标签噪声学习的标准协议（多噪声类型、多噪声率、与 LNL 策略族系统对比）研究过**。GSDN 附带报告过单一 asymmetric 噪声类型下与 GNN 系方法的对比曲线，但未与任何 LNL 策略对比。

本文把训练标签噪声作为中心问题，研究 graph-free 推理约束下的鲁棒训练。我们的出发点是一个通道层面的观察：结构信息进入 MLP 已有三条被充分探索的通道——**输入侧**（将传播特征或位置嵌入拼入输入，如 DND-NET、NOSMOG）、**表征侧**（对齐 MLP 与 GNN 编码器，如 SimMLP）、**预测侧**（teacher 蒸馏或自蒸馏，如 GLNN、GSDN）。这三条通道都把结构用于增强模型，而监督信号本身仍是（可能含噪的）原始标签或由其训练出的 teacher 预测。我们走第四条通道——**监督侧**：把结构转化为监督信号本身的去噪机制。具体地，冻结的对比表征定义了局部邻域，邻域内含噪标签的集体投票构成软监督目标；错误标签在局部多数的稀释下被修正。由于目标构造只依赖表征几何与投票、不依赖模型自身预测，该机制没有自训练方法的噪声自强化回路；由于表征由无监督对比学习获得，噪声标签无法污染分组几何。

本文贡献如下：

1. **问题与通道**：据我们所知，首个以训练标签噪声为中心问题、通过监督侧通道（表征空间局部共识构造抗噪软标签）处理 graph-free 节点分类的系统研究。
2. **受控主实验**：在 5 数据集 × 5 噪声设置上，同一 X-only MLP 骨干下与 11 个 LNL 策略（含 2023 年的 LogitClip、ANL 与 2025 年的 CT）对比，LCSS 在三个数据集全部设置最优、一个数据集 4/5 最优；随机打乱分组的对照证明增益来自真实局部结构（25/25 设置逆转）。
3. **精度扩展**：将共识软标签移至 teacher 侧再蒸馏，X-only 学生在三个数据集上全部超过同协议 GCN 的绝对精度，并在资源对齐的比较中超过现有 graph-free 方法。
4. **诚实的归因与适用域刻画**：包括分组形式的可替代性消融（kNN/kmeans 等效）、共识单元票数的机制分析、学生容量分析（PubMed 边界被识别为容量依赖现象）、三种表征空间的对照（原始特征/原图传播/对比表征）、以及表征直接分类的上界披露。每一项潜在的审稿质疑都被转化为论文自身的分析内容。

**范围声明**：本文主表比较的是 train-target 类抗噪策略（所有方法只使用含噪训练标签）；利用无标签节点的蒸馏属于另一条资源轴，我们在扩展节（第 6 节）单独对照并展示两者如何结合。本文协议为 transductive；inductive 扩展（需要在子图上重新预训练对比编码器）留作未来工作。

---

# 2 相关工作

**通用标签噪声学习。** 样本选择类方法基于"干净样本先被拟合"的记忆效应筛选小损失样本（small-loss、Co-teaching、JoCoR），鲁棒损失类方法设计对噪声不敏感的损失函数（SCE、ELR、ANL），正则类方法约束模型输出（LogitClip），混合类方法结合半监督学习（DivideMix）或特征谱过滤（FINE）、置信度跟踪（CT）。这些方法与图无关；本文将其中 11 个移植到统一的 X-only MLP 骨干作为主表对照，并发现其中样本选择类的优势依赖学生容量（第 5.3 节）。

**图上标签噪声学习。** NRGNN 用边预测连接含噪标签节点与无标签节点，RTGNN 双网络互校，DND-NET 用 PPR 传播特征与邻居不确定性加权伪标签，RNCGLN 用图对比与自注意力联合自纠正。这些方法的去噪机制与推理均需邻接矩阵，部署设定与本文不同；我们在附录参考表中报告其同协议数字并标注该差异。DND-NET 虽以"简单模型"为卖点，但其 MLP 的输入是 L 步传播特征，推理仍需图。

**GNN-to-MLP 与 graph-free 节点分类。** GLNN 确立了 GNN teacher 蒸馏 X-only MLP 的范式；后续工作沿三个方向推进：更丰富的蒸馏信号（FF-G2M 的全频率分解、VQGraph 的结构 code、PGKD 的类原型）、更可靠的蒸馏（KRD 量化教师知识点对特征扰动的稳定性并上采样可靠监督）、以及工程扩展（AdaGMLP 的集成、ProGMLP 的渐进推理、SALE-MLP 的结构感知嵌入）。理论方面，最近的工作从神经正切核角度证明了软标签蒸馏会使 X-only 学生的核与邻接矩阵对齐、并偏好低频平滑特征——这为"软监督能向纯特征学生传递结构偏置"提供了理论依据，与本文监督侧通道的动机一致（我们的软标签来自表征局部共识而非 GNN logits，属类比性支撑）。**该家族的鲁棒性研究不含训练标签噪声**（见引言）；机制上与本文最近的是 PGKD 的类原型——但其原型是按标签分组的全局类均值，标签含噪时类中心直接被污染，而本文的局部共识不依赖标签正确性。就教师端监督的处理而言，已有工作可归为上采样可靠监督（KRD）与下采样不可靠监督（RKD-MLP）两类；本文的共识软标签属于第三类范式：**修复被污染的监督**。NOSMOG 推理需要 DeepWalk 位置嵌入、SA-MLP 需要邻接行输入，均非严格 X-only。GSDN（arXiv，未正式发表）用邻域自蒸馏实现 graph-free 推理并附带单一 asymmetric 噪声评测，其 label-level 自蒸馏会将含噪标签扩散给邻居监督——我们在表 B 中实测了该效应。

**粒球计算。** 粒球计算以自适应多粒度划分为核心思想，在分类、聚类与标签噪声过滤中有系列工作（MgCNL、MgBIF 等）。已发表的成功用法多为端到端训练中球结构参与表征塑造；本文的用法不同：在冻结表征上做一次性离线聚合。我们如实报告：在此角色下，球式聚合与同表征空间的 kNN/kmeans 共识精度相当（第 5.3 节）——本文贡献在于设定与配方组合，而非新的球算法；球式实现的工程优点是免 k 扫描的固定构造。

---

# 3 方法

## 3.1 问题设定

给定图 G=(V, E)，节点特征 X∈R^{N×d}，训练子集 V_L 的标签 Ỹ_L 以噪声率 ρ 被污染（uniform/pair/instance 三种噪声模型），验证与测试标签仅用于评测。目标：训练一个分类器 f，**推理时只接受节点特征 x_v 作为输入**（不使用邻接、不使用任何图衍生嵌入），最大化干净测试集精度。训练期允许使用图（这是 graph-free 文献线的标准设定：GLNN、GSDN、VQGraph 同此）。

## 3.2 LCSS：四步流程

**第一步：冻结对比表征。** 用 GRACE（图对比学习，边丢弃+特征掩码双视图，InfoNCE 目标）在 (X, A) 上无监督预训练，得到节点表征 H∈R^{N×256}，随后冻结。**资源边界说明**：预训练使用完整图的邻接与全部节点特征、不使用任何标签（transductive 协议，与 GLNN/GSDN/VQGraph 的训练期资源约定一致）；graph-free 约束只施加于推理端。关键点有二：(i) 预训练不接触任何标签，因此噪声无法污染表征几何；(ii) 必须使用对比目标——我们在第 5.4 节展示，将含噪监督信号以任意形式混入表征学习（多任务、微调、用训练后网络的隐层）都会破坏表征的局部可聚类性（分组数从约 50 塌缩至 1），监督表征的线性可分性并不等于局部可聚类性。

**第二步：局部共识分组。** 对行 L2 归一化的 H 做递归二分（2-means）：从全体节点出发，若分裂使簇内平方和的改善超过阈值则继续分裂，直到无显著改善或达到上限。全部数据集共用同一套固定构造参数（max_balls=64，min_ball_size=32，split_improvement=0.05），全程未按数据集调整。该过程产生 22~133 个大小自适应的局部组（分组数由数据自身几何决定，上限在多数数据集上并未触发）。此实现源自粒球计算的递归划分思想；我们不主张该形式不可替代（见 5.3 节消融），其工程优点是无需按数据集扫描邻域参数。

**第三步：β=0.5 半硬共识软标签。** 对每个局部组 B，统计组内训练节点的含噪标签分布 p_B 与多数类 m_B，构造组级目标：

t_B = 0.5 · p_B + 0.5 · onehot(m_B)

每个训练节点 v 的最终目标由其原始含噪标签与组级目标按置信度混合：

y_v^target = (1−α_v) · onehot(ỹ_v) + α_v · t_B(v)

其中置信度由组内投票裕度经多类修正得到：α_v = max(0, (margin_B − 1/C) / (1 − 1/C))，margin_B 为组内多数类票占比，C 为类数——组内多数越显著，共识目标权重越大；当组内投票不优于随机（margin ≤ 1/C）时 α_v = 0，目标退回原始含噪 one-hot。**Fallback 规则**：训练节点数不足的局部组（无有效投票）同样退回原始含噪 one-hot，不做修正。直觉：噪声标签在其所在局部组内通常是少数派，共识目标以集体投票稀释个体错误；β=0.5 的半硬化在"保留分布信息"与"强化多数类信号"之间折中（消融见 5.3 节：β=0.5 是唯一在全部试点设置下不劣于纯软目标的取值）。**目标构造全程只使用训练子集的含噪标签**——不接触验证/测试标签、不使用模型预测；后者使本方法与自训练类方法有本质区别：不存在"模型错误→进入目标→强化错误"的回路。

**第四步：X-only 学生。** 两层 MLP（隐层 64，约 9 万参数）以软交叉熵在 {(x_v, y_v^target)} 上训练。推理即普通 MLP 前向。

## 3.3 复杂度

分组为递归 2-means，O(N log N · d)；目标构造 O(N·C)。学生单节点推理 O(d·F + F·C)，与图规模和边数无关。对照：NRGNN 需 46.5 万参数与迭代式训练。GRACE 预训练是一次性成本，且同一 H 在全部噪声类型与噪声率间复用（表征对噪声构造性免疫）。

## 3.4 可选扩展：共识 teacher 蒸馏

train-only 的 LCSS 只利用了 20% 训练节点的监督，绝对精度受监督覆盖限制。当部署需要更高绝对精度且训练期预算允许时，可将同款共识软标签用于训练一个 GCN[X‖H] teacher，再将 teacher 对全部节点的软预测蒸馏给同一个 X-only MLP。该扩展保持推理端 graph-free 不变，实验见第 6 节。

---

# 4 实验设置

**数据集**：Cora（2.7k 节点/7 类，引文）、CiteSeer（3.3k/6）、PubMed（19.7k/3）、Coauthor-CS（18.3k/15，合著）、Amazon-Photo（7.6k/8，共购）。划分 20%/10%/70%（训练/验证/测试）。

**噪声协议**：仅训练标签注入噪声——uniform（0.2/0.4/0.6）、pair（0.4）、instance-dependent（0.4）共 5 种设置；验证与测试标签干净、仅用于评测。

**对比方法**：全部方法共用同一个两层 X-only MLP 骨干与同一套训练超参，唯一差异为抗噪策略：CE、Co-teaching (2018)、Small-loss (2018)、SCE (2019)、ELR (2020)、JoCoR (2020)、DivideMix-lite (2020)、FINE (2021)、LogitClip (2023)、ANL (2023)、CT (2025)。其中 FINE/DivideMix-lite/CT/LogitClip/ANL 为核心机制移植版（标注 adapted），非原域官方复现。另设 **shuffled 对照**：保持 LCSS 全流程不变、仅将局部组归属随机打乱——用于检验增益是否来自真实结构。

**报告口径**：5 个随机种子的 mean±std；配对比较附逐种子方向计数与 paired t-test；小于 1 倍标准差的差异一律表述为 comparable。β=0.5 为初始探索后冻结的保守默认值（唯一在全部试点设置不劣于纯软目标的取值），冻结后在 Coauthor-CS、Amazon-Photo 及 pair/instance 噪声上作 out-of-sample 确认；附录给出选择审计表。

---

# 5 主要结果

## 5.1 主表：同骨干下与 11 个 LNL 策略的对比

代表性数字（uniform 0.4，完整五数据集 × 五设置表见附表）：

| 数据集 | 最强对照 | LCSS (β=0.5) | shuffled 对照 |
|---|---:|---:|---:|
| Cora | Small-loss 51.95±1.97 | **55.59±1.50** | 43.63±1.62 |
| CiteSeer | JoCoR 49.91±3.37 | **53.83±2.75** | 42.35±2.74 |
| PubMed | DivideMix-lite 78.28±0.54 | 71.90±0.88 | 62.72±4.49 |
| Coauthor-CS | JoCoR 89.38±0.61 | **90.62±0.57** | 87.60±1.03 |
| Amazon-Photo | JoCoR 79.57±0.53 | **81.64±0.92** | 72.06±2.97 |

全表计数：**Cora、CiteSeer、Amazon-Photo 全部 5 种噪声设置最优（15/15）**；Coauthor-CS 4/5 最优（uniform 0.2 与最强对照差 0.25pp，小于标准差，comparable）；**PubMed 在 5 设置中均不敌样本选择族**（Small-loss/JoCoR/CT/DivideMix-lite），但仍稳定优于全部损失族方法（含 2023 年的 LogitClip 与 ANL），且真实结构对 shuffled 全部 5/5 胜出。PubMed 边界的机制分析见 5.3 节——它是容量依赖的现象，而非策略族的绝对优势。

**结构必要性（最强证据）**：shuffled 对照在全部 5 数据集 × 5 设置（25/25）中一致低于 LCSS（逐设置 5/5 种子方向一致），平均差约 9pp，且在我们不敌对照方法的 PubMed 上同样成立（5/5 种子）。软标签的增益来自真实局部结构，而非软化目标本身的正则效应。

## 5.2 表 B：graph-free 家族的资源-性能对照

主表回答"同资源下策略孰优"；表 B 回答"与 graph-free 家族相比我们处于什么位置"。我们将 Graph-MLP 与 GSDN 移植到相同噪声协议（固定其原文默认超参，标注 adapted），与 vanilla MLP、LCSS train-only、GLNN 式蒸馏、LCSS 蒸馏版并列，附资源标注列（训练是否用图/是否用无标签节点）。uniform 0.4 代表性结果：

| 方法（推理均 X-only） | 训练资源 | Cora | CiteSeer | PubMed | Coauthor-CS | Amazon-Photo |
|---|---|---:|---:|---:|---:|---:|
| MLP+CE | 仅含噪标签 | 44.24 | 43.50 | 63.77 | 88.42 | 73.90 |
| LCSS train-only | +图（经冻结表征） | 55.59 | 53.83 | 71.90 | **90.62** | **81.64** |
| Graph-MLP (adapted) | +全图边集对比 | 63.23 | 57.23 | 67.54 | 88.95 | 77.05 |
| GSDN (adapted) | +邻域自蒸馏 | 55.97 | 50.36 | 71.42 | 86.55 | 73.31 |
| GLNN 式蒸馏 | +GNN teacher+全部节点 | 77.95 | 66.09 | 84.32 | — | — |
| **LCSS 蒸馏版（球式 teacher）** | +共识 teacher+全部节点 | **80.60** | **69.40** | **85.01** | — | — |

如实呈现三点：(i) Graph-MLP 凭全图边集正则在 Cora/CiteSeer 上强于 train-only LCSS（资源不同类）；(ii) 大图与结构化噪声下 train-only LCSS 反超（Coauthor-CS、Amazon-Photo 及 pair/instance 全设置，高噪声 uniform 0.6 五数据集均值 57.3 vs Graph-MLP 57.0、GSDN 51.9）；(iii) **资源对齐后 LCSS 蒸馏版超过本文适配的全部 graph-free baselines**（对 Graph-MLP +17.4pp，对 GSDN +24.6pp；kNN 式共识 teacher 变体结果相当——80.99/69.57/85.23，与 5.3 节形式可平替结论一致，如实并列）。GSDN 在高噪声与大图上退化最重（Amazon-Photo uniform 0.6 较我们低 14.2pp），符合其 label-level 自蒸馏扩散噪声标签的机制属性——邻居越多，扩散越广。

## 5.3 消融与归因（诚实分析）

**分组形式可替代。** 将球式分组换成同表征空间的 kNN（k=20）或 kmeans（K 匹配分组数），精度相当甚至 kNN 略优（uniform 0.4 下 +0.3~+2.6pp）。进一步的机制实验定位了原因：**有效变量是每个共识单元内的噪声票数（约 20 票），而非分组形式**——将最小组尺寸调大使每组票数从约 10 提升到约 24 后，球式与最优 kNN 的差距基本闭合（CiteSeer 从 −2.58pp 收敛到 −0.77pp，PubMed 打平）；且大组的标签纯度更低但精度更高——20 票的可靠多数胜过 10 票的高纯度。kNN 的最优 k 在四个数据集上均为 20、干净验证集扫描几乎无损，因此我们不将"免 k"作为性能优势，仅作为固定构造的工程便利。构造参数敏感性：合理范围内为 ±1pp 平台，仅分裂阈值过大（0.1）导致全体塌缩为单组（附录敏感性表）。

**三种表征空间的对照（为什么需要对比表征）。** 同一套 β=0.5 共识公式作用于三种空间：原始特征空间（分组塌缩为单组，失效）；原图邻接空间（含噪标签沿边传播 2 步作目标——被表征空间共识稳定压制 6.9~8.3pp，3 数据集全 5/5 种子，原因是原图传播直接扩散噪声标签，与 GSDN 的机制缺陷同源）；冻结对比表征空间（本文，唯一有效）。此外，用含噪监督训练的普通 GCN 隐层替代 GRACE 构组：结构信号存在（6/6 设置优于 shuffled）但不稳定（仅 1/6 设置以 >0.5pp 优于 GRACE），不能系统替代无监督对比表征。

**学生容量分析与 PubMed 边界的重新定性。** 将全部方法的学生换成线性分类器：LCSS-Linear 在 5/5 数据集上均值优于 Linear+CE 与 linear-shuffled（其中 PubMed 相对 Linear+CE 为弱正——+0.68pp、4/5 种子，其余数据集差距明确）——共识监督不是 MLP 容量的伪影；而**样本选择族在线性学生下崩溃**（PubMed 上 linear Small-loss 69.60 对比其 MLP 版 81.88；linear DivideMix-lite 66.32 对比 78.28），LCSS-Linear 在 PubMed 全部 5 设置反超最强线性选择族 +7.4~+14.3pp。机制解释：选择族依赖"干净样本先拟合"产生的损失分离信号，而该信号本身是模型记忆噪声（容量）的副产品——线性学生几乎不记忆噪声，信号消失，选择既无必要也无依据；共识监督不使用该信号，容量无关。因此 **PubMed 边界是"容量 × 策略族"的交互现象**：弱 MLP 下选择族占优（主表如实保留），线性学生下反转。诚实附注：线性选择族为我们的移植版，其崩溃幅度可能部分含移植欠力；低噪声下 Linear+CE 本身已很强。

**表征上界的披露（范围声明的最强形式）。** 冻结 H 上直接训分类器远强于一切 X-only train-target 方法：线性头达 78.61/68.87/83.87，MLP 头达 73.09/68.69/84.05（Cora/CiteSeer/PubMed，u0.4；两种头各有胜场——小图线性更强、PubMed MLP 头略高，故统称"H-access 上界带"而非单一上界）——超过 train-only LCSS 12~23pp。这条路线推理需携带 GRACE 编码器与图。两点回应：(i) 本文贡献限定于 graph-free 推理部署；(ii) 蒸馏扩展使 X-only 学生达到或略超该上界带（80.60 vs 78.61 等，见第 6 节）——即 graph-free 部署可以不付精度代价。另一个如实报告的现象：将 H 直接拼接进弱 MLP 输入（[X‖H]+CE）在小图上反而崩溃（Cora 48.54，低于 train-only LCSS 7pp）——H 的价值需要经监督侧转化或配合鲁棒策略，裸拼接会被噪声记忆吞噬；这本身构成监督侧通道的又一论据。

**低标签率优势。** 训练标签比例从 65% 降至 5%，LCSS 相对最强选择族的优势从 +0.2~+1.4pp 总体扩大到 +5.6~+6.7pp（6 点 5-seed 曲线，趋势见附录图）：标签越稀缺，局部共识监督越有价值。

**干净标签下的代价（limitation）。** 0% 噪声下 LCSS 相对 CE：Cora +0.44、CiteSeer +2.33、PubMed +0.02、Coauthor-CS −0.88、Amazon-Photo −1.20pp——在特征强可分且标签完全干净时，半硬共识有约 1pp 的代价。本方法面向标签含噪场景。

**特征噪声（家族标准协议）。** 测试特征加高斯扰动 X̃=(1−α)X+αε：相对优势在 Cora/CiteSeer/Amazon-Photo 保持到 α=0.9，但 LCSS 的绝对退化在小图上更陡（Cora 34.0 vs CE 22.8pp）——共识目标加深了对特征承载结构的依赖，这是机制的自然代价；本方法不是特征噪声防御机制（X-only 系统的共同边界）。

## 5.4 为什么必须是冻结的无监督对比表征

四组独立实验给出一致答案：将含噪 CE 以 0.1 权重混入 GRACE 预训练，分组从 52 个塌缩为 1 个（纯度 0.79→0.30，下游 −12.6pp）；用训练后 MLP 的隐层重新构组，同样塌缩；部分线性探针精度更高的自监督表征（如 CCA-SSG、GraphMAE，其 512 维嵌入几何不适配递归二分）同样塌缩为单组——**线性可分性与局部可聚类性是两种性质**，GRACE 的 alignment-uniformity 平衡恰好保住后者；普通含噪监督 GCN 隐层有局部结构信号但不稳定（见 5.3）。

---

# 6 扩展：补齐绝对精度

train-only LCSS 只用 20% 节点的监督，绝对精度距 GCN 约 20pp——这是监督覆盖的机制性差距，加大标签比例无法弥合（65% 标签下仍差约 14pp）。graph-free 文献的标准解法是蒸馏。我们将共识软标签移至 teacher 侧：以其训练 GCN[X‖H] teacher，再将 teacher 全节点软预测蒸馏给同一 X-only MLP。

**结果（5-seed）**：学生达 Cora 80.60±0.69 / CiteSeer 69.40±1.08 / PubMed 85.01±0.93（u0.4），**三数据集全部超过同协议 GCN(X) 的自身精度**（78.77/64.90/83.75）；uniform 0.6 下超过 GLNN 式蒸馏 +5.8/+9.0/+7.4pp（15/15 种子）。X-only 推理的精度短板就此闭合，且资源对齐后超过本文适配的全部 graph-free baselines（表 B）。

**三层诚实归因**（对照实验分解）：(i) 增益主力是 teacher 的 [X‖H] 输入（u0.4 +2.3~+2.7pp，u0.6 +3.9~+7.7pp）；(ii) 共识去噪目标的额外增益在中噪声下边际（+0.2~+0.6pp）、在高噪声下真实且稳定（+1.1~+1.8pp，14/15 种子）——噪声越高，teacher 记忆的噪声越多，共识修正越有价值；(iii) 球式与 kNN 式共识 teacher 相当（与 5.3 节平替结论一致）。结构一致性度量（cut value）显示蒸馏变体的预测-拓扑一致性显著高于 train-only（+0.14~+0.25，全 5/5 种子），与 NTK 理论对蒸馏传递结构偏置的刻画相符。

---

# 7 结论与局限

本文提出 LCSS，以冻结对比表征上的局部共识软标签实现 graph-free 推理下的标签噪声鲁棒节点分类，并以一条可选蒸馏扩展补齐绝对精度。5 数据集 × 5 噪声设置的受控实验中，方法在 Cora、CiteSeer、Amazon-Photo 全部设置最优、Coauthor-CS 4/5 设置最优，PubMed 作为边界如实报告并被归因为容量依赖现象；结构必要性由随机分组对照严格确立（25/25 设置一致）；蒸馏扩展使 X-only 学生超过同协议 GCN。

**局限**（均有实验依据）：(1) 训练期需要完整图上的高质量对比预训练——子图上重训 GRACE 会导致分组塌缩，故本文协议为 transductive，inductive 留待未来工作；(2) 分组形式可被同空间 kNN/kmeans 替代，贡献在配方而非分组算法；(3) 弱 MLP 学生下，损失可分性强的数据（PubMed 型）上样本选择族更优（尽管该优势被识别为容量依赖）；(4) 完全干净标签下有约 1pp 代价；(5) 不防御特征噪声，且共识目标加深特征依赖；(6) 噪声为合成注入（instance-dependent 为最接近真实噪声的代理），真实标注噪声基准在图领域尚缺。

---

# 附录

（产出物位于 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706`，LaTeX 版可直接 `\input`；以下为审阅版实体内容，23 项对拍全部通过。）

审阅用文件索引（直接复制路径打开；不使用本地 markdown 链接格式）：

- 资产清单：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\asset_manifest.md`
- 附录 A 完整主表：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_A_complete_main_table.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_A_complete_main_table.tex`
- 附录 B 配对显著性：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_B_paired_significance.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_B_paired_significance.tex`
- 附录 C graph-dependent 参考方法：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_C_graph_dependent_references.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_C_graph_dependent_references.tex`
- 附录 D.1 构造参数敏感性：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D1_parameter_sensitivity.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D1_parameter_sensitivity.tex`
- 附录 D.2 β 审计：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D2_beta_audit.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D2_beta_audit.tex`
- 附录 E 特征噪声图：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_E_feature_noise_curves.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_E_feature_noise_curves.png`
- 附录 F Cut Value 表：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_F_cut_value.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_F_cut_value.tex`
- 附录 G t-SNE 图：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G_tsne_three_coloring.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G_tsne_three_coloring.png`
- 附录 G 粒球覆盖图：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G2_grace_ball_cover.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G2_grace_ball_cover.png`
- 附录 H 低标签率图：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_H_labelrate_curves.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_H_labelrate_curves.png`
- 附录 H 单调性核验：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_H_labelrate_monotonicity.md`
- 附录 I 表 B 完整版：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\table_B_graphfree_family.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\table_B_graphfree_family.tex`
- 附录 I 蒸馏扩展五臂表：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\extension_denoised_teacher_distill.md`；LaTeX：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\extension_denoised_teacher_distill.tex`

## 附录 A：完整主表

正文 5.1 节仅给出 uniform 0.4 的代表性数字，本附录给出全部 5 数据集 × 5 噪声设置 × 13 方法的完整结果（表 A.1–A.5）。数值为 5 种子测试精度 mean±std（%）；**每列数值最高者加粗**（诚实加粗原则：baseline 最高即加粗 baseline）；`*` 表示 LCSS 相对最强非 shuffled baseline 的 paired t-test p<0.05（n=5）。带 "adapted" 的方法为核心机制移植版，非原域官方复现。

**表 A.1 Cora（引文，7 类）**

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
| --- | --- | --- | --- | --- | --- |
| MLP+CE | 54.96±2.22 | 44.24±2.62 | 32.51±3.38 | 42.83±2.19 | 45.33±2.22 |
| Co-teaching (2018) | 58.62±1.97 | 46.83±1.78 | 33.29±3.48 | 44.26±2.30 | 47.35±2.24 |
| Small-loss (2018) | 59.84±1.29 | 51.95±1.97 | 36.72±1.09 | 46.99±2.77 | 51.82±0.94 |
| SCE (2019) | 53.57±2.17 | 43.60±1.86 | 32.66±3.78 | 41.40±2.04 | 43.29±4.35 |
| ELR (2020) | 52.95±2.38 | 44.28±1.38 | 32.82±3.51 | 41.83±1.75 | 45.93±2.26 |
| JoCoR (2020) | 60.12±2.25 | 51.72±1.30 | 36.92±2.14 | 45.91±2.38 | 52.26±1.01 |
| DivideMix-lite (2020, adapted) | 55.19±2.30 | 44.79±2.05 | 32.30±3.61 | 43.34±2.34 | 44.75±3.03 |
| FINE (2021, adapted) | 54.85±1.98 | 43.54±1.48 | 31.77±3.04 | 41.72±1.11 | 44.76±2.74 |
| LogitClip (2023, adapted) | 58.22±2.48 | 48.79±1.49 | 34.09±2.06 | 45.44±1.14 | 47.35±1.27 |
| ANL (2023, adapted) | 54.24±1.67 | 44.71±2.26 | 32.99±2.44 | 42.59±1.89 | 44.14±2.39 |
| CT (2025, adapted) | 59.87±1.05 | 51.45±2.20 | 34.48±1.24 | 45.69±2.83 | 50.43±2.26 |
| LCSS | **62.94±1.60** | **55.59±1.50\*** | **41.04±2.37\*** | **50.29±2.76\*** | **55.67±0.81\*** |
| Shuffled | 53.70±2.06 | 43.63±1.62 | 31.70±2.69 | 39.38±1.28 | 44.03±1.75 |

**表 A.2 CiteSeer（引文，6 类）**

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
| --- | --- | --- | --- | --- | --- |
| MLP+CE | 55.10±1.17 | 43.50±3.02 | 33.35±1.36 | 45.15±1.67 | 44.33±2.04 |
| Co-teaching (2018) | 57.05±0.82 | 46.24±2.89 | 34.19±2.92 | 46.25±2.29 | 46.81±1.68 |
| Small-loss (2018) | 59.26±1.94 | 49.70±3.07 | 37.73±2.86 | 48.16±2.56 | 50.75±3.33 |
| SCE (2019) | 52.12±3.28 | 42.09±3.47 | 30.59±2.83 | 42.22±1.56 | 40.92±4.23 |
| ELR (2020) | 53.62±1.48 | 43.17±3.95 | 30.80±2.70 | 42.69±2.43 | 41.94±3.27 |
| JoCoR (2020) | 59.36±1.43 | 49.91±3.37 | 39.07±2.47 | 48.63±2.35 | 50.86±2.45 |
| DivideMix-lite (2020, adapted) | 55.81±1.54 | 43.93±2.91 | 31.99±1.94 | 45.27±1.84 | 44.19±2.86 |
| FINE (2021, adapted) | 55.86±1.06 | 43.43±3.43 | 32.23±2.69 | 44.24±2.20 | 44.88±2.40 |
| LogitClip (2023, adapted) | 59.17±1.97 | 47.29±3.26 | 33.05±2.13 | 46.94±2.68 | 47.53±2.72 |
| ANL (2023, adapted) | 53.69±2.43 | 41.67±3.20 | 31.53±2.70 | 42.65±1.59 | 42.21±4.40 |
| CT (2025, adapted) | 59.61±1.45 | 49.45±2.57 | 37.29±2.85 | 48.81±2.50 | 49.91±2.50 |
| LCSS | **63.38±1.20\*** | **53.83±2.75\*** | **40.17±2.75** | **50.09±2.40\*** | **54.62±2.25\*** |
| Shuffled | 54.53±1.83 | 42.35±2.74 | 31.28±1.31 | 45.04±1.73 | 42.81±2.34 |

**表 A.3 PubMed（引文，3 类；边界数据集，如实报告）**

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
| --- | --- | --- | --- | --- | --- |
| MLP+CE | 74.63±1.49 | 63.77±4.40 | 45.50±2.45 | 59.48±1.38 | 62.69±2.78 |
| Co-teaching (2018) | 81.12±0.55 | 75.77±0.99 | 54.50±6.19 | 70.41±1.57 | 74.89±0.87 |
| Small-loss (2018) | 81.88±0.85 | 77.30±1.29 | 54.72±5.02 | 73.19±1.70 | 76.47±1.25 |
| SCE (2019) | 79.45±0.62 | 68.37±1.27 | 45.49±3.02 | 61.08±2.08 | 67.75±2.45 |
| ELR (2020) | 76.71±0.53 | 64.03±2.35 | 45.74±2.13 | 59.96±1.85 | 62.80±1.74 |
| JoCoR (2020) | 81.70±0.71 | 77.84±1.10 | **58.12±5.83** | **73.93±1.47** | **76.89±0.80** |
| DivideMix-lite (2020, adapted) | **83.25±0.72** | **78.28±0.54** | 48.57±4.14 | 73.42±2.25 | 76.69±1.51 |
| FINE (2021, adapted) | 74.78±1.13 | 61.55±5.80 | 44.03±3.65 | 52.53±3.22 | 60.05±2.43 |
| LogitClip (2023, adapted) | 79.25±0.66 | 65.69±1.31 | 47.08±2.15 | 64.54±2.38 | 67.08±3.43 |
| ANL (2023, adapted) | 79.69±0.74 | 69.31±2.43 | 46.69±3.00 | 61.17±0.88 | 68.73±1.80 |
| CT (2025, adapted) | 81.52±0.53 | 77.23±0.33 | 53.32±4.35 | 70.97±2.34 | 76.25±1.22 |
| LCSS | 82.44±0.42 | 71.90±0.88 | 47.53±2.42 | 68.49±1.27 | 72.01±1.23 |
| Shuffled | 75.31±0.97 | 62.72±4.49 | 44.27±2.72 | 58.10±1.19 | 61.05±3.37 |

**表 A.4 Coauthor-CS（合著，15 类）**

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
| --- | --- | --- | --- | --- | --- |
| MLP+CE | 91.38±0.54 | 88.42±0.62 | 80.03±1.91 | 75.91±1.34 | 87.44±1.15 |
| Co-teaching (2018) | 91.70±0.29 | 88.80±0.97 | 81.88±0.72 | 79.06±1.30 | 88.95±0.97 |
| Small-loss (2018) | 91.50±0.49 | 89.06±0.88 | 81.65±0.91 | 80.13±1.50 | 88.97±1.02 |
| SCE (2019) | 90.63±0.47 | 88.37±0.53 | 79.43±1.45 | 79.77±2.51 | 87.99±0.56 |
| ELR (2020) | 91.21±0.60 | 88.29±0.73 | 80.54±1.28 | 78.03±1.48 | 87.78±0.94 |
| JoCoR (2020) | **91.81±0.25** | 89.38±0.61 | 82.58±0.41 | 81.02±1.62 | 89.17±0.85 |
| DivideMix-lite (2020, adapted) | 91.14±0.57 | 88.41±0.62 | 80.03±1.91 | 77.38±1.82 | 87.42±1.28 |
| FINE (2021, adapted) | 91.27±0.67 | 87.79±0.78 | 79.94±1.82 | 75.12±1.19 | 87.35±1.16 |
| LogitClip (2023, adapted) | 91.81±0.21 | 88.94±0.24 | 80.48±1.16 | 81.03±0.57 | 89.20±0.76 |
| ANL (2023, adapted) | 91.02±0.49 | 88.91±0.57 | 79.53±1.33 | 79.55±2.56 | 88.33±0.70 |
| CT (2025, adapted) | 91.05±0.50 | 88.37±0.72 | 82.45±0.57 | 77.71±2.02 | 88.34±0.98 |
| LCSS | 91.56±0.25 | **90.62±0.57\*** | **87.17±0.62\*** | **85.96±1.41\*** | **90.50±0.49\*** |
| Shuffled | 90.40±0.69 | 87.60±1.03 | 78.83±1.93 | 76.05±0.54 | 86.89±1.53 |

**表 A.5 Amazon-Photo（共购，8 类）**

| Method | Uniform 0.2 | Uniform 0.4 | Uniform 0.6 | Pair 0.4 | Instance 0.4 |
| --- | --- | --- | --- | --- | --- |
| MLP+CE | 82.34±1.39 | 73.90±1.80 | 58.60±3.77 | 62.61±4.31 | 69.13±3.00 |
| Co-teaching (2018) | 85.21±0.51 | 78.21±1.00 | 63.29±5.00 | 67.24±6.97 | 73.59±1.19 |
| Small-loss (2018) | 84.96±0.58 | 77.25±1.62 | 61.40±3.72 | 66.70±5.73 | 73.00±1.25 |
| SCE (2019) | 84.74±0.94 | 76.08±3.11 | 57.82±4.07 | 66.51±6.96 | 73.08±2.54 |
| ELR (2020) | 82.31±1.98 | 73.93±2.12 | 59.32±4.53 | 66.07±5.25 | 69.41±2.70 |
| JoCoR (2020) | 85.60±1.26 | 79.57±0.53 | 66.48±4.43 | 66.94±6.36 | 74.80±1.79 |
| DivideMix-lite (2020, adapted) | 80.30±1.51 | 72.53±3.69 | 56.62±4.50 | 66.01±7.84 | 70.82±1.52 |
| FINE (2021, adapted) | 82.27±1.07 | 73.02±2.69 | 58.01±5.25 | 61.54±4.53 | 66.38±2.68 |
| LogitClip (2023, adapted) | 83.74±0.67 | 76.29±2.00 | 61.58±4.64 | 68.88±3.40 | 73.06±1.66 |
| ANL (2023, adapted) | 85.36±0.47 | 76.94±2.80 | 59.03±4.96 | 64.90±6.57 | 73.18±2.36 |
| CT (2025, adapted) | 84.40±1.12 | 77.70±1.16 | 60.81±3.08 | 66.86±6.94 | 73.57±2.03 |
| LCSS | **86.20±1.46** | **81.64±0.93** | **70.78±3.19** | **72.51±4.02** | **79.59±1.16** |
| Shuffled | 81.99±1.03 | 72.06±2.97 | 59.65±5.46 | 62.95±4.94 | 68.98±2.44 |

汇总：LCSS 在 Cora/CiteSeer/Amazon-Photo 的全部 15 列最优，Coauthor-CS 4/5 列最优（uniform 0.2 与 JoCoR/LogitClip 差 0.25pp，小于标准差），PubMed 输样本选择族但胜全部损失族——与正文 5.1 与 5.3 节的容量分析一致。

## 附录 B：配对显著性检验

完整表（339 行）见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_B_paired_significance.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_B_paired_significance.tex`。LCSS 与每个 baseline 在每个数据集 × 噪声设置下的逐种子配对差、paired t-test 与 Wilcoxon signed-rank p 值。此处给出汇总计数：

| 数据集 | LCSS vs 各 baseline：t-test p<0.05 的比较数 | vs shuffled：p<0.05 |
|---|---:|---:|
| Cora | 70 / 78 | 5 / 5 |
| CiteSeer | 72 / 78 | 5 / 5 |
| PubMed | 64 / 78 | 5 / 5 |
| Coauthor-CS | 61 / 70 | 5 / 5 |
| Amazon-Photo | 15 / 15 | — |

注：n=5 配对种子；Wilcoxon 在 n=5 下 p 值下限为 0.0625，因此显著性标记以 paired t-test 为准，Wilcoxon 仅作方向一致性佐证，不用于否定稳定性。PubMed 上 p<0.05 的比较包含 LCSS 落后于选择族的方向（检验是双向如实的）。Amazon-Photo 行来自 MLP-12 批次的配对文件（不含 shuffled 配对行，shuffled 对照见附录 A 表 A.5——5/5 设置均值差 +4.2~+11.1pp、逐种子 5/5 正向）。

## 附录 C：graph-dependent 参考方法

**本表方法推理需要图——部署设定与本文不同，数字为同协议参考，不做配对比较。** 5-seed 聚合（uniform 0.4，percent 20/10/70，train-only 噪声）：

| 方法 | 推理需图 | Cora | CiteSeer |
|---|---|---:|---:|
| GCN+CE | 是 | 78.42±1.51 | 65.30±1.21 |
| GCN[X‖H]+CE | 是 | 80.27±0.69 | 67.61±1.81 |
| NRGNN | 是 | 81.07±2.30 | 70.27±1.08 |
| NRGNN[X‖H] | 是 | 81.39±1.70 | 70.90±1.87 |
| SoftDist-GCN[X‖H]（本文共识目标的 GCN 版） | 是 | 81.37±0.67 | 68.91±1.59 |
| LCSS（本文，X-only 推理） | **否** | 55.59±1.50 | 53.83±2.75 |
| LCSS-distill（本文，X-only 推理） | **否** | 80.60±0.69 | 69.40±1.08 |

两点如实说明：(i) 图原生方法在推理可用图时精度更高——这正是本文范围声明的内容；(ii) LCSS-distill 在不使用图推理的前提下达到 NRGNN 级精度（Cora 80.60 vs 81.07；CiteSeer 69.40 vs 70.27），差距 <1pp。TFR 的迁移配置结果与逐种子明细见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_C_graph_dependent_references.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_C_graph_dependent_references.tex`；来源哈希已记录，协议元数据整理在投稿前完成。

## 附录 D：构造参数敏感性与 β 选择审计

**D.1 构造参数敏感性（27 配置网格，uniform 0.4）。** 完整网格见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D1_parameter_sensitivity.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D1_parameter_sensitivity.tex`。三个结论：(i) 合理范围（max_balls∈{32,64,128} × min_ball_size∈{16,32,64} × split_improvement∈{0.01,0.05}）内，相对默认配置的平均差异全部落在 ±0.9pp 内——鲁棒平台；(ii) split_improvement=0.1 的全部 9 个配置在 4/5 数据集塌缩为单组（平均 −7.5pp）——清晰的失效边界；(iii) Cora/CiteSeer 上 max_balls=32 与 128 给出逐种子完全相同的结果（构造自停止于 22–28 组，上限从未触发）——"固定规则 + 自停止构造"的直接证据。5-seed 复核显示 min_ball_size=64 的配置有小幅增益（+0.4~+1.8pp，对应正文 5.3 节的票数机制），但默认配置未更换：候选筛选使用了 seed0 测试集信号，更换将破坏"构造参数从未按数据集调整"的实验纪律，且未经全噪声谱验证。

**D.2 β 选择审计。** 完整表见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D2_beta_audit.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_D2_beta_audit.tex`。若按 per-setting 验证集精度选择 β∈{0,0.5,1}，6 个试点设置中 5 个会选 β=1（与 β=0.5 的验证差距 0.06~2.32pp），仅 PubMed uniform 0.2 选 β=0.5；而 β=1 在该设置为负增益。此审计支撑正文 4 节的措辞：β=0.5 是"初始探索后冻结的保守默认——唯一在全部试点设置不劣于纯软目标的取值"，而非"验证集选出的最优值"；其有效性由冻结后在 Coauthor-CS、Amazon-Photo 及 pair/instance 噪声上的 out-of-sample 确认支撑。

## 附录 E：测试特征噪声完整曲线

图 E.1（PDF：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_E_feature_noise_curves.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_E_feature_noise_curves.png`）：测试特征加高斯扰动 X̃=(1−α)X+αε，α∈{0,0.1,0.3,0.5,0.7,0.9}，5 方法 × 5 数据集，5-seed 均值±std 误差带；α=0 逐种子锚定主表（校验零失败）。代表性读数（α=0.5）：Cora 上 LCSS 45.07 vs Small-loss 41.61、JoCoR 40.84、CE 36.01；Amazon-Photo 上 LCSS 63.82 vs JoCoR 62.84。结论与正文 5.3 节一致：相对优势在多数数据集保持至高噪声，但 LCSS 的绝对退化在小图更陡（共识目标加深特征依赖），且本方法不是特征噪声防御机制。

## 附录 F：Cut Value 结构一致性表

完整表见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_F_cut_value.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_F_cut_value.tex`（6 方法臂 × 5 数据集，uniform 0.4，5-seed）。**必要脚注（如实声明）**：CE 与 shuffled 的高 CV 值部分来自预测坍缩——其逐种子 CV 标准差达 0.15~0.28（如 Cora CE 的逐种子 CV 为 {0.97, 0.99, 0.43, 0.86, 0.99}，高值种子对应预测坍向少数类；全同预测的 CV 趋近 1.0），而蒸馏臂仅 0.003~0.01。因此 **CV 在标签噪声协议下不是干净的结构注入度量**（该指标的既有用法均在干净标签设定），本文不以 train-only 各臂的 CV 差异作机制结论；唯一引用的是蒸馏变体与 train-only 的组内对比——LCSS-distill 的 CV 高于 train-only +0.246/+0.246/+0.139（Cora/CiteSeer/PubMed，全 5/5 种子，蒸馏臂无坍缩污染）——支撑正文 6 节"teacher 蒸馏携带更强拓扑对齐偏置"，与 NTK 理论刻画一致。

## 附录 G：t-SNE 三着色可视化

图 G.1（PDF：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G_tsne_three_coloring.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G_tsne_three_coloring.png`）：Cora 的 GRACE 表征 H 经 t-SNE 降维（固定随机种子），同一坐标三份着色——(a) 真实类别：表征空间呈清晰的类簇结构；(b) 含噪标签（uniform 0.4，seed 0）：噪声标签在各簇内呈散点分布、多为局部少数派；(c) 局部组归属：递归二分产生的分组与类簇结构大体对齐。该图直观展示"噪声标签在其局部组内是少数派、可被共识投票稀释"的机制前提。**插图性质**——分组质量的定量证据是附录 A 的 shuffled 对照与正文 5.3 节的纯度/票数分析，不依赖此可视化。

图 G.2（PDF：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G2_grace_ball_cover.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_G2_grace_ball_cover.png`）：参照粒球聚类文献中常用的"中心-半径覆盖"画法，在五个主数据集的 frozen GRACE H 上重新构造默认粒球，并将 H 线性投影到二维 PCA 平面后以圆形轮廓可视化。每一行对应一个数据集：左列展示最终粒球在投影空间中的局部轮廓，右列放大一个代表性球并标出球心、80% 投影距离分位半径和球内 train noisy label 投票。该图只用于解释粒球在本文中扮演的几何角色：粒球构造本身只读取 H，不读取标签；黑色圆是二维可视化轮廓，不改变算法中的原始最终粒球；标签颜色仅用于说明干净类别几何和后续 soft target 的投票信号。对大图只采样散点用于显示，粒球构造仍使用全体节点。

## 附录 H：低标签率完整曲线

图 H.1（PDF：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_H_labelrate_curves.pdf`；PNG：`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\figures\appendix_H_labelrate_curves.png`）：Cora/CiteSeer，训练标签比例 6 点（5%~65%），LCSS 与 Small-loss/JoCoR/FINE/shuffled 的 5-seed 曲线及误差带。**单调性核验**（`C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\appendix_H_labelrate_monotonicity.md`）：LCSS 相对最强选择族的优势随标签率降低而扩大的趋势，在 Cora 上严格单调、在 CiteSeer 上存在一个回折点（非严格单调）——故正文 5.3 节使用"总体扩大"而非"单调扩大"。结构必要性在全部 12 个数据集 × 标签率点上保持（vs shuffled 全 5/5 种子，+6.6~+13.6pp）。

## 附录 I：正文表格的完整版

**表 I.1（正文表 B 的完整版）**：graph-free 家族对照全表见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\table_B_graphfree_family.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\table_B_graphfree_family.tex`——注意 LCSS-distill（球式 teacher，80.60/69.40/85.01）与 kNN-consensus distill（80.99/69.57/85.23）为独立两行，两者差异小于标准差（正文 5.3 节形式可平替结论的 teacher 侧再现）。**表 I.2（扩展节五臂全表）**：Markdown 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\extension_denoised_teacher_distill.md`；LaTeX 版见 `C:\Users\Admin\Desktop\实验8去噪5.19-\external\NoisyGL\paper_assets_20260706\tables\extension_denoised_teacher_distill.tex`——五个 teacher 臂（T1 GLNN / T2 [X‖H]-CE / T3 球式共识 / T4 kNN 共识 / T5 shuffled 共识）× uniform 0.4/0.6 × 学生与 teacher 精度，附 D1（H 输入增量）与 D2（共识去噪增量）分解行；正文 6 节的三层归因数字均出自此表。一处投稿前待核：表 I.1 资产中 GCN[X‖H]+CE 参照行的 Coauthor-CS 与 Amazon-Photo 两格来源待核对原始 run（正文未引用这两格）。

---

# 数据来源对照表（审阅用，投稿版删除）

| 正文位置 | 来源 |
|---|---|
| 主表数字 | mlp10/mlp12 aggregate（5-seed） |
| shuffled 25/25 | MLP-5/10/12 paired deltas |
| 表 B | mlp14_report + MLP-9 重组 |
| kNN/kmeans 平替、票数机制 | MLP-8 P0 / MLP-11 / MLP-13 |
| 三空间对照 | MLP-15 P3 + X-ball（2026-06-18）|
| 容量分析 / PubMed 重定性 | linear_ablation + MLP-15 P0c/P4 |
| GRACE-H 上界 / [X‖H] 崩溃 | grace_h_ablation_20260704 |
| 低标签率曲线 | MLP-6 Part B |
| clean 列 / β 审计 / 显著性 | MLP-15 P2/P0a/P0b |
| 特征噪声 / cut value | MLP-16 P1/P0 |
| 蒸馏扩展 + 三层归因 | MLP-9 5-seed |
| why-GRACE 四组实验 | Phase 9/20 + P2a/P2b + MLP-15 |
| inductive 塌缩 | MLP-15 P1 |
