# Related Work 切割与写作素材笔记（2026-07-02）

依据：本地精读四篇竞品原文（`文献文件/粒球噪声学习/` 下 `Ding10241192.pdf`=DND-NET、`GLNN.pdf`、`RNCGLN.pdf`、`simlp.pdf`），对照论文骨架 v2（`paper_skeleton_graphfree_gb_v2_20260702.md`）。
用途：写作时直接取用的切割句、分类表、引用锚点与 limitation 素材。**本文件不引入任何新实验、不改任何 claim；所有表述受骨架 v2 §0 "不 claim" 红线约束。**

---

## 0. 一句话净判断

四篇均未做"冻结 label-free 对比表征 + 表征空间几何局部共识软标签 + X-only MLP"的组合。结构进入 MLP 的通道四分法下，**监督侧（把结构转成抗噪软标签目标）无人占据**；处理标签噪声的两篇（DND-NET、RNCGLN）均用模型置信度伪标签（自训练族，有反馈回路），无一篇用"几何共识、目标构造不依赖模型预测、非迭代单遍"。核心配方成立，但下述三处框架级重叠必须主动切割。

---

## 1. 四篇机制速查表（写作时的事实底稿）

| 论文 | 发表处 | 核心机制 | 做标签噪声？ | 推理 graph-free？ | 结构进入 MLP 的通道 |
|---|---|---|---|---|---|
| DND-NET | KDD 2024 | PPR 传播特征（L≈10 步）喂 MLP + NPA 邻居一致性 KL 损失 + 邻居熵加权可靠伪标签 | ✅ 主打 | ❌ 三个组件全要邻接 | 输入侧 |
| GLNN | ICLR 2022 | GNN teacher 全节点 softmax 蒸给 MLP student；loss = λ·CE(真标签) + (1−λ)·KL(teacher) | ❌（其"噪声"消融是**特征**高斯噪声，非标签噪声） | ✅ | 预测侧 |
| RNCGLN | AAAI 2024 | MLP 编码 + 多跳图对比损失（局部）+ 全节点高效自注意力（全局）+ 伪图/伪标签双自提升，端到端 | ✅（置信度换标签，自训练） | ⚠️ 半个：不需要邻接，但推理对全体节点做注意力，非逐节点独立前向 | 表征侧（对比+注意力） |
| SimMLP | WSDM 2025 | 自监督对齐 MLP 编码器与 GNN 编码器嵌入（不变性+重建+两防塌策略），推理只用 MLP 编码器+线性头 | ❌（下游头普通 CE；卖点是特征/边噪声与 label scarcity） | ✅ | 表征侧（编码器对齐） |

协议关键差异（防止任何跨协议数字比较）：
- DND-NET：Cora-ML（≠Cora）/CiteSeer/PubMed/Coauthor-CS，**5%/类 train + 20% val**，20 次重复；uniform 20/40/60 + pair 20/30/40。
- GLNN：CPF 五数据集 + OGB Arxiv/Products，标准稀疏划分，干净标签；transductive + inductive（20-80 hold-out）。
- RNCGLN：Cora/CiteSeer/PubMed/Photo，{graph noise, label noise} 组合 {0.1,0.3}/{0.3,0.1}；其 PubMed 可跑是其自报，我们宿主早期跑它 PubMed OOM（工程观察，旧协议 seeds 3000-3004，不可作对比数字引用）。
- SimMLP：20 数据集，节点/链接/图三任务，cold-start 设定；干净标签。
- 我们：Cora/CiteSeer/PubMed，percent 20/10/70，train-only 噪声，5 seeds。**以上任何绝对精度都不可与我们主表同格放置（MLP-7 铁律）。**

---

## 2. Related Work 结构建议：四通道分类表（取代逐段切割）

建议 §3 增加一张组织表，"结构信息进入 MLP 的通道"四分法，一张表完成全部切割：

| 通道 | 代表方法 | 是否处理标签噪声 |
|---|---|---|
| 输入侧（结构预先灌进特征） | DND-NET（PPR 传播特征）、NOSMOG/GENN（位置嵌入）、GA-MLP/SGC/SIGN | DND-NET ✅（但推理需图） |
| 表征侧（结构塑造编码器） | Graph-MLP、SimMLP、GraphECL、RNCGLN | RNCGLN ✅（但端到端耦合噪声 CE） |
| 预测侧（结构经 teacher 预测传递） | GLNN、VQGraph、CPF | ❌ |
| **监督侧（结构转成抗噪软标签目标）** | **本文（唯一）** | **✅** |

表后接一句定位句（可直接用）：
> "Existing routes inject structure into MLPs through the input (propagated features), the representation (encoder alignment), or the prediction (teacher logits); none constructs the *supervision itself* from structure. We take this fourth route: frozen contrastive representations define local consensus neighborhoods whose noisy-label votes form the training targets — making the supervision, not the model, the noise-robust component."

出处备注：此分类表灵感来自 SimMLP Table 1 的互信息目标对照表（其原表是 MI objective 分类，我们的是通道分类，不重复）。

---

## 3. 三处必须主动切割的框架级重叠

### 3.1 RNCGLN —— 最高危，双鲁棒头牌撞脸

重叠：它的 headline"同时处理 graph noise + label noise"与我们旧 dual-robust 措辞几乎同框；且同为"对比学习注入结构 + MLP 系 + 标签噪声"。

处理（骨架 v2 已把结构轴降为弱 profile 卖点 5，写作执行三点）：
1. 凡提结构不敏感，引 RNCGLN 并声明：我们**不 claim 双噪声头牌**；结构不敏感只是 X-only 推理的部署属性（0.00pp 扰动），不是与 RNCGLN 竞争的贡献轴。
2. 三个真实切割点（均有据）：
   - **推理形态**：RNCGLN 推理对全体节点做高效注意力（自报 O(nd²) 计算 / O(d²+dn) 内存），非逐节点独立 X-only 前向；孤立新节点/逐节点在线服务场景与我们不同。
   - **表征-噪声耦合**：它的表征学习与含噪 CE 端到端联合（L = Llsi + αLgsi 一起反传）。我们的冻结 label-free H 是有本管线证据的刻意反选——P2a 直接证明 noisy CE（哪怕 λ=0.1）进入 stage-1 会使球几何塌缩（52→1 球，−12.6pp）。**表述纪律：只说"在我们的管线中耦合破坏局部可聚类几何，因此选择冻结"，不评价 RNCGLN 自身设计好坏。**
   - **抗噪机制族**：它是置信度伪标签自训练（模型预测反馈进标签，有自强化回路，靠 warm-up 缓解）；我们是表征几何共识，目标构造不含模型预测、非迭代单遍。可关联引用我们自己的 dynamic-alpha/loop 判负作为"为什么反选"的机制证据。
3. 可选（导师沟通包选项，非建议）：RNCGLN 已注册在 NoisyGL 宿主，若导师要一行"最新 MLP 系抗噪"context 数字，可在我们协议下重跑（仅 Cora/CiteSeer，PubMed 会 OOM）；**按 MLP-7 铁律绝不进同 backbone 主表**——它是完整框架非弱 MLP 策略。

### 3.2 GLNN —— §7 扩展节的直系前身，写成"继承+升级"

重叠：我们 T1 臂就是 GLNN 式蒸馏；扩展节整体站在它的范式上。

处理：
1. **三个必须披露的实现差异**：
   - GLNN 原版 loss 含 λ·CE(真标签) 项——标签噪声下该项把噪声直接灌给 student。我们 MLP-8/9 的实现是纯 teacher-target 蒸馏（去掉噪声 CE 项），**这个适配对 GLNN baseline 更有利**（帮它挡噪声）→ 标注 "GLNN-style (adapted: label-CE term removed under label noise)"，如同 "MLP-adapted FINE"。这反而强化胜它的可信度。
   - GLNN 全文假设干净标签（其消融的"噪声"是特征高斯噪声）。
   - **扩展节 first claim 收窄（GSDN 精读后，2026-07-03）**：不能写"首次系统研究标签噪声下 GNN-to-MLP 蒸馏"——GSDN 附录 E 有噪声下与 GLNN 等蒸馏系的对比曲线。**改写为："首次在 train-label noise 下对比 denoised-teacher vs bare-teacher 蒸馏，并给出 H-input / 共识去噪 / 分区形式的三层归因分解"**——这是 MLP-9 的实际独有贡献，GSDN 未做（它没有 teacher 目标去噪的概念，也没有归因分解）。u0.6 数据支撑不变：G1 +7.38pp 3/3 全 5/5；G2 +1.39pp 3/3、14/15 seeds。
2. 三句诚实话照骨架 v2 §7 不动：主力是 teacher 的 [X‖H] 输入；去噪增益 u0.4 边际、u0.6 真实；球式与 kNN 式共识 teacher 相当（T4 在 u0.4 还是数值最佳臂）。

### 3.2b GSDN —— 设定最近邻（2026-07-03 全文精读后新增；本节以原文为准，取代 Codex 转述）

原文事实（arXiv 2210.02097v5，"Under review" 未见正式接收记录；已全文核实）：
- **机制**：纯 MLP backbone 上的邻域自蒸馏——feature-level（目标节点预测与 mixup 可学习插值增强的邻居预测一致 + 均匀负采样推远非邻居）+ label-level（train 节点标签同时监督其邻居的预测头 gγ）。结构只在训练期隐式用作 prior，推理无图依赖。**通道归位：预测侧（自蒸馏族，与 GLNN 同区），非监督侧**——四通道分类表不需要新增通道，我们的"监督侧唯一"定位不变。
- **噪声实验（对 Codex 转述的两点修正）**：确有 label noise 0-0.6 曲线（正文 Fig.4a/b vs GNN 系 + 附录 E Fig.A3 vs 蒸馏/MLP 系，Cora/CiteSeer）。但：(a) 噪声是 **asymmetric (i+1)%C 单一类型**，无 uniform/instance 覆盖；(b) **对照全是 GNN 与蒸馏/MLP 系，没有任何 LNL 策略**（无 small-loss/co-teaching 等）；(c) 协议为 20/类 planetoid 式 split（label rate 1.5-5.2%）+ 未明确声明噪声 train-only，与我们 percent20/train-only 不可直比。定位是其鲁棒性附带评测，不是 LNL 研究。
- **对我们的三点影响**：① 裸设定 first 死（骨架 §0/P2 已改）；② 扩展节 first 收窄（见上 3.2 第 1 条）；③ 正面可用：其 Fig.A3 显示噪声升高时 MLP 系相对 GNN 优势扩大——与我们"弱下游/高噪声下共识更值钱"的规律独立一致，Analysis 可引作佐证。
- **切割句（可直接用）**：*"GSDN also evaluates an MLP-based graph-free model under label noise, but noise there serves as an auxiliary robustness check for its neighborhood self-distillation — a single asymmetric-flip noise type, compared against GNN and distillation baselines only. We instead treat train-label noise as the central problem: five noise settings, a fixed X-only backbone, and systematic comparison against eleven adapted LNL strategies, with the supervision itself (rather than prediction consistency) as the noise-robust component."*
- **要不要跑 GSDN 数字**：不进主表（自蒸馏用了全体无标签节点 = 蒸馏资源轴，MLP-7 铁律）；扩展节可选臂，与 SOP/DSS/RNCGLN-context 并列进导师沟通包。写作引用时如实标注 arXiv/未正式接收。

### 3.3 低标签率卖点三家共用

重叠：DND-NET 卖 1-5%/类低标签曲线全胜；SimMLP 卖 label scarcity（到 0.001）下显著优势。

处理：卖点 3 scope 成"**标签噪声下**、X-only 同 backbone 公平对照内的低标签率优势"，引两篇并注明协议不可直比：DND-NET 是 5%/类 + 推理需图；SimMLP 是干净标签 SSL 泛化。我们的差异变量是"噪声×标签稀缺"交互，他们各只有单变量。

---

## 4. 可直接采纳的写作改进点（零新实验）

1. **§3 四通道分类表**（见 §2）——一张表完成四篇+外围方法的切割。
2. **§6 Analysis 引 GLNN §5.8 的信息论上界**：I(G[i];yi) = I(E[i];yi) + I(X[i];yi|E[i])，MLP 只能最大化后一项 → 为 MLP-6 Part B"加标签追不上 GCN（43.7→64.0 仍差 ~14pp）"提供文献支撑：X-only 天花板是特征信息量，不是标签量。
3. **§4 复杂度节加同类参照**：RNCGLN 自报 O(nd²)/O(d²+dn)；GLNN 自报比 GNN 快 146-273×。我们 student 单节点推理 O(K(D+C))、与 n 和边数无关——并排后 graph-free 轻量卖点有竞品数字锚。
4. **t-SNE 插图**：维持骨架 v2 计划（DND-NET Fig 7/10 同款用法，插图不作证据）。
5. **Limitation 节素材（引 GLNN Fig 5 左）**：GLNN 自己展示特征噪声下 MLP/GLNN 比 GNN 掉得快——X-only 方法天然依赖特征质量。我们主动写一句"本文不处理特征噪声，X-only 推理继承该依赖"，堵住可预见提问。
6. **可选零成本机制量化**：GLNN 的 cut loss（Lcut = Tr(ŶᵀAŶ)/Tr(ŶᵀDŶ)，预测-拓扑一致性）可在**已有** run 的预测输出上直接计算，量化共识软目标给 student 注入的结构量（GLNN 报 SAGE 0.92 / MLP 0.76 / GLNN 0.90 可作参照）。Analysis 节想要机制图时用，可做可不做。
7. **SimMLP 问题的预置答案**（审稿人若问"stage-1 为何不用 SimMLP 式对齐"）：可分性≠可聚性四次独立证据（Phase 9/20 + P2a/P2b）——球构建需要 H 的 2-means 可聚类几何而非 probe 精度；入场规则（球纯度诊断）是现成廉价检验。**用现有证据回答，不为它开新实验。**

---

## 5. 红线自查（写作时逐条回看）

- [ ] 任何 DND-NET/GLNN/RNCGLN/SimMLP 的绝对精度都没有与我们主表同格出现（协议不可直比，最多独立 positioning 表并标注）。
- [ ] 没有写"双噪声鲁棒"headline（RNCGLN 占位；我们结构轴=弱 profile）。
- [ ] GLNN baseline 标注了 "adapted"（去噪声 CE 项）。
- [ ] 低标签率卖点带"标签噪声下 + X-only 公平"限定语并引 DND-NET/SimMLP。
- [ ] 对竞品设计只做定位切割，不做价值贬低；对我们反选的理由只引本管线证据。
- [ ] 未越过骨架 v2 §0 "不 claim" 红线（球不可平替 / 全数据集 SOTA / 同 H-access 赢经典 / u0.4 去噪 teacher 强增益）。

---

## 6. GNN-to-MLP 蒸馏家族 13 篇全文核验（2026-07-06，取代 Codex 转述）

覆盖：GLNN(ICLR22)/GSDN(arXiv22 未发表)/NOSMOG/SA-MLP(arXiv22)/FF-G2M(AAAI23)/KRD(ICML23)/PGKD(COLING25)/VQGraph(ICLR24)/AdaGMLP(KDD24)/SimMLP(WSDM25)/SALE-MLP(IJCAI25)/Demystifying(AAAI26)/ProGMLP(AAAI26)。全部本地 PDF 全文提取核验。

### 6.1 决定性事实：13 篇无一处理 train label noise

- "鲁棒性"在该家族 = **特征高斯噪声**，且是同一个协议 `X̃=(1−α)X+αε`（NOSMOG §5.4、PGKD §6.2、VQGraph 附录 C.3）；KRD 的"可靠性"= 特征扰动下 teacher 熵不变性（ρ_i，X′~N(X,Σ(δ))），也是特征轴；AdaGMLP 管标签稀缺(1-3%) + 测试特征缺失(10-50%)，标签干净；其余（SA-MLP/FF-G2M/SALE-MLP/Demystifying/ProGMLP/SimMLP）无任何噪声主题。
- **gap 句最终版（已 13 篇全文背书）**：*"Existing GNN-to-MLP methods show that graph-induced knowledge can be transferred to MLPs for graph-free inference; however, their robustness studies focus on feature noise, missing features, or label scarcity — training-label noise is absent from this literature line. We study graph-free MLP learning where corrupted training labels are the central problem."*
- 监督信号来源：13 篇全部依赖 **GNN teacher 或图结构损失**产生监督；没有一篇从表征空间局部聚合**构造**监督——监督侧通道确认无人占据（第三次独立确认）。

### 6.2 机制最近邻两篇的精确切割

- **PGKD（机制最近邻）**：prototype = **有标签依赖的全局类均值中心**（按标签分组求均值，每类一个；transductive 用 teacher 预测标签防泄漏）。我们 = **无标签正确性依赖的局部球式共识**。两条切割轴：局部 vs 全局；标签有噪时它的类中心直接被污染（它假设干净）。它的 "noisy nodes" 指表征离群点非标签噪声。
- **KRD（概念最近邻，teacher 端可靠性）**：过滤"对特征扰动不稳定的知识点"，我们修正"被标签噪声污染的目标"——噪声源与作用层都不同。**有用的三分法**：KRD 自称上采样（增可靠监督）、批评 RKD-MLP 过滤软标签是下采样（减监督）；我们的共识软标签 = **修复**（repair），第三种范式——这句切割可直接用。

### 6.3 可引用的理论支撑（Demystifying, AAAI 2026——唯一理论文）

- **Proposition 1 + Lemma 1 + Eq.15**：NTK 视角证明 soft-label 蒸馏使 X-only 学生的 NTK 与邻接矩阵对齐（κ(A,Θ) 上升）、优先关注低 Dirichlet 能量（平滑/低频）特征、对未见样本的预测更新等价于隐式 label propagation。**引用方式**：支撑"共识软标签向 X-only 学生传递结构 bias"——但注明是类比性支撑（其软标签=GNN teacher logits，我们=GRACE 局部共识；假设含 mean-pooling/无限宽）。Lemma 1 的低通偏好恰与球式聚合的平滑本质呼应。
- 附带负面观察可引：纯 label-only KD 会强迫学生过度平滑、丢弃判别性非平滑特征（"Misaligned Feature Preference"）——"纯软标签蒸馏的局限"的文献锚。

### 6.4 引用信息勘误（写作时用对版本）

- NOSMOG：本地 PDF 是 NeurIPS 2022 GLFrontiers Workshop 版措辞——引用时用 ICLR 2023 正式版；其推理需 DeepWalk 位置嵌入（inductive 还要边传播位置特征），**非严格 X-only**——且 AdaGMLP 原文点名批评它"依赖测试节点结构信息"（切割有文献背书）。
- SA-MLP：推理需邻接行输入，**非 X-only**（Codex 表格未说清）；arXiv 未见正式发表，谨慎引用。
- PGKD：COLING 2025（Codex 写 arXiv 2023，勘误）。
- GSDN：arXiv Under review 未正式接收（已知）。
- 家族里推理严格 X-only 的：GLNN/GSDN/FF-G2M/VQGraph/AdaGMLP/SALE-MLP/ProGMLP/DA-MLP；非严格：NOSMOG（位置嵌入）/SA-MLP（邻接行）。

### 6.5 借鉴资产（零/低成本）

1. **Cut Value 结构一致性指标**（4 篇标配：GLNN cut loss、NOSMOG、VQGraph Table 3、SALE-MLP 式8-9）：`CV=tr(ŶᵀAŶ)/tr(ŶᵀDŶ)`，在**已有预测输出**上直接算，量化共识软目标给 X-only 学生注入的结构量。零训练成本，建议做（§6 机制量化）。
2. NOSMOG/GLNN 的信息论分解 I(G;y)=I(E;y)+I(X;y|E)——X-only 适用边界的论证框架（已在笔记 §4.2）。
3. SA-MLP 的"记忆 vs 泛化"批判（GLNN 加宽只在 transductive 提升）——可用于区分"记住脏 teacher 输出 vs 学到鲁棒共识"。
4. FF-G2M 的低/高频知识度量（邻居余弦 + 成对距离 KL）——分析共识目标保留哪类邻域信息。
5. 特征噪声曲线 `(1−α)X+αε`（家族标准协议）——**可选防御臂**：审稿人从这个圈来会问"你们对特征噪声如何"；现有答案是 limitation 一句（X-only 继承特征依赖 + GLNN Fig5 引用）；若导师要数字，75 run 半天，进可选臂清单。
6. 扩展节防御句（若被问"为何不比 KRD/VQGraph 蒸馏"）：我们扩展节的 claim 是 teacher **目标去噪**的效应，与蒸馏**技术**改进（可靠性采样/code/渐进）正交且可组合——引用即可，不跑。
