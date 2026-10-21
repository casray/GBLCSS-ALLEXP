# 论文骨架 v2：Graph-Free 局部共识软监督的标签噪声鲁棒节点分类

日期：2026-07-02（取代 `paper_skeleton_graphfree_gb_20260701.md` v1）
依据：MLP-1~7 全量 + MLP-8 P0/P1/P2 + MLP-9 5-seed 终判 + MLP-12 Amazon-Photo 扩展 + MLP-15 GCN-hidden 表征消融。目标：普通 SCI（英文）。
v2 相对 v1 的四处结构性变更：①"球必要性"改写为"局部共识必要性"，kNN/kmeans 平替诚实披露；②新增"closing the accuracy gap"蒸馏扩展节（MLP-9 5-seed 数据）；③主表扩到 5 数据集并补 co-purchase 领域（MLP-12）；④补充普通 noisy-label GCN hidden 替代 GRACE 的 why-GRACE 消融（MLP-15）。

---

## 0. 一句话贡献（诚实、5-seed 坐实）

> 在 **graph-free 节点分类**设定下（训练期借对比表征吸收结构、**推理只用 X-only 轻量 MLP、不碰邻接**），用**冻结 GRACE 表征上的局部共识软标签（粒球式聚合实现，β=0.5 半硬）**做标签噪声鲁棒训练；同一弱 MLP backbone 下在 citation/coauthor/co-purchase 数据上稳定优于经典与最新 MLP-adapted 抗噪策略、依赖真实局部结构（Amazon-Photo 额外 5/5 设置 > shuffled）、低标签率下增益最大；并给出一条可选蒸馏扩展，使 X-only student 达到并超过 GCN 级绝对精度。**据我们所知，首个以 train-label noise 为中心问题、并把表征空间局部共识软监督（监督侧通道）带入 graph-free 节点分类的工作**——graph-free 系已有工作（GSDN）将标签噪声作为附带鲁棒性评测，未与任何 LNL 策略对比、未把噪声作为中心问题。

**不 claim**：设定首创（"graph-free 推理 × 标签噪声评测"已被 GSDN 附带覆盖——first 只落在"噪声为中心问题 + 监督侧机制"上）/ 粒球形式不可平替（kNN/kmeans 相当，消融如实给出）/ 全数据集 SOTA（PubMed 边界）/ 同等 H-access 下赢经典策略（MLP-6 reframe）/ 高噪声之外的去噪 teacher 强增益（u0.4 下 G2 边际）。

---

## 1. 标题 & 卖点排序（按证据强度）

- 候选标题：*Local-Consensus Soft Supervision on Frozen Contrastive Representations for Label-Noise-Robust Graph-Free Node Classification*（"granular-ball style" 可在方法名/摘要保留，最终措辞待导师定夺——见 §9 sign-off）
- 卖点排序：
  1. **结构必要性**（最硬）：β0.5 > shuffled 在主表数据集上稳定成立；Amazon-Photo 新增 5/5 设置全正，幅度 +4.21~+11.14pp。
  2. **X-only 设定内全胜抗噪策略**（Cora/CiteSeer + Amazon-Photo，含 FINE/DivideMix-lite/CT adapted；PubMed/CoauthorCS 按实际表述为边界或补充）。
  3. **低标签率优势**（6 点 5-seed 曲线）。
  4. **精度扩展**（新）：局部共识 teacher 蒸馏，X-only student 超 GCN 绝对精度（+1.3~+4.5pp @u0.4；+5.8~+10.0pp @u0.6）。
  5. （弱 profile）graph-free 推理结构扰动 0.00pp；轻量。

---

## 2. Introduction —— 每段一句 claim

- **P1 问题**：图节点分类同时面临 (a) 标签噪声 (b) 部署需要 graph-free 推理（**效率/延迟、拓扑隐私——冷启动措辞已删**：MLP-15 P1 显示 observed-subgraph 重训 GRACE 的球塌缩为 1 球 purity 0.21-0.32，inductive 扩展未获支持，协议如实标注 transductive，inductive 留 future work + limitation 一句"训练期需要完整图上的高质量对比预训练"）。
- **P2 gap**：图 LNL 方法（NRGNN/TFR/**DND-NET**…）推理依赖邻接；graph-free 方法（Graph-MLP/**GLNN**/**GSDN**/NOSMOG）以效率/蒸馏为主线，标签噪声至多作为附带鲁棒性评测（GSDN 有 0-0.6 噪声曲线但对手全是 GNN、无 LNL 策略对照）、或只关注特征噪声（NOSMOG）。**gap 措辞 = "label noise 未被作为中心问题在 graph-free 推理设定下用 LNL 协议系统研究"，不写"交集空白/无人做过"。**
- **P3 做法**：GRACE 冻结表征（训练期吸收结构）+ 表征空间局部共识软标签（β=0.5 半硬，粒球式固定构造实现）+ X-only MLP。MLP-11 显示 kNN 的全局 `k=20` 在四数据集上已足够，故 K-free 只保留为"免扫 k / 固定配置"的便利性 profile，不作为性能差异化主 claim。MLP-15 进一步显示，普通 noisy-label GCN hidden 的局部结构不是随机噪声（6/6 设置 > shuffled），但仅 1/6 设置以 >0.5pp 超过 frozen GRACE anchor，因此不能替代 label-free GRACE。
- **P4 贡献列点**：①首个 graph-free 设定下的表征空间局部共识标签噪声鲁棒方法；②同 backbone 稳定优于经典+最新策略（Cora/CiteSeer/Amazon-Photo）+ 真实结构必要性铁证；③低标签率优势；④精度扩展：共识 teacher 蒸馏达 GCN 级精度；⑤诚实适用域与归因刻画（含平替消融）。
- ⚠️ **scope 一句**（Intro 末）："本文主表聚焦 train-target 抗噪策略的同 backbone 对比；利用无标签节点的蒸馏是另一资源轴，我们在扩展节（§7）诚实对照并展示如何结合。"——这句同时挡 GLNN 攻击和 [X‖H] 攻击。

---

## 3. Related Work —— 四块 + 切割句

| 块 | 代表 | 切割句 |
|---|---|---|
| 通用 LNL | small-loss/co-teaching/JoCoR/ELR/SCE/FINE/DivideMix/CT/LogitClip/ANL | 图无关；我们在图节点 + graph-free 设定，结构化局部共识软监督 |
| 图 LNL | NRGNN/TFR/ProCon/**DND-NET(KDD24)**/RNCGLN | 推理依赖邻接。**DND-NET 重点切割**：其 MLP 吃 PPR 传播特征、NPA 损失与邻居不确定性均需图——是解耦 GNN 非 graph-free；协议不可直接比较（Cora-ML≠Cora、5%/类≠percent20，标注）；其低标签率卖点须引用并区分（它 1-5%/类 + 图内推理，我们 percent 曲线 + X-only） |
| **GNN-to-MLP / graph-free 节点分类（13 篇全文核验，cut-notes §6）** | Graph-MLP/**GLNN**/**GSDN**/NOSMOG/SA-MLP/FF-G2M/**KRD**/**PGKD**/VQGraph/AdaGMLP/SimMLP/SALE-MLP/**Demystifying(AAAI26)**/ProGMLP | **gap 句（13 篇背书）**：*"their robustness studies focus on feature noise, missing features, or label scarcity — training-label noise is absent from this literature line"*（NOSMOG/PGKD/VQGraph 的"噪声鲁棒"全是同一协议的特征高斯噪声；KRD 的"可靠性"=特征扰动熵不变性，干净标签；AdaGMLP=标签稀缺+测试特征缺失）。**机制最近邻两个切割**：PGKD 原型=有标签依赖的全局类均值（标签有噪即污染）vs 我们=免标签正确性的局部球共识；KRD 三分法进 related work 结尾——上采样可靠(KRD)/下采样不可靠(RKD-MLP)/**修复被污染(我们，第三范式)**。**X-only 勘误**：NOSMOG 推理需 DeepWalk 位置嵌入、SA-MLP 需邻接行——均非严格 X-only（AdaGMLP 原文点名批评 NOSMOG，切割有文献背书）。GSDN 重点切割照旧（附录 E 噪声曲线单一 asymmetric 类型、无 LNL 对照、arXiv 未发表）；GLNN 蒸馏在扩展节实测对照。**理论引用**：Demystifying Prop.1+Lemma1（NTK-邻接对齐 + 低通偏好 + 隐式标签传播）作"共识软标签向 X-only 学生传结构 bias"的类比性理论支撑（注明其软标签=GNN logits、含无限宽假设） |
| 粒球计算 | MgCNL/MgBIF/CGBC/MOGB | 球多为 end-to-end 重塑表征；我们=冻结表征上离线聚合；**诚实句：我们发现该角色下球式聚合与 kNN/kmeans 共识相当（消融），贡献在设定与配方组合，非新球算法** |

---

## 4. Method —— 组件 + 证据锚

1. **冻结 GRACE 表征**（why-GRACE：alignment-uniformity 平衡使 H 可被 2-means 递归分裂；P2a/P2b 与 MLP-15 共同说明 supervised/noisy-label 表征可能带来局部结构，但不稳定，不能系统替代 frozen GRACE——写进 §6 分析）。
2. **局部共识分组**：行 L2 归一化 H 上递归 2-means 建球（全项目固定 `max_balls=64, split_improvement=0.05`）。与 kNN/kmeans 的差异化不再写成性能不可替代；MLP-11 显示 kNN 的 best-k 在四数据集均为 20，clean-val 选 k 几乎无损，因此球式实现只作为固定构造、免扫描的轻量实现 profile。
3. **β=0.5 半硬软标签**：`target = 0.5·in-ball 分布 + 0.5·onehot(多数票)`（MLP-4 定型，唯一 6/6 稳 ≥ β=0）。
4. **X-only MLP student**（2 层，推理 graph-free）。
5. **（扩展，§7）共识 teacher 蒸馏**：同款软目标训 GCN[X‖H] teacher → 全节点 softmax → 蒸给同一 X-only MLP。
6. **红线**：clean 只评测；噪声只加 train；目标构造不见 clean。
7. **复杂度分析**（组内过审标配，必写）：球构建 O(n log n · d)（递归 2-means）；目标构造 O(n·C)；student 推理 O(n·K·(D+C))、与图规模/边数无关；对照 NRGNN 465k 参数/迭代 vs 本方法 92k/非迭代（MLP-2 效率表）。

---

## 5. Experiments —— 表图清单

| # | 表/图 | 内容 | 来源 run | 撑的 claim |
|---|---|---|---|---|
| Setup | — | Cora/CiteSeer/PubMed/CoauthorCS/Amazon-Photo；uniform0.2/0.4/0.6+pair0.4+instance0.4；percent 20/10/70；同弱 MLP；全 baseline 标 "MLP-adapted" | — | 公平性 |
| T1 | **主表 A**（表名："Comparison with MLP-adapted label-noise learning strategies under graph-free X-only inference"） | β0.5 vs CE/small-loss/co-teaching/JoCoR/ELR/SCE/FINE/DivideMix-lite/CT/LogitClip/ANL + shuffled，5 数据集 5-seed | MLP-5+7a+10+12 | 卖点2+1 |
| **T1b** | **★表 B：graph-free family 定位表** | vanilla MLP / Graph-MLP(adapted) / GSDN(adapted) / Ours β0.5 (train-only) / GLNN-style 蒸馏 / Ours-distill(共识 teacher) / GCN(X)、GCN[X‖H] 参照行 × 5 噪声设定 × 5-seed；**带资源标注列**（训练用图？推理用图？用无标签节点？）。NOSMOG/VQGraph/TGS 维持引用切割 | **MLP-14 + MLP-9 重组** | 回答 graph-free 圈审稿人"我不是只挑弱 MLP baseline"；结果按 profile 写：Graph-MLP 在 Cora/CiteSeer/低噪声强，Ours train-only 在高噪声均值、pair/instance、Coauthor/Amazon 与资源约束下更稳；Ours-distill 在已测三数据集显著强于两者 |
| T2 | **消融** | β{0,0.5,1}；w/o-GRACE(X-ball 塌缩)；ellipsoid/gmm（被 β0.5 吸收）；shuffled；**kNN-in-H / kmeans-H 平替行（P0）+ kNN@k=20 行（MLP-11，u0.4 略优于球，如实）**；**GCN-hidden 替代 GRACE 构组（MLP-15：6/6 > shuffled，但仅 1/6 > GRACE by >0.5pp）** | MLP-4/5a/2 + MLP-8 P0 + **MLP-11/15** | 半硬+GRACE 必要；平替诚实披露 |
| F1 | **低标签率曲线** | 6 点 × 5 方法，5-seed 误差棒 | MLP-6 Part B | 卖点3 |
| T3 | **H-access 对照（诚实）** | β0.5 vs CE/small-loss/JoCoR 的 [X‖H] 版 | MLP-6 Part A | scope 说明 |
| **T4** | **★精度扩展表** | 五臂 student（GLNN/[X‖H]-CE/共识 teacher/kNN teacher/shuffled teacher）× u0.4/u0.6 × 5-seed + teacher acc | **MLP-9** | 卖点4 |
| **T5** | **★归因分解表** | D1(H-input) / D2(共识去噪) / (球−kNN) 三层配对增量 | **MLP-9 decomposition** | 诚实归因（分析贡献） |
| T6 | 结构/效率 profile | X-only 扰动 0.00pp vs GNN 掉 5~23pp；参数/墙钟 | MLP-2 C2 | 弱 profile |
| **T7** | **★附录表 C：graph-dependent 参考表（新，旧数据复用）** | NRGNN / NRGNN[X‖H] / TFR 同协议 5-seed 数字；表头标注 "inference requires the graph — different deployment setting"；Codex 需核对旧行 split/noise hash 与 MLP 线一致 | Phase 24/25/36 | 回应"为什么不比图原生方法"；不做 paired 断言 |
| **T8** | **★Cut Value 结构一致性表（新，零训练后处理）** | CV=tr(ŶᵀAŶ)/tr(ŶᵀDŶ)，家族标准指标（GLNN/NOSMOG/VQGraph/SALE-MLP 同款）：{Ours, CE, shuffled, kNN@20, Ours-distill, GLNN-style} × 5 数据集 × u0.4 × 5-seed | **MLP-16 P0** | §6 机制量化：共识软目标注入的结构量；与 Demystifying NTK 理论互印 |
| **F3** | **★特征噪声曲线（新，家族标准协议防御）** | X̃=(1−α)X+αε，α∈{0,0.1,...,0.9}，{Ours, CE, small-loss, JoCoR, shuffled} × 5 数据集 × u0.4 × 5-seed；α=0 锚定主表 | **MLP-16 P1** | 防御 graph-free 圈标准提问；按预注册分支进鲁棒性小节或 limitation |
| F2 | **★t-SNE 图** | GRACE H 三着色：真类/噪声标签/球归属 | 新绘（现成 H+assignment） | §6 插图（非证据） |

---

## 6. Analysis / 适用域（诚实=加分项）

- **甜区**：Cora/CiteSeer 型 citation + Amazon-Photo co-purchase；Amazon-Photo 是 CoauthorCS 型胜行（5/5 噪声设定数值最佳，vs shuffled +4.21~+11.14pp），说明方法不是只在 citation/coauthor 域成立。**PubMed 边界经 MLP-15 P4 改写为"容量依赖的边界"（重大更新）**：弱 MLP backbone 下选择族胜（主表 A 照实保留）；但把全部方法换成线性学生后，**选择族崩溃**（linear small-loss 69.60 vs 其 MLP 版 81.88 @u0.2；linear DivideMix-lite 66.32 vs 78.28 @u0.4），SoftDist-Linear 在 5/5 噪声设置全部超过最强 linear 选择族 +7.4~+14.3pp。机制解释：选择族依赖"干净样本先拟合"产生的 loss 分离信号，而该信号本身是容量（噪声记忆）的产物——线性学生不怎么记忆噪声，loss 分离消失，选择既无必要也无信号；共识监督不依赖该信号，容量无关。诚实 caveat 两条：①linear 选择族是我们的适配版，崩溃可能部分含适配欠力（FINE 式标注）；②u0.2 低噪声下 Linear+CE（81.42）仍略高于 SoftDist-Linear（80.70）——强线性可分+低噪声时裸 CE 仍是边界。原"角色×数据集"表述保留（蒸馏角色把 PubMed 拉到 85.0），现在是"角色×容量×数据集"三维边界。
- **★容量分析段（MLP-15 P0c，§6 新段）**：SoftDist-Linear 在 5/5 数据集同时胜 Linear+CE 与 linear-shuffled（全 5/5 seeds）——**共识软监督不是 MLP 容量伪影，线性学生同样成立**。且 linear 学生在 4/5 数据集上强于 MLP 学生（PubMed +8.1、AmazonPhoto +4.8、CiteSeer +1.1、CoauthorCS +0.4pp；仅 Cora MLP 略优 +1.5）——"学生容量是噪声记忆旋钮"是 §6 的新分析变量。**红线：主线学生保持 MLP（预注册纪律，MLP-13 同规——不许看 test 换主线）**；linear 行进容量消融表，论文可写一句 "practitioners may prefer a linear student where features are strongly linearly separable"。标尺规律在线性层复现：Linear+CE 越强的数据集球增量越小。
- **三空间共识消融（MLP-15 P3，关死"要 GRACE 干嘛"）**：X 空间（X-ball，塌缩单球）/ 原图空间（LP-target：噪声标签邻接传播 2 步进同一 β0.5 公式，被 H-consensus 压 +6.9~+8.3pp、3 数据集全 5/5）/ H 空间（主线）——**只有冻结对比表征空间的共识有效**；原图传播扩散噪声标签（与 GSDN 同病，两个独立证据互印）。
- **clean 0% 列（MLP-15 P2，混合结论进 limitation）**：Cora +0.44 / CiteSeer +2.33 / PubMed +0.02 / CoauthorCS −0.88 / AmazonPhoto −1.20pp vs CE——**不许写"完全无害"**；limitation 写"半硬局部共识在强特征可分且标签完全干净时可有 ~1pp 代价"。
- **β 选择审计（MLP-15 P0a，Setup 措辞定稿）**：per-setting val 会在 5/6 试点设置选 β=1（差距 ≤2.4pp），仅 PubMed u0.2 选 0.5——**不能写"β 由 validation 选出"**。诚实措辞："β=0.5 是初始探索后冻结的保守默认——唯一在全部试点设置不劣于纯软目标的取值（β=1 在 PubMed u0.2 为负）；冻结后在 CoauthorCS/AmazonPhoto/pair/instance 上 out-of-sample 确认"。审计表进附录。
- **显著性（P0b）**：主表加 paired t-test 标记；Wilcoxon 在 n=5 时下限 0.0625，如实注明 "5 paired seeds"，不作稳定性否定依据。
- **★Cut Value 机制段（MLP-16 P0 结果已定）**：家族标准指标没有支持 train-only Ours 的全局拓扑一致性通道——CV(Ours)>CV(CE) 且 >CV(shuffled) 只在 CiteSeer/CoauthorCS 局部成立，Cora/PubMed/AmazonPhoto 反预期。**根因（逐 seed 核验，2026-07-06 补）：CV 在标签噪声协议下不是干净的"结构注入"度量**——CE/shuffled 的高 CV 来自**预测坍缩**（Cora CE 逐 seed CV={0.97,0.99,0.43,0.86,0.99}，高 CV seed 即预测坍向少数类；全同预测 CV=1.0；CE/shuffled 的 CV std 高达 0.15-0.28，而蒸馏系仅 0.003-0.01），家族文献在干净标签下用 CV，"高=学到结构"的语义在噪声下反转。§6 必须收窄为："train-only 增益不主要由 global topology smoothness 解释；且 CV 在噪声协议下受坍缩污染，如实注明"——CV 表进附录不进正文。但 Ours-distill 在 Cora/CiteSeer/PubMed 全部 5/5 seeds 的 CV 高于 train-only（+0.2458/+0.2460/+0.1388，且蒸馏系无坍缩污染、数字干净），可支撑扩展节："teacher distillation carries stronger topology-aligned bias"——与 Demystifying NTK 理论互印的位置从 train-only 移到 distill 变体。
- **★特征噪声句（MLP-16 P1 结果已定）**：家族标准协议 X̃=(1−α)X+αε；结果符合 X-only 方法随特征质量退化的共同边界。Ours 在 Cora/CiteSeer/AmazonPhoto 多数 α 下相对优势保持（α=0.9 仍 −0.09/+0.29/+0.81pp vs 最强对手），但**绝对 drop 在小图偏陡**（Cora Ours 34.0pp vs CE 22.8pp——共识软目标让模型更依赖特征承载的结构信息，特征坏时掉得更多，这是机制的自然代价）；PubMed 仍输 small-loss/JoCoR，CoauthorCS 中等 α 被 JoCoR 反超（−1.26~−3.05pp）。写法：多数场景的标签噪声优势在测试特征噪声下仍可保持，但本方法不是 feature-noise 防御机制，且共识目标在部分数据集加剧特征依赖（引 GLNN Fig.5 作 X-only 共同边界）；PubMed/CoauthorCS 中等扰动 + 小图陡降如实进 limitation。曲线整条进附录图。
- **结构必要性 vs 形式可平替性**（核心新分析）：shuffled 全线大负（结构必要）；kNN/kmeans 相当（形式可平替）→ 价值在"冻结对比表征上的局部共识"这个配方。MLP-11 进一步削弱强 K-free 叙事：kNN 曲线在本协议下并不敏感，四数据集 best-k 均为 20，单一全局 k 无损。**MLP-13 补上机制统一解释：有效变量是"每个共识单元内的噪声票数"（约 20-24 票），不是分区形式**——把球构造的 min_ball_size 从 32 提到 64（每球票数 ~10→~24）后，u0.4 下球与 kNN best-k 的差距基本闭合（CiteSeer +2.58pp→+0.77pp、5/5 seeds 正；PubMed 打平），且大球纯度更低但精度更高（0.477 vs 0.505）——"20+ 票的可靠多数"比"10 票的高纯度"更值钱。MLP-11 观察到的 kNN 微弱优势主要是默认构造票数偏少的产物，非形式劣势。因此论文写"固定构造免扫 k"的便利性 + 票数机制段，不写"k 选择负担显著"。**主表默认参数维持 64/32/0.05 不换**（MLP-13 候选筛选用 seed0 test delta，换默认 = test 知情调参，毁掉"全项目从未调构造参数"的诚实资产；且仅 u0.4 验证）。
- **附录敏感性表（MLP-13）**：27 配置网格——合理范围内 ±1pp 平台、split_improvement=0.1 全线塌缩单球（−7.5pp）是清晰的失效边界；Cora/CiteSeer 上 max_balls 32/128 逐 seed 结果完全相同（自停止在 22-28 球、上限未触发）= "fixed rules with self-stopping construction" 的直接证据。
- **归因三层（T5）**：teacher 端 H 输入是主力（u0.4 +2.3~+2.7pp，u0.6 +3.9~+7.7pp）；共识去噪目标真实但边际、高噪声下更强（u0.6 G2 3/3、14/15 seeds）；球 vs kNN ≈ 0。主动给出=控制叙事。
- **why-GRACE / 可分性≠可聚性**：不再写成"所有有监督表征都会塌缩"。P2b loop / MLP 隐层重构确实出现球结构退化；P2a 放松球参数后能恢复球数但只追平 anchor；MLP-15 用普通 noisy-label GCN hidden 替代 GRACE 构组，real 在 Cora/CiteSeer × train ratio {5%,10%,20%} 的 6/6 设置中超过 shuffled（说明不是随机结构），但只有 1/6 设置以 >0.5pp 超过 frozen GRACE（CiteSeer 20%, +0.86pp），Cora 10/20 与 CiteSeer 5 均落后。结论：supervised GCN hidden 可作为有信号的对照，但不稳定，不能替代 label-free frozen GRACE；这才是 why-GRACE 的准确表述。
- **H-access scope**：MLP-6 结果如实（[X‖H]+选择反超）→ 贡献条件于 graph-free 推理部署。**GRACE-H 直接分类消融（2026-07-04）把这条 scope 声明补到最强形式，必须进 T3/§6**：frozen H + 线性 CE 在 u0.4 达 78.61/68.87/83.87（Cora/CiteSeer/PubMed），比 train-only SoftDist-MLP 高 12~23pp、0/5 seeds——**H 本身就是最强的"分类器输入"，任何 X-only train-target 方法都追不上它**。三层回应全部写进论文：①该路线推理期需要 GRACE 编码器+图（部署 profile 不同，且新节点需前向整个编码器）；②**Ours-distill 追平或略超它**（80.60 vs 78.61、69.40 vs 68.87、85.01 vs 84.05）——蒸馏扩展的第三个存在理由：X-only 部署拿到 H-carry 路线的精度；③[X‖H]+CE 裸拼接在小图反而塌（Cora 48.54、CiteSeer 51.66，比 train-only 还低）——H 的价值要么经监督侧转化、要么配鲁棒策略、要么整个换成 H-only 输入，裸拼进弱 MLP 会被噪声记忆吞掉（这本身是"为什么走监督侧通道"的新论据）。**红线：这组数字绝不能藏**——"直接用 H 分类"是任何审稿人都能一行代码试出来的对照；主动给出+蒸馏版应答=控制叙事，藏=灾难。同时主线卖点排序不变：主表 A 的"X-only train-target 策略对比"设定内全部结论不受影响（GRACE-H 分类器不属于该资源类——它推理用 H）。

---

## 7. Extension: Closing the Accuracy Gap（新节，MLP-9）

- **动机**（正面回答 GLNN 攻击）：train-only X-only MLP 绝对精度低（Cora u0.4 0.556）因为只用 20% 节点监督；蒸馏是 graph-free 圈利用无标签节点的标准配方。
- **结果（5-seed）**：共识 teacher student 达 Cora 80.60±0.69 / CiteSeer 69.40±1.08 / PubMed 85.01±0.93（u0.4），全部超过 GCN(X) teacher 自身精度（78.77/64.90/83.75）；u0.6 胜 GLNN +5.8/+9.0/+7.4pp（15/15 seeds）。**X-only 推理达到 GCN 级精度，精度短板闭环。**
- **三句诚实话（不可省）**：①主要提升来自 teacher 的 [X‖H] 输入（D1）；②共识去噪目标的额外增益在 u0.4 边际（+0.2~+0.6pp）、在 u0.6 真实（+1.1~+1.8pp，14/15 seeds）——高噪声下 GNN teacher 记忆噪声更多，共识目标价值上升，与噪声钟形规律一致；③球式与 kNN 式共识 teacher 相当（G3 0/5×3），T4 kNN-teacher 在 u0.4 数值上还是最佳臂——如实给出。
- **叙事定位**：这节把最尖的审稿攻击（"为什么不直接蒸馏"）变成论文自己的正面结果："蒸馏确实好，而且把我们的共识软目标放到 teacher 侧比裸 teacher 更好，尤其高噪声"。
- **★MLP-14 新增论据（强化"§7 进正文"）**：表 B 显示 Graph-MLP(adapted) 在 Cora/CiteSeer 上强于我们的 train-only（+3~11pp，10/10 格，靠全图边集对比正则）——这是必须诚实呈现的结果。回应有两层：①资源轴不同（Graph-MLP 用全体节点+全部边，train-only 只用 train 标签，不属主表 A 的 train-target 资源类）；②**资源对齐后 Ours-distill 反超所有 graph-free 竞品**：Cora u0.4 80.99 vs Graph-MLP 63.23（+17.76pp）、vs GSDN（+25pp 级），已测三数据集 6/6 稳定。→ §7 不只是回应 GLNN，它是 graph-augmented 资源级唯一压过 Graph-MLP 的方法，因此更应进正文。

---

## 8. 写作顺序

```
1. Method(§4 含复杂度) + Setup —— 最锁定，先落
2. T1 主表 + T2 消融(含平替行) + F1 曲线 —— 脊梁
3. §7 扩展节 + T4/T5 —— 精度故事
4. §6 Analysis（含 t-SNE 图、三层归因、角色×数据集边界）
5. Related Work(§3 含 DND-NET/GLNN 切割)
6. Intro(§2) 最后写 → Abstract/结论/限制
每节写完回查：claim 有 run 撑吗？越过 §0 "不 claim" 红线没有？
```

---

## 9. 待导师 sign-off 的四件事（写作可先行，这四处留占位）

1. **品牌措辞**：标题/方法名用 "granular-ball" 还是 "local-consensus (ball-style)"——P0+MLP-9 双平替 + MLP-11（kNN@20 在 u0.4 还略优）后必须摊牌，两版措辞都备好。
2. **扩展节位置**：§7 进正文还是附录（进正文=精度故事完整+表 B 有落点；附录=主线更纯）。
3. **投稿目标**：KBS / Neural Networks / Pattern Recognition。
4. **可选实验臂开关**：GSDN/Graph-MLP 已由 MLP-14 实测进入表 B；后续默认不再追加 graph-free family baseline。SOP（机制最近邻，FINE 陷阱风险）＞ RNCGLN-context 行（宿主已注册，仅 Cora/CiteSeer）＞ DSS ＞ AmazonCom 第 6 数据集均维持导师点头才开。
