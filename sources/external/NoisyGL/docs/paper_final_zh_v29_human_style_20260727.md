# 粒球局部共识软监督：带噪标签下的无图推理节点分类

**Granular-Ball Local-Consensus Soft Supervision for Noisy-Label Node Classification with Graph-Free Inference**

---

## Highlights

- GB-LCSS 面向训练标签噪声与无图推理并存的节点分类场景。
- 冻结图对比表征中的粒球将局部标签共识转化为软监督。
- 分类器推理仅使用节点属性，无需图结构或预训练表征。
- GB-LCSS 在多数噪声设置下优于同骨干抗噪方法。
- 共识教师蒸馏将仅属性学生提升至具有竞争力的 GCN 水平。

---

## 摘要

训练阶段可访问图结构而推理阶段只能使用节点属性，是隐私约束、关系缺失和低时延需求下常见的部署场景。当训练标签受到噪声污染时，现有图标签去噪方法仍依赖邻接信息，而现有无图推理方法通常以干净训练标签为前提，对噪声监督的鲁棒性尚缺乏系统研究。为此，本文提出粒球局部共识软监督（Granular-Ball Local-Consensus Soft Supervision，GB-LCSS）框架。GB-LCSS 在无监督图对比学习获得的冻结表征空间中构造粒球，通过球内标签共识生成软监督目标，进而训练仅以节点属性为输入的 MLP 分类器。在五个主结果数据集的完整噪声谱评测中，GB-LCSS 在绝大多数噪声设置下取得了优于同骨干抗噪方法的分类性能。作为扩展，共识教师蒸馏在保持仅特征推理的同时，将学生模型性能提升至具有竞争力的 GCN 水平。结果表明，GB-LCSS 为训练标签含噪且要求无图推理的节点分类任务提供了一种简单有效的监督构造方法。

**关键词**：标签噪声，节点分类，无图推理，图对比学习，粒球计算，知识蒸馏

---

## 1 引言

图神经网络（Graph Neural Network，GNN）通过消息传递聚合邻域信息，已成为节点分类的主流方法 [1,2]。然而，部署端并不总能获得完整的边关系，在线消息传递还会引入邻居检索与多跳传播开销 [3]。在现有无图推理研究中，训练阶段可以利用图结构，而推理阶段的学生模型只使用节点属性，不再访问邻接关系或执行消息传递 [3,14]。本文也沿用这一设置：测试时分类器只使用节点属性，不使用邻接矩阵、邻居节点、图编码器输出或其他图衍生表征。GLNN [3] 及其后续研究表明，通过知识蒸馏、结构编码或结构感知训练，多层感知机（Multilayer Perceptron，MLP）能够在该设置下获得接近 GNN 的分类性能 [5--7,29,30]。

当训练标签受到噪声污染时，这一部署设置面临新的困难。人工标注、自动采集和多源数据整合都可能引入错误节点标签，深度模型还会在训练过程中逐渐拟合这些错误监督 [8]。NRGNN、RTGNN、DND-NET 和 RNCGLN 等图标签噪声方法通过图传播、双网络互校或伪标签机制提高鲁棒性 [9--12]，但其去噪过程或最终推理仍依赖邻接矩阵。另一方面，已有 GNN-to-MLP 和 X-only 节点分类工作大多以干净训练标签为前提，研究重点集中于结构知识迁移、特征扰动、特征缺失或标签稀缺 [3,5--7,14,15]。GSDN [14] 考虑了非对称标签噪声情形，但尚未在统一标签噪声协议下与通用抗噪策略进行系统比较。因此，如何利用训练期图结构抵抗标签噪声，同时获得推理期仅依赖节点属性的分类器，尚未得到系统研究。

现有无图推理方法主要在输入、表征或预测层面向 MLP 传递结构知识 [3,5--7,14,15]，其监督信号通常来自原始训练标签，或来自由这些标签训练的教师。当训练标签含噪时，标签错误也可能通过结构感知训练或知识蒸馏传递给学生模型。这意味着，仅传递图结构知识还不够，还需要利用图结构构造更可靠的监督信号。基于这一思路，本文从无监督图对比表征的局部几何结构中提取标签共识，并据此生成软监督，以降低模型对单个含噪标签的依赖。

本文提出粒球局部共识软监督（Granular-Ball Local-Consensus Soft Supervision，GB-LCSS）框架。GB-LCSS 首先在冻结的无监督图对比表征空间中构造粒球 [34]，再由球内训练节点的含噪标签分布和多数类共同形成组级目标，并按照置信度将其与节点原始标签进行软混合。共识越强，组级目标在最终软标签中的权重越大。共识较弱时，则更多保留节点的原始标签。软目标在分类器训练前一次性生成，用于训练 X-only MLP。整个过程不依赖干净标签或模型预测反馈，训练后的分类器仅使用节点属性。

在 11 个数据集的统一评测和其中 5 个数据集的完整噪声谱实验中，GB-LCSS 在多数设置下优于采用相同 X-only MLP 骨干的通用抗噪方法。基础 GB-LCSS 只在训练节点上构造软目标，图结构信息也只能通过这些节点传递给 X-only MLP。为使更多节点获得监督，并向学生传递更丰富的结构信息，本文进一步引入共识教师蒸馏：先使用 GB-LCSS 目标训练图结构教师，再将教师对所有节点的逐节点预测蒸馏给 X-only 学生。该扩展仍只使用原有的含噪训练标签，不额外使用干净标签，学生在测试时也仍然只使用节点属性。在多个代表性数据集上，X-only 学生的精度接近或超过相同噪声设置下的 GCN 基线。

本文在转导节点分类协议下进行评测。主实验统一使用 X-only MLP 骨干和训练节点的含噪标签，推理依赖图结构的方法列为跨部署条件参照。

本文的主要贡献如下：

1. 面向训练标签含噪且推理端只能使用节点属性的节点分类问题，提出 GB-LCSS 框架。该框架利用训练期图结构构造监督信息，而分类器在推理时仍只使用节点属性。
2. 设计粒球划分与置信度软目标机制。粒球划分不使用标签，球内含噪训练标签用于形成组级目标。共识置信度决定组级目标与节点原始标签的混合比例，整个过程无需干净标签或模型预测反馈。
3. 在 11 个图数据集和多种噪声协议下验证 GB-LCSS 的有效性与适用范围，并通过随机分组对照分析局部几何结构的贡献。共识教师蒸馏进一步将覆盖全部节点的教师预测传递给学生，在保持 X-only 推理的同时提高分类精度。

---

## 2 相关工作

### 2.1 通用标签噪声学习

通用标签噪声学习主要从样本选择、鲁棒损失、标签修正和半监督训练等角度减弱错误监督。Co-teaching 由两个网络相互选择小损失样本，JoCoR 在共同样本选择的基础上约束双网络预测一致性 [17,18]。SCE 组合交叉熵与反向交叉熵，ELR 抑制模型对噪声标签的后期记忆，ANL 和 LogitClip 分别通过负学习目标与 logit 范数约束提高损失鲁棒性 [19--22]。DivideMix 将样本划分为有标签与无标签部分并进行半监督训练，FINE 根据类内特征主方向识别可靠样本，CT 则利用训练过程中的置信度轨迹估计标签可靠性 [23--25]。这些方法为本文提供了同骨干抗噪参照，但其原始设计面向独立样本，不利用训练阶段的图结构构造监督信息。

### 2.2 图上标签噪声学习

图上标签噪声学习需要同时处理标签稀疏性、节点依赖性以及错误监督沿图结构传播的问题。UnionNET 聚合结构邻近节点的标签以估计节点类别分布，并据此进行样本重加权和标签修正 [46]。NRGNN 通过边预测和伪标签挖掘扩充可靠监督，RTGNN 采用双网络、自强化标签修正和一致性约束治理噪声 [9,10]。DND-NET 将高阶图传播与简单分类器解耦，并结合邻居不确定性生成鲁棒伪标签 [11]。它采用简单分类器，但输入仍是图传播特征，因此不属于严格的 X-only 推理。RNCGLN 则联合图对比学习与标签自纠正，同时考虑标签和结构扰动 [12]。

近期方法进一步利用拓扑样本选择、结构重构和多视图一致性。TSS 将图拓扑引入小损失样本选择，以保留类边界附近的有效节点 [47]。TFR 通过拓扑特征重构抑制对噪声标签的过拟合，并据此筛选伪标签 [40]。ProCon 结合特征相似性和邻接关系确定语义 peer，再利用 peer 标签一致性识别噪声并更新类原型 [41]。GD² 依据节点自身视图与结构视图的预测差异检测噪声 [42]。GNN-CFGD 利用训练记忆和混合模型划分可靠节点，再通过图连接与置信度细分补充监督 [44]。

蒸馏方法 BO-NNC 以多个自监督图模型作为教师，通过双层优化学习教师权重并改进标签 [58]。SAMTD 则根据结构和预测一致性对多个教师进行加权 [45]。另一类方法直接提高拓扑或聚合过程的鲁棒性。GTA 通过增补可信边和删除不可信边调整图拓扑 [59]，MILAB 以信息瓶颈约束邻居聚合，并通过对抗扰动下的预测一致性估计节点可靠性 [50]。HALF 面向异质图，根据局部同质性融合低通与高通损失以识别含噪节点 [51]。这些方法与本文处理相同的图节点标签噪声任务，但其噪声识别、监督传播或最终推理通常仍依赖图结构。

### 2.3 GNN-to-MLP 蒸馏与无图推理节点分类

无图推理研究允许在训练阶段利用图结构，同时避免在部署阶段执行消息传递。早期工作 Graph-MLP 通过邻域对比约束训练 MLP，GSDN 则在预测层进行邻域自蒸馏 [4,14]。SimMLP 以自监督方式对齐 MLP 表征和图上下文表征，ES-MLP 进一步通过边拆分机制适配同质图与异质图 [15,33]。NOSMOG 还引入位置与结构特征以增强 MLP [5]。这些方法减少了推理阶段的图计算，但资源边界并不完全相同。NOSMOG 依赖预先生成的 DeepWalk 位置嵌入。SA-MLP 的主要结构分支以邻接矩阵行为输入，其在线变体则另行估计潜在结构 [27]。因此，NOSMOG 与 SA-MLP 的主要设定不能直接视为与本文严格 X-only 设置具有相同的推理资源条件。

另一条路线是将 GNN 的知识蒸馏给 MLP。GLNN 将 GNN 教师的预测传递给仅使用节点属性的学生，建立了基本的 GNN-to-MLP 范式 [3]。后续方法扩展了蒸馏内容和训练方式。FF-G2M 分解低频与高频知识，VQGraph 通过结构 tokenizer 和离散 codebook 表达局部结构，PGKD 传递类原型，TINED 联合教师参数注入与 Dirichlet energy 蒸馏 [6,26,28,43]。KRD 根据特征扰动下的熵稳定性选择可靠的教师知识点 [7]。AdaGMLP、ProGMLP 和 SALE-MLP 分别研究标签稀缺下的自适应蒸馏、精度与推理代价调节，以及结构感知潜在表征 [29--31]。Yu 等 [32] 从神经正切核角度分析了软标签蒸馏向 MLP 传递低频图偏置的过程，其理论对象是 GNN logits，本文将其作为结构知识迁移的类比性解释。

上述工作证明了训练期图知识可以迁移至轻量学生，研究重点主要集中在蒸馏效率、结构知识迁移、特征扰动、特征缺失和标签稀缺。Graph-MLP、GSDN、SimMLP 与 ES-MLP 均以训练标签为既定监督信号。GSDN 虽包含非对称标签噪声实验，其训练目标仍侧重邻域自蒸馏。KRD 根据特征扰动下的稳定性筛选教师知识，PGKD 则由训练标签或教师预测形成类别原型。GB-LCSS 关注监督信号本身的构造：先在冻结的无监督图表征上完成标签无关的粒球划分，再聚合球内含噪训练标签形成软监督。基础版本直接训练 X-only 学生，蒸馏扩展则以 GB-LCSS 目标训练教师，并向学生传递全节点预测。

### 2.4 粒球计算与噪声学习

粒球计算以较少的超球体信息粒替代细粒度样本，并通过多粒度表示兼顾计算效率与噪声鲁棒性 [34]。在分类任务中，GBkNN 在粒球集合上依据查询点到球的距离执行近邻分类，粒球采样则利用球的纯度和边界结构选择代表性样本 [34,37]。VPGB 从粗粒度视角联合完成含噪数据的属性约简与分类 [52]。MgCNL 根据粒球标签置信度筛选可靠样本，MgBIF 聚合球内标签形成伪标签后再进行样本选择 [35,36]。GEAF 进一步采用不依赖标签纯度的自适应粒椭球划分，在局部粒内识别并修正错误标签 [53]。这些方法主要将粒结构用于分类、样本过滤、标签修正或可靠训练集构造。

粒球也开始用于图学习。监督式粒球图粗化和 GBGC 都将粒球作为超节点压缩训练图，前者依据标签纯度分裂，后者依据图的多粒度结构自适应形成粗化结果 [54,55]。MGGR 则利用粒球递归分解图内子区域，并通过层次编码器完成图级分类 [56]。这些方法说明粒球能够承载图结构，但其目标主要是图粗化、训练加速或图级表示，最终模型仍依赖图结构。

粒球与深度模型的结合进一步拓展了其抗噪角色。GBRAIN 在文本分类中动态构造粒球，并以球中心和球级标签形成粗粒度训练表示，从而减弱错误标签对模型的影响 [57]。G-GBC 以 CNN 提取特征，利用高斯混合模型和 Mahalanobis 距离自适应生成粒球，并通过球纯度与权重参与抗噪训练 [48]。BNO-GB 采用无监督、标签无关的粒球聚类，并以球级和样本级指标区分干净样本、分布内噪声和分布外噪声，随后优化开放意图表征 [49]。GB-LCSS 与上述工作的区别在于，它在冻结的无监督图表征上完成不读取标签的粒球划分，再利用球内含噪训练标签构造软监督。粒球不作为最终分类器或图粗化单元，也不与下游模型联合更新。训练完成后，分类器只接收节点属性。

---

## 3 方法

### 3.1 问题设定与符号

给定传导式属性图 \(G=(\mathcal V,\mathcal E,X)\)，其中 \(\mathcal V=\{v_1,\ldots,v_N\}\) 为节点集合，\(\mathcal E\) 为边集合，\(A\in\{0,1\}^{N\times N}\) 为邻接矩阵，\(X=[x_1,\ldots,x_N]^\top\in\mathbb R^{N\times d}\) 为节点属性矩阵。节点属于 \(C\) 个类别。训练节点集合记为 \(\mathcal V_L\)，其可观测标签 \(\tilde y_v\in\{1,\ldots,C\}\) 可能受到未知噪声污染。GB-LCSS 不要求预先知道噪声类型、噪声率或标签转移矩阵。

训练阶段允许使用完整的 \(X\) 与 \(A\) 学习结构表征并构造监督目标。训练结束后仅保留分类器 \(f_\theta\)。对于待预测节点 \(v\)，模型输出

\[
p_\theta(\cdot\mid x_v)=\operatorname{softmax}\!\left(f_\theta(x_v)\right),
\tag{1}
\]

推理时不访问 \(A\)、邻居节点、冻结表征或图编码器。本文研究的是同一传导式图中的仅属性推理，不涉及未见新节点的归纳式泛化。

本文使用的主要符号如下。

| 符号 | 含义 |
|---|---|
| \(H,\bar H\) | GRACE 编码器输出及其行 L2 归一化结果 |
| \(\mathcal B=\{B_j\}_{j=1}^{K}\) | 覆盖全部节点的粒球划分 |
| \(T_j=B_j\cap\mathcal V_L\) | 粒球 \(B_j\) 中的训练节点 |
| \(p_j,m_j,r_j\) | 球内含噪标签分布、多数类及多数类票占比 |
| \(q_j\) | 球级半硬共识目标 |
| \(\alpha_j\) | 球级共识权重 |
| \(t_v\) | 训练节点 \(v\) 的最终软目标 |
| \(\beta\) | 标签分布与多数类 one-hot 的混合系数 |
| \(s_{\min}\) | 构造球内共识所需的最少训练节点数 |

### 3.2 GB-LCSS 总体框架

GB-LCSS 将训练期图结构转化为仅属性分类器可使用的软监督。其基础流程包含三个阶段。首先，使用不读取节点标签的 GRACE 编码器在完整图上学习节点表征。其次，在冻结表征空间中递归构造粒球，并根据每个球内训练节点的含噪标签形成局部共识目标。最后，仅使用节点属性 \(X\) 和这些软目标训练 MLP。图结构和冻结表征只参与训练期监督构造，推理阶段不再需要。图 1 展示了完整的信息流。

![图 1 GB-LCSS 从训练期图结构到仅特征推理的整体流程](figures/fig1_lcss_pipeline_20260716.png)

### 3.3 冻结表征空间中的标签无关粒球构造

#### 3.3.1 冻结图对比表征

本文采用 GRACE [16] 生成粒球空间。GRACE 通过边丢弃和特征掩码得到两个随机图视图，并优化节点级对比目标，使同一节点在两个视图中的表示接近。预训练过程只读取 \(X\) 与 \(A\)，不使用节点标签。训练收敛后取编码器输出

\[
H=g_\psi(X,A)\in\mathbb R^{N\times d_h},
\tag{2}
\]

冻结参数 \(\psi\)，并按节点顺序保存 \(H\)。构球前对每个节点表征进行 L2 归一化

\[
\bar h_v=\frac{h_v}{\max(\lVert h_v\rVert_2,\varepsilon)}.
\tag{3}
\]

因此，后续粒球的成员关系只由冻结表征的几何结构决定。标签仅在粒球生成完成后用于构造局部共识。

#### 3.3.2 递归二分

粒球集合初始化为 \(\mathcal B=\{\mathcal V\}\)。对于任意粒球 \(B\subseteq\mathcal V\)，定义其中心和平均中心距离为

\[
\mu_B=\frac{1}{|B|}\sum_{v\in B}\bar h_v,
\qquad
D(B)=\frac{1}{|B|}\sum_{v\in B}\lVert \bar h_v-\mu_B\rVert_2.
\tag{4}
\]

在每轮划分中，算法收集所有满足 \(|B|\ge 2n_{\min}\) 的候选球，并按 \(D(B)\) 从大到小依次尝试二均值划分。本文实现采用欧氏距离、k-means++ 初始化和 `n_init=10`。设候选球 \(B\) 被划分为 \(B^{(1)}\) 与 \(B^{(2)}\)，划分后的加权平均中心距离为

\[
D_{\mathrm{split}}(B)=
\frac{|B^{(1)}|D(B^{(1)})+|B^{(2)}|D(B^{(2)})}{|B|}.
\tag{5}
\]

当两个子球均不少于 \(n_{\min}\) 个节点，且

\[
D_{\mathrm{split}}(B)\le (1-\delta)D(B)
\tag{6}
\]

时，接受该轮遇到的第一个有效划分，随后重新计算全部候选球及其顺序。若一轮中所有候选划分均不满足式 (6)，或粒球数量达到 \(B_{\max}\)，递归过程停止。正式实验统一使用 \(B_{\max}=64\)、\(n_{\min}=32\) 和 \(\delta=0.05\)。二均值随机种子由下游随机种子、当前分裂轮次和候选位置共同确定，完整规则随代码与配置公开。

### 3.4 粒球局部共识软监督

#### 3.4.1 球级半硬目标

对于粒球 \(B_j\)，记其中训练节点为 \(T_j=B_j\cap\mathcal V_L\)。当 \(|T_j|\ge s_{\min}\) 时，统计其含噪标签分布

\[
p_{jc}=
\frac{1}{|T_j|}
\sum_{v\in T_j}\mathbb I(\tilde y_v=c),
\qquad c=1,\ldots,C.
\tag{7}
\]

令

\[
m_j=\arg\max_c p_{jc},
\qquad
r_j=\max_c p_{jc},
\tag{8}
\]

其中 \(m_j\) 是多数类，\(r_j\) 是多数类票占比。类别并列时按实现中的 `argmax` 规则取索引最小的类别。仅使用分布 \(p_j\) 容易得到过于平滑的监督，只使用多数类又会丢失球内不确定性。GB-LCSS 将二者混合为半硬球级目标

\[
q_j=(1-\beta)p_j+\beta e_{m_j},
\tag{9}
\]

其中 \(e_c\) 表示类别 \(c\) 的 one-hot 向量，基础方法固定 \(\beta=0.5\)。

#### 3.4.2 置信度自适应的节点目标

不同粒球的多数共识强度并不相同。为扣除多分类问题中的随机一致基线，定义

\[
\alpha_j=
\max\left(
0,
\frac{r_j-1/C}{1-1/C}
\right).
\tag{10}
\]

对于 \(v\in T_j\)，最终训练目标为

\[
t_v=(1-\alpha_j)e_{\tilde y_v}+\alpha_j q_j.
\tag{11}
\]

当球内投票完全均匀时，\(\alpha_j=0\)，目标退回原始含噪 one-hot。当前实现取 \(s_{\min}=2\)。若 \(|T_j|<2\)，该球中的训练节点同样保留原始含噪 one-hot。默认方法允许节点自身的含噪标签参与所属球的统计。第 5.3 节使用 leave-one-out 变体单独分析自票影响，该变体不改变基础算法的定义。

粒球成员关系的生成不使用任何标签。式 (7) 至式 (11) 只读取训练节点的观测标签，不读取验证标签、测试标签或模型预测。干净标签仅用于实验后的诊断与最终评测，不参与训练目标的计算。

#### 3.4.3 仅属性学生训练

学生 \(f_\theta\) 为两层 MLP，其计算顺序为线性层、ReLU、BatchNorm、dropout 和输出线性层。隐层宽度为 \(F=64\)，dropout 为 0.5。学生概率由式 (1) 给出，并在训练节点上最小化软交叉熵

\[
\mathcal L_{\mathrm{LCSS}}
=
-\frac{1}{|\mathcal V_L|}
\sum_{v\in\mathcal V_L}
\sum_{c=1}^{C}
t_{vc}\log p_\theta(c\mid x_v).
\tag{12}
\]

模型选择与优化器设置见第 4 节。训练完成后只保留 \(f_\theta\)，预测过程不再执行图消息传递、GRACE 编码或粒球检索。

**算法 1** GB-LCSS：粒球局部共识软标签生成与 X-only MLP 训练

```text
输入: 图 G=(V,E,X)，含噪训练标签 {ỹ_v | v∈V_L}，类别数 C
      构造参数 B_max, n_min, δ, s_min，目标系数 β
输出: 仅属性分类器 f_θ，训练节点软目标 {t_v | v∈V_L}

 1: 使用 GRACE 在 (X,A) 上进行标签无关预训练，得到 H
 2: 对 H 逐行 L2 归一化，初始化粒球集合 𝓑 ← {V}
 3: while |𝓑| < B_max do
 4:     取所有满足 |B| ≥ 2n_min 的候选球，并按 D(B) 降序排列
 5:     accepted ← false
 6:     for B in candidates do
 7:         使用 2-means 将 B 划分为 B^(1), B^(2)
 8:         if min(|B^(1)|, |B^(2)|) ≥ n_min
 9:            and D_split(B) ≤ (1−δ)D(B) then
10:             以 B^(1), B^(2) 替换 B
11:             accepted ← true
12:             break
13:         end if
14:     end for
15:     if accepted = false then break
16: end while
17: 对全部 v∈V_L 初始化 t_v ← e_{ỹ_v}
18: for B_j ∈ 𝓑 do
19:     T_j ← B_j ∩ V_L
20:     if |T_j| < s_min then continue
21:     按式 (7)–(10)计算 p_j, m_j, r_j, q_j 和 α_j
22:     对每个 v∈T_j，按式 (11)更新 t_v
23: end for
24: 使用式 (12)训练两层 MLP f_θ
25: return f_θ, {t_v | v∈V_L}
```

### 3.5 共识教师蒸馏扩展

基础 GB-LCSS 只在训练节点上提供直接监督。为了把训练期结构知识传递到全部节点，本文进一步构造共识教师蒸馏扩展。该扩展沿用第 3.3 节的粒球划分和式 (7)、式 (8) 的球内统计，但教师侧使用完整的球内标签分布作为局部目标，即设置 \(\beta_T=0\)。因此

\[
q_j^{T}=p_j,
\qquad
t_v^{T}=(1-\alpha_j)e_{\tilde y_v}+\alpha_j p_j.
\tag{13}
\]

教师 \(g_\phi\) 为两层 GCN，输入为原始属性与归一化冻结表征的拼接

\[
\tilde x_v=[x_v\Vert\bar h_v].
\tag{14}
\]

教师使用邻接矩阵 \(A\)，并仅在训练节点上最小化

\[
\mathcal L_T
=
-\frac{1}{|\mathcal V_L|}
\sum_{v\in\mathcal V_L}
\sum_{c=1}^{C}
t_{vc}^{T}
\log
\operatorname{softmax}
\left(g_\phi(\tilde X,A)_v\right)_c.
\tag{15}
\]

选定教师检查点后固定 \(g_\phi\)，并为全部节点生成温度软化后的预测

\[
\pi_v=
\operatorname{softmax}
\left(
\frac{g_\phi(\tilde X,A)_v}{\tau}
\right).
\tag{16}
\]

X-only 学生在全部 \(N\) 个节点上最小化

\[
\mathcal L_{\mathrm{KD}}
=
-\frac{\tau^2}{N}
\sum_{v\in\mathcal V}
\sum_{c=1}^{C}
\pi_{vc}
\log
\operatorname{softmax}
\left(
\frac{f_\theta(x_v)}{\tau}
\right)_c.
\tag{17}
\]

当前配置取 \(\tau=1\)，学生损失中不再加入原始含噪硬标签交叉熵。训练结束后丢弃教师、邻接矩阵和冻结表征，仅保留学生 MLP。该扩展增加训练期资源，但不改变推理输入。

### 3.6 复杂度与部署开销

GRACE 预训练记为一次性开销 \(T_{\mathrm{GRACE}}\)，其输出表征需要 \(O(Nd_h)\) 的存储。同一份冻结 \(H\) 可在一个数据集的不同噪声设置和下游种子之间复用。

粒球构造至多接受 \(B_{\max}-1\) 次分裂。当前实现每轮扫描候选分区并执行若干次二均值，保守时间复杂度写为

\[
O(B_{\max}Nd_h+T_{\mathrm{2means}}),
\tag{18}
\]

其中第一项表示候选统计与排序所需的累计扫描，\(T_{\mathrm{2means}}\) 表示全部二均值调用的累计代价。粒球节点归属向量的存储为 \(O(N)\)。当前实现显式初始化 \(N\times C\) 的目标矩阵，因此软目标阶段的时间和空间复杂度均包含 \(O(NC)\)。

基础学生采用全批量训练。若训练轮数为 \(T_S\)，其训练复杂度为 \(O(T_SN(dF+FC))\)。训练后的单节点推理复杂度为

\[
O(dF+FC),
\tag{19}
\]

且不随边数 \(|\mathcal E|\) 增长。蒸馏扩展额外承担标准两层 GCN 教师的训练成本，并在全部节点上训练学生。教师与冻结表征均在训练结束后移除。实际墙钟时间、显存和吞吐量在第 5.5 节单独报告。

---

## 4 实验设置

### 4.1 数据集与评测范围

实验覆盖 11 个公开图数据集，统计信息见表 0。所有数据集均采用 20%/10%/70% 的训练、验证和测试划分，并在 uniform 0.4 下完成五种子评测。方法首先在 Cora 和 CiteSeer 上开发。固定 \(\beta\) 与全部构造参数后，进一步在 CoraFull [38]、Coauthor-CS 和 Amazon-Photo [39] 上评测 uniform 0.2/0.4/0.6、pair 0.4 和 instance 0.4 五种噪声设置。PubMed 也完成了相同的噪声谱评测，其结果作为边界分析报告。Amazon-Computers、Cora_ML、Attributed-Wiki、CiteSeerFull 和 Coauthor-Physics 用于补充考察跨数据集表现，其 uniform 0.4 结果列于表 2。

**表 0** 数据集统计。边数按无向边计。同质性为 clean class label 在非自环边两端一致的比例，仅用于描述数据集，不参与训练、超参数选择或目标构造。训练节点数按 20% percent split 计算。

| 数据集 | 节点 | 无向边 | 特征维数 | 类别 | 训练节点 | 边同质性 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 2,708 | 5,278 | 1,433 | 7 | 541 | 0.810 |
| CiteSeer | 3,327 | 4,552 | 3,703 | 6 | 665 | 0.736 |
| CoraFull | 19,793 | 63,421 | 8,710 | 70 | 3,958 | 0.567 |
| Coauthor-CS | 18,333 | 81,894 | 6,805 | 15 | 3,666 | 0.808 |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | 1,530 | 0.827 |
| PubMed | 19,717 | 44,324 | 500 | 3 | 3,943 | 0.802 |
| Amazon-Computers | 13,752 | 245,861 | 767 | 10 | 2,750 | 0.777 |
| Cora_ML | 2,995 | 8,158 | 2,879 | 7 | 599 | 0.789 |
| Attributed-Wiki | 2,405 | 8,990 | 4,973 | 17 | 481 | 0.669 |
| CiteSeerFull | 4,230 | 5,337 | 602 | 6 | 846 | 0.949 |
| Coauthor-Physics | 34,493 | 247,962 | 8,415 | 5 | 6,898 | 0.931 |

### 4.2 标签噪声与数据划分

标签噪声仅注入训练集，验证集与测试集保持干净。对于 \(C\) 类分类，uniform 噪声满足 \(P_{ii}=1-\rho\) 和 \(P_{ij}=\rho/(C-1)\ (i\ne j)\)，其中 \(\rho\in\{0.2,0.4,0.6\}\)。pair 噪声采用 \(\rho=0.4\) 和类映射 \(i\rightarrow(i-1)\bmod C\)。instance-dependent 噪声同样设为 \(\rho=0.4\)。对训练节点 \(v\)，先从截断至 \([0,1]\) 的正态分布 \(q_v\sim\mathcal{N}(\rho,0.1^2)\) 采样翻转率，再由节点特征 \(x_v\) 与类别相关随机映射 \(W_{y_v}\) 计算非真类概率。最终保留真类的概率为 \(1-q_v\)，其余概率质量按 softmax 分配给非真类。

同一数据集和随机种子下的全部方法共享数据划分与噪声实例，验证集用于模型选择。Small-loss、Co-teaching 和 JoCoR 按已知的合成噪声率设置 forget-rate schedule，CE、SCE、ELR、LogitClip 和 GB-LCSS 不使用该信息。该设置遵循 NoisyGL 的合成标签噪声评测协议 [13]。

### 4.3 对比方法与实现细节

主实验中的全部方法采用相同的两层 MLP 骨干，包括 64 维隐层、BatchNorm 和 0.5 dropout。模型使用 Adam 训练，学习率为 0.01，权重衰减为 \(5\times10^{-4}\)，最多训练 200 个 epoch，并根据验证集性能选择检查点。正式对照包括 MLP+CE、Small-loss、Co-teaching [17]、JoCoR [18]、SCE [19]、ELR [20] 和 LogitClip [22]。随机分组对照保持 GB-LCSS 的目标构造与训练流程不变，仅打乱节点的分组归属。

在 uniform 0.4 下，本文还评测了 ANL-CE [21]、GMM+CT [25] 和 FINE [24] 的 MLP-adapted 版本。适配过程保留各方法的核心目标或样本选择机制，同时统一骨干、优化器和节点分类协议。ANL-CE 使用 NCE、NNCE 与 \(L_1\) 的组合目标。GMM+CT 结合逐类置信差趋势与低损失双高斯选择。FINE 按 noisy class 计算未中心化 SVD 主方向，并对投影分数进行逐类双高斯划分。这三项结果用于比较核心机制在统一节点分类协议下的表现，原方法的图像训练流程和完整超参数空间以原论文为准。

### 4.4 统计报告与复现信息

主表报告 5 个配对随机种子的均值、标准差、逐种子方向计数、配对均值差及 95% 置信区间。每个设置采用双侧配对 \(t\) 检验。确认性比较限定为 GB-LCSS 相对 MLP+CE、随机分组和 JoCoR 的三个 25-setting 检验族，并分别进行 Holm 校正。其他方法间比较作为探索性结果报告。由于每个设置仅有 5 个种子，本文结合效应量、置信区间和方向一致性解释结果。

主表中每个数据集的 5 个下游种子共享同一份冻结表征 \(H\)。为估计预训练随机性的影响，本文在 Cora、CiteSeer 和 CoraFull 的 uniform 0.4 设置上额外运行 3 个 GRACE 种子与 3 个下游种子的分层实验。全部实验在单卡 AMD RX 7700 XT（ROCm）上完成。

正式实验中，Cora 的表征维数为 128，其余完整噪声谱数据集为 256。同一数据集的各噪声设置和下游种子复用同一份 \(H\)。补充材料提供表征形状、配置来源、SHA256 及预训练资产清单，第 8 节进一步说明不同数据集的归档范围。

---

## 5 实验结果与分析

### 5.1 与标签噪声学习策略的比较

表 1 比较 GB-LCSS 与七种同骨干标签噪声学习方法在 uniform 0.4 下的性能。表 2 进一步给出其余六个数据集上的结果，以展示方法在统一协议下的跨数据集表现。其他四种噪声设置及逐种子结果见附录 A。

**表 1** uniform 0.4 噪声下的测试精度（%，5 种子均值±标准差）。所有训练方法采用相同的 X-only 两层 MLP。粗体为每列最高均值。

| 方法 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---:|---:|---:|---:|---:|
| MLP+CE | 44.24±2.62 | 43.50±3.02 | 31.98±0.98 | 88.42±0.62 | 73.90±1.80 |
| Small-loss | 51.95±1.97 | 49.70±3.07 | 33.04±0.50 | 89.06±0.88 | 77.25±1.62 |
| Co-teaching [17] | 46.83±1.78 | 46.24±2.89 | 34.39±0.24 | 88.80±0.97 | 78.21±1.00 |
| JoCoR [18] | 51.72±1.30 | 49.91±3.37 | 35.70±0.32 | 89.38±0.61 | 79.57±0.53 |
| SCE [19] | 43.60±1.86 | 42.09±3.47 | 29.29±0.45 | 88.37±0.53 | 76.08±3.11 |
| ELR [20] | 44.28±1.38 | 43.17±3.95 | 32.14±0.67 | 88.29±0.73 | 73.93±2.12 |
| LogitClip [22] | 48.79±1.49 | 47.29±3.26 | 34.69±1.17 | 88.94±0.24 | 76.29±2.00 |
| 随机化分组 | 43.63±1.62 | 42.35±2.74 | 32.27±0.81 | 87.60±1.03 | 72.06±2.97 |
| GB-LCSS | **55.59±1.50** | **53.83±2.75** | **37.33±0.81** | **90.62±0.57** | **81.64±0.93** |

在七个正式同骨干对照中，GB-LCSS 在五个数据集的 uniform 0.4 设置上均取得最高平均精度。扩展到完整噪声谱后，它在 Cora、CiteSeer、CoraFull 和 Amazon-Photo 的 20 个设置中全部领先，在 Coauthor-CS 的 5 个设置中领先 4 个。参数冻结后的三个数据集对应 14/15 个最高均值，计入两个开发数据集后为 24/25。ANL-CE、GMM+CT 和 FINE 的补充评测未改变最强对照，具体结果见附录 B。

真实几何分组为性能增益提供了稳定支撑。仅打乱分组归属后，25 个设置的精度全部下降。GB-LCSS 相对 MLP+CE、随机分组和 JoCoR 的平均增量分别为 7.16、7.83 和 2.87 个百分点，Holm 校正后分别有 24、25 和 8 个设置达到显著水平。相对 JoCoR 的领先幅度较小，23/25 个设置仍有至少 4 个种子保持同向。

与 uniform 0.4 相比，GB-LCSS 在 CoraFull、Coauthor-CS 和 Amazon-Photo 的 pair 与 instance-dependent 噪声下通常取得更大的相对增益，Cora 和 CiteSeer 则未呈现一致趋势。结构化噪声可能放大局部共识的作用，但该现象具有数据集依赖性。

**表 2** 其余六个数据集上的 uniform 0.4 结果（测试精度 %，5 种子均值±标准差）。

| 数据集 | GB-LCSS | 最强对照 | 随机化分组 | GB-LCSS−最强对照 | GB-LCSS−随机化分组 |
|---|---:|---:|---:|---:|---:|
| PubMed | 71.90±0.88 | JoCoR 77.84±1.10 | 62.72±4.49 | −5.94 | +9.18 |
| Amazon-Computers | 75.53±0.87 | JoCoR 76.81±0.85 | 69.89±1.19 | −1.28 | +5.64 |
| Cora_ML | 55.78±3.37 | Small-loss 55.56±3.42 | 49.07±4.10 | +0.22 | +6.71 |
| Attributed-Wiki | 65.42±1.20 | LogitClip 67.83±1.51 | 59.48±1.91 | −2.41 | +5.94 |
| CiteSeerFull | 59.40±1.73 | JoCoR 59.48±1.60 | 46.25±3.18 | −0.08 | +13.14 |
| Coauthor-Physics | 93.11±0.25 | JoCoR 93.09±0.28 | 88.90±2.05 | +0.02 | +4.21 |

表 2 显示，局部几何共识的收益具有明显的数据集依赖性。GB-LCSS 在 PubMed、Amazon-Computers 和 Attributed-Wiki 上明显优于随机分组，同时低于最强同骨干方法。它在 Cora_ML、CiteSeerFull 和 Coauthor-Physics 上与最强对照的差值不超过 0.25 个百分点。方法排序还会随噪声强度变化，例如 CiteSeerFull 从 uniform 0.2 下领先 2.28 个百分点转为 uniform 0.6 下落后 4.71 个百分点。分组由此提供了可测的局部信息，其分类收益还受到数据几何、噪声强度和学生训练策略的共同影响。

### 5.2 与无图推理方法族的比较

表 3 比较 GB-LCSS 与代表性无图推理方法。GLNN-style 使用 GNN 教师的响应蒸馏，Graph-MLP 通过 NContrast 引入多跳邻接监督，FF-G2M 蒸馏低频邻域与高频边差异知识，GSDN 使用邻域自蒸馏。KRD 与 VQGraph 行为机制代理，用于补充分析，不参与正式方法排名。

**表 3** 无图推理机制比较（uniform 0.4，测试精度 %，5 种子均值±标准差，全部方法推理仅使用节点特征）。粗体为每列最高均值。protocol port/reimplementation 表示在统一噪声协议下保留原方法的核心目标，inspired proxy 仅作机制参照。完整表与 uniform 0.6 结果见附录 I。

| 方法 | 训练资源 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---|---:|---:|---:|---:|---:|
| MLP+CE | 含噪标签 | 44.24±2.62 | 43.50±3.02 | 31.98±0.98 | 88.42±0.62 | 73.90±1.80 |
| Graph-MLP (protocol port) | +多跳邻接 NContrast | 64.91±2.04 | 59.87±1.73 | 44.51±0.55 | 89.97±0.39 | 80.44±0.65 |
| GSDN-inspired (mechanism adaptation) | +邻域自蒸馏 | 55.97±4.29 | 50.36±3.39 | 35.80±0.66 | 86.55±0.78 | 73.31±2.00 |
| GB-LCSS | +图（经冻结表征） | 55.59±1.50 | 53.83±2.75 | 37.33±0.81 | 90.62±0.57 | 81.64±0.93 |
| GLNN 式蒸馏 | +GNN 教师+无标签节点 | 77.95±2.62 | 66.09±1.38 | **58.74±0.82** | 93.20±0.09 | 89.86±0.77 |
| FF-G2M (common-backbone reimplementation) | +GNN 教师+LFD/HFD | 77.65±2.97 | 65.80±0.49 | 56.58±0.93 | 91.58±0.24 | 84.82±0.84 |
| KRD-inspired reliability proxy | +GNN 教师+全局可靠性筛选 | 71.77±2.20 | 64.96±1.03 | 47.12±0.95 | 91.39±0.49 | 85.17±0.94 |
| VQGraph-inspired token proxy | +GNN 教师+软码本代理 | 78.89±2.34 | 65.64±1.72 | 58.52±0.77 | 93.25±0.09 | **90.54±0.85** |
| GB-LCSS 蒸馏扩展（球式共识教师） | +球式去噪共识教师+无标签节点 | **80.60±0.69** | **69.40±1.08** | 58.22±0.69 | **93.75±0.22** | 90.49±1.05 |

注：所有方法使用 percent 0.2/0.1/0.7 划分，uniform 0.4 噪声仅注入训练节点，并按相同的五种子协议汇总。

训练阶段使用更丰富的图信息可以明显提高 X-only 学生的精度。Graph-MLP 在三个引文数据集上高于基础 GB-LCSS，而在 Coauthor-CS 和 Amazon-Photo 上分别低 0.65 和 1.20 个百分点。GLNN-style、FF-G2M 和 GB-LCSS 蒸馏扩展进一步利用 GNN 教师与无标签节点，其中 GB-LCSS 蒸馏扩展在 Cora、CiteSeer 和 Coauthor-CS 上取得该组正式方法中的最高均值，在 CoraFull 上低于 GLNN-style。表 5 将进一步分解 H 输入、全节点覆盖和局部共识目标的贡献。

### 5.3 消融与机制分析

#### 5.3.1 局部共识与标签平滑

默认目标中的 \(p_j\) 包含节点自身的含噪标签。为分离外部局部共识与一般标签平滑的作用，本文比较 leave-one-out GB-LCSS（LOO）、LOO 随机分组、均匀标签平滑和含噪训练先验平滑。LOO 在计算节点目标时移除其自身投票，其余训练配置保持不变。下表中的默认 GB-LCSS 来自同批次复跑，配对差值均以该批次为准。

| 数据集 | 当前 GB-LCSS | LOO-GB-LCSS | LOO-随机分组 | 均匀平滑 | 先验平滑 | LOO−当前 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 55.59±1.50 | **57.15±0.84** | 42.79±1.11 | 44.69±2.90 | 44.89±2.38 | +1.56 |
| CiteSeer | 53.83±2.75 | **55.21±2.34** | 41.62±2.80 | 43.69±3.31 | 43.43±2.99 | +1.37 |
| CoraFull | **37.33±0.81** | 37.18±0.48 | 32.18±0.73 | 32.59±1.12 | 32.63±0.84 | −0.15 |
| Coauthor-CS | **90.62±0.57** | 90.61±0.62 | 87.46±1.08 | 88.25±0.80 | 88.04±0.73 | −0.01 |
| Amazon-Photo | 81.59±0.91 | **82.07±1.15** | 72.83±2.67 | 74.18±1.72 | 74.72±1.44 | +0.47 |
| PubMed | 71.69±0.81 | **72.02±0.37** | 63.38±4.60 | 63.67±5.08 | 64.25±4.52 | +0.32 |

移除自票后，LOO 在六个数据集上平均提高 0.60 个百分点，并在全部数据集上优于两种显式平滑和 LOO 随机分组。其 Brier 分数也一致下降，Cora 由 0.488 降至 0.450。这说明其他球内训练节点确实提供了独立监督信息。与此同时，LOO 在提高噪声标签纠正率的同时也增加了干净标签误伤率，Cora 的 CDR 由 0.215 升至 0.302。单个数据集上的 LOO 增量均未通过 Holm 校正，因此本文保留默认目标，并将 LOO 用作机制分析。默认目标体现了外部局部共识与保守组感知平滑之间的折中。

#### 5.3.2 分组形式与有效票数

同一 GRACE 表征上的 kNN 和 kmeans 可以获得与粒球相近的局部共识效果。uniform 0.4 下，\(k=20\) 的 kNN 比默认粒球高 0.3 至 2.6 个百分点。进一步分析显示，该差异主要与共识单元中的有效训练票数有关。默认球平均包含约 10 个训练节点。将最小球规模由 32 增至 64 后，平均票数提高到约 24，CiteSeer 上粒球与 kNN 的差距由 2.58 缩小至 0.77 个百分点，PubMed 上两者数值持平。

较高的有效票数即使伴随球内纯度下降，也能改善下游性能。较大球的纯度由 0.505 降至 0.477，精度却进一步提高。kNN 消融呈现相同趋势，四个数据集的最优 \(k\) 均为 20，而 \(k=5\) 在 Cora 上得到 53.93，低于粒球的 55.59，这与有效票数不足的解释一致。kNN 对照保留查询节点自身，与默认粒球目标中的自票设置一致。在当前 20% 标签比例下，稳定的多数统计比过细分组更重要。粒球以固定预算和近线性构造成本获得了与 kNN 相近的局部共识精度。参数敏感性结果见附录 D.1。

#### 5.3.3 共识空间

局部共识的有效性取决于分组所在的表征空间。在原始特征空间中，递归二分退化为单一分组。直接在邻接空间传播含噪标签得到的目标比 GRACE 空间共识低 6.9 至 8.3 个百分点，三个数据集的五个种子均保持同向。以含噪监督 GCN 隐层构造分组时，6/6 个设置仍优于随机分组，但只有 1 个设置比 GRACE 高出 0.5 个百分点以上。在本文比较的三种空间中，冻结 GRACE 表征为粒球提供了最稳定的局部几何基础。

#### 5.3.4 学生容量与 PubMed

将学生模型替换为线性分类器后，GB-LCSS 在五个数据集上均高于线性 CE 和线性随机分组。PubMed 上相对线性 CE 的增量为 0.68 个百分点，其余数据集的差距更明显。与之相比，样本选择方法在更弱的学生上退化较大。PubMed uniform 0.2 下，线性 Small-loss 与 MLP Small-loss 分别为 69.60 和 81.88。uniform 0.4 下，线性 DivideMix-lite 与对应的 MLP 代理分别为 66.32 和 78.28。GB-LCSS 线性版本在 PubMed 的五种噪声设置中均高于对应的最强线性选择方法，差值为 7.4 至 14.3 个百分点。

这一反转表明，PubMed 上的方法排序与学生容量有关。样本选择依赖训练过程形成的损失分离，局部共识目标则在训练前一次性构造，对学生记忆噪声能力的依赖较弱。因此，表 2 中的结果体现了弱 MLP 与训练策略之间的交互。局部共识在线性学生上仍保持稳定增益。线性选择方法均为统一协议下的适配实现，详细结果见附录。

#### 5.3.5 H-access 参照

冻结表征 \(H\) 携带完整图上的结构信息。下表比较 X-only GB-LCSS 与直接使用 \(H\) 的分类器，所有结果均为 uniform 0.4 下的五种子均值。使用 \(H\) 的方法在推理时需要预计算的图派生表征，并在图更新或新节点加入后重新运行编码器。

| 分类路线 | 推理输入 H | Cora | CiteSeer | PubMed |
|---|:---:|---:|---:|---:|
| GB-LCSS（X-only） | 否 | 55.59 | 53.83 | 71.90 |
| H-only 线性分类头 | 是 | 78.61 | 68.87 | 83.87 |
| H-only MLP 分类头 | 是 | 73.09 | 68.69 | 84.05 |
| MLP[X‖H]+CE | 是 | 48.54 | 51.66 | 83.68 |
| MLP[X‖H]+Small-loss | 是 | 57.70 | 56.30 | **84.83** |
| MLP[X‖H]+JoCoR | 是 | **60.13** | **58.60** | 84.79 |

直接使用 \(H\) 可以大幅提高分类精度，但收益取决于训练策略。[X‖H]+CE 在 Cora 上仅为 48.54，而 [X‖H]+JoCoR 达到 60.13，高于 X-only GB-LCSS。H-only 分类头在三个数据集上进一步达到 68.69 至 84.05。该结果明确区分了两类部署条件：直接使用 \(H\) 的方法具有更强输入，而 GB-LCSS 的最终分类器仅接收 \(X\)。第 6 节通过教师蒸馏研究如何将这部分结构知识转移给 X-only 学生。

#### 5.3.6 标签率与输入扰动

低标签率会放大局部共识的相对价值。在 Cora 和 CiteSeer 上，训练标签比例由 65% 降至 5% 时，GB-LCSS 相对最强样本选择方法的优势由 0.2 至 1.4 个百分点扩大到 5.6 至 6.7 个百分点，完整曲线见附录 H。零噪声下，GB-LCSS 相对 CE 在 Cora、CiteSeer、PubMed、Coauthor-CS 和 Amazon-Photo 上的差值分别为 +0.44、+2.33、+0.02、−0.88 和 −1.20 个百分点，表明共识修正主要适用于监督受污染的场景。

测试特征扰动揭示了 X-only 部署的另一项限制。加入高斯噪声后，GB-LCSS 在 Cora、CiteSeer 和 Amazon-Photo 上仍保持相对优势，但在 Cora 和 CiteSeer 上的绝对精度下降快于 CE，Cora 的全程降幅为 34.0 个百分点，而 CE 为 22.8 个百分点。这说明标签噪声下的优势不能直接转化为特征噪声鲁棒性。

### 5.4 冻结无监督对比表征的经验选择

冻结 GRACE 表征在当前粒球构造中表现出最稳定的局部可聚类性。向 GRACE 预训练加入权重为 0.1 的含噪交叉熵后，分组由 52 个塌缩为 1 个，类别纯度由 0.79 降至 0.30，下游精度下降 12.6 个百分点。训练后 MLP 隐层、CCA-SSG 和 GraphMAE 表征也在相同构造参数下退化为单一分组。后两者虽然具有更高的线性探针精度，却未形成适合局部共识的几何结构。这些结果说明，线性可分性与局部可聚类性是不同的表征性质。附录 G 给出相应可视化。

预训练随机性没有改变上述结论。在 Cora、CiteSeer 和 CoraFull 的 \(3\) 个 GRACE 种子 \(\times 3\) 个下游种子实验中，不同 \(H\)-seed 下平均精度的标准差分别为 0.53、0.07 和 0.44 个百分点，与下游随机性处于相近量级。每份 \(H\) 上，GB-LCSS 相对随机分组、CE 和 JoCoR 的增量均保持为正。不同 \(H\) 产生的具体分区差异较大，adjusted Rand index 仅为 0.25 至 0.37，但下游收益仍较稳定。这表明性能依赖跨分区稳定的局部共识统计，对具体成员划分并不敏感。


### 5.5 计算与部署开销

表 4 将一次性预处理、下游训练与部署开销分开报告。下游计时使用同一 ROCm GPU 和 seed 0，推理延迟为最佳模型 100 次 full-batch 前向的均值。Cora、CiteSeer 和 CoraFull 的 GRACE 平均预训练时间分别为 0.07、0.13 和 74.48 分钟，峰值显存分别为 0.40、0.67 和 12.73 GiB。预训练在每个数据集上执行一次，其表征可供全部噪声设置复用。

**表 4** 下游训练与 X-only 推理的计算开销（seed 0）。GRACE 预训练成本单独报告，不计入下游墙钟。

| 数据集 | 方法 | 参数量 (K) | FP32 参数 (MiB) | 下游墙钟 (s) | 构球 (s) | 推理 (ms/全图) | 吞吐 (M 节点/s) | 峰值推理显存 (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | MLP+CE | 92.4 | 0.35 | 2.16 | 0 | 0.388 | 6.97 | 126.2 |
| Cora | GB-LCSS | 92.4 | 0.35 | 5.20 | 4.18 | 0.389 | 6.95 | 126.2 |
| Cora | GCN | 92.2 | 0.35 | 2.39 | 0 | 2.171 | 1.25 | 160.4 |
| CiteSeer | MLP+CE | 237.6 | 0.91 | 1.32 | 0 | 1.121 | 2.97 | 193.4 |
| CiteSeer | GB-LCSS | 237.6 | 0.91 | 4.99 | 3.62 | 1.111 | 3.00 | 193.5 |
| CiteSeer | GCN | 237.4 | 0.91 | 2.02 | 0 | 2.364 | 1.41 | 196.1 |
| CoraFull | MLP+CE | 562.2 | 2.15 | 8.83 | 0 | 15.120 | 1.31 | 821.7 |
| CoraFull | GB-LCSS | 562.2 | 2.15 | 13.34 | 4.49 | 14.987 | 1.32 | 827.7 |
| CoraFull | GCN | 562.1 | 2.14 | 7.15 | 0 | 10.996 | 1.80 | 846.8 |

GB-LCSS 与 MLP+CE 使用相同的学生模型，因此二者的参数量、推理延迟和显存基本一致，额外下游成本主要是约 4 秒的粒球构造。MLP 在 Cora 和 CiteSeer 上快于 GCN，而 CoraFull 的高维稠密特征使其前向耗时更高。由此可见，GB-LCSS 的稳定部署收益来自推理阶段无需读取邻接矩阵或图衍生表征，实际延迟仍受特征维度和执行方式影响。

---

## 6 扩展：以共识教师蒸馏弥合精度差距

基础 GB-LCSS 将图结构压缩为训练节点上的软目标，未标记节点不参与学生的直接监督。即使将 Cora 的训练比例提高到 65%，其 uniform 0.4 精度仍为 64.0，低于仅使用 20% 标签的 GCN 约 14 个百分点。共识教师蒸馏通过图教师为全部节点生成逐节点软预测，扩大了监督覆盖，也保留了更细粒度的结构信息。

BO-NNC 与 SAMTD 通过多教师组合和权重学习提高鲁棒性 [45,58]。本文固定单一 GCN 教师结构，只改变教师的训练目标。表 5 比较含噪标签、随机分组、kNN 共识和粒球共识四种监督来源，以分析 H 输入、全节点覆盖和局部共识去噪对学生性能的影响。

**表 5** 共识教师蒸馏（测试精度 %，5 种子均值±标准差）。粗体为每列最高均值。学生均为同一仅特征 MLP，仅教师不同。粒球教师使用完整球内分布目标（\(\beta_T=0\)）。kNN 行按每个 seed 的平均球内训练票数匹配 \(k\)，以控制有效共识票数。完整增益分解见附录 I。

| 教师 | Cora u0.4 | CiteSeer u0.4 | CoraFull u0.4 | Cora u0.6 | CiteSeer u0.6 | CoraFull u0.6 |
|---|---:|---:|---:|---:|---:|---:|
| GCN(X)+CE（GLNN 式） | 77.95±2.62 | 66.09±1.38 | 58.74±0.82 | 65.50±3.28 | 53.30±4.42 | 51.63±2.02 |
| GCN[X‖H]+CE | 80.23±1.57 | 68.81±1.12 | **59.45±0.63** | 69.44±2.18 | 61.04±3.08 | **52.75±0.88** |
| 随机分组共识教师 | 78.81±1.63 | 68.60±1.14 | 58.98±0.41 | 68.51±1.82 | 61.07±2.76 | 52.20±0.72 |
| kNN（票数匹配）共识教师 | **80.99±1.02** | **69.57±1.46** | 58.95±0.56 | 71.07±2.29 | 61.96±2.98 | 52.73±0.95 |
| 球式共识教师（本文） | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 | **71.27±2.46** | **62.28±2.30** | 51.82±0.86 |

共识教师蒸馏明显缩小了 X-only 学生与 GCN 的精度差距。由球式共识教师蒸馏得到的学生在 Cora 和 CiteSeer uniform 0.4 上达到 80.60 和 69.40，分别高于 GLNN 式学生 2.65 和 3.31 个百分点。uniform 0.6 下，该优势进一步扩大到 5.77 和 8.98 个百分点。CoraFull 上，球式共识学生低于 GCN[X‖H]+CE 和 kNN 教师对应的学生，表明该数据集的主要收益来自 \(H\) 输入与全节点覆盖。

H 输入提供了蒸馏扩展中的主要精度增量，局部共识目标在较高噪声下进一步改善教师监督。票数匹配 kNN 教师与球式教师的六个差值均小于 1 个百分点，说明两种局部共识实现具有相近效果。附录 F 中，蒸馏学生的 cut value 比 train-only 学生高 0.14 至 0.25，且五个种子方向一致。该现象与软标签蒸馏向纯特征学生传递图偏置的理论分析 [32] 相符。

表 6 将蒸馏学生与推理阶段仍使用图结构的 GCN、NRGNN [9]、TFR [40]、ProCon [41] 和 GD² [42] 置于同一噪声协议下比较。两类方法的训练资源与推理输入不同，因此该表用于衡量 X-only 学生与图模型之间的精度距离。其他图方法的运行与协议记录见附录 C。

**表 6** 图领域专用去噪方法参照（uniform 0.4，测试精度 %，5 种子均值±标准差）。粗体为每列最高均值。图依赖方法在推理阶段需要邻接矩阵，本表作为跨部署资源参照，仅列入完成五数据集统一协议评测的方法。

| 方法 | 推理需图 | Cora | CiteSeer | CoraFull | Coauthor-CS | Amazon-Photo |
|---|---|---:|---:|---:|---:|---:|
| GCN(X) | 是 | 78.97±1.63 | 64.53±1.54 | 61.28±0.52 | 92.06±0.34 | 90.77±0.43 |
| GCN[X‖H] | 是 | 80.70±1.54 | 67.48±1.39 | 61.42±0.44 | 92.05±0.19 | **91.87±0.64** |
| NRGNN | 是 | **81.76±1.28** | **71.03±1.01** | **62.70±0.63** | 93.56±0.11 | 89.02±1.08 |
| NRGNN[X‖H] | 是 | 81.47±1.28 | 70.74±1.04 | 62.06±0.52 | 93.72±0.18 | 89.53±1.53 |
| TFR-GCN | 是 | 78.75±0.11 | 68.92±1.77 | 42.12±1.14 | 78.36±2.39 | 35.36±9.43 |
| ProCon | 是 | 77.19±2.17 | 61.07±3.32 | 36.42±3.48 | 86.08±6.85 | 20.74±4.82 |
| GD2 | 是 | 76.89±3.51 | 63.07±1.58 | 59.88±0.79 | 91.57±0.18 | 91.04±1.00 |
| **GB-LCSS 蒸馏扩展（学生）** | **否** | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 | **93.75±0.22** | 90.49±1.05 |

注：TFR-GCN、ProCon 与 GD2 为统一协议下的本地适配或重实现结果。

蒸馏后的 X-only 学生已接近强图模型的精度。在 Cora、CiteSeer 和 CoraFull 上，它比 NRGNN 分别低 1.16、1.63 和 4.48 个百分点。在 Coauthor-CS 上，学生达到 93.75，与最强图依赖方法数值接近。Amazon-Photo 上的差距约为 1.4 个百分点。结果表明，共识教师蒸馏能够在不改变学生推理输入的条件下，将基础 GB-LCSS 从同骨干抗噪训练方法提升到具有竞争力的 GCN 精度区间。

---

## 7 结论

本文面向带噪训练标签下的仅属性节点分类，提出粒球局部共识软监督方法 GB-LCSS。该方法在冻结图对比表征上构造粒球，将组内含噪标签聚合为软监督目标，并训练推理阶段只接收节点属性的分类器。在七个正式同骨干对照中，GB-LCSS 在参数冻结后完成完整噪声谱评测的三个数据集上，于 15 个设置中的 14 个取得最高平均精度。纳入两个开发数据集后，该计数为 24/25。

实验进一步揭示了局部共识发挥作用的条件。真实分组在全部完整设置中均优于随机分组，移除节点自身投票后，球内其他训练节点仍能提供有效监督。粒球、kNN 和 kmeans 的比较表明，有效训练票数比具体分组形式更关键，粒球提供的是一种固定预算、计算简洁的局部共识实现。共识教师蒸馏则将训练期图结构转化为全节点软监督，使 X-only 学生在多个数据集上达到具有竞争力的 GCN 精度。

GB-LCSS 依赖能够形成稳定局部几何的冻结图表征，目前采用 transductive 预训练，并在完全干净标签和部分数据集上表现出有限收益。未来工作将面向真实标注噪声、inductive 表征学习和更大规模图，进一步研究可部署的局部共识估计与不确定性建模。

---

## 8 数据、代码与结果资产

本文使用的图数据集均为公开基准。投稿补充材料归档完整配置、逐种子结果、split/noise hash、统计脚本、冻结 H 及其 SHA256。匿名代码仓库链接将在投稿排版阶段补入。Cora、CiteSeer 与 CoraFull 的三预训练种子 checkpoint、H、loss 与成本清单已归档，可重新提取和核验对应 H。Coauthor-CS 与 Amazon-Photo 通过归档 H 复现下游结果，现有资产未包含精确重建原始表征所需的 encoder checkpoint。artifact manifest 将按数据集标明可复现范围。

---

## 参考文献

[1] T. N. Kipf, M. Welling, Semi-supervised classification with graph convolutional networks, in: Proc. ICLR, 2017.

[2] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, Y. Bengio, Graph attention networks, in: Proc. ICLR, 2018.

[3] S. Zhang, Y. Liu, Y. Sun, N. Shah, Graph-less neural networks: Teaching old MLPs new tricks via distillation, in: Proc. ICLR, 2022.

[4] Y. Hu, H. You, Z. Wang, Z. Wang, E. Zhou, Y. Gao, Graph-MLP: Node classification without message passing in graph, arXiv preprint arXiv:2106.04051, 2021.

[5] Y. Tian, C. Zhang, Z. Guo, X. Zhang, N. V. Chawla, Learning MLPs on graphs: A unified view of effectiveness, robustness, and efficiency, in: Proc. ICLR, 2023.

[6] L. Yang, et al., VQGraph: Rethinking graph representation space for bridging GNNs and MLPs, in: Proc. ICLR, 2024.

[7] L. Wu, H. Lin, Y. Huang, S. Z. Li, Quantifying the knowledge in GNNs for reliable distillation into MLPs, in: Proc. ICML, 2023.

[8] C. Zhang, S. Bengio, M. Hardt, B. Recht, O. Vinyals, Understanding deep learning (still) requires rethinking generalization, Communications of the ACM 64 (3) (2021) 107–115.

[9] E. Dai, C. Aggarwal, S. Wang, NRGNN: Learning a label noise resistant graph neural network on sparsely and noisily labeled graphs, in: Proc. ACM SIGKDD, 2021.

[10] S. Qian, et al., Robust training of graph neural networks via noise governance, in: Proc. WSDM, 2023.

[11] K. Ding, et al., Divide and denoise: Empowering simple models for robust semi-supervised node classification against label noise, in: Proc. ACM SIGKDD, 2024.

[12] Y. Zhu, L. Feng, Z. Deng, Y. Chen, R. Amor, M. Witbrock, Robust node classification on graph data with graph and label noise, in: Proc. AAAI, 2024.

[13] Z. Wang, et al., NoisyGL: A comprehensive benchmark for graph neural networks under label noise, in: Proc. NeurIPS Datasets and Benchmarks Track, 2024.

[14] L. Wu, J. Xia, H. Lin, Z. Gao, Z. Liu, G. Zhao, S. Z. Li, Teaching yourself: Graph self-distillation on neighborhood for node classification, arXiv preprint arXiv:2210.02097, 2022.

[15] Z. Wang, Z. Zhang, C. Zhang, Y. Ye, Training MLPs on graphs without supervision, in: Proc. WSDM, 2025.

[16] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, L. Wang, Deep graph contrastive representation learning, in: ICML Workshop on Graph Representation Learning and Beyond, 2020.

[17] B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. W. Tsang, M. Sugiyama, Co-teaching: Robust training of deep neural networks with extremely noisy labels, in: Proc. NeurIPS, 2018.

[18] H. Wei, L. Feng, X. Chen, B. An, Combating noisy labels by agreement: A joint training method with co-regularization, in: Proc. CVPR, 2020.

[19] Y. Wang, X. Ma, Z. Chen, Y. Luo, J. Yi, J. Bailey, Symmetric cross entropy for robust learning with noisy labels, in: Proc. ICCV, 2019.

[20] S. Liu, J. Niles-Weed, N. Razavian, C. Fernandez-Granda, Early-learning regularization prevents memorization of noisy labels, in: Proc. NeurIPS, 2020.

[21] X. Ye, X. Li, S. Dai, T. Liu, Y. Sun, W. Tong, Active negative loss functions for learning with noisy labels, in: Proc. NeurIPS, 2023.

[22] H. Wei, H. Zhuang, R. Xie, L. Feng, G. Niu, B. An, Y. Li, Mitigating memorization of noisy labels by clipping the model prediction, in: Proc. ICML, 2023.

[23] J. Li, R. Socher, S. C. H. Hoi, DivideMix: Learning with noisy labels as semi-supervised learning, in: Proc. ICLR, 2020.

[24] T. Kim, J. Ko, S. Cho, J. Choi, S.-Y. Yun, FINE samples for learning with noisy labels, in: Proc. NeurIPS, 2021.

[25] W. Pan, W. Wei, F. Zhu, Y. Deng, Enhanced sample selection with confidence tracking: Identifying correctly labeled yet hard-to-learn samples in noisy data, in: Proc. AAAI, 39 (19) (2025) 19795–19803. https://doi.org/10.1609/aaai.v39i19.34180.

[26] L. Wu, H. Lin, Y. Huang, T. Fan, S. Z. Li, Extracting low-/high-frequency knowledge from graph neural networks and injecting it into MLPs: An effective GNN-to-MLP distillation framework, in: Proc. AAAI, 2023.

[27] J. Chen, M. Bai, S. Chen, J. Gao, J. Zhang, J. Pu, SA-MLP: Distilling graph knowledge from GNNs into structure-aware MLP, Transactions on Machine Learning Research (2024).

[28] T. Wu, Z. Zhao, J. Wang, X. Bai, L. Wang, N. Wong, Y. Yang, Edge-free but structure-aware: Prototype-guided knowledge distillation from GNNs to MLPs, in: Proc. COLING, 2025, pp. 5656–5667.

[29] W. Lu, et al., AdaGMLP: AdaBoosting GNN-to-MLP knowledge distillation, in: Proc. ACM SIGKDD, 2024.

[30] W. Lu, Z. Guan, W. Zhao, Y. Yang, Y. Sun, Z. Liang, Y. Zhan, D. Tao, ProGMLP: A progressive framework for GNN-to-MLP knowledge distillation with efficient trade-offs, in: Proc. AAAI, 40 (29) (2026) 24088–24096. https://doi.org/10.1609/aaai.v40i29.39587.

[31] H. Pal, S. Malik, R. Patel, A. Malhotra, SALE-MLP: Structure aware latent embeddings for GNN to graph-free MLP distillation, in: Proc. IJCAI, 2025, pp. 6003–6011. https://doi.org/10.24963/ijcai.2025/668.

[32] Z. Yu, M. Lin, W. Li, Z. Yin, S. Xiao, S. Lu, Demystifying GNN-to-MLP knowledge transfer: Theoretical grounding and dual-stream distillation method, in: Proc. AAAI, 40 (33) (2026) 28005–28013. https://doi.org/10.1609/aaai.v40i33.40025.

[33] M. Kohn, M. Hoffmann, A. Scherp, Edge-splitting MLP: Node classification on homophilic and heterophilic graphs without message passing, in: Proc. Third Learning on Graphs Conference, PMLR 269 (2025) 11:1--11:21. https://proceedings.mlr.press/v269/kohn25a.html.

[34] S. Xia, Y. Liu, X. Ding, G. Wang, H. Yu, Y. Luo, Granular ball computing classifiers for efficient, scalable and robust learning, Information Sciences 483 (2019) 136–152.

[35] F. Dunkin, X. Li, H. Li, G. Wu, C. Hu, S. S. Ge, MgCNL: A sample separation approach via multi-granularity balls for fault diagnosis with the interference of noisy labels, IEEE Transactions on Automation Science and Engineering 22 (2025) 7748–7761. https://doi.org/10.1109/TASE.2024.3469000.

[36] F. Dunkin, X. Li, C. Hu, G. Wu, H. Li, X. Lu, Z. Zhang, Like draws to like: A multi-granularity ball-intra fusion approach for fault diagnosis models to resists misleading by noisy labels, Advanced Engineering Informatics 60 (2024) 102425. https://doi.org/10.1016/j.aei.2024.102425.

[37] S. Xia, S. Zheng, G. Wang, X. Gao, B. Wang, Granular ball sampling for noisy label classification or imbalanced classification, IEEE Transactions on Neural Networks and Learning Systems 34 (4) (2023) 2144–2155. https://doi.org/10.1109/TNNLS.2021.3105984.

[38] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, T. Eliassi-Rad, Collective classification in network data, AI Magazine 29 (3) (2008) 93–106.

[39] O. Shchur, M. Mumme, A. Bojchevski, S. Gunnemann, Pitfalls of graph neural network evaluation, in: Relational Representation Learning Workshop, NeurIPS, 2018. https://www.cs.cit.tum.de/daml/gnn-benchmark/.

[40] Z. Wang, Y. Bei, S. Zhou, Z. Zhou, J. Fan, H. Xue, H. Wang, J. Bu, Learning from graph: Mitigating label noise on graph through topological feature reconstruction, in: Proc. CIKM, 2025, pp. 3261–3270. https://doi.org/10.1145/3746252.3761185.

[41] K. Li, J. Sun, J. Lou, Z. Feng, H. Zhou, C. Wu, G. Xue, W. Zhao, J. Li, Leveraging peer-informed label consistency for robust graph neural networks with noisy labels, in: Proc. IJCAI, 2025, pp. 5598–5606. https://doi.org/10.24963/ijcai.2025/623.

[42] K. Li, J. Lou, J. Sun, H. Zeng, W. Li, C. Wu, Y. Luo, W. Zhao, S. Du, J. Li, GD²: Robust graph learning under label noise via dual-view prediction discrepancy, Advances in Neural Information Processing Systems 38 (2025).

[43] Z. Zhou, Z. Ding, J. Shi, L. Qing, S. Shen, TINED: GNNs-to-MLPs by teacher injection and Dirichlet energy distillation, in: Proc. ICML, PMLR 267 (2025) 78616--78632. https://proceedings.mlr.press/v267/zhou25a.html.

[44] S. Li, B. Zhang, J. Song, G. Ruan, C. Wang, J. Xie, Graph neural networks with coarse- and fine-grained division for mitigating label noise and sparsity, Neural Networks 187 (2025) 107338. https://doi.org/10.1016/j.neunet.2025.107338.

[45] B. Hu, G. Wen, Y. Huang, Y. Liu, X. Zhu, Structure-aware multi-teacher distillation for noise node classification, Information Processing & Management 63 (7) (2026) 104877. https://www.sciencedirect.com/science/article/pii/S0306457326002682.

[46] Y. Li, J. Yin, L. Chen, Unified robust training for graph neural networks against label noise, in: Proc. PAKDD, Lecture Notes in Computer Science 12712 (2021) 528--540. https://doi.org/10.1007/978-3-030-75762-5_42.

[47] Y. Wu, J. Yao, X. Xia, J. Yu, R. Wang, B. Han, T. Liu, Mitigating label noise on graphs via topological sample selection, in: Proc. ICML, PMLR 235 (2024) 53944--53972. https://proceedings.mlr.press/v235/wu24ae.html.

[48] J. Yang, Q. Liu, X. Cao, F. Li, X. Yang, G-GBC: A Gaussian mixture-based granular ball computing method for robust CNN classification under noisy labels, Knowledge-Based Systems 341 (2026) 115754. https://doi.org/10.1016/j.knosys.2026.115754.

[49] Y. Li, X. Ouyang, J. Zhang, C. Pan, L. Ren, X. Yang, Beneficial noise learning for open intent classification via granular-ball representation, Pattern Recognition 177 (2026) 113283. https://doi.org/10.1016/j.patcog.2026.113283.

[50] Z. Liu, H. Yu, X. Luo, Mitigating label noise in graph learning with information bottleneck-guided aggregation and adversarial consistency, Pattern Recognition 180 (2026) 114271. https://doi.org/10.1016/j.patcog.2026.114271.

[51] S. Li, B. Zhang, M. Cao, J. Song, X. Cheng, C. Wang, HALF: A homophily-aware loss fusion for robust learning under label noise in heterophilic graphs, Information Sciences 754 (2026) 123714. https://doi.org/10.1016/j.ins.2026.123714.

[52] X. Peng, P. Wang, S. Xia, C. Wang, W. Chen, VPGB: A granular-ball based model for attribute reduction and classification with label noise, Information Sciences 611 (2022) 504--521. https://doi.org/10.1016/j.ins.2022.08.066.

[53] J. Zhang, X. Sun, B. Huang, T. Wang, H. Li, X. Jia, GEAF: A label noise filtering method based on adaptive partitioning of granular ellipsoids, Information Sciences 742 (2026) 123324. https://doi.org/10.1016/j.ins.2026.123324.

[54] S. Xia, X. Ma, Z. Liu, C. Liu, S. Zhao, G. Wang, Graph coarsening via supervised granular-ball for scalable graph neural network training, in: Proc. AAAI, 39 (12) (2025) 12872--12880. https://doi.org/10.1609/aaai.v39i12.33404.

[55] S. Xia, G. Wang, G. Xu, S. Zhao, G. Wang, GBGC: Efficient and adaptive graph coarsening via granular-ball computing, in: Proc. IJCAI, 2025, pp. 3489--3497. https://doi.org/10.24963/ijcai.2025/388.

[56] J. Ni, S. Xia, Y. Guan, G. Xu, L. Chen, Y. Liu, Y. Wang, Multi-granularity graph refinement via granular ball for graph classification, Neurocomputing 686 (2026) 133713. https://doi.org/10.1016/j.neucom.2026.133713.

[57] Z. Wang, T. Zhang, S. Xia, L. Lin, G. Wang, GBRAIN: Combating textual label noise by granular-ball based robust training, in: Proc. ACM ICMR, 2024, pp. 357--365. https://doi.org/10.1145/3652583.3658084.

[58] Y. Liu, Z. Wu, Z. Lu, C. Nie, G. Wen, Y. Zhu, X. Zhu, Noisy node classification by bi-level optimization based multi-teacher distillation, in: Proc. AAAI, 39 (18) (2025) 19033--19040. https://doi.org/10.1609/aaai.v39i18.34095.

[59] M. Zhou, X. Han, W. Wei, G. Yan, Graph topology adaptive judgment against node label noise, Knowledge-Based Systems 328 (2025) 114162. https://doi.org/10.1016/j.knosys.2025.114162.

---

## 附录索引（投稿时编排为 Supplementary Material）

补充材料包含完整结果、统计分析、机制消融、复现资产和协议审计，正文表格与补充材料中的对应数据已完成一致性核验。

- **附录 A** 完整主表：核心 5 数据集（Cora/CiteSeer/CoraFull/Coauthor-CS/Amazon-Photo）× 5 噪声设置，正式排名包含 GB-LCSS、随机化消融与 7 个完整噪声谱对照。uniform 0.4 另列 ANL-CE、GMM+CT 和 FINE 三项统一协议重实现。机制代理方法单独标注，不计入已发表方法的胜出统计。附录同时给出 5 种子均值±标准差、配对方向与统计检验，并逐式给出 uniform/pair/instance 转移规则、pair 类映射、截断正态参数和随机种子绑定方式。PubMed、Amazon-Computers、Cora_ML、Attributed-Wiki、CiteSeerFull 与 Coauthor-Physics 另列为适用域审计。
- **附录 B** 预声明主比较族统计：GB-LCSS 分别相对 MLP+CE、随机化分组和 JoCoR 的逐设置配对差、95% 置信区间、方向计数及 family-wise Holm 校正结果。其余方法间检验单列为探索性结果，不据此作显著性 claim。n=5 下不以未拒绝零假设证明等效。
- **附录 C** 图依赖参考方法：正文表 6 的完整版（逐种子数字、可行性记录与协议审计）。所有纳入比较的结果均采用 `noise_scope=train`，并通过验证集与测试集标签零污染核验。UnionNET 和 TSS 仅完成 Cora/CiteSeer 评测，未覆盖最终五数据集统一矩阵，其结果和运行记录保留在补充材料中。RNCGLN 的单种子或预测坍缩记录，以及 RTGNN/CRGNN 在 Amazon-Photo 上的高方差结果一并报告。
- **附录 D** 构造参数敏感性与 β 选择审计。敏感性实验覆盖 27 个配置。合理参数范围内的波动约为 ±1 个百分点，δ=0.1 会导致分组塌缩为单组。组数上限 32 与 128 在 Cora 和 CiteSeer 上给出逐种子相同结果，说明小图对该范围不敏感。五个主结果数据集共形成 52 至 64 个分组，并在 25 个设置中的 19 个达到球数上限。β 选择审计提供逐设置的验证集选择记录，用于支撑正文对默认参数的说明。
- **附录 E** 测试特征噪声完整曲线（5 方法 × 5 数据集 × 6 个扰动水平，零扰动点逐种子锚定主表数字）。
- **附录 F** Cut value 结构一致性表，附该指标在标签噪声协议下受预测坍缩污染的说明（CE 与随机分组对照的逐种子方差达 0.15–0.28，蒸馏臂仅 0.003–0.01）。正文仅引用蒸馏变体与仅训练标签变体之间的组内差值。
- **附录 G** 表征空间可视化：t-SNE 三着色图（真实类别/噪声标签/分组归属）、粒球覆盖示意图与噪声软修正机制图（图 G.1–G.3，均为机制示意，定量证据以正文消融为准）。
- **附录 H** 低标签率完整曲线（6 个标签比例 × 5 方法）及趋势单调性核验（Cora 严格单调，CiteSeer 存在一个回折点，故正文表述为总体扩大）。
- **附录 I** 正文表 3 与表 5 的完整版本（全部噪声设置、训练资源标注列、教师精度与三层增益分解）。正文表 5 的 kNN 行按每个 seed 的平均球内训练票数匹配 k。当前 CoraFull 已补齐表 3 的 graph-free family 行、正文表 5 的教师分解与正文表 6 的主要图依赖参照。RTGNN、CLNode、CRGNN 与 RNCGLN 的 CoraFull 行暂不进入正文表。
