# B领域：图粗化、传播保持、训练迁移与效率

> 检索截止：2026-10-19。完成七篇核心论文的方法与理论/实验关键部分阅读；其中 ConvMatch 为 WWW 2024 companion workshop paper，非主会论文。主读范围限定于原文方法、关键理论及实验协议；本报告将作者结论与本领域推断分开。

## 检索范围与选文理由

本领域重点不是泛泛列举“压缩图”，而是区分三个不同问题：

1. **粗化（coarsening）**：把原节点按硬分区合并成超节点，通常仍有原节点归属映射；粗边由原边加权聚合，超节点特征由成员聚合得到。
2. **凝缩（condensation）**：直接优化合成节点/边或合成无图特征，使小数据集替代大图训练；其表示不要求对应任何原节点集合。
3. **传播/任务保持**：谱相似、信号能量保持、特定消息传递算子近似、GNN轨迹/梯度匹配，目标不同，保证范围不能互换。

本领域主读的七篇按本轮分工选取为 SCAL、Loukas、FGC、ConvMatch、Message-Passing Guarantees、GCond、SFGC。前三篇是结构/特征粗化和受限谱保证；ConvMatch与Message-Passing Guarantees分别提供传播响应目标和直接的传播算子保证；GCond与SFGC是合成凝缩，不对应原节点分区。SGBGC/GBGC、ACE/CPF、PLGC/RobGC归其他领域，不计入本领域七篇核心数。

## 已主读论文

### 1. SCAL：Scaling Up Graph Neural Networks Via Graph Coarsening

**题录与来源。** Zengfeng Huang, Shengzhong Zhang, Chong Xi, Tang Liu, Min Zhou. KDD 2021。作者预印本：[arXiv:2106.05150](https://arxiv.org/abs/2106.05150)；DOI: [10.1145/3447548.3467256](https://doi.org/10.1145/3447548.3467256)。全文作者版本可在线读取。方法阅读：§3.1–3.2，pp. 4–5；理论：§4.1–4.5，pp. 5–10；实验协议及结果：§6.1–6.2，pp. 11–13；重点核对了粗图算子、Theorem 4.7、Table 1/2。代码链接需继续查证（预印本文本未在已读取页面中确认仓库入口）。

**问题、输入输出与监督。** 输入为带节点属性图及少数带标签节点；输出硬分区 (P)、由原图聚合而成的加权粗图及训练后的GNN参数。每个粗节点都对应确定的原节点簇。训练在粗图上进行，最后把学到的参数放回原图预测，因此该方法把“参数迁移”与“对原图输出精确保持”区别开来，并没有声称两者等价。

**关键机制。** 先用 one-hot 分区矩阵 \hat P 聚合边：(A_c=\hat P^T A\hat P)，从而簇间边权是原边数/权重总和；节点度也相应求和。令 (C=mathrm{diag}(|B_1|,ldots,|B_k|))，正交化后的 (P=\hat P C^{-1/2})。论文叙述把 (X_c=P^TX) 称为超节点均值，但按其明确定义，这实际等于每个球内特征和除以 √|B|，即 √|B| 乘算术均值。粗图传播用球大小校正后的自环与度项（式(3)），不是不加说明地直接套用标准归一化邻接。对 APPNP，作者把传播写成低维约束二次优化；当粗化投影子空间逼近目标信号子空间时，低维传播解可逼近相应最优解。Theorem 4.7 的前提是目标子空间上的相对 Laplacian 投影误差小于 1，它保持的是图正则/能量结构相关的解，不是一般多层非线性节点分类的逐点输出保证。

**标签处理与协议。** 粗标签通过 (P^TY) 后取 argmax；球内标签混合时，实验选择从训练/验证集合删去混合球，作者明确说这一操作常能提高准确率。它是监督筛选，不是软标签保留，也与本项目只用合法训练标签和加权 soft CE 的目标不同。Cora/CiteSeer/PubMed使用每类20个训练、500验证、1000测试的公开 split；Coauthor Physics/DBLP随机按每类20训练、30验证、其余测试，并另测5-shot；结果报告20次均值和标准差。粗图训练后测试仍在原图上用原始节点特征和原始图结构完成。

**证据、成本与边界。** 作者报告十倍压缩下多数准确率仍有竞争力；不同 coarsener 中 variation-based 方法比谱聚类更快。Coarsening 单独作为预处理，作者报告 coarsening time；粗图训练降低传播/特征张量开销；原图上的最终预测仍需完整图推理。消融主要比较 coarsening ratio 和 coarsener，并非直接检验噪声标签下软标签统计，也没有逐层传播误差曲线。结论限于其数据和GCN/APPNP协议。

**对候选方案的具体含义。** 不应只写 `A_c=P^TAP, X_c=mean(X)` 而省略传播矩阵 S 的尺度：硬归属矩阵、列归一化矩阵、均值算子分别产生不同的边权、特征幅度和自环/度归一化。建议把“聚合与传播算子”作为独立模块及消融，明确球质量如何进入传播。训练后原图推理可以保留作最低可用方案，但要单独测 `P^T Φ_θ(A,X)` 与 `Φ_θ(A_c,X_c)` 的差距，且报告完整成本。作者实验支持粗图训练可减少训练成本；从“特定传播二次问题的子空间误差”推广到我们多层GNN的风险判断属于本报告推断。

### 2. Graph Reduction with Spectral and Cut Guarantees

**题录与来源。** Andreas Loukas. JMLR 20(116):1–42, 2019。正式期刊全文：[JMLR HTML/PDF](https://jmlr.org/papers/v20/18-680.html)，作者仓库：[graph-coarsening v1.1](https://github.com/loukasa/graph-coarsening/tree/v1.1)。全文42页。阅读：§2.1–2.4，pp. 3–8；§3（restricted spectral approximation及切/谱嵌入推论），pp. 10–14；§4.1–4.3算法及复杂度，pp. 14–19；§5实验，pp. 20–26；Appendix证明抽查。本文是理论/算法论文，不以GNN节点分类为验证目标。

**问题、输入输出与映射。** 输入无特征的加权无向图及需要保留的向量子空间 ℛ；输出由多个收缩层组成的硬分区粗图。每一层将原节点映射到唯一粗节点，并要求每个收缩集合诱导连通子图。作者将 (x_c=P x) 定义为簇内算术平均；提升算子把簇平均复制回每个成员。Laplacian-consistent coarse matrix 用该提升矩阵对原 Laplacian 作 Galerkin 型压缩，使常量零空间和粗图 Laplacian 形式保留。

**关键保证及其假设。** 定义 ((ℛ,ε))-similarity，要求对所有 (xinℛ)，原向量与其粗化再提升后的差在 L 半范数下不超过 ε 倍原 L 半范数。若目标子空间取低频 Laplacian 特征空间，可推出特征值/特征向量、谱聚类嵌入和相应图切的近似关系。这里的“保留”由选定的结构子空间约束；它不等同于任意属性、标签或下游分类边界保持。若目标信号不在所选子空间，或者层级压缩导致局部变化大，保证就不支持任务保持结论。作者在结论中也指出，给定图和目标压缩率下可达到多小误差、最优压缩问题的可解保证仍开放。

**方法与成本。** Local Variation 每步先为候选连通收缩集合计算局部变化代价，再按代价升序选择互不重叠集合，在达到节点数/误差阈值时输出粗图；可用 edge-based 或 neighborhood-based 候选集。多层误差上界累积为 ∏(1+σ_l)-1，局部变化代价解耦后可贪心降低其上界。边候选时论文给出近线性量级的理论复杂度（~O(ckM+ck³+k²N)一类项，变量由候选规模和目标子空间维数决定），精确搜索和候选族宽度会改变成本；多层压缩的成本取决于每层重新生成的图和候选，不能简化为只看最终球数。实验在 Yeast、Airfoil、Bunny、Minnesota 等图上评估谱误差；多种方法确定性只跑一次，Kron reduction重复10次。70%缩减时误差受图和子空间影响显著；因此“固定同球数就能有同样结构误差”不成立。

**对候选方案的具体含义。** 该文提供可测的结构保持基线和把压缩映射明确化的模板，不给标签感知选球标准。建议用原始 A/传播签名计算候选局部变异，作为 (g(B)) 的无监督支路或诊断项；不要把谱/切指标直接作为分类任务成功判据。对每个目标M，报告局部变化与任务指标的关系，并检查异配时低频平滑子空间是否与预测相关。作者定理支持受限谱/切性质；“这可以辅助约束粗图训练的传播漂移”是待验证迁移假设。

### 3. Featured Graph Coarsening with Similarity Guarantees (FGC)

**题录与来源。** Manoj Kumar, Anurag Sharma, Shashwat Saxena, Sandeep Kumar. ICML 2023, PMLR 202:17953–17975。官方全文：[PMLR PDF](https://proceedings.mlr.press/v202/kumar23a/kumar23a.pdf)，题录/作者页：[PMLR](https://proceedings.mlr.press/v202/kumar23a.html)。阅读：§2.2–2.3，pp. 2–3；§3.2–3.6及算法，pp. 3–6；§4实验，pp. 6–9；检查 Appendix A.6–A.8 对算法收敛和相似保证证明。代码入口在本轮浏览的正式论文/目录页未明确，暂记未知。

**问题、输入输出与监督。** 输入带节点属性的连通无向图；输出粗图 Laplacian、粗特征和到粗节点的 loading。理想定义中的分配矩阵 C 每行仅有一个非零项，对应硬原节点分区；实际算法优化的是满足非负/范数等约束的连续 loading 松弛，并用分组稀疏惩罚推动稀疏。作者在大 λ 时观察到列近似正交，不能把算法实现简化说成“每步都硬分配”。与纯凝缩方法不同，它仍有原节点 loading 映射；但 C 和粗图边权由优化学习，不是固定分区后机械求 (S^TAS)。

**关键目标与理论。** 目标由 −γ log det(̃L+J)（推动连通性）、粗图特征 Dirichlet energy、α/2||C̃X−X||²_F 特征重构惩罚、λ/2||Cᵀ||²_{1,2} 组成；随后用 ̃L=CᵀLC。作者交替做 C 的 block majorization-minimization 更新和 ̃X 的闭式凸子问题更新，给出收敛到 KKT 点的结果。它的 ε-similarity 定义比较原/粗图特征在各自 Laplacian 下的能量范数，并非任务损失、标签分布或GNN logits的保证。特征重构SSE只约束 (C̃X) 逼近 X；它不会自动保证任务信号被保留，特征维数/低秩表示、图的平滑假设或不合适的 α/λ 都可能造成偏移。

**实验协议与成本口径。** 节点分类时，先用 FGC 得分配，再从 (PY) 对超节点标签 argmax，粗图训练 GNN，最后用学得参数预测原图未标注节点；结果用10-fold交叉验证，比较GCond/SCAL，论文并未完整论证与常用固定train/val/test mask严格可比。报告的复杂度为每次主迭代最坏 O(p²k)，由 C 梯度矩阵乘法主导；作者在表5对比时将“coarsening + node classification”与原图分类时间并列，FGC在Cora/Citeseer/Pubmed/Co-CS的报告时间分别为1.66/2.22/30.4/34.4，基线表的训练/推理和计时硬件细节需要谨慎对待，不能将数字外推到本机或大规模图。分类表展示不同粗化比和多个数据集，但核心对照仍较少，且监督是argmax球标签。

**对候选方案的具体含义。** FGC说明“特征+图结构联合粗化、用可证明或可测的失真目标约束硬分区”有明确可探索空间，也提醒不要把特征 SSE 当任务保持。对我们的候选实现，先保留固定硬树+粗图训练，避免一开始引入昂贵的C连续优化；以节点级hold-out的传播/预测失真、带噪标签mask下的监督差异证据、与纯 feature-SSE 指标的分离作为机制评估。对粗标签优先用训练节点数加权 soft CE；在固定球预测 (p_B) 时，∑_{i∈B∩train}CE(y_i,p_B)=n_train(B)CE(q_B,p_B)，所以这只在共享预测下精确汇总节点损失，不能称自动去噪。这是我们根据交叉熵代数得到的推断，不是FGC或其他论文的抗噪实验结论。

### 4. Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training (ConvMatch)

**题录与来源。** Charles Dickens, Edward W. Huang, Aishwarya Reganti, Jiong Zhu, Karthik Subbian, Danai Koutra. WWW 2024 Companion Proceedings，pp.1502–1510；这是 companion/workshop 论文，不是 WWW 主会论文。作者预印本：[arXiv:2312.15520](https://arxiv.org/abs/2312.15520)；正式收录信息可由 NeurIPS 2024 论文参考文献[11](https://proceedings.neurips.cc/paper_files/paper/2024/file/d041d6bb47c01a4ce327a1773703e9a0-Paper-Conference.pdf)交叉核验；作者代码：[amazon-science/convolution-matching](https://github.com/amazon-science/convolution-matching)。细读作者版§3–5及Appendix B–G，含式(6)–(9)、Theorem 1、Algorithm 1、Table 2–5和复杂度分析；另只读仓库节点分类 coarsener、实验入口、聚合实现，未运行代码。

**对象、目标与监督。** 每个原节点唯一归属一个粗节点，采用硬分区矩阵 (P)。(A'=P^TAP) 聚合原始边质量，(C^{-1}P^TX) 是按成员数平均的粗特征；再以 (A'+C)、(D'+C) 修正 self-loop 和度归一化。方法不是仅匹配属性或图切，而是最小化原图一层 GCN 输出与“粗图输出提升回原节点”的差异，使用 (L_1) 传播响应损失。候选合并边由原图 SGC 表示的近邻构造，节点特征相似可使原图相距很远甚至不连通的节点进入同一球。聚合边上的合并代价可做到 (O(d_u+d_v))；A-ConvMatch 用上界近似以便批量合并。

**标签、评估与复杂度边界。** 论文覆盖六个图与两类任务：Cora/CiteSeer含节点分类和链接预测，OGBN-Arxiv/OGBN-Products做节点分类，OGBL-Collab/OGBL-Citation2做链接预测；以不同压缩比、粗化时间、训练时间/显存与任务表现评估，调参中使用验证集，因此端到端成本应把摘要、调参、粗图训练和原图验证分开看。作者代码显示：节点分类标签聚合前先把非训练标签置零；粗节点 train mask 是其成员 train mask 的 OR；多数标签只依据训练成员的计数选择。节点分类实验先在粗图训练，预测时使用原图和同一参数。代码可用性较强，但复现仍需匹配论文版本、数据预处理和硬件；本次未执行代码。

**限制与设计影响。** 它最直接支持“任务传播响应可作为粗化目标”，但目标限于作者的GCN层/候选合并策略；不能据此声称非线性深层GNN、标签噪声鲁棒或泛化无偏。相比候选 `X_c=mean(X)`，论文做了粗图 self-loop/度的质量修正，但最终的原图推理仍会重算传播，传播分布漂移未消失。建议首版诊断采用固定树，分别算 `P^TΦ(A,X)` 与 `Φ_c(A_c,X_c)` 的球级传播响应差和 lift-logit 差，再区分该误差由特征平均、粗邻接或归一化产生。代码中的标签聚合说明了如何守 train mask，但“多数票标签”可改为合法训练标签上的 `n_train`-加权 soft CE；后者等价于共享 lift 预测下的节点 CE 汇总，这是代数结论，不是 ConvMatch 的鲁棒性结果。

### 5. Graph Coarsening with Message-Passing Guarantees

**题录与来源。** Antonin Joly, Nicolas Keriven. NeurIPS 2024。正式版：[NeurIPS 论文与 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d041d6bb47c01a4ce327a1773703e9a0-Paper-Conference.pdf)，DOI 10.52202/079017-3648；[作者代码](https://gitlab.inria.fr/anjoly/mp-guarantees-graph-coarsening)。逐段读§2–3.3、§4、Theorem 1/2，重点 Eq.(4)–(11)、脚注3；抽读 Appendix A.1–A.3、B.2–B.6 的证明、训练流程与实验设置。

**算子与定理。** 令 `Q ∈ R^(n×N)` 是从原节点限制到粗节点的聚合矩阵，`Q+ ∈ R^(N×n)` 是其伪逆/提升。对硬球，`Q=C^(-1)P^T` 是球内平均，`Q+=P` 将粗表示复制回成员。给定原图传播矩阵 `S`，作者不是在 `P^TAP` 上再次调用通用 GCN normalization，而定义粗传播 `S_c^MP=Q S Q+`（Eq.6，p.5）。它一般非对称；对未归一化 `S=A` 和等大分区，其项是粗边权除以源粗节点大小。Theorem 1 将 restricted spectral approximation（RSA）转成一步传播误差界；它需要 S 与投影保持拉普拉斯核及目标子空间 R。

**训练损失与适用范围。** Eq.(9)（p.6）定义 `R_c(θ)=J(Q+ Φ_θ(X_c,S_c^MP))`，其中 `X_c=QX`：粗图 forward 后先 lift 回原节点，再在原图训练节点上计算 CE/MSE 并反传。脚注3明确：可以先聚合标签、在粗图直接算损失，但该路线不是本文研究对象。Theorem 2 控制粗图训练最优参数的 excess risk，受 RSA 常数、传播范数、网络深度、参数范围、损失 Lipschitz 条件、特征处于保留子空间等因素控制。关键边界是 Assumption 4 的激活必须 R-preserving、在 L-范数下 Lipschitz，并与提升算子可交换；作者明言目前可保证满足的例子只有 identity 激活/线性 SGC。不能把定理直接外推到 ReLU GCN，更不能据此宣称异配、多模态特征下任务保持。

**实验和成本。** 在合成随机几何图上比较六步传播误差；在 Cora/CiteSeer 及 Reddit 上测试 SGC/GCN。粗图传播比普通对称归一化等选择更稳定，SGC 下优势更明显；作者解释符合线性 SGC 理论，而 GCN 的 ReLU 未必满足假设。Cora/CiteSeer 每种设置10次随机训练。Eq.(9) 训练复杂度从约 (O(T(N+E)D)) 变成 (O(T(n+e)D+TN))，其中 (TN) 是为原训练节点 lift/算损失的成本。这个公式不是“零成本全图监督”。作者还指出其实验粗化算法需要大型矩阵逆和SVD，因而限制在中型图；这是 RSA/谱空间粗化流程的计算瓶颈，不能误写成构造 (QSQ^+) 对硬分区本身必需稠密求逆。硬球 (Q=C^{-1}P^T, Q^+=P) 可直接稀疏构造。

**对候选方案的具体含义。** 这篇最直接修正了候选粗图传播：若用平均特征，则用同一平均限制 (Q) 和提升 (Q^+) 定义 (S_c=Q S Q^+)，不要把它换成对 (A_c) 再次自动加自环/归一化的 `GCNConv`。节点级 lift 训练可用 (n_{train})-加权软标签 CE 精确汇总成粗球级损失，前提是每个成员共享同一 logits；此等式是本项目代数推导，不是该文定理，也不证明标签抗噪。对其余架构，最安全的首轮验证对象是线性 SGC/滤波模型；原图推理必须在原图 (S) 上用相同参数重算，另测与 lift 粗 logits 的漂移。该文保证不涉及标签质量、分裂树选择或粗化总成本；论文明确成本瓶颈甚至可能盖过训练节省。

### 6. Graph Condensation for Graph Neural Networks (GCond)

**题录与来源。** Wei Jin, Lingxiao Zhao, Shichang Zhang, Yozen Liu, Jiliang Tang, Neil Shah. ICLR 2022。正式版：[OpenReview PDF](https://openreview.net/pdf?id=WLEx3Jo4QaB)；作者预印本：[arXiv:2110.07580](https://arxiv.org/abs/2110.07580)；[作者代码](https://github.com/ChandlerBang/GCond)。细读§3.1–3.2、§4.1–4.4、Appendix A.2–A.3、B Algorithm 1、C.3复杂度；核对表1–4、表8–9。正文有作者正式版本标记，实验配置与代码入口也可查。

**对象与方法。** GCond 输出的是可学习的合成节点特征 (X')、合成结构 (A') 和预设类别标签 (Y')，合成节点不对应原节点成员，不是 (S^TAS) 意义上的粗球。原始双层目标要求在合成数据上训练的模型能替代原训练集上训练的模型；实际用梯度匹配近似训练轨迹匹配，按类别匹配原始/合成图上的参数梯度。(A') 通过共享的 MLP (Phi) 从 (X') 生成，(A'_{ij}=mathrm{sigmoid}((MLP([x_i';x_j'])+MLP([x_j';x_i']))/2))，再交替优化 (Phi) 和 (X')、阈值稀疏化；(Y') 固定而非同优化。GCond-x 还显示可以不显式合成图结构，把图信息蒸馏进合成特征。

**监督、迁移与实验证据。** 半监督 transductive 情形允许 (Y) 部分标记；公开 split 的训练标签及图特征/邻接用于凝缩，测试时模型用原始测试节点及其原图结构预测。inductive Flickr/Reddit 测试图不参与训练。论文在 Cora/CiteSeer/OGBN-Arxiv/Flickr/Reddit 与不同 condensation ratio 上比较；Cora 2.6%时 GCond 80.1±0.6，原数据 81.2±0.2；另测跨架构迁移、20次架构搜索相关性等。性能很强，但来自长程凝缩优化而非一次预处理，也依赖指定的condensation backbone/优化协议；与固定粗化树不能直接比较。

**成本与设计影响。** 梯度匹配每个外层步骤需在大图采样邻域上算梯度、在小合成图做反传，并反复采样模型初始化；论文复杂度仍线性依赖原图 (N)，还涉及合成图二阶梯度/近似。它验证了“合成数据可跨参数训练后迁移至原图推理”及“任务目标可比特征/谱目标更直接”，但不是低成本硬分区粗化的对照上限。对当前方案的具体影响是把标签生成和特征/边合并拆开：我们的硬球仍有可解释原节点映射、预处理可一次复用；若额外用标签优化树或做 gradient matching，须把访问的 train labels、原图反复前向/反向和验证选择成本显式核算，不能只报最后粗图训练时间。参数维度可迁移的一条底线是使用共享特征/滤波权重，不能让线性层直接依赖原始节点索引。

### 7. Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data (SFGC)

**题录与来源。** Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan. NeurIPS 2023 Main Conference Track。官方版：[NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/13183a224208671a6fc33ba1aa661ec4-Paper-Conference.pdf)；作者 HTML 预印本：[arXiv:2306.02664](https://arxiv.org/abs/2306.02664)；[作者代码](https://github.com/Amanda-Zheng/SFGC)。主读作者版§2.2–2.4、§3.1–3.2、§4、Algorithm 1及Appendix E.2–E.3；核对式(3)–(7)、Table 1–3、Table A4/A5。

**对象、目标和训练迁移。** SFGC生成合成特征 `X̃` 与合成标签 `Ỹ`，合成拓扑为单位阵；不会产生原节点归属，也不保留粗图边。它先在原图训练多条专家 GNN 轨迹，再将小型 graph-free 数据上的学生 GNN 长程参数变化与同初始化的专家轨迹区间匹配，更新合成特征。每步用 GNTK 闭式 ridge 评分 `γ_gnf` 衡量验证子图标签预测误差，并选得分最低的合成数据快照。也就是说，它把图结构知识隐式编码进特征，目标是压缩训练集替代性而非图传播保持。

**实验和成本口径。** 节点分类覆盖 transductive Cora/CiteSeer/OGBN-Arxiv 与 inductive Flickr/Reddit，采用公开划分及多个压缩比；还比较 MLP、GAT、APPNP、Cheby、GCN、SAGE、SGC 跨骨干。作者报告不同结构合成变体下 identity 版本表现更好，表中在 Cora 2.6% 下跨模型平均 80.3，OGBN-Arxiv 0.25% 平均 64.3。图结构拓扑被去掉不意味着凝缩免费：流程先训练 (K) 条大图专家轨迹、做多步 meta-matching，并反复计算 GNTK 评分；作者以核闭式解替代每个候选快照从头训练学生，且用验证子图减内存。作者未把这套前期凝缩开销变成硬球粗化的直接成本优势。

**限制与设计影响。** 这是与粒球粗化不同的合成凝缩路线。标签/验证信息可以在凝缩过程和快照选择中使用；若把它与当前受噪训练标签/干净验证集协议比较，必须标明监督用量。其优点是无须维护合成图边、能跨多种模型验证；代价是完全失去真实节点映射，生成成本和复用需求都高。对本方案的意义是：有原节点成员关系时，应把可追溯的 (Q/Q^+) 与合成压缩明确区分；若只要训练数据蒸馏性能，SFGC 是相关基线，但它不回答任务保持传播算子的设计问题。

## 跨文献阶段性综合

- “原节点分区超节点”与“优化合成数据点”必须分开编码。SCAL、Loukas、FGC、ConvMatch、Message-Passing Guarantees属于前者；GCond、SFGC属于后者。若首版选硬分区 `P∈{0,1}^{N×n}`，球质量矩阵为 `C=diag(n_b)`，则平均限制 `Q=C^(-1)P^T`，提升 `Q+=P`，二者不可混写为同一个“归属矩阵”。
- (A_c=P^TAP) 保留粗节点间总边质量，不意味着原图的标准对称度归一化传播与粗图标准传播相同。SCAL用成员规模校正；MP Guarantees 更直接用 (S_c=Q S Q^+)，明确先限制原传播再提升，不在 (A_c) 上二次自动归一化；ConvMatch匹配特定GCN传播响应。硬分区时计算 (Q,Q^+) 及稀疏 (QSQ^+) 可直接依边聚合，RSA选基/某些谱粗化才会有论文所述逆矩阵/SVD开销。
- 粗特征平均会造成球大小/类内多模态信息变化。SCAL的归一化和均值叙述存在一个必须显式处理的尺度差：(P^TX) 对其正交化 (P) 等于 √|B|×mean；Loukas及MP Guarantee的限制算子使用算术平均。我们的 (X_c=QX=mathrm{mean}(X)) 与 (S_c=Q S Q^+) 在映射上更自然一致，但要注意 (Q^+QX) 会把球内细节变成常值。原图推理仍吃原始逐节点 (X)，所以输入分布/输出漂移要在验证节点上量化。
- 下游保持指标至少要测 propagation/logit/loss 层面的目标，而不能仅测 Laplacian 相似、特征 SSE 或标签纯度。映射误差、粗图训练输入与原图推理输入漂移、训练损失是否严格按合法训练mask加权，是三个独立量。
- 效率须拆成预处理/构树、分区聚边、特征与标签聚合、粗图训练、每轮验证、最终原图推理。若预处理耗时 (T_c)，每轮全图训练 (T_f)，粗图训练 (T_g)，复用R次后才有净省时条件 (T_c+R T_g<R T_f)；逐轮原图验证时应把相应验证成本计入双方。这个 break-even 式是本报告的成本核算建议，不是前述论文的统一复杂度定理。

## 对当前提案的首轮建议

1. **保留**原节点二分树、固定节点预算M作为自适应资源预算，以及合法训练标签的节点数加权 softCE；软标签可减少人为 argmax 信息损失，且可与原节点监督作等价聚合对照。
2. **具体化** assignment S 至少有两种明确定义：(R) 为列正交的质量归一化矩阵，(Q) 为 one-hot 归属。粗特征均值算子应写 (D_B^{-1}Q^T X)；粗边总权写 (Q^TAQ)；不得直接把 (R^TAR) 和 (Q^TAQ) 混称为同一个“粗图邻接”。粗传播选择须由下文MP文献核对后确定。
3. **补入最低限度的任务保持诊断**：在不含训练标签的固定候选树上，比较 (Q S Q^+QX) 与 (QSX) 等限制—传播响应误差，并在相同参数下比较 (Q^+f_c(QX,QSQ^+)) 与 (f(X,S)) 的节点级 logits/损失差；优先用线性 SGC，因为 MP Guarantees 的激活保持假设只已知 identity 符合。此诊断不是新监督泄漏，也不许可查看测试标签调参。
4. **不把特征重构 SSE、谱误差或相同粗图精度之一单独当成功标准**。最小比较应包括普通无标签二分树、标签纯度树或原有机制、拟议的统计支持差异树，在相同候选坐标、球预算、GNN、mask和seed协议下报告机制值、准确率和总时间。
5. **暂缓复杂可微分软分配、凝缩轨迹匹配**。前者使硬分区归属、标签聚合和传播权重难审计；GCond/SFGC的轨迹凝缩会引入多次原图优化、核评估或二阶梯度成本。只有固定分区粗图对照证明传播/任务损失是主要瓶颈后，再追加一个有针对性的模块。

## 粗传播、损失与最小可验证骨干

### 传播公式的建议

首版明确 (P) 为 one-hot 硬成员矩阵，(C=P^TP)，(Q=C^{-1}P^T) 为球内均值限制，(Q^+=P) 为按成员复制。对原图定义一次完整的稀疏传播矩阵 (S)（包括是否加自环、归一化方式和方向），粗图传播固定为 (S_c=Q S Q^+)。粗邻接 (A_c=P^TAP) 仍可保存用于统计和可视化，但不能把 `GCNConv(A_c)` 的自动自环/度归一化当成 (S_c)；那是另一个传播算子，必须单列为基线。

这回答“(P_c=RPS) 是否匹配 meanX”的问题：若 (R=Q)、右侧映射为 (Q^+)，那么“限制—原图传播—提升到粗空间”的形式与平均 (X_c=QX) 是自洽的。更准确地说，粗传播的矩阵乘法右端作用在粗表示时，应写成 (S_c=Q S Q^+)，其中 (Q^+X_c) 将粗表示展开为原节点常值信号，再由原始传播 (S) 聚合，最后由 (Q) 做球内平均。它优先适配线性传播骨干，而非任意标准 GCNConv。

### 最小可验证骨干

用固定 (K) 阶线性滤波并保留自身通道：

\[
Z(X,S)=\big[X,\;SX,\;S^2X,\ldots,S^KX\big]W,
\qquad W\in\mathbb{R}^{(K+1)F\times C_{cls}}.
\]

粗图训练用 (X_c=QX) 和 (S_c=QSQ^+)：

\[
Z_c=\big[X_c,\;S_cX_c,\ldots,S_c^KX_c\big]W,
\]

然后可直接算粗节点损失，或将 (Q^+Z_c) 提升到原 train mask 上算 CE。最终原图推理用同一参数 (W) 和原始 (S,X) 计算 (Z(X,S))。(W) 的维数只依赖特征维数、滤波阶数和类别数，不依赖 (N)，因此可由粗图迁移到原图；它也天然避免 LINKX 一类若 adjacency-MLP 第一层直接接收 (N)-维邻接行时参数随节点数变化、无法跨 (n) 粗图迁移的问题。此骨干是最小实验建议，不是以上论文验证过的新方法。

自通道能让分类头在传播特征之外保留节点自身属性，尤其适合作为检验“平均特征造成的信息损失”的起点；但球内差异在粗图训练时已被均值压掉，故这种结构不能修复所有类内多模态漂移。第一轮只比较 (K=0/1/2)、带/不带 (X) 自身通道、(S_c=QSQ^+) 与普通粗图再归一化，报告粗/原 propagation feature 的相对误差、lift logits差异、训练mask损失和原图分类结果。SGC理论只支持线性/保留子空间假设；若在异配图上 (S) 的低频保留子空间不含有用特征，应将其作为边界，后续再考虑高通/双通道滤波并单独证明/验证，不能套用该定理。

### 标签与成本口径

每个球只汇总合法 train mask 内的标签计数 (n_{b,c})，软标签 (q_{b,c}=n_{b,c}/n_b^{train})，粗损失使用 (n_b^{train}\cdot CE(q_b,p_b))，零训练标签球不进入监督损失。若同球成员共享由 lift 得到的预测 (p_b)，则该目标严格等于该球原始训练节点的 CE 总和；这是等式，不代表抗噪或原图预测不变。与此不同，粗传播映射和特征平均改变模型输入与可表达输出，应分别测量。

完整效率至少列 (T_{tree}+T_{aggregate}+T_{coarseTrain}+T_{validation}+T_{fullInference})。若粗化只训练/调参一次，保守计入全部构图和原图验证；只有同一个粗图被复用 (R) 轮、且 (T_{pre}+R T_{coarse}<R T_{full}) 时才能宣称摊销训练收益。每轮原图 validation、早停及最后原图 inference 都不能被粗图训练计时掩盖。MP Guarantee给出了 lift loss 的 (TN) 项，SCAL、ConvMatch则显示最终原图推理和验证本来就需要完整图路径。

以上修改是由文献证据推出的设计建议；粗传播和损失选择现已按七篇主读收敛为首轮可验证规格，性能与节省仍需实验。
