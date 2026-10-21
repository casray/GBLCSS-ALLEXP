# 面向 noisy-label attributed graph 节点分类的 Graph-aware Granular Ball 研究设计

## 结论与总推荐

这条方向值得继续，但更适合走“双轨落地”。第一条轨道服务于你当前工程的最小验证，直接沿用现有 frozen-GRACE noisy-label gate，把当前 ordinary embedding-space GB 换成 graph-aware GB，先证明“球本身有没有独立价值，且是否优于 kmeans、topk、PPR-only、random/shuffled ball”。第二条轨道服务于对外可比实验，放到 NoisyGL 的统一 benchmark 里，使用 RTGNN 作为主宿主 baseline，系统对比 TSS、NRGNN、UnionNet、RNCGLN、CRGNN。这样做能同时继承你现有工程资产、吸收历史负结果边界，又获得统一噪声协议和强基线对照。fileciteturn0file2 fileciteturn0file8 fileciteturn0file10 citeturn8view0turn11view0turn35view0

我给出的最终推荐如下表所示。

| 决策项 | 最终推荐 | 主要理由 | 结论类型 |
|---|---|---|---|
| 对外统一实验框架 | **NoisyGL** | 它提供统一数据加载、噪声注入、Base Predictor API、HPO，并且当前公开仓库已内置 `nrgnn`、`unionnet`、`rtgnn`、`rncgln`、`crgnn`、`tss` 等实现，适合公平复现与增量开发。citeturn8view0turn11view0 | 【外部文献支持】 |
| 主宿主 baseline | **RTGNN** | 原论文有公开代码，NoisyGL 也有复现；机制上已经有 clean/noisy 划分、标签纠正、伪标签与一致性正则，最容易把球级软标签 `q_B` 和节点置信度 `r_i` 接成“软纠正 + 样本加权”。citeturn35view0turn37view0turn52view2turn11view0 | 【外部文献支持】加【设计判断】 |
| 次宿主 baseline | **UnionNet** | 如果你想做最少代码改动的 v1，UnionNet 的“标签聚合、样本重加权、标签纠正”与球级 `q_B` 的接口最贴近；NoisyGL 已提供公开实现。citeturn38view0turn8view0turn11view0 | 【外部文献支持】加【设计判断】 |
| 强竞争对手 | **TSS、NRGNN、UnionNet、RNCGLN、CRGNN** | TSS 代表拓扑 sample selection，NRGNN/RTGNN 代表 graph augmentation + pseudo-label，UnionNet 代表 aggregation/reweighting，RNCGLN 代表图噪声和标签噪声联合鲁棒，CRGNN 代表 contrastive robust GNN。citeturn29view0turn46view0turn34view0turn48view0turn30view0 | 【外部文献支持】 |
| 一周最小验证入口 | **保留你现有 frozen-GRACE gate** | 你当前工程已经具备 `ce/global/kmeans/topk_matched/gb` 及 coverage-aware 诊断，最适合快速加上 ordinary GB 与 graph-aware GB 的对照。文档也明确要求先做机制验证，再决定是否扩大主线。fileciteturn0file2 fileciteturn0file8 fileciteturn0file9 | 【项目文档支持】 |

有一条需要单独强调。NoisyGL 论文中的统一协议默认会污染训练集和验证集，只保留测试集干净；你当前项目的 noisy-label downstream 设计则是“只污染 train，val/test 保持干净”，而且核心入口是 frozen GRACE 表示空间中的后处理分类。两套协议都合理，但不能混在同一张表里直接比较，必须拆成 public benchmark track 和 project-aligned track。citeturn19view0 fileciteturn0file7 fileciteturn0file8

## 基线与实验框架选择

NoisyGL 应视为实验框架，不宜与方法基线并列。它在论文里给出统一 split、统一标签污染、统一 API 与多维分析；在当前公开仓库 README 里，方法清单已经扩展到 `rtgnn`、`nrgnn`、`unionnet`、`rncgln`、`crgnn`、`tss`，数据集扩展到 Cora、Citeseer、PubMed、Amazon-Computers、Amazon-Photos、DBLP、BlogCatalog、Flickr、Amazon-Ratings、Roman-Empire，噪声类型扩展到 `clean/pair/uniform/random/instance`。这意味着你可以把“Graph-aware GB”实现成一个新的 predictor，直接复用它的 split、noise、logging、HPO 与多方法对照。citeturn19view0turn8view0turn11view0

下面这张表给出我建议重点盯住的几类 baseline。

| 方法 | 核心机制 | 已确认公开代码 | 已确认支持的数据与噪声 | 已确认实验协议 | 角色建议 |
|---|---|---|---|---|---|
| **NoisyGL** | benchmark，统一数据、噪声与复现接口 | 官方 repo，且有 predictor 模板。citeturn8view0 | Cora/Citeseer/PubMed/Amazon-Computers/Amazon-Photos/DBLP/BlogCatalog/Flickr/Amazon-Ratings/Roman-Empire；`clean/pair/uniform/random/instance`。citeturn8view0 | 论文基准默认 noisy train+val, clean test；10 runs；仓库有 HPO。citeturn19view0turn8view0 | **框架** |
| **RTGNN** | noisy/clean label division，self-reinforcement，inter-view 与 intra-view consistency，伪标签 | 原论文给出官方代码，NoisyGL 有复现。citeturn37view0turn52view2turn11view0 | 原论文在 Cora、Citeseer、BlogCatalog 做 uniform/pair noise；NoisyGL 把它纳入统一多数据集多噪声接口。citeturn37view0turn8view0 | 原论文 5%/15%/80% split，训练集与验证集都加噪，5 runs。citeturn37view0 | **主宿主 baseline** |
| **TSS** | 基于 Class-conditional Betweenness Centrality 的 topological sample selection，逐步从 easy 到 hard 选择节点 | 本次检索中我确认到的是 NoisyGL 的公开实现。citeturn11view0 | 原论文使用 Cora、CiteSeer、PubMed、WikiCS、Facebook、Physics、DBLP，支持 symmetric、pairflip、instance-dependent noise。citeturn46view0 | 原论文使用 noisy validation，2-layer GCN，预训练加再训练，5 trials。citeturn46view0 | **强竞争对手** |
| **NRGNN** | 通过 link prediction/refined graph 把 unlabeled 节点接到相似 labeled 节点，再做 pseudo-label mining | 原论文与官方 repo都公开，NoisyGL 有复现。citeturn34view0turn51view0turn11view0 | 原论文使用 Cora、Citeseer、PubMed、DBLP，支持 uniform/pair noise；NoisyGL 统一扩展到更多数据与噪声。citeturn48view0turn48view3turn8view0 | 原论文在 Cora/Citeseer 采 5% label rate，PubMed/DBLP 采 1% label rate，污染 train+val，5 runs。citeturn48view0turn48view3 | **强竞争对手** |
| **UnionNet** | neighborhood/context label aggregation，sample reweighting，label correction | NoisyGL 有公开实现。citeturn11view0turn8view0 | 原论文强调多种噪声类型和水平；NoisyGL 统一支持多数据集多噪声。citeturn38view0turn8view0 | 在 NoisyGL 中走统一协议。citeturn19view0turn8view0 | **次宿主**加**强竞争对手** |
| **RNCGLN** | 联合处理图噪声与标签噪声，局部 graph contrastive learning，加全局注意力，伪图与伪标签联合 | NoisyGL 有公开实现。citeturn11view0turn30view0 | 在 NoisyGL 中支持统一多数据集多噪声。citeturn8view0turn11view0 | NoisyGL benchmark 显示其计算代价很高，PubMed 会 OOM。citeturn22view3 | **强但昂贵的竞争对手** |
| **CRGNN** | feature masking + edge dropping 的 contrastive robust GNN，动态 CE 与 cross-space consistency | NoisyGL 有公开实现。citeturn11view0turn30view0 | 在 NoisyGL 中支持统一多数据集多噪声。citeturn8view0turn11view0 | 统一协议下可直接比较，尤其适合作为“加入 GCL embedding”阶段的竞争对手。citeturn19view0turn30view0 | **强竞争对手** |

基于 NoisyGL 的统一结果和原始论文，我不建议把 RNCGLN 当主宿主。原因不在于它弱，原因在于它太重。NoisyGL 报告它在 Cora 上比 GCN 慢 66.8 倍，在 DBLP 上慢 2945.8 倍，而且在 PubMed 上会内存溢出；相对地，NRGNN 在性能和效率之间有更合理的折中。对你的 v1 方法来说，先把球的独立价值跑出来，比一开始就绑到最重的联合图噪声方法更稳。citeturn22view3

从“可嫁接性”看，RTGNN 与 UnionNet 最合适。RTGNN 的 clean/noisy candidate 划分、自强化和一致性正则，天然容纳 `r_i` 与 `q_B`；UnionNet 的 label aggregation、sample reweighting、label correction，天然容纳“球级分布替换邻域分布”。从“对外说服力”看，RTGNN 更适合当主宿主，因为它有原作者公开代码、NoisyGL 复现、原论文明确的强结果，而且其机制与“图标签噪声治理”高度贴合。citeturn37view0turn52view2turn38view0turn8view0

你现有工程的 baseline 价值也很高。当前 README 已经把 `ce`、`global`、`kmeans`、`topk_matched`、`gb`、`gb_prior`、`gb_ca_ce`、`gb_ca_global` 组织成 gate 实验，而且 success gate 已经要求 GB 变体必须同时压过 `ce/global/kmeans/topk_matched`，这正适合拿来做“一周最小验证”而不是立刻大范围扩展。fileciteturn0file2

## Graph-aware Granular Ball v1 方法设计

你当前代码里的 ordinary GB 已经非常清楚，它是在 embedding 空间里做递归二分，分裂依据是 KMeans 二分后 child mean radius 相对 parent mean radius 的下降，整个构球过程只看 `embeddings`，没有显式拓扑项，也没有预测一致性项。它是很好的 ordinary GB 基线，但它还不是 graph-aware GB。fileciteturn0file4

这正好对应你导师提出的核心空缺。上传文档已经把这个问题说得很清楚：普通粒球默认把样本当成独立点，只看特征或表示距离；图节点还有边、邻域、局部连通、路径与拓扑位置。文档里也已经给出了一个很好的方向，即图粒球至少要同时考虑 embedding 相似性、图拓扑邻近性和局部预测一致性，并把 noisy label 只当弱参考。fileciteturn0file3

从外部文献看，这个方向有明确支撑。SGBGC 把 graph granular-ball 定义成 connected subgraph，并且通过 purity threshold 做监督式 split，用于 graph coarsening；CGBC 则把 contrastive learning 与 granular-ball 层结合，用 coarse-grained 中心和球标签来缓解噪声标签，但它的样本并非图节点，粒球仍主要依赖 embedding 聚类。更新近的 GBGC 和 SCGNN 也都在强化一个共识，即在图上，granular-ball 更像 group-level structural or semantic unit，而不是单纯的欧氏球。你的工作空间就在这里：你要做的是“面向 noisy-label robust node classification 的 graph-aware granular ball”，既不走 SGBGC 的 purity split，也不走 CGBC 的非图实例式 embedding-only grouping。citeturn41academia2turn41academia0turn41academia1turn41academia3

### 正式定义

给定 attributed graph  
\[
G=(V,E,X),\quad |V|=n
\]
以及由 backbone encoder 得到的节点表示  
\[
H=[h_1,\dots,h_n]\in\mathbb{R}^{n\times d},
\]
和 stop-gradient 的教师预测分布  
\[
P=[p_1,\dots,p_n],\quad p_i\in\Delta^{C-1},
\]
定义一个 **Graph-aware Granular Ball** 为
\[
B=(V_B,E_B,c_B,\mu_B,\bar p_B,Q(B),q_B),
\]
其中 \(V_B\subseteq V\) 是球内节点集，\(E_B\) 是诱导边集，\(c_B\in V_B\) 是球中心节点，\(\mu_B\) 是球中心表示，\(\bar p_B\) 是球级平均预测分布，\(Q(B)\) 是球质量，\(q_B\) 是球级软标签。球的构造只依赖 \(H\)、图拓扑邻近性和预测一致性，不直接依赖 noisy label purity。noisy labels 只在球构好之后，用于生成 \(q_B\) 和 \(r_i\)。这是为了避免 label leakage 和把错误标签固化进构球过程。fileciteturn0file3 fileciteturn0file6 citeturn41academia2turn41academia0

记拓扑邻近矩阵 \(T\) 为 truncated personalized PageRank 或 top-M PPR 邻近度，归一化到 \([0,1]\)。Graph-aware membership 分数定义为
\[
s(i,B)=\lambda_h s_h(i,B)+\lambda_t s_t(i,B)+\lambda_p s_p(i,B),
\]
其中
\[
s_h(i,B)=\frac{1+\cos(h_i,\mu_B)}{2},
\]
\[
s_t(i,B)=T_{i,c_B},
\]
\[
s_p(i,B)=1-\frac{\mathrm{JS}(p_i,\bar p_B)}{\log C}.
\]
这里 \(s_h\) 衡量表示相似性，\(s_t\) 衡量拓扑邻近性，\(s_p\) 衡量预测一致性。Graph-aware GB v1 的构球准则是，在中心节点的 K-hop 或 top-M PPR 候选集内，贪心吸收满足 \(s(i,B)\ge \tau_{\text{add}}\) 且不会明显拉低球质量 \(Q(B)\) 的节点。fileciteturn0file3

### 与 ordinary GB、SGBGC、CGBC 的区别

| 形态 | 构球主信号 | 标签在构球中的角色 | 输出 | 对 noisy-label node classification 的适配度 |
|---|---|---|---|---|
| **ordinary GB** | 仅 embedding 距离与半径收益 | 无 | 球分组、中心、半径 | 可做 baseline，但忽略图拓扑和预测一致性。fileciteturn0file4 |
| **SGBGC** | 图连通子图，监督 purity threshold | 强监督 split 依据 | coarsened graph / supernodes | 对 coarsening 很合适，对 noisy label 场景有 leakage 风险。citeturn41academia2 |
| **CGBC** | contrastive representation + GB clustering | 球标签用于噪声纠正 | 粗粒度表示、球标签 | 对 noisy-label 有启发，但未显式建模图拓扑。citeturn41academia0 |
| **Graph-aware GB v1** | embedding + topology + prediction consistency + ball quality | 只在构球后弱参与 `q_B` | 球级软标签、节点置信度、样本权重 | 与你的任务最对齐。 |

### 球质量、球级软标签和节点置信度

球质量定义为
\[
Q(B)=g_{\text{size}}(|B|)\cdot\big(w_h\,\mathrm{Comp}(B)+w_t\,\mathrm{Topo}(B)+w_p\,\mathrm{Pred}(B)\big),
\]
其中
\[
\mathrm{Comp}(B)=\frac{1}{|B|}\sum_{i\in B}\frac{1+\cos(h_i,\mu_B)}{2},
\]
\[
\mathrm{Topo}(B)=\frac{1}{|B|-1}\sum_{i\in B\setminus\{c_B\}}T_{i,c_B},
\]
\[
\mathrm{Pred}(B)=1-\frac{1}{|B|\log C}\sum_{i\in B}\mathrm{JS}(p_i,\bar p_B),
\]
\[
g_{\text{size}}(|B|)=1-\exp\!\left(-\frac{|B|-1}{m_0}\right).
\]
这里 `Comp` 管紧凑性，`Topo` 管局部连通/拓扑邻近，`Pred` 管预测一致性，`g_size` 管小球惩罚。这样做的目的很明确：你不再只优化 compactness。fileciteturn0file3 fileciteturn0file6

对每个球 \(B\)，令 \(T_B\) 为球内有 noisy train label 的节点集合，令
\[
\hat y_B=\text{EmpDist}\{\tilde y_i:i\in T_B\}
\]
为 noisy label 分布，若 \(|T_B|<m_{\text{train}}\) 则使用全局 train prior \(\pi\) 做 Dirichlet fallback。再令
\[
\bar p_B=\frac{\sum_{i\in B}\omega_i p_i}{\sum_{i\in B}\omega_i},\qquad \omega_i=s(i,B).
\]
球级软标签定义为
\[
q_B=\mathrm{Norm}\big((1-\eta_B)\bar p_B+\eta_B\hat y_B\big),\qquad \eta_B=\lambda_y\cdot Q(B)\cdot \min(1,\frac{|T_B|}{m_{\text{train}}}).
\]
这意味着 noisy label 只占一个受 `Q(B)` 和 `|T_B|` 控制的小权重；标签越少、球质量越差，\(\eta_B\) 越小，球的软标签越接近预测共识而非 noisy majority。fileciteturn0file3

对节点 \(i\in B\)，定义节点置信度
\[
r_i = Q(B)\cdot\big(\alpha_h s_h(i,B)+\alpha_t s_t(i,B)+\alpha_p s_p(i,B)\big),
\]
并裁剪到 \([0,1]\)。最终节点训练目标写成
\[
\tilde q_i=(1-r_i)\,e_{\tilde y_i}+r_i\,q_B,
\]
对应样本权重可以取
\[
w_i=w_{\min}+(1-w_{\min})r_i.
\]
这样 `r_i` 同时服务两件事，一件是 soft correction interpolation，另一件是 per-node loss reweighting。用于噪声检测 AUROC 的节点怀疑分数建议定义为
\[
u_i = 1-r_i\cdot q_B[\tilde y_i].
\]
这个分数越高，表示“球不支持该 noisy label，且节点本身也不可靠”。这是一个比单纯 `1-r_i` 更有解释力的检测分数。这个定义目前是**研究假设**，但与 RTGNN/TSS 通过 early prediction、小损失或 confident subset 识别可靠样本的思路是一致的。citeturn35view0turn29view0

### Graph-aware GB v1 伪代码

```python
# Input:
#   G=(A, X), noisy train labels y_tilde
#   encoder f_theta, classifier g_phi
#   ball update period T_update
#   K-hop / top-M PPR neighborhood, lambda_h, lambda_t, lambda_p
# Output:
#   trained robust classifier with graph-aware ball targets

initialize host baseline  # RTGNN or UnionNet or current frozen-GRACE classifier
warm up for E_warm epochs

for epoch in range(E_total):

    H = stop_grad_or_forward_embeddings(f_theta, A, X)      # frozen SSL H or current hidden H
    P = stop_grad(EMA(g_phi(H)))                            # teacher predictions

    if epoch % T_update == 0:
        T = truncated_PPR(A, topM=M)                        # or K-hop normalized proximity
        unassigned = set(all_nodes)
        Balls = []

        while unassigned:
            c = argmax_i [ max(P[i]) * log(1 + degree(i)) ] over i in unassigned
            B = {c}
            mu_B = H[c]
            pbar_B = P[c]

            candidates = topological_candidates(c, T, K_hop=K, restrict=unassigned)
            candidates = sort_by_descending(
                lambda i: lambda_h * emb_sim(H[i], mu_B)
                        + lambda_t * topo_sim(T[i, c])
                        + lambda_p * pred_cons(P[i], pbar_B)
            )

            for i in candidates:
                B_new = B union {i}
                if score(i, B) >= tau_add and Q(B_new) >= Q(B) - delta_q:
                    B = B_new
                    mu_B = weighted_mean(H[B])
                    pbar_B = weighted_mean(P[B])

            if not connected_induced_subgraph(A, B):
                B = largest_connected_component(A, B)

            if oversized_or_low_quality(B):
                split_into_two_medoids_and_push_back(B)
            else:
                Balls.append(B)
                unassigned -= B

        for each ball B in Balls:
            compute mu_B, pbar_B, Q(B)
            compute noisy-label dist yhat_B using train nodes only with Dirichlet fallback
            q_B = normalize((1 - eta_B) * pbar_B + eta_B * yhat_B)

        for each train node i in assigned ball B(i):
            r_i = clip(Q(B(i)) * (
                    alpha_h * emb_sim(H[i], mu_B)
                  + alpha_t * topo_sim_to_center(i, B(i))
                  + alpha_p * pred_cons(P[i], pbar_B)
                ), 0, 1)
            q_i = (1 - r_i) * onehot(y_tilde[i]) + r_i * q_B(i)
            w_i = w_min + (1 - w_min) * r_i

    # host baseline update
    loss = sum_i w_i * CE(pred_i, q_i) + original_host_regularizers
    update model parameters

return model
```

### 如何接入 RTGNN 或 NoisyGL 训练流程

如果主宿主选 RTGNN，我建议只动四个接口。第一，在 warm-up 之后，每隔 \(T_{\text{update}}\) 个 epoch 用当前 hidden states 和 EMA predictions 构球；第二，把 noisy candidate nodes 的 self-reinforcement 目标从“硬自预测”换成 `q_i=(1-r_i)e_{\tilde y_i}+r_i q_B`；第三，把 RTGNN 对 noisy candidate 的 down-weight 因子改成 node-wise `w_i`；第四，把“是否把 unlabeled node 纳入 pseudo-label set”的阈值与 `Q(B)` 和 `r_i` 绑定。edge predictor、inter-view mimicry、intra-view regularization 全部保持原样。这样改是低侵入的，既保留 RTGNN 主体机制，又能把你的 Graph-aware GB 明确定位成 robust local prior。citeturn36view0turn37view0

如果走 NoisyGL，我建议新建 `GAGBRTGNN_Predictor` 或 `GAGBUnionNET_Predictor`，直接继承它的 `Base_predictor`，复用其 data loader、noise injection、config 和 logging。NoisyGL README 已经明确提供 generic Base predictor class 和 generic noise injection schemes，这样你的方法可以被写成“原 baseline + graph-aware ball module”的最小增量。citeturn8view0

如果要加 GCL embedding，也建议先用**冻结**的 GRACE/BGRL/GraphCL 表示，只把它用于构球空间，或再做一个小型 feature augmentation 版本。项目文档已经多次显示：GB 深度介入 view generation、same-ball hard positive、直接控制 augmentations 这些路线反复不稳；post-pretraining 的弱监督与 soft prior 更安全。fileciteturn0file0 fileciteturn0file6 fileciteturn0file10

## 实验设计与表格模板

你的实验设计最好分成 **project-aligned track** 和 **public benchmark track**。前者用于证明“graph-aware ball 是否优于 ordinary GB 和 generic local grouping”，后者用于证明“该设计在公开 benchmark 下是否仍有竞争力”。这两条线共享方法定义与控制组，但协议不能混表。fileciteturn0file7 fileciteturn0file8 citeturn19view0

### 推荐实验矩阵

| 轨道 | 数据集 | 噪声类型 | 噪声率 | seeds | 主要用途 |
|---|---|---|---|---|---|
| project-aligned | Cora, CiteSeer, PubMed | symmetric/uniform 必做，pair 第二优先 | 0.2, 0.4, 0.5 | 5 | 快速验证 ordinary GB 与 graph-aware GB 的独立贡献 |
| public benchmark | Cora, CiteSeer, PubMed, Amazon-Photo, DBLP | uniform, pair；第二阶段加 instance | 0.2, 0.4, 0.5 | 开发期 5，最终主表 10 | 与 RTGNN/TSS/NRGNN/UnionNet/RNCGLN/CRGNN 公平对比 |
| 后续扩展 | BlogCatalog, Flickr, Roman-Empire 或 Amazon-Ratings | uniform, pair, instance | 0.3, 0.5 | 5 | 验证在低同配或更复杂图上的稳健性 |

这样选有两个原因。第一，NoisyGL benchmark 指出 pair noise 更有破坏性，图结构传播会加剧噪声影响，因此 uniform 和 pair 是主噪声类型；仓库当前还支持 random 和 instance，可在第二阶段扩大。第二，项目文档已经说明只靠 Cora 单数据集、单 seed、局部提升不足以支撑主_claim。citeturn19view0turn8view0 fileciteturn0file10

### 必做对照与消融

| 组别 | 具体设置 | 作用 |
|---|---|---|
| host baseline | RTGNN 或 UnionNet 原版 | 主对照 |
| + ordinary GB | 用你当前 embedding-only `build_granular_balls`，其余全部不变 | 证明 graph-aware 的增益是否来自“图感知”而非“分个球” |
| + graph-aware GB | 仅加入 topology + prediction consistency + quality | 主方法 |
| + graph-aware GB + GCL | 构球空间换成 frozen GRACE/BGRL/GraphCL 表示 | 检验 SSL 表示是否进一步放大 graph-aware ball 价值 |
| kmeans matched | K 等于 ball count | 排除“只是做分组” |
| topk neighbor | k 等于 matched average ball size | 排除“只是做局部平滑” |
| PPR-only | \(\lambda_h=0,\lambda_p=0\) | 检验纯拓扑邻近是否足够 |
| ordinary GB only | \(\lambda_t=0,\lambda_p=0\) | 检验纯 embedding grouping |
| pred-only grouping | \(\lambda_h=0,\lambda_t=0\) | 检验纯预测一致性是否足够 |
| shuffled ball | 保持 ball size multiset，随机打乱成员 | 排除 size prior |
| random partition | 保持 ball count/size 匹配，随机分组 | 排除 generic grouping |
| no-Q(B) | 移除质量因子 | 检验 quality 的必要性 |
| q_B pred-only | \(\eta_B=0\) | 检验 noisy-label weak prior 是否有帮助 |
| q_B noisy-only | \(\bar p_B\) 去掉，只用 noisy label dist | 检验是否退回“球内多数即真值”的错误路线 |

这些控制组不是可选项，必须做。项目文档已经多次强调，no-ball、random、shuffled、kmeans、topk 控制缺失时，非常容易产生 overclaim；当前 frozen GRACE + GB local smoothing 的历史实验，也已经表明 coverage 拉满并不自动带来 GB-specific value，尤其 CiteSeer 上会被 kmeans 和 generic local smoothing 追平甚至反超。fileciteturn0file7 fileciteturn0file8 fileciteturn0file10

### 必报指标与定义

| 指标 | 定义建议 | 用途 |
|---|---|---|
| Accuracy mean/std | test accuracy over seeds | 主性能 |
| Clean purity | 用 clean labels 统计 weighted ball purity | 球是否捕获真实语义 |
| Label entropy | 球内 noisy-train label entropy，另报 `q_B` entropy | 球内标签混乱度 |
| Noise detection AUROC | suspicion score \(u_i = 1-r_i q_B[\tilde y_i]\) 对 noisy/clean train nodes 的 AUROC | 球是否识别噪声 |
| Noise correction rate | noisy train nodes 中，\(\arg\max \tilde q_i = y_i^*\) 的比例 | 真正修正了多少噪声 |
| Clean damage rate | clean train nodes 中，被改坏的比例 | 鲁棒方法副作用 |
| Effective coverage | 有效 `r_i > 0` 或 `w_i > w_min` 的 train node 比例 | 机制可解释性 |
| Ball diagnostics | ball count、avg size、singleton ratio、Q(B) 均值 | 方法行为诊断 |

其中 clean purity、noise detection AUROC、noise correction rate、clean damage rate 这四个指标非常关键，因为它们能回答“球到底在帮什么”。如果 accuracy 提升了，纯度与 AUROC 没动，往往说明增益来自 generic smoothing 或额外正则；如果 purity/AUROC 提升了，但 clean damage rate 很高，则说明方法在误伤 clean nodes。fileciteturn0file7 fileciteturn0file8 fileciteturn0file10

### 推荐主表模板

**主表一，公开 benchmark 性能表**

| Dataset | Noise | Rate | RTGNN | RTGNN+OrdGB | RTGNN+GraphAwareGB | RTGNN+GraphAwareGB+GCL | TSS | NRGNN | UnionNet | RNCGLN | CRGNN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

**主表二，机制诊断表**

| Dataset | Noise | Rate | Method | Clean Purity | Label Entropy | AUROC | Noise Correction Rate | Clean Damage Rate | Coverage | Test Acc |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|

**主表三，分组控制表**

| Dataset | Noise | Rate | Host | +kmeans | +topk | +PPR-only | +OrdGB | +GraphAwareGB | +random partition | +shuffled ball |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

**主表四，组件消融表**

| Variant | Remove Topology | Remove Pred Consistency | Remove Q(B) | q_B pred-only | q_B noisy-only | r_i only | q_B only | Acc | AUROC | NCR | CDR |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|

## 风险审查与不可重复的失败模式

项目文档里已经把几类高风险模式写得很清楚，这些边界应直接写进本轮实验设计的“禁止事项”。

| 旧失败模式 | 历史证据 | 本轮硬约束 |
|---|---|---|
| 把 same-ball membership 当 hard positive | 文档明确要求不要把 same-ball membership 直接当二值正样本标签，且多轮复盘都说明 hard/broad same-ball 语义过粗。fileciteturn0file6 fileciteturn0file10 | 同球关系只作为 soft prior、`q_B`、`r_i` 的来源，不进入 hard positive 或 hard negative 规则。 |
| 深度改 view generation | 多份历史总结都指出 GB 深度介入 augmentation/view generation 会破坏健康 GRACE，长期低于 baseline。fileciteturn0file0 fileciteturn0file10 | GCL 只用 frozen embedding，先不改 augmentations，不改 view sampler。 |
| 只优化 compactness | 上传文档已明确建议图粒球同时考虑 topology proximity、prediction consistency 和 quality；只做球内紧凑很容易退回普通聚类。fileciteturn0file3 | `Q(B)` 必须同时含 Comp、Topo、Pred 三项，且要有 matched kmeans/PPR-only/random 对照。 |
| 只看 Cora 单 seed | 跨方向复盘反复强调 Cora 单 seed 和小于 std 的提升极易 overclaim。fileciteturn0file6 fileciteturn0file10 | 最小验证也要 5 seeds；完整实验至少 Cora/CiteSeer/PubMed。 |
| coverage 提高就当成功 | coverage-aware GB 历史实验已经达到 full coverage，但 `alpha` 很小，且没有稳定压过 kmeans/topk，CiteSeer 仍不稳。fileciteturn0file8 | 不能只报 coverage，必须同时过 kmeans/topk/paired-delta/AUROC 四道门。 |
| 用 `gb_prior` 包装 strict GB 成功 | 历史文档明确提醒 strict GB 与 prior-corrected 版本不可混为一谈。fileciteturn0file9 fileciteturn0file10 | ordinary GB、Graph-aware GB、prior-corrected variants 分开报表，绝不共用 claim。 |

这里最关键的一条，是“当前项目的负结果并没有否定 post-pretraining GB 全部可能性，但已经很明确地否定了 GB 作为强语义真值、强 view controller、强硬关系规则的角色”。所以你这轮方法应把粒球定位成 **robust local structure prior**、**soft label generator** 和 **confidence signal**，这与项目文档当前研究偏好完全一致。fileciteturn0file6

## 论文表述与执行计划

### 论文创新点表述

下面这些表述适合写进论文引言与方法贡献里。我把它们按证据类型区分开了。

| 表述 | 建议写法 | 结论类型 |
|---|---|---|
| 任务缺口 | “Existing granular-ball methods for noisy-label learning mainly operate in embedding spaces, while graph node classification additionally requires topology-aware local grouping.” 这个缺口由 CGBC 的非图样本式粒球和 SGBGC/GBGC/SCGNN 的图粒球方向共同支撑。citeturn41academia0turn41academia2turn41academia1turn41academia3 | 【外部文献支持】 |
| 方法定义 | “We define a graph-aware granular ball for noisy-label node classification, where membership is jointly determined by embedding similarity, topology proximity, and prediction consistency, instead of noisy-label purity.” citeturn41academia2turn29view0turn35view0 | 【外部文献支持】加【设计判断】 |
| 方法输出 | “The proposed graph-aware balls generate ball-level soft labels and node confidences, which are injected into robust node classification as soft correction and sample weighting signals.” 这是你的方法主创新。 | 【研究假设】 |
| 实验主张 | “The value of graph-aware balls should be validated against matched local grouping controls including ordinary GB, kmeans, top-k neighbors, PPR-only, random partitions, and shuffled balls.” 这条来自你项目历史经验和当前 benchmark 认识。fileciteturn0file7 fileciteturn0file10 | 【项目文档支持】 |

我建议论文里的主 claim 先写得克制一些。一个稳妥版本是：

> We propose a graph-aware granular-ball mechanism for noisy-label node classification on attributed graphs. Unlike ordinary embedding-space granular balls, our method jointly models representation similarity, topology proximity, and prediction consistency to produce ball-level soft labels and node-level confidence scores. The method is plug-and-play with robust GNN training pipelines and is evaluated against matched local-grouping controls to isolate graph-aware ball specific value.

这个表述有几个优点。第一，它没有把“same-ball = semantic truth”写成结论。第二，它没有把“粒球必然优于所有聚类或邻域方法”写得过满。第三，它天然兼容你的一周最小验证版和后续完整 benchmark 版。这个写法也与项目文档要求的 claim boundary 一致。fileciteturn0file6 fileciteturn0file7

### 阶段化执行计划

| 阶段 | 时间 | 目标 | 具体执行 | Go / No-Go 标准 | 证据类型 |
|---|---|---|---|---|---|
| 最小验证 | 1 周 | 证明 graph-aware 构球优于 ordinary GB 或至少能显著提升机制指标 | 直接在你现有 frozen-GRACE gate 上实现 `GraphAwareGBV1`，只跑 Cora/CiteSeer，uniform 0.2/0.4，5 seeds；对照 `ce/global/kmeans/topk_matched/OrdGB/PPR-only/random/shuffled`；主报 purity、entropy、AUROC、NCR、CDR、acc。fileciteturn0file2 fileciteturn0file8 | 如果 GraphAwareGB 没法稳定优于 OrdGB 且 AUROC/NCR 无提升，这条主线暂停，先只保留为 negative diagnostic。 | 【项目文档支持】 |
| 完整小图实验 | 2 到 3 周 | 做出可写论文的完整 small-graph 版 | 把方法接到 NoisyGL 的 RTGNN predictor 上，跑 Cora/CiteSeer/PubMed，uniform 与 pair，0.2/0.4/0.5，5 seeds 开发、最终主表 10 seeds；同时补 UnionNet 次宿主与 TSS/NRGNN/UnionNet/RNCGLN/CRGNN 竞争对照。citeturn8view0turn35view0turn46view0turn48view0 | 如果 RTGNN+GraphAwareGB 在至少两个数据集、至少一种高噪声设置下稳定超过 host 与 OrdGB，且 clean damage rate 可控，就进入扩展阶段。 | 【外部文献支持】加【设计判断】 |
| 后续扩展 | 后续 | 补强泛化、SSL 与低同配场景 | 加 frozen GCL embedding 版本；扩到 Amazon-Photo、DBLP、BlogCatalog 或 Roman-Empire；补 instance-dependent noise；再做一个 UnionNet-host 的轻量版验证“方法是否依赖 RTGNN”。citeturn8view0turn41academia3 | 如果方法只能在 Cora/CiteSeer citation 图上成立，不能扩到更复杂图，则主 claim 降级为“在 homophilous citation graphs 上的 robust local prior”。 | 【研究假设】 |

最后给出一个最短的落地顺序，方便你马上开做。

先做 **现有 frozen-GRACE gate + GraphAwareGBV1**。  
只换构球与 `q_B/r_i`，不改 encoder，不改 views，不改 split。  
先证明它相对 `OrdGB/kmeans/topk/PPR-only/random/shuffled` 的独立价值。  
这个环节过关后，再把同一套 GraphAwareGB 接到 **RTGNN in NoisyGL**。  
再过关后，最后才加 **frozen GCL embedding**。  

这个顺序与项目已有经验是一致的，也与当前公开 benchmark 的可比性要求一致。fileciteturn0file0 fileciteturn0file8 fileciteturn0file10 citeturn8view0turn35view0