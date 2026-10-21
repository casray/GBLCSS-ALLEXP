# Literature and Research Map for GRACE + Granular-Ball Contrastive Learning

> 目的：这是给 AI 在设计下一轮实验前快速读取的研究知识库。它不是综述论文草稿，而是“实验方向决策文件”。
> 核心任务：在节点级图对比学习（以 GRACE 为中心）中，探索 Granular Ball Computing 是否能作为可靠的局部结构先验、软关系教师、置信度权重或增强控制器，帮助形成可发表的实验方向。
> 文件定位：本文件是 **AI-facing experiment decision layer**，用于快速决定下一轮实验。完整逐篇文献证据、更多 `Unknown/TODO` 和详细 Paper Notes 保留在 `01_literature_and_research_map.md`。当本文件中的判断需要追溯来源或写论文相关表述时，必须回查详细版或原始 PDF。

---

## 0. How AI Should Use This File

### 0.1 读取优先级

AI 每次辅助设计实验时，应按以下顺序读取：

1. `1. Core Thesis`：先确定哪些方向允许做、哪些方向禁止做。
2. `2. Decision Matrix`：直接选择下一轮实验方向。
3. `4. Paper Cards`：只在需要文献依据时查具体论文。
4. `5. Experiment Blueprints`：将 idea 转成可跑的实验。
5. `6. Claim Boundary and Failure Rules`：判断实验结果是否足以写论文。
6. `01_literature_and_research_map.md`：当需要逐篇证据、数据集细节、代码可复现性、公式或不确定信息时，回查详细版；不要只凭本优化版写论文 claim。

### 0.2 绝对规则

- 本项目默认是 **无监督 / 自监督节点级 GCL 预训练**，不得把标签、label purity、class boundary、supervised coarsening 直接放进 pretraining loss。
- `same-ball membership` 不能直接等价于正样本标签；只能作为 **soft relation / confidence / denoising prior**。
- 粒球不应首先替代 GRACE 的 evaluation protocol，也不应改变 train / validation / test split。
- 任何提升必须跨数据集、跨 seed、看 mean/std；小于标准差的单点提升不能支撑论文 claim。
- 如果某个文献是多视图聚类、监督分类、半监督 GCN 或图粗化，只能迁移其中的机制，不可原样搬到 GRACE。

### 0.3 Evidence Tags

- `Direct`：与节点级 GCL 或图表征学习接近，可直接指导主实验。
- `Transferable`：任务不同，但可迁移为 soft relation、confidence、denoising、augmentation control。
- `Diagnostic`：依赖标签或任务特定目标，只能用于离线分析、可视化或 ablation。
- `Warning`：提示某种迁移方式有风险，必须进入失败标准。

---

## 1. Core Thesis

### 1.1 当前最稳妥主线

最稳妥的论文主线不是“粒球提供新的硬标签”，而是：

> 粒球为节点级图对比学习提供鲁棒的局部结构先验，通过软关系教师、置信度加权和结构感知增强控制，减少随机增强带来的语义破坏和错误对比关系。

推荐主线可以表述为：

```text
GRACE baseline
+ weak-view granular-ball relation teacher
+ node/ball confidence weighting
+ boundary-distance local topology prior
+ augmentation control or denoising auxiliary loss
```

### 1.2 为什么不做 hard same-ball positive

文献共同提示：粒球位于 instance 与 cluster/prototype 之间，适合表达“局部关系倾向”，但不保证同球节点一定同语义。

风险包括：

- 球形假设不一定匹配图节点表示空间。
- 训练早期 embedding 波动会导致粒球 membership 不稳定。
- 同球内可能混入边界节点、噪声节点、异类近邻。
- 过强 center pulling 会把 contrastive learning 变成伪聚类，导致过平滑或 representation collapse。

因此，粒球关系应进入：

- soft relation matrix；
- node-to-ball confidence；
- edge/feature augmentation probability；
- denoising / uncertainty weighting；
- offline diagnostic and visualization。

不应直接进入：

- hard positive labels；
- label purity driven loss；
- class-wise boundary loss；
- supervised coarsened graph training。

---

## 2. Decision Matrix for Next Experiments

| Priority | Direction | Use GB As | Why Worth Trying | Main Risk | Decision |
|---|---|---|---|---|---|
| P0 | Stable GB diagnostic | stability probe | 先验证粒球在 raw/H/Z 空间是否稳定 | 不稳定则后续 loss 全无意义 | 必做 |
| P1 | Weak-view relation teacher | soft relation distribution | CLSA 支持 weak→strong distribution distillation；GRACE 需要增强一致性 | teacher 本身不可靠 | 优先主线 |
| P1 | Confidence-weighted soft same-ball relation | weight, not label | 利用球内距离、半径、密度、size、stability 降低错误关系影响 | 权重过强造成过平滑 | 优先主线 |
| P1 | Boundary-distance ball relation | ball-to-ball soft topology | GBCM-GCN/AW-GBGAE 支持用 GB 构图捕获 local/global | 图构造成本和噪声 | 优先主线 |
| P2 | Augmentation control | edge/feature drop guard | GRACE/CLSA/SPARROW 均提示增强可能破坏语义 | 控制太保守导致增强不足 | 可做 ablation |
| P2 | Granular denoising auxiliary loss | uncertain node down-weighting | GFSAF/GBG++ 支持 noise stripping/outlier idea | 仅降权收益可能弱 | 可做辅助 |
| P3 | Graph coarsening / sampling | efficiency module | SGBGC/PGBSF 支持效率与压缩 | 监督信息泄漏，改变节点级协议 | 暂缓主线 |
| P3 | Prototype center pulling | center/prototype | GB-RVFL/GBRIP/MOGB 支持中心摘要 | 强拉近风险高 | 只做诊断 |

---

## 3. Corrected High-Level Literature Map

### 3.1 图对比学习基础

- **GRACE**：节点级 GCL baseline。核心是生成两个 graph views，并最大化同一节点在两个视图中的表示一致性；view corruption 包括 edge removing 与 feature masking。启发：粒球若加入 GRACE，最好围绕“增强一致性、局部结构关系、负样本/增强噪声诊断”，不要替代 GRACE 的基本协议。
- **SPARROW / augmentation-free GCL**：说明 GraphDA 可能破坏图语义，也说明 encoder output 与 projector output 可以作为不同关系空间来诊断。启发：projector space 不应默认用于生成粒球，可作为诊断对照。
- **CLSA**：强增强直接进 contrastive loss 会退化；用 weak view 的相对分布监督 strong view 更安全。启发：GRACE + GB 最自然的版本是 weak-view ball-relation teacher，而不是 strong-view hard positives。

### 3.2 粒球 + 对比学习

- **MGBCC**：在 instance-level 与 cluster-level 之间引入 granular-ball-level contrastive learning，缓解 false negative 与局部结构缺失。启发：GB 可作为中间粒度关系单元；限制：多视图聚类，不是节点级图 GCL。
- **GFSAF**：用 granular-ball fuzzy contrastive learning 提取 mutual representation，并用 noise stripping loss 处理 complementary representation。启发：GB 更适合作为 fuzzy / robust / noise-aware relation，而不是硬分配。

### 3.3 粒球生成与质量控制

- **GBC classifiers**：将点输入替换为粒球输入，定义球标签与球间距离，强调效率、鲁棒性和大粒度优先。启发：GB 是表示单元，不只是 preprocessing。
- **GBG++**：强调 GB generation 的稳定性、效率、outlier detection 与 ball quality。启发：在 GCL loss 前，必须先诊断粒球稳定性、局部 outlier、size-aware quality。
- **Adaptive GB generation**：用 division 替代 k-means 加速，并考虑 overlap elimination 等因素，使 GB generation 更自适应。启发：不要手工固定 ball number；可以把 overlap/covering quality 转为 uncertainty。

### 3.4 粒球进入 GNN / GAE / 图构建

- **AW-GBGAE**：针对无边结构的高维无标签数据，用 weighted GB 进行 feature weighting 与 intra/inter-ball edge construction，再送入 GAE。启发：GB 可构造 local/global edge prior；限制：它是 general data clustering，不是已有图上的节点级 GCL。
- **GBCM-GCN**：用无监督 GB generation 与 boundary distance 构建多视图拓扑，同时捕获 local/global 信息。启发：boundary distance 是比 hard membership 更适合迁移到 GRACE 的 ball-to-ball soft relation。
- **SGBGC**：监督粒球图粗化，用 label similarity 与 purity threshold 生成 super nodes。启发：可做效率或诊断参考；限制：明显依赖标签，不可进入无监督 GRACE pretraining。
- **PGBSF**：用 GB sampling 生成递增 subgraphs，结合 parameter sharing 提高 GCN 训练效率。启发：可做效率实验；限制：半监督 node classification 与 purity threshold 设置，不适合作为无监督 GCL 主线。
- **GBExplainer**：用自适应粒度生成解释子图。启发：适合后验解释和可视化，证明 GB relation 是否对应重要结构；不适合直接设计 pretraining loss。

### 3.5 在本优化版中未展开为 Paper Card 的扩展文献

以下文献在详细版 `01_literature_and_research_map.md` 中有更完整记录，但在本优化版中只保留机制启发。AI 可以用它们辅助实验方向判断；如果要引用具体方法细节、数据集、公式或复现信息，必须回查详细版或原始 PDF。

- Ball k-Means：支持快速自适应聚类和粒度控制。
- Granular-ball Density Peaks：支持 density-aware relation / cluster center idea。
- GEC / Granular Ellipsoid：提醒球形假设可能不足，非球形结构可能需要椭球或局部形状诊断。
- GE-SMOTE / EvoSampling / GAdaBoost：支持噪声、边界、样本质量加权；不支持直接 negative filtering 作为唯一主线。
- GBRIP / MOGB / GB-RVFL：支持 center/radius/margin confidence；警示 prototype pulling 不可过强。
- Graph-based Representation for Image / GBGM：支持 structure-aware augmentation guard；迁移到图时应控制 edge drop 与 feature mask。
- MGNR：支持 multi-granularity neighbor relation；提醒固定 k 或固定 ball number 可能有 density bias。

---

## 4. Paper Cards

### 4.1 GRACE - Deep Graph Contrastive Representation Learning

**Evidence**: Direct.

**Setting**: unsupervised node-level graph representation learning.

**Mechanism**:

- Generate two corrupted graph views.
- Corruptions: edge removing + feature masking.
- Optimize node-level contrastive agreement across views.
- Intra-view negatives also matter and can regularize smoothing.

**What it teaches this project**:

- GRACE should remain baseline and evaluation center.
- GB should not replace the two-view GCL protocol.
- GB can guide which augmentations are safer and which relations should be trusted.
- Sparse/noisy features make topology-aware priors valuable.

**Do not copy wrongly**:

- Do not redefine positives as all same-ball nodes.
- Do not change public splits or use labels in pretraining.

**Experiment hook**:

```text
baseline_loss = GRACE(edge_drop, feature_mask)
add GB only as auxiliary relation / confidence / augmentation guard
```

---

### 4.2 SPARROW - Simplified GCL Without Augmentation

**Evidence**: Direct / Warning.

**Setting**: augmentation-free graph contrastive learning.

**Mechanism**:

- Treat encoder representation and projection-head representation as positive pair.
- Remove GraphDA and multi-branch architecture.
- Focus on alignment-uniformity balance.

**What it teaches this project**:

- Graph augmentations can corrupt semantics and add hyperparameter burden.
- Encoder output `H` and projector output `Z` should be separated in diagnostics.
- If GB in `Z` is unstable or overly loss-specific, generate balls in `H` or weak-view `H_w` instead.

**Do not copy wrongly**:

- Do not abandon GRACE baseline too early.
- Do not assume projector space is semantically better for GB generation.

---

### 4.3 CLSA - Contrastive Learning with Stronger Augmentations

**Evidence**: Transferable / Warning.

**Setting**: visual self-supervised contrastive learning.

**Mechanism**:

- Strong augmentation can expose useful patterns but direct contrastive use can degrade identity retrieval.
- Use weakly augmented view distribution as teacher for strongly augmented query.
- Optimize contrastive loss + distribution divergence minimization.

**What it teaches this project**:

- For GRACE, strong edge/feature corruption should not be blindly forced to match one-hot positives.
- Weak-view GB relation can supervise strong-view GB relation softly.
- This directly supports `weak_relation_teacher` rather than hard same-ball positives.

**Experiment hook**:

```text
R_w = ball_relation_distribution(H_weak)
R_s = ball_relation_distribution(H_strong)
loss_relation = KL(stopgrad(R_w), R_s)
loss = loss_GRACE + lambda_rel * loss_relation
```

---

### 4.4 MGBCC - Multi-view Granular-ball Contrastive Clustering

**Evidence**: Transferable.

**Setting**: multi-view clustering, not node-level GCL.

**Mechanism**:

- Instance-level contrastive learning may introduce false negatives.
- Cluster-level methods may be too coarse and miss local structure.
- Granular balls lie between instances and clusters.
- Establish intra-view and cross-view ball associations, then contrast associated balls in shared latent space.

**What it teaches this project**:

- GB is a good intermediate granularity.
- Ball-to-ball relation is safer than node-to-node hard relation.
- Relationship should be soft and confidence-weighted.

**Do not copy wrongly**:

- Multi-view samples are not the same as GRACE graph augmentations.
- Cross-view ball intersection rules may not directly apply to node augmentations.

---

### 4.5 GFSAF - Granular-ball Fuzzy Split and Attention Fusion

**Evidence**: Transferable.

**Setting**: multi-view clustering.

**Mechanism**:

- Representation degradation occurs when fusion over-relies on mutual information and loses complementary information.
- Granular-ball fuzzy contrastive learning extracts robust mutual representation.
- Noise stripping loss reduces noise influence for complementary representation.
- Cross-view attention fuses mutual/complementary representations.

**What it teaches this project**:

- GB relation should be fuzzy, not binary.
- Noise/low-quality regions should be down-weighted rather than hard removed.
- If using weak/strong views in GRACE, preserve both shared relation and view-specific useful variation.

**Do not copy wrongly**:

- Do not import multi-view attention as main novelty unless the graph setting justifies it.
- Do not claim GFSAF proves node-level graph benefits.

---

### 4.6 Granular Ball Computing Classifiers

**Evidence**: Transferable.

**Setting**: traditional supervised classification.

**Mechanism**:

- Introduces granular ball as computing unit.
- Defines ball label and distance between balls.
- Replaces point inputs with GBs to improve efficiency, scalability and robustness.

**What it teaches this project**:

- Ball center/radius/distance are basic reusable signals.
- Larger granularity can remove noise, but may lose fine information.
- In GCL, use GB to summarize local structure, not to invent labels.

**Reusable signals**:

```text
center c_b
radius r_b
ball_size |B|
node_distance_to_center d(i, c_b)
ball_distance d(B_i, B_j)
```

---

### 4.7 GBG++ - Fast and Stable GB Generation

**Evidence**: Transferable / Diagnostic.

**Setting**: supervised classification; GB generation.

**Mechanism**:

- Existing GBG is unstable/time-consuming due to k-means/k-division dependence.
- GBG++ uses attention/data-driven splitting.
- Includes local outlier detection and conflict handling.
- Considers GB quality, including sample size influence.

**What it teaches this project**:

- Before adding GB loss, evaluate GB stability across seeds/views/epochs.
- Local outliers should have smaller relation weights.
- GB quality should affect loss weight.

**Do not copy wrongly**:

- Supervised purity and label-based quality cannot enter unsupervised GCL.
- Use only label-free analogues: size, radius, overlap, embedding stability, density, boundary distance.

---

### 4.8 Adaptive GB Generation

**Evidence**: Transferable / Diagnostic.

**Setting**: supervised classification; adaptive GB generation.

**Mechanism**:

- Replaces k-means with division for efficiency.
- Considers overlap elimination and other factors.
- Aims for parameter-free/adaptive GB generation.

**What it teaches this project**:

- Fixed ball number is a weak default.
- Overlap ratio can become uncertainty.
- Adaptive granularity is promising, but online regeneration may be expensive.

**Label-free conversion**:

```text
overlap_uncertainty(B_i, B_j) = overlap_volume_or_membership_conflict
ball_quality = f(size, radius, density, stability, overlap)
```

---

### 4.9 AW-GBGAE

**Evidence**: Transferable.

**Setting**: high-dimensional unlabeled data clustering without predefined graph edges.

**Mechanism**:

- Feature weighting handles irrelevant features.
- Weighted GBs construct intra-ball and inter-ball edges.
- GAE learns embeddings from constructed weighted graph.
- Dynamic graph update adjusts neighbor relationships.

**What it teaches this project**:

- GB can build a graph prior when edge structure is missing or noisy.
- In GRACE, GB can provide an auxiliary soft topology rather than replacing original adjacency.
- Feature weighting suggests feature-mask probability can be GB-aware.

**Do not copy wrongly**:

- AW-GBGAE is clustering-oriented and graph-construction-oriented.
- It does not prove same-ball nodes are positives in graph GCL.

---

### 4.10 GBCM-GCN

**Evidence**: Transferable.

**Setting**: multi-view semi-supervised classification.

**Mechanism**:

- Uses unsupervised GB generation.
- Constructs topology using boundary distance between balls.
- Captures both local and global information.
- Collaborative matrix fuses local/global feature information across views.

**Most useful formula**:

```text
Dist(GB_i, GB_j) = ||c_i - c_j|| - (r_i + r_j)
```

Interpretation:

- Smaller boundary distance = more similar balls.
- Negative distance may indicate overlap or very close local structure.
- Positive but small distance can indicate global-neighbor relation.

**What it teaches this project**:

- Boundary distance is better than binary same-ball membership.
- Ball-to-ball topology prior can be converted into soft relation weights.

**Experiment hook**:

```text
S_ball(i,j) = exp(- boundary_dist(B_i, B_j) / tau_b)
R_node(u,v) = S_ball(ball(u), ball(v)) * conf(u) * conf(v)
```

---

### 4.11 SGBGC - Supervised Granular-Ball Graph Coarsening

**Evidence**: Diagnostic / Warning.

**Setting**: supervised graph coarsening for scalable GNN training.

**Mechanism**:

- Iteratively splits graph into GBs using label similarity / purity threshold.
- Uses GBs as super vertices.
- Reports scalability and robustness under noise injection.

**What it teaches this project**:

- GB can summarize graph structure and improve scalability.
- GB can be useful for noisy graph settings.

**Do not copy wrongly**:

- This is supervised and label-dependent.
- Coarsened-graph training changes the node-level GRACE protocol.
- Use only for diagnostic/efficiency ablation, not main unsupervised loss.

---

### 4.12 PGBSF - Progressive Granular Ball Sampling Fusion

**Evidence**: Diagnostic / Warning.

**Setting**: semi-supervised GCN node classification.

**Mechanism**:

- Divides graph into GB-based subgraphs.
- Trains progressively on increasing mini-batch graphs.
- Shares parameters between stages.
- Targets efficiency, convergence and memory.

**What it teaches this project**:

- GB can act as sampling / curriculum unit.
- Progressive training may reduce instability.

**Do not copy wrongly**:

- Sampling fusion is not contrastive relation learning.
- Purity threshold and label-rate experiments imply caution for unsupervised settings.

---

### 4.13 GBExplainer

**Evidence**: Diagnostic.

**Setting**: GNN explanation.

**Mechanism**:

- Uses adaptive granularity to avoid fixed subgraph size.
- Combines coarse structural-level and fine edge-level information.

**What it teaches this project**:

- After training, use GB to explain whether learned relation matches important graph structures.
- Can visualize failure cases: unstable balls, boundary nodes, noisy edges.

**Do not copy wrongly**:

- Explanation is post-hoc; it does not directly define a pretraining objective.

---

## 5. Experiment Blueprints

### 5.0 Baseline Before Any GB Loss

Always run and record:

```text
GRACE original
GRACE with same seeds + same splits
GRACE with weak augmentation only
GRACE with strong augmentation only
GRACE with tuned edge_drop/feature_mask
```

Record:

```text
dataset, seed, split, encoder, hidden_dim, proj_dim, tau,
edge_drop rates, feature_mask rates,
linear eval acc, mean/std, training time, memory
```

---

### 5.1 Experiment A - Stable GB Diagnostic

**Question**: Is GB membership/relation stable enough to enter training?

**Spaces to compare**:

```text
X_raw              # raw node feature
H_clean            # encoder output on original graph
H_weak             # encoder output on weak view
H_strong           # encoder output on strong view
Z_projector        # projector output, diagnostic only
```

**Metrics**:

```text
membership_NMI across seeds
membership_NMI across epochs
ball_count mean/std
avg_ball_size and size distribution
radius distribution
node_center_margin = r_b - d(node, center_b)
boundary_distance distribution
relation_matrix_correlation across views
```

**Pass condition**:

- Ball count and relation matrix stable across seeds/views.
- Boundary nodes are identifiable.
- Weak-view relations are more stable than strong-view relations.

**Fail condition**:

- Membership NMI is low across adjacent epochs.
- Ball count swings heavily.
- Most nodes are boundary/low-confidence.

**Action after failure**:

- Use raw feature or warmed-up encoder output instead of online embedding.
- Freeze GB generation for several epochs.
- Reduce GB loss weight or only use GB for augmentation guard.

---

### 5.2 Experiment B - Weak-view GB Relation Teacher

**Hypothesis**: Weak-view GB relation is a more reliable teacher for strong-view relation than hard node identity alone.

**Basic form**:

```text
H_w1, H_w2 = encoder(weak_view_1), encoder(weak_view_2)
H_s = encoder(strong_view)
R_w = relation_distribution(GB(H_w1), GB(H_w2))
R_s = relation_distribution(GB(H_s))
loss_rel = KL(stopgrad(R_w), R_s)
loss = loss_GRACE + lambda_rel * loss_rel
```

**Relation choices**:

```text
same_ball_soft: 1 if same ball else 0, then smooth by confidence
boundary_distance_soft: exp(-Dist(GB_i, GB_j)/tau_b)
center_similarity: cosine(c_i, c_j)
rank_relation: top-k related balls from weak view
```

**Ablations**:

```text
no relation teacher
hard same-ball teacher
soft same-ball teacher
boundary-distance teacher
teacher from raw X
teacher from H_weak
teacher from Z_projector
```

**Expected publishable claim**:

> A weak-view granular-ball relation teacher improves robustness to stronger graph augmentations by preserving local structural consistency without using labels.

---

### 5.3 Experiment C - Confidence-weighted Same-ball Soft Relation

**Hypothesis**: Same-ball relation is useful only for high-confidence interior nodes and should be down-weighted for boundary/outlier nodes.

**Node confidence options**:

```text
conf_center(i) = 1 - d(h_i, c_b) / (r_b + eps)
conf_size(b) = clip(log(|B_b|) / log(max_size), min_conf, 1)
conf_stability(i) = frequency_same_ball_across_views_or_epochs
conf_density(i) = local_density_normalized
conf_boundary(i) = sigmoid((r_b - d(h_i,c_b))/tau)
conf_node(i) = product_or_weighted_sum(conf_center, conf_stability, conf_size)
```

**Loss idea**:

```text
w_ij = conf_node(i) * conf_node(j) * soft_relation(i,j)
loss_aux = weighted_relation_loss(i,j,w_ij)
```

**Critical ablation**:

```text
uniform same-ball weight
center-distance confidence
stability confidence
size-aware confidence
all confidence combined
```

**Warning**:

If this improves only training loss but not linear eval, it may be over-smoothing.

---

### 5.4 Experiment D - Boundary-distance Ball-to-ball Local Structure Preservation

**Hypothesis**: Boundary distance between balls provides better local/global topology prior than binary membership.

**Ball relation**:

```text
D_bi_bj = ||c_i - c_j|| - (r_i + r_j)
S_bi_bj = exp(-max(D_bi_bj, 0) / tau_b)
```

**Node relation**:

```text
R_ij = S_ball(ball(i), ball(j)) * conf(i) * conf(j)
```

**Loss choices**:

```text
relation_distillation: KL(R_weak, R_strong)
neighbor_preservation: preserve top-k ball neighbors across views
soft_positive_weight: add R_ij as positive weight in contrastive denominator/numerator
```

**Ablations**:

```text
kNN relation in node embedding
same-ball relation
boundary-distance relation
random ball relation
raw graph adjacency relation
```

**Expected claim**:

> Boundary-distance GB relation provides a label-free, multi-granularity local topology prior for graph contrastive learning.

---

### 5.5 Experiment E - GB-guided Augmentation Control

**Hypothesis**: Random edge dropping and feature masking may destroy local semantics; GB confidence can control augmentation strength.

**Edge drop control**:

```text
if edge(u,v) connects high-confidence close balls:
    lower drop probability
elif edge(u,v) connects low-confidence/noisy/boundary nodes:
    higher or normal drop probability
else:
    default GRACE probability
```

**Feature mask control**:

```text
feature_importance = optional feature weight from GB or simple variance/MI proxy
mask_prob(node, feature) = base_mask * uncertainty(node) * inverse_feature_importance
```

**Ablations**:

```text
GRACE random augmentation
GB edge-drop guard only
GB feature-mask guard only
GB both edge and feature guard
confidence-free GB guard
```

**Warning**:

Too conservative augmentation can make contrastive task trivial. Track alignment/uniformity.

---

### 5.6 Experiment F - Denoising Auxiliary Loss

**Hypothesis**: GB can identify uncertain/boundary/outlier nodes and reduce their harmful contribution to relation loss.

**Methods**:

```text
low confidence nodes down-weighted in auxiliary relation loss
boundary nodes excluded only in ablation, not main method
outlier ball gets small lambda
```

**Do not make this the only novelty**:

Negative filtering or node removal alone is usually too weak as a paper contribution.

---

## 6. Claim Boundary and Failure Rules

### 6.1 Claims allowed if experiments succeed

Allowed:

```text
Granular balls provide a label-free local structure prior for node-level graph contrastive learning.
Soft GB relation distillation improves robustness to stronger graph augmentations.
Confidence-weighted GB relations reduce noisy or unstable contrastive constraints.
Boundary-distance ball relation is more reliable than hard same-ball membership.
```

Not allowed:

```text
Same-ball nodes are true positives.
Granular balls discover labels without supervision.
Supervised GB purity improves unsupervised pretraining.
Graph coarsening results prove node-level GCL representation improvement.
```

### 6.2 Failure patterns and interpretation

| Failure | Likely Meaning | Next Action |
|---|---|---|
| GB loss improves train loss but hurts eval | over-smoothing / prototype pulling | reduce lambda, use confidence, remove hard positives |
| Only one dataset improves | dataset-specific density artifact | add datasets and noise analysis |
| Strong-view teacher worse than baseline | strong augmentation destroys relation | use weak teacher / lower augmentation |
| Projector-space GB unstable | projector serves loss not semantics | use encoder H or raw X |
| Boundary nodes dominate | ball generation poor or representation noisy | warm-up, adaptive GB, confidence threshold |
| GB-guided augmentation no gain | augmentation already tuned or guard too weak | run stronger augmentation and noisy graph setting |

### 6.3 Minimum publishable experimental package

A credible paper direction should include:

```text
1. GRACE baseline reproduced.
2. At least 3 node classification datasets.
3. At least 10 or 20 seeds if feasible.
4. Mean/std and significance relative to std.
5. Ablation: no GB, hard same-ball, soft relation, confidence, boundary distance.
6. Stability diagnostic of GB generation.
7. Noise or strong augmentation scenario showing why GB helps.
8. Complexity / runtime overhead report.
9. Visualization: ball relation vs embedding / graph structure.
```

---

## 7. Recommended Execution Order

### Step 1 - Reproduce and freeze baseline

```text
Run GRACE with official-like hyperparameters.
Save embeddings H and Z every fixed number of epochs.
Record all seeds and splits.
```

### Step 2 - Offline GB diagnostic only

```text
Generate GBs on X_raw, H_clean, H_weak, H_strong, Z.
Measure stability and relation consistency.
Choose best space for GB generation.
```

### Step 3 - Add weak relation teacher

```text
Use weak-view GB relation as teacher.
Train strong view relation with KL/MSE soft distribution loss.
Start lambda very small.
```

### Step 4 - Add confidence weighting

```text
Compute node confidence from center distance, radius margin and stability.
Weight only auxiliary loss, not core GRACE loss.
```

### Step 5 - Add boundary-distance relation

```text
Replace same-ball soft relation with boundary-distance ball-to-ball relation.
Compare with kNN and adjacency relation.
```

### Step 6 - Add augmentation guard

```text
Use GB confidence to modulate edge drop and feature mask.
Test mainly under stronger augmentation/noisy graph settings.
```

---

## 8. Compact Prompt for Future AI Sessions

When starting a new experiment, give AI this prompt:

```text
Read 01_literature_and_research_map_ai_optimized.md.
Design the next GRACE + Granular-Ball experiment.
Constraints:
- no labels or purity in pretraining;
- no hard same-ball positives;
- preserve GRACE evaluation protocol;
- first check GB stability;
- prefer weak-view soft relation teacher, confidence weighting, and boundary-distance relation;
- include ablations, failure criteria, and expected claim boundary.
```

---

## 9. Final Takeaways

1. **GRACE 是中心，不是要被粒球替代。** 粒球应作为辅助局部结构信号进入增强、关系、置信度或去噪模块。
2. **CLSA 给出最强迁移逻辑：weak teacher + soft distribution。** 这比 hard positive 更适合处理强增强带来的语义扭曲。
3. **MGBCC/GFSAF 证明 GB 适合中间粒度和 fuzzy relation。** 但它们是多视图聚类，不是节点级 GCL。
4. **GBCM-GCN/AW-GBGAE 证明 GB 可构造 local/global topology。** 最值得迁移的是 boundary distance 与 intra/inter-ball relation。
5. **GBG++/Adaptive GB generation 提醒先做稳定性诊断。** 不稳定的粒球不能进入主 loss。
6. **SGBGC/PGBSF 的价值主要是效率和诊断。** 其监督/purity机制不能进入无监督 GRACE pretraining。
7. **最容易犯的错误是把同球当硬标签。** 本项目的关键词应是 soft、confidence、boundary、weak-teacher、denoising、augmentation-guard。
