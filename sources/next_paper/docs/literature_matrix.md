# 第二篇论文文献矩阵（2022–2026）

> 检索日期：2026-07-14
> 目标：判断“粒球 + 对比学习 + 局部原型/开放集/OOD/噪声标签”中哪些问题仍有论文空间，并为候选方向统一评分。
> 证据边界：以下方法描述只采用出版社论文页、正式会议论文页、arXiv/SSRN 原文页或论文明确给出的官方代码仓库。搜索结果、二手博客和本项目历史记录只用于发现线索，不作为最终事实来源。

## 1. 检索协议

### 1.1 研究问题

1. 2022–2026 年间，是否已有方法把粒球直接用于对比表示学习、局部多原型、开放集边界、异常/OOD 检测或噪声标签鲁棒训练？
2. 本项目已有的 `frozen representation + post-hoc granular-ball boundary`、`local soft target + small-loss` 等候选是否与现有论文构成实质重复？
3. 哪些非粒球强基线必须进入后续实验，才能证明收益来自粒球结构而非普通 kNN/KMeans/GMM/原型或训练技巧？

### 1.2 检索式与来源

- 主题组合：`granular-ball contrastive learning`、`granular-ball open intent/OOD/anomaly`、`granular-ball noisy label`、`multi-prototype open set`、`local prototype OOD`、`KNN contrastive open intent`、`graph neural network noisy label contrastive`。
- 一手来源：AAAI Proceedings、ACL Anthology、PMLR、NeurIPS Proceedings、IJCAI Proceedings、IEEE Xplore、ScienceDirect、arXiv、SSRN、论文作者公开仓库。
- 时间范围以 2022–2026 为主；ADB（2021）和 GRACE（2020）仅作为不可缺少的基础方法，不计入“近五年新颖性”证据。

### 1.3 纳入与排除

- 纳入：与至少一个核心维度直接相关——粒球、对比学习、局部/多原型、开放意图/OOD/异常检测、噪声标签、图节点学习。
- 排除：只有“ball”字样但并非 granular-ball computing；无原文元数据的转述；与候选机制没有可检验关系的泛 OOD/聚类论文。
- 代码状态：只有论文页或作者仓库明确给出链接才记为“有”；未找到不等于不存在，统一写“未核验到官方代码”。

## 2. 核心文献矩阵

| # | 标题 | 年份 / venue | 官方论文链接 | 官方代码 | 任务 / 模态 | 对比学习形式 | 局部区域表示 | 开放集机制 | 与粒球关系 | 可作为何种基线 | 未解决问题 | 对本项目的重复风险 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Multi-view Granular-ball Contrastive Clustering (MGBCC) | 2025 / AAAI | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34274) | [GitHub](https://github.com/Duo-laimi/mgbcc) | 多视图聚类 | 在 instance 与 cluster 之间增加 granular-ball-level contrast | 粒球及跨视图球关联 | 无 | 粒球是中间粒度对比单元，用于缓解 false negatives 和局部结构缺失 | “粒球确实进入对比目标”的最近直接先例 | 非图节点任务；球关系能否在无标签图表示中可靠估计仍未解 | **高**：任何“球级对比/跨球拉近”都需与其区分；硬迁移还与本地失败证据冲突 |
| 2 | Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary (MOGB) | 2025 / AAAI | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34630) | [GitHub](https://github.com/Liyanhuaa/MOGB) | 开放意图 / 文本 | 表示学习阶段含层级/子中心约束，不是纯后处理 | 每类多个自适应粒球、子中心和半径 | 多粒度球边界拒识未知意图 | 粒球直接表示复杂类内区域和决策边界 | 开放意图主基线；也定义了“真 GB vs 多中心”的最近任务先例 | 粒球价值是否独立于联合 representation learning；冻结表示时是否退化 | **极高**：本地 frozen-space Pilot 已显示 post-hoc GB 不优于 size-matched KMeans，方向应关闭 |
| 3 | Ellipsoid-Based Decision Boundaries for Open Intent Classification (EliDecide) | 2026 / AAAI | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40837) | 未核验到官方代码 | 开放意图 / 文本 | supervised contrastive learning | 每类可学习椭球矩阵，表达方向相关尺度 | 以已知覆盖和伪开放样本收缩构成 dual loss | 非粒球；是方向性边界的强替代 | 椭球/协方差边界强基线 | 多模态类内分布仍可能需要多区域；伪开放样本质量依赖训练 | **极高**：任何“椭球粒球 + 对比边界”都必须说明超越 EliDecide 的必要性 |
| 4 | Federated Open Intent Classification via Granular-Ball Knowledge Representation (FedOC-GB) | 2026 / Neural Networks | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608026002790) | [GitHub](https://github.com/jiezhang64/FedOC-GB) | 联邦开放意图 / 文本 | 结构化表示学习，不以经典 InfoNCE 为核心 | 客户端类内多粒球；服务器聚合全局粒球知识库 | 结构感知伪未知 + 全局多粒度边界 | 粒球同时承担知识压缩、多原型、边界 | 若做联邦扩展则为直接主基线 | 非联邦集中式环境是否需要粒球知识聚合没有回答 | **高**：多原型知识库、伪未知生成和多粒度边界均已有直接先例；只有联邦问题本身仍具清晰差异 |
| 5 | Beneficial Noise Learning for Open Intent Classification via Granular-Ball Representation (BNO-GB) | 2026 / Pattern Recognition | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320326002487) | [GitHub](https://github.com/Liyanhuaa/BNO_GB) | 噪声开放意图 / 文本 | 噪声引导的表示学习，含 prototype losses | 球级 outlier tendency + 样本级 inlier consistency | 区分 clean、IND noise、OOD noise，并利用 OOD noise | 粒球用于多级噪声识别和局部结构 | 噪声开放集、局部一致性、原型损失的直接主基线 | 对无噪声开放集、图节点或无标签构球的可迁移性不明 | **极高**：把粒球用于开放集噪声识别、伪标签修正或“有益噪声”已被占据 |
| 6 | Robust Smart Contract Vulnerability Detection via Contrastive Learning-Enhanced Granular-ball Training (CGBC) | 2026 / arXiv 预印本 | [arXiv](https://arxiv.org/abs/2603.27734) | 未核验到官方代码 | 噪声分类 / 智能合约文本代码 | 语义保持增强的无监督对比预训练 | GB center、majority label、intra/inter-GB losses | 无 | 对比表示先行，粒球作为粗粒度鲁棒训练层 | “contrastive warm-up → GB noisy-label training”机制基线 | 强依赖监督 purity/majority label；跨模态和正式同行评审仍缺失 | **高**：弱原型/紧致性虽可差异化，但整体训练顺序与主张已相近 |
| 7 | GBmooer: Granular-ball Text Label Smoothing with Hypersphere Contrastive Learning | 2026 / SSRN 预印本 | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6022343) | 未核验到官方代码 | 噪声文本分类 | hypersphere angular contrastive learning | 球中心、角半径、球内标签分布 | 无 | 对比表示后用粒球局部标签分布做 smoothing | 角空间/局部平滑/粒球置信度基线 | 尚未正式同行评审；无图结构；官方代码未确认 | **极高**：`LCSS soft target` 的“局部球分布平滑”叙事与其高度接近 |
| 8 | G-GBC: A Gaussian Mixture-Based Granular Ball Computing Method for Robust CNN Classification under Noisy Labels | 2026 / Knowledge-Based Systems | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0950705126004855) | 未核验到官方代码 | 噪声图像分类 | 无显式对比学习 | GMM、Mahalanobis、soft assignment、动态 purity 权重 | 无 | 用概率混合改造粒球并嵌入 CNN | GMM/椭球/软分配粒球基线 | 是否真优于同成本 GMM soft target；标签 purity 的泄漏/偏差 | **高**：任何 Gaussian/ellipsoid ball 与 noisy-label robust training 都需正面对照 |
| 9 | MgCNL: A Sample Separation Approach via Multi-Granularity Balls for Fault Diagnosis with the Interference of Noisy Labels | 2024 / IEEE T-ASE | [IEEE Xplore](https://ieeexplore.ieee.org/document/10706599) | 未核验到官方代码 | 噪声标签 / 工业故障时序 | 每轮含对比/分布分离步骤，核心是高置信样本筛选 | 多粒度球与标签置信评价 | 无 | 粒球用于未知噪声率下样本分离 | 粒球 sample-selection 的直接基线 | 领域特化；与普通 small-loss/JSD 选择相比的结构增益需严格隔离 | **极高**：`GB + small-loss/sample selection` 已有非常直接先例 |
| 10 | Like Draws to Like: A Multi-Granularity Ball-Intra Fusion Approach for Fault Diagnosis Models to Resists Misleading by Noisy Labels (MgBIF) | 2024 / Advanced Engineering Informatics | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1474034624000739) | 未核验到官方代码 | 噪声标签 / 工业故障时序 | 无显式对比学习 | 以样本为中心的多粒度邻域球、球内标签融合 | 无 | 球内伪标签校正 + 高置信样本选择，噪声率未知 | local correction + selection 直接基线 | 局部同类假设在复杂/异配结构上会失败；计算成本与同成本 kNN 对照不足 | **极高**：与 `LCSS + soft-small-loss` 的功能分解几乎同构，虽模态不同也会削弱新颖性 |
| 11 | Identifying Outliers via Local Granular-Ball Density (GBDO) | 2025 / IEEE TNNLS | [IEEE Xplore](https://ieeexplore.ieee.org/document/11073180/) | [GitHub](https://github.com/Mxeron/GBDO) | 无监督离群检测 / 表格 | 无 | k-similarity 粒球邻域、fuzzy relation、局部可达相似密度 | 局部粒球 outlier factor | 粒球作为多粒度密度计算单元 | GB-local-density 强基线 | 高维深表示、类别条件 OOD、图节点场景的可靠性仍未知 | **高**：任何 local GB density/OOD score 都必须与 GBDO 和普通 LOF/kNN 密度比较 |
| 12 | Finding Time Series Anomalies Using Granular-Ball Vector Data Description (GBOC) | 2026 / AAAI | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39722) | 未核验到官方代码 | 时序异常检测 | 无经典对比学习；训练做中心对齐 | 密度引导层次粒球；球中心是局部 normal prototypes | 最近球中心距离为 anomaly score | 粒球承担正常模式压缩与一类边界 | 多局部 normal prototype / 异常检测基线 | 强 center pulling 是否损害细粒度结构；能否扩展到语义 OOD | **高**：memory bank/local prototype/anomaly 叙事已被直接覆盖 |
| 13 | Effective Open Intent Classification with K-center Contrastive Learning and Adjustable Decision Boundary (CLAB) | 2023 / AAAI | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26560) | [GitHub](https://github.com/lxk00/CLAP) | 开放意图 / 文本 | K-center contrastive learning | 每类 K 个中心；自适应球形边界 | ADBES 扩缩每类边界 | 非粒球，但在功能上是最关键的多中心替代 | 必须进入所有开放意图粒球实验 | K 的选择、中心与边界的联合稳定性 | **极高**：如果 GB 不能胜过 size-matched K-center，其结构贡献不可成立 |
| 14 | KNN-Contrastive Learning for Out-of-Domain Intent Classification (KNNCL) | 2022 / ACL | [ACL Anthology](https://aclanthology.org/2022.acl-long.352/) | 论文页未列出可核验官方代码 | OOD intent / 文本 | 以 IND 样本 kNN 构造局部对比关系 | 非参数局部邻域 | density-based novelty detection | 非粒球；最直接局部邻域替代 | 局部对比 + 密度拒识强基线 | kNN 邻域受表示质量和密度不均衡影响 | **高**：任何“局部球关系提升开放检测”都必须证明优于 kNNCL |
| 15 | Out-of-Distribution Detection with Deep Nearest Neighbors | 2022 / ICML | [PMLR](https://proceedings.mlr.press/v162/sun22d.html) | [GitHub](https://github.com/deeplearning-wisc/knn-ood) | 通用 OOD / 图像 | 使用预训练深表示，不修改训练目标 | 样本级 kNN 距离 | 非参数距离阈值 | 非粒球 | frozen-space Pilot 的首要强对照 | 存储/检索成本高；不提供结构压缩和解释 | **高**：粒球若只压缩 kNN 且不改善性能，贡献只能是效率而非检测质量 |
| 16 | Investigating Why Contrastive Learning Benefits Robustness against Label Noise | 2022 / ICML | [PMLR](https://proceedings.mlr.press/v162/xue22a.html) | 论文页未列出可核验官方代码 | 噪声标签 / 图像 | 自监督对比预训练后 robust fine-tuning | 子类相关低秩表示结构（非显式局部球） | 无 | 非粒球；解释对比表示为何抗噪 | 机制/理论背景和 contrastive-pretrain baseline | 理论条件与图节点/局部结构不完全一致 | **中**：说明“先对比后抗噪”本身不是粒球新意，必须证明 GB 的额外结构作用 |
| 17 | NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise | 2024 / NeurIPS Datasets & Benchmarks | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html) | [GitHub](https://github.com/eaglelab-zju/NoisyGL) | 图节点噪声标签 | 汇总多类方法；本身是统一 benchmark | 取决于具体 predictor | 无 | 无直接粒球模块 | 本项目第一/第二篇 noisy-graph 实验协议底座 | 统一协议不自动保证新方法公平，需要同 seed、split、encoder 与调参预算 | **中**：不抢方法新意，但强制更高的实验可比性标准 |
| 18 | Leveraging Peer-Informed Label Consistency for Robust GNNs with Noisy Labels (ProCon) | 2025 / IJCAI | [IJCAI](https://www.ijcai.org/proceedings/2025/623) | 未核验到官方代码 | 图节点噪声标签 | 非经典 InfoNCE；语义 peer consistency | feature similarity + adjacency peers；prototype feedback | 无 | 非粒球，但功能覆盖局部一致性/原型/GMM clean split | `LCSS + selection` 的最近图任务强基线 | 迭代 prototype feedback 成本和错误累积；官方代码状态未确认 | **极高**：第二篇若仍是图噪声局部一致性，必须超越而非只避开 ProCon |
| 19 | Rethinking Graph Contrastive Learning Through Relative Similarity Preservation (RELGCL) | 2025 / IJCAI | [IJCAI](https://www.ijcai.org/proceedings/2025/358) | 未核验到官方代码 | 图表示学习 | pairwise/listwise relative similarity preservation | 按结构距离组织相对关系，而非固定簇/原型 | 无 | 非粒球；提供更自然的软关系建模 | weak relation teacher 的强替代 | 关系来自图结构距离，未解决标签噪声或开放集 | **高**：任何“soft local relation teacher”必须说明粒球比相对结构关系多提供什么 |
| 20 | Deep Open Intent Classification with Adaptive Decision Boundary (ADB) | 2021 / AAAI（基础项） | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17690) | 论文页未列出可核验官方代码 | 开放意图 / 文本 | 先监督训练表示 | 每类单中心、单半径 | 自适应球形边界 | 非粒球；MOGB/CLAB/EliDecide 的直接祖先 | 必须保留的基础边界基线 | 单球假设过强，难覆盖多模态/方向异性 | **低（基础）**：仅超过 ADB 不足以支持 2026 年新颖性 |
| 21 | Deep Graph Contrastive Representation Learning (GRACE) | 2020 / ICML Workshop/正式版本（基础项） | [arXiv](https://arxiv.org/abs/2006.04131) | [GitHub](https://github.com/CRIPAC-DIG/GRACE) | 节点级图表示 | 双增强视图、节点级 InfoNCE | 无显式局部区域 | 无 | 无粒球；本项目冻结表示来源 | 表示学习基线与固定 encoder | 增强可能破坏语义；不处理噪声标签/开放集 | **低（基础）**：本项目方法必须保留其评估协议，不能把 encoder 改动混成 GB 收益 |
| 22 | Breaking Down OOD Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities | 2022 / ICML | [PMLR](https://proceedings.mlr.press/v162/bitterwolf22a.html) | 论文页未列出可核验官方代码 | 通用 OOD / 图像 | 分析多类 OOD-training objectives | 无特定局部原型 | 分解 density ratio / class probability 等核心量 | 非粒球 | 解释不同 OOD loss 可能等价的审计基线 | 未覆盖粒球结构和文本开放意图 | **中**：提醒“新 score 组合”可能只是已有核心量重参数化，需用条件增益证明独立信息 |
| 23 | Towards Total Recall in Industrial Anomaly Detection (PatchCore) | 2022 / CVPR | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | [GitHub](https://github.com/amazon-research/patchcore-inspection) | 工业异常检测 / 图像 patch | 无 | 代表性 nominal patch-feature memory bank；greedy coreset subsampling | 最近 memory feature 的异常距离 | 非粒球；直接覆盖“压缩 memory bank 以降低存储和推理成本” | memory-efficient nearest-neighbor anomaly/OOD 的核心强基线 | 工业 patch anomaly 与文本 intent OOD 的语义不同，但问题和评测轴高度相似 | **极高**：仅把 greedy coreset 换成 GB coreset 不足以形成主创新 |
| 24 | Far Away in the Deep Space: Dense Nearest-Neighbor-Based Out-of-Distribution Detection (cDNP) | 2023 / ICCV Workshops | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023W/UnCV/papers/Galesso_Far_Away_in_the_Deep_Space_Dense_Nearest-Neighbor-Based_Out-of-Distribution_Detection_ICCVW_2023_paper.pdf) | 论文页未核验到官方仓库 | dense OOD / 道路语义分割 | 无 | ID feature bank；实验比较 random subsampling、greedy coreset、per-class greedy coreset | kNN density/距离与 parametric score 结合 | 非粒球 | 相同预算下 coreset selection 的直接强基线 | 模态为 dense segmentation；未研究自适应粒球 | **极高**：class-conditional coreset + NN-OOD 的实验问题已被直接研究 |
| 25 | SupLID: Geometrical Guidance for OOD Detection in Semantic Segmentation | 2025 / arXiv 预印本 | [arXiv](https://arxiv.org/abs/2511.18816) | [GitHub](https://github.com/hdnugit/SupLID) | dense OOD / 语义分割 | 无 | Linear Intrinsic Dimensionality 引导的几何 coreset；superpixel 表示 | 几何信号引导 classifier OOD score，强调实时推理 | 非粒球 | geometrical coreset + post-hoc OOD + 效率的最近先例 | 尚是预印本；文本/图任务可迁移性未明 | **极高**：几何 coreset 提升 OOD 质量–效率的主张已出现 |

## 3. 证据综合

### 3.1 已被高度占据的机制空间

1. **冻结表示上的 post-hoc 多粒球开放边界**：MOGB 已覆盖自适应多粒度粒球边界，CLAB 已覆盖 K-center contrastive + adjustable boundary，EliDecide 已覆盖 supervised contrastive + directional ellipsoid。FedOC-GB 又覆盖多粒球知识库和多粒度推理。仅把球形换成椭球、把单中心换成多中心，已不足以构成新方法。
2. **粒球局部标签融合 + 高置信样本选择**：MgBIF、MgCNL 已把球内标签融合、伪标签修正、未知噪声率下样本筛选作为核心；GBmooer 把 hypersphere contrastive + 局部粒球标签分布 smoothing 串联；BNO-GB 又把多级局部一致性用于 IND/OOD noise 分型。`LCSS + soft-small-loss` 虽有强本地效果，但论文新颖性风险极高。
3. **粒球局部原型/异常分数**：GBDO 已给出局部粒球密度 outlier factor，GBOC 已将粒球中心作为 normal prototypes 并以最近中心距离评分。普通 kNN OOD 又是冻结表示下的强基线。单纯建立“粒球 memory bank”或“最近球距离”很难形成独立贡献。
4. **粒球直接进入对比目标**：MGBCC 是明确先例；本项目多轮 same-ball positive、prototype pulling 和 direct consistency 又已有负结果。因此此方向既有文献重复又有本地机制风险。
5. **用 coreset 压缩 kNN/OOD memory bank**：PatchCore 已把 greedy coreset 作为 nominal memory bank 的关键组件，cDNP 已直接比较 random/GCS/per-class GCS，SupLID 又提出几何 coreset 与实时 dense OOD。将“真粒球不更准”改写成“GB coreset 更省”是合理工程问题，但已不是清晰的论文空白。

### 3.2 仍可能存在、但必须满足严格条件的空白

1. **条件增益而非替代打分**：不再问“GB score 是否高”，而问在 kNN/KMeans/GMM/energy/原型分数已经给定后，某个标签无关的粒球稳定性或边界不确定性是否仍提供可重复的条件信息。必须使用配对预测、同一冻结表示和 size-matched controls。
2. **图特有、且不等同于局部标签平滑的作用**：例如粒球只估计 `representation reliability`，再用于校准图噪声方法，而不直接生成标签、正样本或训练筛选。需要证明该信号无法被 feature kNN、结构一致性、loss history 或 model confidence 替代。
3. **效率/压缩作为次级工程问题**：如果性能不超过 kNN，但粒球能在明确误差上界或 recall 约束下显著压缩 memory bank、降低检索成本，可形成工程分析；但 PatchCore/cDNP/SupLID 已占据 coreset OOD 主问题，除非出现可证明的新 GB 特性，否则不得作为主论文方向。
4. **联邦/隐私或流式更新**：FedOC-GB 已覆盖联邦粒球知识聚合，但图节点、漂移/持续学习、隐私预算下的局部结构压缩仍可能有任务差异。该方向成本显著更高，不适合作为当前最小 Pilot。

### 3.3 对本项目候选方向的直接含义

| 本地候选 | 文献结论 | 本地证据 | 当前判断 |
|---|---|---|---|
| Frozen GRACE/intent representation + post-hoc multi-local GB/ellipsoid open-set score | 与 MOGB、CLAB、EliDecide、KNNCL、FedOC-GB 高重合 | BANKING77 真 GB 0/7 稳定胜出；StackOverflow 退化为单 prototype；KMeans size-matched 不弱 | **No-Go，不能作为主线** |
| LCSS + soft-small-loss noisy-label learning | 效果链条与 MgBIF、MgCNL、GBmooer、BNO-GB 高度接近；图任务又有 ProCon | 多数据集多 seed 效果强，但机制仍是局部软修正 + selection | **效果候选，论文新颖性候选不足；只作 backup/工程结果** |
| Weak/soft GB relation 直接进入 GCL | MGBCC 已提供球级对比先例；RELGCL 提供非 GB 相对关系强替代 | 本地 hard/soft/prototype/direct relation 多轮失败 | **停止重复调参** |
| 粒球条件可靠性/校准信号 | 尚未发现完全同构论文；但 GBDO、BNO-GB 已覆盖局部 outlier/consistency，竞争强 | 本地尚缺“给定强基线后的条件增益”检验 | **可进入统一评分，但先做零训练诊断** |
| 粒球压缩 kNN/memory bank | PatchCore、cDNP、SupLID 已覆盖 coreset memory、per-class coreset 和几何 coreset OOD；GBOC 覆盖 prototype compression | 本地尚未做系统时间/内存/误差曲线 | **降级为工程分析；不再作为主论文方向** |

## 4. 证据质量与限制

- **正式同行评审且官方页面可核验**：MGBCC、MOGB、EliDecide、FedOC-GB、BNO-GB、G-GBC、MgCNL、MgBIF、GBDO、GBOC、CLAB、KNNCL、Deep kNN、NoisyGL、ProCon、RELGCL、PatchCore、cDNP/Far Away。
- **预印本，只能作为新颖性风险证据，不能作为已建立结论**：CGBC（arXiv）、GBmooer（SSRN）、SupLID（arXiv）。
- **代码状态限制**：表中“未核验到官方代码”只表示本轮在论文页/官方仓库未确认，不应写成“没有代码”。
- **任务迁移限制**：工业故障、图像、智能合约和文本开放意图的正结果不能直接证明图节点任务有效；它们主要用于判断机制先例和设计必要对照。
- **时间边界**：2026 年论文仍可能陆续上线；正式投稿前必须再次更新 2026 年检索，尤其是 open intent、granular-ball noisy-label 和 prototype OOD 三组关键词。

## 5. 文献阶段结论

已经被文献支持的结论是：粒球可以充当中间粒度对比单元、局部多原型、开放边界、局部密度单元或噪声标签缓冲层；这些角色在 2024–2026 年已形成密集工作。仅把上述角色迁移到图节点或换一个 loss，难以支撑强新颖性。

合理但尚未验证的判断是：本项目更可能从“**粒球是否提供强基线之外的条件可靠性信息**”或“**在可控误差下压缩局部 memory / retrieval 成本**”获得独立问题，而不是再次证明粒球能做局部聚类。

已被负结果削弱的方向是：冻结表示上的 post-hoc GB 开放边界、硬 same-ball positive、强 prototype pulling、简单负样本过滤，以及不与 size-matched KMeans/kNN/GMM 比较的粒球分数。

下一阶段所有候选方向必须统一回答三问：

1. 相比最强非粒球局部基线，新增信息是什么？
2. 该信息在多数据集、多 seed、同协议下是否超出标准差并通过配对检验？
3. 若精度不提升，是否存在可量化且足够大的效率、压缩或校准收益？
