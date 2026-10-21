# 五领域文献阅读索引

2026-10-19。Luna/max分领域阅读，主代理复核关键设计断言。不是逐页精读全部文献或穷尽SOTA的声明。经典/代表性依据方法关联性，未编造被引次数。

| 领域 | 核心篇数 |
|---|---:|
| 粒球生成与粗粒度表示 | 7 |
| 图粗化、凝缩与计算保持 | 7 |
| 带噪图学习与鲁棒压缩 | 8 |
| 异配表示、传播与粗化 | 7 |
| 可靠分裂、剪枝与预算分配 | 7 |

共 36 篇核心条目；题名及已登记DOI/arXiv去重通过。缺标识符或异名仍需结合题录人工复核。

## 粒球生成与粗粒度表示

| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |
|---|---|---|---|
| [Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training](https://ojs.aaai.org/index.php/AAAI/article/view/33404) | 2025 / AAAI-25, 39(12), 12872–12880 | AAAI正式版 §3（Definition 1–2、SGBGC、coarsened graph、GNN training）、Algorithm 1、§4和Tables 1–3；作者扩展全文附录E及Tables 4–7（五图、5%/10%/15%/20%噪声、多粗化率）；公开代码调用链 gb_division.py、tools/corse_split.py、tools/split_ball_purity.py、tools/new_graph.py、split_test.py、main.py。 | 核心 |
| [GBGC: Efficient and Adaptive Graph Coarsening via Granular-ball Computing](https://www.ijcai.org/proceedings/2025/388) | 2025 / IJCAI-25, pp. 3489–3497 | IJCAI正式版 §3.1–3.3、Theorem 1及证明、§4.1–4.6、Tables 1–4；本地作者全文逐段核对。 | 核心 |
| [MGNR: A Multi-Granularity Neighbor Relationship and Its Application in KNN Classification and Clustering Methods](https://ieeexplore.ieee.org/document/10530064/) | 2024 / IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12), 7956–7972 | TPAMI作者版 §III-B–D（Definitions 3–9、Algorithms 1–7）、§III-E复杂度分析、§IV分类/聚类实验及Tables IV–VIII。 | 核心 |
| [AW-GBGAE: An Adaptive Weighted Graph Autoencoder Based on Granular-Balls for General Data Clustering](https://doi.org/10.1109/TPAMI.2025.3596615) | 2025 / IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(12), 11298–11314 | TPAMI正式版 §III-A–D（Definitions 1–7、Algorithms 1–3）、复杂度分析、§IV实验及Tables II–V、Figs. 5–11。 | 核心 |
| [Granular-ball Representation Learning for Deep CNN on Learning with Label Noise](https://arxiv.org/abs/2409.03254) | 2024 / arXiv preprint, arXiv:2409.03254v1 | 作者全文§3.2–3.5、Algorithm 1、梯度分配/经验池机制、§4.1–4.3及Tables 1–5；核对噪声实验、阈值消融和局限。 | 核心 |
| [Efficient and Scalable Granular-ball Graph Coarsening Method for Large-scale Graph Node Classification](https://arxiv.org/abs/2603.29148) | 2026 / arXiv preprint, arXiv:2603.29148v2 | arXiv v2作者全文§IV-A–D（Definition 1、Eq. 10–16、Theorem 1、GCN训练和复杂度）、§V-A–F、Tables II–VI。 | 核心 |
| [GBG++: A Fast and Stable Granular Ball Generation Method for Classification](https://doi.org/10.1109/TETCI.2024.3359091) | 2024 / IEEE Transactions on Emerging Topics in Computational Intelligence, 8(2), 2022–2036 | TETCI发表版§III-A–E、Definitions 1–6、Algorithms 1–2、§V-A–G、Tables I–V、Figs. 8–11；与IEEE DOI和arXiv作者版元数据核对。 | 核心 |

## 图粗化、凝缩与计算保持

| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |
|---|---|---|---|
| [Scaling Up Graph Neural Networks Via Graph Coarsening](https://doi.org/10.1145/3447548.3467256) | 2021 / KDD 2021 | 作者版本§3.1–3.2 pp.4–5；§4.1–4.5 pp.5–10；§6.1–6.2 pp.11–13；式(3)、Theorem 4.7、Table 1/2。 | 核心 |
| [Graph Reduction with Spectral and Cut Guarantees](https://jmlr.org/papers/v20/18-680.html) | 2019 / Journal of Machine Learning Research 20(116):1–42 | §2.1–2.4 pp.3–8；§3 pp.10–14；§4.1–4.3 pp.14–19；§5 pp.20–26；Appendix相关证明抽查。 | 核心 |
| [Featured Graph Coarsening with Similarity Guarantees](https://proceedings.mlr.press/v202/kumar23a.html) | 2023 / ICML 2023, PMLR 202:17953–17975 | §2.2–2.3 pp.2–3；§3.2–3.6 pp.3–6；§4 pp.6–9；Appendix A.6–A.8证明。 | 核心 |
| [Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training](https://dl.acm.org/doi/10.1145/3589335.3651920) | 2024 / Companion Proceedings of the ACM Web Conference 2024, pp.1502–1510 (companion/workshop; not main track) | 作者版§3–5；Eq.(6)–(9)、Theorem 1、Algorithm 1、Tables 2–5、Appendix B–G；作者代码只读标签聚合、训练/评估入口和聚合实现，未运行。 | 核心 |
| [Graph Coarsening with Message-Passing Guarantees](https://proceedings.neurips.cc/paper_files/paper/2024/file/d041d6bb47c01a4ce327a1773703e9a0-Paper-Conference.pdf) | 2024 / NeurIPS 2024 Main Conference Track | §2–3.3 pp.3–7；§4 pp.8–10；Eq.(4)–(11)、Theorem 1/2和脚注3；Appendix A.1–A.3及B.2–B.6抽查。 | 核心 |
| [Graph Condensation for Graph Neural Networks](https://openreview.net/pdf?id=WLEx3Jo4QaB) | 2022 / ICLR 2022 | §3.1–3.2、§4.1–4.4；Appendix A.2–A.3、B Algorithm 1、C.3复杂度；Tables 1–4、8–9。 | 核心 |
| [Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data](https://proceedings.neurips.cc/paper_files/paper/2023/file/13183a224208671a6fc33ba1aa661ec4-Paper-Conference.pdf) | 2023 / NeurIPS 2023 Main Conference Track | §2.2–2.4、§3.1–3.2、§4、Algorithm 1；Appendix E.2–E.3；Eq.(3)–(7)、Tables 1–3及A4/A5。 | 核心 |

## 带噪图学习与鲁棒压缩

| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |
|---|---|---|---|
| [NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise](https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 / NeurIPS 2024, Datasets and Benchmarks Track | 正式稿pp.1-2,4-6; §2-4、RQ2/RQ3/RQ5、App. A6; arXiv HTML v2同章节；官方README数据与噪声接口 | 核心 |
| [NRGNN: Learning a Label Noise-Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs](https://dl.acm.org/doi/10.1145/3447548.3467364) | 2021 / ACM SIGKDD 2021, pp. 227-236 | arXiv HTML §3定理3.1/3.2、§4式(8)-(14)、§5.1-5.5协议和表1-3；核对KDD正式题录 | 核心 |
| [Robust Training of Graph Neural Networks via Noise Governance](https://doi.org/10.1145/3539597.3570369) | 2023 / WSDM 2023, pp. 607-615 | 作者PDF §4.2-4.4式(7)-(16)、§5.1-5.4表2-4及附录；题录核对DOI正式记录。PDF首页DOI为占位符，未据此引文。 | 核心 |
| [PLGC: Pseudo-Labeled Graph Condensation](https://jmlr.org/tmlr/papers/) | 2026 / Transactions on Machine Learning Research (TMLR), June 2026 | arXiv v1 §3式(2)-(7)、§4理论与假设、§5.1.3-5.1.4噪声实验及表图；官方TMLR目录、代码README核对。 | 核心 |
| [Divide and Denoise: Empowering Simple Models for Robust Graph Semi-Supervised Learning against Label Noise](https://doi.org/10.1145/3637528.3671798) | 2024 / ACM SIGKDD 2024, pp. 574-584 | 论文§4.1-4.4、式(4)-(12)、§5.1-5.3、表1/2和图2/4；核查官方代码main.py与train.py标签污染/验证接口。 | 核心 |
| [RobGC: Towards Robust Graph Condensation](https://doi.org/10.1109/TKDE.2025.3569629) | 2025 / IEEE Transactions on Knowledge and Data Engineering, 37(8), 4791-4804 | arXiv HTML §II-IV、式(7)-(14)、主要表图、消融和复杂度；核对TKDE元数据及作者代码。 | 核心 |
| [Robust Graph Condensation via Classification Complexity Mitigation](https://proceedings.neurips.cc/paper_files/paper/2025/file/ce1a7774429d5c06cc744ff4260a8e9f-Abstract-Conference.html) | 2025 / NeurIPS 2025 Main Conference | NeurIPS正式PDF与arXiv HTML §3-4、定理1/2、表1/2、附录C.1成本与攻击协议。 | 核心 |
| [Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction](https://proceedings.mlr.press/v306/ju26c.html) | 2026 / ICML 2026, Proceedings of Machine Learning Research 306:54828-54846 | arXiv v2 §2-4式(2)-(8)、定理3.1、表1-3、§4.7-4.9、附录B-E复杂度/理论/消融；PMLR正式题录和官方代码train.py、utils.py。 | 核心 |

## 异配表示、传播与粗化

| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |
|---|---|---|---|
| [Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs](https://proceedings.neurips.cc/paper/2020/hash/58ae23d878a47004366189884c2f8440-Abstract.html) | 2020 / NeurIPS 2020 | NeurIPS正式PDF §2、§3.1 Theorems 1–3、§3.2 Eq.5–8与复杂度、§5.1–5.2及Tables 1/4/5、Fig.3；PDF pp.2–8；另核附录D.2非线性消融指引。 | 核心 |
| [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP) | 2021 / ICLR 2021 | 作者arXiv v6（2021-10-26）：§3 Eq.1、§4 Theorems 4.1–4.2、Appendix A.1/A.3正式定理与证明条件、§5、Appendix A.5/A.9数据生成/协议/效率；同时用ICLR 2021官方目录核对发表版本。OpenReview PDF访问触发challenge，未将其误称已阅读。 | 核心 |
| [Revisiting Heterophily For Graph Neural Networks](https://papers.neurips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html) | 2022 / NeurIPS 2022 | NeurIPS正式PDF §2.2–3 (pp.2–4), §4.1–4.2 (pp.5–6, Eq.4–11及Theorem 1), §6 (pp.7–9, Table 1及主比较), §7 limitations。 | 核心 |
| [A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?](https://iclr.cc/Downloads/2023) | 2023 / ICLR 2023 | 作者全文arXiv v2 (2024-03-02)：§3.1–3.2、§4、§5.1–5.2、Appendix A–B；另查ICLR 2023官方目录确认正式发表。ICLR官网文章跳转当前抓取失败，正文按作者arXiv版本核读。 | 核心 |
| [Understanding Heterophily for Graph Neural Networks](https://proceedings.mlr.press/v235/wang24u.html) | 2024 / ICML 2024, PMLR 235:50489-50529 | PMLR正式元数据；作者v2全文§2–5、Theorem 1–4、Eq.4/7/8/9、Appendix A.2–A.4、B.4、C.1–C.3。PDF定位：Assumption 1–2约pp.2–3，Theorem 2/Eq.4 p.3，Theorem 3/Eq.7 p.4，Theorem 4/Eq.8–9 pp.4–5，实验pp.6–8，Appendix B.4 p.29、C.2 p.30。 | 核心 |
| [Partition-wise Graph Filtering: A Unified Perspective Through the Lens of Graph Coarsening](https://doi.org/10.1145/3711896.3737075) | 2025 / KDD 2025, Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Volume 2 | 作者全文§2 RSA、§3.1–3.6、Eq.6–11、§4.1–4.3/Tables 1/3/5/6、Appendix F.1、G.1；PDF定位：RSA约p.2，Eq.7–8和Theorem 3.2约pp.2–3，Table 1约pp.5–6，复杂度§3.6约p.6；Appendix F.1按章节核对split与调参。 | 核心 |
| [Remedying Coarsening-Based GNN Training under Heterophily via Adaptive Complementary Enhancement](https://proceedings.mlr.press/v337/li26j.html) | 2026 / UAI 2026, Proceedings of Machine Learning Research 337:3738–3761 | PMLR/UAI正式记录；作者全文§2.1–2.2、Prop.2.2/Table 1、§3.1/Eq.5、§3.2.1–3.2.3/Eq.6–11/Prop.3.1、§4.1–4.5/Tables 3–4；Appendix D.1–D.4；静态核仓库README、backbones.py、coarse_train.py、refined_projector.py。PDF约pp.1–3理论/初始损失，pp.3–4投影细化/ASR，pp.4–5目标/复杂度，pp.5–8实验效率，pp.16–17代码附录。 | 核心 |

## 可靠分裂、剪枝与预算分配

| 论文与原始来源 | 年份/发表 | 已读范围 | 状态 |
|---|---|---|---|
| [On the Robustness of Decision Tree Learning under Label Noise](https://link.springer.com/chapter/10.1007/978-3-319-57454-7_53) | 2017 / PAKDD 2017, Lecture Notes in Computer Science 10234, pp. 685–697 | 作者arXiv稿§2.2–3.4、§4、附录A；核对正式PAKDD题录、卷页与DOI。 | 核心 |
| [Robust Loss Functions for Training Decision Trees with Noisy Labels](https://ojs.aaai.org/index.php/AAAI/article/view/29516) | 2024 / Proceedings of the AAAI Conference on Artificial Intelligence 38(14), 15859–15867 | 正式AAAI PDF §2–4、定理3–5、表1、图2–6及实验附录；核对官方论文页和代码入口。 | 核心 |
| [Unbiased Recursive Partitioning: A Conditional Inference Framework](https://doi.org/10.1198/106186006X133933) | 2006 / Journal of Computational and Graphical Statistics 15(3), 651–674 | 作者PDF§2–6、算法步骤、条件置换定义、Table 1、Figure 4及基准结果Figures 6–7。 | 核心 |
| [Mining High-Speed Data Streams](https://dl.acm.org/doi/10.1145/347090.347107) | 2000 / Proceedings of the 6th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 71–80 | 作者KDD论文§2–4、表1、脚注2、Theorem 1及Figures 1–7；核对正式题录。 | 核心 |
| [Hierarchical testing of variable importance](https://academic.oup.com/biomet/article-abstract/95/2/265/230264?login=false) | 2008 / Biometrika 95(2), 265–278 | 作者PDF§1–3、式(1)–(12)、Theorems 1–2、Figures 2–4和四组数值模拟。 | 核心 |
| [Recursive partitioning for heterogeneous causal effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941430/) | 2016 / Proceedings of the National Academy of Sciences 113(27), 7353–7360 | PNAS正式全文§2–4、诚实分区/EMSE式、Theorem与Table 1三设计模拟；核对arXiv v3和PNAS正式题录。 | 核心 |
| [Generalized and Scalable Optimal Sparse Decision Trees](https://proceedings.mlr.press/v119/lin20g.html) | 2020 / Proceedings of the 37th International Conference on Machine Learning, PMLR 119:6150–6160 | PMLR正式PDF§2目标式(1)、§3–4 DPB/界/算法、§5实验和Figures 2–6；PMLR正式页面核题录。 | 核心 |
