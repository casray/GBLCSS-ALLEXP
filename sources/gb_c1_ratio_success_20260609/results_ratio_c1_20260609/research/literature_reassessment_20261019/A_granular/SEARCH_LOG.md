# A 领域检索日志

日期：2026-10-19（Asia/Shanghai）。当前检索为定向范围复审，不声称穷尽所有粒球文献。核心计数仅含已读方法与结果/理论部分的原文；最终计入7篇。

## 初始范围与选择

- 任务书与研究边界：`research/literature_reassessment_20261019/00_READING_BRIEF.md`、`research/core_redesign_20261019/Claude方案评议_20261019.md`。
- 本地全文材料：`C:/Users/Admin/Desktop/实验8去噪5.19-/文献文件/_literature_review_work/extracted_text/`；用文件名定向定位，不修改本地文献库。旧综述 `research/algorithm_redesign_20261013/reviews` 仅用于背景，不能替代原文。
- 主检索：AAAI OJS、IJCAI proceedings、arXiv作者版本；另检索SGBGC作者代码仓库以审计是否可核对标签mask。

## 已检索并纳入

1. `Graph Coarsening via Supervised Granular-Ball for Scalable Graph Neural Network Training` — 官方AAAI页面与作者arXiv扩展全文。已读正式版方法/主实验及扩展附录E、Tables 4–7：五图，5%–20%四种均匀错标率，多粗化率；噪声实验文本没有说清扰动只加在训练标签还是全体标签。结果不是所有图/压缩率/噪声率下都第一。代码审计已完成：默认normal按节点顺序连续切60/20/20（不是论文所述随机split）；构球只mask test、原val标签仍参与；球级另随机重切train/val。`tools/new_graph.py`构建无权nx.Graph，gb_division仅输出edge_index与二值拓扑一致，未发现权重丢失。结论只限公开代码默认路径，不能证明论文表格由此路径运行。官方：[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33404)，作者版：[arXiv v1](https://arxiv.org/html/2412.13842v1)，代码：[GitHub](https://github.com/mxjun17/GBNC)。
2. `GBGC: Efficient and Adaptive Graph Coarsening via Granular-ball Computing` — 查询：`IJCAI 2025 GBGC unsupervised granular ball graph coarsening learning`；IJCAI正式页确认作者、题名、页码和DOI 10.24963/ijcai.2025/388；作者版arXiv:2506.19224。已读方法、理论证明和完整实验章节。[IJCAI](https://www.ijcai.org/proceedings/2025/388)，[arXiv HTML](https://arxiv.org/html/2506.19224)。

## 后续完成并计入核心阅读

- MGNR：IEEE Xplore官方题录与本地作者版全文；DOI 10.1109/TPAMI.2024.3400281。已读粒球定义/分裂的Definitions 3–5、Algorithms 1–7、复杂度分析及分类/聚类实验。核对出DM是平均距而半径是最大距；该方法在KNN仍查点级训练样本，在密度聚类才用球心。作者复杂度包含组内距离矩阵O(N²/Ψ)，不能笼统称线性。另检查Lemma 2：从均距不等式推出加权球体积不等式缺少凸函数幂运算条件。[IEEE Xplore](https://ieeexplore.ieee.org/document/10530064/)，[作者代码](https://github.com/xjnine/MGNR)。
- AW-GBGAE：本地IEEE发表版全文和作者代码，DOI 10.1109/TPAMI.2025.3596615。已读Definitions 1–7、Algorithms 1–3、复杂度分析与九数据集实验。`sqrt(N)`进入邻居数阈值k_nei，不是球数；簇数k为输入。球只指导N个原始样本节点的加权图构造，GAE仍处理N个点且有O(N²H)重构项。[DOI](https://doi.org/10.1109/TPAMI.2025.3596615)，[PubMed题录](https://pubmed.ncbi.nlm.nih.gov/40773392/)，[作者代码](https://github.com/xjnine/AWGBGAE)。
- GBC-CNN：`Granular-ball Representation Learning for Deep CNN on Learning with Label Noise`，arXiv:2409.03254。已读§3.2–3.5、Algorithm 1、梯度分配/经验池机制、§4.1–4.3和Tables 1–5；核对纯度阈值、噪声实验和backbone仍逐图像前向的效率边界。正式题名/作者：[arXiv](https://arxiv.org/abs/2409.03254)。
- GB-CGNN：`Efficient and Scalable Granular-ball Graph Coarsening Method for Large-scale Graph Node Classification`，arXiv:2603.29148v2。已读§IV–V、Theorem 1、复杂度、12数据集结果与消融；区分了√N初始METIS球数与最终球数，也核对“球级子图采样、球内节点仍跑GCN”的路径。作者版：[arXiv v2](https://arxiv.org/html/2603.29148v2)。
- GBG++：`GBG++: A Fast and Stable Granular Ball Generation Method for Classification`，TETCI 2024，DOI 10.1109/TETCI.2024.3359091，arXiv:2305.18450。替换全文不可得的ICDE候选。已读发表版§III-A–E、Definitions 1–6、Algorithms 1–2、§V-A–G、Tables I–V和Figs. 8–11；核对监督生成、mean-distance半径、orphan删除、复杂度假设、20/24数据集口径矛盾和噪声/消融实验。[IEEE DOI](https://doi.org/10.1109/TETCI.2024.3359091)，[arXiv](https://arxiv.org/abs/2305.18450)。

## 版本与访问记录

- SGBGC：AAAI正式版与arXiv作者扩展版并用；正文依据正式版，附录E依据作者扩展全文。噪声结果的文字说明不够明确，按表格限定结论。
- GBGC：IJCAI正式版元数据与arXiv作者HTML核对；按正式论文主文阅读。
- ICDE PLGB-FSC：核对官方ICDE列表、IEEE题录、作者代码仓库；IEEE Xplore因机器人验证只返回不可读页面，未找到作者全文。本次只掌握摘要，明确不计入核心；因此以本地全文GBG++补足第七篇。

## 当前排除/未计数

- ICDE PLGB-FSC只读到题录/摘要，IEEE Xplore全文被机器人验证拦截；明确不计入七篇核心阅读。以GBG++替代，原因是本地发表全文可逐项核查方法、复杂度与实验，且其标签监督/纯度机制直接相关。
- `Graph Coarsening via Supervised Granular-Ball...` 作者仓库进行了网页只读调用链审计，没有执行代码；报告结论限定在默认代码路径。
- 不纳入2027在线日期的ScienceDirect页面，时间晚于本任务截止日2026-10-19。
