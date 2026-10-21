# B领域检索日志

检索日期：2026-10-19（UTC/Asia Shanghai日期同日）

## 核心检索式与题录核验

- `Scaling Up Graph Neural Networks Via Graph Coarsening SCAL KDD 2021 authors official PDF`：作者预印本 arXiv:2106.05150核验作者、题名、章节和公式；DOI 10.1145/3447548.3467256核验KDD 2021出版入口。纳入。
- `Graph Reduction with Spectral and Cut Guarantees Loukas JMLR 2019 full text`：JMLR正式版42页核验作者、页码、定义、算法、复杂度与实验；确认作者仓库 v1.1。纳入。
- `Featured Graph Coarsening with Similarity Guarantees FGC ICML 2023 paper authors`：PMLR官方论文条目与PDF核验作者、ICML/PMLR卷页、优化、KKT收敛、节点分类与复杂度。纳入。

## 原文阅读定位

1. **SCAL** — 作者版全文 [arXiv](https://arxiv.org/abs/2106.05150)：§3.1–3.2、§4.1–4.5、§6.1–6.2；Table 1/2；重点检查式(3)、Theorem 4.7、标签聚合方式、train/validation/test split、20-run统计和coarsening-time表。KDD DOI：[10.1145/3447548.3467256](https://doi.org/10.1145/3447548.3467256)。代码链接在当前读取的作者版页面未确认，继续检索。
2. **Loukas** — 正式JMLR PDF [全文](https://jmlr.org/papers/volume20/18-680/18-680.pdf)：pp.3–8、10–19、20–26及Appendix相关证明；核对Definition 11、Theorems 14–15、Algorithm 2、复杂度、结果与限制。作者代码：[v1.1](https://github.com/loukasa/graph-coarsening/tree/v1.1)。
3. **FGC** — ICML/PMLR官方PDF [全文](https://proceedings.mlr.press/v202/kumar23a/kumar23a.pdf)：pp.2–9及Appendix A.6–A.8；核对Eq. (6)–(7)、Algorithm 1、Theorem 1、Theorem 3.1、Algorithm 2、Tables 4–5。官方题录：[PMLR](https://proceedings.mlr.press/v202/kumar23a.html)。作者代码入口在目前检查的PMLR页面/论文文本中未确认，记为未知。

## 本地材料和排除

- 已读取当前任务书 `research/literature_reassessment_20261019/00_READING_BRIEF.md` 及方案评议 `research/core_redesign_20261019/Claude方案评议_20261019.md`，作为研究问题和边界；这两份不是论文证据。
- 当前目录的首轮定名检索没有发现这三篇论文的已提取全文副本。此阶段直接使用期刊/会议/作者版本原文，不以第三方摘要替代。
- 第三方二手摘要/AI paper summary页面未作为方法或结果证据。

## 补充检索与原文阅读（完成）

- `Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training arXiv 2312.15520 WWW companion code`：以作者预印本 [arXiv HTML](https://arxiv.org/abs/2312.15520) 读§3–5、Eq.(6)–(9)、Theorem 1、Algorithm 1、Tables 2–5及附录B–G；以作者仓库 [amazon-science/convolution-matching](https://github.com/amazon-science/convolution-matching) 只读 NodeClassificationCoarsener、训练/测试入口和边/特征聚合实现；未运行。NeurIPS 2024论文References[11]核对 Companion Proceedings ACM Web Conference 2024 pp.1502–1510；ACM DOI 10.1145/3589335.3651920。记录其workshop companion身份，不误写主会论文。
- `Graph Coarsening with Message-Passing Guarantees Joly Keriven NeurIPS 2024 full text code`：使用 [NeurIPS官方PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/d041d6bb47c01a4ce327a1773703e9a0-Paper-Conference.pdf)、官方题录/DOI 10.52202/079017-3648及文内作者GitLab入口。读取§2–3.3、§4、Eq.(4)–(11)、Theorem 1/2、脚注3，附录A.1–A.3/B.2–B.6抽查。直接核实 (S_c=QSQ^+)、lift后原训练节点CE、label coarsening未研究、激活假设、复杂度 (O(T(n+e)D+TN))；区分RSA谱粗化/实验初始化的大矩阵逆与硬分区稀疏构造算子。
- `Graph Condensation for Graph Neural Networks ICLR 2022 method gradient matching code`：以 [OpenReview正式PDF](https://openreview.net/pdf?id=WLEx3Jo4QaB)、作者 [arXiv HTML](https://arxiv.org/html/2110.07580)、作者代码 [GCond](https://github.com/ChandlerBang/GCond) 核对版本、方法、训练split、迁移、结果和复杂度。阅读§3.1–3.2、§4.1–4.4、Appendix A.2–A.3/B Algorithm 1/C.3及Tables 1–4/8–9。纳入为凝缩代表，不把合成节点误作原节点分区。
- `Structure-free Graph Condensation NeurIPS 2023 SFGC author code`：以 [NeurIPS官方论文页/PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/13183a224208671a6fc33ba1aa661ec4-Paper-Conference.pdf)、[作者arXiv HTML](https://arxiv.org/html/2306.02664)、[作者代码](https://github.com/Amanda-Zheng/SFGC)核验。官方PDF本轮web open偶发内部错误，但官方页面的搜索摘录可用，方法及实验以可完整读取的作者HTML为准。读取§2.2–2.4、§3.1–3.2、§4、Algorithm 1及Appendix E.2–E.3。核对图free合成特征、专家轨迹、GNTK验证快照选择与时间成本。
- FGC卡片按独立复核修正：理想定义中的loading是硬assignment；实际算法优化连续非负loading松弛，通过group sparsity推动稀疏，作者报告大 λ 下观察到近似正交；不再说优化时逐步硬分配。
- 全部七篇均读到方法和关键理论/实验部分，满足本领域核心数；未执行任何论文仓库、安装依赖或训练。本轮未完成全局去重；跨A/C/D引用不在本地再次计数。
