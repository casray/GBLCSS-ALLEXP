# GBGCL：Multi-Granularity Graph Contrastive Learning Framework via Granular-Ball on Heterogeneous Graphs

2026-10-14补充：OFA引用链已核，[42]为HeterGCL、[64]为HOG-GCN、[65]为CCA-SSG。具体方法、证据与限制见[HeterGCL/OFA](HeterGCL_OFA.md)、[HOG-GCN](HOG_GCN_Wang2022.md)、[CCA-SSG](CCA_SSG_OFA_theory.md)。本项目已完成三异配图原GB64基线，不能继续把原稿适配写为“尚未启动”；结果见workspace experiments/heterophily_original_baseline_20261014/实验结论.md。OFA尚未在本项目实现或测试。

更新：2026-10-13；用户补充全文后的更正版本。此前摘要卡依据题名判断“多类型异构图”，该判断错误，以下以全文为准。

## 文献信息与证据

- 作者：Shuyin Xia, Guan Wang, Cheng Tan, Chao Huang, Sen Zhao, Yi Liu, Yi Wang, Wei Wang, Guoyin Wang, Xinbo Gao。文件名的 Wang 等不是正式作者顺序依据。
- IEEE TPAMI，2026 接收作者版，DOI [10.1109/TPAMI.2026.3720194](https://doi.org/10.1109/TPAMI.2026.3720194)。[PubMed 元数据](https://pubmed.ncbi.nlm.nih.gov/42566369/) 记载 2026-08-07 ahead-of-print。页眉的 2021 是 LaTeX 模板残留，不能作发表年份。
- 用户全文：C:/Users/Admin/Downloads/Wang 等 - 2026 - Multi-Granularity Graph Contrastive Learning Framework via Granular Ball on Heterogeneous Graphs.pdf，14 页。
- 已阅读全文提取，视觉核对 PDF 第 5 页公式 (7)–(14)、第 9 页 Table III/IV；未复现代码。来源哈希见 [GBGCL_source.json](../../sources/GBGCL_source.json)，提取 [GBGCL_fulltext.txt](../../sources/GBGCL_fulltext.txt)。文本提取有 6 个非法代理字符以替代符处理，数学以原 PDF 为准。
- 文内代码为 https://anonymous.4open.science/r/GBGCL/ ；本轮 web 工具访问失败，不声称完成代码审计。

## 研究问题：本文实际研究 heterophily

虽然题名采用 heterogeneous graphs，导言用相邻节点类别/特征相异讨论问题，形式化仅为普通无向 G=(V,E)，实验为 Cornell、Texas、Wisconsin、Chameleon、Squirrel、Film。本文核心是异配图表示学习，并非通过节点类型、关系类型或元路径定义的多类型异构图任务。两个通用概念仍须区分；对这篇论文不能只按题名下结论。

## 方法与信息使用

1. GBA 初始化：最高度节点作为中心，BFS 收集约 floor(sqrt(N)) 个节点，移除后重复（p5，式4–6）。
2. 质量：球内同类边占比 quality(B)=sum 1(Y_u=Y_v)/|E_B|，明确说明 Y_i 是节点标签（p5，式7）。父球质量低于两个子球质量最大值时继续分裂，子球用两个高度中心的 BFS 划分（式8）。
3. 因此，**按公开公式构球依赖类别标签**。Table III 可用信息列也将 GBGCL 标为 X,A,Y，而 GRACE、HeterGCL 等为 X,A。正文没有在该质量式处明确限定 Y 为训练标签、伪标签，或说明缺失标签如何处理。该不透明处需核代码；不能仅凭公式指控实际使用测试真值。
4. 同球节点为正对、跨球节点为负对（式10），球原始属性均值组织为 GB_X（式11–12）。这不同于 GB-LCSS 冻结 H 上无标签 2-means 后只生成软监督。
5. MLP 编码 X 得 H0；属性掩蔽得 X1/H1。L_c 为球相关对比项；L_o 含两视图不变性和 Gram 矩阵去相关；GMM 产生软分配 R，L_l 约束原图相邻节点分配接近（p6，式16–20）。总损失为 alpha L_c+(1-alpha)(L_o+L_l)。
6. MLP 的前向属性输入不等于整个训练不使用图或标签。Algorithm 1 在训练循环之前构球，不支持“每步在线重新分球”的表述。
7. 式11中正对集合与节点求和记号、式14中中心索引/维度不够清楚，严格复现需要代码解释。不要补写未证实的实现。

## 实验核读与局限

分类用 10%/10%/80% train/validation/test、10 次运行平均（p8）；不同于当前 GB-LCSS 的 20/10/70 固定协议。全文实验没有给出独立标签噪声注入评测。因此，本论文不能证明其构球在 noisy Y 下仍可靠。

Table III（p9，准确率，百分点）：

| 数据集 | GBGCL | 去掉 GBA | 均值差 |
|---|---:|---:|---:|
| Cornell | 79.2±2.8 | 71.8±2.9 | +7.4 |
| Texas | 78.2±2.4 | 71.0±3.3 | +7.2 |
| Wisconsin | 78.1±2.5 | 74.2±3.1 | +3.9 |
| Chameleon | 45.3±1.6 | 46.6±1.0 | −1.3 |
| Squirrel | 33.9±0.7 | 33.7±0.0 | +0.2 |
| Film | 36.4±0.4 | 36.3±0.6 | +0.1 |

这些均值差不是配对显著性检验。粒球收益主要集中于三张很小的 WebKB 图；不能称六图均显著改善。该表 GBGCL 也不是六图全面最优：Chameleon/Squirrel 的 GCN 更高，Film 的 HeterGCL 更高。

Table IV 的 MLP 变体高于论文所选 GNN 变体：Cornell 79.2 对56.0、Texas78.2对65.0、Wisconsin78.1对59.2、Film36.4对35.8。只能支持该受测架构对照，不能证明 MLP 对所有异配 GNN 都占优。原始 Chameleon/Squirrel 的重复节点问题另见 ICLR2023基准论文，本文未在表中明确标注 filtered 版本。

## 对 GB-LCSS 的价值与研究边界

- 值得借鉴：结构/属性各自组织关系，MLP 保留自属性，按区域分辨率组织证据，比较 GNN 与 MLP。
- 不直接迁移：依赖 Y 的质量分裂；同球全正、跨球全负；将球纯度当作可靠概率；噪声环境中未经诊断的邻边同簇约束。
- 更有意义的差异：在不读取干净标签构球的条件下，为带噪训练标签构造可验证的局部监督，并检验其对 X-only 学生是否有增益。这是研究假设，非已确定新颖性。
- GBGCL 足以否定“首次粒球图对比”宽泛主张；SAGHL、HALF、ES-MLP、MGNR 等仍须分别比较任务、信息、训练及部署。见 [异配路线审查](../../05_异配图路线审查与建议.md)。

