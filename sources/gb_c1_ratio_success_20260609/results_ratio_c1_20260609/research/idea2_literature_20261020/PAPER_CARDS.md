# Idea 2 文献卡片

核查日期：2026-10-20。共 26 张卡片：20 篇方法论文，6 项统计/历史文献或官方工具文档。

**阅读状态**：“正文”表示读到与本问题有关的方法、公式或实验段落，并不表示复核了整篇论文的所有证明。“局部正文”不冒充全文；“仅摘要”不据此断言正文没有某机制。NC=不知道哪些配对出错；PVP=部分对应已知、其余对应未知；IV=视图缺失。三者不能混同。

**数字状态**：NR=所检查的原文表格未报告该档位；U=尚未可靠核实，不能解释为没有报告；NA=任务/指标不适用。后续论文重新实现的结果单独标注，不能冒充该方法原论文的结果。数值集中保存在 NUMERIC_EVIDENCE.md。

## A1 — RMCNC

Robust Multi-View Clustering with Noisy Correspondence，TKDE 36(12):9150–9162，2024。

- 证据：[作者正文](https://xlearning-lab.com/markdowns/2024-tkde-robust-multi-view-clustering-with-noisy-correspondence/)，方法 III、实验 IV、Table III；正文。
- ①任务：NC，也分别测 PVP、完整对齐。
- ②粒度：样本对；鲁棒概率损失，学习后按距离重匹配。
- ③共同成员：没有以独立组的交集建立关系。
- ④机会校正：未见超几何/随机重叠校正。
- ⑤粒度选择：损失参数 q，不是球容量，也未给出容量—错配率定律。
- ⑥协议：NC 含 20/40/60/80%，PVP 单独设置；第一视图作锚，其他视图打乱。实际移动数、同类内错配需代码记录。
- ⑦数字：Table III 的 20/40/60% 已摘录；Caltech 为 9146 样本、102 类版本，不能与 8677 样本版本拼表。
- ⑧[作者代码](https://github.com/sunyuan-cs/2024-TKDE-RMCNC)。
- ⑨重叠：中。问题相同，修正对象和统计量不同。

## A2 — CANDY

Robust Contrastive Multi-view Clustering against Dual Noisy Correspondence，NeurIPS 2024。

- 证据：[会议页](https://papers.nips.cc/paper_files/paper/2024/hash/dbe81b08f7dc4dd8b43bc62dedfd9662-Abstract-Conference.html)、[作者 slides](https://nips.cc/media/neurips-2024/Slides/96528.pdf)、[model.py](https://raw.githubusercontent.com/XLearning-SCU/2024-NeurIPS-CANDY/main/model.py)、[loader](https://raw.githubusercontent.com/XLearning-SCU/2024-NeurIPS-CANDY/main/dataset_loader.py)。主 PDF 获取失败；局部正文/作者演示及核心代码核查。
- ①任务：同时处理错正配对与同类假负对。
- ②粒度：样本及高阶邻域；上下文亲和矩阵乘积，SVD 去噪。
- ③共同成员：不是硬分组成员交集，存在软邻域关系聚合。
- ④机会校正：核心代码未见超几何检验。
- ⑤粒度：邻域与谱阈值；未核实组容量随噪声变化的规则。
- ⑥协议：loader 固定随机源 1234，选取 floor(fp_ratio·N) 行，再在所选行内置换；允许不动点、同类错配。推理 extract_feature 默认 realign=True。
- ⑦数字：原论文 20/40/60=U/U/U；CorreGen Table 1 重实现含 20%，另见数值文件。
- ⑧[代码实际已公开](https://github.com/XLearning-SCU/2024-NeurIPS-CANDY)，仓库旧简介仍写“即将公开”，不可据此判定没有代码。
- ⑨重叠：高。都聚合关系证据；本候选的硬组计数与零假设校准不同。

## A3 — ROLL

ROLL: Robust Noisy Pseudo-label Learning for Multi-View Clustering with Noisy Correspondence，CVPR 2025，30732–30741。

- 证据：[会议页](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_ROLL_Robust_Noisy_Pseudo-label_Learning_for_Multi-View_Clustering_with_Noisy_CVPR_2025_paper.html)、[原 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_ROLL_Robust_Noisy_Pseudo-label_Learning_for_Multi-View_Clustering_with_Noisy_CVPR_2025_paper.pdf)的检索可读段落 3.6、4.1–4.3；完整 PDF 打开失败。局部正文。
- ①任务：NC 与伪标签错误共同存在。
- ②粒度：实例伪标签和样本对；预热后进行鲁棒伪标签、鲁棒对比训练。
- ③共同成员：已核实部分未使用独立组交集。
- ④机会校正：已核实部分未见。
- ⑤粒度：分类头 K 为类别数；不是按错配率调局部组大小。
- ⑥协议：第一视图固定，其他视图随机打乱 m%；档位 0/20/50/80，5 次运行，表征重匹配后拼接聚类。
- ⑦数字：原论文 20%=U；40/60%=NR（上述主实验档位）。CorreGen 的重实现另列。
- ⑧[代码](https://github.com/sunyuan-cs/2025-CVPR-ROLL)含训练、预热及数据加载文件，尚未本机运行。
- ⑨重叠：中。直接性能竞品，主要通过鲁棒损失处理错误。

## A4 — CorreGen

Uncover Underlying Correspondence for Robust Multi-view Clustering，ICLR 2026。

- 证据：[正式会议页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html)、[作者全文](https://xlearning-lab.com/markdowns/2026-iclr-uncover-underlying-correspondence-for-robust-multi-view-clustering/)，3.2、Table 1、附录 C；正文。
- ①任务：NC、语义假负对、无可用对应的损坏样本。
- ②粒度：样本级软多对多；GMM 边缘分布、OT、虚拟样本，EM 更新。
- ③共同成员：不是硬组共同成员计数。
- ④机会校正：OT 边缘约束不等于超几何机会校正。
- ⑤粒度：GMM/批次设置，不是平均球容量按噪声增长。
- ⑥协议：MR 控制随机置换比例；CR 控制高斯损坏；评估时在 512 样本批次中重匹配。
- ⑦数字：Table 1 有 0/20/50/80%；40/60%=NR；20% 摘录于数值文件。
- ⑧[代码](https://github.com/XLearning-SCU/2026-ICLR-CorreGen)。
- ⑨重叠：高。多对多和拒绝无效对应已有；区别须落到计数证据和粒度，不可只说“区域对齐”。

## A5 — SURE

Robust Multi-View Clustering With Incomplete Information，TPAMI，2022 在线发表、2023 卷期 45(1):1055–1069。

- 证据：[作者全文](https://xlearning-lab.com/markdowns/2022-tpami-robust-multi-view-clustering-with-incomplete-information/)，方法和实验；正文。
- ①任务：PVP+视图缺失，重点是随机负采样的假负对。
- ②粒度：实例及语义类别对应，鲁棒对比损失。
- ③共同成员：没有独立组的交集计数。
- ④机会校正：未见。
- ⑤粒度：距离/损失规则，不是区域容量。
- ⑥协议：可用已知对齐信息、缺失掩码；不能无说明转成“所有配对状态未知”的 NC。
- ⑦数字：原生 20/40/60 NC 表不作为本卡已核实项；RMCNC Table III 重实现数字另列。
- ⑧[代码](https://github.com/XLearning-SCU/2022-TPAMI-SURE)。
- ⑨重叠：中。是重要历史基线，不能仅因名称 robust 就视为最新 NC 最强基线。

## A6 — DIVIDE

Decoupled Contrastive Multi-View Clustering with High-Order Random Walks，AAAI 2024，14193–14201。

- 证据：[作者全文](https://xlearning-lab.com/markdowns/2024-aaai-decoupled-contrastive-multi-view-clustering-with-high-order-random-walks/)，方法、Table 2；正文。
- ①任务：完整/缺失视图聚类，处理邻域内假正、邻域外假负。
- ②粒度：高阶随机游走邻域，软关系目标；分离视图内/跨视图学习。
- ③共同成员：软邻域，不是硬簇交集。
- ④机会校正：未见随机交集校准。
- ⑤粒度：随机游走步数，论文分析 3–7 步；不等于组大小—错配率定律。
- ⑥协议：原文主要是完整/缺失视图；后续工作适配到 NC。
- ⑦数字：原生 NC 20/40/60=NR；CorreGen Table 1 的重实现另列。
- ⑧[代码](https://github.com/XLearning-SCU/2024-AAAI-DIVIDE)。
- ⑨重叠：中高。软关系学习的基础骨干，不能把“利用邻域上下文”当新增点。

## A7 — MvCLN

Partially View-aligned Representation Learning with Noise-robust Contrastive Loss，CVPR 2021。

- 证据：[作者全文 PDF](https://xlearning-lab.com/assets/2021-CVPR-Partially-View-aligned-Representation-Learning-with-Noise-robust-Contrastive-Loss.pdf)，3.1–3.4、补充材料；正文。
- ①任务：PVP；已知配对作正例，随机负采样含同类假负。
- ②粒度：样本对，但目标是找到同类别对应，不要求同一实例。
- ③共同成员：不使用组交集。
- ④机会校正：不使用。
- ⑤粒度：负采样比例、距离间隔、两阶段切换。
- ⑥协议：已知对齐子集；评估含类别对齐率 CAR。
- ⑦数字：原生 20/40/60 未按未知错配协议核实；RMCNC Table III 重实现见数值文件。
- ⑧[代码](https://github.com/XLearning-SCU/2021-CVPR-MvCLN)。
- ⑨重叠：中。语义对应替代实例对应已有先例。

## A8 — PLCI【新增、最接近的机制先例】

Robust Multi-view Clustering against Imperfect Information，arXiv:2606.04343v1，2026-06-03；本次未核实正式录用，按预印本处理。

- 证据：[全文](https://arxiv.org/html/2606.04343)，2.2–2.4、Eq.(7)–(9)、附录 B/C；正文。
- ①任务：视图缺失+NC，也测 NC-only。
- ②粒度：实例可靠性+原型级关系。
- ③共同成员：是软加权共现 `U=Σγ_i q_iᵀq_i`。硬分配且 γ=1 时退化为成员交集计数。
- ④机会校正：Eq.(8) 用 `-U/||U||∞` 作 OT 成本，未见扣除随机共现基线。
- ⑤粒度：每视图 C 个语义原型；未见噪声驱动局部容量规律。
- ⑥协议：IVR/NCR 分开控制；需要读代码确认置换细节。不能把混合 IIR 当纯 NC。
- ⑦数字：NC-only Table 4 有 20/50/80%；20% Scene15 ACC 44.92；40/60%=NR。
- ⑧本次未找到作者公开实现；不捏造仓库、不冒称可直接复现。
- ⑨重叠：高。必须比较；有意义的待检验差别是“模型置信加权共现”与“机会校准的局部证据”。

## B1 — PVC

Partially View-aligned Clustering，NeurIPS 2020。

- 证据：[作者全文](https://xlearning-lab.com/markdowns/2020-neurips-partially-view-aligned-clustering/)，方法；正文。
- ①任务：已知部分配对，其余未知。
- ②粒度：实例，匈牙利匹配的可微替代。
- ③共同成员：不靠组交集。
- ④机会校正：无。
- ⑤粒度：实例排列矩阵，不是局部组容量。
- ⑥协议：利用可信配对训练；不能在 NC 比较中额外给它真实错误掩码。
- ⑦数字：原生 PVP 指标不作 NC；RMCNC Table III 重实现见数值文件。
- ⑧[代码](https://github.com/hi-zhenyu/PVC)。
- ⑨重叠：中。匹配先例，任务信息条件不同。

## B2 — CPSPAN

Deep Incomplete Multi-view Clustering with Cross-view Partial Sample and Prototype Alignment，CVPR 2023。

- 证据：[全文](https://arxiv.org/html/2303.15689)，3.2–3.3、Eq.(7)–(11)；正文。
- ①任务：视图缺失导致原型偏移。不是未知错误配对的 NC。
- ②粒度：样本+原型；以已观测配对为监督，优化原型排列。
- ③共同成员：按原型距离匹配，不是交集统计。
- ④机会校正：没有超几何校准。
- ⑤粒度：K 个原型；无容量—NC 关系。
- ⑥协议：缺失率实验，不能当错配率实验。
- ⑦数字：20/40/60 NC=NA。
- ⑧本轮未核实作者仓库地址，留空。
- ⑨重叠：中高。证明“跨视图原型匹配”不是空白；机会校准的证据来源不同。

## B3 — SMART

Semantic Matching Contrastive Learning for Partially View-Aligned Clustering，arXiv:2512.15396v1，2025-12-17；按预印本处理。

- 证据：[全文](https://arxiv.org/html/2512.15396)，III；正文。
- ①任务：PVP。
- ②粒度：实例语义图；对齐协方差，发现语义匹配。
- ③共同成员：无硬组交集。
- ④机会校正：协方差/相关系数不是成员交集检验。
- ⑤粒度：图阈值、批次，不是按 NC 调球容量。
- ⑥协议：已知对齐样本参与分布校准。
- ⑦数字：未知 NC 20/40/60=NA。
- ⑧[作者代码](https://github.com/THPengL/SMART)。
- ⑨重叠：中。进一步限制“放松实例一一匹配”的新颖性主张。

## B4 — OPTION

Optimal Transport–Guided Flow Matching for Incomplete and Unaligned Multi-View Clustering，ICML 2026。

- 证据：[PMLR 官方页](https://proceedings.mlr.press/v306/zhou26l.html)；**仅摘要**，PDF 获取失败。
- ①任务：理想、缺失、未对齐场景分别评估。
- ②粒度：条件流补全+Gromov–Wasserstein 启发的几何结构对齐。
- ③共同成员：摘要未说明，不能断言正文没有。
- ④机会校正：未核实。
- ⑤粒度：未核实；摘要未给容量—噪声定律。
- ⑥协议：未对齐不自动等于 NC；具体可信掩码和评估需核查。
- ⑦数字：20/40/60=U/U/U。
- ⑧[作者链接代码](https://github.com/TimoZhou1024/OPTION)。
- ⑨重叠：中（暂定）。作为结构匹配的新近边界，不冒称已精读。

## C1 — MGBCC

Multi-view Granular-ball Contrastive Clustering，AAAI 2025，20637–20645。

- 证据：[全文](https://arxiv.org/html/2412.13550v1)，Eq.(5)、跨视图关联、参数分析；正文及前轮作者代码核查。
- ①任务：通常对齐的多视图聚类。
- ②粒度：k-means 生成局部球，中心对比。
- ③共同成员：`|B∩D|/min(|B|,|D|)` 阈值关系。
- ④机会校正：没有随机交集基线。
- ⑤粒度：`K=max(floor(N/p),1)`；p 是平均容量而非每球恰好 p 人。分析 p=1/2/4/8/16；没有 NC 联动实验。
- ⑥协议：本次未发现其原文的人为错配率曲线；NoisyMNIST 的特征噪声不等于 NC。
- ⑦数字：NC 20/40/60=NR；干净 Table 3 的 p=2 ACC=72.63，不能当40%错配成绩。
- ⑧[代码](https://github.com/poensu/mgbcc_main)。
- ⑨重叠：高。应作为主要改造对象。代码负号/对数已核查，不再指控损失方向写反。

## C2 — HOARD

Hierarchical Cross-View Alignment for Multi-View Clustering via Decoupled Information Distillation，AAAI 2026。

- 证据：[正式 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/40136/44097)，Eq.(7)–(10)、原型对齐段；正文。
- ①任务：多视图信息对齐/蒸馏。
- ②粒度：粒球+原型，多层共享/特有表示。
- ③共同成员：Eq.(7) 与 MGBCC 同类交集/min-size，默认阈值0.1。
- ④机会校正：没有该规则的随机期望修正。
- ⑤粒度：多层结构已有；未见容量与 NC 率联动。
- ⑥协议：已检查实验未见 20/40/60 人为 NC 曲线。
- ⑦数字：NC 20/40/60=NR。
- ⑧本次未找到可确认的作者公开代码，不能仅由印刷公式判断实现错误。
- ⑨重叠：高。“先局部再原型”已有；本候选需证明关联规则增量。

## C3 — GFSAF

Views Attention Fusion of Granular-ball Fuzzy Representations Split for Improved Multi-View Clustering，AAAI 2026。

- 证据：[正式 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39556/43517)，GFCL、Eq.(1)–(6)；正文。
- ①任务：不同视图质量造成表示退化。
- ②粒度：球级对比，公共/互补表示拆分，注意力融合。
- ③共同成员：正文以跨视图同一样本、视图内重叠定义正球对；**未核实它等于 MGBCC 的具体比例公式**。
- ④机会校正：已检查部分未见。
- ⑤粒度：多粒度；生成细节及大小选择尚需完整代码审计。
- ⑥协议：噪声剥离针对表示噪声，不能等同错误配对。
- ⑦数字：NC 20/40/60=NR（已检查实验）。
- ⑧[代码](https://github.com/Lsy235/GFSAF)。
- ⑨重叠：中高。可借鉴保留互补信息；先不要叠加整个融合模块。

## C4 — MGCF-Net

MGCF-Net: a multi-view granular-ball contrastive fusion network for spatial transcriptomics clustering，Frontiers in Physiology，2026-10-07；用户指定的范围外例外。

- 证据：[官方全文](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1967219/full)，Cross-View Association、Eq.(10)–(13)；正文。
- ①任务：空间转录组 spot 聚类。
- ②粒度：spot 与粒球；结构图/特征图、融合图。
- ③共同成员：独立分球后的成员重叠，阈值生成跨视图关系。
- ④机会校正：该关联段未见。
- ⑤粒度：空间结构组织，不是 NC 驱动容量。
- ⑥协议：多图视图对应同一批 spot，不是打乱实体对应。
- ⑦数字：NC 20/40/60=NA。
- ⑧本次未核实公开代码。
- ⑨重叠：中高。进一步说明交集建球关系已有应用，但不构成 NC 效果证据。

## D1 — NCR

Learning with Noisy Correspondence for Cross-modal Matching，NeurIPS 2021。

- 证据：[作者正文](https://xlearning-lab.com/markdowns/2021-neurips-learning-with-noisy-correspondence-for-cross-modal-matching/)，方法；正文。
- ①任务：图文匹配中的错正配对。
- ②粒度：实例；GMM 对损失建模，双网络互教，软化对应。
- ③共同成员：无组交集。
- ④机会校正：低损失后验不是随机重叠检验。
- ⑤粒度：无球容量。
- ⑥协议：图文检索；与 MVC 的类聚合评价不同。
- ⑦数字：原文20/40/60细表U；DECL Table 1 有重实现结果，单独列。
- ⑧[作者研究页代码入口](https://xlearning-lab.com/researches/noisy-correspondence/)。
- ⑨重叠：低至中。可靠性估计先例，不能直接当聚类ACC基线。

## D2 — DECL

Deep Evidential Learning with Noisy Correspondence for Cross-modal Retrieval，ACM MM 2022；用户指定例外。

- 证据：[作者 PDF](https://xlearning-lab.com/assets/2022-ACMMM-Deep-Evidential-Learning-with-Noisy-Correspondence-for-Cross-modal-Retrieval.pdf)，§3、§4.3、Table 1；正文。
- ①任务：错配图文检索。
- ②粒度：实例/查询候选的证据分布，Dirichlet 不确定性，动态难负例。
- ③共同成员：不以分组交集建对应。
- ④机会校正：概率不确定性不等于超几何校准。
- ⑤粒度：无局部球容量。
- ⑥协议：随机打乱指定比例图像；Flickr30K、MS-COCO，另有自然噪声CC152K。
- ⑦数字：Table 1 的20/40/60%检索R@1摘录，禁止与聚类ACC比较。
- ⑧[代码](https://github.com/QinYang79/DECL)。
- ⑨重叠：低至中。为“不强制相信所有配对”提供方法参照。

## D3 — BiCro

Noisy Correspondence Rectification for Multi-Modality Data via Bi-Directional Cross-Modal Similarity Consistency，CVPR 2023。

- 证据：[全文](https://arxiv.org/html/2303.12419)，3.3、Table 1；正文。
- ①任务：图文配对有强弱匹配程度。
- ②粒度：样本+可靠锚点关系；Beta 混合模型选锚，双向相似性一致性估计软对应。
- ③共同成员：没有独立组交集。
- ④机会校正：没有随机成员重叠校准。
- ⑤粒度：锚点/近邻选择不是球大小噪声定律。
- ⑥协议：Flickr30K/MS-COCO人工错配，CC152K自然噪声；细节需与MVC区别。
- ⑦数字：本轮未转录原Table 1的20/40/60检索数值（U），不编造。
- ⑧[代码](https://github.com/xu5zhao/BiCro)。
- ⑨重叠：中。已经使用集合式上下文，不能再称kNN没有集合性质。

## D4 — Norton

Multi-granularity Correspondence Learning from Long-term Noisy Videos，ICLR 2024。

- 证据：[全文](https://arxiv.org/html/2401.16702)，3.2、附录D；正文。
- ①任务：视频片段—字幕及帧—词错配，含无法对齐内容。
- ②粒度：多层序列/局部内容；鲁棒时间OT、不可对齐桶、关键帧词匹配。
- ③共同成员：不依赖跨模态对象ID交集计数。
- ④机会校正：未见超几何校准。
- ⑤粒度：多语义层次，不是每球样本数随错配率变化。
- ⑥协议：时间异步和自然字幕噪声；不能直接视为MVC随机置换。
- ⑦数字：MVC20/40/60=NA；不从图估读检索曲线。
- ⑧[作者项目及代码入口](https://lin-yijie.github.io/projects/Norton)。
- ⑨重叠：中。支持允许部分关联/拒绝对应，不能声称该思想首次提出。

## E1 — polyClustR【历史先例】

Eason, Nyamundanda, Sadanandam，BMC Bioinformatics 19:182，2018；突破年份限制的必要历史项。

- 证据：[出版社全文](https://link.springer.com/article/10.1186/s12859-018-2204-4)，Methods: Hypergeometric test、Network analysis；正文。
- ①任务：同一批病人由不同算法得到的亚型分组整合。
- ②粒度：簇对及簇网络。
- ③共同成员：是。
- ④机会校正：**已有超几何重叠显著性+FDR；另一规则PMI明确为交集/min组大小**（Methods: Proportion of maximum intersection）。它与MGBCC所用比例同形，不能与经典pointwise mutual information混同。
- ⑤粒度：各算法自行选簇数，之后找社区；不是NC率驱动容量。
- ⑥协议：无跨视图错配注入。
- ⑦20/40/60：NA。
- ⑧[作者代码](https://github.com/syspremed/polyClustR)。
- ⑨重叠：高（统计操作），中（研究任务）。不得声称首次用超几何建立簇间边。

## E2 — Hubert & Arabie

Comparing partitions，Journal of Classification 2:193–218，1985。

- 证据：[出版社页面](https://link.springer.com/article/10.1007/BF01908075)；摘要/书目信息；细节由E3交叉核对。
- ①任务：分区评价，非训练方法。
- ②粒度：完整划分。
- ③共同成员：列联表/样本对计数。
- ④机会校正：ARI 的随机一致性校正。
- ⑤粒度：给定分区边缘，不提供球容量选择。
- ⑥协议：固定边缘随机分区模型，非部分错配生成器。
- ⑦20/40/60：NA。
- ⑧代码：非训练基线。
- ⑨重叠：统计背景。ARI 与单条组关联的显著性不是同一个量。

## E3 — Vinh, Epps & Bailey

Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance，JMLR 11:2837–2854，2010。

- 证据：[JMLR官方页及论文入口](https://jmlr.org/papers/v11/vinh10a.html)；摘要和校正框架。
- ①任务：比较聚类结果。
- ②粒度：全列联表。
- ③共同成员：是。
- ④机会校正：互信息的期望校正/标准化。
- ⑤粒度：强调簇数影响，不给NC容量定律。
- ⑥协议：随机分区背景模型。
- ⑦20/40/60：NA。
- ⑧代码：非神经训练基线。
- ⑨重叠：统计基础。可比较PMI/标准化残差，不能把“减期望”包装成新统计量。

## E4 — Hoeffding

Probability Inequalities for Sums of Bounded Random Variables，JASA 58(301):13–30，1963；DOI 10.1080/01621459.1963.10500830。

- 证据：[原论文扫描件](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)，第6节/定理4；扫描文字不可检索，本轮同时用E5原文核对定理归属，不能说完成逐行校勘。
- ①任务：有界随机量集中界。
- ②粒度：样本和。
- ③共同成员：可将其表达为0/1总体上的抽样和。
- ④机会校正：不是校正量，提供偏差概率控制。
- ⑤粒度：给定信号边际时的样本量要求，不决定语义混合后的最优球数。
- ⑥协议：有限总体无放回抽样仍可获相应界，不能假称样本独立。
- ⑦20/40/60：NA。
- ⑧代码：本目录 theory_check.py 仅验证我们具体置换恒等式，不是该定理的证明。
- ⑨重叠：理论工具，不作原创贡献。

## E5 — Bardenet & Maillard

Concentration inequalities for sampling without replacement，Bernoulli 21(3):1361–1385，2015。

- 证据：[原文](https://arxiv.org/pdf/1309.4029)，引言、Hoeffding reduction、后续有限总体界；相关正文。
- ①任务：无放回抽样集中。
- ②粒度：样本和。
- ③共同成员：可用于0/1交集样本。
- ④机会校正：无；浓缩界。
- ⑤粒度：样本数和有限总体修正；不保证粗球更好。
- ⑥协议：明确总体、样本量、抽样机制。
- ⑦20/40/60：NA。
- ⑧代码：非训练基线。
- ⑨重叠：支持严格处理置换依赖。

## E6 — Fisher/超几何官方实现说明

SciPy fisher_exact 官方文档；工具文档，不计为方法论文。

- 证据：[官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)，Notes；正文。
- ①任务：2×2表的独立性检验。
- ②粒度：组对。
- ③共同成员：`[[c,n-c],[m-c,N-n-m+c]]`。
- ④机会校正：固定边缘下超几何零分布，富集用单侧greater。
- ⑤粒度：检验随n、m、N变化；不能仅用交集比例。
- ⑥协议：完全随机关联零假设，不等于部分置换的完整生成模型。
- ⑦20/40/60：NA。
- ⑧代码：SciPy；本轮不安装依赖。
- ⑨重叠：现成工具。p值不是“关系为假的概率”，未显著也不等于语义不同。
