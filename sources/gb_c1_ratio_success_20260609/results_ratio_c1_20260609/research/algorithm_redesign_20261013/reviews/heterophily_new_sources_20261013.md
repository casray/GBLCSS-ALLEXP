# 异配表示、含噪标签与 X-only 推理：新增文献核查

核查日期：2026-10-13  
范围：HALF、GNN-CFGD、ES-MLP，以及 GraphACL、HLCL、HeterGCL、GREET 的任务边界。只据论文、出版社/会议页面和作者公开来源记录；不把摘要没有说明的细节推成事实。

## 阅读深度与访问边界

| 文献 | 阅读深度 | 可用证据边界 |
|---|---|---|
| HALF | 官方出版社摘要及其索引到的引言、方法与实验片段；未能完整打开正文 PDF | 可核验任务、核心机制、uniform/pairwise label noise 和局部同配使用；数据划分与 clean validation 协议未核实。出版社页面当前列入 *Information Sciences* 754、123714，并显示 2026-10-25 卷期日期；该日期晚于本次核查日期 2026-10-13，故这里只记录出版社所列卷期，不据此断言卷期已正式出版。 |
| GNN-CFGD | 官方出版社页摘要及其索引出的完整方法/实验章节片段，并核对作者 arXiv 摘要；未逐页核查附录 | 可核验粗/细粒度划分、peer GCN、clean-label-oriented linking 与伪标签机制；数据集清单和 clean validation 规则尚未逐项核对。 |
| ES-MLP | PMLR 官方摘要/书目信息；并核对 OpenReview 作者版 PDF 的索引文本首页，未逐页完整审读 | 可核验模型任务、edge-splitting、训练时图关系损失及“不用 edges 做 inference”的摘要主张；细节和 clean validation 规则未核查。 |
| GraphACL | NeurIPS 正式论文全文中与方法、实验协议和数据集有关的章节 | 可核验异配表征目标、下游线性评估、验证集调参；不是含噪标签鲁棒训练论文。 |
| HLCL | PMLR 官方摘要/书目信息 | 可核验其特征相似度分出的 homophilic/heterophilic 子图及双滤波对比目标；具体 probe 与 clean validation 协议未从可访问页面确认。 |
| HeterGCL | IJCAI 正式论文全文中与表示、语义分支、实验协议有关的章节 | 可核验严格无监督预训练、latent feature clustering 与 10/10/80 下游分类划分；不是噪声标签纠正任务。 |
| GREET | AAAI 正式论文全文中方法、实验协议与异配结果章节 | 可核验无监督预训练、特征/结构边判别、kNN 正则、下游逻辑回归及验证集选参；不是噪声标签纠正任务。 |

## 三篇新增文献

### HALF：问题组合与本方案高度相邻

HALF 的任务已经明确组合了**异配图、节点标签噪声和局部同配自适应**。方法先由节点特征构造同配/异配双结构，再按节点级 local homophily 融合低通与高通监督损失，以 memorization effect 区分含噪/较干净的已标注节点；另外用 dual-pass alignment loss 对齐未标注节点的两路预测。论文报告 uniform 与 pairwise label noise。由此，不能把“首次把异配图与噪声节点分类结合”或“首次使用局部结构可靠性处理异配图标签噪声”写成贡献。

与拟议方案的可辨区别是**可靠性信号的来源和输出部署边界**：HALF 以有噪标签上的低/高通损失及 local homophily 作融合信号，模型显式依赖图滤波；拟议方案希望先用无标签图表示构造多粒度球和跨球 kNN，再用权重有效票数、JS 稳定性给跨节点投票分布 q 加权，最后由 X-only MLP 预测。这个差异足以形成待验证的机制假设，但不是新颖性已成立的证明。必须直接把 HALF 纳入任务匹配对照，并验证球/kNN 权重是否改善 q 的正确性与纠错效果；若要主张识别目标节点是否翻错，须另行定义使用目标观测标签的检测机制。

**标签与验证协议：**训练过程明确使用观测标签来计算监督损失并分 clean/noisy；摘要和当前可访问的出版社片段没有充分说明 validation/test 标签是否保持干净、是否用干净验证标签调参。此点记为未核实，不能写成 HALF 使用或不使用 clean validation。

**图依赖：**双图结构、低/高通滤波与预测对齐明确进入方法的训练/监督信号，因此可以确认训练路径使用图结构。当前可访问摘要/片段没有明示预测阶段是否必须输入边；这里只能说尚未核明 HALF 有 edge-free 或 X-only inference，不能把推理依赖写成已核实事实。

### GNN-CFGD：噪声标签处理近邻，未确认为异配专用方法

GNN-CFGD 面向**标签稀疏与标签噪声下的半监督节点分类**。其粗粒度阶段对节点损失拟合 GMM，并并行训练 peer GCN 协同识别较干净标签；随后实施 clean-label-oriented link，将未标节点连到识别为较干净的已标节点；细粒度阶段按预测置信度进一步处理可疑已标节点和未标节点，并生成额外监督。

它与拟议方案共享“训练集里的节点可靠性控制/伪标签可用性”这一任务层面，但信号是**有噪标签上的模型损失与 peer prediction**，并且通过重构图传播监督。当前核查到的官方摘要与方法摘要没有把异配作为目标，也没有足够证据断言它在异配基准上专门验证；因此应称作噪声/稀疏标签对照，不应把它当成已解决异配标签噪声的直接先例。其路线使用 GCN 与重构邻接关系，不能支持 X-only inference 的主张。

clean validation 是否被假定或用于调参，在这轮可访问的官方摘要/章节片段中未核实。不要从常见噪声实验习惯反推其协议。

### ES-MLP：会议年份与出版年份不同；X-only inference 已有先例

ES-MLP 是监督式节点分类方法。它把 Graph-MLP 的邻居对比损失与 ES-GNN 的边拆分机制结合，在训练损失中区分 task-relevant 与 task-irrelevant 特征对/邻接关系。PMLR 摘要明确报告，推理时**不使用 edges**，并报告对多种 edge noise 的鲁棒性；本轮只核对摘要和作者版首页，未逐式核对其推理实现，因此将此表述限定为作者摘要主张。摘要没有声称对 noisy labels 鲁棒；这篇不能替代标签噪声基线。

年份应同时写清两个口径：OpenReview 作者版将论文标为 *Proceedings of the Third Learning on Graphs Conference (LoG 2024)*；PMLR 正式书目记录则把论文收入 PMLR 269，并标为 2025。严谨写法是“LoG 2024 会议论文，PMLR 269（2025）出版”，而不是把两个年份当作互相矛盾的版本。

对拟议方法的含义是：**训练时用图、推理时不使用边**的部署形式已有论文摘要先例；是否精确等同本方案的 X-only 输入实现，仍需核对 ES-MLP 全文和代码。可能的新点应落在如何由异配图上的多尺度球级邻域证据产生并验证标签可靠性权重，而不是单独主张 edge-free inference。

## 四种异配表征方法：目标相邻，标签噪声任务不同

| 方法 | 预训练/训练目标 | 标签如何出现 | 图是否进入表示/推理 | 与拟议方案的交集 |
|---|---|---|---|---|
| GraphACL (NeurIPS 2023) | 无监督非对称邻域对比；论证一跳上下文和二跳 monophily 可帮助异配节点表示 | 编码器无监督训练；冻结表示后训练线性分类器；论文明确按验证准确率选超参数 | GCN 编码器使用图，产生表示需要图 | 异配表示先验；不是噪声标签检测器。 |
| HLCL (UAI 2024) | 按节点特征 cosine similarity 识别同配/异配子图，用低通/高通滤波视图做对比 | 官方摘要称 self-supervised；具体 probe 与验证协议本轮未核实 | 两类图视图与频率滤波是核心方法 | 双结构异配表征已存在；不是标签噪声纠正。 |
| HeterGCL (IJCAI 2024) | 严格无监督、transductive 预训练；ANA 分离多阶邻居，另有 MLP 语义分支与潜在特征聚类 | 预训练不依赖人工标签；下游用 logistic regression，随机 10/10/80 划分；聚类评价使用真值类别数 | ANA 使用邻接结构；总体表示不是 X-only | latent feature K-means/GMM 软分配已存在；因此“聚类/软分组 + 异配表征”不宜单独作为新颖点。 |
| GREET (AAAI 2023) | 无监督边判别器从特征和结构估计同配/异配边，并以双通道表示/对比互相优化 | 表示预训练无标签；冻结表示后 logistic regression；超参数在 validation 选 | 使用判别边视图和低/高通编码，表示生成依赖图 | 尤其接近“异配表示 + kNN”：GREET 的对比损失已使用特征 kNN 找相似节点；它没有多粒度粒球或含噪标签纠正目标。 |

这些方法的标准节点分类结果主要评价**无监督异配表示对常规标注分类的可分性**，不是在训练标签被翻转后恢复干净标签的鲁棒性结果。下游线性/逻辑回归和验证集选参会使用标签，但不能因此把无监督编码器说成使用了噪声标签；反过来，也不能把它们的 clean-label downstream accuracy 当作 HALF/GNN-CFGD 式噪声鲁棒证据。

## 对拟议方案的新颖性边界

拟议流程为“无标签异配表示 → 多粒度粒球与跨球 kNN → 权重有效票数/JS 稳定性得到 alpha → X-only MLP 训练含噪节点分类”。拆开后，各组成已有明确邻近先例：异配自监督表示（GraphACL/HLCL/HeterGCL/GREET）；基于特征相似度的跨节点 kNN（GREET）；潜在特征聚类及软隶属（HeterGCL）；节点级异配结构自适应的噪声标签识别（HALF）；小损失/GMM 与置信度伪标签（GNN-CFGD）；训练时利用边、推理时去边（ES-MLP）。

因此可 defend 的潜在贡献不是这些零件的并列清单，也不是“异配 + noisy labels”这个任务组合，而应是：**球的几何构造与权重计算不读取标签，但跨球投票分布 q 会聚合其他训练节点的观测标签，因此整体可靠性信号不能称为“无标签”；需研究多粒度球是否能比单一 kNN、普通聚类或已有图信号更可靠地构造 q，并检验 q 的正确性、目标 Brier 与基于 q 的纠错 risk-coverage。权重有效票数和跨视图/扰动 JS 稳定性可以作为 q 的可靠性权重；若权重严格不读取本节点观测标签，在 uniform 独立翻转下，它不能单独识别本节点是否翻错。若另以 q 与本节点观测标签的分歧识别噪声，那是加入了本节点标签信息的另一种检测机制，需单独定义、评估和消融。**

当前证据只支持“存在机制差异”，不支持“已确认原创”。尤其要防止把 JS 稳定性、投票和 kNN 当作本身新颖；需把它们放到已有标签噪声方法和异配 kNN 表征方法中做强基线对比。

## 基准与任务协议：异配不等于标签噪声

异配节点分类基准回答的是：给定图结构、节点特征及训练标签，模型能否预测常规测试节点标签。异配是结构/标签关系属性；标签噪声是训练监督被污染的实验条件。基准里节点类别标签本身并不表示已经有 noisy-label corruption。作者发布的 heterophilous-graphs 套件包含 Roman-empire、Amazon-ratings、Minesweeper、Tolokers、Questions 等数据，覆盖不同图来源和分类任务；要研究标签噪声，需另行明确仅污染 train labels、噪声类型/比例、clean validation/test 规则和划分。对二分类/不平衡任务还应沿用合适指标，不能把所有数据集只报 accuracy。

对照时固定同一数据划分、噪声实现、种子、训练预算和验证规则。clean validation 可以作为调参条件，但必须声明；若研究主张是不依赖干净标签的 alpha，则 alpha 的构造、阈值选择和停止条件都不能窥视 clean validation/test 标签。用于机制解释的真实噪声身份只能在方法冻结后离线评估排序/校准，不能参与 alpha 生成。

## 建议

1. **把 HALF 作为首要直接基线。** 在同一异配数据、相同 train-only uniform/pair noise、相同 clean validation/test 规则下复现或公平适配；对比重点放在纠错质量和同一 edge-free 学生性能，而不只比不同论文表格中的精度。若方案另加入基于 q 与目标观测标签分歧的噪声检测器，再单独比较目标翻错识别质量。
2. **按信息依赖评估 q 与 alpha。** 若 alpha 不读取本节点观测标签或噪声身份，且噪声是 uniform 独立翻转，不要求 alpha 单独预测本节点是否翻错；应报告 q 的正确性、目标 Brier，以及按 alpha 覆盖率变化的 q-based correction/defer risk-coverage。若另用 q 与本节点观测标签的分歧作 noise detector，应把它列为加入本节点标签信息的独立机制，并单独报告其检测指标。报告权重有效票数、JS 稳定性及其组合的增量，干净/噪声身份只用于冻结方案后的事后评估。
3. **正面排除 GREET/HeterGCL 的替代解释。** 对照原始特征 kNN、GREET 式特征邻居、单尺度聚类、匹配簇数的 K-means/GMM、多粒度粒球和随机分组；否则结果不能归因于粒球。
4. **把图训练和 X-only 推理拆开报告。** 对齐 Graph-MLP/ES-MLP、MLP 与图模型，报告最终推理是否读 A、边扰动后性能和推理开销；不能把“推理不读边”写成首创。
5. **按任务选基准并报告协议。** 至少同时覆盖不同类型异配节点分类图，公开噪声类型、训练标签污染比例、是否有干净验证标签、split 和配对 seeds；错误检测指标与最终分类指标都要报告。

## 原始来源

- HALF, *Information Sciences*: [ScienceDirect article/abstract](https://www.sciencedirect.com/science/article/abs/pii/S0020025526006456), DOI: [10.1016/j.ins.2026.123714](https://doi.org/10.1016/j.ins.2026.123714)
- GNN-CFGD, *Neural Networks*: [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S0893608025002175); 作者预印本 [arXiv:2411.03744](https://arxiv.org/abs/2411.03744)
- ES-MLP: [PMLR 269 official entry](https://proceedings.mlr.press/v269/kohn25a.html); [OpenReview author-version PDF](https://openreview.net/pdf?id=BQEb4r21cm)
- GraphACL: [NeurIPS 2023 paper PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/3430bcc30cdaabd0bf6c5d0c31bda67c-Paper-Conference.pdf)
- HLCL: [PMLR 244 official entry](https://proceedings.mlr.press/v244/yang24a.html)
- HeterGCL: [IJCAI 2024 official paper PDF](https://www.ijcai.org/proceedings/2024/0265.pdf)
- GREET: [AAAI official paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/25573/25345)
- Heterophilous Graph Benchmark: [ICLR 2023 authors' paper preprint](https://arxiv.org/abs/2302.11640); [authors' official dataset repository](https://github.com/yandex-research/heterophilous-graphs)
