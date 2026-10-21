# Luna / max 阅读分工与详细提示词

共同提示词见00_READING_BRIEF.md；本文件保存各领域任务，供复查与继续。每域要求至少7篇原始论文达到方法+实验或理论+假设的阅读深度，抽象页不计完成。

## A：粒球生成与表示

代理luna_granular_review；gpt-6-luna/max。

主读候选：SGBGC、GBGC、MGNR、AW-GBGAE、GBC-CNN/Granular-ball Representation Learning for Deep CNN on Learning with Label Noise、ICDE2025伪标签无监督粒球划分与快速谱聚类、arxiv2603.29148大规模粒球图粗化。核准题录；缺方法全文需替换并保留缺口。

问题：√N究竟是初始球数、规模阈值还是邻居数？分裂用半径、均距、SSE、密度、标签或伪标签哪一个？球如何进入实际神经计算和梯度？是否真正减少点级计算？是否隐藏N²构图？训练标签mask与未标注球如何处理？噪声下纯度驱动的风险是作者证据还是我们的推断？生成效率、局部语义保持、表示学习之间哪些经验可迁移？提出1–3个具体设计改动，不能无限开方向。

## B：图粗化/凝缩与计算保持

代理luna_coarsening_review；gpt-6-luna/max。

主读候选：SCAL KDD2021、Loukas谱与割保持JMLR2019、FGC ICML2023、ConvMatch WWW2024companion、Message-Passing Guarantees NeurIPS2024、GCond ICLR2022、SFGC NeurIPS2023。SGBGC/GBGC/ACE/CPF/PLGC不重复算本域核心。

问题：原节点分区还是合成节点？hard/soft归属如何归一化？质量/度/特征如何加权？粗图传播与原图传播差异如何界定？粗图均值特征会否改变训练输入分布？训练时原图是否仍参与梯度或验证？成本是否只在多次复用时摊销成立？我们的Ac=SᵀAS、meanX、nTrain加权softCE、原图推理哪里需要修正？具体区分最低可行版本与后续复杂扩展。

## C：带噪图学习与鲁棒压缩

代理luna_noisy_review；gpt-6-luna/max。

主读候选8篇：NoisyGL、NRGNN、RTGNN、Divide and Denoise KDD2024、PLGC TMLR2026、RobGC TKDE2025、MRGC NeurIPS2025、ICGNN ICML2026。至少7篇实读。

问题：标签噪声、结构噪声、属性噪声、对抗噪声不可混淆；是否用额外干净标签（PLGC需重点看正式/预印本差别）；噪声矩阵未知时能否可靠读取标签差异；标签稀疏/不平衡如何影响分组？软CE聚合是否只是原损失汇总？区域选择与目标同批标签如何引入适应性？实际鲁棒收益来自关系、纠正还是损失？怎样避免把粗化与复杂纠错全部堆叠？

## D：异配表示、传播和粗化

第二批派发Luna/max。

建议核心：H2GCN/Beyond Homophily in Graph Neural Networks NeurIPS2020；GPR-GNN/Adaptive Universal Generalized PageRank Graph Neural Network ICLR2021；ACM/Adaptive Channel Mixing NeurIPS2022；A Critical Look at the Evaluation of GNNs under Heterophily ICLR2023；Understanding Heterophily for Graph Neural Networks ICML2024；Partition-wise Graph Filtering CPF KDD2025；ACE arxiv2607.21885作者注明UAI2026。若新近强方法比其中某篇更直接，可补充但至少7篇。

关键：异配并非结构噪声；同球混类比例不是唯一压缩质量；节点自身/邻居、低高通、不同阶关系分离为何有效；U/PU/P²U作为候选坐标会保留什么/丢失什么？关系分布模型的可辨识条件？普通GCN原图→粗图下降是否混淆骨干弱点？哪些指标优于边同配率？ACE具体是否纠正训练/推理失配、是否保留原图监督、效率成本？CPF真正在球/分区计算还是全节点计算？不直接把两论文损失叠加，给最小候选表示与骨干匹配建议。

## E：统计分裂、剪枝和预算分配

第二批派发Luna/max。

建议核心：Kass1980 CHAID；Hothorn/Hornik/Zeileis2006 Unbiased Recursive Partitioning: A Conditional Inference Framework；Domingos/Hulten2000 Mining High-Speed Data Streams(VFDT)；Meinshausen2008 Hierarchical testing of variable importance；Athey/Imbens2016 Recursive partitioning for heterogeneous causal effects（借鉴honesty而不迁移因果结论）；Hu/Rudin/Seltzer2019 Optimal Sparse Decision Trees；Lin等2020 Generalized and Scalable Optimal Sparse Decision Trees(GOSDT)。可根据直接相关性替换一篇为近期稳健分裂/剪枝研究，但至少7篇。CART专著可补充、不单独充当一篇论文。

关键：检验控制误报并非最优预算排序；p值不是收益；相同噪声标签选择树再估计分布的偏差；固定无标签树能解决哪部分选择偏差、不能解决哪部分？多重/层次检验的null是否嵌套，不能把变量重要性定理直接套到父子分布相等；XOR问题与贪心停机；诚实样本分割在极少训练标签下是否得不偿失；叶子预算与复杂度惩罚；剪枝DP vs贪心及大预算复杂度；高类别数样本量导致Hoeffding过保守，是否有可用效应量/方差修正替代？集中不等式在条件独立翻转而节点非iid时哪些步骤仍成立？最终为g+λr建议一个更有依据但不过度复杂的可检验目标。

## 主代理职责

### E领域检索后的选文调整

主代理检索到两个更直接连接“噪声—分裂准则”的来源，要求优先纳入：Ghosh/Manwani/Sastry, On the Robustness of Decision Tree Learning under Label Noise，arxiv1605.06296（正式PAKDD2017题录需核）；Wilton/Ye, Robust Loss Functions for Training Decision Trees with Noisy Labels，AAAI2024，DOI10.1609/aaai.v38i14.29516，官方PDF https://ojs.aaai.org/index.php/AAAI/article/download/29516/30856 。

因此E的核心七篇调整为上述两篇、CIT2006、VFDT2000、Meinshausen2008、Athey/Imbens2016、GOSDT2020。CHAID与OSDT可作追溯但不强制增加工作量。重点核对2017的对称噪声大样本排序保持、2024保守损失的早停/欠拟合权衡，不能将不纯度降低、统计显著性、有限预算计算价值混为一谈。新论文不因已有鲁棒树先例被否定，而应把它们变成必要对照。

不把代理结论直接视为事实。检查来源、阅读深度、论文去重和协议；针对会改变算法的断言回读原文；保存最终跨域汇总、设计v1→v2改动及未解决问题。不训练模型。

## 实际分配与范围变更记录

- A由luna_granular_review完成7篇。ICDE2025 PLGB-FSC仅获题录/摘要而缺可读全文，明确不计数；用方法与实验全文可读的GBG++替代。之后该代理只进行旧PRUNE与新骨架的窄范围交叉复核，结果另存A_granular/CROSS_CHECK.md。
- B完成后复用luna_coarsening_review做D；C完成8篇后复用luna_noisy_review做E。均保持用户要求的gpt-6-luna/max；未创建额外用户聊天。
- 期间代理因额度中断，在额度重置后从已保存文件恢复；D又发生一次模型capacity错误，重试恢复。没有把报错前未落盘的工作当成完成。
- 主代理额外读SIGN作为效率基线，不用它重复充当任一领域七篇名额。
- 最终交叉核查特别要求：MRGC的单列标签攻击不得被淡化为仅结构邻近工作；NE全称为negative exponential；连通性不能误写为所有表示空间粒球的硬要求；Pc=RPH不因非行随机就额外重归一化。
