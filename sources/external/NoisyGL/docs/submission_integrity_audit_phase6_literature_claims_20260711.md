# 投稿完整性审计 Phase 6：文献与 Claim 核查

核查日期：2026-07-11。核查范围包括 graph-free/GNN-to-MLP、图标签噪声、粒球复杂度及正文参考文献。优先采用论文官方页面、会议论文集与出版社页面。

## 总体判定

- **研究设定成立，但“首次”必须收窄。** GSDN 已在 Cora/CiteSeer 上测试多噪声率的非对称训练标签噪声，因此不能声称“graph-free MLP 从未研究训练标签噪声”。当前可守的版本是：据我们所知，现有工作尚未把训练标签噪声作为中心问题，并在多噪声类型/噪声率下与标签噪声学习策略族进行统一骨干的系统比较。
- **部署动机中的延迟/数据依赖有一手文献支持。** GLNN 和 GSDN 均明确把多跳邻居抓取与在线推理延迟作为动机，并报告 MLP 推理加速。
- **隐私动机只能作为可能场景，不能写成隐私保证。** 图边确有隐私风险，但 X-only 推理本身没有差分隐私或防拓扑推断保证。应写“边可能因访问控制、数据隔离或隐私边界而不可用”，不要写“LCSS 保护隐私”。
- **当前参考文献表不能投稿。** 多条仍是中文占位符，SA-MLP venue 过时，MgCNL venue 错误，表 5 的 TFR/ProCon/GD2 等正文方法没有正式引用。
- **当前复杂度公式没有被引用文献证明。** 2019 GBC 论文支持其 GBkNN 框架“近 O(n)”与暴力 kNN 的 O(n^2)，不能直接推出当前 `ordinary_gb.py` 的递归 2-means 为 `O(N d log B_max)`。

## Claim 逐项核查

| 正文说法 | 判定 | 证据与修改 |
|---|---|---|
| GNN 在线推理受多跳邻居抓取延迟限制 | 支持 | GLNN 与 GSDN 的摘要/引言均明确说明 data dependency 与 neighborhood fetching latency。 |
| graph-free 方法均未研究训练标签噪声 | 不支持 | GSDN §5.2 明确注入 asymmetric train-label noise，并在 Cora/CiteSeer 上扫描噪声率。 |
| 训练标签噪声尚未在 graph-free 文献中被系统研究 | 暂可支持 | GSDN 只做单一非对称噪声和 GNN/蒸馏/MLP 对照，没有多噪声类型和完整 LNL 策略族比较。必须保留“据我们所知/尚未系统研究”。 |
| NOSMOG 的 robustness 属于标签噪声 | 不支持 | NOSMOG 的噪声轴是 node feature noise/adversarial feature augmentation；其输入还包含 DeepWalk 位置特征，并非严格 X-only。 |
| KRD 处理标签噪声 | 不支持 | KRD 以特征扰动下 teacher 信息熵不变性量化 knowledge reliability，噪声源不是训练标签。 |
| VQGraph 是结构 tokenizer/codebook 蒸馏 | 支持 | ICLR 2024 官方摘要明确为 VQ-VAE 式结构 tokenizer、离散 codebook 与 soft code assignments。 |
| SA-MLP 仅为 2022 arXiv | 过时 | 正式版本发表于 TMLR 2024；而且 student 读取结构输入，不能与严格 X-only 方法混为一类。 |
| LCSS/graph-free inference 提供隐私保护 | 不支持 | 无图推理减少部署端对邻接的访问，但未提供形式化隐私保证；训练所得模型仍可能携带图信息。 |
| 当前递归 2-means 的复杂度为 O(N d log B_max) [34] | 不支持 | [34] 分析的是其 GBC/GBkNN 框架。当前代码每次成功分裂前会扫描全部球并重新计算距离，最多执行 B_max-1 次；保守上界应包含 O(B_max N d) 扫描项及 2-means 迭代成本。 |
| 当前 kNN target 构造为 O(|V_L|^2 d) | 支持当前实现 | `build_knn_soft_targets_from_features` 显式构造 `train_features @ train_features.T`，因此其时间/内存分别约为 O(|V_L|^2 d)/O(|V_L|^2)。这不是所有近邻算法的普遍下界，只是当前精确实现。 |

## 建议使用的 Gap 句

> Existing graph-free and GNN-to-MLP methods show that graph-induced knowledge can be transferred to MLPs for efficient inference. Their robustness studies, however, primarily concern feature perturbations, missing features, label scarcity, or unreliable teacher predictions. GSDN additionally reports an asymmetric train-label-noise stress test, but training-label noise is not treated as the central problem through multiple noise models and systematic comparisons with learning-with-noisy-label strategies. To our knowledge, this intersection remains insufficiently studied.

中文版本：

> 现有无图推理与 GNN-to-MLP 工作已证明图结构知识可以迁移到 MLP，其鲁棒性研究主要关注特征扰动、特征缺失、标签稀缺或教师预测可靠性。GSDN 进一步附带测试了非对称训练标签噪声，但没有在多种噪声模型下将训练标签噪声作为中心问题，并与标签噪声学习策略族进行系统比较。据我们所知，这一交叉设定仍缺少系统研究。

## 参考文献 P0 修正清单

1. `[25] CT` 不能保留占位符。完整信息：W. Pan, W. Wei, F. Zhu, Y. Deng, *Enhanced Sample Selection with Confidence Tracking: Identifying Correctly Labeled Yet Hard-to-Learn Samples in Noisy Data*, AAAI 2025, 39(19):19795-19803, DOI `10.1609/aaai.v39i19.34180`。
2. `[27] SA-MLP` 更新为 TMLR 2024 正式版本，不再只写 arXiv 2022。
3. `[28] PGKD` 补全作者 Taiqiang Wu 等及 COLING 2025 正式出处。
4. `[30] ProGMLP` 补全 AAAI 2026, 40(29):24088-24096, DOI `10.1609/aaai.v40i29.39587`。
5. `[31] SALE-MLP` 补全 H. Pal, S. Malik, R. Patel, A. Malhotra, IJCAI 2025, 6003-6011, DOI `10.24963/ijcai.2025/668`。
6. `[32] Demystifying...` 补全 Z. Yu, M. Lin, W. Li, Z. Yin, S. Xiao, S. Lu, AAAI 2026, 40(33):28005-28013, DOI `10.1609/aaai.v40i33.40025`。
7. `[33] RKD-MLP` 是 ICLR 2023 投稿记录而非已确认正式论文。不要把它当正式发表成果；可删除，或明确标注 OpenReview submission。
8. `[35] MgCNL` 的 venue 不是 IEEE TII。正式刊载为 IEEE Transactions on Automation Science and Engineering, vol. 22, 2025；DOI `10.1109/TASE.2024.3469000`（2024 online-first）。
9. `[36] MgBIF` 补全 Fir Dunkin 等，*Advanced Engineering Informatics* 60 (2024) 102425, DOI `10.1016/j.aei.2024.102425`。
10. `[37] Granular Ball Sampling...` 正式卷期为 IEEE TNNLS 34(4), 2023, 2144-2155；DOI `10.1109/TNNLS.2021.3105984`。可注明 2021 online-first，但不能只写 2021 而省略正式卷期。
11. 表 5 正文中的 TFR、ProCon、GD2、CLNode、CRGNN 必须各有引用。至少补入：
    - TFR: Z. Wang et al., *Learning from Graph: Mitigating Label Noise on Graph through Topological Feature Reconstruction*, CIKM 2025, 3261-3270, DOI `10.1145/3746252.3761185`；
    - ProCon: K. Li et al., *Leveraging Peer-Informed Label Consistency for Robust Graph Neural Networks with Noisy Labels*, IJCAI 2025, 5598-5606, DOI `10.24963/ijcai.2025/623`；
    - GD2: K. Li et al., *GD2: Robust Graph Learning under Label Noise via Dual-View Prediction Discrepancy*, NeurIPS 2025。

## 复杂度的安全改写

不要继续写：

> 递归 2-means 的时间复杂度为 O(N d log B_max) [34]。

建议写：

> 在本实现中，粒球数量由常数上限 B_max=64 约束。每次接受分裂前需要扫描当前分区并对候选球执行 2-means；因此保守复杂度可写为 O(B_max N d + T_2means)，其中 T_2means 取决于 k-means 迭代次数和初始化次数。当 B_max 视为固定常数时，该离线构造对 N 近似线性。相比之下，本文使用的精确 kNN 对照显式构造训练节点相似度矩阵，时间为 O(|V_L|^2 d)、内存为 O(|V_L|^2)。

这一区分只比较“当前实现”，不能概括所有近似 kNN、索引树或 ANN 方法。

## 主要一手来源

- GLNN: https://arxiv.org/abs/2110.08727
- GSDN: https://arxiv.org/abs/2210.02097
- NOSMOG: https://arxiv.org/abs/2208.10010
- KRD: https://proceedings.mlr.press/v202/wu23m.html
- VQGraph: https://proceedings.iclr.cc/paper_files/paper/2024/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html
- NoisyGL: https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html
- 2019 GBC: https://doi.org/10.1016/j.ins.2019.01.010
- SA-MLP TMLR: https://openreview.net/forum?id=MZ2kKZc8m7
- PGKD: https://openreview.net/forum?id=3aMK3rg377
- SALE-MLP: https://www.ijcai.org/proceedings/2025/668
- ProGMLP: https://ojs.aaai.org/index.php/AAAI/article/view/39587
- Demystifying GNN-to-MLP: https://ojs.aaai.org/index.php/AAAI/article/view/40025
- TFR: https://github.com/eaglelab-zju/TFR
- ProCon: https://www.ijcai.org/proceedings/2025/623
- GD2: https://papers.nips.cc/paper_files/paper/2025/hash/c132c02176577c4319a878f6417a331a-Abstract-Conference.html

## 结论边界

本轮搜索支持“该交叉设定缺少系统研究”，但不能数学意义上证明不存在任何未被索引、同期投稿或非英文工作。因此所有 first claim 必须保留 `to our knowledge`，并在附录提供逐篇筛选矩阵，而不是写成无条件事实。
