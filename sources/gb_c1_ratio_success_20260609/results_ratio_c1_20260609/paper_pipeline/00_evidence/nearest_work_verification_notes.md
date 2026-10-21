# Nearest-work verification notes

核验日期：2026-07-27  
目标期刊：Elsevier *Information Sciences*  
用途：限定 GB-LCSS 的新颖性、资源边界和必须补充的直接基线。本文档不是计量学系统综述；没有逐数据库完成全量检索前，禁止使用“首次”“尚无工作”或“从未研究”等排他性表述。

## 1. 直接机制邻近工作

| 工作 | 原始来源 | 已核验事实 | 与 GB-LCSS 的关系 | 稿件和实验边界 |
|---|---|---|---|---|
| PosteL, AAAI 2026 | AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/39321；官方代码: https://github.com/ml-postech/PosteL | 由训练节点标签与图上的邻域标签共现统计估计后验软标签；官方实现包含 MLP；no-IPL 代码在构造训练目标时屏蔽 validation/test labels；完整 IPL 版本使用 validation/test 节点的模型伪标签更新全局统计。 | 直接否定“把图结构转成软目标，再训练 MLP”本身可作为本文新颖性的说法。GB-LCSS 的剩余区别是：冻结自监督表征中的标签无关分组、并列安全局部含噪分布、球数上限受控的共识单元及统一噪声协议。 | 主表使用 `postel_mlp_noipl_rm`；只能称“官方代码忠实的 PosteL 目标 + 项目 MLP/统一协议”。IPL 单独列为资源更强的 transductive 补充，不与 no-IPL 混排。 |
| ALS, SDM 2023 | SIAM: https://epubs.siam.org/doi/10.1137/1.9781611977653.ch7；arXiv: https://arxiv.org/abs/2108.13555 | 先传播标签得到局部邻域分布，再用可学习的全局类别关系矩阵在线细化软标签；目标包含 KL 正则与 pacing。原任务动机是大规模子图批训练的 label bias，而不是错误标签防御。 | 与 GB-LCSS 同属“图结构辅助软标签”，但 ALS 的软标签和类别关系在线学习；GB-LCSS 的分组先冻结，学生训练不反向改变局部关系单元。 | `lp_mlp_softdist_beta05` 不是 ALS。必须实现完整传播、可学习关系矩阵、KL 与 pacing，标记为“机制忠实的统一 MLP 适配”，不能写成官方 ALS 数值。 |
| SALS, arXiv 2021 | https://arxiv.org/abs/2112.00499 | 用邻居类别分布和均匀分布构造结构感知平滑目标，原文以 GNN 为主要骨干，并未给出官方 MLP 主结果。 | 直接覆盖“邻域标签分布生成软目标”，但默认依赖已知邻居标签与同质性语义；GB-LCSS 先在冻结表征中定义局部关系，再只聚合 noisy training labels。 | 实现名必须是 `sals_mlp_adapted`；只使用 train-labeled neighbors，无此类邻居时回退均匀分布并记录。必须称 MLP adaptation，不得称官方 SALS-MLP。 |
| GSDN, arXiv:2210.02097 | https://arxiv.org/abs/2210.02097 | 纯 MLP 框架在训练时以图邻域为自蒸馏先验，推理只使用节点属性；附录包含不同标签噪声率实验。 | 同时覆盖“训练用图、推理 X-only”和标签噪声评测，是部署属性上最直接的先行工作之一。区别不能写成“GSDN 未研究噪声”，而应落在监督构造机制和统一协议。 | 历史 `gsdn` 是 mechanism-inspired adaptation。新实现 `gsdn_faithful_core` 是 **equation-grounded common-protocol adaptation**：按 v5 Eq. (3)/(7)/(8) 建模，共享主干和双头；将项目的成对有向 arcs 规范为唯一无向关系，Eq. (7) 仍显式计算正反方向，Eq. (8) 对每个有标注端点传播其 noisy label。每批按唯一关系数归一化，而论文 Algorithm 1 写全局 \(1/B\)，所以不得称 equation-faithful、exact reproduction 或官方代码。 |
| GLNN, ICLR 2022 | https://openreview.net/forum?id=4p6_5HBWPCw | GNN 向 MLP 蒸馏，学生推理不依赖图，并讨论 transductive/inductive 场景。 | 说明“部署时移除图”不是本文首创。GB-LCSS 基础版不需要监督 GNN 教师；蒸馏扩展属于资源更强层。 | 摘要不足以确认其错误标签协议；不得声称 GLNN 完全没有噪声实验。 |
| NoisyGL, NeurIPS 2024 Datasets and Benchmarks | https://proceedings.neurips.cc/paper_files/paper/2024/hash/436ffa18e7e17be336fd884f8ebb5748-Abstract-Datasets_and_Benchmarks_Track.html | 提供图标签噪声统一数据、划分、预处理和方法接口。 | 本项目复用其合成标签噪声评测思路，同时增加 X-only 部署约束和冻结表征局部共识。 | NoisyGL 的 benchmark 首创主张不能转化为本文任务首创主张。 |

## 2. 粒球与节点分类的直接工作

| 工作 | 原始来源 | 已核验事实 | 公平性/定位风险 |
|---|---|---|---|
| SGBGC, AAAI 2025 | 论文: https://doi.org/10.1609/aaai.v39i12.33404；代码: https://github.com/mxjun17/GBNC | 直接把粒球计算用于图节点分类；球划分读取标签和 purity；论文在 CiteSeer 上报告 5%、10%、15%、20% 标签噪声。官方 `gb_division.py` 只屏蔽 test labels，没有同样屏蔽 validation labels；`main.py` 对粗化球重新划分 train/validation，并最终在原图上用 GNN 测试。 | 必须正面引用，不能再声称粒球与节点级图学习尚未结合。建议区分 `SGBGC-author` 与 `SGBGC-strict`；只有严格屏蔽 validation/test clean labels 的版本可进入公平跨资源比较。其推理依赖图，不与 X-only 方法混成一个同资源排名。 |
| SCGNN, arXiv:2605.02617v2 | https://arxiv.org/abs/2605.02617 | 以原始特征和训练标签/purity 构造粒球，形成 anchor augmented graph 与 LCC；默认分类器是图依赖 GNN。文中噪声评估主要对应伪标签误差，并非本项目的训练标签人工注入协议。 | 是“粒球 + 图节点分类”的直接近期工作，但标签使用方式、噪声定义、训练资源和部署输入均不同。只能逐项比较，不能用其作者表格冒充统一噪声协议结果。 |
| SAGHL, EAAI 2026 online/in press | https://doi.org/10.1016/j.engappai.2026.115428 | 使用拓扑驱动、标签无关的粒球超图构造并加入对比正则；模型推理依赖图/超图。论文声称缓解标签噪声，但本轮未核验其是否使用与本文一致的人工错误标签协议。 | 直接削弱“标签无关粒球图结构是本文独有”的宽泛表述。区别应限定到冻结自监督表示、tie-safe noisy-label 聚合、X-only 学生及统一多噪声评测。 |
| PGBSF, *Information Sciences* 2024 | https://doi.org/10.1016/j.ins.2024.120831 | 粒球采样结合 GCN 节点分类，主要报告 label-rate，而不是错误训练标签噪声。 | 必须列入粒球图学习相关工作；不能把 label scarcity 写成 label corruption。 |
| GB-CGNN, arXiv:2603.29148 | https://arxiv.org/abs/2603.29148 | 使用粒球进行可扩展图粗化和节点分类，最终分类器依赖 GCN/图。 | 说明粒球还可作为图粗化单元；GB-LCSS 不以粒球替换原图或作为最终图分类器。 |
| Granular-ball computing classifiers, *Information Sciences* 2019 | https://www.sciencedirect.com/science/article/pii/S0020025519300106 | 定义以球替代点的分类框架和 GBkNN/GBSVM。 | 其复杂度结论不能自动转移到本实现；本稿必须按实际候选重扫、排序、`n_init` 和迭代报告成本。 |

## 3. 统一协议实现矩阵

| 实现名 | 训练图 | 推理图 | 标签资源 | 身份表述 | 主比较资格 |
|---|---:|---:|---|---|---|
| `postel_mlp_noipl_rm` | 是 | 否 | noisy train labels；不使用 val/test clean labels 或伪标签 | official-code-faithful target + project MLP | 同资源主表 |
| `gsdn_faithful_core` | 是 | 否 | noisy train labels；唯一无向关系上的邻域自蒸馏 | equation-grounded common-protocol adaptation；independent, not official code | 同资源主表 |
| `als_mlp_rm` | 是 | 否 | noisy train labels；传播与可学习类别关系 | mechanism-faithful project-MLP adaptation | 同资源主表 |
| `sals_mlp_adapted` | 是 | 否 | noisy train labels；train-labeled neighbors only | mechanism-faithful project-MLP adaptation | 同资源主表 |
| `postel_mlp_ipl_rm` | 是 | 否 | noisy train labels + val/test model pseudo-labels | stronger-resource transductive variant | 补充表 |
| `SGBGC-strict` | 是 | 是 | noisy train labels；val/test clean labels严格屏蔽 | strict protocol port | 跨资源表 |
| `SGBGC-author` | 是 | 是 | 按作者代码资源流 | author-protocol reproduction | 仅补充/协议敏感性 |

## 4. 投稿定位结论

1. PosteL 已直接覆盖“图结构后验软标签 + MLP”。本文的新颖性不能建立在这条宽泛管线上。
2. GSDN 已直接覆盖训练期利用图、推理期 X-only，并触及标签噪声。本文不能声称这一资源组合此前无人研究。
3. SGBGC、SCGNN、SAGHL、PGBSF 和 GB-CGNN 证明粒球已经用于节点分类、图增强或图粗化。本文不能声称“首次把粒球引入节点级图学习”。
4. 当前最窄且可防御的区别是：在冻结的自监督图表征中进行标签无关分组，只以 noisy training labels 形成并列安全局部软监督，并训练 X-only 学生；所有比较采用固定 split/noise/seed 和可追溯协议。
5. 项目内部历史 kNN 诊断在若干数据集上高于默认粒球，因此 novelty 不能建立在“粒球天然优于一般局部邻域”上。粒球应被定位为球数上限受控的局部共识接口实例。
6. 新的 PosteL/GSDN/ALS/SALS 同资源网格完成前，v30 不得保留“七个正式同骨干对照中全部最优”或把早期最小索引并列规则下的表格当作当前并列安全主表。

## 5. 静态实现状态（2026-07-28）

- `postel_mlp_noipl_rm`、`als_mlp_rm` 与 `sals_mlp_adapted` 已通过目标单测和静态审查；暂停前的单种子 ROCm smoke 只证明功能路径，不是论文性能证据。
- `gsdn_faithful_core` 已通过独立 Eq. (7)/(8) oracle、无向关系规范化、两端标签传播、训练标签隔离和 X-only inference 测试。主任务复核命令 `py -3 -m unittest tests.test_gsdn_faithful tests.test_standard_runner_baselines` 为 31 tests / OK。
- 最近工作编排器已移除非 dry-run `NotImplementedError`，其 25-command fail-fast、exact resume、不可变 raw、375+125 evidence 和统计输出契约通过 14 tests / OK。
- 真实输入 CLI dry-run 生成 25 个 planned commands，显式固定 ROCm Python 路径，`subprocess` 调用数为 0；GSDN 新身份精确匹配，旧 `equation_faithful_reimplementation` 身份被测试明确拒绝。
- 以上内容是正式训练前的静态门禁。用户解除暂停后，GSDN ROCm smoke
  和 500 行正式网格均已完成；最终性能证据见第 7 节。

## 6. 尚待完成

- 用数据库、检索式、截止日期、纳入/排除规则补齐系统检索日志；
- 运行 `SGBGC-strict`，并把作者协议版本限制在补充材料；
- 对 SCGNN/SAGHL 是否提供可移植官方实现及其噪声协议作代码级复核；
- 将历史 Graph-MLP/GLNN/FF-G2M/KRD/VQGraph 表降为资源分层诊断，
  不与当前五方法共同协议表混合排名。

## 7. 正式共同协议结果（2026-07-29）

正式目录为
`../../external/NoisyGL/runs/nearest_work_full_grid_20260728/`。

完整性与谱系：

- 25/25 个命令块完成，25/25 个不可变 CSV 的 SHA-256 均匹配；
- 500/500 个原生最近工作结果，失败、缺失、重复均为 0；
- 375 个 PosteL/ALS/SALS 结果与 125 个冻结 GB-LCSS v2 结果组成
  500 行独立证据；
- 加入 125 个 GSDN 结果后形成 625 行五方法联合证据，每种方法 125 行；
- 125/125 个 dataset--noise--seed 组通过 split/noise/污染计数审计；
- 第二次 exact resume 启动 0 个训练子进程，25 个命令全部为
  `skipped_exact_complete`。

完整 25-setting 配对结论：

| 对照 | GB-LCSS 平均差 (pp) | 正向设置 | 至少 4/5 seeds 正向 | 25 项族内 Holm |
|---|---:|---:|---:|---:|
| PosteL-noIPL | +5.202 | 24/25 | 24/25 | 18/25 |
| ALS | +7.215 | 25/25 | 25/25 | 24/25 |
| SALS | +4.202 | 24/25 | 23/25 | 13/25 |
| GSDN | -0.698 | 10/25 | 8/25 | 2/25 |

GB-LCSS 对 GSDN 的五设置平均差在 Cora、CiteSeer、CoraFull、
Coauthor-CS 和 Amazon-Photo 上分别为 -0.96、+2.69、-5.07、+2.17 和
-2.33 个百分点。五方法联合排名中，GB-LCSS 在 10/25 个设置取得最高
均值，GSDN 在其余 15/25 个设置取得最高均值。

允许表述：

- GB-LCSS 在共同协议下稳定优于 PosteL、ALS 和 SALS；
- GB-LCSS 与 GSDN 的相对性能具有明显数据集依赖性；
- 两者在不同数据集上的优势区间与组级离线共识和逐节点在线自蒸馏的
  机制差异相容，但当前实验不能证明该解释的因果性。

禁止表述：

- GB-LCSS 优于全部同资源 X-only 最近工作；
- GB-LCSS 是 25 个设置中的整体最佳方法；
- GSDN 的较高方差可以抵消其 15/25 个设置上的均值优势；
- 将项目 GSDN 适配写成 official、exact、faithful reproduction。
## V32 method-status amendment (2026-07-29)

- PosteL, ALS, and SALS retain their verified adapted-method status and
  enter the current 25-setting comparison grid.
- The old `gsdn` implementation is retained only as a **provisional
  adapted implementation** in the temporary Table B and validation-label
  sensitivity experiment.
- `gsdn_faithful_core` is not used by v32. Its earlier 25-setting result
  must not be cited, ranked, or combined with current headline counts.
- A new GSDN implementation will replace the provisional row in a later,
  independently frozen revision.
