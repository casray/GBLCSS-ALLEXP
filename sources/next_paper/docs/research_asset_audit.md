# 下一篇论文研究资产审计

> 审计日期：2026-07-14
> 项目根目录：`C:\Users\Admin\Desktop\实验8去噪5.19-`
> 当前 Git 分支：`codex/stage-b-gb-compactness`
> 审计原则：本文件只记录可由本地文档、配置、代码、原始日志或结果表追溯的事实。合理解释、尚未验证假设和负结果分别标注，不把单 seed 或小于标准差的提升升级为稳定结论。

## 0. 执行摘要

1. 第一篇论文的已确认主线是 **SoftDist / LCSS**：先用 label-free GRACE 获得冻结图表示 `H`，在 `H` 中构造 label-blind 粒球，再用 train noisy labels 形成局部 soft target，最后训练 GCN `[X||H]` 或 graph-free X-only MLP。第一篇的安全 claim 是“真实粒球分组相对同表示 CE 与 size-matched shuffled control 有稳定受控增量”，不是全面 SOTA。
2. 目标附件优先建议的“冻结对比表示 + 多局部粒球/粒椭球开放集拒识”已经完成 BANKING77 和 StackOverflow Pilot，并被判为 **no-go**。BANKING77 中 KMeans 在 7 组粒球超参 × 3 个聚类种子下稳定平替真球；补充 real GB 结构 seeds 0–4 后，full 仅 2/5 positive、ALL-safe 0/5 positive。StackOverflow 五个结构 seeds 均因每类约一球退化为 prototype。该方向不能以相同机制重新启动。
3. 2026-07-12 至 2026-07-14 已形成更强的第二篇候选：**LCSS soft target + soft-small-loss robust selection**。在 uniform 0.6 的六个数据集上，组合相对 `max(LCSS-only, Small-loss-only)` 的 paired synergy 全部 5/5 seeds 正向；Pair/Instance 0.4 也有多数据集五种子证据。该方向已经具备 Pilot 之后的实证成熟度，但与第一篇“标签噪声 + 局部软标签”故事接近，必须通过文献重复性与论文差异审计后才能定为下一篇主方向。
4. 当前工作区有大量用户修改和未跟踪实验资产。后续不得在现工作区直接切换/重置分支；实现新方向时应使用独立 worktree 或独立目录，并保留现有第一篇论文代码、日志和结果。

## 1. 已完成研究问题

### 1.1 第一篇论文最终解决的问题

**问题**：在图节点训练标签被 uniform/pair/instance noise 污染、验证与测试标签干净的协议下，能否利用 label-free 图对比表示中的局部结构，为下游分类提供可审计的局部软监督？

**核心方法**：

1. GRACE 仅使用 `X,A` 进行自监督预训练，获得冻结表示 `H`。
2. 在 `H` 上进行几何递归二分，得到约 64 个 label-blind 粒球。
3. 仅统计每个球内 train noisy labels，生成 multiclass-alpha 或固定 `beta=0.5` 的 soft target。
4. 训练新的 GCN `[X||H]` 或 X-only 两层 MLP；推理阶段 graph-free 变体不进行图传播。

核心实现：

- `external/NoisyGL/utils/ordinary_gb.py`
- `external/NoisyGL/utils/frozen_grace_targets.py`
- `external/NoisyGL/predictor/FrozenGraceGB_GCN_Predictor.py`
- `gb_graph_aware_minival_6_1/src/noisy_label_gb_grace/`

### 1.2 稳定实验结论

#### 已被实验支持

- SoftDist `[X||H]` 在 Cora、CiteSeer、Coauthor-CS 的 uniform 0.4、5-seed 协议中，真实球相对同 `H` 的 CE 与 shuffled control 有稳定配对增量。原始总结见 `项目核心总结_新窗口上下文.md`；方向历史见 `02_experiment_history_and_lessons 11.md`。
- 公平地给 NRGNN 同样的 `[X||H]` 后，SoftDist 在 Cora 基本持平，在 CiteSeer/Coauthor-CS 落后约 1.2–2pp。因此粒球增量可归因，但不能写成全面超过强图原生方法。
- X-only MLP 变体证明：下游不需要把 `H` 作为输入，也不需要推理时图传播；但有效 soft target 仍依赖冻结 `H` 中的粒球，raw-X 构球会塌缩。
- 真实局部共识相对 shuffled 的贡献可扩展到 Cora_ML、Attributed-Wiki、CiteSeerFull、Coauthor-Physics，但在部分数据集上会被 JoCoR、Small-loss、LogitClip 等强鲁棒策略吸收。证据见 `02_experiment_history_and_lessons 20.md`。

#### 合理解释，不是因果定理

- 下游越弱，粒球局部共识的边际价值越大；这一规律与 GCN `[X||H]`、GCN(X)、MLP(X) 的效果量顺序一致，但没有完成严格因果隔离。
- ball-clusterability 比规模、类数或同质性更接近适用域变量；现有相关性主要是小样本跨数据集诊断，不应写成一般定律。

### 1.3 当前论文 claim 上限

可以写：

- 在固定表示、固定 split/noise 协议和相同 backbone 下，真实局部粒球 soft target 相对 CE 与 shuffled control 提供稳定受控增量。
- 方法适用于部分 citation/coauthor 图与中高噪声区间，并存在清晰数据集和噪声率边界。

不能写：

- 全数据集标签噪声 SOTA；
- 粒球稳定优于 kNN/KMeans；
- 球质量高必然带来下游提升；
- 单次运行或小于 std 的提升具有显著性；
- raw-X 几何可替代冻结 GRACE `H`。

## 2. 已失败或不稳定方向

| 方向 | 设计动机 | 实际结果 / 失败原因 | 复用价值 | 下一篇处置 |
|---|---|---|---|---|
| GB-guided view generation / augmentation | 用粒球保护局部语义 | 多轮 long/formal 低于健康 GRACE；改变 augmentation distribution 导致 view drift | 负结果与诊断 | 排除 |
| GB-BGRL / GB-GBT 目标替换 | 用非 InfoNCE 目标修复 view mismatch | 未修复问题并引入更多变量 | baseline 代码 | 排除 |
| hard/full/ratio same-ball 正负关系 | 把同球当语义标签 | 关系过粗，误伤有效负样本与 clean edge | 作为反例 | 排除，除非有置信度与完整消融 |
| prototype pulling / direct relation regularization | 强化球内紧致 | 跨项目反复不稳，收益被简单局部方法吸收 | 损失模板 | 排除 |
| 纯粒球 edge denoise | 球关系识别噪声边 | 噪声命中低且 clean collateral damage 高 | edge scorer 诊断 | 不作主线 |
| GB-FN 新公式购物 | 改 adaptive/mass/cross-view 抑制 | 没有稳定超过 fixed top-k / no-ball top-k | conservative top-k 对照 | 停止调公式 |
| GRACE encoder 在 noisy CE 下微调 | 让下游反馈修正表示 | linear probe 约 0.804 降到 classifier 约 0.570 | 冻结策略证据 | 排除 |
| 动态 alpha / 二值 gate / early learning | 自适应控制软目标 | 从过度校正跳到几乎不校正，跨数据集不稳；模型反馈存在自强化 | gate 测试框架 | 排除 |
| ball consistency / 无标签节点扩展 | 扩大球级监督覆盖 | 球级目标分辨率不足，无稳定增益 | ablation | 排除 |
| 自适应粒度（变细） | 大图 64 球过粗 | Coauthor-Physics 64→186 球但 seed0 仅 +0.10pp；未过 +0.50pp 门 | 构球策略与测试 | 排除统一策略 |
| 中等票数粗球 | 降低小球投票方差 | holdout 平均不增益，Attributed-Wiki 明显回退 | vote-target 诊断 | 排除统一策略 |
| 粒椭球 / GMM responsibility | 建模各向异性与软归属 | 离线质量常不能转成 downstream；DBLP 5-seed 增量仅 +0.19pp、3/5 正 | 协方差、shrinkage、PCA 代码 | 只作边界/备选模块 |
| GEAF-style hard correction | 显式修正高置信噪声标签 | 多数数据集离线过门但 downstream 失败；仅 AmazonCom seed0 有局部正例 | hard correction 诊断 | 不作普适主线 |
| 图节点噪声检测 | 球 LOO 分数检测错标 | 绝对 AUROC 高，但被 H-kNN/GCN confidence 压制；集成增量约 0.001–0.003 | 条件增益诊断 | 排除主线 |
| 跨视角球冲突检测 | X-ball 与 H-ball 冲突发现独立噪声 | 冲突节点反而更干净，条件 AUROC 失败 | 多视角诊断 | 排除 |
| 免 K 图聚类 | 利用自适应球数替代预设 K | 比 true-K KMeans 低 10–15pp，且不具 wrong-K 鲁棒性 | 聚类控制 | 排除 |
| 冻结表征多粒球开放意图拒识 | 多局部边界覆盖多模态已知类 | BANKING77 被 KMeans 在 7×3 控制和 5 structural seeds 中平替；StackOverflow 五个结构 seeds 均退化为 prototype；encoder/model 仍仅 seed0 | 完整 OOD score/metric/条件增益工具链 | 相同 post-hoc 机制禁止复活 |
| robust teacher/plugin | 将 LCSS+small-loss 原样迁移到 GCN teacher | PubMed 大幅回退，整体平均为负 | teacher/student distillation 框架 | 关闭直接插件方案 |
| noisy-only 噪声率估计 | 从球内 LOO 分歧估计真实噪声率 | Cora/CiteSeer/CoraFull 绝对误差 0.120/0.132/0.233 | 失败基线 | 禁止包装成部署模块 |

## 3. 当前第二篇候选资产

### 3.1 LCSS soft target + soft-small-loss

**研究问题**：局部共识 target correction 与训练期 small-loss sample selection 是否提供互补而非重复的信息？

**已有证据**：

- uniform 0.6：Cora/CiteSeer/CiteSeerFull/CoraFull/PubMed/Coauthor-Physics 的 synergy 分别约 `+4.75/+3.63/+6.24/+5.02/+5.19/+3.27pp`，全部 5/5 paired seeds 正向；real-shuffled 同样全部 5/5。
- uniform 0.4：六个代表数据集平均 synergy 约 `+2.84pp`；其中四个数据集 5/5 正向，Amazon-Photo 与 Coauthor-Physics 只有 3/5，属于条件性中噪声收益。
- Pair/Instance 0.4：五数据集平均 synergy 约 `+1.49/+3.04pp`，real-shuffled 均 5/5 数据集稳定通过。
- PubMed 强基线：Pair/Instance 相对 JoCoR 约 `+3.78/+4.05pp`，均 5/5；uniform 0.6 只有 3/5，不支持三噪声全胜。
- clean-val 从 `{0,0.2,0.4,0.6}` 选择 forget rate 时，15/15 setting 的五种子均值在 true-rate oracle 1pp 内，全局 selected-oracle `+0.11pp`，相对 fixed 0.5 全局 `+1.26pp`。它是训练强度选择，不是噪声率估计，也不是 fully noise-rate-free。

**机制证据**：LCSS 主要增加 clean-class probability mass、改善 Brier 并重排 soft loss；它并非大量硬翻标签。Pair/Instance 中 selected-target accuracy 相对 Small-loss 平均提高约 3.60/6.60pp，final mask turnover 明显下降。该 trace 为 clean-dependent seed0 事后解释，不能替代五种子性能证据。

**创新风险**：与第一篇共享冻结 GRACE `H`、粒球局部 soft target、图标签噪声任务。下一阶段必须证明新论文的核心研究问题是“target correction 与 robust selection 的协同及高噪声适用域”，而不是“第一篇再加一个 small-loss 模块”；还必须检索相近的 soft-label co-teaching、prototype/cluster denoising、local consensus 和 graph noisy-label 方法。

关键资产：

- 方法：`external/NoisyGL/predictor/LCSSRobust_Predictor.py`
- robust-only：`external/NoisyGL/predictor/MLPNoiseBaselines_Predictor.py`
- 机制 trace：`external/NoisyGL/predictor/MLP34SelectionTrace_Predictor.py`
- clean-val：`external/NoisyGL/scripts/run_mlp35_cleanval_grid.py`
- 测试：`external/NoisyGL/tests/test_mlp28_softtarget_robust.py`、`test_mlp34_selection_trace.py`、`test_mlp35_cleanval_repeatability.py`
- 结果：`external/NoisyGL/runs/mlp31_noise_rate_5seed_20260712/`、`mlp33_u06_graphnative_sensitivity_noise_20260713/`、`mlp34_pubmed_rhohat_mechanism_20260714/`、`mlp35_cleanval_repeatability_20260714/`

## 4. 可复用代码

| 模块 | 文件 / 入口 | 输入 | 输出 | 修改成本与注意事项 |
|---|---|---|---|---|
| GRACE 编码器与表示提取 | `gb_graph_aware_minival_6_1/src/noisy_label_gb_grace/upstream_grace.py`；`extract_encoder_h` | 图 `X,A`、预训练模型 | L2-normalized `H` | 低；必须保持 label-free 和节点顺序哈希 |
| 数据集与固定 split | `.../data.py`；`load_dataset`、`apply_split` | 数据集名、train/val/test 比例、seed | data + masks | 低；新任务不得改变历史 split |
| 标签噪声注入 | `.../noise/label_noise.py` 与 NoisyGL loader | clean train labels、噪声类型/率、seed | noisy train labels + hash | 低；val/test 必须零污染 |
| 普通递归粒球 | `external/NoisyGL/utils/ordinary_gb.py`；`build_ordinary_granular_balls` | embeddings、max balls、min size、split threshold | `GranularBallResult(assignment, stats)` | 低；已有 fixed/sqrt、shuffled size-matched 控制 |
| 纯度自适应球 | 同文件 `build_purity_adaptive_granular_balls` | embeddings、train mask/labels | assignment + stats | 中；开放集/弱监督要警惕标签泄漏 |
| 粒椭球 | 同文件 `build_size_adaptive_ellipsoid_balls`；`frozen_grace_targets.py` 的 ellipsoid/PCA/shrinkage 函数 | assignment、features、train labels | Mahalanobis 权重 soft target | 中；高维小样本退化已有 fallback 与诊断 |
| soft target / 置信度 | `external/NoisyGL/utils/frozen_grace_targets.py` | assignment、noisy labels、train mask、beta/alpha | node-level soft targets + coverage/quality diagnostics | 低；实现大而复杂，新增机制宜封装为独立文件 |
| kNN/KMeans/GMM 对照 | 同文件 `build_*knn*`、`build_*kmeans*`、`apply_gmm_resp_*` | frozen features、labels、mask | size-matched local targets | 低；下一篇必须保留这些强平替对照 |
| graph-free robust student | `external/NoisyGL/predictor/LCSSRobust_Predictor.py` | X、LCSS soft targets、forget schedule | MLP logits、selection diagnostics | 低；已有 known-rate 与 clean-val runner |
| 多种子运行/汇总 | `external/NoisyGL/scripts/run_ssl_swap_downstream.py` 及 MLP28–35 runner/summarizer | config、datasets、methods、seeds | raw CSV/JSON、aggregate、paired deltas、gate | 低；优先复用现有哈希和完整性检查 |
| 开放集 split/BERT | `开放意图粒球新方向0630/code/oi_gb/data.py`、`bert_intent.py` | TSV 文本、known ratio、seed | known/unknown split、features、logits | 低到中；当前为 BERT closed-set 训练，不是 supervised contrastive encoder |
| OOD 基线与指标 | `.../oi_gb/baselines.py`、`metrics.py`、`conditionals.py` | train/eval features、logits、unknown labels | MSP/Energy/kNN/Maha/OpenMax/ReAct/ViM/GEN；AUROC/AUPR/FPR95/open metrics | 低；阈值函数支持只用 validation known scores |
| 粒球开放集 score | `.../oi_gb/gb_scores.py`、`posthoc.py` | frozen train/eval features | center/radius/density/purity/overlap/multi score | 低；已有 KMeans/shuffled control 和归一化 |
| 开放集离线平替探针 | `.../offline_probe_generic.py` | 已落盘 NPZ | 真球 vs KMeans 条件增益诊断 | 低；适合作为任何新域第一道 kill-test |
| 统计与报告 | `.../evaluation/summary.py`、各 `summarize_mlp*.py` | raw rows | mean/std、paired deltas、gate decisions | 低；显著性/置信区间需在下一阶段统一补充 |

## 5. 可复用实验经验

1. **同源吸收检查是第一门**：球在哪个表示空间构造，就必须与该空间的 kNN、KMeans、GMM、prototype 做平替性比较。
2. **real > shuffled 只证明真实分组有信息，不证明粒球最优**；还要比较 size-matched KMeans、随机中心和相同参数量原型。
3. **offline diagnostic 不能替代 downstream**。clean mass、purity、AUROC 或几何紧致度改善，可能完全不改变分类决策边界。
4. **seed0 只止损**。任何进入论文的性能结论必须使用相同 split/noise hash 的 paired 5-seed mean±std；局部机制 trace 要与性能证据分开。
5. **阈值和选择只能使用 train/validation**。测试 unknown 标签、clean train 标签不得进入构球、阈值选择或模型选择。
6. **公平资源控制**：对手必须获得同样表示、输入和训练协议；FINE/DivideMix-lite 等只能诚实标记为 MLP-adapted strategy baseline。
7. **预设 Gate 不后验放宽**：Pilot 失败时停止扩 seeds，不为局部有利数据集单独调参；若无带时间戳的正式预注册文件，只能称为固定审计 Gate。
8. **ROCm 非确定性按运行路径审计**：标准 runner 三重复最大 range 约 0.10pp；一次 trace 的 2.76pp 差异不能被泛化为所有结果的误差带。

## 6. 关键证据索引

- 项目总览：`项目核心总结_新窗口上下文.md`
- 最新交接：`99_session_handoff.md`
- 跨方向复盘：`03_cross_direction_meta_review.md`
- 第一篇主线历史：`02_experiment_history_and_lessons 11.md`
- 开放集 no-go：`开放意图粒球新方向0630/复盘_open-intent方向收口_20260630.md`
- 开放集 raw artifacts：`开放意图粒球新方向0630/runs/gb_score_conditional_gain_20260630/`
- 开放集五结构 seed 审计：`next_paper/runs/frozen_open_intent_structural_seeds_20260714/`
- 第二篇候选议程：`external/NoisyGL/docs/next_paper_research_agenda_20260712.md`
- 第二篇结果历史：`02_experiment_history_and_lessons 23.md` 至 `25.md`
- 第一篇当前稿：`external/NoisyGL/docs/paper_final_zh_v5_mlp23_breadth_20260710.md`

## 7. 审计缺口与进入下一阶段的条件

1. `01_literature_and_research_map.md` 截止 2026-06-22，不能替代面向下一篇的 2022–2026 专项检索；尤其需要核实 MOGB、BNO-GB、EliDecide、prototype/multi-prototype OOD 与 soft-label robust selection 的真实出版信息和官方代码。
2. 开放集 Pilot 的 encoder/model 和 known-class split 只有 seed0；新增五个 structural seeds 只验证构球随机性。两个数据集仍被强简单控制平替，可支持“相同 post-hoc 机制不推进”，不能支持一般性“粒球对开放集无效”或跨模型 seed 的总体性能结论。
3. 当前第二篇候选虽有强实验结果，但需要回答：相对第一篇是否是独立研究问题；相对已有 soft-label small-loss/co-teaching 是否有新机制；对比学习是否不可替换；粒球是否比 kNN/KMeans 局部共识有独立价值。
4. 现有报告以 mean/std 和 paired wins 为主；新论文最终报告还应统一增加置信区间或配对显著性检验，并预声明多重比较处理。
5. 当前工作区极度脏且包含嵌套 `external/NoisyGL` 与 ADB Git 仓库。新方向代码实现前必须建立隔离 worktree/目录，且不得覆盖现有用户改动。

## 8. 第一阶段结论

**资产审计结论：完成。** 本地证据已经推翻“直接把多粒域开放集识别作为默认主方向”的先验；该机制在当前项目中已有可追溯 no-go。下一阶段应以真实文献重复性审计为门，至少比较：

1. 开放集/开放意图的多局部边界或粒椭球方向（高重复风险，已有本地 no-go）；
2. 粒球原型记忆库（尚无本地 Pilot，粒球必要性风险高）；
3. 开放集半监督或未知污染下的粒球鲁棒伪标签对比学习（与第一篇重复风险中高）；
4. LCSS soft target + robust selection（本地证据最强，但论文差异与新颖性风险中高）；
5. label-free local-clusterability-aware contrastive representation（创新潜力高、成本和失败风险高）。

在 `literature_matrix.md` 完成并核实关键论文前，不创建复杂新模型、不重跑大规模实验。
