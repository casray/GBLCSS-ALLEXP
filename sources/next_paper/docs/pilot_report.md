# 冻结表示多粒度局部边界 Pilot 报告

> 实验执行日期：2026-06-30
> 独立复算与统计审计日期：2026-07-14
> 五结构随机种子审计日期：2026-07-14
> 最终状态：**No-Go；停止“冻结表示 + post-hoc 粒球局部边界/条件分数”方向，不进入联合训练或大规模实验。**

## 1. 详细结论

本 Pilot 要验证的必要条件不是“某个 GB 分数偶尔高于 MSP”，而是：在完全相同的冻结 BERT 表示、训练集、已知类划分和代表数量下，**自适应真粒球是否提供 size-matched KMeans 不能替代的开放集信息**。

答案是否定的，而且两个数据集呈现两种互补失败：

1. **BANKING77 是显著被平替**。固定 G2 `dist+margin` 分数的真球 AUROC 为 0.7741，三次 KMeans 初始化为 `0.7867±0.0086`，差值 −1.26pp；paired stratified bootstrap 95% CI 为 `[−2.08, −0.41]pp`，`p=0.002`。在 MSP、Prototype、kNN 都判为相对安全的 ALL-safe 子集上，真球为 0.7525，KMeans 为 `0.7882±0.0176`，差值 −3.57pp，95% CI `[−5.14, −2.05]pp`，`p<0.001`。7 组粒球超参 × 3 个 KMeans seeds 中，真球在 `dist+margin` 与 ALL-safe 均为 **0/7** 胜出。
2. **StackOverflow 是结构退化**。25% known 只有 5 个已知类，真球最终仅 6 个，约等于每类一个中心。真球和三个 KMeans controls 的 `dist+margin` AUROC 都是 0.85704637，ALL-safe 都是 0.85926631，paired bootstrap 差值的 95% CI 在数值精度内为 `[0, 0]`、`p=1.0`。这不是“统计功效不足”，而是实现输出逐样本等价；shuffled control 在 ALL-safe 上甚至略高（0.8617）。
3. **五个结构随机种子没有改变结论**。在固定 encoder/model seed0 NPZ 上改变真实粒球内部二分 seed=`0..4`，并配对 KMeans seeds=`13..17`、shuffle seeds=`1013..1017`。BANKING77 full 的 real−KMeans 为 `−0.0110±0.0215`、仅 2/5 positive；ALL-safe 为 `−0.0402±0.0446`、0/5 positive，结构 seed bootstrap CI 为 `[−0.0798,−0.0130]`。StackOverflow 五个 seeds 的 real 与 KMeans 仍逐 seed 完全相等，shuffle 均值反而略高。两个数据集的结构稳定性 Gate 均失败。

因此，被实验支持的结论是：**在本 Pilot 的冻结表示与 post-hoc 协议下，自适应粒球没有显示相对等量 KMeans 的独立局部边界价值。** 该结论足以触发固定停止条件，但不能外推成“所有端到端粒球开放集方法都无效”。

## 2. 研究假设、粒球角色与动机

### 2.1 研究假设

- H1（条件信息）：当 MSP、Prototype、kNN 已经给出判断后，粒球半径、边界 margin、密度、purity proxy、覆盖与重叠仍能识别被强基线漏掉的 unknown。
- H2（结构必要性）：真粒球必须在 `dist+margin` 与 ALL-safe 两个固定口径同时超过 size-matched KMeans；否则收益来自普通局部聚类/中心距离，而不是粒球结构。
- H3（跨数据集最低复现）：若 BANKING77 出现平替，StackOverflow 作为类内多模态更强的有利场景应至少显示同方向增益，才值得扩展到 4 datasets × 5 seeds。

### 2.2 粒球扮演的角色

粒球是冻结表示空间中的**局部语义区域与 soft boundary confidence**：每个球包含 center、radius、size、purity proxy、density 和 overlap 等统计。它不作为 hard positive、hard negative filter 或强 prototype pulling 模块，也不修改 BERT 表示。

### 2.3 方法动机

单中心 ADB 难以表达多模态类内结构；MOGB 表明多粒度边界在联合 representation learning 中可能有效。本 Pilot 刻意冻结表示，分离“局部边界本身”与“表征学习”的贡献，并加入等量 KMeans 与 shuffled controls，防止把普通多中心收益误归因于粒球。

## 3. 代码、配置与可追溯资产

### 3.1 代码

- 主运行：`开放意图粒球新方向0630/code/run_oi1_smoke.py`
- StackOverflow 运行：`开放意图粒球新方向0630/code/run_oi1b_stackoverflow.py`
- 粒球与 controls：`开放意图粒球新方向0630/code/oi_gb/gb_scores.py`
- 基础 OOD scores：`开放意图粒球新方向0630/code/oi_gb/baselines.py`
- 公平阈值与指标：`开放意图粒球新方向0630/code/oi_gb/metrics.py`
- 通用复算：`开放意图粒球新方向0630/code/offline_probe_generic.py`
- 7×3 稳健性：`开放意图粒球新方向0630/code/offline_probe_B_hparam.py`
- 五结构种子审计：`next_paper/open_intent_pilot/structural_seed_audit.py`
- 五结构种子配置：`next_paper/open_intent_pilot/configs/structural_seed_audit.json`

2026-07-14 审计没有修改上述代码或重新训练 BERT，只在原始 NPZ 上复算 controls、置信区间和测试。

### 3.2 共同配置

| 项目 | 设置 |
|---|---|
| encoder | `bert-base-uncased` |
| fine-tuning | 只训练最后一层，3 epochs |
| max length | 64 |
| batch / eval batch | 64 / 128 |
| known ratio | 0.25 |
| model/data split seed | 0 |
| GB default | max balls/class 16；min ball size 8；split improvement 0.05 |
| KMeans controls | 与真球逐类、逐配置等球数；random states 13/14/15 |
| threshold | 只在 known validation scores 上取目标 5% FPR 的分位数 |
| test unknown labels | 只用于 AUROC/AUPR/FPR95/open-classification 评测 |

### 3.3 数据规模

| 数据集 | known classes | train-known | val-known | test | test known / unknown | 真球数 |
|---|---:|---:|---:|---:|---:|---:|
| BANKING77 | 19 | 2,107 | 235 | 3,080 | 760 / 2,320 | 150 |
| StackOverflow | 5 | 3,000 | 500 | 6,000 | 1,500 / 4,500 | 6 |

### 3.4 原始结果

- BANKING77：`开放意图粒球新方向0630/runs/gb_score_conditional_gain_20260630/banking77_r25_seed0_20260630_172920/`
- StackOverflow：`开放意图粒球新方向0630/runs/gb_score_conditional_gain_20260630/stackoverflow_r25_seed0_20260630_201825/`
- 每个完整 run 均包含 `config.json`、`features_logits_seed0.npz`、`score_metrics.csv`、`conditional_gain.csv`、`correlations.json`、`seed0_report.md`。
- 独立复算：`next_paper/runs/frozen_open_intent_pilot_audit_20260714/offline_probe_log.md`
- 统计结果：`next_paper/runs/frozen_open_intent_pilot_audit_20260714/statistics.json`
- 五结构种子 raw/summary/JSON/log：`next_paper/runs/frozen_open_intent_structural_seeds_20260714/`

## 4. 强基线与公平协议

### 4.1 基线

统一冻结表示上计算：MSP、ODIN-temperature approximation、Energy、Prototype、kNN、Mahalanobis、GEN、ReAct+Energy、OpenMax-tail approximation、ViM-residual approximation，以及 random score。报告明确：ODIN/OpenMax/ViM 是 fixed-feature transfer approximations，不是原论文端到端忠实复现。

粒球 controls 包括：

- `GB_shuffled`：保持球规模结构但随机分组；
- `GB_kmeans_proto`：逐类、等球数 KMeans；
- 离线 `KMeans seeds 13/14/15`：对每组真球超参重建 size-matched controls；
- 7 组 GB granularity/split hyperparameters。

### 4.2 无测试泄漏审计

代码数据流显示：

1. train/val/test 从同一已知类 split 产生；val 仅含已知类。
2. 所有 baseline scores 与 GB scores 都基于相同的冻结 train/eval features。
3. `threshold_for_known_fpr(val_scores[name], target_fpr=0.05)` 只使用 known validation scores。
4. `test_open_labels == -1` 在 score 生成后才转成 `y_unknown`，只传给评测函数。
5. 球拟合只接收 `train_features`、`train_labels` 和 train predictions，不接收 test labels。

### 4.3 正式拒识阈值与 ALL-safe 条件诊断的边界

正式 open-set classification 的 rejection threshold 只从 known validation scores 估计，目标 known FPR 固定为 5%。离线 `offline_probe_generic.py` 与五结构 seed 审计中的 ALL-safe 不是该拒识阈值：它分别在当前 test score 分布上取 MSP、Prototype、kNN 的中位数安全侧并求交集。

该子集构造不读取 test unknown labels，但使用 test score distribution，因此属于 **label-free transductive conditional diagnostic**。test unknown labels 只在子集固定后用于 AUROC；ALL-safe 结果不得写成 validation-derived threshold，也不参与最终 open classification 决策。

## 5. 固定成功与失败标准

本节 Gate 可由 2026-06-30 原始 probe 代码与日志追溯，但 `experiment_plan.md` 是 2026-07-14 结果后重建的审计协议，不将其冒充为正式事前注册。

| Gate | 成功标准 | 失败标准 | 实际结果 |
|---|---|---|---|
| G1 条件信息 | ALL-safe AUROC 有实质识别力，且不是靠退化到单一中心距离 | 组合被无信息分量稀释，或仅 `s_dist` 有效 | BANKING77 组合需要退化到 distance；StackOverflow strong baselines 更强 |
| G2 真球必要性 | real GB 在 `dist+margin` 和 ALL-safe **同时严格高于** size-matched KMeans | 任一口径 `real ≤ KMeans` | 两数据集均失败 |
| G3 扩展资格 | G2 通过后才做 4 datasets × seeds 0–4 | G2 失败立即停止 | 未扩展，遵守 stop rule |

## 6. 基础开放集结果

### 6.1 BANKING77 seed0

| 方法/分数 | AUROC | AUPR | FPR95 | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| GB distance component | **0.7999** | 0.8948 | 0.5829 | 0.3101 | 0.3813 |
| ViM residual approximation | 0.7524 | 0.8754 | 0.6566 | 0.2675 | 0.3635 |
| kNN | 0.7453 | 0.8700 | 0.6539 | 0.2656 | 0.3620 |
| GB equal-weight composite | 0.7330 | 0.8705 | 0.7355 | 0.3006 | 0.3657 |
| KMeans composite | 0.7285 | 0.8701 | 0.7053 | 0.2873 | 0.3640 |
| Shuffled composite | 0.6318 | 0.8230 | 0.8592 | 0.2312 | 0.3437 |

单个 distance component 看似最好，但它就是最近局部中心距离；决定粒球必要性的离线等量 KMeans probe 表明该信号不稳定属于真球：默认配置只高 KMeans 0.19pp，7 组超参仅 3/7 略胜，且固定联合分数与条件子集均 0/7 胜出。

### 6.2 StackOverflow seed0

| 方法/分数 | AUROC | AUPR | FPR95 | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| GEN | **0.8880** | 0.9382 | 0.2920 | 0.4500 | 0.4728 |
| ODIN-T approximation | 0.8857 | 0.9376 | 0.2980 | 0.4597 | 0.4776 |
| MSP | 0.8851 | 0.9361 | 0.2953 | 0.4587 | 0.4802 |
| kNN | 0.8776 | 0.9368 | 0.3247 | 0.3915 | 0.4351 |
| GB distance component | 0.8565 | 0.9182 | 0.3473 | 0.3165 | 0.3935 |
| GB composite | 0.8469 | 0.9194 | 0.3813 | 0.4913 | 0.4836 |
| KMeans composite | 0.8469 | 0.9194 | 0.3813 | 0.4913 | 0.4836 |
| Shuffled composite | 0.8444 | 0.9177 | 0.4600 | 0.4118 | 0.4352 |

这里真球不仅没有超过强 logits baselines，而且 composite 与 KMeans 完全一致；这与“6 个球覆盖 5 类”的结构退化一致。

## 7. 真球 vs KMeans 多随机种子与统计

### 7.1 默认配置

| 数据集/口径 | real GB | KMeans mean±sample std（3 seeds） | 差值（real−KMeans） | paired bootstrap 95% CI | p |
|---|---:|---:|---:|---:|---:|
| BANKING77 full `dist+margin` | 0.7741 | 0.7867±0.0086 | −0.0126 | [−0.0208, −0.0041] | 0.002 |
| BANKING77 ALL-safe | 0.7525 | 0.7882±0.0176 | −0.0357 | [−0.0514, −0.0205] | <0.001 |
| StackOverflow full `dist+margin` | 0.857046 | 0.857046±0.000000 | 0.0000 | [≈0, ≈0] | 1.000 |
| StackOverflow ALL-safe | 0.859266 | 0.859266±0.000000 | 0.0000 | [≈0, ≈0] | 1.000 |

bootstrap 为同一 test set 上的 known/unknown 分层 paired resampling，3,000 次。KMeans mean±std 衡量 control 初始化，不是 encoder/model seed 方差；两者都不能冒充跨训练 seed 的泛化不确定性。

### 7.2 BANKING77 超参消融

| GB 配置 | balls | real/KMeans DM | real/KMeans distance | real/KMeans ALL-safe |
|---|---:|---:|---:|---:|
| max16/min8/split0.05 | 150 | 0.7741/0.7867 | 0.7999/0.7980 | 0.7525/0.7882 |
| max8/min8/split0.05 | 127 | 0.7734/0.7793 | 0.7949/0.7964 | 0.7552/0.7612 |
| max32/min4/split0.05 | 262 | 0.7713/0.7875 | 0.8083/0.8115 | 0.7497/0.7837 |
| max16/min16/split0.05 | 66 | 0.6997/0.7390 | 0.7720/0.7709 | 0.5932/0.6902 |
| max16/min8/split0.01 | 152 | 0.7742/0.7859 | 0.7997/0.7981 | 0.7530/0.7873 |
| max16/min8/split0.20 | 19 | 0.6610/0.6610 | 0.7116/0.7116 | 0.5571/0.5571 |
| max64/min4/split0.01 | 271 | 0.7708/0.8009 | 0.8108/0.8142 | 0.7530/0.7973 |

### 7.3 五个结构随机种子

| 数据集/口径 | real GB mean±std | 配对 KMeans mean±std | real−KMeans mean±std | positive seeds | seed bootstrap 95% CI | exact sign-flip p | shuffle mean±std |
|---|---:|---:|---:|---:|---:|---:|---:|
| BANKING77 full | 0.7697±0.0245 | 0.7807±0.0071 | −0.0110±0.0215 | 2/5 | [−0.0303, 0.0032] | 0.4375 | 0.6787±0.0067 |
| BANKING77 ALL-safe | 0.7375±0.0516 | 0.7777±0.0133 | −0.0402±0.0446 | 0/5 | [−0.0798, −0.0130] | 0.0625 | 0.5896±0.0160 |
| StackOverflow full | 0.857046±0.000000 | 0.857046±0.000000 | 0.000000±0.000000 | 0/5 | [0, 0] | 1.0000 | 0.857091±0.000221 |
| StackOverflow ALL-safe | 0.859266±0.000000 | 0.859266±0.000000 | 0.000000±0.000000 | 0/5 | [0, 0] | 1.0000 | 0.862507±0.000602 |

这里的 seed=`0..4` 只改变真实粒球内部递归二分；每个 seed 分别配对一个等球数 KMeans 和一个严格保持真实球大小的 shuffle。seed bootstrap 使用 10,000 次重采样，BANKING77/StackOverflow seeds 分别为 20260716/20260717；exact sign-flip 使用全部 \(2^5\) 种符号组合。五个 seed 的离散检验最小双侧 p 为 0.0625，因此 BANKING77 ALL-safe 的 sign-flip p 没有低于 0.05；这不构成正方向证据，因为均值为负、0/5 positive 且 bootstrap CI 完全小于 0。

## 8. 测试与复现状态

- `py -3.12 -m pytest 开放意图粒球新方向0630/tests -q --ignore=.../test_stackoverflow_runner.py`：**13 passed，3 warnings**。
- `py -3.12 -m pytest next_paper/open_intent_pilot/tests/test_structural_seed_audit.py -q`：**9 passed，3 warnings**，包含真实 GB/KMeans/shuffle 的 synthetic-NPZ 集成测试和端到端五文件输出测试。
- 五结构 seed 正式命令：`py -3.12 next_paper/open_intent_pilot/structural_seed_audit.py --source-workspace C:\Users\Admin\Desktop\实验8去噪5.19- --config next_paper/open_intent_pilot/configs/structural_seed_audit.json --output-dir next_paper/runs/frozen_open_intent_structural_seeds_20260714`。
- 完整 pytest collection 在当前 Python 3.12 环境因缺少可选依赖 `transformers` 而在 import 阶段停止；`test_stackoverflow_runner.py` 的断言没有执行。原始 BERT runs 和 NPZ 已存在，本次离线 probe 可完整复现。
- 首次调用裸 `python` 返回 9009，根因是 PATH 优先命中 WindowsApps 占位符；复算均使用明确的 `py -3.12`。没有修改项目 PATH 或安装包。

## 9. 模型多 seed 限制与停止逻辑

本 Pilot 仍只有 encoder/model seed0，**没有模型 5-seed mean±std**。新增实验提供的是五个 frozen-representation structural seeds，验证构球随机性，不是训练随机性。原执行计划采用顺序 kill-test：seed0 必须先在最有利场景通过真球必要性 G2，才允许扩到多模型 seed；现在结构 seeds 0–4 又显示 BANKING77 仅 2/5 full positive、0/5 ALL-safe positive，StackOverflow 仍逐 seed 等价，因此继续训练 model seeds 1–4 不能回答“当前粒球结构是否有独立价值”，只会违反停止规则。

因此本报告的 No-Go 是对一个**必要条件**的否决，不是跨随机初始化估计总体平均效果。若未来改变表示学习机制、球构造或任务，必须重新预注册并运行独立多模型 seeds；不得复用本报告的 CI 声称新方法稳定。

## 10. Claim 审计

### 已被实验支持

- 固定 BERT 表示 + post-hoc 真粒球在 BANKING77 的固定 DM/ALL-safe 口径显著弱于等量 KMeans。
- 在五个结构 seeds 上，BANKING77 没有稳定超过 KMeans；ALL-safe 0/5 positive，结构 seed bootstrap CI 完全小于 0。
- 在 StackOverflow 五个结构 seeds 下，真球均退化到近单原型结构，与 KMeans outputs 等价。
- 单个 `s_dist` 的高 AUROC 不能证明粒球结构必要性。

### 合理但仍是解释

- BANKING77 的 GB 几何被普通局部聚类吸收；StackOverflow 的球数过少导致退化。
- MOGB 的正效果更可能来自 HRL 联合塑造表示，而非冻结空间中的多球边界本身。该解释与 MOGB 消融方向一致，但本 Pilot 没有复现 MOGB 全模型，不能写成因果定论。

### 尚未验证

- 端到端、流式、资源约束 coreset 或联邦场景中的粒球是否有效。
- 其他 known ratios、其他 encoder、CLINC/SNIPS 或模型 seeds 下是否存在不同现象。
- 粒球是否能以效率/压缩而非精度贡献成立。

### 已被负结果削弱

- 冻结表示上继续调 `dist/margin/density/purity/overlap` 组合；
- 把最近球中心距离包装成新的粒球 OOD score；
- 不带 size-matched KMeans/k-center 对照的多局部边界 claim；
- 因 seed0 某个单分量较高就直接进入联合训练。

## 11. 最终决策

**决策：放弃当前 D5“GB Conditional Calibration / 冻结表示局部边界精度”方向。** 不扩 4 datasets × 5 model seeds，不进入联合 BERT–GB training，也不追加同类分数组合。

允许的后续只有研究问题发生实质改变的路线：

1. Stable-GB Reliability：先做零训练条件增益诊断，证明跨 view 稳定性提供 kNN/confidence 之外的信息；
2. Transition-aware LCSS：只先做已知 synthetic `T` 的 oracle 上限，过 1pp 门后才讨论估计；
3. GB-Coreset 降级为效率型工程分析，因为 PatchCore/cDNP/SupLID 已覆盖 coreset OOD 主问题；
4. 保留 LCSS + soft-small-loss 作为效果型 backup，但必须正视 MgBIF/MgCNL/GBmooer/ProCon 的重复风险。

在任一新方向的最小 Pilot 显示明确、稳定、不可被强 control 平替的价值之前，禁止进入联合训练或大规模实验。
