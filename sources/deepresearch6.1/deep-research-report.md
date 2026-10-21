# GB 与图对比学习六轮实验的失败收缩与下一步决策

引注说明：F1=`02_experiment_history_and_lessons.md`，F2=`02_experiment_history_and_lessons 2.md`，F3=`02_experiment_history_and_lessons 3.md`，F4=`02_experiment_history_and_lessons 4.md`，F5=`02_experiment_history_and_lessons 5.md`，F6=`02_experiment_history_and_lessons 6.md`。文中 `Lx-Ly` 为本地文件行号。

## 适用范围限制

本报告用于指导“粒球计算与图对比学习结合”的下一轮实验。除非作为必要 baseline、ablation 或强对照，**暂不考虑没有和粒球结合的对比学习实验**。因此，x-space percentile / degree-protect edge denoise 这类 non-GB 正信号只能作为边界参考或对照项，不能作为本轮推荐主线，也不能替代 GB 相关机制问题。

## Executive Summary

- 目前最明确应停止继续投入的，不是某个小变体，而是一整类“让 GB 直接当强语义机制”的路线：GB 深度参与 view generation、GB-BGRL/GB-GBT、direct same-ball positive/prototype/consistency、pure GB membership edge denoise，都已经出现了“机制能激活，但 formal/long protocol 不赢健康 baseline”或“sanity 都不过线”的共同失败模式。它们更适合作为 negative result / motivation，而不是主线继续调参。 (F5 L5-L14, L20-L26, L57-L96, L584-L671; F1 L319-L405, L427-L521)

- 当前最可保留的 **GB 主线** 只有一个：**GB-FN-GRACE v2 topk-soft**。但它的现实位置不是“已经成功的方法”，而是“最保守、破坏最小、尚未被彻底证伪的候选”。现有提升只有 Cora `+0.0014`、CiteSeer `+0.0010`、PubMed 基本持平，而且与 GRACE 的 std 明显重叠；更关键的是，**no-ball topk 几乎复现了同样收益**。 (F5 L195-L276; F2 L188-L300, L648-L681)

- 在本报告的 GB 范围限制下，当前最值得保留的正向信号是 **RDCL** 和 **GB-FN**。RDCL 在 PubMed 上有最强信号，但 random / shuffle 控制已经明显削弱了“语义粒球是收益来源”的解释；x-space percentile 在噪声图上更强，但主信号来自直接 edge similarity，不属于本轮候选主线，只能作为 non-GB 强对照或边界参考。 (F6 L251-L337; F5 L456-L554)

- 你现在最需要做的不是发明新公式，而是做 **高信息增益的否证实验**：  
  一是 RDCL 的 fixed / size-matched partition 控制，判断 strongest positive 到底是不是“语义分组”带来的；  
  二是 GB-FN 的 no-ball / candidate-overlap / per-seed paired delta 机制包，判断 GB 是否真的提供了 no-ball 之外的独立信息。 (F6 L549-L593; F1 L1052-L1067; F5 L1177-L1210, L1247-L1275)

- 多个文件都给出同一个警告：**不能把 protected_precision、candidate Precision@k、ball activation、mean_num_balls 等诊断指标，直接等同于最终下游有效性。** 高 purity 在 RDCL 上没有保证更高 F1；GB ball-topk 的 candidate precision 也没有稳定超过 no-ball topk。 (F6 L341-L359; F5 L813-L855; F5 L90-L96)

- 如果目标是写论文，最安全的写法不是“GB 显著提升 GCL”，而是更弱、更诚实的表述：**GB 也许只能作为低侵入的结构先验或候选范围 prior；当前 strongest positive 中，很多收益并不能唯一归因于 GB。** 任何忽略 std overlap、no-ball 对照、shuffle/random/fixed 控制、以及跨数据集不足的写法，都会明显过界。 (F5 L218-L276, L941-L959, L1125-L1143; F6 L298-L337, L723-L755)

- 如果只剩一周算力，最值得先做的不是再调 GB-FN 公式，也不是继续改 view，而是 **RDCL 的 fixed/size-matched partition 控制实验**。因为它会最快告诉你：你手里“最像论文主结果”的信号，到底是真机制，还是 generic local regularizer。这个实验失败，能直接收缩大半叙事空间；成功，才值得进入下一轮扩展。 (F6 L549-L593, L687-L747)

## Stop List

**应停止继续投入的方向**

| 方向 | 核心假设 | 实验结果支持了什么 | 实验结果削弱了什么 | 主要失败原因 | 当前决策 | 只有在什么新证据下才允许复活 |
|---|---|---|---|---|---|---|
| GB-guided view generation / augmentation control | 用 same-ball / cross-ball 结构重写双视图，会比随机 GRACE 增强更“语义一致” | 粒球确实能激活，不是退化成单球；短程局部轮次可接近或偶尔略高于弱 baseline | 一旦进入 formal / long protocol，GB-GCL 反复低于健康 GRACE；decay、freeze、anchor-explore 只能缓解，不能翻盘 | GB 直接改变了有效的 view distribution，导致长训练动态被破坏；这是 **role mismatch**，不是“没构出球” | **abandon** | 必须先出现 protocol-matched、multi-seed、至少两数据集超过 GRACE 的结果，同时给出 view-drift / useful-edge damage 诊断，否则不应复活。 (F5 L20-L24, L57-L96; F5 L895-L913) |
| Direct same-ball positive / prototype / consistency regularization | same-ball、prototype、consistency 可以直接当额外正关系 | 分支都能激活，工程上可实现 | prototype/positive/consistency 在 Cora/CiteSeer 的 sanity 中没有稳定 validation gain，Cora 还明显下降 | same-ball 作为 **hard positive** 太粗，容易伤害 instance discrimination；consistency loss 还常常太弱 | **abandon** | 只有在同一 sanity protocol 下，先稳定超过 short v2 validation，且证明 loss scale 合理、不压倒 InfoNCE，才值得重启。 (F1 L319-L405) |
| GB-BGRL / GB-GBT | 非 InfoNCE objective 能吸收 GB view 的噪声，修复 view-side 失败 | GB-GBT 比 GB-BGRL 更稳 | GB-GBT 仍明显低于 GRACE；GB-BGRL 大幅崩坏，loss 近零但表示无效 | 这不是“InfoNCE 独有缺陷”，而是 **GB view 与训练目标整体错配**；同时改 view 和 objective 风险过大 | **abandon** | 只有在先拿到健康 non-GB BGRL/GBT baseline，再证明 GB 接入后 multi-seed 仍能增益，才有复活资格。现阶段不应继续。 (F5 L584-L671; F4 L317-L380) |
| Pure GB membership edge denoise | same-ball / cross-ball membership 本身足以识别 noisy edges | cross-ball 的确有一点噪声富集偏置；说明方向不是完全无信号 | precision/recall 太弱，clean collateral damage 明显；formal hybrid 也不如更简单的 percentile scorer | membership 太粗，给不出足够强的 edge-risk signal；clean 图上还会误伤有用边 | **abandon as main method** | 只有先在 **edge scorer diagnostic** 上明显超过 x-space percentile 的 AUC/precision，再考虑训练级验证；否则不许重启。 (F4 L413-L475; F5 L22-L24, L539-L546) |
| GB-FN 的 hard deletion / full-soft / ratio broad suppression / adaptive-mass-cross-view 公式堆叠 | 更强的 same-ball 处理或更复杂公式能带来更大提升 | topk-soft 的确比这些更安全，说明“弱介入”方向对 | hard、full-soft、ratio、v21/v22/v23 都没超过 v2 topk，很多还更差 | 可修正的 FN 比例本来就很小；一旦范围变宽，就变成粗暴误伤；复杂度增加但证据没有增加 | **stop formula churn** | 只有诊断先明确指出固定 topk 的具体瓶颈，否则不要再加新公式。 (F5 L195-L276, L918-L939) |

**只能作为 diagnostic / negative result 保留的方向**

| 方向 | 现在还能做什么 | 不能做什么 |
|---|---|---|
| 构球质量 / protected_precision / mean_num_balls / max_group_size 诊断 | 用来解释为什么某个 GB 机制可能太粗、太碎或塌缩；用于 stop/go 早筛 | 不能把高 protected_precision、高 ball activation、更多 balls 直接写成“方法有效”或“会带来更高下游指标”。 (F6 L75-L102, L341-L359; F5 L90-L96) |
| no-ball / candidate Precision@k / candidate overlap 诊断 | 用来回答 GB-FN 到底有没有独立信息，是否只是 pure similarity topk 的轻微近似 | 不能把 post-hoc Precision@k 直接等同于训练有效性，更不能把它写成有监督证据。 (F2 L648-L686; F5 L813-L855) |
| reliable-ball / topology percentile / partition Jaccard 诊断 | 用来判断 topology tweak 是否真的移动了 partition / candidate set | 如果 partition 几乎不动，就不应继续 formal 训练，更不能把“改了边”写成“改了机制”。 (F5 L796-L855, L964-L982) |
| 视图生成、alternative objective、GB membership edge denoise 的失败结果 | 可作为 negative result / motivation / appendix，支持“GB 不适合直接当强语义机制” | 不能再作为后续主线调参的理由。 (F5 L895-L913; F5 L584-L671; F4 L444-L475) |

## Survivable Ideas

| 想法 | 为什么不是“概念已死” | 只能以什么安全形式继续 | 必须新增什么证据，否则停止 | 论文定位 |
|---|---|---|---|---|
| GB-FN-GRACE v2 topk-soft | 它至少满足“改 loss 不改 view”，而且在 Cora/CiteSeer 上有 tiny positive mean；失败更多像 **effect 太小**，而不是方向彻底错 | 只保留 **same-ball candidate scope + small top-k + fixed-alpha soft suppression**；不再改公式，不再放大 suppress scope | 必须同时补齐 no-ball topk、candidate overlap、masked ratio、per-seed paired delta、candidate Precision@k；若 GB 与 no-ball 在候选和 loss 上不可区分，就降级为 appendix/diagnostic | **主线候选，但 claim 必须极弱**。 (F5 L195-L276, L1177-L1210; F1 L1052-L1067) |
| RDCL local relation consistency | PubMed 上有目前最强的正信号；说明“local relation regularization”本身可能有效 | 先把它当 **local regularizer**，不是 semantic GB proof；先做 fixed / size-matched / shuffle / random 对照，不先调 alpha/tau | 如果 true groups 仍然不能稳定超过 fixed/random/shuffle controls，就停止讲“语义粒球收益”，把它改写成 generic local regularizer 或直接降级 | **最值得推进的候选**，但 GB claim 需要严格过控。 (F6 L251-L337, L549-L593, L687-L747) |
| 噪声鲁棒场景中的 GB | 不是“GB 主 denoise 有效”，而是“GB 可能充其量只是辅助 edge-risk feature / diagnostic” | 仅考虑 **GB + edge scorer** 的结合形式；x-space percentile 只能作为 non-GB 强对照，不能作为独立候选主线 | 如果 GB+ x_pct 连 edge AUC / noisy-edge keep separation 都提升不了，就停止在 robustness 线上继续为 GB 寻找位置 | **决策型方向**，用于划清 GB 能做什么、不能做什么；不把 non-GB robustness 写成本报告主方向。 (F5 L456-L554; F4 L444-L475) |
| GB-PCL positive expansion | 它有 Cora 单 seed 正信号，但 best saved runs 还是 kmeans/kNN，不是 original GB | 只做“一次 minimal multi-seed verification”，而不是继续 sweep；而且它最多只能作为副实验，不应抢主线 | 若 Cora 5-seed 不能稳定为正，或 PubMed 一上就退，就立即停止 | **低优先级 ablation**，不进 top-3。 (F6 L153-L223, L757-L789) |

**跨方向失败模式已经足够清楚，应该转成后续实验的硬约束：**

- **GB 不能再被当成强 view generator。** 以后任何让 GB 主导 view 的方案，必须先给出 view-drift 或 useful-edge damage 诊断，并在 protocol-matched 对照中真正赢过 GRACE；否则不许开新一轮训练。 (F5 L57-L96, L895-L913)

- **same-ball 不是 hard semantic truth。** 以后不允许再把 same-ball 当 hard positive、hard delete、整球 broad suppression 的依据；GB 只能作为 soft / conservative prior。 (F1 L375-L405; F5 L251-L262, L872-L890)

- **公式复杂度不能先于诊断证据。** v21/v22/v23、mass preserve、cross-view、adaptive weighting 这类路线，只有在诊断先指出固定 topk 的具体失败点时才允许进入实验；否则直接判为“高复杂度低信息增益”。 (F5 L198-L202, L257-L276, L918-L939)

- **GB-FN 必须永远和 no-ball topk 成对出现。** 如果 no-ball tie / beat GB，或者 candidate overlap 高到接近重合，就不允许写 GB-specific claim。 (F2 L648-L686; F1 L1052-L1067)

- **clean citation graph 上 tiny delta 一律按 inconclusive 处理。** 只要 std overlap 明显，或者 paired seed deltas 不稳，就不允许写成“有效提升”。 (F5 L218-L276, L941-L959)

- **诊断 purity 不等于 downstream validity。** protected_precision、candidate Precision@k、mean_num_balls 只可当解释变量，不能当成功判据。 (F6 L341-L359; F5 L813-L855)

- **RDCL 的语义解释必须先过 control。** random、shuffle、fixed partition、size-matched control 必须先跑，再谈 alpha/tau；如果控制组也涨，叙事就必须退回到“regularization form”而不是“语义粒球”。 (F6 L280-L337, L387-L405, L549-L593)

- **噪声鲁棒方向必须严格区分 GB 与非 GB 贡献。** x-space percentile 的 positive result 不能写成 GB 结果，也不进入本轮推荐主线；任何 hybrid 也必须先证明“比 x_pct 单独更强”，才允许说 GB 提供增量。 (F5 L498-L554; F4 L431-L475)

- **不再接受 Cora-only single-seed 主张。** 特别是 GB-PCL 这类方向，不到 multi-seed 和至少一个非 Cora 数据集，不进主表。 (F6 L364-L382, L191-L223)

## Best Next Directions

**第一名：RDCL local relation consistency，但要先做 fixed / size-matched partition 机制隔离。**  
这是目前最值得优先推进的方向，不是因为它已经证明了“GB 语义分组有效”，而是因为它同时满足四点：一，当前正信号最强，PubMed 3-seed 有 `+0.0030` 和 `+0.0025` 的可见增益；二，失败点非常集中，主要就是 mechanism control 没过；三，最小验证实验极便宜，只需加 fixed/size-matched 分组而不是再发明新 loss；四，这个实验一旦成功，就能形成论文级贡献；一旦失败，也能立刻收缩叙事，不再误把 strongest positive 归因给“语义粒球”。当前最合理的定位不是“GB 成功”，而是“RDCL 可能是有效的 local relation regularizer，GB/kmeans 是候选 grouping prior”。 (F6 L251-L337, L549-L593, L687-L747)

**第二名：GB-FN-GRACE v2 topk-soft，但研究目标必须从‘提精度’改成‘证明 GB 有没有独立信息’。**  
它之所以排第二，是因为它是所有 GB+GCL 结合里对 GRACE 破坏最小、角色最保守的形式；同时多份总结都把它视为“当前最可保留的 GB 主线”。但它也几乎已经到达一个危险边界：Cora/CiteSeer 只是 tiny positive，PubMed 基本持平，no-ball topk 极近，candidate Precision@k 没有稳定优势，masked negative ratio 还只有约 `1e-3`，说明可修正信号本来就非常小。这个方向仍值得做，但只能做 **evidence-strengthening**，不再做 formula invention。 (F5 L195-L276, L1177-L1210; F2 L921-L942)

**第三名：噪声鲁棒分支中的‘GB 角色判定实验’，不是新的 GB denoise 主线。**  
把它放进 top-3，不是因为我认为“GB denoise 还有很大希望”，恰恰相反，是因为你需要一个明确实验来结束这条含混线路：在 synthetic edge noise 下，真正强的是 x-space percentile；GB membership denoise 目前明确更弱。由于本报告暂不考虑未结合粒球的对比学习实验，最有价值的问题只能收缩为：**GB 能否作为 x_pct 之外的微弱附加 feature / diagnostic？** 如果答案是否定的，这条 GB robustness 线就应该正式停止；如果答案是肯定的，也只能写成“GB 是辅因子”，不能写成主机制。x_pct-only 只能保留为强对照，不能作为本轮独立推荐方向。 (F5 L456-L554; F4 L413-L475)

**没有进入前三的方向：GB-PCL。**  
它不是完全没有正信号，而是当前证据结构太差：Cora 单 seed、缺 raw log、缺 matched baseline、最佳结果甚至来自 kmeans/kNN 而不是 original GB。它可以保留成“如果论文需要才做的一次 minimal ablation verification”，但现在不值得抢算力。 (F6 L153-L223)

## Minimal Experiment Plan

下面三张表，分别对应我认为最值得继续的三个方向。表中的“设计依据”一行给出对应文件证据来源。

**RDCL local relation consistency 的最小验证实验**

| 项目 | 设计 |
|---|---|
| Precise hypothesis | 目前 RDCL 的 PubMed 增益，不仅仅来自“给任意局部小组一个额外 regularizer”，而是来自 **meaningful grouping** 相对 fixed / random / shuffle controls 的额外价值。 |
| Method variant | 保持当前 RDCL 主配置不变：`alpha=0.5`, `tau=0.2`, `p=10`, `batch_size=2048`；只新增两类控制：**fixed partition** 与 **size-matched random partition**。再与已有 `random`, `kmeans_shuffle`, `gb_v2_shuffle` 一并比较。 |
| Baseline | 无 RDCL 的 baseline；RDCL-kmeans；RDCL-gb_v2。 |
| Mandatory controls | `random`、`kmeans_shuffle`、`gb_v2_shuffle`、`fixed partition`、`size-matched random partition`；所有控制必须尽量匹配组数与组大小分布；**不要先调 alpha/tau**。 |
| Datasets | **最小集只跑 PubMed**；只有在 PubMed 过关后，才扩到 Cora 5 seeds 或第二个更大数据集。 |
| Seeds | PubMed 先用现有推荐的 3 个 seed：`23344, 1, 2`。若 survive，再扩。 |
| Metrics | F1Mi、F1Ma、mean/std、per-seed paired delta；主比较对象是“main groups vs fixed/random/shuffle”。 |
| Diagnostics | protected_precision、num_groups、max_group_size、组大小直方图、size-match 误差、group assignment stability；但这些只当解释指标，不当成功判据。 |
| Success criteria | main kmeans 或 gb_v2 相对 **fixed/random/shuffle** 至少有 `+0.0015` 以上的 mean gap，并且 3 个 seed 里多数为正；控制组不能再“几乎复现主方法”。 |
| Failure criteria | fixed/random/shuffle 与主方法在 PubMed 上继续打平，或 gap 小到 `<0.001` 且没有稳定 paired wins。 |
| Early stop rule | 3 seed 一跑完，只要 fixed 或 shuffle 继续贴住 main，就立刻停止继续调 alpha/tau、构球方法、group purity。 |
| Expected paper claim if successful | “RDCL 在节点级 GCL 中提供了有效的 local relation regularization，而且 meaningful grouping 在至少一个 setting 上优于 generic size-matched grouping。” |
| What not to claim | 不要写“GB 语义分组是必要原因”；不要写“高 protected_precision 导致更高 F1”；不要写“gb_v2 优于 kmeans”，除非数据真的支持。 |
| 设计依据 | 现有正信号与主要弱点都已在文件中明确：PubMed `+0.0030/+0.0025`、shuffle/random 接近主方法、fixed/size-matched 被直接列为 next priority。 (F6 L251-L337, L549-L593, L687-L747) |

**GB-FN-GRACE v2 topk-soft 的最小验证实验**

| 项目 | 设计 |
|---|---|
| Precise hypothesis | GB 的价值不在于显著提高 overall candidate purity，而在于 **改变了被 soft suppression 的少量 top-k FN 集合**，从而提供了 no-ball topk 没有的轻微结构性 regularization。 |
| Method variant | **完全不改公式**，固定使用现有 `v2 topk fixed-alpha`；先做 checkpoint replay / log-based diagnostics，再决定是否需要补新 seeds。 |
| Baseline | GRACE；**no-ball topk**（同 `topk`、同 `alpha`、同温度、同 protocol）；不需要再把 hard/full-soft/ratio/v21/v22/v23 当主比较对象，它们已足够失败。 |
| Mandatory controls | `no-ball topk`；candidate overlap；masked_negative_ratio；per-seed paired delta；candidate Precision@k；selected candidate similarity；loss contribution；最好再看 per-epoch candidate dynamics。 |
| Datasets | 最小集：Cora、CiteSeer、PubMed。若现有 checkpoint 足够，先做 diagnostics；只有出现 GB-specific signal，才补 PubMed 5 seeds。 |
| Seeds | 复用现有 Cora/CiteSeer 5 seeds、PubMed 3 seeds；**第二阶段** 仅在第一阶段通过时扩到 PubMed 5 seeds。 |
| Metrics | 下游 accuracy/F1 mean/std；paired delta（GB vs no-ball, GB vs GRACE）；candidate Precision@k；candidate overlap；masked_negative_ratio；selected candidate count。 |
| Diagnostics | per-seed candidate overlap；GB vs no-ball selected set Jaccard；selected negatives 对 loss 的实际贡献；ball size / num_balls / first-nontrivial-level coarse-ness。 |
| Success criteria | 不是只看 accuracy 均值，而是同时满足：① 至少一个数据集上 GB vs no-ball candidate overlap **明显低于 90%**；② GB 在 candidate 诊断或 paired delta 上呈现稳定、非噪声级差异；③ 下游上不输 no-ball。 |
| Failure criteria | no-ball ties/beats GB；candidate overlap 高到几乎重合；Precision@k gap 继续接近零；loss contribution 差异极小。 |
| Early stop rule | 如果 Cora/CiteSeer 的 GB-vs-no-ball overlap 已经 >90%，且 paired delta 不稳，就停止把 GB-FN 当 main method，只保留为 appendix / diagnostic。 |
| Expected paper claim if successful | “GB 可以作为一种低侵入的结构先验，为 false-negative suppression 提供轻量 candidate prior。” |
| What not to claim | 不要写“显著优于 GRACE”；不要写“GB candidate selection 明显优于 no-ball similarity topk”；不要写“跨数据集稳定泛化”。 |
| 设计依据 | 多份总结已一致指出：v2 是 safest 形态；问题不在公式，而在证据强度；no-ball 和 candidate precision 是必须补的机制包。 (F5 L195-L276, L1177-L1210, L1247-L1275; F2 L648-L686; F1 L1052-L1067) |

**噪声鲁棒分支里 GB 角色判定的最小验证实验**

| 项目 | 设计 |
|---|---|
| Precise hypothesis | 在 edge rewire noise 下，**direct endpoint similarity** 是主 edge-risk signal；在本报告范围内，只讨论 GB membership 能否给 x_pct 提供可测增量。若不能提供增量，则暂不继续考虑该类没有 GB 贡献的对比学习鲁棒实验。 |
| Method variant | 分两步。**第一步先做 scorer-level diagnostic**：比较 `GB-only`、`x_pct-only`、`x_pct + GB` 三类 scorer 的 noisy-edge separation（AUC / PR / keep-rate split）。只有第二步 diagnostic 证明 `x_pct + GB` 有增量，才进入训练级对比。训练级只保留一个最简单 hybrid，不做复杂融合公式。 |
| Baseline | noisy GRACE；`x_pct-only`；`GB-only`；`shuffled-score` control。这里的 `x_pct-only` 只作为 non-GB 强对照，不作为候选主线。 |
| Mandatory controls | clean/noisy 双表；`x_pct-only vs GB-only vs hybrid`；`shuffled-score` attribution；clean cost；如果 hybrid 不能超过 x_pct，就不能把 GB 写进 robustness main story。 |
| Datasets | 最小集：Cora 与 CiteSeer 的 `edge_noise=0.4`；同时补一个 clean setting 作为代价检查。 |
| Seeds | diagnostic 阶段可先不做 full 5-seed；训练阶段先 3 seeds quick，只有 hybrid 明显超过 x_pct 时才扩到 5 seeds。 |
| Metrics | scorer-level：AUC、PR-AUC、clean-edge keep vs noisy-edge keep；训练级：best-test / last-test、mean/std、clean delta、noise delta。 |
| Diagnostics | score histogram；clean/noisy keep-rate split；hybrid 相对 x_pct 的增量 AUC；如果用 GB feature，还要记录它究竟改变了哪些边。 |
| Success criteria | `x_pct + GB` 在 scorer-level 明显优于 `x_pct-only`，且训练级在 noisy setting 超过 `x_pct-only`，同时 clean cost 不额外恶化。 |
| Failure criteria | hybrid 的 scorer-level 增量不存在，或训练级仍低于 / 等于 x_pct；那就说明 GB 在该分支里最多只是冗余特征。 |
| Early stop rule | 只要第一步 scorer diagnostic 看不见增量，就不进入第二步训练。 |
| Expected paper claim if successful | 最多写成：“在噪声图场景中，GB 可作为 edge-risk estimation 的辅助结构特征。” |
| What not to claim | 不要写“GB edge denoise 是主贡献”；更不能把 x_pct 的 positive result 写成 GB result；也不要把 x_pct-only 写成本轮下一步实验主线。 |
| 设计依据 | 文件已非常明确：x_pct 是当前 noisy graph 上最强信号，GB membership denoise 显著更弱，hybrid 也不如最优 percentile。 (F5 L456-L554; F4 L413-L475) |

## Paper Claim Boundary

**可以谨慎成立的 claim**

| 可以写什么 | 允许的写法边界 | 为什么现在可以这样写 |
|---|---|---|
| GB-FN v2 topk-soft 是一种低侵入的结构先验 | 必须写成 **tiny / mild / conservative / low-intrusion**，并明确说明 Cora/CiteSeer 只是小幅均值正向、PubMed 基本持平、std overlap 明显、no-ball 很接近 | 这与现有证据一致；它保证了不夸大 GB 独立贡献。 (F5 L218-L276; F2 L234-L300) |
| RDCL 在 PubMed 上表现为一个有用的 local relation regularizer | 只能写“local relation regularizer 有信号”，不能写“语义粒球已被证明是原因” | PubMed 3-seed 有真实正信号，但 shuffle/random controls 已削弱语义解释。 (F6 L273-L337, L693-L717) |
| x-space percentile edge reweighting 提升了 synthetic edge-rewire noise 下的鲁棒性 | 在本报告限制下，不能作为下一轮主线或 GB 论文主 claim；只能作为 non-GB 强对照，并同时报告 clean graph cost | 这正是文件中的 strongest noisy positive，但总结也反复提醒不能重写为 GB 贡献。 (F5 L498-L554) |
| GB view generation、direct positive/prototype、alternative objective、GB membership denoise 在当前协议下未优于健康 baseline | 可以写成 negative result / ablation memory / motivation | 这些路线确实已有反复失败证据，而且这类负结果对限定方法边界是有价值的。 (F5 L57-L96, L584-L671; F1 L319-L405, L427-L521) |

**现在绝对不能写的 claim**

| 绝对不能写什么 | 为什么绝对不能写 |
|---|---|
| “GB 显著且稳定优于 GRACE。” | clean citation graphs 上 GB-FN 的增益只有 `0.001x` 量级，std overlap 明显；view generation、alternative objectives 还明显低于 GRACE。 (F5 L218-L276, L941-L959; F5 L57-L96, L620-L670) |
| “GB candidate selection 明显优于 no-ball similarity topk。” | candidate Precision@k 的 GB-vs-no-ball 差异在 Cora/PubMed 还是负的，CiteSeer 也只是极小正差；no-ball accuracy 也几乎贴住 GB-FN。 (F2 L648-L686; F5 L202-L203, L813-L855) |
| “高 protected_precision / candidate precision / ball activation 直接带来更高下游指标。” | RDCL 的 shuffle controls precision 只有约 `0.357`，但 F1 仍接近主方法；view generation 中 mean_num_balls 很健康也没带来更好结果。 (F6 L280-L319, L341-L359; F5 L90-L96) |
| “语义粒球或高纯度 grouping 是 RDCL gains 的必要条件。” | 当前恰恰是 random / kmeans_shuffle / gb_v2_shuffle 复现了大部分 PubMed 收益。 (F6 L280-L337, L723-L755) |
| “GB edge denoise 带来了当前 noisy-graph robustness 的主要收益。” | strongest positive 来自 x-space percentile，不是 GB；pure GB membership denoise 已经被多处判弱。 (F5 L498-L554; F4 L444-L475) |
| “GB-PCL improves GRACE。” | 现在只有 Cora 单 seed、缺 raw log、缺非 Cora；最佳控制甚至来自 kmeans/kNN。 (F6 L153-L223, L757-L789) |
| “已经建立跨数据集泛化。” | Amazon、DBLP、CiteSeer、PubMed 的证据非常不均衡；不少结果仍是 single seed、partial 或 baseline 缺失。 (F5 L12-L14, L225-L229; F6 L298-L301, L800-L817) |

## Final Recommendation

如果你只有 **一周算力**，我建议先做 **RDCL 的 fixed / size-matched partition 控制实验**，只跑 PubMed 3 seeds，`alpha=0.5`、`tau=0.2`、`p=10`、`batch_size=2048` 全部锁死，不要再调超参。原因很简单：这一步对你的研究空间收缩最有效。它直接回答“当前 strongest positive 到底是不是 semantic grouping 贡献”，而这个问题不解决，任何继续写 RDCL 的叙事都站不稳。这个实验成功，你就保住了最像论文主结果的一条线；这个实验失败，你就可以立即停止把 RDCL 讲成 GB 语义方法，避免继续在错误解释上消耗算力。 (F6 L251-L337, L549-L593, L687-L747)

如果你有 **两周算力**，第二个实验不是再发明 GB-FN 公式，而是做 **GB-FN v2 topk-soft 的 no-ball 机制包**：复用现有 Cora/CiteSeer/PubMed checkpoints，先补 candidate overlap、masked ratio、per-seed paired delta、candidate Precision@k、loss contribution；只有这些诊断显示 GB 和 no-ball 的确不同，再决定是否把 PubMed 从 3 seeds 补到 5 seeds。这个顺序的好处是：你先判断 strongest positive 是否真实，再判断最保守的 GB 主线是否还有独立信息。两步都过了，论文才有继续聚焦“GB + GCL”的价值；只要其中一步不过，你就应该显著收缩题目。 (F1 L1052-L1067; F5 L1177-L1210, L1247-L1275)

不建议把这一两周花在以下事情上：继续改 view generation、继续改 GB-BGRL/GB-GBT、继续做 same-ball positive/prototype/consistency，或者继续尝试 pure GB membership denoise。这些方向不是“还差一点”，而是已经反复表现为 **机制层面的不匹配**。同时，由于本报告暂不考虑没有和粒球结合的对比学习实验，x-space percentile robustness 只能保留为强对照和边界参考，不作为本轮独立推进方向，更不能拿来粉饰 GB。 (F5 L57-L96, L584-L671; F1 L319-L405; F4 L444-L475; F5 L498-L554)
