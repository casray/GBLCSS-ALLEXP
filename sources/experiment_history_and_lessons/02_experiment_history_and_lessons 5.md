# 1. Current Overall Judgment

本文件只覆盖当前项目目录 `/Users/qwwr/Downloads/codex-code_2_无粒球抗噪` 中已有的日志、配置、代码、论文草稿和结果汇总；不代表其他项目或所有 GB-GCL 项目的最终判断。

- Confirmed: 当前项目至少测试过 7 个方向：GB 视图生成、GB-FN-GRACE 假负样本抑制、GB membership edge denoise、x/z percentile edge denoise、GB-BGRL/GB-GBT 替代目标、GNNGuard-lite 对照、粒球/候选诊断。
- Confirmed: 直接让 granular ball 深度参与 view generation 的 GB-GCL 路线在长训练和多轮稳定化后仍低于 GRACE，适合作为 negative result / motivation。
- Confirmed: GB-FN-GRACE v2 topk-soft 是当前最稳的 GB 主线形式，但 Cora/CiteSeer 的提升很小且 std 明显重叠；不能写成强统计优势。
- Confirmed: x-based percentile edge denoise 在 edge rewire noise 下有更强的 5-seed 正向证据，但它不是 GB 方法；它更像噪声鲁棒方向或强对照。
- Likely: GB 在本项目中更适合作为温和候选范围/结构先验，而不是作为硬语义标签、整球 suppression 或视图生成主体。
- Likely: Top-k 精细 suppression 比整球 full-soft、ratio 过宽 suppression、hard 删除、复杂 adaptive/mass/cross-view 公式更稳。
- Inconclusive: GB-FN-GRACE 相对 no-ball topk 的额外收益非常小；候选 Precision@k 没有稳定优于 no-ball topk。
- Risk: 多数 clean/citation graph 上的增益远小于 std，若作为论文主线必须用“温和、可解释、低侵入”而不是“大幅提升”叙事。
- Risk: PubMed 只有 formal_3seed_100epoch；Amazon/Coauthor 仍是 partial 或 sanity，不能支撑泛化强结论。
- Next priority: 停止继续堆 GB-FN 公式，优先补强证据组织、候选机制诊断、PubMed/Amazon 种子，或把 noisy edge denoise 作为独立鲁棒方向。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation | 直接控制双视图边/特征扰动 | mainline, decay, stronger recovery, freeze, anchor-explore | Cora | 短程/部分稳定性可正向，长训练低于 GRACE | GB 视图分布漂移，后期不稳 | abandon | negative result |
| DIR-02 | GB-FN-GRACE loss-side false-negative suppression | same-ball 候选范围 + Top-k soft suppression | hard, v1 full-soft, v2 topk, ratio, strict quality, alpha schedule, v21/v22/v23, no-ball controls, multi-dataset | Cora, CiteSeer, PubMed, Amazon, Coauthor sanity, edge noise | Cora/CiteSeer 小幅高于 GRACE；PubMed 持平；Amazon mixed | 增益极小、std 重叠、GB 相对 no-ball 贡献温和 | inconclusive | main-method candidate |
| DIR-03 | GB membership edge denoise | 用 same/cross-ball membership 给边降权 | w025/w050/w075, pct/sim quick screens | Cora edge_noise=0.4 | noisy edges 有弱降权偏置 | precision/recall 弱，clean collateral damage 明显，val 不过 GRACE quick | abandon | negative result |
| DIR-04 | x/z percentile edge denoise | 无 GB；作为 denoise 对照/替代信号 | D_x_pct050_w02, G hard, B warmup, Z_z, shuffled-score, multi-noise formal | Cora/CiteSeer edge noise | 5-seed noisy settings 全部正向，noise 越大收益越明显 | clean Cora 有成本；不是 GB 贡献 | continue | main-method candidate |
| DIR-05 | GB-guided alternative objectives | GB view 接 BGRL/GBT 目标 | GB-BGRL, GB-GBT | Cora | GBT 稳定但不如 GRACE；BGRL 明显崩 | 目标函数和 GB 视图错配，teacher/student gap 或 redundancy objective 不友好 | abandon | negative result |
| DIR-06 | GNNGuard-lite edge reweighting control | 无 GB；GNNGuard 风格边权对照 | sanity5, quick30 | Cora edge_noise=0.4 | best test 只略高于 quick GRACE | 权重几乎全接近 1，val gate 不过 | abandon | ablation only |
| DIR-07 | GB diagnostics and candidate analysis | 解释 GB 分区、候选、topology 影响 | ball partition diag, reliable-ball percentile, FN candidate precision | Cora, CiteSeer, PubMed | 解释了 feature term 主导和 Top-k 候选质量 | 非方法方向，不直接提升 accuracy | diagnostic-only | diagnostic |

# 3. Direction-Level Detailed Records

## DIR-01: GB-guided View Generation

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB-guided view generation
- Time period: Unknown；当前目录中保留多轮 Cora 结果。
- Target problem: 用 granular-ball hierarchy 生成更结构感知的双视图增强，替代或加强 GRACE 随机增强。
- Core hypothesis: same-ball 边/节点应被更稳定保留，cross-ball 或低质量球相关结构可被更强扰动，从而提升节点级 GCL。
- Baseline: GRACE。
- Main related files: `src/augment/view_generator.py`, `src/augment/edge_perturb.py`, `src/augment/granular_ball.py`, `configs/cora_decay_stronger_recovery.yaml`。
- Result files: `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_stability_comparison/conclusion.txt`, `outputs/cora_perturbation_decay_comparison/summary.csv`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`, `outputs/final_main_results.csv`。
- Raw logs: `outputs/cora_decay_stronger_recovery_gb_gcl/`, `outputs/cora_decay_stronger_recovery_freeze_gb_gcl/`, `outputs/cora_anchor_v2_stronger_gb_gcl/`, `outputs/cora_long_gb_mainline_gb_gcl/`。
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 随机 edge drop / feature mask 不利用局部结构语义，可能破坏语义一致边。
- Why GB might help: GB hierarchy 可给出局部结构团块，理论上可保护球内关系、扰动球间关系。
- Expected improvement: 更稳定的正样本视图和更少语义破坏。
- Literature or previous experiment inspiration: 当前项目的 granular-ball construction 与 view generator 代码。
- Expected paper claim: GB-guided augmentation preserves local semantics.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-GCL mainline | 用 GB 控制边扰动和可选球内 feature masking | Cora | `cora_stability_comparison`: 0.7356±0.0112 vs GRACE 0.7286±0.0235；但 `cora_long_comparison`: 0.7656±0.0109 vs GRACE 0.7746±0.0113 | GB 视图是否能激活并稳定 | 粒球确实非退化，但长训练优势不能保持 |
| perturbation decay | 后期降低扰动强度 | Cora | `cora_perturbation_decay_comparison`: decay 0.7472±0.0073 vs GRACE 0.7532±0.0074 | 后期漂移是否可用 decay 修复 | 能追回一部分，但仍低于 GRACE |
| stronger late recovery | 更强后期恢复 | Cora | `cora_decay_tuning_round`: 0.7482±0.0083，仍低于 GRACE=0.7532 | 更保守后期扰动是否足够 | 不足以超过 GRACE |
| anchor-explore v2 | anchor 弱视图 + explore 强视图 | Cora | 0.7354±0.0146，低于 stronger_late_recovery 0.7430±0.0157 与 GRACE 0.7586±0.0112 | 分离稳定视图和探索视图 | anchor 过稳、explore 偏强，未修复主瓶颈 |

### 3.4 Role of Granular Ball

- GB role: 直接决定 view generation 中边保留概率、cross-ball/same-ball 扰动、anchor/explore 视图强弱。
- Where GB is generated: `GBGCLViewGenerator.generate` 中基于 `x` 或 `z` 构建 hierarchy。
- Representation space: 主要为 `z`；部分配置/诊断可能使用 `x`。
- Affects loss: 不直接修改 InfoNCE。
- Affects augmentation: 是，核心作用。
- Affects graph structure: 是，改变 `edge_index`。
- Affects sampling: sampled level / anchor first-nontrivial level。
- Affects evaluation: 不应改变 linear evaluation；结果文件未显示 evaluation 变更作为主因。

### 3.5 Result Summary

- Best result: 早期/部分稳定性结果中 GB-GCL mainline 可略高于某轮 GRACE；但正式主表 Cora `GB-GCL stronger` 为 0.7430±0.0157，低于 GRACE 0.7586±0.0112。
- Worst result: anchor-explore v2 0.7354±0.0146；不同轮次 baseline 不完全等价，需按 round 内比较。
- Average trend: 多轮稳定化后仍低于 GRACE。
- Mean/std: 见 `outputs/final_main_results.csv`, `outputs/cora_decay_tuning_round/conclusion.txt`。
- Baseline mean/std: GRACE Cora formal 0.7586±0.0112；另有 round 内 GRACE 0.7532±0.0074、0.7746±0.0113。
- Delta: formal `GB-GCL stronger` 相对 GRACE 约 -0.0156。
- Std overlap: formal 结果存在 std overlap，但均值方向持续不利；不支持主线。
- Datasets improved: Unknown；当前可靠证据主要 Cora。
- Datasets worsened: Cora formal/long-training。
- Per-seed stability: 有 5 seeds，但详细 per-seed 未在结论中展开；raw logs 可查。
- Runtime/memory cost: Unknown。
- Raw evidence: 上述 output 目录和 `summary.json/log`。

### 3.6 What Is Directly Supported

- GB hierarchy 能稳定激活，不是一直退化为单球：`cora_long_comparison` 记录 mean_num_balls=30.94。
- 直接用 GB 深度参与 view generation 的 Cora 长训练/正式结果低于 GRACE。
- decay / stronger recovery / anchor-explore 只缓解部分问题，不能追回 GRACE。

### 3.7 Reasonable Interpretation

解释：GB view generation 的问题不在“粒球生成不了”，而在 GB 介入视图后改变训练分布，后期仍会损伤 GRACE 原本有效的增强/对齐动态。

### 3.8 What Cannot Be Concluded

- 不能推出所有 GB-guided augmentation 都无效；这里只测试了当前代码和参数族。
- 不能推出 GB 本身无价值；后续 loss-side FN suppression 显示仍有温和信号。
- 不能把早期稳定性小幅正向当作长训练主线成功。

### 3.9 Failure Analysis

- Main failure reason: GB 直接改 view generation 带来额外视图分布漂移。
- Secondary failure reason: anchor/explore 分工中 anchor 过稳、explore 仍偏强，正样本跨粒度对齐信号不足。
- Evidence: `outputs/cora_anchor_explore_v2_round/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`。
- Alternative explanation: 部分结果可能受 round 间 baseline/eval epoch 差异影响；但正式主表仍低于 GRACE。
- Minimum check needed: 若复活，必须在同一 formal protocol 下证明 5-seed、多数据集超过 GRACE 和 GCA，并提供视图分布漂移诊断。

### 3.10 Lessons Learned

- Lesson: 非退化粒球不等于有效增强。Evidence: mean_num_balls 稳定但 accuracy 低于 GRACE。Future constraint: 先验证下游和 loss/augmentation scale，再宣传 GB 激活。
- Lesson: 后期恢复只能追回部分性能。Evidence: stronger_late_recovery 0.7482±0.0083 < GRACE 0.7532。Future constraint: 不继续只调 decay schedule。
- Lesson: anchor/explore 设计需要避免 anchor 退化为近似无扰动。Evidence: anchor-explore v2 低于 stronger_late_recovery。Future constraint: 复活前必须定义弱但非退化 anchor。

### 3.11 What Is Still Useful

- `GBGCLViewGenerator` 的 debug 字段、level statistics、anchor/explore 设计可用于解释失败。
- 该方向是论文中“为什么不把 GB 用作 view generation”的关键 negative result。

### 3.12 Decision

- Decision: abandon
- Reason: 重复版本在正式或长训练比较中低于 GRACE，机制成本高且主失败模式一致。
- What should not be tried again: 单纯加大/减小 same-ball/cross-ball edge perturbation、继续只调 decay schedule、把“粒球非退化”当作成功指标。
- Variant still worth trying: 仅在新的鲁棒/噪声任务中，把 GB 用作轻量 guard 而非双视图主体；当前项目暂无足够证据。
- Evidence required to revive this direction: protocol-matched 5-seed Cora/CiteSeer/PubMed 正向、视图分布漂移下降、GB ablation 去除后显著变差。

### Experimental Setup Summary

- Datasets: Cora；其他数据集 Unknown。
- Seeds: 多数 formal 为 5 seeds。
- Epochs: 100 pretrain，100 linear eval 或 round-specific。
- Encoder: GCN。
- Projector: two-layer projection head。
- Base loss: InfoNCE。
- Added mechanism: GB-driven view edge/feature perturbation。
- GB loss weight: Not applicable。
- Augmentation: GB view generation replaces/controls GRACE-style views。
- Ball generation method: `GranularBallBuilder` on `z` in representative configs。
- Evaluation protocol: frozen linear evaluation。

### Baseline Comparability Check

- Same dataset: yes within each round。
- Same split: likely yes from config; exact split seed details Unknown。
- Same seeds: yes for formal 5-seed Cora rounds。
- Same encoder/projector: yes in representative configs。
- Same augmentation: no；method intentionally changes augmentation。
- Same evaluation script: likely yes (`train.py`)。
- Same linear evaluation: likely yes。
- Baseline rerun: yes in several rounds。
- Comparable conclusion: partial; enough to reject current form as mainline。

### Label Leakage Check

- Labels used in GB generation: no evidence in GB builder; search shows training labels are used for evaluation/datasets, not GB construction。
- Labels used in purity: no evidence for training-time purity。
- Labels used in sampling/coarsening: no evidence。
- Validation/test labels used outside evaluation: no evidence。
- Supervised GB mechanism used in pretraining: no evidence。
- Leakage risk: low

## DIR-02: GB-FN-GRACE Loss-Side False-Negative Suppression

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB-FN-GRACE loss-side false-negative suppression
- Time period: 当前论文计划 dated 2026-04-09，输出目录含多轮正式/补实验。
- Target problem: 标准 InfoNCE 把所有非正样本都当负样本，可能误推远 same-ball 结构近邻。
- Core hypothesis: GB 不应深度改视图，而应只提供高质量 same-ball 候选范围；再用 Top-k soft suppression 轻量修正 false negatives。
- Baseline: GRACE；外部 baseline GCA/DGI；no-ball topk 作为候选范围控制。
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `train.py`, `configs/cora_fn_grace_v2_topk.yaml`, `paper/sections/3_method.tex`, `paper/sections/4_experiments.tex`, `PAPER_PLAN.md`。
- Result files: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, `outputs/fn_grace_v2_mainline_round/conclusion.txt`, `outputs/cora_fn_grace_v2_round/conclusion.txt`, `outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt`, `outputs/cora_fn_grace_grid_round/conclusion.txt`, `outputs/noball_topk_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/pubmed_formal_round/conclusion.txt`, `outputs/amazon_protocol_check_round/conclusion.txt`。
- Raw logs: `outputs/cora_fn_grace_v2_topk_gb_fn_grace/`, `outputs/citeseer_fn_grace_v2_topk_gb_fn_grace/`, `outputs/pubmed_fn_grace_v2_topk_bestmatch_gb_fn_grace/`, `outputs/cora_fn_grace_noball_topk_gb_fn_grace/` 等。
- Current decision: inconclusive

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: false negatives in InfoNCE denominator。
- Why GB might help: GB same-ball relation提供结构约束，限制候选范围，避免纯相似度过贪心或随机噪声。
- Expected improvement: 在不改视图、不改 encoder/eval 的前提下小幅提升表示质量。
- Literature or previous experiment inspiration: DIR-01 说明 GB 直接改 view 失败，因此转向 objective correction。
- Expected paper claim: GB is more useful as auxiliary objective prior than as view generation主体。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| hard | same-ball hard deletion | Cora | 0.7520±0.0217 | hard suppression 是否可行 | 比 GB-GCL 强，但弱于 soft 和 GRACE |
| v1 full-soft | 对整球 high-quality same-ball 做 soft suppression | Cora | 0.7580±0.0167 | 降权是否比删除稳 | 接近 GRACE，但范围太宽 |
| v2 topk fixed-alpha | same-ball 中 Top-k=5, alpha=0.25 | Cora/CiteSeer/PubMed | Cora 0.7600±0.0118；CiteSeer 0.6766±0.0128；PubMed 0.7790±0.0173 | 精细 Top-k 是否优于整球 | 当前最稳配置，但提升很小 |
| v2 ratio | 保留比例候选，范围更宽 | Cora | 0.7544±0.0118；masked ratio 0.072 | suppression 变宽是否有益 | 变宽伤害效果 |
| strict quality | 更严格 ball_quality_threshold | Cora | 与 v2 topk 相同 | 球质量阈值是否关键 | 当前主线有效球本来稳定，阈值不是关键旋钮 |
| alpha schedule | 动态 alpha | Cora | 0.7600±0.0113 | 更强/动态抑制是否有益 | 与 fixed-alpha 基本等价 |
| v21 similarity-adaptive | 相似度连续权重 | Cora | 0.7580 | 候选内部加权是否有益 | 未优于 fixed-alpha |
| v22 mass preserve | negative mass preservation | Cora | 0.7592±0.0112 | 负压力守恒是否有益 | scale 约 1.0009，影响太小 |
| v23 cross-view/cross-only | cross-view ranking/scope | Cora | best 0.7598±0.0120，低于主线 | 排序视图和 scope 是否瓶颈 | 没有额外收益 |
| no-ball topk | 全体负样本 Top-k，不用 GB | Cora/CiteSeer/PubMed | Cora 0.7592±0.0114；CiteSeer 0.6760±0.0147；PubMed 0.7787±0.0179 | GB 候选范围是否有额外价值 | 非常接近，GB 额外收益温和 |
| multi-dataset | PubMed/Amazon/Coauthor | PubMed/Amazon/Coauthor | PubMed 3-seed持平；Amazon mixed；Coauthor sanity | 泛化边界 | citation graphs 更稳，co-purchase mixed |

### 3.4 Role of Granular Ball

- GB role: 只提供 same-ball 候选范围和 high-quality mask。
- Where GB is generated: `train.py` warmup 后周期性用 encoder embedding `z` 构建。
- Representation space: `z`，非 classifier logits；`GranularBallBuilder` 内部 detach。
- Affects loss: 是，修改 InfoNCE denominator 的 selected negative weights。
- Affects augmentation: 不改 GRACE view generation；`GBFNGraceViewGenerator` 继承 GRACE wrapper。
- Affects graph structure: 不直接改变 graph。
- Affects sampling: 在 same-ball candidates 中选 Top-k。
- Affects evaluation: 不应改变 evaluation。

### 3.5 Result Summary

- Best result: Cora v2 topk 0.7600±0.0118 vs GRACE 0.7586±0.0112；CiteSeer 0.6766±0.0128 vs GRACE 0.6756±0.0140。
- Worst result: v2 ratio 0.7544±0.0118；hard 0.7520±0.0217；Amazon-Photo single seed 0.3820 vs GRACE 0.4040。
- Average trend: 相对 GRACE 一致持平或小幅正向于 citation graphs；相对 no-ball topk 差距极小；co-purchase mixed。
- Mean/std: Cora/CiteSeer formal 5-seed；PubMed formal 3-seed；Amazon partial 1-seed。
- Baseline mean/std: `outputs/final_main_results.csv`。
- Delta: Cora +0.0014；CiteSeer +0.0010；PubMed +0.0000；no-ball deltas Cora +0.0008、CiteSeer +0.0006、PubMed +0.0003。
- Std overlap: yes，Cora/CiteSeer/PubMed 均重叠；统计上不能宣称显著优于 GRACE。
- Datasets improved: Cora/CiteSeer 均值小幅；Amazon-Computers single seed 小幅。
- Datasets worsened: Amazon-Photo single seed；Cora edge_noise=0.3 mixed。
- Per-seed stability: formal rows有 mean/std；per-seed raw metrics 在各输出目录。
- Runtime/memory cost: PubMed CPU 成本高；早期 PubMed mainline round 标记 CPU 全量 expensive。
- Raw evidence: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, raw output dirs。

### 3.6 What Is Directly Supported

- v2 Top-k 比 v1 full-soft 和 ratio 过宽 suppression 更合理。
- fixed-alpha 已足够，similarity-adaptive、mass preservation、cross-view ranking/cross-only 都未超过当前主线。
- no-ball topk 很强且接近 GB-FN-GRACE；GB 候选范围的额外收益很小。
- 标签只用于 post-hoc candidate precision analysis，不参与训练；当前未见训练时 label leakage。

### 3.7 Reasonable Interpretation

解释：当前真正稳定的信号是“少量高相似 false-negative candidate 的轻量 soft suppression 不伤害 GRACE，并可带来温和正则化”。GB 的作用更像限制候选范围和提供解释性，而不是显著提高候选标签纯度或带来大幅 accuracy gain。

### 3.8 What Cannot Be Concluded

- 不能说 GB-FN-GRACE 在统计上显著优于 GRACE；std 重叠。
- 不能说 GB 明显优于 no-ball pure similarity topk；差值只有 0.0003-0.0008 量级。
- 不能说方法泛化到 co-purchase/coauthor graphs；Amazon mixed，Coauthor 只有 sanity。
- 不能用 PubMed 早期 31 epochs 低结果判断失败；后续 full/3seed 已追回。

### 3.9 Failure Analysis

- Main failure reason: clean citation graph 上可修正的 false-negative 规模极小，masked_negative_ratio 约 1e-3，导致上限温和。
- Secondary failure reason: GB same-ball 未显著提高候选 Precision@k；Cora/PubMed ball-topk 相对 no-ball 略低，CiteSeer 略高。
- Evidence: `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/noball_topk_round/conclusion.txt`。
- Alternative explanation: 小幅收益可能来自训练时机/结构先验正则化，而非候选标签纯度。
- Minimum check needed: 5-seed PubMed、Amazon-Computers 3+ seeds、GB/no-ball/fixed random 控制下的 mechanism diagnostics。

### 3.10 Lessons Learned

- Lesson: hard same-ball deletion不稳。Evidence: hard 0.7520±0.0217 < soft/full/topk。Future constraint: 不再用 hard membership 当语义标签。
- Lesson: suppression 不能过宽。Evidence: v2 ratio masked ratio 0.072、selected candidates 195.7，test 0.7544。Future constraint: 保持 Top-k 小范围。
- Lesson: 复杂公式收益低。Evidence: v21/v22/v23 均未超过 v2。Future constraint: 不继续堆 adaptive/mass/cross-view。
- Lesson: no-ball topk 是必须 baseline。Evidence: no-ball 在 Cora/CiteSeer/PubMed 与主线极接近。Future constraint: 所有后续 GB-FN claim 必须对 no-ball。

### 3.11 What Is Still Useful

- `FalseNegativeSuppressedInfoNCELoss` 和 diagnostic stats 可复用。
- v2 topk fixed-alpha 是当前最合理的 GB-FN 代表。
- no-ball topk 和 candidate precision 是论文解释中必须保留的强对照。

### 3.12 Decision

- Decision: inconclusive
- Reason: 均值趋势温和正向，但 std 重叠、no-ball 差距极小、跨图类型证据不完整。
- What should not be tried again: hard same-ball deletion、整球 full-soft 作为主线、ratio 过宽 suppression、继续加 similarity-adaptive/mass-preserve/cross-view 公式。
- Variant still worth trying: 不改公式，补强多 seed / noisy robustness / candidate diagnostics；或把 GB 表述为温和结构先验。
- Evidence required to revive this direction as strong main line: 多数据集 5-seed、相对 GRACE 和 no-ball 的 non-overlapping 或至少稳定更大 delta、GB ablation 和 candidate diagnostic 同时支持。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Coauthor-CS sanity, Cora/CiteSeer edge_noise。
- Seeds: Cora/CiteSeer 5；PubMed 3；Amazon 1；Coauthor sanity 1。
- Epochs: 100 pretrain；linear eval 100 for main clean citation，部分 noisy/edge denoise为 50。
- Encoder: GCN。
- Projector: two-layer projection head。
- Base loss: InfoNCE。
- Added mechanism: false-negative soft suppression。
- GB loss weight: Not applicable；uses alpha=0.25 negative weight。
- Augmentation: GRACE edge/feature dropout unchanged。
- Ball generation method: `GranularBallBuilder` on `z`, warmup=20, update interval=5。
- Evaluation protocol: frozen linear evaluation。

### Baseline Comparability Check

- Same dataset: yes within formal tables。
- Same split: likely yes via current dataset configs；exact split reproducibility not fully documented here。
- Same seeds: yes for Cora/CiteSeer clean formal。
- Same encoder/projector: yes。
- Same augmentation: yes versus GRACE for GB-FN mainline。
- Same evaluation script: yes (`train.py`)。
- Same linear evaluation: yes in main formal configs。
- Baseline rerun: yes / summarized in final tables。
- Comparable conclusion: yes for Cora/CiteSeer/PubMed trend；statistical strength remains weak。

### Label Leakage Check

- Labels used in GB generation: no evidence。
- Labels used in purity: not in training; `fn_candidate_precision` uses labels only post-hoc。
- Labels used in sampling/coarsening: no evidence。
- Validation/test labels used outside evaluation: no evidence。
- Supervised GB mechanism used in pretraining: no evidence。
- Leakage risk: low

### Granular-Ball Diagnostics

- Number of balls: Cora mainline exact mean in final table not listed；PubMed full recheck mean_num_balls=1.8/max=2；Amazon mean_num_balls=1.4/max=2。
- Average ball size: TODO from diagnostic CSV。
- Max ball size: TODO。
- Singleton ball ratio: TODO。
- Radius mean/std: Unknown。
- Coverage: high-quality coverage TODO。
- Relation density: same_ball_pair_ratio logged in diagnostics；summaries include some values。
- Confidence distribution: ball_quality stats TODO。
- Cross-seed stability: partially via diagnostics; not fully summarized。
- GB loss scale vs contrastive loss: Not applicable。
- Too coarse: PubMed/Amazon often only 2 balls，可能过粗。
- Too fragmented: Not observed as main issue for current mainline。
- TODO: 统一导出 mainline Cora/CiteSeer GB diagnostic summary。

### Next Minimal Experiment If Modified

Not applicable for formula modification. Evidence-strengthening minimum:

- New hypothesis: GB-FN is a low-cost structural prior with small but consistent benefit over no-ball under controlled seeds。
- Change only: no formula changes。
- Keep fixed: v2 topk fixed-alpha, GRACE augment/eval, no-ball baseline。
- Minimal datasets: PubMed 5 seeds; Amazon-Computers 3 seeds；可选 Cora/CiteSeer rerun with candidate stats。
- Minimal seeds: 5 for citation, 3 for Amazon。
- Success criterion: GB-FN beats GRACE and no-ball by a delta larger than current noise floor or shows consistent per-seed wins plus mechanism support。
- Failure criterion: no-ball ties/beats GB-FN or deltas remain <0.001 with std overlap。
- Why this is the smallest useful test: 当前瓶颈是证据强度，不是公式。

## DIR-03: GB Membership Edge Denoise

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: GB membership edge denoise
- Time period: 当前输出目录中有 quick screen 与 precision diagnosis。
- Target problem: edge rewire 噪声下，利用 GB same/cross-ball 关系给边降权。
- Core hypothesis: noisy edges 更可能跨球或低质量，可通过 GB membership 降权。
- Baseline: GRACE under Cora edge_noise=0.4。
- Main related files: `src/augment/soft_edge_denoise.py`, `train.py`, `configs/cora_gb_edge_denoise_edge_noise_04_w075.yaml`。
- Result files: `outputs/soft_edge_denoise_round/conclusion.txt`, `outputs/edge_denoise_precision_round/conclusion.txt`, `outputs/edge_denoise_percentile_round/conclusion.txt`。
- Raw logs: `outputs/cora_gb_edge_denoise_edge_noise_04_w075_gb_edge_denoise/`, `outputs/cora_gb_edge_denoise_sim_edge_noise_04_quick30_gb_edge_denoise/`。
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: noisy edge corrupts message passing and contrastive views。
- Why GB might help: GB partition could identify local semantic groups and downweight suspicious cross-ball edges。
- Expected improvement: Better robustness under injected edge rewire noise。
- Literature or previous experiment inspiration: robustness/edge denoising intuition。
- Expected paper claim: GB denoising improves robustness under graph noise。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| w025/w050/w075 | same_ball_weight=1, cross_ball_weight varied | Cora edge_noise=0.4 quick | best w075 val=0.5880/test=0.5680 vs quick GRACE val=0.5980/test=0.5610 | GB membership weight是否有效 | test 可略高但 val 不过，不能 formal |
| precision diagnosis | dedup unique edge 统计 | Cora edge_noise=0.4 | precision 0.5596, recall 0.2272 | 是否真打中 noisy edges | 只是弱统计偏置 |
| percentile/sim variants | pct/sim soft/hard quick | Cora edge_noise=0.4 | 未形成 best-val 正信号 | 是否覆盖率可改善 | 不足以推进 formal |

### 3.4 Role of Granular Ball

- GB role: 用 same-ball/cross-ball 和 high-quality gate 决定 edge_weight。
- Where GB is generated: pretrain/eval embed 过程中 periodic refresh。
- Representation space: `z` in representative config。
- Affects loss: 不直接改 loss。
- Affects augmentation: 不改 random augment，但改变 GCN edge weights。
- Affects graph structure: edge weights changed, not necessarily edge_index。
- Affects sampling: no。
- Affects evaluation: `apply_during_eval_embed: true`，会影响 embedding evaluation path；这是方法设定的一部分。

### 3.5 Result Summary

- Best result: w075 quick test +0.007 vs quick GRACE，但 val -0.010。
- Worst result: Unknown。
- Average trend: mechanism active but performance gate failed。
- Mean/std: quick mostly 1 seed or small screen；precision diagnosis no formal 5-seed。
- Baseline mean/std: quick GRACE val=0.5980/test=0.5610。
- Delta: val negative，test small positive。
- Std overlap: Unknown；quick screen不能支撑。
- Datasets improved: none formal。
- Datasets worsened: Cora quick validation。
- Per-seed stability: Unknown。
- Runtime/memory cost: Unknown。
- Raw evidence: `outputs/soft_edge_denoise_round/conclusion.txt`, `outputs/edge_denoise_precision_round/conclusion.txt`。

### 3.6 What Is Directly Supported

- GB membership denoise 确实让 noisy edges 平均权重更低，但 precision 不高、recall 低。
- clean-edge collateral damage 明显。
- 当前版本不值得进入 formal。

### 3.7 Reasonable Interpretation

解释：same/cross-ball membership 对 noisy edge 有弱相关性，但不是足够精确的 edge denoising 信号；GB 分区过粗或边界不稳会让 clean edges 被误伤。

### 3.8 What Cannot Be Concluded

- 不能说 GB edge denoise 在所有噪声下无效；只做了 Cora edge_noise=0.4 quick/diagnosis。
- 不能把 best-test 小正向当作成功；best-val gate 未过。

### 3.9 Failure Analysis

- Main failure reason: low precision/recall and high clean collateral damage。
- Secondary failure reason: same/cross-ball 是粗信号，不够区分 injected noisy edges。
- Evidence: precision 0.5596, recall 0.2272, clean downweight rate 0.1171。
- Alternative explanation: quick setting/参数不足；但机制诊断已显示信号弱。
- Minimum check needed: 若复活，先在 edge-level precision/recall 上明显超过当前，再跑 formal。

### 3.10 Lessons Learned

- Lesson: cross-ball != noisy edge。Evidence: downweighted edges precision only modestly above base rate。Future constraint: 不用 hard GB membership 单独做 denoise 主信号。
- Lesson: val gate 重要。Evidence: test 小正向但 val 负向。Future constraint: formal 前必须同时过 best-val 和 mechanism gate。

### 3.11 What Is Still Useful

- edge-level precision diagnosis 代码与指标可复用于 DIR-04。
- 该方向证明 GB membership alone 不足以做可靠 edge denoise。

### 3.12 Decision

- Decision: abandon
- Reason: 当前 GB membership denoise 没过 quick val gate，机制打击噪声不够精准。
- What should not be tried again: 只靠 same/cross-ball membership 统一降权。
- Variant still worth trying: 结合 endpoint similarity / uncertainty，而不是 pure membership。
- Evidence required to revive this direction: precision/recall、clean collateral、5-seed val/test 同时改善。

### Baseline Comparability Check

- Same dataset: yes for quick Cora edge_noise=0.4。
- Same split: likely yes。
- Same seeds: mostly no/quick；formal不足。
- Same encoder/projector: yes。
- Same augmentation: yes except edge weights。
- Same evaluation script: yes。
- Baseline rerun: quick reused/new mixed。
- Comparable conclusion: partial；足够停止当前 branch，不足以写全局结论。

## DIR-04: x/z Percentile Edge Denoise

### 3.1 Direction Summary

- Direction ID: DIR-04
- Direction name: x/z percentile edge denoise
- Time period: 当前输出目录中有 validation/generalization/formal/shuffled-control 多轮结果。
- Target problem: edge rewire 噪声下，按 endpoint feature/representation similarity 给低相似边降权。
- Core hypothesis: injected noisy edges 在 endpoint similarity 上更低，用 percentile soft weights 可保留 clean graph message passing。
- Baseline: noisy GRACE。
- Main related files: `src/augment/soft_edge_denoise.py`, `configs/cora_x_pct_edge_denoise_D_edge_noise_04_formal_main100.yaml`, `scripts/summarize_x_pct_formal_main_table.py`。
- Result files: `outputs/x_pct_edge_denoise_formal_main_table/conclusion.txt`, `outputs/x_pct_edge_denoise_ablation_round/conclusion.txt`, `outputs/x_pct_edge_denoise_low_noise_round/conclusion.txt`, `outputs/x_pct_edge_denoise_generalization_round/conclusion.txt`, `outputs/randomized_score_control_round/conclusion.txt`。
- Raw logs: `outputs/cora_x_pct_edge_denoise_D_edge_noise_04_formal_main100_x_pct_edge_denoise/`, `outputs/citeseer_x_pct_edge_denoise_D_edge_noise_04_formal_main100_x_pct_edge_denoise/` 等。
- Current decision: continue

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: noisy input graph corrupts edge structure。
- Why GB might help: 本方向本身不使用 GB；它作为 edge denoise 替代信号和强对照出现。
- Expected improvement: 在 edge_noise 设置下提升 GRACE robustness。
- Literature or previous experiment inspiration: GNNGuard/edge denoise intuition；DIR-03 暗示 pure GB membership 不够，需要更直接 edge score。
- Expected paper claim: x-based edge denoise improves robustness under graph noise；不是 GB 主 claim。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| D_x_pct050_w02_warm0 | x source, percentile_soft p=0.50, min_weight=0.2 | Cora/CiteSeer edge noise | formal selected noisy settings全正向 | 主候选 | 当前最强 noisy setting 方法 |
| G_x_hard_pct035 | hard percentile | Cora edge_noise=0.4 | best-test 接近 D，但 val 弱 | hard vs soft | soft D 更平衡 |
| Z_z_pct035 | z source | Cora edge_noise=0.4 | 低于 D | x vs z | evidence directionally favors x |
| B_x_pct035_warm5 | warmup=5 | Cora edge_noise=0.4 | 低于 D | early intervention | warmup=0 更好，但非完全正交 |
| shuffled-score | 保持权重预算，随机分配 score | Cora/CiteSeer | weaker than aligned D | attribution | 支持 edge selection 而非 generic regularization |
| clean/low-noise | edge_noise=0.0/0.1 | Cora/CiteSeer | clean Cora 有成本，low-noise多为正向 | 边界条件 | 更适合 mid/high noise，不是 universal enhancement |

### 3.4 Role of Granular Ball

- GB role: none in main D；这是 non-GB denoise/control direction。
- Where GB is generated: Not applicable。
- Representation space: `x` for D；`z` only in Z ablation。
- Affects loss: no。
- Affects augmentation: no；but affects GCN edge_weight during pretrain/eval embed。
- Affects graph structure: edge weights, not edge_index。
- Affects sampling: no。
- Affects evaluation: `apply_during_eval_embed: true` in configs。

### 3.5 Result Summary

- Best result: formal 5-seed main table: Cora noise=0.4 +0.0322 best-test vs GRACE；CiteSeer noise=0.4 +0.0420。
- Worst result: clean Cora best-test -0.0103, last-test -0.0137 in 3-seed clean/low-noise round。
- Average trend: noisy settings positive；clean/low-noise mixed；benefit grows with noise on Cora selected settings。
- Mean/std: formal 5-seed noisy settings available。
- Baseline mean/std: same formal round GRACE。
- Delta: Cora noise=0.2 +0.0190, 0.3 +0.0250, 0.4 +0.0322；CiteSeer noise=0.4 +0.0420。
- Std overlap: Some std overlap likely remains, but deltas larger than DIR-02 and directionally consistent。
- Datasets improved: Cora noisy, CiteSeer noisy。
- Datasets worsened: Cora clean。
- Per-seed stability: summarized by std; per-seed raw logs exist。
- Runtime/memory cost: edge weighting overhead；exact runtime Unknown。
- Raw evidence: `outputs/x_pct_edge_denoise_formal_main_table/conclusion.txt`。

### 3.6 What Is Directly Supported

- x_pct D beats noisy GRACE in selected 5-seed formal noisy settings。
- aligned edge scores beat shuffled score controls in 3-seed attribution round。
- x source currently beats z source directionally in retained ablation。
- method has clean graph cost in at least Cora clean setting。

### 3.7 Reasonable Interpretation

解释：edge rewire noise creates low endpoint-similarity edges; x-based percentile soft weighting directly targets this artifact, so it is more effective than GB membership denoise under synthetic edge noise.

### 3.8 What Cannot Be Concluded

- 不能说这是 GB 的贡献。
- 不能说 clean graph universal harmless。
- 不能推广到 PubMed/other noise types；当前 formal scope 明确受限。
- 不能 claim x-vs-z/soft-vs-hard/warmup 的完全正交因果；ablation 文件明确标记不是 fully factorial。

### 3.9 Failure Analysis

- Main failure reason: clean Cora cost，说明低相似边不总是噪声。
- Secondary failure reason: synthetic edge_rewire-specific；真实噪声泛化 Unknown。
- Evidence: `x_pct_edge_denoise_low_noise_round`: Cora clean best-test -0.0103。
- Alternative explanation: clean cost可能来自 eval_embed edge weighting；需要 ablation。
- Minimum check needed: true clean/noise tradeoff table, real noise benchmarks, PubMed noisy scaling。

### 3.10 Lessons Learned

- Lesson: direct edge similarity is stronger than pure GB membership for synthetic edge noise。Evidence: DIR-04 formal > DIR-03 quick fail。Future constraint: edge denoise主线应优先用 direct edge score。
- Lesson: shuffled score control is necessary。Evidence: shuffled weaker on all completed settings。Future constraint: denoise claim必须证明不是权重预算正则化。
- Lesson: robustness method may cost clean performance。Evidence: Cora clean negative delta。Future constraint: main table must include clean cost/limitation。

### 3.11 What Is Still Useful

- 可作为独立鲁棒方向继续推进。
- 可作为 GB denoise 的强对照，说明 GB membership 不是最强 edge score。

### 3.12 Decision

- Decision: continue
- Reason: 在 noisy selected settings 下有 5-seed protocol-matched 正向证据和 shuffled-control attribution。
- What should not be tried again: 把该方向写成 GB 贡献；忽略 clean cost。
- Variant still worth trying: real-noise datasets, PubMed/CiteSeer more noise levels, eval_embed-only ablation。
- Evidence required to revive/strengthen: 5-seed clean/noisy完整曲线、realistic noise、runtime cost。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer。
- Seeds: formal noisy 5 seeds；some ablations 3 seeds。
- Epochs: 100 pretrain, 50 linear eval for formal noisy。
- Encoder: GCN。
- Projector: GRACE projection head。
- Base loss: InfoNCE。
- Added mechanism: edge weights from endpoint similarity percentile。
- Augmentation: GRACE unchanged。
- Evaluation protocol: frozen linear eval with edge weights applied during eval embedding。

### Baseline Comparability Check

- Same dataset: yes。
- Same split: likely yes。
- Same seeds: yes within formal。
- Same encoder/projector: yes。
- Same augmentation: yes。
- Same evaluation script: yes。
- Baseline rerun: yes。
- Comparable conclusion: yes for selected noisy settings。

## DIR-05: GB-Guided Alternative Objectives

### 3.1 Direction Summary

- Direction ID: DIR-05
- Direction name: GB-guided alternative objectives
- Time period: 当前目录含 GB-BGRL 与 GB-GBT round。
- Target problem: 避免 InfoNCE false negatives 或增强 instability，转向 bootstrap / redundancy-reduction objective。
- Core hypothesis: 结合 GB view 与 BGRL/Barlow Twins 目标可能比 InfoNCE 更稳。
- Baseline: GRACE, GB-GCL stronger_late_recovery。
- Main related files: `src/gb_gcl/models/contrastive.py`, `src/gb_gcl/losses/bgrl_loss.py`, `src/gb_gcl/losses/barlow_twins.py`, `src/augment/view_generator.py`。
- Result files: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`。
- Raw logs: `outputs/cora_gb_bgrl_main_gb_bgrl/`, `outputs/cora_gb_gbt_main_gb_gbt/`。
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: InfoNCE false negatives 和 GB view instability。
- Why GB might help: weak/stable teacher or redundancy reduction might absorb GB view noise。
- Expected improvement: 比 GB-GCL view generation 更稳定。
- Literature or previous experiment inspiration: BGRL/Barlow Twins style objectives。
- Expected paper claim: Weak-view relation teaching or non-negative objectives stabilize GB-GCL。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-BGRL | 原图 teacher + GB student，EMA target | Cora | 0.4164±0.0456 | bootstrap 是否稳定 | 明显弱于 GRACE/GB-GCL |
| GB-GBT | weak-stable + GB view for Barlow Twins | Cora | 0.7304±0.0138 | redundancy objective 是否更稳 | 比 BGRL/部分 GB-GCL 稳，但仍低于 GRACE |

### 3.4 Role of Granular Ball

- GB role: 仍主要影响 view generation / student view。
- Where GB is generated: `GBBGRLViewGenerator`, `GBGBTViewGenerator` wrapping GBGCL generator。
- Representation space: likely `z` in configs; exact config not reopened here。
- Affects loss: loss objective changed to BGRL/GBT。
- Affects augmentation: yes。
- Affects graph structure: yes through GB view。
- Affects sampling: level/weak view selection。
- Affects evaluation: no intended change。

### 3.5 Result Summary

- Best result: GB-GBT 0.7304±0.0138。
- Worst result: GB-BGRL 0.4164±0.0456。
- Average trend: 不接近 GRACE 0.7586±0.0112。
- Mean/std: available in conclusions。
- Baseline mean/std: GRACE 0.7586±0.0112；GB-GCL stronger_late_recovery 0.7430±0.0157。
- Delta: GB-GBT vs GRACE -0.0282；GB-BGRL vs GRACE -0.3422。
- Std overlap: GB-GBT 不足以支持；GB-BGRL clearly bad。
- Datasets improved: none。
- Datasets worsened: Cora。
- Per-seed stability: summarized。
- Runtime/memory cost: Unknown。
- Raw evidence: output round conclusions。

### 3.6 What Is Directly Supported

- 当前 GB-BGRL 最小实现不能作为主线。
- GB-GBT 虽更稳，但仍无法超过 GRACE。

### 3.7 Reasonable Interpretation

解释：问题不只是 InfoNCE false negative；当 GB 视图仍是主体时，替换 objective 不会自动解决视图-目标错配。

### 3.8 What Cannot Be Concluded

- 不能说所有 BGRL/GBT graph SSL 失败；只针对当前 GB-guided variants。
- 不能说 teacher-student 永远没价值；但当前实现不应继续主线。

### 3.9 Failure Analysis

- Main failure reason: objective 与当前 GB view distribution 错配。
- Secondary failure reason: BGRL teacher 原图过稳、student/teacher gap 过大，loss 过快趋零但表示无效。
- Evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`。
- Alternative explanation: objective hyperparams 未充分调优；但差距太大，不适合当前优先级。
- Minimum check needed: teacher/student gap diagnostics and comparable 5-seed improvements。

### 3.10 Lessons Learned

- Lesson: 换 objective 不能弥补错误 view mechanism。Evidence: GB-GBT < GRACE。Future constraint: 先确认 view signal，再换目标。
- Lesson: BGRL 需要有信息量的 target view。Evidence: GB-BGRL loss 近零但 accuracy 崩。Future constraint: 不用原图 teacher + 强 GB student 作为默认。

### 3.11 What Is Still Useful

- 可作为 negative result，说明“绕开 InfoNCE”不是当前最小路径。
- GB-GBT weak view 统计可用于未来 weak-view 设计参考。

### 3.12 Decision

- Decision: abandon
- Reason: 当前 Cora 结果明显低于 GRACE，不适合主线。
- What should not be tried again: 直接把 GB view 接到 BGRL/GBT，期待 objective 自动修复。
- Variant still worth trying: 若未来研究 teacher-student，先缩小 view gap 并做 collapse diagnostics。
- Evidence required to revive this direction: 至少 Cora/CiteSeer 5-seed 接近/超过 GRACE，且 loss/embedding collapse 指标正常。

## DIR-06: GNNGuard-Lite Edge Reweighting Control

### 3.1 Direction Summary

- Direction ID: DIR-06
- Direction name: GNNGuard-lite edge reweighting control
- Time period: sanity/quick rounds only。
- Target problem: 用 GNNGuard-style neighbor importance 给 noisy edges 降权，作为非 GB edge denoise baseline。
- Core hypothesis: representation similarity + memory smoothing 可保护 clean edges。
- Baseline: GRACE and x_pct_edge_denoise quick。
- Main related files: `src/augment/gnn_guard.py`, `configs/cora_gnn_guard_edge_noise_04_sanity.yaml`。
- Result files: `outputs/gnn_guard_lite_sanity_round/conclusion.txt`, `outputs/gnn_guard_lite_quick_round/conclusion.txt`。
- Raw logs: `outputs/cora_gnn_guard_edge_noise_04_sanity_gnn_guard_lite/`, `outputs/cora_gnn_guard_edge_noise_04_quick30_gnn_guard_lite/`。
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: noisy graph edges。
- Why GB might help: 不使用 GB；作为 guard-style baseline。
- Expected improvement: 通过边重要性记忆平滑噪声。
- Literature or previous experiment inspiration: GNNGuard。
- Expected paper claim: stronger external edge reweighting baseline does/does not explain denoise gains。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| sanity5 | z_memory_min02, 5 epochs | Cora edge_noise=0.4 | test 0.488 | 能否跑通 | 跑通但很弱 |
| quick30 | z_memory_min02, 30 epochs | Cora edge_noise=0.4 | val same as GRACE, test +0.002；val gate false | quick value | 不值得 formal |

### 3.4 Role of Granular Ball

- GB role: none。
- Where GB is generated: Not applicable。
- Representation space: `z`。
- Affects loss: no。
- Affects augmentation: no。
- Affects graph structure: edge weights。
- Affects sampling: no。
- Affects evaluation: edge weights apply during eval embed。

### 3.5 Result Summary

- Best result: quick30 best val 0.5980, test 0.5630 vs GRACE 0.5980/0.5610。
- Worst result: sanity5 test 0.488。
- Average trend: too weak to pursue。
- Mean/std: mostly 1 seed quick/sanity。
- Baseline mean/std: quick GRACE 0.5980/0.5610。
- Delta: test +0.002, val +0.000。
- Std overlap: Unknown；not formal。
- Datasets improved: none formal。
- Datasets worsened: sanity。
- Per-seed stability: Unknown。
- Runtime/memory cost: Unknown。
- Raw evidence: `outputs/gnn_guard_lite_quick_round/conclusion.txt`。

### 3.6 What Is Directly Supported

- Current GNNGuard-lite implementation does not pass quick val gate。
- Edge weights nearly all close to 1，mechanism too weak compared with x_pct。

### 3.7 Reasonable Interpretation

解释：normalized neighbor importance 在当前 graph/representation 下太接近均匀，不能形成有效 denoise。

### 3.8 What Cannot Be Concluded

- 不能否定 full GNNGuard；这里只是 lightweight in-repo control。

### 3.9 Failure Analysis

- Main failure reason: weight signal too weak; mean weights near 0.995。
- Secondary failure reason: clean and noisy downweight rates both high，precision不够。
- Evidence: quick conclusion mechanism table。
- Alternative explanation: 需要更长训练或不同 threshold；但 quick gate不支持优先投入。
- Minimum check needed: stronger separation and validation gain。

### 3.10 Lessons Learned

- Lesson: 不是所有 similarity edge weighting 都有效。Evidence: x_pct strong, guard-lite weak。Future constraint: 必须看 weight distribution and val gate。

### 3.11 What Is Still Useful

- 作为 baseline 说明 x_pct 的收益不是任何 similarity memory 都能得到。

### 3.12 Decision

- Decision: abandon
- Reason: quick/sanity evidence too weak。
- What should not be tried again: 当前 z_memory_min02 直接扩 formal。
- Variant still worth trying: full GNNGuard faithful reproduction only if论文需要。
- Evidence required to revive this direction: quick 3-seed best-val/test clear positive and edge weight separation。

## DIR-07: GB Diagnostics and Candidate Analysis

### 3.1 Direction Summary

- Direction ID: DIR-07
- Direction name: GB diagnostics and candidate analysis
- Time period: 当前目录有多轮 post-hoc/diagnostic outputs。
- Target problem: 解释 GB 分区、topology 修改、候选标签命中率与 method 结果之间的关系。
- Core hypothesis: 即使方法增益小，诊断可以说明失败机制和可保留信号。
- Baseline: Not applicable / paired method outputs。
- Main related files: `src/gb_gcl/analysis/fn_candidate_precision.py`, `src/gb_gcl/analysis/ball_partition_diagnosis.py`, `scripts/summarize_ball_partition_diag.py`。
- Result files: `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/ball_partition_diag_round/conclusion.txt`, `outputs/reliable_ball_percentile_round/conclusion.txt`, `outputs/noball_large_graph_round/conclusion.txt`。
- Raw logs: related diagnostic output dirs and raw method dirs。
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 方法小幅增益难解释，需要判断 GB 是否真的改善 candidate quality 或 partition。
- Why GB might help: diagnostics reveal whether GB is meaningful or merely no-op。
- Expected improvement: Not direct accuracy improvement。
- Literature or previous experiment inspiration: mechanism interpretability。
- Expected paper claim: GB provides a mild structural candidate prior; topology-only changes do not move partition boundaries。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| FN candidate precision | label used post-hoc only | Cora/CiteSeer/PubMed | ball-topk ≈ no-ball topk >> random | candidate quality | pure similarity is already strong；GB extra purity not stable |
| ball partition diag | topology percentile perturbation | Cora edge_noise=0.4 | assignment almost unchanged | topology term effect | feature term dominates partition |
| reliable-ball percentile | pct90/80/70 topology filter | Cora edge_noise=0.4 quick | metrics identical | reliable topology filter是否影响 GB-FN | 删除 topology edges 仍不改候选 |
| no-ball large graph | PubMed no-ball vs GB-FN | PubMed | both 0.7940 single-seed/full recheck; formal 3-seed no-ball 0.7787 vs GB 0.7790 | GB value on larger graph | GB 相对 no-ball 没拉开 |

### 3.4 Role of Granular Ball

- GB role: 被诊断对象，而非训练主方法。
- Where GB is generated: from saved configs/checkpoints/raw reruns。
- Representation space: method-dependent，mainly `z`。
- Affects loss: only in analyzed GB-FN method。
- Affects augmentation: only in analyzed GB-GCL method。
- Affects graph structure: diagnostic may perturb topology but not main method。
- Affects sampling: candidate set construction。
- Affects evaluation: post-hoc labels used only analysis。

### 3.5 Result Summary

- Best result: candidate Top-k far better than random in label precision；e.g. Cora no-ball Precision@k 0.8040 vs random 0.1796。
- Worst result: ball-topk not consistently higher than no-ball; Cora -0.0029, PubMed -0.0003。
- Average trend: GB explains candidate restriction mildly, not dramatically。
- Mean/std: candidate precision no std in conclusion；TODO if needed。
- Baseline mean/std: Not applicable。
- Delta: topology perturbation changed assignments max 0.001108 only。
- Std overlap: Not applicable。
- Datasets improved: diagnostic only。
- Datasets worsened: Not applicable。
- Per-seed stability: limited。
- Runtime/memory cost: Unknown。
- Raw evidence: listed conclusion files。

### 3.6 What Is Directly Supported

- Top-k similarity candidates are much more label-consistent than random。
- GB ball-topk does not consistently beat no-ball topk in Precision@k。
- Topology percentile deletion can change edges but barely changes Cora partition/candidates。
- Feature margin dominates topology margin in the tested Cora diagnostic。

### 3.7 Reasonable Interpretation

解释：当前 GB-FN 的微小收益更可能来自 conservative structural prior / training-time distribution regularization，而不是显著更纯的 candidate labels。

### 3.8 What Cannot Be Concluded

- 不能用 post-hoc labels 证明 unsupervised mechanism 在训练时知道类别。
- 不能把 Cora topology diagnosis 推广到所有图。
- 不能用 candidate Precision@k 直接等价最终 accuracy。

### 3.9 Failure Analysis

- Main failure reason: diagnostics reveal GB signal too mild to support strong claim。
- Secondary failure reason: current partition often too coarse, topology term too weak。
- Evidence: assignment change max 0.001108；candidate precision near no-ball。
- Alternative explanation: different levels/representations may show stronger GB signal。
- Minimum check needed: cross-seed/epoch stability and candidate overlap diagnostics on Cora/CiteSeer/PubMed。

### 3.10 Lessons Learned

- Lesson: post-hoc label precision must be kept separate from training evidence。Evidence: labels only post-hoc。Future constraint: no leakage in method claims。
- Lesson: topology-only reliable-ball filter can be no-op at candidate level。Evidence: candidate_set_jaccard_mean min 0.996681。Future constraint: diagnose partition movement before formal。
- Lesson: no-ball topk is a strong diagnostic baseline。Evidence: no-ball Precision@k close to ball-topk。Future constraint: always include no-ball。

### 3.11 What Is Still Useful

- Strong support for paper nuance: GB is a mild prior, not a dramatic semantic purifier。
- Provides anti-repetition memory against topology-only tweaks。

### 3.12 Decision

- Decision: diagnostic-only
- Reason: Not a method direction; it explains why several methods succeed/fail.
- What should not be tried again: Reliable topology percentile filter without first showing partition/candidate movement。
- Variant still worth trying: candidate overlap over epochs/seeds; ball size/radius/quality distribution plots。
- Evidence required to revive this direction: diagnostics that directly explain per-seed wins/losses。

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: Hard or Broad Same-Ball Semantics Are Unstable

### Status
confirmed

### Related Directions
DIR-02, DIR-03

### Observation
Hard deletion underperforms soft; full-soft and ratio broad suppression are weaker than Top-k; GB membership edge denoise has low precision/recall.

### Possible Reason
same-ball membership is a structural prior, not a reliable semantic label for every pair or edge.

### Evidence
`outputs/cora_fn_grace_round/conclusion.txt`, `outputs/final_ablation_results.csv`, `outputs/edge_denoise_precision_round/conclusion.txt`。

### Future Constraint
Use GB as soft/confidence/conservative prior only；never treat all same-ball nodes or cross-ball edges as hard truth.

### When to Revisit
Only if a diagnostic shows high same-ball semantic purity and low clean collateral across multiple datasets/seeds.

## FP-2: Direct GB View Generation Damages Long-Training GRACE Dynamics

### Status
confirmed

### Related Directions
DIR-01, DIR-05

### Observation
GB-GCL stronger and alternative objectives remain below GRACE despite active non-degenerate balls.

### Possible Reason
GB changes the view distribution too much; changing objective later does not fix view-target mismatch.

### Evidence
`outputs/final_main_results.csv`, `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`。

### Future Constraint
Do not make GB the main view generator without a view-drift diagnostic and protocol-matched win.

### When to Revisit
If noisy graph settings need GB as guard and x_pct/direct edge scores are insufficient.

## FP-3: Formula Complexity Adds Little After Top-k

### Status
confirmed

### Related Directions
DIR-02

### Observation
similarity-adaptive, mass preservation, cross-view ranking, cross-only suppression all fail to beat v2 topk fixed-alpha.

### Possible Reason
Top-k selected set is already tiny; extra weighting changes loss mass by too little or weakens uniformity.

### Evidence
`outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt`。

### Future Constraint
Stop adding GB-FN formulas unless a diagnostic identifies a specific bottleneck.

### When to Revisit
After larger candidate scope/noisy graph evidence shows current fixed-alpha underfits.

## FP-4: Standard Deviations Overlap on Clean Citation Graph Gains

### Status
confirmed

### Related Directions
DIR-02

### Observation
Cora/CiteSeer gains over GRACE are +0.0014/+0.0010 with std around 0.011-0.014.

### Possible Reason
Current correction touches too few negatives to create large accuracy deltas.

### Evidence
`outputs/final_main_results.csv`。

### Future Constraint
Mark clean graph GB-FN accuracy claims as statistically weak/inconclusive.

### When to Revisit
If 10+ seeds or larger datasets show stable per-seed wins.

## FP-5: Topology-Only Ball Tweaks Can Be No-Ops

### Status
confirmed

### Related Directions
DIR-07

### Observation
Reliable topology percentile filters delete edges but barely change ball assignment or Top-k candidates.

### Possible Reason
Feature term dominates assignment margin; current first nontrivial Cora partition is too coarse.

### Evidence
`outputs/ball_partition_diag_round/conclusion.txt`, `outputs/reliable_ball_percentile_round/conclusion.txt`。

### Future Constraint
Before formal runs, verify that a topology change actually moves partition/candidate sets.

### When to Revisit
On datasets where topology margin is comparable to feature margin.

## FP-6: Clean Graph Robustness Methods Can Have Cost

### Status
likely

### Related Directions
DIR-04

### Observation
x_pct edge denoise helps noisy settings but hurts Cora clean best/last test.

### Possible Reason
Low endpoint similarity can represent useful heterophily/long-range relation, not only noise.

### Evidence
`outputs/x_pct_edge_denoise_low_noise_round/conclusion.txt`。

### Future Constraint
Always report clean cost for denoise methods.

### When to Revisit
With adaptive gating that disables denoise on clean graphs.

# 5. Cross-Direction Positive Signal Ledger

## PS-1: GB Is Safer as Loss-Side Candidate Prior Than View Generator

### Status
likely

### Related Directions
DIR-01, DIR-02

### Observation
GB-GCL view generation underperforms GRACE, while GB-FN topk ties/slightly exceeds GRACE on citation graphs.

### Why It May Matter
It gives the paper a coherent negative-to-positive story: move GB from augmentation主体 to objective-side auxiliary prior.

### Required Verification
Stronger no-ball comparisons and multi-seed PubMed/Amazon evidence.

### Possible Paper Use
motivation / weak main method

## PS-2: Top-k Fine Suppression Is Better Than Broad Suppression

### Status
confirmed

### Related Directions
DIR-02

### Observation
v2 topk beats v1 full-soft and v2 ratio on Cora.

### Why It May Matter
Shows same-ball membership must be filtered down to a tiny high-confidence subset.

### Required Verification
Mechanism stats across Cora/CiteSeer/PubMed.

### Possible Paper Use
main method / ablation

## PS-3: Direct x-based Edge Score Is Strong Under Edge Rewire Noise

### Status
confirmed for selected synthetic settings

### Related Directions
DIR-04

### Observation
D_x_pct050_w02 beats noisy GRACE across selected 5-seed Cora/CiteSeer settings.

### Why It May Matter
It is the strongest robustness signal in the current project, though not GB.

### Required Verification
Clean cost, real noise, PubMed, runtime.

### Possible Paper Use
main method for a robustness paper / strong control for GB paper

## PS-4: Shuffled-Score Control Supports Edge Selection Causality

### Status
likely

### Related Directions
DIR-04

### Observation
Aligned x_pct edge scores beat shuffled score assignment while preserving weight budget.

### Why It May Matter
It helps attribute denoise gains to edge selection, not generic regularization.

### Required Verification
5-seed shuffled formal if used in main text.

### Possible Paper Use
ablation / appendix

## PS-5: Diagnostics Explain Why Some GB Tweaks Fail

### Status
confirmed

### Related Directions
DIR-07

### Observation
Topology deletion barely changes Cora ball assignments/candidates; candidate precision shows no-ball topk is already strong.

### Why It May Matter
Prevents repeated topology-only and overclaiming candidate purity.

### Required Verification
More datasets/seeds.

### Possible Paper Use
diagnostic / limitations

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GB-GCL direct view generation as mainline | Active balls did not translate to long-training gains over GRACE | `outputs/final_main_results.csv`, `outputs/cora_long_comparison/conclusion.txt` | 5-seed multi-dataset win plus view-drift diagnostic |
| Hard same-ball negative deletion | Hard version weaker and higher variance than soft | `outputs/cora_fn_grace_round/conclusion.txt` | High same-ball semantic precision and no collapse/variance increase |
| Full-ball or ratio broad suppression | selected candidates/masked ratio too high and accuracy lower | `outputs/final_ablation_results.csv` | Evidence that broad suppression improves multiple datasets |
| similarity-adaptive/mass-preserve/cross-view GB-FN formulas | Did not beat fixed-alpha topk | v21/v22/v23 conclusion files | Diagnostic showing fixed-alpha is the bottleneck |
| GB membership-only edge denoise | Low precision/recall and validation failure | `outputs/edge_denoise_precision_round/conclusion.txt` | Edge-level precision/recall and best-val both improve |
| Reliable topology percentile filter | Deleted topology edges but partition/candidates unchanged | `outputs/ball_partition_diag_round/conclusion.txt` | Demonstrated assignment/candidate movement before accuracy runs |
| Current GNNGuard-lite z_memory_min02 | Quick val gate failed and weights nearly uniform | `outputs/gnn_guard_lite_quick_round/conclusion.txt` | Stronger edge weight separation plus 3-seed val/test gain |
| Claiming clean GB-FN statistical superiority | Deltas far smaller than std | `outputs/final_main_results.csv` | More seeds or larger deltas beyond overlap |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB as view generator | GB captures local grouping | GB as diagnostic or light loss-side prior |
| hard same-ball positive/negative treatment | same-ball relation has weak semantic signal | soft confidence/Top-k gated weighting |
| GB membership edge denoise | cross-ball edges enriched for noise weakly | combine with endpoint similarity or learned uncertainty |
| topology-only partition filtering | topology can matter in principle | first prove topology margin affects assignments |
| BGRL/GBT with strong GB views | weak-view/teacher relation may help | reduce view gap and add collapse diagnostics first |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same dataset/split/seeds/encoder/projector/augmentation/evaluation。
- Multi-seed evidence: 5 seeds for Cora/CiteSeer/PubMed where feasible；3 seeds minimum for expensive exploratory。
- Multi-dataset evidence: at least two standard citation graphs, plus clearly marked third/mixed graph evidence。
- Ablation removing GB signal: GRACE and no-ball topk for GB-FN；shuffled score for edge denoise。
- Mechanism diagnostic: masked ratio/candidate precision/edge precision/ball diagnostics as appropriate。
- No label leakage: labels only in post-hoc analysis or evaluation, never in unsupervised pretraining。

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation while testing GB mechanism.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as reliable semantic positive/negative without confidence or soft weighting.
- Every new direction must define the exact GB role before coding: view, loss, edge weight, diagnostic, or control.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing the GB signal when GB is claimed.
- If improvement is smaller than overlapping std, mark it `inconclusive`.
- If a direction only works for one seed, do not move it into the paper main line.
- If clean graph gain is tiny, test noisy graph or strong augmentation before making robustness claims.
- If complexity increases, require stronger evidence than a simple baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.
- For this project specifically, no-ball topk is a required baseline for any GB-FN claim.
- For this project specifically, topology-only ball changes must first prove candidate/assignment movement.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| GB-guided view generation | GB directly builds stronger views | Repeatedly below GRACE in long/formal runs | View distribution drift | Yes, motivation/negative result | abandon |
| GB-FN-GRACE | GB same-ball Top-k soft FN suppression | Tiny positive/flat on citation graphs, mixed elsewhere | Very small correction scale and std overlap | Yes, conservative main candidate | inconclusive |
| GB membership edge denoise | same/cross-ball edge weights | Mechanism weak, val gate failed | GB membership too coarse | Yes, negative result | abandon |
| x_pct edge denoise | endpoint similarity percentile soft weights | Strong noisy 5-seed evidence | Synthetic-noise-specific and clean cost | Yes, robustness direction | continue |
| GB-BGRL/GB-GBT | alternative objectives with GB views | Below GRACE, BGRL poor | Objective-view mismatch | Yes, negative result | abandon |
| GNNGuard-lite | guard-style non-GB edge weights | Quick/sanity weak | Weights too uniform | Limited baseline value | abandon |
| Diagnostics | candidate/partition analysis | Explains mild GB effect and no-op topology filters | Not a method | Yes | diagnostic-only |

# 9. Recommended Next Directions

## NEXT-DIR-1: Evidence-Strengthened GB-FN-GRACE, No Formula Change

### Why This Direction Is Recommended
Current GB-FN formula is already locally saturated; the bottleneck is evidence strength, not another weighting variant.

### Previous Failure It Addresses
Avoids repeating v21/v22/v23 formula churn.

### Core Hypothesis
GB-FN is a conservative structural prior with small but consistent benefit over GRACE/no-ball under citation graph protocols.

### GB Role
same-ball candidate prior only.

### What Must Be Different From Failed Directions
No new formula; focus on seeds, no-ball comparison, diagnostics, and honest std overlap.

### Minimal Implementation
Reuse v2 topk fixed-alpha; rerun/summarize PubMed 5 seeds if feasible and Amazon-Computers 3 seeds if co-purchase claim is needed.

### Minimal Datasets and Seeds
PubMed 5 seeds; Amazon-Computers 3 seeds; optional Cora/CiteSeer diagnostic rerun.

### Success Criteria
Stable per-seed wins or meaningful delta over GRACE and no-ball; diagnostics show GB changes candidate distribution.

### Failure Criteria
no-ball ties/beats GB-FN or deltas remain below 0.001 with overlapping std.

### Paper Claim If Successful
GB provides a mild, low-intrusion structural prior for false-negative suppression.

### Reason to Stop Early
If PubMed/no-ball remains tied and candidate diagnostics show no GB-specific signal.

## NEXT-DIR-2: Noisy Edge Robustness via x_pct Edge Denoise

### Why This Direction Is Recommended
It has the strongest 5-seed positive evidence in the current project.

### Previous Failure It Addresses
Avoids weak GB membership denoise by using direct endpoint similarity.

### Core Hypothesis
Under edge rewire noise, endpoint feature similarity identifies noisy edges better than GB membership.

### GB Role
None; this is a robustness branch/control, not a GB mainline.

### What Must Be Different From Failed Directions
Must openly state non-GB contribution and include clean cost.

### Minimal Implementation
Keep D_x_pct050_w02_warm0; add clean/noisy formal table and real-noise or PubMed noisy check.

### Minimal Datasets and Seeds
Cora/CiteSeer clean + edge_noise 0.1-0.4 with 5 seeds; optional PubMed 3 seeds.

### Success Criteria
Positive noisy deltas with acceptable clean cost and shuffled-score attribution.

### Failure Criteria
Clean cost too large or benefits disappear outside synthetic edge_rewire.

### Paper Claim If Successful
Feature-similarity edge reweighting improves robustness to edge rewire noise.

### Reason to Stop Early
If clean cost dominates or real-noise benchmarks show no value.

## NEXT-DIR-3: GB Diagnostic Pack for Paper Honesty

### Why This Direction Is Recommended
Current method claims are subtle; diagnostics prevent overclaiming.

### Previous Failure It Addresses
Prevents repeated topology-only and candidate-purity assumptions.

### Core Hypothesis
GB’s value is mild candidate prior/regularization, not large semantic purification.

### GB Role
diagnostic object.

### What Must Be Different From Failed Directions
No new training method; summarize existing logs and maybe add plots.

### Minimal Implementation
Aggregate Cora/CiteSeer/PubMed diagnostic CSV: num_balls, masked ratio, selected candidates, candidate overlap, candidate Precision@k.

### Minimal Datasets and Seeds
Cora/CiteSeer formal 5 seeds; PubMed 3 seeds.

### Success Criteria
Clear explanation of when GB differs from no-ball and when it does not.

### Failure Criteria
Diagnostics show GB and no-ball are indistinguishable everywhere.

### Paper Claim If Successful
GB is a conservative structural prior with interpretable but small effect.

### Reason to Stop Early
If diagnostics require expensive reruns without changing any paper conclusion.

## NEXT-DIR-4: Conditional GB + Edge Denoise Hybrid

### Why This Direction Is Recommended
GB membership alone is weak, x_pct is strong but non-GB; a hybrid could test whether GB can gate direct edge scores.

### Previous Failure It Addresses
Addresses DIR-03 coarse GB membership and DIR-04 non-GB limitation.

### Core Hypothesis
GB should gate or calibrate endpoint-similarity denoise only where partition confidence is high.

### GB Role
confidence/guard signal, not primary edge score.

### What Must Be Different From Failed Directions
Do not use same/cross-ball as the score itself; use it only as confidence gate.

### Minimal Implementation
Take D_x_pct050_w02 and reduce intervention on low-quality/unstable balls; add pure D and no-GB controls.

### Minimal Datasets and Seeds
Cora edge_noise=0.4 3 seeds quick; only expand if best-val/test beat D or reduce clean cost.

### Success Criteria
Hybrid matches D noisy gain while reducing clean Cora cost.

### Failure Criteria
No gain over D or worse clean/noisy tradeoff.

### Paper Claim If Successful
GB confidence calibrates robust edge denoising.

### Reason to Stop Early
If GB diagnostics show partition too coarse to gate edge scores.

# 10. Paper Claim Implications

## Claim-1

### Claim
GB provides a robust local structure prior for node-level GCL.

### Supported By Which Directions
DIR-02 weakly; DIR-07 as diagnostic nuance.

### Weakened By Which Directions
DIR-01, DIR-03, DIR-05.

### Current Evidence
Cora/CiteSeer GB-FN small positive; PubMed tie；GB view/edge membership routes fail.

### Missing Evidence
Non-overlapping std, stronger no-ball gap, more datasets.

### Required Experiments
PubMed 5 seeds, no-ball diagnostics, Amazon 3 seeds if claiming beyond citation graphs.

### Risk
High if phrased broadly.

### Current Readiness
weak

## Claim-2

### Claim
GB-guided augmentation preserves local semantics.

### Supported By Which Directions
Little direct support.

### Weakened By Which Directions
DIR-01.

### Current Evidence
GB-GCL active but lower than GRACE in long/formal results.

### Missing Evidence
Positive multi-dataset GB augmentation result.

### Required Experiments
New view-drift-controlled augmentation, not current versions.

### Risk
High.

### Current Readiness
not ready

## Claim-3

### Claim
GB-FN topk-soft is better than broad same-ball suppression.

### Supported By Which Directions
DIR-02.

### Weakened By Which Directions
None direct.

### Current Evidence
Cora v2 topk 0.7600±0.0118 > v1 full-soft 0.7580±0.0167 > ratio 0.7544±0.0118; masked ratios support mechanism.

### Missing Evidence
CiteSeer/PubMed full ablation.

### Required Experiments
Optional multi-dataset ablation if paper needs broad claim.

### Risk
Medium.

### Current Readiness
moderate

## Claim-4

### Claim
GB is more useful as an auxiliary prior than as direct replacement for contrastive positives/views.

### Supported By Which Directions
DIR-01 negative result + DIR-02 mild positive.

### Weakened By Which Directions
DIR-02 std overlap/no-ball gap small.

### Current Evidence
GB view generation below GRACE；loss-side correction best among GB routes.

### Missing Evidence
Stronger GB-FN no-ball separation.

### Required Experiments
No formula change; evidence-strengthening and diagnostics.

### Risk
Medium.

### Current Readiness
moderate

## Claim-5

### Claim
GB denoising improves robustness under graph noise.

### Supported By Which Directions
DIR-03 weakly at mechanism level only.

### Weakened By Which Directions
DIR-03 performance failure; DIR-04 non-GB method stronger.

### Current Evidence
GB membership denoise has weak noisy-edge bias but not formal success.

### Missing Evidence
GB-specific noisy 5-seed win.

### Required Experiments
Hybrid GB confidence + x_pct only if necessary.

### Risk
High.

### Current Readiness
not ready

## Claim-6

### Claim
x-based percentile edge denoise improves robustness under edge rewire noise.

### Supported By Which Directions
DIR-04.

### Weakened By Which Directions
Clean Cora cost.

### Current Evidence
5-seed noisy Cora/CiteSeer selected settings all positive; shuffled-control supports edge selection.

### Missing Evidence
Real noise, PubMed, clean formal full curve.

### Required Experiments
Clean/noisy formal extension and real-noise benchmark.

### Risk
Medium because non-GB and synthetic-noise-specific.

### Current Readiness
moderate

# 11. TODO List

## Missing raw logs

- TODO: If paper mainline uses GB-FN diagnostics, link exact per-seed `diagnostic_stats.csv` rows for Cora/CiteSeer/PubMed.
- TODO: Collect raw PubMed formal 3-seed paths into a single manifest.

## Missing baseline checks

- TODO: Confirm exact split consistency and seed mapping for all final tables.
- TODO: Decide whether Amazon unified protocol should be in main text or appendix only.

## Missing seeds/std

- TODO: PubMed GB-FN/GRACE/no-ball to 5 seeds if strong citation-graph claim is needed.
- TODO: Amazon-Computers to 3 seeds before claiming co-purchase positive signal.
- TODO: GNNGuard-lite and GB edge denoise remain quick only; do not compare as formal.

## Missing datasets

- TODO: Coauthor-CS only sanity; no claim.
- TODO: No real-world noisy graph benchmark for DIR-04.

## Missing ablations

- TODO: Multi-dataset v2 topk vs v1 full-soft/ratio if ablation claim must generalize beyond Cora.
- TODO: For x_pct, eval_embed-only vs pretrain-only edge weighting ablation.

## Missing GB diagnostics

- TODO: Ball count/size/radius/quality distribution over seeds for Cora/CiteSeer/PubMed.
- TODO: Candidate overlap and selected similarity for GB-FN mainline; current final mainline lacks fully symmetric no-ball comparison metric.
- TODO: Cross-epoch and cross-seed stability of first nontrivial level.

## Missing visualizations

- TODO: Top-k vs full/ratio masked_negative_ratio plot.
- TODO: x_pct clean/noisy weight distribution plot.
- TODO: ball assignment stability/topology margin plot if using diagnostics in paper.

## Missing explanation for failed directions

- TODO: Add concise appendix table for GB-GCL, GB-BGRL, GB-GBT, GB edge denoise negative results.
- TODO: Separate “method failed” from “GB signal still useful” in paper narrative.

## Missing paper-claim evidence

- TODO: Do not claim statistical superiority over GRACE without acknowledging std overlap.
- TODO: Do not claim GB candidate purity improvement unless candidate diagnostics are expanded and favorable.
- TODO: Do not claim GB noise robustness from current GB membership edge denoise.

# 12. Experiment 7 Linkage Note

- Added on 2026-05-19: `02_experiment_history_and_lessons 7 .md` now records `实验7noisy_label_gb_grace` as a separate direction-level history.
- Do not merge that file into this document's direction count or DIR numbering; this document remains a separate historical review.
- Relevant new conclusion: frozen GRACE + strict GB local label smoothing under noisy train labels is Cora-positive only in selected medium-label high-noise settings and is cross-dataset unstable on CiteSeer.
- Future constraint imported from Experiment 7: any noisy-label GB direction must report effective smoothing coverage and must beat CE/global/kmeans/topk_matched, not only global smoothing.
