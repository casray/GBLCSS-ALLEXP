# GB-GCL Experiment History and Lessons

本文件只基于当前项目目录中的 README、代码、配置、日志、结果表、论文草稿，以及 `outputs/`、`gb_noisy_label_validation/`、`gb_graph_aware_minival_6_1/` 等子实验目录的证据整理。它不读取、不混用、也不代表其他项目目录中的 GB-GCL 实验最终判断。

# 1. Current Overall Judgment

- Confirmed: 当前项目已经测试 7 个方向：GB 视图增强、GB-FN negative suppression、直接 GB 关系正则、GB soft edge denoise、GB-BGRL/GB-GBT 替代目标、diagnostic-only 分析，以及 frozen-embedding noisy-label 场景下的 GAGB local target repair。
- Confirmed: 没有一个方向达到 strict `continue` 标准。没有方向同时满足“多数据集稳定提升、std 不重叠、机制清楚、无额外复杂度风险”。
- Confirmed: GB 深度参与 view generation 在 Cora 正式/长训练结果中反复低于 GRACE，虽然粒球结构本身能激活。
- Confirmed: 当前最可保留的主线是 GB-FN-GRACE v2 topk-soft，但提升很小：Cora +0.0014，CiteSeer +0.0010，PubMed 持平，且均与 baseline std overlap。
- Confirmed: GAGB 在 frozen GRACE noisy-label setting 中已经证明“target repair 机制信号存在”：v2.1 在 4/4 设置中把 NCR 从 0 拉成正值，且 CDR 保持在 0.03 以下；但它还没有证明强 GAGB-specific value。
- Likely: GB 更适合作为温和的 false-negative candidate prior，而不是作为 hard semantic label、直接正样本、prototype pulling 或大范围视图重写。
- Likely: 在 noisy-label 下，GB/GAGB 更适合作为 local soft-label / confidence prior，而不是 grouping 一旦形成就应直接推高准确率的“语义真值单元”。
- Likely: Top-k 精细 suppression 比整球 full-soft 或 ratio 扩大 suppression 更安全。
- Inconclusive: GB 相对 no-ball topk 的额外收益很小，候选标签命中率也没有一致优于 no-ball topk。
- Inconclusive: GAGB v1 的 purity/entropy 提升和 v2.1 的正 NCR，还不足以推出 graph-aware grouping 本身优于 ordinary GB、khop、kmeans、topk 或 intervention-matched shuffled/replay。
- Risk: 如果只看 Cora 或只看均值差，不看 std overlap/no-ball 对照，会高估当前方法。
- Risk: 如果只看 purity、entropy、AUROC、NCR，而不看 class-level accuracy、CDR、matched controls 和 Cora 0.4 这类弱设置，会高估 GAGB。
- Risk: 继续堆叠 similarity-adaptive、mass preserve、cross-view ranking、fuzzy affinity、prototype/positive/consistency 分支，容易增加复杂度但不增加证据。
- Next priority: 不再扩展 encoder-side 强 GB 机制。优先保留两条“软角色”方向：GB-FN 作为轻量结构先验，GAGB 作为 frozen-embedding noisy-label target repair；其中 GAGB 当前更需要补齐 fair controls、matched replay 和 boundary-shift 诊断，而不是继续堆新构球公式。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation / augmentation control | 用粒球层级控制边保留、anchor/explore、扰动衰减 | gb_gcl mainline, decay, stronger recovery, freeze, anchor/explore, long/stability rounds | Cora | 粒球激活但正式/长训练低于 GRACE | 视图分布漂移、扰动过强、后期不稳 | abandon | motivation |
| DIR-02 | GB-FN-GRACE false-negative suppression | 用 same-ball 结构先验筛选并 soft downweight 少量负样本 | hard/soft v1, v2 topk, ratio, grid, topk1, no-ball, v21/v22/v23, stable ranking, fuzzy affinity, noise/multidataset | Cora, CiteSeer, PubMed, Amazon, Coauthor sanity | Cora/CiteSeer 小幅正向，PubMed 持平；std overlap | 额外收益极小，no-ball topk 非常接近 | modify | main-method candidate |
| DIR-03 | Direct GB relation regularization | same-ball/prototype/consistency 直接拉近或正则节点 | prototype, shared-ball soft positive, ball consistency | Cora, CiteSeer | sanity 激活但无 validation gain | 直接关系约束不带来收益，Cora 常降 test | abandon | negative result |
| DIR-04 | GB-guided soft edge denoising | 用 same-ball/feature similarity 给边加 soft reliability weight | edge_denoise_only, FN+edge_denoise, reliable-ball tau screen, edge similarity diagnosis | Cora | clean setting 低于 v2/GRACE；固定 tau screen 近 no-op | 固定相似度阈值不分离 clean/noisy edge，正式证据不足 | modify | diagnostic |
| DIR-05 | GB with alternative SSL objectives | 将 GB 视图迁移到 BGRL/Barlow Twins 类目标 | gb_bgrl, gb_gbt | Cora | gb_gbt 比 gb_bgrl 稳但仍低于 GRACE/GB-GCL | 目标函数与当前 GB 视图分布错配 | abandon | negative result |
| DIR-06 | Diagnostic-only analysis | 解释候选质量、边相似度、baseline 健康性和外部 baseline | fn candidate precision, edge similarity, GRACE health, GCA/DGI/PubMed baseline | Cora, CiteSeer, PubMed, Amazon | Top-k 远好于 random；ball vs no-ball 不一致 | 诊断支持温和 claim，不支持强方法 claim | diagnostic-only | diagnostic |
| DIR-07 | GAGB noisy-label local target repair | 用 graph-aware grouping 生成局部 soft target / confidence prior，并在 frozen embedding 下做保守 repair | ordinary GB validation, GAGB v1, fair-control patch, v2 gated target, v2.1 fair repair, v2.2 class failure, v2.3 class-safe calibration | Cora, CiteSeer | v2.1 在 4/4 设置中得到正 NCR 且 CDR<0.03；v2.3 缓解 Cora 0.4 失败 | GAGB-specific value 不稳，Cora 0.4 存在间接 boundary movement，matched replay 仍缺 | modify | diagnostic |

# 3. Direction-Level Detailed Records

## DIR-01: GB-guided view generation / augmentation control

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB-guided view generation / augmentation control
- Time period: 2026-04；精确起止 Unknown
- Target problem: 让 granular-ball 直接控制图增强，保护 same-ball 边、扰动 cross-ball 边，并产生结构感知双视图。
- Core hypothesis: 粒球层级能提供比随机 edge dropout 更语义一致的 view generation。
- Baseline: GRACE，统一 encoder/projector/linear evaluation 时作为主要 baseline。
- Main related files: `src/augment/view_generator.py`, `src/augment/edge_perturb.py`, `src/augment/granular_ball.py`, `configs/cora_decay_stronger_recovery.yaml`
- Result files: `outputs/cora_long_comparison/summary.csv`, `outputs/cora_stability_comparison/summary.csv`, `outputs/cora_decay_tuning_round/summary.csv`, `outputs/cora_anchor_explore_v2_round/summary.csv`, `outputs/final_main_results.csv`
- Raw logs: `outputs/cora_decay_stronger_recovery_gb_gcl/seed_*/train.log`, `outputs/cora_anchor_v2_stronger_gb_gcl/seed_*/train.log`, other matching `outputs/cora_*_gb_gcl/seed_*/train.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 随机 view generation 可能破坏局部结构语义。
- Why GB might help: GB hierarchy 提供 same-ball/cross-ball 结构，可区分边保留概率。
- Expected improvement: 更语义一致的 augmentation，提升 node classification。
- Literature or previous experiment inspiration: Granular-ball 作为结构划分/局部先验。
- Expected paper claim: GB-guided augmentation preserves local semantics and improves GCL.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| gb_gcl mainline/stability | GB 控制 view edge keep | Cora | `outputs/cora_stability_comparison`: GB-GCL 0.7356 vs GRACE 0.7286，但该 GRACE 较弱 | 粒球是否能激活并稳定 | 能激活，不足以证明正式优于健康 GRACE |
| long comparison | 长训练协议 | Cora | GB-GCL 0.7656 vs GRACE 0.7746 | 长训练是否保留优势 | 优势不能保留 |
| stronger_late_recovery / decay | 后期减少扰动 | Cora | best 0.7482，仍低于 GRACE 0.7532 或 final GRACE 0.7586 | 视图扰动能否后期修复 | 追回一部分但不够 |
| freeze hierarchy | 冻结粒球层级 | Cora | 0.7460，低于 stronger recovery 0.7482 | 分区漂移是否主因 | 冻结不解决 |
| anchor/explore v2 | anchor 弱视图 + explore 强视图 | Cora | 0.7354，低于 stronger 0.7430 和 GRACE 0.7586 | 分工视图是否更稳 | anchor 过稳、explore 仍强 |

### 3.4 Role of Granular Ball

- GB role: 直接控制 view generation 和 edge perturbation。
- Where GB is generated: `GBGCLViewGenerator.generate()` 内按 epoch 用 `GranularBallBuilder` 生成。
- Representation space: `ball_source=z` 时使用 encoder embedding；也支持 scan 中 `x`，但主线多用 `z`。
- Affects loss: 不直接改 InfoNCE。
- Affects augmentation: 是，控制 same-ball/cross-ball edge keep、anchor/explore view。
- Affects graph structure: 是，重写增强后的 `edge_index`。
- Affects sampling: 是，按 sampled hierarchy level。
- Affects evaluation: 不直接影响 linear evaluation。

### 3.5 Result Summary

- Best result: `outputs/cora_long_comparison/summary.csv` 中 GRACE 0.7746 +/- 0.0113，GB-GCL mainline 0.7656 +/- 0.0109；GB-GCL 不是 best overall。
- Worst result: `outputs/cora_anchor_explore_v2_round/summary.csv` 中 anchor/explore 0.7354 +/- 0.0146，低于 GRACE 0.7586 +/- 0.0112。
- Average trend: 粒球结构能激活，但 clean graph 下正式比较通常低于健康 GRACE。
- Mean/std: 关键 Cora 结果均有 5 seeds；跨数据集 formal evidence 缺失。
- Baseline mean/std: GRACE 0.7586 +/- 0.0112 in `outputs/final_main_results.csv`；另有 long comparison GRACE 0.7746 +/- 0.0113。
- Delta: final table GB-GCL stronger 0.7430 vs GRACE 0.7586，约 -0.0156。
- Std overlap: 有 overlap，但均值方向对 GB-GCL 不利。
- Datasets improved: 当前正式健康 baseline 下 Unknown / none。
- Datasets worsened: Cora。
- Per-seed stability: 粒球激活相对稳定；accuracy 不稳定或不占优。
- Runtime/memory cost: 高于 GRACE；具体 runtime summary 不完整。
- Raw evidence: `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前项目直接支持：GB view generation 可以形成非退化多球结构，但“结构激活”不等于“下游准确率提升”。在 Cora 正式/长训练证据里，它不能作为优于 GRACE 的主线方法。

### 3.7 Reasonable Interpretation

解释：GB 深度介入 view generation 会改变边保留分布，训练后期仍持续引入视图分布漂移。即使分区冻结，视图噪声和层级采样仍可能破坏 GRACE 的有效正样本对齐。

### 3.8 What Cannot Be Concluded

不能断言“granular-ball 本身无效”。已有日志显示粒球能激活，失败更像 GB 在 view generation 中的角色不合适。也不能把早期/弱 baseline 下的小幅优势写成正式优于健康 GRACE。

### 3.9 Failure Analysis

- Main failure reason: 直接用 GB 重写增强视图，扰动强度和层级采样导致 representation 收敛不稳。
- Secondary failure reason: 后期 decay/freeze/anchor-explore 都增加机制复杂度，但没有稳定超过 GRACE。
- Evidence: `outputs/final_main_results.csv`, `outputs/cora_long_comparison/summary.csv`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`
- Alternative explanation: 可能是 Cora-only 结论，其他图类型缺 formal evidence。
- Minimum check needed: 如果未来复活，必须先在 Cora 之外的一个数据集做 protocol-matched 5-seed，并且不改 evaluation。

### 3.10 Lessons Learned

- Lesson: 粒球结构激活不能作为成功标准。Evidence: long comparison 中 mean_num_balls=30.94 但 test 低于 GRACE。Future constraint: 任何 GB view 方法必须同时报告 accuracy 和 GB 诊断。
- Lesson: 后期减弱扰动只能追回一部分。Evidence: stronger_late_recovery 0.7482 仍低于 GRACE 0.7532/0.7586。Future constraint: 不再只靠 schedule 微调挽救此方向。
- Lesson: anchor/explore 分工不是自动稳定器。Evidence: anchor v2 0.7354 低于 stronger 和 GRACE。Future constraint: 若再试，必须先给出视图距离/edge distribution 诊断。

### 3.11 What Is Still Useful

可保留 `GBGCLViewGenerator`、edge perturbation 诊断、mean_num_balls/same_ball_edge_ratio/cross_ball_edge_ratio 统计，以及“GB 不适合深度参与 view generation”的论文 motivation。

### 3.12 Decision

- Decision: abandon
- Reason: repeated Cora formal/long attempts低于健康 GRACE，修补方式未解决主机制问题。
- What should not be tried again: 不要继续只改 GB view schedule、freeze epoch、anchor/explore level policy 来追 Cora clean accuracy。
- Variant still worth trying: 仅作为 diagnostic 或 robustness stressor，不作为主方法。
- Evidence required to revive this direction: 至少两个数据集 protocol-matched 5-seed，稳定高于 GRACE 且说明 view distribution 为什么更好。

### Baseline Comparability Check

- Same dataset: Cora yes.
- Same split: Planetoid split / config-level same, but some early rounds used different epoch/eval settings.
- Same seeds: formal rounds mostly 5 seeds; sanity/early rounds partial.
- Same encoder/projector: mostly same.
- Same augmentation: no，GB-GCL 改 augmentation，本来就是方向变量。
- Same evaluation script: yes, `src/gb_gcl/utils/evaluation.py`。
- Same linear evaluation: mostly yes.
- Baseline rerun: yes in key rounds.
- Comparable conclusion: partial to yes for final Cora negative conclusion.

### Label Leakage Check

- Labels used in GB generation: none found. `src/augment/granular_ball.py` uses `x`, `z`, `edge_index`; no `data.y`.
- Labels used in purity: no label purity. `boundary_purity` in `src/augment/ball_quality.py` means boundary mixing, not class label purity.
- Validation/test labels used outside evaluation: no evidence.
- Leakage risk: none for training; post-hoc diagnostics handled separately in DIR-06.

## DIR-02: GB-FN-GRACE false-negative suppression

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB-FN-GRACE false-negative suppression
- Time period: 2026-04；精确起止 Unknown
- Target problem: 标准 InfoNCE 把所有非正样本当负样本，可能惩罚结构/语义相近节点。
- Core hypothesis: GB same-ball prior 能把候选范围限制到更可信的 false negatives；只对 Top-k 候选 soft downweight 能避免整球过粗。
- Baseline: GRACE, no-ball topk, GCA, DGI。
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `train.py`, `configs/cora_fn_grace_v2_topk.yaml`, `PAPER_PLAN.md`
- Result files: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, `outputs/cora_fn_grace_v2_round/summary.csv`, `outputs/cora_fn_grace_grid_round/summary.csv`, `outputs/noball_topk_round/summary.csv`, `outputs/pubmed_formal_round/summary.csv`, `outputs/stable_ranking_round/summary.csv`
- Raw logs: `outputs/cora_fn_grace_*_gb_fn_grace/seed_*/train.log`, `outputs/citeseer_fn_grace_*_gb_fn_grace/seed_*/train.log`, `outputs/pubmed_fn_grace_*_gb_fn_grace/seed_*/train.log`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: false negatives in InfoNCE denominator。
- Why GB might help: GB 提供非监督结构分组，same-ball 可作为 conservative candidate scope。
- Expected improvement: 小幅提升或至少不伤害 GRACE，同时比 GB view generation 更少干预。
- Literature or previous experiment inspiration: 从 DIR-01 失败转向 objective correction。
- Expected paper claim: GB is more useful as an auxiliary false-negative prior than as view generation.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| v1 hard/full-soft | same-ball 整球 hard/soft suppression | Cora | hard 0.7520，soft 0.7580；GRACE 0.7586 | hard vs soft | soft safer；整球仍粗 |
| v2 topk fixed-alpha | same-ball + Top-k=5 + alpha=0.25 | Cora, CiteSeer, PubMed | Cora 0.7600, CiteSeer 0.6766, PubMed 0.7790 | 少量高置信 suppression | 小幅/持平；std overlap |
| v2 ratio | top ratio 扩大候选 | Cora | 0.7544，低于 v2 | 更宽 suppression | suppression 过宽伤害 |
| strict quality | 更严格 quality threshold | Cora | 与 v2 一样 0.7600 | quality threshold 是否关键 | 不是主要旋钮 |
| grid/topk1 | topk/alpha sensitivity | Cora, CiteSeer | topk5 alpha0.35 0.7602；topk1 clean 无一致收益 | 局部超参 | 不值得更新主线 |
| v21/v22/v23 | similarity-adaptive, mass preserve, cross-view ranking/scope | Cora | 都未超过 v2 | 复杂公式是否增益 | 复杂化无收益 |
| stable ranking | stable_min/mean candidate ranking | Cora, CiteSeer | Cora stable_mean 0.7604，但 CiteSeer 低于 v2 | 跨视图稳定排序 | 不稳定，不升级 |
| no-ball topk | 全体负样本 Top-k，无 GB | Cora, CiteSeer, PubMed | 非常接近；Cora/CiteSeer 略低，PubMed 几乎持平 | GB 是否只是复杂度 | GB 额外收益很小 |
| noise/multidataset | edge noise, Amazon, Coauthor sanity | Cora, CiteSeer, PubMed, Amazon | Cora heavy noise 有小信号；CiteSeer mixed；Amazon 1 seed mixed | 边界条件 | robustness/co-purchase 证据不足 |

### 3.4 Role of Granular Ball

- GB role: 候选范围和 quality prior。
- Where GB is generated: `train.py::_maybe_refresh_fn_state()`，warmup 后按 interval 用 model embedding 构球。
- Representation space: encoder-level `model.embed(data)`，不是 classifier logits；代码中 `GB-FN-GRACE currently requires ball_source='z'`。
- Affects loss: yes，`FalseNegativeSuppressedInfoNCELoss` 修改 InfoNCE denominator weights。
- Affects augmentation: no，GB-FN-GRACE view generator 保持 GRACE augmentation。
- Affects graph structure: no，除非额外 edge denoise 分支；主线不改。
- Affects sampling: Top-k same-ball candidate selection。
- Affects evaluation: no。

### 3.5 Result Summary

- Best result: Cora stable_mean 0.7604 +/- 0.0111，但未在 CiteSeer 稳定；主线 v2 Cora 0.7600 +/- 0.0118。
- Worst result: Cora v2 ratio 0.7544 +/- 0.0118；Amazon-Photo 1 seed 0.3820 vs GRACE 0.4040。
- Average trend: Cora/CiteSeer 小幅高于 GRACE，PubMed 持平；相对 no-ball topk 只有极小优势。
- Mean/std: Cora/CiteSeer 5 seeds，PubMed 3 seeds；Amazon/Coauthor 多为 1 seed 或 sanity。
- Baseline mean/std: GRACE Cora 0.7586 +/- 0.0112；CiteSeer 0.6756 +/- 0.0140；PubMed 0.7790 +/- 0.0173。
- Delta: Cora +0.0014；CiteSeer +0.0010；PubMed +0.0000；均远小于 std。
- Std overlap: yes，统计上 inconclusive。
- Datasets improved: Cora/CiteSeer 均值小幅；Amazon-Computers 1 seed +0.0030。
- Datasets worsened: Amazon-Photo 1 seed -0.0220；CiteSeer edge_noise=0.4 低于 no-ball。
- Per-seed stability: 有 5-seed formal，但缺 paired seed significance summary。
- Runtime/memory cost: 比 GRACE 多 GB refresh，PubMed CPU 成本高；具体完整 runtime Unknown。
- Raw evidence: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, `outputs/noball_topk_round/conclusion.txt`, `outputs/pubmed_formal_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前项目直接支持：Top-k soft suppression 比 full-soft/ratio 更安全；GB-FN-GRACE v2 在 Cora/CiteSeer/PubMed 上至少不崩；粒球候选相对 no-ball topk 的收益很小但在 Cora/CiteSeer 均值略高。

### 3.7 Reasonable Interpretation

解释：真正稳定的机制可能不是“same-ball 一定语义同类”，而是“GB 限制了 pure similarity Top-k 的贪心候选范围”，产生温和正则化。GB 是 candidate quality prior，不是强监督标签。

### 3.8 What Cannot Be Concluded

不能声称 GB-FN-GRACE 显著优于 GRACE；当前提升均与 std overlap。不能声称 GB 候选标签命中率稳定优于 no-ball topk；post-hoc precision 显示 Cora/PubMed 略低，CiteSeer 略高。不能声称 co-purchase/coauthor 图上已经稳定有效。

### 3.9 Failure Analysis

- Main failure reason: GB 相对 no-ball topk 的边际信息很小，accuracy delta 太小。
- Secondary failure reason: same-ball partition 在大图上常只有约 1-2 个球，信号偏粗；在噪声/新图上激活晚或混合。
- Evidence: `outputs/noball_topk_round/conclusion.txt`, `outputs/pubmed_formal_round/summary.csv`, `outputs/fn_candidate_precision_round/conclusion.txt`
- Alternative explanation: 当前 evaluation/CPU budget 限制使部分新图只有 1 seed；不能排除更多 seed 后趋势变化。
- Minimum check needed: paired seed delta + candidate overlap/precision + loss-scale diagnostic，至少 Cora/CiteSeer/PubMed。

### 3.10 Lessons Learned

- Lesson: 精细 Top-k 比整球 suppression 更安全。Evidence: Cora v2 topk 0.7600 vs v1 full-soft 0.7580，ratio 0.7544。Future constraint: 不要扩大 suppression 范围作为默认改法。
- Lesson: soft downweight 比 hard delete 更稳。Evidence: Cora soft 0.7580 vs hard 0.7520。Future constraint: 不要重试 hard same-ball masking，除非有新置信度证据。
- Lesson: 复杂 weighting 没有明显增益。Evidence: v21/v22/v23/stable ranking 均未稳定超过 v2。Future constraint: 不要继续 formula shopping。
- Lesson: no-ball topk 是强对照。Evidence: Cora 0.7592、CiteSeer 0.6760、PubMed 0.7787。Future constraint: 任何新 GB-FN claim 必须同时过 no-ball topk。

### 3.11 What Is Still Useful

保留 v2 topk-soft 实现、no-ball topk 对照、post-hoc fn candidate precision、noise robustness 表，以及论文主张中“GB 是温和结构先验，不是强语义标签”的更严格表达。

### 3.12 Decision

- Decision: modify
- Reason: 当前方向最接近主线，但证据强度不足以 strict continue；应降低 claim 强度并补机制诊断，而不是继续发明复杂变体。
- What should not be tried again: 不要重试 hard masking、ratio 宽 suppression、mass-preserve、similarity-adaptive、cross-only、stable_min、topk1、fuzzy affinity 作为主线升级。
- Variant still worth trying: v2 topk-soft 作为 light prior；stable_mean 只可作为 appendix weak signal，不作为默认。
- Evidence required to revive this direction as `continue`: 至少 Cora/CiteSeer/PubMed 上 paired multi-seed delta 不被 std 淹没，并明确优于 no-ball topk 或给出充分机制解释。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer, PubMed formal；Amazon-Photo/Computers partial；Coauthor-CS sanity。
- Seeds: Cora/CiteSeer 5 seeds；PubMed 3 seeds；Amazon/Coauthor mostly 1 seed。
- Epochs: 主表多为 100 pretrain / 100 eval；sanity 为 30/50 或 1 epoch。
- Encoder: GCN encoder via `GRACEModel`。
- Projector: 2-layer projection head。
- Base loss: InfoNCE。
- Added mechanism: soft negative downweight on selected candidates。
- GB loss weight: alpha=0.25 for suppression；不是 additive GB loss。
- Augmentation: GRACE edge dropout + feature masking unchanged。
- Ball generation method: `GranularBallBuilder`, `ball_source=z`, warmup 20, interval 5。
- Evaluation protocol: frozen encoder linear evaluation，best val selects test。

### Baseline Comparability Check

- Same dataset: yes for main table。
- Same split: Planetoid fixed; Amazon/Coauthor RandomNodeSplit but only partial。
- Same seeds: yes for Cora/CiteSeer；PubMed 3 seeds；partial datasets no。
- Same encoder/projector: yes。
- Same augmentation: yes for GB-FN mainline vs GRACE。
- Same evaluation script: yes。
- Baseline rerun: yes in final main results。
- Comparable conclusion: yes for Cora/CiteSeer/PubMed small/flat trend；not enough for Amazon/Coauthor。

### Label Leakage Check

- Labels used in GB generation: no evidence。`src/augment/granular_ball.py` only uses `x`, `z`, `edge_index` and detaches representations。
- Labels used in purity: no class-label purity。`boundary_purity` is structural boundary mixing。
- Labels used in sampling/coarsening: no evidence。
- Validation/test labels used outside evaluation: no evidence in training。
- Supervised GB mechanism used in pretraining: no。
- Leakage risk: none for training; labels appear in evaluation and DIR-06 post-hoc diagnostics only。

### Granular-Ball Diagnostics

- Number of balls: Cora v2 mainline short summary 1.33; Cora stable/formal variants 1.65；PubMed formal 1.72；GB-GCL view direction often ~30。
- Average ball size: Cora short high-quality ball size ~451-880 depending protocol；PubMed average not fully summarized。
- Max ball size: Unknown across all formal seeds。
- Singleton ball ratio: TODO。
- Radius mean/std: TODO。
- Coverage: implied full assignment but not explicitly summarized。
- Overlap/conflict: TODO。
- Relation density: same_ball_pair_ratio Cora around 0.35；PubMed around 0.40；CiteSeer short around 0.17。
- Confidence distribution: ball quality distribution TODO。
- Cross-epoch stability: partial via refresh/suppression epochs; detailed stability TODO。
- Cross-seed stability: mean/std available for accuracy, not for assignment overlap。
- GB loss scale vs contrastive loss: suppression ratio ~0.0012 on Cora；not additive loss。
- Too coarse: PubMed and mainline often only 1-2 balls, likely coarse。
- Too fragmented: not supported for v2; more relevant to GB view direction。
- TODO: radius, singleton ratio, per-seed assignment stability。

### Next Minimal Experiment If Modified

- New hypothesis: GB-FN 的价值是温和结构 prior，而非显著提升 accuracy 的强方法。
- Change only: 不改训练公式；补 paired diagnostics：candidate overlap、per-seed delta、selected candidates 的 label precision/affinity/loss contribution。
- Keep fixed: v2 topk-soft, no-ball topk, GRACE, same seeds/splits/evaluation。
- Minimal datasets: Cora + CiteSeer + PubMed existing checkpoints first；只在缺 stats 时重跑。
- Minimal seeds: Cora/CiteSeer 5, PubMed 3。
- Success criterion: 能解释 GB 相对 no-ball 的差异，且不依赖单一数据集。
- Failure criterion: 若 candidate quality/overlap/loss scale 仍无法区分 GB 与 no-ball，则论文 claim 降级为 appendix/diagnostic。
- Why this is the smallest useful test: 当前瓶颈不是又一个公式，而是机制证据不足。

## DIR-03: Direct GB relation regularization

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: Direct GB relation regularization
- Time period: 2026-04；精确起止 Unknown
- Target problem: GB-FN v2 的负样本修正太弱，尝试让 same-ball/prototype/consistency 直接提供额外正向约束。
- Core hypothesis: 同球节点、球原型或同层级一致性可以作为额外 semantic relation。
- Baseline: same-protocol short GB-FN-GRACE v2。
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `configs/cora_fn_grace_v3_proto.yaml`, `configs/cora_fn_grace_v3_ball_positive_top1_lam002.yaml`, `configs/cora_fn_grace_v3_ball_consistency_lam001.yaml`
- Result files: `outputs/ball_prototype_round/summary.csv`, `outputs/ball_positive_round/summary.csv`, `outputs/ball_consistency_round/summary.csv`
- Raw logs: matching `outputs/cora_fn_grace_v3_*_gb_fn_grace/seed_0/train.log`, `outputs/citeseer_fn_grace_v3_*_gb_fn_grace/seed_0/train.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: v2 只 downweight 极少负样本，信号可能过弱。
- Why GB might help: 同球节点、prototype 和 consistency 可能增强局部语义紧致性。
- Expected improvement: 在 v2 上增加额外正则，提升 validation/test。
- Literature or previous experiment inspiration: prototype/center constraints and positive-pair construction。
- Expected paper claim: GB relation constraints can improve node-level GCL beyond false-negative suppression.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| v3 prototype | node-to-ball prototype positive branch | Cora, CiteSeer | Cora test -0.008；CiteSeer val/test -0.004/-0.003 vs short v2 | prototype pulling | 机制激活但不增益 |
| v3.2 ball consistency | single-hierarchy consistency regularization, lambda 0.01/0.05 | Cora, CiteSeer | Cora test -0.001；CiteSeer tied | consistency loss | loss 很小，信号弱 |
| v3.3 shared-ball positives | same-ball soft positives top1/top2, lambda 0.02 | Cora, CiteSeer | Cora -0.002 val / -0.007 test；CiteSeer tied | direct positives | top1/top2 均不达成功条件 |

### 3.4 Role of Granular Ball

- GB role: 直接构造正关系、prototype 或 consistency regularizer。
- Where GB is generated: 复用 GB-FN 的 `z` 空间 hierarchy。
- Representation space: encoder embedding `z`。
- Affects loss: yes，additive prototype/consistency/positive loss。
- Affects augmentation: no。
- Affects graph structure: no。
- Affects sampling: same-ball/topm/prototype anchor 选择。
- Affects evaluation: no。

### 3.5 Result Summary

- Best result: CiteSeer consistency/positive 与 short v2 持平。
- Worst result: Cora prototype test 0.6550 vs short v2 0.6630；Cora ball positive 0.6560 vs 0.6630。
- Average trend: 机制可激活，但没有 validation improvement。
- Mean/std: 只有 sanity 1 seed，std 不可用。
- Baseline mean/std: same-protocol short v2，1 seed。
- Delta: Cora 多数负；CiteSeer 多为平。
- Std overlap: Unknown。
- Datasets improved: none。
- Datasets worsened: Cora；CiteSeer prototype slight worse。
- Per-seed stability: Unknown。
- Runtime/memory cost: 小幅增加；具体 Unknown。
- Raw evidence: `outputs/ball_prototype_round/conclusion.txt`, `outputs/ball_positive_round/conclusion.txt`, `outputs/ball_consistency_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前 sanity 证据支持：prototype/positive/consistency 分支技术上能激活，但没有产生正向 validation signal，因此没有进入 formal 5-seed 的价值。

### 3.7 Reasonable Interpretation

解释：hard/shared-ball relation 对 node-level GCL 来说仍过粗；直接拉近或正则同球节点容易削弱 instance discrimination，或者信号太弱到不起作用。

### 3.8 What Cannot Be Concluded

不能断言所有 prototype/positive 方法都失败；当前只测试了本项目实现、短程 1 seed sanity 和固定小权重。不能作为 formal 多 seed 负结果。

### 3.9 Failure Analysis

- Main failure reason: 同球关系不足以直接当正样本或 prototype 目标。
- Secondary failure reason: consistency loss scale 很小；positive loss 虽激活但没有验证增益。
- Evidence: `outputs/ball_consistency_round/summary.csv`, `outputs/ball_positive_round/summary.csv`
- Alternative explanation: sanity protocol 可能过短。
- Minimum check needed: 必须先在 sanity 中超过 short v2 validation，再考虑 formal。

### 3.10 Lessons Learned

- Lesson: 直接正样本/原型拉近比 negative downweight 风险更高。Evidence: Cora prototype/positive test 明显下降。Future constraint: 不要把 same-ball 当 hard semantic positive。
- Lesson: consistency loss scale 过小不会解决 v2 信号弱问题。Evidence: raw consistency loss ~5.8e-4 / 3.3e-4。Future constraint: 先看 loss scale，再跑 formal。
- Lesson: topm 增加不等于更好。Evidence: positive top1/top2 结果 tied。Future constraint: 不用 topm sweep 替代机制诊断。

### 3.11 What Is Still Useful

可保留实现和 sanity 负结果，用于论文 appendix 或 future work 说明：GB relation 直接作为 positive/prototype 不可靠。

### 3.12 Decision

- Decision: abandon
- Reason: 多个直接关系分支在 sanity stop rule 下未产生正向 validation signal。
- What should not be tried again: 不要在当前框架下继续微调 prototype weight、positive topm 或 consistency lambda。
- Variant still worth trying: 若要复活，必须改成 confidence-weighted/uncertainty-aware auxiliary diagnostic，而非直接拉近。
- Evidence required to revive this direction: 同一 sanity protocol 下 Cora/CiteSeer 均超过 short v2 validation，并报告 loss scale 不压倒 InfoNCE。

### Baseline Comparability Check

- Same dataset: yes for sanity Cora/CiteSeer。
- Same split: yes。
- Same seeds: only seed 0。
- Same encoder/projector: yes。
- Same augmentation: yes。
- Same evaluation script: yes。
- Baseline rerun: same-protocol short v2 reused。
- Comparable conclusion: partial；足够做 stop/go，不足以做 formal claim。

### Granular-Ball Diagnostics

- Number of balls: 约 1.333，max 2。
- Active anchor ratio: prototype/positive/consistency mostly 0.3333。
- Mean num prototypes/positives/consistency balls: 0.3333-0.6667。
- Selected positive similarity: Cora ~0.32；CiteSeer ~0.33。
- GB loss scale vs contrastive loss: positive raw loss ~2.27 with weight 0.02；consistency raw loss ~1e-4 到 1e-3。
- TODO: additive loss vs InfoNCE per-epoch ratio。

## DIR-04: GB-guided soft edge denoising

### 3.1 Direction Summary

- Direction ID: DIR-04
- Direction name: GB-guided soft edge denoising
- Time period: 2026-04；精确起止 Unknown
- Target problem: 输入图可能有 noisy edges，GB/embedding similarity 可给边估计 reliability。
- Core hypothesis: 给低可靠边较低 edge_weight 可以提高 robustness。
- Baseline: GRACE, GB-FN-GRACE v2, no-ball topk。
- Main related files: `src/gb_gcl/edge_reliability.py`, `train.py`, `configs/cora_fn_grace_v3_soft_edge_denoise_both.yaml`, `configs/cora_fn_grace_soft_edge_denoise_only.yaml`
- Result files: `outputs/cora_fn_grace_v3_soft_edge_denoise_both_gb_fn_grace/summary.json`, `outputs/cora_fn_grace_soft_edge_denoise_only_gb_fn_grace/summary.json`, `outputs/edge_similarity_diag_round/summary.csv`, `outputs/noiseaware_ball_round/summary.csv`
- Raw logs: `outputs/cora_fn_grace_v3_soft_edge_denoise*_gb_fn_grace/seed_*/train.log`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: graph corruption/noisy edges undercut augmentation and message passing。
- Why GB might help: same-ball + feature similarity 可以给边软可靠性估计。
- Expected improvement: clean graph 不伤害，noisy graph 下提升 robustness。
- Literature or previous experiment inspiration: edge denoising and soft reliability weighting。
- Expected paper claim: GB denoising improves robustness under graph noise.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| edge_denoise_only | 不启用 FN，只给边 soft weight | Cora clean | 0.7560 +/- 0.0172，低于 GRACE 0.7586 | 单独 edge reliability | clean 下不增益 |
| FN + edge_denoise both | v2 topk + edge reliability | Cora clean | 0.7552 +/- 0.0164，低于 v2 0.7600 | 组合是否更强 | 组合拖累 |
| reliable-ball tau screen | reliable topology for ball construction | Cora edge_noise=0.4 quick screen | tau 0.3/0.5/0.7 identical，keep_ratio=1.0 | 固定 tau 是否筛边 | near no-op |
| edge similarity diagnosis | clean/noisy edge cosine distribution | Cora edge_noise=0.4 | fixed tau 不清楚分离；建议 percentile filter | 诊断阈值 | 固定绝对阈值 brittle |

### 3.4 Role of Granular Ball

- GB role: edge reliability score 中的 same-ball signal，以及 reliable-ball topology construction。
- Where GB is generated: `train.py::_build_edge_denoise_state()` and `GranularBallBuilder`。
- Representation space: encoder embedding。
- Affects loss: 间接通过 edge weights affecting model forward。
- Affects augmentation: yes，view edges 获得 `edge_weight`。
- Affects graph structure: 不删除主图边，但赋 soft weights；reliable-ball variant 可筛 ball construction edges。
- Affects sampling: no。
- Affects evaluation: no。

### 3.5 Result Summary

- Best result: edge_denoise_only Cora clean 0.7560 +/- 0.0172，仍低于 GRACE/v2。
- Worst result: FN+edge_denoise both 0.7552 +/- 0.0164，低于 v2 0.7600。
- Average trend: clean graph 不增益；noisy graph 下固定阈值方案没有形成 formal positive evidence。
- Mean/std: Cora clean 5 seeds；reliable tau screen 1 seed/30 epoch。
- Baseline mean/std: GRACE 0.7586 +/- 0.0112；v2 0.7600 +/- 0.0118。
- Delta: edge_denoise_only vs GRACE -0.0026；both vs v2 -0.0048。
- Std overlap: yes。
- Datasets improved: none in formal clean setting。
- Datasets worsened: Cora clean。
- Per-seed stability: formal summary exists but no paired significance。
- Runtime/memory cost: higher due edge scoring；exact Unknown。
- Raw evidence: `outputs/edge_similarity_diag_round/conclusion.txt`, `outputs/noiseaware_ball_round/conclusion.txt`

### 3.6 What Is Directly Supported

当前项目直接支持：现有 fixed sigmoid/absolute-threshold edge reliability 方案不应作为主线；edge similarity diagnosis 显示 clean/noisy edge cosine 分布不是固定 tau 容易分开的。

### 3.7 Reasonable Interpretation

解释：表示空间的 cosine scale 会随训练漂移，固定阈值不稳；edge denoise 如果要继续，应转成 percentile/adaptive filter，而不是继续调固定 tau。

### 3.8 What Cannot Be Concluded

不能断言 GB edge denoising 永远无效；当前没有 percentile filter formal 结果，也没有多数据集 noisy graph formal 结果。

### 3.9 Failure Analysis

- Main failure reason: fixed reliability score 对 clean/noisy edge 分离不足。
- Secondary failure reason: clean setting 中 edge weighting 增加复杂度并可能扰乱 GRACE/FN 表示。
- Evidence: `outputs/edge_similarity_diag_round/summary.csv`, `outputs/noiseaware_ball_round/summary.csv`
- Alternative explanation: edge_noise protocol 或 quick screen 太短。
- Minimum check needed: percentile-based denoise on Cora edge_noise=0.4/0.5 with GRACE/no-ball/v2 controls。

### 3.10 Lessons Learned

- Lesson: fixed tau 不能直接依赖。Evidence: reliable_edge_keep_ratio=1.0，tau 0.3/0.5/0.7 identical。Future constraint: 不再跑 fixed-tau formal。
- Lesson: clean graph edge weighting 不是免费增益。Evidence: clean Cora 低于 v2/GRACE。Future constraint: robustness claim 必须在 noisy setting 证明。
- Lesson: edge similarity diagnosis 比盲跑 formal 更有价值。Evidence: diag 建议 percentile filter。Future constraint: 先诊断分布，再设计 denoiser。

### 3.11 What Is Still Useful

保留 `edge_reliability.py`、edge similarity hist/summary、clean/noisy edge metadata pipeline。这个方向适合做 robustness appendix 或 future work。

### 3.12 Decision

- Decision: modify
- Reason: 当前实现失败，但诊断给出了明确的最小改法：percentile/adaptive filter。
- What should not be tried again: 不要重跑 fixed tau=0.3/0.5/0.7 或 clean-only edge denoise formal。
- Variant still worth trying: percentile_filter / per-epoch adaptive threshold，仅在 noisy graph 下。
- Evidence required to revive this direction: noisy Cora/CiteSeer 5-seed 中相对 GRACE/no-ball/v2 有明确提升，并提供 clean/noisy edge keep 诊断。

### Next Minimal Experiment If Modified

- New hypothesis: percentile edge reliability 比 fixed tau 更能适应 cosine scale 漂移。
- Change only: reliability threshold 从 fixed tau 改为 percentile filter；其他 v2 topk-soft 不动。
- Keep fixed: Cora edge_noise=0.4/0.5, existing split/seeds/eval。
- Minimal datasets: Cora first；CiteSeer only if Cora passes。
- Minimal seeds: 3 seeds quick formal, then 5 seeds if >= +0.005。
- Success criterion: test mean 高于 GRACE/no-ball/v2，且 std 不完全淹没；noisy edge keep 明显低于 clean edge keep。
- Failure criterion: edge keep 分布无分离或 accuracy <= v2。
- Why this is the smallest useful test: 直接针对已发现的 fixed tau no-op。

## DIR-05: GB with alternative SSL objectives

### 3.1 Direction Summary

- Direction ID: DIR-05
- Direction name: GB with alternative SSL objectives
- Time period: 2026-04；精确起止 Unknown
- Target problem: InfoNCE/GRACE 与 GB view 可能不匹配，尝试 bootstrap 或 redundancy-reduction 目标。
- Core hypothesis: BGRL/Barlow Twins 类目标可能更能容忍 GB weak/strong views。
- Baseline: GRACE, GB-GCL stronger_late_recovery。
- Main related files: `src/gb_gcl/models/contrastive.py`, `src/gb_gcl/losses/bgrl_loss.py`, `src/gb_gcl/losses/barlow_twins.py`, `train.py`
- Result files: `outputs/cora_gb_bgrl_round/summary.csv`, `outputs/cora_gbt_round/summary.json`
- Raw logs: `outputs/cora_gb_bgrl_main_gb_bgrl/seed_*/train.log`, `outputs/cora_gb_gbt_main_gb_gbt/seed_*/train.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: contrastive objective may conflict with GB view perturbation。
- Why GB might help: weak/strong GB views could fit bootstrap/BT objectives。
- Expected improvement: 比 GB-GCL stronger 更稳，追回 GRACE。
- Literature or previous experiment inspiration: BGRL and Barlow Twins style SSL。
- Expected paper claim: GB works beyond InfoNCE as a general graph SSL prior.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-BGRL | teacher/student bootstrap | Cora | 0.4164 +/- 0.0456 | bootstrap with GB views | 明显失败 |
| GB-GBT | Barlow Twins objective | Cora | 0.7304 +/- 0.0138 | redundancy reduction | 比 BGRL 稳，但低于 GRACE 0.7586 和 GB-GCL 0.7430 |

### 3.4 Role of Granular Ball

- GB role: 构造 weak/student/teacher graph views。
- Where GB is generated: view generator。
- Representation space: encoder `z` for ball construction。
- Affects loss: 通过换 objective；GB 不直接做 additive loss。
- Affects augmentation: yes。
- Affects graph structure: yes。
- Affects sampling: yes，hierarchy level。
- Affects evaluation: no。

### 3.5 Result Summary

- Best result: GB-GBT 0.7304 +/- 0.0138。
- Worst result: GB-BGRL 0.4164 +/- 0.0456。
- Average trend: 替代目标没有救回 GB view direction。
- Mean/std: Cora 5 seeds。
- Baseline mean/std: GRACE 0.7586 +/- 0.0112；GB-GCL stronger 0.7430 +/- 0.0157。
- Delta: GB-GBT vs GRACE -0.0282；GB-BGRL vs GRACE -0.3422。
- Std overlap: GB-GBT 与 GRACE 不构成正向；GB-BGRL 明显失败。
- Datasets improved: none。
- Datasets worsened: Cora。
- Per-seed stability: available；BGRL high variance。
- Runtime/memory cost: BGRL has target model; exact Unknown。
- Raw evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`

### 3.6 What Is Directly Supported

在当前项目里，GB-BGRL/GB-GBT 不适合作为论文主线。目标函数替换没有解决 GB view 与表示学习之间的错配。

### 3.7 Reasonable Interpretation

解释：GB-BGRL 的 teacher 原图和 student 强 GB view 差距过大，loss 快速趋近零但表示没有足够信息约束。GB-GBT 更稳定但仍没有提供更好的线性评估。

### 3.8 What Cannot Be Concluded

不能说 BGRL/Barlow Twins 在图上无效；只说明当前 GB view + 当前实现组合失败。

### 3.9 Failure Analysis

- Main failure reason: objective 与 GB view distribution 错配。
- Secondary failure reason: BGRL teacher/student gap 和 collapse-like near-zero loss。
- Evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`
- Alternative explanation: 需要单独为 BGRL/GBT 调视图强度，但这会变成新项目。
- Minimum check needed: 若复活，先做 1 seed loss/embedding collapse diagnostic，而非 formal。

### 3.10 Lessons Learned

- Lesson: 换 objective 不是自动修复 GB view 的捷径。Evidence: GBT 低于 GRACE 和 GB-GCL。Future constraint: 不要用新目标掩盖 view 机制问题。
- Lesson: Bootstrap target 需要严格控制 view gap。Evidence: BGRL 0.4164。Future constraint: 不再用原图 teacher + 强 GB student 作为默认。

### 3.11 What Is Still Useful

可作为负结果和实现对照；如果论文需要说明“我们聚焦 InfoNCE objective correction 而非替换 SSL 框架”，此方向可提供背景。

### 3.12 Decision

- Decision: abandon
- Reason: Cora 5-seed 明显低于现有主线和 GRACE。
- What should not be tried again: 不要在当前项目继续微调 GB-BGRL/GB-GBT 作为主线。
- Variant still worth trying: only diagnostic, with much smaller teacher/student view gap。
- Evidence required to revive this direction: 先超过 GB-GCL stronger，再接近 GRACE；同时证明没有 collapse。

## DIR-06: Diagnostic-only analysis

### 3.1 Direction Summary

- Direction ID: DIR-06
- Direction name: Diagnostic-only analysis
- Time period: 2026-04；精确起止 Unknown
- Target problem: 解释为什么 GB-FN 小幅有效/不显著，以及 baseline 是否健康。
- Core hypothesis: 诊断能约束 paper claim，避免把弱信号讲成强方法。
- Baseline: not a method baseline；分析 GRACE/GCA/DGI/no-ball/GB-FN。
- Main related files: `src/gb_gcl/analysis/fn_candidate_precision.py`, `src/gb_gcl/analysis/edge_similarity.py`, `outputs/grace_health_check_round/*`
- Result files: `outputs/fn_candidate_precision_round/summary.csv`, `outputs/edge_similarity_diag_round/summary.csv`, `outputs/grace_health_check_round/summary.csv`, `outputs/external_baselines_round/summary.csv`, `outputs/dgi_baseline_round/conclusion.txt`, `outputs/pubmed_gca_round/conclusion.txt`
- Raw logs: Diagnostics use existing checkpoints/logs; label precision produces summary only.
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: unclear whether accuracy gains come from true false-negative quality, baseline weakness, or artifact。
- Why GB might help: Diagnostics can show selected candidates/edges are meaningful。
- Expected improvement: Not an accuracy method; expected better explanation。
- Literature or previous experiment inspiration: method validation and ablation discipline。
- Expected paper claim: GB-FN is a light structural prior, with limitations.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| fn candidate precision | labels used post-hoc only | Cora, CiteSeer, PubMed | Top-k far above random; ball vs no-ball mixed | 候选是否像 false negatives | 相似度 Top-k 是主信号，GB 不是大幅纯度提升 |
| edge similarity diag | clean/noisy edge cosine distribution | Cora edge_noise | not clearly separable | fixed tau denoise 是否合理 | fixed threshold brittle |
| GRACE health check | baseline tuning screen | PubMed, Amazon | defaults not clearly unhealthy | baseline 是否坏 | 低结果不是简单 baseline bug |
| external baselines | GCA/DGI/PubMed GCA | Cora, CiteSeer, PubMed | GCA strong on CiteSeer, weak on PubMed; DGI lower | main table context | claim must be dataset-specific |

### 3.4 Role of Granular Ball

- GB role: Diagnostic subject, candidate scope, post-hoc replay。
- Where GB is generated: Existing checkpoints/runs。
- Representation space: saved best checkpoint embeddings。
- Affects loss: no，diagnostic-only。
- Affects augmentation: no。
- Affects graph structure: no。
- Affects sampling: only replayed candidate selection。
- Affects evaluation: no。

### 3.5 Result Summary

- Best result: Top-k selected candidates have much higher label Precision@k than random on all analyzed datasets.
- Worst result: ball-topk does not consistently beat no-ball topk in Precision@k。
- Average trend: diagnostics weaken strong GB claim but support a conservative prior story。
- Mean/std: fn precision Cora/CiteSeer 5 seeds；PubMed 1 seed in this diagnostic。
- Baseline mean/std: random topk included。
- Delta: ball vs no-ball Precision@k Cora -0.0029，CiteSeer +0.0010，PubMed -0.0003。
- Std overlap: yes/mixed。
- Datasets improved: diagnostic only。
- Datasets worsened: diagnostic only。
- Per-seed stability: available for Cora/CiteSeer precision。
- Runtime/memory cost: analysis only。
- Raw evidence: `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/edge_similarity_diag_round/conclusion.txt`

### 3.6 What Is Directly Supported

Labels are used only for post-hoc analysis and never participate in training or candidate selection updates, per `src/gb_gcl/analysis/fn_candidate_precision.py`. Selected Top-k candidates are far more label-consistent than random, but GB does not consistently improve label precision over no-ball topk.

### 3.7 Reasonable Interpretation

解释：GB 的作用更像限制 pure similarity 的候选分布/时机，而不是显著提高同类命中率。论文应强调 lightweight prior and negative results，而不是大幅候选质量跃迁。

### 3.8 What Cannot Be Concluded

不能用 post-hoc labels 反推训练中有监督信息；也不能用 Precision@k 直接替代最终 accuracy。

### 3.9 Failure Analysis

- Main failure reason: 诊断没有证明 GB 相比 no-ball 有强候选质量优势。
- Secondary failure reason: PubMed diagnostic seed 不完整。
- Evidence: `outputs/fn_candidate_precision_round/summary.csv`
- Alternative explanation: GB benefit may lie in training dynamics, not final checkpoint candidate purity。
- Minimum check needed: candidate overlap/loss impact over training epochs。

### 3.10 Lessons Learned

- Lesson: no-ball topk 是必须 baseline。Evidence: precision and accuracy both close。Future constraint: paper main line must include it。
- Lesson: labels can be used post-hoc but must be explicitly separated from training。Evidence: code note `label_usage`。Future constraint: every label diagnostic must state leakage boundary。
- Lesson: external baselines can reverse dataset story。Evidence: GCA > GB-FN on CiteSeer but < GRACE on PubMed under current protocol。Future constraint: avoid universal SOTA wording。

### 3.11 What Is Still Useful

这些诊断是当前项目最有价值的论文约束：能支持“温和、可解释、有限增益”的 claim，并阻止过度包装。

### 3.12 Decision

- Decision: diagnostic-only
- Reason: 该方向不是主方法，但能解释和约束主方法。
- What should not be tried again: 不要把 post-hoc label precision 写成训练时 supervised signal。
- Variant still worth trying: per-epoch candidate overlap and loss contribution diagnostic。
- Evidence required to revive this direction: Not applicable; already diagnostic-only。

### Label Leakage Check

- Labels used in GB generation: no。
- Labels used in post-hoc precision: yes，explicitly diagnostic only。
- Validation/test labels used outside evaluation: only post-hoc analysis after training。
- Leakage risk: low if described honestly; high if written as training evidence。

## DIR-07: GAGB noisy-label local target repair

### 3.1 Direction Summary

- Direction ID: DIR-07
- Direction name: GAGB noisy-label local target repair
- Time period: 2026-05 to 2026-06
- Target problem: 在 frozen GRACE embeddings 上做带噪标签节点分类，验证 graph-aware granular balls 是否能作为局部 soft-label / confidence prior，帮助保守地修复 noisy train labels。
- Core hypothesis: 如果 graph-aware grouping 比 ordinary embedding-only GB 更能定义可靠局部区域，那么 ball-level soft target、teacher-aware q_mix 和 conflict gate 应该在少量 train nodes 上产生有效纠错，同时保持低 clean damage。
- Baseline: `gb_noisy_label_validation` 的 ordinary GB local smoothing，以及 `gb_graph_aware_minival_6_1` 中与 `gb`、`gagb`、`khop_only`、`kmeans_gagb_matched`、`topk_gagb_train_matched`、shuffled controls 的同协议比较。
- Main related files: `gb_graph_aware_minival_6_1\src\noisy_label_gb_grace\balls\graph_aware_ball.py`, `gb_graph_aware_minival_6_1\src\noisy_label_gb_grace\smoothing\targets.py`, `gb_graph_aware_minival_6_1\scripts\run_noisy_classifier.py`, `gb_graph_aware_minival_6_1\scripts\diagnose_gagb_v2_class_failure.py`, `gb_graph_aware_minival_6_1\scripts\diagnose_gagb_v2_3_class_safe_calibration.py`, `claude_gagb_experiment_pack.html`
- Result files: `gb_noisy_label_validation\reports\2026-05-21_coverage_aware_validation_report.md`, `gb_graph_aware_minival_6_1\reports\2026-06-01_gagb_minival_result.md`, `gb_graph_aware_minival_6_1\reports\2026-06-01_gagb_fair_minival_result.md`, `gb_graph_aware_minival_6_1\reports\2026-06-01_gagb_v2_target_minival_result.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_gated_diagnostic_20260601\gated_nodes_diagnostic_report.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_1_fair_repair_fixed_20260602\fair_repair_report.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_2_class_failure_20260602\cora_0p4_failure_report.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_3_class_safe_calibration_20260602\v2_3_report.md`
- Raw logs: `gb_graph_aware_minival_6_1\results\**\*.json`, `gb_graph_aware_minival_6_1\reports\**\*.csv`, `gb_graph_aware_minival_6_1\reports\**\confusion_matrices.json`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 之前的 GB view generation、hard positive、compactness encoder regularization 都太强；如果继续在 encoder 侧用 GB，容易破坏原始 GCL。
- Why GB might help: frozen embedding 阶段更安全，graph-aware grouping 可能比 ordinary GB 定义更可靠的局部区域，用来生成 soft target、teacher prior 和 conflict gate。
- Expected improvement: 在不重训 GRACE 的前提下，修复少量 noisy train labels，并优于 ordinary GB、khop、kmeans、topk 和 shuffled controls。
- Literature or previous experiment inspiration: 来自 ordinary GB noisy-label validation 的弱正信号，以及 Stage A/B 对“不要强行训练 encoder”的负结果。
- Expected paper claim: GAGB provides a conservative local repair prior for noisy-label node classification under frozen GCL embeddings.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| ordinary GB validation | 只用 ordinary GB / coverage-aware GB 做 local smoothing | Cora, CiteSeer | Cora 局部正向，CiteSeer 不稳 | local smoothing 本身是否值得继续 | 只能作为弱 baseline，不能直接立主线 |
| GAGB v1 minival | embedding + topology greedy grouping，alpha=0.3 local smoothing | Cora, CiteSeer | purity/entropy 更好，但 accuracy 极小波动；Cora 0.4 明显更弱 | graph-aware grouping 是否天然更优 | 独立价值未证明 |
| v1 fair-control patch | 加入 emb-only、kmeans matched、topk matched、alpha-matched shuffled | Cora, CiteSeer | 仍不能稳定打过 khop / shuffled / matched clustering | v1 是否只是对照不公平 | 不是；v1 机制本身就不够强 |
| v2 target-only | 保持构球不变，只加入 noisy-label-only teacher、q_pred、conflict gate | Cora, CiteSeer | accuracy mixed，表面上 NCR 仍为 0 | q_mix 是否有纠错信号 | 有信号，但被旧 alpha03 update 掩盖 |
| gated diagnostic | 统计 gated noisy ratio、possible before alpha、lost by alpha | Cora, CiteSeer | 84%-99% possible corrections lost by alpha | v2 失败是否因“没有信号” | 不是；是 target mixing 太弱 |
| v2.1 fair repair | `alpha09` / `replace_qmix` + bug-fixed fair controls | Cora, CiteSeer | 4/4 settings NCR>0 且 CDR<0.03；Cora 0.4 仍失败 | repair 是否比 inert alpha03 更有机制价值 | mechanism-positive，部分 method-positive |
| v2.2 class failure | per-class NCR/CDR/confusion/failure report | 重点 Cora 0.4 | class 1 / 6 大幅下跌但 direct clean damage 很小 | Cora 0.4 为什么失败 | 主要是间接 boundary movement，不是大规模 clean damage |
| v2.3 class-safe calibration | alpha06/07/08、class-capped、class-balanced、boundary-safe、replay | Cora, CiteSeer | 缓解 Cora 0.4，但仍未超过 gb；replay control 过粗 | 是否是 target strength / class concentration 问题 | 是；但仍未完成 GAGB-specific 证明 |

### 3.4 Role of Granular Ball

- GB role: frozen-embedding 阶段的 graph-aware local grouping、soft-label prior、confidence prior 和 repair gate support。
- Where GB is generated: `build_graph_aware_balls()` 在冻结的 GRACE embedding 和图拓扑上构球。
- Representation space: saved frozen GRACE embedding + k-hop topology；当前 v1/v2 builder 主要是 `lambda_h=0.7`, `lambda_t=0.3` 的 embedding+topology greedy grouping。
- Affects loss: yes，通过局部 soft target / weighted CE 间接影响 classifier loss。
- Affects augmentation: no。
- Affects graph structure: no；不重写 pretraining graph。
- Affects sampling: no；只改变 target/weight，不改 split。
- Affects evaluation: no；评估仍在固定 split 上进行。

### 3.5 Result Summary

- Best result: v2.1 中 `Cora 0.2` 的 `replace_qmix` 达到 `0.8090`，`alpha09` 达到 `0.8083`，均显著高于同设置 `alpha03=0.8039`，同时 `NCR=0.2519`、`CDR=0.0143`；`CiteSeer 0.2` 的 `alpha09=0.7087` 高于 `alpha03=0.7058`。
- Worst result: `Cora 0.4` 是持续弱设置；v2.1 `alpha09=0.7907`、`replace=0.7886`，均低于 `alpha03=0.7949`；v1 raw `gagb` 也在该设置输给 ordinary `gb`。
- Average trend: v1 grouping alone 主要改善 purity/entropy，但不稳定改善 accuracy；v2.1 repair 首次把 4/4 设置的 NCR 从 0 变成正值，且 CDR 有界；v2.3 class-safe calibration 明显缓解 Cora 0.4，但仍未稳定打过 `gb`。
- Mean/std: 核心 minival 基于 Cora/CiteSeer、noise 0.2/0.4、5 seeds；v2.3 的 `Cora 0.4` 中 `gb=0.7968±0.0056`，`class_capped_alpha09=0.7958±0.0065`，`class_balanced_alpha09=0.7957±0.0058`，`boundary_safe_alpha09=0.7959±0.0043`。
- Baseline mean/std: within-direction baseline 主要是 ordinary `gb`、plain `gagb`、`alpha03`、`khop_only`、kmeans/topk matched 和 shuffled matched controls。
- Delta: v1 paired deltas 只有 `±0.000x` 到 `-0.0023` 量级；v2.1 `alpha09` 相对 `alpha03` 赢 `2/4`，`replace_qmix` 赢 `3/4`；两者相对 `gb` 都是 `3/4`，但 `Cora 0.4` 都是负向。
- Std overlap: 大多数正向 setting 仍与 baseline std overlap；当前更像 mechanism-positive / partially method-positive，而不是 strong win。
- Datasets improved: repair 的较强设置主要出现在 `Cora 0.2` 和 `CiteSeer 0.2`；`CiteSeer 0.4` 只有很弱改进。
- Datasets worsened: `Cora 0.4` 是最一致的弱设置。
- Per-seed stability: 5-seed 协议存在，但同一方法在不同 dataset-noise setting 上表现分化明显。
- Runtime/memory cost: 相比重训 GRACE 更低；当前方向主要是 frozen embedding + linear teacher + downstream classifier。
- Raw evidence: 上述 v1/v2/v2.1/v2.2/v2.3 报告和 CSV/JSON 诊断文件。

### 3.6 What Is Directly Supported

- ordinary GB noisy-label smoothing 只有弱信号，不能单独立为主方法。
- GAGB v1 能构造 clean purity 更高、noisy-label entropy 更低的局部区域，但这本身不保证 accuracy 更好。
- v2 gated diagnostic 直接证明：旧 `alpha03` 的 convex update 会把大量 pre-alpha 可纠正节点“冲回去”，所以 `NCR=0` 不等于“没有纠错信号”。
- v2.1 fair repair 直接证明：`alpha09` 和 `replace_qmix` 在 4/4 setting 中都能得到正 NCR，且 CDR 保持在 `0.03` 以下。
- v2.2 直接证明：`Cora 0.4` 的失败主要不是 direct clean damage，而是少量 target edits 导致的间接 decision-boundary movement，尤其影响 class 1 / 6。
- v2.3 直接证明：class-safe / boundary-safe calibration 能缓解 `Cora 0.4` 的 class 1 / 6 下跌，但没有把结果稳定推到 `gb` 之上。

### 3.7 Reasonable Interpretation

解释：当前最可能有价值的不是“graph-aware grouping 一形成就更准”，而是“GAGB 提供了一个可用于保守 repair 的局部坐标系”。它的真实价值更像 local target prior / confidence prior，而不是直接 accuracy booster。另一个合理解释是：当前收益有一部分可能来自 generic target intervention，而不是 GAGB grouping 本身，所以还必须做更强的 matched replay/shuffled control。

### 3.8 What Cannot Be Concluded

- 不能声称 GAGB grouping 已经显著优于 ordinary GB、khop、kmeans、topk 或 shuffled grouping。
- 不能声称 purity/entropy/AUROC 的提升自动转化成下游优势。
- 不能声称 GAGB 已经是 paper-ready main method。
- 不能把 v2.3 class-safe calibration 写成“方法成功”，因为它主要是 failure diagnosis / safer calibration。

### 3.9 Failure Analysis

- Main failure reason: GAGB-specific value 还没有被干净隔离；v1 主要赢 purity，不赢 accuracy；v2.1 主要赢 repair signal，但对 matched controls 的优势不够强。
- Secondary failure reason: target update strength、class concentration 和少量 edited nodes 会引发间接 classifier-boundary movement，`Cora 0.4` 是最明显证据。
- Evidence: `fair_repair_report.md` 显示 `alpha09` 和 `replace` 相对 `gb` 虽然各赢 `3/4`，但 `Cora 0.4` 明显失败；`cora_0p4_failure_report.md` 显示 class 1 / 6 的下降大于 direct intervention footprint；`v2_3_report.md` 显示 class-capped 和 boundary-safe 能恢复弱设置但仍略低于 `gb`。
- Alternative explanation: 现有 replay/shuffled control 还不够 intervention-strength matched，因此一部分收益仍可能来自 generic target perturbation，而不一定来自 GAGB 内容。
- Minimum check needed: 在同一协议下，对 `alpha07`、`class_capped_alpha09`、`boundary_safe_alpha09` 做 full fair-control rerun，并补 target-class-matched replay / prediction-shift diagnostics。

### 3.10 Lessons Learned

- Lesson: 更纯的局部球不等于更好的分类器。Evidence: v1 中 `gagb` 的 purity/entropy 优于 `gb`，但 paired accuracy 只有 `±0.000x` 到 `-0.0023`。Future constraint: 任何 GAGB claim 都必须同时报 accuracy、NCR/CDR 和 matched controls。
- Lesson: `NCR=0` 可能只是 update 太弱，不等于没有纠错信号。Evidence: gated diagnostic 中 `possible_before_alpha` 在四个设置达到 `0.84-0.99`。Future constraint: 先查 pre-alpha signal，再判定方向无效。
- Lesson: 少量 edited nodes 也会带来 class-level boundary shift。Evidence: `Cora 0.4` 的 class 1 / 6 下跌显著大于 gated node 数量。Future constraint: 必须做 class-level diagnostics，而不只看 aggregate CDR。
- Lesson: fair controls 比单纯新方法更重要。Evidence: v2.1 之前的 routing bug 会把 shuffled/khop control 误路由到 GAGB grouping。Future constraint: 任何新变体都要先保证 matched control 真正生效。
- Lesson: clean labels 只能做 diagnostics。Evidence: 当前 teacher、grouping、target construction 都只用 noisy train labels。Future constraint: noisy-label 论文叙事必须明确写出 leakage boundary。

### 3.11 What Is Still Useful

- `graph_aware_ball.py` 提供了一个不改 pretraining 的 graph-aware grouping 实现。
- v1/v2.1/v2.2/v2.3 报告形成了完整的“从 grouping purity 到 repair signal，再到 boundary failure”的证据链。
- fair-control suite（khop、kmeans matched、topk matched、alpha-matched shuffled）是之后任何 noisy-label 粒球方法都应保留的基线框架。
- `possible_before_alpha` / `lost_by_alpha` / class-level NCR-CDR-confusion 这套诊断非常有价值，能避免把“无效 update”误判成“无效 grouping”。

### 3.12 Decision

- Decision: modify
- Reason: 当前方向已经有真实的机制信号和清晰的下一步，但还没有足够证据把收益归因给 GAGB-specific grouping。
- What should not be tried again: 不要再把 v1 grouping-alone 的 purity 改善包装成方法成功；不要再用旧 `alpha03` 的 `NCR=0` 直接宣判 GAGB 无效；不要只靠 Cora 单设置或 replay 不公平对照做结论。
- Variant still worth trying: `alpha07`、`class_capped_alpha09`、`boundary_safe_alpha09` 的 full fair-control rerun，以及 intervention-strength-matched shuffled/replay。
- Evidence required to revive this direction: 至少在 `Cora/CiteSeer × noise 0.2/0.4` 中稳定满足“相对 `alpha03` / `gb` / shuffled 多数 setting 为正，相对 kmeans/topk/khop 至少部分稳定为正，Cora 0.4 不再是明显弱设置”，并证明 matched replay 不能复现同等收益。

### Experimental Setup Summary

- Datasets: Cora, CiteSeer
- Seeds: 0,1,2,3,4
- Encoder: frozen GRACE embeddings
- Teacher: noisy-label-only linear teacher
- Added mechanism: graph-aware grouping + local soft targets + conflict gate + repair variants
- Ball generation method: embedding + k-hop topology greedy grouping
- Evaluation protocol: fixed split noisy-label node classification，best-val test acc
- Train/val/test split: ratio split；ordinary GB validation 包含 train ratio 0.1/0.2，GAGB 主 minival 主要固定 train ratio 0.2、val 0.1

### Baseline Comparability Check

- Same dataset: yes within DIR-07。
- Same split: yes within each report family；主 minival 使用固定 ratio split。
- Same seeds: yes，核心结果均为 5 seeds。
- Same encoder/projector: yes，使用 frozen GRACE embedding，不重训 encoder。
- Same augmentation: yes，DIR-07 不改 GRACE augmentation。
- Same evaluation script: yes，主实验通过同一 noisy-label classifier pipeline 比较。
- Baseline rerun: yes within the fair-control suite。
- Comparable conclusion: yes within DIR-07；但它与 DIR-01~DIR-05 的 unsupervised pretraining 方向不是同一协议，不应直接混成一个主表。

### Label Leakage Check

- Labels used in GB generation: no；构球依赖 frozen embedding 和拓扑。
- Labels used in purity: clean labels 只用于后验 purity / NCR / CDR / confusion diagnostics。
- Labels used in sampling/coarsening: no。
- Validation/test labels used outside evaluation: no。
- Supervised GB mechanism used in pretraining: no；DIR-07 不重训 pretraining encoder。
- Leakage risk: low if paper里明确写清 “teacher / target construction 仅用 noisy train labels，clean labels 只用于 diagnostics”。

### Granular-Ball Diagnostics

- Number of balls / average ball size / coverage / alpha stats / clean purity / noisy-label entropy / noise_detection_AUROC / NCR / CDR / class-level confusion 均已具备。
- Too coarse: ordinary GB、khop、kmeans matched controls 已经说明“球存在”不等于方法成立。
- Too fragmented: 当前不是主问题；更大的问题是 target intervention content 与 strength。
- TODO: intervention-strength-matched replay、prediction-shift / logit-margin diagnostics。

### Next Minimal Experiment If Modified

- New hypothesis: class-safe calibration 之所以有效，是因为它保留了真实的 local correction signal，同时减少了 class concentration 和 indirect boundary movement；如果这一点成立，它应当在更强的 matched replay/shuffled 控制下仍保留优势。
- Change only: 不改 GRACE、不改 GAGB builder、不改 teacher；只补 `alpha07` / `class_capped_alpha09` / `boundary_safe_alpha09` 的 full fair-control rerun，以及 target-class-matched replay / boundary-shift diagnostics。
- Keep fixed: frozen embeddings、split、noise protocol、teacher training、grouping builder、Cora/CiteSeer minival 协议。
- Minimal datasets: Cora, CiteSeer
- Minimal seeds: 5
- Success criterion: 相对 `alpha03` / `gb` / shuffled controls 在至少 3/4 setting 保持正向；相对 kmeans/topk/khop 在至少 2/4 setting 保持正向；`Cora 0.4` 不再明显低于 `gb`。
- Failure criterion: matched replay / khop / kmeans 复现同等收益，或 `Cora 0.4` 继续成为稳定弱设置。
- Why this is the smallest useful test: 它在不改 pretraining 和构球定义的前提下，直接检验“收益是不是来自 GAGB-specific grouping”。

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: GB deep view intervention hurts clean GRACE

### Status
confirmed

### Related Directions
DIR-01, DIR-05

### Observation
GB-GCL view generation and alternative objectives repeatedly underperform healthy GRACE on Cora formal/long protocols.

### Possible Reason
GB-controlled views introduce distribution drift or view gap that disrupts representation learning.

### Evidence
`outputs/final_main_results.csv`, `outputs/cora_long_comparison/summary.csv`, `outputs/cora_gbt_round/conclusion.txt`

### Future Constraint
Do not make GB view generation the paper main method without fresh multi-dataset evidence.

### When to Revisit
Only for robustness/noisy graph settings with diagnostic proof that GB preserves useful edges.

## FP-2: Hard or wide same-ball relation is too coarse

### Status
confirmed

### Related Directions
DIR-02, DIR-03

### Observation
Hard suppression, full-soft, ratio-wide suppression, prototype and ball-positive branches do not outperform conservative topk-soft.

### Possible Reason
Same-ball membership is a structural prior, not a semantic label.

### Evidence
`outputs/cora_fn_grace_round/conclusion.txt`, `outputs/final_ablation_results.csv`, `outputs/ball_positive_round/summary.csv`

### Future Constraint
No hard same-ball positives or broad same-ball suppression without confidence/soft diagnostics.

### When to Revisit
Only if a confidence signal predicts false negatives better than no-ball topk.

## FP-3: Formula complexity does not buy measurable gain

### Status
confirmed

### Related Directions
DIR-02

### Observation
similarity-adaptive, negative mass preservation, cross-view-aware ranking, cross-only, stable_min and fuzzy affinity do not produce stable improvements.

### Possible Reason
The useful candidate set is already tiny; extra weighting mostly changes negligible denominator mass.

### Evidence
`outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt`, `outputs/stable_ranking_round/conclusion.txt`

### Future Constraint
Stop formula shopping; require mechanism diagnostics before any new variant.

### When to Revisit
If future diagnostic shows current selected candidates are consistently wrong in a fixable way.

## FP-4: Accuracy gains are within overlapping std

### Status
confirmed

### Related Directions
DIR-02, DIR-04

### Observation
Cora/CiteSeer/PubMed gains are tiny relative to standard deviations.

### Possible Reason
GB-FN modifies only ~0.1% negatives, so effect size is small.

### Evidence
`outputs/final_main_results.csv`, `outputs/pubmed_formal_round/summary.csv`

### Future Constraint
Do not call a direction successful solely from mean delta smaller than std.

### When to Revisit
With paired seed tests or larger, consistent effect sizes.

## FP-5: Fixed edge similarity thresholds are brittle

### Status
likely

### Related Directions
DIR-04

### Observation
Reliable-ball tau screen kept all edges and produced identical metrics; edge similarity distributions drift high.

### Possible Reason
Absolute cosine thresholds do not adapt to representation scale.

### Evidence
`outputs/noiseaware_ball_round/conclusion.txt`, `outputs/edge_similarity_diag_round/summary.csv`

### Future Constraint
Use percentile/adaptive filters before any further edge-denoise formal runs.

### When to Revisit
After implementing a percentile filter and verifying clean/noisy edge separation.

## FP-6: Label leakage risk is low in training but easy to misstate

### Status
confirmed

### Related Directions
DIR-02, DIR-06

### Observation
Training GB generation uses `x/z/edge_index`; labels appear in evaluation and post-hoc diagnostics.

### Possible Reason
Project separates training mechanism from label diagnostics.

### Evidence
`src/augment/granular_ball.py`, `src/gb_gcl/analysis/fn_candidate_precision.py`

### Future Constraint
Every paper claim using label precision must say labels are post-hoc only.

### When to Revisit
If any future config introduces supervised purity or label-aware coarsening.

## FP-7: Better local purity does not guarantee better downstream accuracy

### Status
confirmed

### Related Directions
DIR-06, DIR-07

### Observation
GAGB v1 consistently improves clean purity and lowers noisy-label entropy relative to ordinary GB, but paired accuracy gains stay tiny or negative; v2.1 repair gets positive NCR with low CDR, yet `Cora 0.4` still loses on class 1 / 6.

### Possible Reason
局部球质量、局部纠错和最终 classifier boundary 不是同一个层面的成功标准。少量 target edits 也可能通过训练动态放大成 class-level boundary shift。

### Evidence
`gb_graph_aware_minival_6_1\reports\2026-06-01_gagb_minival_result.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_2_class_failure_20260602\cora_0p4_failure_report.md`

### Future Constraint
任何 noisy-label GB/GAGB 方法都必须同时报告 purity/entropy/NCR/CDR、class-level accuracy 和 matched controls，不能只看局部诊断。

### When to Revisit
只有在补齐 prediction-shift / matched replay 之后，才能判断 purity 改善是否真的能转化成分类收益。

# 5. Cross-Direction Positive Signal Ledger

## PS-1: Conservative Top-k suppression is safer than broad same-ball suppression

### Status
confirmed

### Related Directions
DIR-02

### Observation
v2 topk beats v1 full-soft and ratio-wide Cora ablations.

### Why It May Matter
It identifies the smallest currently useful GB role: candidate scoping for a tiny number of negatives.

### Required Verification
Paired seed significance and no-ball comparison mechanism diagnostics.

### Possible Paper Use
main method

## PS-2: Soft downweight is safer than hard masking

### Status
confirmed

### Related Directions
DIR-02, DIR-03

### Observation
Hard same-ball suppression trails soft in Cora.

### Why It May Matter
Supports using GB as soft prior, not binary semantic label.

### Required Verification
Already sufficient as ablation; do not overstate.

### Possible Paper Use
ablation

## PS-3: GB-FN does not collapse on citation graphs

### Status
likely

### Related Directions
DIR-02

### Observation
Cora/CiteSeer small positive and PubMed formal 3-seed tied with GRACE.

### Why It May Matter
Shows objective-side GB prior is safer than view-side GB intervention.

### Required Verification
Clarify std overlap and no-ball closeness.

### Possible Paper Use
main method / limitation

## PS-4: Heavy Cora edge noise shows a modest robustness hint

### Status
possible

### Related Directions
DIR-02, DIR-04

### Observation
Cora edge_noise=0.4/0.5 has GB-FN v2 above GRACE/no-ball in mean; CiteSeer is mixed.

### Why It May Matter
Robustness may be a better setting for GB priors than clean graph accuracy.

### Required Verification
Formal paired noisy graph results across Cora and CiteSeer, plus edge/candidate diagnostics.

### Possible Paper Use
appendix / future work

## PS-5: Diagnostics are strong enough to prevent overclaiming

### Status
confirmed

### Related Directions
DIR-06

### Observation
fn candidate precision shows Top-k >> random but GB vs no-ball mixed.

### Why It May Matter
It supports a nuanced paper claim and helps avoid false success.

### Required Verification
Optional per-epoch candidate dynamics.

### Possible Paper Use
diagnostic

## PS-6: Pre-alpha q_mix can contain real correction signal even when final NCR is zero

### Status
confirmed

### Related Directions
DIR-07

### Observation
Gated-node diagnostics show `possible_before_alpha` reaches roughly `0.84-0.99` across the four main Cora/CiteSeer settings, while old `alpha03` leaves final `NCR=0`.

### Why It May Matter
它说明“无效的最终 target update”不能直接等价成“无效的 grouping / teacher signal”。这为 target-only repair 提供了明确入口。

### Required Verification
Keep the same diagnostics in every new repair variant, especially when new gates or replay controls are added.

### Possible Paper Use
diagnostic

## PS-7: Conservative target repair is safer than direct grouping claims

### Status
likely

### Related Directions
DIR-07

### Observation
v2.1 repair turns NCR positive in 4/4 settings with `CDR < 0.03`; v2.3 class-safe calibration mitigates the `Cora 0.4` failure without causing large direct clean damage.

### Why It May Matter
如果 GAGB 还有论文价值，更可能体现在 conservative local repair prior，而不是“graph-aware grouping 一定更优”的强 claim。

### Required Verification
Full fair-control rerun for `alpha07` / `class_capped_alpha09` / `boundary_safe_alpha09`, plus intervention-strength-matched replay.

### Possible Paper Use
motivation

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GB view schedule/freeze/anchor tweaks for clean Cora | Repeatedly below healthy GRACE | `outputs/cora_decay_tuning_round`, `outputs/cora_hierarchy_freeze_round`, `outputs/cora_anchor_explore_v2_round` | Multi-dataset 5-seed positive plus view-distribution diagnostic |
| Hard same-ball negative masking | Lower than soft and risks deleting valid negatives | `outputs/cora_fn_grace_round/conclusion.txt` | Confidence model showing masked candidates are true false negatives |
| Ratio-wide or full-ball suppression | Suppression becomes too broad; Cora ratio drops to 0.7544 | `outputs/final_ablation_results.csv` | Mechanism evidence that broader candidates are cleaner than current Top-k |
| Mass-preserve / adaptive / cross-view formula variants | No stable gain over fixed-alpha v2 | v21/v22/v23 conclusions | Candidate-level diagnostic identifying a specific fixed-alpha failure |
| Prototype / same-ball positive / consistency loss sweeps | Sanity no validation gain or Cora test drop | `outputs/ball_*_round` | Same sanity protocol beats v2 on both Cora and CiteSeer |
| Fixed tau reliable-ball denoise | keep_ratio=1.0, identical metrics | `outputs/noiseaware_ball_round/summary.csv` | Percentile/adaptive separation of clean/noisy edges |
| GB-BGRL / GB-GBT as mainline | Cora 5-seed below GRACE | `outputs/cora_gb_bgrl_round`, `outputs/cora_gbt_round` | First beat GB-GCL stronger and diagnose no collapse |
| 用 purity/entropy/AUROC 直接替代下游验证 | GAGB v1 表明局部球更纯不等于 accuracy 更高 | `gb_graph_aware_minival_6_1\reports\2026-06-01_gagb_minival_result.md` | 需要 matched controls、NCR/CDR 和 class-level accuracy 一起为正 |
| 用旧 alpha03 的 `NCR=0` 直接判定 GAGB 无信号 | gated diagnostic 已证明大量可纠正节点被 weak convex update 冲掉 | `gb_graph_aware_minival_6_1\reports\gagb_v2_gated_diagnostic_20260601\gated_nodes_diagnostic_report.md` | 先检查 pre-alpha q_mix、possible_before_alpha、lost_by_alpha |
| 在没有 intervention-matched replay 的情况下宣称 GAGB-specific value | 当前 v2.1/v2.3 还不能完全排除 generic target perturbation 解释 | `gb_graph_aware_minival_6_1\reports\gagb_v2_1_fair_repair_fixed_20260602\fair_repair_report.md`, `gb_graph_aware_minival_6_1\reports\gagb_v2_3_class_safe_calibration_20260602\v2_3_report.md` | 需要 matched replay/shuffled 和 prediction-shift diagnostics |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB deep view generation | GB can detect non-degenerate local structure | Use GB only as diagnostic or light objective prior |
| Full same-ball suppression | Some same-ball nodes may be false negatives | Top-k soft suppression with no-ball control |
| Direct same-ball positive/prototype | Local relation may be informative | Confidence-weighted, diagnostic-only relation scores |
| Fixed edge threshold denoise | Edge reliability may matter under noise | Percentile/adaptive reliability filter |
| Alternative SSL objective with GB views | Weak/strong view design may matter | First diagnose view gap, then minimal objective test |
| Graph-aware grouping as direct accuracy-improving smoother | graph-aware local region may still be useful as target prior | Conservative repair with class-safe gate and matched controls |
| Strong target replacement (`replace_qmix`) on weak settings | local correction signal may exist but be too aggressive | Alpha-calibrated, class-capped or boundary-safe repair |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: Same dataset, split, seeds, encoder/projector, augmentation unless augmentation is the variable, and same linear evaluation.
- Multi-seed evidence: At least 5 seeds on Cora/CiteSeer; PubMed at least current 3 seeds unless runtime blocks are stated.
- Multi-dataset evidence: At minimum Cora + CiteSeer + PubMed for citation graph claim.
- Ablation removing GB signal: no-ball topk or equivalent must be included.
- Mechanism diagnostic: candidate precision/overlap/loss scale or edge reliability distribution.
- No label leakage: labels only in evaluation or clearly post-hoc analysis.

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation when comparing method variants.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as a reliable semantic positive without confidence or soft weighting.
- Every new direction must define the exact GB role before coding: augmentation, loss, edge, sampling, diagnostic, or evaluation.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing the GB signal, usually no-ball topk.
- If improvement is smaller than overlapping std, mark it `inconclusive` or `modify`, not `continue`.
- If a direction only works for one seed, do not move it into the paper main line.
- If clean graph gain is tiny, test noisy graph or strong augmentation before making robustness claims.
- If complexity increases, require stronger evidence than a simple GRACE/no-ball baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.
- In this project, the reusable signals are conservative Top-k false-negative suppression and conservative target repair diagnostics, not broad GB supervision.
- For noisy-label GB/GAGB directions, do not use clean labels anywhere in grouping, teacher training, target construction, gating, class budgeting or replay sampling.
- For noisy-label GB/GAGB directions, do not claim success without kmeans/topk/khop/shuffled or replay controls.
- For noisy-label GB/GAGB directions, always inspect class-level failure on weak settings before sweeping thresholds.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-01 | GB controls view generation | Negative on clean Cora formal/long | View drift / perturbation mismatch | Motivation and diagnostics | abandon |
| DIR-02 | GB scopes Top-k false negatives | Small positive/tie; statistically weak | Signal too small, no-ball close | Candidate mainline with weaker claim | modify |
| DIR-03 | Direct GB positive/prototype/consistency | Sanity stop; no validation gain | Same-ball too coarse as direct relation | Negative result | abandon |
| DIR-04 | GB soft edge reliability | Clean no gain; noisy diagnostics incomplete | Fixed threshold brittle | Future robustness direction | modify |
| DIR-05 | GB-BGRL/GB-GBT | Negative Cora 5-seed | Objective/view mismatch | Negative result | abandon |
| DIR-06 | Diagnostics | Useful explanation | Not a main method | Strong paper constraint | diagnostic-only |
| DIR-07 | GAGB noisy-label target repair | Mechanism-positive but main claim still weak | Grouping value not isolated; Cora 0.4 boundary movement | Strong diagnostic chain, possible local repair prior | modify |

# 9. Recommended Next Directions

## NEXT-DIR-1: Intervention-matched GAGB target repair completion

### Why This Direction Is Recommended
这是当前唯一已经明确出现“正 NCR + 低 CDR”机制信号、且仍有清晰最小下一步的 GAGB 方向。

### Previous Failure It Addresses
它直接回应了 v1 purity-only claim、v2.1 `Cora 0.4` 失败，以及 v2.3 之后“收益是否只是 generic target intervention”的疑问。

### Core Hypothesis
如果 GAGB 真的有独立价值，那么保守的 class-safe / boundary-safe repair 在 full fair-control 和 intervention-matched replay 下仍应优于 `gb`、`khop`、kmeans/topk matched 和 shuffled/replay controls。

### GB Role
graph-aware local grouping + teacher-aware local repair prior。

### What Must Be Different From Failed Directions
不改 GRACE、不改 GAGB builder、不再堆新的 grouping 公式；只补 full fair controls、matched replay 和 boundary-shift diagnostics。

### Minimal Implementation
固定 frozen embeddings、teacher 和 GAGB builder，重跑 `alpha07`、`class_capped_alpha09`、`boundary_safe_alpha09`，同时包含 `gb`、`gagb`、`khop_only`、`kmeans_gagb_matched`、`topk_gagb_train_matched`、shuffled control 和 target-class-matched replay；输出 class-level confusion、prediction shift、logit margin 诊断。

### Minimal Datasets and Seeds
Cora + CiteSeer；noise 0.2/0.4；5 seeds。

### Success Criteria
相对 `alpha03` / `gb` / shuffled 在至少 3/4 settings 为正，相对 kmeans/topk/khop 在至少 2/4 settings 为正，且 `Cora 0.4` 不再是稳定弱设置。

### Failure Criteria
matched replay / shuffled / khop 能复现同等收益，或 class-safe 变体仍普遍低于 `gb`。

### Paper Claim If Successful
GAGB provides a conservative local repair prior for noisy-label node classification under frozen GCL embeddings.

### Reason to Stop Early
如果 `Cora 0.4` 继续失败，且 matched replay 能逼近主方法，则应把 GAGB 降级为 diagnostic-only，不再推进为主方法。

## NEXT-DIR-2: Mechanism-first GB-FN consolidation

### Why This Direction Is Recommended
当前最接近主线，但缺少足够机制证据解释 GB vs no-ball 的小差异。

### Previous Failure It Addresses
避免继续 formula shopping。

### Core Hypothesis
GB 的收益来自候选范围/训练时机的温和正则，而不是候选 label precision 大幅提升。

### GB Role
same-ball candidate scope only。

### What Must Be Different From Failed Directions
不改公式；只补 paired diagnostics。

### Minimal Implementation
复用现有 checkpoints，计算 per-seed candidate overlap、GB vs no-ball selected set、loss contribution、per-epoch candidate dynamics。

### Minimal Datasets and Seeds
Cora/CiteSeer 5 seeds，PubMed 3 seeds。

### Success Criteria
能解释 GB 相对 no-ball 的差异，并支持温和 claim。

### Failure Criteria
GB 与 no-ball 在候选和 loss 上不可区分。

### Paper Claim If Successful
GB is a lightweight structural prior that regularizes false-negative suppression.

### Reason to Stop Early
若 Cora/CiteSeer candidate overlap >90% 且 loss contribution 差异极小，则停止。

## NEXT-DIR-3: Robustness-focused GB-FN under heavy edge noise

### Why This Direction Is Recommended
Cora heavy edge noise 有小幅正信号，但 CiteSeer mixed。

### Previous Failure It Addresses
clean graph gain太小，可能不是 GB 最适合的场景。

### Core Hypothesis
GB-FN 在强边噪声下比 GRACE/no-ball 更稳定。

### GB Role
same-ball candidate prior，不改 view generation。

### What Must Be Different From Failed Directions
必须固定 v2 topk-soft，不加入 edge denoise 或新公式。

### Minimal Implementation
重用现有 noise configs，补齐 Cora/CiteSeer edge_noise=0.4/0.5 paired 5-seed。

### Minimal Datasets and Seeds
Cora + CiteSeer, 5 seeds。

### Success Criteria
GB-FN 在两个数据集重噪声下均高于 GRACE/no-ball，且 delta 不只是单 seed。

### Failure Criteria
CiteSeer 继续 mixed 或 no-ball 更强。

### Paper Claim If Successful
GB prior improves robustness under severe edge noise.

### Reason to Stop Early
3-seed quick check 没有正向均值或机制激活。

## NEXT-DIR-4: Percentile-based edge reliability

### Why This Direction Is Recommended
edge similarity diagnosis 明确指出 fixed tau brittle。

### Previous Failure It Addresses
DIR-04 fixed tau no-op。

### Core Hypothesis
per-epoch percentile filter can separate noisy edges better than absolute cosine tau.

### GB Role
same-ball + feature similarity edge reliability。

### What Must Be Different From Failed Directions
只在 noisy graph 测；clean graph 不作为成功标准。

### Minimal Implementation
Add percentile threshold to edge reliability, log clean/noisy keep rates。

### Minimal Datasets and Seeds
Cora edge_noise=0.4, 3 seeds first。

### Success Criteria
Noisy edge keep lower than clean edge keep and accuracy > v2/no-ball。

### Failure Criteria
Edge keep 不分离或 accuracy 不提升。

### Paper Claim If Successful
GB denoising can support robustness when thresholds adapt to representation scale.

### Reason to Stop Early
If quick screen repeats fixed-tau no-op behavior。

# 10. Paper Claim Implications

## Claim-1

### Claim
GB provides robust local structure prior for node-level GCL.

### Supported By Which Directions
DIR-02 partly; DIR-06 diagnostics partly。

### Weakened By Which Directions
DIR-01, DIR-03, DIR-04, DIR-05。

### Current Evidence
Cora/CiteSeer small positive, PubMed tie；no-ball topk very close。

### Missing Evidence
Stronger mechanism diagnostic and non-overlapping effect。

### Required Experiments
NEXT-DIR-2 and possibly NEXT-DIR-3。

### Risk
Overclaiming tiny deltas。

### Current Readiness
weak

## Claim-2

### Claim
GB confidence reduces noisy relation damage.

### Supported By Which Directions
DIR-02 topk vs full/ratio suggests narrower candidates are safer。

### Weakened By Which Directions
DIR-03 direct relation branches, DIR-06 candidate precision mixed。

### Current Evidence
Top-k conservative suppression works better than broad suppression。

### Missing Evidence
Explicit confidence calibration and ball quality distribution。

### Required Experiments
Candidate confidence diagnostics; no new formula first。

### Risk
Current `ball_quality_threshold=0.0` means confidence claim is weak。

### Current Readiness
weak

## Claim-3

### Claim
GB-guided augmentation preserves local semantics.

### Supported By Which Directions
DIR-01 structural diagnostics show active balls。

### Weakened By Which Directions
DIR-01 formal accuracy below GRACE。

### Current Evidence
Active structure but no accuracy benefit。

### Missing Evidence
Multi-dataset positive results。

### Required Experiments
Not recommended now。

### Risk
Contradicted by current main evidence。

### Current Readiness
not ready

## Claim-4

### Claim
GB denoising improves robustness under graph noise.

### Supported By Which Directions
DIR-02 Cora heavy edge noise possible signal; DIR-04 edge diagnostics。

### Weakened By Which Directions
CiteSeer robustness mixed; fixed tau denoise no-op。

### Current Evidence
Possible but inconsistent。

### Missing Evidence
Percentile/adaptive denoise and multi-dataset noisy formal。

### Required Experiments
NEXT-DIR-3 or NEXT-DIR-4。

### Risk
High。

### Current Readiness
not ready

## Claim-5

### Claim
Hard same-ball positive is not reliable for node-level GCL.

### Supported By Which Directions
DIR-02 hard/full/ratio ablations；DIR-03 positive/prototype negative sanity。

### Weakened By Which Directions
None in current project。

### Current Evidence
Hard/broad/direct relation variants fail or do not improve。

### Missing Evidence
Formal direct-positive 5-seed not run because sanity failed。

### Required Experiments
None unless challenged; current evidence enough for negative-result discussion with caveat。

### Risk
Medium due sanity-only for DIR-03。

### Current Readiness
moderate

## Claim-6

### Claim
GB is more useful as an auxiliary prior than as a direct replacement for contrastive positives or view generation.

### Supported By Which Directions
DIR-01 negative, DIR-02 small positive/tie, DIR-03 negative, DIR-05 negative。

### Weakened By Which Directions
DIR-02 effect size small。

### Current Evidence
Best current direction is auxiliary negative suppression; view/direct/alternative directions fail。

### Missing Evidence
Stronger paired diagnostics and careful no-ball comparison。

### Required Experiments
NEXT-DIR-2。

### Risk
Moderate；claim should be phrased as project evidence, not universal theorem。

### Current Readiness
moderate

## Claim-7

### Claim
Graph-aware local grouping plus conservative target repair can provide mechanism-positive noisy-label correction under frozen GCL embeddings.

### Supported By Which Directions
DIR-07。

### Weakened By Which Directions
DIR-07 v1 fair controls、DIR-07 `Cora 0.4` class failure、当前 replay control 仍不够强。

### Current Evidence
v2.1 在 4/4 settings 得到正 NCR 且 `CDR < 0.03`；v2.2 说明弱设置主要是间接 boundary movement；v2.3 说明 class-safe / boundary-safe calibration 能缓解该弱设置。

### Missing Evidence
Full fair-control win over `gb` / `khop` / kmeans / topk / shuffled / matched replay；更强的 prediction-shift diagnostics；Cora/CiteSeer 之外的数据集。

### Required Experiments
NEXT-DIR-1。

### Risk
中高。当前收益仍可能部分来自 generic target intervention，而不是 GAGB-specific grouping。

### Current Readiness
weak

# 11. TODO List

## Missing raw logs

- Round-level aggregate docs exist, but some direction decisions reference `summary.csv/conclusion.txt` more than individual seed logs.
- If preparing appendix, map every table row to exact `seed_*/train.log` and `metrics.json` paths.

## Missing baseline checks

- Amazon-Photo/Computers and Coauthor-CS are not formal multi-seed comparisons.
- PubMed DGI is not run.
- DIR-07 还缺 intervention-strength-matched replay / shuffled control 的最终定版比较。

## Missing seeds/std

- DIR-03 prototype/positive/consistency only sanity 1 seed。
- DIR-04 reliable-ball tau only quick screen 1 seed。
- PubMed fn_candidate_precision diagnostic currently reports 1 seed。
- DIR-07 未来若转 NoisyGL / RTGNN benchmark，需要重新建立与当前 frozen-GRACE minival 分开的 seeds/std 体系。

## Missing datasets

- GB-GCL view generation formal evidence is mostly Cora。
- Edge denoise lacks CiteSeer/PubMed formal noisy results。
- Coauthor-CS only sanity。
- DIR-07 目前只有 Cora/CiteSeer；缺 PubMed 或更真实 noisy-label benchmark。

## Missing ablations

- For DIR-02, need paired GB-vs-no-ball candidate overlap and per-seed delta tests。
- For DIR-04, need percentile filter ablation if direction continues。
- For DIR-07, need full fair-control rerun for `alpha07` / `class_capped_alpha09` / `boundary_safe_alpha09`。
- For DIR-07, need target-class-matched replay / shuffled control to isolate GAGB-specific value。

## Missing GB diagnostics

- Radius mean/std。
- Singleton ball ratio。
- Cross-seed assignment stability。
- Cross-epoch assignment stability beyond refresh counts。
- Ball quality distribution and confidence calibration。
- GB loss scale vs InfoNCE for additive branches。
- For DIR-07, need prediction-shift / class-boundary / logit-margin diagnostics on weak settings such as `Cora 0.4`。

## Missing visualizations

- Candidate overlap GB vs no-ball。
- Per-epoch masked_negative_ratio and selected similarity。
- Edge clean/noisy percentile separation。
- Ball size distribution over epochs。
- For DIR-07, need per-class confusion deltas and changed-node class distribution plots。

## Missing explanation for failed directions

- DIR-05 needs embedding collapse / variance diagnostic if ever discussed beyond a short negative result。
- DIR-01 needs view distribution distance if someone revisits augmentation。
- DIR-07 still lacks a final matched-replay explanation for how much gain comes from grouping content versus intervention strength。

## Missing paper-claim evidence

- Strong robustness claim not ready。
- Strong superiority over no-ball topk not ready。
- Strong “GB improves candidate purity” not ready。
- Strong “GAGB-specific grouping is necessary for noisy-label repair” not ready。

# 12. Turning Point: Drop Graph-Aware Ball, Adopt Reliable Mutual-kNN Ball (2026-06-05 to 2026-06-07)

本节记录一次明确的方向转折：从 DIR-07 的 GAGB / 图感知粒球，转向「**对比表示上的 reliable mutual-kNN 粒球去噪**」。证据来自 `gb_graph_aware_minival_6_1` 的 DCGB v2.8 诊断、raw-vs-GRACE 诊断，以及 `external/NoisyGL` 的 ReliableGB-GCN S1 smoke。

## 12.1 Core Conclusions

1. **图感知粒球在 Cora/CiteSeer 当前设置下不应作为主创新点。** DCGB v2.8 加 raw-vs-GRACE 诊断（Cora/CiteSeer × noise 0.2/0.4 × 5 seed）显示：拓扑相对特征互惠 kNN 的增量 `dcgb - dcgb_emb_only` 在全部 8 个条件中 purity ±0.002、AUROC ±0.001，近似为 0。原因：同配引文图上邻接信息已被词袋特征与 GRACE 嵌入吸收，拓扑冗余。证据 `gb_graph_aware_minival_6_1/reports/dcgb_v2_8_raw_vs_grace_20260605/`、`reports/dcgb_v2_8_grouping_diag_20260605/`。

2. **reliable mutual-kNN 构球是真正稳定的正信号。** 在 raw 特征上 ordinary GB purity 崩塌（Cora 0.303、CiteSeer 0.210），而 reciprocal mutual-kNN balls 显著恢复 purity（0.78–0.84）；在 GRACE 空间上 dcgb purity 0.89、noise-AUROC 0.79 且相对 `gb` 的 AUROC 提升在 4/4 设置 std 不重叠。可靠性来自**构球方式（互惠最近邻），与表示空间无关、与图拓扑无关**。

3. **监督表示在标签噪声下不可靠（single-seed smoke 下的工作假设，非普适结论）。** NoisyGL ReliableGB-GCN S1（Cora uniform 0.4 **seed3000 单 seed smoke**）：reliable-GB reweight 在 GCN 宿主上 test 持平 GCN（0.705），`best_phase=warmup`；修正 no-ball 对照 bug 后，real-GB gated precision 0.50 与有效 no-ball 0.50 持平。**观察到的风险（非普适定理）**：GCN val-acc 约 epoch 6 见顶后记忆 40% 噪声标签，构球建在已污染表示上。**工作假设（仅由该 single-seed smoke 支持，未经多 seed/多设置确认）**：在此 smoke 中瓶颈看起来不在球质量/去噪公式/时机，而更可能在监督表示本身；这是一个待验证假设，**不是**已确证的普适规律。基于该假设转向 S2a——在 label-free 对比表示（GRACE/GCA，不看标签故不记忆噪声）上构同一套 reliable 球，正是为了检验该假设。证据 `external/NoisyGL/runs/reliablegb_s1_20260607/s1_report.md`、`external/NoisyGL/runs/orgingb_noisygl_experiment_history_20260604.md` 第 9 节。

## 12.2 New Anti-Repetition Entries

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| 把图感知（特征+拓扑）粒球当 Cora/CiteSeer 主创新点 | dcgb ≈ dcgb_emb_only，拓扑增量 ±0.001 | `reports/dcgb_v2_8_raw_vs_grace_20260605/raw_vs_grace_report.md` | 异配/特征稀疏图上拓扑与特征真正分歧（用户当前不做异配图） |
| 在监督 GCN 表示上构球去噪当主线 | 表示记忆噪声，球建在污染表示上，ties GCN/no-ball | `external/NoisyGL/runs/reliablegb_s1_20260607/s1_report.md` | 在 label-free 对比表示上构同一套 reliable 球并超过对照 |
| no-ball/random 对照仅靠类属性、被 config `gb:` 键覆盖 | `_conf_value` 覆盖类属性，对照静默失效 | buggy vs corrected CSV | 诊断 CSV 中确认控制字段（suspicion_mode/random_control）真正不同 |

## 12.3 Decided Next Direction (S2a, not yet run)

- 复用 `gb_graph_aware_minival_6_1` 的 frozen-GRACE classifier-gate 管线，比较「GCN hidden 构球 vs frozen GRACE 构球」的球质量与下游去噪；`build_dcgb_balls(mode="emb_only")` 即 reliable mutual-kNN 构造，可直接复用。
- 成功标准：对比表示上 reliable 球的球质量更高且下游超过 GCN/对照；失败则把 reliable-GB 去噪降为 diagnostic，论文主线转向「表示空间 × 球质量」系统研究。
- 论文主线候选：**label-free contrastive representation + mutual-kNN reliable granular ball + conservative noise-aware training**（非图感知）。
- **详细执行方案（防跑偏）见** `gb_graph_aware_minival_6_1/reports/s2a_plan_20260607/s2a_execution_plan.md`。要点：minival lane 只支持 symmetric 噪声 / seeds 0-4（first signal）；uniform+pair / seeds 3000-3004 属于 NoisyGL main-table lane，两条 protocol 不可混报。

## 12.4 S2a Phase 1 first-signal + headroom 判生死（2026-06-07）

- **S2a Phase 1 ratio-split smoke 未通过**（Cora / symmetric 0.4 / seed 0 / frozen GRACE / `ratio_train0.2_val0.1` / pretrain0；**不是 planetoid，也不是 NoisyGL protocol**）。`reliable mutual-kNN 球（dcgb_emb_only）`在 GRACE 上 test 0.7941，**不及 ordinary gb 0.7962、no-ball topk 0.7951，且与破坏球结构的 shuffled 0.7941 精确打平**——尽管其 purity 0.889 / AUROC 0.739 远高于 shuffled 的 0.495 / 0.489。
- **机制**：primary α=0.3 全部方法 NCR=0（去噪退化为 label-smoothing）；唯 `qsafe_α09`（NCR=0.0323、CDR=0、仅纠 14 个标签）有少量干净纠错，但 test 反而更低——**不是"把 gate 放强"就能解决，下游分类器没从这批正确纠错里受益**。
- **Headroom 判生死（clean-oracle 诊断）**：同 frozen GRACE / 同 split，clean-oracle CE = 0.8221，noisy CE = 0.7878 → **headroom = 3.43pp（>1pp，非 no-headroom 分支）**。最好的 `gb` 只吃到 0.84pp（≈25%），`dcgb_emb_only` 0.63pp 且与 shuffled 打平 → **球的对应内容吃到 ~0 可恢复 headroom**。clean-oracle 0.8221 ≈ 该嵌入 clean 线性上界，说明**冻结嵌入上唯一杠杆是标签处理**。证据 `gb_graph_aware_minival_6_1/reports/s2a_plan_20260607/s2a_phase1_smoke_result.md`、`.../s2a_headroom_oracle.json`、`scripts/diagnose_clean_oracle_headroom.py`（clean 仅作 oracle 诊断，绝不进方法）。
- **结论与方向**：否定的是 **"frozen-head + soft-target/reweight"这一条转化链路**，**不是整个 S2**（headroom 还在）。瓶颈精确定位到**转化**。下一步在两者间二选一（不调 alpha）：**S2-redesign** = 把球信号换成能提交决策的形态（ball-guided sample selection / co-teaching 小损失 / loss correction），或 **S3** = 把"高质量球为何不转化"做成系统研究（表示空间 × 球质量 × 转化失败机制）。
- **反重复**：(1) 不要再调 `alpha/gate/top_ratio/软目标公式`当方法（formula shopping，已多次失败，qsafe 证明 NCR>0 也不转化）；(2) 不要再用 frozen-head soft-target 形态期待涨点；(3) pair/high-noise headroom 是单独 NoisyGL lane，不与 ratio-split symmetric 混报。

## 12.5 S2-redesign-v1：commit-style 选择，gate 未过 → 关闭 reliable-ball 方法线，转 S3（2026-06-07）

把"软目标/重加权"换成**提交式 sample selection**（每 epoch 按 `z(−loss)+z(rel)` 选 top-`k`，固定 schedule W20/ramp20/ρ=1−noise=0.6，method 与所有对照**同 schedule，仅 rel 来源不同**）。同一格单 seed smoke。证据 `gb_graph_aware_minival_6_1/reports/s2a_plan_20260607/s2a_redesign_v1_selection_report.md`、`.../s2a_redesign_v1_selection.{csv,json}`、`scripts/run_selection_experiment.py`、`src/noisy_label_gb_grace/evaluation/selection.py`。

- **预注册 gate 未过 → 关闭 reliable-mutual-kNN 方法线。** `sel_dcgb_emb` test=0.7788 **不及 gb_soft 0.7962、亦不及 ce 0.7878**（criterion 1 FAIL）。按预注册规则：不扩 seed、不调 ρ、不进 NoisyGL pair lane，转 S3。
- **重要次级发现（单格，生成假设、未升级为结论）**：
  1. **提交机制让球信号真正转化——soft-target 才是瓶颈（已确认）**。与 Phase 1 不同，`sel_dcgb_emb` 在 test 与 selection precision 上**大幅碾压** no-ball/shuffled/random（0.779 vs 0.66–0.68；0.874 vs 0.65–0.78），criteria 2–4 全过。
  2. **互惠-kNN「reliable」球并非最好的选择信号**：同机制下 **`sel_gb` 0.8094 > `sel_topk` 0.7930 > `sel_dcgb_emb` 0.7788**。普通 GB / kNN 共识选得更好——尽管 dcgb 的 ball purity 0.889 远高于 gb。**球 purity 不预测 selection/下游质量。**
  3. `sel_gb` 吃到 ~63% 的 3.43pp headroom 且超过 gb_soft/ce——但**单格、非预注册方法**，现在推它就是 cherry-picking，故仅作 S3 假设。
  4. loss-only 选择有害（no-ball 0.6642 < ce），靠球共识救回——reliability 信号确实在做功。
- **S3 重定向（决定方向）**：主线问题锐化为 **"为什么 ball purity/AUROC（dcgb≫gb）不预测 selection/下游质量（sel_gb>sel_dcgb）"**——表示 × 构球 × "哪种 reliability 信号真正转化"。S3 用多 seed + 对照系统研究 ordinary-GB / kNN-consensus / reliable-ball 选择信号 vs 其球质量；`sel_gb` 只作候选调查对象，**不**从单格宣称新方法。
- **反重复**：(4) 不要为了让 `sel_dcgb_emb` 过线去调 schedule/ρ（已禁止）；(5) 不要从这单格把 `sel_gb` 当成"找到的新方法"宣称，必须多 seed + 对照后再说。

## 12.6 S3 Phase 1：seed0 倒挂被多 seed 证伪，reliable 球稳健（2026-06-07）

Cora / symmetric 0.4 / `ratio_train0.2` / **seeds 0–4**（每 seed s 用 pretrain{s}/noise{s}/clf{s}）/ frozen GRACE，commit-style selection，固定 schedule。预注册 A/B/C 机械判定。证据 `gb_graph_aware_minival_6_1/reports/s3_phase1_20260607/s3_phase1_report.md`、`.../s3_phase1_rows.csv`、`scripts/run_s3_phase1.py`。

- **A 不成立（1/5，mean −3.00pp）→ seed0 的 `sel_gb > sel_dcgb_emb` 倒挂是 seed 噪声。** 跨 seed 均值 `sel_dcgb_emb 0.8002 > sel_topk 0.7969 > sel_gb 0.7702`；`sel_gb` 在 seed3/4 崩到 0.72。**§12.5 那条"互惠-kNN 球不是最好选择信号"被证伪——它只是单格方差。reliable 球反而是稳的那个。**
- **B 不成立（2/5，mean −2.71pp）→ `sel_gb` 不稳超 gb_soft**。§12.5 中"sel_gb 吃到 63% headroom"是单格，作废。
- **C 成立（5/5/5 满分，mean +8.4~9.2pp）→ commit-style selection + reliable 球是稳健正信号。** 破坏球结构（shuffled/random）purity 掉到 ~0.51、test 掉到 ~0.71；loss-only 0.708。球结构对 selection 有真实因果贡献。
- **球质量 vs 转化（修正解读）**：dcgb 与 gb 的 group-purity 几乎相同（0.892 vs 0.893），但 dcgb 的 **node-level selection precision 0.904 ≫ gb 0.800**、noise-excl 0.856 ≫ 0.589 → **group-purity 不区分，node-level 选择质量才区分**。`sel_topk`（kNN 共识，不显式建球）rel-AUROC/sel-precision 最高、test≈dcgb。
- **方向**：A/B 证伪 → 不围绕 `sel_gb` 倒挂写论文，`sel_gb` 不作方法。C 稳健 → S2/S3 不收缩。Phase 2 机制问题**重锚**为"为什么 group-purity 相同而 node-level selection 质量不同"+"显式建球 vs kNN 共识差异来源"，**不是**已被证伪的 seed0 倒挂。
- **教训（写给未来）**：**任何单格倒挂/涨点先多 seed 证伪再投入机制解释。** §12.5 据 seed0 写下的两条次级发现（sel_gb 更优、reliable 球非最佳）被 §12.6 的 5 seed 直接推翻——这正是"不从单 seed 判定"红线的又一次实证。

## 12.7 S3 Phase 2-Q2：显式建球 vs kNN 共识不可区分 → 论文主线降级（2026-06-07）

复用 §12.6 的 5-seed 数据做 `sel_dcgb_emb`（显式互惠-kNN 球）vs `sel_topk_matched`（纯 kNN 共识，k 逐 seed 匹配球大小、coverage 同为 0.600）的 paired 必要性验证。证据 `gb_graph_aware_minival_6_1/reports/s3_phase1_20260607/s3_phase2_q2_dcgb_vs_topk_report.md`。

- **DCGB − TopK paired**：test mean **+0.327pp**、std 1.03pp、**t≈0.71（不显著）**，4/5 wins 但 mean <0.5pp；**selection precision 0/5 输（−1.85pp）、noise-excl recall 0/5 输（−2.76pp）**。同覆盖同 schedule，已核验非假象。
- **预注册命中第三档（降级）**：显式建球相对 kNN 共识**无可证独立增量**——test 优势不显著不稳，selection 维度反而系统性更差。
- **主线判定**：稳健的是 **commit-style selection + 局部一致性可靠性信号**（Phase 1 的 C 满分），但该信号用**互惠-kNN 球**还是**纯 kNN 共识**实现下游不可区分。**论文 claim 从"粒球（granular ball）方法"降级为"局部一致性可靠性信号 + commit-style 选择"，粒球仅作实现细节，不作独立创新点/显著增益来源。** 与转折点一脉相承：从图感知球→reliable 球→Phase 1 现象→Q2，每一步"球的特异结构"都未转化为可证下游增量。
- **反重复**：(6) 不要再把"粒球"当独立卖点宣称，除非先在某设置（更高/pair 噪声、更弱表示、多尺度/非对称邻域）证明球相对 kNN 共识有独立增量；(7) Q1（gb vs dcgb group-purity 相同但 node-level 选择质量不同）是解释性机制章节，不是主线生死，可后做。

## 12.8 S3 Phase 1c：frozen 表示上球结构无救，粒球须进 encoder coupling（2026-06-07）

读 CGBC（合约漏洞，`2603.27734v1.pdf`）与 GBmooer（文本噪声，`ssrn-6022343.pdf`）原文后，定位它们"为什么有效"的根因：**表示是为球而生的（CGBC 用语义增强对比 + intra/inter-GB 损失反馈进 encoder；GBmooer 用超球角距离对比专门让球可分），且球结构反向塑造表示**——二者都不是把球套在冻结的通用表示上。据此做最后一道 frozen-space 检查：在 frozen GRACE 上引入 GBmooer 式**球级密度结构**信号 `rel=density(B)×compactness(i,B)×noisy_consensus(B)`，看能否拉开 kNN 共识。证据 `gb_graph_aware_minival_6_1/reports/s3_phase1c_density_20260607/s3_phase1c_report.md`。

- **先验证避免无效实验**：L2 归一化表示上 cosine top-k 与角距离 top-k 完全一致（300/300），故仅换角距离是恒等变换；改测真正的"球级密度结构"。
- **预注册否定（机械）**：`sel_density_gb` test 均值 0.7714 **< topk 0.7969 < dcgb 0.8002**，density>topk 仅 1/5、mean −2.56pp；sel-precision 0.807<0.923、noise-recall 0.665<0.884，全面更差。density 赢自己的 shuffled/random（+5.97/+3.02pp）说明它捕捉到结构，**但该结构被 kNN 共识覆盖且覆盖得更好**。
- **结论**：**frozen GRACE 上球级结构（图感知/reliable/density 全部）无独立增量，且不如 kNN 共识。** frozen-space"球有没有救"穷尽 → **没有**。与 §12.1/12.5/12.7 一脉相承：表示冻结且非为球而生时，球必退化成局部一致性。
- **方向（按预注册）**：保留粒球主线**只剩**让表示为球而生 / 球反馈进 encoder 一条路——CGBC（intra/inter-GB 损失 + 联合训练）或 GBmooer（超球对比专训球可分 + 多尺度平滑）。这是 encoder coupling（工程更大、项目历史联合训练不稳、须避开 hard same-ball positive）。
- **反重复**：(8) 不要再在 frozen 表示上找"更聪明的球信号"——图感知/reliable/density 已三次证明无独立增量；下一步要么 encoder coupling，要么接受粒球降为实现细节。

## 12.9 S3 Phase 2：最小 encoder-coupling 球质量未改善 → 早停（2026-06-07）

frozen-space 穷尽后，按 CGBC/GBmooer 的根因（表示为球而生 + 球反馈进 encoder）做最小弱耦合冒烟：GRACE init 全 encoder 小 lr（1e-4）微调 `L=L_sup+λcl·L_GRACE+λgb·L_ball`（λ 全固定不调），L_ball=周期构 dcgb 球 + 仅高质量球（label-free gate）+ confidence-weighted intra（可微 H 现算中心）+ 不同-noisy-majority 球中心 margin inter，无 hard positive/无 view generation/clean 仅诊断。证据 `gb_graph_aware_minival_6_1/reports/s3_phase2_coupling_20260607/s3_phase2_coupling_report.md`、`src/noisy_label_gb_grace/training/ball_coupling.py`。

- **预注册早停（球质量中间诊断）**：coupled-dcgb clean_purity 0.8856 / noise-AUROC 0.8885 **vs frozen 0.8893/0.8920**（purity −0.37pp、AUROC −0.35pp，略降非升）；**coupled vs baseline（同 init、λgb=0）purity ±0.00pp** → 球损失对球质量零作用。按规则不跑 selection、不扩 seed、不调 λ。
- **归因**：不是"球质量变好但不转化"，而是"最小耦合根本没让球质量变好"。根因 = **球损失被对比损失淹没**（log：L_GRACE≈6.84 ≫ intra≈0.02/inter≈0.2），150 epoch 未把表示推向球更纯；护栏禁止调 λ 刷结果故止此。
- **边界**：这是"最小、不破坏 GRACE 的弱耦合"的负结果，**不**否定强耦合——CGBC/GBmooer 有效来自**球损失主导训练**（CGBC `L=L_clustering+L_SCE` 球结构一等损失；GBmooer 超球对比专为球可分），代价是放弃 frozen GRACE 稳定性、接近从头训。
- **方向收敛**：frozen 改球（§12.1/12.5/12.7/12.8）+ 最小弱耦合（本节）全无首信号 → **在"保持 GRACE 稳定"前提下粒球结构无独立增量**。保留粒球主线只剩**强耦合/球损失主导训练**（更大工程、与项目"联合训练历史不稳"相悖）。诚实替代主线：**"局部一致性可靠性信号 + commit-style 选择"为主体，粒球为一种实现 + "为何 frozen 表示下球不转化、弱耦合不足"的负结果机制分析。**
- **反重复**：(9) 不要再用"最小/弱"耦合期待球质量提升——球损失被 L_GRACE 量级淹没已证；要么球损失主导（接受不稳与大工程），要么粒球降为实现细节。

## 12.10 两格生死验证：local consistency + commit selection 通过方法论文门槛，但主力不是显式粒球（2026-06-07）

按 `gb_graph_aware_minival_6_1/reports/执行计划_两格生死验证_交Codex_20260607.md` 执行两格低成本生死验证。新实现保持 minival 协议内比较：新增 `src/noisy_label_gb_grace/evaluation/coteaching.py`（双线性头 co-teaching，同一 keep-ratio schedule）、`pair` 噪声生成（train-only, `c -> (c+1)%C`），以及 `scripts/run_deathmatch_citeseer.py` / `scripts/run_deathmatch_pair.py`。所有数字来自同一 frozen-GRACE minival 协议，**未混入 NoisyGL co-teaching 数字**。证据：

- `gb_graph_aware_minival_6_1/reports/deathmatch_citeseer_20260607/report.md`
- `gb_graph_aware_minival_6_1/reports/deathmatch_citeseer_20260607/deathmatch_rows.csv`
- `gb_graph_aware_minival_6_1/reports/deathmatch_citeseer_20260607/deathmatch_gates.json`
- `gb_graph_aware_minival_6_1/reports/deathmatch_pair_20260607/report.md`
- `gb_graph_aware_minival_6_1/reports/deathmatch_pair_20260607/deathmatch_rows.csv`
- `gb_graph_aware_minival_6_1/reports/deathmatch_pair_20260607/deathmatch_gates.json`

### 生死格 1：CiteSeer symmetric 0.4，5 seeds

均值：

| method | mean test |
|---|---:|
| `sel_topk` | 0.7082 |
| `sel_dcgb` | 0.7000 |
| `ce` | 0.6979 |
| `sel_noball` | 0.6977 |
| `coteaching` | 0.6958 |
| `sel_random` | 0.6944 |
| `sel_shuffled` | 0.7040 |
| `sel_clean_oracle` | 0.7104 |

机械 gate：

- **Gate A PASS**：`sel_topk` 5/5 seeds 超过 `coteaching`，mean delta +1.24pp；`sel_dcgb` 4/5 但 mean delta +0.42pp，未达到 +0.5pp 阈值。
- **Gate B PASS**：由 `sel_topk` 通过，分别 5/5 超过 `sel_noball`、4/5 超过 `sel_shuffled`、5/5 超过 `sel_random`。`sel_dcgb` 在 CiteSeer 上没有通过结构对照 gate。
- **Gate C PASS**：best structured (`max(sel_topk, sel_dcgb)`) 相对 `ce` mean delta +1.03pp，超过 symmetric 阈值 +1.0pp。

### 生死格 2：Cora pair 0.4，5 seeds

均值：

| method | mean test |
|---|---:|
| `sel_topk` | 0.6082 |
| `ce` | 0.5920 |
| `sel_dcgb` | 0.5834 |
| `sel_random` | 0.5646 |
| `sel_shuffled` | 0.5430 |
| `coteaching` | 0.5225 |
| `sel_noball` | 0.5218 |
| `sel_clean_oracle` | 0.8143 |

机械 gate：

- **Gate A PASS**：`sel_topk` 5/5 超过 `coteaching`，mean delta +8.57pp；`sel_dcgb` 4/5 超过 `coteaching`，mean delta +6.09pp。
- **Gate B PASS**：由 `sel_topk` 通过，5/5 超过 `sel_noball`、`sel_shuffled`、`sel_random`。`sel_dcgb` 对 `sel_random` 只有 3/5，未通过该项。
- **Gate C PASS**：best structured 相对 `ce` mean delta +2.11pp，超过 pair/high-noise 阈值 +1.5pp。

### 结论

- **方法论文路线获得继续铺主表的资格。** 两格均通过预注册 Gate A/B/C，说明 `local consistency reliability + commit-style selection` 不只是击败自造 no-ball/shuffled/random 控制，也能在 minival 协议内打过一个真实近邻机制基线 `co-teaching`，并在 pair 噪声主场拉开更大差距。
- **主力信号是 `sel_topk`，不是显式粒球。** 两格通过主要由 kNN consensus (`sel_topk`) 支撑：CiteSeer 上 `sel_dcgb` 未通过 Gate A 的 margin，也未通过 Gate B；Cora pair 上 `sel_dcgb` 虽超过 co-teaching，但仍弱于 `sel_topk` 且对 random control 不稳。该结果进一步支持 §12.7/§12.8 的降级判断：论文主线应写成**局部一致性可靠性信号 + commit-style 选择**，显式粒球/DCGB 是一个结构化实现与 ablation，不应作为独立增益来源。
- **pair/结构化噪声是更有价值的主场。** Cora pair 0.4 下 `sel_topk` 相对 `ce` +1.62pp、相对 `coteaching` +8.57pp，明显强于 Cora/CiteSeer symmetric 中的小幅增益，后续主表应优先包含 pair/asymmetric 噪声。
- **clean oracle 仍显示大量 headroom。** Cora pair `sel_clean_oracle` 0.8143 vs `sel_topk` 0.6082，说明当前 selection 还远未接近上限；后续不应把当前方法写成强 SOTA，而应主张 label-free、协议内可比、结构化噪声下稳定改进。

### 下一步

若继续方法论文，应铺正式主表但保持 claim 克制：

```text
Cora, CiteSeer, PubMed
noise: symmetric + pair/asymmetric, 0.2 + 0.4
methods: CE, co-teaching, sel_noball, sel_topk, sel_dcgb, shuffled/random controls, clean oracle diagnostic
primary method: sel_topk / local-consistency selection
DCGB/granular ball: implementation variant + mechanism/ablation
```

反重复新增：(10) 不要把两格生死验证解读成"粒球翻身"；它证明的是 local consistency reliability 有方法价值，且 `sel_topk` 当前比 `sel_dcgb` 更适合做主方法。若要重新抬高粒球地位，必须先证明 `sel_dcgb` 在某个正式设置中稳定超过 `sel_topk`。

### Paired t 复核补充（2026-06-08）

对两格生死验证进一步做 5-seed paired t 复核后，§12.10 的方向需要更精确地表述为：

1. **真正稳定显著的赢家是 `sel_topk`，不是 `sel_dcgb` / 粒球。**
   - CiteSeer symmetric 0.4：`sel_topk - coteaching = +1.24pp, t=2.96`（SIG），`sel_topk - sel_noball = +1.05pp, t=3.40`（SIG）；但 `sel_dcgb - sel_topk = -0.82pp, t=-5.31`，0/5 seeds，显著更差。
   - Cora pair 0.4：`sel_topk - coteaching = +8.57pp, t=4.74`（SIG），`sel_topk - sel_noball = +8.65pp, t=4.24`（SIG）；`sel_dcgb - sel_topk = -2.48pp`，仍更差。
   - 结论：主方法固定为 **kNN neighborhood-consensus reliability (`sel_topk`) + commit-style selection**；`sel_dcgb` / 粒球仅作 ablation 和机制分析对象。

2. **相对 CE 的增量过了 mean gate，但尚不显著。**
   - CiteSeer symmetric 0.4：`sel_topk - ce = +1.03pp, t=2.00`（ns）。
   - Cora pair 0.4：`sel_topk - ce = +1.63pp, t=2.19`（ns）。
   - 结论：论文不得写"显著超过 CE / 碾压 baseline"；更稳的 claim 是 **显著优于 co-teaching / pure small-loss 等去噪选择基线，且 label-free、即插即用**。

3. **pair / 高噪声是主场，symmetric 是补充。**
   - Cora pair 0.4 中 `sel_topk` 相对 co-teaching / no-ball 的优势达到 +8pp 量级；CiteSeer symmetric 0.4 中相对 shuffled 仅 +0.41pp 且 ns。
   - 结论：正式主表优先展示 pair/asymmetric 与较高噪声率；symmetric 作为补充而非主卖点。

反重复新增：(11) 主表以后必须报告 paired delta + t 值 + SIG/ns。只报 wins/mean 会掩盖"相对 CE 不显著"这类关键信息。

## 12.11 正式主表第一阶段：Cora 有信号，CiteSeer pair 失败 → 不进入 PubMed/第二阶段（2026-06-08）

按 `gb_graph_aware_minival_6_1/reports/执行计划_正式主表_交Codex_20260608.md` 执行第一阶段正式主表。范围严格限制为 minival 协议内：

```text
datasets: Cora, CiteSeer
noise: pair 0.4, pair 0.6, symmetric 0.4
seeds: 0..4
primary method: sel_topk
ablation: sel_dcgb
baselines: ce, coteaching, sel_noball, sel_shuffled, sel_random, clean oracle diagnostic
```

实现与产物：

- 主表 runner：`gb_graph_aware_minival_6_1/scripts/run_maintable.py`
- 总报告：`gb_graph_aware_minival_6_1/reports/maintable_20260608/report.md`
- 汇总表：`gb_graph_aware_minival_6_1/reports/maintable_20260608/maintable_summary.csv`
- 原始行：`gb_graph_aware_minival_6_1/reports/maintable_20260608/maintable_rows.csv`
- 每格子报告：`gb_graph_aware_minival_6_1/reports/maintable_20260608/{Dataset}_{noise}_{rate}/report.md`

### 第一阶段主表结果

| dataset | noise | ce | coteaching | sel_topk | sel_dcgb | topk-coteach | topk-CE | dcgb-topk | Gate1 | Gate2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Cora | pair0.4 | 0.5920 | 0.5225 | 0.6082 | 0.5834 | +8.57pp / t=4.74 SIG | +1.63pp / t=2.19 ns | -2.48pp / t=-1.54 ns | PASS | PASS |
| Cora | pair0.6 | 0.2506 | 0.2448 | 0.2721 | 0.2592 | +2.73pp / t=1.42 ns | +2.15pp / t=1.30 ns | -1.29pp / t=-1.77 ns | PASS | PASS |
| Cora | symmetric0.4 | 0.7935 | 0.7064 | 0.7969 | 0.8002 | +9.05pp / t=8.13 SIG | +0.35pp / t=0.78 ns | +0.33pp / t=0.71 ns | PASS | PASS |
| CiteSeer | pair0.4 | 0.5956 | 0.6521 | 0.6293 | 0.6082 | -2.29pp / t=-4.30 SIG | +3.37pp / t=4.21 SIG | -2.10pp / t=-1.69 ns | FAIL | FAIL |
| CiteSeer | pair0.6 | 0.3114 | 0.3284 | 0.3088 | 0.3088 | -1.97pp / t=-0.61 ns | -0.27pp / t=-1.00 ns | 0.00pp / t=0.00 ns | FAIL | FAIL |
| CiteSeer | symmetric0.4 | 0.6979 | 0.6958 | 0.7082 | 0.7000 | +1.24pp / t=2.96 SIG | +1.03pp / t=2.00 ns | -0.82pp / t=-5.31 SIG | PASS | PASS |

### 机械判定

- **Cora 三格均通过 Gate1/Gate2**，但 pair0.6 的 `sel_topk - coteaching` 和 `sel_topk - CE` 都是 ns；不能写成显著提升，只能写成弱正信号。
- **CiteSeer pair0.4 / pair0.6 均失败**：
  - pair0.4 中 `coteaching` 显著强于 `sel_topk`（`sel_topk - coteaching = -2.29pp, t=-4.30 SIG`），这是强负证据。
  - pair0.6 中 `sel_topk` 低于 `coteaching` 和 CE，且所有结构信号几乎退化，说明该设置下 local consistency selection 没有方法优势。
- **CiteSeer symmetric0.4 通过**，延续两格生死验证结论：`sel_topk` 可显著超过 co-teaching，但相对 CE 仍 ns。

### 结论

1. **不能进入 PubMed / 低噪声第二阶段。** 预注册规则要求 pair 主场整体支撑方法论文路线；但 CiteSeer pair 两格失败，尤其 pair0.4 被 co-teaching 显著反超。因此第一阶段正式主表未达到继续扩表条件。

2. **`sel_topk` 仍是唯一值得保留的主信号，但 claim 必须进一步收缩。** 它在 Cora pair/symmetric 和 CiteSeer symmetric 上能打过 co-teaching；但在 CiteSeer pair 上失败，说明方法对数据集/噪声结构敏感。论文不能写成"结构化噪声下稳定优于 co-teaching"，只能写成"在部分同配图与噪声设置下，local consistency reliability 可以显著优于 small-loss/co-teaching；其有效性依赖表示空间与噪声模式"。

3. **粒球主线进一步降级。** `sel_dcgb` 在多数正式格子中弱于 `sel_topk`，且从未显示独立优势。DCGB/粒球只能作为 ablation / 机制分析对象，不应作为方法核心。

4. **pair 噪声不是普适主场。** Cora pair0.4 是强正例，但 CiteSeer pair0.4 是强负例；"pair/high-noise 是主场"需要改成更谨慎的条件性表述：**Cora pair 是主场，CiteSeer pair 不是**。

### 下一步决策

当前不建议继续铺正式方法主表。更合理的下一步是回到机制分析，解释为什么 CiteSeer pair 失败：

```text
Q: 为什么 local kNN consensus 在 Cora pair 有效，但在 CiteSeer pair 失败？
候选诊断：
- pair flip 是否与 CiteSeer 表示空间的近邻结构冲突；
- co-teaching 是否在 CiteSeer pair 中受益于小损失排序，而 local consensus 错选同噪声簇；
- topk reliability 的 selection_precision / noise_exclusion 是否在 CiteSeer pair 低于 Cora pair；
- 各类 pair flip 的类别混淆矩阵和 selected class balance。
```

反重复新增：(12) 不要基于两格生死验证直接铺 PubMed/大主表；正式第一阶段已经显示跨数据集不稳定。继续方法论文前，必须先解释并修复 CiteSeer pair 失败，否则只能收缩成"方法+机制分析"或机制论文。

## 12.12 适用边界论文 12 格诊断：C1/C2/C4 成立，label-free 预警未达核心贡献线（2026-06-08）

按 `gb_graph_aware_minival_6_1/reports/执行计划_适用边界论文_交Codex_20260608.md` 执行边界分析实验。范围严格限制在 minival frozen-GRACE 协议：

```text
datasets: Cora, CiteSeer
noise: symmetric, pair
rates: 0.2, 0.4, 0.6
seeds: 0..4
primary method: sel_topk
ablation: sel_dcgb
baselines: ce, coteaching, sel_noball, sel_shuffled, sel_random, clean oracle diagnostic
```

执行说明：第一次 `reports/boundary_20260608/` 运行被用户要求检查 GPU 后中止，属于不完整 CPU/未确认 GPU 输出，不作为论文结果使用。正式结果来自显式 `--gpu-id 0` 的 ROCm run，输出到：

- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/report.md`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/boundary_setting_summary.csv`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/boundary_rows.csv`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/boundary_claims.json`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/boundary_correlations.json`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/figure_A_validity_boundary.png`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/figure_B_label_free_warning.png`
- `gb_graph_aware_minival_6_1/reports/boundary_20260608_gpu/figure_C_selection_vs_downstream.png`

新增代码：

- `src/noisy_label_gb_grace/evaluation/failure_signal.py`：label-free 失效预警指标，只用 embedding + noisy labels + train mask，不接收 clean labels。
- `scripts/run_boundary.py`：12 格边界实验 runner。
- `scripts/summarize_boundary.py`：生成边界 summary、correlation、C1-C4 判定和三张图。
- `tests/test_failure_signal.py`、`tests/test_boundary_summary.py`：覆盖 clean-label-free 签名、toy 区分能力、summary/gate。

### 12 格边界结果（sel_topk 主方法）

| dataset | noise | sel_topk | CE | co-teaching | selection gap (`sel_prec - random clean`) | noise_excl | topk-CE | topk-coteach |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | symmetric0.2 | 0.8117 | 0.8113 | 0.7773 | +0.1608 | 0.8426 | +0.04pp | +3.44pp |
| Cora | symmetric0.4 | 0.7969 | 0.7935 | 0.7064 | +0.3225 | 0.8839 | +0.35pp | +9.05pp |
| Cora | symmetric0.6 | 0.6725 | 0.7136 | 0.6646 | +0.0674 | 0.1729 | -4.11pp | +0.79pp |
| Cora | pair0.2 | 0.7824 | 0.7696 | 0.7414 | +0.1138 | 0.6537 | +1.28pp | +4.10pp |
| Cora | pair0.4 | 0.6082 | 0.5920 | 0.5225 | +0.0972 | 0.5465 | +1.63pp | +8.57pp |
| Cora | pair0.6 | 0.2721 | 0.2506 | 0.2448 | -0.1172 | 0.4012 | +2.15pp | +2.73pp |
| CiteSeer | symmetric0.2 | 0.7106 | 0.7064 | 0.7086 | +0.1085 | 0.6241 | +0.42pp | +0.20pp |
| CiteSeer | symmetric0.4 | 0.7082 | 0.6979 | 0.6958 | +0.2301 | 0.7233 | +1.03pp | +1.24pp |
| CiteSeer | symmetric0.6 | 0.6743 | 0.6685 | 0.6668 | +0.1223 | 0.3220 | +0.58pp | +0.75pp |
| CiteSeer | pair0.2 | 0.7009 | 0.6906 | 0.6970 | +0.0807 | 0.5218 | +1.03pp | +0.40pp |
| CiteSeer | pair0.4 | 0.6293 | 0.5956 | 0.6521 | +0.1185 | 0.5767 | +3.37pp | -2.29pp |
| CiteSeer | pair0.6 | 0.3088 | 0.3114 | 0.3284 | -0.0006 | 0.0000 | -0.27pp | -1.97pp |

### 预注册判定

`boundary_claims.json`：

| Claim | Result | Evidence |
|---|---|---|
| C1 symmetric positive | PASS | symmetric0.4 两数据集均 `sel_topk > coteaching` 且 SIG。 |
| C2 pair boundary | PASS | Cora pair 的 selection gap 随 rate 0.2→0.4→0.6 单调下降，并在 0.6 低于 0。 |
| C3 label-free warning | **FAIL** | 三个 noisy-label+geometry 预警指标最大 `|Pearson r| = 0.4757 < 0.6`，未达核心贡献线。 |
| C4 ball no increment | PASS | 12 格中 `sel_dcgb - sel_topk` 均不构成显著正增益；粒球仍无独立方法优势。 |

### 结论

1. **论文不能写成"三件事完整版"。** C1/C2/C4 成立，但 C3 失败。也就是说，正向有效性、边界机制、粒球无增益都被支持；但当前设计的 label-free 失效预警指标不能可靠预测真实失效，最多作为失败候选/未来工作，不能作为核心贡献。

2. **适用边界主线成立，但需要更精确：边界不只是 pair vs symmetric，而是 noise type × noise rate。** Cora pair0.6 失效非常清楚（selection gap -0.1172），CiteSeer pair0.6 完全退化（noise_excl=0）；同时 Cora symmetric0.6 也出现下游负增益（topk-CE -4.11pp），说明高噪声本身也会破坏邻域一致性选择。

3. **local consistency selection 的正向价值仍然成立，但 claim 必须限定。** symmetric0.4 两数据集稳定优于 co-teaching；Cora pair0.2/0.4 也强于 co-teaching。但 CiteSeer pair0.4/0.6 失败，Cora symmetric0.6 相对 CE 失败。论文不能写"结构化噪声主场"，应写成"在中等噪声或局部一致性仍能区分 clean/noisy 的设置下有效；当噪声率过高或结构化噪声形成局部一致错误时失效"。

4. **粒球继续降级。** `sel_dcgb` 没有在 12 格中证明相对 `sel_topk` 的独立增益；后续论文中粒球只能是 reliability unit / ablation / 机制参照，不应作为主方法卖点。

### 下一步建议

不进入 PubMed 或 NoisyGL 主表。更合理的是把论文骨架固定为 **"local-consistency reliability 的有效性与失效边界分析"**，并补两件小事：

- 机制图/文字：用 `selection gap` 和 `noise_excl_recall` 解释何时有效、何时失效。
- 若仍想保留 label-free 预警贡献，必须重新设计更强的 deployment-time 指标；当前三个候选指标未过线，不应硬写。

反重复新增：(13) 不要把 label-free warning 当作已成功贡献；当前最大相关性 0.476，未过预注册阈值。若以后继续做预警指标，必须先过 C3，再写成核心贡献。
反重复新增：(14) 不要把"pair 噪声"或"高噪声"简单说成主场；边界实验显示 pair0.6 和 symmetric0.6 都可能失效，关键变量是 local consistency signal 是否仍高于随机干净率。

## D2 Stable-GB Reliability A0/A1 索引（2026-07-14）

完整记录见 `02_experiment_history_and_lessons 26.md`，详细报告见 `next_paper/docs/stable_gb_reliability_a0a1_report.md`。

- A0：Cora/CiteSeer 候选量、类别、coverage、分组折、9 视图与粒球非坍缩检查全部通过。
- A1：两个数据集的五类预注册 Gate 全部失败；Cora 主增量为负，CiteSeer 小正点估计的 CI 跨 0 且未超过 permutation p95。
- 决策：D2 NO-GO，不进入 A2、真实 GRACE views 或训练；不得事后改门槛或公式救结果。
# 2026-08-05 supplemental direction: robust-teacher verification of GB consensus

- **Direction:** keep the frozen label-free granular-ball partition and formal LOO target, while using an NRGNN teacher only to attenuate the ball correction weight from non-training-node posterior agreement.
- **Protocol:** Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed0, six target controls, no downstream classifier before the target-quality gate.
- **Result:** complete 36/36 target rows. UTV improved Brier over formal LOO and shuffled in 0/6 cells; mean target-accuracy delta versus formal was -1.335pp. Real grouping beat shuffled in Brier on 6/6, but clean-oracle verification had 0/6 Brier headroom.
- **Supported conclusion:** teacher posterior information is useful as direct soft supervision, but multiplicative confidence attenuation nearly removes formal GB correction and worsens calibration. The failed component is the verifier-only information path, not NRGNN knowledge or local geometry by themselves.
- **Decision:** `abandon`. Do not tune another product, threshold, or confidence normalization under MLP-49. Revisit only if the teacher supplies residual target content or GB regularizes teacher predictions through a separately preregistered direction.
- **Evidence:** `external/NoisyGL/runs/mlp49_utv_gblcss_20260805/mlp49_report.md`.

# 2026-08-05 supplemental direction: LOO minimum-vote gating

- **Hypothesis:** after removing each target node's self-vote, corrections supported by only 2--4 other training labels may be statistically unstable. Raising the activation threshold from `s_min=2` to `s_min=5` might improve target quality without changing frozen GRACE H, ball construction, beta, or the student.
- **Protocol:** Cora/CiteSeer/CoraFull, uniform0.4/0.6, train-only noise, seed0; formal `distribution_only + leave_one_out`; real and size-matched shuffled assignments; `s_min in {2,5,10}`. Only `s_min=5` was a candidate and `s_min=10` was frozen as a stress arm.
- **Availability diagnostic:** no real ball had zero training votes in any of the six cells. Mean other-vote support was 11.05 on Cora, 11.82 on CiteSeer, and 151.25 on CoraFull. Therefore, explicitly ignoring label-empty balls is a no-op under the current 20% split and medium-size balls.
- **Result:** complete 36/36 target rows. `s_min=5` improved Brier on 0/6 cells and worsened it by `+0.003886` on average. Mean target-accuracy delta was only `+0.137pp`, below the preregistered `+0.5pp` gate. Real `s_min=5` still beat shuffled in Brier on 6/6 cells.
- **Mechanism conclusion:** the few low-support nodes are not the dominant failure source. Formal LOO alpha already softens uncertain local votes; replacing this continuous confidence with a hard support cutoff removes useful probability information. At `s_min=10`, Cora/CiteSeer coverage falls to roughly 56%/53% and Brier degrades sharply.
- **Decision:** `abandon`. Keep formal `s_min=2`; do not run downstream Phase B and do not tune another support threshold. MLP-43 remains historical self-inclusive evidence and must not be mixed with this formal LOO result.
- **Evidence:** `external/NoisyGL/runs/mlp50_loo_min_vote_gate_20260805/mlp50_report.md` and `phase_a_gate.json`.
