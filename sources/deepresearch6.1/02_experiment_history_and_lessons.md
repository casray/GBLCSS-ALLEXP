# GB-GCL Experiment History and Lessons

本文件只基于当前项目目录 `/Users/qwwr/Downloads/codex-code` 中的 README、代码、配置、日志、结果表、论文草稿和本目录下的 `outputs/` 证据整理。它不读取、不混用、也不代表其他项目目录中的 GB-GCL 实验最终判断。

# 1. Current Overall Judgment

- Confirmed: 当前项目已经测试 6 个方向：GB 视图增强、GB-FN negative suppression、直接 GB 关系正则、GB soft edge denoise、GB-BGRL/GB-GBT 替代目标、diagnostic-only 分析。
- Confirmed: 没有一个方向达到 strict `continue` 标准。没有方向同时满足“多数据集稳定提升、std 不重叠、机制清楚、无额外复杂度风险”。
- Confirmed: GB 深度参与 view generation 在 Cora 正式/长训练结果中反复低于 GRACE，虽然粒球结构本身能激活。
- Confirmed: 当前最可保留的主线是 GB-FN-GRACE v2 topk-soft，但提升很小：Cora +0.0014，CiteSeer +0.0010，PubMed 持平，且均与 baseline std overlap。
- Likely: GB 更适合作为温和的 false-negative candidate prior，而不是作为 hard semantic label、直接正样本、prototype pulling 或大范围视图重写。
- Likely: Top-k 精细 suppression 比整球 full-soft 或 ratio 扩大 suppression 更安全。
- Inconclusive: GB 相对 no-ball topk 的额外收益很小，候选标签命中率也没有一致优于 no-ball topk。
- Risk: 如果只看 Cora 或只看均值差，不看 std overlap/no-ball 对照，会高估当前方法。
- Risk: 继续堆叠 similarity-adaptive、mass preserve、cross-view ranking、fuzzy affinity、prototype/positive/consistency 分支，容易增加复杂度但不增加证据。
- Next priority: 保留 GB-FN-GRACE v2 作为“轻量结构先验/候选质量 prior”的候选主线，同时补机制诊断和严格表述；不要再盲目发明新公式。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation / augmentation control | 用粒球层级控制边保留、anchor/explore、扰动衰减 | gb_gcl mainline, decay, stronger recovery, freeze, anchor/explore, long/stability rounds | Cora | 粒球激活但正式/长训练低于 GRACE | 视图分布漂移、扰动过强、后期不稳 | abandon | motivation |
| DIR-02 | GB-FN-GRACE false-negative suppression | 用 same-ball 结构先验筛选并 soft downweight 少量负样本 | hard/soft v1, v2 topk, ratio, grid, topk1, no-ball, v21/v22/v23, stable ranking, fuzzy affinity, noise/multidataset | Cora, CiteSeer, PubMed, Amazon, Coauthor sanity | Cora/CiteSeer 小幅正向，PubMed 持平；std overlap | 额外收益极小，no-ball topk 非常接近 | modify | main-method candidate |
| DIR-03 | Direct GB relation regularization | same-ball/prototype/consistency 直接拉近或正则节点 | prototype, shared-ball soft positive, ball consistency | Cora, CiteSeer | sanity 激活但无 validation gain | 直接关系约束不带来收益，Cora 常降 test | abandon | negative result |
| DIR-04 | GB-guided soft edge denoising | 用 same-ball/feature similarity 给边加 soft reliability weight | edge_denoise_only, FN+edge_denoise, reliable-ball tau screen, edge similarity diagnosis | Cora | clean setting 低于 v2/GRACE；固定 tau screen 近 no-op | 固定相似度阈值不分离 clean/noisy edge，正式证据不足 | modify | diagnostic |
| DIR-05 | GB with alternative SSL objectives | 将 GB 视图迁移到 BGRL/Barlow Twins 类目标 | gb_bgrl, gb_gbt | Cora | gb_gbt 比 gb_bgrl 稳但仍低于 GRACE/GB-GCL | 目标函数与当前 GB 视图分布错配 | abandon | negative result |
| DIR-06 | Diagnostic-only analysis | 解释候选质量、边相似度、baseline 健康性和外部 baseline | fn candidate precision, edge similarity, GRACE health, GCA/DGI/PubMed baseline | Cora, CiteSeer, PubMed, Amazon | Top-k 远好于 random；ball vs no-ball 不一致 | 诊断支持温和 claim，不支持强方法 claim | diagnostic-only | diagnostic |

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

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB deep view generation | GB can detect non-degenerate local structure | Use GB only as diagnostic or light objective prior |
| Full same-ball suppression | Some same-ball nodes may be false negatives | Top-k soft suppression with no-ball control |
| Direct same-ball positive/prototype | Local relation may be informative | Confidence-weighted, diagnostic-only relation scores |
| Fixed edge threshold denoise | Edge reliability may matter under noise | Percentile/adaptive reliability filter |
| Alternative SSL objective with GB views | Weak/strong view design may matter | First diagnose view gap, then minimal objective test |

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
- In this project, the reusable signal is conservative Top-k false-negative suppression, not broad GB supervision.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-01 | GB controls view generation | Negative on clean Cora formal/long | View drift / perturbation mismatch | Motivation and diagnostics | abandon |
| DIR-02 | GB scopes Top-k false negatives | Small positive/tie; statistically weak | Signal too small, no-ball close | Candidate mainline with weaker claim | modify |
| DIR-03 | Direct GB positive/prototype/consistency | Sanity stop; no validation gain | Same-ball too coarse as direct relation | Negative result | abandon |
| DIR-04 | GB soft edge reliability | Clean no gain; noisy diagnostics incomplete | Fixed threshold brittle | Future robustness direction | modify |
| DIR-05 | GB-BGRL/GB-GBT | Negative Cora 5-seed | Objective/view mismatch | Negative result | abandon |
| DIR-06 | Diagnostics | Useful explanation | Not a main method | Strong paper constraint | diagnostic-only |

# 9. Recommended Next Directions

## NEXT-DIR-1: Mechanism-first GB-FN consolidation

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

## NEXT-DIR-2: Robustness-focused GB-FN under heavy edge noise

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

## NEXT-DIR-3: Percentile-based edge reliability

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
NEXT-DIR-1 and possibly NEXT-DIR-2。

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
NEXT-DIR-2 or NEXT-DIR-3。

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
NEXT-DIR-1。

### Risk
Moderate；claim should be phrased as project evidence, not universal theorem。

### Current Readiness
moderate

# 11. TODO List

## Missing raw logs

- Round-level aggregate docs exist, but some direction decisions reference `summary.csv/conclusion.txt` more than individual seed logs.
- If preparing appendix, map every table row to exact `seed_*/train.log` and `metrics.json` paths.

## Missing baseline checks

- Amazon-Photo/Computers and Coauthor-CS are not formal multi-seed comparisons.
- PubMed DGI is not run.

## Missing seeds/std

- DIR-03 prototype/positive/consistency only sanity 1 seed。
- DIR-04 reliable-ball tau only quick screen 1 seed。
- PubMed fn_candidate_precision diagnostic currently reports 1 seed。

## Missing datasets

- GB-GCL view generation formal evidence is mostly Cora。
- Edge denoise lacks CiteSeer/PubMed formal noisy results。
- Coauthor-CS only sanity。

## Missing ablations

- For DIR-02, need paired GB-vs-no-ball candidate overlap and per-seed delta tests。
- For DIR-04, need percentile filter ablation if direction continues。

## Missing GB diagnostics

- Radius mean/std。
- Singleton ball ratio。
- Cross-seed assignment stability。
- Cross-epoch assignment stability beyond refresh counts。
- Ball quality distribution and confidence calibration。
- GB loss scale vs InfoNCE for additive branches。

## Missing visualizations

- Candidate overlap GB vs no-ball。
- Per-epoch masked_negative_ratio and selected similarity。
- Edge clean/noisy percentile separation。
- Ball size distribution over epochs。

## Missing explanation for failed directions

- DIR-05 needs embedding collapse / variance diagnostic if ever discussed beyond a short negative result。
- DIR-01 needs view distribution distance if someone revisits augmentation。

## Missing paper-claim evidence

- Strong robustness claim not ready。
- Strong superiority over no-ball topk not ready。
- Strong “GB improves candidate purity” not ready。

