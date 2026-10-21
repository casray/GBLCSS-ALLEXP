# 1. Current Overall Judgment

本文件只覆盖当前项目目录 `/Users/qwwr/Downloads/codex-code_22` 中可见的代码、配置、日志、结果和笔记，不代表其他项目或所有 GB-GCL 项目的最终总判断。

- Confirmed: 本项目至少已经测试 6 个方向：GB 视图生成、GB-FN-GRACE 假负样本抑制、GB 边去噪、GB-BGRL/GB-GBT 非 InfoNCE 迁移、GB 候选有效性诊断、跨数据集/协议健康检查。
- Confirmed: GB 直接参与 view generation 的路线在 Cora formal/long 对比中反复低于 GRACE，应作为负结果和动机，不应继续当主方法堆参数。
- Confirmed: GB-FN-GRACE v2 topk-soft 是当前最稳的正向方法形态，但 Cora/CiteSeer 对 GRACE 的提升约 0.1 个百分点，明显小于 std，主线 claim 仍偏弱。
- Confirmed: full same-ball / ratio-style 宽范围 suppression 会扩大 masked negative ratio 并伤害结果，不应重复。
- Confirmed: similarity-adaptive、mass-preserve、cross-view/cross-only 等复杂变体没有超过 fixed topk 主线。
- Likely: GB 更适合作为轻量辅助先验、诊断信号或鲁棒性场景下的结构约束，而不是直接替代 GRACE 的增强或目标。
- Likely: 当前 z-space 构球经常只有 1-2 个非平凡球，GB 对候选集合的新增信息弱，容易被 pure similarity topk 解释掉。
- Inconclusive: PubMed/Amazon/Coauthor 的跨数据集结论不足，多数是 1 seed、partial 或 sanity-only。
- Risk: 如果只报告 Cora/CiteSeer tiny delta，而不报告 std overlap、no-ball topk 和 PubMed equal result，论文主张会被认为贡献不足。
- Next priority: 先做最小 no-ball 对照、GB 候选诊断和噪声鲁棒性验证，再决定是否把 GB-FN-GRACE 写入 main method。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation / augmentation control | 直接控制同球/跨球边保留和特征 masking | fixed, mild/slow schedule, perturb mainline/stronger, decay, freeze, anchor/explore, long | Cora | 少数短程对比可接近 baseline，但 formal/long 低于 GRACE | 改 view 后破坏 GRACE 的增强分布，复杂度上升但收益不稳 | abandon | negative result |
| DIR-02 | GB-FN-GRACE loss-side false-negative suppression | 用 same-ball 高质量候选对 InfoNCE 负样本做 soft downweight | hard, v1 full-soft, v2 topk, ratio, alpha schedule, strict quality, v21 adaptive, v22 mass-preserve, v23 cross-view | Cora, CiteSeer, PubMed partial, Amazon partial | v2 topk 在 Cora/CiteSeer 略高于 GRACE | 提升小于 std，no-ball topk 接近，PubMed 1 seed 无差异 | modify | main-method candidate |
| DIR-03 | GB-guided soft edge denoising / reliability | 用 same-ball、feature similarity、topology 给边加 reliability weight | denoise only, denoise+FN, feature protect, gamma cross, softball variants | Cora clean, Cora edge_noise_0.5 | 噪声边低权重有 AUC 信号，edge_noise 下略优于 GRACE | clean 图退化，噪声提升小且与 std 重叠，机制到性能转化弱 | modify | diagnostic |
| DIR-04 | GB-guided non-InfoNCE objective replacement | 把 GB view 迁移到 BGRL/GBT 等非 InfoNCE 目标 | GB-GBT, GB-BGRL, stronger baselines | Cora | 无稳定正向趋势 | GB-BGRL 明显崩，GB-GBT 低于 GRACE；机制改变过大 | abandon | negative result |
| DIR-05 | GB candidate precision and no-ball diagnostic | 诊断 same-ball topk 是否比 pure similarity topk 更有信息 | fn_candidate_precision, no-ball topk, PubMed large graph point validation | Cora, CiteSeer, PubMed | no-ball 比 random 强很多，GB 相对 no-ball 无明显优势 | GB same-ball 没打开 candidate precision gap | diagnostic-only | diagnostic |
| DIR-06 | Cross-dataset and protocol health check | 检查 GB-FN 跨数据集、外部 baseline、统一协议可比性 | multi_dataset, external baselines, DGI/GCA, Amazon protocol | Cora, CiteSeer, PubMed, Amazon, Coauthor-CS | Cora/CiteSeer 有完整 baseline，PubMed/Amazon 多为 partial | seed/std 缺失，协议差异大，不能支撑泛化 claim | inconclusive | diagnostic |

# 3. Direction-Level Detailed Records

## DIR-01: GB-guided view generation / augmentation control

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB-guided view generation / augmentation control
- Time period: 2026-04 outputs in current project
- Target problem: 用 GB 结构保护同球边、扰动跨球边，改善 GRACE 视图增强质量。
- Core hypothesis: 同球节点更可能语义一致，跨球连接更可能可扰动；GB 可生成更语义保持的增强视图。
- Baseline: GRACE, same encoder/projector/evaluation where summary marks formal comparable.
- Main related files: `src/augment/view_generator.py`, `src/augment/edge_perturb.py`, `src/augment/granular_ball.py`, `configs/cora_decay_stronger_recovery.yaml`
- Result files: `outputs/cora_ablation_round/summary.json`, `outputs/cora_decay_tuning_round/summary.json`, `outputs/cora_perturbation_decay_comparison/summary.json`, `outputs/cora_long_comparison/summary.json`, `outputs/cora_anchor_explore_v2_round/summary.json`
- Raw logs: per-seed `outputs/cora_*_gb_gcl/seed_*/train.log`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: 随机 edge drop / feature mask 可能破坏局部语义结构。
- Why GB might help: GB 层级提供同球/跨球边先验，可调节 edge keep probability。
- Expected improvement: 更少破坏同语义邻域，更强鲁棒性和稳定性。
- Literature or previous experiment inspiration: `PAPER_PLAN.md` 将其作为早期失败路线和动机。
- Expected paper claim: GB 直接参与 view generation 可以提高 GCL 表示。

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| fixed / mild / slow schedule | 调整 GB view 的 edge keep schedule | Cora | `0.7316-0.7356`，低于 formal GRACE `0.7586` | schedule 是否缓解破坏 | schedule 不足以修复 |
| perturb conservative/mainline/stronger | 调整扰动强度 | Cora | strongest `0.7398 ± 0.0161` | 更强或更弱扰动 | 仍低于 GRACE |
| decay / stronger recovery / later onset | 后期恢复 edge keep | Cora | best `0.7482 ± 0.0083`，低于相近 GRACE `0.7532 ± 0.0074` | 后期减少扰动 | 只缩小差距 |
| freeze hierarchy | 冻结后期球层级 | Cora | freeze `0.7460 ± 0.0077` | 避免层级漂移 | 没有改善 |
| anchor/explore v2 | anchor/explore 异步层级 | Cora | `0.7354 ± 0.0146` | 稳定弱视图/强视图 | 不如 stronger 和 GRACE |
| long comparison | 更长训练 | Cora | GRACE `0.7746 ± 0.0113` vs GB-GCL `0.7656 ± 0.0109` | 长训练收敛 | GB view 仍落后 |

### 3.4 Role of Granular Ball

- GB role: view augmentation control.
- Where GB is generated: 训练中基于当前 encoder 表示或输入特征构建层级。
- Representation space: 主要 `ball_source: z`；builder 内部 detach。
- Affects loss: no direct loss modification.
- Affects augmentation: yes, 同球/跨球 edge keep、feature masking、weak/anchor/explore views。
- Affects graph structure: yes, view edge set.
- Affects sampling: no explicit node sampling.
- Affects evaluation: no.

### 3.5 Result Summary

- Best result: decay stronger recovery `0.7482 ± 0.0083` in `outputs/cora_decay_tuning_round/summary.json`; long GB-GCL `0.7656 ± 0.0109` in long protocol.
- Worst result: schedule/perturb variants around `0.7310-0.7356`.
- Average trend: formal comparable results usually below GRACE by about `0.005-0.016`.
- Mean/std: available for Cora 5 seeds.
- Baseline mean/std: GRACE `0.7586 ± 0.0112` formal; long GRACE `0.7746 ± 0.0113`; perturbation GRACE `0.7532 ± 0.0074`.
- Delta: mostly negative.
- Std overlap: yes.
- Datasets improved: none with reliable formal evidence.
- Datasets worsened: Cora formal/long.
- Per-seed stability: 5 seeds available for major Cora variants.
- Runtime/memory cost: higher than GRACE due periodic GB construction and diagnostics; exact overhead Unknown.
- Raw evidence: `outputs/cora_ablation_round/summary.json`, `outputs/cora_long_comparison/summary.json`, per-seed logs.

### 3.6 What Is Directly Supported

- GB view generation did not beat comparable GRACE in the current project's formal Cora summaries.
- Increasing or scheduling GB perturbation did not produce a robust main-method gain.
- Long training does not rescue the direction under the current implementation.

### 3.7 Reasonable Interpretation

Interpretation: GB-controlled view generation changes the core augmentation distribution too much. Even when edge keep rates recover, the method may reduce useful augmentation diversity or introduce unstable partition-dependent perturbations.

### 3.8 What Cannot Be Concluded

- Cannot conclude all GB augmentation is impossible on all graphs.
- Cannot conclude noisy graph robustness is impossible from clean Cora only.
- Cannot claim GB view generation is paper main method in this project.

### 3.9 Failure Analysis

- Main failure reason: GB directly controls view perturbation and damages the GRACE objective.
- Secondary failure reason: layer/partition choices add complexity without enough signal.
- Evidence: formal and long summaries above.
- Alternative explanation: hyperparameter budget might still be insufficient, but many adjacent variants already failed.
- Minimum check needed: only revisit if a new diagnostic shows exactly which edges should be protected/dropped and improves a noisy-graph setting.

### 3.10 Lessons Learned

- Lesson: Do not keep tuning GB view schedules without a new mechanism.
- Evidence: `outputs/cora_ablation_round/summary.json`, `outputs/cora_decay_tuning_round/summary.json`.
- Future constraint: Any future augmentation-control direction must include edge-level diagnostic and no-GB augmentation ablation.

- Lesson: Long training favors GRACE over current GB-GCL.
- Evidence: `outputs/cora_long_comparison/summary.json`.
- Future constraint: Do not use short-run wins as proof for this direction.

### 3.11 What Is Still Useful

- Reusable code: GB hierarchy builder, edge perturbation stats, diagnostic logging.
- Paper value: negative result motivating why the later paper plan moved GB to objective correction.
- Diagnostic value: mean_num_balls, same/cross edge ratio, edge keep rate logs reveal when GB is too intrusive.

### 3.12 Decision

- Decision: abandon
- Reason: repeated Cora formal/long variants underperform comparable GRACE.
- What should not be tried again: more schedule/freeze/anchor tweaks of the same GB view-generation mechanism.
- Variant still worth trying: only a diagnostic-only edge protection experiment under explicit graph noise.
- Evidence required to revive this direction: multi-seed noisy-graph gain over GRACE and no-GB denoising, plus edge-level diagnostic showing GB identifies corrupted edges.

### Experimental Setup Summary

- Datasets: Cora.
- Seeds: mostly 5 seeds.
- Epochs: 100 and long protocol.
- Encoder: GCN encoder from project configs.
- Projector: GRACE projection head.
- Base loss: InfoNCE.
- Added mechanism: GB-guided view edge/feature perturbation.
- Augmentation: edge dropout + feature masking, replaced/controlled by GB variants.
- Ball generation method: hierarchical granular balls, mostly `ball_source: z`.
- Evaluation protocol: frozen linear evaluation.

### Baseline Comparability Check

- Same dataset: yes for Cora comparisons.
- Same split: expected same project dataset pipeline; exact split per summary not always repeated.
- Same seeds: yes for major formal 5-seed comparisons.
- Same encoder/projector: yes by configs, unless noted.
- Same augmentation: no, this direction intentionally changes augmentation.
- Same evaluation script: yes by project training pipeline.
- Same linear evaluation: yes for formal summaries.
- Baseline rerun: yes in listed summaries.
- Comparable conclusion: partial to yes; augmentation changes are the intended treatment.

### Label Leakage Check

- Labels used in GB generation: none found in GB builder/configs.
- Labels used in purity: `boundary_purity` in code is boundary-mixing based, not class-label purity.
- Labels used in sampling/coarsening: none found.
- Validation/test labels used outside evaluation: not found for training; analysis scripts use labels post-hoc.
- Supervised GB mechanism used in pretraining: no evidence.
- Leakage risk: low.

### Granular-Ball Diagnostics

- Number of balls: Cora GB-GCL variants often around 20-50 mean balls; max sampled level up to 8.
- Average ball size: Unknown in aggregate summaries.
- Coverage: all nodes assigned by hierarchy.
- Relation density: same/cross edge ratios logged.
- Cross-epoch stability: split/freeze/level usage logged, but no explicit stability metric.
- Cross-seed stability: mean/std for num_balls available in several summaries.
- Too coarse: not the dominant issue in this direction.
- Too fragmented: possible in fixed/schedule variants with high ball counts; exact singleton ratio Unknown.
- TODO: add singleton ratio and cross-seed partition agreement before any revival.

## DIR-02: GB-FN-GRACE loss-side false-negative suppression

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB-FN-GRACE false-negative suppression
- Time period: 2026-04 outputs in current project.
- Target problem: 标准 InfoNCE 将除正样本外的节点都当负样本，可能误伤语义近邻。
- Core hypothesis: 高质量 same-ball 中的少量 top-k 高相似节点更可能是假负样本，soft downweight 可减少误伤。
- Baseline: GRACE with same encoder/projector/evaluation; no-ball topk diagnostic baseline.
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `src/augment/granular_ball.py`, `configs/cora_fn_grace_v2_topk.yaml`, `PAPER_PLAN.md`
- Result files: `outputs/fn_grace_v2_mainline_round/summary.json`, `outputs/cora_fn_grace_v2_round/summary.json`, `outputs/cora_fn_grace_v21_round/summary.json`, `outputs/cora_fn_grace_v22_round/summary.json`, `outputs/cora_fn_grace_v23_round/summary.json`, `outputs/noball_topk_round/summary.json`, `outputs/noball_large_graph_round/summary.json`
- Raw logs: per-seed `outputs/*_gb_fn_grace/seed_*/train.log`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: false negatives in node-level InfoNCE denominator.
- Why GB might help: same-ball membership may provide conservative structural prior without changing views.
- Expected improvement: keep GRACE views unchanged while downweighting a few high-confidence false negatives.
- Literature or previous experiment inspiration: paper plan explicitly reframes GB from view generation to objective correction.
- Expected paper claim: GB-FN-GRACE is a minimally invasive objective correction that improves GRACE.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| hard | hard remove same-ball negatives | Cora | raw summary exists, not promoted in plan | hard membership as semantic positive/negative filter | Too risky; not current mainline |
| v1 full-soft | downweight all same-ball candidates | Cora | `0.7580 ± 0.0167`, masked ratio `~0.13` | broad soft suppression | Similar mean but worse val/std; over-broad |
| v2 topk fixed alpha | same-ball + quality + topk=5 + alpha=0.25 | Cora/CiteSeer | Cora `0.7600 ± 0.0118`; CiteSeer `0.6766 ± 0.0128` | fine-grained suppression | Best current form, but tiny delta |
| v2 ratio | top_ratio=0.2 | Cora | `0.7544 ± 0.0118`, selected candidates `~195.7` | wider candidate selection | Too wide hurts |
| strict quality | quality threshold 0.2 | Cora | same as v2 | stronger quality filter | No extra gain |
| alpha schedule | scheduled alpha | Cora | same mean as v2 | alpha tuning | No meaningful gain |
| v21 similarity-adaptive | candidate weight based on similarity | Cora | `0.7580 ± 0.0123` | smarter weighting | No improvement over fixed |
| v22 mass-preserve | preserve denominator mass | Cora | `0.7592 ± 0.0112` | avoid mass shrink | No improvement |
| v23 cross-view/cross-only | rank/suppress across views | Cora | cross-both `0.7598`, cross-only `0.7580` | view-aware ranking/scope | Not better than same-view both |
| no-ball topk | remove GB candidate restriction | Cora/CiteSeer/PubMed | close/equal to GB | Is GB needed beyond similarity? | Weakens GB-specific claim |

### 3.4 Role of Granular Ball

- GB role: objective-side false-negative candidate prior.
- Where GB is generated: periodically during training.
- Representation space: mainly encoder `z`; not classifier logits.
- Affects loss: yes, modifies InfoNCE denominator weights.
- Affects augmentation: no for current mainline; view generation remains GRACE-like.
- Affects graph structure: no.
- Affects sampling: only selects negative candidates in loss.
- Affects evaluation: no.

### 3.5 Result Summary

- Best result: Cora v2 topk `0.7600 ± 0.0118`; CiteSeer v2 topk `0.6766 ± 0.0128`.
- Worst result: ratio `0.7544 ± 0.0118`; wider suppression underperforms.
- Average trend: v2 topk is slightly above GRACE on Cora/CiteSeer, but deltas are tiny.
- Mean/std: available for Cora/CiteSeer 5 seeds; PubMed/Amazon mostly partial.
- Baseline mean/std: Cora GRACE `0.7586 ± 0.0112`; CiteSeer GRACE `0.6756 ± 0.0140`.
- Delta: Cora `+0.0014`; CiteSeer `+0.0010`; both inside std.
- Std overlap: yes.
- Datasets improved: Cora/CiteSeer mean only; Amazon-Computers 1 seed tiny positive; Cora edge noise 0.4/0.5 has small positive trend in robustness round.
- Datasets worsened: Amazon-Photo 1 seed worsens; Cora feature_noise 0.1 worsens.
- Per-seed stability: 5 seeds on Cora/CiteSeer; only 1 seed on PubMed/Amazon.
- Runtime/memory cost: periodic GB construction and full similarity selection add cost; exact overhead Unknown.
- Raw evidence: listed summary files and per-seed logs.

### 3.6 What Is Directly Supported

- Top-k soft suppression is better than full same-ball/ratio suppression within this project.
- More complex weighting/mass/cross-view variants did not beat fixed topk.
- Current GB-specific advantage over pure similarity topk is weak.
- Labels are not used for training-side candidate selection; diagnostic labels are post-hoc only.

### 3.7 Reasonable Interpretation

Interpretation: The useful part may be "select a few high-similarity candidates and softly suppress them"; current same-ball restriction may mostly reduce candidate count rather than add semantic precision. GB may still be useful as a conservative gate under noise or when similarity alone is unreliable, but current clean citation evidence is too weak.

### 3.8 What Cannot Be Concluded

- Cannot claim stable statistically significant improvement over GRACE.
- Cannot claim GB candidate selection is better than no-ball similarity topk.
- Cannot claim PubMed/Amazon generalization from 1-seed partial evidence.
- Cannot claim hard same-ball membership is reliable semantic supervision.

### 3.9 Failure Analysis

- Main failure reason: effect size is smaller than variance and no-ball topk explains much of the gain.
- Secondary failure reason: current GB partitions are often very coarse, e.g. mainline Cora mean_num_balls around `1.65`, so same-ball can be too broad and topk becomes mostly similarity-driven.
- Evidence: `outputs/noball_topk_round/summary.json`, `outputs/fn_candidate_precision_round/summary.json`, `outputs/noball_large_graph_round/summary.json`.
- Alternative explanation: current GB generation space (`z`) or first-nontrivial level selection may be too coarse.
- Minimum check needed: equal candidate count/similarity no-ball baseline, PubMed >=3 seeds, and ablation with no GB signal.

### 3.10 Lessons Learned

- Lesson: Fine topk suppression is safer than full same-ball suppression.
- Evidence: v2 topk `0.7600 ± 0.0118` vs v1 full-soft `0.7580 ± 0.0167` and ratio `0.7544 ± 0.0118`.
- Future constraint: never suppress hundreds of same-ball candidates without a strong precision diagnostic.

- Lesson: More complex formula changes are not automatically useful.
- Evidence: v21/v22/v23 all fail to beat v2.
- Future constraint: keep the loss rule simple unless a diagnostic predicts why complexity helps.

- Lesson: GB contribution must be separated from pure similarity topk.
- Evidence: no-ball topk Cora `0.7592 ± 0.0114`, CiteSeer `0.6760 ± 0.0147`, PubMed equal 1 seed.
- Future constraint: every future GB-FN result needs no-ball topk side-by-side.

### 3.11 What Is Still Useful

- Reusable code: `FalseNegativeSuppressedInfoNCELoss`, topk/ratio/adaptive/mass/cross-view selection, diagnostics.
- Paper motivation: objective-side GB is less destructive than view-side GB.
- Positive signal: small but consistent mean improvements on Cora/CiteSeer under formal 5-seed protocol.
- Diagnostic signal: selected candidate stats, masked_negative_ratio, candidate precision analysis.

### 3.12 Decision

- Decision: modify
- Reason: current form is the best method candidate but not strong enough for main-line paper claim without stricter evidence.
- What should not be tried again: full same-ball soft suppression, ratio=0.2 broad selection, similarity-adaptive/mass-preserve/cross-only as standalone improvements.
- Variant still worth trying: minimal GB-FN v2 with stricter no-ball controls, alternative GB generation space, and noisy-graph condition where similarity topk is less reliable.
- Evidence required to revive this direction: > std-overlap improvement on at least two datasets or a clear robustness setting where GB beats no-ball topk.

### Experimental Setup Summary

- Datasets: Cora, CiteSeer; PubMed/Amazon partial.
- Seeds: 5 for Cora/CiteSeer formal; 1 for several large datasets.
- Epochs: 100 pretrain / 100 eval in formal runs.
- Encoder: GCN.
- Projector: GRACE projection head.
- Base loss: InfoNCE.
- Added mechanism: soft negative downweighting.
- GB loss weight: not a separate additive loss; alpha downweights selected negatives.
- Augmentation: GRACE edge/feature dropout unchanged in mainline.
- Ball generation method: hierarchical split by representation/topology quality.
- Evaluation protocol: frozen linear evaluation.

### Baseline Comparability Check

- Same dataset: yes for Cora/CiteSeer formal.
- Same split: expected same project data pipeline; exact split seed should be kept explicit in future.
- Same seeds: yes for Cora/CiteSeer.
- Same encoder/projector: yes.
- Same augmentation: yes for mainline vs GRACE.
- Same evaluation script: yes.
- Same linear evaluation: yes.
- Baseline rerun: yes in current outputs.
- Comparable conclusion: yes for Cora/CiteSeer mean comparison; no/partial for PubMed/Amazon.

### Label Leakage Check

- Labels used in GB generation: none found.
- Labels used in purity: no class-label purity in training; `boundary_purity` is boundary mixing.
- Labels used in sampling/coarsening: none found.
- Validation/test labels used outside evaluation: diagnostic scripts use labels only post-hoc.
- Supervised GB mechanism used in pretraining: no evidence.
- Leakage risk: low.

### Granular-Ball Diagnostics

- Number of balls: Cora v2 topk mean `1.65`; CiteSeer `1.75`; PubMed partial `1.8`.
- Average ball size: Cora high-quality size around `880.1`; CiteSeer around `1247.625`.
- Singleton ball ratio: Unknown.
- Radius mean/std: Unknown.
- Coverage: all nodes assigned.
- Relation density: same_ball_pair_ratio around `0.35-0.38` on Cora/CiteSeer.
- Confidence distribution: quality threshold logged, but full distribution Unknown.
- Cross-epoch stability: ball refresh epochs logged, no partition agreement metric.
- Cross-seed stability: partial via mean/std of number of balls; actual assignment stability Unknown.
- GB loss scale vs contrastive loss: no additive GB loss; negative_mass_scale mostly `1.0`.
- Too coarse: yes, likely, because first nontrivial level often has 1-2 balls.
- TODO: add singleton ratio, ball size distribution, candidate precision per ball.

### Next Minimal Experiment If Modified

- New hypothesis: GB only helps when it filters similarity topk in cases where similarity is unreliable, especially under edge noise.
- Change only: compare same-ball topk vs no-ball topk vs GRACE with identical topk/alpha/warmup/evaluation.
- Keep fixed: encoder, projector, augmentations, evaluation, seed list.
- Minimal datasets: Cora edge_noise 0.4/0.5 and CiteSeer edge_noise 0.4/0.5.
- Minimal seeds: 5 for Cora/CiteSeer; PubMed only after a Cora/CiteSeer signal appears.
- Success criterion: GB-FN beats both GRACE and no-ball topk by more than overlapping std on at least two noise settings.
- Failure criterion: no-ball equals/exceeds GB or deltas remain < std.
- Why this is the smallest useful test: it directly tests the remaining GB-specific contribution without adding new method complexity.

## DIR-03: GB-guided soft edge denoising / reliability

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: GB-guided soft edge denoising / reliability
- Time period: 2026-04 outputs in current project.
- Target problem: noisy or semantically weak edges may hurt contrastive pretraining.
- Core hypothesis: same-ball relation plus feature similarity can assign lower reliability to noisy edges and improve robustness.
- Baseline: GRACE, GB-FN-GRACE v2 topk, no edge denoising.
- Main related files: `src/gb_gcl/edge_reliability.py`, `scripts/analyze_cora_soft_edge_denoise.py`, `scripts/analyze_cora_gamma_cross_scan.py`, `configs/cora_fn_grace_soft_edge_denoise_only.yaml`
- Result files: `outputs/soft_edge_denoise_round/summary.json`, `outputs/soft_edge_denoise_mechanism_round/summary.json`, `outputs/soft_edge_denoise_softball_round/summary.json`, `outputs/gamma_cross_scan_round/summary.json`
- Raw logs: per-seed `outputs/cora_fn_grace_soft_edge_denoise*_gb_fn_grace/seed_*/train.log`
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: random graph corruption and noisy edges.
- Why GB might help: same-ball edges may be more reliable, cross-ball or low-similarity edges may be downweighted.
- Expected improvement: improve edge-noise robustness without hard deleting edges.
- Literature or previous experiment inspiration: negative view-generation results suggested softer edge-level control.
- Expected paper claim: GB denoising improves robustness under graph noise.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| denoise only | edge reliability, FN disabled | Cora clean/noise | clean `0.7560 ± 0.0172`; edge_noise0.5 `0.5480 ± 0.0245` | edge weights only | clean lower than GRACE; noise slightly above GRACE |
| v3 denoise + FN | combine edge denoise and FN | Cora clean/noise | clean `0.7552 ± 0.0164`; edge_noise0.5 `0.5476 ± 0.0247` | composability | no benefit over v2 |
| mechanism analysis | noisy-edge low-weight AUC | Cora edge_noise0.5 | AUC around `0.633`, low-10 precision around `0.694` | whether weights identify noise | diagnostic signal exists |
| feature protect | protect high feature-sim edges | Cora sanity | clean seed0 `0.7630`; edge_noise seed0 `0.5420` | avoid damaging clean semantics | single-seed, mixed |
| gamma cross scan | soft cross-ball affinity | Cora sanity | larger gamma hurts clean/noise | whether cross-ball should be partly trusted | gamma > 0 not helpful |
| softball variants | soft-ball affinity variants | Cora sanity | edge_noise often worse | smoother GB relation | not promising in current form |

### 3.4 Role of Granular Ball

- GB role: edge reliability prior.
- Where GB is generated: from embeddings/hierarchy before computing edge weights.
- Representation space: current embeddings; exact checkpoint/epoch per script.
- Affects loss: no direct loss except when combined with FN.
- Affects augmentation: yes, edge weights affect contrastive views/training graph.
- Affects graph structure: soft edge weights, not hard deletion.
- Affects sampling: no.
- Affects evaluation: no.

### 3.5 Result Summary

- Best result: edge_noise0.5 GB-FN v2 topk `0.5482 ± 0.0274`; denoise only `0.5480 ± 0.0245`; GRACE `0.5424 ± 0.0280`.
- Worst result: clean denoise+FN `0.7552 ± 0.0164`, below GRACE `0.7586 ± 0.0112`; gamma/softball sanity variants sometimes worse.
- Average trend: edge denoising has diagnostic signal but weak performance gains.
- Mean/std: formal 5-seed for clean and edge_noise0.5; many variants single-seed sanity.
- Baseline mean/std: GRACE clean `0.7586 ± 0.0112`; edge_noise0.5 `0.5424 ± 0.0280`.
- Delta: edge_noise0.5 denoise only `+0.0056` over GRACE, inside std.
- Std overlap: yes.
- Datasets improved: Cora edge_noise0.5 mean only.
- Datasets worsened: Cora clean.
- Per-seed stability: formal for main edge-denoise rows; sanity variants insufficient.
- Runtime/memory cost: edge scoring adds per-edge computation; exact overhead Unknown.
- Raw evidence: listed summary files and analysis scripts.

### 3.6 What Is Directly Supported

- Edge reliability scores can rank noisy edges better than random under synthetic edge noise.
- Current edge-denoise variants do not produce a robust accuracy improvement.
- Combining edge denoising with GB-FN does not improve over GB-FN v2.

### 3.7 Reasonable Interpretation

Interpretation: The denoiser sees some noise signal, but the score is not strong or calibrated enough to improve representation learning. It may need to be used only as diagnostic or with a stricter, minimal intervention.

### 3.8 What Cannot Be Concluded

- Cannot claim robust denoising main method.
- Cannot claim gamma-cross or softball variants are generally invalid because most are single-seed sanity.
- Cannot claim clean graph gains.

### 3.9 Failure Analysis

- Main failure reason: edge-level diagnostic signal does not translate into stable downstream gain.
- Secondary failure reason: clean graph downweighting may suppress useful edges.
- Evidence: `outputs/soft_edge_denoise_round/summary.json`, `outputs/soft_edge_denoise_mechanism_round/summary.json`.
- Alternative explanation: current min/max weights or feature protect thresholds are poorly calibrated.
- Minimum check needed: compare against pure feature-sim/topology denoiser and no-GB denoiser under identical noise.

### 3.10 Lessons Learned

- Lesson: AUC signal is not enough for main-method claim.
- Evidence: AUC `~0.633` but edge_noise accuracy gain within std.
- Future constraint: require both mechanism and downstream lift.

- Lesson: Do not combine denoising with FN before each works independently.
- Evidence: v3 denoise+FN does not beat v2 topk.
- Future constraint: keep one GB role per experiment.

### 3.11 What Is Still Useful

- Edge-origin masks and noisy-edge AUC diagnostics.
- Reliability code for future robustness-focused experiments.
- Paper appendix/motivation: GB can expose noisy-edge structure but is not yet a robust denoiser.

### 3.12 Decision

- Decision: modify
- Reason: diagnostic signal exists, but current method is not accuracy-stable.
- What should not be tried again: gamma-cross scan and softball variants as 1-seed performance evidence; combining denoise+FN before isolating edge signal.
- Variant still worth trying: minimal edge_noise-only test with pure feature-sim denoiser, GB-only denoiser, and GB+feature denoiser.
- Evidence required to revive this direction: multi-seed edge-noise gain over GRACE, no-ball/no-GB denoiser, and non-overlapping or practically meaningful delta.

### Baseline Comparability Check

- Same dataset: yes for Cora clean/noise formal rows.
- Same split: expected same project pipeline.
- Same seeds: yes for formal rows.
- Same encoder/projector: yes.
- Same augmentation: edge denoising intentionally changes edge weights.
- Same evaluation script: yes.
- Same linear evaluation: yes.
- Baseline rerun: yes in robustness outputs.
- Comparable conclusion: partial; method changes graph weights by design.

### Label Leakage Check

- Labels used in GB generation: none.
- Labels used in denoising: no training-side label use found.
- Labels used in analysis: yes, post-hoc same-label and noisy-edge analysis in scripts.
- Leakage risk: low for training; medium if post-hoc label analysis is misreported as training evidence.

### Granular-Ball Diagnostics

- Number of balls: denoise-only summaries often show `mean_num_balls=1.0` for some clean runs, indicating no nontrivial same-ball signal.
- Edge reliability: noisy-edge low-weight AUC around `0.633`.
- Same-ball edge ratio: logged in diagnostics.
- Too coarse: yes in some denoise-only runs.
- TODO: report node_to_ball nontrivial rate for each edge-denoise run.

### Next Minimal Experiment If Modified

- New hypothesis: GB helps denoising only when pure feature similarity cannot identify noisy edges.
- Change only: swap edge reliability components: feature-only, GB-only, GB+feature.
- Keep fixed: no FN suppression, same noise generation, same GRACE encoder/eval.
- Minimal datasets: Cora edge_noise0.5 and CiteSeer edge_noise0.4.
- Minimal seeds: 5.
- Success criterion: GB+feature beats feature-only and GRACE by more than std-overlap or shows consistent per-seed wins.
- Failure criterion: feature-only equals/exceeds GB+feature.
- Why this is the smallest useful test: it directly isolates whether GB adds edge-denoise information.

## DIR-04: GB-guided non-InfoNCE objective replacement

### 3.1 Direction Summary

- Direction ID: DIR-04
- Direction name: GB-guided non-InfoNCE objective replacement
- Time period: 2026-04 outputs in current project.
- Target problem: Test whether GB-guided views work better with BGRL/GBT-style objectives than InfoNCE.
- Core hypothesis: Non-contrastive or redundancy-reduction objectives may benefit from stable weak/GB views.
- Baseline: GRACE and GB-GCL stronger_late_recovery.
- Main related files: `src/gb_gcl/models/contrastive.py`, `src/gb_gcl/losses/bgrl_loss.py`, `src/gb_gcl/losses/barlow_twins.py`, `src/augment/view_generator.py`
- Result files: `outputs/cora_gbt_round/summary.json`, `outputs/cora_gb_bgrl_round/summary.json`
- Raw logs: `outputs/cora_gb_gbt_*`, `outputs/cora_gb_bgrl_*`
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: InfoNCE false-negative issue and sensitivity to augmentations.
- Why GB might help: GB could define teacher/student or weak/strong views.
- Expected improvement: avoid negative-sample issue and stabilize representation learning.
- Literature or previous experiment inspiration: BGRL/GBT-style graph SSL baselines.
- Expected paper claim: GB is a general view prior across contrastive and non-contrastive objectives.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-GBT | GB weak-stable plus GB view with Barlow Twins loss | Cora | `0.7304 ± 0.0138` | redundancy-reduction objective | Below GRACE and GB-GCL stronger |
| GB-BGRL | teacher-student BGRL with GB-guided student view | Cora | `0.4164 ± 0.0456` | non-contrastive bootstrap | Severe collapse/underperformance |

### 3.4 Role of Granular Ball

- GB role: view construction for non-InfoNCE objective.
- Where GB is generated: training-time view generator.
- Representation space: `z`/feature depending config.
- Affects loss: objective family changes to BGRL/GBT.
- Affects augmentation: yes.
- Affects graph structure: yes through views.
- Affects sampling: no.
- Affects evaluation: no.

### 3.5 Result Summary

- Best result: GB-GBT `0.7304 ± 0.0138`.
- Worst result: GB-BGRL `0.4164 ± 0.0456`.
- Average trend: strongly negative vs GRACE `0.7586 ± 0.0112`.
- Mean/std: 5 seeds.
- Baseline mean/std: GRACE `0.7586 ± 0.0112`.
- Delta: GB-GBT about `-0.0282`; GB-BGRL about `-0.3422`.
- Std overlap: GB-GBT gap large; GB-BGRL clearly failed.
- Datasets improved: none.
- Datasets worsened: Cora.
- Per-seed stability: GB-BGRL unstable and weak.
- Runtime/memory cost: additional model/objective complexity; exact overhead Unknown.
- Raw evidence: summary files above.

### 3.6 What Is Directly Supported

- Current GB-BGRL and GB-GBT implementations are not viable main methods on Cora.
- Moving away from InfoNCE did not solve the GB augmentation problem.

### 3.7 Reasonable Interpretation

Interpretation: This direction changed too many axes at once: objective, view structure, teacher/student dynamics, and GB hierarchy. Failures cannot isolate a small useful GB signal.

### 3.8 What Cannot Be Concluded

- Cannot conclude BGRL or GBT are bad generally.
- Cannot conclude all non-InfoNCE GB methods fail.
- Cannot reuse these results as criticism of external BGRL/GBT papers.

### 3.9 Failure Analysis

- Main failure reason: objective replacement plus GB view perturbation is too large a mechanism shift.
- Secondary failure reason: BGRL collapse or undertraining likely, but raw diagnosis is insufficient.
- Evidence: `outputs/cora_gb_bgrl_round/summary.json`, `outputs/cora_gbt_round/summary.json`.
- Alternative explanation: implementation/hyperparameters may be immature.
- Minimum check needed: first establish a healthy no-GB BGRL/GBT baseline in this repo.

### 3.10 Lessons Learned

- Lesson: Do not claim GB generalizes across SSL objectives from these runs.
- Evidence: both non-InfoNCE routes underperform.
- Future constraint: any new objective must have a healthy no-GB baseline before adding GB.

### 3.11 What Is Still Useful

- Negative result for appendix/method-selection rationale.
- Code may be useful only after no-GB BGRL/GBT baseline health is proven.

### 3.12 Decision

- Decision: abandon
- Reason: clear Cora underperformance and excessive mechanism shift.
- What should not be tried again: GB-BGRL/GB-GBT as direct continuation without first validating no-GB objective baselines.
- Variant still worth trying: none for current paper timeline.
- Evidence required to revive this direction: healthy BGRL/GBT baseline plus minimal one-change GB ablation.

### Baseline Comparability Check

- Same dataset: yes, Cora.
- Same split/seeds: 5-seed summaries.
- Same encoder/projector: partially; objective/model head changes.
- Same augmentation: no.
- Same evaluation script: yes.
- Comparable conclusion: partial; strong negative enough to stop current direction.

### Label Leakage Check

- Labels used in GB generation: none found.
- Leakage risk: low.

## DIR-05: GB candidate precision and no-ball diagnostic

### 3.1 Direction Summary

- Direction ID: DIR-05
- Direction name: GB candidate precision and no-ball diagnostic
- Time period: 2026-04 outputs in current project.
- Target problem: Determine whether GB same-ball topk adds information beyond pure embedding similarity topk.
- Core hypothesis: If GB is useful, same-ball topk should select more semantically correct false-negative candidates than no-ball similarity topk.
- Baseline: random topk and no-ball topk.
- Main related files: `src/gb_gcl/analysis/fn_candidate_precision.py`, `tests/test_fn_candidate_precision.py`, `outputs/fn_candidate_precision_round/summary.json`
- Result files: `outputs/fn_candidate_precision_round/summary.json`, `outputs/noball_topk_round/summary.json`, `outputs/noball_large_graph_round/summary.json`
- Raw logs: selected experiment per-seed metrics and diagnostic outputs.
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: unclear whether GB-FN gains come from GB or from similarity topk.
- Why GB might help: same-ball constraint should remove false positives from pure similarity.
- Expected improvement: better label precision/hit rate for selected candidates.
- Literature or previous experiment inspiration: mainline paper claim needed mechanism proof.
- Expected paper claim: GB provides a meaningful structural prior over candidate selection.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| candidate precision replay | same anchors, compare ball-topk/no-ball/random | Cora, CiteSeer, PubMed | GB vs no-ball precision delta Cora `-0.0029`, CiteSeer `+0.0010`, PubMed `-0.0003` | candidate semantic precision | GB not consistently better |
| no-ball topk formal | all-negatives topk with same alpha/topk | Cora, CiteSeer | no-ball close to GB | performance contribution | GB-specific delta tiny |
| PubMed large graph point | 1 seed PubMed no-ball vs GB | PubMed | all equal `0.7940` | large citation graph gap | stop condition met; no gap |

### 3.4 Role of Granular Ball

- GB role: diagnostic candidate gate.
- Where GB is generated: from best checkpoint representations for replay/diagnostic.
- Representation space: trained embedding space.
- Affects loss: diagnostic replay and no-ball ablation compare candidate scopes.
- Affects augmentation: no.
- Affects graph structure: no.
- Affects sampling: yes, candidate selection diagnostic.
- Affects evaluation: no training evaluation change.

### 3.5 Result Summary

- Best result: CiteSeer precision delta slightly positive `+0.0010 ± 0.0035`, but inconclusive.
- Worst result: Cora precision delta negative `-0.0029 ± 0.0008`.
- Average trend: no-ball topk is as good or better than GB topk in precision/similarity.
- Mean/std: Cora/CiteSeer multi-seed; PubMed std missing/zero due 1 seed.
- Baseline mean/std: no-ball and random diagnostics available.
- Delta: no-ball vs random precision strongly positive, GB vs no-ball near zero or negative.
- Std overlap: yes for CiteSeer; Cora negative is clearer but small.
- Datasets improved: none with robust evidence.
- Datasets worsened: Cora diagnostic.
- Per-seed stability: available for Cora/CiteSeer.
- Runtime/memory cost: diagnostic-only.
- Raw evidence: `outputs/fn_candidate_precision_round/summary.json`.

### 3.6 What Is Directly Supported

- Pure similarity topk is a very strong candidate selector compared with random.
- Current same-ball restriction does not reliably improve candidate precision over no-ball topk.
- Labels in this diagnostic are explicitly post-hoc and not used for training.

### 3.7 Reasonable Interpretation

Interpretation: Current GB partitions are too coarse or too aligned with embedding similarity to add much information. The mainline improvement, if real, may come from conservative topk downweighting rather than GB-specific semantics.

### 3.8 What Cannot Be Concluded

- Cannot claim GB has no value in all candidate spaces.
- Cannot claim no-ball topk is a final method without full protocol and paper framing.
- Cannot use post-hoc label precision as unsupervised training signal.

### 3.9 Failure Analysis

- Main failure reason: GB does not open a diagnostic precision gap.
- Secondary failure reason: first-nontrivial GB level often too coarse.
- Evidence: precision deltas and PubMed stop decision.
- Alternative explanation: current anchor set biases toward cases where similarity already works.
- Minimum check needed: evaluate cases where no-ball similarity is wrong, not just all selected anchors.

### 3.10 Lessons Learned

- Lesson: Always include no-ball topk before claiming GB-specific false-negative correction.
- Evidence: `outputs/noball_topk_round/summary.json`, `outputs/noball_large_graph_round/summary.json`.
- Future constraint: no GB-FN table should omit no-ball topk.

### 3.11 What Is Still Useful

- Diagnostic scripts and replay method.
- Strong evidence for anti-repetition: do not overclaim same-ball membership.
- Paper use: diagnostic/appendix or limitation.

### 3.12 Decision

- Decision: diagnostic-only
- Reason: It explains why DIR-02 is weak and constrains future claims.
- What should not be tried again: claiming GB candidate precision without no-ball evidence.
- Variant still worth trying: hard-case diagnostic where no-ball topk selects high-similarity wrong-label nodes.
- Evidence required to revive this direction: GB same-ball improves precision/hit by a meaningful margin across datasets and seeds.

### Label Leakage Check

- Labels used in GB generation: no.
- Labels used in diagnostic precision: yes, post-hoc only.
- Validation/test labels used outside evaluation: post-hoc analysis only; do not feed into training.
- Leakage risk: low for training; high if reported as training-side unsupervised criterion.

## DIR-06: Cross-dataset and protocol health check

### 3.1 Direction Summary

- Direction ID: DIR-06
- Direction name: Cross-dataset and protocol health check
- Time period: 2026-04 outputs in current project.
- Target problem: Verify whether Cora/CiteSeer claims generalize and whether external baselines/protocols are healthy.
- Core hypothesis: A paper claim needs comparable baselines and multi-dataset evidence.
- Baseline: GRACE, GCA, DGI; Amazon protocol checks.
- Main related files: `outputs/multi_dataset_round/summary.json`, `outputs/external_baselines_round/summary.json`, `outputs/dgi_baseline_round/summary.json`, `outputs/pubmed_gca_round/summary.json`, `outputs/amazon_protocol_check_round/summary.json`, `PAPER_PLAN.md`
- Result files: same as above.
- Raw logs: per-seed external baseline outputs.
- Current decision: inconclusive

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: mainline evidence may be Cora-only or weak across datasets.
- Why GB might help: if GB is a structural prior, it should not only work on Cora.
- Expected improvement: Cora/CiteSeer/PubMed/Amazon consistent trend.
- Literature or previous experiment inspiration: paper plan's main result table.
- Expected paper claim: GB-FN-GRACE has competitive multi-dataset performance.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| external GCA | add GCA baseline | Cora/CiteSeer, PubMed 3 seeds | CiteSeer GCA `0.6860 ± 0.0151` > GB-FN `0.6766` | stronger baseline | GB-FN not best external baseline |
| DGI | classic DGI baseline | Cora/CiteSeer | weaker than GRACE/GB-FN | low baseline | useful but not decisive |
| PubMed partial | GRACE vs GB-FN | PubMed | both `0.7940`, 1 seed | large graph extension | inconclusive/no gap |
| Amazon partial | GRACE vs GB-FN | Amazon-Photo/Computers | Photo worse, Computers tiny positive | product graphs | inconclusive |
| Coauthor-CS sanity | loader/training sanity | Coauthor-CS | 1 epoch only | feasibility | not result evidence |
| Amazon protocol check | compare unified vs literature-like configs | Amazon | literature-guided config worse in repo protocol | protocol health | do not mix protocols |

### 3.4 Role of Granular Ball

- GB role: evaluation target for GB-FN generalization.
- Where GB is generated: as in DIR-02 when GB-FN runs.
- Representation space: mostly `z`.
- Affects loss: only for GB-FN rows.
- Affects augmentation: no for GB-FN mainline.
- Affects graph structure: no.
- Affects sampling: candidate selection.
- Affects evaluation: no.

### 3.5 Result Summary

- Best result: Cora/CiteSeer formal comparisons exist; PubMed GRACE/GB-FN both `0.7940` in 1 seed.
- Worst result: Amazon-Photo GB-FN `0.3820` vs GRACE `0.4040`, 1 seed.
- Average trend: cross-dataset evidence is incomplete and mixed.
- Mean/std: Cora/CiteSeer yes; PubMed/Amazon mostly missing std or 1 seed.
- Baseline mean/std: GCA and DGI available for Cora/CiteSeer; PubMed GCA 3 seeds.
- Delta: Cora/CiteSeer tiny; Amazon mixed; PubMed zero.
- Std overlap: yes where std exists.
- Datasets improved: Cora/CiteSeer tiny; Amazon-Computers 1 seed tiny.
- Datasets worsened: Amazon-Photo 1 seed; CiteSeer vs GCA.
- Per-seed stability: insufficient outside Cora/CiteSeer.
- Runtime/memory cost: CPU cost noted in summaries.
- Raw evidence: listed summary files.

### 3.6 What Is Directly Supported

- GCA is a stronger external baseline than GB-FN on CiteSeer in current outputs.
- DGI is weaker than GRACE/GB-FN on Cora/CiteSeer.
- PubMed/Amazon evidence is not enough for a main paper generalization claim.
- Amazon protocol summary includes old absolute path strings; this review did not follow/read those external paths and treats only current project's output summary as evidence.

### 3.7 Reasonable Interpretation

Interpretation: Current mainline is a small refinement over GRACE on citation datasets, not a broad SOTA method. Protocol and seed gaps are a larger threat than method polishing.

### 3.8 What Cannot Be Concluded

- Cannot claim PubMed success.
- Cannot claim Amazon generalization.
- Cannot compare to literature numbers directly under different split/evaluation/pretrain protocols.

### 3.9 Failure Analysis

- Main failure reason: missing multi-seed large-dataset evidence.
- Secondary failure reason: protocol mismatch with literature baselines.
- Evidence: `outputs/multi_dataset_round/summary.json`, `outputs/amazon_protocol_check_round/summary.json`.
- Alternative explanation: CPU budget limited the extension.
- Minimum check needed: PubMed at least 3 seeds with GRACE, no-ball topk, GB-FN under identical protocol.

### 3.10 Lessons Learned

- Lesson: Do not promote 1-seed PubMed/Amazon to main table claim.
- Evidence: partial statuses in multi-dataset summaries.
- Future constraint: mark partial rows as Unknown/TODO in paper planning.

- Lesson: Do not mix repo unified protocol with literature protocol numbers.
- Evidence: Amazon protocol check.
- Future constraint: every table row needs protocol label.

### 3.11 What Is Still Useful

- External baselines and protocol documentation.
- Clear TODO list for paper-readiness.
- Evidence that CiteSeer GCA is a strong comparator.

### 3.12 Decision

- Decision: inconclusive
- Reason: important diagnostic work exists, but missing seeds/std prevent direction-level generalization.
- What should not be tried again: direct literature-number comparison or 1-seed main claims.
- Variant still worth trying: PubMed 3-seed minimal protocol with GRACE/no-ball/GB-FN.
- Evidence required to revive this direction: multi-seed comparable results on PubMed and at least one non-citation graph.

### Baseline Comparability Check

- Same dataset: yes within repo rows.
- Same split: current repo unified split; literature split differs on Amazon.
- Same seeds: incomplete outside Cora/CiteSeer.
- Same encoder/projector: mostly yes within repo.
- Same augmentation: yes for GRACE/GB-FN mainline; external baselines differ by method.
- Same evaluation script: repo unified evaluation.
- Same linear evaluation: yes within repo.
- Baseline rerun: yes for Cora/CiteSeer; partial for others.
- Comparable conclusion: partial/inconclusive.

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: GB direct view generation is unstable as main method

### Status
confirmed

### Related Directions
DIR-01, DIR-04

### Observation
GB-GCL view variants and GB-GBT/BGRL variants underperform GRACE on Cora formal/long comparisons.

### Possible Reason
GB perturbation changes the original GRACE augmentation distribution and can reduce useful augmentation diversity.

### Evidence
`outputs/cora_ablation_round/summary.json`, `outputs/cora_long_comparison/summary.json`, `outputs/cora_gbt_round/summary.json`, `outputs/cora_gb_bgrl_round/summary.json`

### Future Constraint
Do not treat GB view generation as paper mainline without a new mechanism diagnostic.

### When to Revisit
Only under explicit noisy-edge robustness, with no-GB denoising controls.

## FP-2: Broad same-ball suppression is too coarse

### Status
confirmed

### Related Directions
DIR-02

### Observation
v1 full-soft and ratio-style selection suppress far more candidates and do not improve over v2 topk.

### Possible Reason
Hard membership or broad same-ball membership includes many valid negatives.

### Evidence
`outputs/fn_grace_v2_mainline_round/summary.json`

### Future Constraint
Use small topk, not full/ratio same-ball suppression.

### When to Revisit
Only if GB diagnostics show high same-ball label precision and fine-grained balls.

## FP-3: Method complexity increases without enough gain

### Status
confirmed

### Related Directions
DIR-01, DIR-02, DIR-03, DIR-04

### Observation
similarity-adaptive, mass-preserve, cross-view, freeze, anchor/explore, BGRL/GBT and denoise combinations did not create robust gains.

### Possible Reason
The base signal is weak; extra machinery mostly adds variance and confounds.

### Evidence
`outputs/cora_fn_grace_v21_round/summary.json`, `outputs/cora_fn_grace_v22_round/summary.json`, `outputs/cora_fn_grace_v23_round/summary.json`, `outputs/cora_anchor_explore_v2_round/summary.json`

### Future Constraint
Each new direction must change exactly one GB role and include a no-GB ablation.

### When to Revisit
After a diagnostic predicts the mechanism.

## FP-4: GB candidate selection does not clearly beat pure similarity topk

### Status
confirmed

### Related Directions
DIR-02, DIR-05

### Observation
No-ball topk is close to GB-FN on Cora/CiteSeer and equal on PubMed 1 seed; candidate precision delta is near zero or negative.

### Possible Reason
Current same-ball partitions are too coarse or redundant with embedding similarity.

### Evidence
`outputs/noball_topk_round/summary.json`, `outputs/noball_large_graph_round/summary.json`, `outputs/fn_candidate_precision_round/summary.json`

### Future Constraint
No future GB-FN claim without no-ball topk.

### When to Revisit
If alternative GB generation gives higher candidate precision across seeds.

## FP-5: Improvement smaller than overlapping std

### Status
confirmed

### Related Directions
DIR-02, DIR-03, DIR-06

### Observation
Cora/CiteSeer GB-FN gains and edge-noise gains are within standard deviations.

### Possible Reason
Signal is small relative to seed variance.

### Evidence
`outputs/fn_grace_v2_mainline_round/summary.json`, `outputs/soft_edge_denoise_round/summary.json`

### Future Constraint
Mark as inconclusive unless effect clears std overlap or has strong per-seed paired evidence.

### When to Revisit
With paired seed deltas and additional datasets.

## FP-6: Cross-dataset evidence is incomplete

### Status
confirmed

### Related Directions
DIR-02, DIR-06

### Observation
PubMed/Amazon rows are mostly 1 seed or sanity-only; Coauthor-CS is 1 epoch.

### Possible Reason
CPU/runtime constraints stopped full validation.

### Evidence
`outputs/multi_dataset_round/summary.json`

### Future Constraint
Do not write generalization claims from these rows.

### When to Revisit
After PubMed >=3 seeds and one non-citation dataset full protocol.

# 5. Cross-Direction Positive Signal Ledger

## PS-1: Top-k soft suppression is safer than broad same-ball suppression

### Status
confirmed

### Related Directions
DIR-02

### Observation
v2 topk is the best GB-FN form on Cora/CiteSeer.

### Why It May Matter
Small, conservative intervention preserves GRACE while testing false-negative correction.

### Required Verification
No-ball topk and multi-dataset checks.

### Possible Paper Use
main method, but only with stronger evidence

## PS-2: GB is more plausible as auxiliary prior than as view replacement

### Status
likely

### Related Directions
DIR-01, DIR-02, DIR-03

### Observation
View-generation variants fail more clearly than loss-side topk.

### Why It May Matter
It gives a defensible research lesson and paper motivation.

### Required Verification
Report negative view-generation rows honestly.

### Possible Paper Use
motivation

## PS-3: Edge reliability has diagnostic noise signal

### Status
possible

### Related Directions
DIR-03

### Observation
Noisy-edge low-weight AUC around `0.633`; low-10 bucket precision around `0.694`.

### Why It May Matter
GB/feature reliability may be useful for robustness or explanation.

### Required Verification
Feature-only/no-GB denoiser and multi-seed downstream lift.

### Possible Paper Use
diagnostic / appendix

## PS-4: Diagnostics can explain failed mainline claims

### Status
confirmed

### Related Directions
DIR-05, DIR-06

### Observation
No-ball candidate and protocol checks reveal why current mainline is weak.

### Why It May Matter
Prevents overclaiming and guides next smallest experiments.

### Required Verification
Keep diagnostics tied to raw outputs and avoid post-hoc leakage.

### Possible Paper Use
appendix / limitation

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GB view-generation schedule/freeze/anchor tuning | Repeatedly below GRACE on Cora formal/long | `outputs/cora_ablation_round/summary.json`, `outputs/cora_long_comparison/summary.json` | Edge-level diagnostic plus noisy-graph multi-seed gain |
| Full same-ball soft suppression | Masks too many negatives, worse than topk | `outputs/fn_grace_v2_mainline_round/summary.json` | Fine balls with high candidate precision |
| Ratio-based same-ball suppression | Selected ~195 candidates and dropped performance | `outputs/fn_grace_v2_mainline_round/summary.json` | Proof that wider suppression improves precision and accuracy |
| Similarity-adaptive / mass-preserve / cross-only variants | Added complexity without beating v2 fixed topk | `outputs/cora_fn_grace_v21_round/summary.json`, `outputs/cora_fn_grace_v22_round/summary.json`, `outputs/cora_fn_grace_v23_round/summary.json` | Diagnostic showing fixed alpha is the bottleneck |
| GB-BGRL / GB-GBT direct continuation | Strong underperformance, too many changed mechanisms | `outputs/cora_gb_bgrl_round/summary.json`, `outputs/cora_gbt_round/summary.json` | Healthy no-GB BGRL/GBT baseline first |
| Claiming GB-FN without no-ball topk | No-ball is close/equal and candidate precision gap is absent | `outputs/noball_topk_round/summary.json`, `outputs/fn_candidate_precision_round/summary.json` | GB beats no-ball topk across seeds/datasets |
| PubMed/Amazon main claim from 1 seed | Missing std and mixed signs | `outputs/multi_dataset_round/summary.json` | >=3 seeds and matched protocol |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB-GCL view replacement | GB may identify local structure | Diagnostic-only edge protection under noise |
| Full same-ball suppression | Some same-ball nodes may be false negatives | Small topk, confidence-gated suppression |
| Ratio suppression | Suppression breadth matters | Per-anchor topk with cap and no-ball control |
| Edge denoise + FN combined | Edge reliability has noise AUC | Isolate edge denoise first |
| GB-BGRL/GBT | GB view may be usable with other objectives | Validate objective baseline before adding GB |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same dataset, split, seeds, encoder/projector, augmentation and evaluation unless the treatment intentionally changes one item.
- Multi-seed evidence: at least 5 seeds for Cora/CiteSeer, at least 3 seeds before PubMed claims.
- Multi-dataset evidence: at least two full datasets beyond a single Cora result.
- Ablation removing GB signal: no-ball topk / no-GB denoiser / GRACE baseline as appropriate.
- Mechanism diagnostic: candidate precision, partition quality, edge-noise AUC, or per-seed paired deltas.
- No label leakage: labels only in evaluation or explicitly post-hoc diagnostics.

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation silently.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as reliable semantic positive without confidence or soft weighting.
- Every new direction must define the exact GB role before coding.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing the GB signal.
- If improvement is smaller than overlapping std, mark it `inconclusive`.
- If a direction only works for one seed, do not move it into the paper main line.
- If clean graph gain is tiny, test noisy graph or strong augmentation before abandoning robustness claims.
- If complexity increases, require stronger evidence than a simple baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.
- In this project, no-ball topk is mandatory for future GB-FN claims.
- In this project, GB view generation should be cited as negative evidence unless a new mechanism is introduced.
- In this project, PubMed/Amazon rows remain TODO until multi-seed std is present.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-01 | GB controls view augmentation | Repeatedly below GRACE | View distribution damage | Yes, negative motivation | abandon |
| DIR-02 | GB-FN topk soft denominator correction | Tiny positive on Cora/CiteSeer, weak vs no-ball | GB adds little beyond similarity topk | Yes, but needs stricter proof | modify |
| DIR-03 | GB edge reliability denoising | Noise AUC signal, weak accuracy lift | Score not calibrated to downstream learning | Yes, diagnostic | modify |
| DIR-04 | GB with BGRL/GBT objectives | Clear underperformance | Too many mechanism changes | Limited negative result | abandon |
| DIR-05 | Candidate precision/no-ball diagnostic | GB not clearly better than no-ball | Coarse/redundant GB partitions | Yes, crucial diagnostic | diagnostic-only |
| DIR-06 | Multi-dataset/protocol check | Incomplete and mixed | Missing seeds/protocol mismatch | Yes, TODO map | inconclusive |

# 9. Recommended Next Directions

## NEXT-DIR-1: Minimal GB-FN contribution proof against no-ball topk

### Why This Direction Is Recommended

It directly addresses the weakest point of the current mainline: GB-specific value.

### Previous Failure It Addresses

DIR-02 and DIR-05 show no-ball topk is close/equal.

### Core Hypothesis

GB helps only when same-ball gating removes high-similarity but semantically wrong no-ball candidates.

### GB Role

Candidate gate for topk false-negative suppression.

### What Must Be Different From Failed Directions

No new weighting formula; strict matched no-ball baseline and paired diagnostics.

### Minimal Implementation

Reuse existing `candidate_scope: same_ball` vs `all_negatives`; add paired per-seed delta table and candidate precision by hard cases.

### Minimal Datasets and Seeds

Cora and CiteSeer, 5 seeds; PubMed 3 seeds only if signal remains.

### Success Criteria

GB beats no-ball by a meaningful paired margin and improves candidate precision on hard anchors.

### Failure Criteria

No-ball equals/exceeds GB or candidate precision gap remains near zero.

### Paper Claim If Successful

GB provides a conservative structural gate for false-negative suppression.

### Reason to Stop Early

If Cora/CiteSeer no-ball remains equal within paired deltas.

## NEXT-DIR-2: Robustness-only edge reliability with no-GB denoiser controls

### Why This Direction Is Recommended

DIR-03 has mechanism signal but weak accuracy lift; isolate whether GB adds anything beyond feature similarity.

### Previous Failure It Addresses

Denoise+FN and gamma/softball variants mixed too many changes.

### Core Hypothesis

GB adds value only under edge noise and only beyond feature-only edge reliability.

### GB Role

Soft edge reliability component.

### What Must Be Different From Failed Directions

No FN suppression; compare feature-only, GB-only, GB+feature, GRACE.

### Minimal Implementation

Existing `edge_reliability.py` with component toggles.

### Minimal Datasets and Seeds

Cora edge_noise0.5 and CiteSeer edge_noise0.4/0.5, 5 seeds.

### Success Criteria

GB+feature beats feature-only and GRACE with nontrivial paired margin.

### Failure Criteria

Feature-only equals/exceeds GB+feature.

### Paper Claim If Successful

GB improves robustness by identifying unreliable edges under graph noise.

### Reason to Stop Early

If feature-only diagnostic AUC and accuracy dominate GB.

## NEXT-DIR-3: GB partition quality and representation-space diagnostic

### Why This Direction Is Recommended

Current GB-FN often has only 1-2 nontrivial balls, making same-ball too coarse.

### Previous Failure It Addresses

Coarse partitions weaken DIR-02 and DIR-05.

### Core Hypothesis

The failure is partly representation-space/level-selection driven.

### GB Role

Diagnostic metadata only.

### What Must Be Different From Failed Directions

No training change; only compare `x`, encoder `z`, pre-projector, and optional normalized spaces.

### Minimal Implementation

Run post-hoc builder on saved checkpoints and output ball count, singleton ratio, size distribution, candidate precision.

### Minimal Datasets and Seeds

Cora/CiteSeer 5 seeds.

### Success Criteria

Find a space/level with finer balls and better candidate precision than current first-nontrivial `z`.

### Failure Criteria

All spaces remain coarse or no better than no-ball.

### Paper Claim If Successful

Current GB use needs calibrated representation space; may justify a revised method.

### Reason to Stop Early

If precision remains no better than no-ball on Cora.

## NEXT-DIR-4: PubMed minimal completion for paper-readiness

### Why This Direction Is Recommended

PubMed is the biggest current evidence gap.

### Previous Failure It Addresses

DIR-06 partial rows and no-ball equal 1-seed result.

### Core Hypothesis

Current mainline either has no PubMed gap or needs different GB settings.

### GB Role

Same as DIR-02, no new method.

### What Must Be Different From Failed Directions

Do not tune before completing comparable seeds.

### Minimal Implementation

Run GRACE, no-ball topk, GB-FN v2 topk with identical configs/protocol for seeds 0,1,2.

### Minimal Datasets and Seeds

PubMed, 3 seeds.

### Success Criteria

GB-FN beats both baselines with consistent paired deltas.

### Failure Criteria

Equal to GRACE/no-ball again.

### Paper Claim If Successful

GB-FN generalizes beyond small citation graphs.

### Reason to Stop Early

If first 3 paired seeds show exact tie or negative GB delta.

# 10. Paper Claim Implications

## Claim-1

### Claim

GB provides robust local structure prior for node-level GCL.

### Supported By Which Directions

Weakly by DIR-02; diagnostically by DIR-03.

### Weakened By Which Directions

DIR-01, DIR-04, DIR-05, DIR-06.

### Current Evidence

Tiny Cora/CiteSeer GB-FN mean deltas; edge denoise AUC signal.

### Missing Evidence

Non-overlapping or paired multi-seed gains, no-ball controls, PubMed completion.

### Required Experiments

NEXT-DIR-1 and NEXT-DIR-4.

### Risk

High: no-ball topk explains much of the current gain.

### Current Readiness
weak

## Claim-2

### Claim

GB-guided augmentation preserves local semantics.

### Supported By Which Directions

None strongly.

### Weakened By Which Directions

DIR-01 and DIR-04.

### Current Evidence

GB view-generation variants underperform GRACE.

### Missing Evidence

Noisy-graph edge-protection evidence.

### Required Experiments

NEXT-DIR-2 only if reframed as robustness/denoising.

### Risk

Very high.

### Current Readiness
not ready

## Claim-3

### Claim

Hard same-ball positive or broad same-ball suppression is not reliable for node-level GCL.

### Supported By Which Directions

DIR-02, DIR-05.

### Weakened By Which Directions

None in current project.

### Current Evidence

full-soft/ratio underperform topk; candidate precision not clearly better than no-ball.

### Missing Evidence

More direct hard-mode summary details if this is a paper claim.

### Required Experiments

Optional hard-vs-soft appendix table from existing logs.

### Risk

Low to medium.

### Current Readiness
moderate

## Claim-4

### Claim

GB is more useful as an auxiliary prior than as a direct replacement for contrastive positives or views.

### Supported By Which Directions

DIR-01 negative result plus DIR-02 weak positive.

### Weakened By Which Directions

DIR-05 weakens GB-specific strength but not the relative positioning.

### Current Evidence

View-generation fails more clearly than loss-side topk.

### Missing Evidence

Stronger loss-side/no-ball proof.

### Required Experiments

NEXT-DIR-1.

### Risk

Medium.

### Current Readiness
weak

## Claim-5

### Claim

GB denoising improves robustness under graph noise.

### Supported By Which Directions

DIR-03 partially.

### Weakened By Which Directions

DIR-03 performance/std overlap and feature-protect sanity mixed results.

### Current Evidence

Cora edge_noise0.5 small mean lift; noisy-edge low-weight AUC around `0.633`.

### Missing Evidence

No-GB denoiser control and CiteSeer robust evidence.

### Required Experiments

NEXT-DIR-2.

### Risk

High.

### Current Readiness
not ready

# 11. TODO List

## Missing raw logs

- Summaries exist for many directions, but this review did not inspect every per-seed raw log in full.
- TODO: for any paper table row, link exact `seed_*/train.log` and `metrics.json`.

## Missing baseline checks

- TODO: record exact split seed and evaluation config for each main result row.
- TODO: avoid using old absolute paths embedded in summaries as source paths; replace with current-project relative paths.

## Missing seeds/std

- PubMed GRACE/GB-FN/no-ball: TODO >=3 seeds.
- Amazon-Photo/Computers: TODO >=3 or 5 seeds before claims.
- Coauthor-CS: current rows are sanity-only, not evidence.

## Missing datasets

- PubMed full comparison.
- At least one non-citation dataset under comparable protocol.

## Missing ablations

- Mandatory no-ball topk for every GB-FN claim.
- Feature-only/no-GB edge denoiser for every edge-denoise claim.
- Remove GB signal ablation for any new direction.

## Missing GB diagnostics

- Singleton ball ratio.
- Ball size distribution.
- Radius mean/std.
- Cross-seed partition stability.
- Candidate precision by ball size and by hard anchors.
- GB computation overhead.

## Missing visualizations

- Candidate precision comparison plot.
- Ball size distribution across epochs/seeds.
- Edge reliability histogram for clean vs noisy edges.
- Paired seed delta plots.

## Missing explanation for failed directions

- GB-BGRL collapse needs only optional diagnosis unless revived.
- GB view-generation failure should be explained via augmentation distribution and edge keep diagnostics.

## Missing paper-claim evidence

- Strong evidence that GB adds value beyond no-ball similarity topk.
- Strong PubMed or larger-graph evidence.
- Robustness evidence not explained by feature similarity alone.

# 12. Experiment 7 Linkage Note

- Added on 2026-05-19: `02_experiment_history_and_lessons 7 .md` now records `实验7noisy_label_gb_grace` as a separate direction-level history.
- Do not merge that file into this document's direction count or DIR numbering; this document remains a separate historical review.
- Relevant new conclusion: frozen GRACE + strict GB local label smoothing under noisy train labels is Cora-positive only in selected medium-label high-noise settings and is cross-dataset unstable on CiteSeer.
- Future constraint imported from Experiment 7: any noisy-label GB direction must report effective smoothing coverage and must beat CE/global/kmeans/topk_matched, not only global smoothing.
