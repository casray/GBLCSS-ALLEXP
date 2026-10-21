# 02 Experiment History and Lessons

本文件只复盘当前项目目录 `/Users/qwwr/Downloads/contrast learning` 中能找到的日志、配置、源码、结果表和项目笔记，不代表其他项目或所有项目的最终总判断。若本文件中的方法名、配置名与其他目录相似，也只以当前目录证据为准。

# 1. Current Overall Judgment

- Confirmed: 当前项目至少测试过 8 个方向；其中 4 个方向已不适合作为主线继续推进，1 个方向为 diagnostic-only，2 个方向值得修改后保留，1 个方向有强结果但不是 GB 贡献。
- Confirmed: 让 granular-ball 深度参与 view generation 的 GB-GCL 路线在长训练和正式对比中反复低于 GRACE；粒球能激活，但激活不等于下游表示变好。
- Confirmed: GB-BGRL 和 GB-GBT 没有解决 GB view 与训练目标错配问题，当前实现明显低于 GRACE。
- Likely: GB 更适合做轻量 objective correction / candidate prior，而不是主导视图生成或替换目标函数。
- Inconclusive: GB-FN-GRACE v2 在 Cora/CiteSeer/PubMed 上相对 GRACE 持平或小幅正向，但提升小于或接近 std，且 no-ball Top-k 极其接近；不能写成强主线胜利。
- Confirmed: x-space percentile / degree-protect edge denoising 在 noisy graph 上有当前项目最强、最稳定的性能信号，但该信号主要不是 GB 造成的。
- Risk: `gb_edge_denoise` 方法名容易误导；当前最强 `pct/s2` 结果的 `mean_num_balls` 常为 1 或 GB role 为 none，不能归因给 granular-ball。
- Risk: 在 clean Cora 上 aggressive percentile edge weighting 会回退，clean-edge collateral damage 是明确风险。
- Next priority: 如果继续 GB 线，应做最小诊断型实验，证明 GB 相对 no-ball Top-k 或 x-space pct 的独立增益；如果写 robustness 线，应把 paper claim 转向 non-GB edge denoising。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB-guided view generation | Direct view augmentation / edge-feature perturbation | gb_gcl mainline, decay, freeze, anchor-explore, stronger_late_recovery | Cora | Active balls but below GRACE | View distribution drift / over-perturbation | abandon | negative result |
| DIR-02 | GB objective replacement | GB views coupled to BGRL/GBT objectives | gb_bgrl, gb_gbt | Cora | Both below GRACE | Objective-view mismatch; bootstrap collapse risk | abandon | negative result |
| DIR-03 | GB-FN-GRACE loss-side false-negative suppression | Same-ball candidate prior for Top-k soft negative downweight | hard, v1 full-soft, v2 topk, alpha schedule, adaptive, mass-preserve, cross-view, no-ball, PubMed/Amazon checks | Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Coauthor-CS sanity | Small positive / tie vs GRACE on citation graphs | Tiny deltas, std overlap, no-ball Top-k nearly matches | modify | main-method candidate |
| DIR-04 | GB diagnostics and topology sensitivity | Diagnostic-only ball partition and candidate analysis | reliable-ball percentile, ball partition diagnosis, candidate precision | Cora, CiteSeer, PubMed | Explains why GB effect is small | Topology changes barely alter balls/candidates | diagnostic-only | diagnostic |
| DIR-05 | GB same/cross-ball edge denoising | Edge weights from same-ball/cross-ball membership or hybrid ball-percentile | soft edge denoise, similarity scoring, hybrid ball percentile | Cora noisy | Weak noisy-edge bias only | Low precision/recall and clean collateral damage | abandon | negative result |
| DIR-06 | Percentile / degree-protect edge denoising | None or incidental; uses x-space endpoint similarity / degree | pct, clean-safe, noise-gated, degree-protect S2 | Cora, CiteSeer, PubMed, Photo | Strong positive on noisy graphs | Clean graph risk; not a GB claim | continue | main-method candidate |
| DIR-07 | Risk-aware augmentation control | None or incidental; risk controls view drops rather than encoder edge weights | train-only, train-test, augmentation-only, mix05/mix07 | Cora | Train-test edge weighting wins; augmentation-only weaker | Augmentation replacement loses denoising strength | abandon | negative result |
| DIR-08 | Weak/strong relation teaching with CLSA | None or incidental; S2 risk used for strong view | s2_clsa_aug, beta/edge ablations, mechanism checks | Cora, CiteSeer, PubMed, Photo | Mixed; positive on several noisy settings, fails CiteSeer | Not robust across datasets; weaker than S2 full often | modify | ablation only |

# 3. Direction-Level Detailed Records

## DIR-01: GB-guided view generation

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB-guided view generation
- Time period: Unknown; artifacts exist in current `outputs/`.
- Target problem: Replace or improve random GRACE augmentation with granular-ball-aware graph views.
- Core hypothesis: Same-ball edges/features should be preserved more, cross-ball or low-quality regions should be perturbed more, producing better contrastive views.
- Baseline: GRACE.
- Main related files: `src/augment/view_generator.py`, `src/augment/edge_perturb.py`, `src/augment/granular_ball.py`, `configs/cora_decay_*`, `configs/cora_anchor_*`, `configs/cora_long_*`.
- Result files: `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`, `outputs/final_main_results.csv`.
- Raw logs: `outputs/cora_decay_gb_mainline_gb_gcl/seed_*/train.log`, `outputs/cora_decay_stronger_recovery_gb_gcl/seed_*/train.log`, `outputs/cora_anchor_v2_*_gb_gcl/seed_*/train.log`.
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: random edge drop and feature mask do not use graph hierarchy.
- Why GB might help: granular balls could distinguish local stable regions from boundary/cross-ball regions.
- Expected improvement: better semantic-preserving augmentation and robustness.
- Literature or previous experiment inspiration: granular-ball local structure prior; project README and `PAPER_PLAN.md` frame this as an early route.
- Expected paper claim: GB can improve GCL by constructing better graph views.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| gb_gcl mainline | GB-based edge/feature perturbation | Cora | Early stability sometimes positive | Whether GB activation improves views | Activation alone is insufficient |
| stronger_late_recovery / decay | Recover late-stage perturbation strength | Cora | Still below GRACE | Whether weaker late perturbation fixes drift | Some recovery, not enough |
| freeze | Freeze hierarchy after epoch | Cora | Worse than stronger_late_recovery | Whether frozen balls remove drift | Freezing does not fix view distribution issue |
| anchor-explore | Weak anchor + GB explore view | Cora | 0.7354 +/- 0.0146, below stronger and GRACE | Whether asymmetric views stabilize contrast | Anchor became too stable / explore gap too large |
| long comparison | Longer training | Cora | GB-GCL below GRACE | Whether advantage persists | Longer training does not preserve advantage |

### 3.4 Role of Granular Ball

- GB role: Directly controls view construction.
- Where GB is generated: `GBGCLViewGenerator.generate`.
- Representation space: mostly `z`; some scan support for `x`.
- Affects loss: indirectly through views.
- Affects augmentation: yes, central mechanism.
- Affects graph structure: yes, edge perturbation by same/cross-ball probabilities.
- Affects sampling: yes, sampled level / anchor level.
- Affects evaluation: no direct change.

### 3.5 Result Summary

- Best result: Some early Cora stability runs showed modest positive trend, but later formal/longer rounds did not preserve it.
- Worst result: Final main table: Cora `gb_gcl stronger` test `0.7430 +/- 0.0157` vs GRACE `0.7586 +/- 0.0112`.
- Average trend: Balls activate but performance remains lower than GRACE under formal/longer comparisons.
- Mean/std: Available in `outputs/final_main_results.csv`, `outputs/cora_long_comparison/conclusion.txt`, and per-run summary JSON.
- Baseline mean/std: GRACE Cora `0.7586 +/- 0.0112` in `outputs/final_main_results.csv`.
- Delta: `gb_gcl stronger` is about `-0.0156` test vs GRACE in final main table.
- Std overlap: Yes; but mean trend is negative across later checks.
- Datasets improved: Unknown beyond early Cora stability.
- Datasets worsened: Cora formal/longer.
- Per-seed stability: Balls stable; accuracy not consistently better.
- Runtime/memory cost: Increased due to hierarchy construction and view generation.
- Raw evidence: `outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`.

### 3.6 What Is Directly Supported

GB partitions can become non-trivial and remain active; this does not imply improved downstream linear evaluation. Multiple stabilizers did not recover GRACE-level performance.

### 3.7 Reasonable Interpretation

Interpretation: GB view generation adds a moving, structured perturbation distribution on top of GRACE. Even when balls are meaningful enough to split, sampled levels, cross-ball perturbation, and feature masking can make the two-view task drift during later training.

### 3.8 What Cannot Be Concluded

Cannot conclude GB itself is useless. This only rejects the current form where GB directly controls view construction.

### 3.9 Failure Analysis

- Main failure reason: View distribution drift / excessive coupling between hierarchy refresh and augmentation.
- Secondary failure reason: Later stabilizers reduced drift but also reduced useful contrastive signal.
- Evidence: `outputs/cora_hierarchy_freeze_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt`.
- Alternative explanation: Hyperparameters may still be suboptimal, but multiple variants already failed to beat simple GRACE.
- Minimum check needed: A no-extra-complexity view guard that preserves GRACE augmentation distribution and only masks a tiny verified noisy subset.

### 3.10 Lessons Learned

- Lesson: Non-trivial GB activation is a diagnostic, not a success metric.
- Evidence: `outputs/cora_long_comparison/conclusion.txt`.
- Future constraint: Any future GB-view method must report both ball health and downstream accuracy against same-seed GRACE.

- Lesson: Freezing hierarchy does not automatically fix view drift.
- Evidence: `outputs/cora_hierarchy_freeze_round/conclusion.txt`.
- Future constraint: Do not retry freeze-only variants without a new mechanism.

### 3.11 What Is Still Useful

Reusable components: granular-ball builder, hierarchy diagnostics, sampled-level stats, edge perturbation stats, anchor/explore debug fields. Paper value: negative motivation for moving GB from view generation to loss-side or diagnostic roles.

### 3.12 Decision

- Decision: abandon
- Reason: Repeated formal/longer Cora variants remain below GRACE despite active balls.
- What should not be tried again: More schedules, freeze-only, or anchor/explore tuning without a new GB role.
- Variant still worth trying: Only a very small augmentation guard with fixed GRACE views and explicit noisy-edge diagnostics.
- Evidence required to revive this direction: Multi-seed improvement over GRACE on at least Cora + CiteSeer, no std-overlap-only claim, and a mechanism diagnostic showing GB avoids useful-edge damage.

## DIR-02: GB objective replacement

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB objective replacement
- Time period: Unknown.
- Target problem: Check whether InfoNCE itself is the bottleneck for GB-GCL.
- Core hypothesis: Teacher-student bootstrap or Barlow Twins-style objectives may tolerate GB views better than contrastive InfoNCE.
- Baseline: GRACE and GB-GCL stronger_late_recovery.
- Main related files: `src/augment/view_generator.py`, `src/gb_gcl/losses/bgrl_loss.py`, `src/gb_gcl/losses/barlow_twins.py`, `src/gb_gcl/models/contrastive.py`.
- Result files: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`.
- Raw logs: `outputs/cora_gb_bgrl_main_gb_bgrl/seed_*/train.log`, `outputs/cora_gb_gbt_main_gb_gbt/seed_*/train.log`.
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: InfoNCE sensitivity to negatives and view mismatch.
- Why GB might help: GB views might be easier to align with bootstrap or redundancy-reduction objectives.
- Expected improvement: Better stability than GB-GCL view generation.
- Literature or previous experiment inspiration: BGRL and Barlow Twins-style SSL.
- Expected paper claim: GB works if paired with a more suitable objective.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-BGRL | Original teacher view + GB student view | Cora | `0.4164 +/- 0.0456` | Bootstrap without negatives | Loss can collapse to easy alignment; representation poor |
| GB-GBT | Weak stable view + GB view with Barlow Twins loss | Cora | `0.7304 +/- 0.0138` | Redundancy reduction with GB views | More stable than BGRL, still below GRACE |

### 3.4 Role of Granular Ball

- GB role: Generates student/GB view, not directly a loss term.
- Where GB is generated: `GBBGRLViewGenerator`, `GBGBTViewGenerator`.
- Representation space: mostly `z`.
- Affects loss: through selected objective and views.
- Affects augmentation: yes.
- Affects graph structure: yes.
- Affects sampling: yes.
- Affects evaluation: no direct change.

### 3.5 Result Summary

- Best result: GB-GBT `0.7304 +/- 0.0138`.
- Worst result: GB-BGRL `0.4164 +/- 0.0456`.
- Average trend: Both below GRACE `0.7586 +/- 0.0112`.
- Mean/std: Available in `outputs/cora_gb_bgrl_round/summary.json` and `outputs/cora_gbt_round/summary.json`.
- Baseline mean/std: GRACE `0.7586 +/- 0.0112`, GB-GCL stronger `0.7430 +/- 0.0157`.
- Delta: GB-GBT about `-0.0282` vs GRACE; GB-BGRL about `-0.3422` vs GRACE.
- Std overlap: GB-GBT remains clearly lower in mean; GB-BGRL clearly failed.
- Datasets improved: None supported.
- Datasets worsened: Cora.
- Per-seed stability: GB-BGRL highly unstable and weak.
- Runtime/memory cost: Higher than GRACE.
- Raw evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`.

### 3.6 What Is Directly Supported

Current GB-BGRL and GB-GBT implementations are not viable main methods on Cora.

### 3.7 Reasonable Interpretation

Interpretation: The issue is not only InfoNCE. The GB view distribution must be compatible with the objective. A stable-looking loss does not guarantee useful embeddings.

### 3.8 What Cannot Be Concluded

Cannot conclude all teacher-student or redundancy-reduction methods fail; only the current minimal GB-coupled versions fail.

### 3.9 Failure Analysis

- Main failure reason: Objective-view mismatch.
- Secondary failure reason: GB-BGRL teacher/student gap and easy near-zero loss.
- Evidence: `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`.
- Alternative explanation: More carefully matched teacher/student augmentations may work, but that becomes a new direction.
- Minimum check needed: A non-GB BGRL/GBT baseline under identical protocol before attributing failure to GB.

### 3.10 Lessons Learned

- Lesson: Replacing InfoNCE is not a shortcut around GB view instability.
- Evidence: GB-BGRL and GB-GBT both below GRACE.
- Future constraint: Do not introduce a new SSL objective before proving the GB signal itself adds value under the existing objective.

### 3.11 What Is Still Useful

The implementations are useful as negative baselines and for understanding objective-view compatibility.

### 3.12 Decision

- Decision: abandon
- Reason: Both tested alternatives are below GRACE and do not explain a path to minimal improvement.
- What should not be tried again: Current teacher-original / student-GB BGRL setup; current GB-GBT as mainline.
- Variant still worth trying: Only after a healthy non-GB BGRL/GBT baseline exists and GB is introduced as a tiny diagnostic prior.
- Evidence required to revive this direction: Comparable non-GB objective baseline plus GB variant improvement across multiple seeds.

## DIR-03: GB-FN-GRACE loss-side false-negative suppression

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: GB-FN-GRACE loss-side false-negative suppression
- Time period: Project notes dated 2026-04-09; artifact timestamps not encoded here.
- Target problem: Standard GRACE treats all non-positive nodes as negatives, creating false-negative risk.
- Core hypothesis: Same-ball high-quality candidates can identify a small set of likely false negatives; soft Top-k downweight is safer than hard or full-ball suppression.
- Baseline: GRACE, GCA, DGI, no-ball Top-k.
- Main related files: `src/gb_gcl/losses/fn_grace_loss.py`, `train.py`, `configs/cora_fn_grace_v2_topk.yaml`, `configs/citeseer_fn_grace_v2_topk.yaml`, `configs/pubmed_fn_grace_v2_topk_bestmatch.yaml`.
- Result files: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, `outputs/cora_fn_grace_v2_round/conclusion.txt`, `outputs/fn_grace_v2_mainline_round/conclusion.txt`, `outputs/noball_topk_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/gb_fn_recheck_round/conclusion.txt`.
- Raw logs: `outputs/cora_fn_grace_v2_topk_gb_fn_grace/seed_*/train.log`, `outputs/citeseer_fn_grace_v2_topk_gb_fn_grace/seed_*/train.log`, `outputs/pubmed_fn_grace_v2_topk_bestmatch_gb_fn_grace/seed_*/train.log` where present.
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: false negatives in InfoNCE denominator.
- Why GB might help: granular balls provide a structure-constrained candidate set before similarity ranking.
- Expected improvement: Safer loss-side correction without changing GRACE views.
- Literature or previous experiment inspiration: Project's shift away from GB view generation; false-negative suppression literature.
- Expected paper claim: GB is more useful as objective correction than as view generation.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| hard | Hard remove same-ball negatives | Cora | Better than GB-GCL, below/unstable vs GRACE | Whether hard membership works | Hard deletion too risky |
| v1 full-soft | Downweight all high-quality same-ball candidates | Cora | `0.7580 +/- 0.0167` | Whether soft same-ball helps | Safer than hard but broad |
| v2 topk fixed-alpha | Top-k same-ball candidates, alpha=0.25 | Cora/CiteSeer/PubMed | Cora `0.7600 +/- 0.0118`; CiteSeer `0.6766 +/- 0.0128`; PubMed `0.7790 +/- 0.0173` formal 3-seed in final table | Whether narrow correction works | Small positive/tie; strongest GB loss-side form |
| alpha schedule / alpha grid | Change alpha schedule or alpha=0.35 | Cora | Mostly tied; best +0.0002 | Whether alpha drives gain | Tuning not worth continuing |
| v2.1 similarity-adaptive | Continuous weight by similarity | Cora | `0.7580`, below fixed | Whether adaptive weighting helps | No gain |
| v2.2 mass-preserve | Preserve negative mass after suppression | Cora | `0.7592`, below fixed | Whether pressure conservation helps | No material effect |
| v2.3 cross-view / cross-only | Rank or suppress cross-view candidates | Cora | At best `0.7598`, below current | Whether same-view ranking is bottleneck | No |
| no-ball Top-k | All-negative Top-k without balls | Cora/CiteSeer/PubMed | Very close to GB-FN | Whether GB adds beyond similarity | GB advantage is tiny |
| Amazon / Coauthor checks | Larger/co-purchase/coauthor graphs | Amazon-Photo/Computers/Coauthor-CS | Mixed / sanity only | Generality | Evidence weak outside citation graphs |

### 3.4 Role of Granular Ball

- GB role: Candidate filter for false-negative suppression.
- Where GB is generated: `build_false_negative_suppression_state` from current `z` hierarchy.
- Representation space: `z`; current code requires `ball_source='z'`.
- Affects loss: yes, InfoNCE denominator weights.
- Affects augmentation: no; GRACE views remain.
- Affects graph structure: no direct train graph rewrite.
- Affects sampling: Top-k candidates within same-ball set.
- Affects evaluation: no direct change.

### 3.5 Result Summary

- Best result: Cora v2 topk `0.7600 +/- 0.0118`, GRACE `0.7586 +/- 0.0112`; CiteSeer v2 `0.6766 +/- 0.0128`, GRACE `0.6756 +/- 0.0140`; PubMed formal 3-seed v2 `0.7790 +/- 0.0173`, GRACE `0.7790 +/- 0.0173`.
- Worst result: Amazon-Photo 1 seed v2 `0.3820` vs GRACE `0.4040`.
- Average trend: Citation graphs are small positive or tied; co-purchase graphs mixed; no-ball Top-k nearly matches.
- Mean/std: Available for Cora/CiteSeer/PubMed in `outputs/final_main_results.csv`.
- Baseline mean/std: Same file.
- Delta: Cora +0.0014; CiteSeer +0.0010; PubMed 0.0000; all small relative to std.
- Std overlap: Yes. Treat statistical strength as inconclusive.
- Datasets improved: Cora and CiteSeer by tiny margins; Amazon-Computers single seed by +0.003.
- Datasets worsened: Amazon-Photo single seed; noisy CiteSeer robustness mixed.
- Per-seed stability: Formal 5-seed for Cora/CiteSeer; PubMed only formal 3-seed in final table; Amazon 1-seed.
- Runtime/memory cost: Extra periodic hierarchy construction.
- Raw evidence: `outputs/final_main_results.csv`, `outputs/final_ablation_results.csv`, `outputs/noball_topk_round/conclusion.txt`.

### 3.6 What Is Directly Supported

GB-FN-GRACE v2 is safer than GB view generation and better than v1 full-soft on Cora. It is not directly supported that GB substantially improves candidate purity over no-ball Top-k; post-hoc precision is near-equal or slightly worse on Cora/PubMed.

### 3.7 Reasonable Interpretation

Interpretation: The robust part is narrow Top-k suppression. GB may provide a mild regularizing candidate prior, but the project evidence does not yet show a large independent GB effect beyond representation similarity.

### 3.8 What Cannot Be Concluded

Cannot claim strong superiority over GRACE, GCA, or no-ball Top-k. Cannot claim GB significantly improves label precision of selected false negatives. Cannot claim robustness under edge noise from GB-FN; CiteSeer noise evidence is mixed.

### 3.9 Failure Analysis

- Main failure reason: Effect size is too small relative to std and no-ball Top-k.
- Secondary failure reason: Ball partitions often have few balls on larger graphs; masked-negative ratio is tiny (`1e-4` to `1e-3` in some checks).
- Evidence: `outputs/noball_topk_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/gb_fn_recheck_round/conclusion.txt`.
- Alternative explanation: The benefit may be regularization/timing rather than candidate purity.
- Minimum check needed: A paired analysis showing GB-FN beats no-ball Top-k across seeds and datasets with a non-overlapping or at least consistent delta.

### 3.10 Lessons Learned

- Lesson: Soft Top-k is safer than hard or full-ball suppression.
- Evidence: Cora v1/v2 ablation in `outputs/final_ablation_results.csv`.
- Future constraint: Do not expand suppression ratio unless accuracy and candidate diagnostics both improve.

- Lesson: Candidate similarity alone is a very strong baseline.
- Evidence: `outputs/noball_topk_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`.
- Future constraint: Every GB-FN claim must include no-ball Top-k.

- Lesson: Paper claims must be modest.
- Evidence: Cora/CiteSeer/PubMed deltas are tiny and overlap std.
- Future constraint: Write "lightweight structural prior" rather than "large improvement".

### 3.11 What Is Still Useful

Main reusable pieces: loss implementation, Top-k candidate diagnostics, no-ball control, post-hoc precision analysis, paper motivation that GB is safer as objective correction than as view generation.

### 3.12 Decision

- Decision: modify
- Reason: Direction is plausible and safer than earlier GB routes, but current evidence is too weak for a strong main-method claim.
- What should not be tried again: More formula variants around adaptive weights, mass preservation, cross-view ranking, or alpha schedules without a new diagnostic target.
- Variant still worth trying: A minimal paired GB-vs-no-ball experiment where only candidate scope differs, with identical seeds and candidate precision/accuracy logged.
- Evidence required to revive this direction: Consistent multi-dataset delta over no-ball Top-k, raw logs, seed-level paired deltas, and a mechanism diagnostic not based on training labels.

### Experimental Setup Summary

- Datasets: Cora, CiteSeer, PubMed, Amazon-Photo, Amazon-Computers, Coauthor-CS sanity.
- Seeds: Cora/CiteSeer 5 seeds; PubMed 3 seeds in final table; Amazon 1 seed; Coauthor-CS 1 epoch sanity.
- Epochs: Mostly 100 pretrain / 100 eval.
- Encoder: GCN-based GRACE model.
- Projector: GRACE projection head.
- Base loss: InfoNCE.
- Added mechanism: same-ball Top-k soft negative downweight.
- GB loss weight: Not separate; alpha downweights negatives.
- Augmentation: unchanged GRACE random edge drop + feature mask.
- Ball generation method: current `z` hierarchy.
- Evaluation protocol: frozen linear evaluation.

### Baseline Comparability Check

- Same dataset: yes for main Cora/CiteSeer/PubMed tables.
- Same split: appears yes under repo configs; exact split seed beyond configs not fully restated in summaries.
- Same seeds: yes for Cora/CiteSeer; partial for PubMed/Amazon.
- Same encoder/projector: yes.
- Same augmentation: yes for GRACE vs GB-FN.
- Same evaluation script: yes in repo.
- Same linear evaluation: yes.
- Baseline rerun: yes in current outputs.
- Comparable conclusion: partial; strong only for Cora/CiteSeer, weaker for PubMed/Amazon.

### Label Leakage Check

- Labels used in GB generation: no evidence.
- Labels used in purity: no for training; labels used in post-hoc `fn_candidate_precision_round` analysis only.
- Labels used in sampling/coarsening: no evidence.
- Validation/test labels used outside evaluation: no evidence.
- Supervised GB mechanism used in pretraining: no.
- Leakage risk: low for training, medium if post-hoc label precision is accidentally promoted as training evidence.

### Granular-Ball Diagnostics

- Number of balls: Cora v2 often max 2 in final mainline summaries; PubMed/Amazon checks also max 2 in recheck note.
- Average ball size: Cora v2 summary shows average high-quality ball size around 880 in seed-level examples.
- Max ball size: available in per-run `summary.json`; not exhaustively copied.
- Singleton ball ratio: Unknown.
- Radius mean/std: Unknown.
- Coverage: high-quality coverage available in diagnostics but not fully summarized here.
- Overlap/conflict: Unknown.
- Relation density: same-ball pair ratio available; Cora v2 around 0.35-0.36 in sample summary.
- Confidence distribution: Unknown.
- Cross-epoch stability: partially available via ball refresh/split epochs.
- Cross-seed stability: partially available.
- GB loss scale vs contrastive loss: Not a separate loss.
- Too coarse: likely on PubMed/Amazon where max balls often 2.
- Too fragmented: not observed for main v2.
- TODO: Add a compact script-generated table of ball counts, candidate counts, and masked ratio per dataset/seed.

### Next Minimal Experiment If Modified

- New hypothesis: GB adds value only when it changes candidate scope in a way that pure similarity cannot.
- Change only: `candidate_scope=same_ball` vs `candidate_scope=all_negatives`.
- Keep fixed: GRACE augmentations, seeds, alpha, Top-k, evaluation, epochs.
- Minimal datasets: Cora + CiteSeer; add PubMed only if GPU available.
- Minimal seeds: 5 for Cora/CiteSeer.
- Success criterion: GB-FN beats no-ball Top-k on paired seed mean by at least +0.003 on both datasets or shows a mechanism metric that predicts final gains.
- Failure criterion: Delta remains below +0.001 or sign flips across datasets.
- Why this is the smallest useful test: It isolates the GB prior from Top-k similarity.

## DIR-04: GB diagnostics and topology sensitivity

### 3.1 Direction Summary

- Direction ID: DIR-04
- Direction name: GB diagnostics and topology sensitivity
- Time period: Unknown.
- Target problem: Understand why GB-FN and reliable-ball changes produce small effects.
- Core hypothesis: Modifying topology or analyzing selected candidates can reveal whether GB changes actual candidate sets.
- Baseline: raw GB-FN v2 and no-ball Top-k.
- Main related files: `src/gb_gcl/analysis/fn_candidate_precision.py`, `src/gb_gcl/analysis/ball_partition_diagnosis.py`, `scripts/summarize_*`.
- Result files: `outputs/reliable_ball_percentile_round/conclusion.txt`, `outputs/ball_partition_diag_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`.
- Raw logs: diagnostic CSV/JSON under each round; exact seed logs available in output dirs.
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: unclear mechanism behind small GB-FN gains.
- Why GB might help: if topology-filtered balls or candidate diagnostics show clear changes, GB role becomes explainable.
- Expected improvement: diagnostic clarity, not necessarily accuracy.
- Literature or previous experiment inspiration: Need to separate direct evidence from plausible narrative.
- Expected paper claim: GB provides a mild structure prior, or diagnostics show why it does not.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| reliable-ball pct70/80/90 | Build balls on filtered topology | Cora noisy quick | Same val/test and same candidate stats | Whether topology pruning changes balls | Almost no change |
| ball partition diagnosis | Compare assignment/candidate changes | Cora noisy | diagnosis-only | Why reliable topology no-ops | Feature term dominates assignment |
| FN candidate precision | Post-hoc label precision | Cora/CiteSeer/PubMed | ball-topk near no-ball | Whether GB improves candidate purity | No large purity gain |

### 3.4 Role of Granular Ball

- GB role: Diagnostic object and candidate generator.
- Where GB is generated: Same hierarchy code as GB-FN.
- Representation space: `z` for mainline; x/topology variants for diagnostics.
- Affects loss: only in diagnostic reruns, not final conclusion.
- Affects augmentation: no.
- Affects graph structure: reliable topology affects ball construction only.
- Affects sampling: candidate set analysis.
- Affects evaluation: no direct change.

### 3.5 Result Summary

- Best result: Diagnostic clarity; not an accuracy direction.
- Worst result: reliable topology did not materially change partitions or candidates.
- Average trend: GB partition/candidate changes are too small in tested settings.
- Mean/std: Not central.
- Baseline mean/std: Not central.
- Delta: reliable pct quick val/test identical across pct70/80/90.
- Std overlap: Not applicable.
- Datasets improved: Not applicable.
- Datasets worsened: Not applicable.
- Per-seed stability: Unknown for diagnostics; several are quick/diagnostic-only.
- Runtime/memory cost: Moderate diagnostic overhead.
- Raw evidence: `outputs/ball_partition_diag_round/conclusion.txt`.

### 3.6 What Is Directly Supported

Filtered topology can remove edges but still leave the first non-trivial ball partition and Top-k candidates almost unchanged. Post-hoc label precision does not show a strong ball-topk advantage over no-ball Top-k.

### 3.7 Reasonable Interpretation

Interpretation: In current GB-FN, features/current representations dominate ball assignments more than modest topology filtering; therefore many GB tweaks do not propagate into different loss weights.

### 3.8 What Cannot Be Concluded

Cannot claim all topology-aware GB methods are no-ops. This conclusion applies to tested percentile topology filtering and current ball assignment.

### 3.9 Failure Analysis

- Main failure reason: The diagnostic change did not affect candidate sets enough to change training.
- Secondary failure reason: Candidate purity is already high from similarity Top-k alone.
- Evidence: `outputs/ball_partition_diag_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`.
- Alternative explanation: Different ball builder or representation space may be more sensitive.
- Minimum check needed: Show assignment and candidate changes before formal training.

### 3.10 Lessons Learned

- Lesson: Do not run formal experiments before checking whether a GB tweak changes balls/candidates.
- Evidence: reliable-ball percentile changed edges but not candidates.
- Future constraint: Every GB topology variant must include a partition-change diagnostic first.

### 3.11 What Is Still Useful

These diagnostics are strong appendix material and prevent repeated no-op GB variants.

### 3.12 Decision

- Decision: diagnostic-only
- Reason: It is useful for explanation and pruning, not a method direction.
- What should not be tried again: reliable topology pct70/80/90 formal runs without changed partition evidence.
- Variant still worth trying: diagnostic-first alternative ball builder or representation source.
- Evidence required to revive this direction: candidate-set Jaccard changes that correlate with accuracy improvements.

## DIR-05: GB same/cross-ball edge denoising

### 3.1 Direction Summary

- Direction ID: DIR-05
- Direction name: GB same/cross-ball edge denoising
- Time period: Unknown.
- Target problem: Improve robustness under injected edge noise by downweighting suspicious graph edges.
- Core hypothesis: Same-ball edges are more reliable; cross-ball edges should receive lower encoder weight.
- Baseline: GRACE under edge noise.
- Main related files: `src/augment/soft_edge_denoise.py`, `train.py`, `configs/cora_gb_edge_denoise_*`.
- Result files: `outputs/soft_edge_denoise_round/conclusion.txt`, `outputs/edge_denoise_precision_round/conclusion.txt`, `outputs/hybrid_ball_quick_summary.json`.
- Raw logs: `outputs/cora_gb_edge_denoise_*_gb_edge_denoise/seed_*/train.log`.
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: noisy edges damage message passing and contrastive views.
- Why GB might help: same-ball membership may identify clean local structure.
- Expected improvement: reduce noisy-edge influence while preserving clean edges.
- Literature or previous experiment inspiration: GB as local structure prior / edge denoising.
- Expected paper claim: GB denoising improves robustness under graph noise.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| same/cross soft weights | Same-ball weight high, cross-ball low | Cora edge_noise=0.4 quick | best val below GRACE quick | Whether GB relation identifies noise | Weak bias, insufficient precision |
| w025/w050/w075 | Cross-ball penalty sweep | Cora edge_noise=0.4 quick | w075 best among GB variants | Strength of cross-ball penalty | Still not enough |
| similarity edge weights + GB penalty | Endpoint similarity with optional GB penalty | Cora | mixed/quick | Whether representation similarity helps | Similarity stronger than GB relation |
| hybrid ball percentile | Percentile weights plus ball bonus/penalty | Cora quick | not promoted | Whether GB improves pct | No clear formal-worthy signal |

### 3.4 Role of Granular Ball

- GB role: Edge relation prior.
- Where GB is generated: `_maybe_refresh_edge_denoise_state`.
- Representation space: usually `z`; hybrid used x-space ball source in code path.
- Affects loss: no direct loss change.
- Affects augmentation: no direct view generation; encoder graph edge weights change.
- Affects graph structure: soft edge weights.
- Affects sampling: no.
- Affects evaluation: by default eval may use unweighted data unless configured; check per config.

### 3.5 Result Summary

- Best result: quick test could be slightly above GRACE, but validation was worse and precision weak.
- Worst result: not promoted to formal.
- Average trend: GB edge relation has only weak noisy-edge bias.
- Mean/std: mostly quick/diagnostic; not sufficient.
- Baseline mean/std: GRACE quick in `outputs/soft_edge_denoise_round/conclusion.txt`.
- Delta: best quick validation delta vs GRACE was negative.
- Std overlap: Unknown due quick screen.
- Datasets improved: Unknown.
- Datasets worsened: validation on Cora quick.
- Per-seed stability: Unknown / quick only.
- Runtime/memory cost: extra edge-weight refresh and snapshots.
- Raw evidence: `outputs/edge_denoise_precision_round/conclusion.txt`.

### 3.6 What Is Directly Supported

GB same/cross-ball edge denoising leans toward downweighting noisy edges, but precision and recall are not high enough; many downweighted edges are clean.

### 3.7 Reasonable Interpretation

Interpretation: Same-ball membership is too coarse for edge noise. Cross-ball does not equal noisy; same-ball does not equal clean.

### 3.8 What Cannot Be Concluded

Cannot claim GB edge denoising improves robustness. Cannot use later pct/S2 results as evidence for this GB mechanism.

### 3.9 Failure Analysis

- Main failure reason: Low noisy-edge coverage and clean collateral damage.
- Secondary failure reason: GB partitions are coarse and not aligned with injected edge noise.
- Evidence: `outputs/edge_denoise_precision_round/conclusion.txt`.
- Alternative explanation: A learned edge scorer could use GB as one feature, but that is a new direction.
- Minimum check needed: Downweight precision/recall above x-space pct under same budget.

### 3.10 Lessons Learned

- Lesson: Cross-ball is not a reliable noisy-edge label.
- Evidence: modest precision and recall in edge denoise diagnosis.
- Future constraint: Any GB edge-denoise method must beat x-space percentile hit-rate before formal training.

### 3.11 What Is Still Useful

Edge-weight snapshot tooling and precision diagnostics are reusable. The negative result motivates non-GB edge-similarity denoising.

### 3.12 Decision

- Decision: abandon
- Reason: Current GB edge prior is too weak and too damaging.
- What should not be tried again: Same/cross-ball edge weights alone, hybrid ball-percentile without diagnostic lift.
- Variant still worth trying: GB as an auxiliary feature in a diagnostic scorer, not as a hard rule.
- Evidence required to revive this direction: Better noisy-edge AUC/AP or precision/recall than pct at the same downweight budget.

## DIR-06: Percentile / degree-protect edge denoising

### 3.1 Direction Summary

- Direction ID: DIR-06
- Direction name: Percentile / degree-protect edge denoising
- Time period: Unknown; later outputs include final paper tables.
- Target problem: Robustness under edge noise.
- Core hypothesis: Low endpoint feature-similarity edges are more likely noisy; downweighting them improves robust GCL.
- Baseline: GRACE, DropEdge, clean-safe variants.
- Main related files: `src/augment/soft_edge_denoise.py`, `train.py`, `configs/*pct*`, `configs/*s2_degree*`.
- Result files: `outputs/final_paper_table.json`, `outputs/statistical_boundary_round/conclusion.txt`, `outputs/pct_multi_noise_formal/conclusion.txt`, `outputs/s2_degree_protect_expanded_formal_round/conclusion.txt`, `outputs/pct_edge_hit_rate_round/conclusion.txt`, `outputs/clean_graph_risk_round/conclusion.txt`.
- Raw logs: `outputs/*pct*gb_edge_denoise/seed_*/train.log`, `outputs/*s2_degree*gb_edge_denoise/seed_*/train.log`.
- Current decision: continue

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: injected noisy edges corrupt message passing.
- Why GB might help: initially explored under `gb_edge_denoise`, but final positive mechanism does not require GB.
- Expected improvement: suppress noisy edges and improve robustness.
- Literature or previous experiment inspiration: graph denoising / robust GCL.
- Expected paper claim: A lightweight edge-similarity prior improves noisy-graph robustness. This is not a GB-specific claim.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| pct_soft quick | x-space percentile edge weights | Cora noise=0.4 | quick positive test but val mixed | Whether endpoint similarity separates noise | Positive signal |
| pct formal | low_ratio/min_weight formal sweeps | Cora/CiteSeer/PubMed/Photo | positive on noisy settings | Whether robust signal generalizes | Strong noisy robustness |
| clean-safe | less severe clean collateral damage | Cora | safer but lower than pct under noise | Whether less damage improves clean/noisy tradeoff | Clean safety can weaken robustness |
| noise-gated | disable/conservative/aggressive by noise level | Cora | diagnostic | Whether avoid clean harm | Useful idea, not final here |
| s2 degree-protect | percentile plus degree protection | Cora/CiteSeer/PubMed/Photo | often matches/slightly beats pct under noise | Whether protect high-degree edges | strongest robust extension |

### 3.4 Role of Granular Ball

- GB role: None in the main positive pct/S2 mechanism; incidental method namespace only.
- Where GB is generated: not needed for `use_percentile_scoring` / `use_degree_protect_percentile`.
- Representation space: raw x-space endpoint cosine.
- Affects loss: indirectly through weighted encoder graph.
- Affects augmentation: no, unless paired with risk-aware augmentation in other directions.
- Affects graph structure: soft edge weights.
- Affects sampling: no.
- Affects evaluation: train/eval edge weighting depends on config; train-test weighting is strongest in current evidence.

### 3.5 Result Summary

- Best result: S2 degree-protect expanded formal: noisy settings all positive vs GRACE; Photo noise=0.4 S2 `0.3134 +/- 0.0251` vs GRACE `0.2644 +/- 0.0280`; CiteSeer noise=0.4 S2 `0.6192 +/- 0.0150` vs GRACE `0.5610 +/- 0.0150`.
- Worst result: Clean Cora pct regresses: pct `0.7432 +/- 0.0207` vs GRACE `0.7588 +/- 0.0110`.
- Average trend: Noisy settings strong; clean setting risky.
- Mean/std: Available in `outputs/statistical_boundary_round/conclusion.txt` and `outputs/s2_degree_protect_expanded_formal_round/conclusion.txt`.
- Baseline mean/std: same result files.
- Delta: Noisy Cora deltas grow from +0.0018 at noise=0.1 to +0.0478 at noise=0.5 for pct.
- Std overlap: Some deltas overlap; several noisy p-values are significant in statistical boundary summary.
- Datasets improved: Cora noisy, CiteSeer noisy, PubMed noisy, Photo noisy.
- Datasets worsened: clean Cora for aggressive pct/S2.
- Per-seed stability: 5-seed formal for main noisy settings.
- Runtime/memory cost: low to moderate edge-weight computation.
- Raw evidence: `outputs/final_paper_table.json`, `outputs/statistical_boundary_round/conclusion.txt`.

### 3.6 What Is Directly Supported

Endpoint feature-similarity percentile edge weighting improves noisy-graph robustness across current noisy formal settings. Degree-protect is a useful variant that sometimes gives additional small gains.

### 3.7 Reasonable Interpretation

Interpretation: Injected noisy edges tend to have lower endpoint feature similarity, so x-space percentile scoring has a stronger noise signal than same/cross-ball membership.

### 3.8 What Cannot Be Concluded

Cannot call this a granular-ball success. Cannot claim clean-graph improvement on Cora. Cannot claim perfect noisy-edge identification; hit-rate diagnosis shows collateral damage remains.

### 3.9 Failure Analysis

- Main failure reason: Clean-edge collateral damage, especially on clean Cora.
- Secondary failure reason: Rule is statistical, not a clean separator.
- Evidence: `outputs/clean_graph_risk_round/conclusion.txt`, `outputs/pct_edge_hit_rate_round/conclusion.txt`.
- Alternative explanation: Learned edge scorer may improve precision/recall.
- Minimum check needed: label-free scorer with lower clean damage at same noisy-edge recall.

### 3.10 Lessons Learned

- Lesson: Strongest current robustness signal is edge-similarity denoising, not GB.
- Evidence: `outputs/final_paper_table.json`.
- Future constraint: Separate robustness paper claims from GB claims.

- Lesson: Clean graph risk must be explicit.
- Evidence: Cora clean regression in `outputs/statistical_boundary_round/conclusion.txt`.
- Future constraint: use noise gate or report clean/noisy boundary before mainline promotion.

### 3.11 What Is Still Useful

This is the strongest current main-method candidate if the paper pivots to noisy-graph robustness. It also provides a strong baseline that future GB edge methods must beat.

### 3.12 Decision

- Decision: continue
- Reason: Strong formal noisy-graph evidence across datasets and seeds.
- What should not be tried again: Presenting this as GB evidence or ignoring clean Cora regression.
- Variant still worth trying: noise-gated pct/S2, learned unsupervised edge scorer, or confidence-calibrated degree protect.
- Evidence required to revive GB relation here: GB features must add beyond pct/S2 in hit-rate and final accuracy.

### Label Leakage Check

- Labels used in edge scoring: no evidence; x-space similarity and degree are label-free.
- Labels used in noisy-edge diagnostics: injected edge-origin flags are used for analysis, not training.
- Validation/test labels used outside evaluation: no evidence.
- Leakage risk: low for method, medium if diagnostic noisy labels are used to choose final paper claims without held-out validation.

## DIR-07: Risk-aware augmentation control

### 3.1 Direction Summary

- Direction ID: DIR-07
- Direction name: Risk-aware augmentation control
- Time period: Unknown.
- Target problem: Use pct/S2 risk as view augmentation, not as persistent encoder edge weight.
- Core hypothesis: Risk-aware edge drop should create stronger robust contrastive views.
- Baseline: GRACE and S2 full edge-weight method.
- Main related files: `src/augment/view_generator.py`, `configs/*s2_aug*`, `configs/*pct_aug*`.
- Result files: `outputs/gcl_boundary_aug_round/conclusion.txt`, `outputs/s2_clsa_aug_round/conclusion.txt`.
- Raw logs: `outputs/*s2_aug*grace/seed_*/train.log`, `outputs/*pct_aug*grace/seed_*/train.log`.
- Current decision: abandon

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: random views do not focus on risky/noisy edges.
- Why GB might help: This direction is not GB-specific; it tests edge-risk augmentation after GB edge denoise proved weak.
- Expected improvement: better robust view diversity.
- Literature or previous experiment inspiration: augmentation-control route.
- Expected paper claim: risk-aware augmentation can replace train-test edge weighting.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| train-test pct/S2 | Edge weights active in train/eval graph | Cora noisy | strongest | Full denoising | Works |
| train-only | Only during training | Cora | much weaker | Whether eval weighting matters | Eval graph matters |
| aug_drop | Drop risky edges in augmentation | Cora | below full S2 | Whether view-only is enough | Not enough |
| aug_mix05/mix07 | mixed random/risk-aware drop | Cora | better than GRACE under noise but below full S2 | Whether mix protects diversity | Partial only |

### 3.4 Role of Granular Ball

- GB role: None in main evidence.
- Where GB is generated: Not relevant.
- Representation space: x-space pct/S2 risk.
- Affects loss: indirectly through views.
- Affects augmentation: yes.
- Affects graph structure: view-only edge drops.
- Affects sampling: edge-drop sampling.
- Affects evaluation: no for augmentation-only variants.

### 3.5 Result Summary

- Best result: train-test pct/S2 remains strongest.
- Worst result: augmentation-only often loses most of the S2 full gain.
- Average trend: augmentation can help vs GRACE under noise but not replace edge weighting.
- Mean/std: `outputs/gcl_boundary_aug_round/conclusion.txt`.
- Baseline mean/std: same.
- Delta: Cora noise=0.4 best augmentation-only around +0.018 vs GRACE but about -0.022 vs S2 full.
- Std overlap: often yes.
- Datasets improved: Cora noisy vs GRACE.
- Datasets worsened: clean Cora or vs S2 full.
- Per-seed stability: formal 5-seed in round summary.
- Runtime/memory cost: low.
- Raw evidence: `outputs/gcl_boundary_aug_round/conclusion.txt`.

### 3.6 What Is Directly Supported

Risk-aware augmentation alone is weaker than using edge weights in the graph seen by the encoder/evaluation pipeline.

### 3.7 Reasonable Interpretation

Interpretation: The robust gain comes from denoising message passing, not only from creating a harder/noisier contrastive view.

### 3.8 What Cannot Be Concluded

Cannot conclude all robust augmentations fail. Only tested pct/S2 augmentation forms are weaker than full edge weighting.

### 3.9 Failure Analysis

- Main failure reason: View-only changes do not remove noisy-edge influence from the base encoder graph enough.
- Secondary failure reason: aggressive risk drop can reduce useful augmentation diversity or damage clean structure.
- Evidence: `outputs/gcl_boundary_aug_round/conclusion.txt`.
- Alternative explanation: A stronger teacher/student objective may extract more from risk-aware views, but current CLSA is mixed.
- Minimum check needed: Must beat S2 full, not just GRACE.

### 3.10 Lessons Learned

- Lesson: Do not replace a working denoising edge weight with augmentation-only control unless it beats S2 full.
- Evidence: gcl boundary table.
- Future constraint: compare every risk-aware augmentation to full edge-weight baseline.

### 3.11 What Is Still Useful

Useful as ablation: it shows where robust signal acts, supporting the interpretation that graph denoising is more important than view replacement.

### 3.12 Decision

- Decision: abandon
- Reason: It does not beat the simpler full edge-weight route.
- What should not be tried again: More mix ratios as main method without new mechanism.
- Variant still worth trying: view augmentation as auxiliary to CLSA or diagnostics, not replacement.
- Evidence required to revive this direction: multi-dataset gains over S2 full with equal or lower clean damage.

## DIR-08: Weak/strong relation teaching with CLSA

### 3.1 Direction Summary

- Direction ID: DIR-08
- Direction name: Weak/strong relation teaching with CLSA
- Time period: Unknown.
- Target problem: Make risk-aware augmentation more useful by teaching distributional relation from weak/strong views.
- Core hypothesis: A CLSA-style distributional loss can stabilize strong risk-aware views and recover fair-GCL gains.
- Baseline: GRACE, pct, S2 full, S2 aug mix05.
- Main related files: `src/gb_gcl/losses/clsa.py`, `train.py`, `configs/*s2_clsa*`.
- Result files: `outputs/s2_clsa_aug_round/conclusion.txt`, `outputs/s2_clsa_aug_expanded_round/conclusion.txt`, `outputs/s2_clsa_ablation_round/conclusion.txt`, `outputs/s2_clsa_mechanism_round/conclusion.txt`.
- Raw logs: `outputs/*s2_clsa*grace/seed_*/train.log`.
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: risk-aware strong views alone are unstable/weak.
- Why GB might help: This direction is not GB-specific; it belongs to robust GCL exploration in this project.
- Expected improvement: fair GCL candidate with better robustness and less train-test edge-weight dependence.
- Literature or previous experiment inspiration: CLSA-style weak/strong distillation.
- Expected paper claim: Weak-view relation teaching stabilizes strong robust views.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| s2_clsa_aug | distributional distillation with S2 strong view | Cora | clean + noisy positives | Whether CLSA helps risk aug | Can help Cora |
| expanded | Cora/CiteSeer/PubMed/Photo | mixed | Generality | Positive in 4/5 new settings vs GRACE, but CiteSeer fails |
| beta/edge ablations | beta, strong keep changes | Cora/CiteSeer partial | no repair | Which parameter matters | Not enough |
| mechanism | loss/strong-view stats | Cora/CiteSeer | many N/A fields | Explain behavior | diagnostics incomplete |

### 3.4 Role of Granular Ball

- GB role: None.
- Where GB is generated: Not central.
- Representation space: S2 risk uses x-space endpoint similarity/degree.
- Affects loss: yes, CLSA distributional loss.
- Affects augmentation: yes, strong risk-aware view.
- Affects graph structure: view-only.
- Affects sampling: risk-aware edge drop.
- Affects evaluation: no direct change.

### 3.5 Result Summary

- Best result: Photo noise=0.4 s2_clsa_aug `0.3586 +/- 0.0339` vs GRACE `0.2644 +/- 0.0280`.
- Worst result: CiteSeer noise=0.4 s2_clsa_aug `0.5564 +/- 0.0416` vs GRACE `0.5610 +/- 0.0150` and far below S2 full `0.6192 +/- 0.0150`.
- Average trend: useful but inconsistent; often weaker than S2 full.
- Mean/std: Available in `outputs/s2_clsa_aug_expanded_round/conclusion.txt`.
- Baseline mean/std: same.
- Delta: new settings average vs GRACE +0.0288, but not robust across datasets.
- Std overlap: often overlaps.
- Datasets improved: Cora, PubMed, Photo; mixed on CiteSeer.
- Datasets worsened: CiteSeer vs GRACE and S2 full.
- Per-seed stability: formal 5-seed in expanded round.
- Runtime/memory cost: extra strong view and CLSA loss.
- Raw evidence: `outputs/s2_clsa_aug_expanded_round/conclusion.txt`.

### 3.6 What Is Directly Supported

CLSA can improve risk-aware augmentation on several noisy settings, but it is not a uniformly reliable replacement for S2 full edge weighting.

### 3.7 Reasonable Interpretation

Interpretation: Relation teaching may help when strong risk-aware views preserve enough class/structure signal, but it can fail when the strong view is too misaligned or dataset-specific.

### 3.8 What Cannot Be Concluded

Cannot claim strong GB-GCL value. Cannot claim CLSA is robust across all datasets. Cannot claim mechanism is fully understood because many diagnostic fields are N/A.

### 3.9 Failure Analysis

- Main failure reason: Cross-dataset inconsistency.
- Secondary failure reason: Mechanism diagnostics incomplete; strong-view precision/recall not always logged.
- Evidence: `outputs/s2_clsa_ablation_round/conclusion.txt`, `outputs/s2_clsa_mechanism_round/conclusion.txt`.
- Alternative explanation: CiteSeer may require different risk source or teacher temperature.
- Minimum check needed: Complete mechanism logging and repair rule for CiteSeer.

### 3.10 Lessons Learned

- Lesson: A relation-teaching objective can help augmentation-only robustness, but must be compared to full edge denoising.
- Evidence: expanded CLSA table.
- Future constraint: Do not promote CLSA as mainline unless it beats S2 full or solves a fairness/clean-risk problem S2 full cannot.

### 3.11 What Is Still Useful

Potential appendix/future-work direction for fair-GCL style robust augmentation. It may be useful on Photo-like noisy settings.

### 3.12 Decision

- Decision: modify
- Reason: Positive but inconsistent; mechanism incomplete.
- What should not be tried again: More CLSA hyperparameters without fixing CiteSeer diagnostics.
- Variant still worth trying: minimal CLSA with complete strong-view precision/recall and a pre-declared repair criterion.
- Evidence required to revive this direction: Positive vs GRACE and not worse than S2 full on Cora/CiteSeer/PubMed, or a clearly different paper claim.

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: Direct GB view generation is unstable as a main method

### Status
confirmed

### Related Directions
DIR-01, DIR-02

### Observation
GB partitions activate but long/formal performance remains below GRACE.

### Possible Reason
Moving hierarchy and sampled granularity alter view distribution across epochs.

### Evidence
`outputs/cora_long_comparison/conclusion.txt`, `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`.

### Future Constraint
Do not tune schedules/freeze/anchor variants as the next GB mainline without a new GB role.

### When to Revisit
Only with a diagnostic proving the GB view changes preserve useful edges/features better than GRACE augmentation.

## FP-2: Hard or broad GB membership is too coarse

### Status
confirmed

### Related Directions
DIR-03, DIR-05

### Observation
v1 full-soft and hard suppression are less safe than v2 Top-k; same/cross-ball edge denoise has weak precision.

### Possible Reason
Same-ball membership is not a semantic label and cross-ball membership is not a noise label.

### Evidence
`outputs/final_ablation_results.csv`, `outputs/edge_denoise_precision_round/conclusion.txt`.

### Future Constraint
Use confidence, Top-k, or diagnostics before using membership as a rule.

### When to Revisit
If ball quality/confidence predicts candidate correctness better than similarity-only baselines.

## FP-3: GB-specific gains are often smaller than simple similarity baselines

### Status
likely

### Related Directions
DIR-03, DIR-04, DIR-06

### Observation
No-ball Top-k nearly matches GB-FN; x-space pct/S2 outperforms GB edge denoise.

### Possible Reason
Current representations/features already encode the local similarity signal that GB attempts to discretize.

### Evidence
`outputs/noball_topk_round/conclusion.txt`, `outputs/fn_candidate_precision_round/conclusion.txt`, `outputs/final_paper_table.json`.

### Future Constraint
Always include no-ball/similarity-only controls.

### When to Revisit
If GB changes candidate distribution in a measurable way and that predicts accuracy.

## FP-4: Clean graph collateral damage

### Status
confirmed

### Related Directions
DIR-05, DIR-06, DIR-07

### Observation
Aggressive pct/S2 improves noisy settings but clean Cora regresses.

### Possible Reason
Low-similarity edges include clean but heterophilous/bridging edges.

### Evidence
`outputs/clean_graph_risk_round/conclusion.txt`, `outputs/statistical_boundary_round/conclusion.txt`.

### Future Constraint
Report clean/noisy boundary and avoid clean-graph claims unless clean settings pass.

### When to Revisit
With a noise gate or confidence mechanism that disables aggressive downweighting on clean graphs.

## FP-5: Complexity increases without enough gain

### Status
confirmed

### Related Directions
DIR-02, DIR-03, DIR-08

### Observation
Adaptive weighting, mass preservation, cross-view ranking, BGRL/GBT, and some CLSA ablations do not justify complexity.

### Possible Reason
The useful signal is small and already captured by simpler mechanisms.

### Evidence
`outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt`, `outputs/cora_gb_bgrl_round/conclusion.txt`, `outputs/cora_gbt_round/conclusion.txt`.

### Future Constraint
Every complex variant needs a diagnostic target before formal training.

### When to Revisit
Only when the simpler method fails on a clearly identified mechanism and the complex variant directly addresses it.

# 5. Cross-Direction Positive Signal Ledger

## PS-1: GB is safer as a loss-side auxiliary prior than as view generation

### Status
likely

### Related Directions
DIR-01, DIR-03

### Observation
GB-FN v2 is close to or slightly above GRACE on citation graphs, while GB-GCL view generation is below GRACE.

### Why It May Matter
It suggests GB should not dominate the training signal.

### Required Verification
Paired GB-FN vs no-ball Top-k with stronger mechanism diagnostics.

### Possible Paper Use
motivation

## PS-2: Narrow Top-k suppression is safer than broad suppression

### Status
confirmed

### Related Directions
DIR-03

### Observation
v2 Top-k beats v1 full-soft and ratio variants on Cora.

### Why It May Matter
It gives a concrete constraint for future false-negative correction.

### Required Verification
Repeat on larger graphs and noisy settings.

### Possible Paper Use
ablation

## PS-3: x-space edge similarity is a strong noisy-edge prior

### Status
confirmed

### Related Directions
DIR-06

### Observation
pct/S2 improves 8/8 noisy settings in statistical boundary summary.

### Why It May Matter
This is the strongest main-method signal in current project artifacts.

### Required Verification
Clarify paper scope: robust edge denoising, not GB.

### Possible Paper Use
main method

## PS-4: Degree protection can reduce or stabilize percentile denoising risk

### Status
likely

### Related Directions
DIR-06

### Observation
S2 is >= pct - 0.005 in all newly requested noisy settings and positive vs GRACE.

### Why It May Matter
It may preserve important high-degree structure while retaining noisy-edge suppression.

### Required Verification
Clean graph and p-value checks beyond current summary.

### Possible Paper Use
main method / ablation

## PS-5: Diagnostics are predictive enough to stop bad branches early

### Status
confirmed

### Related Directions
DIR-04, DIR-05

### Observation
Reliable-ball topology and GB edge denoise diagnostics explain why formal runs would be low value.

### Why It May Matter
Future experiments can be cheaper and less repetitive.

### Required Verification
Keep diagnostics as pre-formal gates.

### Possible Paper Use
appendix / diagnostic

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| GB-GCL view schedules / freeze / anchor-explore | Balls activate but accuracy stays below GRACE | `outputs/cora_decay_tuning_round/conclusion.txt`, `outputs/cora_hierarchy_freeze_round/conclusion.txt`, `outputs/cora_anchor_explore_v2_round/conclusion.txt` | Same-seed multi-dataset gain and diagnostic showing reduced useful-edge damage |
| GB-BGRL current teacher/student setup | Loss becomes easy but embeddings poor | `outputs/cora_gb_bgrl_round/conclusion.txt` | Healthy non-GB BGRL baseline and controlled teacher/student gap |
| GB-GBT current weak/GB view setup | Stable balls but below GRACE | `outputs/cora_gbt_round/conclusion.txt` | Demonstrated objective-view compatibility and GRACE-beating result |
| Full same-ball suppression / hard suppression | Too broad or too brittle | `outputs/cora_fn_grace_round/conclusion.txt`, `outputs/final_ablation_results.csv` | Candidate confidence that beats Top-k fixed-alpha |
| Adaptive/mass-preserve/cross-view GB-FN formula tweaks | No material gain over v2 | `outputs/cora_fn_grace_v21_round/conclusion.txt`, `outputs/cora_fn_grace_v22_round/conclusion.txt`, `outputs/cora_fn_grace_v23_round/conclusion.txt` | A diagnostic showing current v2 failure is caused by that exact formula issue |
| Reliable-ball percentile topology pct70/80/90 | Edge pruning barely changes balls/candidates | `outputs/ball_partition_diag_round/conclusion.txt` | Partition/candidate Jaccard changes before training |
| Same/cross-ball edge denoise alone | Weak noisy-edge precision/recall and clean collateral damage | `outputs/edge_denoise_precision_round/conclusion.txt` | Better hit-rate than x-space pct at same budget |
| Calling pct/S2 a GB result | Main positive mechanism does not require balls | `src/augment/soft_edge_denoise.py`, `outputs/final_paper_table.json` | A GB-added variant that beats pct/S2 with diagnostics |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| GB view generation | GB can describe local hierarchy | Diagnostic prior or tiny loss-side confidence |
| Hard/full same-ball negative suppression | Some same-ball high-sim candidates may be false negatives | Top-k soft with no-ball control |
| Cross-ball edge penalty | Edge relation can weakly correlate with noise | Use GB as one feature in a scorer, not a rule |
| Clean-safe pct | Clean-edge damage matters | Noise gate or calibrated degree protection |
| CLSA on S2 strong view | Weak/strong teaching can help some graphs | Complete mechanism logging and dataset-specific repair rule |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same dataset, split, seeds, encoder, augmentation, evaluation.
- Multi-seed evidence: at least 5 seeds for Cora/CiteSeer; PubMed at least 3 seeds marked clearly if not 5.
- Multi-dataset evidence: not only Cora.
- Ablation removing GB signal: no-ball Top-k or pct-only required for GB claims.
- Mechanism diagnostic: candidate precision/Jaccard or edge hit-rate before formal expansion.
- No label leakage: labels/noisy-edge flags only for diagnostics, never training.

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation when comparing methods.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as reliable semantic positive without confidence or soft weighting.
- Every new direction must define the exact GB role before coding.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing the GB signal.
- If improvement is smaller than overlapping std, mark the paper claim as inconclusive or weak.
- If a direction only works for one seed, do not move it into the paper main line.
- If clean graph gain is tiny or negative, test noisy graph or strong augmentation before making robustness claims.
- If complexity increases, require stronger evidence than a simple baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.
- For this project specifically: every GB-FN run must include no-ball Top-k; every GB edge-denoise run must include pct/S2 edge-similarity baselines.
- For this project specifically: `pct/s2_degree_protect` can support a robustness claim, but not a granular-ball claim unless a GB-added variant beats it.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| DIR-01 | GB controls graph views | Active but below GRACE | View drift / over-perturbation | Yes, as negative motivation | abandon |
| DIR-02 | Replace InfoNCE with BGRL/GBT | Clearly below GRACE | Objective-view mismatch | Yes, as negative result | abandon |
| DIR-03 | GB same-ball Top-k false-negative suppression | Tiny positive/tie on citation graphs | Effect too small and no-ball nearly matches | Yes, but claim must be modest | modify |
| DIR-04 | Diagnostics for GB candidates/topology | Explains no-op and small gains | Not method direction | Yes, strongly | diagnostic-only |
| DIR-05 | GB same/cross-ball edge denoise | Weak noisy-edge signal | Membership too coarse | Tooling useful | abandon |
| DIR-06 | x-space pct/S2 edge denoise | Strong noisy robustness | Clean collateral damage; non-GB | Yes, strongest main candidate if pivoting | continue |
| DIR-07 | Risk-aware augmentation-only | Weaker than full edge weights | View-only insufficient | Yes, ablation | abandon |
| DIR-08 | CLSA weak/strong teaching | Mixed positive | Dataset inconsistency | Yes, future/appendix | modify |

# 9. Recommended Next Directions

## NEXT-DIR-1: Minimal GB-vs-no-ball false-negative proof

### Why This Direction Is Recommended
It directly tests whether GB adds anything beyond similarity Top-k.

### Previous Failure It Addresses
DIR-03's no-ball Top-k nearly matches GB-FN.

### Core Hypothesis
GB improves the timing or distribution of Top-k suppression even when candidate label precision is similar.

### GB Role
Candidate scope only: same-ball vs all-negatives.

### What Must Be Different From Failed Directions
No formula changes; isolate candidate scope.

### Minimal Implementation
Use existing configs; ensure both variants log selected similarity, candidate Jaccard, masked ratio, and seed-level deltas.

### Minimal Datasets and Seeds
Cora + CiteSeer, 5 seeds.

### Success Criteria
GB-FN beats no-ball Top-k by >= +0.003 mean test on both datasets or shows a robust mechanism metric predicting gains.

### Failure Criteria
Delta <= +0.001 or sign flip.

### Paper Claim If Successful
GB provides a mild but real structural prior beyond pure similarity.

### Reason to Stop Early
Candidate sets and metrics are nearly identical after first diagnostic run.

## NEXT-DIR-2: Noise-gated pct/S2 edge denoising

### Why This Direction Is Recommended
It addresses the strongest current method's main weakness: clean Cora regression.

### Previous Failure It Addresses
DIR-06 clean collateral damage.

### Core Hypothesis
Disable or weaken edge downweighting when noise evidence is low.

### GB Role
None unless used as an auxiliary diagnostic feature.

### What Must Be Different From Failed Directions
Do not claim GB; optimize robustness boundary.

### Minimal Implementation
Use existing `use_noise_gated_percentile` path; compare off/conservative/aggressive on clean and noisy Cora.

### Minimal Datasets and Seeds
Cora clean, Cora noise=0.3/0.4/0.5, 5 seeds.

### Success Criteria
No clean Cora regression worse than -0.003 and noisy gains remain > +0.02 at noise >= 0.4.

### Failure Criteria
Gate removes noisy gains or still regresses clean Cora.

### Paper Claim If Successful
A label-free noise-aware edge denoising prior improves robustness while limiting clean damage.

### Reason to Stop Early
Clean/noisy edge weights show no separable gate signal.

## NEXT-DIR-3: GB as auxiliary feature for edge scoring

### Why This Direction Is Recommended
It salvages GB edge information without treating same/cross-ball as a hard rule.

### Previous Failure It Addresses
DIR-05 same/cross-ball edge denoising was too coarse.

### Core Hypothesis
GB relation helps only when combined with x-space similarity and degree, not alone.

### GB Role
Auxiliary feature in a label-free or diagnostic scorer.

### What Must Be Different From Failed Directions
First pass must be diagnostic-only; no formal training before hit-rate improvement.

### Minimal Implementation
Compute edge score features: endpoint cosine, degree rank, same-ball, ball quality, boundary margin. Evaluate unsupervised score variants against injected-noise flags for diagnostics only.

### Minimal Datasets and Seeds
Cora noise=0.4 and CiteSeer noise=0.4, 1 diagnostic seed first.

### Success Criteria
GB-added score improves AUC/AP or precision/recall over pct/S2 at same budget.

### Failure Criteria
No hit-rate lift over pct/S2.

### Paper Claim If Successful
GB contributes complementary structure information to edge denoising.

### Reason to Stop Early
GB features have near-zero incremental ranking value.

## NEXT-DIR-4: CLSA repair with complete diagnostics

### Why This Direction Is Recommended
CLSA has positive settings but fails CiteSeer and lacks complete mechanism logs.

### Previous Failure It Addresses
DIR-08 mixed evidence and N/A diagnostics.

### Core Hypothesis
CLSA works only when strong-view noisy-edge precision/recall and teacher entropy are in a safe range.

### GB Role
None.

### What Must Be Different From Failed Directions
Add complete strong-view precision/recall, entropy, KL logging before new hyperparameters.

### Minimal Implementation
Rerun existing s2_clsa_aug with diagnostics fixed; only then test one repair candidate.

### Minimal Datasets and Seeds
CiteSeer noise=0.4 and Cora noise=0.4, 5 seeds if diagnostics pass.

### Success Criteria
CiteSeer becomes positive vs GRACE and not more than -0.005 vs S2 full.

### Failure Criteria
CiteSeer remains negative or diagnostics stay N/A.

### Paper Claim If Successful
Weak-view relation teaching stabilizes risk-aware augmentation under noisy graphs.

### Reason to Stop Early
Strong-view precision/recall is poor or teacher/student relation is uninformative.

# 10. Paper Claim Implications

## Claim-1

### Claim
GB provides robust local structure prior for node-level GCL.

### Supported By Which Directions
Weakly by DIR-03.

### Weakened By Which Directions
DIR-01, DIR-04, DIR-05.

### Current Evidence
GB-FN v2 is tiny positive/tie on citation graphs, but direct GB view/edge roles fail.

### Missing Evidence
Independent gain over no-ball Top-k and pct/S2.

### Required Experiments
NEXT-DIR-1 and/or NEXT-DIR-3.

### Risk
High.

### Current Readiness
weak

## Claim-2

### Claim
GB confidence reduces noisy relation damage.

### Supported By Which Directions
Limited support from DIR-03 Top-k being safer than full-soft.

### Weakened By Which Directions
DIR-05 and candidate precision diagnostics.

### Current Evidence
Top-k soft works better than broad suppression, but evidence points to narrowness more than GB confidence.

### Missing Evidence
Confidence metric that predicts correct candidates.

### Required Experiments
Candidate confidence analysis without labels in training.

### Risk
High.

### Current Readiness
not ready

## Claim-3

### Claim
GB-guided augmentation preserves local semantics.

### Supported By Which Directions
None strongly.

### Weakened By Which Directions
DIR-01 and DIR-07.

### Current Evidence
GB-GCL view generation underperforms GRACE.

### Missing Evidence
Multi-seed gains from GB view guard.

### Required Experiments
Only if new augmentation mechanism exists.

### Risk
Very high.

### Current Readiness
not ready

## Claim-4

### Claim
GB denoising improves robustness under graph noise.

### Supported By Which Directions
Not supported by current GB-specific denoising.

### Weakened By Which Directions
DIR-05.

### Current Evidence
GB same/cross edge denoise has weak precision/recall; pct/S2 robustness is non-GB.

### Missing Evidence
GB-added edge scorer beating pct/S2.

### Required Experiments
NEXT-DIR-3.

### Risk
High.

### Current Readiness
not ready

## Claim-5

### Claim
Hard same-ball positive/negative rewiring is not reliable for node-level GCL.

### Supported By Which Directions
DIR-01, DIR-03, DIR-05.

### Weakened By Which Directions
None in current project.

### Current Evidence
Hard/broad GB mechanisms are unstable or too coarse; soft/narrow is safer.

### Missing Evidence
More datasets would strengthen, but current evidence is enough as project lesson.

### Required Experiments
None before using as motivation/negative result.

### Risk
Low.

### Current Readiness
moderate

## Claim-6

### Claim
GB is more useful as an auxiliary prior than as a direct replacement for contrastive positives/views.

### Supported By Which Directions
DIR-01, DIR-02, DIR-03.

### Weakened By Which Directions
DIR-03's small effect size weakens strength, but not directionality.

### Current Evidence
GB-FN is safer than GB view/objective replacement; effect is small.

### Missing Evidence
Independent gain over similarity-only controls.

### Required Experiments
NEXT-DIR-1.

### Risk
Medium.

### Current Readiness
weak

## Claim-7

### Claim
Feature-similarity edge denoising improves noisy-graph robustness.

### Supported By Which Directions
DIR-06.

### Weakened By Which Directions
Clean Cora regression.

### Current Evidence
Noisy formal settings are consistently positive; statistical boundary supports stronger noisy settings.

### Missing Evidence
Clean/noise gate and clearer non-GB paper framing.

### Required Experiments
NEXT-DIR-2.

### Risk
Medium.

### Current Readiness
moderate

# 11. TODO List

## Missing raw logs

- Add an index of exact raw log paths for each direction and seed.
- Some summaries are enough for direction judgment, but raw seed logs were not exhaustively inspected line by line.

## Missing baseline checks

- Confirm split/evaluation parity from configs for every non-Cora expansion.
- Amazon literature protocol mismatch is documented; do not mix literature scores with current unified protocol.

## Missing seeds/std

- PubMed GB-FN in final table is formal 3-seed, not 5-seed.
- Amazon-Photo and Amazon-Computers GB-FN are 1-seed only.
- Coauthor-CS is sanity only.

## Missing datasets

- GB-FN lacks strong formal evidence beyond citation graphs.
- GB-specific edge denoise lacks multi-dataset formal evidence because quick/diagnostic screens stopped it.

## Missing ablations

- GB-FN needs a strict paired no-ball control in every future main table.
- GB edge scorer needs pct/S2 removal/addition ablations before any claim.

## Missing GB diagnostics

- Singleton ratio, radius mean/std, coverage, overlap/conflict, cross-seed stability, and confidence distribution are mostly Unknown.
- Add a script-generated table summarizing ball counts and candidate stats per dataset/seed.

## Missing visualizations

- Candidate Jaccard / selected-candidate overlap plots for GB-FN vs no-ball.
- Clean-vs-noisy edge weight distributions for pct/S2 and any GB-added scorer.

## Missing explanation for failed directions

- GB-GCL view drift has plausible interpretation but needs a more direct metric of view distribution drift.
- GB-BGRL collapse needs a non-GB BGRL baseline to isolate objective vs GB contribution.

## Missing paper-claim evidence

- Strong GB main-method claim is not ready.
- Robust edge denoising claim is more ready but not GB-specific.
- If paper remains GB-focused, the strongest safe claim is currently modest: GB is safer as a lightweight auxiliary prior, and hard/direct GB roles are risky.

# 12. Experiment 7 Linkage Note

- Added on 2026-05-19: `02_experiment_history_and_lessons 7 .md` now records `实验7noisy_label_gb_grace` as a separate direction-level history.
- Do not merge that file into this document's direction count or DIR numbering; this document remains a separate historical review.
- Relevant new conclusion: frozen GRACE + strict GB local label smoothing under noisy train labels is Cora-positive only in selected medium-label high-noise settings and is cross-dataset unstable on CiteSeer.
- Future constraint imported from Experiment 7: any noisy-label GB direction must report effective smoothing coverage and must beat CE/global/kmeans/topk_matched, not only global smoothing.
