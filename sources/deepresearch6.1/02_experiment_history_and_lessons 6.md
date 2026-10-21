# 0. Scope

本文件只覆盖当前项目目录：

`C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master`

证据只来自本目录内的代码、配置、日志、`outputs/` 和 `analysis/`。没有读取或混用其他项目目录的结果。父目录中的研究笔记或 prompt 文件不作为本文件的结果证据。本复盘不代表所有项目的最终总判断。

# 1. Current Overall Judgment

- Confirmed: 当前项目已经形成 3 个方向级实验单元：构球诊断、GB-PCL 正样本扩展、RDCL 局部关系一致性。
- Confirmed: PubMed 上 RDCL 有最强正信号，3-seed kmeans 为 F1Mi 0.8626 vs baseline 0.8596，delta +0.0030；gb_v2 为 0.8621，delta +0.0025。
- Confirmed: RDCL 机制对照削弱了“高纯度语义分组是必要原因”的解释。random / kmeans_shuffle / gb_v2_shuffle 在 PubMed 上仍然接近或高于主方法，但 protected_precision 只有约 0.357。
- Likely: 当前 RDCL 的收益更像来自 local relation regularization / group-size constraint form，而不是已经被证明来自真实语义粒球。
- Inconclusive: GB-PCL 在 Cora 单 seed 有若干正结果，但没有多 seed、多数据集和完整 raw log 支撑。
- Risk: 继续只调 alpha/tau 或构球方法，不先做机制隔离，会重复消耗算力但不能增强论文主张。
- Next priority: 做一个更干净的 size-matched / fixed-partition local regularizer 对照，明确 RDCL 的有效成分。

# 2. Direction Index

| Direction ID | Direction Name | GB Role | Versions Included | Datasets | Best Trend | Main Failure | Decision | Paper Value |
|---|---|---|---|---|---|---|---|---|
| DIR-01 | GB construction diagnostics | diagnostic grouping quality only | random, knn, kmeans, gb, gb_v2 variants, ga_gb variants | Cora, PubMed | Cora ga_gb precision up to 0.8488; PubMed kmeans_fast precision 0.7978 | high diagnostic precision did not guarantee downstream RDCL gain | diagnostic-only | diagnostic |
| DIR-02 | GB-PCL positive expansion | same-group / kNN positives added to GRACE numerator | gb, gb_v2, kmeans, knn, lambda sweeps, weighted variants | Cora | best Cora F1Mi 0.8358 on kmeans lambda=1.5 tau_w=0.2 | single seed, raw logs missing, no PubMed/DBLP confirmation | inconclusive | ablation only |
| DIR-03 | RDCL local relation consistency | auxiliary group relation loss | kmeans, gb_v2, ga_gb, random, kmeans_shuffle, gb_v2_shuffle, alpha/tau screen | Cora, CiteSeer, PubMed, DBLP | PubMed kmeans 3-seed +0.0030 over baseline | shuffled/random controls reproduce most PubMed gain | modify | main-method candidate with cautious claim |

# 3. Direction-Level Detailed Records

## DIR-01: GB construction diagnostics

### 3.1 Direction Summary

- Direction ID: DIR-01
- Direction name: GB construction diagnostics
- Time period: 2026-04-29 to 2026-05-04, based on current artifact timestamps.
- Target problem: evaluate whether different grouping methods produce semantically or structurally useful node groups before using them in GCL losses.
- Core hypothesis: higher protected_precision, coverage, NMI, and edge_ratio should indicate more reliable grouping priors.
- Baseline: no training baseline; this is diagnostic-only.
- Main related files: `preexp_gb_pcl.py`, `granular_ball/gb_utils.py`
- Result files: `outputs/preexp/Cora/*`, `outputs/preexp/PubMed/*`
- Raw logs: Unknown. Current directory contains result JSON/CSV, but no matching raw console logs were found.
- Current decision: diagnostic-only

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: before adding GB constraints, the project needed to know whether same-ball membership is a plausible semantic relation.
- Why GB might help: compact groups may protect same-label or graph-local nodes.
- Expected improvement: use grouping diagnostics to choose safer GB or kmeans variants for downstream loss.
- Expected paper claim: GB construction can reveal or provide robust local structure priors.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| random | fixed random groups | Cora, PubMed | low precision: Cora 0.1769, PubMed 0.3569 at p=10 | random baseline | random grouping is weak by diagnostics |
| knn | kNN positives | Cora, PubMed | high precision: Cora 0.8193, PubMed 0.7996 at k=5 | local nearest-neighbor relation | kNN is semantically cleaner but creates many tiny groups |
| kmeans | embedding clusters | Cora, PubMed | Cora 0.7844, PubMed 0.7827 at p=10 | cluster-based grouping | stable diagnostic baseline |
| gb | adaptive granular balls | Cora, PubMed | Cora 0.7075, PubMed 0.7598 at p=10 | original GB split rule | bounded-size groups but usually weaker than kmeans |
| gb_v2 variants | 2-means and stop variants | Cora, PubMed | 2means_only close to kmeans; old gb_v2 sometimes collapsed | GB construction refinement | implementation details can dominate diagnostics |
| ga_gb variants | graph-aware GB refinement | Cora, PubMed | Cora beta=0.3 precision 0.8488; PubMed fast GA-GB below kmeans_fast | topology-aware refinement | promising on Cora diagnostics, not clearly superior on PubMed |

### 3.4 Role of Granular Ball

- GB role: diagnostic grouping object.
- Where GB is generated: pretrained embeddings from `preexp_gb_pcl.py`.
- Representation space: model embedding after GRACE-style pretraining.
- Affects loss: no.
- Affects augmentation: no.
- Affects graph structure: no.
- Affects sampling: no.
- Affects evaluation: only diagnostic metrics.

### 3.5 Result Summary

- Best diagnostic signal: Cora `ga_gb p=10,beta=0.3` protected_precision 0.8488, NMI 0.4297.
- PubMed useful diagnostic: `kmeans_fast p=10,n_init=1` protected_precision 0.7978, better than `kmeans_reference` 0.7827 in the saved comparison.
- GB caution: old `gb_v2` collapsed on Cora to 1 group with max_group_size 2708 and precision 0.1793; on PubMed it collapsed to 10 large groups with max_group_size 4066 and precision 0.5260.
- Raw evidence: `outputs/preexp/Cora/preexp_compare_metrics.csv`, `outputs/preexp/PubMed/preexp_compare_metrics.csv`, `outputs/preexp/PubMed/pubmed_ga_gb_v2_compare.csv`.

### 3.6 What Is Directly Supported

- kNN and kmeans are strong diagnostic baselines in this project.
- Some GB variants can collapse into overly large groups if stopping rules are not controlled.
- GA-GB improves Cora diagnostic precision, but PubMed GA-GB fast variants are not clearly better than kmeans_fast.

### 3.7 Reasonable Interpretation

- Interpretation: grouping diagnostics are necessary but insufficient. Later RDCL results show downstream F1 is not determined by protected_precision alone.
- Interpretation: bounded max group size is a useful engineering constraint because collapsed groups destroy the intended local prior.

### 3.8 What Cannot Be Concluded

- Cannot conclude that higher protected_precision will improve GCL downstream metrics.
- Cannot conclude GA-GB is a better training method from diagnostics alone.
- Cannot conclude GB construction is stable across training epochs; current diagnostic files are mostly snapshot-level.

### 3.9 Failure Analysis

- Main failure reason: diagnostics alone did not predict downstream value.
- Secondary failure reason: some GB variants collapse to large groups.
- Evidence: `outputs/preexp/*/preexp_compare_metrics.csv` and RDCL mechanism controls in `analysis/pubmed_mechanism_controls.csv`.
- Minimum check needed: pair diagnostic metrics with downstream controls where shuffled groups preserve group sizes.

### 3.10 Lessons Learned

- Lesson: keep random, kNN, and kmeans diagnostics as baselines.
- Evidence: they anchor precision and edge_ratio ranges in `outputs/preexp/*`.
- Future constraint: every new GB method should first beat random and be compared to kmeans/kNN diagnostics.

- Lesson: max_group_size must be monitored.
- Evidence: old `gb_v2` collapsed to very large groups.
- Future constraint: reject grouping variants that create global or near-global groups unless the method explicitly needs them.

### 3.11 What Is Still Useful

- `preexp_gb_pcl.py` and `outputs/preexp/*` provide reusable diagnostics for method selection.
- GA-GB remains useful as a diagnostic idea, especially on Cora.

### 3.12 Decision

- Decision: diagnostic-only
- Reason: useful for explaining and filtering methods, but not a training contribution by itself.
- What should not be tried again: using protected_precision alone as success criterion.
- Variant still worth trying: group stability across epochs and seeds.
- Evidence required to revive as paper claim: diagnostic improvements must correlate with downstream gains after shuffled controls.

## DIR-02: GB-PCL positive expansion

### 3.1 Direction Summary

- Direction ID: DIR-02
- Direction name: GB-PCL positive expansion
- Time period: 2026-04-29 to 2026-05-01, based on current artifact timestamps.
- Target problem: extend GRACE positives beyond node identity using GB/kmeans/kNN same-group candidates.
- Core hypothesis: same-group or nearest-neighbor positives can enrich the contrastive numerator and improve node representations.
- Baseline: Cora GRACE baseline seed 39788 from later multiseed log is F1Mi 0.8317, but exact original GB-PCL baseline raw log is Unknown.
- Main related files: `train.py`, `losses/gb_pcl_loss.py`, `granular_ball/gb_utils.py`
- Result files: `outputs/gb_pcl/Cora/*/final_eval.json`, `outputs/gb_pcl/Cora/*/config.json`, `outputs/gb_pcl/Cora/*/group_stats.json`
- Raw logs: Unknown. Current directory contains GB-PCL JSON/CSV outputs but no separate GB-PCL raw console logs.
- Current decision: inconclusive

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: identity-only positives may be too sparse for node-level learning.
- Why GB might help: nodes inside the same granular ball may be safe additional positives.
- Expected improvement: higher F1 by adding same-group positives with a controlled `gb_lambda`.
- Expected paper claim: GB expands positive pairs safely.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| GB-PCL gb | original GB groups | Cora seed 39788 | best visible around F1Mi 0.8325 | same-ball positives | small or unstable gain |
| GB-PCL gb_v2 | adaptive 2-means GB | Cora seed 39788 | best visible F1Mi 0.8343 at lambda=1.0 | improved bounded GB positives | better than gb, still single-seed |
| GB-PCL kmeans | kmeans groups | Cora seed 39788 | best F1Mi 0.8358 at lambda=1.5, tau_w=0.2 | cluster positives | best saved GB-PCL result |
| GB-PCL knn | kNN positives | Cora seed 39788 | best F1Mi 0.8355 at lambda=1.5, top_m=10 | local neighbor positives | competitive with kmeans |
| weighted variants | `gb_tau_w=0.2` weighting | Cora seed 39788 | mixed | confidence-weighted positives | not enough evidence for broad claim |

### 3.4 Role of Granular Ball

- GB role: positive-pair construction.
- Where GB is generated: periodically in training from current embeddings.
- Representation space: model embedding CPU copy.
- Affects loss: yes, through `gb_pcl_loss`.
- Affects augmentation: no.
- Affects graph structure: no.
- Affects sampling: positive candidate set only.
- Affects evaluation: no.

### 3.5 Result Summary

- Best saved run: `outputs/gb_pcl/Cora/20260501_203926`, method kmeans, lambda=1.5, top_m=5, tau_w=0.2, F1Mi 0.8358, F1Ma 0.8233.
- Competitive saved run: `outputs/gb_pcl/Cora/20260430_101854`, method knn, lambda=1.5, top_m=10, F1Mi 0.8355, F1Ma 0.8247.
- Reference baseline issue: comparable Cora seed39788 baseline from `logs_cora_multiseed/cora_baseline_seed39788.log` is F1Mi 0.8317, but this baseline was produced later and may not exactly match the original GB-PCL experiment batch.
- Mean/std: only evaluation std is stored in final_eval; training seed std is not available for GB-PCL.
- Raw evidence: `outputs/gb_pcl/Cora/*`.

### 3.6 What Is Directly Supported

- On Cora single seed, kmeans and kNN positive expansion can exceed the available Cora baseline reference.
- GB-PCL has saved final_eval outputs and group diagnostics.
- There is no current-directory evidence that GB-PCL generalizes beyond Cora.

### 3.7 Reasonable Interpretation

- Interpretation: positive expansion can help on Cora, but the gain may be dataset-specific or seed-specific.
- Interpretation: kNN and kmeans are safer than original GB for positive expansion in the current saved runs.

### 3.8 What Cannot Be Concluded

- Cannot claim GB-PCL is stable without multi-seed runs.
- Cannot claim it works on PubMed, DBLP, or CiteSeer because current GB-PCL outputs are Cora-only.
- Cannot claim original GB is better than kmeans/kNN.

### 3.9 Failure Analysis

- Main failure reason: evidence is too narrow for a paper main line.
- Secondary failure reason: raw logs are missing, and the exact baseline comparability is partial.
- Evidence: `outputs/gb_pcl/Cora/*` and absence of matching `logs_gb_pcl/` in current directory.
- Minimum check needed: rerun Cora 5 seeds and PubMed 3 seeds only if GB-PCL becomes relevant again.

### 3.10 Lessons Learned

- Lesson: single-seed Cora improvements are not enough.
- Evidence: GB-PCL best Cora run is positive but lacks multi-seed confirmation.
- Future constraint: do not promote GB-PCL without at least Cora 5 seeds and one non-Cora dataset.

- Lesson: kmeans/kNN should be retained as strong non-GB controls.
- Evidence: best GB-PCL runs are kmeans/kNN, not original GB.
- Future constraint: any GB positive method must beat kmeans/kNN controls.

### 3.11 What Is Still Useful

- `gb_pcl_loss.py` is a reusable implementation of weighted multi-positive contrastive loss.
- Saved Cora runs are useful as ablation evidence and negative/partial result memory.

### 3.12 Decision

- Decision: inconclusive
- Reason: positive Cora signal exists, but evidence lacks seeds, datasets, and raw logs.
- What should not be tried again: broad GB-PCL sweeps on Cora single seed.
- Variant still worth trying: only a minimal multi-seed check if the paper needs a positive-expansion ablation.
- Evidence required to revive this direction: kmeans/kNN/GB-PCL must show stable gain over comparable baseline across seeds and at least one larger dataset.

## DIR-03: RDCL local relation consistency

### 3.1 Direction Summary

- Direction ID: DIR-03
- Direction name: RDCL local relation consistency
- Time period: 2026-05-02 to 2026-05-05, based on current artifact timestamps.
- Target problem: add a weak local relation consistency loss to GRACE without replacing the base contrastive objective.
- Core hypothesis: local groups can preserve relation distributions across two augmented views.
- Baseline: GRACE baseline from `train.py` with dataset-specific `config.yaml`.
- Main related files: `train.py`, `losses/rdcl_loss.py`, `granular_ball/gb_utils.py`
- Result files: `analysis/rdcl_experiment_summary.md`, `analysis/rdcl_experiment_summary.csv`, `analysis/pubmed_ablation_screen.csv`, `analysis/pubmed_mechanism_controls.csv`
- Raw logs: `logs_cora_multiseed/`, `logs_pubmed_multiseed/`, `logs_pubmed_ablation/`, `logs_pubmed_mechanism/`, `logs_dblp_rdcl/`, plus root PubMed logs.
- Current decision: modify

### 3.2 Why This Direction Was Tried

- GRACE/GCL weakness targeted: instance-level contrast may not enforce stable local relational structure.
- Why GB might help: GB/kmeans groups can define local subsets over which intra-group relation distributions are matched.
- Expected improvement: a small auxiliary relation loss should improve F1 without collapsing the base objective.
- Expected paper claim: local relation consistency improves graph contrastive learning; GB/kmeans provide candidate grouping priors.

### 3.3 Versions / Attempts Included

| Version | Main Change | Dataset | Result Trend | What It Tested | What It Showed |
|---|---|---|---|---|---|
| RDCL-kmeans main | kmeans, alpha=0.5, tau=0.2 | PubMed 3 seeds | F1Mi 0.8626 vs baseline 0.8596, +0.0030 | main RDCL signal | strongest current positive |
| RDCL-gb_v2 main | bounded GB v2 groups | PubMed 3 seeds | F1Mi 0.8621, +0.0025 | GB replacement for kmeans | close to kmeans but not better |
| Cora multi-seed | kmeans and gb_v2 | Cora 5 seeds | kmeans +0.0013, gb_v2 +0.0022 | stability on small graph | weak positive |
| DBLP single-seed | kmeans and gb_v2 | DBLP | kmeans -0.0016, gb_v2 -0.0010 | transfer to DBLP | negative single-seed |
| CiteSeer outputs | kmeans alpha/tau variants | CiteSeer seed 38108 | F1Mi 0.7195 to 0.7213 | trend screen | baseline not found in current directory, delta Unknown |
| ga_gb PubMed | graph-aware groups beta=0.3 | PubMed seed 23344 | F1Mi 0.8606 | graph-aware GB training value | lower than kmeans and gb_v2 main |
| PubMed ablation | alpha/tau screen | PubMed seed 23344 | best kmeans alpha=0.5 tau=0.1 F1Mi 0.8633 | local hyperparameter optimum | no +0.0010 trigger over main |
| mechanism controls | random, kmeans_shuffle, gb_v2_shuffle | PubMed 3 seeds | random +0.0014, kmeans_shuffle +0.0031, gb_v2_shuffle +0.0029 | remove true semantic grouping | true grouping not proven necessary |

### 3.4 Role of Granular Ball

- GB role: group definition for relation distribution consistency loss.
- Where GB is generated: periodic RDCL group updates from current embeddings.
- Representation space: model embedding CPU copy.
- Affects loss: yes, through `rdcl_loss`.
- Affects augmentation: no.
- Affects graph structure: no.
- Affects sampling: groups define local relation distributions.
- Affects evaluation: no.

### 3.5 Result Summary

- PubMed baseline 3 seeds: F1Mi_mean 0.8596, F1Mi_seed_std 0.0018.
- PubMed RDCL-kmeans: F1Mi_mean 0.8626, seed_std 0.0016, delta +0.0030, mean_precision 0.7995.
- PubMed RDCL-gb_v2: F1Mi_mean 0.8621, seed_std 0.0009, delta +0.0025, mean_precision 0.7874.
- Cora RDCL-kmeans: F1Mi_mean 0.8299 vs baseline 0.8287, delta +0.0013.
- Cora RDCL-gb_v2: F1Mi_mean 0.8309 vs baseline 0.8287, delta +0.0022.
- DBLP RDCL-kmeans: F1Mi 0.8425 vs baseline 0.8441, delta -0.0016.
- DBLP RDCL-gb_v2: F1Mi 0.8431 vs baseline 0.8441, delta -0.0010.
- PubMed mechanism controls: kmeans_shuffle F1Mi_mean 0.8627, gb_v2_shuffle 0.8626, random 0.8611, all with precision around 0.357.
- Raw evidence: `analysis/rdcl_experiment_summary.md`, `analysis/pubmed_mechanism_controls.csv`.

### 3.6 What Is Directly Supported

- RDCL has a clear PubMed performance signal under current 3-seed setup.
- gb_v2 is close to kmeans on PubMed and Cora.
- DBLP is not supportive in the current single-seed run.
- Random and shuffled controls reproduce much of the PubMed improvement.

### 3.7 Reasonable Interpretation

- Interpretation: RDCL is useful as local relation regularization.
- Interpretation: group-size/local constraint form may be more important than semantic purity on PubMed.
- Interpretation: meaningful grouping may still matter under noise, harder datasets, or more controlled settings, but current evidence does not prove it.

### 3.8 What Cannot Be Concluded

- Cannot claim semantic/pure GB grouping is necessary for PubMed gains.
- Cannot claim gb_v2 is superior to kmeans.
- Cannot claim CiteSeer effect because no comparable baseline was found in current directory.
- Cannot claim DBLP supports RDCL without further evidence.

### 3.9 Failure Analysis

- Main failure reason: mechanism controls undermine the original semantic grouping explanation.
- Secondary failure reason: dataset transfer is mixed.
- Evidence: `analysis/pubmed_mechanism_controls.csv`, `analysis/rdcl_experiment_summary.md`.
- Alternative explanation: PubMed gains may come from regularizing relation distributions over size-matched local subsets, regardless of semantic grouping.
- Minimum check needed: fixed-partition or size-matched local regularizer baseline.

### 3.10 Lessons Learned

- Lesson: shuffled controls are mandatory for any GB-GCL mechanism claim.
- Evidence: PubMed kmeans_shuffle and gb_v2_shuffle match main methods.
- Future constraint: do not write paper claims around semantic grouping unless main groups beat shuffled controls.

- Lesson: protected_precision does not guarantee F1 gain.
- Evidence: main kmeans/gb_v2 precision near 0.79, shuffled controls near 0.357, yet F1 remains similar.
- Future constraint: use precision as diagnostic, not as success proxy.

- Lesson: Cora weak positives should not drive the paper story.
- Evidence: Cora deltas are +0.0013 and +0.0022 with seed std around 0.0075 to 0.0079.
- Future constraint: PubMed remains the strongest evidence anchor.

### 3.11 What Is Still Useful

- RDCL implementation, logging, and analysis tables are useful.
- gb_v2 remains useful as a bounded-size alternative to kmeans.
- Mechanism controls are valuable paper ablations.

### 3.12 Decision

- Decision: modify
- Reason: performance signal exists, but mechanism claim must become more cautious.
- What should not be tried again: more PubMed alpha/tau sweeps before isolating mechanism.
- Variant still worth trying: size-matched fixed partition / no-embedding local relation regularizer.
- Evidence required to revive stronger GB claim: true kmeans/gb_v2 grouping must beat random/shuffled controls across seeds or in robustness/noise settings.

# 4. Cross-Direction Failed Pattern Ledger

## FP-1: Diagnostic purity does not imply downstream gain

### Status
confirmed

### Related Directions
DIR-01, DIR-03

### Observation
High protected_precision groups do not necessarily produce better RDCL F1; low-precision shuffled groups remain competitive on PubMed.

### Possible Reason
The loss form and group-size distribution may regularize representations even without semantic purity.

### Evidence
`outputs/preexp/*/preexp_compare_metrics.csv`, `analysis/pubmed_mechanism_controls.csv`

### Future Constraint
Every diagnostic claim must be paired with downstream control experiments.

### When to Revisit
After a dataset or noise setting shows true grouping beating shuffled controls.

## FP-2: Single-seed Cora improvement is not enough

### Status
confirmed

### Related Directions
DIR-02, DIR-03

### Observation
GB-PCL Cora single-seed results look positive, but Cora RDCL multi-seed is only weak positive.

### Possible Reason
Cora has high variance and small absolute deltas.

### Evidence
`outputs/gb_pcl/Cora/*`, `analysis/rdcl_experiment_summary.md`

### Future Constraint
No Cora-only single-seed result should enter the main paper table.

### When to Revisit
Only when a method has PubMed or another larger dataset support.

## FP-3: More alpha/tau tuning has low marginal value

### Status
likely

### Related Directions
DIR-03

### Observation
PubMed ablation screen found best single-seed kmeans alpha=0.5 tau=0.1, but only +0.0007 over the main config, below expansion threshold.

### Possible Reason
Current gains are not bottlenecked by small alpha/tau changes.

### Evidence
`analysis/pubmed_ablation_screen.csv`

### Future Constraint
Do not continue parameter sweeps before answering mechanism questions.

### When to Revisit
After a new mechanism or dataset changes the objective.

## FP-4: GB construction can collapse without constraints

### Status
confirmed

### Related Directions
DIR-01

### Observation
Old `gb_v2` diagnostic outputs created very large groups on Cora and PubMed.

### Possible Reason
Stopping criterion allowed over-merged partitions.

### Evidence
`outputs/preexp/Cora/preexp_compare_metrics.csv`, `outputs/preexp/PubMed/preexp_compare_metrics.csv`

### Future Constraint
Always report max_group_size and reject global-like groups.

### When to Revisit
Only with bounded split/stop rules and max size diagnostics.

# 5. Cross-Direction Positive Signal Ledger

## PS-1: RDCL local relation regularization

### Status
confirmed on PubMed, weak on Cora

### Related Directions
DIR-03

### Observation
PubMed RDCL-kmeans and RDCL-gb_v2 improve over baseline across 3 seeds.

### Why It May Matter
It is the strongest current project-level method signal.

### Required Verification
Mechanism-isolating controls that separate loss form, group size, and semantic grouping.

### Possible Paper Use
main method candidate with cautious mechanism statement.

## PS-2: Bounded gb_v2 as a practical alternative to kmeans

### Status
possible

### Related Directions
DIR-01, DIR-03

### Observation
gb_v2 is close to kmeans on PubMed and slightly higher than kmeans on Cora aggregate, while keeping max group size much smaller.

### Why It May Matter
It may be easier to present as a granular-ball construction if mechanism controls are handled honestly.

### Required Verification
gb_v2 must beat shuffled or random controls in at least one meaningful setting.

### Possible Paper Use
ablation or implementation alternative.

## PS-3: Positive expansion can work on Cora

### Status
possible

### Related Directions
DIR-02

### Observation
GB-PCL kmeans and kNN variants reach Cora F1Mi around 0.8355 to 0.8358 in saved single-seed outputs.

### Why It May Matter
It provides an auxiliary ablation direction if the project needs a comparison between positive expansion and relation consistency.

### Required Verification
multi-seed and non-Cora validation.

### Possible Paper Use
appendix or ablation only.

# 6. Anti-Repetition Memory

## Do Not Retry Without New Evidence

| Pattern | Why It Failed | Evidence | Required New Evidence Before Retry |
|---|---|---|---|
| Claiming high-purity GB groups cause RDCL gains | shuffled/random controls match most PubMed gain | `analysis/pubmed_mechanism_controls.csv` | true groups beat shuffled controls across seeds/datasets |
| Broad GB-PCL sweeps on Cora single seed | result cannot establish stability | `outputs/gb_pcl/Cora/*` | Cora 5 seeds plus one non-Cora dataset |
| More PubMed alpha/tau sweeps | previous screen did not produce a materially stronger config | `analysis/pubmed_ablation_screen.csv` | a new mechanism changes the objective or success threshold |
| Treating DBLP as supportive | current single-seed DBLP regressed | `analysis/rdcl_experiment_summary.md` | DBLP multi-seed turns positive |
| Using collapsed GB variants | old gb_v2 created very large groups | `outputs/preexp/*/preexp_compare_metrics.csv` | bounded max group size and stable precision |

## Failed Form, Not Necessarily Failed Idea

| Failed Form | Underlying Signal That May Remain Useful | Safer Future Form |
|---|---|---|
| semantic-group-only explanation | weak local relation regularization | compare main groups against random/shuffled/fixed partitions |
| GB-PCL as main method from Cora only | positive expansion can help under some conditions | keep as ablation after multi-seed validation |
| protected_precision as success proxy | diagnostics still explain group quality | use precision with downstream controls |
| old unbounded gb_v2 | bounded GB construction remains useful | enforce max group size and monitor collapse |

## Minimum Bar Before a Direction Enters Paper Main Line

- Comparable baseline: same dataset, seed list, batch size, epochs, encoder, and evaluation.
- Multi-seed evidence: at least 3 seeds for PubMed-level claims; 5 seeds for Cora if used.
- Multi-dataset evidence: PubMed plus at least one additional non-negative dataset.
- Ablation removing GB signal: random, shuffled, or fixed-partition controls.
- Mechanism diagnostic: protected_precision, num_groups, max_group_size, edge_ratio.
- No label leakage: labels may be used only for diagnostics, not pretraining group construction.

# 7. Direction-Level Lessons for Future Experiments

- Do not judge success only from Cora.
- Do not change baseline evaluation when testing a new GB-GCL direction.
- Do not use labels or supervised purity in unsupervised pretraining.
- Do not treat hard same-ball membership as reliable semantic positive without confidence or soft weighting.
- Every new direction must define the exact GB role before coding.
- Every direction must include at least one mechanism diagnostic.
- Every direction must include at least one ablation removing or corrupting the GB signal.
- If improvement is smaller than overlapping std, mark it inconclusive.
- If a direction only works for one seed, do not move it into the paper main line.
- If complexity increases, require stronger evidence than a simple baseline.
- If a direction fails, extract the smaller reusable signal before abandoning it.

# 8. Current Multi-Direction Summary

| Direction | Main Idea | Overall Result | Most Likely Failure Reason | Still Useful? | Decision |
|---|---|---|---|---|---|
| GB construction diagnostics | compare grouping quality before training | useful diagnostics; not predictive enough alone | diagnostic metrics do not isolate training mechanism | yes | diagnostic-only |
| GB-PCL positive expansion | add same-group/kNN positives to contrastive numerator | Cora single-seed positive but weak evidence | no multi-seed or non-Cora confirmation | partly | inconclusive |
| RDCL local relation consistency | add auxiliary relation distribution loss | PubMed positive, Cora weak, DBLP negative | mechanism not specific to semantic grouping | yes | modify |

# 9. Recommended Next Directions

## NEXT-DIR-1: Size-matched fixed local regularizer baseline

### Why This Direction Is Recommended

It directly tests whether RDCL gains come from group size/local relation loss form rather than semantic grouping.

### Previous Failure It Addresses

It addresses the random/shuffled control weakness in DIR-03.

### Core Hypothesis

If a fixed or size-matched random partition matches RDCL, then the main contribution is regularization form, not GB semantics.

### GB Role

GB is a candidate grouping prior, not assumed necessary.

### What Must Be Different From Failed Directions

Do not change alpha/tau first; isolate group semantics first.

### Minimal Implementation

Add or reuse a fixed partition / deterministic size-matched grouping method, keeping `rdcl_alpha=0.5`, `rdcl_tau=0.2`, `p=10`, `batch_size=2048`.

### Minimal Datasets and Seeds

PubMed seeds 23344, 1, 2.

### Success Criteria

True kmeans or gb_v2 beats fixed/random/shuffled controls by at least +0.0015 mean F1Mi.

### Failure Criteria

Controls remain equal or better.

### Paper Claim If Successful

Meaningful local grouping contributes beyond generic relation regularization.

### Reason to Stop Early

If fixed/random controls match main methods on PubMed.

## NEXT-DIR-2: Robustness/noise setting for RDCL

### Why This Direction Is Recommended

Clean PubMed does not prove semantic grouping value; noise may expose the benefit of stable groups.

### Previous Failure It Addresses

It tests whether true grouping matters when graph views are less reliable.

### Core Hypothesis

GB/kmeans groups may help more under edge noise or stronger augmentation.

### GB Role

Auxiliary local relation prior.

### What Must Be Different From Failed Directions

Include shuffled controls from the start.

### Minimal Implementation

Use existing RDCL path, add a controlled noisy-edge or stronger-drop setting if already supported; otherwise keep this as TODO.

### Minimal Datasets and Seeds

PubMed 3 seeds first.

### Success Criteria

True grouping beats shuffled controls and baseline under noise.

### Failure Criteria

All controls move together.

### Paper Claim If Successful

GB/RDCL improves robustness rather than clean-graph accuracy only.

### Reason to Stop Early

If clean-setting mechanism remains unexplained and noise support is absent.

## NEXT-DIR-3: Minimal GB-PCL verification only if needed

### Why This Direction Is Recommended

GB-PCL has saved positive Cora single-seed evidence but is not currently paper-ready.

### Previous Failure It Addresses

It checks whether the positive-expansion direction is real or just Cora noise.

### Core Hypothesis

Positive expansion may help, but only if kmeans/kNN controls remain stable over seeds.

### GB Role

Positive-pair construction.

### What Must Be Different From Failed Directions

Run seeds instead of parameter sweeps.

### Minimal Implementation

Pick one best kmeans/knn config and one gb_v2 config from saved outputs; compare with matched baseline.

### Minimal Datasets and Seeds

Cora 5 seeds, PubMed 3 seeds only if Cora is clearly positive.

### Success Criteria

Mean F1Mi gain at least +0.003 on Cora and non-negative on PubMed.

### Failure Criteria

Cora gain disappears or PubMed regresses.

### Paper Claim If Successful

Positive expansion is a secondary ablation, not the main method.

### Reason to Stop Early

If Cora 5-seed gain is weak or negative.

# 10. Paper Claim Implications

## Claim-1

### Claim

RDCL provides a useful local relation regularizer for node-level GCL.

### Supported By Which Directions

DIR-03 PubMed 3-seed results.

### Weakened By Which Directions

DIR-03 DBLP single-seed negative result and mechanism controls.

### Current Evidence

PubMed kmeans +0.0030 and gb_v2 +0.0025 over baseline.

### Missing Evidence

Mechanism-specific gap over fixed/random/shuffled controls.

### Required Experiments

Size-matched fixed local regularizer baseline.

### Risk

Moderate. Performance signal is real on PubMed, but mechanism statement is not yet strong.

### Current Readiness

moderate

## Claim-2

### Claim

Semantic or high-purity GB/kmeans grouping is necessary for RDCL gains.

### Supported By Which Directions

Not supported by current project evidence.

### Weakened By Which Directions

DIR-03 mechanism controls.

### Current Evidence

kmeans_shuffle and gb_v2_shuffle match main methods while precision is around 0.357.

### Missing Evidence

True groups must beat shuffled controls across seeds or conditions.

### Required Experiments

Fixed/random/shuffled controls on more datasets or noise settings.

### Risk

High.

### Current Readiness

not ready

## Claim-3

### Claim

GB-PCL positive expansion improves GRACE.

### Supported By Which Directions

DIR-02 Cora single-seed saved outputs.

### Weakened By Which Directions

Lack of raw logs, seeds, and non-Cora validation.

### Current Evidence

Best Cora saved F1Mi is 0.8358, but comparable baseline and seed stability are partial.

### Missing Evidence

Multi-seed and non-Cora confirmation.

### Required Experiments

Minimal seed verification only if GB-PCL becomes relevant.

### Risk

High.

### Current Readiness

weak

# 11. TODO List

## Missing raw logs

- GB-PCL raw console logs were not found in current directory.
- Preexp raw console logs were not found in current directory.

## Missing baseline checks

- CiteSeer RDCL outputs exist, but no comparable baseline log was found in current directory.
- GB-PCL exact original baseline log is Unknown.

## Missing seeds/std

- GB-PCL has evaluation std, not training seed std.
- DBLP RDCL is single-seed only.
- CiteSeer RDCL is single-seed and lacks baseline.

## Missing datasets

- GB-PCL lacks PubMed/DBLP/CiteSeer training evidence in current outputs.
- RDCL needs one more supportive non-PubMed dataset for stronger generality.

## Missing ablations

- RDCL needs fixed-partition local regularizer control.
- GB-PCL needs kmeans/kNN/GB controls over matched seeds if revived.

## Missing GB diagnostics

- Cross-epoch group stability is not summarized.
- Cross-seed group stability is not summarized.

## Missing visualizations

- Optional: precision vs F1 plot for RDCL and GB-PCL.
- Optional: group size distribution plots for kmeans, gb_v2, shuffled controls.

## Missing explanation for failed directions

- DBLP negative RDCL result needs analysis only if DBLP becomes paper-critical.
- CiteSeer cannot be judged without baseline.

## Missing paper-claim evidence

- Need evidence that meaningful GB/kmeans grouping beats random/shuffled/fixed controls.

# 12. Experiment 7 Linkage Note

- Added on 2026-05-19: `02_experiment_history_and_lessons 7 .md` now records `实验7noisy_label_gb_grace` as a separate direction-level history.
- Do not merge that file into this document's direction count or DIR numbering; this document remains a separate historical review.
- Relevant new conclusion: frozen GRACE + strict GB local label smoothing under noisy train labels is Cora-positive only in selected medium-label high-noise settings and is cross-dataset unstable on CiteSeer.
- Future constraint imported from Experiment 7: any noisy-label GB direction must report effective smoothing coverage and must beat CE/global/kmeans/topk_matched, not only global smoothing.
