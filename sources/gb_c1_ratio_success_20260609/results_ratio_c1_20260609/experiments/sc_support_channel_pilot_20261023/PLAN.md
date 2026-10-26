# SC support selection and representation/prediction transfer

Status: designed, not run. Authorized on 2026-10-23 in the current conversation.
Earliest execution: 2026-10-23 02:09:00 Asia/Shanghai (2026-10-22 18:09:00 UTC).
The user requested design now and experiment startup about two hours later.

## Scope and hypotheses

- E1: Can noisy-training-label agreement select a useful voting support without clean labels for that selection?
- E2: Is R2 alignment budget limiting performance? Lower alignment loss alone is not evidence of better classification or unseen-node predictability.
- E4: Does label-free representation initialization improve fixed-teacher prediction transfer beyond either channel and beyond additional training updates?
- E3 and E5 are conditional later phases, not an unconditional overnight search.
- Balls organize shared training-label counts. No new GCE fusion, noise head, supervised geometry, or dynamic representation update is added to E1.

Keep old experiments, manuscript, data, and remote repositories unchanged. New code, configs, outputs and failures live under this experiment directory. All prior negative results remain applicable until new evidence says otherwise.

## Sources and fixed protocol

Read the workspace literature/history entry files and research/privileged_graph_label_noise_20261022/REPORT.md before implementation.
Reuse the verified environment D:/codex_temp/noisygl-rocm721/Scripts/python.exe. GPU training requires torch.cuda.is_available() in this ROCm environment; CPU fallback is not permitted for model fits.

Source implementations and assets:

- experiments/neighbor_preserving_ball_20261022/generator.py: deterministic geometry tree, cut, LCSS target construction.
- experiments/neighbor_support_ablation_20261022/kernels.py: support-only dynamic-programming costs.
- experiments/neighbor_support_ablation_20261022/run.py: three-dataset paired R0/R2 controls.
- experiments/r2_support_5dataset_20261022/run.py: five-dataset raw data, H, splits, R2 alignments.
- experiments/teacher_support_lcss_20261022: ordinary two-layer GCN teachers, NOT ERASE.
- experiments/teacher_support_distillation_20261022: all-node soft cross-entropy KD, temperature 1; NOT the correlation loss from the paper named DIST.
- GB-LCSS_code/src/gblcss/data.py: independent per-node transition sampling, train-only corruption.

Datasets: cora, citeseer, corafull, coauthorcs, amazonpho.
Conditions: clean, uniform0.4, uniform0.6, pair0.4. Protocol seeds 0,1,2.
Keep exact historical 20/10/70 splits, noise realization, normalized frozen GRACE H, attribute preprocessing, optimizer and first-best clean-validation checkpoint selection. These are repeated protocol/student seeds, not independent GRACE pretraining replicates.
No test scoring during training or hyperparameter selection. Freeze selections and completed predictions before one phase-end test analysis. A test-selected oracle is labeled an optimistic retrospective reference, never a deployed selector or a parameter source.
beta=.5 and min_votes=2 stay fixed in these diagnostic phases for historical comparability. beta=.5 has development/test-selection history; these runs are not independent confirmation of its choice.

## E1: fixed-candidate agreement selector

### Mathematical scope

For each fixed candidate s, exclude the query's own label from its donor histogram. Let h_i(s) be uniform on the maximum-count classes (a one-hot vector for a unique winner). Score SC(s)=mean_i h_i(s)[z_i]. This integrates randomized tie breaking exactly and avoids class-index or observed-label-favoring ties.
Under independent symmetric corruption, E[SC(s)] = eta/(C-1) + gamma E[mean_i h_i(s)[y_i]], gamma=1-C*eta/(C-1)>0. Both expectations include the other nodes' noise. This is an expected ranking identity, not a guarantee that one empirical maximizer is correct. Shared donors induce dependent scores.
Global selection s_hat from all z can depend on z_i, so the final adaptive target does not automatically inherit fixed-partition leave-one-out independence. Do not reuse the original concentration/improvement theorem for the selected algorithm. Cross-fitted selection is a separate future method, not silently added here.
Pair noise is outside this affine identity. Do not apply a scalar symmetric-noise gamma to it. Training-label selection does not eliminate clean-validation dependence of checkpoint selection.

### Candidate construction, controls, and scoring

Use one label-value-free tree per dataset/seed for every s in {4,8,16,32,64,128}, shared across all four conditions. Freeze min_nodes=8, min_train=3, max_leaves=4096, snapshot_leaves=64. Splits use H and training positions only. The deeper family is a declared change from historical min_nodes=32/max_leaves=256, not a hidden rescue.
Record actual support min, quartiles, mean, max; fraction within [s/2,2s]; reached maximum leaf count; assignment hashes; and duplicate cuts. Deduplicate identical assignments for fitting while retaining all nominal s rows.
Keep BOTH the historical SUPPORT32 partition and fixed s=32 on the new tree. Only SC versus new-tree fixed32 identifies selection improvement; comparison with historical SUPPORT includes tree-family changes.
All training nodes must be scored on the same set. min_train=3 ensures at least two leave-one-out donors. If this fails, stop for a correctness issue; do not drop difficult nodes or fall back to z_i in the SC score.
SC ties within numerical tolerance 1e-12 choose the largest nominal s. Freeze selections before reading clean training labels for diagnostics. Clean train labels may be used to recreate the explicitly specified corruption, but cannot enter scoring, geometry, tie rules or targets except in the clean condition.

### E1a: offline first, no model fits

All five datasets, seeds 0-2, four conditions, six candidates: 360 score/target cells.
Outputs: geometry_profiles.csv, candidate_scores.csv, selected_support.json, targets/*.pt, input/source hashes, and posthoc_vote_diagnostics.csv.
After selection is frozen, report clean-label vote accuracy, SC-vote Spearman, ties/coverage, target Brier, observed-clean/corrupted strata, class-balanced diagnostic accuracy, and actual support distribution.
Do not use oracle diagnostics to choose a new score formula, beta, tree depth, or support grid in this run.

### E1b: R0 pilot, then confirmation gate

Pilot: cora, corafull, amazonpho; seed0; all four conditions. Fit each unique candidate and CE, GCE(q=.7), OLD64, historical SUPPORT32, same-count k-means, and support-matched kNN controls. Use existing 64-hidden R0 MLP, dropout .5, Adam lr .01/wd .0005, 200 epochs. Reuse exact old cells only after input/init/selection/source audit; no approximate reuse.
For k-means control, use the number of groups selected by SC and the same LCSS rule. For kNN, use each query's actual SC-partition donor count and exclude itself. This pilot tests alternatives to the selected partition, not an independently SC-tuned k-means algorithm.
SC selects before fits. VAL selects among candidate-s models by validation accuracy, ties larger s. ORACLE chooses test best only after all selections and artifacts are frozen. All use the same per-candidate validation-selected checkpoint.
Pilot continuation, assessed on validation metrics only: uniform-noise macro-mean SC-VAL >= -.5 pp; SC-fixed32 positive on at least 2/3 pilot datasets averaged across the two uniform rates; clean SC-CE >= -1 pp on each pilot dataset. Pair is descriptive. Failure means finish the pilot report and stop E1 expansion, not retune.
If passed, add missing five-dataset/three-seed R0 cells using the same frozen design. Keep seed0 in a clearly labeled development table; report seed1/2 separately and combined descriptively, not as fresh independent research confirmation.
Full practical gate: uniform macro-mean SC-VAL >= -.5 pp, SC-fixed32 positive on >=3/5 datasets, clean SC-CE >= -1 pp on each dataset. Report macro-F1 and per-class accuracy regardless of gate. Gate is not a significance test.
Only after the R0 gate passes extend the selected-target/fixed32/CE comparison to GCN teacher training. No 720-fit R0+GCN support sweep at startup.

## E2: alignment audit and bounded extension

First inspect all 15 existing alignment logs; output all curves, epoch100 loss, best epoch/loss, last10 OLS slope and last-minus-best. Existing all-node alignment means validation-node cosine/CKA are in-sample fit diagnostics, NOT unseen-node generalization estimates.
Then seed0 cora/coauthorcs/amazonpho alignment-length diagnostic: one deterministic 600-epoch trajectory per dataset with prefix checkpoints at 100/300/600 chosen by minimum label-free evaluation alignment within that prefix. Preserve the historical100 checkpoint as an explicit reference. Exact historical bridge at epoch100 must pass; otherwise record divergence and use new100 for paired length comparisons rather than mix runs.
At each prefix train frozen-head CE and fixed historical SUPPORT32 heads for 100 epochs on four conditions. Use identical head init and head optimization. Report cosine and linear CKA plus validation-selected downstream metrics. No claim of irreducible X-to-H error from these in-sample quantities.
If a longer prefix improves average validation accuracy by >=1 pp versus new100, with neither supervision arm losing on average, choose the shortest qualifying prefix and confirm seed1/2; otherwise keep100. Do not change the E4 primary100 initialization mid-experiment.
No exhaustive unseen-node mapping experiment or inductive benchmark in this first batch.

## E4: fixed-teacher representation/prediction pilot

Reuse the verified SUPPORT_LCSS GCN probability matrices for noisy conditions. Clean matrices do not exist: train one clean SUPPORT_LCSS teacher per needed dataset/seed using the frozen teacher protocol, record this additional cost, never call it cached.
Pilot: all five datasets, seed0, four conditions. Six student arms:

| Arm | Initialization | Stage after initialization | Updates |
|---|---|---|---|
| A200 | random | all-node teacher soft CE, full MLP | 200 |
| A300 | random | all-node teacher soft CE, full MLP | 300 |
| B | label-free R2 encoder100, paired fresh head | same KD, full MLP | 200 |
| D | same R2 encoder100/head | same KD, frozen encoder and BN | 200 |
| C_full | same R2 encoder100/head | training-node SUPPORT LCSS, full MLP | 200 |
| C_head | same R2 encoder100/head | training-node SUPPORT LCSS, frozen encoder and BN | 200 |

Architecture, initialization mapping, BN behavior and deployable X-only forward must be bridge-tested before using caches. A200 versus B is equal KD duration; A300 versus B is equal total student update count including alignment. Neither is equal FLOPs by definition: report actual runtime/update counts and preprocessing costs.
All B/C/D use identical initial encoder and classifier head. A200/A300 share identical initial parameters and first200-step trajectory; their checkpoint-selection windows differ. Teacher output remains fixed across all six arms. No use of clean test or teacher-correctness strata for training weights.
Preselect one candidate (B or D) using only across-dataset pilot validation scores, tie B. Comparator is the strongest A200/A300/C_full/C_head per cell by validation, never test. Require candidate uniform+pair validation gain >=1 pp on >=3/5 datasets and clean deficit <=1 pp on every dataset before expanding to seed1/2. Report both B and D in full; do not select per test dataset.
After phase freeze, compare paired test accuracy and macro-F1, teacher-correct/wrong strata, teacher-following vs correction, R2-only-correct vs KD-only-correct. An oracle union is a diagnostic upper reference, not deployable performance. Failure stops module expansion; no reliability gate added post hoc.

## E3 and E5: conditional scope, not launch queue

E3 requires a passing E1 or E4 pilot and a separate frozen search manifest prepared before fits. Up to30 paired sampled configurations per method, identical common hyperparameter candidates, validation-only selection. Add beta selection for LCSS (0,.5,1), with the same total fit cap; do not add free extra trials. GCE q search likewise counts within the cap. Separate method-specific vs common optimization budgets in the report.
Use development seeds0-2. Audit whether seeds3-7 have already influenced this project; do not call them unused without evidence. Choose and record confirmation seeds before their results are read. Fixed GRACE means split-seed confirmation, not independent representation confirmation. Report default and tuned results, all negative results, selection cost, and historical beta provenance.
E5 requires E1 R0 continuation. Six previously used extension datasets are PubMed, Amazon-Computers, Cora_ML, Attributed-Wiki, CiteSeerFull, Coauthor-Physics; describe them as extension validation, not untouched datasets. Implement cache/path/protocol checks first; use existing splits without changing fractions. Include CE/GCE/OLD64/SC-SUPPORT and same-count k-means, four conditions, seeds0-2. No tuning on extension test results or selecting only favorable datasets.
E3/E5 are not automatically launched by the first scheduled wake. Finish the first-batch report and present gates/costs before deciding their scale. The current authorization covers the bounded first batch and its explicitly stated seed expansions.

## Implementation, runtime, and record policy

At scheduled startup run the prepared diagnostic entry, then implement the fit adapters in this new directory, reusing the inspected historical helpers without editing them. The current prepared script covers E1a and the historical-log portion of E2; GPU fit adapters are deliberately conditional on diagnostic outputs and are not yet implemented.
Training must retain config, raw stdout/stderr, epoch history, validation selection, checkpoint, all-node probabilities, initialization and source hashes, runtime, seed, paired results, mean/std, accuracy/macro-F1 and decision.json.
One GPU job at a time. Do not terminate other users' jobs. If GPU is occupied, preserve ready state and report the conflict; do not silently use CPU. Preflight checks device and free disk before large caches.
Use a per-stage lock and completed-artifact manifest to prevent duplicate scheduled starts. Never silently overwrite successful old results. Engineering failures are recorded separately from scientific negatives. Fix demonstrable local code defects with an amendment and rerun only affected cells; never change scientific thresholds after seeing outcomes.
No automatic cleanup/deletion, remote upload, package upgrades, or new data download in this first batch.

## Success/failure reporting

For each phase write REPORT_zh.md with hypothesis, ball role, motivation, exact changes, expected mechanism, predefined gates, results, deviations, limitations, and a continue/modify/defer/stop decision. Separate observed facts, plausible explanations, untested hypotheses, and weakened directions. Three seeds and practical thresholds do not establish significance.
Update this directory's HANDOFF.md after each completed stage. Update the workspace history with a short provenance-linked entry only after a stage has actual results. The submitted manuscript is out of scope.
