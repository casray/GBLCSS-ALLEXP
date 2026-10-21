# R2 mechanism controls: frozen single-seed screening plan

Date: 2026-10-21. Authorized by user after the representation-supervision review.
Goal: separate useful representation transfer, class-directed local targets, and granular-ball grouping.
Architecture: import the completed representation_supervision_20261021 sources read-only; create targets,
controls, checkpoints and reports only in this new directory. No submitted manuscript changes.
Stack: existing ROCm PyTorch GPU environment, NumPy, scikit-learn KMeans, threadpoolctl.

## Questions and role of granular balls

H1: R2 gains require meaningful node-wise GRACE correspondence, not merely a frozen random encoder.
H2: LCSS adds useful class direction beyond generic regularization and the same per-node correction mass.
H3: granular balls add value over matched kNN, k-means and random partitions on the same frozen R2 encoder.
Balls only organize fixed, label-free regions and leave-one-out class statistics. No hard positive pairs,
dynamic partitions, model-capacity changes, small-loss selection or parameter search in this batch.

## Fixed protocol and budget

Cora, CiteSeer, CoraFull; clean, uniform0.4, uniform0.6, pair0.4; protocol/student seed 0.
Reuse the previous exact raw X, ordered 20/10/70 split, training-only noise, normalized H, and historical GB assignment.
Use clean validation labels only for head checkpoint selection, first strict maximum over 100 full-batch updates.
All classifier fits: Adam lr=0.01, weight_decay=0.0005, same initial classifier, no early training termination.
All deployed students remain the original two-layer X-only MLP, hidden64, frozen first layer/BN, no stage-2 dropout.
The real R2 checkpoint is reused. Refit CE/LCSS as reproducibility anchors and compare with the previous 24 cells.
Test metrics are computed only after all 156 formal fits complete. Seed std is null; no significance claims.

## Arms (156 new classifier fits)

Real R2: CE, GB-LCSS, LS, GCE, BOOT, MASS, KNN, KM, RANDOM: 9 x 12 = 108 fits.
Random frozen encoder and shuffled-H aligned encoder: each CE and GB-LCSS: 2 x 2 x 12 = 48 fits.
Three shuffled-H pretrains (100 updates each), shared across conditions/rules; three random encoders saved without updates.
No newly trained GRACE, no c-group fine-tuning/deeper head, no seed expansion.

### a. Regularizers

- CE: ordinary observed-label cross entropy.
- LS: PyTorch-style label smoothing epsilon0.1, target=(1-epsilon)e_z+epsilon/C.
- GCE: mean (1 - p_z^0.7)/0.7, q=0.7; all samples, no truncation.
- BOOT: soft bootstrapping target=0.8 e_z+0.2 stop_gradient(p), refreshed each update.
- MASS: take GB target's own observed-class probability a=t_z; assign a to z and (1-a)/(C-1) to each other class.
  This preserves each node's actual off-observed correction mass, not just its alpha. It deliberately retains
  GB-derived correction strength, so it isolates class direction, not the value of geometry-free supervision.
All use the same fixed optimizer and budget. These are preset comparisons, not best-tuned versions of all robust losses.
Log full optimization traces and final validation, do not infer universal failure from a single fixed learning rate.

### b. Grouping replacements

- GB: exact previous leave-one-out targets, beta0.5, min_votes2, distribution-only on ties.
- KNN: for each training query select exactly as many OTHER training nodes as its GB leave-one-out support.
  Cosine similarity on normalized H; stable descending sort, training-order tie breaking; exclude self explicitly.
  No labels in neighbor selection. Same LCSS transformation of the resulting vote counts.
- KM: scikit-learn KMeans, k=actual number of historical balls, k-means++, n_init10,
  max_iter300, tol1e-4, random_state0; all nodes' H, no labels/split masks in fitting. Same target rule.
  Equal number of regions, not equal per-node support. Report support and target differences.
- RANDOM: shuffle GB assignments separately within train and nontrain nodes with seed1701.
  Preserves each group's total size AND number of training voters exactly. Uses masks, never class labels.
  Matches support distribution, not each query's exact original support. Same target rule.

### d. Representation controls

- RANDOM_ENCODER: original seed0 model, no alignment updates, freeze encoder/BN and train same head100.
- SHUFFLED_H: fixed all-node H row permutation, seed1702; preserve H marginal and optimization100,
  select minimum label-free evaluation alignment, then freeze encoder/BN and train same head100.
- Both use CE and the same REAL-GRACE GB targets. Thus d isolates the representation branch conditional on
  fixed supervision; GB-target arms are not fully graph-free training controls. Compare CE arms first.
Random encoder has zero stage-1 updates and is not compute matched; shuffled-H is the matched100+100 control.

## Frozen exploratory gates and failure rules

Mean = macro-average across 3 datasets x 3 noisy conditions, not independent experimental replicates.
1. Transfer signal: real R2+CE exceeds each random and shuffled+CE by >=1pp noisy macro and positive
   dataset-mean differences on >=2/3 datasets. Evaluate separately, not a proof of a unique mechanism.
2. Local direction signal: GB exceeds MASS by >=0.5pp noisy macro, >=6/9 positive cells, >=2/3 positive
   dataset means. Report clean effects too.
3. Generic regularizer comparison: choose ONE global LS/GCE/BOOT by mean noisy VALIDATION accuracy,
   tie in listed order, then compare GB on all cells. Also report a stricter descriptive per-cell test envelope,
   explicitly not a deployable selector. Require GB noisy >=0.5pp vs the globally selected rule, >=6/9 positive.
4. GB-specific signal: require GB > KM by >=0.5pp noisy macro, positive >=6/9; > RANDOM by >=0.5pp;
   report KNN regardless, and do not claim GB necessity if KNN matches/exceeds. KNN comparison is a separate gate.
5. Retain the previous clean-safety cap: GB loss versus CE must not exceed1pp on ANY clean dataset.
   Failing safety means no automatic expansion even if a noisy mechanism gate passes.
Failing a gate is a resource decision, not proof of universal impossibility. Keep negative outputs.
No test-driven changes to targets, weights, thresholds, schedule or checkpoint choice after freezing.
Engineering failures stop the affected stage; record traceback and reviewed fix before resuming.
Per-fit timeout900s; preprocessing timeout1800s per graph; no automatic new branches.

## Implementation and verification sequence

1. Add test_controls.py with failing feature assertions and exact numeric checks.
2. Implement kernels.py (losses, mass control, votes, support, permutations); run CPU tests.
3. Add run.py/config.json; prepare matched inputs, GPU smoke and anchor parity before test scoring.
4. Freeze PLAN/config/source/checkpoint/input hashes; train full grid sequentially on GPU.
5. Score held-out test after complete marker; export results, paired differences and gate decisions.
6. Reload all X-only deployment checkpoints, check BN freeze, first strict validation max, exact budgets,
   original input hashes, group sizes/support/self-exclusion, and anchor equality. Recompute test accuracy independently.
7. Write detailed Chinese report, mean/std status, failure/success decision and durable handoff.

## Results and final decision

Pending execution. Execution status and findings will be recorded in WORK_LOG.md and 综合结论与接续.md;
this frozen plan will not be retroactively edited to match outcomes.
