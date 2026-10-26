# Neighbor-preserving granular balls: seed0 implementation and experiment

Date: 2026-10-22. User approved execution of the reviewed design in
research/neighbor_preserving_ball_design_20261022/算法设计与多视图依据.md.

## Hypothesis and role

Selecting a geometric-tree cut to preserve each query's label-free local voting
weights may improve LCSS supervision beyond radius reduction or a fixed ball
budget. Balls share class counts only; there is no representation training change,
GCE, soft membership, multiview fusion, label-purity split or test-driven tuning.
Past support/budget and label-risk-pruning failures remain relevant controls.

## Scope and protocol

Cora, CiteSeer, CoraFull; clean, uniform0.4, uniform0.6, pair0.4; protocol/student
seed0. R0 primary, R2 secondary. Same ordered20/10/70 split, historical GRACE H,
raw X, observed training labels and seed0 R2 encoder as the completed GCE-LCSS
experiment. R0 200 epochs; R2 frozen encoder/BN head100; hidden64, dropout.5,
Adam lr.01 wd.0005. First strictly best clean-validation accuracy selects epoch.
Test metrics are computed after all runs, never for arm selection. This is a
development smoke/diagnostic seed, not independent multi-seed confirmation.

Eight arms: CE, OLD64, TREE64, FINE, NP, KM, KNN32, KNNMATCH.
192 grid cells;48 CE/OLD64 anchors may be reused after exact input/state/protocol
checks.144 new fits at most; aliases must be explicitly recorded. No new GRACE
or R2 pretraining. Old sources and manuscripts stay unchanged.

## Algorithm frozen before outcomes

Use double-precision row-normalized historical H. All tree/neighbor inputs are
features and training positions only. Child minimum total size32; training
members3 (two LOO donors). Up to256 fine leaves. KMeans++ n_init10,
max_iter300, Lloyd, random_state=(seed+node_id) mod (2**32-1), one BLAS thread.
Canonical child order is smallest global node id. Legal proposed splits enter a
max-SSE-gain queue, ties use smallest member id then tree id. Reject nonpositive
numerical SSE gain <=1e-12*max(1,parent_SSE), or illegal children; no projection
fallback, radius threshold, label criterion, or forced count.
TREE64 is the exact frontier after63 accepted splits, or final frontier if fewer.

k=min(32,L-1). Exact double squared Euclidean neighbors among training nodes,
self excluded; ties break by global node id, irrespective of train-index order.
For B, s=|B intersect T|-1, a_i=number of reference neighbors in B;
J(B)=sum_i[1-a_i/max(k,s)]. Bottom-up F=min(J,F_left+F_right),
with ties within1e-12 per training node selecting parent. NP is the recovered cut.
KM uses the same normalized H, actual NP region count, n_init10, same seed;
KM may have different training-support distribution, which is reported.
KNN32 uses the reference neighbors; KNNMATCH exactly matches each NP query's
LOO support. All targets use existing beta.5, min_votes2 and tie-safe LCSS.

## Files and checks

- generator.py: label-free tree, exact neighbors, costs and dynamic program.
- test_generator.py: direct TV, exhaustive cuts, invariants, prefix stability,
  tie handling, donor exclusion, target reference and invalid input tests.
- run.py: frozen caches, CPU preparation, reused anchors, GPU fits and logs.
- analyze.py: independent reconstruction/metrics, gates and reporting tables.
- config.json, frozen_inputs.json, per-run configs, raw logs, targets, tree,
  selected states, predictions, result CSVs, audit, report and handoff retained.

Implementation sequence: tests RED; kernels GREEN; prepare+independent targets;
freeze sources/inputs; short GPU bridge smokes; full192-cell grid; audits;
report and context update. Live status is in WORK_LOG.md, not this frozen plan.
No full project git checkout is created: this authorized isolated experiment
folder reads the existing untracked research assets; no tracked source is edited.

## Success/failure and continuation

Apply the numerical criteria per route at seed0 as provisional screens only:
repair: all3 clean differences vsCE>=-1pp; CoraFull clean vsOLD64>=1pp;
noisy macro vsOLD64>=0, >=6/9 nonnegative cells, no cell<-1pp.
incremental: against each TREE64/FINE/KM, noisy macro>=.5pp and >=6/9 positive
cells, plus clean safety. Multi-seed >=2/3 sign criteria cannot be judged here.
Report both kNN controls, not a hidden pass/fail surrogate. Proxy J improvement
is an optimization property, not evidence of improved classification.

If both repair and incremental screens fail for R0, stop at seed0. If R0 passes
either full provisional screen, this run reports a candidate for a separately
frozen seed1/2 confirmation; do not automatically add new methods or multiview.
Do not lower old GCE gates, tune k/count/support, or combine modules after failure.

## Evidence and risks

True train labels are used only after targets are frozen for diagnostics;
neither label values nor raw.y go into generation. Noise affects training only.
Shared donors/balls and conditions are dependent observations; no node-level
p-values. Seed std is unavailable for seed0. Cached pretraining costs are
excluded from new-stage timing and clearly identified. Full neighbor/tree/KM
construction time is reported; no presumed speedup over kNN. GPU is mandatory
for training; per fit timeout900s. Exact KNN may be built on CPU float64 to keep
stable numerical ordering. Tree depth and actual count, not log/count guesses,
are recorded. Final decision must say continue/modify/defer/stop with negatives.
