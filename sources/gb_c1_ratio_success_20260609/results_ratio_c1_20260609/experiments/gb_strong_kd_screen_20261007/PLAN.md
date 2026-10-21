# Strong-KD regional increment screen

Date: 2026-10-07. Status: planned before training; user approved this screen.

## Research question and hypothesis

Does the existing regional probability-mean KL retain an incremental benefit
when pointwise soft-label cross entropy is replaced by published DIST or DGKD
losses, for exactly the same frozen ERASE teacher and attribute-only MLP?
Granular balls supply a regional soft constraint, not hard class labels. This
is a post-submission transfer experiment, not validation of original LOO theory.

## Evidence and risks

Read the literature map, original experiment history, history 39, and regional
k-means report. GB-KD improved ordinary KD by 0.296753 pp over three student
seeds; matched k-means improved it by 0.281181 pp. These are fixed protocol
seeds, not independent teacher/noise/split repetitions. No GB-specific accuracy
claim follows. Stronger KD might absorb the whole regional increment.

DIST source: hunto/image_classification_sota commit
6f7a70a1dd57fd88413d83909d3d53243068f701, linked by hunto/DIST_KD.
DGKD source: xsk160/DGKD commit ae8a495a054578d4b5e36aa5e7d3a3f4872037f6.
Downloaded source snapshots are in vendor/. Inspect, do not execute the full
upstream training drivers. Upstream DGKD's transductive driver sends labels of
idx_train + idx_val + idx_test into its target-class-dependent loss. This is not
permitted by our protocol. DGKD-T uses frozen teacher argmax for ALL nodes and
is explicitly an adapted loss baseline, not a faithful full-paper reproduction.
No observed or clean labels enter any distillation objective in this screen.

## Frozen protocol

- Datasets: cora, citeseer, corafull, coauthorcs, amazonpho.
- Uniform noise 0.4 and 0.6; original split/noise/ERASE/GRACE seed 0.
- Student seed 0. Same cached teacher probabilities and original GB membership.
- Raw X-only MLP: 2 layers, width 64, BN, dropout .5; Adam lr .01,
  weight decay .0005; 200 full-batch epochs; same initial state and RNG.
- First strict maximum of clean validation accuracy selects checkpoint.
- No labels outside the validation set are read for candidate selection.
- Cached temperature-1 teacher probabilities give equivalent logits log(P).
  Clamp only at dtype tiny; record the number of exact zeros. Higher-temperature
  probabilities are softmax(log(P)/T). This reconstruction loses information
  if teacher probabilities underflowed; disclose zero counts.
- Plain KD candidates T=1 and T=4. DIST candidates T=1 and T=4, beta=gamma=1
  (official constructor defaults). DGKD-T candidates T=.7 and T=1,
  alpha=15, beta=3 (weights and .7 come from official train_and_eval.py).
- All losses include T^2 scaling. Plain KD T=1 uses the old exact CE call for
  replay. DIST and DGKD core functions are imported unmodified from vendor/.
- Each candidate has a paired +GB arm with regional coefficient 1. Regional KL
  uses temperature-1 probabilities and size/N weights, as in the prior study.
- Retain the native DGKD loss scale; log gradient ratios. A weak regional
  increment under its larger loss scale is not proof that regions are useless.
- No noisy-label CE, feature augmentation, teacher retraining, backbone change,
  or extra epochs. GPU mandatory, CPU fallback forbidden; four CPU threads.

## Grid and selection

3 families x 2 candidates x 2 region states x 10 cells = 120 final fits.
Reuse the 20 exact plain-KD T=1 R0/R1 references after bit-exact replay checks.
Run 100 new fits plus 2 replay fits. One seed implies seed std is unavailable,
not zero. Dataset variation must not be reported as seed std.

Primary paired comparison: within each family/cell, select the candidate using
the NO-region validation accuracy (ties: listed candidate order), then compare
its matched +GB fit at that SAME candidate. This isolates the regional addition.
Secondary practical comparison: independently validation-select the candidate
for each region state (same two-candidate budget). Also compare the overall
validation-selected no-region challenger to its same-family/candidate +GB fit.
All selection is frozen BEFORE loading test labels in the separate analysis.
No method is selected by test accuracy. Report every candidate, not just winners.

## Success, failure and decision

Exploratory continue gate: +GB improves the validation-selected no-region
challenger by at least 0.20 pp on average and in at least 7/10 paired cells.
Also report paired increments for each family and worst-cell regressions.
This is a practical screening threshold, not statistical significance.
If unmet: pause adding new regional modules; check optimization/temperature and
loss-scale evidence before any broader rejection. If met: propose independent
full-protocol seeds and a second teacher. Neither outcome supports SOTA or
granular-ball superiority over k-means. No follow-on grid is automatic.

## Implementation plan

- [ ] Test loss wrappers against official losses on CPU/GPU, gradients, teacher
  detachment, class permutation, temperature reconstruction and stable extremes.
- [ ] Implement loss-only wrappers, runner, validation-only selector and analysis.
- [ ] Freeze source/input hashes, environment, source versions and candidate list.
- [ ] Replay ordinary and +GB Cora .4 seed0, requiring bit-exact probabilities,
  identical validation trajectory and checkpoint epoch.
- [ ] Run all 100 new fits, keeping per-epoch loss/validation logs, gradient and
  region diagnostics, best state/probabilities and fixed-epoch20/50/100/200 files.
- [ ] Validate initialization, protocol, files and selected-model reload; summarize
  all positives/negatives, then update project handoff/history.

## Results and final decision

Pending execution. Do not fill from expectations or omit failed fits.
