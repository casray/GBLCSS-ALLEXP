# Support versus neighbor identity: fixed ablation

Date: 2026-10-22. Authorized by the user's approval of the preceding review.

## Hypothesis, motivation and ball role

The previous NP result may come from matching the reference support size rather
than preserving the identities of nearby donors. Decompose each query cost as
1-a/max(k,s) = abs(s-k)/max(k,s) + (min(k,s)-a)/max(k,s).
Balls remain label-free partitions sharing leave-one-out class counts. No new
loss, representation, minority gate or soft membership is introduced.

## Frozen protocol and configurations

Cora, CiteSeer, CoraFull; clean, uniform0.4, uniform0.6, pair0.4; paired protocol
and student seeds 0,1,2. Historical GRACE is shared, not independently pretrained.
Seed0 is already explored; seeds1/2 are additional paired replications of this
frozen comparison, not previously untouched datasets or SSL confirmation.
Use the GCE-fusion experiment's exact splits, labels, raw X, initializations and
frozen R2 encoders. Ordered20/10/70, train-only noise, first strictly best clean
validation accuracy, R0 200 epochs, R2 head100. Full grid is frozen before test
evaluation. No hyperparameter selection, k=32 throughout, beta=.5, min_votes2.

Candidate trees and exact float64 neighbor ordering use the previous generator
unchanged: max256 fine leaves, min32 total/min3 training members, SSE-gain
priority, KMeans++ n_init10/max_iter300/Lloyd/thread1, seed+tree-node-id.
Read only H and training positions when constructing any partition.

Eight arms (576 cells):
- CE: observed one-hot targets.
- OLD64: exact historical partition and LCSS targets for each seed.
- NP: original full TV cost, same dynamic program as previous experiment.
- SUPPORT: same tree and DP, only m*abs((m-1)-k)/max(m-1,k) per tree node.
- COUNT: same tree, exact NP leaf count, greedily split available largest SSE
  gain with canonical member-id/node-id tie breaking. Count only, no neighbors.
- KMNP: same normalized H and NP count, KMeans n_init10/max_iter300/seed/thread1.
- KMOLD: identical KMeans implementation, historical actual region count.
- KNNMATCH: exactly NP's per-query donor count, same exact nearest neighbors.

Reuse216 historical cells only after target, initialization, epoch, input and
source hash checks: all144 CE/OLD64 plus72 seed0 NP/KMNP/KNNMATCH. Fit360 new
cells. Aliases are recorded, never counted as independent models. No new GRACE
or R2 pretraining. Code changes are confined to this new experiment directory.

## Expected mechanism and decision criteria

Primary contrasts: NP-SUPPORT (neighbor identity), NP-COUNT (placement at fixed
count), KMNP-KMOLD (count within one implementation), NP-OLD64 (repair),
NP-KMNP and NP-KNNMATCH (competitive controls). R0 primary, R2 secondary.
Report per-cell and per-seed paired differences with sample standard deviations.

For a retained repair: all three seed-level noisy macros NP-OLD64 >=0, at least
6/9 cell means >=0, no cell mean <-1pp, each clean dataset mean NP-CE >=-1pp,
and CoraFull clean NP-OLD64 >=1pp. Report worst seed clean separately.
Neighbor-identity contribution supported only if NP-SUPPORT noisy macro >=.5pp,
at least6/9 cell means >0 and at least2/3 seed macros >0. Same criterion against
COUNT and KMNP separately. If abs(NP-SUPPORT macro)<.5pp, call it no material
increment detected, NOT a formal equivalence result. Do not quantify a causal
fraction of the old improvement from these outcomes.

If SUPPORT matches/beats NP, retain simpler support objective as a candidate;
if NP wins, retain donor identities as a mechanism candidate; if neither beats
KMNP, no independent granular-ball advantage is claimed. Failed clean or repair
gates remain failures. No small-loss combination or parameter rescue follows.

## Verification, results and final decision

Tests: exact algebra, tiny-tree exhaustive DP, count constraints/ties, invalid
inputs, independent LOO target reference. Preparation verifies historical
protocols and seed0 cached partitions. Short GPU smokes bridge historical
first-three losses and validation counts. Freeze all sources and inputs before
full fits. Final audit reloads every unique checkpoint, verifies first-best
epoch and R2 encoder/BN invariance, reconstructs targets and paired tables.
Keep config, raw logs, predictions, mean/std, source manifest, failures, complete
result table and Chinese interpretation. Clean labels for diagnostics and test
metrics are read only after grid completion. Target Brier is not assumed to
predict accuracy; macro-F1 and clean damage are reported alongside it.

Results and final continue/modify/defer/stop decision go in the final report,
not into this frozen protocol. No result is currently assumed.
