# R2 boundary diagnostic and completed coarsening review

Date2026-10-21. Scope authorized after review: first two non-training tasks only.
No GCE combinations, fine-tuning, deeper head, SSL, seed expansion, submitted-paper changes or new figures.
This is a posthoc mechanism diagnostic motivated by already observed rankings, not confirmatory evidence.

## Research hypothesis and GB role

Fixed disjoint regions may miss useful near neighbors across region assignments. GB is the existing
fixed label-free region, not a new algorithm. Test whether target disadvantages concentrate near radial
outskirts and where matched kNN voters cross region assignments. Neither relation is assumed true.
Revisit existing ball-level computation experiments separately; do not transfer R2 conclusions to them.

## Inputs and definitions frozen before calculation

Read R2 mechanism-controls inputs/targets/probabilities and previous representation-supervision assets,
raw clean training labels for OFFLINE DIAGNOSIS ONLY; no optimization or method selection.
Cora/CiteSeer/CoraFull, clean/uniform0.4/uniform0.6/pair0.4, seed0. Same12 protocols, same historical groups.
Evaluate only train nodes; target differences do not directly explain held-out prediction differences.

- Radial feature = distance to all-node ball centroid / ball's mean all-node radius, on normalized GRACE H.
  Mean radius is not a hard outer boundary, and this feature is not distance to a split surface.
- Crossing feature = fraction of the query's matched-support OTHER training kNN voters assigned to other GBs.
  Since support matches, this also measures replacement of original same-ball voters.
- Vote TV = half L1 difference between original GB-LOO and kNN vote proportions. Label-dependent descriptive
  mediator, not independent geometric evidence of the mechanism.
- Main target disadvantage = squared Euclidean/Brier(GBtarget,clean onehot) minus Brier(KNNtarget,clean onehot).
  Positive favors kNN. Also report class probability difference and observed-label off-mass.
- Originally clean means observed train label equals original clean class. Originally corrupted means not equal.
- Softening is any mass moved away from observed class, not an error. Strict wrong-target event requires
  some OTHER class probability > true-class probability by >1e-7; true-class ties reported separately.
- Existing model TRAIN error difference is descriptive only, not a new held-out evaluation or causal explanation.

## Stratification and confounding

Compute geometry once without labels. Quartile cutpoints based on all eligible train nodes (GB LOO support>=2),
shared across noise conditions; do not force tied feature values into different bins. Drop duplicate edges,
report actual bin count. Four groups only when four distinct intervals exist.
Report all/originally-clean/originally-corrupted subsets with explicit denominators.
Spearman ranks and Qlast-Qfirst gaps per dataset-condition; also within-ball and within-clean-class rank
associations to expose group-level confounding. These do not remove all confounding or establish causality.
No pooled-node p-values: training nodes/votes overlap; repeated noise conditions share geometry and seed.

## Descriptive interpretation rules (not rescue gates)

For each of radial and crossing features, check whether BOTH Spearman>0 and end-quartile target gap>0:
all3 clean graphs AND at least6/9 noisy cells would be broad descriptive support. Four-bin monotonicity
reported separately, never forced by pooling. Mixed/negative results mean qualify/reject a general account,
not prove an alternative cause. VoteTV has no independent causal gate.
Previous GB continuation decision stays false regardless of these results.

## Completed coarsening review

Read existing 1080raw JSON records, main/paired/E2/cost CSVs and audit. Recompute stored paired means/std
and aggregate signals; independently cross-check summary tables and raw metrics, but do NOT claim independent
retraining or a repeat audit of all prediction tensors. Separate C ball-level training vs B matched targets,
C vs full LCSS, R3 label-driven cut vs R0 geometry, and cold/end-to-end costs vs per-epoch savings.
Do not call WikiCS unused: other historical projects already used it. Check lack of strong partition/noisy-GNN
controls before promoting a positive signal. Existing1080 report remains historical, no rewriting.

## Implementation and checks

Create diagnostic_utils.py, test_diagnostic.py, diagnose.py. First failing kernel tests, implement and rerun.
Freeze source/input hashes before analysis; verify unchanged after analysis. Expected20656train-node rows,
12protocols; unit-test Brier, error/tie semantics, tie-safe bins, geometry, crossing, within-group ranks.
Cross-check global target Brier by two numeric expressions; recompute prior R2 pairwise accuracy; no new fits.
Export node_metrics.csv, binned_metrics.csv, correlations.csv, decision.json, budget_review.json/CSV,
audit.json, raw_log and Chinese detailed report, plus handoff. All seed std=null for new diagnostic.
Failure: inconsistent input, targets, identities, or old summary stops execution and records traceback.
Final decision: interpret only; no automatic further experiments.
