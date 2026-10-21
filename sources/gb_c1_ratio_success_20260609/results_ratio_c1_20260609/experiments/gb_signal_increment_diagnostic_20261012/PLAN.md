# Regional signal increment: frozen diagnostic plan

Date: 2026-10-12. Stage: exploratory screening, followed by confirmation only if the screen passes.

## Material Passport

- User request: design and execute a new experiment; submitted manuscript work is deferred.
- Sources: original literature map; experiment histories 13, 18, 41; cross-direction review;
  gb_dist_protocol_seeds_20261007; gb_regional_kmeans_20261007.
- Assets: five datasets, uniform noise 0.4/0.6, protocol seeds 1/2, fixed GRACE and ERASE caches.
- Scope: offline diagnostics, 20 protocol cells, no new teacher/student optimization.
- Status: registered before computing diagnostic outcomes.

## Hypothesis, motivation, and role of granular balls

Earlier vote-only detectors and kNN-ball ensembles failed to yield substantial independent information.
This screen specifically tests whether regional geometry, support, and vote statistics add information
to a strong teacher-confidence / peer-consistency baseline and to that baseline augmented with kNN.
Granular balls organize features, not hard labels or positive pairs. Equal feature definitions are
also supplied to flat k-means and bisecting k-means. A size-matched random partition is a control.
No feature is described as unique to granular balls.

The target is TRAINING ANNOTATION CORRUPTION, clean_label != observed_label. Teacher and student errors
are not interchangeable with this target and are not primary outcomes of this experiment.
Clean training labels are used only as diagnostic responses and to train the diagnostic metamodel.
That metamodel is an oracle probe, NOT a deployable denoising algorithm.
Validation/test labels are never used as diagnostic responses or in feature construction.

## Fixed design

- Reuse exactly the 20 cached protocols, teacher probabilities, student probabilities, and GB assignments.
  Check assignment fingerprints and probability shapes, record SHA-256 before/after.
- Use the same NumPy float32 row-normalized GRACE representation as historical grouping experiments.
- GB: original cached assignment, no rebuilding or parameter selection.
- KM: match number of GB regions, k-means++, n_init=10, random_state=0, max_iter=300, tol=1e-4.
  Reuse the earlier KM cache only when embedding SHA-256 and region count match.
- BKM: same region count, bisecting_strategy=biggest_inertia, k-means++, n_init=10,
  random_state=0, max_iter=300, tol=1e-4. This does not match GB child-size constraints.
- Random: shuffle GB assignment with random seed 100, preserving every region size.
- kNN: cosine neighbors among OTHER TRAINING NODES, k=20, batched exact GPU retrieval.
- Peer control inspired by ProCon: cosine peers among all other graph nodes, two-hop adjacency
  gets full similarity and other candidates get 0.5 times similarity, k=20. Prototypes are means
  of GRACE representations grouped by observed TRAINING labels; nearest-prototype pseudo labels
  provide peer consistency. This is a fixed diagnostic adaptation, not complete ProCon (no EMA,
  iterative clean selection, GMM, contrastive updates, or ProCon training).
  Source: https://www.ijcai.org/proceedings/2025/623 and existing NoisyGL ProCon implementation.
- Peer selection uses GRACE geometry rather than raw-X geometry; all diagnostic arms share it.
- Features are built without clean labels. Labels or masks embedded inside the H archive are ignored.
- Root branch/hierarchy features are deferred: original GB cache has no split tree. Do not invent a
  recovered hierarchy or add hierarchical features only to one grouping rule.

## Frozen feature families

Base: teacher observed-class probability, maximum probability, normalized entropy, top-two margin,
observed-class deficit, peer prototype-label agreement with observed class, peer teacher mean
observed-class probability, peer teacher mean maximum, cosine to observed-label prototype,
nearest-prototype cosine minus observed-label cosine, and current-prototype/observed agreement.

Neighbor: observed-label share, maximum share, normalized entropy, observed-label deficit, mean
cosine, and cosine standard deviation among 20 other training nodes.

Region (same columns for GB/KM/BKM/Random): log region size, log OTHER training-node support,
observed-label share, maximum share, normalized entropy, observed-label deficit, distance to region
center / mean radius, within-region distance percentile, distance to next-closest region center
minus own distance (scaled by mean radius), OTHER training-node teacher mean observed probability,
and missing-vote indicator. Mean radius is not a containing sphere boundary.

Nested arms: Base; Base+kNN; Base+GB/KM/BKM/Random;
Base+kNN+GB/KM/BKM/Random. All are fixed logistic models; no outcome-driven feature selection.

## Cross-validation and outcomes

- Five outer folds, leave one DATASET out. Both noise rates and both seeds of that dataset stay
  together. Never randomly split duplicated nodes or seeds into train/evaluation folds.
- StandardScaler and logistic regression fitted on the other four datasets only.
- LogisticRegression C=1, L2, lbfgs, max_iter=1000, no class weights. Each source protocol cell has
  equal total weight; weights are normalized to mean 1 in the training fold. Scaler uses same weights.
- Main per-cell metric: AUROC. Also report average precision, log loss, Brier error, and at 20%
  flagged coverage: precision, recall, and fraction of clean annotations mistakenly flagged.
  Ties at the threshold use fractional expected selection, independent of true labels.
- Macro-average first within each dataset over four cells, then across five datasets.
  Mean/std use ddof=1 across dataset-level paired differences. Save per-cell differences as well.
- These datasets/seeds were previously inspected. Dataset-held-out metamodel evaluation prevents
  within-screen fit leakage, but DOES NOT make the research direction externally confirmed.
- No node-level significance test or independent-node confidence interval is claimed.

## Pre-registered screening gates and stop policy

These are investment thresholds, not statistical significance or equivalence tests.

G1 (regional information): Base+kNN+GB improves macro AUROC over Base+kNN by at least 0.02,
with positive dataset-level increments on at least 3/5 datasets.

G2 (GB-specific case): Base+kNN+GB exceeds EACH of Base+kNN+KM and Base+kNN+BKM by at least
0.02 macro AUROC, and each comparison is positive on at least 3/5 datasets.

G3 (decision relevance): relative to Base+kNN, the GB arm improves precision at 20% coverage by
at least 0.02 on the macro scale, and does not increase the macro clean-annotation false flag rate.

- G1+G2+G3 pass: design fresh-data confirmation of the fixed signals; do not claim a deployable method.
- G1+G3 pass, G2 fails: regional evidence only; GB is one implementation and cannot be the unique claim.
- G1 or G3 fails: stop this teacher-label regional-confidence branch. Do not search new feature
  weights, radii, k, thresholds, or selected datasets on these outcomes.
- Before any algorithm training, require a clean-label-free rule and a paired teacher experiment.
- This offline run has a 60-minute execution budget; the broader branch has a two-week cap.
  Engineering repairs are allowed and logged; scientific thresholds cannot change after metrics.

## Code/config changes, execution, and records

New isolated directory only: PLAN.md, run_diagnostic.py, test_diagnostic.py, features, partitions,
raw_log.txt, environment.json, input_manifest.json, evaluation predictions, per_cell_metrics.csv,
dataset_metrics.csv, paired_deltas.csv, decision.json, and Chinese report/handoff.
Existing experiment assets and submitted manuscripts are not edited.

Runtime: D:/codex_temp/noisygl-rocm721/Scripts/python.exe; AMD RX 7700 XT / ROCm PyTorch.
CPU libraries handle clustering and logistic regression; GPU handles exact batched neighbors.
Commands:

```powershell
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' -m pytest experiments/gb_signal_increment_diagnostic_20261012/test_diagnostic.py -q
& 'D:/codex_temp/noisygl-rocm721/Scripts/python.exe' experiments/gb_signal_increment_diagnostic_20261012/run_diagnostic.py --run
```

Final decision and numerical outcomes are written after execution, including all negative results.
