# MLP-52 Cross-Partition Vote Pooling Plan

## Research hypothesis

MLP-51 showed that three independent GB partitions improve Brier in 6/6 cells, but averaging three already mixed soft targets suppresses useful hard corrections. Pooling the leave-one-out label votes before computing a single `q` and `alpha` may retain cross-partition variance reduction without repeatedly mixing in the node's noisy one-hot label.

## Granular-ball role

Each GB remains a fixed, label-free, medium-scale local semantic unit. The method pools the three multisets of LOO votes associated with a node. A training neighbor appearing in multiple H-seed balls receives repeated votes, which represents cross-representation membership stability. Labels do not change any partition.

## Frozen protocol

- Same datasets, noise settings, split, seed, three H artifacts, GB parameters, LOO rule, tie rule, and beta as MLP-51.
- `pooled_gb_3h`: sum three per-node LOO class-count vectors, normalize once, then compute one tie-safe target.
- `pooled_knn_3h`: exact same pooling rule over three exclude-self, vote-matched kNN neighborhoods.
- `pooled_gb_3h_shuffled`: same operation on independently size-matched shuffled assignments.
- `mean_target_gb_3h`: MLP-51 equal final-target mean, retained only to test whether the new aggregation restores corrections.
- No target weighting, learned gate, validation selection, or hyperparameter search.

## Preregistered gate

1. Pooled GB improves Brier over single-H GB in at least 4/6 cells.
2. Pooled GB forms a kNN Pareto win in at least 4/6 cells: lower Brier and target accuracy no more than 0.5pp below pooled kNN.
3. Pooled GB target accuracy is at least mean-target GB accuracy in at least 4/6 cells, showing that vote pooling repairs over-conservative target averaging.
4. No pooled-GB target-accuracy regression over single-H GB exceeds 0.5pp.
5. Pooled real GB has lower Brier than pooled shuffled GB in 6/6 cells.
6. All integrity and provenance checks pass.

Failure closes vote pooling. Missing only one count by one cell with no adverse regression is a near-pass; otherwise proceed to robust-teacher latent balls for round 3.

## Outputs

- `runs/mlp52_crosspartition_vote_pool_20260806/phase_a_raw.csv`
- `runs/mlp52_crosspartition_vote_pool_20260806/per_view_diagnostics.csv`
- `runs/mlp52_crosspartition_vote_pool_20260806/phase_a_gate.json`
- `runs/mlp52_crosspartition_vote_pool_20260806/mlp52_report.md`

