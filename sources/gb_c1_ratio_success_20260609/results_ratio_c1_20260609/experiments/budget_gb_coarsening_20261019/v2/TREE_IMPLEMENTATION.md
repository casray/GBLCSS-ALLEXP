# Candidate tree and exact-budget cut implementation

Date: 2026-10-19. Scope: CPU construction and cut selection only; no GNN training.

## Files and API

- `granularity.py`: `BallTree`, `build_tree(z, max_leaves, seed=20261019, min_leaf=2)`, and `select_cut(tree, budget, labels, train_mask, n_classes, rule='R0', beta=1.0)`.
- `test_granularity.py`: independent exhaustive-cut comparison and behavioral checks.
- This file records mechanism, command, verification, and runtime.

`BallTree` fields are `members`, `children`, `parent`, `costs`; root ID is 0, every child ID exceeds its parent, leaves use `[-1,-1]`. Costs are within-node input-space SSE, without further normalization. Caller-side z normalization therefore determines D's scale. `num_leaves` reports actual candidate leaves. The tree is pickle-compatible; serialization belongs to the runner.

## Unlabeled construction

The constructor accepts no labels or train/validation/test masks. A highest-SSE-first priority queue expands eligible leaves. Each split uses CPU scikit-learn 2-means with `n_init=3`, `max_iter=100`, `algorithm='lloyd'`, and the fixed construction seed (default 20261019), under `threadpool_limits(limits=4)`.

When either 2-means child violates `min_leaf`, use a stable median split along the maximum-variance z coordinate. This permits splitting constant/duplicate embeddings instead of silently losing feasible budgets. Eligible leaves have at least `2*min_leaf` nodes. Construction stops at the requested maximum or when no eligible leaf remains; callers must inspect `num_leaves`. Asking the selector for more than the actual leaves raises `ValueError`.

## Additive cut objectives and inheritance

R0 uses only D and does not inspect either labels or train_mask. R3 and O2 use

`cost(B) = D(B) + beta * (n_B/L) * U_inh(B)/(1-1/C)`.

Here `n_B` counts labeled training support inside the node, `L` is total training support, and C is the class count. Only `labels[train_mask]` participates. O2 uses exactly the same mechanism as R3; the caller must pass clean training labels for O2.

For support `n_B>=2`, `U(B)=1-sum_c count_c*(count_c-1)/(n_B*(n_B-1))`. A node with less than two labeled support points inherits its parent's already inherited U, thus reaching the closest supported ancestor. Zero-support nodes contribute zero because their support weight is zero. If global training support is below 2, the fallback is the uniform class prior `1-1/C`, explicitly recorded in `global_label_fallback`. This fallback is independent of all labels; no true-prior estimate is claimed.

The inherited-label fraction in diagnostics is the fraction of all labeled support falling into selected nodes with support below 2. The histogram counts selected nodes with zero, one, or two-plus support points. A mixed two-label parent and its singleton children have identical total label cost; splitting singleton labels receives no artificial purity reward.

## Exact cardinality optimization

Bottom-up tree dynamic programming computes the minimum additive cost for every cardinality up to `min(budget, subtree_leaf_count)`. Selecting a node supplies cardinality 1. A split combines a positive cardinality from each child. For each value in the smaller child's table, all compatible values in the other child are evaluated together using a NumPy vector. Stored left-child cardinalities allow exact traceback.

This is an exact tree-knapsack computation, not a Lagrange multiplier search. Every integer cardinality between 1 and a binary subtree's leaf count is feasible. The output contains exactly M selected nodes, a contiguous assignment `0..M-1`, complete coverage, and no overlap. The implementation never mutates the input tree. Tie choices are deterministic; equal-objective alternative cuts need not have the same node IDs as another solver.

## Verification record

Run from this directory:

```powershell
& 'D:\codex_temp\noisygl-rocm721\Scripts\python.exe' -m unittest test_granularity -v
```

Test-first red: before `granularity.py` existed, all 9 tests failed with `AssertionError: granularity implementation is missing` (9 failures, 0.002 s). Green: all 9 tests passed in 0.707 s. One environment warning came from joblib's inability to query physical core count; it fell back to logical core count. KMeans and tests completed successfully, with numerical CPU threads bounded by threadpoolctl.

Checks cover:

- For R0, R3, and O2, all budgets 1 through 6 agree with exhaustive enumeration on a branching synthetic tree.
- Altering every held-out label to an out-of-range sentinel does not change R3 assignment or objective.
- R0 accepts `None` labels/mask and ignores arbitrary invalid label values.
- Singleton descendants inherit mixed-parent disagreement and do not receive an artificial purity bonus.
- Label weights use labeled-support count rather than total node membership.
- Global one-label and zero-label fallbacks are explicit and finite.
- Construction is deterministic and costs equal independently calculated SSE.
- Repeated identical input points reach 8 leaves of size 2 through median fallback.
- Exact cardinality, coverage, no overlap, contiguous IDs, tree immutability, and explicit errors for unavailable budgets.

## CPU performance smoke check

Synthetic Gaussian input, generator seed 20261019, N=20,000, D=16, requested candidate leaves=8,000, min_leaf=2. R3 used five classes, 1,000 uniformly sampled training support points, and target M=2,000. Measured with `time.perf_counter` in the stated Python environment:

| Quantity | Value |
|---|---:|
| Actual candidate leaves | 8,000 |
| Tree nodes | 15,999 |
| Construction | 21.9179 seconds |
| R3 exact selection | 0.4143 seconds |
| Actual selected M | 2,000 |
| Selected support histogram (0 / 1 / 2+) | 1,238 / 567 / 195 |

The performance check verifies feasibility and exact budget on a representative size, not graph-learning accuracy or a research hypothesis. It indicates construction should be cached independently of split/noise seeds and cut selection can be rerun cheaply. The initial benchmark command included the Unix-only `resource` import and failed on Windows before executing any work; rerunning without that unused import produced the results above.
