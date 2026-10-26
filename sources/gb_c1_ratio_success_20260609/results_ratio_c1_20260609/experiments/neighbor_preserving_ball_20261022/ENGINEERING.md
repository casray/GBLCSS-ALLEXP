# Engineering trace

## Scope

Only this new experiment directory and final project context entries are edited.
Training/generation sources and inputs were frozen in frozen_inputs.json before
smokes/full fits. analyze.py and audit_geometry.py are read-only postprocessing
programs; their hashes are recorded in their audit outputs, not retroactively
added to the original freeze. No old code, labels, models or manuscript edited.

## Test-first sequence

1. test_generator.py ran before generator.py existed: expected import failure.
2. First implementation passed8/9 tests. The unanimous-vote test expected exact
   float32 zero, but the unchanged historical LCSS arithmetic leaves5.96e-8.
   Added absolute tolerance1e-7 to that assertion, not to the model formula.
3. All9 tests passed, including243 exhaustive small-tree cut cases, direct TV
   comparisons, duplicate-distance global-ID ties, self exclusion, prefix
   stability, invalid inputs and class-permutation equivariance.
4. All96 prepared target matrices matched an independent NumPy formula within
   3e-7; all ball targets matched historical PyTorch arithmetic exactly.
5. Sixteen GPU smokes passed. Four CE/OLD64 route bridges reproduced the first
   three historical epochs' validation counts and losses exactly (delta0).

## Additional geometry audit

Initial audit wrongly required the union of the three evaluation splits to cover
all graph nodes. The historical percent_split_indices function independently
floors .2N/.1N/.7N, leaving up to2 nodes outside these sets. Inspected the original
function and replaced that assertion with exact ordered historical split
reproduction, disjointness and valid index bounds. The geometric partitions
still must cover every graph node. This change affects no training or input.

Sklearn/joblib warns that physical-core detection is unavailable and falls back
to logical cores. KMeans BLAS thread count remains explicitly1; this is not a
failed split or fallback algorithm.

## Reproduction commands

Use D:/codex_temp/noisygl-rocm721/Scripts/python.exe from the workspace root:

```text
-m unittest discover -s experiments/neighbor_preserving_ball_20261022 -p test_*.py -v
experiments/neighbor_preserving_ball_20261022/run.py prepare
experiments/neighbor_preserving_ball_20261022/run.py smoke
experiments/neighbor_preserving_ball_20261022/analyze.py smoke
experiments/neighbor_preserving_ball_20261022/run.py run
experiments/neighbor_preserving_ball_20261022/audit_geometry.py
experiments/neighbor_preserving_ball_20261022/analyze.py full
```

prepare refuses an already-frozen directory; do not rerun it over these inputs.
The run phase skips completed results and refuses a failed/running result until
it is explicitly reviewed. No parameter rescue or result-dependent retries.
