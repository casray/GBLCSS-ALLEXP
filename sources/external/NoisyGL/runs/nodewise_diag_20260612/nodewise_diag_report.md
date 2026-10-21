# Phase 12 Nodewise Target Diagnostic Report

## Protocol

- Split: percent 0.2/0.1/0.7.
- Noise: uniform 0.4.
- Seeds: 0, 1, 2 unless overridden.
- Balls: fixed geometric anchor, max_balls=64, min_ball_size=32.
- Clean labels are used only for diagnostics.

## Aggregate

| Dataset | Variant | tau | gamma | NCR | CDR | CDR reduction | NCR loss | KL | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| citeseer | baseline |  |  | 0.0557 | 0.0128 | 0.0000 | 0.0000 | 0.5098 | 0 |
| citeseer | distance_alpha |  | 0.5 | 0.0083 | 0.0008 | 0.9524 | 0.8758 | 0.6326 | 0 |
| citeseer | distance_alpha |  | 1.0 | 0.0014 | 0.0000 | 1.0000 | 0.9855 | 0.7388 | 0 |
| citeseer | distance_alpha |  | 2.0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9014 | 0 |
| citeseer | sim_weighted | 0.1 |  | 0.0350 | 0.0048 | 0.6595 | 0.4572 | 0.5731 | 0 |
| citeseer | sim_weighted | 0.5 |  | 0.0543 | 0.0111 | 0.1310 | 0.0145 | 0.5194 | 0 |
| citeseer | sim_weighted | 1.0 |  | 0.0557 | 0.0128 | 0.0000 | 0.0000 | 0.5144 | 0 |
| citeseer | sim_weighted_distance | 0.1 | 0.5 | 0.0083 | 0.0008 | 0.9524 | 0.8758 | 0.6862 | 0 |
| citeseer | sim_weighted_distance | 0.1 | 1.0 | 0.0014 | 0.0000 | 1.0000 | 0.9855 | 0.7830 | 0 |
| citeseer | sim_weighted_distance | 0.1 | 2.0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9306 | 0 |
| citeseer | sim_weighted_distance | 0.5 | 0.5 | 0.0083 | 0.0008 | 0.9524 | 0.8758 | 0.6410 | 0 |
| citeseer | sim_weighted_distance | 0.5 | 1.0 | 0.0014 | 0.0000 | 1.0000 | 0.9855 | 0.7459 | 0 |
| citeseer | sim_weighted_distance | 0.5 | 2.0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9063 | 0 |
| citeseer | sim_weighted_distance | 1.0 | 0.5 | 0.0083 | 0.0008 | 0.9524 | 0.8758 | 0.6366 | 0 |
| citeseer | sim_weighted_distance | 1.0 | 1.0 | 0.0014 | 0.0000 | 1.0000 | 0.9855 | 0.7422 | 0 |
| citeseer | sim_weighted_distance | 1.0 | 2.0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.9038 | 0 |
| cora | baseline |  |  | 0.0768 | 0.0049 | 0.0000 | 0.0000 | 0.4569 | 0 |
| cora | distance_alpha |  | 0.5 | 0.0201 | 0.0000 | 0.6667 | 0.7345 | 0.5917 | 0 |
| cora | distance_alpha |  | 1.0 | 0.0015 | 0.0000 | 0.6667 | 0.9804 | 0.7086 | 0 |
| cora | distance_alpha |  | 2.0 | 0.0000 | 0.0000 | 0.6667 | 1.0000 | 0.8866 | 0 |
| cora | sim_weighted | 0.1 |  | 0.0576 | 0.0019 | 0.3889 | 0.2474 | 0.5231 | 0 |
| cora | sim_weighted | 0.5 |  | 0.0768 | 0.0029 | 0.2778 | 0.0000 | 0.4667 | 0 |
| cora | sim_weighted | 1.0 |  | 0.0768 | 0.0049 | 0.0000 | 0.0000 | 0.4616 | 0 |
| cora | sim_weighted_distance | 0.1 | 0.5 | 0.0083 | 0.0000 | 0.6667 | 0.8899 | 0.6486 | 0 |
| cora | sim_weighted_distance | 0.1 | 1.0 | 0.0015 | 0.0000 | 0.6667 | 0.9804 | 0.7557 | 0 |
| cora | sim_weighted_distance | 0.1 | 2.0 | 0.0000 | 0.0000 | 0.6667 | 1.0000 | 0.9174 | 0 |
| cora | sim_weighted_distance | 0.5 | 0.5 | 0.0167 | 0.0000 | 0.6667 | 0.7798 | 0.6004 | 0 |
| cora | sim_weighted_distance | 0.5 | 1.0 | 0.0015 | 0.0000 | 0.6667 | 0.9804 | 0.7161 | 0 |
| cora | sim_weighted_distance | 0.5 | 2.0 | 0.0000 | 0.0000 | 0.6667 | 1.0000 | 0.8916 | 0 |
| cora | sim_weighted_distance | 1.0 | 0.5 | 0.0201 | 0.0000 | 0.6667 | 0.7345 | 0.5959 | 0 |
| cora | sim_weighted_distance | 1.0 | 1.0 | 0.0015 | 0.0000 | 0.6667 | 0.9804 | 0.7122 | 0 |
| cora | sim_weighted_distance | 1.0 | 2.0 | 0.0000 | 0.0000 | 0.6667 | 1.0000 | 0.8890 | 0 |

## Gate Decision

- No variant passes the preregistered gate on both datasets.

## Distance-Bin Diagnostic

The raw distance-bin CDR/NCR table is written to `distance_bins.csv`. It directly measures whether clean-label damage concentrates near ball boundaries.

## Research Record

- Hypothesis: ball-level SoftDist loses information because all train nodes in a ball receive the same vote distribution and alpha.
- Granular-ball role: unchanged support boundary; only within-ball vote weighting and per-node alpha are diagnosed.
- Code change: added `scripts/diagnose_nodewise_targets.py` and unit tests.
- Config change: none to anchor SoftDist [X||H].
- Success standard: CDR reduction >= 30% and NCR loss <= 10%.
- Result: 0 aggregate rows pass the single-dataset gate; downstream Phase 13 requires a variant that passes both datasets.

## Difference From Closed Directions

- Unlike C1_topk/kNN, the support set remains restricted to the geometric ball.
- Unlike previous gates, distance alpha tests geometric membership confidence rather than noisy-label purity.
- Unlike dynamic alpha, there is no model prediction feedback loop.
- Unlike unlabeled expansion, only train-node targets are changed.
- Unlike adaptive granularity, ball construction is unchanged.
