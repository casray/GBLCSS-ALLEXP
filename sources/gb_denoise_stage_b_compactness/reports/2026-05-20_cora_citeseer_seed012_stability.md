# Stage B Stability Gate: Cora and CiteSeer Seeds 0/1/2

## Research Hypothesis

If true granular-ball compactness is useful as a label-free local structure prior, B1 should improve over B0 across multiple seeds and datasets. B3/B4 should perform worse than B1 if true assignment matters.

## Code State

Runs were executed in a clean detached worktree at commit `fc62447` to avoid uncommitted changes in the main working tree.

Worktree:

`C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs`

## Configuration

- Datasets: Cora, CiteSeer
- Seeds: 0, 1, 2
- Methods: B0, B1, B3, B4
- Epochs: 1000
- Warm-up epochs: 300
- SSL objective: negative-free
- Ball space: H
- Reliable ball min size: 3
- Reliable ball max size: 256
- Reliable ball AD percentile: 50

Local Planetoid raw files were copied from Stage A cache to avoid network download failures.

## Aggregate Results

### Linear Evaluation Accuracy

| dataset | method | mean | std |
|---|---:|---:|---:|
| Cora | B0 | 0.427333 | 0.114583 |
| Cora | B1 | 0.436000 | 0.081191 |
| Cora | B3 | 0.130000 | 0.000000 |
| Cora | B4 | 0.130000 | 0.000000 |
| CiteSeer | B0 | 0.450333 | 0.161017 |
| CiteSeer | B1 | 0.387667 | 0.044770 |
| CiteSeer | B3 | 0.077000 | 0.000000 |
| CiteSeer | B4 | 0.077000 | 0.000000 |

### B1 Paired Accuracy Delta Against B0

| dataset | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Cora | +0.076 | -0.016 | -0.034 |
| CiteSeer | +0.033 | +0.069 | -0.290 |

### Assignment Controls

B3 and B4 were consistently worse than B0 on both datasets and all seeds.

For Cora:

- B3 deltas: -0.214, -0.250, -0.428
- B4 deltas: -0.214, -0.250, -0.428

For CiteSeer:

- B3 deltas: -0.272, -0.289, -0.559
- B4 deltas: -0.272, -0.289, -0.559

## Interpretation

This batch weakens the claim that B1 is already stable. The method has a positive mean on Cora, but only one of three seeds improves. On CiteSeer, B1 improves seed 0 and seed 1 but fails badly on seed 2, giving a negative mean.

The assignment controls remain informative: B3/B4 consistently collapse downstream accuracy. This supports the narrower claim that arbitrary compactness is harmful and that true assignment matters. However, it does not yet prove that the current true-assignment B1 is robust.

## Claim Boundary

Supported:

- Compactness is active after size-aware splitting.
- Random or size-matched random assignment compactness is harmful across these runs.
- Radius/AD improvements alone are not sufficient success criteria.

Not supported yet:

- B1 is stable across seeds.
- B1 is stable across datasets.
- Current hyperparameters are ready for paper-level claims.

## Decision

Modify before expanding to PubMed.

The next experiment should not be PubMed yet. First fix the seed instability on Cora/CiteSeer because PubMed would be more expensive and less informative if the current method remains unstable.

Recommended next changes:

1. Add automatic summary columns for `compact_nonzero_epochs`, `covered_nodes_max`, and `covered_nodes_last`.
2. Lower the risk of over-compression by testing smaller `lambda_gb` values, for example `0.001` and `0.0005`.
3. Add an anti-collapse guard or early warning based on effective rank.
4. Re-run Cora/CiteSeer seeds 0/1/2 before running PubMed.
