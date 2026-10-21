# Stage B Lambda Stability Sweep

## Research Hypothesis

The previous B1 instability may be caused by an overly strong granular-ball compactness weight, `lambda_gb=0.005`. If this is true, smaller weights should reduce over-compression and make B1 more stable across Cora/CiteSeer seeds.

## Granular-Ball Role

Granular balls remain a label-free soft local structure prior. Labels are used only for linear evaluation and diagnostic purity.

## Code State

Runs were executed in a clean detached worktree at commit `fc62447`.

Worktree:

`C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs`

The main worktree had unrelated uncommitted code changes, so it was not used for the runs.

## Configuration

- Datasets: Cora, CiteSeer
- Seeds: 0, 1, 2
- Methods: B0 reused baseline, B1 rerun
- B1 `lambda_gb`: 0.001, 0.0005, 0.0001
- Epochs: 1000
- Warm-up epochs: 300
- SSL objective: negative-free
- Ball space: H
- Reliable ball max size: 256
- Results directory: `results_lambda_sweep/`

## Results

### B1 Accuracy Mean Over Seeds

| dataset | lambda_gb | B1 acc mean | B1 acc std |
|---|---:|---:|---:|
| Cora | 0.001 | 0.442667 | 0.187905 |
| Cora | 0.0005 | 0.396000 | 0.121791 |
| Cora | 0.0001 | 0.332333 | 0.053725 |
| CiteSeer | 0.001 | 0.463333 | 0.049652 |
| CiteSeer | 0.0005 | 0.478667 | 0.073894 |
| CiteSeer | 0.0001 | 0.460000 | 0.119512 |

### B1 Paired Accuracy Delta Against B0

| dataset | lambda_gb | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| Cora | 0.001 | +0.217 | -0.154 | -0.017 |
| Cora | 0.0005 | +0.036 | -0.097 | -0.033 |
| Cora | 0.0001 | +0.031 | -0.108 | -0.208 |
| CiteSeer | 0.001 | +0.129 | +0.042 | -0.132 |
| CiteSeer | 0.0005 | +0.157 | +0.029 | -0.101 |
| CiteSeer | 0.0001 | +0.042 | +0.025 | -0.038 |

### Mean Paired Deltas Across All 6 Dataset-Seed Pairs

| lambda_gb | mean delta acc | std delta acc | mean delta effective rank | mean delta AD | mean delta purity |
|---:|---:|---:|---:|---:|---:|
| 0.001 | +0.014167 | 0.145409 | -0.903821 | -0.008415 | -0.023011 |
| 0.0005 | -0.001500 | 0.097439 | +0.131533 | -0.004119 | +0.002561 |
| 0.0001 | -0.042667 | 0.098792 | +0.693578 | -0.000405 | -0.022420 |

All B1 runs activated compactness:

- `compact_nonzero_epochs = 700` for all lambda settings and all seeds.
- `covered_nodes_max > 0` for all lambda settings and all seeds.

## Interpretation

Lowering `lambda_gb` does not solve the stability problem.

The best mean accuracy delta is from `lambda_gb=0.001`, but it is not stable: Cora seed 0 improves strongly, while Cora seed 1 and CiteSeer seed 2 still drop. `lambda_gb=0.0005` is the most balanced setting by accuracy variance and purity delta, but its mean accuracy delta is approximately zero. `lambda_gb=0.0001` is too weak to produce a reliable gain.

This suggests that the issue is not only that `lambda_gb=0.005` was too large. The current B1 mechanism itself is seed-sensitive.

## What Is Supported

- Size-aware splitting keeps compactness active across all tested lambda values.
- Smaller `lambda_gb` reduces the catastrophic drops seen at `0.005`, especially on CiteSeer seed 2.
- `lambda_gb=0.0005` is safer than `0.005`, but not clearly beneficial.

## Reasonable Guess

The instability likely comes from using a fixed ball assignment after warm-up. If the warm-up representation is seed-sensitive, B1 may lock onto a poor local structure and keep pulling toward it.

## Not Yet Verified

- Whether refreshing balls during training (`B2`) improves stability.
- Whether a better stable encoder, such as BN-stabilized negative-free or InfoNCE, makes GB compactness useful.
- Whether a soft reliability weighting is better than the current binary reliable-ball mask.

## Weakened Direction

Simply lowering `lambda_gb` is not enough to make B1 robust. The next experiment should not continue only with smaller compactness weights.

## Final Decision

Modify before PubMed.

Recommended next experiment:

1. Test ball refresh (`B2`) with `lambda_gb=0.0005` and `0.001`.
2. Keep Cora/CiteSeer seeds 0/1/2.
3. Compare against B0 and existing B1 results.
4. Accept only if most dataset-seed pairs improve or remain within noise, and effective rank does not degrade materially.

## Evidence Files

- `C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs\gb_denoise_stage_b_compactness\results_lambda_sweep\stage_b_summary.csv`
- `C:\Users\Admin\Desktop\实验8去噪5.19-_stageb_fc62447_runs\gb_denoise_stage_b_compactness\results_lambda_sweep\stage_b_paired_deltas.csv`
