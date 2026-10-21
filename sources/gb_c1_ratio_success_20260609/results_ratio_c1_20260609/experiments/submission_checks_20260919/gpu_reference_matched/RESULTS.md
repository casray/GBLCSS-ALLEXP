# GPU Supplementary Results

Accuracy: mean +/- sample SD (%); parentheses: paired mean change from the freshly rerun default (percentage points).

| Arm | Cora | CiteSeer | CoraFull |
|---|---:|---:|---:|
| default | 57.52 +/- 0.71 (+0.00) | 55.24 +/- 2.04 (+0.00) | 37.22 +/- 0.66 (+0.00) |
| beta_0 | 55.13 +/- 1.88 (-2.39) | 52.94 +/- 3.67 (-2.30) | 37.11 +/- 0.36 (-0.10) |
| beta_0.25 | 56.44 +/- 1.49 (-1.08) | 54.47 +/- 2.63 (-0.77) | 37.36 +/- 0.44 (+0.14) |
| beta_0.75 | 57.59 +/- 1.16 (+0.07) | 56.36 +/- 1.83 (+1.12) | 37.02 +/- 1.34 (-0.19) |
| beta_1 | 57.56 +/- 1.11 (+0.04) | 56.98 +/- 2.07 (+1.74) | 37.08 +/- 1.18 (-0.14) |
| max_balls_32 | 57.55 +/- 1.09 (+0.03) | 56.26 +/- 1.93 (+1.02) | 35.12 +/- 0.61 (-2.10) |
| max_balls_128 | 57.52 +/- 0.71 (+0.00) | 55.46 +/- 2.07 (+0.21) | 38.79 +/- 0.14 (+1.57) |
| min_train_per_ball_5 | 56.09 +/- 1.43 (-1.42) | 54.74 +/- 2.47 (-0.50) | 37.24 +/- 0.67 (+0.02) |
| min_train_per_ball_10 | 50.25 +/- 1.48 (-7.27) | 50.02 +/- 1.92 (-5.22) | 37.12 +/- 0.65 (-0.10) |
| min_ball_size_1 | 55.87 +/- 1.71 (-1.65) | 55.94 +/- 1.80 (+0.70) | 37.17 +/- 0.66 (-0.05) |
| min_ball_size_16 | 56.58 +/- 1.53 (-0.94) | 56.49 +/- 1.18 (+1.25) | 36.92 +/- 0.59 (-0.30) |
| min_ball_size_64 | 57.45 +/- 1.30 (-0.07) | 56.23 +/- 2.78 (+0.99) | 37.60 +/- 0.57 (+0.38) |
| split_improvement_0.01 | 57.52 +/- 0.71 (+0.00) | 55.24 +/- 2.04 (+0.00) | 37.22 +/- 0.66 (+0.00) |
| split_improvement_0.1 | 44.37 +/- 2.31 (-13.15) | 42.90 +/- 2.50 (-12.34) | 31.86 +/- 1.05 (-5.36) |
| mlp_ce | 44.24 +/- 2.62 (-13.28) | 43.50 +/- 3.02 (-11.74) | 31.98 +/- 0.98 (-5.24) |
| shuffled | 43.31 +/- 1.40 (-14.21) | 42.00 +/- 2.91 (-13.24) | 32.30 +/- 0.85 (-4.91) |
| raw_x | 44.37 +/- 2.31 (-13.15) | 42.90 +/- 2.50 (-12.34) | 31.86 +/- 1.05 (-5.36) |
| a2x | 44.37 +/- 2.31 (-13.15) | 42.90 +/- 2.50 (-12.34) | 31.86 +/- 1.05 (-5.36) |

## Checks

- Logical evaluations: 270.
- Actual GPU training runs: 214.
- Exact-target reuse: 56; not independent repeats.
- Maximum default Brier deviation from archive: 6.92e-08.
- All devices, pair fingerprints, stored targets, losses and predictions passed checks.
- All 15 raw-X / A^2X pairs have one ball and identical targets.
- Original algorithm/config files still match the pre-run snapshot.
- Accuracy differences from archived training runs are listed separately in archive_comparison.csv.
