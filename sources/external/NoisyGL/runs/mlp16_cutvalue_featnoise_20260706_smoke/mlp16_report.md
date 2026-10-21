# MLP-16 Cut Value + Feature Noise Report

## Material Passport
- Role: writing-stage mechanism quantification and graph-free-family feature-noise defense.
- Protocol: train-only uniform label noise, rate 0.4, percent split 0.2/0.1/0.7.
- P0 Cut Value is post-hoc only; P1 perturbs test features only.
- Trained/evaluated model rows: 8
- Missing model groups: 0
- Alpha=0 anchor failures: 0

## P0 Cut Value Aggregate
| Dataset | Method | n | CV mean | CV std |
|---|---|---:|---:|---:|
| cora | Ours beta=0.5 train-only | 1 | 0.6262 | 0.0000 |
| cora | Shuffled local consensus | 1 | 0.5154 | 0.0000 |
| cora | kNN@matched beta=0.5 | 1 | 0.6213 | 0.0000 |
| cora | MLP+CE | 1 | 0.9725 | 0.0000 |
| cora | GLNN-style T1 | 1 | 0.9085 | 0.0000 |
| cora | Ours-distill T3 | 1 | 0.9093 | 0.0000 |

## P0 Pre-registered Comparisons
| Comparison | Dataset | n | Delta | Pos |
|---|---|---:|---:|---:|
| cv_ours_minus_ce | cora | 1 | -0.3463 | 0/1 |
| cv_ours_minus_shuffled | cora | 1 | 0.1108 | 1/1 |
| cv_ours_minus_knn | cora | 1 | 0.0049 | 1/1 |
| cv_distill_minus_ours | cora | 1 | 0.2830 | 1/1 |
| cv_distill_minus_glnn | cora | 1 | 0.0008 | 1/1 |

## P1 Feature Noise Curves
| Dataset | Method | alpha | n | Test mean | Test std |
|---|---|---:|---:|---:|---:|
| cora | Ours beta=0.5 train-only | 0.0 | 1 | 55.46 | 0.00 |
| cora | Ours beta=0.5 train-only | 0.1 | 1 | 55.36 | 0.00 |
| cora | Ours beta=0.5 train-only | 0.3 | 1 | 53.09 | 0.00 |
| cora | Ours beta=0.5 train-only | 0.5 | 1 | 43.43 | 0.00 |
| cora | Ours beta=0.5 train-only | 0.7 | 1 | 32.30 | 0.00 |
| cora | Ours beta=0.5 train-only | 0.9 | 1 | 22.53 | 0.00 |
| cora | Shuffled local consensus | 0.0 | 1 | 43.85 | 0.00 |
| cora | Shuffled local consensus | 0.1 | 1 | 43.80 | 0.00 |
| cora | Shuffled local consensus | 0.3 | 1 | 41.27 | 0.00 |
| cora | Shuffled local consensus | 0.5 | 1 | 34.41 | 0.00 |
| cora | Shuffled local consensus | 0.7 | 1 | 27.92 | 0.00 |
| cora | Shuffled local consensus | 0.9 | 1 | 20.63 | 0.00 |
| cora | MLP+CE | 0.0 | 1 | 41.74 | 0.00 |
| cora | MLP+CE | 0.1 | 1 | 41.42 | 0.00 |
| cora | MLP+CE | 0.3 | 1 | 38.84 | 0.00 |
| cora | MLP+CE | 0.5 | 1 | 34.30 | 0.00 |
| cora | MLP+CE | 0.7 | 1 | 26.81 | 0.00 |
| cora | MLP+CE | 0.9 | 1 | 20.42 | 0.00 |
| cora | JoCoR | 0.0 | 1 | 50.82 | 0.00 |
| cora | JoCoR | 0.1 | 1 | 50.50 | 0.00 |
| cora | JoCoR | 0.3 | 1 | 46.60 | 0.00 |
| cora | JoCoR | 0.5 | 1 | 38.58 | 0.00 |
| cora | JoCoR | 0.7 | 1 | 28.50 | 0.00 |
| cora | JoCoR | 0.9 | 1 | 21.21 | 0.00 |
| cora | Small-loss | 0.0 | 1 | 49.50 | 0.00 |
| cora | Small-loss | 0.1 | 1 | 49.02 | 0.00 |
| cora | Small-loss | 0.3 | 1 | 43.85 | 0.00 |
| cora | Small-loss | 0.5 | 1 | 36.15 | 0.00 |
| cora | Small-loss | 0.7 | 1 | 28.07 | 0.00 |
| cora | Small-loss | 0.9 | 1 | 18.79 | 0.00 |

## P1 Drop From alpha=0 to alpha=0.9
| Dataset | Method | n | Drop | std |
|---|---|---:|---:|---:|
| cora | Ours beta=0.5 train-only | 1 | 32.93pp | 0.00 |
| cora | MLP+CE | 1 | 21.32pp | 0.00 |
| cora | Small-loss | 1 | 30.71pp | 0.00 |
| cora | JoCoR | 1 | 29.60pp | 0.00 |
| cora | Shuffled local consensus | 1 | 23.22pp | 0.00 |

## P1 Ours vs Baselines by alpha
| Dataset | Baseline | alpha | n | Delta | Pos |
|---|---|---:|---:|---:|---:|
| cora | mlp_ce | 0.0 | 1 | 13.72pp | 1/1 |
| cora | mlp_small_loss | 0.0 | 1 | 5.96pp | 1/1 |
| cora | mlp_jocor | 0.0 | 1 | 4.64pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 1 | 11.61pp | 1/1 |
| cora | mlp_ce | 0.1 | 1 | 13.93pp | 1/1 |
| cora | mlp_small_loss | 0.1 | 1 | 6.33pp | 1/1 |
| cora | mlp_jocor | 0.1 | 1 | 4.85pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 1 | 11.56pp | 1/1 |
| cora | mlp_ce | 0.3 | 1 | 14.25pp | 1/1 |
| cora | mlp_small_loss | 0.3 | 1 | 9.23pp | 1/1 |
| cora | mlp_jocor | 0.3 | 1 | 6.49pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 1 | 11.82pp | 1/1 |
| cora | mlp_ce | 0.5 | 1 | 9.13pp | 1/1 |
| cora | mlp_small_loss | 0.5 | 1 | 7.28pp | 1/1 |
| cora | mlp_jocor | 0.5 | 1 | 4.85pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 1 | 9.02pp | 1/1 |
| cora | mlp_ce | 0.7 | 1 | 5.49pp | 1/1 |
| cora | mlp_small_loss | 0.7 | 1 | 4.22pp | 1/1 |
| cora | mlp_jocor | 0.7 | 1 | 3.80pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 1 | 4.38pp | 1/1 |
| cora | mlp_ce | 0.9 | 1 | 2.11pp | 1/1 |
| cora | mlp_small_loss | 0.9 | 1 | 3.75pp | 1/1 |
| cora | mlp_jocor | 0.9 | 1 | 1.32pp | 1/1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 1 | 1.90pp | 1/1 |

## Alpha=0 Anchor Check
| Dataset | Method | seed | alpha0 | anchor | abs diff | pass |
|---|---|---:|---:|---:|---:|---:|
| cora | frozengrace_gb_mlp_softdist_beta05 | 0 | 55.46 | 55.46 | 0.000e+00 | 1 |
| cora | mlp_ce | 0 | 41.74 | 41.74 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 43.85 | 43.85 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 0 | 49.50 | 49.50 | 0.000e+00 | 1 |
| cora | mlp_jocor | 0 | 50.82 | 50.82 | 0.000e+00 | 1 |

## Completeness
| Dataset | Method | expected | observed | missing |
|---|---|---:|---:|---|
| cora | frozengrace_gb_mlp_softdist_beta05 | 1 | 1 |  |
| cora | mlp_ce | 1 | 1 |  |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 1 |  |
| cora | knn_mlp_softdist_beta05 | 1 | 1 |  |
| cora | mlp_distill_softdist_xh | 1 | 1 |  |
| cora | mlp_distill_gcn | 1 | 1 |  |
| cora | mlp_small_loss | 1 | 1 |  |
| cora | mlp_jocor | 1 | 1 |  |
