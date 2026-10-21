# MLP-16 Cut Value + Feature Noise Report

## Material Passport
- Role: writing-stage mechanism quantification and graph-free-family feature-noise defense.
- Protocol: train-only uniform label noise, rate 0.4, percent split 0.2/0.1/0.7.
- P0 Cut Value is post-hoc only; P1 perturbs test features only.
- Trained/evaluated model rows: 180
- Missing model groups: 0
- Alpha=0 anchor failures: 0

## Figures
- Feature-noise curves: `mlp16_feature_noise_curves.png`

## P0 Cut Value Aggregate
| Dataset | Method | n | CV mean | CV std |
|---|---|---:|---:|---:|
| amazonpho | Ours beta=0.5 train-only | 5 | 0.8120 | 0.0169 |
| amazonpho | Shuffled local consensus | 5 | 0.8408 | 0.0154 |
| amazonpho | kNN@matched beta=0.5 | 5 | 0.8178 | 0.0200 |
| amazonpho | MLP+CE | 5 | 0.8135 | 0.0070 |
| citeseer | Ours beta=0.5 train-only | 5 | 0.7042 | 0.0431 |
| citeseer | Shuffled local consensus | 5 | 0.6566 | 0.1862 |
| citeseer | kNN@matched beta=0.5 | 5 | 0.7537 | 0.0712 |
| citeseer | MLP+CE | 5 | 0.6648 | 0.2764 |
| citeseer | GLNN-style T1 | 5 | 0.9509 | 0.0128 |
| citeseer | Ours-distill T3 | 5 | 0.9502 | 0.0034 |
| coauthorcs | Ours beta=0.5 train-only | 5 | 0.7623 | 0.0101 |
| coauthorcs | Shuffled local consensus | 5 | 0.7402 | 0.0208 |
| coauthorcs | kNN@matched beta=0.5 | 5 | 0.7691 | 0.0129 |
| coauthorcs | MLP+CE | 5 | 0.7293 | 0.0133 |
| cora | Ours beta=0.5 train-only | 5 | 0.6674 | 0.0927 |
| cora | Shuffled local consensus | 5 | 0.7187 | 0.2478 |
| cora | kNN@matched beta=0.5 | 5 | 0.7200 | 0.1462 |
| cora | MLP+CE | 5 | 0.8476 | 0.2423 |
| cora | GLNN-style T1 | 5 | 0.9056 | 0.0112 |
| cora | Ours-distill T3 | 5 | 0.9132 | 0.0058 |
| pubmed | Ours beta=0.5 train-only | 5 | 0.8401 | 0.0773 |
| pubmed | Shuffled local consensus | 5 | 0.9103 | 0.1521 |
| pubmed | kNN@matched beta=0.5 | 5 | 0.8134 | 0.0691 |
| pubmed | MLP+CE | 5 | 0.8722 | 0.2003 |
| pubmed | GLNN-style T1 | 5 | 0.9795 | 0.0037 |
| pubmed | Ours-distill T3 | 5 | 0.9789 | 0.0032 |

## P0 Pre-registered Comparisons
| Comparison | Dataset | n | Delta | Pos |
|---|---|---:|---:|---:|
| cv_ours_minus_ce | cora | 5 | -0.1803 | 1/5 |
| cv_ours_minus_shuffled | cora | 5 | -0.0513 | 3/5 |
| cv_ours_minus_knn | cora | 5 | -0.0526 | 1/5 |
| cv_distill_minus_ours | cora | 5 | 0.2458 | 5/5 |
| cv_distill_minus_glnn | cora | 5 | 0.0076 | 4/5 |
| cv_ours_minus_ce | citeseer | 5 | 0.0394 | 3/5 |
| cv_ours_minus_shuffled | citeseer | 5 | 0.0477 | 4/5 |
| cv_ours_minus_knn | citeseer | 5 | -0.0495 | 1/5 |
| cv_distill_minus_ours | citeseer | 5 | 0.2460 | 5/5 |
| cv_distill_minus_glnn | citeseer | 5 | -0.0006 | 4/5 |
| cv_ours_minus_ce | pubmed | 5 | -0.0320 | 2/5 |
| cv_ours_minus_shuffled | pubmed | 5 | -0.0702 | 1/5 |
| cv_ours_minus_knn | pubmed | 5 | 0.0267 | 3/5 |
| cv_distill_minus_ours | pubmed | 5 | 0.1388 | 5/5 |
| cv_distill_minus_glnn | pubmed | 5 | -0.0005 | 2/5 |
| cv_ours_minus_ce | coauthorcs | 5 | 0.0329 | 5/5 |
| cv_ours_minus_shuffled | coauthorcs | 5 | 0.0221 | 4/5 |
| cv_ours_minus_knn | coauthorcs | 5 | -0.0068 | 1/5 |
| cv_ours_minus_ce | amazonpho | 5 | -0.0015 | 2/5 |
| cv_ours_minus_shuffled | amazonpho | 5 | -0.0288 | 0/5 |
| cv_ours_minus_knn | amazonpho | 5 | -0.0058 | 1/5 |

## P1 Feature Noise Curves
| Dataset | Method | alpha | n | Test mean | Test std |
|---|---|---:|---:|---:|---:|
| amazonpho | Ours beta=0.5 train-only | 0.0 | 5 | 81.59 | 0.91 |
| amazonpho | Ours beta=0.5 train-only | 0.1 | 5 | 80.32 | 1.08 |
| amazonpho | Ours beta=0.5 train-only | 0.3 | 5 | 74.65 | 2.78 |
| amazonpho | Ours beta=0.5 train-only | 0.5 | 5 | 63.82 | 4.03 |
| amazonpho | Ours beta=0.5 train-only | 0.7 | 5 | 45.93 | 2.40 |
| amazonpho | Ours beta=0.5 train-only | 0.9 | 5 | 26.81 | 1.09 |
| amazonpho | Shuffled local consensus | 0.0 | 5 | 72.06 | 2.97 |
| amazonpho | Shuffled local consensus | 0.1 | 5 | 70.29 | 3.71 |
| amazonpho | Shuffled local consensus | 0.3 | 5 | 64.90 | 5.37 |
| amazonpho | Shuffled local consensus | 0.5 | 5 | 54.31 | 5.65 |
| amazonpho | Shuffled local consensus | 0.7 | 5 | 39.51 | 3.09 |
| amazonpho | Shuffled local consensus | 0.9 | 5 | 25.39 | 0.96 |
| amazonpho | MLP+CE | 0.0 | 5 | 73.90 | 1.80 |
| amazonpho | MLP+CE | 0.1 | 5 | 72.34 | 2.34 |
| amazonpho | MLP+CE | 0.3 | 5 | 66.64 | 3.27 |
| amazonpho | MLP+CE | 0.5 | 5 | 56.94 | 3.29 |
| amazonpho | MLP+CE | 0.7 | 5 | 41.43 | 2.23 |
| amazonpho | MLP+CE | 0.9 | 5 | 25.47 | 1.63 |
| amazonpho | JoCoR | 0.0 | 5 | 79.57 | 0.53 |
| amazonpho | JoCoR | 0.1 | 5 | 78.60 | 0.82 |
| amazonpho | JoCoR | 0.3 | 5 | 74.07 | 1.41 |
| amazonpho | JoCoR | 0.5 | 5 | 62.84 | 2.12 |
| amazonpho | JoCoR | 0.7 | 5 | 44.77 | 1.86 |
| amazonpho | JoCoR | 0.9 | 5 | 25.85 | 1.50 |
| amazonpho | Small-loss | 0.0 | 5 | 77.25 | 1.62 |
| amazonpho | Small-loss | 0.1 | 5 | 75.94 | 2.02 |
| amazonpho | Small-loss | 0.3 | 5 | 71.31 | 2.53 |
| amazonpho | Small-loss | 0.5 | 5 | 61.27 | 2.32 |
| amazonpho | Small-loss | 0.7 | 5 | 44.68 | 2.02 |
| amazonpho | Small-loss | 0.9 | 5 | 25.99 | 1.70 |
| citeseer | Ours beta=0.5 train-only | 0.0 | 5 | 53.83 | 2.75 |
| citeseer | Ours beta=0.5 train-only | 0.1 | 5 | 53.83 | 2.51 |
| citeseer | Ours beta=0.5 train-only | 0.3 | 5 | 51.38 | 2.34 |
| citeseer | Ours beta=0.5 train-only | 0.5 | 5 | 44.53 | 2.20 |
| citeseer | Ours beta=0.5 train-only | 0.7 | 5 | 32.54 | 1.70 |
| citeseer | Ours beta=0.5 train-only | 0.9 | 5 | 21.16 | 0.58 |
| citeseer | Shuffled local consensus | 0.0 | 5 | 42.35 | 2.74 |
| citeseer | Shuffled local consensus | 0.1 | 5 | 41.88 | 2.69 |
| citeseer | Shuffled local consensus | 0.3 | 5 | 39.78 | 2.56 |
| citeseer | Shuffled local consensus | 0.5 | 5 | 34.01 | 2.11 |
| citeseer | Shuffled local consensus | 0.7 | 5 | 26.65 | 2.00 |
| citeseer | Shuffled local consensus | 0.9 | 5 | 19.79 | 0.56 |
| citeseer | MLP+CE | 0.0 | 5 | 43.50 | 3.02 |
| citeseer | MLP+CE | 0.1 | 5 | 42.69 | 3.04 |
| citeseer | MLP+CE | 0.3 | 5 | 39.24 | 2.66 |
| citeseer | MLP+CE | 0.5 | 5 | 33.61 | 3.40 |
| citeseer | MLP+CE | 0.7 | 5 | 26.52 | 1.79 |
| citeseer | MLP+CE | 0.9 | 5 | 19.74 | 0.83 |
| citeseer | JoCoR | 0.0 | 5 | 49.91 | 3.37 |
| citeseer | JoCoR | 0.1 | 5 | 49.72 | 3.64 |
| citeseer | JoCoR | 0.3 | 5 | 46.92 | 3.79 |
| citeseer | JoCoR | 0.5 | 5 | 40.35 | 4.08 |
| citeseer | JoCoR | 0.7 | 5 | 30.52 | 1.81 |
| citeseer | JoCoR | 0.9 | 5 | 20.73 | 0.22 |
| citeseer | Small-loss | 0.0 | 5 | 49.70 | 3.07 |
| citeseer | Small-loss | 0.1 | 5 | 49.24 | 3.58 |
| citeseer | Small-loss | 0.3 | 5 | 46.39 | 3.60 |
| citeseer | Small-loss | 0.5 | 5 | 40.34 | 3.68 |
| citeseer | Small-loss | 0.7 | 5 | 30.21 | 2.12 |
| citeseer | Small-loss | 0.9 | 5 | 20.87 | 0.75 |
| coauthorcs | Ours beta=0.5 train-only | 0.0 | 5 | 90.62 | 0.57 |
| coauthorcs | Ours beta=0.5 train-only | 0.1 | 5 | 89.59 | 0.91 |
| coauthorcs | Ours beta=0.5 train-only | 0.3 | 5 | 83.99 | 2.40 |
| coauthorcs | Ours beta=0.5 train-only | 0.5 | 5 | 72.12 | 3.49 |
| coauthorcs | Ours beta=0.5 train-only | 0.7 | 5 | 55.30 | 2.64 |
| coauthorcs | Ours beta=0.5 train-only | 0.9 | 5 | 31.14 | 0.82 |
| coauthorcs | Shuffled local consensus | 0.0 | 5 | 87.60 | 1.04 |
| coauthorcs | Shuffled local consensus | 0.1 | 5 | 86.10 | 1.57 |
| coauthorcs | Shuffled local consensus | 0.3 | 5 | 80.26 | 3.02 |
| coauthorcs | Shuffled local consensus | 0.5 | 5 | 69.46 | 4.01 |
| coauthorcs | Shuffled local consensus | 0.7 | 5 | 53.35 | 3.46 |
| coauthorcs | Shuffled local consensus | 0.9 | 5 | 29.14 | 0.64 |
| coauthorcs | MLP+CE | 0.0 | 5 | 88.42 | 0.62 |
| coauthorcs | MLP+CE | 0.1 | 5 | 87.41 | 0.86 |
| coauthorcs | MLP+CE | 0.3 | 5 | 82.53 | 2.02 |
| coauthorcs | MLP+CE | 0.5 | 5 | 72.16 | 2.27 |
| coauthorcs | MLP+CE | 0.7 | 5 | 54.98 | 2.13 |
| coauthorcs | MLP+CE | 0.9 | 5 | 28.14 | 1.15 |
| coauthorcs | JoCoR | 0.0 | 5 | 89.38 | 0.61 |
| coauthorcs | JoCoR | 0.1 | 5 | 88.76 | 0.78 |
| coauthorcs | JoCoR | 0.3 | 5 | 85.25 | 1.53 |
| coauthorcs | JoCoR | 0.5 | 5 | 75.17 | 2.51 |
| coauthorcs | JoCoR | 0.7 | 5 | 55.87 | 3.34 |
| coauthorcs | JoCoR | 0.9 | 5 | 29.63 | 1.10 |
| coauthorcs | Small-loss | 0.0 | 5 | 89.06 | 0.88 |
| coauthorcs | Small-loss | 0.1 | 5 | 88.30 | 1.02 |
| coauthorcs | Small-loss | 0.3 | 5 | 84.41 | 1.92 |
| coauthorcs | Small-loss | 0.5 | 5 | 73.99 | 3.26 |
| coauthorcs | Small-loss | 0.7 | 5 | 55.25 | 4.03 |
| coauthorcs | Small-loss | 0.9 | 5 | 29.22 | 1.33 |
| cora | Ours beta=0.5 train-only | 0.0 | 5 | 55.59 | 1.50 |
| cora | Ours beta=0.5 train-only | 0.1 | 5 | 55.50 | 1.67 |
| cora | Ours beta=0.5 train-only | 0.3 | 5 | 53.23 | 1.52 |
| cora | Ours beta=0.5 train-only | 0.5 | 5 | 45.07 | 2.55 |
| cora | Ours beta=0.5 train-only | 0.7 | 5 | 32.37 | 1.38 |
| cora | Ours beta=0.5 train-only | 0.9 | 5 | 21.79 | 1.12 |
| cora | Shuffled local consensus | 0.0 | 5 | 43.63 | 1.62 |
| cora | Shuffled local consensus | 0.1 | 5 | 43.40 | 1.78 |
| cora | Shuffled local consensus | 0.3 | 5 | 40.82 | 2.24 |
| cora | Shuffled local consensus | 0.5 | 5 | 35.32 | 1.65 |
| cora | Shuffled local consensus | 0.7 | 5 | 27.91 | 1.93 |
| cora | Shuffled local consensus | 0.9 | 5 | 21.17 | 2.02 |
| cora | MLP+CE | 0.0 | 5 | 44.24 | 2.62 |
| cora | MLP+CE | 0.1 | 5 | 43.76 | 2.42 |
| cora | MLP+CE | 0.3 | 5 | 41.79 | 2.15 |
| cora | MLP+CE | 0.5 | 5 | 36.01 | 2.58 |
| cora | MLP+CE | 0.7 | 5 | 28.08 | 1.97 |
| cora | MLP+CE | 0.9 | 5 | 21.42 | 1.42 |
| cora | JoCoR | 0.0 | 5 | 51.72 | 1.30 |
| cora | JoCoR | 0.1 | 5 | 51.65 | 1.80 |
| cora | JoCoR | 0.3 | 5 | 48.37 | 1.83 |
| cora | JoCoR | 0.5 | 5 | 40.84 | 2.27 |
| cora | JoCoR | 0.7 | 5 | 31.21 | 1.87 |
| cora | JoCoR | 0.9 | 5 | 21.89 | 1.90 |
| cora | Small-loss | 0.0 | 5 | 51.95 | 1.97 |
| cora | Small-loss | 0.1 | 5 | 51.58 | 2.09 |
| cora | Small-loss | 0.3 | 5 | 48.15 | 2.69 |
| cora | Small-loss | 0.5 | 5 | 41.61 | 3.61 |
| cora | Small-loss | 0.7 | 5 | 30.84 | 2.07 |
| cora | Small-loss | 0.9 | 5 | 20.58 | 1.88 |
| pubmed | Ours beta=0.5 train-only | 0.0 | 5 | 71.82 | 0.91 |
| pubmed | Ours beta=0.5 train-only | 0.1 | 5 | 70.58 | 1.21 |
| pubmed | Ours beta=0.5 train-only | 0.3 | 5 | 65.08 | 2.34 |
| pubmed | Ours beta=0.5 train-only | 0.5 | 5 | 57.44 | 2.25 |
| pubmed | Ours beta=0.5 train-only | 0.7 | 5 | 48.47 | 1.04 |
| pubmed | Ours beta=0.5 train-only | 0.9 | 5 | 40.26 | 1.52 |
| pubmed | Shuffled local consensus | 0.0 | 5 | 62.71 | 4.51 |
| pubmed | Shuffled local consensus | 0.1 | 5 | 62.03 | 4.37 |
| pubmed | Shuffled local consensus | 0.3 | 5 | 58.95 | 3.97 |
| pubmed | Shuffled local consensus | 0.5 | 5 | 54.05 | 3.70 |
| pubmed | Shuffled local consensus | 0.7 | 5 | 47.58 | 2.97 |
| pubmed | Shuffled local consensus | 0.9 | 5 | 41.02 | 2.14 |
| pubmed | MLP+CE | 0.0 | 5 | 63.77 | 4.40 |
| pubmed | MLP+CE | 0.1 | 5 | 62.77 | 4.41 |
| pubmed | MLP+CE | 0.3 | 5 | 59.07 | 4.88 |
| pubmed | MLP+CE | 0.5 | 5 | 53.55 | 4.66 |
| pubmed | MLP+CE | 0.7 | 5 | 47.42 | 3.62 |
| pubmed | MLP+CE | 0.9 | 5 | 40.69 | 2.36 |
| pubmed | JoCoR | 0.0 | 5 | 77.84 | 1.10 |
| pubmed | JoCoR | 0.1 | 5 | 77.59 | 0.95 |
| pubmed | JoCoR | 0.3 | 5 | 75.06 | 1.13 |
| pubmed | JoCoR | 0.5 | 5 | 67.30 | 0.83 |
| pubmed | JoCoR | 0.7 | 5 | 54.48 | 0.62 |
| pubmed | JoCoR | 0.9 | 5 | 41.66 | 0.48 |
| pubmed | Small-loss | 0.0 | 5 | 77.30 | 1.29 |
| pubmed | Small-loss | 0.1 | 5 | 77.07 | 1.44 |
| pubmed | Small-loss | 0.3 | 5 | 74.22 | 0.93 |
| pubmed | Small-loss | 0.5 | 5 | 66.43 | 0.97 |
| pubmed | Small-loss | 0.7 | 5 | 53.34 | 0.63 |
| pubmed | Small-loss | 0.9 | 5 | 41.24 | 1.05 |

## P1 Drop From alpha=0 to alpha=0.9
| Dataset | Method | n | Drop | std |
|---|---|---:|---:|---:|
| cora | Ours beta=0.5 train-only | 5 | 33.79pp | 1.49 |
| cora | MLP+CE | 5 | 22.82pp | 2.27 |
| cora | Small-loss | 5 | 31.37pp | 1.25 |
| cora | JoCoR | 5 | 29.83pp | 1.66 |
| cora | Shuffled local consensus | 5 | 22.46pp | 2.55 |
| citeseer | Ours beta=0.5 train-only | 5 | 32.67pp | 2.99 |
| citeseer | MLP+CE | 5 | 23.75pp | 2.46 |
| citeseer | Small-loss | 5 | 28.83pp | 2.52 |
| citeseer | JoCoR | 5 | 29.18pp | 3.32 |
| citeseer | Shuffled local consensus | 5 | 22.57pp | 2.20 |
| pubmed | Ours beta=0.5 train-only | 5 | 31.56pp | 1.32 |
| pubmed | MLP+CE | 5 | 23.08pp | 3.99 |
| pubmed | Small-loss | 5 | 36.06pp | 0.92 |
| pubmed | JoCoR | 5 | 36.18pp | 0.73 |
| pubmed | Shuffled local consensus | 5 | 21.68pp | 3.62 |
| coauthorcs | Ours beta=0.5 train-only | 5 | 59.48pp | 0.81 |
| coauthorcs | MLP+CE | 5 | 60.28pp | 1.12 |
| coauthorcs | Small-loss | 5 | 59.84pp | 1.05 |
| coauthorcs | JoCoR | 5 | 59.75pp | 1.06 |
| coauthorcs | Shuffled local consensus | 5 | 58.46pp | 1.25 |
| amazonpho | Ours beta=0.5 train-only | 5 | 54.79pp | 1.69 |
| amazonpho | MLP+CE | 5 | 48.43pp | 2.89 |
| amazonpho | Small-loss | 5 | 51.25pp | 2.62 |
| amazonpho | JoCoR | 5 | 53.72pp | 1.58 |
| amazonpho | Shuffled local consensus | 5 | 46.67pp | 3.30 |

## P1 Ours vs Baselines by alpha
| Dataset | Baseline | alpha | n | Delta | Pos |
|---|---|---:|---:|---:|---:|
| cora | mlp_ce | 0.0 | 5 | 11.35pp | 5/5 |
| cora | mlp_small_loss | 0.0 | 5 | 3.64pp | 4/5 |
| cora | mlp_jocor | 0.0 | 5 | 3.87pp | 5/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 5 | 11.96pp | 5/5 |
| cora | mlp_ce | 0.1 | 5 | 11.75pp | 5/5 |
| cora | mlp_small_loss | 0.1 | 5 | 3.93pp | 5/5 |
| cora | mlp_jocor | 0.1 | 5 | 3.85pp | 5/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 5 | 12.11pp | 5/5 |
| cora | mlp_ce | 0.3 | 5 | 11.44pp | 5/5 |
| cora | mlp_small_loss | 0.3 | 5 | 5.09pp | 5/5 |
| cora | mlp_jocor | 0.3 | 5 | 4.87pp | 5/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 5 | 12.41pp | 5/5 |
| cora | mlp_ce | 0.5 | 5 | 9.06pp | 5/5 |
| cora | mlp_small_loss | 0.5 | 5 | 3.45pp | 5/5 |
| cora | mlp_jocor | 0.5 | 5 | 4.22pp | 5/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 5 | 9.74pp | 5/5 |
| cora | mlp_ce | 0.7 | 5 | 4.28pp | 5/5 |
| cora | mlp_small_loss | 0.7 | 5 | 1.53pp | 4/5 |
| cora | mlp_jocor | 0.7 | 5 | 1.16pp | 3/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 5 | 4.46pp | 5/5 |
| cora | mlp_ce | 0.9 | 5 | 0.37pp | 3/5 |
| cora | mlp_small_loss | 0.9 | 5 | 1.21pp | 4/5 |
| cora | mlp_jocor | 0.9 | 5 | -0.09pp | 2/5 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 5 | 0.62pp | 3/5 |
| citeseer | mlp_ce | 0.0 | 5 | 10.34pp | 5/5 |
| citeseer | mlp_small_loss | 0.0 | 5 | 4.13pp | 5/5 |
| citeseer | mlp_jocor | 0.0 | 5 | 3.93pp | 5/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 5 | 11.48pp | 5/5 |
| citeseer | mlp_ce | 0.1 | 5 | 11.14pp | 5/5 |
| citeseer | mlp_small_loss | 0.1 | 5 | 4.60pp | 5/5 |
| citeseer | mlp_jocor | 0.1 | 5 | 4.12pp | 5/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 5 | 11.95pp | 5/5 |
| citeseer | mlp_ce | 0.3 | 5 | 12.14pp | 5/5 |
| citeseer | mlp_small_loss | 0.3 | 5 | 4.99pp | 5/5 |
| citeseer | mlp_jocor | 0.3 | 5 | 4.47pp | 5/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 5 | 11.61pp | 5/5 |
| citeseer | mlp_ce | 0.5 | 5 | 10.92pp | 5/5 |
| citeseer | mlp_small_loss | 0.5 | 5 | 4.18pp | 5/5 |
| citeseer | mlp_jocor | 0.5 | 5 | 4.18pp | 5/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 5 | 10.52pp | 5/5 |
| citeseer | mlp_ce | 0.7 | 5 | 6.02pp | 5/5 |
| citeseer | mlp_small_loss | 0.7 | 5 | 2.34pp | 4/5 |
| citeseer | mlp_jocor | 0.7 | 5 | 2.03pp | 4/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 5 | 5.89pp | 5/5 |
| citeseer | mlp_ce | 0.9 | 5 | 1.42pp | 4/5 |
| citeseer | mlp_small_loss | 0.9 | 5 | 0.29pp | 3/5 |
| citeseer | mlp_jocor | 0.9 | 5 | 0.43pp | 4/5 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 5 | 1.37pp | 5/5 |
| pubmed | mlp_ce | 0.0 | 5 | 8.05pp | 5/5 |
| pubmed | mlp_small_loss | 0.0 | 5 | -5.49pp | 0/5 |
| pubmed | mlp_jocor | 0.0 | 5 | -6.02pp | 0/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 5 | 9.11pp | 5/5 |
| pubmed | mlp_ce | 0.1 | 5 | 7.81pp | 5/5 |
| pubmed | mlp_small_loss | 0.1 | 5 | -6.49pp | 0/5 |
| pubmed | mlp_jocor | 0.1 | 5 | -7.02pp | 0/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 5 | 8.55pp | 5/5 |
| pubmed | mlp_ce | 0.3 | 5 | 6.01pp | 4/5 |
| pubmed | mlp_small_loss | 0.3 | 5 | -9.14pp | 0/5 |
| pubmed | mlp_jocor | 0.3 | 5 | -9.98pp | 0/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 5 | 6.13pp | 4/5 |
| pubmed | mlp_ce | 0.5 | 5 | 3.89pp | 4/5 |
| pubmed | mlp_small_loss | 0.5 | 5 | -8.99pp | 0/5 |
| pubmed | mlp_jocor | 0.5 | 5 | -9.86pp | 0/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 5 | 3.39pp | 4/5 |
| pubmed | mlp_ce | 0.7 | 5 | 1.05pp | 2/5 |
| pubmed | mlp_small_loss | 0.7 | 5 | -4.87pp | 0/5 |
| pubmed | mlp_jocor | 0.7 | 5 | -6.01pp | 0/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 5 | 0.89pp | 3/5 |
| pubmed | mlp_ce | 0.9 | 5 | -0.43pp | 2/5 |
| pubmed | mlp_small_loss | 0.9 | 5 | -0.99pp | 1/5 |
| pubmed | mlp_jocor | 0.9 | 5 | -1.40pp | 1/5 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 5 | -0.77pp | 2/5 |
| coauthorcs | mlp_ce | 0.0 | 5 | 2.19pp | 5/5 |
| coauthorcs | mlp_small_loss | 0.0 | 5 | 1.55pp | 5/5 |
| coauthorcs | mlp_jocor | 0.0 | 5 | 1.23pp | 5/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 5 | 3.02pp | 5/5 |
| coauthorcs | mlp_ce | 0.1 | 5 | 2.18pp | 5/5 |
| coauthorcs | mlp_small_loss | 0.1 | 5 | 1.29pp | 5/5 |
| coauthorcs | mlp_jocor | 0.1 | 5 | 0.83pp | 5/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 5 | 3.49pp | 5/5 |
| coauthorcs | mlp_ce | 0.3 | 5 | 1.45pp | 4/5 |
| coauthorcs | mlp_small_loss | 0.3 | 5 | -0.42pp | 2/5 |
| coauthorcs | mlp_jocor | 0.3 | 5 | -1.26pp | 1/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 5 | 3.73pp | 5/5 |
| coauthorcs | mlp_ce | 0.5 | 5 | -0.04pp | 2/5 |
| coauthorcs | mlp_small_loss | 0.5 | 5 | -1.87pp | 2/5 |
| coauthorcs | mlp_jocor | 0.5 | 5 | -3.05pp | 1/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 5 | 2.65pp | 4/5 |
| coauthorcs | mlp_ce | 0.7 | 5 | 0.32pp | 3/5 |
| coauthorcs | mlp_small_loss | 0.7 | 5 | 0.05pp | 2/5 |
| coauthorcs | mlp_jocor | 0.7 | 5 | -0.57pp | 2/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 5 | 1.95pp | 3/5 |
| coauthorcs | mlp_ce | 0.9 | 5 | 3.00pp | 5/5 |
| coauthorcs | mlp_small_loss | 0.9 | 5 | 1.92pp | 4/5 |
| coauthorcs | mlp_jocor | 0.9 | 5 | 1.51pp | 4/5 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 5 | 2.00pp | 5/5 |
| amazonpho | mlp_ce | 0.0 | 5 | 7.69pp | 5/5 |
| amazonpho | mlp_small_loss | 0.0 | 5 | 4.35pp | 5/5 |
| amazonpho | mlp_jocor | 0.0 | 5 | 2.02pp | 5/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.0 | 5 | 9.54pp | 5/5 |
| amazonpho | mlp_ce | 0.1 | 5 | 7.98pp | 5/5 |
| amazonpho | mlp_small_loss | 0.1 | 5 | 4.38pp | 5/5 |
| amazonpho | mlp_jocor | 0.1 | 5 | 1.73pp | 4/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.1 | 5 | 10.04pp | 5/5 |
| amazonpho | mlp_ce | 0.3 | 5 | 8.00pp | 5/5 |
| amazonpho | mlp_small_loss | 0.3 | 5 | 3.34pp | 4/5 |
| amazonpho | mlp_jocor | 0.3 | 5 | 0.58pp | 3/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.3 | 5 | 9.74pp | 5/5 |
| amazonpho | mlp_ce | 0.5 | 5 | 6.88pp | 4/5 |
| amazonpho | mlp_small_loss | 0.5 | 5 | 2.54pp | 4/5 |
| amazonpho | mlp_jocor | 0.5 | 5 | 0.98pp | 3/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.5 | 5 | 9.51pp | 5/5 |
| amazonpho | mlp_ce | 0.7 | 5 | 4.50pp | 4/5 |
| amazonpho | mlp_small_loss | 0.7 | 5 | 1.25pp | 3/5 |
| amazonpho | mlp_jocor | 0.7 | 5 | 1.17pp | 2/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.7 | 5 | 6.43pp | 5/5 |
| amazonpho | mlp_ce | 0.9 | 5 | 1.34pp | 4/5 |
| amazonpho | mlp_small_loss | 0.9 | 5 | 0.81pp | 3/5 |
| amazonpho | mlp_jocor | 0.9 | 5 | 0.96pp | 4/5 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0.9 | 5 | 1.42pp | 4/5 |

## Alpha=0 Anchor Check
| Dataset | Method | seed | alpha0 | anchor | abs diff | pass |
|---|---|---:|---:|---:|---:|---:|
| cora | frozengrace_gb_mlp_softdist_beta05 | 0 | 55.46 | 55.46 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05 | 1 | 54.88 | 54.88 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05 | 2 | 54.62 | 54.62 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05 | 3 | 58.21 | 58.21 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05 | 4 | 54.78 | 54.78 | 0.000e+00 | 1 |
| cora | mlp_ce | 0 | 41.74 | 41.74 | 0.000e+00 | 1 |
| cora | mlp_ce | 1 | 44.96 | 44.96 | 0.000e+00 | 1 |
| cora | mlp_ce | 2 | 42.06 | 42.06 | 0.000e+00 | 1 |
| cora | mlp_ce | 3 | 48.23 | 48.23 | 0.000e+00 | 1 |
| cora | mlp_ce | 4 | 44.22 | 44.22 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 43.85 | 43.85 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 43.01 | 43.01 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 2 | 42.22 | 42.22 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 3 | 46.33 | 46.33 | 0.000e+00 | 1 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 4 | 42.74 | 42.74 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 0 | 49.50 | 49.50 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 1 | 51.13 | 51.13 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 2 | 54.62 | 54.62 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 3 | 53.14 | 53.14 | 0.000e+00 | 1 |
| cora | mlp_small_loss | 4 | 51.35 | 51.35 | 0.000e+00 | 1 |
| cora | mlp_jocor | 0 | 50.82 | 50.82 | 0.000e+00 | 1 |
| cora | mlp_jocor | 1 | 50.61 | 50.61 | 0.000e+00 | 1 |
| cora | mlp_jocor | 2 | 53.67 | 53.67 | 0.000e+00 | 1 |
| cora | mlp_jocor | 3 | 52.40 | 52.40 | 0.000e+00 | 1 |
| cora | mlp_jocor | 4 | 51.08 | 51.08 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 0 | 55.33 | 55.33 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 1 | 55.84 | 55.84 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 2 | 53.09 | 53.09 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 3 | 55.58 | 55.58 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 4 | 49.31 | 49.31 | 0.000e+00 | 1 |
| citeseer | mlp_ce | 0 | 46.09 | 46.09 | 0.000e+00 | 1 |
| citeseer | mlp_ce | 1 | 41.92 | 41.92 | 0.000e+00 | 1 |
| citeseer | mlp_ce | 2 | 43.77 | 43.77 | 0.000e+00 | 1 |
| citeseer | mlp_ce | 3 | 46.48 | 46.48 | 0.000e+00 | 1 |
| citeseer | mlp_ce | 4 | 39.22 | 39.22 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 42.23 | 42.23 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 40.59 | 40.59 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 2 | 44.16 | 44.16 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 3 | 45.83 | 45.83 | 0.000e+00 | 1 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 4 | 38.96 | 38.96 | 0.000e+00 | 1 |
| citeseer | mlp_small_loss | 0 | 49.05 | 49.05 | 0.000e+00 | 1 |
| citeseer | mlp_small_loss | 1 | 50.26 | 50.26 | 0.000e+00 | 1 |
| citeseer | mlp_small_loss | 2 | 49.79 | 49.79 | 0.000e+00 | 1 |
| citeseer | mlp_small_loss | 3 | 53.99 | 53.99 | 0.000e+00 | 1 |
| citeseer | mlp_small_loss | 4 | 45.40 | 45.40 | 0.000e+00 | 1 |
| citeseer | mlp_jocor | 0 | 47.59 | 47.59 | 0.000e+00 | 1 |
| citeseer | mlp_jocor | 1 | 52.28 | 52.28 | 0.000e+00 | 1 |
| citeseer | mlp_jocor | 2 | 50.77 | 50.77 | 0.000e+00 | 1 |
| citeseer | mlp_jocor | 3 | 53.52 | 53.52 | 0.000e+00 | 1 |
| citeseer | mlp_jocor | 4 | 45.36 | 45.36 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 0 | 71.60 | 71.60 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 1 | 72.05 | 72.05 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 2 | 72.49 | 72.49 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 3 | 72.60 | 72.60 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 4 | 70.36 | 70.36 | 0.000e+00 | 1 |
| pubmed | mlp_ce | 0 | 62.62 | 62.62 | 0.000e+00 | 1 |
| pubmed | mlp_ce | 1 | 71.47 | 71.47 | 0.000e+00 | 1 |
| pubmed | mlp_ce | 2 | 61.87 | 61.87 | 0.000e+00 | 1 |
| pubmed | mlp_ce | 3 | 62.53 | 62.53 | 0.000e+00 | 1 |
| pubmed | mlp_ce | 4 | 60.35 | 60.35 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 60.89 | 60.89 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 70.63 | 70.63 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 2 | 62.00 | 62.00 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 3 | 60.39 | 60.39 | 0.000e+00 | 1 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 4 | 59.63 | 59.63 | 0.000e+00 | 1 |
| pubmed | mlp_small_loss | 0 | 77.46 | 77.46 | 0.000e+00 | 1 |
| pubmed | mlp_small_loss | 1 | 77.02 | 77.02 | 0.000e+00 | 1 |
| pubmed | mlp_small_loss | 2 | 78.94 | 78.94 | 0.000e+00 | 1 |
| pubmed | mlp_small_loss | 3 | 77.74 | 77.74 | 0.000e+00 | 1 |
| pubmed | mlp_small_loss | 4 | 75.37 | 75.37 | 0.000e+00 | 1 |
| pubmed | mlp_jocor | 0 | 77.78 | 77.78 | 0.000e+00 | 1 |
| pubmed | mlp_jocor | 1 | 77.39 | 77.39 | 0.000e+00 | 1 |
| pubmed | mlp_jocor | 2 | 79.35 | 79.35 | 0.000e+00 | 1 |
| pubmed | mlp_jocor | 3 | 78.29 | 78.29 | 0.000e+00 | 1 |
| pubmed | mlp_jocor | 4 | 76.38 | 76.38 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 0 | 90.35 | 90.35 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 1 | 90.03 | 90.03 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 2 | 91.43 | 91.43 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 3 | 90.96 | 90.96 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 4 | 90.31 | 90.31 | 0.000e+00 | 1 |
| coauthorcs | mlp_ce | 0 | 88.48 | 88.48 | 0.000e+00 | 1 |
| coauthorcs | mlp_ce | 1 | 87.46 | 87.46 | 0.000e+00 | 1 |
| coauthorcs | mlp_ce | 2 | 88.44 | 88.44 | 0.000e+00 | 1 |
| coauthorcs | mlp_ce | 3 | 89.21 | 89.21 | 0.000e+00 | 1 |
| coauthorcs | mlp_ce | 4 | 88.51 | 88.51 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 87.84 | 87.84 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 85.89 | 85.89 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 2 | 87.47 | 87.47 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 3 | 88.40 | 88.40 | 0.000e+00 | 1 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 4 | 88.40 | 88.40 | 0.000e+00 | 1 |
| coauthorcs | mlp_small_loss | 0 | 88.90 | 88.90 | 0.000e+00 | 1 |
| coauthorcs | mlp_small_loss | 1 | 88.44 | 88.44 | 0.000e+00 | 1 |
| coauthorcs | mlp_small_loss | 2 | 90.16 | 90.16 | 0.000e+00 | 1 |
| coauthorcs | mlp_small_loss | 3 | 89.75 | 89.75 | 0.000e+00 | 1 |
| coauthorcs | mlp_small_loss | 4 | 88.07 | 88.07 | 0.000e+00 | 1 |
| coauthorcs | mlp_jocor | 0 | 89.17 | 89.17 | 0.000e+00 | 1 |
| coauthorcs | mlp_jocor | 1 | 88.64 | 88.64 | 0.000e+00 | 1 |
| coauthorcs | mlp_jocor | 2 | 90.06 | 90.06 | 0.000e+00 | 1 |
| coauthorcs | mlp_jocor | 3 | 89.96 | 89.96 | 0.000e+00 | 1 |
| coauthorcs | mlp_jocor | 4 | 89.08 | 89.08 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 0 | 81.44 | 81.44 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 1 | 81.03 | 81.03 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 2 | 80.63 | 80.63 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 3 | 81.89 | 81.89 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 4 | 82.99 | 82.99 | 0.000e+00 | 1 |
| amazonpho | mlp_ce | 0 | 74.49 | 74.49 | 0.000e+00 | 1 |
| amazonpho | mlp_ce | 1 | 74.96 | 74.96 | 0.000e+00 | 1 |
| amazonpho | mlp_ce | 2 | 71.02 | 71.02 | 0.000e+00 | 1 |
| amazonpho | mlp_ce | 3 | 75.63 | 75.63 | 0.000e+00 | 1 |
| amazonpho | mlp_ce | 4 | 73.41 | 73.41 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 0 | 73.02 | 73.02 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 1 | 73.97 | 73.97 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 2 | 71.34 | 71.34 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 3 | 74.72 | 74.72 | 0.000e+00 | 1 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 4 | 67.25 | 67.25 | 0.000e+00 | 1 |
| amazonpho | mlp_small_loss | 0 | 74.77 | 74.77 | 0.000e+00 | 1 |
| amazonpho | mlp_small_loss | 1 | 77.98 | 77.98 | 0.000e+00 | 1 |
| amazonpho | mlp_small_loss | 2 | 77.24 | 77.24 | 0.000e+00 | 1 |
| amazonpho | mlp_small_loss | 3 | 79.18 | 79.18 | 0.000e+00 | 1 |
| amazonpho | mlp_small_loss | 4 | 77.07 | 77.07 | 0.000e+00 | 1 |
| amazonpho | mlp_jocor | 0 | 79.37 | 79.37 | 0.000e+00 | 1 |
| amazonpho | mlp_jocor | 1 | 79.68 | 79.68 | 0.000e+00 | 1 |
| amazonpho | mlp_jocor | 2 | 79.23 | 79.23 | 0.000e+00 | 1 |
| amazonpho | mlp_jocor | 3 | 80.45 | 80.45 | 0.000e+00 | 1 |
| amazonpho | mlp_jocor | 4 | 79.12 | 79.12 | 0.000e+00 | 1 |

## Completeness
| Dataset | Method | expected | observed | missing |
|---|---|---:|---:|---|
| cora | frozengrace_gb_mlp_softdist_beta05 | 5 | 5 |  |
| cora | mlp_ce | 5 | 5 |  |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 5 |  |
| cora | knn_mlp_softdist_beta05 | 5 | 5 |  |
| cora | mlp_distill_softdist_xh | 5 | 5 |  |
| cora | mlp_distill_gcn | 5 | 5 |  |
| cora | mlp_small_loss | 5 | 5 |  |
| cora | mlp_jocor | 5 | 5 |  |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 5 | 5 |  |
| citeseer | mlp_ce | 5 | 5 |  |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 5 |  |
| citeseer | knn_mlp_softdist_beta05 | 5 | 5 |  |
| citeseer | mlp_distill_softdist_xh | 5 | 5 |  |
| citeseer | mlp_distill_gcn | 5 | 5 |  |
| citeseer | mlp_small_loss | 5 | 5 |  |
| citeseer | mlp_jocor | 5 | 5 |  |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 5 | 5 |  |
| pubmed | mlp_ce | 5 | 5 |  |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 5 |  |
| pubmed | knn_mlp_softdist_beta05 | 5 | 5 |  |
| pubmed | mlp_distill_softdist_xh | 5 | 5 |  |
| pubmed | mlp_distill_gcn | 5 | 5 |  |
| pubmed | mlp_small_loss | 5 | 5 |  |
| pubmed | mlp_jocor | 5 | 5 |  |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 5 | 5 |  |
| coauthorcs | mlp_ce | 5 | 5 |  |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 5 |  |
| coauthorcs | knn_mlp_softdist_beta05 | 5 | 5 |  |
| coauthorcs | mlp_small_loss | 5 | 5 |  |
| coauthorcs | mlp_jocor | 5 | 5 |  |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | 5 | 5 |  |
| amazonpho | mlp_ce | 5 | 5 |  |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 5 |  |
| amazonpho | knn_mlp_softdist_beta05 | 5 | 5 |  |
| amazonpho | mlp_small_loss | 5 | 5 |  |
| amazonpho | mlp_jocor | 5 | 5 |  |
