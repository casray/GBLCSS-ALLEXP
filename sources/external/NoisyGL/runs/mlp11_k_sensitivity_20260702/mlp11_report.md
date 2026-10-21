# MLP-11 k-Sensitivity Report

## Material Passport
- Protocol: train-only uniform 0.4, percent split 0.2/0.1/0.7.
- kNN target: `knn_mlp_softdist_beta05`, multiclass alpha, beta=0.5.
- Ball references are reused; no ball reruns.

## C1/C2/C3
| Condition | Pass | Detail |
|---|---:|---|
| C1_k_variation | 0 | best_k min=20 max=20 ratio=1.00 |
| C2_k_sensitive | 0 | datasets_loss_ge_2pp=0 |
| C3_ball_tracks_best_k | 1 | pass_count=3/4 |
| C3_detail_cora | 1 | gap=0.016148, best_std=0.022642 |
| C3_detail_citeseer | 0 | gap=0.025773, best_std=0.023065 |
| C3_detail_pubmed | 1 | gap=0.005724, best_std=0.009463 |
| C3_detail_coauthorcs | 1 | gap=0.002946, best_std=0.006165 |

Global k*: `20`

## k Curve
| Dataset | k | n | Test mean±std | Valid mean |
|---|---:|---:|---:|---:|
| cora | 3 | 5 | 51.37±3.27 | 53.33 |
| cora | 5 | 5 | 53.93±0.88 | 55.93 |
| cora | 10 | 5 | 55.27±1.26 | 58.07 |
| cora | 20 | 5 | 57.20±2.26 | 58.81 |
| cora | 50 | 5 | 55.03±1.35 | 56.89 |
| cora | 100 | 5 | 54.30±2.04 | 55.41 |
| citeseer | 3 | 5 | 50.26±1.50 | 54.82 |
| citeseer | 5 | 5 | 53.51±2.02 | 56.75 |
| citeseer | 10 | 5 | 55.83±2.47 | 58.55 |
| citeseer | 20 | 5 | 56.41±2.31 | 60.00 |
| citeseer | 50 | 5 | 56.30±2.43 | 60.30 |
| citeseer | 100 | 5 | 54.94±2.59 | 58.67 |
| pubmed | 3 | 5 | 66.34±0.47 | 67.11 |
| pubmed | 5 | 5 | 68.43±0.73 | 69.83 |
| pubmed | 10 | 5 | 70.80±0.31 | 72.10 |
| pubmed | 20 | 5 | 72.48±0.95 | 73.39 |
| pubmed | 50 | 5 | 72.35±0.57 | 73.40 |
| pubmed | 100 | 5 | 72.27±0.59 | 73.40 |
| coauthorcs | 3 | 5 | 89.64±0.68 | 90.17 |
| coauthorcs | 5 | 5 | 90.32±0.55 | 90.68 |
| coauthorcs | 10 | 5 | 90.83±0.58 | 91.26 |
| coauthorcs | 20 | 5 | 90.91±0.62 | 91.45 |
| coauthorcs | 50 | 5 | 90.83±0.58 | 91.34 |
| coauthorcs | 100 | 5 | 90.44±0.62 | 91.05 |

## Ball/Shuffled References
| Dataset | Method | n | Test mean±std |
|---|---|---:|---:|
| cora | frozengrace_gb_mlp_softdist_beta05 | 5 | 55.59±1.50 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 43.63±1.62 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | 5 | 53.83±2.75 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 42.35±2.74 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | 5 | 71.90±0.88 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 62.72±4.49 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | 5 | 90.62±0.57 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 87.60±1.03 |

## Best-k
| Dataset | Best k | Best test | Best std |
|---|---:|---:|---:|
| cora | 20 | 57.20 | 2.26 |
| citeseer | 20 | 56.41 | 2.31 |
| pubmed | 20 | 72.48 | 0.95 |
| coauthorcs | 20 | 90.91 | 0.62 |

## Global-k Loss
| Dataset | Global k | Global test | Best k | Best test | Loss |
|---|---:|---:|---:|---:|---:|
| cora | 20 | 57.20 | 20 | 57.20 | +0.00pp |
| citeseer | 20 | 56.41 | 20 | 56.41 | +0.00pp |
| pubmed | 20 | 72.48 | 20 | 72.48 | +0.00pp |
| coauthorcs | 20 | 90.91 | 20 | 90.91 | +0.00pp |

## R3 Val-selected k
| Dataset | Val k | Val-selected test | Test best k | Test best | Δ vs best | Ball | Δ vs ball |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 20 | 57.20 | 20 | 57.20 | +0.00pp | 55.59 | +1.61pp |
| citeseer | 50 | 56.30 | 20 | 56.41 | +0.11pp | 53.83 | +2.47pp |
| pubmed | 100 | 72.27 | 20 | 72.48 | +0.20pp | 71.90 | +0.37pp |
| coauthorcs | 20 | 90.91 | 20 | 90.91 | +0.00pp | 90.62 | +0.29pp |

## Regression Check
| Dataset | Seed | P0 matched-k | New k | P0 test | New test | Delta | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| cora | 0 | 10 | 10 | 55.46 | 55.46 | +0.00pp | pass |
| cora | 1 | 10 | 10 | 55.94 | 55.94 | +0.00pp | pass |
| cora | 2 | 10 | 10 | 56.52 | 56.52 | +0.00pp | pass |
| cora | 3 | 10 | 10 | 55.25 | 55.25 | +0.00pp | pass |
| cora | 4 | 10 | 10 | 53.19 | 53.19 | +0.00pp | pass |
| citeseer | 0 | 10 | 10 | 56.70 | 56.70 | +0.00pp | pass |
| citeseer | 1 | 10 | 10 | 55.03 | 55.03 | +0.00pp | pass |
| citeseer | 2 | 10 | 10 | 55.24 | 55.24 | +0.00pp | pass |
| citeseer | 3 | 10 | 10 | 59.45 | 59.45 | +0.00pp | pass |
| citeseer | 4 | 11 | 10 | 51.55 | 52.75 |  | skipped_p0_matched_k_11 |

## Completeness
- Expected groups: 24
- Missing groups: 0
