# MLP-33 High-Noise Graph-Native, Sensitivity, and Noise-Type Report

## Uniform 0.6 graph-native seed-0 comparison

| Dataset | Direct combo | Frozen X-only distill | NRGNN | NRGNN[X||H] | RTGNN | Best X-only - best graph |
|---|---:|---:|---:|---:|---:|---:|
| cora | 44.96 | 68.65 (mlp_distill_knn_xh) | 74.99 | 71.50 | 71.06 | -6.33pp |
| citeseer | 41.80 | 59.19 (mlp_distill_softdist_xh) | 68.08 | 68.00 | 60.78 | -8.89pp |
| corafull | 33.23 | 52.61 (mlp_distill_gcn_xh) | 58.44 | 57.84 | NA | -5.83pp |
| amazonpho | 72.01 | 84.65 (mlp_distill_gcn_xh) | 88.40 | 88.37 | 24.95 | -3.75pp |
| pubmed | 65.32 | 77.99 (mlp_distill_softdist_xh) | 56.88 | 77.20 | 65.50 | +0.78pp |

Gate: 1/5 datasets are within 2pp or win; pass=0; decision=stop_after_seed0.

## Forget-rate misspecification at true uniform 0.6 (seed 0)

| Dataset | Assumed rate | Accuracy | Delta vs correct 0.6 |
|---|---:|---:|---:|
| cora | 0.4 | 43.48 | -1.48pp |
| cora | 0.5 | 44.70 | -0.26pp |
| cora | 0.6 | 44.96 | +0.00pp |
| citeseer | 0.4 | 41.28 | -0.52pp |
| citeseer | 0.5 | 40.76 | -1.03pp |
| citeseer | 0.6 | 41.80 | +0.00pp |
| corafull | 0.4 | 32.09 | -1.14pp |
| corafull | 0.5 | 33.66 | +0.43pp |
| corafull | 0.6 | 33.23 | +0.00pp |

## Pair 0.4 five-seed comparison

| Dataset | LCSS | Small-loss | Combination | Shuffled combination | Synergy | Positive | Real-shuffled | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 50.29+/-2.47 | 46.99+/-2.48 | 51.34+/-1.54 | 42.04+/-2.00 | +1.04pp | 4/5 | +9.30pp | 5/5 |
| citeseer | 50.09+/-2.15 | 48.16+/-2.29 | 51.15+/-2.26 | 45.87+/-1.21 | +1.07pp | 4/5 | +5.28pp | 5/5 |
| corafull | 34.41+/-1.16 | 30.15+/-1.00 | 35.31+/-0.97 | 30.09+/-0.74 | +0.90pp | 5/5 | +5.22pp | 5/5 |
| amazonpho | 72.47+/-3.15 | 66.70+/-5.13 | 72.37+/-3.54 | 66.63+/-7.00 | -0.10pp | 3/5 | +5.74pp | 4/5 |
| pubmed | 68.64+/-1.17 | 73.19+/-1.52 | 77.71+/-0.73 | 65.99+/-0.43 | +4.52pp | 5/5 | +11.73pp | 5/5 |

Gate: pass=1; synergy=4/5; mean=+1.49pp; minimum=-0.10pp; structure=5/5; reasons=pass.

## Instance 0.4 five-seed comparison

| Dataset | LCSS | Small-loss | Combination | Shuffled combination | Synergy | Positive | Real-shuffled | Positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 55.67+/-0.73 | 51.82+/-0.84 | 59.22+/-1.10 | 46.13+/-1.83 | +3.55pp | 5/5 | +13.09pp | 5/5 |
| citeseer | 54.62+/-2.01 | 50.75+/-2.98 | 58.20+/-2.05 | 46.41+/-1.57 | +3.57pp | 5/5 | +11.79pp | 5/5 |
| corafull | 37.08+/-0.61 | 32.90+/-0.78 | 39.77+/-0.54 | 33.57+/-0.47 | +2.69pp | 5/5 | +6.20pp | 5/5 |
| amazonpho | 79.78+/-1.04 | 73.00+/-1.12 | 80.68+/-0.86 | 71.76+/-1.28 | +0.90pp | 4/5 | +8.91pp | 5/5 |
| pubmed | 71.57+/-1.29 | 76.47+/-1.12 | 80.94+/-0.93 | 73.76+/-1.26 | +4.47pp | 5/5 | +7.18pp | 5/5 |

Gate: pass=1; synergy=5/5; mean=+3.04pp; minimum=+0.90pp; structure=5/5; reasons=pass.
