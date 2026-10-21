# MLP-8 Replacement + Expansion Report

## Material Passport
- Artifact type: experiment result
- Phase covered in this report: P0 replacement control
- Protocol: X-only weak MLP, train-only label noise, percent split 0.2/0.1/0.7
- Reused anchor: MLP-5 beta=0.5 and shuffled rows

## Completeness
- Expected P0 rows: 60
- Observed P0 rows: 60
- Missing P0 rows: 0

## P0 Aggregate
| Dataset | Method | Mean +/- Std | matched k | matched K |
|---|---|---:|---:|---:|
| citeseer | true ball beta=0.5 | 53.83+/-2.75 |  |  |
| citeseer | shuffled ball beta=0.5 | 42.35+/-2.74 |  |  |
| citeseer | kmeans-H beta=0.5 | 55.32+/-1.79 |  | 63.8 |
| citeseer | kNN-in-H beta=0.5 | 55.59+/-2.87 | 10.2 | 63.8 |
| cora | true ball beta=0.5 | 55.59+/-1.50 |  |  |
| cora | shuffled ball beta=0.5 | 43.63+/-1.62 |  |  |
| cora | kmeans-H beta=0.5 | 55.54+/-2.16 |  | 52.4 |
| cora | kNN-in-H beta=0.5 | 55.27+/-1.26 | 10.0 | 52.4 |
| pubmed | true ball beta=0.5 | 71.90+/-0.88 |  |  |
| pubmed | shuffled ball beta=0.5 | 62.72+/-4.49 |  |  |
| pubmed | kmeans-H beta=0.5 | 72.17+/-0.86 |  | 64.0 |
| pubmed | kNN-in-H beta=0.5 | 72.63+/-0.52 | 62.0 | 64.0 |

## P0 Paired Deltas: true ball - replacement
| Dataset | Comparison | Mean delta | Wins | Decision aid |
|---|---|---:|---:|---|
| cora | ball_minus_knn_mlp_softdist_beta05 | 0.32pp | 2/5 | not-pass |
| cora | ball_minus_kmeans_mlp_softdist_beta05 | 0.05pp | 3/5 | not-pass |
| cora | ball_minus_frozengrace_gb_mlp_softdist_beta05_shuffled | 11.96pp | 5/5 | pass |
| citeseer | ball_minus_knn_mlp_softdist_beta05 | -1.76pp | 1/5 | not-pass |
| citeseer | ball_minus_kmeans_mlp_softdist_beta05 | -1.49pp | 1/5 | not-pass |
| citeseer | ball_minus_frozengrace_gb_mlp_softdist_beta05_shuffled | 11.48pp | 5/5 | pass |
| pubmed | ball_minus_knn_mlp_softdist_beta05 | -0.72pp | 0/5 | not-pass |
| pubmed | ball_minus_kmeans_mlp_softdist_beta05 | -0.27pp | 2/5 | not-pass |
| pubmed | ball_minus_frozengrace_gb_mlp_softdist_beta05_shuffled | 9.18pp | 5/5 | pass |

## P0 Verdict
Verdict: **p0_replacement_not_rejected**
- cora: ball-knn=+0.0032 (2/5), ball-kmeans=+0.0005 (3/5)
- citeseer: ball-knn=-0.0176 (1/5), ball-kmeans=-0.0149 (1/5)
- pubmed: ball-knn=-0.0072 (0/5), ball-kmeans=-0.0027 (2/5)

## Claim Boundary
- P0 tests only the train-target role of granular balls.
- If replacement is not rejected, it does not automatically invalidate P1 expansion/teacher roles.

## P1 Expansion + GLNN Seed0
- Phase covered in this section: P1 seed0 lambda scan
- Protocol: X-only weak MLP, train-only uniform label noise, percent split 0.2/0.1/0.7
- Gate note: G1 uses a global lambda per noise rate, not a per-dataset cherry-picked lambda.

### P1 Aggregate
| Dataset | Noise | Method | Mean | lambda | Expansion coverage |
|---|---:|---|---:|---:|---:|
| citeseer | 0.4 | true ball beta=0.5 | 55.33 |  |  |
| citeseer | 0.4 | true ball beta=0.5 + expansion (lambda=0.1) | 60.22 | 0.1 | 0.18332081141998496 |
| citeseer | 0.4 | true ball beta=0.5 + expansion (lambda=0.3) | 61.17 | 0.3 | 0.18332081141998496 |
| citeseer | 0.4 | true ball beta=0.5 + expansion (lambda=1) | 61.04 | 1.0 | 0.18332081141998496 |
| citeseer | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 46.01 | 0.1 | 0.0236664162283997 |
| citeseer | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 40.55 | 0.3 | 0.0236664162283997 |
| citeseer | 0.4 | shuffled ball beta=0.5 + expansion (lambda=1) | 38.32 | 1.0 | 0.0236664162283997 |
| citeseer | 0.4 | shuffled ball beta=0.5 | 42.23 |  |  |
| citeseer | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 64.05 | 0.1 | 0.44402704733283244 |
| citeseer | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 64.05 | 0.3 | 0.44402704733283244 |
| citeseer | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=1) | 64.43 | 1.0 | 0.44402704733283244 |
| citeseer | 0.4 | MLP CE | 46.09 |  |  |
| citeseer | 0.4 | MLP distill noisy GCN teacher | 64.82 |  |  |
| citeseer | 0.6 | true ball beta=0.5 | 38.62 |  |  |
| citeseer | 0.6 | true ball beta=0.5 + expansion (lambda=0.1) | 44.03 | 0.1 | 0.06874530428249437 |
| citeseer | 0.6 | true ball beta=0.5 + expansion (lambda=0.3) | 44.85 | 0.3 | 0.06874530428249437 |
| citeseer | 0.6 | true ball beta=0.5 + expansion (lambda=1) | 46.13 | 1.0 | 0.06874530428249437 |
| citeseer | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 30.76 | 0.1 | 0.0 |
| citeseer | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 30.76 | 0.3 | 0.0 |
| citeseer | 0.6 | shuffled ball beta=0.5 + expansion (lambda=1) | 30.76 | 1.0 | 0.0 |
| citeseer | 0.6 | shuffled ball beta=0.5 | 30.76 |  |  |
| citeseer | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 48.80 | 0.1 | 0.2434259954921112 |
| citeseer | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 50.56 | 0.3 | 0.2434259954921112 |
| citeseer | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=1) | 52.28 | 1.0 | 0.2434259954921112 |
| citeseer | 0.6 | MLP CE | 32.13 |  |  |
| citeseer | 0.6 | MLP distill noisy GCN teacher | 47.68 |  |  |
| cora | 0.4 | true ball beta=0.5 | 55.46 |  |  |
| cora | 0.4 | true ball beta=0.5 + expansion (lambda=0.1) | 65.86 | 0.1 | 0.3770189201661283 |
| cora | 0.4 | true ball beta=0.5 + expansion (lambda=0.3) | 65.49 | 0.3 | 0.3770189201661283 |
| cora | 0.4 | true ball beta=0.5 + expansion (lambda=1) | 65.65 | 1.0 | 0.3770189201661283 |
| cora | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 41.85 | 0.1 | 0.023534840793724043 |
| cora | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 40.95 | 0.3 | 0.023534840793724043 |
| cora | 0.4 | shuffled ball beta=0.5 + expansion (lambda=1) | 37.63 | 1.0 | 0.023534840793724043 |
| cora | 0.4 | shuffled ball beta=0.5 | 43.85 |  |  |
| cora | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 73.19 | 0.1 | 0.596677434240886 |
| cora | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 73.30 | 0.3 | 0.596677434240886 |
| cora | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=1) | 73.83 | 1.0 | 0.596677434240886 |
| cora | 0.4 | MLP CE | 41.74 |  |  |
| cora | 0.4 | MLP distill noisy GCN teacher | 74.35 |  |  |
| cora | 0.6 | true ball beta=0.5 | 38.94 |  |  |
| cora | 0.6 | true ball beta=0.5 + expansion (lambda=0.1) | 41.74 | 0.1 | 0.10613751730503 |
| cora | 0.6 | true ball beta=0.5 + expansion (lambda=0.3) | 40.79 | 0.3 | 0.10613751730503 |
| cora | 0.6 | true ball beta=0.5 + expansion (lambda=1) | 40.37 | 1.0 | 0.10613751730503 |
| cora | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 28.81 | 0.1 | 0.0 |
| cora | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 28.81 | 0.3 | 0.0 |
| cora | 0.6 | shuffled ball beta=0.5 + expansion (lambda=1) | 28.81 | 1.0 | 0.0 |
| cora | 0.6 | shuffled ball beta=0.5 | 28.81 |  |  |
| cora | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 47.86 | 0.1 | 0.2182741116751269 |
| cora | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 48.76 | 0.3 | 0.2182741116751269 |
| cora | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=1) | 48.13 | 1.0 | 0.2182741116751269 |
| cora | 0.6 | MLP CE | 30.71 |  |  |
| cora | 0.6 | MLP distill noisy GCN teacher | 62.74 |  |  |
| pubmed | 0.4 | true ball beta=0.5 | 72.24 |  |  |
| pubmed | 0.4 | true ball beta=0.5 + expansion (lambda=0.1) | 78.16 | 0.1 | 0.7355775326486623 |
| pubmed | 0.4 | true ball beta=0.5 + expansion (lambda=0.3) | 80.08 | 0.3 | 0.7355775326486623 |
| pubmed | 0.4 | true ball beta=0.5 + expansion (lambda=1) | 82.63 | 1.0 | 0.7355775326486623 |
| pubmed | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 58.74 | 0.1 | 0.06941802966907569 |
| pubmed | 0.4 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 58.43 | 0.3 | 0.06941802966907569 |
| pubmed | 0.4 | shuffled ball beta=0.5 + expansion (lambda=1) | 55.42 | 1.0 | 0.06941802966907569 |
| pubmed | 0.4 | shuffled ball beta=0.5 | 60.89 |  |  |
| pubmed | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 79.00 | 0.1 | 0.6686319259541017 |
| pubmed | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 82.08 | 0.3 | 0.6686319259541017 |
| pubmed | 0.4 | kNN-in-H beta=0.5 + expansion (lambda=1) | 83.58 | 1.0 | 0.6686319259541017 |
| pubmed | 0.4 | MLP CE | 62.62 |  |  |
| pubmed | 0.4 | MLP distill noisy GCN teacher | 83.68 |  |  |
| pubmed | 0.6 | true ball beta=0.5 | 49.75 |  |  |
| pubmed | 0.6 | true ball beta=0.5 + expansion (lambda=0.1) | 50.33 | 0.1 | 0.047736782046405475 |
| pubmed | 0.6 | true ball beta=0.5 + expansion (lambda=0.3) | 49.76 | 0.3 | 0.047736782046405475 |
| pubmed | 0.6 | true ball beta=0.5 + expansion (lambda=1) | 51.60 | 1.0 | 0.047736782046405475 |
| pubmed | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.1) | 44.39 | 0.1 | 0.04608850006339546 |
| pubmed | 0.6 | shuffled ball beta=0.5 + expansion (lambda=0.3) | 43.26 | 0.3 | 0.04608850006339546 |
| pubmed | 0.6 | shuffled ball beta=0.5 + expansion (lambda=1) | 43.90 | 1.0 | 0.04608850006339546 |
| pubmed | 0.6 | shuffled ball beta=0.5 | 46.25 |  |  |
| pubmed | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.1) | 50.58 | 0.1 | 0.04108025865348041 |
| pubmed | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=0.3) | 53.39 | 0.3 | 0.04108025865348041 |
| pubmed | 0.6 | kNN-in-H beta=0.5 + expansion (lambda=1) | 58.05 | 1.0 | 0.04108025865348041 |
| pubmed | 0.6 | MLP CE | 47.21 |  |  |
| pubmed | 0.6 | MLP distill noisy GCN teacher | 68.32 |  |  |

### P1 Gates
| Noise | lambda | Gate | Mean delta | Positive | Pass-count | Decision |
|---:|---:|---|---:|---:|---:|---|
| 0.4 | 0.1 | G1_expansion_minus_train_only | +7.07pp | 3/3 | 3/3 | pass |
| 0.4 | 0.1 | G3_expansion_minus_glnn | -6.20pp | 0/3 | 0/3 | not-pass |
| 0.4 | 0.1 | G4_expansion_minus_shuffled_expansion | +19.22pp | 3/3 | 3/3 | pass |
| 0.4 | 0.1 | G4_expansion_minus_knn_expansion | -4.00pp | 0/3 | 0/3 | not-pass |
| 0.4 | 0.3 | G1_expansion_minus_train_only | +7.90pp | 3/3 | 3/3 | pass |
| 0.4 | 0.3 | G3_expansion_minus_glnn | -5.37pp | 0/3 | 0/3 | not-pass |
| 0.4 | 0.3 | G4_expansion_minus_shuffled_expansion | +22.27pp | 3/3 | 3/3 | pass |
| 0.4 | 0.3 | G4_expansion_minus_knn_expansion | -4.23pp | 0/3 | 0/3 | not-pass |
| 0.4 | 1.0 | G1_expansion_minus_train_only | +8.76pp | 3/3 | 3/3 | pass |
| 0.4 | 1.0 | G3_expansion_minus_glnn | -4.51pp | 0/3 | 0/3 | not-pass |
| 0.4 | 1.0 | G4_expansion_minus_shuffled_expansion | +25.98pp | 3/3 | 3/3 | pass |
| 0.4 | 1.0 | G4_expansion_minus_knn_expansion | -4.17pp | 0/3 | 0/3 | not-pass |
| 0.6 | 0.1 | G1_expansion_minus_train_only | +2.93pp | 3/3 | 1/3 | not-pass |
| 0.6 | 0.1 | G3_expansion_minus_glnn | -14.22pp | 0/3 | 0/3 | not-pass |
| 0.6 | 0.1 | G4_expansion_minus_shuffled_expansion | +10.71pp | 3/3 | 3/3 | pass |
| 0.6 | 0.1 | G4_expansion_minus_knn_expansion | -3.71pp | 0/3 | 0/3 | not-pass |
| 0.6 | 0.3 | G1_expansion_minus_train_only | +2.69pp | 3/3 | 1/3 | not-pass |
| 0.6 | 0.3 | G3_expansion_minus_glnn | -14.45pp | 0/3 | 0/3 | not-pass |
| 0.6 | 0.3 | G4_expansion_minus_shuffled_expansion | +10.86pp | 3/3 | 3/3 | pass |
| 0.6 | 0.3 | G4_expansion_minus_knn_expansion | -5.77pp | 0/3 | 0/3 | not-pass |
| 0.6 | 1.0 | G1_expansion_minus_train_only | +3.60pp | 3/3 | 1/3 | not-pass |
| 0.6 | 1.0 | G3_expansion_minus_glnn | -13.55pp | 0/3 | 0/3 | not-pass |
| 0.6 | 1.0 | G4_expansion_minus_shuffled_expansion | +11.55pp | 3/3 | 3/3 | pass |
| 0.6 | 1.0 | G4_expansion_minus_knn_expansion | -6.78pp | 0/3 | 0/3 | not-pass |

### P1 Claim Boundary
- P1 is seed0 only unless a gate passes; it is a smoke test, not a stable paper table.
- GLNN here is an MLP-adapted noisy-GCN-teacher distillation baseline, not an official GLNN reproduction.
- Clean labels are used only through final test accuracy and offline reporting, not in target construction.
