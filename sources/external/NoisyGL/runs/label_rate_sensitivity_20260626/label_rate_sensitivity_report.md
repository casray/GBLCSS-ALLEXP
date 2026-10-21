# Label-Rate Sensitivity for SoftDist [X||H]

## Protocol

- Split: `percent`, train ratios `0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4`.
- Validation ratio is fixed at `0.10`; test ratio is `1 - train - val`.
- Noise: `uniform`, rate `0.4`, scope `train`.
- Seeds: `0,1,2,3,4`.
- Core comparison: SoftDist [X||H] against GCN+CE [X||H] and shuffled SoftDist.
- This is a sensitivity analysis, not a cross-ratio leaderboard.

## Research Record

- Hypothesis: SoftDist's ball soft target should help most when labels are scarce enough that GCN [X||H] has not saturated, but not necessarily improve monotonically with larger train ratios.
- Granular-ball role: balls act as noisy-label soft target providers and local structure priors, not hard positive labels.
- Method motivation: test whether the ball component adds value beyond the same `[X||H]` input and beyond shuffled ball structure across label-rate regimes.
- Code changes: no model, loss, baseline, or evaluation logic changes; only this summarizer was added to combine old and new raw CSV files with explicit split metadata.
- Config: existing Phase 16 coverage files are reused where available; newly added ratios are run with the same protocol, methods, and seeds.
- Expected mechanism: low/mid train ratios should leave room for ball-level consensus; high train ratios may let GCN [X||H] absorb local structure directly, reducing ball marginal gain.
- Success standard: both `SoftDist - GCN [X||H]` and `SoftDist - shuffled` pass the pre-registered mean-delta and positive-seed thresholds.
- Failure standard: no train ratio passes both controls, or gains only appear against plain GCN but disappear against GCN [X||H] or shuffled.
- Results: see Main Results and paired-delta CSV.
- Final decision: use this as a sensitivity/mechanism curve, not as a replacement for the 20/10/70 main protocol.

## Completeness Audit

| Dataset | Train | Method | Present | Bad status | Missing seeds |
|---|---:|---|---:|---:|---|
| all | all | all | complete | 0 |  |

## Main Results

| Dataset | Train | GCN | GCN [X||H] | SoftDist [X||H] | Shuffled | Soft-GCN[X||H] | Soft-Shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 0.05 | 0.6367 +/- 0.0343 | 0.6887 +/- 0.0166 | 0.7035 +/- 0.0214 | 0.6807 +/- 0.0145 | +1.48pp (5/5) strong | +2.28pp (5/5) strong |
| cora | 0.10 | 0.7325 +/- 0.0109 | 0.7543 +/- 0.0255 | 0.7675 +/- 0.0150 | 0.7499 +/- 0.0178 | +1.32pp (4/5) strong | +1.76pp (5/5) strong |
| cora | 0.15 | 0.7579 +/- 0.0232 | 0.7828 +/- 0.0249 | 0.7889 +/- 0.0191 | 0.7760 +/- 0.0298 | +0.61pp (3/5) | +1.29pp (5/5) strong |
| cora | 0.20 | 0.7842 +/- 0.0151 | 0.8027 +/- 0.0069 | 0.8137 +/- 0.0067 | 0.7921 +/- 0.0097 | +1.10pp (4/5) strong | +2.16pp (5/5) strong |
| cora | 0.25 | 0.8039 +/- 0.0121 | 0.8231 +/- 0.0150 | 0.8269 +/- 0.0217 | 0.8169 +/- 0.0120 | +0.39pp (4/5) | +1.00pp (4/5) |
| cora | 0.30 | 0.8079 +/- 0.0094 | 0.8224 +/- 0.0131 | 0.8319 +/- 0.0185 | 0.8262 +/- 0.0184 | +0.95pp (3/5) | +0.57pp (5/5) |
| cora | 0.40 | 0.8344 +/- 0.0074 | 0.8486 +/- 0.0043 | 0.8387 +/- 0.0099 | 0.8322 +/- 0.0110 | -0.99pp (1/5) | +0.65pp (4/5) |
| citeseer | 0.05 | 0.4937 +/- 0.0552 | 0.5539 +/- 0.0449 | 0.5721 +/- 0.0506 | 0.5501 +/- 0.0528 | +1.83pp (4/5) strong | +2.20pp (4/5) strong |
| citeseer | 0.10 | 0.6005 +/- 0.0191 | 0.6455 +/- 0.0211 | 0.6613 +/- 0.0180 | 0.6374 +/- 0.0297 | +1.59pp (4/5) strong | +2.39pp (4/5) strong |
| citeseer | 0.15 | 0.6276 +/- 0.0195 | 0.6642 +/- 0.0149 | 0.6782 +/- 0.0150 | 0.6595 +/- 0.0171 | +1.39pp (5/5) strong | +1.87pp (5/5) strong |
| citeseer | 0.20 | 0.6530 +/- 0.0121 | 0.6761 +/- 0.0181 | 0.6891 +/- 0.0160 | 0.6696 +/- 0.0234 | +1.30pp (5/5) strong | +1.95pp (5/5) strong |
| citeseer | 0.25 | 0.6648 +/- 0.0123 | 0.6868 +/- 0.0132 | 0.7017 +/- 0.0081 | 0.6891 +/- 0.0073 | +1.49pp (5/5) strong | +1.26pp (5/5) strong |
| citeseer | 0.30 | 0.6791 +/- 0.0120 | 0.6985 +/- 0.0101 | 0.7042 +/- 0.0106 | 0.6971 +/- 0.0096 | +0.57pp (4/5) strong | +0.71pp (4/5) |
| citeseer | 0.40 | 0.6996 +/- 0.0088 | 0.7103 +/- 0.0093 | 0.7162 +/- 0.0122 | 0.7112 +/- 0.0127 | +0.59pp (4/5) strong | +0.49pp (3/5) |
| coauthorcs | 0.05 | 0.9019 +/- 0.0070 | 0.8998 +/- 0.0065 | 0.9022 +/- 0.0051 | 0.8959 +/- 0.0100 | +0.23pp (4/5) | +0.63pp (4/5) |
| coauthorcs | 0.10 | 0.9139 +/- 0.0019 | 0.9130 +/- 0.0033 | 0.9173 +/- 0.0021 | 0.9111 +/- 0.0037 | +0.44pp (5/5) | +0.63pp (5/5) |
| coauthorcs | 0.15 | 0.9181 +/- 0.0015 | 0.9168 +/- 0.0030 | 0.9201 +/- 0.0026 | 0.9158 +/- 0.0024 | +0.33pp (5/5) | +0.43pp (5/5) |
| coauthorcs | 0.20 | 0.9206 +/- 0.0034 | 0.9205 +/- 0.0019 | 0.9230 +/- 0.0014 | 0.9187 +/- 0.0023 | +0.25pp (5/5) | +0.43pp (5/5) |
| coauthorcs | 0.25 | 0.9232 +/- 0.0019 | 0.9226 +/- 0.0008 | 0.9240 +/- 0.0025 | 0.9208 +/- 0.0021 | +0.13pp (4/5) | +0.31pp (5/5) |
| coauthorcs | 0.30 | 0.9228 +/- 0.0027 | 0.9237 +/- 0.0023 | 0.9271 +/- 0.0010 | 0.9234 +/- 0.0036 | +0.34pp (5/5) | +0.37pp (4/5) |
| coauthorcs | 0.40 | 0.9251 +/- 0.0018 | 0.9274 +/- 0.0017 | 0.9288 +/- 0.0015 | 0.9272 +/- 0.0019 | +0.14pp (3/5) | +0.16pp (5/5) |

## Pre-Registered Interpretation

- A train ratio is strong evidence only when both SoftDist-vs-GCN[X||H] and SoftDist-vs-shuffled pass the positive-seed and mean-delta thresholds.
- Weak or negative high-label results support the saturation interpretation rather than a monotonic coverage claim.
- The main 20/10/70 protocol should remain the main table; this curve explains when the ball soft target contributes.
- Cora/CiteSeer support the low/mid-label-rate mechanism; Coauthor-CS shows smaller gains below the strong-evidence threshold, so the claim should be dataset- and regime-conditional.

## Artifacts

- Combined raw: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_combined_raw.csv`
- Aggregate: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_aggregate.csv`
- Paired deltas: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_paired_deltas.csv`
- Audit: `runs/label_rate_sensitivity_20260626/label_rate_sensitivity_audit.csv`
