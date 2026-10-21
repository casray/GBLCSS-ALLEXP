# MLP-51 Multi-Partition Granular-Ball Consensus report

## Frozen protocol

- Three archived, independently pretrained GRACE H views; fixed equal target averaging.
- Formal tie-safe LOO targets: `distribution_only`, `beta=0.5`, `s_min=2`.
- Exact kNN excludes self and matches each view's mean train votes per ball.
- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.

## Target quality

| Dataset | Noise | Method | Acc | Brier | NCR | CDR |
|---|---:|---|---:|---:|---:|---:|
| citeseer | 0.4 | mp_gb_3h | 0.5669 | 0.5606 | 0.0238 | 0.0027 |
| citeseer | 0.4 | mp_gb_3h_shuffled | 0.5579 | 0.7374 | 0.0000 | 0.0000 |
| citeseer | 0.4 | mp_knn_3h | 0.5669 | 0.5618 | 0.0272 | 0.0054 |
| citeseer | 0.4 | single_h0_gb | 0.5774 | 0.5913 | 0.0714 | 0.0216 |
| citeseer | 0.4 | single_h0_knn | 0.5714 | 0.5749 | 0.0442 | 0.0108 |
| citeseer | 0.6 | mp_gb_3h | 0.4135 | 0.8549 | 0.0000 | 0.0000 |
| citeseer | 0.6 | mp_gb_3h_shuffled | 0.4135 | 0.9721 | 0.0000 | 0.0000 |
| citeseer | 0.6 | mp_knn_3h | 0.4135 | 0.8346 | 0.0051 | 0.0073 |
| citeseer | 0.6 | single_h0_gb | 0.4286 | 0.8868 | 0.0256 | 0.0000 |
| citeseer | 0.6 | single_h0_knn | 0.4195 | 0.8374 | 0.0128 | 0.0036 |
| cora | 0.4 | mp_gb_3h | 0.6081 | 0.4665 | 0.0783 | 0.0000 |
| cora | 0.4 | mp_gb_3h_shuffled | 0.5749 | 0.7124 | 0.0000 | 0.0000 |
| cora | 0.4 | mp_knn_3h | 0.6248 | 0.4424 | 0.1174 | 0.0000 |
| cora | 0.4 | single_h0_gb | 0.6137 | 0.4879 | 0.1391 | 0.0354 |
| cora | 0.4 | single_h0_knn | 0.6488 | 0.4590 | 0.1826 | 0.0064 |
| cora | 0.6 | mp_gb_3h | 0.3752 | 0.8801 | 0.0059 | 0.0000 |
| cora | 0.6 | mp_gb_3h_shuffled | 0.3715 | 1.0418 | 0.0000 | 0.0000 |
| cora | 0.6 | mp_knn_3h | 0.3789 | 0.8890 | 0.0118 | 0.0000 |
| cora | 0.6 | single_h0_gb | 0.3789 | 0.9028 | 0.0147 | 0.0050 |
| cora | 0.6 | single_h0_knn | 0.3937 | 0.8941 | 0.0353 | 0.0000 |
| corafull | 0.4 | mp_gb_3h | 0.5978 | 0.6054 | 0.0000 | 0.0000 |
| corafull | 0.4 | mp_gb_3h_shuffled | 0.5978 | 0.7654 | 0.0000 | 0.0000 |
| corafull | 0.4 | mp_knn_3h | 0.5993 | 0.5802 | 0.0038 | 0.0000 |
| corafull | 0.4 | single_h0_gb | 0.5993 | 0.6266 | 0.0050 | 0.0008 |
| corafull | 0.4 | single_h0_knn | 0.5993 | 0.5949 | 0.0038 | 0.0000 |
| corafull | 0.6 | mp_gb_3h | 0.3982 | 0.9849 | 0.0000 | 0.0000 |
| corafull | 0.6 | mp_gb_3h_shuffled | 0.3982 | 1.1504 | 0.0000 | 0.0000 |
| corafull | 0.6 | mp_knn_3h | 0.3982 | 0.9575 | 0.0000 | 0.0000 |
| corafull | 0.6 | single_h0_gb | 0.3982 | 1.0059 | 0.0000 | 0.0000 |
| corafull | 0.6 | single_h0_knn | 0.3982 | 0.9698 | 0.0000 | 0.0000 |

## Preregistered gate

- PASS: `0`.
- Near-pass: `0`.
- Reason: `mp_knn_brier_gate_failed`.
- Brier wins vs single-H GB: `6/6`.
- Brier wins vs 3H kNN mirror: `2/6`.
- Brier wins vs shuffled 3H GB: `6/6`.
- Accuracy-regression cells: `['citeseer:0.4', 'citeseer:0.6', 'cora:0.4']`.

## Decision

Phase A did not pass. No downstream training is authorized by this plan.

## Detailed interpretation

The three-partition average is not a null result in every sense. Its Brier score is lower than the single-H granular-ball target in all six cells and lower than its size-matched shuffled control in all six cells. Independent GRACE pretraining seeds therefore contain complementary local information, and the gain cannot be reduced to generic target smoothing.

The candidate nevertheless fails as an algorithmic improvement. It beats the identically averaged three-H kNN mirror in only 2/6 cells, so the useful information is supplied by multiple representations rather than by granular balls. Averaging final targets also reduces decisive corrections: target accuracy drops by more than 0.5 percentage points on CiteSeer u0.4, CiteSeer u0.6, and Cora u0.4. The lower Brier score is partly produced by conservative probability averaging, not by recovering more corrupted labels.

This result supports a diagnostic claim only: representation-seed diversity improves calibration. It does not support replacing the formal single-H GB-LCSS method, claiming a granular-ball advantage over kNN, or starting a downstream multi-seed experiment.
