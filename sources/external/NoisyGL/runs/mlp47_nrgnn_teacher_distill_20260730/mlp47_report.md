# MLP-47 NRGNN Robust-Teacher Distillation Report

All values use five paired seeds. NRGNN teachers are retrained independently
inside each seed; every student is a two-layer X-only MLP.

## Uniform 0.4

| Method | Cora | CiteSeer | CoraFull |
|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | 77.95±2.62 | 66.09±1.38 | 58.74±0.82 |
| GCN[X||H]+CE | 80.23±1.57 | 68.81±1.12 | 59.45±0.63 |
| Random-group consensus teacher | 78.81±1.63 | 68.60±1.14 | 58.98±0.41 |
| kNN consensus teacher (exclude self) | 81.61±0.74 | 69.65±1.61 | 58.58±0.55 |
| Granular-ball consensus teacher | 80.60±0.69 | 69.40±1.08 | 58.22±0.69 |
| NRGNN teacher | 81.07±0.55 | 70.87±1.06 | 57.71±0.58 |
| NRGNN[X||H] teacher | 80.37±1.29 | 70.43±0.98 | 56.77±0.59 |

## Uniform 0.6

| Method | Cora | CiteSeer | CoraFull |
|---|---:|---:|---:|
| GCN(X)+CE (GLNN-style) | 65.50±3.28 | 53.30±4.42 | 51.63±2.02 |
| GCN[X||H]+CE | 69.44±2.18 | 61.04±3.08 | 52.75±0.88 |
| Random-group consensus teacher | 68.51±1.82 | 61.07±2.76 | 52.20±0.72 |
| kNN consensus teacher (exclude self) | 71.80±2.63 | 62.90±2.26 | 52.65±0.93 |
| Granular-ball consensus teacher | 71.27±2.46 | 62.28±2.30 | 51.82±0.86 |
| NRGNN teacher | 74.81±1.17 | 66.62±1.27 | 53.55±1.00 |
| NRGNN[X||H] teacher | 74.50±2.25 | 67.75±1.79 | 52.78±0.62 |

## Paired comparisons

| Rate | Dataset | Comparison | Mean delta | Positive seeds |
|---:|---|---|---:|---:|
| 0.4 | citeseer | mlp_distill_nrgnn_student_vs_own_teacher | -0.16pp | 1/5 |
| 0.4 | citeseer | mlp_distill_nrgnn_xh_student_vs_own_teacher | -0.31pp | 1/5 |
| 0.4 | citeseer | nrgnn_vs_ball | +1.47pp | 4/5 |
| 0.4 | citeseer | nrgnn_vs_glnn | +4.78pp | 5/5 |
| 0.4 | citeseer | nrgnn_vs_knn | +1.22pp | 5/5 |
| 0.4 | citeseer | nrgnn_xh_vs_ball | +1.03pp | 3/5 |
| 0.4 | citeseer | nrgnn_xh_vs_gcn_xh | +1.62pp | 4/5 |
| 0.4 | citeseer | nrgnn_xh_vs_knn | +0.78pp | 3/5 |
| 0.6 | citeseer | mlp_distill_nrgnn_student_vs_own_teacher | +0.52pp | 4/5 |
| 0.6 | citeseer | mlp_distill_nrgnn_xh_student_vs_own_teacher | -0.25pp | 2/5 |
| 0.6 | citeseer | nrgnn_vs_ball | +4.34pp | 5/5 |
| 0.6 | citeseer | nrgnn_vs_glnn | +13.32pp | 5/5 |
| 0.6 | citeseer | nrgnn_vs_knn | +3.72pp | 4/5 |
| 0.6 | citeseer | nrgnn_xh_vs_ball | +5.47pp | 4/5 |
| 0.6 | citeseer | nrgnn_xh_vs_gcn_xh | +6.71pp | 4/5 |
| 0.6 | citeseer | nrgnn_xh_vs_knn | +4.85pp | 4/5 |
| 0.4 | cora | mlp_distill_nrgnn_student_vs_own_teacher | -0.70pp | 2/5 |
| 0.4 | cora | mlp_distill_nrgnn_xh_student_vs_own_teacher | -1.10pp | 0/5 |
| 0.4 | cora | nrgnn_vs_ball | +0.46pp | 3/5 |
| 0.4 | cora | nrgnn_vs_glnn | +3.11pp | 5/5 |
| 0.4 | cora | nrgnn_vs_knn | -0.55pp | 2/5 |
| 0.4 | cora | nrgnn_xh_vs_ball | -0.23pp | 2/5 |
| 0.4 | cora | nrgnn_xh_vs_gcn_xh | +0.14pp | 2/5 |
| 0.4 | cora | nrgnn_xh_vs_knn | -1.25pp | 1/5 |
| 0.6 | cora | mlp_distill_nrgnn_student_vs_own_teacher | -1.22pp | 0/5 |
| 0.6 | cora | mlp_distill_nrgnn_xh_student_vs_own_teacher | -0.38pp | 2/5 |
| 0.6 | cora | nrgnn_vs_ball | +3.54pp | 5/5 |
| 0.6 | cora | nrgnn_vs_glnn | +9.31pp | 5/5 |
| 0.6 | cora | nrgnn_vs_knn | +3.01pp | 5/5 |
| 0.6 | cora | nrgnn_xh_vs_ball | +3.23pp | 5/5 |
| 0.6 | cora | nrgnn_xh_vs_gcn_xh | +5.07pp | 5/5 |
| 0.6 | cora | nrgnn_xh_vs_knn | +2.70pp | 5/5 |
| 0.4 | corafull | mlp_distill_nrgnn_student_vs_own_teacher | -5.00pp | 0/5 |
| 0.4 | corafull | mlp_distill_nrgnn_xh_student_vs_own_teacher | -5.30pp | 0/5 |
| 0.4 | corafull | nrgnn_vs_ball | -0.52pp | 1/5 |
| 0.4 | corafull | nrgnn_vs_glnn | -1.04pp | 1/5 |
| 0.4 | corafull | nrgnn_vs_knn | -0.87pp | 0/5 |
| 0.4 | corafull | nrgnn_xh_vs_ball | -1.46pp | 0/5 |
| 0.4 | corafull | nrgnn_xh_vs_gcn_xh | -2.68pp | 0/5 |
| 0.4 | corafull | nrgnn_xh_vs_knn | -1.81pp | 0/5 |
| 0.6 | corafull | mlp_distill_nrgnn_student_vs_own_teacher | -5.15pp | 0/5 |
| 0.6 | corafull | mlp_distill_nrgnn_xh_student_vs_own_teacher | -5.61pp | 0/5 |
| 0.6 | corafull | nrgnn_vs_ball | +1.73pp | 5/5 |
| 0.6 | corafull | nrgnn_vs_glnn | +1.93pp | 4/5 |
| 0.6 | corafull | nrgnn_vs_knn | +0.90pp | 4/5 |
| 0.6 | corafull | nrgnn_xh_vs_ball | +0.95pp | 4/5 |
| 0.6 | corafull | nrgnn_xh_vs_gcn_xh | +0.03pp | 2/5 |
| 0.6 | corafull | nrgnn_xh_vs_knn | +0.13pp | 2/5 |

## Interpretation boundary

This experiment reports all pre-registered outcomes. It tests whether a
mature robust graph teacher can replace the consensus teacher under the
same X-only inference requirement; it does not change the base GB-LCSS
main-table claims.
