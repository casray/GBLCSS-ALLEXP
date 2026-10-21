# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `cora,citeseer,dblp,pubmed,amazonpho`
- Seeds: `0,1,2`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazonpho | 3 | 0.4039 | 129.00 | 0.8698 | 0.9467 | 0.7581 | 0.9483 | 0.6884 | 0.4095 | possible orthogonal signal |
| citeseer | 3 | 0.3739 | 64.00 | 0.6991 | 0.8383 | 0.7651 | 0.8286 | 0.7257 | 0.4705 | possible orthogonal signal |
| cora | 3 | 0.3758 | 52.33 | 0.7871 | 0.8957 | 0.9296 | 0.8460 | 0.8751 | 0.7001 | possible orthogonal signal |
| dblp | 3 | 0.3932 | 256.00 | 0.8377 | 0.9059 | 0.9598 | 0.6955 | 0.9059 | 0.7797 | possible orthogonal signal |
| pubmed | 3 | 0.3975 | 256.00 | 0.8289 | 0.8703 | 0.9405 | 0.6633 | 0.8714 | 0.7057 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| amazonpho | co_teaching_intersection | keep_0.5_each | 3 | 0.0143 | 0.3896 | 0.3495 |  |
| amazonpho | co_teaching_intersection | keep_0.6_each | 3 | 0.0640 | 0.3399 | 0.4353 |  |
| amazonpho | co_teaching_intersection | keep_0.7_each | 3 | 0.1478 | 0.2561 | 0.5401 |  |
| amazonpho | filter | top_0.2 | 3 | 0.2642 | 0.1397 | 0.7845 | 0.9124 |
| amazonpho | filter | top_0.3 | 3 | 0.1810 | 0.2230 | 0.6834 | 0.8850 |
| amazonpho | filter | top_0.4 | 3 | 0.0865 | 0.3174 | 0.5882 | 0.8574 |
| amazonpho | reweight | 1_minus_s_ball | 3 | 0.1582 | 0.2458 | 1.0000 |  |
| citeseer | co_teaching_intersection | keep_0.5_each | 3 | 0.0753 | 0.2986 | 0.3529 |  |
| citeseer | co_teaching_intersection | keep_0.6_each | 3 | 0.1155 | 0.2585 | 0.4541 |  |
| citeseer | co_teaching_intersection | keep_0.7_each | 3 | 0.1610 | 0.2129 | 0.5639 |  |
| citeseer | filter | top_0.2 | 3 | 0.2696 | 0.1044 | 0.7890 | 0.7643 |
| citeseer | filter | top_0.3 | 3 | 0.2035 | 0.1704 | 0.6607 | 0.7055 |
| citeseer | filter | top_0.4 | 3 | 0.1578 | 0.2161 | 0.5905 | 0.6856 |
| citeseer | reweight | 1_minus_s_ball | 3 | 0.2202 | 0.1537 | 1.0000 |  |
| cora | co_teaching_intersection | keep_0.5_each | 3 | 0.0225 | 0.3533 | 0.4085 |  |
| cora | co_teaching_intersection | keep_0.6_each | 3 | 0.0474 | 0.3285 | 0.5071 |  |
| cora | co_teaching_intersection | keep_0.7_each | 3 | 0.1030 | 0.2729 | 0.6106 |  |
| cora | filter | top_0.2 | 3 | 0.2670 | 0.1089 | 0.7985 | 0.8073 |
| cora | filter | top_0.3 | 3 | 0.1883 | 0.1876 | 0.6784 | 0.7709 |
| cora | filter | top_0.4 | 3 | 0.1195 | 0.2564 | 0.5866 | 0.7395 |
| cora | reweight | 1_minus_s_ball | 3 | 0.1730 | 0.2028 | 1.0000 |  |
| dblp | co_teaching_intersection | keep_0.5_each | 3 | 0.0259 | 0.3672 | 0.4351 |  |
| dblp | co_teaching_intersection | keep_0.6_each | 3 | 0.0572 | 0.3360 | 0.5382 |  |
| dblp | co_teaching_intersection | keep_0.7_each | 3 | 0.1077 | 0.2855 | 0.6238 |  |
| dblp | filter | top_0.2 | 3 | 0.2665 | 0.1267 | 0.7935 | 0.8806 |
| dblp | filter | top_0.3 | 3 | 0.1883 | 0.2049 | 0.6873 | 0.8438 |
| dblp | filter | top_0.4 | 3 | 0.1079 | 0.2852 | 0.5753 | 0.7795 |
| dblp | reweight | 1_minus_s_ball | 3 | 0.2198 | 0.1734 | 1.0000 |  |
| pubmed | co_teaching_intersection | keep_0.5_each | 3 | 0.0388 | 0.3587 | 0.4213 |  |
| pubmed | co_teaching_intersection | keep_0.6_each | 3 | 0.0695 | 0.3280 | 0.5245 |  |
| pubmed | co_teaching_intersection | keep_0.7_each | 3 | 0.1187 | 0.2788 | 0.6104 |  |
| pubmed | filter | top_0.2 | 3 | 0.2763 | 0.1212 | 0.7849 | 0.8395 |
| pubmed | filter | top_0.3 | 3 | 0.2106 | 0.1869 | 0.6913 | 0.8157 |
| pubmed | filter | top_0.4 | 3 | 0.1512 | 0.2463 | 0.5952 | 0.7596 |
| pubmed | reweight | 1_minus_s_ball | 3 | 0.2692 | 0.1283 | 1.0000 |  |

## Decision

At least one big graph passes the pre-registered conditional orthogonality gate. Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above.
Small-graph mean conditional ball AUC: `0.8373`; big-graph mean conditional ball AUC: `0.7691`.
