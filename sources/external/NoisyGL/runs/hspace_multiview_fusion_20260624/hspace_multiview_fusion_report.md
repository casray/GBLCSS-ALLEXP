# Phase 40a: H-space Multi-view Rank Fusion Diagnostic

## Protocol

- Datasets: `cora,citeseer,coauthorcs`; seeds: `0,1,2,3,4`.
- Noise: `uniform` 0.4; split: percent 0.2/0.1/0.7.
- Single detectors: `s_knn`, `s_gcn_conf`, `s_ball_loo`, `s_graphnbr`.
- Primary fusion detectors: F1-F5 percentile-rank averages.
- Oracle logreg rows use clean/noisy flags from other seeds and are diagnostic only.

## Summary

| Dataset | Family | Detector | AUROC | ΔAUROC vs best single | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K | best single counts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| citeseer | fusion | F1_rank_knn_gcn | 0.8664 | -0.0186 | 0/5 | 0.3992 | 0.0132 | 0.7339 | -0.0295 | s_knn:5 |
| citeseer | fusion | F2_rank_knn_ball | 0.8735 | -0.0116 | 0/5 | 0.3876 | 0.0016 | 0.7363 | -0.0271 | s_knn:5 |
| citeseer | fusion | F3_rank_gcn_ball | 0.8413 | -0.0438 | 0/5 | 0.3548 | -0.0313 | 0.7052 | -0.0583 | s_knn:5 |
| citeseer | fusion | F4_rank_knn_gcn_ball | 0.8772 | -0.0079 | 1/5 | 0.4005 | 0.0145 | 0.7429 | -0.0205 | s_knn:5 |
| citeseer | fusion | F5_rank_knn_gcn_ball_graphnbr | 0.8519 | -0.0332 | 0/5 | 0.3710 | -0.0150 | 0.7090 | -0.0544 | s_knn:5 |
| citeseer | oracle | O1_logreg_knn_gcn | 0.8907 | 0.0057 | 4/5 | 0.4214 | 0.0354 | 0.7720 | 0.0086 | s_knn:5 |
| citeseer | oracle | O2_logreg_knn_gcn_ball | 0.8909 | 0.0059 | 5/5 | 0.4209 | 0.0349 | 0.7711 | 0.0077 | s_knn:5 |
| citeseer | single | s_ball_loo | 0.8257 | -0.0593 | 0/5 | 0.2947 | -0.0914 | 0.6892 | -0.0742 | s_knn:5 |
| citeseer | single | s_gcn_conf | 0.7549 | -0.1301 | 0/5 | 0.2876 | -0.0984 | 0.6138 | -0.1497 | s_knn:5 |
| citeseer | single | s_graphnbr | 0.6202 | -0.2649 | 0/5 | 0.0779 | -0.3081 | 0.4969 | -0.2665 | s_knn:5 |
| citeseer | single | s_knn | 0.8850 | 0.0000 | 0/5 | 0.3860 | 0.0000 | 0.7634 | 0.0000 | s_knn:5 |
| coauthorcs | fusion | F1_rank_knn_gcn | 0.9915 | -0.0023 | 0/5 | 0.5812 | -0.0055 | 0.9570 | -0.0060 | s_gcn_conf:5 |
| coauthorcs | fusion | F2_rank_knn_ball | 0.9768 | -0.0171 | 0/5 | 0.5432 | -0.0436 | 0.9231 | -0.0399 | s_gcn_conf:5 |
| coauthorcs | fusion | F3_rank_gcn_ball | 0.9867 | -0.0072 | 0/5 | 0.5727 | -0.0140 | 0.9375 | -0.0255 | s_gcn_conf:5 |
| coauthorcs | fusion | F4_rank_knn_gcn_ball | 0.9880 | -0.0059 | 0/5 | 0.5731 | -0.0136 | 0.9451 | -0.0179 | s_gcn_conf:5 |
| coauthorcs | fusion | F5_rank_knn_gcn_ball_graphnbr | 0.9824 | -0.0114 | 0/5 | 0.5630 | -0.0237 | 0.9223 | -0.0407 | s_gcn_conf:5 |
| coauthorcs | oracle | O1_logreg_knn_gcn | 0.9944 | 0.0006 | 5/5 | 0.5871 | 0.0004 | 0.9659 | 0.0028 | s_gcn_conf:5 |
| coauthorcs | oracle | O2_logreg_knn_gcn_ball | 0.9944 | 0.0006 | 5/5 | 0.5871 | 0.0003 | 0.9661 | 0.0031 | s_gcn_conf:5 |
| coauthorcs | single | s_ball_loo | 0.9614 | -0.0324 | 0/5 | 0.4980 | -0.0887 | 0.8994 | -0.0636 | s_gcn_conf:5 |
| coauthorcs | single | s_gcn_conf | 0.9938 | 0.0000 | 0/5 | 0.5867 | 0.0000 | 0.9630 | 0.0000 | s_gcn_conf:5 |
| coauthorcs | single | s_graphnbr | 0.7572 | -0.2366 | 0/5 | 0.2092 | -0.3775 | 0.6681 | -0.2949 | s_gcn_conf:5 |
| coauthorcs | single | s_knn | 0.9808 | -0.0131 | 0/5 | 0.5468 | -0.0400 | 0.9359 | -0.0271 | s_gcn_conf:5 |
| cora | fusion | F1_rank_knn_gcn | 0.9550 | 0.0076 | 5/5 | 0.5465 | 0.0365 | 0.8640 | 0.0079 | s_knn:5 |
| cora | fusion | F2_rank_knn_ball | 0.9401 | -0.0073 | 1/5 | 0.5170 | 0.0071 | 0.8281 | -0.0280 | s_knn:5 |
| cora | fusion | F3_rank_gcn_ball | 0.9426 | -0.0047 | 1/5 | 0.5210 | 0.0110 | 0.8295 | -0.0266 | s_knn:5 |
| cora | fusion | F4_rank_knn_gcn_ball | 0.9543 | 0.0070 | 4/5 | 0.5434 | 0.0334 | 0.8570 | 0.0009 | s_knn:5 |
| cora | fusion | F5_rank_knn_gcn_ball_graphnbr | 0.9394 | -0.0079 | 2/5 | 0.5175 | 0.0075 | 0.8266 | -0.0295 | s_knn:5 |
| cora | oracle | O1_logreg_knn_gcn | 0.9563 | 0.0090 | 5/5 | 0.5467 | 0.0367 | 0.8680 | 0.0119 | s_knn:5 |
| cora | oracle | O2_logreg_knn_gcn_ball | 0.9580 | 0.0106 | 5/5 | 0.5506 | 0.0406 | 0.8660 | 0.0099 | s_knn:5 |
| cora | single | s_ball_loo | 0.9016 | -0.0458 | 0/5 | 0.4073 | -0.1027 | 0.7762 | -0.0799 | s_knn:5 |
| cora | single | s_gcn_conf | 0.9253 | -0.0221 | 0/5 | 0.5224 | 0.0124 | 0.8134 | -0.0427 | s_knn:5 |
| cora | single | s_graphnbr | 0.6771 | -0.2703 | 0/5 | 0.1242 | -0.3857 | 0.5202 | -0.3359 | s_knn:5 |
| cora | single | s_knn | 0.9474 | 0.0000 | 0/5 | 0.5100 | 0.0000 | 0.8561 | 0.0000 | s_knn:5 |

## Gate Decision

FAIL: rank fusion does not show stable value over the per-seed best single detector.

Success requires a rank fusion detector to improve mean AUROC by at least +0.01 over the per-seed best single detector, with at least 4/5 positive seeds, on at least 2/3 datasets, without AUPRC/P@K degradation.

## Key Check

`F4_rank_knn_gcn_ball` vs per-seed best single: citeseer: ΔAUROC=-0.0079, pos=1/5; coauthorcs: ΔAUROC=-0.0059, pos=0/5; cora: ΔAUROC=0.0070, pos=4/5
