# Phase 40A: kNN + ball Ensemble Probe

## Protocol

- Datasets: `cora,citeseer,coauthorcs`; seeds: `0,1,2,3,4`.
- Noise: `uniform` 0.4; split: percent 0.2/0.1/0.7.
- Primary deployable probes: `rank_mean_knn_ball`, `z_mean_knn_ball`.
- Oracle diagnostic: `loo_logreg_knn_ball` uses labels from other seeds and is not deployable.

## Summary

| Dataset | Detector | AUROC | ΔAUROC vs kNN | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | loo_logreg_knn_ball | 0.8866 | 0.0016 | 4/5 | 0.4028 | 0.0168 | 0.7608 | -0.0026 |
| citeseer | rank_mean_knn_ball | 0.8735 | -0.0116 | 0/5 | 0.3876 | 0.0016 | 0.7363 | -0.0271 |
| citeseer | s_ball_loo | 0.8257 | -0.0593 | 0/5 | 0.2947 | -0.0914 | 0.6892 | -0.0742 |
| citeseer | s_knn | 0.8850 | 0.0000 | 0/5 | 0.3860 | 0.0000 | 0.7634 | 0.0000 |
| citeseer | z_mean_knn_ball | 0.8748 | -0.0102 | 0/5 | 0.3900 | 0.0039 | 0.7355 | -0.0280 |
| coauthorcs | loo_logreg_knn_ball | 0.9819 | 0.0011 | 5/5 | 0.5511 | 0.0043 | 0.9412 | 0.0052 |
| coauthorcs | rank_mean_knn_ball | 0.9768 | -0.0040 | 0/5 | 0.5432 | -0.0036 | 0.9231 | -0.0129 |
| coauthorcs | s_ball_loo | 0.9614 | -0.0194 | 0/5 | 0.4980 | -0.0488 | 0.8994 | -0.0365 |
| coauthorcs | s_knn | 0.9808 | 0.0000 | 0/5 | 0.5468 | 0.0000 | 0.9359 | 0.0000 |
| coauthorcs | z_mean_knn_ball | 0.9810 | 0.0002 | 3/5 | 0.5496 | 0.0029 | 0.9378 | 0.0019 |
| cora | loo_logreg_knn_ball | 0.9505 | 0.0031 | 5/5 | 0.5312 | 0.0213 | 0.8587 | 0.0026 |
| cora | rank_mean_knn_ball | 0.9401 | -0.0073 | 1/5 | 0.5170 | 0.0071 | 0.8281 | -0.0280 |
| cora | s_ball_loo | 0.9016 | -0.0458 | 0/5 | 0.4073 | -0.1027 | 0.7762 | -0.0799 |
| cora | s_knn | 0.9474 | 0.0000 | 0/5 | 0.5100 | 0.0000 | 0.8561 | 0.0000 |
| cora | z_mean_knn_ball | 0.9451 | -0.0022 | 2/5 | 0.5222 | 0.0122 | 0.8439 | -0.0122 |

## Gate Decision

FAIL: deployable kNN+ball ensemble does not show stable auxiliary value over kNN.

Success requires a deployable ensemble to improve mean AUROC by at least +0.01 over `s_knn`, with at least 4/5 positive seeds, on at least 2/3 datasets, without AUPRC/P@K degradation.
