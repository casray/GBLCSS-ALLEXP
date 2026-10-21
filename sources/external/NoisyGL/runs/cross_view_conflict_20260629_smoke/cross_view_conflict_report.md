# Phase 40: Cross-view conflict diagnostic

Gate decision: STOP

## Protocol

- Datasets: `cora`; seeds: `0`.
- Noise: `uniform` 0.4; scope `train`.
- Split: `percent` 0.2/0.1/0.7.
- kNN k: `20`; pass_auc: `0.6`; double-clean min noisy: `8`.
- Scores use train noisy labels only. Clean labels are used only for evaluation.

## Gate Summary

| Dataset | Detector | H-clean AUC | X-clean AUC | Both-clean minus combo | H pass seeds | X pass seeds | Conflict noise delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cora | s_xview | 0.3303 | 0.4642 | -0.3731 | 0/5 | 0/5 | -0.0888 | FAIL |
| cora | s_xview_soft | 0.4244 | 0.6001 | -0.2065 | 0/5 | 1/5 | -0.0888 | FAIL |

## Detector Summary

| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |
|---|---|---|---:|---:|---:|---:|
| cora | random | baseline | 0.4957 | 0.0001 | 0.4261 | 1.0000 |
| cora | s_ball_loo_H | baseline | 0.8780 | 0.3652 | 0.7826 | 1.0000 |
| cora | s_ball_loo_X | baseline | 0.6460 | 0.1027 | 0.5391 | 1.0000 |
| cora | s_combo_2knn_max | baseline | 0.8839 | 0.3753 | 0.7739 | 1.0000 |
| cora | s_combo_2knn_mean | baseline | 0.9305 | 0.4608 | 0.8391 | 1.0000 |
| cora | s_knn_H | baseline | 0.9225 | 0.4389 | 0.8261 | 1.0000 |
| cora | s_knn_X | baseline | 0.8135 | 0.2875 | 0.6913 | 1.0000 |
| cora | s_xview | primary | 0.4605 | -0.0180 | 0.3913 | 1.0000 |
| cora | s_xview_disagree | exploratory | 0.6329 | 0.2326 | 0.6130 | 1.0000 |
| cora | s_xview_soft | primary | 0.5274 | 0.0118 | 0.4435 | 1.0000 |

## Conditional AUC Summary

| Dataset | Detector | H clean50 | X clean50 | H∩X clean50 | H∩X minus combo | corr H | corr X |
|---|---|---:|---:|---:|---:|---:|---:|
| cora | s_combo_2knn_max | 0.6928 | 0.8755 | 0.6870 | 0.0000 | 0.7908 | 0.8776 |
| cora | s_combo_2knn_mean | 0.8073 | 0.9209 | 0.8321 | 0.1451 | 0.9318 | 0.8341 |
| cora | s_xview | 0.3303 | 0.4642 | 0.3139 | -0.3731 | 0.0166 | 0.0147 |
| cora | s_xview_disagree | 0.3455 | 0.6267 | 0.3153 | -0.3717 | 0.1999 | 0.1176 |
| cora | s_xview_soft | 0.4244 | 0.6001 | 0.4805 | -0.2065 | -0.1951 | -0.0213 |

## Conflict Summary

| Dataset | Conflict rate | Conflict noise | Non-conflict noise | Delta | Eligible |
|---|---:|---:|---:|---:|---:|
| cora | 0.6802 | 0.3967 | 0.4855 | -0.0888 | 1.0000 |

## Decision

- STOP: no primary satisfied the preregistered cross-view non-homologous gate on >=2/3 datasets.
- This supports closing new granular-ball variants for graph-node denoising and using the result as boundary evidence.
