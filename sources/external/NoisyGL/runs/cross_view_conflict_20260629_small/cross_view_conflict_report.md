# Phase 40: Cross-view conflict diagnostic

Gate decision: STOP

## Protocol

- Datasets: `cora,citeseer`; seeds: `0,1,2,3,4`.
- Noise: `uniform` 0.4; scope `train`.
- Split: `percent` 0.2/0.1/0.7.
- kNN k: `20`; pass_auc: `0.6`; double-clean min noisy: `8`.
- Scores use train noisy labels only. Clean labels are used only for evaluation.

## Gate Summary

| Dataset | Detector | H-clean AUC | X-clean AUC | Both-clean minus combo | H pass seeds | X pass seeds | Conflict noise delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| citeseer | s_xview | 0.4959 | 0.4900 | -0.3036 | 0/5 | 0/5 | -0.0314 | FAIL |
| citeseer | s_xview_soft | 0.4878 | 0.4959 | -0.2788 | 0/5 | 0/5 | -0.0314 | FAIL |
| cora | s_xview | 0.4485 | 0.5179 | -0.2868 | 0/5 | 0/5 | -0.0317 | FAIL |
| cora | s_xview_soft | 0.4098 | 0.5266 | -0.2950 | 0/5 | 1/5 | -0.0317 | FAIL |

## Detector Summary

| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |
|---|---|---|---:|---:|---:|---:|
| citeseer | random | baseline | 0.5060 | 0.0140 | 0.4035 | 1.0000 |
| citeseer | s_ball_loo_H | baseline | 0.8290 | 0.2976 | 0.7027 | 1.0000 |
| citeseer | s_ball_loo_X | baseline | 0.5732 | 0.0741 | 0.4567 | 1.0000 |
| citeseer | s_combo_2knn_max | baseline | 0.8543 | 0.3297 | 0.7233 | 1.0000 |
| citeseer | s_combo_2knn_mean | baseline | 0.8900 | 0.3949 | 0.7769 | 1.0000 |
| citeseer | s_knn_H | baseline | 0.8771 | 0.3647 | 0.7580 | 1.0000 |
| citeseer | s_knn_X | baseline | 0.8058 | 0.2722 | 0.6700 | 1.0000 |
| citeseer | s_xview | primary | 0.4916 | -0.0030 | 0.4091 | 1.0000 |
| citeseer | s_xview_disagree | exploratory | 0.6796 | 0.2132 | 0.6407 | 1.0000 |
| citeseer | s_xview_soft | primary | 0.4989 | 0.0002 | 0.3997 | 1.0000 |
| cora | random | baseline | 0.5251 | 0.0257 | 0.4367 | 1.0000 |
| cora | s_ball_loo_H | baseline | 0.8876 | 0.3773 | 0.7716 | 1.0000 |
| cora | s_ball_loo_X | baseline | 0.6387 | 0.1033 | 0.5084 | 1.0000 |
| cora | s_combo_2knn_max | baseline | 0.9004 | 0.3986 | 0.7870 | 1.0000 |
| cora | s_combo_2knn_mean | baseline | 0.9408 | 0.4842 | 0.8434 | 1.0000 |
| cora | s_knn_H | baseline | 0.9361 | 0.4676 | 0.8348 | 1.0000 |
| cora | s_knn_X | baseline | 0.8341 | 0.3087 | 0.7034 | 1.0000 |
| cora | s_xview | primary | 0.4868 | -0.0057 | 0.3929 | 1.0000 |
| cora | s_xview_disagree | exploratory | 0.6839 | 0.2636 | 0.6621 | 1.0000 |
| cora | s_xview_soft | primary | 0.5090 | 0.0047 | 0.3977 | 1.0000 |

## Conditional AUC Summary

| Dataset | Detector | H clean50 | X clean50 | H∩X clean50 | H∩X minus combo | corr H | corr X |
|---|---|---:|---:|---:|---:|---:|---:|
| citeseer | s_combo_2knn_max | 0.7738 | 0.8680 | 0.7669 | 0.0000 | 0.8139 | 0.8865 |
| citeseer | s_combo_2knn_mean | 0.8369 | 0.8939 | 0.8291 | 0.0622 | 0.9316 | 0.8592 |
| citeseer | s_xview | 0.4959 | 0.4900 | 0.4633 | -0.3036 | 0.0215 | 0.0523 |
| citeseer | s_xview_disagree | 0.6420 | 0.6739 | 0.5939 | -0.1730 | 0.2129 | 0.1806 |
| citeseer | s_xview_soft | 0.4878 | 0.4959 | 0.4881 | -0.2788 | -0.1002 | -0.0703 |
| cora | s_combo_2knn_max | 0.7438 | 0.9145 | 0.7689 | 0.0000 | 0.7981 | 0.9041 |
| cora | s_combo_2knn_mean | 0.8159 | 0.9385 | 0.8783 | 0.1094 | 0.9445 | 0.8467 |
| cora | s_xview | 0.4485 | 0.5179 | 0.4821 | -0.2868 | 0.0073 | 0.0450 |
| cora | s_xview_disagree | 0.5556 | 0.7148 | 0.5930 | -0.1759 | 0.2140 | 0.1811 |
| cora | s_xview_soft | 0.4098 | 0.5266 | 0.4738 | -0.2950 | -0.1614 | -0.0679 |

## Conflict Summary

| Dataset | Conflict rate | Conflict noise | Non-conflict noise | Delta | Eligible |
|---|---:|---:|---:|---:|---:|
| citeseer | 0.7708 | 0.4036 | 0.4351 | -0.0314 | 1.0000 |
| cora | 0.7242 | 0.3970 | 0.4287 | -0.0317 | 1.0000 |

## Decision

- STOP: no primary satisfied the preregistered cross-view non-homologous gate on >=2/3 datasets.
- This supports closing new granular-ball variants for graph-node denoising and using the result as boundary evidence.
