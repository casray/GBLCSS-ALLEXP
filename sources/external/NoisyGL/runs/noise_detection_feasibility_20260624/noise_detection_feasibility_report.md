# Phase 39: Label Noise Detection Feasibility Diagnostic

## Protocol

- Datasets: `cora,citeseer,coauthorcs`; seeds: `0,1,2,3,4`.
- Noise: `uniform` 0.4; split: percent 0.2/0.1/0.7.
- Primary detectors: `s_ball_loo`, `s_graphdens`; pass_auc=0.6.
- Clean labels are used only for evaluation labels (`noisy_train != clean_train`).

## Detector Summary

| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |
|---|---|---|---:|---:|---:|---:|
| citeseer | random | baseline | 0.4862 | -0.0060 | 0.3807 | 1.0000 |
| citeseer | s_ball | reference | 0.8304 | 0.3308 | 0.6950 | 1.0000 |
| citeseer | s_ball_loo | primary | 0.8257 | 0.2947 | 0.6892 | 1.0000 |
| citeseer | s_ballnbr | exploratory | 0.5399 | 0.0458 | 0.4184 | 1.0000 |
| citeseer | s_center | exploratory | 0.5277 | 0.0290 | 0.4021 | 1.0000 |
| citeseer | s_combo_H | exploratory | 0.6779 | 0.1376 | 0.5270 | 1.0000 |
| citeseer | s_combo_HG | exploratory | 0.6638 | 0.1215 | 0.4931 | 1.0000 |
| citeseer | s_entropy | exploratory | 0.4978 | 0.0067 | 0.3898 | 1.0000 |
| citeseer | s_gcn_conf | baseline | 0.7549 | 0.2876 | 0.6138 | 1.0000 |
| citeseer | s_gcn_entropy | baseline | 0.7191 | 0.2001 | 0.5638 | 1.0000 |
| citeseer | s_gcn_smallloss | baseline | 0.7549 | 0.2876 | 0.6138 | 1.0000 |
| citeseer | s_graphdens | primary | 0.5425 | 0.0466 | 0.4199 | 1.0000 |
| citeseer | s_graphdens_shuffled | shuffled | 0.5010 | 0.0076 | 0.3837 | 0.9994 |
| citeseer | s_graphnbr | baseline | 0.6202 | 0.0779 | 0.4969 | 0.3744 |
| citeseer | s_jsd_loo | exploratory | 0.8255 | 0.2883 | 0.6909 | 1.0000 |
| citeseer | s_knn | baseline | 0.8850 | 0.3860 | 0.7634 | 1.0000 |
| citeseer | s_lp | baseline | 0.5020 | 0.0020 | 0.4010 | 1.0000 |
| citeseer | s_shuffled | shuffled | 0.5210 | 0.0129 | 0.4028 | 0.9994 |
| coauthorcs | random | baseline | 0.5035 | 0.0019 | 0.4066 | 1.0000 |
| coauthorcs | s_ball | reference | 0.9663 | 0.5229 | 0.8998 | 0.9999 |
| coauthorcs | s_ball_loo | primary | 0.9614 | 0.4980 | 0.8994 | 0.9999 |
| coauthorcs | s_ballnbr | exploratory | 0.5107 | 0.0178 | 0.4121 | 0.9999 |
| coauthorcs | s_center | exploratory | 0.5060 | 0.0039 | 0.4076 | 1.0000 |
| coauthorcs | s_combo_H | exploratory | 0.7719 | 0.2331 | 0.6375 | 1.0000 |
| coauthorcs | s_combo_HG | exploratory | 0.7302 | 0.1804 | 0.5905 | 1.0000 |
| coauthorcs | s_entropy | exploratory | 0.4962 | -0.0043 | 0.3959 | 0.9999 |
| coauthorcs | s_gcn_conf | baseline | 0.9938 | 0.5867 | 0.9630 | 1.0000 |
| coauthorcs | s_gcn_entropy | baseline | 0.5651 | 0.0513 | 0.4610 | 1.0000 |
| coauthorcs | s_gcn_smallloss | baseline | 0.9938 | 0.5867 | 0.9630 | 1.0000 |
| coauthorcs | s_graphdens | primary | 0.5075 | 0.0200 | 0.4109 | 0.9999 |
| coauthorcs | s_graphdens_shuffled | shuffled | 0.5142 | 0.0117 | 0.4229 | 1.0000 |
| coauthorcs | s_graphnbr | baseline | 0.7572 | 0.2092 | 0.6681 | 0.7115 |
| coauthorcs | s_jsd_loo | exploratory | 0.9612 | 0.4982 | 0.8999 | 0.9999 |
| coauthorcs | s_knn | baseline | 0.9808 | 0.5468 | 0.9359 | 1.0000 |
| coauthorcs | s_lp | baseline | 0.5000 | 0.0000 | 0.3998 | 1.0000 |
| coauthorcs | s_shuffled | shuffled | 0.5951 | 0.0606 | 0.4864 | 1.0000 |
| cora | random | baseline | 0.4853 | -0.0140 | 0.3508 | 1.0000 |
| cora | s_ball | reference | 0.9011 | 0.4231 | 0.7720 | 1.0000 |
| cora | s_ball_loo | primary | 0.9016 | 0.4073 | 0.7762 | 1.0000 |
| cora | s_ballnbr | exploratory | 0.5149 | 0.0197 | 0.3911 | 1.0000 |
| cora | s_center | exploratory | 0.4895 | 0.0054 | 0.3649 | 1.0000 |
| cora | s_combo_H | exploratory | 0.6973 | 0.1426 | 0.5224 | 1.0000 |
| cora | s_combo_HG | exploratory | 0.6677 | 0.1114 | 0.4883 | 1.0000 |
| cora | s_entropy | exploratory | 0.4990 | -0.0005 | 0.3732 | 1.0000 |
| cora | s_gcn_conf | baseline | 0.9253 | 0.5224 | 0.8134 | 1.0000 |
| cora | s_gcn_entropy | baseline | 0.6966 | 0.1711 | 0.5432 | 1.0000 |
| cora | s_gcn_smallloss | baseline | 0.9253 | 0.5224 | 0.8134 | 1.0000 |
| cora | s_graphdens | primary | 0.5280 | 0.0338 | 0.4039 | 1.0000 |
| cora | s_graphdens_shuffled | shuffled | 0.5115 | 0.0116 | 0.3932 | 1.0000 |
| cora | s_graphnbr | baseline | 0.6771 | 0.1242 | 0.5202 | 0.4902 |
| cora | s_jsd_loo | exploratory | 0.9035 | 0.4199 | 0.7801 | 1.0000 |
| cora | s_knn | baseline | 0.9474 | 0.5100 | 0.8561 | 1.0000 |
| cora | s_lp | baseline | 0.5000 | 0.0000 | 0.3693 | 1.0000 |
| cora | s_shuffled | shuffled | 0.5531 | 0.0348 | 0.4205 | 1.0000 |

## Orthogonality Summary

| Dataset | Primary | AUC | real-shuffled | AUC in kNN-clean50 | AUC in GCN-clean50 | AUC in kNN∩GCN-clean50 | corr kNN | corr GCN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | s_ball_loo | 0.8257 | 0.3048 | 0.7211 | 0.8233 | 0.7071 | 0.7836 | 0.4583 |
| citeseer | s_graphdens | 0.5425 | 0.0415 | 0.6279 | 0.6229 | 0.6355 | 0.2263 | -0.0207 |
| coauthorcs | s_ball_loo | 0.9614 | 0.3662 | 0.7043 | 0.8192 | nan | 0.8968 | 0.8005 |
| coauthorcs | s_graphdens | 0.5075 | -0.0067 | 0.6926 | 0.7624 | nan | 0.1139 | 0.1252 |
| cora | s_ball_loo | 0.9016 | 0.3485 | 0.7156 | 0.8485 | 0.6747 | 0.8442 | 0.7029 |
| cora | s_graphdens | 0.5280 | 0.0165 | 0.7046 | 0.6644 | 0.8849 | 0.2745 | 0.1720 |

## Graph Increment Summary

| Dataset | combo_HG-H | graphdens-graphnbr |
|---|---:|---:|
| citeseer | -0.0141 | -0.0777 |
| coauthorcs | -0.0417 | -0.2498 |
| cora | -0.0296 | -0.1490 |

## Gate Notes

- G1 is not auto-declared by this report; inspect paired direction counts and the raw `orthogonality_raw.csv`.
- `best-of` and combo detectors are exploratory only and must not trigger go/no-go.
