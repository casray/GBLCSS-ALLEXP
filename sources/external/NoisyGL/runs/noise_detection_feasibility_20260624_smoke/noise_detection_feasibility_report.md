# Phase 39: Label Noise Detection Feasibility Diagnostic

## Protocol

- Datasets: `cora`; seeds: `0`.
- Noise: `uniform` 0.4; split: percent 0.2/0.1/0.7.
- Primary detectors: `s_ball_loo`, `s_graphdens`; pass_auc=0.6.
- Clean labels are used only for evaluation labels (`noisy_train != clean_train`).

## Detector Summary

| Dataset | Detector | Role | AUROC | AUPRC lift | P@K | Eligible |
|---|---|---|---:|---:|---:|---:|
| cora | random | baseline | 0.4520 | -0.0371 | 0.2857 | 1.0000 |
| cora | s_ball | reference | 0.8879 | 0.4139 | 0.7302 | 1.0000 |
| cora | s_ball_loo | primary | 0.8856 | 0.3894 | 0.7566 | 1.0000 |
| cora | s_ballnbr | exploratory | 0.5313 | 0.0477 | 0.3915 | 1.0000 |
| cora | s_center | exploratory | 0.5027 | 0.0055 | 0.3598 | 1.0000 |
| cora | s_combo_H | exploratory | 0.7045 | 0.1350 | 0.4868 | 1.0000 |
| cora | s_combo_HG | exploratory | 0.6778 | 0.1114 | 0.4709 | 1.0000 |
| cora | s_entropy | exploratory | 0.5133 | 0.0044 | 0.3704 | 1.0000 |
| cora | s_gcn_conf | baseline | 0.9383 | 0.5377 | 0.7937 | 1.0000 |
| cora | s_gcn_entropy | baseline | 0.6914 | 0.1519 | 0.4974 | 1.0000 |
| cora | s_gcn_smallloss | baseline | 0.9383 | 0.5377 | 0.7937 | 1.0000 |
| cora | s_graphdens | primary | 0.5283 | 0.0589 | 0.3757 | 1.0000 |
| cora | s_graphdens_shuffled | shuffled | 0.4996 | 0.0000 | 0.3704 | 1.0000 |
| cora | s_graphnbr | baseline | 0.6704 | 0.1188 | 0.4921 | 0.4658 |
| cora | s_jsd_loo | exploratory | 0.8894 | 0.4174 | 0.7566 | 1.0000 |
| cora | s_knn | baseline | 0.9462 | 0.5382 | 0.8571 | 1.0000 |
| cora | s_lp | baseline | 0.5000 | 0.0000 | 0.3862 | 1.0000 |
| cora | s_shuffled | shuffled | 0.5597 | 0.0361 | 0.4074 | 1.0000 |

## Orthogonality Summary

| Dataset | Primary | AUC | real-shuffled | AUC in kNN-clean50 | AUC in GCN-clean50 | AUC in kNN∩GCN-clean50 | corr kNN | corr GCN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cora | s_ball_loo | 0.8856 | 0.3260 | 0.6426 | 0.7870 | nan | 0.8336 | 0.6858 |
| cora | s_graphdens | 0.5283 | 0.0287 | 0.6891 | 0.6708 | nan | 0.3107 | 0.2483 |

## Graph Increment Summary

| Dataset | combo_HG-H | graphdens-graphnbr |
|---|---:|---:|
| cora | -0.0267 | -0.1422 |

## Gate Notes

- G1 is not auto-declared by this report; inspect paired direction counts and the raw `orthogonality_raw.csv`.
- `best-of` and combo detectors are exploratory only and must not trigger go/no-go.
