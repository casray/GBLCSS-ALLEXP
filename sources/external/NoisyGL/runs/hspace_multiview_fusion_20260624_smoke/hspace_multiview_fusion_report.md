# Phase 40a: H-space Multi-view Rank Fusion Diagnostic

## Protocol

- Datasets: `cora`; seeds: `0`.
- Noise: `uniform` 0.4; split: percent 0.2/0.1/0.7.
- Single detectors: `s_knn`, `s_gcn_conf`, `s_ball_loo`, `s_graphnbr`.
- Primary fusion detectors: F1-F5 percentile-rank averages.
- Oracle logreg rows use clean/noisy flags from other seeds and are diagnostic only.

## Summary

| Dataset | Family | Detector | AUROC | ΔAUROC vs best single | pos/5 | AUPRC lift | ΔAUPRC lift | P@K | ΔP@K | best single counts |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cora | fusion | F1_rank_knn_gcn | 0.9576 | 0.0114 | 1/1 | 0.5679 | 0.0297 | 0.8624 | 0.0053 | s_knn:1 |
| cora | fusion | F2_rank_knn_ball | 0.9333 | -0.0129 | 0/1 | 0.5270 | -0.0112 | 0.8042 | -0.0529 | s_knn:1 |
| cora | fusion | F3_rank_gcn_ball | 0.9412 | -0.0050 | 0/1 | 0.5167 | -0.0215 | 0.8254 | -0.0317 | s_knn:1 |
| cora | fusion | F4_rank_knn_gcn_ball | 0.9520 | 0.0058 | 1/1 | 0.5520 | 0.0138 | 0.8519 | -0.0053 | s_knn:1 |
| cora | fusion | F5_rank_knn_gcn_ball_graphnbr | 0.9350 | -0.0112 | 0/1 | 0.5135 | -0.0247 | 0.8201 | -0.0370 | s_knn:1 |
| cora | oracle | O1_logreg_knn_gcn | 0.5000 | -0.4462 | 0/1 | 0.0000 | -0.5382 | 0.3862 | -0.4709 | s_knn:1 |
| cora | oracle | O2_logreg_knn_gcn_ball | 0.5000 | -0.4462 | 0/1 | 0.0000 | -0.5382 | 0.3862 | -0.4709 | s_knn:1 |
| cora | single | s_ball_loo | 0.8856 | -0.0606 | 0/1 | 0.3894 | -0.1487 | 0.7566 | -0.1005 | s_knn:1 |
| cora | single | s_gcn_conf | 0.9383 | -0.0079 | 0/1 | 0.5377 | -0.0005 | 0.7937 | -0.0635 | s_knn:1 |
| cora | single | s_graphnbr | 0.6704 | -0.2757 | 0/1 | 0.1188 | -0.4194 | 0.4921 | -0.3651 | s_knn:1 |
| cora | single | s_knn | 0.9462 | 0.0000 | 0/1 | 0.5382 | 0.0000 | 0.8571 | 0.0000 | s_knn:1 |

## Gate Decision

FAIL: rank fusion does not show stable value over the per-seed best single detector.

Success requires a rank fusion detector to improve mean AUROC by at least +0.01 over the per-seed best single detector, with at least 4/5 positive seeds, on at least 2/3 datasets, without AUPRC/P@K degradation.

## Key Check

`F4_rank_knn_gcn_ball` vs per-seed best single: cora: ΔAUROC=0.0058, pos=1/1
