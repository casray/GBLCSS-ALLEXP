# Phase 2: Ball Expansion Precision Diagnostic

Date: 2026-06-10

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`
- Datasets: Cora, CiteSeer
- Seeds: 0, 1, 2
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Grid: `min_train_per_ball in {5,8,10}`, `margin_threshold in {0.5,0.6,0.7}`
- Real/shuffled comparison uses the same ball-size distribution.

## Gate Summary

Pre-defined gates:

- Precision >= 75%, ideally >= 80%.
- Coverage over non-train nodes >= 10%.
- Real precision - shuffled precision >= 20pp.

All Cora grid points pass the three gates. CiteSeer passes 6/9 grid points.

## Mean over Seeds 0-2

| Dataset | min_train | margin | Real precision | Real coverage | Avg candidates | Shuffled precision | Gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | 5 | 0.5 | 0.8507 | 0.5722 | 1240.0 | 0.2608 | +58.99pp | yes |
| Cora | 5 | 0.6 | 0.8955 | 0.3263 | 707.0 | 0.2325 | +66.30pp | yes |
| Cora | 5 | 0.7 | 0.9290 | 0.1647 | 357.0 | 0.1206 | +80.84pp | yes |
| Cora | 8 | 0.6 | 0.9139 | 0.2626 | 569.0 | 0.2325 | +68.14pp | yes |
| Cora | 10 | 0.5 | 0.8754 | 0.3692 | 800.0 | 0.2360 | +63.94pp | yes |
| Cora | 10 | 0.6 | 0.9282 | 0.1947 | 422.0 | 0.2360 | +69.23pp | yes |
| CiteSeer | 5 | 0.5 | 0.7135 | 0.4669 | 1243.0 | 0.1832 | +53.03pp | no: precision |
| CiteSeer | 5 | 0.6 | 0.7564 | 0.2626 | 699.0 | 0.2239 | +53.25pp | yes |
| CiteSeer | 5 | 0.7 | 0.7519 | 0.1213 | 323.0 | 0.0000 | +75.19pp | yes |
| CiteSeer | 8 | 0.5 | 0.7906 | 0.3356 | 893.3 | 0.1690 | +62.17pp | yes |
| CiteSeer | 8 | 0.6 | 0.8031 | 0.1882 | 501.0 | 0.0625 | +74.06pp | yes |
| CiteSeer | 10 | 0.5 | 0.8224 | 0.2595 | 690.7 | 0.1864 | +63.61pp | yes |
| CiteSeer | 10 | 0.6 | 0.8203 | 0.1514 | 403.0 | 0.0000 | +82.03pp | yes |

## Phase 3 Setting

Use one global gate for training:

- `expansion_min_train_per_ball=10`
- `expansion_margin_threshold=0.5`

Reason: among settings where both datasets reach at least 80% real precision, this setting has the largest candidate coverage on both datasets. It avoids per-dataset threshold tuning while keeping the diagnostic precision above the stronger 80% target.

## Artifacts

- Raw CSV: `runs/ball_expansion_diag_20260610/ball_expansion_diag_raw_20260610.csv`
- Log: `runs/ball_expansion_diag_20260610/ball_expansion_diag_20260610.log`
