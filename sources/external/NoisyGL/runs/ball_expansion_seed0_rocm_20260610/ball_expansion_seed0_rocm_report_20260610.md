# Phase 3: Ball Expansion Training Seed0 ROCm Report

Date: 2026-06-10

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`
- Datasets: Cora, CiteSeer
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Seed: 0
- Base target: SoftDist static train target.
- Expansion gate: `expansion_min_train_per_ball=10`, `expansion_margin_threshold=0.5`.
- Expansion loss: add `lambda_u * CE(unlabeled_candidate, ball_train_distribution)`.

## Seed0 Results

| Dataset | SoftDist static anchor | lambda_u=0.05 | Delta | lambda_u=0.10 | Delta |
|---|---:|---:|---:|---:|---:|
| Cora | 0.8042 | 0.7757 | -2.85pp | 0.7784 | -2.58pp |
| CiteSeer | 0.6748 | 0.6680 | -0.68pp | 0.6735 | -0.13pp |

## Gate Diagnostics Used for Training

Seed0 diagnostic for the selected gate:

| Dataset | Precision | Coverage over non-train | Candidate count | Correct candidates |
|---|---:|---:|---:|---:|
| Cora | 0.8308 | 0.4610 | 999 | 830 |
| CiteSeer | 0.8627 | 0.1559 | 415 | 358 |

## Decision

Phase 3 fails the seed0 smoke criterion. Despite high diagnostic precision, unlabeled expansion supervision hurts Cora substantially and does not improve CiteSeer. Do not expand this variant to 5 seeds.

The likely issue is not candidate precision alone. The expansion loss supervises many non-train nodes with a static ball distribution, which can over-regularize the classifier and interfere with validation-selected checkpoints. This is especially visible on Cora, where coverage is high (46.1% of non-train nodes) and the original SoftDist/GCN baselines are already strong.

## Artifacts

- Raw summary: `runs/ball_expansion_seed0_rocm_20260610/ball_expansion_seed0_rocm_raw_summary_20260610.csv`
- Logs: `runs/ball_expansion_seed0_rocm_20260610/logs/`
