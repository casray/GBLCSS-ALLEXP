# Phase 23 TSS CiteSeer Completion Report

Date: 2026-06-15  
Purpose: fill the missing CiteSeer TSS entry from the baseline table.

## Protocol

- Method: `tss`
- Dataset: `citeseer`
- Noise: `uniform`, `noise_rate=0.4`
- Split: `split_type=percent`, train/val/test = 0.2 / 0.1 / 0.7
- Seeds: 0-4
- Device: CPU
- Output: `runs/tss_citeseer_completion_20260615/tss_citeseer_cpu_summary.csv`
- Raw log: `runs/tss_citeseer_completion_20260615/tss_citeseer_cpu_5seed.log`

## Implementation Note

The previous ROCm run failed because TSS computes a dense Personalized PageRank matrix and hit `HIPBLAS_STATUS_ALLOC_FAILED`. The CPU retry first required a small device fix in `predictor/module/TSS.py`: `Personalized_PageRank` no longer hard-codes `.cuda()` and instead uses the adjacency tensor's device. This does not change the TSS algorithm.

## Results

| Seed | Test accuracy |
|---:|---:|
| 0 | 0.4674 |
| 1 | 0.5842 |
| 2 | 0.4601 |
| 3 | 0.5266 |
| 4 | 0.4893 |

Mean ± std: `0.5055 ± 0.0510`.

## Decision

- The missing CiteSeer TSS table cell can be filled.
- The value is weak and high-variance, so it should remain a baseline completion entry, not a method claim.
- No further TSS tuning is recommended in this project; the goal was table completion, not improving TSS.
