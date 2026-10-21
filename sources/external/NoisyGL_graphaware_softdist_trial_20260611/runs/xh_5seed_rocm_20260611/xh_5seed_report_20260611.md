# Phase 5: [X||H] 5-Seed ROCm Confirmation

Date: 2026-06-11

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`, AMD Radeon RX 7700 XT
- Datasets: Cora, CiteSeer
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Seeds: 0-4
- New runs: `gcn_xh_ce`, `frozengrace_gb_gcn_softdist_xh`, `frozengrace_gb_gcn_softdist_xh_shuffled`
- Existing anchors reused from `runs/softdist_5seed_rocm_20260610`.

## Mean ± Std

| Dataset | Method | Test acc |
|---|---|---:|
| Cora | GCN+CE | 0.7842 ± 0.0151 |
| Cora | NRGNN | 0.8107 ± 0.0230 |
| Cora | SoftDist static | 0.7954 ± 0.0094 |
| Cora | SoftDist shuffled | 0.7832 ± 0.0167 |
| Cora | GCN+CE [X||H] | 0.8027 ± 0.0069 |
| Cora | SoftDist [X||H] | **0.8137 ± 0.0067** |
| Cora | SoftDist [X||H] shuffled | 0.7921 ± 0.0097 |
| CiteSeer | GCN+CE | 0.6530 ± 0.0121 |
| CiteSeer | NRGNN | **0.7027 ± 0.0108** |
| CiteSeer | SoftDist static | 0.6737 ± 0.0113 |
| CiteSeer | SoftDist shuffled | 0.6578 ± 0.0116 |
| CiteSeer | GCN+CE [X||H] | 0.6761 ± 0.0181 |
| CiteSeer | SoftDist [X||H] | 0.6891 ± 0.0159 |
| CiteSeer | SoftDist [X||H] shuffled | 0.6696 ± 0.0234 |

## Paired Deltas

| Dataset | Comparison | Mean delta | Positive seeds | Per-seed deltas |
|---|---|---:|---:|---|
| Cora | `gcn_xh_ce - GCN+CE` | +1.86pp | 5/5 | +1.32, +1.05, +3.43, +1.48, +2.00 |
| Cora | `softdist_xh - gcn_xh_ce` | +1.10pp | 4/5 | -0.48, +1.32, +2.01, +1.21, +1.43 |
| Cora | `softdist_xh - SoftDist` | +1.83pp | 5/5 | +0.16, +3.38, +3.43, +0.53, +1.64 |
| Cora | `softdist_xh - xh_shuffled` | +2.16pp | 5/5 | +2.53, +2.69, +3.33, +0.58, +1.69 |
| CiteSeer | `gcn_xh_ce - GCN+CE` | +2.31pp | 4/5 | +4.55, +0.77, -0.04, +2.49, +3.78 |
| CiteSeer | `softdist_xh - gcn_xh_ce` | +1.30pp | 5/5 | +0.69, +1.76, +1.81, +1.59, +0.65 |
| CiteSeer | `softdist_xh - SoftDist` | +1.54pp | 4/5 | +2.11, +0.04, -1.37, +5.58, +1.33 |
| CiteSeer | `softdist_xh - xh_shuffled` | +1.95pp | 5/5 | +2.06, +0.94, +3.53, +1.41, +1.81 |

## Decision

Phase 5 passes.

- H contribution is stable: `gcn_xh_ce - GCN+CE` is positive on Cora 5/5 seeds and CiteSeer 4/5 seeds.
- Ball contribution on top of H is also stable enough: `softdist_xh - gcn_xh_ce` is positive on Cora 4/5 seeds and CiteSeer 5/5 seeds.
- Real ball structure is necessary: `softdist_xh - softdist_xh_shuffled` is positive on both datasets 5/5 seeds.
- Cora seed0 was misleading. In 5 seeds, `softdist_xh` is the best mean result on Cora and is slightly above NRGNN mean (+0.31pp), though the paired margin is within NRGNN's variance. CiteSeer still trails NRGNN by 1.36pp mean.

## Implication for Phase 4

Use `[X||H]` as the feature setting for early-learning arbitration diagnostics. The remaining target is not to prove that balls help on Cora from scratch; 5-seed already supports that. The realistic Phase 4 goal is to reduce residual CDR and close the remaining CiteSeer gap to NRGNN without damaging the now-strong Cora result.

## Artifacts

- Raw XH summary: `runs/xh_5seed_rocm_20260611/xh_5seed_raw_summary_20260611.csv`
- Aggregate summary: `runs/xh_5seed_rocm_20260611/xh_5seed_aggregate_20260611.csv`
- Paired deltas: `runs/xh_5seed_rocm_20260611/xh_5seed_paired_deltas_20260611.csv`
- Logs: `runs/xh_5seed_rocm_20260611/logs/`
