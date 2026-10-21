# Phase 1: [X||H] Classifier Seed0 ROCm Report

Date: 2026-06-10

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`, AMD Radeon RX 7700 XT
- Datasets: Cora, CiteSeer
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Seed: 0
- New methods:
  - `gcn_xh_ce`: GCN trained with noisy CE on `[X||H]`; no ball target.
  - `frozengrace_gb_gcn_softdist_xh`: SoftDist target + `[X||H]`.
  - `frozengrace_gb_gcn_softdist_xh_shuffled`: shuffled-ball control + `[X||H]`.

## Seed0 Results

| Dataset | GCN+CE anchor | SoftDist anchor | SoftDist shuffled anchor | GCN+CE [X||H] | SoftDist [X||H] | SoftDist [X||H] shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.7974 | 0.8042 | 0.8047 | 0.8106 | 0.8058 | 0.7805 |
| CiteSeer | 0.6435 | 0.6748 | 0.6491 | 0.6890 | 0.6959 | 0.6753 |

## Attribution Deltas

| Dataset | `gcn_xh_ce - GCN+CE` | `softdist_xh - SoftDist` | `softdist_xh - gcn_xh_ce` | `softdist_xh - xh_shuffled` |
|---|---:|---:|---:|---:|
| Cora | +1.32pp | +0.16pp | -0.48pp | +2.53pp |
| CiteSeer | +4.55pp | +2.11pp | +0.69pp | +2.06pp |

## Gate Decision

Phase 1 passes the pre-defined gate:

- `softdist_xh` improves over SoftDist by +2.11pp on CiteSeer.
- The other dataset, Cora, does not drop; it is +0.16pp over SoftDist.
- Real ball structure remains useful under `[X||H]`: +2.53pp on Cora and +2.06pp on CiteSeer over the shuffled-ball control.

## Interpretation Boundary

This is still a seed0 smoke result. The strongest signal is that frozen GRACE embeddings are useful as classifier features, especially on CiteSeer. The ball contribution under `[X||H]` is positive versus shuffled on both datasets, but on Cora `gcn_xh_ce` is slightly higher than `softdist_xh`, so the current evidence supports proceeding to Phase 2 diagnostics rather than claiming a stable final method.

## Artifacts

- Raw summary: `runs/xh_seed0_rocm_20260610/xh_seed0_rocm_raw_summary_20260610.csv`
- Logs: `runs/xh_seed0_rocm_20260610/logs/`
