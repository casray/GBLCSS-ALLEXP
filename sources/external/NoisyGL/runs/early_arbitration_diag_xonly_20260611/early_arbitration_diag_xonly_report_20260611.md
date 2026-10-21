# Phase 4a-2: X-Only Early-Learning Arbitration Diagnostic

Date: 2026-06-11

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`
- Datasets: Cora, CiteSeer
- Seeds: 0-2
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Warmup features: `x` only. The warmup GCN does not use frozen GRACE embedding H.
- Ball target source remains unchanged: SoftDist balls are still built on H.
- Warmup: fresh GCN, noisy CE only, 10 epochs.
- Early signal: mean softmax over epochs 3-10.
- Tau grid: 0.5, 0.6, 0.7, 0.8.

Clean labels are used only for diagnostic metrics. They do not enter warmup training, ball construction, target construction, or the protection rule.

## Mean over Seeds 0-2

| Dataset | tau | Baseline NCR | Protected NCR | NCR loss | Baseline CDR | Protected CDR | CDR reduction | Protected count | Protection precision | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | 0.5 | 0.6821 | 0.6724 | 0.0143 | 0.2061 | 0.1514 | 0.2609 | 25.0 | 0.7230 | no |
| Cora | 0.6 | 0.6821 | 0.6773 | 0.0071 | 0.2061 | 0.1738 | 0.1551 | 13.7 | 0.7863 | no |
| Cora | 0.7 | 0.6821 | 0.6821 | 0.0000 | 0.2061 | 0.1933 | 0.0606 | 5.3 | 0.7937 | no |
| Cora | 0.8 | 0.6821 | 0.6821 | 0.0000 | 0.2061 | 0.1991 | 0.0329 | 3.0 | 0.5714 | no |
| CiteSeer | 0.5 | 0.6029 | 0.3963 | 0.3416 | 0.2916 | 0.1111 | 0.6198 | 162.3 | 0.4644 | no |
| CiteSeer | 0.6 | 0.6029 | 0.4631 | 0.2309 | 0.2916 | 0.1415 | 0.5153 | 121.7 | 0.5135 | no |
| CiteSeer | 0.7 | 0.6029 | 0.5202 | 0.1358 | 0.2916 | 0.1846 | 0.3675 | 78.7 | 0.5669 | no |
| CiteSeer | 0.8 | 0.6029 | 0.5775 | 0.0425 | 0.2916 | 0.2316 | 0.2060 | 36.0 | 0.6921 | no |

## Gate Check

Required gates:

- CDR relative reduction >= 30% on both datasets.
- NCR relative loss <= 10%.
- Protection precision >= 70%.

No tau passes all three gates on both datasets.

## Decision

Phase 4 is permanently closed.

The X-only warmup removes the H-coupling concern from Phase 4a, but the early-learning protection signal remains insufficient. It protects too few nodes on Cora to reduce CDR by 30%, and on CiteSeer it protects too many noisy labels unless tau is high; high tau then fails the CDR reduction gate.

This supports the mechanism boundary: early-learning confidence, whether coupled with H (`xh` warmup) or independent from ball geometry (`x` warmup), is not reliable enough to safely protect clean labels in this pipeline.

## Artifacts

- Raw CSV: `runs/early_arbitration_diag_xonly_20260611/early_arbitration_diag_xonly_raw_20260611.csv`
- Aggregate CSV: `runs/early_arbitration_diag_xonly_20260611/early_arbitration_diag_xonly_aggregate_20260611.csv`
- Log: `runs/early_arbitration_diag_xonly_20260611/early_arbitration_diag_xonly_20260611.log`
