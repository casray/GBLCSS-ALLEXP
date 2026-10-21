# Phase 4a: Early-Learning Arbitration Diagnostic

Date: 2026-06-11

## Protocol

- Runtime: `C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe`
- Device: `cuda:0`
- Datasets: Cora, CiteSeer
- Seeds: 0-2
- Split: `split_type=percent`, `train_percent=0.2`, `val_percent=0.1`, `test_percent=0.7`
- Noise: `noise_type=uniform`, `noise_rate=0.4`
- Feature setting: `[X||H]`, following Phase 5.
- Warmup model: fresh GCN, noisy CE only, 10 epochs.
- Early signal: mean softmax over epochs 3-10.
- Protection rule: for SoftDist train corrections where `pseudo_label != noisy_label`, protect original label if `p_early(y_noisy) >= tau_protect`.
- Tau grid: 0.5, 0.6, 0.7, 0.8.

Clean labels are used only for this diagnostic table. They do not enter warmup training, ball voting, target construction, or the protection rule.

## Mean over Seeds 0-2

| Dataset | tau | Baseline NCR | Protected NCR | NCR loss | Baseline CDR | Protected CDR | CDR reduction | Protected count | Protection precision | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | 0.5 | 0.6821 | 0.6661 | 0.0237 | 0.2061 | 0.1538 | 0.2521 | 24.7 | 0.7343 | no |
| Cora | 0.6 | 0.6821 | 0.6790 | 0.0045 | 0.2061 | 0.1785 | 0.1335 | 12.0 | 0.7900 | no |
| Cora | 0.7 | 0.6821 | 0.6821 | 0.0000 | 0.2061 | 0.1932 | 0.0624 | 5.3 | 0.8333 | no |
| Cora | 0.8 | 0.6821 | 0.6821 | 0.0000 | 0.2061 | 0.2000 | 0.0299 | 2.3 | 0.6111 | no |
| CiteSeer | 0.5 | 0.6029 | 0.4025 | 0.3313 | 0.2916 | 0.1073 | 0.6331 | 159.7 | 0.4805 | no |
| CiteSeer | 0.6 | 0.6029 | 0.4739 | 0.2137 | 0.2916 | 0.1511 | 0.4822 | 113.0 | 0.5163 | no |
| CiteSeer | 0.7 | 0.6029 | 0.5341 | 0.1143 | 0.2916 | 0.1791 | 0.3865 | 75.7 | 0.6188 | no |
| CiteSeer | 0.8 | 0.6029 | 0.5761 | 0.0450 | 0.2916 | 0.2287 | 0.2163 | 38.3 | 0.6815 | no |

## Gate Check

Pre-defined gates:

- CDR relative reduction >= 30% on both datasets.
- NCR relative loss <= 10%.
- Protection precision >= 70%.

No tau passes all three gates on both datasets.

- Cora: protection precision is acceptable at tau 0.5-0.7 and NCR loss is small, but CDR reduction never reaches 30%.
- CiteSeer: low tau values reduce CDR strongly, but they protect many actually noisy labels, causing excessive NCR loss and protection precision below 70%. Tau 0.8 controls NCR loss but CDR reduction drops below 30% and protection precision still stays below 70%.

## Decision

Phase 4a fails. Do not implement Phase 4b training.

The early-learning signal is not a reliable enough clean-label protector under this protocol. It either protects too few nodes on Cora or protects too many noisy nodes on CiteSeer. This keeps the method from safely reducing CDR without sacrificing NCR.

## Artifacts

- Raw CSV: `runs/early_arbitration_diag_20260611/early_arbitration_diag_raw_20260611.csv`
- Aggregate CSV: `runs/early_arbitration_diag_20260611/early_arbitration_diag_aggregate_20260611.csv`
- Log: `runs/early_arbitration_diag_20260611/early_arbitration_diag_20260611.log`
