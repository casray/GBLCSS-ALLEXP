# R2LP and Legacy Baseline Audit

Date: 2026-06-23

## Scope

- Protocol: 20% / 10% / 70%, uniform noise 0.4, train-only corruption.
- Datasets audited: Cora, CiteSeer, Coauthor-CS.
- Methods audited: R2LP and the legacy appendix baselines.

## Confirmed R2LP Defect

The previous NoisyGL R2LP adapter initialized its noisy-label pool from the
first `pre_unknown * N` natural node indices. This pool crossed the randomized
train/validation/test split and directly inserted clean validation and test
labels into `y_unknown`.

For Cora, the leaked pool contained 938-967 test nodes per seed, approximately
half of the test set. Because validation and test labels were no longer
corrupted under the corrected train-only protocol, 90.9%-92.8% of the labels
in this pool were correct. The previous Cora result, 0.8874 +/- 0.0165, is
therefore invalid.

## Fix

- `predictor/R2LP_Predictor.py` now builds `idx_unknown` exclusively from
  `train_mask`.
- `y_unknown` is zero outside the training split.
- A regression test verifies that validation and test nodes cannot enter the
  initial R2LP label state.
- The R2LP training loop, propagation formula, model config, and validation
  selection logic were not changed.

## Corrected Results

| Dataset | Corrected R2LP | Previous invalid R2LP | Mean change |
|---|---:|---:|---:|
| Cora | 0.6095 +/- 0.0340 | 0.8874 +/- 0.0165 | -27.79 pp |
| CiteSeer | 0.3431 +/- 0.0532 | 0.4229 +/- 0.1014 | -7.97 pp |

SoftDist [X||H] exceeds corrected R2LP by 20.46 pp on Cora and 34.23 pp on
CiteSeer.

## Protocol Re-audit

All 10 corrected R2LP runs satisfy:

- status is `ok`;
- `modified_val_count = 0`;
- `modified_test_count = 0`;
- split hash matches the formal main-table run for the same dataset and seed;
- noise hash matches the formal main-table run for the same dataset and seed;
- initial label pool equals the complete train split;
- validation overlap is zero;
- test overlap is zero;
- `y_unknown` has no nonzero entry outside train.

The first no-clean-set iteration has an empty `idx_clean`, so the diagnostic
training accuracy is NaN. The empty NLL loss produces zero gradients rather
than NaN parameters. Label propagation from the train-only noisy pool then
bootstraps the selected clean set. This is retained as a reporting caveat, not
treated as label leakage.

## Legacy Baseline Failures

| Dataset / Method | Runs | Confirmed cause | Reporting decision |
|---|---:|---|---|
| CiteSeer / TSS | 5 | Dense PPR inversion caused ROCm `HIPBLAS_STATUS_ALLOC_FAILED` | OOM / incomplete |
| Coauthor-CS / UnionNET | 5 | No `unionnet_coauthorcs.yaml` | unsupported / incomplete |
| Coauthor-CS / CRGNN | 5 | No `crgnn_coauthorcs.yaml` | unsupported / incomplete |
| Coauthor-CS / CGNN | 5 | No `cgnn_coauthorcs.yaml` | unsupported / incomplete |
| Coauthor-CS / TSS | 5 | No `tss_coauthorcs.yaml` | unsupported / incomplete |
| Coauthor-CS / R2LP | 5 | No `r2lp_coauthorcs.yaml` | unsupported / incomplete |

No Coauthor-CS hyperparameters were copied from another dataset. These cells
must remain incomplete unless a separately documented tuning protocol is run.

## Artifact Lineage

- Invalid appendix backup: `appendix_raw_pre_r2lp_audit.csv`
- Invalid summary backup: `summary_pre_r2lp_audit.csv`
- Corrected raw R2LP runs: `r2lp_trainonly_fixed_raw.csv`
- Updated appendix table: `appendix_raw.csv`
- Updated combined summary: `summary.csv`

## Final Decision

- Use the corrected R2LP Cora and CiteSeer rows.
- Never cite the previous R2LP Cora 0.8874 result.
- Keep CiteSeer TSS and all five Coauthor-CS legacy methods incomplete.
- Do not create Coauthor-CS configs by transferring Cora or CiteSeer
  hyperparameters.
