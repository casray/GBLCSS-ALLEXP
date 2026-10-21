# Submission Integrity Audit - Phase 1: Core Main Table

## Verdict

- Status: PASS for the core 5-dataset x 5-setting numerical claim.
- Recomputed rows: 1625 = 5 datasets x 5 settings x 13 methods x 5 seeds.
- All formal source rows report `noise_scope=train`, `modified_val_count=0`, `modified_test_count=0`, and `modified_count == modified_train_count`.
- Split and noise fingerprints are identical across paired methods within every dataset/setting/seed.
- LCSS exceeds shuffled grouping in 25/25 settings.
- LCSS is best or tied among the 11 non-LCSS robust baselines in 24/25 settings.

## Per-dataset Claim Check

| Dataset | LCSS best/tied settings |
|---|---:|
| Cora | 5/5 |
| CiteSeer | 5/5 |
| CoraFull | 5/5 |
| Coauthor-CS | 4/5 |
| Amazon-Photo | 5/5 |

## Uniform 0.4 Recomputed Slice

| Dataset | LCSS | Best non-LCSS baseline | Shuffled |
|---|---:|---:|---:|
| Cora | 55.59+-1.50 | mlp_small_loss 51.95+-1.97 | 43.63+-1.62 |
| CiteSeer | 53.83+-2.75 | mlp_jocor 49.91+-3.37 | 42.35+-2.74 |
| CoraFull | 37.33+-0.81 | mlp_jocor 35.70+-0.32 | 32.27+-0.81 |
| Coauthor-CS | 90.62+-0.57 | mlp_jocor 89.38+-0.61 | 87.60+-1.03 |
| Amazon-Photo | 81.64+-0.93 | mlp_jocor 79.57+-0.53 | 72.06+-2.97 |

## Scope of This Pass

This phase verifies the core train-only label-noise table from formal seed rows and their source CSVs. It does not yet certify Tables 3-5, literature claims, adapted-baseline fidelity, GRACE checkpoint provenance, or exact reproducibility from a clean checkout.

## Submission Blockers Already Visible

- `single_exp.py` and `total_exp.py` still call `label_process` without an eligible train mask; those legacy entry points corrupt all labels. The citable main-table rows audited here came from `scripts/run_standard_downstream.py`, which does pass the train mask, but the public reproduction interface is unsafe until fixed or removed.
- The manuscript reference list still contains placeholder entries and a duplicated `## 参考文献` heading; citation integrity cannot pass while these remain.
- Several adapted/reimplemented baselines require a separate algorithm-fidelity audit before their names can be presented without stronger qualification.
