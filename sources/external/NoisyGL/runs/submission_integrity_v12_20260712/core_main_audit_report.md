# Submission Integrity Audit - Phase 1: Core Main Table

## Verdict

- Status: PASS for the core 5-dataset x 5-setting numerical claim.
- Recomputed rows: 1625 = 5 datasets x 5 settings x 13 methods x 5 seeds.
- All formal source rows report `noise_scope=train`, `modified_val_count=0`, `modified_test_count=0`, and `modified_count == modified_train_count`.
- Split and noise fingerprints are identical across paired methods within every dataset/setting/seed.
- LCSS exceeds shuffled grouping in 25/25 settings.
- In the historical 13-arm matrix, LCSS is best or tied in 24/25 settings; proxy rows in this matrix are not counted as published-method wins.

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

This phase verifies the historical core train-only label-noise matrix and its source CSVs. Published-method ranking is certified separately using the proxy-free Phase 4 scope; this pass does not by itself certify method fidelity, GRACE checkpoint provenance, or exact reproducibility from a clean checkout.

## Submission Blockers Already Visible

- The 13-arm historical matrix remains useful for source/protocol integrity, but its proxy rows are excluded from the formal published-method ranking; the faithful 25-setting claim is recomputed separately by `summarize_phase4_faithful_claims_20260712.py`.
- Exact reproduction from a clean checkout still requires packaging the frozen H files or a deterministic encoder-checkpoint regeneration path.
