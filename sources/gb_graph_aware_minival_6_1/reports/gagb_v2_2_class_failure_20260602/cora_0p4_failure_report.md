# GAGB v2.2 Cora 0.4 Failure Analysis

## Purpose

Diagnose why v2.1 obtains positive NCR but unstable or lower accuracy on Cora noise=0.4. Clean labels are used only for diagnostics/evaluation.

## Per-class Summary

| Method | Class | Test Acc | NCR | CDR | Changed | Gated Noisy | Gated Clean | Teacher Gated Acc | Corrected Purity | Damaged Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gb | 0 | 0.6862 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 1 | 0.7921 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 2 | 0.9315 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 3 | 0.8455 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 4 | 0.7866 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 5 | 0.7731 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 6 | 0.5492 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 0 | 0.6846 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.8000 | 0.0833 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 1 | 0.7803 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 2 | 0.9342 | 0.0000 | 0.0000 | 0.0000 | 8.6000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 3 | 0.8472 | 0.0000 | 0.0000 | 0.0000 | 3.8000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 4 | 0.7866 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 5 | 0.7654 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 6 | 0.5365 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha09 | 0 | 0.6732 | 0.0067 | 0.0202 | 1.2000 | 0.4000 | 0.8000 | 0.0833 | 1.0000 | 0.8440 |
| gagb_qb_conflict_gate_alpha09 | 1 | 0.7329 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha09 | 2 | 0.9418 | 0.2751 | 0.0000 | 8.6000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 3 | 0.8472 | 0.0566 | 0.0000 | 3.8000 | 3.8000 | 0.0000 | 1.0000 | 0.9826 | NA |
| gagb_qb_conflict_gate_alpha09 | 4 | 0.7906 | 0.1410 | 0.0000 | 4.0000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 5 | 0.7712 | 0.0251 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 6 | 0.5159 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_replace_qmix | 0 | 0.6724 | 0.0067 | 0.0202 | 1.2000 | 0.4000 | 0.8000 | 0.0833 | 1.0000 | 0.8440 |
| gagb_qb_conflict_gate_replace_qmix | 1 | 0.7053 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_replace_qmix | 2 | 0.9418 | 0.2751 | 0.0000 | 8.6000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 3 | 0.8517 | 0.0566 | 0.0000 | 3.8000 | 3.8000 | 0.0000 | 1.0000 | 0.9826 | NA |
| gagb_qb_conflict_gate_replace_qmix | 4 | 0.7906 | 0.1410 | 0.0000 | 4.0000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 5 | 0.7702 | 0.0251 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_alpha09 | 0 | 0.6911 | 0.0333 | 0.0050 | 1.2000 | 1.0000 | 0.2000 | 0.6667 | 1.0000 | 0.8462 |
| khop_only_alpha09 | 1 | 0.7592 | 0.0505 | 0.0135 | 1.2000 | 0.8000 | 0.4000 | 0.8333 | 1.0000 | 0.8874 |
| khop_only_alpha09 | 2 | 0.9322 | 0.0941 | 0.0199 | 4.2000 | 3.2000 | 1.0000 | 0.5667 | 0.9286 | 0.7500 |
| khop_only_alpha09 | 3 | 0.8493 | 0.0296 | 0.0020 | 2.2000 | 2.0000 | 0.2000 | 1.0000 | 0.9327 | 0.9286 |
| khop_only_alpha09 | 4 | 0.7872 | 0.0865 | 0.0039 | 2.8000 | 2.6000 | 0.2000 | 0.8000 | 1.0000 | 0.7500 |
| khop_only_alpha09 | 5 | 0.7673 | 0.0249 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_alpha09 | 6 | 0.4984 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_replace | 0 | 0.6821 | 0.0333 | 0.0050 | 1.2000 | 1.0000 | 0.2000 | 0.6667 | 1.0000 | 0.8462 |
| khop_only_replace | 1 | 0.7618 | 0.0505 | 0.0135 | 1.2000 | 0.8000 | 0.4000 | 0.8333 | 1.0000 | 0.8874 |
| khop_only_replace | 2 | 0.9322 | 0.0941 | 0.0199 | 4.2000 | 3.2000 | 1.0000 | 0.5667 | 0.9286 | 0.7500 |
| khop_only_replace | 3 | 0.8479 | 0.0296 | 0.0020 | 2.2000 | 2.0000 | 0.2000 | 1.0000 | 0.9327 | 0.9286 |
| khop_only_replace | 4 | 0.7866 | 0.0865 | 0.0039 | 2.8000 | 2.6000 | 0.2000 | 0.8000 | 1.0000 | 0.7500 |
| khop_only_replace | 5 | 0.7712 | 0.0249 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_replace | 6 | 0.5175 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |

## Largest Cora 0.4 Class-level Accuracy Drops

- `gagb_qb_conflict_gate_alpha09` vs `gb` worst class `1`: delta `-0.0592`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gb` worst class `1`: delta `-0.0868`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha09` vs `gagb_qb_conflict_gate_alpha03` worst class `1`: delta `-0.0474`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gagb_qb_conflict_gate_alpha03` worst class `1`: delta `-0.0750`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.

## Confusion Matrix Pointers

- `gagb_qb_conflict_gate_alpha03` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_replace_qmix` aggregate matrices are in `confusion_matrices.json`.
- `gb` aggregate matrices are in `confusion_matrices.json`.
- `khop_only_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `khop_only_replace` aggregate matrices are in `confusion_matrices.json`.

## Automatic Decision Hint

- No obvious high per-class clean damage or low damaged-ball purity was detected from aggregate means; inspect confusion matrices and qmix/teacher class bias before threshold sweep.
- Do not run gate threshold sweep until this report is reviewed against the confusion matrices.

## Manual Interpretation After Matrix Inspection

- The Cora 0.4 failure is not primarily explained by high aggregate or per-class CDR. For GAGB alpha09 / replace, direct clean damage is mostly concentrated in class `0`, with CDR about `0.0202`; the largest test accuracy drop is instead class `1`.
- The worst class-level drop is class `1`: alpha09 vs GB is `-0.0592`, replace vs GB is `-0.0868`. However class `1` has only about `0.4` gated noisy nodes on average and `0.0` gated clean nodes, so the class-level accuracy drop is larger than the direct intervention footprint.
- Class `6` also drops, but it has `0` gated noisy nodes and `0` gated clean nodes for GAGB repair. This is indirect classifier-boundary movement, not direct label correction or direct clean damage.
- GAGB qmix confusion on gated nodes is mostly clean-aligned for classes `2`, `3`, `4`, and `5`; class `0` has a small number of wrong qmix argmax cases, including redirects to classes `1/2/3/4`.
- K-hop repair changes more classes and causes more direct CDR, especially classes `1`, `2`, and `3`, but its class-level accuracy is sometimes less bad than GAGB on class `1`. This weakens a strong GAGB-builder claim.

## Next-step Decision From This Diagnostic

The next step should not be a broad gate-threshold sweep yet. The most supported next experiment is target-strength / boundary-sensitivity diagnosis:

1. Compare `replace_qmix`, `alpha09`, `alpha06/alpha07`, and a per-class capped update on Cora 0.4 only.
2. Track class `1` and class `6` test accuracy as primary failure indicators.
3. Add an intervention-strength matched shuffled or replay control that changes the same number of nodes/classes as GAGB, because current shuffled controls usually change almost nothing.
4. Keep ball-quality filtering as a secondary candidate; this diagnostic did not show low damaged-ball purity as the main failure source.
