# GAGB v2.2 Cora 0.4 Failure Analysis

## Purpose

Diagnose why v2.1 obtains positive NCR but unstable or lower accuracy on Cora noise=0.4. Clean labels are used only for diagnostics/evaluation.

## Per-class Summary

| Method | Class | Test Acc | NCR | CDR | Changed | Gated Noisy | Gated Clean | Teacher Gated Acc | Corrected Purity | Damaged Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gb | 0 | 0.6138 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 1 | 0.8947 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 2 | 0.9041 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 3 | 0.8252 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 4 | 0.8893 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 5 | 0.6971 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gb | 6 | 0.5556 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 0 | 0.6301 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 1 | 0.8947 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 2 | 0.9110 | 0.0000 | 0.0000 | 0.0000 | 13.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 3 | 0.8234 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 4 | 0.8859 | 0.0000 | 0.0000 | 0.0000 | 8.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 5 | 0.6587 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha03 | 6 | 0.5238 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha09 | 0 | 0.5894 | 0.0000 | 0.0278 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | NA | 0.9524 |
| gagb_qb_conflict_gate_alpha09 | 1 | 0.8882 | 0.2000 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha09 | 2 | 0.9384 | 0.3824 | 0.0000 | 13.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 3 | 0.8164 | 0.0303 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 4 | 0.8893 | 0.3200 | 0.0000 | 8.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 5 | 0.6635 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha09 | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_replace_qmix | 0 | 0.6016 | 0.0000 | 0.0278 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | NA | 0.9524 |
| gagb_qb_conflict_gate_replace_qmix | 1 | 0.8882 | 0.2000 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_replace_qmix | 2 | 0.9384 | 0.3824 | 0.0000 | 13.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 3 | 0.8304 | 0.0303 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 4 | 0.8893 | 0.3200 | 0.0000 | 8.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_replace_qmix | 5 | 0.6538 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_replace_qmix | 6 | 0.4444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_alpha09 | 0 | 0.6748 | 0.0294 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_alpha09 | 1 | 0.8750 | 0.2000 | 0.0303 | 3.0000 | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| khop_only_alpha09 | 2 | 0.9247 | 0.2647 | 0.0200 | 10.0000 | 9.0000 | 1.0000 | 0.9000 | 0.9286 | 0.7500 |
| khop_only_alpha09 | 3 | 0.8322 | 0.0152 | 0.0102 | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| khop_only_alpha09 | 4 | 0.8926 | 0.1200 | 0.0000 | 3.0000 | 3.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_alpha09 | 5 | 0.6394 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_alpha09 | 6 | 0.3889 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_replace | 0 | 0.6382 | 0.0294 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_replace | 1 | 0.8882 | 0.2000 | 0.0303 | 3.0000 | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| khop_only_replace | 2 | 0.9281 | 0.2647 | 0.0200 | 10.0000 | 9.0000 | 1.0000 | 0.9000 | 0.9286 | 0.7500 |
| khop_only_replace | 3 | 0.8269 | 0.0152 | 0.0102 | 2.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9286 |
| khop_only_replace | 4 | 0.8859 | 0.1200 | 0.0000 | 3.0000 | 3.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| khop_only_replace | 5 | 0.6587 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| khop_only_replace | 6 | 0.4603 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |

## Largest Cora 0.4 Class-level Accuracy Drops

- `gagb_qb_conflict_gate_alpha09` vs `gb` worst class `6`: delta `-0.0556`, NCR `0.0000`, CDR `0.0000`, gated noisy `0.0000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gb` worst class `6`: delta `-0.1111`, NCR `0.0000`, CDR `0.0000`, gated noisy `0.0000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha09` vs `gagb_qb_conflict_gate_alpha03` worst class `0`: delta `-0.0407`, NCR `0.0000`, CDR `0.0278`, gated noisy `0.0000`, gated clean `1.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gagb_qb_conflict_gate_alpha03` worst class `6`: delta `-0.0794`, NCR `0.0000`, CDR `0.0000`, gated noisy `0.0000`, gated clean `0.0000`.

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