# GAGB v2.3 Class-Safe Target Calibration Report

## Purpose

Test whether Cora noise=0.4 failure is caused by over-strong target updates, class-imbalanced correction, or indirect decision-boundary movement. Clean labels are used only for diagnostics/evaluation.

## Protocol Boundary

- GRACE embeddings, GAGB construction, teacher training, split, noise protocol, and classifier protocol are unchanged from v2.2.
- Train-time target construction uses only noisy labels, train mask, qmix, teacher probabilities, and grouping.
- `gagb_replay_same_class_count` is intentionally not implemented as a trainable method because it would require clean class counts.

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
| gagb_qb_conflict_gate_alpha07 | 0 | 0.5854 | 0.0000 | 0.0278 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | NA | 0.9524 |
| gagb_qb_conflict_gate_alpha07 | 1 | 0.8882 | 0.2000 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha07 | 2 | 0.9281 | 0.3529 | 0.0000 | 12.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha07 | 3 | 0.8182 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha07 | 4 | 0.8893 | 0.2000 | 0.0000 | 5.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha07 | 5 | 0.6779 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha07 | 6 | 0.5079 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha09 | 0 | 0.5894 | 0.0000 | 0.0278 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | NA | 0.9524 |
| gagb_qb_conflict_gate_alpha09 | 1 | 0.8882 | 0.2000 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha09 | 2 | 0.9384 | 0.3824 | 0.0000 | 13.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 3 | 0.8164 | 0.0303 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 4 | 0.8893 | 0.3200 | 0.0000 | 8.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha09 | 5 | 0.6635 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha09 | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_class_capped_alpha09 | 0 | 0.5935 | 0.0000 | 0.0278 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | NA | 0.9524 |
| gagb_class_capped_alpha09 | 1 | 0.8947 | 0.2000 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_class_capped_alpha09 | 2 | 0.9144 | 0.1176 | 0.0000 | 4.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 3 | 0.8269 | 0.0303 | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 4 | 0.8826 | 0.1600 | 0.0000 | 4.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 5 | 0.6635 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_class_capped_alpha09 | 6 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_boundary_safe_alpha09 | 0 | 0.5976 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | NA | NA |
| gagb_boundary_safe_alpha09 | 1 | 0.8947 | 0.1000 | 0.0000 | 1.0000 | 2.0000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_boundary_safe_alpha09 | 2 | 0.9212 | 0.1765 | 0.0000 | 6.0000 | 13.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_boundary_safe_alpha09 | 3 | 0.8234 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_boundary_safe_alpha09 | 4 | 0.8826 | 0.1200 | 0.0000 | 3.0000 | 8.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_boundary_safe_alpha09 | 5 | 0.7019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_boundary_safe_alpha09 | 6 | 0.5476 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_replay_random_same_count | 0 | 0.5854 | 0.0000 | 0.0833 | 3.0000 | 0.0000 | 3.0000 | 0.3333 | NA | 0.8462 |
| gagb_replay_random_same_count | 1 | 0.8882 | 0.1000 | 0.0909 | 5.0000 | 2.0000 | 3.0000 | 0.8000 | 0.5000 | 0.9467 |
| gagb_replay_random_same_count | 2 | 0.9007 | 0.0000 | 0.0600 | 4.0000 | 1.0000 | 3.0000 | 1.0000 | NA | 1.0000 |
| gagb_replay_random_same_count | 3 | 0.8077 | 0.0000 | 0.0612 | 6.0000 | 0.0000 | 6.0000 | 0.6667 | NA | 0.9722 |
| gagb_replay_random_same_count | 4 | 0.8859 | 0.0000 | 0.0667 | 4.0000 | 0.0000 | 4.0000 | 1.0000 | NA | 1.0000 |
| gagb_replay_random_same_count | 5 | 0.7212 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_replay_random_same_count | 6 | 0.4683 | 0.0000 | 0.1053 | 4.0000 | 2.0000 | 2.0000 | 1.0000 | NA | 0.6579 |

## Largest Cora 0.4 Class-level Accuracy Drops

- `gagb_qb_conflict_gate_alpha07` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0103`, NCR `0.3529`, CDR `0.0000`, gated noisy `13.0000`, gated clean `0.0000`.
- `gagb_class_capped_alpha09` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0240`, NCR `0.1176`, CDR `0.0000`, gated noisy `13.0000`, gated clean `0.0000`.
- `gagb_boundary_safe_alpha09` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0171`, NCR `0.1765`, CDR `0.0000`, gated noisy `13.0000`, gated clean `0.0000`.
- `gagb_replay_random_same_count` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0377`, NCR `0.0000`, CDR `0.0600`, gated noisy `1.0000`, gated clean `3.0000`.
- `gagb_qb_conflict_gate_alpha09` vs `gb` worst class `6`: delta `-0.0556`, NCR `0.0000`, CDR `0.0000`, gated noisy `0.0000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha09` vs `gagb_qb_conflict_gate_alpha03` worst class `0`: delta `-0.0407`, NCR `0.0000`, CDR `0.0278`, gated noisy `0.0000`, gated clean `1.0000`.

## Class 1 / Class 6 Delta Focus

- `gagb_qb_conflict_gate_alpha07` class `1`: acc `0.8882`, delta vs alpha09 `0.0000`, delta vs GB `-0.0066`, NCR `0.2000`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha07` class `6`: acc `0.5079`, delta vs alpha09 `0.0079`, delta vs GB `-0.0476`, NCR `0.0000`, CDR `0.0000`.
- `gagb_class_capped_alpha09` class `1`: acc `0.8947`, delta vs alpha09 `0.0066`, delta vs GB `0.0000`, NCR `0.2000`, CDR `0.0000`.
- `gagb_class_capped_alpha09` class `6`: acc `0.5000`, delta vs alpha09 `0.0000`, delta vs GB `-0.0556`, NCR `0.0000`, CDR `0.0000`.
- `gagb_boundary_safe_alpha09` class `1`: acc `0.8947`, delta vs alpha09 `0.0066`, delta vs GB `0.0000`, NCR `0.1000`, CDR `0.0000`.
- `gagb_boundary_safe_alpha09` class `6`: acc `0.5476`, delta vs alpha09 `0.0476`, delta vs GB `-0.0079`, NCR `0.0000`, CDR `0.0000`.
- `gagb_replay_random_same_count` class `1`: acc `0.8882`, delta vs alpha09 `0.0000`, delta vs GB `-0.0066`, NCR `0.1000`, CDR `0.0909`.
- `gagb_replay_random_same_count` class `6`: acc `0.4683`, delta vs alpha09 `-0.0317`, delta vs GB `-0.0873`, NCR `0.0000`, CDR `0.1053`.

## Confusion Matrix Pointers

- `gagb_boundary_safe_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_class_capped_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha03` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha07` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_replay_random_same_count` aggregate matrices are in `confusion_matrices.json`.
- `gb` aggregate matrices are in `confusion_matrices.json`.

## Required Decision Questions

1. Does alpha06/07/08 recover Cora 0.4 accuracy while keeping NCR > 0?
2. Does class_capped_alpha09 reduce class 1 and class 6 drops?
3. Does class_balanced_alpha09 reduce correction concentration?
4. Does boundary_safe_alpha09 reduce CDR and indirect boundary damage?
5. Is GAGB still better than khop_only repair after calibration?
6. Does replay control suggest that failure is due to intervention strength rather than GAGB target content?
7. Which variant should be kept for the next fair-control run?