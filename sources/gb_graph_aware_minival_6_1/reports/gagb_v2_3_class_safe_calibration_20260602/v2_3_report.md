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
| gagb_qb_conflict_gate_alpha06 | 0 | 0.6756 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.8000 | 0.0833 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 1 | 0.7618 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 2 | 0.9370 | 0.0000 | 0.0000 | 0.0000 | 8.6000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 3 | 0.8462 | 0.0000 | 0.0000 | 0.0000 | 3.8000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 4 | 0.7879 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 5 | 0.7740 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 1.0000 | NA | NA |
| gagb_qb_conflict_gate_alpha06 | 6 | 0.5413 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha07 | 0 | 0.6756 | 0.0000 | 0.0056 | 0.2000 | 0.4000 | 0.8000 | 0.0833 | NA | 0.9524 |
| gagb_qb_conflict_gate_alpha07 | 1 | 0.7671 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha07 | 2 | 0.9404 | 0.2692 | 0.0000 | 8.4000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha07 | 3 | 0.8458 | 0.0325 | 0.0000 | 2.2000 | 3.8000 | 0.0000 | 1.0000 | 0.9783 | NA |
| gagb_qb_conflict_gate_alpha07 | 4 | 0.7906 | 0.1037 | 0.0000 | 3.0000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha07 | 5 | 0.7740 | 0.0160 | 0.0000 | 0.4000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha07 | 6 | 0.5397 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_qb_conflict_gate_alpha08 | 0 | 0.6740 | 0.0067 | 0.0202 | 1.2000 | 0.4000 | 0.8000 | 0.0833 | 1.0000 | 0.8440 |
| gagb_qb_conflict_gate_alpha08 | 1 | 0.7553 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_qb_conflict_gate_alpha08 | 2 | 0.9404 | 0.2751 | 0.0000 | 8.6000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha08 | 3 | 0.8476 | 0.0566 | 0.0000 | 3.8000 | 3.8000 | 0.0000 | 1.0000 | 0.9826 | NA |
| gagb_qb_conflict_gate_alpha08 | 4 | 0.7886 | 0.1410 | 0.0000 | 4.0000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha08 | 5 | 0.7760 | 0.0251 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_qb_conflict_gate_alpha08 | 6 | 0.5381 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
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
| gagb_class_capped_alpha09 | 0 | 0.6797 | 0.0067 | 0.0056 | 0.6000 | 0.4000 | 0.8000 | 0.0833 | 1.0000 | 0.9524 |
| gagb_class_capped_alpha09 | 1 | 0.7776 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_class_capped_alpha09 | 2 | 0.9370 | 0.0956 | 0.0000 | 3.0000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 3 | 0.8465 | 0.0412 | 0.0000 | 2.8000 | 3.8000 | 0.0000 | 1.0000 | 0.9826 | NA |
| gagb_class_capped_alpha09 | 4 | 0.7926 | 0.0757 | 0.0000 | 2.2000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 5 | 0.7673 | 0.0251 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_capped_alpha09 | 6 | 0.5413 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_class_balanced_alpha09 | 0 | 0.6797 | 0.0067 | 0.0056 | 0.6000 | 0.4000 | 0.8000 | 0.0833 | 1.0000 | 0.9524 |
| gagb_class_balanced_alpha09 | 1 | 0.7724 | 0.0400 | 0.0000 | 0.4000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_class_balanced_alpha09 | 2 | 0.9370 | 0.1028 | 0.0000 | 3.2000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_balanced_alpha09 | 3 | 0.8476 | 0.0474 | 0.0000 | 3.2000 | 3.8000 | 0.0000 | 1.0000 | 0.9826 | NA |
| gagb_class_balanced_alpha09 | 4 | 0.7933 | 0.0837 | 0.0000 | 2.4000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_balanced_alpha09 | 5 | 0.7702 | 0.0251 | 0.0000 | 0.6000 | 0.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_class_balanced_alpha09 | 6 | 0.5349 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
| gagb_boundary_safe_alpha09 | 0 | 0.6821 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.8000 | 0.0833 | NA | NA |
| gagb_boundary_safe_alpha09 | 1 | 0.7724 | 0.0200 | 0.0000 | 0.2000 | 0.4000 | 0.0000 | 1.0000 | 0.9524 | NA |
| gagb_boundary_safe_alpha09 | 2 | 0.9370 | 0.1115 | 0.0000 | 3.4000 | 8.6000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_boundary_safe_alpha09 | 3 | 0.8444 | 0.0061 | 0.0000 | 0.4000 | 3.8000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_boundary_safe_alpha09 | 4 | 0.7906 | 0.0551 | 0.0000 | 1.6000 | 4.0000 | 0.0000 | 1.0000 | 1.0000 | NA |
| gagb_boundary_safe_alpha09 | 5 | 0.7769 | 0.0000 | 0.0000 | 0.0000 | 0.6000 | 0.0000 | 1.0000 | NA | NA |
| gagb_boundary_safe_alpha09 | 6 | 0.5429 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | NA |
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
| gagb_replay_random_same_count | 0 | 0.6846 | 0.0000 | 0.0500 | 2.2000 | 0.2000 | 2.0000 | 0.8667 | NA | 0.7946 |
| gagb_replay_random_same_count | 1 | 0.7737 | 0.0200 | 0.0333 | 1.8000 | 0.8000 | 1.0000 | 0.7667 | 0.5000 | 0.9822 |
| gagb_replay_random_same_count | 2 | 0.9281 | 0.0130 | 0.0340 | 3.0000 | 1.2000 | 1.8000 | 1.0000 | 1.0000 | 0.9792 |
| gagb_replay_random_same_count | 3 | 0.8325 | 0.0062 | 0.0400 | 5.2000 | 1.4000 | 3.8000 | 0.7833 | 1.0000 | 0.8728 |
| gagb_replay_random_same_count | 4 | 0.7852 | 0.0111 | 0.0561 | 3.6000 | 0.6000 | 3.0000 | 0.8583 | 0.6579 | 0.9583 |
| gagb_replay_random_same_count | 5 | 0.7846 | 0.0000 | 0.0164 | 1.4000 | 0.8000 | 0.6000 | 1.0000 | NA | 0.8000 |
| gagb_replay_random_same_count | 6 | 0.5206 | 0.0000 | 0.0211 | 1.4000 | 1.0000 | 0.4000 | 0.3333 | NA | 0.6579 |

## Largest Cora 0.4 Class-level Accuracy Drops

- `gagb_qb_conflict_gate_alpha06` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0048`, NCR `0.0000`, CDR `0.0000`, gated noisy `8.6000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha07` vs `gagb_qb_conflict_gate_alpha09` worst class `3`: delta `-0.0014`, NCR `0.0325`, CDR `0.0000`, gated noisy `3.8000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha08` vs `gagb_qb_conflict_gate_alpha09` worst class `4`: delta `-0.0020`, NCR `0.1410`, CDR `0.0000`, gated noisy `4.0000`, gated clean `0.0000`.
- `gagb_class_capped_alpha09` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0048`, NCR `0.0956`, CDR `0.0000`, gated noisy `8.6000`, gated clean `0.0000`.
- `gagb_class_balanced_alpha09` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0048`, NCR `0.1028`, CDR `0.0000`, gated noisy `8.6000`, gated clean `0.0000`.
- `gagb_boundary_safe_alpha09` vs `gagb_qb_conflict_gate_alpha09` worst class `2`: delta `-0.0048`, NCR `0.1115`, CDR `0.0000`, gated noisy `8.6000`, gated clean `0.0000`.
- `gagb_replay_random_same_count` vs `gagb_qb_conflict_gate_alpha09` worst class `3`: delta `-0.0147`, NCR `0.0062`, CDR `0.0400`, gated noisy `1.4000`, gated clean `3.8000`.
- `gagb_qb_conflict_gate_alpha09` vs `gb` worst class `1`: delta `-0.0592`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gb` worst class `1`: delta `-0.0868`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_alpha09` vs `gagb_qb_conflict_gate_alpha03` worst class `1`: delta `-0.0474`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` vs `gagb_qb_conflict_gate_alpha03` worst class `1`: delta `-0.0750`, NCR `0.0400`, CDR `0.0000`, gated noisy `0.4000`, gated clean `0.0000`.

## Class 1 / Class 6 Delta Focus

- `gagb_qb_conflict_gate_alpha06` class `1`: acc `0.7618`, delta vs alpha09 `0.0289`, delta vs GB `-0.0303`, NCR `0.0000`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha06` class `6`: acc `0.5413`, delta vs alpha09 `0.0254`, delta vs GB `-0.0079`, NCR `0.0000`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha07` class `1`: acc `0.7671`, delta vs alpha09 `0.0342`, delta vs GB `-0.0250`, NCR `0.0400`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha07` class `6`: acc `0.5397`, delta vs alpha09 `0.0238`, delta vs GB `-0.0095`, NCR `0.0000`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha08` class `1`: acc `0.7553`, delta vs alpha09 `0.0224`, delta vs GB `-0.0368`, NCR `0.0400`, CDR `0.0000`.
- `gagb_qb_conflict_gate_alpha08` class `6`: acc `0.5381`, delta vs alpha09 `0.0222`, delta vs GB `-0.0111`, NCR `0.0000`, CDR `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` class `1`: acc `0.7053`, delta vs alpha09 `-0.0276`, delta vs GB `-0.0868`, NCR `0.0400`, CDR `0.0000`.
- `gagb_qb_conflict_gate_replace_qmix` class `6`: acc `0.5000`, delta vs alpha09 `-0.0159`, delta vs GB `-0.0492`, NCR `0.0000`, CDR `0.0000`.
- `gagb_class_capped_alpha09` class `1`: acc `0.7776`, delta vs alpha09 `0.0447`, delta vs GB `-0.0145`, NCR `0.0400`, CDR `0.0000`.
- `gagb_class_capped_alpha09` class `6`: acc `0.5413`, delta vs alpha09 `0.0254`, delta vs GB `-0.0079`, NCR `0.0000`, CDR `0.0000`.
- `gagb_class_balanced_alpha09` class `1`: acc `0.7724`, delta vs alpha09 `0.0395`, delta vs GB `-0.0197`, NCR `0.0400`, CDR `0.0000`.
- `gagb_class_balanced_alpha09` class `6`: acc `0.5349`, delta vs alpha09 `0.0190`, delta vs GB `-0.0143`, NCR `0.0000`, CDR `0.0000`.
- `gagb_boundary_safe_alpha09` class `1`: acc `0.7724`, delta vs alpha09 `0.0395`, delta vs GB `-0.0197`, NCR `0.0200`, CDR `0.0000`.
- `gagb_boundary_safe_alpha09` class `6`: acc `0.5429`, delta vs alpha09 `0.0270`, delta vs GB `-0.0063`, NCR `0.0000`, CDR `0.0000`.
- `khop_only_alpha09` class `1`: acc `0.7592`, delta vs alpha09 `0.0263`, delta vs GB `-0.0329`, NCR `0.0505`, CDR `0.0135`.
- `khop_only_alpha09` class `6`: acc `0.4984`, delta vs alpha09 `-0.0175`, delta vs GB `-0.0508`, NCR `0.0000`, CDR `0.0000`.
- `khop_only_replace` class `1`: acc `0.7618`, delta vs alpha09 `0.0289`, delta vs GB `-0.0303`, NCR `0.0505`, CDR `0.0135`.
- `khop_only_replace` class `6`: acc `0.5175`, delta vs alpha09 `0.0016`, delta vs GB `-0.0317`, NCR `0.0000`, CDR `0.0000`.
- `gagb_replay_random_same_count` class `1`: acc `0.7737`, delta vs alpha09 `0.0408`, delta vs GB `-0.0184`, NCR `0.0200`, CDR `0.0333`.
- `gagb_replay_random_same_count` class `6`: acc `0.5206`, delta vs alpha09 `0.0048`, delta vs GB `-0.0286`, NCR `0.0000`, CDR `0.0211`.

## Confusion Matrix Pointers

- `gagb_boundary_safe_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_class_balanced_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_class_capped_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha03` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha06` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha07` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha08` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `gagb_qb_conflict_gate_replace_qmix` aggregate matrices are in `confusion_matrices.json`.
- `gagb_replay_random_same_count` aggregate matrices are in `confusion_matrices.json`.
- `gb` aggregate matrices are in `confusion_matrices.json`.
- `khop_only_alpha09` aggregate matrices are in `confusion_matrices.json`.
- `khop_only_replace` aggregate matrices are in `confusion_matrices.json`.

## Required Decision Questions

1. Does alpha06/07/08 recover Cora 0.4 accuracy while keeping NCR > 0?
2. Does class_capped_alpha09 reduce class 1 and class 6 drops?
3. Does class_balanced_alpha09 reduce correction concentration?
4. Does boundary_safe_alpha09 reduce CDR and indirect boundary damage?
5. Is GAGB still better than khop_only repair after calibration?
6. Does replay control suggest that failure is due to intervention strength rather than GAGB target content?
7. Which variant should be kept for the next fair-control run?