# MLP-49 UTV-GB-LCSS Phase A report

## Protocol

- Fixed GRACE H and fixed recursive 2-means balls; labels never alter Stage 2.
- Formal tie-safe leave-one-out targets; NRGNN only attenuates alpha from non-training-node evidence.
- Uniform 0.4/0.6, train-only noise, percent 0.2/0.1/0.7, seed 0.
- Grid complete: **True**.

## Target quality

| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Coverage | Mean alpha |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| citeseer | 0.4 | clean_oracle_verifier | 0.5639 | 0.6860 | 0.0136 | 0.0000 | 0.9940 | 0.1712 |
| citeseer | 0.4 | formal_loo_gb | 0.5805 | 0.5733 | 0.0612 | 0.0081 | 0.9940 | 0.3446 |
| citeseer | 0.4 | nrgnn_teacher_target | 0.7068 | 0.6000 | 0.6633 | 0.2588 | 1.0000 | 0.0761 |
| citeseer | 0.4 | teacher_weighted_noisy | 0.5579 | 0.8842 | 0.0000 | 0.0000 | 1.0000 | 0.0761 |
| citeseer | 0.4 | utv_gb | 0.5579 | 0.8688 | 0.0000 | 0.0000 | 0.9940 | 0.0114 |
| citeseer | 0.4 | utv_gb_shuffled | 0.5579 | 0.8831 | 0.0000 | 0.0000 | 0.9955 | 0.0012 |
| citeseer | 0.6 | clean_oracle_verifier | 0.4165 | 1.0239 | 0.0051 | 0.0000 | 0.9940 | 0.1159 |
| citeseer | 0.6 | formal_loo_gb | 0.4045 | 0.8901 | 0.0205 | 0.0509 | 0.9940 | 0.2881 |
| citeseer | 0.6 | nrgnn_teacher_target | 0.6827 | 0.6601 | 0.6282 | 0.2400 | 1.0000 | 0.0468 |
| citeseer | 0.6 | teacher_weighted_noisy | 0.4135 | 1.1729 | 0.0000 | 0.0000 | 1.0000 | 0.0468 |
| citeseer | 0.6 | utv_gb | 0.4135 | 1.1646 | 0.0000 | 0.0000 | 0.9940 | 0.0061 |
| citeseer | 0.6 | utv_gb_shuffled | 0.4135 | 1.1712 | 0.0000 | 0.0000 | 1.0000 | 0.0014 |
| cora | 0.4 | clean_oracle_verifier | 0.5970 | 0.5689 | 0.0609 | 0.0064 | 0.9982 | 0.2643 |
| cora | 0.4 | formal_loo_gb | 0.6396 | 0.4778 | 0.1957 | 0.0322 | 1.0000 | 0.4145 |
| cora | 0.4 | nrgnn_teacher_target | 0.8410 | 0.4750 | 0.8174 | 0.1415 | 1.0000 | 0.1582 |
| cora | 0.4 | teacher_weighted_noisy | 0.5749 | 0.8503 | 0.0000 | 0.0000 | 1.0000 | 0.1582 |
| cora | 0.4 | utv_gb | 0.5749 | 0.8112 | 0.0000 | 0.0000 | 1.0000 | 0.0293 |
| cora | 0.4 | utv_gb_shuffled | 0.5749 | 0.8476 | 0.0000 | 0.0000 | 0.9963 | 0.0030 |
| cora | 0.6 | clean_oracle_verifier | 0.3715 | 1.0343 | 0.0000 | 0.0000 | 0.9852 | 0.1377 |
| cora | 0.6 | formal_loo_gb | 0.3734 | 0.8945 | 0.0118 | 0.0149 | 1.0000 | 0.2822 |
| cora | 0.6 | nrgnn_teacher_target | 0.7301 | 0.6279 | 0.6941 | 0.2090 | 1.0000 | 0.0751 |
| cora | 0.6 | teacher_weighted_noisy | 0.3715 | 1.2569 | 0.0000 | 0.0000 | 1.0000 | 0.0751 |
| cora | 0.6 | utv_gb | 0.3715 | 1.2419 | 0.0000 | 0.0000 | 1.0000 | 0.0088 |
| cora | 0.6 | utv_gb_shuffled | 0.3715 | 1.2549 | 0.0000 | 0.0000 | 0.9963 | 0.0017 |
| corafull | 0.4 | clean_oracle_verifier | 0.5978 | 0.7152 | 0.0000 | 0.0000 | 0.9997 | 0.0985 |
| corafull | 0.4 | formal_loo_gb | 0.5978 | 0.6335 | 0.0000 | 0.0000 | 1.0000 | 0.2254 |
| corafull | 0.4 | nrgnn_teacher_target | 0.6776 | 0.6351 | 0.6074 | 0.2751 | 1.0000 | 0.2614 |
| corafull | 0.4 | teacher_weighted_noisy | 0.5978 | 0.8044 | 0.0000 | 0.0000 | 1.0000 | 0.2614 |
| corafull | 0.4 | utv_gb | 0.5978 | 0.7789 | 0.0000 | 0.0000 | 1.0000 | 0.0255 |
| corafull | 0.4 | utv_gb_shuffled | 0.5978 | 0.8038 | 0.0000 | 0.0000 | 1.0000 | 0.0008 |
| corafull | 0.6 | clean_oracle_verifier | 0.3982 | 1.1156 | 0.0000 | 0.0000 | 0.9997 | 0.0594 |
| corafull | 0.6 | formal_loo_gb | 0.3982 | 1.0048 | 0.0000 | 0.0000 | 1.0000 | 0.1500 |
| corafull | 0.6 | nrgnn_teacher_target | 0.6321 | 0.7595 | 0.5831 | 0.2938 | 1.0000 | 0.1490 |
| corafull | 0.6 | teacher_weighted_noisy | 0.3982 | 1.2036 | 0.0000 | 0.0000 | 1.0000 | 0.1490 |
| corafull | 0.6 | utv_gb | 0.3982 | 1.1900 | 0.0000 | 0.0000 | 1.0000 | 0.0086 |
| corafull | 0.6 | utv_gb_shuffled | 0.3982 | 1.2031 | 0.0000 | 0.0000 | 1.0000 | 0.0004 |

## Preregistered decision

- Pass: **False**
- Reason: `brier_controls_failed`
- Brier better than formal and shuffled: 0/6
- Brier better than shuffled: 6/6
- Clean-oracle verifier headroom: 0/6
- Mean target-accuracy delta vs formal: -1.335pp

Phase B is permitted only when every frozen gate passes. A failure does not authorize a second verifier formula.

## Detailed conclusion

### Hypothesis and granular-ball role

MLP-49 kept the label-free GRACE partition and formal tie-safe LOO local target unchanged. The robust NRGNN teacher could only attenuate each node's ball-correction weight using non-training-node posterior agreement and confidence.

### Direct evidence

- UTV improved Brier over both formal LOO and shuffled UTV in 0/6 cells. Its Brier delta versus formal LOO ranged from +0.1454 to +0.3474; all six were adverse.
- Mean target-accuracy delta versus formal LOO was -1.335pp. Cora u0.4 and CiteSeer u0.4 crossed the preregistered 1pp adverse boundary.
- Real balls beat size-matched shuffled balls in Brier in 6/6 cells. Thus geometry still carried local information, but teacher attenuation did not improve the formal target.
- Clean-oracle verification improved formal LOO Brier in 0/6 cells. This removes the premise that a more accurate verifier alone can rescue the multiplicative rule.
- The direct NRGNN posterior target had 63.21%--84.10% target accuracy, but also 14.15%--29.38% clean-label damage. The teacher contains useful information, but it is not a safe drop-in correction target.
- UTV mean alpha was only 0.0061--0.0293. MC stability was high, while posterior entropy confidence was low; multiplying agreement and entropy confidence suppressed nearly all hard correction changes.

### Supported conclusion

The preregistered Phase A gate failed with reason `brier_controls_failed`; no downstream MLP run is authorized. Robust-teacher verification by multiplicatively shrinking the formal LOO alpha is closed under this frozen formulation.

### Interpretation boundary

This result does not show that robust teacher knowledge is useless: the direct teacher targets are substantially more accurate, and MLP-47 already showed useful NRGNN distillation. It shows that confidence-only attenuation discards beneficial GB corrections faster than it removes harmful ones. Replacing the product by a tuned threshold or a second attenuation formula would be post-result tuning and is not permitted. A future method would need a genuinely different information path, such as teacher residual content or GB regularization of teacher predictions.
