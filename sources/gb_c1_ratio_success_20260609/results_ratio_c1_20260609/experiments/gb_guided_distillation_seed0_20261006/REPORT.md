# Fixed ERASE teacher: GB-guided distillation screening

Seed 0 only; uniform training-label noise 0.4 and 0.6. No tuning or teacher retraining.

D0: ordinary distillation; D1: noisy-label anchor; D2: constant-weight GB target; D3: adaptive GB target; D4: shuffled size-matched GB target.

| Dataset | Noise | Teacher | D0 | D1 | D2 | D3 | D4 | D3-D0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 0.4 | 81.953 | 83.113 | 83.061 | 83.061 | 82.269 | 83.061 | -0.844 |
| cora | 0.6 | 77.150 | 77.784 | 78.100 | 77.467 | 78.153 | 78.153 | 0.369 |
| citeseer | 0.4 | 68.514 | 68.342 | 68.514 | 68.385 | 68.385 | 68.385 | 0.043 |
| citeseer | 0.6 | 61.770 | 62.887 | 63.015 | 62.887 | 63.187 | 63.144 | 0.301 |
| corafull | 0.4 | 59.747 | 58.549 | 59.062 | 58.946 | 58.564 | 58.990 | 0.014 |
| corafull | 0.6 | 57.510 | 56.131 | 56.247 | 56.052 | 56.139 | 55.908 | 0.007 |
| coauthorcs | 0.4 | 90.961 | 93.376 | 93.423 | 93.369 | 93.353 | 93.423 | -0.023 |
| coauthorcs | 0.6 | 90.961 | 93.665 | 93.673 | 93.657 | 93.657 | 93.657 | -0.008 |
| amazonpho | 0.4 | 92.082 | 92.493 | 92.941 | 93.109 | 92.848 | 92.848 | 0.355 |
| amazonpho | 0.6 | 89.281 | 89.300 | 89.468 | 88.889 | 89.505 | 89.132 | 0.205 |

## Candidate paired comparisons

| Reference | Mean delta (pp) | Wins/ties/losses | Worst delta |
|---|---:|---|---:|
| D0 | 0.0419 | 7/0/3 | -0.8443 |
| D1 | -0.1444 | 3/0/7 | -0.7916 |
| D2 | 0.0238 | 4/2/4 | -0.7916 |
| D4 | -0.0640 | 3/4/3 | -0.7916 |
| teacher | 0.6131 | 7/0/3 | -1.3713 |

## Predefined gates

- extra_gain: False
- local_membership: False
- better_than_labels: False
- better_than_constant: True
- retain_teacher: True

Decision: stop this fixed candidate.

All targets were fixed before the new student fits. Clean training labels enter only the post-freeze diagnostics.
D1 and D4 reuse the real-GB weights; they are mechanism controls, not complete no-ball methods.
D2 preserves total weight mass, not identical per-node probability displacement.
No standard deviations, significance tests, equivalence conclusions or SOTA claims follow from this one-seed screening.

Audit: 50 outputs, 150 recomputed metrics, 40 strict-validation checkpoint selections, all frozen input hashes unchanged. See output_audit.json.

See diagnostics.csv for target coverage, weights, L1 movement, Brier scores and correction/damage counts.
