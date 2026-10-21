# MLP-58 Formal LCSS Target Teacher Report

The GCN teacher consumes X and A. Frozen GRACE H is used only to build granular balls.
Teacher supervision is the formal GB-LCSS target: beta=0.5, leave-one-out voting, and distribution-only tie handling.
The student is the same X-only MLP used by the existing distillation experiments.

## Protocol audit

- New rows: 60/60.
- Paired dataset/rate/seed groups: 30/30; all passed=True.
- Hash-paired control comparisons: 60/60; all passed=True.
- Noise scope: training labels only; validation and test labels remain clean.

## Aggregate accuracy

| noise | dataset | method | mean+/-std (%) | n |
|---|---|---|---:|---:|
| u0.4 | citeseer | mlp_distill_formal_lcss_x | 69.27+/-1.57 | 5 |
| u0.4 | citeseer | mlp_distill_formal_lcss_x_shuffled | 64.75+/-1.64 | 5 |
| u0.4 | cora | mlp_distill_formal_lcss_x | 81.35+/-1.05 | 5 |
| u0.4 | cora | mlp_distill_formal_lcss_x_shuffled | 76.33+/-3.04 | 5 |
| u0.4 | corafull | mlp_distill_formal_lcss_x | 57.29+/-0.24 | 5 |
| u0.4 | corafull | mlp_distill_formal_lcss_x_shuffled | 58.51+/-0.31 | 5 |
| u0.6 | citeseer | mlp_distill_formal_lcss_x | 58.93+/-4.62 | 5 |
| u0.6 | citeseer | mlp_distill_formal_lcss_x_shuffled | 52.53+/-3.14 | 5 |
| u0.6 | cora | mlp_distill_formal_lcss_x | 71.34+/-3.91 | 5 |
| u0.6 | cora | mlp_distill_formal_lcss_x_shuffled | 63.98+/-2.14 | 5 |
| u0.6 | corafull | mlp_distill_formal_lcss_x | 52.32+/-1.26 | 5 |
| u0.6 | corafull | mlp_distill_formal_lcss_x_shuffled | 52.19+/-0.84 | 5 |

## Paired comparisons

| noise | dataset | comparison | mean delta (pp) | positive seeds |
|---|---|---|---:|---:|
| u0.4 | citeseer | vs_beta0 | +1.29 | 4/5 |
| u0.4 | citeseer | vs_gcn | +3.18 | 5/5 |
| u0.4 | citeseer | vs_shuffled | +4.52 | 5/5 |
| u0.4 | cora | vs_beta0 | +1.47 | 4/5 |
| u0.4 | cora | vs_gcn | +3.40 | 5/5 |
| u0.4 | cora | vs_shuffled | +5.02 | 5/5 |
| u0.4 | corafull | vs_beta0 | -0.90 | 1/5 |
| u0.4 | corafull | vs_gcn | -1.46 | 0/5 |
| u0.4 | corafull | vs_shuffled | -1.23 | 0/5 |
| u0.6 | citeseer | vs_beta0 | +2.84 | 5/5 |
| u0.6 | citeseer | vs_gcn | +5.64 | 5/5 |
| u0.6 | citeseer | vs_shuffled | +6.40 | 5/5 |
| u0.6 | cora | vs_beta0 | +2.44 | 5/5 |
| u0.6 | cora | vs_gcn | +5.84 | 5/5 |
| u0.6 | cora | vs_shuffled | +7.36 | 5/5 |
| u0.6 | corafull | vs_beta0 | -0.19 | 1/5 |
| u0.6 | corafull | vs_gcn | +0.69 | 3/5 |
| u0.6 | corafull | vs_shuffled | +0.13 | 3/5 |

## Pre-registered gates

| noise | gate | passing datasets | average delta (pp) | worst delta (pp) | passed |
|---|---|---:|---:|---:|---|
| u0.4 | vs_gcn | 2/3 | +1.71 | -1.46 | True |
| u0.4 | vs_beta0 | 3/3 | +0.62 | -0.90 | True |
| u0.4 | vs_shuffled | 2/3 | +2.77 | -1.23 | True |
| u0.6 | vs_gcn | 2/3 | +4.06 | +0.69 | True |
| u0.6 | vs_beta0 | 3/3 | +1.70 | -0.19 | True |
| u0.6 | vs_shuffled | 2/3 | +4.63 | +0.13 | True |

## Decision

All pre-registered gates passed: True.
If all gates pass, the formal target becomes the preferred granular-ball teacher for the manuscript.
A failed gate would only reject target transfer to the teacher route; it would not invalidate the base GB-LCSS method.
