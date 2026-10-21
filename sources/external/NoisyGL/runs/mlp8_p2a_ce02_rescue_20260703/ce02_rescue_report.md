# MLP-8 P2a CE-0.2 Rescue Report

## Protocol

- Dataset: Cora
- Seed: 0
- Noise: train-only uniform 0.4
- Split: percent 0.2/0.1/0.7
- Pretrain candidate: GRACE + `0.2 * noisy CE`
- Student: X-only MLP with `frozengrace_gb_mlp_softdist_beta05`
- Claim limit: seed0 diagnostic only

## Results

| Arm | Representation | Ball params | Real/Shuffled | Balls | True train purity | Valid | Test |
|---|---|---|---|---:|---:|---:|---:|
| anchor default | frozen GRACE | 64/32/0.05 | real | 52 | 0.7893 | 57.04 | 55.46 |
| anchor loose | frozen GRACE | 64/16/0.0 | real | 64 | 0.8022 | 57.41 | 53.61 |
| CE-0.2 default | GRACE+0.2 noisy CE | 64/32/0.05 | real | 39 | 0.7967 | 57.78 | 55.46 |
| CE-0.2 default shuffled | GRACE+0.2 noisy CE | 64/32/0.05 | shuffled | 39 | 0.3530 | 47.04 | 42.64 |
| CE-0.2 loose | GRACE+0.2 noisy CE | 64/16/0.0 | real | 64 | 0.8219 | 57.78 | 53.67 |
| CE-0.2 loose shuffled | GRACE+0.2 noisy CE | 64/16/0.0 | shuffled | 64 | 0.3833 | 47.41 | 42.16 |

The original generation script reports CE-0.2 default at 55.67, while the independent downstream reload of the saved H gives 55.46. The discrepancy is +0.21pp and is treated as comparable, not as a successful improvement.

## Gate Check

| Gate | Result |
|---|---|
| CE-0.2 default > anchor default by >=0.5pp | Fail: 0.00pp in independent rerun, +0.21pp in script-internal run |
| CE-0.2 default > shuffled | Pass: +12.82pp |
| CE-0.2 loose >= anchor loose | Comparable: +0.05pp |
| CE-0.2 loose > anchor default | Fail: -1.79pp |

## Decision

Do not expand P2a CE-0.2 to 5 seeds.

CE-0.2 fixes the severe collapse observed in CE-0.1 default construction. It yields 39 balls under the default threshold and 64 balls under loose construction, and its real grouping strongly beats shuffled. However, the downstream accuracy only matches the frozen-GRACE anchor under the default construction and drops under loose construction. Therefore supervised noisy-CE-assisted GRACE does not provide a new downstream advantage over label-free frozen GRACE.
