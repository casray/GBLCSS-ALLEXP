# MLP-50 LOO minimum-vote gate report

## Protocol

- Formal `tie_mode=distribution_only`, `leave_one_out=true` targets.
- `s_min` counts other training votes after removing the target node.
- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, seed 0.
- `s_min=5` is the only candidate; `s_min=10` is diagnostic only.

## Target quality

| Dataset | Noise | Method | Acc | Brier | NCR | CDR | Coverage |
|---|---:|---|---:|---:|---:|---:|---:|
| citeseer | 0.4 | real_s10 | 0.5714 | 0.7044 | 0.0340 | 0.0027 | 0.5263 |
| citeseer | 0.4 | real_s2 | 0.5805 | 0.5733 | 0.0612 | 0.0081 | 0.9940 |
| citeseer | 0.4 | real_s5 | 0.5805 | 0.5785 | 0.0578 | 0.0054 | 0.9774 |
| citeseer | 0.4 | shuffled_s10 | 0.5579 | 0.8109 | 0.0000 | 0.0000 | 0.6256 |
| citeseer | 0.4 | shuffled_s2 | 0.5504 | 0.7739 | 0.0000 | 0.0135 | 0.9910 |
| citeseer | 0.4 | shuffled_s5 | 0.5564 | 0.7736 | 0.0000 | 0.0027 | 0.9474 |
| citeseer | 0.6 | real_s10 | 0.4135 | 0.9975 | 0.0026 | 0.0036 | 0.5263 |
| citeseer | 0.6 | real_s2 | 0.4045 | 0.8901 | 0.0205 | 0.0509 | 0.9940 |
| citeseer | 0.6 | real_s5 | 0.4090 | 0.8906 | 0.0205 | 0.0400 | 0.9774 |
| citeseer | 0.6 | shuffled_s10 | 0.4135 | 1.0697 | 0.0000 | 0.0000 | 0.6256 |
| citeseer | 0.6 | shuffled_s2 | 0.4060 | 0.9995 | 0.0000 | 0.0182 | 0.9940 |
| citeseer | 0.6 | shuffled_s5 | 0.4105 | 1.0029 | 0.0000 | 0.0073 | 0.9504 |
| cora | 0.4 | real_s10 | 0.6155 | 0.6394 | 0.1043 | 0.0064 | 0.5619 |
| cora | 0.4 | real_s2 | 0.6396 | 0.4778 | 0.1957 | 0.0322 | 1.0000 |
| cora | 0.4 | real_s5 | 0.6414 | 0.4869 | 0.1826 | 0.0193 | 0.9667 |
| cora | 0.4 | shuffled_s10 | 0.5749 | 0.7716 | 0.0000 | 0.0000 | 0.6377 |
| cora | 0.4 | shuffled_s2 | 0.5730 | 0.7204 | 0.0087 | 0.0096 | 1.0000 |
| cora | 0.4 | shuffled_s5 | 0.5767 | 0.7283 | 0.0043 | 0.0000 | 0.9298 |
| cora | 0.6 | real_s10 | 0.3752 | 1.0662 | 0.0059 | 0.0000 | 0.5619 |
| cora | 0.6 | real_s2 | 0.3734 | 0.8945 | 0.0118 | 0.0149 | 1.0000 |
| cora | 0.6 | real_s5 | 0.3752 | 0.9023 | 0.0118 | 0.0100 | 0.9667 |
| cora | 0.6 | shuffled_s10 | 0.3715 | 1.1355 | 0.0000 | 0.0000 | 0.6377 |
| cora | 0.6 | shuffled_s2 | 0.3641 | 1.0593 | 0.0000 | 0.0199 | 0.9963 |
| cora | 0.6 | shuffled_s5 | 0.3697 | 1.0721 | 0.0000 | 0.0050 | 0.9261 |
| corafull | 0.4 | real_s10 | 0.5978 | 0.6350 | 0.0000 | 0.0000 | 0.9904 |
| corafull | 0.4 | real_s2 | 0.5978 | 0.6335 | 0.0000 | 0.0000 | 1.0000 |
| corafull | 0.4 | real_s5 | 0.5978 | 0.6338 | 0.0000 | 0.0000 | 0.9990 |
| corafull | 0.4 | shuffled_s10 | 0.5978 | 0.7670 | 0.0000 | 0.0000 | 0.9924 |
| corafull | 0.4 | shuffled_s2 | 0.5978 | 0.7666 | 0.0000 | 0.0000 | 1.0000 |
| corafull | 0.4 | shuffled_s5 | 0.5978 | 0.7666 | 0.0000 | 0.0000 | 0.9987 |
| corafull | 0.6 | real_s10 | 0.3982 | 1.0067 | 0.0000 | 0.0000 | 0.9904 |
| corafull | 0.6 | real_s2 | 0.3982 | 1.0048 | 0.0000 | 0.0000 | 1.0000 |
| corafull | 0.6 | real_s5 | 0.3982 | 1.0052 | 0.0000 | 0.0000 | 0.9990 |
| corafull | 0.6 | shuffled_s10 | 0.3982 | 1.1492 | 0.0000 | 0.0000 | 0.9924 |
| corafull | 0.6 | shuffled_s2 | 0.3982 | 1.1481 | 0.0000 | 0.0000 | 1.0000 |
| corafull | 0.6 | shuffled_s5 | 0.3982 | 1.1483 | 0.0000 | 0.0000 | 0.9987 |

## Label-availability diagnostic

| Dataset | Noise | Control | Empty balls | Empty-node fraction | Mean other votes | <5-vote train nodes |
|---|---:|---|---:|---:|---:|---:|
| citeseer | 0.4 | real | 0 | 0.0000 | 11.82 | 15 |
| citeseer | 0.4 | shuffled | 0 | 0.0000 | 12.73 | 33 |
| citeseer | 0.6 | real | 0 | 0.0000 | 11.82 | 15 |
| citeseer | 0.6 | shuffled | 0 | 0.0000 | 12.73 | 33 |
| cora | 0.4 | real | 0 | 0.0000 | 11.05 | 18 |
| cora | 0.4 | shuffled | 0 | 0.0000 | 11.43 | 38 |
| cora | 0.6 | real | 0 | 0.0000 | 11.05 | 18 |
| cora | 0.6 | shuffled | 0 | 0.0000 | 11.43 | 38 |
| corafull | 0.4 | real | 0 | 0.0000 | 151.25 | 4 |
| corafull | 0.4 | shuffled | 0 | 0.0000 | 148.67 | 5 |
| corafull | 0.6 | real | 0 | 0.0000 | 151.25 | 4 |
| corafull | 0.6 | shuffled | 0 | 0.0000 | 148.67 | 5 |

## Preregistered gate

- PASS: `0`.
- Reason: `brier_failed`.
- Brier-improved cells: `0/6`.
- Mean Brier delta, s5-s2: `+0.003886`.
- Mean target-accuracy delta, s5-s2: `+0.137pp`.
- Real-structure Brier wins: `6/6`.

## Decision

Phase A failed. Keep formal s_min=2 and do not run downstream Phase B.
