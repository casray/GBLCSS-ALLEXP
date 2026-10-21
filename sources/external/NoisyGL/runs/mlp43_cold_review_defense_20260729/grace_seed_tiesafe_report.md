# MLP-43 Tie-Safe GRACE-Seed Stability

Protocol: three archived GRACE seeds x three downstream seeds on Cora, CiteSeer, and CoraFull. The downstream target rule is `tie_mode=distribution_only`.

| Dataset | H0 | H1 | H2 | H-seed SD | downstream RMS SD | ratio |
|---|---:|---:|---:|---:|---:|---:|
| cora | 55.69 +/- 0.32 | 54.74 +/- 0.70 | 55.29 +/- 0.82 | 0.48pp | 0.65pp | 0.73 |
| citeseer | 55.14 +/- 1.06 | 55.41 +/- 0.56 | 55.10 +/- 0.93 | 0.17pp | 0.88pp | 0.19 |
| corafull | 37.28 +/- 0.13 | 38.19 +/- 0.08 | 37.85 +/- 0.54 | 0.46pp | 0.32pp | 1.42 |

## Pre-registered interpretation

- cora: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.
- citeseer: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.
- corafull: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.

Cross-H assignment ARI mean=0.305, range=0.247-0.407.

Passing supports only stability across the three archived pretraining seeds. It does not remove the transductive fixed-graph scope.
