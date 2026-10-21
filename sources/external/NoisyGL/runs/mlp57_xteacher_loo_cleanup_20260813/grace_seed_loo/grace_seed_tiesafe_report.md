# MLP-43 Tie-Safe GRACE-Seed Stability

Protocol: three archived GRACE seeds x three downstream seeds on Cora, CiteSeer, and CoraFull. The downstream target rule is `tie_mode=distribution_only` and `leave_one_out=true`.

| Dataset | H0 | H1 | H2 | H-seed SD | downstream RMS SD | ratio |
|---|---:|---:|---:|---:|---:|---:|
| cora | 56.69 +/- 1.08 | 56.45 +/- 0.63 | 56.50 +/- 0.49 | 0.13pp | 0.78pp | 0.17 |
| citeseer | 56.31 +/- 1.28 | 56.03 +/- 0.54 | 56.90 +/- 0.45 | 0.45pp | 0.84pp | 0.53 |
| corafull | 37.39 +/- 0.35 | 37.95 +/- 0.38 | 38.59 +/- 0.59 | 0.60pp | 0.45pp | 1.33 |

## Pre-registered interpretation

- cora: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.
- citeseer: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.
- corafull: LCSS-CE is positive for 3/3 H seeds; cross-H dispersion gate=PASS.

Cross-H assignment ARI mean=0.300, range=0.247-0.365.

Passing supports only stability across the three archived pretraining seeds. It does not remove the transductive fixed-graph scope.
