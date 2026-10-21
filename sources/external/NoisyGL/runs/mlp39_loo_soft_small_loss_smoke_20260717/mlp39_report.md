# MLP-39 LOO-LCSS + Soft-Small-Loss Seed0 Smoke

| Dataset | LOO combo | Default combo | LOO-LCSS | Small-loss | LOO shuffled | LOO-default | LOO structure |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 59.37 | 60.16 | 56.99 | 49.50 | 45.91 | -0.79pp | +13.46pp |
| citeseer | 60.22 | 59.84 | 56.36 | 49.05 | 43.99 | +0.39pp | +16.24pp |
| corafull | 40.45 | 41.27 | 37.64 | 33.05 | 33.97 | -0.82pp | +6.49pp |

## Gate

- Pass: 0
- Positive vs default: 1/3
- Mean LOO conditional increment: -0.41pp
- Minimum dataset increment: -0.82pp
- Structure deltas >=1pp: 3/3
- Reasons: fewer_than_2_of_3_positive_vs_default;mean_increment_below_0p5pp;dataset_regression_below_minus_0p5pp
