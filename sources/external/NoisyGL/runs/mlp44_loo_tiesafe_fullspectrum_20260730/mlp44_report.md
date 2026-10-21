# MLP-44 Tie-Safe LOO-GB-LCSS Full-Spectrum Report

Formal lineage: `tie_mode=distribution_only`, `leave_one_out=1` for both real and shuffled LOO targets. Uniform 0.4 is reused only from the audited v2 artifact.

| Dataset | Noise | Base GB | LOO-GB | LOO-base | LOO shuffled | Real-shuffled | Seeds vs base | Seeds vs shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cora | uniform0.2 | 63.24+/-1.76 | 63.25+/-2.46 | +0.01 | 53.57+/-1.99 | +9.68 | 2/5 | 5/5 |
| cora | uniform0.4 | 56.28+/-1.54 | 57.49+/-0.76 | +1.20 | 43.31+/-1.40 | +14.17 | 4/5 | 5/5 |
| cora | uniform0.6 | 40.21+/-2.94 | 41.86+/-2.30 | +1.65 | 32.18+/-2.54 | +9.68 | 5/5 | 5/5 |
| cora | pair0.4 | 50.62+/-2.46 | 51.45+/-2.80 | +0.83 | 39.28+/-1.34 | +12.17 | 4/5 | 5/5 |
| cora | instance0.4 | 55.74+/-1.48 | 56.84+/-1.67 | +1.11 | 43.34+/-1.70 | +13.51 | 5/5 | 5/5 |
| citeseer | uniform0.2 | 63.57+/-0.88 | 63.92+/-1.12 | +0.35 | 55.13+/-1.85 | +8.79 | 3/5 | 5/5 |
| citeseer | uniform0.4 | 53.52+/-2.71 | 55.24+/-2.04 | +1.72 | 42.08+/-2.88 | +13.16 | 5/5 | 5/5 |
| citeseer | uniform0.6 | 40.40+/-2.68 | 42.02+/-3.14 | +1.62 | 30.36+/-1.88 | +11.66 | 5/5 | 5/5 |
| citeseer | pair0.4 | 51.01+/-2.57 | 51.70+/-2.14 | +0.70 | 43.92+/-1.67 | +7.78 | 4/5 | 5/5 |
| citeseer | instance0.4 | 54.79+/-2.23 | 55.58+/-2.54 | +0.78 | 43.07+/-2.00 | +12.51 | 5/5 | 5/5 |
| corafull | uniform0.2 | 43.21+/-0.42 | 43.46+/-0.57 | +0.25 | 41.02+/-0.92 | +2.44 | 4/5 | 5/5 |
| corafull | uniform0.4 | 37.30+/-0.57 | 37.39+/-0.51 | +0.09 | 32.30+/-0.85 | +5.09 | 3/5 | 5/5 |
| corafull | uniform0.6 | 27.35+/-0.85 | 27.43+/-0.89 | +0.08 | 21.41+/-0.84 | +6.02 | 2/5 | 5/5 |
| corafull | pair0.4 | 34.48+/-1.20 | 34.46+/-1.23 | -0.02 | 28.49+/-0.78 | +5.98 | 3/5 | 5/5 |
| corafull | instance0.4 | 36.89+/-0.80 | 36.95+/-0.53 | +0.06 | 31.43+/-0.79 | +5.52 | 3/5 | 5/5 |
| coauthorcs | uniform0.2 | 91.32+/-0.47 | 91.54+/-0.26 | +0.22 | 90.26+/-0.70 | +1.28 | 3/5 | 5/5 |
| coauthorcs | uniform0.4 | 90.62+/-0.47 | 90.71+/-0.59 | +0.09 | 87.46+/-1.07 | +3.25 | 3/5 | 5/5 |
| coauthorcs | uniform0.6 | 87.39+/-0.35 | 87.19+/-0.69 | -0.20 | 78.78+/-1.82 | +8.41 | 2/5 | 5/5 |
| coauthorcs | pair0.4 | 86.02+/-1.15 | 86.24+/-1.18 | +0.21 | 75.99+/-0.60 | +10.25 | 4/5 | 5/5 |
| coauthorcs | instance0.4 | 90.45+/-0.55 | 90.51+/-0.43 | +0.06 | 86.87+/-1.56 | +3.65 | 3/5 | 5/5 |
| amazonpho | uniform0.2 | 86.55+/-0.96 | 86.07+/-1.05 | -0.49 | 81.03+/-1.23 | +5.03 | 2/5 | 5/5 |
| amazonpho | uniform0.4 | 81.33+/-0.70 | 82.12+/-1.08 | +0.79 | 71.45+/-3.29 | +10.66 | 4/5 | 5/5 |
| amazonpho | uniform0.6 | 70.97+/-2.95 | 71.45+/-3.75 | +0.47 | 58.30+/-4.17 | +13.15 | 4/5 | 5/5 |
| amazonpho | pair0.4 | 73.04+/-3.74 | 73.89+/-3.73 | +0.84 | 62.56+/-4.24 | +11.32 | 4/5 | 5/5 |
| amazonpho | instance0.4 | 79.75+/-0.95 | 79.59+/-0.88 | -0.15 | 67.56+/-2.89 | +12.03 | 1/5 | 5/5 |

## Dataset-level LOO minus base

| Dataset | Mean across five settings | Minimum setting | Non-negative settings |
|---|---:|---:|---:|
| cora | +0.96pp | +0.01pp | 5/5 |
| citeseer | +1.03pp | +0.35pp | 5/5 |
| corafull | +0.09pp | -0.02pp | 4/5 |
| coauthorcs | +0.08pp | -0.20pp | 4/5 |
| amazonpho | +0.29pp | -0.49pp | 3/5 |

## Preregistered gate

- PASS: `1`.
- Global LOO-base: `+0.49pp`.
- Minimum setting LOO-base: `-0.49pp`.
- Datasets with non-negative cross-setting LOO-base: `5/5`.
- Positive real-shuffled setting means: `25/25`.
- Real-shuffled settings with at least 4/5 positive seeds: `25/25`.
- Reasons: `pass`.

## Decision

Tie-safe LOO-GB-LCSS satisfies all preregistered gates and may replace the self-inclusive target.
