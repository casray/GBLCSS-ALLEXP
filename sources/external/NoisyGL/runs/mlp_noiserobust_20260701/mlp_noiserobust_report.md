# MLP Noise-Robust Baseline Report

## Material Passport

- Artifact type: experiment result
- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, train-only label noise, ROCm `cuda:0`
- Backbone: X-only MLP for every compared noise-robust strategy
- Verification status: ANALYZED

## Completeness

- Expected rows: 320
- Missing rows: 0

## Main Table

| Dataset | Noise | Rate | CE | Small-loss | Co-teaching | JoCoR | ELR | SCE | SoftDist | Shuffled | G1 | Strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| citeseer | pair | 0.4 | 45.15+/-1.67 | 48.16+/-2.56 | 46.25+/-2.29 | 48.63+/-2.35 | 42.69+/-2.43 | 42.22+/-1.56 | 49.96+/-2.17 | 45.15+/-2.20 | pass | pass |
| citeseer | uniform | 0.2 | 55.10+/-1.17 | 59.26+/-1.94 | 57.05+/-0.82 | 59.36+/-1.43 | 53.62+/-1.48 | 52.12+/-3.28 | 62.29+/-0.83 | 56.63+/-1.06 | pass | pass |
| citeseer | uniform | 0.4 | 43.50+/-3.02 | 49.70+/-3.07 | 46.24+/-2.89 | 49.91+/-3.37 | 43.17+/-3.95 | 42.09+/-3.47 | 51.62+/-2.70 | 43.48+/-2.35 | pass | pass |
| citeseer | uniform | 0.6 | 33.35+/-1.36 | 37.73+/-2.86 | 34.19+/-2.92 | 39.07+/-2.47 | 30.80+/-2.70 | 30.59+/-2.83 | 37.35+/-2.42 | 31.63+/-1.82 | fail | fail |
| cora | pair | 0.4 | 42.83+/-2.19 | 46.99+/-2.77 | 44.26+/-2.30 | 45.91+/-2.38 | 41.83+/-1.75 | 41.40+/-2.04 | 47.89+/-2.12 | 41.11+/-1.55 | pass | pass |
| cora | uniform | 0.2 | 54.96+/-2.22 | 59.84+/-1.29 | 58.62+/-1.97 | 60.12+/-2.25 | 52.95+/-2.38 | 53.57+/-2.17 | 61.41+/-1.24 | 55.05+/-1.85 | pass | pass |
| cora | uniform | 0.4 | 44.24+/-2.62 | 51.95+/-1.97 | 46.83+/-1.78 | 51.72+/-1.30 | 44.28+/-1.38 | 43.60+/-1.86 | 51.84+/-1.37 | 44.11+/-1.87 | fail | fail |
| cora | uniform | 0.6 | 32.51+/-3.38 | 36.72+/-1.09 | 33.29+/-3.48 | 36.92+/-2.14 | 32.82+/-3.51 | 32.66+/-3.78 | 37.12+/-2.88 | 33.01+/-2.69 | pass | fail |

## Paired Deltas Against SoftDist

| Dataset | Noise | Rate | Baseline | Delta | Wins |
|---|---|---:|---|---:|---:|
| citeseer | pair | 0.4 | MLP+CE | +4.81pp | 5/5 |
| citeseer | pair | 0.4 | MLP+small-loss | +1.80pp | 5/5 |
| citeseer | pair | 0.4 | MLP+co-teaching | +3.70pp | 5/5 |
| citeseer | pair | 0.4 | MLP+JoCoR | +1.32pp | 5/5 |
| citeseer | pair | 0.4 | MLP+ELR | +7.27pp | 5/5 |
| citeseer | pair | 0.4 | MLP+SCE | +7.74pp | 5/5 |
| citeseer | pair | 0.4 | SoftDist-MLP shuffled | +4.81pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+CE | +7.18pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+small-loss | +3.02pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+co-teaching | +5.23pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+JoCoR | +2.93pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+ELR | +8.67pp | 5/5 |
| citeseer | uniform | 0.2 | MLP+SCE | +10.16pp | 5/5 |
| citeseer | uniform | 0.2 | SoftDist-MLP shuffled | +5.65pp | 5/5 |
| citeseer | uniform | 0.4 | MLP+CE | +8.12pp | 5/5 |
| citeseer | uniform | 0.4 | MLP+small-loss | +1.92pp | 5/5 |
| citeseer | uniform | 0.4 | MLP+co-teaching | +5.38pp | 5/5 |
| citeseer | uniform | 0.4 | MLP+JoCoR | +1.71pp | 4/5 |
| citeseer | uniform | 0.4 | MLP+ELR | +8.45pp | 5/5 |
| citeseer | uniform | 0.4 | MLP+SCE | +9.53pp | 5/5 |
| citeseer | uniform | 0.4 | SoftDist-MLP shuffled | +8.14pp | 5/5 |
| citeseer | uniform | 0.6 | MLP+CE | +3.99pp | 5/5 |
| citeseer | uniform | 0.6 | MLP+small-loss | -0.39pp | 2/5 |
| citeseer | uniform | 0.6 | MLP+co-teaching | +3.15pp | 5/5 |
| citeseer | uniform | 0.6 | MLP+JoCoR | -1.73pp | 1/5 |
| citeseer | uniform | 0.6 | MLP+ELR | +6.55pp | 5/5 |
| citeseer | uniform | 0.6 | MLP+SCE | +6.75pp | 5/5 |
| citeseer | uniform | 0.6 | SoftDist-MLP shuffled | +5.71pp | 5/5 |
| cora | pair | 0.4 | MLP+CE | +5.07pp | 5/5 |
| cora | pair | 0.4 | MLP+small-loss | +0.91pp | 3/5 |
| cora | pair | 0.4 | MLP+co-teaching | +3.63pp | 5/5 |
| cora | pair | 0.4 | MLP+JoCoR | +1.98pp | 5/5 |
| cora | pair | 0.4 | MLP+ELR | +6.07pp | 5/5 |
| cora | pair | 0.4 | MLP+SCE | +6.49pp | 5/5 |
| cora | pair | 0.4 | SoftDist-MLP shuffled | +6.79pp | 5/5 |
| cora | uniform | 0.2 | MLP+CE | +6.46pp | 5/5 |
| cora | uniform | 0.2 | MLP+small-loss | +1.57pp | 4/5 |
| cora | uniform | 0.2 | MLP+co-teaching | +2.80pp | 5/5 |
| cora | uniform | 0.2 | MLP+JoCoR | +1.30pp | 3/5 |
| cora | uniform | 0.2 | MLP+ELR | +8.46pp | 5/5 |
| cora | uniform | 0.2 | MLP+SCE | +7.84pp | 5/5 |
| cora | uniform | 0.2 | SoftDist-MLP shuffled | +6.36pp | 5/5 |
| cora | uniform | 0.4 | MLP+CE | +7.60pp | 5/5 |
| cora | uniform | 0.4 | MLP+small-loss | -0.11pp | 1/5 |
| cora | uniform | 0.4 | MLP+co-teaching | +5.01pp | 5/5 |
| cora | uniform | 0.4 | MLP+JoCoR | +0.13pp | 2/5 |
| cora | uniform | 0.4 | MLP+ELR | +7.56pp | 5/5 |
| cora | uniform | 0.4 | MLP+SCE | +8.24pp | 5/5 |
| cora | uniform | 0.4 | SoftDist-MLP shuffled | +7.74pp | 5/5 |
| cora | uniform | 0.6 | MLP+CE | +4.61pp | 4/5 |
| cora | uniform | 0.6 | MLP+small-loss | +0.40pp | 3/5 |
| cora | uniform | 0.6 | MLP+co-teaching | +3.83pp | 4/5 |
| cora | uniform | 0.6 | MLP+JoCoR | +0.20pp | 2/5 |
| cora | uniform | 0.6 | MLP+ELR | +4.30pp | 5/5 |
| cora | uniform | 0.6 | MLP+SCE | +4.45pp | 5/5 |
| cora | uniform | 0.6 | SoftDist-MLP shuffled | +4.11pp | 4/5 |

## Gate Decision

- Uniform-rate G1 counts: cora=2/3, citeseer=2/3.
- Pair-0.4 G1: cora=1, citeseer=1.
- G1 overall under preregistered weak-MLP criterion: GO
- G1 counts CE, small-loss, co-teaching, ELR, and shuffled. Strict additionally counts JoCoR and SCE.
