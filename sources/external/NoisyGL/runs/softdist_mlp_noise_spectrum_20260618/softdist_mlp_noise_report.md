# SoftDist-MLP Noise Spectrum Report

## Material Passport

- Artifact type: experiment result
- Status: COMPLETED
- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, ROCm `cuda:0`
- Verification status: ANALYZED

## Completeness

- Expected new rows: 180
- Missing new rows: 0
- Existing uniform 0.4 rows are reused from the preregistered MLP diagnostic.

## Results

| Dataset | Noise | Rate | MLP+CE | SoftDist-MLP | Shuffled | vs CE | vs shuffled | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| citeseer | clean | 0.0 | 64.55+/-0.75 | 66.68+/-0.33 | 64.59+/-0.79 | +2.12pp (5/5) | +2.09pp (5/5) | stable increment; clean safe |
| citeseer | instance | 0.4 | 43.75+/-2.07 | 51.51+/-1.32 | 44.98+/-1.95 | +7.77pp (5/5) | +6.53pp (5/5) | stable increment |
| citeseer | pair | 0.4 | 43.17+/-1.14 | 45.65+/-3.04 | 42.93+/-2.42 | +2.48pp (4/5) | +2.72pp (4/5) | stable increment |
| citeseer | uniform | 0.2 | 56.67+/-2.11 | 63.35+/-1.23 | 57.47+/-1.46 | +6.68pp (5/5) | +5.88pp (5/5) | stable increment |
| citeseer | uniform | 0.4 | 44.53+/-4.11 | 52.30+/-2.84 | 44.93+/-3.46 | +7.77pp (5/5) | +7.37pp (5/5) | stable increment |
| citeseer | uniform | 0.5 | 37.93+/-2.26 | 45.67+/-4.60 | 39.34+/-1.71 | +7.74pp (5/5) | +6.33pp (5/5) | stable increment |
| citeseer | uniform | 0.6 | 32.66+/-4.66 | 36.14+/-2.92 | 32.37+/-4.71 | +3.48pp (4/5) | +3.77pp (4/5) | stable increment |
| cora | clean | 0.0 | 66.49+/-1.19 | 66.88+/-1.17 | 64.12+/-1.38 | +0.39pp (3/5) | +2.77pp (5/5) | clean safe |
| cora | instance | 0.4 | 44.39+/-2.95 | 53.53+/-2.43 | 45.14+/-1.81 | +9.14pp (5/5) | +8.39pp (5/5) | stable increment |
| cora | pair | 0.4 | 40.41+/-2.30 | 46.97+/-1.72 | 40.39+/-1.30 | +6.55pp (5/5) | +6.58pp (5/5) | stable increment |
| cora | uniform | 0.2 | 55.08+/-2.60 | 61.42+/-2.24 | 55.48+/-2.31 | +6.34pp (5/5) | +5.94pp (5/5) | stable increment |
| cora | uniform | 0.4 | 45.88+/-3.44 | 54.63+/-1.46 | 46.51+/-1.92 | +8.75pp (5/5) | +8.12pp (5/5) | stable increment |
| cora | uniform | 0.5 | 40.02+/-2.37 | 45.42+/-2.30 | 39.50+/-2.36 | +5.40pp (5/5) | +5.92pp (5/5) | stable increment |
| cora | uniform | 0.6 | 31.91+/-1.65 | 38.73+/-3.52 | 31.66+/-3.32 | +6.83pp (5/5) | +7.07pp (5/5) | stable increment |

## Decision

- Cora stable noisy regimes: 6/6.
- CiteSeer stable noisy regimes: 6/6.
- Clean safety: citeseer=pass; cora=pass.
- Claim: cross-noise robustness is supported for transfer to an X-only MLP.
- Boundary: this does not establish classifier-independent or model-independent robustness.
