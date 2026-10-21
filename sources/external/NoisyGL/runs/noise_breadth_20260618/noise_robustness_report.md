# Noise Robustness Completion Report (2026-06-18)

## Material Passport

- Artifact type: experiment result
- Status: COMPLETED
- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, ROCm `cuda:0`
- Verification status: ANALYZED

## Requested-run completeness

- Expected new per-seed rows: 150
- Missing new per-seed rows: 0

## Regime decisions

The decision column evaluates the controlled granular-ball contribution only; it does not claim an overall win over NRGNN.

| Dataset | Noise | Rate | SoftDist-GCN[XH] | SoftDist-shuffled | SoftDist-NRGNN | Ball contribution |
|---|---:|---:|---:|---:|---:|---|
| citeseer | instance | 0.4 | +1.53pp (4/5) | +2.28pp (5/5) | -3.20pp (0/5) | retain increment |
| citeseer | pair | 0.2 | -1.81pp (2/5) | -1.41pp (3/5) | -3.56pp (0/5) | no stable increment |
| citeseer | pair | 0.4 | +1.32pp (3/5) | +1.96pp (5/5) | -4.23pp (0/5) | no stable increment |
| citeseer | uniform | 0.2 | +0.54pp (3/5) | -0.13pp (3/5) | -0.81pp (0/5) | no stable increment |
| citeseer | uniform | 0.4 | +1.30pp (5/5) | +1.95pp (5/5) | -1.36pp (1/5) | retain increment |
| citeseer | uniform | 0.5 | +0.83pp (4/5) | +1.74pp (5/5) | -3.80pp (0/5) | retain increment |
| citeseer | uniform | 0.6 | +2.22pp (4/5) | +2.29pp (4/5) | -7.12pp (0/5) | retain increment |
| coauthorcs | instance | 0.4 | +0.65pp (5/5) | +0.58pp (5/5) | -1.29pp (0/5) | retain increment |
| coauthorcs | pair | 0.4 | +1.66pp (3/5) | -0.59pp (3/5) | -10.37pp (0/5) | no stable increment |
| coauthorcs | uniform | 0.2 | +0.26pp (4/5) | +0.29pp (5/5) | -1.12pp (0/5) | retain increment |
| coauthorcs | uniform | 0.4 | +0.53pp (5/5) | +0.55pp (5/5) | -1.13pp (0/5) | retain increment |
| coauthorcs | uniform | 0.6 | +0.08pp (3/5) | +0.21pp (4/5) | -1.96pp (0/5) | no stable increment |
| cora | instance | 0.4 | +1.71pp (5/5) | +1.94pp (5/5) | +0.24pp (2/5) | retain increment |
| cora | pair | 0.2 | +1.91pp (3/5) | +1.73pp (4/5) | -0.32pp (2/5) | no stable increment |
| cora | pair | 0.4 | +1.84pp (4/5) | +3.57pp (5/5) | +0.52pp (3/5) | retain increment |
| cora | uniform | 0.2 | -0.37pp (2/5) | -0.25pp (3/5) | +1.05pp (3/5) | no stable increment |
| cora | uniform | 0.4 | +1.10pp (4/5) | +2.16pp (5/5) | +0.31pp (3/5) | retain increment |
| cora | uniform | 0.5 | +1.52pp (5/5) | +3.34pp (5/5) | -1.72pp (1/5) | retain increment |
| cora | uniform | 0.6 | +0.66pp (4/5) | -3.59pp (2/5) | -11.20pp (0/5) | no stable increment |

## Interpretation boundary

Uniform 0.5 is a predeclared transition diagnostic, not a new tuning point. Pair 0.6 is not treated as a generic high-noise setting because the designated wrong class becomes the population majority above rate 0.5. High-noise failures are retained as applicability evidence.
