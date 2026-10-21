# MLP Dual-Robust Report

## Material Passport

- Artifact type: experiment result
- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, train-only label noise, ROCm `cuda:0`
- Label-noise table: X-only MLP backbone for all robust strategies
- Structure-noise table: clean-label training, inference-time graph perturbation only
- Claim discipline: clean labels are evaluation-only; structural noise is not used during training

## Completeness

- Label main table missing rows: 0 / 600
- Ablation missing rows: 0 / 75
- Structure table missing rows: 0 / 315

## Main Table 1: Label Noise Robustness

| Dataset | Noise | Method | n | Test acc mean±std |
|---|---:|---|---:|---:|
| citeseer | instance0.4 | MLP+CE | 5 | 44.33±2.04 |
| citeseer | instance0.4 | MLP+small-loss | 5 | 50.75±3.33 |
| citeseer | instance0.4 | MLP+co-teaching | 5 | 46.81±1.68 |
| citeseer | instance0.4 | MLP+JoCoR | 5 | 50.86±2.45 |
| citeseer | instance0.4 | MLP+ELR | 5 | 41.94±3.27 |
| citeseer | instance0.4 | MLP+SCE | 5 | 40.92±4.23 |
| citeseer | instance0.4 | SoftDist-MLP | 5 | 51.81±1.61 |
| citeseer | instance0.4 | SoftDist shuffled | 5 | 43.82±1.43 |
| citeseer | pair0.4 | MLP+CE | 5 | 45.15±1.67 |
| citeseer | pair0.4 | MLP+small-loss | 5 | 48.16±2.56 |
| citeseer | pair0.4 | MLP+co-teaching | 5 | 46.25±2.29 |
| citeseer | pair0.4 | MLP+JoCoR | 5 | 48.63±2.35 |
| citeseer | pair0.4 | MLP+ELR | 5 | 42.69±2.43 |
| citeseer | pair0.4 | MLP+SCE | 5 | 42.22±1.56 |
| citeseer | pair0.4 | SoftDist-MLP | 5 | 49.96±2.17 |
| citeseer | pair0.4 | SoftDist shuffled | 5 | 45.15±2.20 |
| citeseer | uniform0.2 | MLP+CE | 5 | 55.10±1.17 |
| citeseer | uniform0.2 | MLP+small-loss | 5 | 59.26±1.94 |
| citeseer | uniform0.2 | MLP+co-teaching | 5 | 57.05±0.82 |
| citeseer | uniform0.2 | MLP+JoCoR | 5 | 59.36±1.43 |
| citeseer | uniform0.2 | MLP+ELR | 5 | 53.62±1.48 |
| citeseer | uniform0.2 | MLP+SCE | 5 | 52.12±3.28 |
| citeseer | uniform0.2 | SoftDist-MLP | 5 | 62.29±0.83 |
| citeseer | uniform0.2 | SoftDist shuffled | 5 | 56.63±1.06 |
| citeseer | uniform0.4 | MLP+CE | 5 | 43.50±3.02 |
| citeseer | uniform0.4 | MLP+small-loss | 5 | 49.70±3.07 |
| citeseer | uniform0.4 | MLP+co-teaching | 5 | 46.24±2.89 |
| citeseer | uniform0.4 | MLP+JoCoR | 5 | 49.91±3.37 |
| citeseer | uniform0.4 | MLP+ELR | 5 | 43.17±3.95 |
| citeseer | uniform0.4 | MLP+SCE | 5 | 42.09±3.47 |
| citeseer | uniform0.4 | SoftDist-MLP | 5 | 51.62±2.70 |
| citeseer | uniform0.4 | SoftDist shuffled | 5 | 43.48±2.35 |
| citeseer | uniform0.6 | MLP+CE | 5 | 33.35±1.36 |
| citeseer | uniform0.6 | MLP+small-loss | 5 | 37.73±2.86 |
| citeseer | uniform0.6 | MLP+co-teaching | 5 | 34.19±2.92 |
| citeseer | uniform0.6 | MLP+JoCoR | 5 | 39.07±2.47 |
| citeseer | uniform0.6 | MLP+ELR | 5 | 30.80±2.70 |
| citeseer | uniform0.6 | MLP+SCE | 5 | 30.59±2.83 |
| citeseer | uniform0.6 | SoftDist-MLP | 5 | 37.35±2.42 |
| citeseer | uniform0.6 | SoftDist shuffled | 5 | 31.63±1.82 |
| coauthorcs | instance0.4 | MLP+CE | 5 | 87.44±1.15 |
| coauthorcs | instance0.4 | MLP+small-loss | 5 | 88.97±1.02 |
| coauthorcs | instance0.4 | MLP+co-teaching | 5 | 88.95±0.97 |
| coauthorcs | instance0.4 | MLP+JoCoR | 5 | 89.17±0.85 |
| coauthorcs | instance0.4 | MLP+ELR | 5 | 87.78±0.94 |
| coauthorcs | instance0.4 | MLP+SCE | 5 | 87.99±0.56 |
| coauthorcs | instance0.4 | SoftDist-MLP | 5 | 89.70±0.82 |
| coauthorcs | instance0.4 | SoftDist shuffled | 5 | 87.21±1.13 |
| coauthorcs | pair0.4 | MLP+CE | 5 | 75.91±1.34 |
| coauthorcs | pair0.4 | MLP+small-loss | 5 | 80.13±1.50 |
| coauthorcs | pair0.4 | MLP+co-teaching | 5 | 79.06±1.30 |
| coauthorcs | pair0.4 | MLP+JoCoR | 5 | 81.02±1.62 |
| coauthorcs | pair0.4 | MLP+ELR | 5 | 78.03±1.48 |
| coauthorcs | pair0.4 | MLP+SCE | 5 | 79.77±2.51 |
| coauthorcs | pair0.4 | SoftDist-MLP | 5 | 79.43±1.57 |
| coauthorcs | pair0.4 | SoftDist shuffled | 5 | 75.99±0.75 |
| coauthorcs | uniform0.2 | MLP+CE | 5 | 91.38±0.54 |
| coauthorcs | uniform0.2 | MLP+small-loss | 5 | 91.50±0.49 |
| coauthorcs | uniform0.2 | MLP+co-teaching | 5 | 91.70±0.29 |
| coauthorcs | uniform0.2 | MLP+JoCoR | 5 | 91.81±0.25 |
| coauthorcs | uniform0.2 | MLP+ELR | 5 | 91.21±0.60 |
| coauthorcs | uniform0.2 | MLP+SCE | 5 | 90.63±0.47 |
| coauthorcs | uniform0.2 | SoftDist-MLP | 5 | 91.46±0.49 |
| coauthorcs | uniform0.2 | SoftDist shuffled | 5 | 90.40±0.79 |
| coauthorcs | uniform0.4 | MLP+CE | 5 | 88.42±0.62 |
| coauthorcs | uniform0.4 | MLP+small-loss | 5 | 89.06±0.88 |
| coauthorcs | uniform0.4 | MLP+co-teaching | 5 | 88.80±0.97 |
| coauthorcs | uniform0.4 | MLP+JoCoR | 5 | 89.38±0.61 |
| coauthorcs | uniform0.4 | MLP+ELR | 5 | 88.29±0.73 |
| coauthorcs | uniform0.4 | MLP+SCE | 5 | 88.37±0.53 |
| coauthorcs | uniform0.4 | SoftDist-MLP | 5 | 89.95±0.70 |
| coauthorcs | uniform0.4 | SoftDist shuffled | 5 | 87.72±0.83 |
| coauthorcs | uniform0.6 | MLP+CE | 5 | 80.03±1.91 |
| coauthorcs | uniform0.6 | MLP+small-loss | 5 | 81.65±0.91 |
| coauthorcs | uniform0.6 | MLP+co-teaching | 5 | 81.88±0.72 |
| coauthorcs | uniform0.6 | MLP+JoCoR | 5 | 82.58±0.41 |
| coauthorcs | uniform0.6 | MLP+ELR | 5 | 80.54±1.28 |
| coauthorcs | uniform0.6 | MLP+SCE | 5 | 79.43±1.45 |
| coauthorcs | uniform0.6 | SoftDist-MLP | 5 | 84.89±0.96 |
| coauthorcs | uniform0.6 | SoftDist shuffled | 5 | 79.31±1.72 |
| cora | instance0.4 | MLP+CE | 5 | 45.33±2.22 |
| cora | instance0.4 | MLP+small-loss | 5 | 51.82±0.94 |
| cora | instance0.4 | MLP+co-teaching | 5 | 47.35±2.24 |
| cora | instance0.4 | MLP+JoCoR | 5 | 52.26±1.01 |
| cora | instance0.4 | MLP+ELR | 5 | 45.93±2.26 |
| cora | instance0.4 | MLP+SCE | 5 | 43.29±4.35 |
| cora | instance0.4 | SoftDist-MLP | 5 | 52.96±2.49 |
| cora | instance0.4 | SoftDist shuffled | 5 | 44.63±1.37 |
| cora | pair0.4 | MLP+CE | 5 | 42.83±2.19 |
| cora | pair0.4 | MLP+small-loss | 5 | 46.99±2.77 |
| cora | pair0.4 | MLP+co-teaching | 5 | 44.26±2.30 |
| cora | pair0.4 | MLP+JoCoR | 5 | 45.91±2.38 |
| cora | pair0.4 | MLP+ELR | 5 | 41.83±1.75 |
| cora | pair0.4 | MLP+SCE | 5 | 41.40±2.04 |
| cora | pair0.4 | SoftDist-MLP | 5 | 47.89±2.12 |
| cora | pair0.4 | SoftDist shuffled | 5 | 41.11±1.55 |
| cora | uniform0.2 | MLP+CE | 5 | 54.96±2.22 |
| cora | uniform0.2 | MLP+small-loss | 5 | 59.84±1.29 |
| cora | uniform0.2 | MLP+co-teaching | 5 | 58.62±1.97 |
| cora | uniform0.2 | MLP+JoCoR | 5 | 60.12±2.25 |
| cora | uniform0.2 | MLP+ELR | 5 | 52.95±2.38 |
| cora | uniform0.2 | MLP+SCE | 5 | 53.57±2.17 |
| cora | uniform0.2 | SoftDist-MLP | 5 | 61.41±1.24 |
| cora | uniform0.2 | SoftDist shuffled | 5 | 55.05±1.85 |
| cora | uniform0.4 | MLP+CE | 5 | 44.24±2.62 |
| cora | uniform0.4 | MLP+small-loss | 5 | 51.95±1.97 |
| cora | uniform0.4 | MLP+co-teaching | 5 | 46.83±1.78 |
| cora | uniform0.4 | MLP+JoCoR | 5 | 51.72±1.30 |
| cora | uniform0.4 | MLP+ELR | 5 | 44.28±1.38 |
| cora | uniform0.4 | MLP+SCE | 5 | 43.60±1.86 |
| cora | uniform0.4 | SoftDist-MLP | 5 | 51.84±1.37 |
| cora | uniform0.4 | SoftDist shuffled | 5 | 44.11±1.87 |
| cora | uniform0.6 | MLP+CE | 5 | 32.51±3.38 |
| cora | uniform0.6 | MLP+small-loss | 5 | 36.72±1.09 |
| cora | uniform0.6 | MLP+co-teaching | 5 | 33.29±3.48 |
| cora | uniform0.6 | MLP+JoCoR | 5 | 36.92±2.14 |
| cora | uniform0.6 | MLP+ELR | 5 | 32.82±3.51 |
| cora | uniform0.6 | MLP+SCE | 5 | 32.66±3.78 |
| cora | uniform0.6 | SoftDist-MLP | 5 | 37.12±2.88 |
| cora | uniform0.6 | SoftDist shuffled | 5 | 33.01±2.69 |

## C1 Label Robustness Gate

| Dataset | Noise | Robust baselines beaten | Complete | Pass |
|---|---:|---:|---:|---:|
| cora | instance0.4 | 5/5 | 5/5 | 1 |
| citeseer | instance0.4 | 5/5 | 5/5 | 1 |
| coauthorcs | uniform0.2 | 2/5 | 5/5 | 0 |
| coauthorcs | uniform0.4 | 5/5 | 5/5 | 1 |
| coauthorcs | uniform0.6 | 5/5 | 5/5 | 1 |
| coauthorcs | pair0.4 | 2/5 | 5/5 | 0 |
| coauthorcs | instance0.4 | 5/5 | 5/5 | 1 |

- C1 summary: 5/7 complete new settings pass (pass means SoftDist beats at least 3/5 robust MLP strategies with paired seed support).

## Main Table 2: Structure Noise Robustness

| Dataset | Method | Mode | p | n | Test acc mean±std |
|---|---|---|---:|---:|---:|
| citeseer | SoftDist-MLP | clean | 0.00 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | edge_drop | 0.25 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | edge_drop | 0.50 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | edge_drop | 0.75 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | isolated | 0.25 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | isolated | 0.50 | 5 | 66.68±0.33 |
| citeseer | SoftDist-MLP | isolated | 0.75 | 5 | 66.68±0.33 |
| citeseer | gcn | clean | 0.00 | 5 | 73.21±0.62 |
| citeseer | gcn | edge_drop | 0.25 | 5 | 72.62±0.96 |
| citeseer | gcn | edge_drop | 0.50 | 5 | 71.57±0.76 |
| citeseer | gcn | edge_drop | 0.75 | 5 | 70.83±1.07 |
| citeseer | gcn | isolated | 0.25 | 5 | 72.53±0.59 |
| citeseer | gcn | isolated | 0.50 | 5 | 71.98±0.69 |
| citeseer | gcn | isolated | 0.75 | 5 | 71.37±0.90 |
| citeseer | gin | clean | 0.00 | 5 | 69.91±1.04 |
| citeseer | gin | edge_drop | 0.25 | 5 | 68.57±1.23 |
| citeseer | gin | edge_drop | 0.50 | 5 | 66.80±1.27 |
| citeseer | gin | edge_drop | 0.75 | 5 | 64.68±1.59 |
| citeseer | gin | isolated | 0.25 | 5 | 66.86±1.14 |
| citeseer | gin | isolated | 0.50 | 5 | 64.74±1.95 |
| citeseer | gin | isolated | 0.75 | 5 | 62.69±2.19 |
| coauthorcs | SoftDist-MLP | clean | 0.00 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | edge_drop | 0.25 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | edge_drop | 0.50 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | edge_drop | 0.75 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | isolated | 0.25 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | isolated | 0.50 | 5 | 92.69±0.37 |
| coauthorcs | SoftDist-MLP | isolated | 0.75 | 5 | 92.69±0.37 |
| coauthorcs | gcn | clean | 0.00 | 5 | 93.13±0.12 |
| coauthorcs | gcn | edge_drop | 0.25 | 5 | 92.29±0.14 |
| coauthorcs | gcn | edge_drop | 0.50 | 5 | 91.00±0.11 |
| coauthorcs | gcn | edge_drop | 0.75 | 5 | 88.93±0.32 |
| coauthorcs | gcn | isolated | 0.25 | 5 | 92.18±0.37 |
| coauthorcs | gcn | isolated | 0.50 | 5 | 91.50±0.58 |
| coauthorcs | gcn | isolated | 0.75 | 5 | 90.91±0.75 |
| coauthorcs | gin | clean | 0.00 | 5 | 88.63±0.27 |
| coauthorcs | gin | edge_drop | 0.25 | 5 | 86.88±0.40 |
| coauthorcs | gin | edge_drop | 0.50 | 5 | 83.88±0.64 |
| coauthorcs | gin | edge_drop | 0.75 | 5 | 77.18±1.94 |
| coauthorcs | gin | isolated | 0.25 | 5 | 80.34±2.07 |
| coauthorcs | gin | isolated | 0.50 | 5 | 72.65±4.08 |
| coauthorcs | gin | isolated | 0.75 | 5 | 65.37±5.90 |
| cora | SoftDist-MLP | clean | 0.00 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | edge_drop | 0.25 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | edge_drop | 0.50 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | edge_drop | 0.75 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | isolated | 0.25 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | isolated | 0.50 | 5 | 66.88±1.17 |
| cora | SoftDist-MLP | isolated | 0.75 | 5 | 66.88±1.17 |
| cora | gcn | clean | 0.00 | 5 | 85.74±0.37 |
| cora | gcn | edge_drop | 0.25 | 5 | 84.33±0.55 |
| cora | gcn | edge_drop | 0.50 | 5 | 82.31±0.52 |
| cora | gcn | edge_drop | 0.75 | 5 | 79.93±1.13 |
| cora | gcn | isolated | 0.25 | 5 | 83.61±0.68 |
| cora | gcn | isolated | 0.50 | 5 | 81.06±0.70 |
| cora | gcn | isolated | 0.75 | 5 | 79.09±0.94 |
| cora | gin | clean | 0.00 | 5 | 83.26±0.73 |
| cora | gin | edge_drop | 0.25 | 5 | 80.56±1.05 |
| cora | gin | edge_drop | 0.50 | 5 | 77.29±1.48 |
| cora | gin | edge_drop | 0.75 | 5 | 71.77±2.48 |
| cora | gin | isolated | 0.25 | 5 | 77.17±1.66 |
| cora | gin | isolated | 0.50 | 5 | 71.53±2.13 |
| cora | gin | isolated | 0.75 | 5 | 67.37±2.79 |

## C2 Structure Robustness Gate

| Dataset | Mode | SoftDist drop @0.75 | Worst GNN drop @0.75 | Pass |
|---|---|---:|---:|---:|
| cora | edge_drop | 0.00 | -11.49 | 1 |
| cora | isolated | 0.00 | -15.89 | 1 |
| citeseer | edge_drop | 0.00 | -5.23 | 1 |
| citeseer | isolated | 0.00 | -7.23 | 1 |
| coauthorcs | edge_drop | 0.00 | -11.46 | 1 |
| coauthorcs | isolated | 0.00 | -23.26 | 1 |

## Ablation: SoftDist vs Components

| Dataset | Baseline removed/changed | n | SoftDist delta mean±std | Seedwise wins |
|---|---|---:|---:|---:|
| citeseer | w/o GRACE (X balls) | 5 | 8.87±1.43 | 5/5 |
| citeseer | hard majority | 5 | -5.11±1.75 | 0/5 |
| citeseer | SoftDist shuffled | 5 | 8.14±2.09 | 5/5 |
| citeseer | MLP+CE | 5 | 8.12±1.64 | 5/5 |
| coauthorcs | w/o GRACE (X balls) | 5 | 0.56±0.12 | 5/5 |
| coauthorcs | hard majority | 5 | -1.84±0.27 | 0/5 |
| coauthorcs | SoftDist shuffled | 5 | 2.23±0.74 | 5/5 |
| coauthorcs | MLP+CE | 5 | 1.53±0.60 | 5/5 |
| cora | w/o GRACE (X balls) | 5 | 7.25±2.75 | 5/5 |
| cora | hard majority | 5 | -7.20±1.99 | 0/5 |
| cora | SoftDist shuffled | 5 | 7.74±1.87 | 5/5 |
| cora | MLP+CE | 5 | 7.60±3.29 | 5/5 |

## Interpretation Checklist

- C1 decides whether label-robustness extends beyond the MLP-1 Cora/CiteSeer grid.
- C2 tests the structural half of the dual-robust framing; SoftDist-MLP should be invariant because inference ignores adjacency.
- C3 is supported only if GRACE balls, soft targets, and real ball structure all beat their removals on paired seeds.
