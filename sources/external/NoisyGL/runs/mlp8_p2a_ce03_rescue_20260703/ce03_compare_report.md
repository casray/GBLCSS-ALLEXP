# MLP-8 P2a CE-0.3 Compare Report

## Protocol

- Dataset: Cora
- Seed: 0
- Noise: train-only uniform 0.4
- Split: percent 0.2/0.1/0.7
- Candidates: GRACE + noisy CE with lambda 0.2 and 0.3
- Downstream: complete SoftDist-MLP standard runner (`run_ssl_swap_downstream.py`)

## Comparison

| Lambda | Params | Real test | Shuffled test | Real - shuffled | Balls | True train purity |
|---:|---|---:|---:|---:|---:|---:|
| 0.2 | default 64/32/0.05 | 55.46 | 42.64 | +12.82pp | 39 | 0.7967 |
| 0.2 | loose 64/16/0.0 | 53.67 | 42.16 | +11.50pp | 64 | 0.8219 |
| 0.3 | default 64/32/0.05 | 54.99 | 44.06 | +10.92pp | 50 | 0.8299 |
| 0.3 | loose 64/16/0.0 | 53.09 | 42.37 | +10.71pp | 64 | 0.8430 |

## Selected Complete Rerun

The highest candidate is CE=0.2 with default ball parameters. It was rerun as a clean complete SoftDist-MLP standard pipeline and wrote `selected_complete_ce02_default.csv`.

| Method | Test | Balls | True train purity |
|---|---:|---:|---:|
| CE-0.2 default real | 55.46 | 39 | 0.7967 |
| CE-0.2 default shuffled | 42.64 | 39 | 0.3530 |

## Decision

CE=0.3 does not improve over CE=0.2. CE=0.2 default is the best candidate, but it only matches the frozen-GRACE default anchor. The supervised noisy-CE-assisted representation produces strong real-vs-shuffled structure, but it does not provide additional downstream accuracy over label-free frozen GRACE.

Do not expand P2a to 5 seeds.
