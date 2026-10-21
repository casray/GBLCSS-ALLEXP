# MLP-15 GCN-Hidden Balls Report

## Protocol

- Scope: why-GRACE diagnostic; use noisy-label GCN hidden embeddings as ball features, then train the normal SoftDist-MLP student.
- Datasets: Cora, CiteSeer.
- Train ratios: 0.05, 0.10, 0.20; validation ratio fixed at 0.10.
- Noise: train-only uniform 0.4; seed 0.
- Arms: frozen-GRACE anchor, GCN-hidden real, GCN-hidden shuffled, MLP+CE, and GCN+CE reference.

## Results

| Dataset | train% | Method | Test | Valid | Balls | True train purity |
|---|---:|---|---:|---:|---:|---:|
| citeseer | 5 | GCN+CE reference | 49.59 | 52.11 |  |  |
| citeseer | 5 | GCN-hidden real | 40.08 | 41.87 | 17 | 0.5662650602409639 |
| citeseer | 5 | GCN-hidden shuffled | 35.27 | 38.25 | 17 | 0.3597560975609756 |
| citeseer | 5 | GRACE anchor | 44.43 | 41.27 | 64 | 0.738562091503268 |
| citeseer | 5 | MLP+CE | 35.02 | 38.25 |  |  |
| citeseer | 10 | GCN+CE reference | 58.29 | 55.12 |  |  |
| citeseer | 10 | GCN-hidden real | 50.43 | 48.80 | 23 | 0.6404833836858006 |
| citeseer | 10 | GCN-hidden shuffled | 41.75 | 44.88 | 23 | 0.3293051359516616 |
| citeseer | 10 | GRACE anchor | 50.36 | 53.01 | 64 | 0.7112462006079028 |
| citeseer | 10 | MLP+CE | 40.40 | 42.17 |  |  |
| citeseer | 20 | GCN+CE reference | 64.86 | 68.37 |  |  |
| citeseer | 20 | GCN-hidden real | 56.19 | 59.04 | 26 | 0.6827067669172933 |
| citeseer | 20 | GCN-hidden shuffled | 43.60 | 46.69 | 26 | 0.281203007518797 |
| citeseer | 20 | GRACE anchor | 55.33 | 59.04 | 64 | 0.7007518796992481 |
| citeseer | 20 | MLP+CE | 46.09 | 45.48 |  |  |
| cora | 5 | GCN+CE reference | 62.06 | 64.07 |  |  |
| cora | 5 | GCN-hidden real | 41.24 | 46.67 | 18 | 0.6666666666666666 |
| cora | 5 | GCN-hidden shuffled | 34.72 | 35.93 | 18 | 0.40601503759398494 |
| cora | 5 | GRACE anchor | 40.98 | 41.11 | 52 | 0.819672131147541 |
| cora | 5 | MLP+CE | 33.12 | 32.96 |  |  |
| cora | 10 | GCN+CE reference | 67.91 | 72.96 |  |  |
| cora | 10 | GCN-hidden real | 45.89 | 52.22 | 40 | 0.7969924812030075 |
| cora | 10 | GCN-hidden shuffled | 39.80 | 46.30 | 40 | 0.40671641791044777 |
| cora | 10 | GRACE anchor | 47.51 | 58.52 | 52 | 0.8208955223880597 |
| cora | 10 | MLP+CE | 37.95 | 40.37 |  |  |
| cora | 20 | GCN+CE reference | 79.26 | 78.52 |  |  |
| cora | 20 | GCN-hidden real | 55.04 | 59.63 | 31 | 0.8096118299445472 |
| cora | 20 | GCN-hidden shuffled | 44.01 | 46.67 | 31 | 0.3456561922365989 |
| cora | 20 | GRACE anchor | 55.46 | 57.04 | 52 | 0.789279112754159 |
| cora | 20 | MLP+CE | 41.74 | 47.04 |  |  |

## Paired Deltas

| Dataset | train% | Comparison | Delta |
|---|---:|---|---:|
| citeseer | 5 | gcn_hidden_real_minus_GRACE anchor | -4.35pp |
| citeseer | 5 | gcn_hidden_real_minus_GCN-hidden shuffled | +4.81pp |
| citeseer | 5 | gcn_hidden_real_minus_MLP+CE | +5.06pp |
| citeseer | 10 | gcn_hidden_real_minus_GRACE anchor | +0.08pp |
| citeseer | 10 | gcn_hidden_real_minus_GCN-hidden shuffled | +8.68pp |
| citeseer | 10 | gcn_hidden_real_minus_MLP+CE | +10.03pp |
| citeseer | 20 | gcn_hidden_real_minus_GRACE anchor | +0.86pp |
| citeseer | 20 | gcn_hidden_real_minus_GCN-hidden shuffled | +12.59pp |
| citeseer | 20 | gcn_hidden_real_minus_MLP+CE | +10.09pp |
| cora | 5 | gcn_hidden_real_minus_GRACE anchor | +0.26pp |
| cora | 5 | gcn_hidden_real_minus_GCN-hidden shuffled | +6.52pp |
| cora | 5 | gcn_hidden_real_minus_MLP+CE | +8.13pp |
| cora | 10 | gcn_hidden_real_minus_GRACE anchor | -1.62pp |
| cora | 10 | gcn_hidden_real_minus_GCN-hidden shuffled | +6.09pp |
| cora | 10 | gcn_hidden_real_minus_MLP+CE | +7.94pp |
| cora | 20 | gcn_hidden_real_minus_GRACE anchor | -0.42pp |
| cora | 20 | gcn_hidden_real_minus_GCN-hidden shuffled | +11.03pp |
| cora | 20 | gcn_hidden_real_minus_MLP+CE | +13.30pp |

## Decision

- GCN-hidden real beats GRACE anchor by >0.5pp in 1/6 settings.
- GCN-hidden real beats its shuffled control in 6/6 settings.
- This is a seed0 diagnostic only. Expand only if it beats GRACE anchor in at least 4/6 settings and beats shuffled in at least 5/6 settings.
