# MLP-20 Structure Robustness Report

Protocol: Cora/CiteSeer/Coauthor-CS, train-only uniform label noise rate 0.4, percent split 0.2/0.1/0.7, seeds 0-4. Structure perturbations are applied only at inference time.

- Raw rows: 300
- Status counts: {'ok': 300}
- Max modified val/test labels: 0 / 0

## Structure perturbation p=0.5

| Dataset | Method | Clean | Edge drop test | Edge drop delta | Isolated test | Isolated delta |
|---|---:|---:|---:|---:|---:|---:|
| cora | GCN | 78.97±1.63 | 74.29±1.91 | -4.68±0.56 | 70.72±1.82 | -8.24±0.54 |
| cora | GIN | 66.26±2.25 | 57.32±1.76 | -8.94±2.75 | 48.02±4.73 | -18.24±5.45 |
| cora | LCSS train-only | 55.59±1.50 | 55.59±1.50 | +0.00±0.00 | 55.59±1.50 | +0.00±0.00 |
| cora | LCSS-distill | 80.60±0.69 | 80.60±0.69 | +0.00±0.00 | 80.60±0.69 | +0.00±0.00 |
| citeseer | GCN | 64.53±1.54 | 61.70±1.02 | -2.83±0.80 | 61.23±1.50 | -3.30±0.61 |
| citeseer | GIN | 49.82±2.25 | 44.46±2.43 | -5.36±1.39 | 40.98±2.67 | -8.84±2.49 |
| citeseer | LCSS train-only | 53.83±2.75 | 53.83±2.75 | +0.00±0.00 | 53.83±2.75 | +0.00±0.00 |
| citeseer | LCSS-distill | 69.40±1.08 | 69.40±1.08 | +0.00±0.00 | 69.40±1.08 | +0.00±0.00 |
| coauthorcs | GCN | 92.06±0.34 | 90.19±0.48 | -1.87±0.22 | 91.33±1.15 | -0.72±0.84 |
| coauthorcs | GIN | 57.94±4.73 | 55.18±3.73 | -2.75±2.70 | 48.56±6.25 | -9.38±7.77 |
| coauthorcs | LCSS train-only | 90.62±0.57 | 90.62±0.57 | +0.00±0.00 | 90.62±0.57 | +0.00±0.00 |
| coauthorcs | LCSS-distill | 93.75±0.22 | 93.75±0.22 | +0.00±0.00 | 93.75±0.22 | +0.00±0.00 |

## Structure perturbation p=0.75

| Dataset | Method | Clean | Edge drop test | Edge drop delta | Isolated test | Isolated delta |
|---|---:|---:|---:|---:|---:|---:|
| cora | GCN | 78.97±1.63 | 70.36±1.39 | -8.60±0.51 | 67.99±2.24 | -10.98±0.89 |
| cora | GIN | 66.26±2.25 | 48.06±4.47 | -18.20±5.23 | 41.53±6.27 | -24.73±7.15 |
| cora | LCSS train-only | 55.59±1.50 | 55.59±1.50 | +0.00±0.00 | 55.59±1.50 | +0.00±0.00 |
| cora | LCSS-distill | 80.60±0.69 | 80.60±0.69 | +0.00±0.00 | 80.60±0.69 | +0.00±0.00 |
| citeseer | GCN | 64.53±1.54 | 59.95±1.28 | -4.58±0.53 | 59.82±1.48 | -4.71±0.71 |
| citeseer | GIN | 49.82±2.25 | 41.01±2.91 | -8.81±2.59 | 38.03±3.49 | -11.79±3.72 |
| citeseer | LCSS train-only | 53.83±2.75 | 53.83±2.75 | +0.00±0.00 | 53.83±2.75 | +0.00±0.00 |
| citeseer | LCSS-distill | 69.40±1.08 | 69.40±1.08 | +0.00±0.00 | 69.40±1.08 | +0.00±0.00 |
| coauthorcs | GCN | 92.06±0.34 | 88.29±0.77 | -3.76±0.49 | 91.15±1.43 | -0.91±1.12 |
| coauthorcs | GIN | 57.94±4.73 | 50.89±5.07 | -7.04±6.51 | 44.60±9.04 | -13.34±11.08 |
| coauthorcs | LCSS train-only | 90.62±0.57 | 90.62±0.57 | +0.00±0.00 | 90.62±0.57 | +0.00±0.00 |
| coauthorcs | LCSS-distill | 93.75±0.22 | 93.75±0.22 | +0.00±0.00 | 93.75±0.22 | +0.00±0.00 |

## Key conclusion

- LCSS train-only and LCSS-distill are graph-free at inference, so their edge_drop/isolated deltas are exactly 0.00pp in every dataset/seed/perturbation.
- GCN and GIN depend on the inference graph; their performance degrades under edge removal and isolated-node simulation, especially at p=0.75.
- This supports the structural robustness claim as a deployment-property claim: label-noise robustness comes from local-consensus soft supervision, while structure-noise robustness comes from graph-free inference.