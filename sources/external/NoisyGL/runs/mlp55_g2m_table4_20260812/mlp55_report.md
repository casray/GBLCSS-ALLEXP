# MLP-55 GNN-to-MLP Table 4 Report

## Protocol

- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, percent 0.2/0.1/0.7.
- Five paired seeds; every dataset--noise--seed independently retrains teacher and student.
- All students use X-only inference.
- FF-G2M is a common-backbone paper reimplementation; KRD and VQGraph rows are mechanism adaptations.

## New Methods

| Method | Dataset | Noise | Mean+-std (%) |
|---|---|---:|---:|
| FF-G2M (common-backbone reimplementation) | cora | 0.4 | 77.65+-2.97 |
| KRD-inspired reliability adaptation | cora | 0.4 | 71.77+-2.20 |
| VQGraph-inspired token adaptation | cora | 0.4 | 78.89+-2.34 |
| FF-G2M (common-backbone reimplementation) | cora | 0.6 | 63.63+-2.16 |
| KRD-inspired reliability adaptation | cora | 0.6 | 61.28+-1.17 |
| VQGraph-inspired token adaptation | cora | 0.6 | 65.75+-2.72 |
| FF-G2M (common-backbone reimplementation) | citeseer | 0.4 | 65.80+-0.49 |
| KRD-inspired reliability adaptation | citeseer | 0.4 | 64.96+-1.03 |
| VQGraph-inspired token adaptation | citeseer | 0.4 | 65.64+-1.72 |
| FF-G2M (common-backbone reimplementation) | citeseer | 0.6 | 52.32+-4.20 |
| KRD-inspired reliability adaptation | citeseer | 0.6 | 54.17+-3.14 |
| VQGraph-inspired token adaptation | citeseer | 0.6 | 53.34+-4.47 |
| FF-G2M (common-backbone reimplementation) | corafull | 0.4 | 56.58+-0.93 |
| KRD-inspired reliability adaptation | corafull | 0.4 | 47.12+-0.95 |
| VQGraph-inspired token adaptation | corafull | 0.4 | 58.52+-0.77 |
| FF-G2M (common-backbone reimplementation) | corafull | 0.6 | 45.78+-2.90 |
| KRD-inspired reliability adaptation | corafull | 0.6 | 41.61+-2.43 |
| VQGraph-inspired token adaptation | corafull | 0.6 | 52.01+-0.65 |

## Table 4

| Training route | Student inference | cora u0.4 | citeseer u0.4 | corafull u0.4 | cora u0.6 | citeseer u0.6 | corafull u0.6 |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Base GB-LCSS (without distillation) | X-only | 57.49+-0.76 | 55.24+-2.04 | 37.39+-0.51 | 41.86+-2.30 | 42.02+-3.14 | 27.43+-0.89 |
| GCN(X)+CE teacher (GLNN-style) | X-only | 77.95+-2.62 | 66.09+-1.38 | 58.74+-0.82 | 65.50+-3.28 | 53.30+-4.42 | 51.63+-2.02 |
| FF-G2M (common-backbone reimplementation) | X-only | 77.65+-2.97 | 65.80+-0.49 | 56.58+-0.93 | 63.63+-2.16 | 52.32+-4.20 | 45.78+-2.90 |
| KRD-inspired reliability adaptation | X-only | 71.77+-2.20 | 64.96+-1.03 | 47.12+-0.95 | 61.28+-1.17 | 54.17+-3.14 | 41.61+-2.43 |
| VQGraph-inspired token adaptation | X-only | 78.89+-2.34 | 65.64+-1.72 | 58.52+-0.77 | 65.75+-2.72 | 53.34+-4.47 | 52.01+-0.65 |
| Granular-ball consensus teacher | X-only | 80.60+-0.69 | 69.40+-1.08 | 58.22+-0.69 | 71.27+-2.46 | 62.28+-2.30 | 51.82+-0.86 |
| NRGNN teacher | X-only | 81.07+-0.55 | 70.87+-1.06 | 57.71+-0.58 | 74.81+-1.17 | 66.62+-1.27 | 53.55+-1.00 |

## Paired Deltas

| Method | Baseline | Dataset | Noise | Delta (pp) | Positive seeds |
|---|---|---|---:|---:|---:|
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | cora | 0.4 | -0.31 | 2/5 |
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | cora | 0.6 | -1.87 | 1/5 |
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.4 | -0.29 | 3/5 |
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.6 | -0.98 | 0/5 |
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | corafull | 0.4 | -2.16 | 0/5 |
| FF-G2M (common-backbone reimplementation) | GCN(X)+CE teacher (GLNN-style) | corafull | 0.6 | -5.85 | 0/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | cora | 0.4 | -2.96 | 1/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | cora | 0.6 | -7.64 | 0/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | citeseer | 0.4 | -3.60 | 0/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | citeseer | 0.6 | -9.96 | 0/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | corafull | 0.4 | -1.64 | 0/5 |
| FF-G2M (common-backbone reimplementation) | Granular-ball consensus teacher | corafull | 0.6 | -6.04 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | cora | 0.4 | -3.42 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | cora | 0.6 | -11.18 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | citeseer | 0.4 | -5.07 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | citeseer | 0.6 | -14.30 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | corafull | 0.4 | -1.12 | 0/5 |
| FF-G2M (common-backbone reimplementation) | NRGNN teacher | corafull | 0.6 | -7.77 | 0/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | cora | 0.4 | -6.18 | 0/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | cora | 0.6 | -4.22 | 0/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.4 | -1.13 | 1/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.6 | +0.87 | 3/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | corafull | 0.4 | -11.63 | 0/5 |
| KRD-inspired reliability adaptation | GCN(X)+CE teacher (GLNN-style) | corafull | 0.6 | -10.02 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | cora | 0.4 | -8.83 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | cora | 0.6 | -9.99 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | citeseer | 0.4 | -4.44 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | citeseer | 0.6 | -8.11 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | corafull | 0.4 | -11.11 | 0/5 |
| KRD-inspired reliability adaptation | Granular-ball consensus teacher | corafull | 0.6 | -10.21 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | cora | 0.4 | -9.30 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | cora | 0.6 | -13.53 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | citeseer | 0.4 | -5.91 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | citeseer | 0.6 | -12.45 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | corafull | 0.4 | -10.59 | 0/5 |
| KRD-inspired reliability adaptation | NRGNN teacher | corafull | 0.6 | -11.95 | 0/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | cora | 0.4 | +0.94 | 5/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | cora | 0.6 | +0.25 | 2/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.4 | -0.46 | 1/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | citeseer | 0.6 | +0.04 | 3/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | corafull | 0.4 | -0.23 | 2/5 |
| VQGraph-inspired token adaptation | GCN(X)+CE teacher (GLNN-style) | corafull | 0.6 | +0.39 | 3/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | cora | 0.4 | -1.71 | 1/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | cora | 0.6 | -5.52 | 0/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | citeseer | 0.4 | -3.76 | 0/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | citeseer | 0.6 | -8.93 | 0/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | corafull | 0.4 | +0.29 | 3/5 |
| VQGraph-inspired token adaptation | Granular-ball consensus teacher | corafull | 0.6 | +0.19 | 3/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | cora | 0.4 | -2.17 | 1/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | cora | 0.6 | -9.06 | 0/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | citeseer | 0.4 | -5.23 | 0/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | citeseer | 0.6 | -13.27 | 0/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | corafull | 0.4 | +0.81 | 4/5 |
| VQGraph-inspired token adaptation | NRGNN teacher | corafull | 0.6 | -1.54 | 1/5 |

## Completeness

- New formal records: 90/90.
