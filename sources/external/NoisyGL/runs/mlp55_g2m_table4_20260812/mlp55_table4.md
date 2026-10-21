| Training route | Student inference | cora u0.4 | citeseer u0.4 | corafull u0.4 | cora u0.6 | citeseer u0.6 | corafull u0.6 |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Base GB-LCSS (without distillation) | X-only | 57.49+-0.76 | 55.24+-2.04 | 37.39+-0.51 | 41.86+-2.30 | 42.02+-3.14 | 27.43+-0.89 |
| GCN(X)+CE teacher (GLNN-style) | X-only | 77.95+-2.62 | 66.09+-1.38 | 58.74+-0.82 | 65.50+-3.28 | 53.30+-4.42 | 51.63+-2.02 |
| FF-G2M (common-backbone reimplementation) | X-only | 77.65+-2.97 | 65.80+-0.49 | 56.58+-0.93 | 63.63+-2.16 | 52.32+-4.20 | 45.78+-2.90 |
| KRD-inspired reliability adaptation | X-only | 71.77+-2.20 | 64.96+-1.03 | 47.12+-0.95 | 61.28+-1.17 | 54.17+-3.14 | 41.61+-2.43 |
| VQGraph-inspired token adaptation | X-only | 78.89+-2.34 | 65.64+-1.72 | 58.52+-0.77 | 65.75+-2.72 | 53.34+-4.47 | 52.01+-0.65 |
| Granular-ball consensus teacher | X-only | 80.60+-0.69 | 69.40+-1.08 | 58.22+-0.69 | 71.27+-2.46 | 62.28+-2.30 | 51.82+-0.86 |
| NRGNN teacher | X-only | 81.07+-0.55 | 70.87+-1.06 | 57.71+-0.58 | 74.81+-1.17 | 66.62+-1.27 | 53.55+-1.00 |
