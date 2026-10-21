# MLP-28 Soft-JoCoR Kill Smoke

| Dataset | LCSS | JoCoR-only | Soft-small-loss | Soft-JoCoR | Shuffled Soft-JoCoR | vs soft-small | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| cora | 38.94 | 36.68 | 44.96 | 45.49 | 33.09 | +0.53pp | +12.40pp |
| citeseer | 38.62 | 37.80 | 41.80 | 45.27 | 33.81 | +3.48pp | +11.47pp |
| citeseerfull | 39.68 | 41.30 | 49.11 | 51.00 | 36.00 | +1.89pp | +14.99pp |

## Gate

- Pass: 1
- Mean gain vs soft-small-loss: +1.97pp
- Minimum gain: +0.53pp
- Positive datasets: 3/3
- Real-shuffled >=1pp datasets: 3/3
- Reasons: pass
