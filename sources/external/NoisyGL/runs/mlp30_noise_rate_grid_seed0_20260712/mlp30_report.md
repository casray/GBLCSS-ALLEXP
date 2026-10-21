# MLP-30 LCSS + Soft-Small-Loss Noise-Rate Grid Seed0

## Per-setting results

| Dataset | Rate | LCSS | Small-loss | Combination | Shuffled | Synergy | Combo-LCSS | Combo-Small | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 0.2 | 63.06 | 58.52 | 64.27 | 51.19 | +1.21pp | +1.21pp | +5.75pp | +13.09pp |
| cora | 0.4 | 55.46 | 49.50 | 60.16 | 45.70 | +4.70pp | +4.70pp | +10.66pp | +14.46pp |
| cora | 0.6 | 38.94 | 37.68 | 44.96 | 34.56 | +6.02pp | +6.02pp | +7.28pp | +10.40pp |
| citeseer | 0.2 | 61.51 | 57.60 | 63.27 | 55.50 | +1.76pp | +1.76pp | +5.67pp | +7.77pp |
| citeseer | 0.4 | 55.33 | 49.05 | 59.84 | 44.33 | +4.51pp | +4.51pp | +10.78pp | +15.51pp |
| citeseer | 0.6 | 38.62 | 36.94 | 41.80 | 31.92 | +3.18pp | +3.18pp | +4.85pp | +9.88pp |
| citeseerfull | 0.2 | 69.64 | 65.32 | 72.34 | 63.96 | +2.70pp | +2.70pp | +7.02pp | +8.38pp |
| citeseerfull | 0.4 | 59.81 | 56.91 | 66.67 | 48.94 | +6.86pp | +6.86pp | +9.76pp | +17.73pp |
| citeseerfull | 0.6 | 39.68 | 38.26 | 49.11 | 30.56 | +9.42pp | +9.42pp | +10.84pp | +18.54pp |
| corafull | 0.2 | 43.29 | 41.36 | 44.55 | 41.75 | +1.26pp | +1.26pp | +3.19pp | +2.80pp |
| corafull | 0.4 | 37.39 | 33.05 | 41.27 | 34.43 | +3.88pp | +3.88pp | +8.22pp | +6.84pp |
| corafull | 0.6 | 28.13 | 22.40 | 33.23 | 23.83 | +5.10pp | +5.10pp | +10.83pp | +9.40pp |
| coauthorcs | 0.2 | 91.61 | 91.06 | 91.37 | 90.64 | -0.23pp | -0.23pp | +0.31pp | +0.73pp |
| coauthorcs | 0.4 | 90.35 | 88.90 | 90.77 | 88.45 | +0.42pp | +0.42pp | +1.88pp | +2.32pp |
| coauthorcs | 0.6 | 86.22 | 81.33 | 88.87 | 80.36 | +2.65pp | +2.65pp | +7.54pp | +8.52pp |
| amazonpho | 0.2 | 84.03 | 84.41 | 84.09 | 82.33 | -0.32pp | +0.06pp | -0.32pp | +1.76pp |
| amazonpho | 0.4 | 81.44 | 74.77 | 81.20 | 76.15 | -0.24pp | -0.24pp | +6.42pp | +5.04pp |
| amazonpho | 0.6 | 66.63 | 55.20 | 72.01 | 57.74 | +5.38pp | +5.38pp | +16.81pp | +14.27pp |
| pubmed | 0.2 | 82.38 | 81.62 | 83.52 | 80.73 | +1.14pp | +1.14pp | +1.90pp | +2.79pp |
| pubmed | 0.4 | 71.60 | 77.46 | 81.24 | 73.61 | +3.78pp | +9.64pp | +3.78pp | +7.63pp |
| pubmed | 0.6 | 49.75 | 58.82 | 65.32 | 51.48 | +6.50pp | +15.57pp | +6.50pp | +13.84pp |
| amazoncom | 0.2 | 77.55 | 80.03 | 77.60 | 75.92 | -2.43pp | +0.05pp | -2.43pp | +1.68pp |
| amazoncom | 0.4 | 75.33 | 76.63 | 76.15 | 74.40 | -0.48pp | +0.82pp | -0.48pp | +1.75pp |
| amazoncom | 0.6 | 68.92 | 64.74 | 71.64 | 62.78 | +2.72pp | +2.72pp | +6.90pp | +8.86pp |
| coauthorph | 0.2 | 94.04 | 94.68 | 93.73 | 91.73 | -0.96pp | -0.31pp | -0.96pp | +2.00pp |
| coauthorph | 0.4 | 93.05 | 93.05 | 92.71 | 90.92 | -0.34pp | -0.34pp | -0.34pp | +1.79pp |
| coauthorph | 0.6 | 86.90 | 89.15 | 91.81 | 85.02 | +2.66pp | +4.91pp | +2.66pp | +6.80pp |

## Rate gates

- uniform 0.2: pass=0, synergy-positive=5/9, mean=+0.46pp, regressions>1pp=1, structure>=1pp=8/9, reasons=fewer_than_6_of_9_synergy_positive;mean_synergy_below_1pp
- uniform 0.4: pass=1, synergy-positive=6/9, mean=+2.57pp, regressions>1pp=0, structure>=1pp=9/9, reasons=pass
- uniform 0.6: pass=1, synergy-positive=9/9, mean=+4.85pp, regressions>1pp=0, structure>=1pp=9/9, reasons=pass

## Synergy trends

| Dataset | u0.2 | u0.4 | u0.6 | Monotonic |
|---|---:|---:|---:|---:|
| cora | +1.21pp | +4.70pp | +6.02pp | 1 |
| citeseer | +1.76pp | +4.51pp | +3.18pp | 0 |
| citeseerfull | +2.70pp | +6.86pp | +9.42pp | 1 |
| corafull | +1.26pp | +3.88pp | +5.10pp | 1 |
| coauthorcs | -0.23pp | +0.42pp | +2.65pp | 1 |
| amazonpho | -0.32pp | -0.24pp | +5.38pp | 1 |
| pubmed | +1.14pp | +3.78pp | +6.50pp | 1 |
| amazoncom | -2.43pp | -0.48pp | +2.72pp | 1 |
| coauthorph | -0.96pp | -0.34pp | +2.66pp | 1 |
