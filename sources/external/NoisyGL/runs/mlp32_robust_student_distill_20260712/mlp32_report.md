# MLP-32 Robust-Student Distillation Seed0 Report

| Dataset | Direct | Plain distill | Robust student | Shuffled | Teacher | vs plain | vs direct | real-shuffled | retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 60.16 | 79.42 | 79.58 | 78.58 | 79.58 | +0.16pp | +19.42pp | +1.00pp | 100.0% |
| citeseer | 59.84 | 69.46 | 68.73 | 67.74 | 67.74 | -0.73pp | +8.89pp | +0.99pp | 101.5% |
| corafull | 41.27 | 58.61 | 58.51 | 58.48 | 61.42 | -0.09pp | +17.24pp | +0.03pp | 95.3% |
| amazonpho | 81.20 | 89.92 | 88.81 | 87.64 | 89.43 | -1.10pp | +7.62pp | +1.18pp | 99.3% |
| pubmed | 81.24 | 83.81 | 83.44 | 82.91 | 82.06 | -0.36pp | +2.20pp | +0.53pp | 101.7% |
| coauthorph | 92.71 | 95.22 | 94.36 | 94.44 | 95.25 | -0.86pp | +1.65pp | -0.08pp | 99.1% |

## Gate

- Pass: 0
- Above plain distillation: 1/6
- Mean robust minus plain: -0.50pp
- Above direct method: 6/6
- Real-shuffled >=1pp: 2/6
- Teacher retention >=95%: 6/6
- Regressions over 1pp: 1
- Reasons: fewer_than_4_of_6_above_plain_distillation;mean_plain_distillation_gain_below_1pp;fewer_than_4_of_6_structure_gains_1pp;plain_distillation_regression_over_1pp
