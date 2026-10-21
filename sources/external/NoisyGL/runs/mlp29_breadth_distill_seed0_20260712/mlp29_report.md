# MLP-29 Breadth and Robust-Teacher Distillation Seed0

## Direct X-only MLP

| Dataset | LCSS | Small-loss | Combination | Shuffled | Synergy | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|
| corafull | 28.13 | 22.40 | 33.23 | 23.83 | +5.10pp | +9.40pp |
| coauthorcs | 86.22 | 81.33 | 88.87 | 80.36 | +2.65pp | +8.52pp |
| amazonpho | 66.63 | 55.20 | 72.01 | 57.74 | +5.38pp | +14.27pp |
| pubmed | 49.75 | 58.82 | 65.32 | 51.48 | +6.50pp | +13.84pp |
| amazoncom | 68.92 | 64.74 | 71.64 | 62.78 | +2.72pp | +8.86pp |
| coauthorph | 86.90 | 89.15 | 91.81 | 85.02 | +2.66pp | +6.80pp |

## X-only student after teacher distillation

| Dataset | Bare | Beta0.5 SoftDist | Beta0.5 kNN | Robust SoftDist | Robust shuffled | Robust-plain | Robust-kNN | Real-shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corafull | 52.61 | 51.65 | 52.85 | 53.78 | 54.02 | +2.13pp | +0.92pp | -0.24pp |
| coauthorcs | 89.04 | 92.60 | 93.05 | 93.99 | 92.36 | +1.39pp | +0.94pp | +1.64pp |
| amazonpho | 84.65 | 87.56 | 87.84 | 90.27 | 74.77 | +2.71pp | +2.43pp | +15.50pp |
| pubmed | 76.38 | 76.90 | 77.67 | 60.96 | 61.05 | -15.94pp | -16.71pp | -0.09pp |
| amazoncom | 78.69 | 75.89 | 75.10 | 78.97 | 71.44 | +3.09pp | +3.87pp | +7.53pp |
| coauthorph | 92.96 | 94.50 | 94.71 | 94.97 | 94.26 | +0.48pp | +0.27pp | +0.71pp |

## Gates

- direct: pass=1, positive=6/6, mean=+4.17pp, min=+2.65pp, structure>=1pp=6/6, reasons=pass
- distill: pass=0, positive=5/6, mean=-1.02pp, min=-15.94pp, structure>=1pp=3/6, reasons=mean_teacher_gain_below_0p5pp;teacher_regression_over_1pp;fewer_than_4_of_6_teacher_structure_gain_1pp
