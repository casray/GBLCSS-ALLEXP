# MLP-41 Fixed k=5 kNN Teacher Report

Protocol: Cora/CiteSeer/CoraFull, train-only uniform 0.4/0.6, percent 0.2/0.1/0.7, seeds 0-4.

| Dataset | Noise | kNN k=5 teacher | Ball teacher | Original matched-k teacher | k5-ball | k5-matched |
|---|---:|---:|---:|---:|---:|---:|
| citeseer | 0.4 | 68.82+-1.76 | 69.40+-1.08 | 69.57+-1.46 | -0.58pp (3/5) | -0.75pp (2/5) |
| citeseer | 0.6 | 62.41+-2.30 | 62.28+-2.30 | 61.96+-2.98 | +0.13pp (2/5) | +0.45pp (3/5) |
| cora | 0.4 | 81.23+-0.89 | 80.60+-0.69 | 80.99+-1.02 | +0.63pp (4/5) | +0.24pp (4/5) |
| cora | 0.6 | 71.13+-2.55 | 71.27+-2.46 | 71.07+-2.29 | -0.14pp (2/5) | +0.06pp (2/5) |
| corafull | 0.4 | 59.44+-0.54 | 58.22+-0.69 | 58.95+-0.56 | +1.21pp (5/5) | +0.49pp (4/5) |
| corafull | 0.6 | 53.25+-0.65 | 51.82+-0.86 | 52.73+-0.95 | +1.42pp (4/5) | +0.52pp (4/5) |

## Interpretation Boundary

This experiment compares the ball teacher with a deliberately fixed local kNN scale (k=5). It does not replace the historical evidence that scale-matched kNN can equal or exceed the ball teacher.
