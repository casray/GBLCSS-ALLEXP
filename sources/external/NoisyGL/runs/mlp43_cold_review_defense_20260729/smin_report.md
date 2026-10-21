# MLP-43 Minimum-Vote Sensitivity

Only the minimum number of noisy training votes required to activate a ball target is changed. The main method remains `s_min=2`.

| Dataset | s_min | Accuracy | Target coverage | Delta vs 2 | Positive seeds |
|---|---:|---:|---:|---:|---:|
| amazonpho | 2 | 81.33 +/- 0.70 | 100.00% | 0.00 | - |
| amazonpho | 3 | 81.56 +/- 1.02 | 99.97% | +0.23 | 1/5 |
| amazonpho | 5 | 81.43 +/- 0.99 | 99.83% | +0.10 | 2/5 |
| amazonpho | 10 | 80.95 +/- 0.75 | 92.73% | -0.38 | 1/5 |
| citeseer | 2 | 53.52 +/- 2.71 | 100.00% | 0.00 | - |
| citeseer | 3 | 53.58 +/- 2.67 | 99.82% | +0.06 | 1/5 |
| citeseer | 5 | 54.62 +/- 1.46 | 97.56% | +1.10 | 4/5 |
| citeseer | 10 | 50.83 +/- 2.13 | 69.74% | -2.69 | 0/5 |
| coauthorcs | 2 | 90.62 +/- 0.47 | 100.00% | 0.00 | - |
| coauthorcs | 3 | 90.62 +/- 0.47 | 100.00% | +0.00 | 0/5 |
| coauthorcs | 5 | 90.60 +/- 0.48 | 99.94% | -0.02 | 0/5 |
| coauthorcs | 10 | 90.56 +/- 0.43 | 99.26% | -0.06 | 2/5 |
| cora | 2 | 56.28 +/- 1.54 | 100.00% | 0.00 | - |
| cora | 3 | 56.28 +/- 1.54 | 100.00% | +0.00 | 0/5 |
| cora | 5 | 55.98 +/- 1.98 | 98.00% | -0.31 | 3/5 |
| cora | 10 | 51.46 +/- 1.88 | 65.77% | -4.82 | 0/5 |
| corafull | 2 | 37.30 +/- 0.57 | 100.00% | 0.00 | - |
| corafull | 3 | 37.30 +/- 0.57 | 100.00% | +0.00 | 0/5 |
| corafull | 5 | 37.30 +/- 0.57 | 100.00% | +0.00 | 0/5 |
| corafull | 10 | 37.11 +/- 0.75 | 99.46% | -0.20 | 2/5 |
