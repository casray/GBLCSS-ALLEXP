# MLP-43 Corrected Grouping Alternatives

Protocol: uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, seeds 0-4. All kNN rows exclude the query node and all grouping alternatives use `tie_mode=distribution_only`.

| Dataset | GB-LCSS | kNN | kNN-GB | k-means | k-means-GB |
|---|---:|---:|---:|---:|---:|
| cora | 56.28 +/- 1.54 | 57.21 +/- 1.19 | +0.93 (3/5) | 56.01 +/- 0.70 | -0.27 (3/5) |
| citeseer | 53.52 +/- 2.71 | 56.17 +/- 3.24 | +2.65 (5/5) | 55.07 +/- 1.85 | +1.55 (4/5) |
| corafull | 37.30 +/- 0.57 | 40.50 +/- 0.51 | +3.20 (5/5) | 39.28 +/- 0.52 | +1.97 (5/5) |
| coauthorcs | 90.62 +/- 0.47 | 90.83 +/- 0.57 | +0.21 (4/5) | 90.93 +/- 0.64 | +0.31 (4/5) |
| amazonpho | 81.33 +/- 0.70 | 82.23 +/- 1.20 | +0.90 (3/5) | 81.47 +/- 0.90 | +0.14 (3/5) |

## Decision

- The corrected kNN result is not explained by self-neighbor leakage.
- Granular balls are not an accuracy-unique grouping mechanism in the frozen GRACE space; kNN is stronger on several datasets and k-means is competitive.
- The manuscript should attribute the main effect to local-consensus soft supervision and present granular balls as a fixed-budget implementation, not as an irreplaceable accuracy source.
