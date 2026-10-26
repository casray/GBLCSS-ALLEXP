# Locked posthoc tables

All classification runs use seed0. Accuracy and macro-F1 are percentages. No sample standard deviation is estimable from one seed; summary.csv uses null, not zero.
Noisy means are equal-weight averages over uniform0.4, uniform0.6 and pair0.4. Selection and continuation used validation only.

## E1: R0 noisy accuracy

| Dataset | CE | GCE | OLD64 | LEGACY32 | FIXED32 | SC | VAL | KM | KNN |
|---|---|---|---|---|---|---|---|---|---|
| cora | 38.100 | 37.555 | 48.566 | 50.150 | 50.150 | 49.077 | 50.800 | 48.109 | 48.549 |
| corafull | 27.566 | 25.170 | 33.184 | 35.395 | 35.511 | 35.434 | 35.742 | 36.141 | 37.462 |
| amazonpho | 64.002 | 64.538 | 75.331 | 74.865 | 73.881 | 74.230 | 75.363 | 75.020 | 74.672 |

## E1: clean accuracy / macro-F1

| Dataset | CE | SC | FIXED32 | KM | KNN |
|---|---|---|---|---|---|
| cora | 64.908 / 62.267 | 66.227 / 62.613 | 67.441 / 64.880 | 67.757 / 64.546 | 68.549 / 66.242 |
| corafull | 47.254 / 37.318 | 49.679 / 38.939 | 48.704 / 36.772 | 49.636 / 39.163 | 49.044 / 36.967 |
| amazonpho | 87.937 / 85.428 | 87.059 / 83.560 | 86.555 / 83.596 | 86.760 / 83.792 | 86.312 / 82.119 |

## E1: selected supports

| Dataset | Condition | SC | VAL | SC-VAL validation pp |
|---|---|---|---|---|
| cora | clean | 8 | 64 | -0.741 |
| cora | uniform0.4 | 64 | 8 | -2.593 |
| cora | uniform0.6 | 16 | 32 | -5.926 |
| cora | pair0.4 | 8 | 64 | -5.185 |
| corafull | clean | 8 | 16 | -0.556 |
| corafull | uniform0.4 | 16 | 16 | 0.000 |
| corafull | uniform0.6 | 32 | 16 | -0.303 |
| corafull | pair0.4 | 8 | 16 | -1.112 |
| amazonpho | clean | 8 | 8 | 0.000 |
| amazonpho | uniform0.4 | 16 | 32 | -1.046 |
| amazonpho | uniform0.6 | 32 | 4 | -2.484 |
| amazonpho | pair0.4 | 128 | 128 | 0.000 |

## E2: mean test accuracy, all four conditions and both heads

| Dataset | 100 | 300 | 600 |
|---|---|---|---|
| cora | 71.102 | 70.640 | 69.063 |
| coauthorcs | 89.819 | 88.194 | 88.055 |
| amazonpho | 78.896 | 78.954 | 79.967 |

## E4: noisy accuracy / macro-F1

| Dataset | A200 | A300 | B | D | C_full | C_head |
|---|---|---|---|---|---|---|
| cora | 76.183 / 70.506 | 76.183 / 70.506 | 75.796 / 70.079 | 74.881 / 69.527 | 49.903 / 43.909 | 69.798 / 65.663 |
| citeseer | 60.767 / 51.886 | 61.412 / 52.608 | 62.257 / 53.235 | 63.989 / 54.230 | 48.697 / 45.630 | 63.187 / 55.923 |
| corafull | 54.601 / 37.642 | 54.666 / 37.655 | 54.397 / 37.165 | 45.995 / 27.372 | 34.743 / 23.153 | 45.358 / 29.843 |
| coauthorcs | 92.088 / 87.672 | 92.127 / 87.789 | 92.434 / 88.519 | 90.797 / 86.085 | 88.797 / 82.617 | 89.992 / 83.727 |
| amazonpho | 86.598 / 75.319 | 87.289 / 76.212 | 86.723 / 76.265 | 80.871 / 69.774 | 74.846 / 66.533 | 79.627 / 72.650 |

## E4: clean accuracy

| Dataset | A200 | A300 | B | D | C_full | C_head |
|---|---|---|---|---|---|---|
| cora | 85.119 | 85.119 | 85.119 | 82.691 | 65.435 | 80.739 |
| citeseer | 73.883 | 73.883 | 74.570 | 72.637 | 66.624 | 72.165 |
| corafull | 62.685 | 62.685 | 63.327 | 51.122 | 47.311 | 51.534 |
| coauthorcs | 94.109 | 94.109 | 93.852 | 93.805 | 91.304 | 93.602 |
| amazonpho | 91.148 | 91.839 | 92.026 | 86.331 | 86.200 | 86.555 |

## E4: B versus validation-selected comparator (test deltas)

| Dataset | Noisy accuracy delta pp | Noisy macro-F1 delta pp | Clean accuracy delta pp |
|---|---|---|---|
| cora | -0.387 | -0.427 | 0.000 |
| citeseer | -0.931 | -2.688 | 0.687 |
| corafull | -0.269 | -0.490 | 0.642 |
| coauthorcs | 0.307 | 0.730 | -0.257 |
| amazonpho | -0.566 | 0.053 | 0.187 |

## Scope and audit

E1 144 fits; E2 72 heads plus 3 alignment trajectories; E4 120 student fits plus 5 new clean GCN teachers. Four 3-epoch smoke fits are excluded from all result tables.
No seed1/2 fit expansion passed. E3/E5 were not launched. Cached noisy teachers are ordinary SUPPORT-LCSS GCNs, not ERASE; KD is all-node soft cross-entropy, not the DIST paper loss.
Two analysis-only implementation errors were repaired after the first test release; original logs/source and locked selections were preserved. See ANALYSIS_AMENDMENT.md.
