# MLP-42 Table 4 Breadth Report

Protocol: Coauthor-CS/Amazon-Photo, train-only uniform 0.4/0.6, percent 0.2/0.1/0.7, seeds 0-4.
The kNN teacher uses the same vote-matched diagnostic definition as the frozen Table 4; k is not selected by validation/test accuracy.

## Uniform 0.4

| Teacher | Coauthor-CS | Amazon-Photo |
|---|---:|---:|
| GCN(X)+CE (GLNN-style) | 93.20+-0.09 | 89.86+-0.77 |
| GCN[X||H]+CE | 93.35+-0.55 | 90.51+-1.08 |
| Shuffled-consensus teacher | 93.11+-0.39 | 89.29+-0.26 |
| Vote-matched kNN-consensus teacher | 93.69+-0.43 | 90.15+-0.93 |
| Ball-consensus teacher | 93.75+-0.22 | 90.49+-1.05 |

## Uniform 0.6

| Teacher | Coauthor-CS | Amazon-Photo |
|---|---:|---:|
| GCN(X)+CE (GLNN-style) | 91.13+-0.58 | 84.80+-1.48 |
| GCN[X||H]+CE | 91.16+-1.32 | 85.10+-3.27 |
| Shuffled-consensus teacher | 91.23+-0.50 | 80.11+-9.78 |
| Vote-matched kNN-consensus teacher | 91.66+-0.90 | 86.82+-0.61 |
| Ball-consensus teacher | 92.25+-0.69 | 85.87+-2.25 |

## Paired attribution

| Dataset | Noise | Ball-GCN(X) | Ball-GCN[X||H] | Ball-shuffled | Ball-kNN |
|---|---:|---:|---:|---:|---:|
| coauthorcs | 0.4 | +0.55pp (5/5) | +0.40pp (4/5) | +0.64pp (5/5) | +0.06pp (3/5) |
| coauthorcs | 0.6 | +1.12pp (5/5) | +1.09pp (5/5) | +1.02pp (5/5) | +0.59pp (4/5) |
| amazonpho | 0.4 | +0.63pp (4/5) | -0.02pp (2/5) | +1.20pp (4/5) | +0.34pp (4/5) |
| amazonpho | 0.6 | +1.07pp (2/5) | +0.77pp (3/5) | +5.76pp (4/5) | -0.95pp (1/5) |

## Vote-matched k audit

- coauthorcs: [57]
- amazonpho: [24]

## Interpretation boundary

This is an exploratory breadth extension. It does not modify the frozen v15 manuscript unless the author later elects to revise Table 4. Vote-matched kNN controls effective consensus size and must not be described as an accuracy-tuned kNN method.
