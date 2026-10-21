# MLP-16 Distill Table B Completion Report

Protocol: uniform 0.4, train-only noise, percent split 20/10/70, seeds 0-4, ROCm cuda:0.

## Completeness
- Formal rows completed: 50/50 status=ok.
- New datasets: Coauthor-CS, Amazon-Photo.
- Note: `run_standard_downstream.py` output for these runs records final student train/valid/test, but not teacher_test columns; Table B uses student test accuracy, so no teacher metric was fabricated.

## New 5-seed Aggregates
| Dataset | T1 GLNN | T2 GCN[X||H] | T3 Ball teacher | T4 kNN teacher | T5 Shuffled |
|---|---:|---:|---:|---:|---:|
| coauthorcs | 93.20±0.09 | 93.35±0.55 | 93.75±0.22 | 93.69±0.43 | 93.11±0.39 |
| amazonpho | 89.86±0.77 | 90.51±1.08 | 90.49±1.05 | 90.15±0.93 | 89.29±0.26 |

## Paired Deltas (mean, percentage points)
| Dataset | T3-T1 | T3-T2 | T3-T4 | T3-T5 | T4-T2 | T2-T1 |
|---|---:|---:|---:|---:|---:|---:|
| coauthorcs | +0.55 (5/5) | +0.40 (4/5) | +0.06 (3/5) | +0.64 (5/5) | +0.34 (4/5) | +0.16 (3/5) |
| amazonpho | +0.63 (4/5) | -0.02 (2/5) | +0.34 (4/5) | +1.20 (4/5) | -0.37 (2/5) | +0.65 (5/5) |

## Appendix Table B Distill Rows (Five Datasets)
| Method | Cora | CiteSeer | PubMed | Coauthor-CS | Amazon-Photo |
|---|---:|---:|---:|---:|---:|
| GLNN-style GCN teacher | 77.95±2.62 | 66.09±1.38 | 84.32±0.47 | 93.20±0.09 | 89.86±0.77 |
| GLNN-style GCN[X||H] teacher | 80.23±1.57 | 68.81±1.12 | 84.80±0.85 | 93.35±0.55 | 90.51±1.08 |
| LCSS-distill (T3 ball teacher) | 80.60±0.69 | 69.40±1.08 | 85.01±0.93 | 93.75±0.22 | 90.49±1.05 |
| kNN-consensus distill (T4) | 80.99±1.02 | 69.57±1.46 | 85.23±0.94 | 93.69±0.43 | 90.15±0.93 |
| Shuffled-consensus distill (T5) | 78.81±1.63 | 68.60±1.14 | 85.08±0.93 | 93.11±0.39 | 89.29±0.26 |

## Main Conclusions
- Coauthor-CS: ball teacher is best among five distillation arms (93.75±0.22), slightly above kNN teacher (93.69±0.43) and clearly above shuffled (93.11±0.39).
- Amazon-Photo: GCN[X||H] teacher is best (90.51±1.08); ball teacher (90.49±1.05) is essentially tied with it and higher than kNN teacher (90.15±0.93), but shuffled is lower (89.29±0.26).
- The new runs strengthen the appendix statement: ball-style and kNN-style consensus are close, but on these two larger datasets ball is not worse than kNN in the distillation role.