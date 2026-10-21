# MLP-43 Existing Tie-Safe Asset Audit

Date: 2026-07-29

## Scope

This audit determines which manuscript results already have complete tie-safe v2
artifacts and therefore do not require new training. The accepted target rule is
`tie_mode=distribution_only`.

## Completeness and protocol checks

| Artifact | Rows | Successful | Seeds | Noise scope | Val/test modified rows |
|---|---:|---:|---|---|---:|
| `runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv` | 60 | 60 | 0-4 | train | 0 |
| `runs/v31_tiesafe_affected_20260728/smoothing_uniform0p4.csv` | 60 | 60 | 0-4 | train | 0 |
| `runs/v31_tiesafe_affected_20260728/breadth_uniform0p4.csv` | 60 | 60 | 0-4 | train | 0 |

All rows use uniform noise at rate 0.4. No failed, NaN, or leakage rows were
found.

## Recomputed five-seed aggregates

### LOO and LOO-shuffled

| Dataset | LOO | LOO-shuffled |
|---|---:|---:|
| Cora | 57.49 +/- 0.76 | 43.31 +/- 1.40 |
| CiteSeer | 55.24 +/- 2.04 | 42.08 +/- 2.88 |
| CoraFull | 37.39 +/- 0.51 | 32.30 +/- 0.85 |
| Coauthor-CS | 90.71 +/- 0.59 | 87.46 +/- 1.07 |
| Amazon-Photo | 82.12 +/- 1.08 | 71.45 +/- 3.29 |
| PubMed | 72.56 +/- 0.64 | 62.77 +/- 4.59 |

### Explicit smoothing controls

| Dataset | Uniform smoothing | Prior smoothing |
|---|---:|---:|
| Cora | 44.69 +/- 2.90 | 44.89 +/- 2.38 |
| CiteSeer | 43.69 +/- 3.31 | 43.43 +/- 2.99 |
| CoraFull | 32.59 +/- 1.12 | 32.63 +/- 0.84 |
| Coauthor-CS | 88.25 +/- 0.80 | 88.04 +/- 0.73 |
| Amazon-Photo | 74.18 +/- 1.72 | 74.72 +/- 1.44 |
| PubMed | 63.67 +/- 5.08 | 64.25 +/- 4.52 |

### Six supplementary datasets

| Dataset | GB-LCSS tie-safe v2 | Shuffled |
|---|---:|---:|
| PubMed | 72.03 +/- 0.44 | 62.95 +/- 4.28 |
| Amazon-Computers | 76.12 +/- 0.62 | 70.38 +/- 1.64 |
| Cora_ML | 55.89 +/- 3.00 | 49.08 +/- 3.19 |
| Attributed-Wiki | 65.25 +/- 1.49 | 60.17 +/- 1.77 |
| CiteSeerFull | 59.56 +/- 1.54 | 47.48 +/- 2.82 |
| Coauthor-Physics | 93.11 +/- 0.26 | 88.91 +/- 2.08 |

## Manuscript disposition

- Appendix LOO, smoothing, and supplementary-dataset tables should be replaced
  from the audited CSVs above. No new training is needed.
- In the H-access table, only the GB-LCSS row changes to the tie-safe main-grid
  values: Cora `56.28 +/- 1.54`, CiteSeer `53.52 +/- 2.71`.
- H-only, `X||H`+CE, Small-loss, and JoCoR do not construct GB targets and are
  not affected by the tie rule.
- The consensus-teacher experiment uses the complete ball label distribution
  with `target_beta=0`. Hence the majority one-hot branch, including its tie
  handling, is inactive. Those rows are mathematically unchanged.

## Remaining current-rule work

- Corrected kNN targets must explicitly exclude the query node and use
  `distribution_only` on ties.
- k-means grouping alternatives must use the same tie-safe target rule.
- The archived GRACE-seed study and representation-source comparison require
  current-rule refreshes before their numerical claims can be retained.

