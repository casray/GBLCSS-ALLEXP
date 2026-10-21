# Phase 41: GB Clustering Feasibility

Decision: **NO-GO**

## Protocol

- Datasets: `cora`; seeds: `0`.
- Split improvements: `0.05`; main split: `0.05`.
- No label noise and no train/val/test split is used for clustering decisions; labels are evaluation-only.
- Ball graph k: `10`; edge density weight: `0.0`.

## Gate Summary

| Dataset | K true | K auto | Balls | GB ACC | KMeans trueK ACC | ACC gap | GB NMI | KMeans trueK NMI | NMI ratio | Wrong-K ACC adv | G1 | G2 | G3 | G4 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cora | 7 | 3.00 | 52.00 | 0.4978 | 0.6304 | -0.1326 | 0.3472 | 0.5209 | 0.6666 | -0.1011 | 0 | 0 | 0 | 1 | FAIL |

## Metric Summary

| Dataset | Split | Method | ACC | NMI | ARI | K auto | Balls |
|---|---:|---|---:|---:|---:|---:|---:|
| cora | 0.05 | gb_cluster | 0.4978±0.0000 | 0.3472±0.0000 | 0.3030±0.0000 | 3.00 | 52.00 |
| cora | 0.05 | kmeans_H_autoK | 0.4926±0.0000 | 0.3615±0.0000 | 0.2942±0.0000 | 3.00 | 52.00 |
| cora | 0.05 | kmeans_H_trueK | 0.6304±0.0000 | 0.5209±0.0000 | 0.4079±0.0000 | 3.00 | 52.00 |
| cora | 0.05 | spectral_H_trueK | 0.6041±0.0000 | 0.4733±0.0000 | 0.2836±0.0000 | 3.00 | 52.00 |

## Wrong-K Robustness

| Dataset | Split | GB ACC | Wrong-K KMeans ACC | ACC advantage | NMI advantage |
|---|---:|---:|---:|---:|---:|
| cora | 0.05 | 0.4978 | 0.5989 | -0.1011±0.0278 | -0.1526 |

## Decision Notes

- NO-GO: G1 and/or G2 did not pass on at least 2/3 evaluated datasets.
- G1 uses `NMI >= 90% of kmeans_H_trueK` and `ACC gap >= -5pp`.
- G2 uses mean ACC advantage over mis-specified-K k-means >= +5pp.
- G3/G4 are diagnostic gates for K_auto reasonableness and nontrivial ball agglomeration.
