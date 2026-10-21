# Phase 41: GB Clustering Feasibility

Decision: **NO-GO**

## Protocol

- Datasets: `cora,citeseer,pubmed`; seeds: `0,1,2,3,4`.
- Split improvements: `0.02,0.05,0.1`; main split: `0.05`.
- No label noise and no train/val/test split is used for clustering decisions; labels are evaluation-only.
- Ball graph k: `10`; edge density weight: `0.0`.

## Gate Summary

| Dataset | K true | K auto | Balls | GB ACC | KMeans trueK ACC | ACC gap | GB NMI | KMeans trueK NMI | NMI ratio | Wrong-K ACC adv | G1 | G2 | G3 | G4 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| citeseer | 6 | 4.40 | 63.80 | 0.5342 | 0.6395 | -0.1053 | 0.2942 | 0.3782 | 0.7779 | +0.0240 | 0 | 0 | 1 | 0 | FAIL |
| cora | 7 | 3.00 | 52.40 | 0.4823 | 0.6317 | -0.1493 | 0.3404 | 0.5193 | 0.6555 | -0.1208 | 0 | 0 | 0 | 0 | FAIL |
| pubmed | 3 | 3.40 | 64.00 | 0.5248 | 0.6766 | -0.1518 | 0.1918 | 0.3107 | 0.6172 | +0.0096 | 0 | 0 | 1 | 0 | FAIL |

## Metric Summary

| Dataset | Split | Method | ACC | NMI | ARI | K auto | Balls |
|---|---:|---|---:|---:|---:|---:|---:|
| citeseer | 0.05 | gb_cluster | 0.5342±0.0285 | 0.2942±0.0319 | 0.2770±0.0371 | 4.40 | 63.80 |
| citeseer | 0.05 | kmeans_H_autoK | 0.6204±0.0449 | 0.3520±0.0371 | 0.3569±0.0385 | 4.40 | 63.80 |
| citeseer | 0.05 | kmeans_H_trueK | 0.6395±0.0009 | 0.3782±0.0009 | 0.3808±0.0013 | 4.40 | 63.80 |
| citeseer | 0.05 | spectral_H_trueK | 0.6658±0.0000 | 0.4113±0.0000 | 0.4205±0.0000 | 4.40 | 63.80 |
| cora | 0.05 | gb_cluster | 0.4823±0.0119 | 0.3404±0.0059 | 0.2889±0.0102 | 3.00 | 52.40 |
| cora | 0.05 | kmeans_H_autoK | 0.4923±0.0002 | 0.3609±0.0004 | 0.2938±0.0002 | 3.00 | 52.40 |
| cora | 0.05 | kmeans_H_trueK | 0.6317±0.0014 | 0.5193±0.0018 | 0.4066±0.0016 | 3.00 | 52.40 |
| cora | 0.05 | spectral_H_trueK | 0.6041±0.0000 | 0.4733±0.0000 | 0.2836±0.0000 | 3.00 | 52.40 |
| pubmed | 0.05 | gb_cluster | 0.5248±0.1020 | 0.1918±0.0566 | 0.1596±0.0761 | 3.40 | 64.00 |
| pubmed | 0.05 | kmeans_H_autoK | 0.6013±0.1032 | 0.2863±0.0335 | 0.2451±0.0588 | 3.40 | 64.00 |
| pubmed | 0.05 | kmeans_H_trueK | 0.6766±0.0000 | 0.3107±0.0000 | 0.2880±0.0000 | 3.40 | 64.00 |

## Wrong-K Robustness

| Dataset | Split | GB ACC | Wrong-K KMeans ACC | ACC advantage | NMI advantage |
|---|---:|---:|---:|---:|---:|
| citeseer | 0.05 | 0.5342 | 0.5103 | +0.0240±0.0603 | -0.0304 |
| cora | 0.05 | 0.4823 | 0.6031 | -0.1208±0.0243 | -0.1630 |
| pubmed | 0.05 | 0.5248 | 0.5153 | +0.0096±0.0973 | -0.0898 |

## Decision Notes

- NO-GO: G1 and/or G2 did not pass on at least 2/3 evaluated datasets.
- G1 uses `NMI >= 90% of kmeans_H_trueK` and `ACC gap >= -5pp`.
- G2 uses mean ACC advantage over mis-specified-K k-means >= +5pp.
- G3/G4 are diagnostic gates for K_auto reasonableness and nontrivial ball agglomeration.
