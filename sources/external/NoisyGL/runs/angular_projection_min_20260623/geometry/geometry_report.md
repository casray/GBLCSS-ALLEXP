# Phase 20a Embedding Geometry Diagnostic

Gate uses only geometry: non-collapsed ball count and true weighted ball purity.
Probe accuracy is reported as reference only and is not used for pass/fail.

## Summary

| Dataset | Method | Num balls | Uniformity | True all purity | Clean probe | Noisy probe |
|---|---|---:|---:|---:|---:|---:|
| citeseer | grace | 64.0000 | -1.5480 | 0.6991 | 0.7109 | 0.6833 |
| citeseer | grace_angproj_knn10_anchor1 | 61.3333 | -2.5276 | 0.6881 | 0.7002 | 0.6830 |
| cora | grace | 52.3333 | -1.0107 | 0.7871 | 0.8294 | 0.7819 |
| cora | grace_angproj_knn10_anchor1 | 47.3333 | -2.6074 | 0.7741 | 0.8313 | 0.7791 |

## Gate Decisions

| Dataset | Method | Num balls OK | Purity OK | Pass |
|---|---|---:|---:|---:|
| citeseer | grace_angproj_knn10_anchor1 | 1 | 0 | 0 |
| cora | grace_angproj_knn10_anchor1 | 1 | 0 | 0 |
