# Phase 20a Embedding Geometry Diagnostic

Gate uses only geometry: non-collapsed ball count and true weighted ball purity.
Probe accuracy is reported as reference only and is not used for pass/fail.

## Summary

| Dataset | Method | Num balls | Uniformity | True all purity | Clean probe | Noisy probe |
|---|---|---:|---:|---:|---:|---:|
| citeseer | grace | 64.0000 | -1.5480 | 0.6991 | 0.7109 | 0.7015 |
| citeseer | grace_unif0p1 | 1.0000 | -3.6773 | 0.2107 | 0.6957 | 0.6534 |
| citeseer | grace_unif0p5 | 1.0000 | -3.7903 | 0.2107 | 0.7006 | 0.6143 |
| citeseer | grace_unif1 | 1.0000 | -3.8045 | 0.2107 | 0.6927 | 0.5968 |
| cora | grace | 52.3333 | -1.0107 | 0.7871 | 0.8294 | 0.7949 |
| cora | grace_unif0p1 | 1.0000 | -3.3627 | 0.3021 | 0.8303 | 0.7840 |
| cora | grace_unif0p5 | 1.0000 | -3.5755 | 0.3021 | 0.8218 | 0.7692 |
| cora | grace_unif1 | 1.0000 | -3.6397 | 0.3021 | 0.8218 | 0.7602 |

## Gate Decisions

| Dataset | Method | Num balls OK | Purity OK | Pass |
|---|---|---:|---:|---:|
| citeseer | grace_unif0p1 | 0 | 0 | 0 |
| citeseer | grace_unif0p5 | 0 | 0 | 0 |
| citeseer | grace_unif1 | 0 | 0 | 0 |
| cora | grace_unif0p1 | 0 | 0 | 0 |
| cora | grace_unif0p5 | 0 | 0 | 0 |
| cora | grace_unif1 | 0 | 0 | 0 |
