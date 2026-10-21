# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `cora`
- Seeds: `0`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cora | 1 | 0.3494 | 52.00 | 0.7873 | 0.8879 | 0.9383 | 0.7844 | 0.9080 | 0.6910 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| cora | co_teaching_intersection | keep_0.5_each | 1 | 0.0185 | 0.3308 | 0.3993 |  |
| cora | co_teaching_intersection | keep_0.6_each | 1 | 0.0366 | 0.3127 | 0.5046 |  |
| cora | co_teaching_intersection | keep_0.7_each | 1 | 0.0785 | 0.2708 | 0.6118 |  |
| cora | filter | top_0.2 | 1 | 0.2431 | 0.1063 | 0.7985 | 0.7706 |
| cora | filter | top_0.3 | 1 | 0.1453 | 0.2041 | 0.6617 | 0.7486 |
| cora | filter | top_0.4 | 1 | 0.1013 | 0.2481 | 0.5841 | 0.6978 |
| cora | reweight | 1_minus_s_ball | 1 | 0.1603 | 0.1891 | 1.0000 |  |

## Decision

No big graph passes the conditional orthogonality gate. Ball suspicion may have marginal signal, but it does not provide enough information inside the GCN-clean subset to justify a big-graph training variant.
Small-graph mean conditional ball AUC: `0.7844`; big-graph mean conditional ball AUC: `nan`.
