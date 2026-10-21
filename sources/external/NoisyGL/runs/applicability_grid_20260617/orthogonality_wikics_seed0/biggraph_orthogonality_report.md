# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `wikics`
- Seeds: `0`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| wikics | 1 | 0.4077 | 64.00 | 0.2724 | 0.6590 | 0.8587 | 0.5282 | 0.8677 | 0.3035 | redundant / no orthogonal gain |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| wikics | co_teaching_intersection | keep_0.5_each | 1 | 0.1192 | 0.2885 | 0.3658 |  |
| wikics | co_teaching_intersection | keep_0.6_each | 1 | 0.1807 | 0.2270 | 0.4684 |  |
| wikics | co_teaching_intersection | keep_0.7_each | 1 | 0.2318 | 0.1759 | 0.5825 |  |
| wikics | filter | top_0.2 | 1 | 0.3549 | 0.0528 | 0.7996 | 0.6183 |
| wikics | filter | top_0.3 | 1 | 0.3252 | 0.0825 | 0.6979 | 0.5983 |
| wikics | filter | top_0.4 | 1 | 0.3096 | 0.0981 | 0.5991 | 0.5544 |
| wikics | reweight | 1_minus_s_ball | 1 | 0.3423 | 0.0654 | 1.0000 |  |

## Decision

No big graph passes the conditional orthogonality gate. Ball suspicion may have marginal signal, but it does not provide enough information inside the GCN-clean subset to justify a big-graph training variant.
