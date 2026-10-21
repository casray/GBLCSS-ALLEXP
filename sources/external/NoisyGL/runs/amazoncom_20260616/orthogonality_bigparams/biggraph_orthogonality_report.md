# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `amazoncom`
- Seeds: `0,1,2`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazoncom | 3 | 0.3976 | 237.33 | 0.7850 | 0.9273 | 0.7965 | 0.9219 | 0.7233 | 0.4610 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| amazoncom | co_teaching_intersection | keep_0.5_each | 3 | 0.0258 | 0.3718 | 0.3598 |  |
| amazoncom | co_teaching_intersection | keep_0.6_each | 3 | 0.0692 | 0.3283 | 0.4530 |  |
| amazoncom | co_teaching_intersection | keep_0.7_each | 3 | 0.1420 | 0.2556 | 0.5479 |  |
| amazoncom | filter | top_0.2 | 3 | 0.2625 | 0.1351 | 0.7833 | 0.8862 |
| amazoncom | filter | top_0.3 | 3 | 0.1935 | 0.2040 | 0.6945 | 0.8614 |
| amazoncom | filter | top_0.4 | 3 | 0.1040 | 0.2936 | 0.5914 | 0.8226 |
| amazoncom | reweight | 1_minus_s_ball | 3 | 0.1634 | 0.2342 | 1.0000 |  |

## Decision

At least one big graph passes the pre-registered conditional orthogonality gate. Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above.
