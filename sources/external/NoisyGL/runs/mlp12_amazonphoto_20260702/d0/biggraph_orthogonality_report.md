# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `amazonpho`
- Seeds: `0`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | homophily | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazonpho | 1 | 0.8272 | 0.4026 | 133.00 | 0.8716 | 0.9468 | 0.8294 | 0.9407 | 0.7464 | 0.5004 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| amazonpho | co_teaching_intersection | keep_0.5_each | 1 | 0.0144 | 0.3882 | 0.3627 |  |
| amazonpho | co_teaching_intersection | keep_0.6_each | 1 | 0.0570 | 0.3456 | 0.4471 |  |
| amazonpho | co_teaching_intersection | keep_0.7_each | 1 | 0.1066 | 0.2961 | 0.5582 |  |
| amazonpho | filter | top_0.2 | 1 | 0.2489 | 0.1537 | 0.7667 | 0.9076 |
| amazonpho | filter | top_0.3 | 1 | 0.1702 | 0.2324 | 0.6758 | 0.8871 |
| amazonpho | filter | top_0.4 | 1 | 0.0752 | 0.3274 | 0.5739 | 0.8436 |
| amazonpho | reweight | 1_minus_s_ball | 1 | 0.1590 | 0.2436 | 1.0000 |  |

## Decision

At least one big graph passes the pre-registered conditional orthogonality gate. Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above.
