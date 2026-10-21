# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `lastfmasia`
- Seeds: `0,1,2`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | homophily | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lastfmasia | 3 | 0.8739 | 0.4014 | 145.33 | 0.2542 | 0.6399 | 0.9514 | 0.5995 | 0.9604 | 0.2264 | weak or unstable |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| lastfmasia | co_teaching_intersection | keep_0.5_each | 3 | 0.0221 | 0.3792 | 0.3248 |  |
| lastfmasia | co_teaching_intersection | keep_0.6_each | 3 | 0.0790 | 0.3224 | 0.4195 |  |
| lastfmasia | co_teaching_intersection | keep_0.7_each | 3 | 0.1684 | 0.2330 | 0.5427 |  |
| lastfmasia | filter | top_0.2 | 3 | 0.3576 | 0.0437 | 0.7787 | 0.5534 |
| lastfmasia | filter | top_0.3 | 3 | 0.3356 | 0.0658 | 0.6680 | 0.5322 |
| lastfmasia | filter | top_0.4 | 3 | 0.3144 | 0.0869 | 0.5866 | 0.5248 |
| lastfmasia | reweight | 1_minus_s_ball | 3 | 0.3358 | 0.0655 | 1.0000 |  |

## Decision

No big graph passes the conditional orthogonality gate. Ball suspicion may have marginal signal, but it does not provide enough information inside the GCN-clean subset to justify a big-graph training variant.
