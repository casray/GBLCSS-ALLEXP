# Phase 33: Big-Graph Ball Orthogonality Diagnostic

## Protocol

- Datasets: `coauthorcs`
- Seeds: `0,1,2`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Warmup GCN: `10` epochs, probability window `5-10`
- Clean labels are used only for diagnostic metrics.
- No downstream model is trained by this script.

## Phase 33a Summary

| Dataset | n | train noise | balls | purity | ball AUC | GCN AUC | ball AUC in GCN-clean50 | GCN AUC in ball-clean50 | corr(ball,GCN) | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coauthorcs | 3 | 0.4010 | 64.00 | 0.8939 | 0.9669 | 0.9942 | 0.8420 | 0.9126 | 0.8241 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| coauthorcs | co_teaching_intersection | keep_0.5_each | 3 | 0.0014 | 0.3996 | 0.4522 |  |
| coauthorcs | co_teaching_intersection | keep_0.6_each | 3 | 0.0112 | 0.3898 | 0.5618 |  |
| coauthorcs | co_teaching_intersection | keep_0.7_each | 3 | 0.0793 | 0.3217 | 0.6242 |  |
| coauthorcs | filter | top_0.2 | 3 | 0.2646 | 0.1363 | 0.7958 | 0.9323 |
| coauthorcs | filter | top_0.3 | 3 | 0.1764 | 0.2246 | 0.6983 | 0.9207 |
| coauthorcs | filter | top_0.4 | 3 | 0.0659 | 0.3350 | 0.5986 | 0.9005 |
| coauthorcs | reweight | 1_minus_s_ball | 3 | 0.0671 | 0.3338 | 1.0000 |  |

## Decision

At least one big graph passes the pre-registered conditional orthogonality gate. Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above.
