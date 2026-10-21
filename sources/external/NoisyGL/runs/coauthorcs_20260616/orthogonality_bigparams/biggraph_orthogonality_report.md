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
| coauthorcs | 3 | 0.4010 | 256.00 | 0.9052 | 0.9669 | 0.9942 | 0.8533 | 0.9328 | 0.7936 | possible orthogonal signal |

## Phase 33b Offline Usage Summary

| Dataset | Usage | Param | n | effective noise | reduction | retention | noise precision in removed set |
|---|---|---|---:|---:|---:|---:|---:|
| coauthorcs | co_teaching_intersection | keep_0.5_each | 3 | 0.0014 | 0.3996 | 0.4493 |  |
| coauthorcs | co_teaching_intersection | keep_0.6_each | 3 | 0.0107 | 0.3903 | 0.5610 |  |
| coauthorcs | co_teaching_intersection | keep_0.7_each | 3 | 0.0728 | 0.3282 | 0.6213 |  |
| coauthorcs | filter | top_0.2 | 3 | 0.2603 | 0.1407 | 0.7915 | 0.9343 |
| coauthorcs | filter | top_0.3 | 3 | 0.1682 | 0.2328 | 0.6903 | 0.9199 |
| coauthorcs | filter | top_0.4 | 3 | 0.0606 | 0.3404 | 0.5934 | 0.8977 |
| coauthorcs | reweight | 1_minus_s_ball | 3 | 0.1117 | 0.2893 | 1.0000 |  |

## Decision

At least one big graph passes the pre-registered conditional orthogonality gate. Phase 34 should be considered only for the passing dataset(s), using the best offline usage mode above.
