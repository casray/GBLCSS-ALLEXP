# Phase 21 Ball Count Sensitivity

## Protocol

- Methods: `frozengrace_gb_gcn_softdist_xh,frozengrace_gb_gcn_softdist_xh_shuffled`
- Datasets: `cora,citeseer`
- Seeds: `0,1,2,3,4`
- max_balls: `16,32,64,128`
- min_ball_size: fixed `32`
- Noise: `uniform` `0.4`
- Split: percent 0.2 / 0.1 / 0.7
- Mechanism: SoftDist [X||H] anchor unchanged; only `gb.max_balls` is overridden in memory.

## Summary

| Dataset | Method | max_balls | Test mean | Test std | Actual balls | Mean train/ball | NCR | CDR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| citeseer | frozengrace_gb_gcn_softdist_xh | 16 | 0.6892 | 0.0142 | 16.00 | 41.56 | 0.6379 | 0.3256 |
| citeseer | frozengrace_gb_gcn_softdist_xh | 32 | 0.6923 | 0.0129 | 32.00 | 20.78 | 0.6533 | 0.2972 |
| citeseer | frozengrace_gb_gcn_softdist_xh | 64 | 0.6891 | 0.0160 | 63.80 | 10.42 | 0.5826 | 0.2996 |
| citeseer | frozengrace_gb_gcn_softdist_xh | 128 | 0.6901 | 0.0150 | 64.60 | 10.30 | 0.5826 | 0.2996 |
| citeseer | frozengrace_gb_gcn_softdist_xh_shuffled | 16 | 0.6723 | 0.0184 | 16.00 | 41.56 | 0.1875 | 0.7295 |
| citeseer | frozengrace_gb_gcn_softdist_xh_shuffled | 32 | 0.6688 | 0.0200 | 32.00 | 20.78 | 0.1809 | 0.7013 |
| citeseer | frozengrace_gb_gcn_softdist_xh_shuffled | 64 | 0.6696 | 0.0234 | 63.80 | 10.42 | 0.1465 | 0.6615 |
| citeseer | frozengrace_gb_gcn_softdist_xh_shuffled | 128 | 0.6698 | 0.0235 | 64.60 | 10.30 | 0.1481 | 0.6666 |
| cora | frozengrace_gb_gcn_softdist_xh | 16 | 0.8132 | 0.0065 | 16.00 | 33.81 | 0.7006 | 0.2581 |
| cora | frozengrace_gb_gcn_softdist_xh | 32 | 0.8132 | 0.0109 | 32.00 | 16.91 | 0.7171 | 0.2185 |
| cora | frozengrace_gb_gcn_softdist_xh | 64 | 0.8137 | 0.0067 | 52.40 | 10.33 | 0.6871 | 0.2018 |
| cora | frozengrace_gb_gcn_softdist_xh | 128 | 0.8137 | 0.0067 | 52.40 | 10.33 | 0.6871 | 0.2018 |
| cora | frozengrace_gb_gcn_softdist_xh_shuffled | 16 | 0.7987 | 0.0042 | 16.00 | 33.81 | 0.2209 | 0.7042 |
| cora | frozengrace_gb_gcn_softdist_xh_shuffled | 32 | 0.7954 | 0.0087 | 32.00 | 16.91 | 0.1835 | 0.6686 |
| cora | frozengrace_gb_gcn_softdist_xh_shuffled | 64 | 0.7921 | 0.0097 | 52.40 | 10.36 | 0.1554 | 0.6574 |
| cora | frozengrace_gb_gcn_softdist_xh_shuffled | 128 | 0.7921 | 0.0097 | 52.40 | 10.36 | 0.1554 | 0.6574 |

## Decision

- Directly supported: `max_balls=64` is not a cherry-picked point. On Cora, real SoftDist [X||H] stays in a very narrow band from 0.8132 to 0.8137 across `max_balls={16,32,64,128}`. On CiteSeer, the corresponding band is 0.6891 to 0.6923.
- Directly supported: real balls outperform shuffled balls for every tested `max_balls` on both datasets.
- Interpretation: with fixed `min_ball_size=32`, increasing `max_balls` past 64 saturates the geometric splitter rather than producing many additional balls. The effective actual ball counts are about 52 on Cora and 64-65 on CiteSeer.
- Paper use: report this as a sensitivity curve plus one sentence: the default `max_balls=64` lies on a stable plateau, not at an isolated tuned optimum.
