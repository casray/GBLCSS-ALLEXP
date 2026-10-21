# MLP-8 P2 Ball-Parameter Rescue Diagnostic

## Protocol

- Seed: 0
- Noise: uniform 0.4, train-only
- Split: percent 0.2/0.1/0.7
- Scope: geometry diagnostic only; no downstream retraining gate is claimed here.
- Gate: candidate rescued only if ball count recovers to at least 75% of frozen-GRACE default and true-all purity is within 0.05 of frozen-GRACE default.

## Best Candidate Per P2 Failure Mode

| Phase | Dataset | Variant | Best param | Num balls | True all purity | Mean train/ball | Alpha+ train | Gate |
|---|---|---|---|---:|---:|---:|---:|---|
| p2a | cora | anchor_grace | 64/32/0.05 | 52 | 0.7873 | 10.40 | 541 | anchor |
| p2a | cora | grace_ce_l0p1 | 64/16/0 | 64 | 0.8117 | 8.69 | 539 | pass |
| p2b | cora | F0_graceH | 64/32/0.05 | 52 | 0.7873 | 10.40 | 541 | anchor |
| p2b | cora | F1_loop_Z1 | 64/32/0 | 58 | 0.5953 | 11.38 | 535 | fail |
| p2b | citeseer | F0_graceH | 64/32/0.05 | 64 | 0.6991 | 10.39 | 665 | anchor |
| p2b | citeseer | F1_loop_Z1 | 64/16/0 | 64 | 0.5765 | 18.42 | 663 | fail |

## Gate Decisions

| Phase | Dataset | Candidate | Param | Balls vs anchor | Purity delta | Count ok | Purity ok | Gate |
|---|---|---|---|---:|---:|---|---|---|
| p2a | cora | grace_ce_l0p1 | 64/16/0 | 64/52 | +0.0244 | True | True | pass |
| p2b | cora | F1_loop_Z1 | 64/32/0 | 58/52 | -0.1920 | True | False | fail |
| p2b | citeseer | F1_loop_Z1 | 64/16/0 | 64/64 | -0.1226 | True | False | fail |

## Decision

- 1/3 candidate cases pass the geometry rescue gate. These cases may be considered for a separate downstream seed0 check.
- If ball count recovers without true purity recovery, interpret the original P2 failure as representation-level contamination, not merely split-threshold collapse.

## Files

- `p2_ballparam_raw.csv`: full parameter grid.
- `p2_ballparam_best.csv`: anchors plus best candidate rows.
- `p2_ballparam_decisions.csv`: gate decisions.
