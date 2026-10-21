# FeatureGB-MLP Offline Diagnostic

- Ball features: raw X, dense float row-L2 normalized.
- Parameters: max_balls=64, min_ball_size=32, split_improvement=0.05.
- Clean labels are used only for diagnostic purity/NCR/CDR.

## Mechanism Diagnostics

| Dataset | Control | Balls | True-all purity | Noisy-train purity | NCR | CDR | Alpha positive | mean alpha | Eligible coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | real | 1 | 0.3021 | 0.2403 | 0.3280 | 0.6847 | 1.0000 | 0.1137 | 1.0000 |
| cora | shuffled | 1 | 0.3021 | 0.2403 | 0.3280 | 0.6847 | 1.0000 | 0.1137 | 1.0000 |
| citeseer | real | 1 | 0.2107 | 0.1955 | 0.2193 | 0.7828 | 1.0000 | 0.0346 | 1.0000 |
| citeseer | shuffled | 1 | 0.2107 | 0.1955 | 0.2193 | 0.7828 | 1.0000 | 0.0346 | 1.0000 |

## Preregistered Gate

| Dataset | true-all gap | noisy-train gap | eligible coverage | Gate |
|---|---:|---:|---:|---|
| citeseer | +0.0000 | +0.0000 | 1.0000 | fail |
| cora | +0.0000 | +0.0000 | 1.0000 | fail |

Overall gate: FAIL.

A single-ball partition was observed; real and shuffled controls therefore carry the same grouping when fixed X-space geometry does not accept any split.

Seed-0 downstream: NOT RUN under the preregistered stopping rule.

## Claim Boundary

- This diagnostic does not establish an H-free SoftDist pipeline.
- The observed failure is specific to the fixed X-space geometry and preregistered parameters.
- It does not rule out topology-aware or other label-free ball construction, which requires a separate preregistration.
