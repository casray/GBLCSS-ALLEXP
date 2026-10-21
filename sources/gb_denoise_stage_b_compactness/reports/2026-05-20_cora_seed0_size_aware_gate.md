# Stage B Size-Aware Gate: Cora Seed 0

## Research Hypothesis

B1 should improve granular-ball quality over B0 when label-free granular-ball compactness is actually activated. B3 and B4 should test whether the gain depends on true granular-ball assignment rather than arbitrary compactness.

## Granular-Ball Role

Granular balls are used as a label-free local structure prior. Labels are used only for linear evaluation and diagnostic purity.

## Method Motivation

The previous Cora gate failed because ball generation produced one oversized ball and reliable-ball filtering rejected it, leaving `covered_nodes=0`. This run uses size-aware granular-ball generation so oversized balls are forced to split until they can pass the size gate.

## Code Changes

- Added `max_size` to `generate_angular_balls`.
- Forced oversized low-distance balls to split instead of being accepted as one large ball.
- Passed `reliable_ball_max_size` from training into granular-ball generation.
- Added tests proving oversized balls split and B1 activates compactness after warm-up.

## Configuration

- Dataset: Cora
- Seed: 0
- Methods: B0, B1, B3, B4
- Config file: `configs/stage_b.yaml`
- Epochs: 1000
- Warm-up epochs: 300
- SSL objective: negative-free
- Ball space: H
- Reliable ball min size: 3
- Reliable ball max size: 256
- Reliable ball AD percentile: 50

## Expected Mechanism

After warm-up, B1 should build size-valid reliable balls and pull nodes toward detached true assignment centers. B3 and B4 should become harmful or weaker if true assignment matters.

## Success Standard

B1 should have nonzero compactness coverage after warm-up, improve diagnostics over B0, and beat B3/B4 without worsening collapse diagnostics.

## Failure Standard

The gate fails if compactness remains inactive, if B1 matches B3/B4, or if apparent radius/AD gains are caused by representation collapse.

## Results

| method | linear_eval_acc | num_balls | mean_radius | mean_average_angular_distance | effective_rank | mean_pairwise_cosine | label_purity_diagnostic | compact_nonzero_epochs | covered_last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.344 | 24 | 0.0389317 | 0.0206710 | 2.4701464 | 0.9985693 | 0.4383754 | 0 | 0 |
| B1 | 0.420 | 26 | 0.0292557 | 0.0159921 | 2.6148787 | 0.9993501 | 0.5763552 | 700 | 1661 |
| B3 | 0.130 | 37 | 0.0101471 | 0.0064865 | 1.6554857 | 0.9999616 | 0.3947389 | 700 | 1249 |
| B4 | 0.130 | 33 | 0.0121616 | 0.0067467 | 1.7119987 | 0.9999532 | 0.3893352 | 700 | 1059 |

Paired deltas against B0:

| method | delta_acc | delta_mean_radius | delta_ad | delta_effective_rank |
|---|---:|---:|---:|---:|
| B1 | +0.076 | -0.009676 | -0.004679 | +0.144732 |
| B3 | -0.214 | -0.028785 | -0.014184 | -0.814661 |
| B4 | -0.214 | -0.026770 | -0.013924 | -0.758148 |

## Interpretation

This run supports the mechanism at seed 0 only. B1 activated compactness and improved linear evaluation, label-purity diagnostics, radius, AD, and effective rank relative to B0. B3/B4 made balls even tighter, but accuracy and effective rank collapsed, which suggests arbitrary compactness can over-compress the representation.

This is consistent with the research direction: true granular-ball structure may be useful as a soft local prior, while random assignment controls can create misleading geometric compactness.

## Current Claim Boundary

Supported by this run:

- The original failed Cora gate was caused by inactive compactness, not by a tested negative result for B1.
- Size-aware splitting activates compactness on Cora.
- At seed 0, B1 beats B0/B3/B4 on linear evaluation and label-purity diagnostics.

Still only a hypothesis:

- B1 is robustly better across seeds or datasets.
- Negative-free SSL is necessary for this effect.
- The observed B1 gain is large enough beyond variance.

Weakened direction:

- Using radius or AD alone as success criteria is unsafe. B3/B4 reduce radius/AD while collapsing accuracy and effective rank.

## Final Decision

Continue with multi-seed validation.

Recommended next experiment:

1. Run Cora seeds 0/1/2 for B0/B1/B3/B4.
2. Add `compact_nonzero_epochs`, `covered_last`, and collapse metrics to the automatic summary.
3. Treat radius/AD improvements as valid only when effective rank and linear evaluation do not degrade.
