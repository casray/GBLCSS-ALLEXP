# MLP-8 P2a Rescue Downstream Report

## Protocol

- Dataset: Cora
- Seed: 0
- Noise: train-only uniform 0.4
- Split: percent 0.2/0.1/0.7
- Backbone: X-only 2-layer MLP, `frozengrace_gb_mlp_softdist_beta05`
- P2a candidate embedding: `runs/mlp8_p2_advisor_smoke_20260702/embeddings/Cora/grace_ce_l0p1/H.pt`
- Loose ball params: `max_balls=64, min_ball_size=16, split_improvement=0.0`

Clean labels are used only for evaluation and diagnostics.

## Results

| Arm | Ball embedding | Params | Real/Shuffled | Balls | Noisy train purity | True train purity | Valid | Test |
|---|---|---|---|---:|---:|---:|---:|---:|
| A0 anchor_default | frozen GRACE | 64/32/0.05 | real | 52 | 0.4954 | 0.7893 | 57.04 | 55.46 |
| A1 anchor_loose | frozen GRACE | 64/16/0.0 | real | 64 | 0.5009 | 0.8022 | 57.41 | 53.61 |
| B0 p2a_default | GRACE+0.1 noisy CE | 64/32/0.05 | real | 1 | 0.2255 | 0.3198 | 46.30 | 42.96 |
| B0 p2a_default_shuffled | GRACE+0.1 noisy CE | 64/32/0.05 | shuffled | 1 | 0.2255 | 0.3198 | 46.30 | 42.96 |
| B1 p2a_loose | GRACE+0.1 noisy CE | 64/16/0.0 | real | 64 | 0.5028 | 0.7978 | 56.30 | 53.35 |
| B2 p2a_loose_shuffled | GRACE+0.1 noisy CE | 64/16/0.0 | shuffled | 64 | 0.3407 | 0.3870 | 45.93 | 41.58 |

## Gate Check

| Gate | Required | Result |
|---|---|---|
| Collapse rescue | B1 > B0 | Pass: +10.40pp |
| Real local structure | B1 > B2 | Pass: +11.77pp |
| Beats frozen-GRACE default | B1 > A0 | Fail: -2.11pp |
| Beats frozen-GRACE loose | B1 >= A1 | Fail: -0.26pp |

## Decision

P2a should not be expanded to 5 seeds. The rescue confirms that the original P2a failure was partly a construction-threshold collapse: changing `64/32/0.05` to `64/16/0.0` moves the P2a candidate from one ball and 42.96% test accuracy to 64 balls and 53.35% test accuracy. The recovered real grouping also strongly beats the shuffled control.

However, the recovered P2a target is still below the frozen-GRACE anchor. The supervised CE-assisted representation therefore does not provide additional downstream value beyond frozen GRACE. It should be recorded as a negative but informative advisor-suggestion result, not as a new mainline.
