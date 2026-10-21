# C1 Ratio-Split Vote-Coverage Diagnostic

Date: 2026-06-09

Setting:

- Dataset: Cora
- Noise: symmetric 0.4 on train labels only
- Seed: 0
- Base config: `configs/stage_c_smoke.yaml`
- Methods: `C1`, `C1_noball`, `C1_topk`, `C1_shuffled`
- Results dir: `results_ratio_c1_20260609`
- Goal: check whether C1 recovers when train-label coverage is increased from Planetoid low-label split to medium/high-label ratio splits.

## Result Table

| train_ratio | val_ratio | method | classifier_test_acc | linear_eval_acc | label_purity | last_correction_acc |
|---:|---:|---|---:|---:|---:|---:|
| 0.1 | 0.1 | C1 | 0.634 | 0.812 | 0.802 | 0.185 |
| 0.1 | 0.1 | C1_noball | 0.507 | 0.764 | 0.704 | 0.000 |
| 0.1 | 0.1 | C1_topk | 0.459 | 0.787 | 0.752 | 0.435 |
| 0.1 | 0.1 | C1_shuffled | 0.546 | 0.792 | 0.760 | 0.065 |
| 0.2 | 0.1 | C1 | 0.744 | 0.837 | 0.813 | 0.315 |
| 0.2 | 0.1 | C1_noball | 0.570 | 0.804 | 0.719 | 0.000 |
| 0.2 | 0.1 | C1_topk | 0.570 | 0.816 | 0.714 | 0.444 |
| 0.2 | 0.1 | C1_shuffled | 0.617 | 0.839 | 0.756 | 0.060 |
| 0.6 | 0.2 | C1 | 0.814 | 0.847 | 0.818 | 0.643 |
| 0.6 | 0.2 | C1_noball | 0.615 | 0.821 | 0.714 | 0.000 |
| 0.6 | 0.2 | C1_topk | 0.773 | 0.853 | 0.803 | 0.698 |
| 0.6 | 0.2 | C1_shuffled | 0.652 | 0.849 | 0.762 | 0.100 |

## Ball Vote Coverage

Coverage is computed on final C1 embeddings and reliable balls.

| train_ratio | reliable_balls | avg_train_per_reliable_ball | reliable_zero_train | reliable_one_train | reliable_ge2_train | reliable_ge3_train | reliable_ge5_train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 171 | 1.33 | 66 | 52 | 53 | 27 | 11 |
| 0.2 | 199 | 2.46 | 24 | 64 | 111 | 70 | 29 |
| 0.6 | 158 | 9.28 | 0 | 1 | 157 | 149 | 119 |

## Direct Findings

1. C1 improves monotonically with train-label coverage:
   - 10% train: 0.634
   - 20% train: 0.744
   - 60% train: 0.814

2. The improvement tracks ball vote coverage:
   - 10% train: only 27 reliable balls have at least 3 train nodes.
   - 20% train: 70 reliable balls have at least 3 train nodes.
   - 60% train: 149 reliable balls have at least 3 train nodes.

3. Real GB remains better than shuffled GB at every ratio:
   - 10%: 0.634 vs 0.546, +8.8pp.
   - 20%: 0.744 vs 0.617, +12.7pp.
   - 60%: 0.814 vs 0.652, +16.2pp.

4. Real GB remains better than no-ball at every ratio:
   - 10%: 0.634 vs 0.507, +12.7pp.
   - 20%: 0.744 vs 0.570, +17.4pp.
   - 60%: 0.814 vs 0.615, +19.9pp.

5. At 60/20/20, topk also becomes strong:
   - C1_topk reaches 0.773.
   - C1 still beats it at 0.814, but the gap narrows to +4.1pp.

## Interpretation

This diagnostic strongly supports the label-coverage explanation.

The previous Planetoid standard split failure was not evidence that encoder adaptation to GB is impossible. It was mainly caused by a missing prerequisite for GB majority-vote denoising: enough labeled members per ball.

When train-label coverage increases, C1 recovers:

- With 20% train labels, C1 already reaches 0.744 and is clearly above no-ball/topk/shuffled.
- With 60% train labels, C1 reaches 0.814 and passes the old standard-split reference baseline of 0.7878, although that baseline is not directly comparable because the split changed.

The mechanism is therefore conditionally viable:

- Viable under medium/high-label noisy-node classification.
- Not viable as a majority-vote label corrector under the Cora Planetoid low-label split.

## Decision

Do not use Planetoid standard split as the main positive setting for C1.

Recommended next step:

1. Treat 60/20/20 as a CGBC/GBmooer-premise diagnostic.
2. Treat 10% and 20% ratio splits as the more defensible medium-label protocol.
3. Run 5 seeds for Cora and CiteSeer at `train_ratio=0.2`, `val_ratio=0.1`, `noise_rate=0.4` with `C1`, `C1_noball`, `C1_topk`, and `C1_shuffled`.
4. If C1 remains above all controls, frame the claim as:

   `Granular-ball majority-vote denoising is useful for noisy-label node classification when local labeled coverage is sufficient.`

5. Do not claim the method solves the standard low-label Planetoid split.

