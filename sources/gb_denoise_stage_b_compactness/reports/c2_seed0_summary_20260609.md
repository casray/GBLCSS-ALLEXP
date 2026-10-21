# C2 Structure-Regularization Seed0 Check

Date: 2026-06-09

Setting:

- Dataset: Cora
- Noise: symmetric 0.4 on train labels only
- Seed: 0
- Base config: `configs/stage_c_smoke.yaml`
- Results dir: `results_c2_20260609`
- Goal: test whether GB structure regularization has independent value when C1 label correction is disabled.

## Methods

| Method | Supervised loss | GB regularization | Label correction |
|---|---|---|---|
| C2_ce_noball | CE | none | off |
| C2_ce_realgb | CE | real GB intra/inter | off |
| C2_sce_noball | SCE | none | off |
| C2_sce_realgb | SCE | real GB intra/inter | off |
| C2_sce_shuffled | SCE | shuffled GB intra/inter | off |

## Results

| Method | classifier_test_acc | linear_eval_acc | label_purity | effective_rank | final_val_acc |
|---|---:|---:|---:|---:|---:|
| C2_ce_noball | 0.506 | 0.749 | 0.724 | 143.9 | 0.484 |
| C2_ce_realgb | 0.585 | 0.790 | 0.792 | 126.3 | 0.550 |
| C2_sce_noball | 0.495 | 0.738 | 0.714 | 144.6 | 0.482 |
| C2_sce_realgb | 0.565 | 0.784 | 0.782 | 130.2 | 0.554 |
| C2_sce_shuffled | 0.502 | 0.739 | 0.688 | 143.6 | 0.492 |

Reference:

- Previous noisy-CE baseline reported by Claude: 0.7878.
- Previous leakage-free C1 real GB reported by Claude: 0.620.

## Direct Findings

1. Real GB-reg gives a consistent seed0 gain over no-ball:
   - CE: 0.585 vs 0.506, +7.9pp.
   - SCE: 0.565 vs 0.495, +7.0pp.

2. Under SCE, real GB-reg beats shuffled GB-reg:
   - C2_sce_realgb 0.565 vs C2_sce_shuffled 0.502, +6.3pp.
   - This supports an independent real-GB structure signal, not just generic regularization.

3. SCE does not help in this setting:
   - SCE no-ball 0.495 is slightly below CE no-ball 0.506.
   - SCE realGB 0.565 is below CE realGB 0.585.

4. The classifier head remains the bottleneck:
   - C2_ce_realgb linear_eval_acc is 0.790, close to the reference baseline.
   - But its trained classifier_test_acc is only 0.585.
   - GB-reg improves representation quality but does not solve noisy supervised head training.

## Interpretation

The experiment supports a weak but real GB structure-prior signal:

- Real GB-reg improves both classifier_test_acc and linear_eval_acc over no-ball.
- Real GB-reg beats shuffled GB-reg under SCE, so the effect is tied to the true ball partition.

However, C2 is not a successful main method on Cora standard split:

- Best C2 result is C2_ce_realgb at 0.585.
- This remains far below the 0.7878 reference baseline and below leakage-free C1 at 0.620.
- The method improves the encoder representation but cannot train a robust classifier head from sparse noisy labels.

## Decision

Do not expand C2 to 5 seeds yet.

Current decision: mechanism signal exists, but paper-level method evidence is negative on Cora standard split seed0.

Recommended next action:

- Treat C2 as an ablation/motivation signal rather than a main method.
- If continuing, focus on classifier-head training under sparse noisy labels rather than adding more GB formulas.
- A minimal next diagnostic would be frozen-encoder classifier protocols with CE/SCE/bootstrapping, because linear_eval suggests the representation is much better than the Phase2 classifier.

