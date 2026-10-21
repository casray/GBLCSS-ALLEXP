# Stage A Stability Ablation Formal Report

## Research Hypothesis

本轮实验验证 Stage A 的初步观察是否依赖某个偶然设置。核心问题是：negative-free 对比学习得到的粒球更紧，是否只是图增强、encoder 配置或粒球生成参数造成的假象；以及普通 InfoNCE 的负样本排斥是否确实有助于保持全局语义可分性。

## Granular-Ball Role

粒球在本轮实验中只作为 label-free 表示空间诊断器使用。训练和建球不使用标签；标签只用于 `label_purity_diagnostic` 和 `linear_eval_acc` 的后验诊断。

主粒球生成算法为 `gain_split`：先对 embedding 做 L2 normalize，再使用角距离、farthest-pair split、`min_ball_size / max_depth / min_split_gain` 递归切分。该算法用于检验同一表示空间下能否形成紧凑且语义一致的局部结构。

## Method Motivation

上一轮 Stage A 结果显示：negative-free 表示的球半径和平均角距离更小，但线性分类、label purity 和 collapse 诊断变差。本轮做三组稳定性检查：

1. 图增强强度 ablation：检查结论是否由增强过强或过弱导致。
2. encoder / stabilizer ablation：检查 negative-free 是否只是缺少 BN、variance regularizer 或 EMA target。
3. 粒球参数 ablation：固定训练好的 embedding，只改变建球参数，检查结论是否由某个粒球参数偶然造成。

## Experimental Setup

- Datasets: Cora, CiteSeer
- Seeds: 0, 1, 2
- Training runs: 54
- Ball-parameter rebuild rows: 540
- Device: all 54 training runs recorded `device = cuda` under ROCm PyTorch
- Results directory: `results_stability`
- Main files:
  - `results_stability/stage_b_summary.csv`
  - `results_stability/stage_b_grouped_summary.csv`
  - `results_stability/ball_ablation_summary.csv`

## Experiment 1: Augmentation Strength Ablation

Settings:

| group | edge drop | feature mask |
|---|---:|---:|
| aug_weak | view1=0.1, view2=0.2 | view1=0.1, view2=0.2 |
| aug_stage_a | view1=0.2, view2=0.4 | view1=0.3, view2=0.4 |
| aug_strong | view1=0.3, view2=0.5 | view1=0.4, view2=0.5 |

Mean over seeds 0/1/2:

| dataset | group | objective | radius | AD | purity | acc | eff_rank | feat_std |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cora | aug_weak | InfoNCE | 0.9365 | 0.6475 | 0.6762 | 0.7803 | 140.2928 | 0.0526 |
| Cora | aug_weak | negfree | 0.1310 | 0.0495 | 0.4990 | 0.3510 | 5.2764 | 0.0054 |
| Cora | aug_stage_a | InfoNCE | 0.8983 | 0.5858 | 0.7064 | 0.7880 | 135.4512 | 0.0515 |
| Cora | aug_stage_a | negfree | 0.0639 | 0.0140 | 0.5613 | 0.3557 | 2.2150 | 0.0022 |
| Cora | aug_strong | InfoNCE | 0.8749 | 0.5620 | 0.7065 | 0.7950 | 130.5526 | 0.0511 |
| Cora | aug_strong | negfree | 0.0168 | 0.0068 | 0.4709 | 0.2797 | 1.3647 | 0.0012 |
| CiteSeer | aug_weak | InfoNCE | 1.0030 | 0.7146 | 0.5833 | 0.6300 | 152.8038 | 0.0552 |
| CiteSeer | aug_weak | negfree | 0.1869 | 0.0368 | 0.2953 | 0.4447 | 6.6807 | 0.0065 |
| CiteSeer | aug_stage_a | InfoNCE | 0.9398 | 0.6340 | 0.6054 | 0.6213 | 145.8881 | 0.0536 |
| CiteSeer | aug_stage_a | negfree | 0.2452 | 0.0371 | 0.2808 | 0.4033 | 6.8008 | 0.0049 |
| CiteSeer | aug_strong | InfoNCE | 0.9114 | 0.5951 | 0.6373 | 0.6463 | 142.0524 | 0.0533 |
| CiteSeer | aug_strong | negfree | 0.2936 | 0.0403 | 0.2537 | 0.5240 | 7.0821 | 0.0050 |

Interpretation:

- Across weak, Stage-A, and strong augmentations, negative-free consistently gives much smaller radius and AD.
- The same negative-free runs consistently show much lower effective rank and feature std, indicating over-compressed representations.
- InfoNCE keeps much higher effective rank and linear accuracy on both datasets.
- Therefore, the Stage A observation is not explained only by one bad augmentation strength.

Supported claim:

> Under the tested augmentation range, raw negative-free SSL tends to produce geometrically compact but globally less separable representations.

Not supported:

> Smaller granular-ball radius means better granular-ball quality.

## Experiment 2: Encoder / Stabilizer Ablation

Fixed augmentation: `aug_stage_a`.

Mean over seeds 0/1/2:

| dataset | objective | variant | radius | AD | purity | acc | eff_rank | feat_std |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Cora | InfoNCE | plain | 0.8983 | 0.5858 | 0.7064 | 0.7880 | 135.4512 | 0.0515 |
| Cora | negfree | plain | 0.0639 | 0.0140 | 0.5613 | 0.3557 | 2.2150 | 0.0022 |
| Cora | negfree | BN | 1.0917 | 0.6581 | 0.7224 | 0.7910 | 131.8897 | 0.0622 |
| Cora | negfree | variance | 0.0699 | 0.0220 | 0.3021 | 0.7180 | 4.0238 | 0.0014 |
| Cora | negfree | EMA | 0.3460 | 0.0800 | 0.4476 | 0.5593 | 9.7099 | 0.0052 |
| CiteSeer | InfoNCE | plain | 0.9398 | 0.6340 | 0.6054 | 0.6213 | 145.8881 | 0.0536 |
| CiteSeer | negfree | plain | 0.2452 | 0.0371 | 0.2808 | 0.4033 | 6.8008 | 0.0049 |
| CiteSeer | negfree | BN | 1.1034 | 0.6789 | 0.6176 | 0.5857 | 136.4544 | 0.0622 |
| CiteSeer | negfree | variance | 0.0351 | 0.0165 | 0.2107 | 0.4867 | 3.1237 | 0.0011 |
| CiteSeer | negfree | EMA | 0.0948 | 0.0489 | 0.3647 | 0.6127 | 7.6013 | 0.0034 |

Interpretation:

- `negfree_bn` is the strongest counterexample to an overly broad negative-free failure claim.
- With BN, negative-free recovers effective rank, feature std, purity, and linear accuracy close to InfoNCE.
- But BN also removes the extreme compactness: radius and AD become similar to or larger than InfoNCE.
- `negfree_var` and `negfree_ema` do not reliably recover semantic ball quality. They can improve linear accuracy in some cases, but effective rank and feature std remain far below InfoNCE/BN.

Supported claim:

> The problem is not simply “negative-free is bad.” The raw negative-free configuration is unstable and over-compresses; BN can restore semantic separability, but then the compact-ball advantage mostly disappears.

Important implication:

> The desired method should not optimize compactness alone. It must balance local compactness with global separability or anti-collapse constraints.

## Experiment 3: Granular-Ball Parameter Ablation

Fixed saved embeddings were reused; no retraining was performed. Parameter grid:

- `min_ball_size`: 4, 8, 16
- `max_depth`: 8, 12
- `min_split_gain`: 0.0005, 0.001, 0.005

For the Stage-A augmentation setting, the observed metric ranges were:

| dataset | objective | radius range | AD range | purity range | acc | eff_rank |
|---|---|---:|---:|---:|---:|---:|
| Cora | InfoNCE | 0.7174-1.0216 | 0.4595-0.7071 | 0.6123-0.7716 | 0.7880 | 135.4511 |
| Cora | negfree | 0.0145-0.0895 | 0.0093-0.0175 | 0.3919-0.6692 | 0.3557 | 2.2637 |
| CiteSeer | InfoNCE | 0.7212-1.0571 | 0.4696-0.7602 | 0.5406-0.6885 | 0.6213 | 145.8880 |
| CiteSeer | negfree | 0.1281-0.2452 | 0.0289-0.0371 | 0.2808-0.3886 | 0.4033 | 6.8583 |

Best purity settings usually used smaller `min_ball_size=4` and deeper splitting. This can raise diagnostic purity for both InfoNCE and negative-free, but it does not change the underlying embedding quality: linear accuracy, effective rank, and feature std stay fixed because embeddings are reused.

Interpretation:

- Granular-ball parameters can change the diagnostic purity and radius substantially.
- However, parameter tuning cannot repair collapsed or weak global representations.
- The conclusion that raw negative-free has low effective rank and poor linear accuracy is not a ball-parameter artifact.

Supported claim:

> Ball-generation hyperparameters affect how local structure is measured, but they do not explain away the representation collapse signal.

## What The Results Support

1. Negative-free without stabilizing normalization tends to generate very compact balls, but this compactness is often caused by representation over-compression.
2. InfoNCE produces less compact balls, but preserves global semantic separability better under the tested settings.
3. BN is a critical confound: it can make negative-free competitive with InfoNCE in purity and accuracy, but it also removes the “small ball radius” advantage.
4. Particle-ball quality cannot be judged by radius or AD alone. Radius/AD must be interpreted together with label purity, linear evaluation, effective rank, and feature std.
5. The original guess about conflict between negative-sample repulsion and local aggregation is only partially supported: repulsion may hurt compactness, but it appears helpful for global separability.

## What The Results Do Not Support

1. They do not support the claim that negative-free is always worse than InfoNCE.
2. They do not support the claim that lower radius means better granular balls.
3. They do not support using raw compactness loss as a strong training objective without anti-collapse or separation controls.
4. They do not yet prove that a new method will improve downstream denoising; this round diagnoses representation and ball structure only.

## Main Failure Or Instability Reason

The repeated failure pattern is over-compression. Raw negative-free aligns two views without negative repulsion or sufficient normalization constraints. It can pull many nodes into a narrow region of representation space. The ball generator then sees very tight local groups, giving small radius and AD, but the representation loses class-level separability.

BN changes this behavior by maintaining feature scale and representation diversity. This suggests the next method should preserve diversity explicitly rather than only encouraging local compactness.

## Still Useful Signal

The useful signal is not “negative-free wins.” The useful signal is:

> There is a measurable tradeoff between local granular-ball compactness and global semantic separability.

This tradeoff is exactly where a GB-GCL method can contribute: use granular balls as soft local structure priors, but add mechanisms that prevent global collapse and preserve inter-class or inter-region separation.

## Recommended Next Decision

Do not continue with raw negative-free as the main method. Use `negfree_bn` or InfoNCE as the stable encoder baseline, then test a conservative GB loss on top of it.

The next smallest publishable mechanism test should be:

1. Baseline: InfoNCE plain and negative-free BN.
2. Add label-free GB compactness with very small weight.
3. Add anti-collapse/separation diagnostics as hard gates.
4. Accept improvement only if radius/AD improves without hurting linear accuracy, effective rank, or feature std.

## Claim Boundary For Paper Writing

Supported wording:

> In our controlled ablations, compact granular balls alone are not sufficient evidence of high-quality semantic structure. Raw negative-free graph SSL can produce very small granular balls while substantially degrading representation rank and downstream linear evaluation. Stabilizing normalization largely restores semantic quality but reduces the compactness advantage, revealing a tradeoff between local compactness and global separability.

Avoid wording:

> Negative-free contrastive learning is worse than InfoNCE.

Avoid wording:

> InfoNCE negative samples conflict with granular balls, therefore negative-free is better.

Better hypothesis for the next stage:

> A robust GB-GCL method should treat granular balls as a soft local prior while preserving representation diversity and global separability.

## Evidence Files

- `results_stability/stage_b_summary.csv`
- `results_stability/stage_b_grouped_summary.csv`
- `results_stability/ball_ablation_summary.csv`
- `src/gb_stage_b/balls.py`
- `src/gb_stage_b/train.py`
- `configs/stage_b.yaml`
