# Coverage-aware Noisy-label GB Local Smoothing Validation

## Research Hypothesis

Frozen GRACE encoder representations may contain local regions that are useful for noisy-label node classification, but strict ball-only label smoothing is too brittle. A coverage-aware GB target can use ball reliability as a confidence signal and fall back to CE or global smoothing when local ball evidence is weak.

## Granular-Ball Role

Granular balls are classifier-stage local reliability units. They are generated on L2-normalized frozen GRACE encoder output `H` and are not used in GRACE pretraining, graph augmentation, hard positive construction, hard negative filtering, or fine-tuning.

## Method Motivation

Experiment 7 showed that strict GB local smoothing is Cora-positive in selected medium-label high-noise settings but unstable on CiteSeer. The likely issue is not only low coverage; local label distributions also need confidence calibration against generic local baselines such as kmeans and topk matched.

## Code Changes

- Added `gb_ca_ce`: coverage-aware GB smoothing with noisy one-hot fallback.
- Added `gb_ca_global`: coverage-aware GB smoothing with global-prior fallback.
- Kept historical baselines: `ce`, `global`, `kmeans`, `topk_matched`, `gb`, and `gb_prior`.
- Added unit tests for sparse-group confidence and global fallback behavior.

## Configuration Change

Default gate methods now include:

```text
ce global kmeans topk_matched gb gb_prior gb_ca_ce gb_ca_global
```

The validation gate should use the existing medium-label ratio protocol:

```text
datasets = Cora, CiteSeer
split_mode = ratio
train_ratios = 0.1, 0.2
val_ratio = 0.1
noise_type = symmetric
noise_rates = 0.2, 0.4
seeds = 0, 1, 2, 3, 4
primary_alpha = 0.3
mode = frozen
```

## Expected Mechanism

For each train node, the coverage-aware target mixes a fallback target with the ball-level label distribution. The local GB contribution is weighted by train-label count, ball-label entropy, ball radius, and node-to-center distance. Low-confidence balls therefore contribute little instead of forcing strict local smoothing.

## Success Criteria

The direction is worth continuing only if `gb_ca_ce` or `gb_ca_global`:

- beats `ce`, `global`, `kmeans`, and `topk_matched` on both Cora and CiteSeer under the same split, seeds, noisy labels, and frozen embeddings;
- has positive paired deltas that are not driven by a single seed;
- reports interpretable coverage, coverage-by-class, ball train-label count, alpha, and entropy diagnostics.

## Failure Criteria

Downgrade or stop if:

- CiteSeer remains unstable;
- kmeans or topk matched ties or beats the coverage-aware GB variants;
- gains appear only in Cora train 20% / 40% noise;
- the confidence diagnostics cannot explain when GB helps or hurts.

## Smoke Result

An end-to-end smoke run was executed only to verify code paths, not to support a scientific claim:

```text
dataset = Cora
split_mode = ratio
train_ratio = 0.1
val_ratio = 0.1
pretrain_seed = noise_seed = classifier_seed = 0
pretrain_epochs = 1
classifier_epochs = 2
noise_rate = 0.2
methods = ce global kmeans topk_matched gb gb_prior gb_ca_ce gb_ca_global
```

The smoke run produced result rows for both new methods and confirmed nonzero confidence diagnostics:

```text
gb_ca_ce: coverage = 1.0, alpha_mean_nonzero ~= 0.0524
gb_ca_global: coverage = 1.0, alpha_mean_nonzero ~= 0.0255
```

These numbers are not comparable to formal Experiment 7 results because the encoder and classifier were intentionally undertrained for smoke testing.

## Final Decision

Continue to formal validation only as a controlled mechanism test. Do not present this as a new main method unless the full Cora/CiteSeer 5-seed gate beats CE, global, kmeans, and topk matched with stable paired deltas.
