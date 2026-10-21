# Stage B Design: GB Compactness Loss for Encoder Shaping

## Context

This design covers Stage B of the CGBC-inspired granular-ball denoising experiments.
Stage A, running in another window, compares InfoNCE-style contrastive learning with
negative-sample-free graph SSL for granular-ball generation quality. Stage B assumes
that Stage A will identify the better SSL objective, or uses negative-sample-free SSL
as the provisional default because it is closest to the CGBC motivation.

The target hypothesis is:

> A label-free granular-ball compactness objective can back-propagate through the
> encoder and improve granular-ball generation quality without damaging the learned
> graph representation.

This is not a direct copy of CGBC. CGBC uses supervised purity, majority-vote ball
labels, and same-label / different-label inter-ball loss. Those mechanisms must not
enter unsupervised GRACE-style pretraining because they would create label leakage or
over-strong prototype pulling.

## Design Goals

- Verify whether granular-ball structure can act as a conservative representation
  shaping prior.
- Keep the experiment minimal and interpretable before adding more formulas.
- Separate true GB contribution from generic grouping, shuffled grouping, or extra
  regularization effects.
- Preserve the baseline evaluation protocol, splits, seeds, encoder, augmentations,
  and linear evaluation.
- Record enough diagnostics to distinguish real ball-quality improvement from
  representation collapse.

## Non-Goals

- Do not reproduce CGBC's supervised GBC training.
- Do not use true labels, label purity, or majority-vote labels in the training loss.
- Do not add inter-ball supervised separation in the first version.
- Do not use same-ball membership as hard positive labels.
- Do not change baseline train / validation / test splits or evaluation logic.
- Do not tune many hyperparameters before mechanism controls pass.

## Recommended Method

Use warm-up SSL followed by stop-gradient granular-ball compactness:

```text
warm up encoder with SSL
extract normalized encoder embeddings H
construct granular balls with a fixed label-free algorithm
compute ball centers and label-free confidence
continue training with L_ssl + lambda_gb * L_compact
```

The compactness objective is:

```text
L = L_ssl + lambda_gb * L_compact

L_compact = mean_b q_b * mean_{i in b} angular_distance(z_i, stopgrad(c_b))
```

Where:

- `z_i` is the current normalized encoder representation for node `i`.
- `c_b` is the normalized center of ball `b`, computed from detached embeddings.
- `q_b` is a label-free confidence weight.
- `angular_distance(a, b) = arccos(clamp(a^T b, -1 + eps, 1 - eps))`.

The first implementation may use binary reliable-ball gating instead of a complex
confidence formula:

```text
q_b = 1 if min_size <= |b| <= max_size and AD_b <= median(AD)
q_b = 0 otherwise
```

`AD_b` is the average angular distance between nodes in a ball and the ball center.

## Training Schedule

Recommended first schedule:

```text
epochs = 1000
warmup_epochs = 300
enable_gb_loss_epoch = 300
ball_refresh_interval = none for B1
ball_refresh_interval = 100 for B2
lambda_gb = 0.005 first
```

Training flow:

1. Train SSL only for `warmup_epochs`.
2. Extract normalized encoder output `H`.
3. Construct granular balls using the same algorithm as Stage A.
4. Freeze assignments and use stop-gradient centers.
5. Continue training with SSL plus compactness loss.
6. For the dynamic variant, refresh balls every `K` epochs, but keep centers
   detached inside each training interval.

## Experiment Matrix

Run the minimal gate first:

| ID | SSL objective | GB loss | Ball assignment | Purpose |
|---|---|---|---|---|
| B0 | best Stage A SSL, or negative-free default | none | none | baseline |
| B1 | same as B0 | compactness | true fixed balls after warm-up | minimal mechanism test |
| B3 | same as B0 | compactness | shuffled true-ball assignments | checks whether grouping identity matters |
| B4 | same as B0 | compactness | size-matched random balls | checks whether group-size regularization explains gains |

Only if B1 beats B3/B4 should the second gate run:

| ID | SSL objective | GB loss | Ball assignment | Purpose |
|---|---|---|---|---|
| B1-lambda-low | same as B0 | compactness, `lambda=0.001` | true fixed balls | sensitivity |
| B1-lambda-high | same as B0 | compactness, `lambda=0.01` | true fixed balls | sensitivity |
| B2 | same as B0 | compactness | true balls refreshed every 100 epochs | checks dynamic-ball stability |

## Datasets and Seeds

Initial datasets:

- Cora
- CiteSeer
- PubMed if the first two do not fail

Initial seeds:

```text
0, 1, 2
```

Expand to 5 seeds only after the minimal gate shows a real signal. If Stage A already
uses a different canonical seed list, Stage B should reuse that list for paired
comparison.

## Required Diagnostics

Ball-quality diagnostics:

- `num_balls`
- `mean_ball_size`
- `max_ball_size`
- `singleton_ratio`
- `mean_radius`
- `mean_average_angular_distance`
- `label_purity_diagnostic`, diagnostics only, never used in training
- `assignment_jaccard_across_epochs`
- `assignment_jaccard_across_seeds`

Representation diagnostics:

- `linear_eval_acc`
- `embedding_uniformity`
- `effective_rank`
- per-dimension feature standard deviation
- mean pairwise cosine similarity

Denoising diagnostics when noisy edges are enabled:

- edge noise AUC or PR-AUC
- clean-edge keep rate
- noisy-edge keep rate
- downstream noisy-graph paired delta

Loss diagnostics:

- `L_ssl`
- `L_compact`
- `lambda_gb * L_compact`
- compactness contribution ratio relative to SSL loss
- number and percentage of nodes covered by reliable balls

## Success Criteria

Stage B passes only if all of the following hold:

1. B1 improves ball quality over B0: lower `mean_average_angular_distance` or
   `mean_radius`, with no worsening in stability or singleton ratio.
2. B1 beats B3 and B4, showing the true GB grouping matters.
3. Linear evaluation does not materially drop relative to B0.
4. Collapse checks pass: effective rank, feature standard deviation, and uniformity
   remain healthy.
5. Gains are paired-seed consistent, not only a single-seed improvement.

Minimum defensible claim if successful:

> Label-free GB compactness can act as a conservative representation-shaping prior
> that improves granular-ball quality without damaging SSL representations.

## Failure Criteria

Stop or downgrade the direction if any of the following happens:

- B1 does not beat shuffled or size-matched random controls.
- Radius or AD improves only because effective rank collapses.
- Linear evaluation is clearly worse than B0.
- Dynamic refresh causes unstable ball assignments.
- Positive results appear only on Cora and fail on CiteSeer.
- The compactness loss dominates SSL loss.

If this fails, keep the result as evidence that GB compactness is not a safe
unsupervised training objective under the tested protocol.

## Implementation Boundaries

The code should be organized so that GB logic is isolated from SSL training:

```text
balls/
  generator.py
  metrics.py
  assignment_controls.py

ssl/
  grace.py
  negfree.py
  losses.py

experiments/
  run_stage_b.py
  configs/stage_b.yaml

reports/
  stage_b_summary.csv
  stage_b_paired_deltas.csv
```

The training script should accept all method choices from config:

- `ssl_objective`
- `warmup_epochs`
- `enable_gb_loss_epoch`
- `lambda_gb`
- `ball_space`
- `ball_refresh_interval`
- `assignment_control`
- `reliable_ball_min_size`
- `reliable_ball_max_size`
- `reliable_ball_ad_percentile`

## Reporting Format

Every run should write:

- config snapshot
- seed
- dataset
- method ID
- raw training log
- final evaluation metrics
- per-epoch or checkpoint-level ball diagnostics
- collapse diagnostics
- paired comparison against B0

Every result summary must separate:

- experimentally supported conclusions
- reasonable interpretations
- untested hypotheses
- directions weakened by negative results

## Open Questions

- If Stage A selects InfoNCE instead of negative-free as better for ball quality,
  Stage B should run on InfoNCE first and keep negative-free as an ablation.
- If Stage A shows no clear difference, Stage B should start with negative-free
  because it best matches the CGBC-inspired mechanism hypothesis.
- If compactness helps ball diagnostics but hurts linear evaluation, the result
  should be treated as inconclusive rather than a method win.
