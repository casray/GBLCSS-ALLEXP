# AGENT.MD - Noisy-label GB-GRACE Project Plan

## 0. Workspace Boundary

This folder is the only writable workspace for this new project:

```text
C:\Users\Admin\Desktop\ALL_exp\noisy_label_gb_grace
```

All future code, configs, logs, results, notes, scripts, tables, and draft documents for this project must be created or modified inside this folder.

Any modification outside this folder, including files in `C:\Users\Admin\Desktop\ALL_exp`, requires explicit user approval before editing.

Reading files outside this folder is allowed when needed for context, literature review, baseline comparison, or reproducing prior project experience. Editing outside this folder is not allowed without approval.

## 1. Project Positioning

This project studies noisy-label node classification after self-supervised graph contrastive pretraining.

The central task is not to convert GRACE into a supervised method. The task is:

```text
Stage 1: Unsupervised graph contrastive pretraining
X, A -> GRACE -> encoder f_theta -> node representation H

Stage 2: Noisy-label robust node classification
Construct granular balls on H
Use noisy train labels only for local label smoothing / confidence weighting
Train classifier or lightly fine-tune encoder
Evaluate on clean validation and test labels
```

Recommended English framing:

```text
Granular-ball assisted graph contrastive pretraining for noisy-label node classification.
```

Recommended Chinese framing:

```text
图对比学习提供无标签鲁棒表示，粒球在该表示空间中形成局部可信区域，用局部软标签替代单个噪声标签，从而提升噪声标签节点分类的鲁棒性。
```

## 2. Motivation

Prior project experience shows that directly inserting granular balls into clean GRACE pretraining is unstable:

- GB-guided view generation repeatedly underperformed healthy GRACE.
- Hard same-ball positives, prototype pulling, broad same-ball suppression, and direct relation regularization are risky.
- GB-FN topk-soft is safer than hard or broad same-ball rules, but current gains are tiny and close to no-ball topk.
- Clean graph gains smaller than standard deviation cannot support a strong paper claim.

The new direction changes the research question:

```text
Can GRACE representations make granular balls useful for robust classification under noisy training labels?
```

This is closer to the mechanisms in GBmooer and CGBC:

- contrastive learning first improves the representation space;
- granular balls are then used for local label smoothing, denoising, or coarse-grained robust training;
- noisy labels affect only the supervised classification stage, not unsupervised pretraining.

## 3. Core Hypothesis

### Main Hypothesis

GRACE learns label-free node representations in which nearby local regions may have more reliable semantic consistency than individual noisy training labels. Granular balls can summarize these local regions and produce confidence-weighted soft labels for noisy-label node classification.

### Minimal Testable Hypothesis

Under synthetic train-label noise, `GRACE + GB local label smoothing` is more robust than:

- `GRACE + standard noisy-label classifier`;
- `GRACE + global label smoothing`;
- `GRACE + kmeans local label smoothing`;
- `GRACE + no-ball topk local smoothing`.

### Claim Boundary

Allowed claim if successful:

```text
Granular balls can serve as a local, confidence-weighted label smoothing prior for noisy-label node classification after self-supervised graph contrastive pretraining.
```

Not allowed:

```text
Granular balls significantly improve GRACE pretraining.
Same-ball nodes are semantic positives.
GB local smoothing is universally better than all clustering or similarity-based alternatives.
GB improves clean graph representation learning.
```

## 4. Method Design

### 4.1 Safe Version: Frozen Encoder

This is the first version to implement.

```text
1. Train GRACE on the original graph without labels.
2. Extract encoder output H.
3. L2-normalize H.
4. Construct granular balls on H.
5. Inject noise only into train labels.
6. For each ball, compute local noisy-label distribution using train nodes only.
7. Generate confidence-weighted soft labels for train nodes.
8. Train a classifier on H with soft labels.
9. Evaluate on clean validation and test labels.
```

The first version freezes the encoder. It is a gate experiment to test whether the GB smoothing signal exists.

### 4.2 Paper Version: Light Fine-tuning

Run this only if the frozen-encoder version passes the gate.

```text
1. GRACE warm-up without labels.
2. Construct balls on normalized H or weak-view H.
3. Generate GB local soft labels and confidence weights from noisy train labels.
4. Fine-tune encoder and classifier with:
   L = L_GB_soft_label + beta * L_GRACE
5. Keep validation and test labels clean.
```

This version is closer to a full method, but it must not be implemented before the safe version shows signal.

## 5. Local Label Smoothing Rule

For each granular ball `B`, compute the class distribution from noisy labels of train nodes inside the ball:

```text
p_B[c] = count(train node in B with noisy label c) / count(train node in B)
```

For train node `i` in ball `B`:

```text
y_i_soft = (1 - alpha_i) * onehot(y_i_noisy) + alpha_i * p_B
```

Use confidence-gated smoothing:

```text
ball_confidence = 1 / (1 + radius_B)
node_confidence = exp(-dist(h_i, center_B) / (radius_B + eps))
alpha_i = alpha_max * ball_confidence * node_confidence
```

Initial conservative settings:

```text
alpha_max in {0.3, 0.5}
eps = 1e-8
representation = normalized encoder output H
```

Boundary or low-confidence nodes should receive less smoothing, not hard filtering.

### 5.1 Empty or Small-ball Fallback

Planetoid-style splits may contain very few labeled train nodes. Many balls may contain no train node, one train node, or only a few train nodes. In this case, the ball label distribution is unreliable.

Use this default rule:

```text
min_train_per_ball = 3
```

For any ball `B`:

```text
if |B_train| < min_train_per_ball:
    disable GB local smoothing for train nodes in B
    alpha_i = 0
```

Optional fallback after the basic method shows signal:

```text
p_B = (count_B + gamma * p_global) / (|B_train| + gamma)
```

where `p_global` is the noisy train-label distribution over the whole train split. This Dirichlet-style fallback may be ablated only after the simple fallback has been evaluated.

## 6. Noise Injection Protocol

Only training labels may be corrupted.

```text
Train labels: noisy
Validation labels: clean
Test labels: clean
Graph structure A: original in first stage
Node features X: original in first stage
```

Initial noise settings:

```text
datasets = Cora, CiteSeer
noise_type = symmetric
noise_rate = 0%, 20%, 40%
seeds = 5
```

The `0%` setting is a sanity check, not a clean-graph claim. It checks whether GB smoothing damages clean-label classifier training.

Full noise settings after the gate passes:

```text
datasets = Cora, CiteSeer, PubMed
noise_type = symmetric, asymmetric, instance-dependent
noise_rate = 10%, 20%, 30%, 40%, 50%
seeds = 5
```

### Noise Definitions

Symmetric label noise:

```text
Randomly flip a selected train node label to a different class uniformly.
```

Asymmetric label noise:

```text
Flip selected train node labels by a fixed class transition, e.g. c -> (c + 1) mod C.
```

Instance-dependent label noise:

```text
Flip labels with higher probability for uncertain or boundary-like train nodes.
```

Instance-dependent noise is deferred until symmetric and asymmetric settings are implemented correctly.

### 6.1 Seed Control

Record these seeds separately:

```text
split_seed
pretrain_seed
noise_seed
classifier_seed
```

For each `(dataset, split_seed, pretrain_seed, noise_type, noise_rate, noise_seed)`, all methods must use the exact same corrupted train-label file.

Do not compare methods trained with different noisy-label realizations.

## 7. Required Baselines

The method must not only beat a weak noisy-label baseline.

Minimum baselines:

```text
GCN + noisy labels
GCN + standard label smoothing
GRACE + linear classifier on noisy labels
GRACE + global label smoothing
GRACE + kmeans local label smoothing
GRACE + no-ball topk local smoothing
GRACE + GB local label smoothing
```

Critical comparison baselines:

```text
GRACE + global label smoothing
GRACE + kmeans local label smoothing
GRACE + no-ball topk local smoothing
```

If GB local smoothing cannot beat or clearly differentiate from these baselines, do not make a GB-specific claim.

### 7.1 Fair Local Baseline Matching

Local smoothing baselines must match GB granularity as closely as possible.

Rules:

```text
kmeans cluster count K = GB ball count from the same representation and seed
no-ball topk k = average non-singleton GB ball size - 1
```

If the gate experiment shows a positive signal, add:

```text
random partition with matched group count
size-matched random partition if feasible
```

These controls answer whether the gain comes from granular-ball structure or from generic local grouping plus smoothing.

## 8. Gate Experiment

### Purpose

Determine whether GB local smoothing has an independent, useful signal before building a larger method.

### Settings

```text
datasets = Cora, CiteSeer
noise = symmetric 0%, 20%, 40%
seeds = 5
encoder = frozen GRACE encoder
representation = normalized H
```

### Compare

```text
GRACE + CE noisy label
GRACE + global label smoothing
GRACE + kmeans local smoothing
GRACE + no-ball topk local smoothing
GRACE + GB local label smoothing
```

### Metrics

```text
accuracy mean/std
per-seed paired delta
best validation accuracy
test accuracy at best validation
soft-label entropy
ball count
average ball size
singleton ball ratio
radius mean/std
balls with 0 train nodes
balls with 1 train node
balls with 2-4 train nodes
balls with >=5 train nodes
effective smoothing coverage
train-label distribution per ball
GB vs kmeans grouping similarity
GB vs no-ball selected-neighbor overlap
```

Effective smoothing coverage:

```text
coverage = count(train nodes with alpha_i > 0) / count(train nodes)
```

## 9. Success Criteria

The gate experiment passes only if most of the following hold:

- GB local smoothing beats `GRACE + global label smoothing` on both Cora and CiteSeer under at least one noise rate.
- GB local smoothing is not worse than kmeans and no-ball topk under the same protocol.
- Paired seed deltas are mostly positive, not only driven by one seed.
- Mean improvement is not obviously swallowed by standard deviation.
- The method does not degrade badly under 20% noise while helping at 40% noise.
- The 0% noise sanity check does not show severe degradation.
- Diagnostics show GB creates a meaningfully different local smoothing distribution from global smoothing and no-ball topk.
- Effective smoothing coverage is large enough to make the method meaningful.

## 10. Failure Criteria

Stop or downgrade the direction if any of the following happens:

- GB local smoothing cannot beat global label smoothing.
- kmeans local smoothing or no-ball topk matches or beats GB in most settings.
- Improvement appears only on Cora or only at one noise rate.
- Gains are smaller than overlapping standard deviations and paired seed deltas are unstable.
- GB creates too many singleton balls, making smoothing ineffective.
- GB creates overly coarse balls, making local smoothing equivalent to global smoothing.
- Too many balls have fewer than `min_train_per_ball` train nodes, making GB smoothing mostly disabled.
- The 0% noise sanity check shows clear harm, indicating over-smoothing or unreliable balls.
- High ball purity or high diagnostic score does not translate into downstream improvement.

If the gate fails, write the result as:

```text
GB local smoothing did not show independent value over simpler local or global smoothing baselines under the tested noisy-label setting.
```

Do not continue by adding more formulas.

## 11. Decision Boundary for Next Steps

### Continue to Full Experiment If

The gate experiment shows stable GB-specific signal against global smoothing, kmeans smoothing, and no-ball topk.

Next steps:

```text
1. Add PubMed.
2. Add asymmetric noise.
3. Add instance-dependent noise.
4. Add runtime and memory cost.
5. Add ablation for alpha_max and confidence components.
```

### Try Light Fine-tuning If

The frozen-encoder version passes the gate and the GB signal remains visible after adding kmeans and no-ball controls.

Fine-tuning loss:

```text
L = L_GB_soft_label + beta * L_GRACE
```

Constraints:

```text
beta must be small and ablated.
GRACE consistency must not use labels.
Balls must be updated slowly or frozen after warm-up.
```

### Stop If

GB local smoothing is not distinguishable from simpler controls.

Do not rescue the direction by:

- changing evaluation protocol;
- removing hard baselines;
- adding hard same-ball positives;
- moving labels into GRACE pretraining;
- tuning only on Cora;
- reporting only best seed;
- ignoring standard deviation.

## 12. Prohibited Directions

The following are not allowed in this project unless the user explicitly approves a new direction:

- Using labels, label purity, or class boundaries inside GRACE pretraining.
- Treating same-ball membership as hard positive labels.
- Prototype pulling or same-label ball-center pulling during unsupervised pretraining.
- Replacing GRACE view generation with GB-controlled views as the main method.
- Changing train / validation / test splits.
- Polluting validation or test labels.
- Claiming significance from one seed or one dataset.
- Claiming success from improvements smaller than overlapping standard deviations.
- Hiding negative results or omitting kmeans / no-ball controls.

## 13. Required Records for Every Experiment

Every experiment must record:

- research hypothesis;
- GB role;
- dataset and split;
- split_seed;
- noise type and noise rate;
- pretrain_seed;
- noise_seed;
- classifier_seed;
- GRACE pretraining config;
- classifier config;
- ball generation config;
- label smoothing formula;
- baselines;
- raw logs;
- result table;
- mean and standard deviation;
- paired seed deltas;
- diagnostics;
- corrupted train-label file path or reproducible noise config;
- local baseline matching rule;
- small-ball fallback rule;
- effective smoothing coverage;
- success or failure decision.

Missing records make the experiment unusable as paper evidence.

## 14. Recommended Directory Structure

Use this folder structure for future implementation:

```text
noisy_label_gb_grace/
  AGENT.MD
  README.md
  configs/
  src/
  scripts/
  logs/
  results/
  reports/
  checkpoints/
```

Do not create or modify these folders until they are needed.

## 15. Paper Claim Readiness

### Not Ready

```text
GB improves GRACE.
GB significantly improves clean node classification.
Same-ball membership captures true node semantics.
GB local smoothing is better than all local smoothing methods.
```

### Weak, If Gate Passes

```text
GB local label smoothing can provide a conservative local denoising prior after GRACE pretraining.
```

### Moderate, If Full Experiment Passes

```text
Under noisy training labels, granular balls constructed in GRACE representation space improve node classification robustness through confidence-weighted local label smoothing.
```

### Strong, Only With Full Evidence

Requires:

- Cora, CiteSeer, PubMed;
- symmetric, asymmetric, and instance-dependent noise;
- 5 seeds;
- mean/std;
- paired seed deltas;
- kmeans and no-ball topk controls;
- confidence ablation;
- no label leakage;
- clean validation and clean test labels.

## 16. Immediate Next Task

The next task is the frozen-encoder gate experiment. Before running the full
gate, the implementation must include coverage, effective-alpha, and fair
local-baseline diagnostics.

Minimum plan:

```text
1. Identify reusable GRACE code or decide whether to copy minimal code into this folder.
2. Define noise injection utilities.
3. Define frozen-encoder classifier protocol.
4. Define GB, kmeans, no-ball topk smoothing modules.
5. Define result table schema.
6. Add small-ball fallback and smoothing coverage diagnostics.
7. Match kmeans/no-ball local baseline granularity to GB.
8. Run Cora/CiteSeer symmetric 0%, 20%, and 40% noise with 5 seeds.
```

Before copying or modifying any existing code outside this folder, ask the user for approval.

### 16.1 Formal Gate Addendum

For the formal frozen gate, record these additional diagnostics:

```text
avg_train_nodes_per_ball
avg_train_nodes_per_nonempty_ball
balls_with_0_train_nodes
balls_with_1_train_node
balls_with_2_train_nodes
balls_with_3plus_train_nodes
alpha_mean_all_train
alpha_mean_nonzero
alpha_max_actual
GB-kmeans assignment NMI
GB-no-ball selected-neighbor overlap
```

Use `GB-strict` as the main method:

```text
if |B_train| < 3:
    alpha_i = 0
```

Add `GB-prior` only as an auxiliary diagnostic method:

```text
p_B = (count_B + gamma * p_global) / (|B_train| + gamma)
```

Do not present `GB-prior` as the main contribution unless `GB-strict` is
reported separately.

Use a fair no-ball local baseline:

```text
topk_matched k = round(mean(|B_train| - 1 for eligible GB balls)),
with k >= 1.
```

For formal reporting:

```text
primary alpha = 0.3
alpha = 0.5 is sensitivity / ablation only
```

The `topk_matched` baseline must use:

```text
k = round(mean(|B_train| - 1 for eligible GB balls))
k = max(k, 1)
candidate pool = train nodes only
self node = excluded from top-k neighbors
local labels = same corrupted train-label artifact as all other methods
```

Skip/resume completeness must be checked by:

```text
dataset, split_seed, pretrain_seed, noise_seed, classifier_seed,
noise_type, noise_rate, method, alpha
```

All noise rates, including 0%, must save a label artifact. For 0%, the artifact
records zero flipped train labels.

Add these diagnostics:

```text
effective_train_nodes_smoothed
effective_smoothing_coverage_by_class
```

If primary-alpha `GB-strict` effective smoothing coverage averages below 0.30
on Cora/CiteSeer, mark the Planetoid setting as label-sparse and plan a
separate medium-label protocol before drawing a negative mechanism conclusion.

The next command target is:

```powershell
python scripts/run_gate_experiment.py --datasets Cora CiteSeer --noise-rates 0 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 0.5 --save-diagnostics
python scripts/summarize_gate_results.py
```

### 16.2 Formal Gate Result: Planetoid Low-label Split

Formal frozen gate on Cora and CiteSeer under the Planetoid split shows weak
but insufficient `GB-strict` signal.

Observed:

```text
Cora:
  GB-strict has positive paired deltas against global and topk_matched under
  20% and 40% symmetric noise.

CiteSeer:
  GB-strict is limited by extremely low effective smoothing coverage.

Effective coverage:
  Cora ~= 0.197
  CiteSeer ~= 0.075
```

Decision:

```text
Do not claim GB-strict gate success.
Do not proceed to PubMed, asymmetric noise, instance-dependent noise, or
light fine-tuning from this result.
Do not use GB-prior as the main method claim.
Do not reduce min_train_per_ball to 1 just to raise coverage.
```

Interpretation:

```text
The current evidence does not prove that GB local smoothing is ineffective.
The main failure mode is that Planetoid low-label splits provide too few
labeled train nodes per ball to estimate reliable local label distributions.
```

Next experiment:

```text
Run a medium-label frozen gate with ratio splits.
Keep it as a separate protocol from Planetoid-low-label.
```

Medium-label protocol:

```text
datasets = Cora, CiteSeer
split_mode = ratio
train_ratio = 0.1, 0.2
val_ratio = 0.1
test = remaining nodes
noise_type = symmetric
noise_rate = 0.0, 0.2, 0.4
seeds = 0, 1, 2, 3, 4
methods = ce, global, kmeans, topk_matched, gb, gb_prior
primary alpha = 0.3
alpha = 0.5 sensitivity only
```

Medium-label success criteria:

```text
Strong pass:
  GB-strict has stable positive paired deltas against global, kmeans, and
  topk_matched under 20% or 40% noise on both Cora and CiteSeer, with
  effective coverage >= 0.3.

Weak pass:
  GB-strict beats global and is roughly tied with kmeans/topk_matched, while
  coverage is clearly higher than Planetoid-low-label and paired deltas are
  mostly positive.

Fail:
  Coverage improves but GB-strict is still consistently matched or beaten by
  kmeans/topk_matched. In that case, stop or downgrade the GB-strict claim.
```

Target command:

```powershell
python scripts/run_gate_experiment.py --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.1 0.2 --val-ratio 0.1 --noise-rates 0 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 0.5 --save-diagnostics
python scripts/summarize_gate_results.py
```

### 16.3 Medium-label Frozen Gate Result

Medium-label ratio splits increased effective `GB-strict` coverage, but did
not validate the strict GB local smoothing claim across datasets.

Observed:

```text
Coverage improved:
  Cora train 10%:     ~= 0.368
  Cora train 20%:     ~= 0.638
  CiteSeer train 10%: ~= 0.311
  CiteSeer train 20%: ~= 0.580

Cora:
  train 20%, 40% noise is the strongest positive setting.
  GB-strict beats CE, global, kmeans, and topk_matched on average there.

CiteSeer:
  GB-strict remains unstable and is usually matched or beaten by CE, global,
  or kmeans even after coverage improves.
```

Decision:

```text
Do not claim GB-strict local smoothing is validated.
Do not proceed to PubMed, asymmetric noise, instance-dependent noise, or
fine-tuning from this result.
Downgrade GB-strict to a Cora-positive but cross-dataset-unstable signal.
```

Interpretation:

```text
The Planetoid failure was partly caused by low coverage, but medium-label
results show that improving coverage alone is not enough to establish a
GB-specific advantage over kmeans/topk controls.
```

Recommended next direction:

```text
Stop expanding GB-strict as the primary method.
If continuing, reframe the next experiment around coverage-aware or
prior-corrected local smoothing with a new hypothesis and new acceptance
criteria. Do not present it as a direct validation of strict ball-only local
label smoothing.
```
