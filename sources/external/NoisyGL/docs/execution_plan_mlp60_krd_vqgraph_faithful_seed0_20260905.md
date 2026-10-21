# MLP-60: KRD and VQGraph faithful protocol-port seed-0 smoke

## Research hypothesis

The weak KRD and VQGraph rows in the current comparison may be caused by the
earlier mechanism adaptations rather than by the published methods. Replacing
those adaptations with faithful ports of the official training objectives can
materially change their performance under the paper's noisy-label protocol.

## Role of granular balls

Granular balls are not used by KRD or VQGraph. The experiment only audits two
comparison routes against the existing GB-LCSS teacher result under the same
data split, noise realization, and attribute-only student inference setting.

## Method motivation

- KRD samples teacher knowledge from each node's graph neighborhood using a
  reliability probability updated from teacher-student agreement.
- VQGraph trains a graph tokenizer with discrete vector quantization, then
  distills both class predictions and soft code assignments to an MLP.

The existing `krd_adapted` and `vqgraph_adapted` implementations do not retain
these complete pipelines. They remain unchanged and are not reused as faithful
results.

## Source identity

- KRD official repository: `LirongWu/KRD`, commit
  `32be7e42ab65b1d246cf1591d8bc48dfe7fb6b21`.
- VQGraph official repository: `YangLing0818/VQGraph`, commit
  `60aa1c99782d542c316a37f0394eba934bcf97b0`.
- Local read-only source snapshots:
  `third_party_reference/KRD` and `third_party_reference/VQGraph`.

## Code changes

- Add `predictor/G2MFaithful_Predictor.py` with new, separate
  `krd_faithful` and `vqgraph_faithful` predictors.
- Add Cora configurations using the official architecture and optimization
  values.
- Register both methods in `scripts/run_standard_downstream.py` without
  changing the existing adapted entries.
- Add unit tests that anchor the KRD edge-wise loss and sampling rule, the
  VQGraph hard quantizer and token loss, X-only student inference, source
  commits, and runner identity.

## Protocol changes relative to the official repositories

This is a faithful protocol port, not a reproduction of the papers' headline
numbers. The method-specific model, losses, sampling, and optimization schedule
are retained, while the following experiment-level conditions are replaced by
the manuscript protocol:

- Cora split: fixed percent 0.2/0.1/0.7.
- Label noise: uniform 0.4 on training labels only.
- Seed: 0.
- Model selection: clean validation labels. VQGraph's public student script
  selects checkpoints using test accuracy; that behavior is not retained.
- Student evaluation: attribute-only inference on the paper's test split.

## Expected mechanism

The faithful KRD port should differ most from `krd_adapted` because it uses
node-specific neighborhood knowledge points rather than a global reliable-node
subset. The faithful VQGraph port should differ from `vqgraph_adapted` because
the tokenizer is discrete, reconstructs both node and adjacency information,
and exposes the official soft-code distillation target.

## Smoke command

```powershell
& '<ROCm Python>' scripts/run_standard_downstream.py `
  --datasets cora `
  --methods krd_faithful,vqgraph_faithful `
  --seeds 0 `
  --noise_type uniform `
  --noise_rate 0.4 `
  --noise_scope train `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --device cuda:0 `
  --stop_on_error `
  --output runs/mlp60_krd_vqgraph_faithful_seed0_20260905/smoke.csv
```

## Success criteria

The smoke is technically valid only if:

1. both rows finish with finite train, validation, test, teacher, and
   method-specific losses;
2. both rows use the same split and noise hashes;
3. label corruption is confined to training nodes;
4. training uses the graph and unlabeled nodes, while inference is X-only;
5. KRD performs at least one post-warmup reliability update and VQGraph uses
   hard codes with a non-empty active codebook.

Accuracy is not a pass criterion. A single seed can motivate a five-seed run
but cannot support a manuscript claim.

## Failure criteria

- NaN, OOM, collapsed one-class prediction, or incomplete diagnostics.
- Split/noise mismatch between the two methods.
- Any use of test labels for checkpoint selection.
- A core method equation replaced by the earlier adaptation.

## Decision rule

- Continue to paired five-seed Cora experiments if either faithful port differs
  from its adapted seed-0 result by at least 1 percentage point, or is within
  2 points of the current granular-ball teacher seed-0 result.
- Otherwise pause expansion and retain the current rows as adaptations, with
  corrected naming only.

## Post-run status

KRD reaches 72.45% in two bitwise-identical repeats and improves over its
previous adaptation by 4.01 percentage points. Two VQGraph GPU repeats with
the same seed reach 64.91% and 66.91%; both are below the previous soft-token
proxy, and their 2.01-point spread prevents treating either value as a stable
seed-0 estimate. A separate clean-label VQGraph run reaches 82.48%, confirming
that the final port trains normally before label noise is introduced. The full
interpretation and next step are recorded in
`runs/mlp60_krd_vqgraph_faithful_seed0_20260905/mlp60_report.md`.
