# Source and adaptation audit

## Official code provenance

- DIST: https://github.com/hunto/DIST_KD links its classification loss to
  https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py
  Snapshot commit: `6f7a70a1dd57fd88413d83909d3d53243068f701`.
- DGKD: https://github.com/xsk160/DGKD
  Snapshot commit: `ae8a495a054578d4b5e36aa5e7d3a3f4872037f6`.
- All downloaded files are unmodified and SHA-256-pinned in `frozen_inputs.json`.
  These are local reference copies, not a claim that the full upstream training
  environments were reproduced. The only imported upstream code is the loss.

## DGKD target-class handling

In this source snapshot, `dgkd_train_student.py` lines 297-299 set `idx_t` to the
concatenation of train, validation and test nodes. `dgkd_train_and_eval.py` line
369 sets `target_t = labels[idx_t]` and passes it to the DGKD loss at lines
376-377. Our protocol does not permit this training path. This is a statement
about the inspected snapshot, not an audit of every experiment in the paper.

This screen uses teacher argmax for every target-class mask, including training
nodes. Hence DGKD-T is a teacher-only adaptation. It neither reads true test
labels nor adds observed-label information that the KD/DIST arms do not use.
The native DGKD weights (alpha 15, beta 3) and temperature .7 are taken from
`dgkd_train_and_eval.py` line 113. Temperature 1 is a prespecified alternative.

## Other controlled changes

DIST is adapted from image-classification loss to a full batch of graph nodes.
Both methods use the existing MLP rather than their original model or training
schedule. No noisy-label CE term is added. This tests loss mechanisms under the
user's protocol, not reported benchmark accuracies from the source papers.

The teacher cache contains probabilities at temperature 1. Log probabilities
recover logits up to a per-node additive constant, which does not change the
softmax at any temperature. Every cached probability is positive, as recorded
in `environment.json`; no underflowed zero needs approximation in this grid.

Eight tests passed before freezing: CPU and GPU losses/gradients against
official code, teacher detachment, class permutations, temperatures, finite
edge cases, and validation-only selection with deterministic ties. Ordinary KD
and +GB replay both reproduced prior probabilities bit-for-bit, including the
entire validation trajectory and best epoch (`replay_check.json`).

## Interpretation boundaries

Native loss scales differ. Regional coefficient 1 is inherited unchanged; its
gradient contribution is observed, not presumed comparable across methods.
Negative +GB results for one fixed coefficient do not rule out every possible
regional design. Positive results still need full-protocol seeds and another
teacher. No test-driven parameter change or follow-on grid is authorized by
this screen's continue gate.
