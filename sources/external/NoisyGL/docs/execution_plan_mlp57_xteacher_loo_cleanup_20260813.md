# MLP-57 X-only Ball Teacher and Formal LOO Diagnostics Plan

Date: 2026-08-13

## Research questions

1. Does a ball-consensus target improve a `GCN(X)` teacher when the teacher input is held fixed?
2. Do the validation-protocol and frozen-GRACE diagnostics still hold for the formal tie-safe leave-one-out GB-LCSS method?

## Method roles

- The teacher-input experiment keeps the existing ball target (`target_beta=0`) and changes only the teacher input from `[X||H]` to `X`.
- The LOO diagnostics use `tie_mode=distribution_only` and `leave_one_out=true` for both real and size-matched shuffled groups.
- GRACE embeddings remain label-free frozen preprocessing. They are reused and are not retrained in this experiment.

## Fixed protocol

- Datasets: Cora, CiteSeer, CoraFull.
- Split: percent 0.2/0.1/0.7.
- Training-label noise: uniform 0.4 and 0.6 for the teacher-input experiment; uniform 0.4 for LOO diagnostics.
- Seeds: 0--4 for the teacher-input and validation-protocol experiments.
- GRACE seed diagnostic: H seeds 0--2 crossed with downstream seeds 0--2.
- Device: ROCm Python, `cuda:0`.
- No changes to GRACE, ball geometry, MLP, optimizer, target beta, or distillation temperature.

## Phase A: `GCN(X)` ball-consensus teacher

New method: `mlp_distill_softdist_x`.

The method is identical to `mlp_distill_softdist_xh` except:

```text
teacher_use_xh=false
```

Matrix: 3 datasets x 2 noise rates x 5 seeds = 30 new teacher/student runs.

Controls are reused only when dataset, split, noise and seed fields match:

- `mlp_distill_gcn`: `GCN(X)+CE` teacher.
- `mlp_distill_gcn_xh`: `GCN[X||H]+CE` teacher.
- `mlp_distill_softdist_xh`: `GCN[X||H]+ball distribution` teacher.

### Preregistered decision

The X-input ball teacher becomes the main ball-teacher route only if, at each reported noise rate:

1. its mean student accuracy exceeds `mlp_distill_gcn` on at least 2/3 datasets;
2. the paired direction is positive on at least 4/5 seeds for each passing dataset;
3. the three-dataset mean gain is at least 1 percentage point.

Otherwise, `[X||H]` remains the complete high-resource teacher route, and the new result is reported as an input-controlled attribution experiment. The `[X||H] - X` difference is descriptive and is not a model-selection gate.

## Phase B: formal LOO validation-protocol sensitivity

Run the formal real and shuffled methods under clean-validation and noisy-validation protocols:

- `frozengrace_gb_mlp_softdist_beta05_loo`
- `frozengrace_gb_mlp_softdist_beta05_loo_shuffled`

Only these 60 rows are new. Existing MLP+CE, JoCoR, PosteL and GSDN rows may be reused after split, train-noise and validation-noise hashes are audited.

Report absolute accuracy, paired real-minus-shuffled differences, method rankings and Kendall tau. No result is dropped based on ranking.

## Phase C: formal LOO frozen-representation diagnostics

Reuse the existing MLP-37 H and checkpoints.

- GRACE seed matrix: 3 datasets x 3 H seeds x 3 downstream seeds x real/shuffled = 54 LOO runs.
- Representation-source comparison: reuse existing raw-X, A2X and supervised-GCN representation artifacts, but rebuild targets with formal LOO. Any collapsed one-ball source remains reported as collapsed.

The original MLP-37 robustness gates are retained. Results from the old self-inclusive target are not mixed with the formal LOO summary.

## Phase D: manuscript and evidence update

- Keep one non-LOO row only in the leave-one-out ablation and rename it `self-inclusive ablation`.
- Replace Appendix C and D.3 with formal LOO results.
- Remove the old non-LOO six-dataset breadth appendix rather than presenting it as evidence for the formal method.
- Recompute vote-support bins after subtracting the query node's own vote.
- Create a new manuscript version; do not overwrite v56.

## Required outputs

Output root: `runs/mlp57_xteacher_loo_cleanup_20260813/`

- `xteacher_raw.csv`
- `xteacher_aggregate.csv`
- `xteacher_paired_deltas.csv`
- `xteacher_report.md`
- `loo_validation_raw.csv`
- `loo_validation_report.md`
- `loo_grace_seed_raw.csv`
- `loo_grace_seed_report.md`
- `loo_representation_raw.csv`
- `loo_representation_report.md`
- `completeness.csv`
- `protocol_audit.csv`

## Failure conditions

- Any validation/test label used in target construction.
- Any val/test label modification outside the explicitly noisy-validation protocol.
- Missing seeds, NaN, OOM, prediction collapse, or split/noise hash mismatch.
- Teacher input diagnostics inconsistent with the declared route.
- Mixing historical self-inclusive rows into a formal LOO aggregate.

