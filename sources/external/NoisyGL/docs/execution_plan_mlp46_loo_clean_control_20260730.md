# MLP-46 Tie-Safe LOO Clean-Label Control

## Goal

Measure whether the final leave-one-out GB-LCSS target incurs a clean-label
accuracy cost. This is a descriptive control and does not affect the MLP-44
promotion decision.

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Split: percent 0.2/0.1/0.7.
- Noise: uniform 0.0, train scope.
- Seeds: 0--4.
- Methods:
  - `frozengrace_gb_mlp_softdist_beta05_loo`
  - `frozengrace_gb_mlp_softdist_beta05_loo_shuffled`
  - `mlp_ce`
- Formal LOO lineage:
  `tie_mode=distribution_only`, `leave_one_out=1`.
- Frozen GB and MLP hyperparameters are unchanged.

Historical self-inclusive clean-label rows are not used as final-method
evidence.

## Required outputs

`runs/mlp46_loo_clean_control_20260730/`

- `clean_uniform0p0.csv`
- `mlp46_aggregate.csv`
- `mlp46_paired_deltas.csv`
- `mlp46_protocol_audit.csv`
- `mlp46_report.md`

## Integrity checks

1. Exactly 75 successful rows:
   5 datasets x 3 methods x 5 seeds.
2. Every LOO row uses `distribution_only` and `leave_one_out=1`.
3. No validation or test label is modified.
4. All three methods share the same split and noise hashes within each
   dataset and seed.
5. `modified_count=modified_train_count=0`.

## Interpretation

- Report LOO minus MLP+CE and LOO minus LOO-shuffled with mean, standard
  deviation, and paired seed deltas.
- A negative delta is a clean-label trade-off, not a failure of the
  noisy-label promotion gate.
- Do not tune the method from this control.

