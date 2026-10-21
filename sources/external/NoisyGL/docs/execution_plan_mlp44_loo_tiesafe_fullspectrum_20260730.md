# MLP-44 Tie-Safe LOO-GB-LCSS Full-Spectrum Confirmation

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Determine whether tie-safe leave-one-out GB-LCSS should replace the
current self-inclusive GB-LCSS as the formal train-target method.

**Architecture:** Reuse only the existing tie-safe v2 uniform-0.4 rows. Run
the four missing noise settings with the frozen LOO real and size-matched
shuffled predictors. Merge them with the immutable tie-safe v2 anchor grid,
reject mixed-lineage or protocol-mismatched rows, and apply the preregistered
decision gates.

**Tech Stack:** Python, PyTorch/ROCm, pandas-free CSV summaries, pytest.

---

## Research hypothesis

Removing each target node's own noisy label from its granular-ball vote
reduces self-confirmation while retaining the useful signal from other
training nodes in the same label-blind ball.

## Granular-ball role

The frozen GRACE representation and granular-ball assignment are unchanged.
For a train node \(v\) in ball \(B_j\), the LOO method removes \(v\)'s own
vote before computing the local distribution, majority set, confidence, and
soft target. The resulting quantities are node-dependent:

\[
p_{j,-v},\qquad q_{j,-v},\qquad \alpha_{j,-v},\qquad t_v.
\]

## Frozen protocol

- Datasets: Cora, CiteSeer, CoraFull, Coauthor-CS, Amazon-Photo.
- Split: percent 0.2/0.1/0.7.
- Noise scope: train only.
- Settings: uniform 0.2/0.4/0.6, pair 0.4, instance 0.4.
- Seeds: 0--4.
- Real method: `frozengrace_gb_mlp_softdist_beta05_loo`.
- Random control:
  `frozengrace_gb_mlp_softdist_beta05_loo_shuffled`.
- Required target lineage:
  `tie_mode=distribution_only`, `leave_one_out=1`.
- Frozen GB parameters:
  `max_balls=64`, `min_ball_size=32`,
  `split_improvement=0.05`, `target_beta=0.5`.

The historical v1 LOO results are invalid for this decision and must not be
loaded. Uniform 0.4 is reused only from:

`runs/v31_tiesafe_affected_20260728/loo_uniform0p4.csv`

All other settings are newly generated under the v2 rule.

## Required outputs

Run directory:

`runs/mlp44_loo_tiesafe_fullspectrum_20260730/`

- `loo_uniform0p2.csv`
- `loo_uniform0p6.csv`
- `loo_pair0p4.csv`
- `loo_instance0p4.csv`
- `mlp44_combined_raw.csv`
- `mlp44_aggregate.csv`
- `mlp44_seed_deltas.csv`
- `mlp44_setting_deltas.csv`
- `mlp44_dataset_deltas.csv`
- `mlp44_gate_decision.csv`
- `mlp44_report.md`

## Integrity requirements

1. Every formal LOO row has `tie_mode=distribution_only` and
   `leave_one_out=1`.
2. Every LOO real/shuffled row has `status=ok`.
3. `modified_val_count=modified_test_count=0`.
4. `modified_count=modified_train_count`.
5. Within each dataset/setting/seed, all six compared methods have identical
   split and noise hashes.
6. The final matrix contains exactly
   `5 datasets x 5 settings x 5 seeds x 6 methods = 750` rows after anchor
   merge, including 250 LOO rows and 500 immutable anchor rows.
7. Duplicates, missing seeds, wrong noise settings, v1 tie rules, or a
   non-LOO target are fatal errors.

## Preregistered decision gates

LOO becomes the formal GB-LCSS target only when all gates pass:

1. The global mean paired delta `LOO - base` over the 25 dataset-setting
   means is strictly positive.
2. At least 4/5 datasets have non-negative cross-setting mean
   `LOO - base`.
3. No dataset-setting mean `LOO - base` is below `-1.0pp`.
4. `LOO real - LOO shuffled` is positive in at least 24/25
   dataset-settings, and at least 20/25 settings have 4/5 or more positive
   paired seeds.

The LOO comparisons against MLP+CE and JoCoR, target coverage, NCR, and CDR
are reported descriptively and are not used to change these gates.

## Decision branches

- **PASS:** redefine the formal method with
  \(p_{j,-v},q_{j,-v},\alpha_{j,-v}\); retain the self-inclusive version as an
  ablation; use the LOO definition in later Pareto experiments.
- **FAIL:** retain self-inclusive GB-LCSS; keep LOO as a visible mechanism
  ablation; do not tune a second LOO variant.

The teacher-distillation extension is not automatically changed. The prior
MLP-38 LOO-teacher smoke did not pass its preregistered expansion gate.

## Commands

Each missing setting uses:

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  scripts/run_standard_downstream.py `
  --datasets cora,citeseer,corafull,coauthorcs,amazonpho `
  --methods frozengrace_gb_mlp_softdist_beta05_loo,frozengrace_gb_mlp_softdist_beta05_loo_shuffled `
  --seeds 0,1,2,3,4 `
  --noise_type <uniform|pair|instance> `
  --noise_rate <0.2|0.4|0.6> `
  --noise_scope train `
  --device cuda:0 `
  --split_type percent `
  --train_percent 0.2 `
  --val_percent 0.1 `
  --test_percent 0.7 `
  --output runs/mlp44_loo_tiesafe_fullspectrum_20260730/<output>.csv `
  --stop_on_error
```

The summary command is:

```powershell
& "C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe" `
  scripts/summarize_mlp44_loo_tiesafe_fullspectrum.py
```

