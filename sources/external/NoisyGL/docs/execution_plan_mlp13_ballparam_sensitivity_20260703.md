# MLP-13 Ball-Parameter Sensitivity Plan

## Research Hypothesis

The current ball construction defaults (`max_balls=64`, `min_ball_size=32`, `split_improvement=0.05`) are fixed rules with self-stopping behavior. The hypothesis is that they are close to a robust plateau for SoftDist-MLP beta=0.5, rather than a fragile dataset-tuned choice.

## Granular-Ball Role

Granular balls remain a local consensus target builder. This experiment does not change beta, target mode, backbone, split, noise protocol, or training loss. It only changes the fixed construction thresholds before the self-stopping ball split process.

## Motivation

Reviewers may ask whether the default ball construction is tuned to selected datasets. A sensitivity grid can show whether the method is robust to reasonable construction changes. If a single unified setting improves most datasets, it becomes a candidate default only after a later noise-spectrum validation.

## Code Changes

- Add `scripts/run_mlp13_ballparam_sensitivity_20260703.ps1`.
- Add `scripts/summarize_mlp13_ballparam_sensitivity.py`.
- Add `tests/test_mlp13_ballparam_sensitivity.py`.

## Config Changes

No config files are changed. The runner overrides the following values through `scripts/run_ssl_swap_downstream.py`:

- `gb_max_balls`
- `gb_min_ball_size`
- `gb_split_improvement`

## Protocol

- Method: `frozengrace_gb_mlp_softdist_beta05`
- Datasets: `cora`, `citeseer`, `pubmed`, `coauthorcs`, `amazonpho`
- Noise: train-only `uniform 0.4`
- Split: `percent`, train/val/test = `0.2/0.1/0.7`
- Stage A: seed0 full grid
  - `max_balls in {32,64,128}`
  - `min_ball_size in {16,32,64}`
  - `split_improvement in {0.01,0.05,0.1}`
- Stage B: default plus at most two globally selected non-default candidates, seeds 1-4; seed0 is reused from Stage A.

## Candidate Selection

The default `64/32/0.05` is always selected for Stage B. Non-default candidates are selected globally, not per dataset.

Hard rejection:

- Any dataset has `num_balls <= 1`.
- Any dataset has `mean_train_per_ball < 2`.
- Any dataset has `alpha_positive_train_count == 0`.
- At least two datasets lose more than 2pp vs default in Stage A seed0.

Ranking after filtering:

1. Higher mean seed0 delta vs default across all five datasets.
2. More positive-delta datasets.
3. Higher mean `noisy_train_purity_weighted`.
4. Smaller `max_balls`.

If no non-default candidate passes hard filters, the best non-degenerate candidate is selected as a negative-result confirmation.

## Success Standard

A non-default unified setting is only a candidate new default if, in Stage B, it:

- improves mean delta by at least +0.5pp,
- is non-negative on at least 4/5 datasets,
- has no dataset mean loss worse than -0.5pp.

Even then, it does not replace the main default until a later noise-spectrum validation.

## Failure / Keep-Default Standard

Keep `64/32/0.05` if it is within 1pp or within one standard deviation of the best unified Stage B setting on at least 4/5 datasets.

If a change only helps one dataset, it is appendix-only and not used as a new default.

## Required Outputs

All files are written under `runs/mlp13_ballparam_sensitivity_20260703/`:

- `mlp13_stageA_raw.csv`
- `mlp13_stageA_grid_summary.csv`
- `mlp13_stageA_selected_configs.csv`
- `mlp13_stageB_raw.csv`
- `mlp13_stageB_paired_deltas.csv`
- `mlp13_stageB_summary.csv`
- `mlp13_stageB_config_gates.csv`
- `mlp13_completeness.csv`
- `mlp13_report.md`

## Paper Claim Boundary

Use the phrase "fixed rules with self-stopping ball construction." Do not describe these thresholds as learned, adaptive, or dataset-specific hyperparameters.
