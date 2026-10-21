# Latest Baselines Train-Only Run Manifest

Date: 2026-06-22

## Research Hypothesis

TFR, ProCon, and GD2 can be compared with SoftDist only after all methods use the same 20/10/70 split, uniform 0.4 train-only label noise, clean validation, and seeds 0-4.

## Granular-Ball Role

Granular balls appear only in the SoftDist and shuffled-control methods. TFR, ProCon, and GD2 keep their native X input and serve as external mechanism baselines.

## Motivation

The previous NoisyGL runner corrupted labels outside the training split. Its historical results are legacy and cannot support the corrected main table.

## Code and Configuration Changes

- Train-only corruption through `eligible_mask`.
- Explicit `--noise_scope`.
- TFR official-code adaptation.
- ProCon and GD2 paper reimplementations.
- Twelve fixed tuning configurations per latest method and dataset.
- Split/noise fingerprints, config IDs, trainable parameter counts, and wall time.

## Expected Mechanism

- TFR: topological feature reconstruction and high-confidence pseudo-labeling.
- ProCon: semantic peers, GMM clean selection, prototype feedback, and BGRL-style self-supervision.
- GD2: ego/structure prediction discrepancy, GMM clean selection, and mixed-view pseudo-label purification.

## Success and Failure Criteria

- Smoke success: all three methods finish Cora 2 epochs and Coauthor-CS 1 epoch without OOM; validation/test corruption counts remain zero.
- Formal success: all five seeds complete under matching split/noise hashes.
- Fewer than five successful seeds: report incomplete, without mean/std.
- OOM or timeout: record failure without changing the method to obtain a result.

## Commands

```powershell
# CPU smoke
python scripts/run_latest_baseline_experiments.py --mode smoke --device cpu

# ROCm smoke
python scripts/run_latest_baseline_experiments.py --mode smoke --device cuda:0 `
  --output_dir runs/latest_baselines_trainonly_20260622/rocm_smoke

# Formal sequence
python scripts/run_latest_baseline_experiments.py --mode tune --device cuda:0
python scripts/run_latest_baseline_experiments.py --mode formal --device cuda:0
python scripts/run_latest_baseline_experiments.py --mode appendix --device cuda:0
python scripts/run_latest_baseline_experiments.py --mode summarize --device cuda:0
```

## Current Result and Decision

- CPU smoke: 6/6 successful rows.
- ROCm smoke: 6/6 successful rows.
- All successful rows have zero validation/test label modifications.
- Coauthor-CS bounded ProCon peer construction and GD2 three-view training complete without OOM.
- Decision: continue to formal tuning on ROCm. Smoke accuracies are not paper evidence.
