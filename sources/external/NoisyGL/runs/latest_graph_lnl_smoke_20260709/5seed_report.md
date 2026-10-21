# Latest Graph-LNL Baselines 5-Seed Report

## Hypothesis

Recent graph label-noise baselines (`TFR`, `ProCon`, `GD2`) can be evaluated under the paper's local protocol and used as graph-dependent reference methods for Table C.

## Granular-Ball Role

Granular balls are not used by these baselines. This experiment positions LCSS/LCSS-distill against recent graph-dependent noisy-label graph methods.

## Motivation

The previous graph-dependent comparison table relied mainly on NoisyGL-era baselines. To address reviewer concerns about newer methods, this run evaluates 2025 graph-label-noise methods under the same local protocol.

## Code Changes

No algorithm or config changes were made for this run.

## Config / Protocol

- Datasets: Cora, CiteSeer, Coauthor-CS.
- Methods: `tfr_gcn`, `procon`, `gd2`.
- Seeds: 0, 1, 2, 3, 4.
- Noise: uniform 0.4.
- Noise scope: train-only.
- Split: percent, train/val/test = 0.2/0.1/0.7.
- Device: `cuda:0` through ROCm Python.

## Output Files

- `seed0.csv`: seed0 smoke.
- `seeds_1_4.csv`: formal seeds 1-4 run.
- `uniform0p4_5seed.csv`: merged 5-seed raw table.
- `uniform0p4_aggregate.csv`: mean/std table.
- `uniform0p4_paired.csv`: per-seed paired table.

## Verification

- Total rows: 45.
- Status: all `ok`.
- Every dataset-method pair has 5 seeds.
- `modified_val_count` sum = 0.
- `modified_test_count` sum = 0.
- `modified_count == modified_train_count` for all rows.

Therefore the experiment follows the intended train-only noisy-label protocol and does not repeat the previous val/test-noise contamination issue.

## Results

| Dataset | Method | Test Acc (%) |
|---|---|---:|
| Cora | TFR-GCN | 78.75 ± 0.11 |
| Cora | ProCon | 77.19 ± 2.17 |
| Cora | GD2 | 76.89 ± 3.51 |
| CiteSeer | TFR-GCN | 68.92 ± 1.77 |
| CiteSeer | ProCon | 61.07 ± 3.32 |
| CiteSeer | GD2 | 63.07 ± 1.58 |
| Coauthor-CS | TFR-GCN | 78.36 ± 2.39 |
| Coauthor-CS | ProCon | 86.08 ± 6.85 |
| Coauthor-CS | GD2 | 91.57 ± 0.18 |

## Interpretation

- TFR-GCN is the strongest of the three on Cora and CiteSeer under this local protocol.
- GD2 is the strongest of the three on Coauthor-CS and is very stable there.
- ProCon is unstable on Coauthor-CS under this local adapted setup, with a large standard deviation.
- These methods should be reported as graph-dependent reference methods, not as same-resource baselines for the X-only main table.

## Claim Boundary

These are local adapted/reimplemented same-protocol references. They should not be described as official reproductions unless each implementation is audited against the original authors' code and protocol.

## Final Decision

The 5-seed run is complete and suitable for updating the graph-dependent Table C. Use the numbers as same-protocol graph-dependent references with an explicit note that `TFR-GCN`, `ProCon`, and `GD2` require graph-dependent training/inference, while LCSS-distill deploys an X-only student.
