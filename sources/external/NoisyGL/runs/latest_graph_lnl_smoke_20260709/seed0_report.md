# Latest Graph-LNL Baselines Seed0 Smoke

## Hypothesis

Recent graph label-noise baselines (`TFR`, `ProCon`, `GD2`) can be evaluated under the paper's local protocol and used as graph-dependent reference methods if they run cleanly with train-only noise.

## Protocol

- Datasets: Cora, CiteSeer, Coauthor-CS.
- Methods: `tfr_gcn`, `procon`, `gd2`.
- Seed: 0.
- Noise: uniform 0.4, train-only.
- Split: percent, train/val/test = 0.2/0.1/0.7.
- Device: `cuda:0` through ROCm Python.
- Output CSV: `seed0.csv`.

## Granular-Ball Role

Granular balls are not used by these baselines. This experiment only positions LCSS against recent graph-dependent noisy-label graph methods.

## Code / Config Changes

No code or config changes were made for this smoke run.

## Sanity Checks

- Rows: 9/9 completed.
- Status: all `ok`.
- `modified_val_count`: all 0.
- `modified_test_count`: all 0.
- `modified_count == modified_train_count` for all rows.

Therefore the run follows the intended train-only noise protocol.

## Seed0 Results

| Dataset | Method | Test Acc |
|---|---|---:|
| Cora | TFR-GCN | 78.84 |
| Cora | ProCon | 76.20 |
| Cora | GD2 | 74.62 |
| CiteSeer | TFR-GCN | 67.23 |
| CiteSeer | ProCon | 60.74 |
| CiteSeer | GD2 | 63.79 |
| Coauthor-CS | TFR-GCN | 81.23 |
| Coauthor-CS | ProCon | 92.28 |
| Coauthor-CS | GD2 | 91.69 |

## Decision

The smoke run passed. `tfr_gcn`, `procon`, and `gd2` are runnable under the local protocol. If these methods are needed for a paper table, the next step is a 5-seed run for the same datasets. These numbers should be described as local adapted/reimplemented same-protocol references, not official reproductions.
