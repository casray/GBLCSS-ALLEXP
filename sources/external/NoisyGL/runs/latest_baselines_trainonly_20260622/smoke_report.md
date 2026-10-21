# Latest Baselines Smoke Report

## Protocol

- Datasets: Cora and Coauthor-CS
- Split: 20% train / 10% validation / 70% test
- Noise: uniform 0.4 on train labels only
- Seed: 0
- Epochs: Cora 2; Coauthor-CS 1
- Environments: Python 3.12 CPU and PyTorch 2.9.1 ROCm 7.2.1

## Result

| Environment | Dataset | TFR | ProCon | GD2 |
|---|---|---|---|---|
| CPU | Cora | OK | OK | OK |
| CPU | Coauthor-CS | OK | OK | OK |
| ROCm | Cora | OK | OK | OK |
| ROCm | Coauthor-CS | OK | OK | OK |

Protocol audit passed in both environments:

- `modified_val_count = 0`
- `modified_test_count = 0`
- identical split/noise hashes for methods sharing a dataset and seed

## Interpretation

This establishes execution compatibility and bounded-memory feasibility only. Accuracy from a one- or two-epoch smoke run must not enter the paper or method comparison.

## Decision

Proceed to the pre-registered 12-configuration seed0 tuning grid on ROCm. Freeze the selected configuration before five-seed formal runs.
