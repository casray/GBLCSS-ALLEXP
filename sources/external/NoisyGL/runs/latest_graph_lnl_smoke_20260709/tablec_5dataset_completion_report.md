# Table C Latest Graph-LNL Baselines Completion Report

## Hypothesis

Recent graph-dependent noisy-label baselines (`TFR`, `ProCon`, `GD2`) can be evaluated under the paper's local train-only noise protocol on all five Table C datasets.

## Granular-Ball Role

Granular balls are not used by these baselines. The experiment positions LCSS and LCSS-distill against recent graph-dependent label-noise methods.

## Motivation

The previous Table C had complete coverage for older graph baselines but only partial coverage for newer 2025 methods. This run fills Cora, CiteSeer, PubMed, Coauthor-CS, and Amazon-Photo under one protocol.

## Code / Config Changes

No algorithm code was changed.

Two GD2 config files were added because the local GD2 implementation previously had configs only for Cora, CiteSeer, and Coauthor-CS:

- `config/gd2/gd2_pubmed.yaml`
- `config/gd2/gd2_amazonpho.yaml`

Both follow the existing Coauthor-CS GD2 template with `n_hidden=256`.

## Protocol

- Datasets: Cora, CiteSeer, PubMed, Coauthor-CS, Amazon-Photo.
- Methods: `tfr_gcn`, `procon`, `gd2`.
- Seeds: 0, 1, 2, 3, 4.
- Noise: uniform 0.4.
- Noise scope: train-only.
- Split: percent, train/val/test = 0.2/0.1/0.7.
- Device: `cuda:0` through ROCm Python.

## Output Files

- `uniform0p4_5seed.csv`: original 3-dataset 5-seed table.
- `pubmed_amazon_uniform0p4_5seed.csv`: PubMed/Amazon-Photo 5-seed table.
- `tablec_latest_graph_lnl_uniform0p4_5dataset_5seed.csv`: merged 5-dataset raw table.
- `tablec_latest_graph_lnl_aggregate.csv`: mean/std aggregate.
- `tablec_latest_graph_lnl_paired.csv`: per-seed paired table.

## Verification

- Total rows: 75.
- Status: all `ok`.
- Every dataset-method pair has 5 seeds.
- `modified_val_count` sum = 0.
- `modified_test_count` sum = 0.
- `modified_count == modified_train_count` for all rows.

Therefore the full Table C completion run follows the intended train-only noisy-label protocol.

## Results

| Dataset | TFR-GCN | ProCon | GD2 |
|---|---:|---:|---:|
| Cora | 78.75 ± 0.11 | 77.19 ± 2.17 | 76.89 ± 3.51 |
| CiteSeer | 68.92 ± 1.77 | 61.07 ± 3.32 | 63.07 ± 1.58 |
| PubMed | 82.24 ± 0.83 | 50.10 ± 13.27 | 83.13 ± 0.70 |
| Coauthor-CS | 78.36 ± 2.39 | 86.08 ± 6.85 | 91.57 ± 0.18 |
| Amazon-Photo | 35.36 ± 9.43 | 20.74 ± 4.82 | 91.04 ± 1.00 |

## Interpretation

- TFR-GCN is strongest on Cora and CiteSeer among these three recent references.
- GD2 is strongest on PubMed, Coauthor-CS, and Amazon-Photo, with especially stable results on Coauthor-CS and Amazon-Photo.
- ProCon is unstable under this local adapted setup and performs poorly on PubMed/Amazon-Photo. It should not be described as a strong reproduced baseline here.
- TFR-GCN is also unstable or poorly transferred on Amazon-Photo under this local adapted config.

## Claim Boundary

These numbers are same-protocol local adapted/reimplemented references, not official reproductions. They are suitable for a graph-dependent reference table but should not be merged into the same-resource X-only main table.

## Final Decision

Table C can now be completed for `TFR-GCN`, `ProCon`, and `GD2` on all five datasets. The most defensible wording is: "we include recent graph-dependent noisy-label references under our protocol; GD2 is the strongest recent reference on three of five datasets, while TFR-GCN is strongest on Cora/CiteSeer."
