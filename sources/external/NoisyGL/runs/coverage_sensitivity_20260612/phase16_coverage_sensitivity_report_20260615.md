# Phase 16 Coverage Sensitivity Report

Date: 2026-06-15

## Research Record

- Hypothesis: In the current SoftDist [X||H] pipeline, increasing train coverage should make ball voting more reliable, but the old C1 monotonic-gain result must be remeasured because the downstream classifier now directly consumes H.
- Granular-ball role: balls provide train-label soft targets; the classifier also sees `[X||H]`, so part of the former ball advantage may already be absorbed by the input representation.
- Code changes: no mechanism changes. Coverage runs reused the same label-free H anchor and varied only the downstream split.
- Config: Cora/CiteSeer, uniform noise 0.4, train ratios `{0.05, 0.1, 0.2, 0.4}`, val ratio 0.1, test ratio `1-train-val`, seeds `0-4`, GPU `cuda:0`.
- H label-free check: the pretraining script path uses only `(x, edge_index)` during GRACE training and metadata marks `used_labels_or_masks_in_training=False`; old copied Cora/CiteSeer artifacts lack metadata, so the report should state source-code verification plus artifact-metadata caveat.
- Raw outputs: `coverage_xh_train*.csv`, `coverage_standard_train*.csv`, `coverage_ball_diag_train*.csv`.
- Aggregates: `phase16_coverage_aggregate.csv`, `phase16_coverage_paired_deltas.csv`, `phase16_coverage_ball_diag_aggregate.csv`.

## Results

Ball increment over GCN [X||H]:

| Dataset | Train ratio | SoftDist [X||H] | GCN [X||H] | Delta | Positive seeds |
|---|---:|---:|---:|---:|---:|
| Cora | 0.05 | 0.7035 | 0.6887 | +1.48pp | 5/5 |
| Cora | 0.10 | 0.7675 | 0.7543 | +1.32pp | 4/5 |
| Cora | 0.20 | 0.8137 | 0.8027 | +1.10pp | 4/5 |
| Cora | 0.40 | 0.8387 | 0.8486 | -0.99pp | 1/5 |
| CiteSeer | 0.05 | 0.5721 | 0.5539 | +1.83pp | 4/5 |
| CiteSeer | 0.10 | 0.6613 | 0.6455 | +1.59pp | 4/5 |
| CiteSeer | 0.20 | 0.6891 | 0.6761 | +1.30pp | 5/5 |
| CiteSeer | 0.40 | 0.7162 | 0.7103 | +0.59pp | 4/5 |

Ball diagnostics:

| Dataset | Train ratio | Eligible train / ball | NCR | CDR | Weighted purity |
|---|---:|---:|---:|---:|---:|
| Cora | 0.05 | 3.35 | 0.373 | 0.223 | 0.787 |
| Cora | 0.10 | 5.32 | 0.582 | 0.217 | 0.787 |
| Cora | 0.20 | 10.33 | 0.687 | 0.202 | 0.787 |
| Cora | 0.40 | 20.67 | 0.746 | 0.214 | 0.787 |
| CiteSeer | 0.05 | 3.54 | 0.287 | 0.285 | 0.700 |
| CiteSeer | 0.10 | 5.40 | 0.449 | 0.309 | 0.700 |
| CiteSeer | 0.20 | 10.42 | 0.583 | 0.300 | 0.700 |
| CiteSeer | 0.40 | 20.85 | 0.664 | 0.291 | 0.700 |

## Decision

The current pipeline does not reproduce the old C1 monotonic coverage-gain claim. NCR increases with coverage, but CDR remains nontrivial and the classifier already receives H directly. The ball increment is strongest at low/mid coverage and saturates or turns negative at high Cora coverage.

Use in paper: replace old C1 coverage plot with this table. The correct claim is conditional: ball soft targets help most when labels are scarce enough that `[X||H]` classification alone has not saturated.
