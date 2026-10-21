# Phase 37 Execution Log

Date: 2026-06-17

## Scope

This run follows `docs/execution_plan_applicability_grid_20260617.md`.
The goal is to build a multi-variable applicability grid for SoftDist / GBSelect without changing the main mechanism.

## Phase 37a Status

- Added and smoke-tested NoisyGL configs for `coauthorph` and `wikics`.
- Added upstream GRACE loader aliases for `coauthorph` / `wikics`.
- Added H path aliases in `scripts/linear_probe_embeddings.py`.
- Loader smoke passed:
  - `coauthorph`: 34,493 nodes, 8,415 features, 5 classes.
  - `wikics`: 11,701 nodes, 300 features, 10 classes.
- GRACE config fallback is preregistered:
  - `coauthorph` uses the DBLP GRACE config unchanged.
  - `wikics` uses the PubMed GRACE config unchanged.

## Pretraining Feasibility

- `coauthorph` full-matrix GRACE one-epoch smoke did not finish within 900 seconds on the ROCm environment.
- This is treated as an implementation-level infeasibility of the current full-matrix GRACE pretraining path for Coauthor-Physics, not as a downstream negative result.
- `wikics` one-epoch GRACE smoke passed on ROCm:
  - Config source: PubMed.
  - Epochs: 1500.
  - One smoke epoch: about 2.5 seconds.
- `wikics` full GRACE pretraining is running in the background:
  - stdout: `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2\phase37_logs\wikics_grace_seed0.out.log`
  - stderr: `C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2\phase37_logs\wikics_grace_seed0.err.log`

## Current Outputs

- `applicability_method_aggregate.csv`
- `applicability_paired_deltas.csv`
- `applicability_grid_summary.csv`
- `applicability_correlations.csv`
- `correlation_analysis.md`

The first grid is a partial table assembled from existing runs plus metadata. WikiCS downstream rows and Coauthor-Physics H-dependent rows are intentionally blank until H exists.

## WikiCS Seed0 Smoke

GRACE H was generated successfully:

- `H.pt`: `(11701, 256)`
- Node alignment with NoisyGL WikiCS: passed.
- Mean row L2 norm: `1.0`.

Seed0 downstream results:

| Method | Test acc |
|---|---:|
| GCN+CE | 0.7842 |
| GCN+CE [X||H] | 0.7827 |
| SoftDist [X||H] | 0.7645 |
| SoftDist [X||H] shuffled | 0.7757 |
| NRGNN | 0.7263 |
| NRGNN [X||H] | 0.7114 |
| GBSelect | 0.6546 |
| GBSelect GCN-only | 0.7057 |
| GBSelect shuffled | 0.6225 |

Paired seed0 deltas:

- SoftDist - GCN[X||H]: `-1.82pp`.
- SoftDist - shuffled: `-1.12pp`.
- GBSelect - GCN-only: `-5.12pp`.
- GBSelect - shuffled: `+3.21pp`.

WikiCS therefore fails both SoftDist and GBSelect smoke gates. The real ball structure is better than shuffled for GBSelect, but it is not a useful second view beyond GCN-only selection.

## WikiCS 5-Seed Confirmation

The ball-related WikiCS rows were expanded to seeds 0-4.

Aggregate:

| Method | Test acc |
|---|---:|
| GCN+CE | 0.7827 +- 0.0039 |
| GCN clean | 0.8137 +- 0.0021 |
| GCN+CE [X||H] | 0.7820 +- 0.0046 |
| SoftDist [X||H] | 0.7733 +- 0.0075 |
| SoftDist [X||H] shuffled | 0.7777 +- 0.0020 |
| GBSelect | 0.6509 +- 0.0179 |
| GBSelect GCN-only | 0.6934 +- 0.0153 |
| GBSelect shuffled | 0.6277 +- 0.0203 |

Paired 5-seed deltas:

- SoftDist - GCN[X||H]: `-0.87pp`, `1/5` positive seeds.
- SoftDist - shuffled: `-0.44pp`, `2/5` positive seeds.
- GBSelect - GCN-only: `-4.25pp`, `0/5` positive seeds.
- GBSelect - shuffled: `+2.32pp`, `5/5` positive seeds.
- GCN clean - GCN noisy: `+3.10pp`.

Decision: WikiCS is a stable failure/boundary dataset for both SoftDist and GBSelect under the current mechanism.

## WikiCS Orthogonality Diagnostic

Seed0 diagnostic:

- Ball marginal AUC: `0.6590`.
- GCN warmup AUC: `0.8587`.
- Ball AUC inside GCN-clean50: `0.5282`.
- True all-node ball purity: `0.2724`.

Interpretation: the ball score has weak marginal signal, but almost no conditional orthogonal signal once the GCN-clean subset is fixed. This matches the downstream failure and argues against treating WikiCS as a promising expansion dataset under the current mechanism.

## Noise-Harm Seed0 Diagnostic

Clean-vs-noisy GCN seed0 was added for all 9 datasets. Current seed0 noise harm examples:

- Cora: `+5.38pp`.
- CiteSeer: `+9.62pp`.
- Coauthor-CS: `+1.08pp`.
- Amazon-Computers: `+3.33pp`.
- WikiCS: `+2.67pp`.
- Coauthor-Physics: `+0.99pp`.

`noise_harm` currently tracks SoftDist usefulness better than simple class count, but it uses clean-label upper-bound information and is therefore diagnostic only, not a deployable adaptive trigger.
