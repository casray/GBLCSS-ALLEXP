# Graph-aware GB Minival Result - 2026-06-01

## Hypothesis

Graph-aware granular balls may be better local reliable regions than ordinary embedding-only GB for noisy-label node classification, because they combine frozen-GRACE embedding similarity with k-hop topology proximity.

## Ball Role

The ball is a soft local supervision unit and a reliability region. It is not used as a hard positive label, hard negative filter, or prototype-pulling target.

## Code Changes

- Added `noisy_label_gb_grace.balls.graph_aware_ball`.
- Added `gagb`, `gagb_ca_ce`, `gagb_weighted_ce`, `khop_only`, and `gagb_shuffled`.
- Saved `edge_index` in embedding payloads.
- Added diagnostics for clean purity, noisy-label entropy, noise detection AUROC, noise correction rate, clean damage rate, and effective coverage.
- Added unit tests for graph-aware grouping, shuffled grouping, smoothing routing, weighting routing, and summary inclusion.

## Config

Command:

```powershell
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce global kmeans topk_matched gb gb_ca_ce gb_weighted_ce gagb gagb_ca_ce gagb_weighted_ce khop_only gagb_shuffled --save-diagnostics
```

Protocol:

- datasets: Cora, CiteSeer
- split: ratio train 0.2, validation 0.1, split seed 0
- noise: symmetric 0.2 and 0.4
- seeds: 0, 1, 2, 3, 4
- embedding: frozen GRACE
- graph-aware topology: k-hop, no PPR

## Results

Summary files:

- `reports/gagb_minival_20260601/gate_summary.csv`
- `reports/gagb_minival_20260601/gate_paired_deltas.csv`
- raw logs: `logs/gate/20260601_130811`

Paired accuracy deltas against ordinary `gb`:

| Dataset | Noise | Method | Delta vs gb | Positive seeds |
|---|---:|---|---:|---:|
| CiteSeer | 0.2 | gagb | +0.00034 | 3/5 |
| CiteSeer | 0.2 | gagb_ca_ce | +0.00069 | 3/5 |
| CiteSeer | 0.4 | gagb | +0.00043 | 3/5 |
| CiteSeer | 0.4 | gagb_ca_ce | +0.00103 | 2/5 |
| Cora | 0.2 | gagb | +0.00011 | 2/5 |
| Cora | 0.2 | gagb_ca_ce | +0.00106 | 3/5 |
| Cora | 0.4 | gagb | -0.00232 | 1/5 |
| Cora | 0.4 | gagb_ca_ce | -0.00211 | 1/5 |

Controls:

- `gagb` does not consistently beat `khop_only`.
- `gagb` does not consistently beat `gagb_shuffled`.
- `gagb_ca_ce` beats `gagb_shuffled` and `khop_only` on Cora 0.2, but not consistently across all four settings.

Mechanism diagnostics:

- `gagb` clean purity is higher than ordinary `gb` on all four settings.
- `gagb` noisy-label entropy is lower than ordinary `gb` on all four settings.
- `gagb` noise detection AUROC is lower than ordinary `gb` on all four settings.
- `gagb_ca_ce` improves AUROC over `gb` on Cora 0.2 and Cora 0.4, but not on CiteSeer.
- `noise_correction_rate` is 0.0 in this run for these soft-label variants.
- `clean_damage_rate` is 0.0, so there is no observed clean-label damage under the current target argmax diagnostic.

## Decision

The current v1 is mechanism-mixed, not paper-ready.

Supported:

- Graph-aware balls produce cleaner and lower-entropy local regions than ordinary embedding-only GB.
- Accuracy is weakly positive against `gb` in 3 of 4 dataset-noise settings for `gagb` and `gagb_ca_ce`.

Not supported:

- The method does not satisfy the stronger criterion of beating `khop_only` and `gagb_shuffled`.
- The current v1 does not show a stable noise-detection advantage over ordinary GB.

Next decision:

Continue only as a diagnostic direction. The next version should test whether the benefit comes from topology purity rather than graph-aware scoring by adding stronger controls and tuning the target construction, not by adding more modules.
