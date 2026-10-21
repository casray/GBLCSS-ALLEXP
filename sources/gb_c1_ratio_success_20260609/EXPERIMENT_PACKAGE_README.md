# C1 Ratio-Split Success Package

Date: 2026-06-09

This folder is a clean working snapshot for the successful C1 ratio-split experiment.

## Main Evidence

- Formal result directory: `results_ratio_formal_20260609`
- Formal report: `reports/ratio_formal_5seed_report_20260609.md`
- Formal raw table: `reports/ratio_formal_rows_20260609.csv`
- Formal summary: `reports/ratio_formal_summary_20260609.csv`
- Paired deltas: `reports/ratio_formal_paired_delta_summary_20260609.csv`
- Ball vote coverage: `reports/ratio_formal_ball_vote_coverage_20260609.csv`

## Diagnostic Evidence

- Ratio coverage diagnostic directory: `results_ratio_c1_20260609`
- Diagnostic report: `reports/ratio_c1_seed0_summary_20260609.md`
- Diagnostic coverage table: `reports/ratio_c1_ball_vote_coverage_20260609.csv`

## Protocol Boundary

Supported claim:

Granular balls can serve as local noisy-label correction units in graph contrastive representations when local labeled coverage is sufficient.

Current formal protocol:

- Datasets: Cora, CiteSeer
- Split: stratified ratio split
- Train / validation / test: 0.2 / 0.1 / 0.7
- Noise: symmetric 0.4 on train labels only
- Seeds: 0, 1, 2, 3, 4
- Methods: `C1`, `C1_noball`, `C1_topk`, `C1_shuffled`

Not supported by this package:

- Solving the standard Planetoid low-label split.
- Broad robustness across all noise rates, split ratios, and larger graph datasets.
- SOTA claims against external noisy-label GNN methods.

## Audit Notes

- The formal result set contains 40 runs.
- Each run has `config.json`, `metrics.json`, `phase1_history.json`, `phase2_history.json`, and `embedding.pt`.
- The copied code includes source, configs, and tests needed to continue experiments from this snapshot.
- `linear_eval_acc` is a clean-label post-hoc representation diagnostic, not the main noisy-label classifier result.
- `correction_acc` is stored in `phase2_history.json`; report summaries use final or max history values.
