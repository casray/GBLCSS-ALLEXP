# Phase 11 Backbone Generalization Report

Date: 2026-06-12

## Protocol

- Datasets: Cora, CiteSeer.
- Split: `percent`, train/val/test = 0.2/0.1/0.7.
- Noise: `uniform`, noise_rate = 0.4.
- Seeds: 0, 1, 2.
- SSL H: GRACE anchor embeddings.
- Backbones: GAT, APPNP.
- Comparison: SoftDist [X||H] vs CE [X||H] under the same backbone.
- Purpose: appendix-level check only; main table remains GCN.

## Results

| Dataset | Backbone | SoftDist [X||H] | CE [X||H] | Mean delta | Positive seeds |
|---|---|---:|---:|---:|---:|
| Cora | GAT | 0.8044 ± 0.0100 | 0.7986 ± 0.0090 | +0.0058 | 3/3 |
| Cora | APPNP | 0.8169 ± 0.0156 | 0.8074 ± 0.0200 | +0.0095 | 3/3 |
| CiteSeer | GAT | 0.6635 ± 0.0365 | 0.6757 ± 0.0208 | -0.0122 | 1/3 |
| CiteSeer | APPNP | 0.6899 ± 0.0189 | 0.6914 ± 0.0245 | -0.0016 | 2/3 |

Per-seed paired deltas:

| Dataset | Backbone | Deltas |
|---|---|---|
| Cora | GAT | +0.0069; +0.0042; +0.0063 |
| Cora | APPNP | +0.0021; +0.0253; +0.0011 |
| CiteSeer | GAT | +0.0013; -0.0077; -0.0301 |
| CiteSeer | APPNP | -0.0288; +0.0047; +0.0193 |

## Decision

The strong "classifier-independent" claim is not supported.

- Cora supports positive ball increments under both GAT and APPNP.
- CiteSeer does not: GAT is clearly negative and APPNP is approximately tied but slightly negative on mean.

The safe claim is conditional: SoftDist [X||H] can transfer to non-GCN backbones on some datasets, but the main evidence remains the GCN setting. Keep Phase 11 as an appendix/robustness table, not a central claim.

## Research Record

- Hypothesis: the ball target mechanism is not tied to GCN and should show positive increments with other node classifiers.
- Granular-ball role: unchanged SoftDist target producer over frozen GRACE H.
- Code changes: added `GATNet` and `APPNPNet` backbones, plus four Phase 11 predictor/config variants.
- Config changes: Cora/CiteSeer only, seeds 0-2, no hyperparameter tuning beyond fixed backbone defaults (`GAT heads=1`, `APPNP K=10, alpha=0.1`).
- Success standard: SoftDist - CE same-backbone delta positive on at least two backbones.
- Result: positive on Cora, not positive on CiteSeer.
- Final decision: use as conditional appendix evidence; do not claim backbone-agnostic robustness.
