# S2-mini: Is graph-aware granular ball worth it? (raw features vs GRACE embedding)

## Question

The "graph-aware granular ball" innovation assumes node adjacency adds information that features
alone lack. DCGB v2.8 showed topology adds ~0 over feature mutual-kNN on the *strong GRACE*
embedding — but maybe that is only because GRACE already absorbed the graph. So we test the space
where topology should be **most** complementary: raw (row-normalized BoW) features. Classifier-free
ball quality; clean labels diagnostics-only. Cora + CiteSeer × noise 0.2/0.4 × 5 seeds.

Variants (same builder): `gb` = ordinary feature-only hierarchical ball; `dcgb` = feature + topology
corroboration; `dcgb_emb_only` = feature mutual-kNN (no topology gate); `dcgb_topo_only` = topology.

## Results (5-seed mean ± std)

| dataset/noise | space | gb purity/AUROC | dcgb purity/AUROC | emb_only purity/AUROC | **topo increment (dcgb−emb_only)** |
|---|---|---|---|---|---|
| Cora 0.2 | raw_x   | 0.303 / 0.614 | 0.781 / 0.634 | 0.783 / 0.633 | purity −0.002, AUROC +0.001 |
| Cora 0.2 | grace_h | 0.893 / 0.682 | 0.892 / 0.787 | 0.892 / 0.787 | −0.000 / −0.000 |
| Cora 0.4 | raw_x   | 0.303 / 0.621 | 0.781 / 0.611 | 0.783 / 0.610 | −0.002 / +0.001 |
| Cora 0.4 | grace_h | 0.893 / 0.645 | 0.892 / 0.733 | 0.892 / 0.733 | −0.000 / +0.000 |
| CiteSeer 0.2 | raw_x   | 0.210 / 0.580 | 0.844 / 0.553 | 0.844 / 0.553 | +0.000 / +0.000 |
| CiteSeer 0.2 | grace_h | 0.797 / 0.606 | 0.751 / 0.705 | 0.752 / 0.706 | −0.000 / −0.001 |
| CiteSeer 0.4 | raw_x   | 0.210 / 0.558 | 0.844 / 0.551 | 0.844 / 0.551 | +0.000 / +0.000 |
| CiteSeer 0.4 | grace_h | 0.797 / 0.582 | 0.751 / 0.669 | 0.752 / 0.668 | −0.000 / +0.001 |

## Findings

1. **The mutual-kNN construction is the real win, not graph-awareness.** On raw features ordinary `gb`
   purity collapses (Cora 0.303, CiteSeer 0.210) while the reciprocal-kNN balls recover to 0.78–0.84
   — a huge, consistent gap. The reliability comes from *how the ball is grown* (mutual nearest
   neighbours), which is representation-agnostic.
2. **Explicit graph topology is inert on these citation graphs — in EVERY space, including raw.** The
   topology increment `dcgb − dcgb_emb_only` is within ±0.002 purity / ±0.001 AUROC in all 8
   conditions. Even where topology had the most room to help (raw BoW features), it adds nothing over
   feature mutual-kNN. On homophilous, feature-rich citation graphs the adjacency signal is already
   redundant with feature co-occurrence and with the GRACE embedding.
3. `dcgb_topo_only` has very high purity (0.94–0.99) but near-random AUROC (0.50–0.51): pure topology
   makes tight-but-uninformative balls.

## Implication for the "graph-aware granular ball" innovation

On Cora/CiteSeer, graph-awareness will NOT be the source of the gain — the win is the reliable
mutual-kNN partition + reliability-gated denoising. To justify "graph-aware" as a headline novelty
the experiment must move to settings where features and topology genuinely disagree
(heterophilous graphs, or feature-sparse/missing-feature graphs); otherwise the graph-aware ablation
will read as inert. A defensible fallback framing leads with the reliable-ball denoising framework +
representation-space×ball-quality study, and reports graph-awareness as a studied conditional factor.
