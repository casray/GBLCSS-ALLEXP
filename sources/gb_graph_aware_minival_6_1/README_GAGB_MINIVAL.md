# Graph-aware GB Minival 6.1

## Research Hypothesis

Ordinary granular balls only use embedding distance. For graph node noisy-label learning, a local region may need both embedding similarity and k-hop topology proximity. This minival checks whether graph-aware granular balls provide value beyond ordinary GB, k-hop-only grouping, and shuffled grouping under the same frozen-GRACE gate protocol.

## Ball Role

Graph-aware GB is used only as a local soft-label unit and reliability region. The ball construction stage does not use noisy labels or clean labels. Labels are used only after grouping to build smoothing targets, weights, and diagnostics.

## Methods

- `gb`: ordinary embedding-space granular ball.
- `gagb`: embedding similarity plus k-hop topology.
- `gagb_ca_ce`: graph-aware GB with coverage-aware CE fallback.
- `gagb_weighted_ce`: graph-aware GB reliability-weighted CE.
- `emb_only_gagb_local`: same greedy builder and k-hop candidate pool as GAGB, but embedding score only.
- `emb_only_gagb_global`: same greedy builder as GAGB, all unassigned nodes as candidates, embedding score only.
- `khop_only`: topology-only control.
- `kmeans_gagb_matched`: KMeans with K matched to GAGB ball count.
- `topk_gagb_matched`: top-k smoothing with k matched to eligible GAGB train group size.
- `gagb_shuffled_size_only`: preserves graph-aware ball size distribution but shuffles members.
- `gagb_shuffled_alpha_matched`: preserves graph-aware ball size distribution and reuses GAGB per-node alpha/coverage while shuffling members.
- `ce`, `global`, `kmeans`, `topk_matched`: existing controls.

## Minimal Command

```powershell
cd C:\Users\Admin\Desktop\实验8去噪5.19-\gb_graph_aware_minival_6_1
py -3 scripts\run_gate_experiment.py --allow-external-python --python C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe --datasets Cora CiteSeer --split-mode ratio --train-ratios 0.2 --val-ratio 0.1 --noise-rates 0.2 0.4 --seeds 0 1 2 3 4 --mode frozen --alphas 0.3 --methods ce global gb gb_ca_ce gagb gagb_ca_ce emb_only_gagb_local emb_only_gagb_global khop_only kmeans_gagb_matched topk_gagb_matched gagb_shuffled_size_only gagb_shuffled_alpha_matched --save-diagnostics
py -3 scripts\summarize_gate_results.py --results-root results --output-dir reports\gagb_fair_minival_20260601
```

## Success Criteria

- `gagb` beats ordinary `gb` in paired mean on at least 3 of 4 dataset-noise settings.
- `gagb` beats `emb_only_gagb_local`, `emb_only_gagb_global`, `khop_only`, `kmeans_gagb_matched`, `topk_gagb_matched`, and `gagb_shuffled_alpha_matched`; otherwise the graph-aware GB-specific claim is not supported.
- `noise_detection_auroc` improves over ordinary GB in at least 3 of 4 settings.
- `clean_damage_rate` does not exceed ordinary GB by more than 1 percentage point.
