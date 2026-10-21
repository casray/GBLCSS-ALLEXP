# Phase 38 Evidence Matrix Execution Log

Date: 2026-06-17

## Objective

Close the Phase 37d evidence gap without changing the main method. The goal is to fill low-cost missing target deltas and re-evaluate only the three pre-registered deployable proxies:

- `mean_vote_margin_weighted`
- `mean_vote_entropy_weighted`
- `ball_gcn_disagree_confident_train`

The main paper line is not dependent on this phase. Adaptive gating is evaluated only if the evidence matrix reaches n >= 10.

## Changes

Added Cora/CiteSeer configs for the existing GBSelect variants:

- `config/gbselect/gbselect_cora.yaml`
- `config/gbselect/gbselect_citeseer.yaml`
- `config/gbselect_gcn_only/gbselect_gcn_only_cora.yaml`
- `config/gbselect_gcn_only/gbselect_gcn_only_citeseer.yaml`
- `config/gbselect_shuffled/gbselect_shuffled_cora.yaml`
- `config/gbselect_shuffled/gbselect_shuffled_citeseer.yaml`

Added Phase 38 summarizer:

- `scripts/summarize_evidence_matrix.py`

Adjusted CSV reading in `scripts/summarize_applicability_grid.py` to tolerate legacy non-UTF-8 path strings with `errors="replace"`. This affects only string decoding, not numeric values.

## Runs

Smoke:

```powershell
$py='C:\Users\Admin\Desktop\三论文融合\三论文融合\GRACE-master\.venv-rocm\Scripts\python.exe'
& $py scripts\run_ssl_swap_downstream.py --embedding_root ..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2 --datasets cora,citeseer --ssl_methods grace --base_methods gbselect,gbselect_gcn_only,gbselect_shuffled --seeds 0 --device cuda:0 --output runs\evidence_matrix_20260617\gbselect_cora_citeseer_smoke.csv
```

Full Cora/CiteSeer GBSelect 5-seed:

```powershell
& $py scripts\run_ssl_swap_downstream.py --embedding_root ..\..\gb_graph_aware_minival_6_1\checkpoints\embeddings_v2 --datasets cora,citeseer --ssl_methods grace --base_methods gbselect,gbselect_gcn_only,gbselect_shuffled --seeds 0,1,2,3,4 --device cuda:0 --output runs\evidence_matrix_20260617\gbselect_cora_citeseer_5seed.csv
```

Proxy recomputation:

```powershell
& $py scripts\extract_deployable_proxies.py --datasets cora,citeseer,pubmed,amazonpho,dblp,coauthorcs,amazoncom,coauthorph,wikics --seeds 0,1,2 --device cuda:0 --output_dir runs\evidence_matrix_20260617\deployable_proxy
```

Summary:

```powershell
& $py scripts\summarize_applicability_grid.py --output_dir runs\evidence_matrix_20260617 --inputs ... runs\evidence_matrix_20260617\gbselect_cora_citeseer_5seed.csv
& $py scripts\extract_deployable_proxies.py --postprocess_only --output_dir runs\evidence_matrix_20260617\deployable_proxy --target_summary runs\evidence_matrix_20260617\applicability_grid_summary.csv
& $py scripts\summarize_evidence_matrix.py --proxy_summary runs\evidence_matrix_20260617\deployable_proxy\deployable_proxy_summary.csv --target_summary runs\evidence_matrix_20260617\applicability_grid_summary.csv --output_dir runs\evidence_matrix_20260617
```

## Added Target Deltas

Cora GBSelect:

- `gbselect`: 0.8139 +/- 0.0147
- `gbselect_gcn_only`: 0.8075 +/- 0.0227
- `gbselect_shuffled`: 0.7831 +/- 0.0183
- `gbselect - gcn_only`: +0.0064, positive seeds 4/5
- `gbselect - shuffled`: +0.0308, positive seeds 5/5

CiteSeer GBSelect:

- `gbselect`: 0.6924 +/- 0.0180
- `gbselect_gcn_only`: 0.6700 +/- 0.0379
- `gbselect_shuffled`: 0.6303 +/- 0.0636
- `gbselect - gcn_only`: +0.0224, positive seeds 4/5
- `gbselect - shuffled`: +0.0621, positive seeds 5/5

Existing PubMed and Amazon-Photo SoftDist/XH result CSVs were also included, raising SoftDist proxy-target n from 5 to 7 where possible.

## Coauthor-Physics

Coauthor-Physics remains `missing_h`.

The upstream GRACE helper project has a data loader alias for Coauthor-Physics, but the runnable GRACE CLI and `config.yaml` are pre-registered only for Cora/CiteSeer/PubMed/DBLP. Reusing another dataset's GRACE config for Physics would create a new protocol choice, so Phase 38 records this as missing rather than fabricating an H.

## Final Proxy Correlations

Maximum available n:

- SoftDist targets: n = 7
- GBSelect targets: n = 8

Best frozen proxy result:

- GBSelect vs GCN-only, `mean_vote_entropy_weighted`: Spearman = -0.7857, Pearson = -0.7241, n = 8
- GBSelect vs GCN-only, `mean_vote_margin_weighted`: Spearman = 0.6667, Pearson = 0.7952, n = 8

SoftDist frozen proxies remain weak to moderate:

- SoftDist vs GCN[X||H], `mean_vote_margin_weighted`: Spearman = 0.3214, Pearson = 0.4651, n = 7
- SoftDist vs shuffled, `mean_vote_entropy_weighted`: Spearman = -0.5714, Pearson = -0.5607, n = 7

## Decision

Phase 38 does not meet the pre-registered n >= 10 threshold. Therefore:

- Do not implement Phase 38d adaptive switch.
- Treat deployable proxies as preliminary evidence only.
- The paper may state that deployable noisy-label/geometric proxies show suggestive association with GBSelect gains, but sample size remains below the adaptive-trigger bar.
- Do not claim a validated deployable trigger or a causal determining variable.

## Outputs

- `runs/evidence_matrix_20260617/applicability_grid_summary.csv`
- `runs/evidence_matrix_20260617/frozen_proxy_correlations.csv`
- `runs/evidence_matrix_20260617/frozen_proxy_scatter.csv`
- `runs/evidence_matrix_20260617/evidence_matrix_report.md`
- `runs/evidence_matrix_20260617/frozen_proxy_scatter_plots/*.svg`

