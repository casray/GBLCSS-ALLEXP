# Phase 37d Deployable Proxy Report

## Protocol

- Datasets: `cora,citeseer,pubmed,amazonpho,dblp,coauthorcs,amazoncom,coauthorph,wikics`
- Seeds: `0,1,2`
- Noise: `uniform` `0.4`
- Split: percent `0.2/0.1/0.7`
- Proxy computation uses H geometry + noisy train labels only.
- Clean labels/test outcomes are used only in target deltas already computed by Phase 37c.
- Coauthor-Physics is expected to be skipped unless its GRACE H exists.

## Proxy Summary

| Dataset | Config | n | noisy purity | vote margin | entropy | eligible train | GCN-ball disagree | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cora | frozengrace_gb_gcn_softdist_xh | 3 | 0.5243 | 0.3358 | 0.6342 | 1.0000 | 0.2532 | ok |
| cora | gbselect | 0 | NA | NA | NA | NA | NA | missing_config |
| citeseer | frozengrace_gb_gcn_softdist_xh | 3 | 0.4937 | 0.2762 | 0.6916 | 1.0000 | 0.4201 | ok |
| citeseer | gbselect | 0 | NA | NA | NA | NA | NA | missing_config |
| pubmed | frozengrace_gb_gcn_softdist_xh | 3 | 0.5290 | 0.2556 | 0.9076 | 1.0000 | 0.1380 | ok |
| pubmed | gbselect | 3 | 0.5534 | 0.2734 | 0.8534 | 0.9999 | 0.1959 | ok |
| amazonpho | frozengrace_gb_gcn_softdist_xh | 3 | 0.5290 | 0.3852 | 0.6916 | 1.0000 | 0.5288 | ok |
| amazonpho | gbselect | 3 | 0.5368 | 0.3601 | 0.6095 | 1.0000 | 0.5342 | ok |
| dblp | frozengrace_gb_gcn_softdist_xh | 3 | 0.3420 | 0.0528 | 0.9731 | 1.0000 | 0.5811 | ok |
| dblp | gbselect | 3 | 0.5481 | 0.3079 | 0.7649 | 0.9997 | 0.1643 | ok |
| coauthorcs | frozengrace_gb_gcn_softdist_xh | 3 | 0.5492 | 0.4203 | 0.5239 | 0.9999 | 0.0944 | ok |
| coauthorcs | gbselect | 3 | 0.5492 | 0.4203 | 0.5239 | 0.9999 | 0.0944 | ok |
| amazoncom | frozengrace_gb_gcn_softdist_xh | 3 | 0.5012 | 0.3183 | 0.5897 | 0.9998 | 0.4948 | ok |
| amazoncom | gbselect | 3 | 0.5012 | 0.3183 | 0.5897 | 0.9998 | 0.4948 | ok |
| coauthorph | frozengrace_gb_gcn_softdist_xh | 0 | NA | NA | NA | NA | NA | missing_h |
| coauthorph | gbselect | 0 | NA | NA | NA | NA | NA | missing_h |
| wikics | frozengrace_gb_gcn_softdist_xh | 3 | 0.2259 | 0.0534 | 0.8984 | 1.0000 | 0.6332 | ok |
| wikics | gbselect | 3 | 0.2259 | 0.0534 | 0.8984 | 1.0000 | 0.6332 | ok |

## 37d-pre: Noisy Purity Check

Noisy train purity is not numerically washed flat across the available H-backed rows: min=0.2259, max=0.5534, range=0.3275.

| Proxy config | Target | n | Spearman | Pearson |
|---|---|---:|---:|---:|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | 5 | 0.3000 | 0.6358 |
| gbselect | gbselect_vs_gcn_only_mean | 6 | -0.0286 | 0.8205 |

## 37d-main: Top Deployable Proxy Correlations

| Config | Target | Proxy | n | Spearman | Pearson |
|---|---|---|---:|---:|---:|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_ball_size | 5 | -1.0000 | -0.8957 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_train_per_ball | 5 | -1.0000 | -0.8960 |
| gbselect | gbselect_vs_gcn_only_mean | ball_gcn_disagree_confident_train | 5 | -1.0000 | -0.9511 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | ball_gcn_original_label_disagree_train | 5 | -0.9000 | -0.9688 |
| gbselect | gbselect_vs_gcn_only_mean | mean_ball_size | 6 | -0.8857 | -0.9379 |
| gbselect | gbselect_vs_gcn_only_mean | mean_train_per_ball | 6 | -0.8857 | -0.9383 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_margin_weighted | 6 | 0.8286 | 0.8525 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_margin_unweighted | 6 | 0.8286 | 0.8672 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_ball_size | 5 | -0.8000 | -0.6636 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_train_per_ball | 5 | -0.8000 | -0.6645 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_entropy_weighted | 6 | -0.7714 | -0.7577 |
| gbselect | gbselect_vs_shuffled_mean | mean_ball_size | 6 | -0.7714 | -0.4288 |
| gbselect | gbselect_vs_shuffled_mean | mean_train_per_ball | 6 | -0.7714 | -0.4291 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_entropy_unweighted | 6 | -0.7143 | -0.6597 |
| gbselect | gbselect_vs_shuffled_mean | ball_gcn_original_label_disagree_train | 6 | 0.7143 | 0.7466 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | noisy_train_purity_unweighted | 5 | 0.6000 | 0.6535 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | ball_gcn_disagree_train | 5 | -0.6000 | -0.6358 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | ball_gcn_original_label_disagree_train | 5 | -0.6000 | -0.5095 |
| gbselect | gbselect_vs_shuffled_mean | noisy_train_purity_weighted | 6 | -0.6000 | 0.1737 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | num_balls | 5 | -0.5643 | -0.0700 |

## 37d-judge

The largest available correlation sample is n=6, below the pre-registered n >= 7 bar. Therefore this run cannot legally pass or fail the adaptive-trigger criterion; treat it as underpowered evidence and fill more target deltas before implementing an adaptive layer.

Claim boundary: write only that evidence suggests ball-clusterability mediates ball gains. Do not write that a causal determining variable has been found.
