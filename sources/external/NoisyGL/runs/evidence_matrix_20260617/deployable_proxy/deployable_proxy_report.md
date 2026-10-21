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
| cora | gbselect | 3 | 0.5243 | 0.3358 | 0.6342 | 1.0000 | 0.2532 | ok |
| citeseer | frozengrace_gb_gcn_softdist_xh | 3 | 0.4937 | 0.2762 | 0.6916 | 1.0000 | 0.4201 | ok |
| citeseer | gbselect | 3 | 0.4937 | 0.2762 | 0.6916 | 1.0000 | 0.4201 | ok |
| pubmed | frozengrace_gb_gcn_softdist_xh | 3 | 0.5290 | 0.2556 | 0.9076 | 1.0000 | 0.1380 | ok |
| pubmed | gbselect | 3 | 0.5533 | 0.2734 | 0.8535 | 0.9999 | 0.1938 | ok |
| amazonpho | frozengrace_gb_gcn_softdist_xh | 3 | 0.5290 | 0.3852 | 0.6916 | 1.0000 | 0.5288 | ok |
| amazonpho | gbselect | 3 | 0.5368 | 0.3601 | 0.6095 | 1.0000 | 0.5342 | ok |
| dblp | frozengrace_gb_gcn_softdist_xh | 3 | 0.3420 | 0.0528 | 0.9731 | 1.0000 | 0.5811 | ok |
| dblp | gbselect | 3 | 0.5481 | 0.3080 | 0.7648 | 0.9997 | 0.1648 | ok |
| coauthorcs | frozengrace_gb_gcn_softdist_xh | 3 | 0.5492 | 0.4203 | 0.5239 | 0.9999 | 0.0944 | ok |
| coauthorcs | gbselect | 3 | 0.5492 | 0.4203 | 0.5239 | 0.9999 | 0.0944 | ok |
| amazoncom | frozengrace_gb_gcn_softdist_xh | 3 | 0.5012 | 0.3183 | 0.5897 | 0.9998 | 0.4948 | ok |
| amazoncom | gbselect | 3 | 0.5012 | 0.3183 | 0.5897 | 0.9998 | 0.4948 | ok |
| coauthorph | frozengrace_gb_gcn_softdist_xh | 0 | NA | NA | NA | NA | NA | missing_h |
| coauthorph | gbselect | 0 | NA | NA | NA | NA | NA | missing_h |
| wikics | frozengrace_gb_gcn_softdist_xh | 3 | 0.2259 | 0.0534 | 0.8984 | 1.0000 | 0.6332 | ok |
| wikics | gbselect | 3 | 0.2259 | 0.0534 | 0.8984 | 1.0000 | 0.6332 | ok |

## 37d-pre: Noisy Purity Check

Noisy train purity is not numerically washed flat across the available H-backed rows: min=0.2259, max=0.5533, range=0.3274.

| Proxy config | Target | n | Spearman | Pearson |
|---|---|---:|---:|---:|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | 7 | 0.2500 | 0.5763 |
| gbselect | gbselect_vs_gcn_only_mean | 8 | -0.0714 | 0.7885 |

## 37d-main: Top Deployable Proxy Correlations

| Config | Target | Proxy | n | Spearman | Pearson |
|---|---|---|---:|---:|---:|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_ball_size | 7 | -0.8929 | -0.6711 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_train_per_ball | 7 | -0.8929 | -0.6712 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | ball_gcn_original_label_disagree_train | 7 | -0.8571 | -0.9089 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_entropy_weighted | 8 | -0.7857 | -0.7241 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_entropy_unweighted | 8 | -0.7619 | -0.6314 |
| gbselect | gbselect_vs_shuffled_mean | mean_ball_size | 8 | -0.7619 | -0.4517 |
| gbselect | gbselect_vs_shuffled_mean | mean_train_per_ball | 8 | -0.7619 | -0.4523 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | ball_gcn_original_label_disagree_train | 7 | -0.6786 | -0.5802 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_margin_weighted | 8 | 0.6667 | 0.7952 |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_margin_unweighted | 8 | 0.6667 | 0.8201 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | ball_gcn_disagree_train | 7 | -0.6429 | -0.5590 |
| gbselect | gbselect_vs_gcn_only_mean | mean_ball_size | 8 | -0.6190 | -0.9134 |
| gbselect | gbselect_vs_gcn_only_mean | mean_train_per_ball | 8 | -0.6190 | -0.9135 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_ball_size | 7 | -0.6071 | -0.3277 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_train_per_ball | 7 | -0.6071 | -0.3281 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_vote_entropy_weighted | 7 | -0.5714 | -0.5607 |
| gbselect | gbselect_vs_shuffled_mean | noisy_train_purity_weighted | 8 | -0.5714 | 0.1303 |
| gbselect | gbselect_vs_shuffled_mean | num_balls | 8 | -0.5524 | -0.3650 |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | noisy_train_purity_unweighted | 7 | 0.5357 | 0.6050 |
| gbselect | gbselect_vs_gcn_only_mean | ball_gcn_disagree_confident_train | 7 | -0.5357 | -0.5637 |

## 37d-judge

At least one deployable proxy reaches |Spearman| >= 0.7 with n >= 7. This is a legitimate but still small-sample basis for a future adaptive layer; expand to 12+ datasets before claiming a rule.

Claim boundary: write only that evidence suggests ball-clusterability mediates ball gains. Do not write that a causal determining variable has been found.
