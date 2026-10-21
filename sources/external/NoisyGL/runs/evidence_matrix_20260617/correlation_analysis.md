# Phase 37 Applicability Grid

Protocol target: percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4 where available.
Clean labels are used only for diagnostic variables such as homophily and noise harm.

## Grid

| Dataset | N | C | h | noise harm | orthogonal AUC | SoftDist-GCN[XH] | SoftDist-shuffled | GBSelect-GCN-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2708 | 7 | 0.8100 | 0.0538 | 0.8460 | 0.0110 | 0.0216 | 0.0064 |
| CiteSeer | 3327 | 6 | 0.7355 | 0.0962 | 0.8286 | 0.0130 | 0.0195 | 0.0224 |
| PubMed | 19717 | 3 | 0.8024 | 0.0365 | 0.6633 | 0.0024 | 0.0017 | 0.0006 |
| Amazon-Photo | 7650 | 8 | 0.8272 | 0.0383 | 0.9483 | 0.0001 | -0.0016 | 0.0311 |
| DBLP | 17716 | 4 | 0.8279 | 0.0305 | 0.6955 | NA | NA | 0.0035 |
| Coauthor-CS | 18333 | 15 | 0.8081 | 0.0108 | NA | 0.0053 | 0.0055 | 0.0071 |
| Amazon-Computers | 13752 | 10 | 0.7772 | 0.0333 | NA | -0.0070 | 0.0182 | 0.0255 |
| Coauthor-Physics | 34493 | 5 | 0.9314 | 0.0099 | NA | NA | NA | NA |
| WikiCS | 11701 | 10 | 0.6543 | 0.0310 | 0.5282 | -0.0087 | -0.0044 | -0.0425 |

## Correlations

| Target | Variable | n | Spearman | Pearson |
|---|---|---:|---:|---:|
| softdist_vs_gcn_xh_mean | nodes | 7 | -0.3929 | -0.4658 |
| softdist_vs_gcn_xh_mean | classes | 7 | -0.4505 | -0.2682 |
| softdist_vs_gcn_xh_mean | homophily | 7 | 0.2500 | 0.3923 |
| softdist_vs_gcn_xh_mean | num_balls | 5 | -0.3078 | -0.1766 |
| softdist_vs_gcn_xh_mean | noise_harm_mean | 7 | 0.6071 | 0.6007 |
| softdist_vs_gcn_xh_mean | orthogonal_auc | 5 | 0.3000 | 0.6160 |
| softdist_vs_gcn_xh_mean | ball_auc | 5 | 0.1000 | 0.5760 |
| softdist_vs_gcn_xh_mean | true_ball_purity | 5 | 0.0000 | 0.6056 |
| gbselect_vs_gcn_only_mean | nodes | 8 | -0.3333 | -0.2372 |
| gbselect_vs_gcn_only_mean | classes | 8 | 0.2874 | -0.0754 |
| gbselect_vs_gcn_only_mean | homophily | 8 | 0.2143 | 0.6647 |
| gbselect_vs_gcn_only_mean | num_balls | 6 | -0.1765 | 0.0952 |
| gbselect_vs_gcn_only_mean | noise_harm_mean | 8 | 0.3333 | 0.2971 |
| gbselect_vs_gcn_only_mean | orthogonal_auc | 6 | 0.9429 | 0.9174 |
| gbselect_vs_gcn_only_mean | ball_auc | 6 | 0.6000 | 0.8734 |
| gbselect_vs_gcn_only_mean | true_ball_purity | 6 | 0.4857 | 0.8436 |

## Adaptive-Layer Gate

`orthogonal_auc`, `ball_auc`, `true_ball_purity`, `noise_harm`, and homophily are diagnostic variables here; they use clean labels or full dataset labels for analysis and cannot directly drive a deploy-time adaptive layer.
No strong deployable trigger is established yet. If this remains true after WikiCS/Physics rows are filled, an adaptive layer would risk becoming dataset-level cherry-picking.
