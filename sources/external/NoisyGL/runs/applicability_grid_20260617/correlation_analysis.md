# Phase 37 Applicability Grid

Protocol target: percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4 where available.
Clean labels are used only for diagnostic variables such as homophily and noise harm.

## Grid

| Dataset | N | C | h | noise harm | orthogonal AUC | SoftDist-GCN[XH] | SoftDist-shuffled | GBSelect-GCN-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 2708 | 7 | 0.8100 | 0.0538 | 0.8460 | 0.0110 | 0.0216 | NA |
| CiteSeer | 3327 | 6 | 0.7355 | 0.0962 | 0.8286 | 0.0130 | 0.0195 | NA |
| PubMed | 19717 | 3 | 0.8024 | 0.0365 | 0.6633 | NA | NA | 0.0006 |
| Amazon-Photo | 7650 | 8 | 0.8272 | 0.0383 | 0.9483 | NA | NA | 0.0311 |
| DBLP | 17716 | 4 | 0.8279 | 0.0305 | 0.6955 | NA | NA | 0.0035 |
| Coauthor-CS | 18333 | 15 | 0.8081 | 0.0108 | NA | 0.0053 | 0.0055 | 0.0071 |
| Amazon-Computers | 13752 | 10 | 0.7772 | 0.0333 | NA | -0.0070 | 0.0182 | 0.0255 |
| Coauthor-Physics | 34493 | 5 | 0.9314 | 0.0099 | NA | NA | NA | NA |
| WikiCS | 11701 | 10 | 0.6543 | 0.0310 | 0.5282 | -0.0087 | -0.0044 | -0.0425 |

## Correlations

| Target | Variable | n | Spearman | Pearson |
|---|---|---:|---:|---:|
| softdist_vs_gcn_xh_mean | nodes | 5 | -0.5000 | -0.5975 |
| softdist_vs_gcn_xh_mean | classes | 5 | -0.6669 | -0.3754 |
| softdist_vs_gcn_xh_mean | homophily | 5 | 0.4000 | 0.5069 |
| softdist_vs_gcn_xh_mean | num_balls | 3 | 0.0000 | -0.4254 |
| softdist_vs_gcn_xh_mean | noise_harm_mean | 5 | 0.7000 | 0.6004 |
| softdist_vs_gcn_xh_mean | orthogonal_auc | 3 | 0.5000 | 0.9912 |
| softdist_vs_gcn_xh_mean | ball_auc | 3 | 0.5000 | 0.9497 |
| softdist_vs_gcn_xh_mean | true_ball_purity | 3 | 0.5000 | 0.9702 |
| gbselect_vs_gcn_only_mean | nodes | 6 | -0.3714 | -0.1326 |
| gbselect_vs_gcn_only_mean | classes | 6 | 0.2029 | -0.0179 |
| gbselect_vs_gcn_only_mean | homophily | 6 | 0.4286 | 0.8253 |
| gbselect_vs_gcn_only_mean | num_balls | 4 | 0.3162 | 0.4150 |
| gbselect_vs_gcn_only_mean | noise_harm_mean | 6 | 0.3143 | 0.1314 |
| gbselect_vs_gcn_only_mean | orthogonal_auc | 4 | 1.0000 | 0.9412 |
| gbselect_vs_gcn_only_mean | ball_auc | 4 | 1.0000 | 0.9666 |
| gbselect_vs_gcn_only_mean | true_ball_purity | 4 | 1.0000 | 0.9177 |

## Adaptive-Layer Gate

`orthogonal_auc`, `ball_auc`, `true_ball_purity`, `noise_harm`, and homophily are diagnostic variables here; they use clean labels or full dataset labels for analysis and cannot directly drive a deploy-time adaptive layer.
No strong deployable trigger is established yet. If this remains true after WikiCS/Physics rows are filled, an adaptive layer would risk becoming dataset-level cherry-picking.
