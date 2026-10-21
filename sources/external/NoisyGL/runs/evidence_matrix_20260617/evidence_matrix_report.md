# Phase 38 Evidence Matrix: Frozen Deployable Proxy Correlations

Only the three pre-registered deployable proxies are reported:

- `mean_vote_margin_weighted`
- `mean_vote_entropy_weighted`
- `ball_gcn_disagree_confident_train`

Clean labels are not used on the proxy side. Target deltas are downstream evaluation results.

## Correlations

| Method | Target | Proxy | n | Spearman | Pearson | Power note |
|---|---|---|---:|---:|---:|---|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | ball_gcn_disagree_confident_train | 6 | 0.0857 | -0.0636 | underpowered |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_vote_entropy_weighted | 7 | -0.2857 | -0.3150 | informative_but_below_adaptive_bar |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | mean_vote_margin_weighted | 7 | 0.3214 | 0.4651 | informative_but_below_adaptive_bar |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | ball_gcn_disagree_confident_train | 6 | -0.1429 | -0.2294 | underpowered |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_vote_entropy_weighted | 7 | -0.5714 | -0.5607 | informative_but_below_adaptive_bar |
| frozengrace_gb_gcn_softdist_xh | softdist_vs_shuffled_mean | mean_vote_margin_weighted | 7 | 0.2857 | 0.3389 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_gcn_only_mean | ball_gcn_disagree_confident_train | 7 | -0.5357 | -0.5637 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_entropy_weighted | 8 | -0.7857 | -0.7241 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_gcn_only_mean | mean_vote_margin_weighted | 8 | 0.6667 | 0.7952 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_shuffled_mean | ball_gcn_disagree_confident_train | 7 | 0.0714 | 0.2150 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_shuffled_mean | mean_vote_entropy_weighted | 8 | -0.2857 | -0.3868 | informative_but_below_adaptive_bar |
| gbselect | gbselect_vs_shuffled_mean | mean_vote_margin_weighted | 8 | 0.1905 | 0.1658 | informative_but_below_adaptive_bar |

## Judge

Maximum n is 8, below the pre-registered n>=10 bar. Adaptive layer is not evaluated; keep proxy gating as future work.

Claim boundary: evidence can be described as preliminary deployable-proxy evidence. Do not claim a causal determining variable.

Scatter rows: 86.
