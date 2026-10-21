# Phase 37d Deployable Proxy Report

## Protocol

- Datasets: `cora,wikics`
- Seeds: `0`
- Noise: `uniform` `0.4`
- Split: percent `0.2/0.1/0.7`
- Proxy computation uses H geometry + noisy train labels only.
- Clean labels/test outcomes are used only in target deltas already computed by Phase 37c.
- Coauthor-Physics is expected to be skipped unless its GRACE H exists.

## Proxy Summary

| Dataset | Config | n | noisy purity | vote margin | entropy | eligible train | GCN-ball disagree | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cora | frozengrace_gb_gcn_softdist_xh | 1 | 0.5453 | 0.3567 | 0.6090 | 1.0000 | 0.2403 | ok |
| cora | gbselect | 0 | NA | NA | NA | NA | NA | missing_config |
| wikics | frozengrace_gb_gcn_softdist_xh | 1 | 0.2295 | 0.0632 | 0.8989 | 1.0000 | 0.6137 | ok |
| wikics | gbselect | 1 | 0.2295 | 0.0632 | 0.8989 | 1.0000 | 0.6137 | ok |

## 37d-pre: Noisy Purity Check

| Proxy config | Target | n | Spearman | Pearson |
|---|---|---:|---:|---:|
| frozengrace_gb_gcn_softdist_xh | softdist_vs_gcn_xh_mean | 2 | NA | 1.0000 |
| gbselect | gbselect_vs_gcn_only_mean | 1 | NA | NA |

## 37d-main: Top Deployable Proxy Correlations

| Config | Target | Proxy | n | Spearman | Pearson |
|---|---|---|---:|---:|---:|

## 37d-judge

No deployable proxy reaches |Spearman| >= 0.7 with n >= 7. Under the pre-registered rule, the adaptive layer should not be implemented yet; keep it as future work unless more datasets change this.

Claim boundary: write only that evidence suggests ball-clusterability mediates ball gains. Do not write that a causal determining variable has been found.
