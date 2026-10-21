# MLP-43 Representation-Source Attribution

All arms share the same split, noise, ball parameters, tie-safe target rule, and X-only MLP. Only the grouping representation changes. `adj2_x` denotes two applications of the stored normalized adjacency to X. A source is expanded to five seeds only when seed0 does not collapse to one ball.

| Dataset | Source | Test | Balls | Collapsed seeds |
|---|---|---:|---:|---:|
| cora | grace_h | 56.28 +/- 1.54 | 52.4 | 0 |
| cora | raw_x | 42.96 | 1.0 | 1 |
| cora | adj2_x | 42.96 | 1.0 | 1 |
| citeseer | grace_h | 53.52 +/- 2.71 | 63.8 | 0 |
| citeseer | raw_x | 43.56 | 1.0 | 1 |
| citeseer | adj2_x | 43.56 | 1.0 | 1 |
| corafull | grace_h | 37.30 +/- 0.57 | 64.0 | 0 |
| corafull | raw_x | 31.95 | 1.0 | 1 |
| corafull | adj2_x | 31.95 | 1.0 | 1 |
| coauthorcs | grace_h | 90.62 +/- 0.47 | 64.0 | 0 |
| coauthorcs | raw_x | 87.88 | 1.0 | 1 |
| coauthorcs | adj2_x | 89.25 +/- 0.58 | 57.2 | 0 |
| amazonpho | grace_h | 81.33 +/- 0.70 | 64.0 | 0 |
| amazonpho | raw_x | 72.70 | 1.0 | 1 |
| amazonpho | adj2_x | 80.41 +/- 1.12 | 64.0 | 0 |
| citeseer | gcn_hidden_noisy_seed0 | 56.49 | 26.0 | 0 |
| cora | gcn_hidden_noisy_seed0 | 55.83 | 31.0 | 0 |

## Paired label-free comparison against GRACE

- amazonpho/adj2_x: n=5, -0.92pp, 1/5 positive seeds.
- amazonpho/raw_x: n=1, -9.32pp, 0/1 positive seeds.
- citeseer/adj2_x: n=1, -10.70pp, 0/1 positive seeds.
- citeseer/raw_x: n=1, -10.70pp, 0/1 positive seeds.
- coauthorcs/adj2_x: n=5, -1.37pp, 0/5 positive seeds.
- coauthorcs/raw_x: n=1, -2.45pp, 0/1 positive seeds.
- cora/adj2_x: n=1, -12.77pp, 0/1 positive seeds.
- cora/raw_x: n=1, -12.77pp, 0/1 positive seeds.
- corafull/adj2_x: n=1, -5.25pp, 0/1 positive seeds.
- corafull/raw_x: n=1, -5.25pp, 0/1 positive seeds.

## Interpretation boundary

- This table attributes performance to the tested grouping representations; it is not a proof that GRACE is uniquely optimal.
- One-ball seed0 sources are reported as geometric failures and are not expanded solely to obtain repeated classifier scores.
- The supervised GCN-hidden rows are seed-0 diagnostics only and use noisy-label supervision, so they are not ranked with the label-free five-seed sources.
