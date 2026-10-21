# MLP-24 Scale-Adaptive Ball Report

Candidate rule: `max(64, ceil(sqrt(|V|)))`. Only the ball-count cap changes.

## Gate decisions

| Stage | Pass | Reasons |
|---|---:|---|
| geometry | 1 | - |
| seed0_smoke | 0 | physics_adaptive_vs_fixed_below_0p5pp |

## Geometry

| Dataset | Variant | Balls | Mean size | Mean train votes | Build seconds |
|---|---|---:|---:|---:|---:|
| cora | fixed | 52 | 52.08 | 10.40 | 3.79 |
| cora | sqrt_nodes | 52 | 52.08 | 10.40 | 2.40 |
| cora | sqrt_nodes_shuffled | 52 | 52.08 | 10.40 | 2.33 |
| citeseer | fixed | 64 | 51.98 | 10.39 | 3.77 |
| citeseer | sqrt_nodes | 64 | 51.98 | 10.39 | 3.82 |
| citeseer | sqrt_nodes_shuffled | 64 | 51.98 | 10.39 | 3.72 |
| corafull | fixed | 64 | 309.27 | 61.84 | 5.14 |
| corafull | sqrt_nodes | 141 | 140.38 | 28.07 | 18.24 |
| corafull | sqrt_nodes_shuffled | 141 | 140.38 | 28.07 | 19.64 |
| coauthorcs | fixed | 64 | 286.45 | 57.28 | 7.82 |
| coauthorcs | sqrt_nodes | 136 | 134.80 | 26.96 | 19.31 |
| coauthorcs | sqrt_nodes_shuffled | 136 | 134.80 | 26.96 | 18.72 |
| amazonpho | fixed | 64 | 119.53 | 23.91 | 3.07 |
| amazonpho | sqrt_nodes | 88 | 86.93 | 17.39 | 5.95 |
| amazonpho | sqrt_nodes_shuffled | 88 | 86.93 | 17.39 | 5.96 |
| citeseerfull | fixed | 64 | 66.09 | 13.22 | 5.37 |
| citeseerfull | sqrt_nodes | 64 | 66.09 | 13.22 | 5.96 |
| citeseerfull | sqrt_nodes_shuffled | 64 | 66.09 | 13.22 | 6.72 |
| coauthorph | fixed | 64 | 538.95 | 107.78 | 7.99 |
| coauthorph | sqrt_nodes | 186 | 185.45 | 37.09 | 20.06 |
| coauthorph | sqrt_nodes_shuffled | 186 | 185.45 | 37.09 | 20.09 |
| pubmed | fixed | 64 | 308.08 | 61.61 | 4.41 |
| pubmed | sqrt_nodes | 141 | 139.84 | 27.96 | 14.30 |
| pubmed | sqrt_nodes_shuffled | 141 | 139.84 | 27.96 | 14.37 |
| amazoncom | fixed | 64 | 214.88 | 42.97 | 4.02 |
| amazoncom | sqrt_nodes | 118 | 116.54 | 23.31 | 13.44 |
| amazoncom | sqrt_nodes_shuffled | 118 | 116.54 | 23.31 | 13.38 |
| coraml | fixed | 62 | 48.31 | 9.66 | 2.95 |
| coraml | sqrt_nodes | 62 | 48.31 | 9.66 | 3.01 |
| coraml | sqrt_nodes_shuffled | 62 | 48.31 | 9.66 | 3.71 |
| attribwiki | fixed | 41 | 58.66 | 11.73 | 2.87 |
| attribwiki | sqrt_nodes | 41 | 58.66 | 11.73 | 2.93 |
| attribwiki | sqrt_nodes_shuffled | 41 | 58.66 | 11.73 | 2.83 |

## Downstream aggregate

| Dataset | Noise | Variant | n | Accuracy |
|---|---|---|---:|---:|
| amazoncom | uniform 0.4 | fixed | 1 | 75.33 +/- 0.00 |
| amazoncom | uniform 0.4 | mlp_jocor | 1 | 77.08 +/- 0.00 |
| amazoncom | uniform 0.4 | sqrt_nodes | 1 | 76.03 +/- 0.00 |
| amazoncom | uniform 0.4 | sqrt_nodes_shuffled | 1 | 67.98 +/- 0.00 |
| amazonpho | uniform 0.4 | fixed | 1 | 81.44 +/- 0.00 |
| amazonpho | uniform 0.4 | mlp_jocor | 1 | 79.37 +/- 0.00 |
| amazonpho | uniform 0.4 | sqrt_nodes | 1 | 81.48 +/- 0.00 |
| amazonpho | uniform 0.4 | sqrt_nodes_shuffled | 1 | 72.10 +/- 0.00 |
| attribwiki | uniform 0.4 | fixed | 1 | 65.60 +/- 0.00 |
| attribwiki | uniform 0.4 | mlp_jocor | 1 | 64.77 +/- 0.00 |
| attribwiki | uniform 0.4 | sqrt_nodes | 1 | 65.60 +/- 0.00 |
| attribwiki | uniform 0.4 | sqrt_nodes_shuffled | 1 | 59.48 +/- 0.00 |
| citeseer | uniform 0.4 | fixed | 1 | 55.33 +/- 0.00 |
| citeseer | uniform 0.4 | mlp_jocor | 1 | 47.59 +/- 0.00 |
| citeseer | uniform 0.4 | sqrt_nodes | 1 | 55.33 +/- 0.00 |
| citeseer | uniform 0.4 | sqrt_nodes_shuffled | 1 | 42.23 +/- 0.00 |
| citeseerfull | uniform 0.4 | fixed | 1 | 59.81 +/- 0.00 |
| citeseerfull | uniform 0.4 | mlp_jocor | 1 | 58.63 +/- 0.00 |
| citeseerfull | uniform 0.4 | sqrt_nodes | 1 | 59.81 +/- 0.00 |
| citeseerfull | uniform 0.4 | sqrt_nodes_shuffled | 1 | 44.85 +/- 0.00 |
| coauthorcs | uniform 0.4 | fixed | 1 | 90.35 +/- 0.00 |
| coauthorcs | uniform 0.4 | mlp_jocor | 1 | 89.17 +/- 0.00 |
| coauthorcs | uniform 0.4 | sqrt_nodes | 1 | 90.56 +/- 0.00 |
| coauthorcs | uniform 0.4 | sqrt_nodes_shuffled | 1 | 87.59 +/- 0.00 |
| coauthorph | uniform 0.4 | fixed | 1 | 93.05 +/- 0.00 |
| coauthorph | uniform 0.4 | mlp_jocor | 1 | 93.34 +/- 0.00 |
| coauthorph | uniform 0.4 | sqrt_nodes | 1 | 93.15 +/- 0.00 |
| coauthorph | uniform 0.4 | sqrt_nodes_shuffled | 1 | 87.61 +/- 0.00 |
| cora | uniform 0.4 | fixed | 1 | 55.46 +/- 0.00 |
| cora | uniform 0.4 | mlp_jocor | 1 | 50.82 +/- 0.00 |
| cora | uniform 0.4 | sqrt_nodes | 1 | 55.46 +/- 0.00 |
| cora | uniform 0.4 | sqrt_nodes_shuffled | 1 | 43.85 +/- 0.00 |
| corafull | uniform 0.4 | fixed | 1 | 37.39 +/- 0.00 |
| corafull | uniform 0.4 | mlp_jocor | 1 | 35.58 +/- 0.00 |
| corafull | uniform 0.4 | sqrt_nodes | 1 | 38.32 +/- 0.00 |
| corafull | uniform 0.4 | sqrt_nodes_shuffled | 1 | 32.34 +/- 0.00 |
| coraml | uniform 0.4 | fixed | 1 | 51.91 +/- 0.00 |
| coraml | uniform 0.4 | mlp_jocor | 1 | 53.24 +/- 0.00 |
| coraml | uniform 0.4 | sqrt_nodes | 1 | 51.91 +/- 0.00 |
| coraml | uniform 0.4 | sqrt_nodes_shuffled | 1 | 44.13 +/- 0.00 |
| pubmed | uniform 0.4 | fixed | 1 | 71.60 +/- 0.00 |
| pubmed | uniform 0.4 | mlp_jocor | 1 | 77.78 +/- 0.00 |
| pubmed | uniform 0.4 | sqrt_nodes | 1 | 71.97 +/- 0.00 |
| pubmed | uniform 0.4 | sqrt_nodes_shuffled | 1 | 62.67 +/- 0.00 |

## Seed-0 paired effects

| Dataset | Adaptive - fixed | Adaptive - shuffled |
|---|---:|---:|
| citeseerfull | +0.00pp | +14.96pp |
| corafull | +0.93pp | +5.98pp |
| coauthorcs | +0.20pp | +2.97pp |
| amazonpho | +0.04pp | +9.37pp |
| pubmed | +0.37pp | +9.30pp |
| amazoncom | +0.71pp | +8.05pp |
| coauthorph | +0.10pp | +5.53pp |

## Final decision

The seven affected datasets improve by +0.33pp on average at seed 0, but Coauthor-Physics improves by only +0.10pp against the preregistered +0.50pp gate.

**Decision: STOP after Phase B.** Keep fixed `max_balls=64`; do not run Phase C or Phase D, do not change the paper method, and record the scale-adaptive rule as a negative/boundary result.

Interpretation: increasing the cap materially changes large-graph geometry and preserves a strong real-vs-shuffled signal, but the finer partition adds little downstream information beyond the existing fixed-64 consensus targets.
