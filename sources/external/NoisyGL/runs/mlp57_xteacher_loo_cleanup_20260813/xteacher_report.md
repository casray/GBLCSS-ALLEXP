# MLP-57 X-only ball-teacher report

The teacher and student both consume X. Frozen GRACE H is used only to build ball targets.
Teacher targets use beta=0 and the non-LOO distribution target, matching the existing teacher ablation.

## Aggregate

| noise | dataset | method | mean+/-std (%) |
|---:|---|---|---:|
| 0.4 | citeseer | mlp_distill_gcn | 66.09+/-1.38 |
| 0.4 | citeseer | mlp_distill_gcn_xh | 68.81+/-1.12 |
| 0.4 | citeseer | mlp_distill_softdist_x | 67.98+/-0.80 |
| 0.4 | citeseer | mlp_distill_softdist_xh | 69.40+/-1.08 |
| 0.4 | cora | mlp_distill_gcn | 77.95+/-2.62 |
| 0.4 | cora | mlp_distill_gcn_xh | 80.23+/-1.57 |
| 0.4 | cora | mlp_distill_softdist_x | 79.88+/-1.71 |
| 0.4 | cora | mlp_distill_softdist_xh | 80.60+/-0.69 |
| 0.4 | corafull | mlp_distill_gcn | 58.74+/-0.82 |
| 0.4 | corafull | mlp_distill_gcn_xh | 59.45+/-0.63 |
| 0.4 | corafull | mlp_distill_softdist_x | 58.19+/-0.56 |
| 0.4 | corafull | mlp_distill_softdist_xh | 58.22+/-0.69 |
| 0.6 | citeseer | mlp_distill_gcn | 53.30+/-4.42 |
| 0.6 | citeseer | mlp_distill_gcn_xh | 61.04+/-3.08 |
| 0.6 | citeseer | mlp_distill_softdist_x | 56.09+/-4.55 |
| 0.6 | citeseer | mlp_distill_softdist_xh | 62.28+/-2.30 |
| 0.6 | cora | mlp_distill_gcn | 65.50+/-3.28 |
| 0.6 | cora | mlp_distill_gcn_xh | 69.44+/-2.18 |
| 0.6 | cora | mlp_distill_softdist_x | 68.90+/-2.82 |
| 0.6 | cora | mlp_distill_softdist_xh | 71.27+/-2.46 |
| 0.6 | corafull | mlp_distill_gcn | 51.63+/-2.02 |
| 0.6 | corafull | mlp_distill_gcn_xh | 52.75+/-0.88 |
| 0.6 | corafull | mlp_distill_softdist_x | 52.51+/-0.78 |
| 0.6 | corafull | mlp_distill_softdist_xh | 51.82+/-0.86 |

## Paired deltas

- u0.4 citeseer x_ball_vs_gcn_x_ce: +1.89pp, 5/5 positive seeds.
- u0.4 citeseer x_ball_vs_gcn_xh_ce: -0.82pp, 1/5 positive seeds.
- u0.4 citeseer x_ball_vs_xh_ball: -1.42pp, 1/5 positive seeds.
- u0.4 cora x_ball_vs_gcn_x_ce: +1.93pp, 5/5 positive seeds.
- u0.4 cora x_ball_vs_gcn_xh_ce: -0.35pp, 2/5 positive seeds.
- u0.4 cora x_ball_vs_xh_ball: -0.72pp, 1/5 positive seeds.
- u0.4 corafull x_ball_vs_gcn_x_ce: -0.56pp, 1/5 positive seeds.
- u0.4 corafull x_ball_vs_gcn_xh_ce: -1.26pp, 0/5 positive seeds.
- u0.4 corafull x_ball_vs_xh_ball: -0.03pp, 3/5 positive seeds.
- u0.6 citeseer x_ball_vs_gcn_x_ce: +2.79pp, 5/5 positive seeds.
- u0.6 citeseer x_ball_vs_gcn_xh_ce: -4.95pp, 0/5 positive seeds.
- u0.6 citeseer x_ball_vs_xh_ball: -6.19pp, 0/5 positive seeds.
- u0.6 cora x_ball_vs_gcn_x_ce: +3.40pp, 5/5 positive seeds.
- u0.6 cora x_ball_vs_gcn_xh_ce: -0.54pp, 1/5 positive seeds.
- u0.6 cora x_ball_vs_xh_ball: -2.37pp, 0/5 positive seeds.
- u0.6 corafull x_ball_vs_gcn_x_ce: +0.89pp, 3/5 positive seeds.
- u0.6 corafull x_ball_vs_gcn_xh_ce: -0.24pp, 3/5 positive seeds.
- u0.6 corafull x_ball_vs_xh_ball: +0.69pp, 4/5 positive seeds.

## Pre-registered adoption gate

- u0.4: passed=True, passing_datasets=2/3, average gain over GCN(X)+CE=+1.09pp.
- u0.6: passed=True, passing_datasets=2/3, average gain over GCN(X)+CE=+2.36pp.

Controls for Cora/CiteSeer come from the audited MLP-9 table lineage, which predates hash logging.
CoraFull controls come from the MLP-43 audited table lineage and retain split/noise hashes.