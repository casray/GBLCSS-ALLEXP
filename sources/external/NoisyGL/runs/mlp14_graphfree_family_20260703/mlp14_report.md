# MLP-14 Graph-free Family Report

## Material Passport
- Table role: Table B / graph-free family positioning only.
- Protocol: train-only label noise, percent split 0.2/0.1/0.7.
- Methods: adapted GSDN and adapted Graph-MLP with X-only MLP inference.
- These methods use graph structure during training and are not added to main Table A.
- Smoke gate: pass

## Resource Annotation
| Method | train graph | infer graph | unlabeled nodes | all edges | note |
|---|---:|---:|---:|---:|---|
| Vanilla MLP | 0 | 0 | 0 | 0 | X-only supervised MLP baseline |
| Graph-MLP adapted | 1 | 0 | 1 | 1 | NContrast-style graph regularization during training |
| GSDN adapted | 1 | 0 | 1 | 1 | Feature/label neighbor self-distillation during training |
| Ours beta=0.5 train-only | 1 | 0 | 1 | 1 | Frozen GRACE-H preprocessing uses the full graph; downstream MLP is X-only |
| GCN(X)+CE | 1 | 1 | 0 | 1 | Reference message-passing model; not graph-free at inference |
| GCN[X||H]+CE | 1 | 1 | 0 | 1 | Reference message-passing model with frozen-H input |
| GLNN-style GCN teacher | 1 | 0 | 1 | 1 | GCN(X) teacher distills to X-only student |
| GLNN-style GCN[X||H] teacher | 1 | 0 | 1 | 1 | GCN[X||H] teacher distills to X-only student |
| Ours-distill kNN teacher | 1 | 0 | 1 | 1 | GCN[X||H] teacher distills to X-only student |

## Smoke Gate
| Method | Dataset | Noise | Test | MLP CE | Delta vs CE | status | CE gate | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gsdn | cora | uniform0.4 | 51.45 | 41.74 | +9.71pp | 1 | 1 | 1 |
| gsdn | cora | uniform0.6 | 37.10 | 30.71 | NA | 1 | 1 | 1 |
| gsdn | citeseer | uniform0.4 | 50.90 | 46.09 | +4.81pp | 1 | 1 | 1 |
| gsdn | citeseer | uniform0.6 | 37.54 | 32.13 | NA | 1 | 1 | 1 |
| gsdn | pubmed | uniform0.4 | 73.29 | 62.62 | +10.67pp | 1 | 1 | 1 |
| gsdn | pubmed | uniform0.6 | 50.16 | 47.21 | NA | 1 | 1 | 1 |
| gsdn | coauthorcs | uniform0.4 | 86.96 | 88.48 | -1.53pp | 1 | 1 | 1 |
| gsdn | coauthorcs | uniform0.6 | 80.35 | 81.38 | NA | 1 | 1 | 1 |
| gsdn | amazonpho | uniform0.4 | 75.67 | 74.49 | +1.18pp | 1 | 1 | 1 |
| gsdn | amazonpho | uniform0.6 | 53.80 | 57.16 | NA | 1 | 1 | 1 |
| graphmlp | cora | uniform0.4 | 58.21 | 41.74 | +16.46pp | 1 | 1 | 1 |
| graphmlp | cora | uniform0.6 | 44.49 | 30.71 | NA | 1 | 1 | 1 |
| graphmlp | citeseer | uniform0.4 | 57.47 | 46.09 | +11.38pp | 1 | 1 | 1 |
| graphmlp | citeseer | uniform0.6 | 42.96 | 32.13 | NA | 1 | 1 | 1 |
| graphmlp | pubmed | uniform0.4 | 69.16 | 62.62 | +6.54pp | 1 | 1 | 1 |
| graphmlp | pubmed | uniform0.6 | 52.50 | 47.21 | NA | 1 | 1 | 1 |
| graphmlp | coauthorcs | uniform0.4 | 88.81 | 88.48 | +0.33pp | 1 | 1 | 1 |
| graphmlp | coauthorcs | uniform0.6 | 81.04 | 81.38 | NA | 1 | 1 | 1 |
| graphmlp | amazonpho | uniform0.4 | 77.55 | 74.49 | +3.06pp | 1 | 1 | 1 |
| graphmlp | amazonpho | uniform0.6 | 63.14 | 57.16 | NA | 1 | 1 | 1 |

## Graph-free Family Table B
| Dataset | Noise | MLP | Graph-MLP | GSDN | Ours train-only | GCN | GCN[X||H] | GLNN | GLNN[X||H] | Ours-distill |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | uniform0.2 | 54.96+/-2.22 (5) | 73.82+/-1.37 (5) | 66.21+/-1.29 (5) | 62.94+/-1.60 (5) | NA | NA | NA | NA | NA |
| cora | uniform0.4 | 44.24+/-2.62 (5) | 63.23+/-3.07 (5) | 55.97+/-4.29 (5) | 55.59+/-1.50 (5) | 78.77+/-1.87 (5) | 80.05+/-1.41 (5) | 77.95+/-2.62 (5) | 80.23+/-1.57 (5) | 80.99+/-1.02 (5) |
| cora | uniform0.6 | 32.51+/-3.38 (5) | 45.66+/-2.71 (5) | 36.76+/-3.01 (5) | 41.04+/-2.37 (5) | 65.44+/-2.59 (5) | 69.33+/-1.64 (5) | 65.50+/-3.28 (5) | 69.44+/-2.18 (5) | 71.07+/-2.29 (5) |
| cora | pair0.4 | 42.83+/-2.19 (5) | 54.99+/-2.97 (5) | 50.78+/-2.89 (5) | 50.29+/-2.76 (5) | NA | NA | NA | NA | NA |
| cora | instance0.4 | 45.33+/-2.22 (5) | 64.55+/-3.17 (5) | 54.50+/-2.47 (5) | 55.67+/-0.81 (5) | NA | NA | NA | NA | NA |
| citeseer | uniform0.2 | 55.10+/-1.17 (5) | 67.65+/-0.99 (5) | 60.82+/-0.77 (5) | 63.38+/-1.20 (5) | NA | NA | NA | NA | NA |
| citeseer | uniform0.4 | 43.50+/-3.02 (5) | 57.23+/-1.85 (5) | 50.36+/-3.39 (5) | 53.83+/-2.75 (5) | 64.90+/-1.11 (5) | 68.35+/-0.97 (5) | 66.09+/-1.38 (5) | 68.81+/-1.12 (5) | 69.57+/-1.46 (5) |
| citeseer | uniform0.6 | 33.35+/-1.36 (5) | 44.04+/-2.15 (5) | 35.85+/-3.16 (5) | 40.17+/-2.75 (5) | 52.29+/-4.00 (5) | 60.34+/-2.42 (5) | 53.30+/-4.42 (5) | 61.04+/-3.08 (5) | 61.96+/-2.98 (5) |
| citeseer | pair0.4 | 45.15+/-1.67 (5) | 52.51+/-1.69 (5) | 46.98+/-1.44 (5) | 50.09+/-2.40 (5) | NA | NA | NA | NA | NA |
| citeseer | instance0.4 | 44.33+/-2.04 (5) | 57.86+/-2.32 (5) | 51.52+/-2.73 (5) | 54.62+/-2.25 (5) | NA | NA | NA | NA | NA |
| pubmed | uniform0.2 | 74.63+/-1.49 (5) | 76.85+/-1.25 (5) | 80.38+/-0.80 (5) | 82.44+/-0.42 (5) | NA | NA | NA | NA | NA |
| pubmed | uniform0.4 | 63.77+/-4.40 (5) | 67.54+/-3.11 (5) | 71.42+/-1.15 (5) | 71.90+/-0.88 (5) | 83.75+/-0.57 (5) | 83.25+/-0.72 (5) | 84.32+/-0.47 (5) | 84.80+/-0.85 (5) | 85.23+/-0.94 (5) |
| pubmed | uniform0.6 | 45.50+/-2.45 (5) | 47.89+/-3.86 (5) | 50.83+/-1.73 (5) | 47.53+/-2.42 (5) | 66.79+/-5.75 (5) | 73.24+/-3.57 (5) | 68.45+/-4.58 (5) | 74.74+/-2.50 (5) | 75.76+/-2.70 (5) |
| pubmed | pair0.4 | 59.48+/-1.38 (5) | 60.12+/-2.01 (5) | 66.64+/-3.23 (5) | 68.49+/-1.27 (5) | NA | NA | NA | NA | NA |
| pubmed | instance0.4 | 62.69+/-2.78 (5) | 67.69+/-3.41 (5) | 70.91+/-0.43 (5) | 72.01+/-1.23 (5) | NA | NA | NA | NA | NA |
| coauthorcs | uniform0.2 | 91.38+/-0.54 (5) | 91.63+/-0.58 (5) | 89.04+/-0.69 (5) | 91.56+/-0.25 (5) | NA | NA | NA | NA | NA |
| coauthorcs | uniform0.4 | 88.42+/-0.62 (5) | 88.95+/-0.43 (5) | 86.55+/-0.78 (5) | 90.62+/-0.57 (5) | 92.06+/-0.34 (5) | 92.05+/-0.19 (5) | NA | NA | NA |
| coauthorcs | uniform0.6 | 80.03+/-1.91 (5) | 81.28+/-1.32 (5) | 79.32+/-1.05 (5) | 87.17+/-0.62 (5) | NA | NA | NA | NA | NA |
| coauthorcs | pair0.4 | 75.91+/-1.34 (5) | 76.57+/-0.90 (5) | 75.66+/-2.82 (5) | 85.96+/-1.41 (5) | NA | NA | NA | NA | NA |
| coauthorcs | instance0.4 | 87.44+/-1.15 (5) | 88.07+/-1.07 (5) | 86.80+/-1.35 (5) | 90.50+/-0.49 (5) | NA | NA | NA | NA | NA |
| amazonpho | uniform0.2 | 82.34+/-1.39 (5) | 84.65+/-0.94 (5) | 82.10+/-1.41 (5) | 86.20+/-1.46 (5) | NA | NA | NA | NA | NA |
| amazonpho | uniform0.4 | 73.90+/-1.80 (5) | 77.05+/-1.24 (5) | 73.31+/-2.00 (5) | 81.64+/-0.93 (5) | 90.77+/-0.43 (5) | 91.87+/-0.64 (5) | NA | NA | NA |
| amazonpho | uniform0.6 | 58.60+/-3.77 (5) | 65.90+/-4.53 (5) | 56.57+/-1.67 (5) | 70.78+/-3.19 (5) | NA | NA | NA | NA | NA |
| amazonpho | pair0.4 | 62.61+/-4.31 (5) | 63.76+/-5.90 (5) | 64.95+/-3.75 (5) | 72.51+/-4.02 (5) | NA | NA | NA | NA | NA |
| amazonpho | instance0.4 | 69.13+/-3.00 (5) | 70.88+/-3.65 (5) | 69.73+/-3.30 (5) | 79.59+/-1.16 (5) | NA | NA | NA | NA | NA |

## Paired Deltas
| Method | Baseline | Dataset | Noise | n | Delta | Pos |
|---|---|---|---|---:|---:|---:|
| gsdn | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.2 | 5 | +3.26pp | 5/5 stable=1 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.4 | 5 | +0.38pp | 3/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.6 | 5 | -4.28pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | cora | pair0.4 | 5 | +0.49pp | 3/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | cora | instance0.4 | 5 | -1.17pp | 1/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.2 | 5 | -2.57pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.4 | 5 | -3.47pp | 1/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.6 | 5 | -4.32pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | citeseer | pair0.4 | 5 | -3.10pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | citeseer | instance0.4 | 5 | -3.10pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.2 | 5 | -2.06pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.4 | 5 | -0.49pp | 2/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.6 | 5 | +3.30pp | 5/5 stable=1 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | pubmed | pair0.4 | 5 | -1.85pp | 1/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | pubmed | instance0.4 | 5 | -1.10pp | 1/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.2 | 5 | -2.53pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.4 | 5 | -4.07pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.6 | 5 | -7.85pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | pair0.4 | 5 | -10.30pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | instance0.4 | 5 | -3.70pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.2 | 5 | -4.10pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.4 | 5 | -8.32pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.6 | 5 | -14.21pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | amazonpho | pair0.4 | 5 | -7.56pp | 0/5 stable=0 |
| gsdn | frozengrace_gb_mlp_softdist_beta05 | amazonpho | instance0.4 | 5 | -9.87pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | cora | uniform0.4 | 5 | -25.02pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | cora | uniform0.6 | 5 | -34.31pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | citeseer | uniform0.4 | 5 | -19.21pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | citeseer | uniform0.6 | 5 | -26.11pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | pubmed | uniform0.4 | 5 | -13.81pp | 0/5 stable=0 |
| gsdn | mlp_distill_knn_xh | pubmed | uniform0.6 | 5 | -24.93pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.2 | 5 | +10.87pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.4 | 5 | +7.64pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | cora | uniform0.6 | 5 | +4.61pp | 4/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | cora | pair0.4 | 5 | +4.70pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | cora | instance0.4 | 5 | +8.88pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.2 | 5 | +4.26pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.4 | 5 | +3.40pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | citeseer | uniform0.6 | 5 | +3.87pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | citeseer | pair0.4 | 5 | +2.42pp | 3/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | citeseer | instance0.4 | 5 | +3.24pp | 5/5 stable=1 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.2 | 5 | -5.58pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.4 | 5 | -4.36pp | 1/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | pubmed | uniform0.6 | 5 | +0.36pp | 3/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | pubmed | pair0.4 | 5 | -8.37pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | pubmed | instance0.4 | 5 | -4.32pp | 1/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.2 | 5 | +0.06pp | 3/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.4 | 5 | -1.67pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | uniform0.6 | 5 | -5.89pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | pair0.4 | 5 | -9.39pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | instance0.4 | 5 | -2.42pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.2 | 5 | -1.55pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.4 | 5 | -4.59pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | amazonpho | uniform0.6 | 5 | -4.88pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | amazonpho | pair0.4 | 5 | -8.75pp | 0/5 stable=0 |
| graphmlp | frozengrace_gb_mlp_softdist_beta05 | amazonpho | instance0.4 | 5 | -8.72pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | cora | uniform0.4 | 5 | -17.76pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | cora | uniform0.6 | 5 | -25.41pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | citeseer | uniform0.4 | 5 | -12.34pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | citeseer | uniform0.6 | 5 | -17.92pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | pubmed | uniform0.4 | 5 | -17.69pp | 0/5 stable=0 |
| graphmlp | mlp_distill_knn_xh | pubmed | uniform0.6 | 5 | -27.87pp | 0/5 stable=0 |

## Hyperparameter Record
| Method | Dataset | protocol | batch | lf | ll | ln | lc | tau |
|---|---|---|---:|---:|---:|---:|---:|---:|
| graphmlp | amazonpho | fixed_default_adapted_to_protocol | 512.0 |  |  |  | 0.5 | 0.5 |
| graphmlp | citeseer | fixed_default_adapted_to_protocol | 512.0 |  |  |  | 0.5 | 0.5 |
| graphmlp | coauthorcs | fixed_default_adapted_to_protocol | 512.0 |  |  |  | 0.5 | 0.5 |
| graphmlp | cora | fixed_default_adapted_to_protocol | 512.0 |  |  |  | 0.5 | 0.5 |
| graphmlp | pubmed | fixed_default_adapted_to_protocol | 512.0 |  |  |  | 0.5 | 0.5 |
| gsdn | amazonpho | fixed_default_adapted_to_protocol | 512.0 | 0.8 | 0.8 | 0.1 |  |  |
| gsdn | citeseer | fixed_default_adapted_to_protocol | 512.0 | 0.8 | 0.8 | 0.1 |  |  |
| gsdn | coauthorcs | fixed_default_adapted_to_protocol | 512.0 | 0.8 | 0.8 | 0.1 |  |  |
| gsdn | cora | fixed_default_adapted_to_protocol | 512.0 | 0.8 | 0.8 | 0.1 |  |  |
| gsdn | pubmed | fixed_default_adapted_to_protocol | 512.0 | 0.8 | 0.8 | 0.1 |  |  |

## Completeness
- Expected graph-free rows: 250
- Missing dataset/method/regime groups: 0
