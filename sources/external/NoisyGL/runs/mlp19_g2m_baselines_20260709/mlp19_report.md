# MLP-19 GNN-to-MLP Baselines Report

## Material Passport
- Table role: Table B / graph-free GNN-to-MLP family expansion only.
- Protocol: Cora/CiteSeer/PubMed/Coauthor-CS/Amazon-Photo, uniform0.4, train-only noise, percent split 0.2/0.1/0.7.
- New runnable methods are adapted to this repository protocol, not official reproduction.
- VQGraph is implemented as an adapted core mechanism, not a line-by-line official reproduction.

## Resource Annotation
| Method | train graph | infer graph | unlabeled nodes | adapted | note |
|---|---:|---:|---:|---:|---|
| FF-G2M adapted | 1 | 0 | 1 | 1 | GCN teacher plus low/high-frequency adapted distillation to X-only MLP |
| KRD adapted | 1 | 0 | 1 | 1 | GCN teacher reliability estimated from entropy stability, then distilled to X-only MLP |
| VQGraph adapted | 1 | 0 | 1 | 1 | GCN teacher plus adapted structure tokenizer/codebook and soft token distillation to X-only MLP |

## Uniform0.4 Table B Slice
| Dataset | MLP | Graph-MLP | GSDN | Ours train-only | GLNN | LCSS-distill | FF-G2M adapted | KRD adapted | VQGraph adapted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | 44.24+/-2.62 (5) | 63.23+/-3.07 (5) | 55.97+/-4.29 (5) | 55.59+/-1.50 (5) | 77.95+/-2.62 (5) | 80.60+/-0.69 (5) | 78.73+/-2.62 (5) | 71.77+/-2.20 (5) | 78.89+/-2.34 (5) |
| citeseer | 43.50+/-3.02 (5) | 57.23+/-1.85 (5) | 50.36+/-3.39 (5) | 53.83+/-2.75 (5) | 66.09+/-1.38 (5) | 69.40+/-1.08 (5) | 66.17+/-1.30 (5) | 64.96+/-1.03 (5) | 65.64+/-1.72 (5) |
| pubmed | 63.77+/-4.40 (5) | 67.54+/-3.11 (5) | 71.42+/-1.15 (5) | 71.90+/-0.88 (5) | 84.32+/-0.47 (5) | 85.01+/-0.93 (5) | 84.08+/-0.49 (5) | 83.81+/-0.63 (5) | 84.26+/-0.69 (5) |
| coauthorcs | 88.42+/-0.62 (5) | 88.95+/-0.43 (5) | 86.55+/-0.78 (5) | 90.62+/-0.57 (5) | 93.20+/-0.09 (5) | 93.75+/-0.22 (5) | 93.12+/-0.24 (5) | 91.39+/-0.49 (5) | 93.25+/-0.09 (5) |
| amazonpho | 73.90+/-1.80 (5) | 77.05+/-1.24 (5) | 73.31+/-2.00 (5) | 81.64+/-0.93 (5) | 89.86+/-0.77 (5) | 90.49+/-1.05 (5) | 90.01+/-1.14 (5) | 85.17+/-0.94 (5) | 90.54+/-0.85 (5) |

## Paired Deltas
| Method | Baseline | Dataset | n | Delta | Positive |
|---|---|---|---:|---:|---:|
| ffg2m_adapted | frozengrace_gb_mlp_softdist_beta05 | cora | 5 | +23.15pp | 5/5 |
| ffg2m_adapted | frozengrace_gb_mlp_softdist_beta05 | citeseer | 5 | +12.34pp | 5/5 |
| ffg2m_adapted | frozengrace_gb_mlp_softdist_beta05 | pubmed | 5 | +12.18pp | 5/5 |
| ffg2m_adapted | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | 5 | +2.50pp | 5/5 |
| ffg2m_adapted | frozengrace_gb_mlp_softdist_beta05 | amazonpho | 5 | +8.37pp | 5/5 |
| ffg2m_adapted | mlp_distill_softdist_xh | cora | 5 | -1.87pp | 1/5 |
| ffg2m_adapted | mlp_distill_softdist_xh | citeseer | 5 | -3.23pp | 0/5 |
| ffg2m_adapted | mlp_distill_softdist_xh | pubmed | 5 | -0.93pp | 1/5 |
| ffg2m_adapted | mlp_distill_softdist_xh | coauthorcs | 5 | -0.63pp | 0/5 |
| ffg2m_adapted | mlp_distill_softdist_xh | amazonpho | 5 | -0.49pp | 2/5 |
| ffg2m_adapted | graphmlp | cora | 5 | +15.50pp | 5/5 |
| ffg2m_adapted | graphmlp | citeseer | 5 | +8.93pp | 5/5 |
| ffg2m_adapted | graphmlp | pubmed | 5 | +16.54pp | 5/5 |
| ffg2m_adapted | graphmlp | coauthorcs | 5 | +4.17pp | 5/5 |
| ffg2m_adapted | graphmlp | amazonpho | 5 | +12.96pp | 5/5 |
| ffg2m_adapted | gsdn | cora | 5 | +22.77pp | 5/5 |
| ffg2m_adapted | gsdn | citeseer | 5 | +15.81pp | 5/5 |
| ffg2m_adapted | gsdn | pubmed | 5 | +12.67pp | 5/5 |
| ffg2m_adapted | gsdn | coauthorcs | 5 | +6.57pp | 5/5 |
| ffg2m_adapted | gsdn | amazonpho | 5 | +16.69pp | 5/5 |
| krd_adapted | frozengrace_gb_mlp_softdist_beta05 | cora | 5 | +16.18pp | 5/5 |
| krd_adapted | frozengrace_gb_mlp_softdist_beta05 | citeseer | 5 | +11.13pp | 5/5 |
| krd_adapted | frozengrace_gb_mlp_softdist_beta05 | pubmed | 5 | +11.91pp | 5/5 |
| krd_adapted | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | 5 | +0.78pp | 3/5 |
| krd_adapted | frozengrace_gb_mlp_softdist_beta05 | amazonpho | 5 | +3.53pp | 5/5 |
| krd_adapted | mlp_distill_softdist_xh | cora | 5 | -8.83pp | 0/5 |
| krd_adapted | mlp_distill_softdist_xh | citeseer | 5 | -4.44pp | 0/5 |
| krd_adapted | mlp_distill_softdist_xh | pubmed | 5 | -1.20pp | 0/5 |
| krd_adapted | mlp_distill_softdist_xh | coauthorcs | 5 | -2.36pp | 0/5 |
| krd_adapted | mlp_distill_softdist_xh | amazonpho | 5 | -5.32pp | 0/5 |
| krd_adapted | graphmlp | cora | 5 | +8.54pp | 5/5 |
| krd_adapted | graphmlp | citeseer | 5 | +7.72pp | 5/5 |
| krd_adapted | graphmlp | pubmed | 5 | +16.27pp | 5/5 |
| krd_adapted | graphmlp | coauthorcs | 5 | +2.44pp | 5/5 |
| krd_adapted | graphmlp | amazonpho | 5 | +8.12pp | 5/5 |
| krd_adapted | gsdn | cora | 5 | +15.80pp | 5/5 |
| krd_adapted | gsdn | citeseer | 5 | +14.60pp | 5/5 |
| krd_adapted | gsdn | pubmed | 5 | +12.40pp | 5/5 |
| krd_adapted | gsdn | coauthorcs | 5 | +4.84pp | 5/5 |
| krd_adapted | gsdn | amazonpho | 5 | +11.85pp | 5/5 |
| vqgraph_adapted | frozengrace_gb_mlp_softdist_beta05 | cora | 5 | +23.30pp | 5/5 |
| vqgraph_adapted | frozengrace_gb_mlp_softdist_beta05 | citeseer | 5 | +11.80pp | 5/5 |
| vqgraph_adapted | frozengrace_gb_mlp_softdist_beta05 | pubmed | 5 | +12.36pp | 5/5 |
| vqgraph_adapted | frozengrace_gb_mlp_softdist_beta05 | coauthorcs | 5 | +2.64pp | 5/5 |
| vqgraph_adapted | frozengrace_gb_mlp_softdist_beta05 | amazonpho | 5 | +8.91pp | 5/5 |
| vqgraph_adapted | mlp_distill_softdist_xh | cora | 5 | -1.71pp | 1/5 |
| vqgraph_adapted | mlp_distill_softdist_xh | citeseer | 5 | -3.76pp | 0/5 |
| vqgraph_adapted | mlp_distill_softdist_xh | pubmed | 5 | -0.75pp | 1/5 |
| vqgraph_adapted | mlp_distill_softdist_xh | coauthorcs | 5 | -0.50pp | 0/5 |
| vqgraph_adapted | mlp_distill_softdist_xh | amazonpho | 5 | +0.05pp | 2/5 |
| vqgraph_adapted | graphmlp | cora | 5 | +15.66pp | 5/5 |
| vqgraph_adapted | graphmlp | citeseer | 5 | +8.40pp | 5/5 |
| vqgraph_adapted | graphmlp | pubmed | 5 | +16.72pp | 5/5 |
| vqgraph_adapted | graphmlp | coauthorcs | 5 | +4.30pp | 5/5 |
| vqgraph_adapted | graphmlp | amazonpho | 5 | +13.50pp | 5/5 |
| vqgraph_adapted | gsdn | cora | 5 | +22.92pp | 5/5 |
| vqgraph_adapted | gsdn | citeseer | 5 | +15.27pp | 5/5 |
| vqgraph_adapted | gsdn | pubmed | 5 | +12.85pp | 5/5 |
| vqgraph_adapted | gsdn | coauthorcs | 5 | +6.70pp | 5/5 |
| vqgraph_adapted | gsdn | amazonpho | 5 | +17.23pp | 5/5 |

## Not Runnable Records
| Dataset | Method | Seeds | Reason |
|---|---|---|---|
| - | - | - | none |

## Completeness
| Dataset | Method | expected | observed | missing |
|---|---|---:|---:|---|
| cora | ffg2m_adapted | 5 | 5 |  |
| cora | krd_adapted | 5 | 5 |  |
| cora | vqgraph_adapted | 5 | 5 |  |
| citeseer | ffg2m_adapted | 5 | 5 |  |
| citeseer | krd_adapted | 5 | 5 |  |
| citeseer | vqgraph_adapted | 5 | 5 |  |
| pubmed | ffg2m_adapted | 5 | 5 |  |
| pubmed | krd_adapted | 5 | 5 |  |
| pubmed | vqgraph_adapted | 5 | 5 |  |
| coauthorcs | ffg2m_adapted | 5 | 5 |  |
| coauthorcs | krd_adapted | 5 | 5 |  |
| coauthorcs | vqgraph_adapted | 5 | 5 |  |
| amazonpho | ffg2m_adapted | 5 | 5 |  |
| amazonpho | krd_adapted | 5 | 5 |  |
| amazonpho | vqgraph_adapted | 5 | 5 |  |
