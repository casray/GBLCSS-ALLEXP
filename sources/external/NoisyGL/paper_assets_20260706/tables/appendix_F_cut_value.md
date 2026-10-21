# Appendix F. Cut Value table

Footnote: high CV for CE/shuffled can be contaminated by prediction collapse under noisy-label protocols. CV is therefore not used as a clean structure metric here; the distill-vs-train-only rows are the only mechanism evidence quoted in the manuscript.

| dataset | method | method_label | n | mean | std |
| --- | --- | --- | --- | --- | --- |
| amazonpho | frozengrace_gb_mlp_softdist_beta05 | Ours beta=0.5 train-only | 5 | 0.8120 | 0.0169 |
| amazonpho | frozengrace_gb_mlp_softdist_beta05_shuffled | Shuffled local consensus | 5 | 0.8408 | 0.0154 |
| amazonpho | knn_mlp_softdist_beta05 | kNN@matched beta=0.5 | 5 | 0.8178 | 0.0200 |
| amazonpho | mlp_ce | MLP+CE | 5 | 0.8135 | 0.0070 |
| citeseer | frozengrace_gb_mlp_softdist_beta05 | Ours beta=0.5 train-only | 5 | 0.7042 | 0.0431 |
| citeseer | frozengrace_gb_mlp_softdist_beta05_shuffled | Shuffled local consensus | 5 | 0.6566 | 0.1862 |
| citeseer | knn_mlp_softdist_beta05 | kNN@matched beta=0.5 | 5 | 0.7537 | 0.0712 |
| citeseer | mlp_ce | MLP+CE | 5 | 0.6648 | 0.2764 |
| citeseer | mlp_distill_gcn | GLNN-style T1 | 5 | 0.9509 | 0.0128 |
| citeseer | mlp_distill_softdist_xh | Ours-distill T3 | 5 | 0.9502 | 0.0034 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05 | Ours beta=0.5 train-only | 5 | 0.7623 | 0.0101 |
| coauthorcs | frozengrace_gb_mlp_softdist_beta05_shuffled | Shuffled local consensus | 5 | 0.7402 | 0.0208 |
| coauthorcs | knn_mlp_softdist_beta05 | kNN@matched beta=0.5 | 5 | 0.7691 | 0.0129 |
| coauthorcs | mlp_ce | MLP+CE | 5 | 0.7293 | 0.0133 |
| cora | frozengrace_gb_mlp_softdist_beta05 | Ours beta=0.5 train-only | 5 | 0.6674 | 0.0927 |
| cora | frozengrace_gb_mlp_softdist_beta05_shuffled | Shuffled local consensus | 5 | 0.7187 | 0.2478 |
| cora | knn_mlp_softdist_beta05 | kNN@matched beta=0.5 | 5 | 0.7200 | 0.1462 |
| cora | mlp_ce | MLP+CE | 5 | 0.8476 | 0.2423 |
| cora | mlp_distill_gcn | GLNN-style T1 | 5 | 0.9056 | 0.0112 |
| cora | mlp_distill_softdist_xh | Ours-distill T3 | 5 | 0.9132 | 0.0058 |
| pubmed | frozengrace_gb_mlp_softdist_beta05 | Ours beta=0.5 train-only | 5 | 0.8401 | 0.0773 |
| pubmed | frozengrace_gb_mlp_softdist_beta05_shuffled | Shuffled local consensus | 5 | 0.9103 | 0.1521 |
| pubmed | knn_mlp_softdist_beta05 | kNN@matched beta=0.5 | 5 | 0.8134 | 0.0691 |
| pubmed | mlp_ce | MLP+CE | 5 | 0.8722 | 0.2003 |
| pubmed | mlp_distill_gcn | GLNN-style T1 | 5 | 0.9795 | 0.0037 |
| pubmed | mlp_distill_softdist_xh | Ours-distill T3 | 5 | 0.9789 | 0.0032 |


Raw per-seed rows are preserved in `mlp16_cut_value_raw.csv`.
