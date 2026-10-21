# MLP-15 Review Defense P0 Report

Protocol: P0 pure recomputation; no model retraining in this report.

## P0a Beta Validation Audit

| dataset   | noise_type   |   noise_rate | val_best_method                    |   val_best_beta |   val_best_valid |   val_best_test | test_best_method                   |   test_best_beta |   test_best_valid |   test_best_test |   beta05_valid |   beta05_test |   val_best_minus_beta05_valid_pp |   test_best_minus_beta05_test_pp | val_and_test_choose_same_beta   |
|:----------|:-------------|-------------:|:-----------------------------------|----------------:|-----------------:|----------------:|:-----------------------------------|-----------------:|------------------:|-----------------:|---------------:|--------------:|---------------------------------:|---------------------------------:|:--------------------------------|
| citeseer  | uniform      |          0.2 | frozengrace_gb_mlp_softdist_beta1  |             1   |            67.29 |           62.89 | frozengrace_gb_mlp_softdist_beta05 |              0.5 |             67.23 |            63.38 |          67.23 |         63.38 |                             0.06 |                             0    | False                           |
| citeseer  | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta1  |             1   |            58.61 |           54.56 | frozengrace_gb_mlp_softdist_beta1  |              1   |             58.61 |            54.56 |          58.43 |         53.83 |                             0.18 |                             0.73 | True                            |
| cora      | uniform      |          0.2 | frozengrace_gb_mlp_softdist_beta1  |             1   |            66    |           63.68 | frozengrace_gb_mlp_softdist_beta1  |              1   |             66    |            63.68 |          64.89 |         62.94 |                             1.11 |                             0.74 | True                            |
| cora      | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta1  |             1   |            58.22 |           56.79 | frozengrace_gb_mlp_softdist_beta1  |              1   |             58.22 |            56.79 |          57.63 |         55.59 |                             0.59 |                             1.2  | True                            |
| pubmed    | uniform      |          0.2 | frozengrace_gb_mlp_softdist_beta05 |             0.5 |            82.62 |           82.51 | frozengrace_gb_mlp_softdist_beta05 |              0.5 |             82.62 |            82.51 |          82.62 |         82.51 |                             0    |                             0    | True                            |
| pubmed    | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta1  |             1   |            75.68 |           74.24 | frozengrace_gb_mlp_softdist_beta1  |              1   |             75.68 |            74.24 |          73.35 |         71.87 |                             2.32 |                             2.37 | True                            |


## P0b Paired Significance Summary

| source           | dataset    | noise_type   |   noise_rate | ours_label               | baseline_label               |   n |   mean_delta_pp |   positive_seeds |   paired_t_p |   wilcoxon_p |
|:-----------------|:-----------|:-------------|-------------:|:-------------------------|:-----------------------------|----:|----------------:|-----------------:|-------------:|-------------:|
| mlp10_main       | citeseer   | instance     |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           11.81 |                5 |    0.000189  |       0.0625 |
| mlp10_main       | citeseer   | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.29 |                5 |    0.0002789 |       0.0625 |
| mlp10_main       | citeseer   | pair         |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            5.04 |                5 |    0.01033   |       0.0625 |
| mlp10_main       | citeseer   | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            4.94 |                5 |    0.005956  |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.2 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            8.86 |                5 |    2.765e-05 |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.28 |                5 |    0.000107  |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           11.48 |                5 |    0.0006138 |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.34 |                5 |    0.0003458 |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.6 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            8.89 |                5 |    0.0001907 |       0.0625 |
| mlp10_main       | citeseer   | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            6.82 |                5 |    0.005699  |       0.0625 |
| mlp10_main       | coauthorcs | instance     |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            3.61 |                5 |    0.002719  |       0.0625 |
| mlp10_main       | coauthorcs | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            3.06 |                5 |    0.0007609 |       0.0625 |
| mlp10_main       | coauthorcs | pair         |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            9.91 |                5 |    6.275e-05 |       0.0625 |
| mlp10_main       | coauthorcs | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.05 |                5 |    5.019e-05 |       0.0625 |
| mlp10_main       | coauthorcs | uniform      |          0.2 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            1.16 |                5 |    0.004884  |       0.0625 |
| mlp10_main       | coauthorcs | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            0.18 |                3 |    0.3546    |       0.3125 |
| mlp10_main       | coauthorcs | uniform      |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            3.02 |                5 |    0.00233   |       0.0625 |
| mlp10_main       | coauthorcs | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            2.19 |                5 |    0.0008977 |       0.0625 |
| mlp10_main       | coauthorcs | uniform      |          0.6 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            8.33 |                5 |    0.001171  |       0.0625 |
| mlp10_main       | coauthorcs | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.14 |                5 |    0.00161   |       0.0625 |
| mlp10_main       | cora       | instance     |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           11.64 |                5 |    2.991e-05 |       0.0625 |
| mlp10_main       | cora       | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.34 |                5 |    0.0002059 |       0.0625 |
| mlp10_main       | cora       | pair         |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           10.91 |                5 |    0.001435  |       0.0625 |
| mlp10_main       | cora       | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.46 |                5 |    0.001764  |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.2 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            9.25 |                5 |    0.0001443 |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.99 |                5 |    0.0003399 |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           11.96 |                5 |    8.351e-08 |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           11.35 |                5 |    0.0001193 |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.6 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            9.34 |                5 |    0.002143  |       0.0625 |
| mlp10_main       | cora       | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.54 |                5 |    0.005883  |       0.0625 |
| mlp10_main       | pubmed     | instance     |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           10.96 |                5 |    0.001394  |       0.0625 |
| mlp10_main       | pubmed     | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            9.32 |                5 |    0.002238  |       0.0625 |
| mlp10_main       | pubmed     | pair         |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |           10.39 |                5 |    3.473e-05 |       0.0625 |
| mlp10_main       | pubmed     | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            9.01 |                5 |    0.0001411 |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.2 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            7.12 |                5 |    4.456e-05 |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.8  |                5 |    0.0002616 |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.4 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            9.18 |                5 |    0.00922   |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.13 |                5 |    0.01271   |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.6 | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled        |   5 |            3.26 |                5 |    0.008357  |       0.0625 |
| mlp10_main       | pubmed     | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            2.03 |                5 |    0.02426   |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.4 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           12.34 |                5 |    0.0002364 |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.4 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           19.21 |                5 |    0.0001813 |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            3.48 |                4 |    0.02688   |       0.125  |
| mlp14_distill    | citeseer   | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            0.76 |                4 |    0.1883    |       0.1875 |
| mlp14_distill    | citeseer   | uniform      |          0.6 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           17.92 |                5 |    3.651e-05 |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.6 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           26.11 |                5 |    7.074e-05 |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            8.66 |                5 |    0.0002924 |       0.0625 |
| mlp14_distill    | citeseer   | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            0.92 |                4 |    0.07734   |       0.125  |
| mlp14_distill    | cora       | uniform      |          0.4 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           17.76 |                5 |    9.643e-05 |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.4 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           25.02 |                5 |    7.972e-05 |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            3.04 |                5 |    0.02445   |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            0.76 |                4 |    0.08471   |       0.125  |
| mlp14_distill    | cora       | uniform      |          0.6 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           25.41 |                5 |    0.0001436 |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.6 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           34.31 |                5 |    9.848e-05 |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            5.57 |                5 |    0.0005288 |       0.0625 |
| mlp14_distill    | cora       | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            1.64 |                3 |    0.1599    |       0.25   |
| mlp14_distill    | pubmed     | uniform      |          0.4 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           17.69 |                5 |    0.0003107 |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.4 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           13.81 |                5 |    0.0001228 |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            0.91 |                5 |    0.02725   |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.4 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            0.42 |                4 |    0.09882   |       0.1875 |
| mlp14_distill    | pubmed     | uniform      |          0.6 | Ours-distill kNN teacher | Graph-MLP adapted            |   5 |           27.87 |                5 |    1.172e-05 |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.6 | Ours-distill kNN teacher | GSDN adapted                 |   5 |           24.93 |                5 |    5.086e-06 |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN teacher       |   5 |            7.32 |                5 |    0.003775  |       0.0625 |
| mlp14_distill    | pubmed     | uniform      |          0.6 | Ours-distill kNN teacher | GLNN-style GCN[X||H] teacher |   5 |            1.02 |                4 |    0.0504    |       0.125  |
| mlp14_train_only | amazonpho  | instance     |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            8.72 |                5 |    0.002674  |       0.0625 |
| mlp14_train_only | amazonpho  | instance     |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            9.87 |                5 |    0.003484  |       0.0625 |
| mlp14_train_only | amazonpho  | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.46 |                5 |    0.0003847 |       0.0625 |
| mlp14_train_only | amazonpho  | pair         |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            8.75 |                5 |    0.001801  |       0.0625 |
| mlp14_train_only | amazonpho  | pair         |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            7.56 |                5 |    0.0001136 |       0.0625 |
| mlp14_train_only | amazonpho  | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            9.9  |                5 |    6.125e-05 |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.2 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            1.55 |                5 |    0.006059  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.2 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            4.1  |                5 |    0.003704  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            3.86 |                5 |    7.341e-05 |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            4.59 |                5 |    0.001454  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            8.32 |                5 |    0.001458  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.73 |                5 |    0.000562  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.6 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            4.88 |                5 |    0.01388   |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.6 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           14.21 |                5 |    0.000162  |       0.0625 |
| mlp14_train_only | amazonpho  | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           12.19 |                5 |    0.0004629 |       0.0625 |
| mlp14_train_only | citeseer   | instance     |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -3.24 |                0 |    0.01423   |       0.0625 |
| mlp14_train_only | citeseer   | instance     |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            3.1  |                5 |    0.01312   |       0.0625 |
| mlp14_train_only | citeseer   | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.29 |                5 |    0.0002789 |       0.0625 |
| mlp14_train_only | citeseer   | pair         |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -2.42 |                2 |    0.1574    |       0.3125 |
| mlp14_train_only | citeseer   | pair         |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            3.1  |                5 |    0.01498   |       0.0625 |
| mlp14_train_only | citeseer   | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            4.94 |                5 |    0.005956  |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.2 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -4.26 |                0 |    0.0003173 |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.2 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            2.57 |                5 |    0.01154   |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.28 |                5 |    0.000107  |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -3.4  |                0 |    0.01702   |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            3.47 |                4 |    0.0548    |       0.125  |
| mlp14_train_only | citeseer   | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.34 |                5 |    0.0003458 |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.6 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -3.87 |                0 |    0.0001451 |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.6 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            4.32 |                5 |    0.02086   |       0.0625 |
| mlp14_train_only | citeseer   | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            6.82 |                5 |    0.005699  |       0.0625 |
| mlp14_train_only | coauthorcs | instance     |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            2.42 |                5 |    0.001607  |       0.0625 |
| mlp14_train_only | coauthorcs | instance     |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            3.7  |                5 |    0.0009338 |       0.0625 |
| mlp14_train_only | coauthorcs | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            3.06 |                5 |    0.0007609 |       0.0625 |
| mlp14_train_only | coauthorcs | pair         |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            9.39 |                5 |    4.173e-05 |       0.0625 |
| mlp14_train_only | coauthorcs | pair         |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           10.3  |                5 |    0.002306  |       0.0625 |
| mlp14_train_only | coauthorcs | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.05 |                5 |    5.019e-05 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.2 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -0.06 |                2 |    0.7838    |       0.625  |
| mlp14_train_only | coauthorcs | uniform      |          0.2 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            2.53 |                5 |    0.0005095 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            0.18 |                3 |    0.3546    |       0.3125 |
| mlp14_train_only | coauthorcs | uniform      |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            1.67 |                5 |    0.001648  |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            4.07 |                5 |    0.0003702 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            2.19 |                5 |    0.0008977 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.6 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            5.89 |                5 |    0.0007648 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.6 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            7.85 |                5 |    0.0002614 |       0.0625 |
| mlp14_train_only | coauthorcs | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.14 |                5 |    0.00161   |       0.0625 |
| mlp14_train_only | cora       | instance     |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -8.88 |                0 |    0.001298  |       0.0625 |
| mlp14_train_only | cora       | instance     |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            1.17 |                4 |    0.2333    |       0.3125 |
| mlp14_train_only | cora       | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           10.34 |                5 |    0.0002059 |       0.0625 |
| mlp14_train_only | cora       | pair         |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -4.7  |                0 |    0.002962  |       0.0625 |
| mlp14_train_only | cora       | pair         |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           -0.49 |                2 |    0.7286    |       1      |
| mlp14_train_only | cora       | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.46 |                5 |    0.001764  |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.2 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |          -10.87 |                0 |    1.088e-06 |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.2 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           -3.26 |                0 |    0.003428  |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.99 |                5 |    0.0003399 |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -7.64 |                0 |    0.004739  |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           -0.38 |                2 |    0.8429    |       1      |
| mlp14_train_only | cora       | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |           11.35 |                5 |    0.0001193 |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.6 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -4.61 |                1 |    0.09324   |       0.125  |
| mlp14_train_only | cora       | uniform      |          0.6 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            4.28 |                5 |    0.1057    |       0.0625 |
| mlp14_train_only | cora       | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.54 |                5 |    0.005883  |       0.0625 |
| mlp14_train_only | pubmed     | instance     |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            4.32 |                4 |    0.03395   |       0.125  |
| mlp14_train_only | pubmed     | instance     |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            1.1  |                4 |    0.09928   |       0.1875 |
| mlp14_train_only | pubmed     | instance     |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            9.32 |                5 |    0.002238  |       0.0625 |
| mlp14_train_only | pubmed     | pair         |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            8.37 |                5 |    2.68e-05  |       0.0625 |
| mlp14_train_only | pubmed     | pair         |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            1.85 |                4 |    0.143     |       0.125  |
| mlp14_train_only | pubmed     | pair         |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            9.01 |                5 |    0.0001411 |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.2 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            5.58 |                5 |    0.0004621 |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.2 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            2.06 |                5 |    0.003427  |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.2 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            7.8  |                5 |    0.0002616 |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.4 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |            4.36 |                4 |    0.03049   |       0.125  |
| mlp14_train_only | pubmed     | uniform      |          0.4 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |            0.49 |                3 |    0.4497    |       0.4375 |
| mlp14_train_only | pubmed     | uniform      |          0.4 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            8.13 |                5 |    0.01271   |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.6 | SoftDist-MLP beta=0.5    | Graph-MLP adapted            |   5 |           -0.36 |                2 |    0.7691    |       0.8125 |
| mlp14_train_only | pubmed     | uniform      |          0.6 | SoftDist-MLP beta=0.5    | GSDN adapted                 |   5 |           -3.3  |                0 |    0.01864   |       0.0625 |
| mlp14_train_only | pubmed     | uniform      |          0.6 | SoftDist-MLP beta=0.5    | MLP+CE                       |   5 |            2.03 |                5 |    0.02426   |       0.0625 |


## P2 Clean-Label Safety

| dataset    | MLP+CE     | SoftDist-MLP beta=0.5   | SoftDist-MLP shuffled   |
|:-----------|:-----------|:------------------------|:------------------------|
| amazonpho  | 88.82±0.77 | 87.62±0.62              | 86.67±0.93              |
| citeseer   | 64.55±0.75 | 66.88±0.43              | 63.68±0.77              |
| coauthorcs | 92.96±0.27 | 92.08±0.38              | 91.47±0.69              |
| cora       | 66.49±1.19 | 66.93±1.39              | 63.12±0.94              |
| pubmed     | 84.63±0.22 | 84.65±0.41              | 84.87±0.29              |


| dataset    | ours_label            | baseline_label        |   n |   mean_delta_pp |   positive_seeds |   paired_t_p |   wilcoxon_p |
|:-----------|:----------------------|:----------------------|----:|----------------:|-----------------:|-------------:|-------------:|
| amazonpho  | SoftDist-MLP beta=0.5 | MLP+CE                |   5 |           -1.2  |                0 |     0.001264 |       0.0625 |
| amazonpho  | SoftDist-MLP beta=0.5 | SoftDist-MLP shuffled |   5 |            0.95 |                5 |     0.01122  |       0.0625 |
| citeseer   | SoftDist-MLP beta=0.5 | MLP+CE                |   5 |            2.33 |                5 |     0.007858 |       0.0625 |
| citeseer   | SoftDist-MLP beta=0.5 | SoftDist-MLP shuffled |   5 |            3.2  |                5 |     0.002937 |       0.0625 |
| coauthorcs | SoftDist-MLP beta=0.5 | MLP+CE                |   5 |           -0.88 |                0 |     0.009489 |       0.0625 |
| coauthorcs | SoftDist-MLP beta=0.5 | SoftDist-MLP shuffled |   5 |            0.61 |                4 |     0.181    |       0.125  |
| cora       | SoftDist-MLP beta=0.5 | MLP+CE                |   5 |            0.44 |                3 |     0.5771   |       0.8125 |
| cora       | SoftDist-MLP beta=0.5 | SoftDist-MLP shuffled |   5 |            3.81 |                5 |     0.000556 |       0.0625 |
| pubmed     | SoftDist-MLP beta=0.5 | MLP+CE                |   5 |            0.02 |                3 |     0.9264   |       1      |
| pubmed     | SoftDist-MLP beta=0.5 | SoftDist-MLP shuffled |   5 |           -0.22 |                0 |     0.07814  |       0.0625 |


## P3 LP-Target Graph-Space Consensus Control

| dataset   | MLP+CE     | LP-target MLP beta=0.5   | SoftDist-MLP beta=0.5   | SoftDist-MLP shuffled   |
|:----------|:-----------|:-------------------------|:------------------------|:------------------------|
| citeseer  | 43.50±3.02 | 46.98±1.64               | 53.83±2.75              | 42.35±2.74              |
| cora      | 44.24±2.62 | 47.34±2.00               | 55.59±1.50              | 43.63±1.62              |
| pubmed    | 63.77±4.40 | 64.98±0.83               | 71.90±0.88              | 62.72±4.49              |


| dataset   | ours_label             | baseline_label         |   n |   mean_delta_pp |   positive_seeds |   paired_t_p |   wilcoxon_p |
|:----------|:-----------------------|:-----------------------|----:|----------------:|-----------------:|-------------:|-------------:|
| citeseer  | SoftDist-MLP beta=0.5  | LP-target MLP beta=0.5 |   5 |            6.86 |                5 |    0.001187  |       0.0625 |
| citeseer  | LP-target MLP beta=0.5 | MLP+CE                 |   5 |            3.48 |                5 |    0.018     |       0.0625 |
| citeseer  | LP-target MLP beta=0.5 | SoftDist-MLP shuffled  |   5 |            4.62 |                5 |    0.004479  |       0.0625 |
| cora      | SoftDist-MLP beta=0.5  | LP-target MLP beta=0.5 |   5 |            8.25 |                5 |    3.385e-05 |       0.0625 |
| cora      | LP-target MLP beta=0.5 | MLP+CE                 |   5 |            3.09 |                5 |    0.01048   |       0.0625 |
| cora      | LP-target MLP beta=0.5 | SoftDist-MLP shuffled  |   5 |            3.7  |                5 |    0.001595  |       0.0625 |
| pubmed    | SoftDist-MLP beta=0.5  | LP-target MLP beta=0.5 |   5 |            6.92 |                5 |    1.798e-06 |       0.0625 |
| pubmed    | LP-target MLP beta=0.5 | MLP+CE                 |   5 |            1.21 |                4 |    0.5575    |       0.625  |
| pubmed    | LP-target MLP beta=0.5 | SoftDist-MLP shuffled  |   5 |            2.26 |                4 |    0.3125    |       0.625  |


## P4 PubMed Linear Robust-Baseline Fairness

| noise_type   |   noise_rate | Linear+CE   | SoftDist-Linear beta=0.5   | SoftDist-Linear shuffled   |
|:-------------|-------------:|:------------|:---------------------------|:---------------------------|
| instance     |          0.4 | 79.09±1.26  | 79.90±1.36                 | 78.60±1.19                 |
| pair         |          0.4 | 66.61±1.46  | 69.94±1.16                 | 63.67±0.90                 |
| uniform      |          0.2 | 81.42±0.99  | 80.70±1.02                 | 80.30±1.62                 |
| uniform      |          0.4 | 79.35±0.45  | 80.03±0.57                 | 78.15±0.45                 |
| uniform      |          0.6 | 57.32±6.02  | 59.66±5.68                 | 56.05±5.86                 |


| dataset   | noise_type   |   noise_rate |   softdist_linear | best_linear_robust_method   |   best_linear_robust |   delta_vs_best_linear_robust_pp |
|:----------|:-------------|-------------:|------------------:|:----------------------------|---------------------:|---------------------------------:|
| pubmed    | instance     |          0.4 |             79.9  | linear_dividemix_lite       |                65.64 |                            14.26 |
| pubmed    | pair         |          0.4 |             69.94 | linear_small_loss           |                55.84 |                            14.1  |
| pubmed    | uniform      |          0.2 |             80.7  | linear_small_loss           |                69.6  |                            11.1  |
| pubmed    | uniform      |          0.4 |             80.03 | linear_dividemix_lite       |                66.32 |                            13.71 |
| pubmed    | uniform      |          0.6 |             59.66 | linear_dividemix_lite       |                52.27 |                             7.39 |


| noise_type   |   noise_rate | ours_label               | baseline_label           |   n |   mean_delta_pp |   positive_seeds |   paired_t_p |   wilcoxon_p |
|:-------------|-------------:|:-------------------------|:-------------------------|----:|----------------:|-----------------:|-------------:|-------------:|
| instance     |          0.4 | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            0.82 |                5 |    0.0002448 |       0.0625 |
| instance     |          0.4 | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            1.3  |                5 |    0.0002274 |       0.0625 |
| instance     |          0.4 | SoftDist-Linear beta=0.5 | linear_ct                |   5 |           16.38 |                5 |    1.106e-06 |       0.0625 |
| instance     |          0.4 | SoftDist-Linear beta=0.5 | linear_dividemix_lite    |   5 |           14.26 |                5 |    8.712e-07 |       0.0625 |
| instance     |          0.4 | SoftDist-Linear beta=0.5 | linear_jocor             |   5 |           18.8  |                5 |    0.0002845 |       0.0625 |
| instance     |          0.4 | SoftDist-Linear beta=0.5 | linear_small_loss        |   5 |           20.28 |                5 |    0.0005196 |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            3.33 |                5 |    0.0003852 |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            6.27 |                5 |    1.503e-05 |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | linear_ct                |   5 |           17.55 |                5 |    0.0001828 |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | linear_dividemix_lite    |   5 |           19.71 |                5 |    0.0007357 |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | linear_jocor             |   5 |           16.39 |                5 |    3.21e-05  |       0.0625 |
| pair         |          0.4 | SoftDist-Linear beta=0.5 | linear_small_loss        |   5 |           14.1  |                5 |    0.003224  |       0.0625 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |           -0.72 |                0 |    0.01779   |       0.0625 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            0.4  |                4 |    0.4082    |       0.3125 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | linear_ct                |   5 |           11.3  |                5 |    7.476e-05 |       0.0625 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | linear_dividemix_lite    |   5 |           13.02 |                5 |    1.117e-06 |       0.0625 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | linear_jocor             |   5 |           11.69 |                5 |    1.993e-05 |       0.0625 |
| uniform      |          0.2 | SoftDist-Linear beta=0.5 | linear_small_loss        |   5 |           11.1  |                5 |    0.0001137 |       0.0625 |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            0.68 |                4 |    0.1039    |       0.125  |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            1.88 |                5 |    0.003368  |       0.0625 |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | linear_ct                |   5 |           16.98 |                5 |    5.673e-07 |       0.0625 |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | linear_dividemix_lite    |   5 |           13.71 |                5 |    1.802e-06 |       0.0625 |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | linear_jocor             |   5 |           20.24 |                5 |    0.0002011 |       0.0625 |
| uniform      |          0.4 | SoftDist-Linear beta=0.5 | linear_small_loss        |   5 |           16.92 |                5 |    9.312e-07 |       0.0625 |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            2.35 |                4 |    0.04755   |       0.125  |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            3.61 |                5 |    0.009837  |       0.0625 |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | linear_ct                |   5 |            9.48 |                5 |    0.007698  |       0.0625 |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | linear_dividemix_lite    |   5 |            7.39 |                5 |    0.01253   |       0.0625 |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | linear_jocor             |   5 |           16.01 |                5 |    0.001108  |       0.0625 |
| uniform      |          0.6 | SoftDist-Linear beta=0.5 | linear_small_loss        |   5 |           16.2  |                5 |    0.0004188 |       0.0625 |


## P0c Completed Linear Capacity Ablation

| dataset    | MLP+CE     | Linear+CE   | SoftDist-MLP beta=0.5   | SoftDist-Linear beta=0.5   | SoftDist-MLP shuffled   | SoftDist-Linear shuffled   |
|:-----------|:-----------|:------------|:------------------------|:---------------------------|:------------------------|:---------------------------|
| amazonpho  | 73.90±1.80 | 82.76±1.11  | 81.64±0.93              | 86.46±0.37                 | 72.06±2.97              | 81.61±0.81                 |
| citeseer   | 43.50±3.02 | 50.12±1.49  | 53.83±2.75              | 54.88±3.36                 | 42.35±2.74              | 48.47±2.05                 |
| coauthorcs | 88.42±0.62 | 89.59±0.18  | 90.62±0.57              | 91.03±0.30                 | 87.60±1.03              | 89.33±0.11                 |
| cora       | 44.24±2.62 | 47.86±1.30  | 55.59±1.50              | 54.06±2.26                 | 43.63±1.62              | 47.84±1.59                 |
| pubmed     | 63.77±4.40 | 79.35±0.45  | 71.90±0.88              | 80.03±0.57                 | 62.72±4.49              | 78.15±0.45                 |


| dataset    | ours_label               | baseline_label           |   n |   mean_delta_pp |   positive_seeds |   paired_t_p |   wilcoxon_p |
|:-----------|:-------------------------|:-------------------------|----:|----------------:|-----------------:|-------------:|-------------:|
| amazonpho  | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            4.85 |                5 |    0.0005279 |       0.0625 |
| amazonpho  | SoftDist-Linear beta=0.5 | SoftDist-MLP beta=0.5    |   5 |            4.83 |                5 |    0.0001429 |       0.0625 |
| amazonpho  | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            3.7  |                5 |    0.004023  |       0.0625 |
| amazonpho  | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled    |   5 |            9.58 |                5 |    0.00383   |       0.0625 |
| amazonpho  | SoftDist-MLP beta=0.5    | MLP+CE                   |   5 |            7.73 |                5 |    0.000562  |       0.0625 |
| citeseer   | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            6.41 |                5 |    0.00415   |       0.0625 |
| citeseer   | SoftDist-Linear beta=0.5 | SoftDist-MLP beta=0.5    |   5 |            1.05 |                4 |    0.06209   |       0.125  |
| citeseer   | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            4.76 |                5 |    0.00894   |       0.0625 |
| citeseer   | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled    |   5 |           11.48 |                5 |    0.0006138 |       0.0625 |
| citeseer   | SoftDist-MLP beta=0.5    | MLP+CE                   |   5 |           10.34 |                5 |    0.0003458 |       0.0625 |
| coauthorcs | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            1.7  |                5 |    0.0004271 |       0.0625 |
| coauthorcs | SoftDist-Linear beta=0.5 | SoftDist-MLP beta=0.5    |   5 |            0.41 |                3 |    0.2177    |       0.3125 |
| coauthorcs | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            1.44 |                5 |    0.002452  |       0.0625 |
| coauthorcs | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled    |   5 |            3.02 |                5 |    0.00233   |       0.0625 |
| coauthorcs | SoftDist-MLP beta=0.5    | MLP+CE                   |   5 |            2.19 |                5 |    0.0008977 |       0.0625 |
| cora       | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            6.22 |                5 |    9.427e-05 |       0.0625 |
| cora       | SoftDist-Linear beta=0.5 | SoftDist-MLP beta=0.5    |   5 |           -1.53 |                1 |    0.1163    |       0.125  |
| cora       | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            6.2  |                5 |    0.0009157 |       0.0625 |
| cora       | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled    |   5 |           11.96 |                5 |    8.351e-08 |       0.0625 |
| cora       | SoftDist-MLP beta=0.5    | MLP+CE                   |   5 |           11.35 |                5 |    0.0001193 |       0.0625 |
| pubmed     | SoftDist-Linear beta=0.5 | SoftDist-Linear shuffled |   5 |            1.88 |                5 |    0.003368  |       0.0625 |
| pubmed     | SoftDist-Linear beta=0.5 | SoftDist-MLP beta=0.5    |   5 |            8.13 |                5 |    0.0001536 |       0.0625 |
| pubmed     | SoftDist-Linear beta=0.5 | Linear+CE                |   5 |            0.68 |                4 |    0.1039    |       0.125  |
| pubmed     | SoftDist-MLP beta=0.5    | SoftDist-MLP shuffled    |   5 |            9.18 |                5 |    0.00922   |       0.0625 |
| pubmed     | SoftDist-MLP beta=0.5    | MLP+CE                   |   5 |            8.13 |                5 |    0.01271   |       0.0625 |

