# MLP-32 Uniform0.4 Cross-Resource Five-Seed Table

| Dataset | Direct robust X-only | Plain distill X-only | NRGNN | NRGNN[X||H] | Distill-NRGNN | Positive |
|---|---:|---:|---:|---:|---:|---:|
| cora | 59.49+/-1.02 | 80.23+/-1.41 | 81.76+/-1.14 | 81.47+/-1.14 | -1.53pp | 1/5 |
| citeseer | 58.28+/-2.39 | 68.81+/-1.00 | 71.03+/-0.90 | 70.74+/-0.93 | -2.23pp | 0/5 |
| corafull | 40.63+/-0.50 | 59.45+/-0.56 | 62.70+/-0.57 | 62.06+/-0.46 | -3.25pp | 0/5 |
| amazonpho | 82.73+/-0.88 | 90.51+/-0.96 | 89.02+/-0.97 | 89.53+/-1.37 | +1.49pp | 5/5 |
| pubmed | 81.47+/-0.29 | 84.80+/-0.76 | 80.69+/-0.59 | 83.29+/-0.37 | +4.11pp | 5/5 |
| coauthorph | 93.21+/-0.35 | 95.44+/-0.15 | 95.84+/-0.18 | 95.89+/-0.14 | -0.40pp | 0/5 |

## Resource profiles

| Method | Training graph | Inference graph | H at inference |
|---|---|---|---|
| frozengrace_gb_mlp_softdist_beta05_small_loss | GRACE preprocessing | no | no |
| mlp_distill_gcn_xh | GCN[X||H] teacher | no | no |
| nrgnn | yes | yes | no |
| nrgnn_xh | yes | yes | yes |
