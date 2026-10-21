# MLP-23 Dataset Breadth Report

Protocol: percent 0.2/0.1/0.7, train-only uniform 0.4, seeds 0-4.

## Gate Decisions

| Dataset | Decision | Ours | Best baseline | Ours-best | Positive | Ours-shuffled | Positive | Complete/Audit |
|---|---|---:|---|---:|---:|---:|---:|---:|
| coraml | boundary | 55.78 | mlp_small_loss 55.56 | +0.22pp | 2/5 | +6.71pp | 5/5 | 1/1 |
| attribwiki | boundary | 65.42 | mlp_logitclip 67.83 | -2.41pp | 0/5 | +5.94pp | 5/5 | 1/1 |
| citeseerfull | boundary | 59.40 | mlp_jocor 59.48 | -0.08pp | 3/5 | +13.14pp | 5/5 | 1/1 |
| coauthorph | boundary | 93.11 | mlp_jocor 93.09 | +0.02pp | 3/5 | +4.21pp | 5/5 | 1/1 |

## Uniform 0.4 Aggregate

| Dataset | Method | n | Mean±Std |
|---|---|---:|---:|
| attribwiki | mlp_logitclip | 5 | 67.83±1.51 |
| attribwiki | frozengrace_gb_mlp_softdist_beta05 | 5 | 65.42±1.20 |
| attribwiki | mlp_jocor | 5 | 64.80±1.96 |
| attribwiki | mlp_coteaching | 5 | 63.81±2.80 |
| attribwiki | mlp_ct | 5 | 63.73±1.79 |
| attribwiki | mlp_small_loss | 5 | 63.14±2.61 |
| attribwiki | mlp_sce | 5 | 62.96±3.01 |
| attribwiki | mlp_elr | 5 | 62.54±2.58 |
| attribwiki | mlp_anl | 5 | 62.51±3.17 |
| attribwiki | mlp_ce | 5 | 62.45±2.20 |
| attribwiki | mlp_dividemix_lite | 5 | 61.70±3.18 |
| attribwiki | mlp_fine | 5 | 60.27±1.84 |
| attribwiki | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 59.48±1.91 |
| citeseerfull | mlp_jocor | 5 | 59.48±1.60 |
| citeseerfull | frozengrace_gb_mlp_softdist_beta05 | 5 | 59.40±1.73 |
| citeseerfull | mlp_small_loss | 5 | 58.43±2.23 |
| citeseerfull | mlp_ct | 5 | 57.03±1.56 |
| citeseerfull | mlp_coteaching | 5 | 56.33±2.15 |
| citeseerfull | mlp_logitclip | 5 | 52.76±2.38 |
| citeseerfull | mlp_dividemix_lite | 5 | 51.84±1.48 |
| citeseerfull | mlp_anl | 5 | 51.25±3.81 |
| citeseerfull | mlp_sce | 5 | 50.68±4.87 |
| citeseerfull | mlp_elr | 5 | 50.56±3.36 |
| citeseerfull | mlp_ce | 5 | 49.71±3.50 |
| citeseerfull | mlp_fine | 5 | 49.63±2.57 |
| citeseerfull | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 46.25±3.18 |
| coauthorph | frozengrace_gb_mlp_softdist_beta05 | 5 | 93.11±0.25 |
| coauthorph | mlp_jocor | 5 | 93.09±0.28 |
| coauthorph | mlp_anl | 5 | 93.06±0.25 |
| coauthorph | mlp_sce | 5 | 93.02±0.35 |
| coauthorph | mlp_ct | 5 | 92.76±0.37 |
| coauthorph | mlp_small_loss | 5 | 92.71±0.32 |
| coauthorph | mlp_coteaching | 5 | 92.69±0.30 |
| coauthorph | mlp_dividemix_lite | 5 | 92.51±0.29 |
| coauthorph | mlp_logitclip | 5 | 92.19±0.44 |
| coauthorph | mlp_elr | 5 | 92.07±0.55 |
| coauthorph | mlp_ce | 5 | 91.25±0.77 |
| coauthorph | mlp_fine | 5 | 90.30±1.26 |
| coauthorph | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 88.90±2.05 |
| coraml | frozengrace_gb_mlp_softdist_beta05 | 5 | 55.78±3.37 |
| coraml | mlp_small_loss | 5 | 55.56±3.42 |
| coraml | mlp_jocor | 5 | 55.28±2.88 |
| coraml | mlp_ct | 5 | 54.63±3.10 |
| coraml | mlp_logitclip | 5 | 53.02±2.98 |
| coraml | mlp_coteaching | 5 | 51.67±4.02 |
| coraml | mlp_ce | 5 | 50.39±2.88 |
| coraml | mlp_dividemix_lite | 5 | 50.39±2.85 |
| coraml | mlp_fine | 5 | 50.32±3.59 |
| coraml | mlp_sce | 5 | 49.92±3.47 |
| coraml | mlp_elr | 5 | 49.40±3.43 |
| coraml | mlp_anl | 5 | 49.35±3.96 |
| coraml | frozengrace_gb_mlp_softdist_beta05_shuffled | 5 | 49.07±4.10 |
