# MLP-59 final dataset and teacher extension report

## Protocol

All formal runs use percent 0.2/0.1/0.7 splits, train-only synthetic label noise, seeds 0--4, and exact split/noise-hash pairing. The formal GB-LCSS target uses leave-one-out voting with `tie_mode=distribution_only`.

## Teacher extension

| dataset    | noise_type   |   noise_rate | method                             |   mean |   std |   count |
|:-----------|:-------------|-------------:|:-----------------------------------|-------:|------:|--------:|
| amazonpho  | uniform      |          0.4 | mlp_distill_formal_lcss_x          |  90.4  |  0.77 |       5 |
| amazonpho  | uniform      |          0.4 | mlp_distill_formal_lcss_x_shuffled |  87.87 |  1.61 |       5 |
| amazonpho  | uniform      |          0.4 | mlp_distill_gcn                    |  89.86 |  0.77 |       5 |
| coauthorcs | uniform      |          0.4 | mlp_distill_formal_lcss_x          |  93.83 |  0.21 |       5 |
| coauthorcs | uniform      |          0.4 | mlp_distill_formal_lcss_x_shuffled |  92.39 |  0.51 |       5 |
| coauthorcs | uniform      |          0.4 | mlp_distill_gcn                    |  93.2  |  0.09 |       5 |
| amazonpho  | uniform      |          0.6 | mlp_distill_formal_lcss_x          |  86.81 |  1.94 |       5 |
| amazonpho  | uniform      |          0.6 | mlp_distill_formal_lcss_x_shuffled |  80.23 |  6.24 |       5 |
| amazonpho  | uniform      |          0.6 | mlp_distill_gcn                    |  84.8  |  1.48 |       5 |
| coauthorcs | uniform      |          0.6 | mlp_distill_formal_lcss_x          |  92.4  |  0.65 |       5 |
| coauthorcs | uniform      |          0.6 | mlp_distill_formal_lcss_x_shuffled |  89.74 |  0.32 |       5 |
| coauthorcs | uniform      |          0.6 | mlp_distill_gcn                    |  91.13 |  0.58 |       5 |
| amazonpho  | uniform      |          0.4 | ffg2m_reimplemented                |  84.82 |  0.84 |       5 |
| amazonpho  | uniform      |          0.4 | krd_adapted                        |  85.17 |  0.94 |       5 |
| amazonpho  | uniform      |          0.4 | vqgraph_adapted                    |  90.54 |  0.85 |       5 |
| coauthorcs | uniform      |          0.4 | ffg2m_reimplemented                |  91.58 |  0.24 |       5 |
| coauthorcs | uniform      |          0.4 | krd_adapted                        |  91.39 |  0.49 |       5 |
| coauthorcs | uniform      |          0.4 | vqgraph_adapted                    |  93.25 |  0.09 |       5 |
| amazonpho  | uniform      |          0.6 | ffg2m_reimplemented                |  73.08 |  2.82 |       5 |
| amazonpho  | uniform      |          0.6 | krd_adapted                        |  80.3  |  0.51 |       5 |
| amazonpho  | uniform      |          0.6 | vqgraph_adapted                    |  84.77 |  2.19 |       5 |
| coauthorcs | uniform      |          0.6 | ffg2m_reimplemented                |  86.5  |  1.01 |       5 |
| coauthorcs | uniform      |          0.6 | krd_adapted                        |  89.8  |  0.29 |       5 |
| coauthorcs | uniform      |          0.6 | vqgraph_adapted                    |  90.71 |  1.24 |       5 |

Teacher gate:

| phase   |   base_positive_cells |   gcn_positive_cells |   shuffled_pass_cells | passed   |   dataset |   baseline_method |   delta_vs_baseline_pp |   positive_vs_baseline |   delta_vs_shuffled_pp |   positive_vs_shuffled |   noninferior_settings |   average_delta_pp |   worst_delta_pp |
|:--------|----------------------:|---------------------:|----------------------:|:---------|----------:|------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------:|
| teacher |                     4 |                    4 |                     4 | True     |       nan |               nan |                    nan |                    nan |                    nan |                    nan |                    nan |                nan |              nan |

The formal ball-consensus teacher passed the frozen gate on Coauthor-CS and Amazon-Photo. It exceeded the base GB-LCSS student and the GCN teacher in all four dataset-rate cells, and exceeded the matched shuffled teacher by at least 1 pp with at least four positive seeds in all four cells.

## Boundary uniform-0.4 entry test

| dataset      | noise_type   |   noise_rate | method                                          |   mean |   std |   count |
|:-------------|:-------------|-------------:|:------------------------------------------------|-------:|------:|--------:|
| amazoncom    | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  76.2  |  0.56 |       5 |
| amazoncom    | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  70.49 |  1.08 |       5 |
| attribwiki   | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  65.17 |  1.42 |       5 |
| attribwiki   | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  59.83 |  2.13 |       5 |
| citeseerfull | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  60.94 |  1.08 |       5 |
| citeseerfull | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  46.69 |  2.74 |       5 |
| coauthorph   | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  93.13 |  0.25 |       5 |
| coauthorph   | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  88.8  |  2.12 |       5 |
| coraml       | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  56.26 |  2.61 |       5 |
| coraml       | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  47.81 |  3.34 |       5 |
| pubmed       | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  72.6  |  0.66 |       5 |
| pubmed       | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  63.04 |  4.49 |       5 |
| citeseerfull | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo          |  60.94 |  1.08 |       5 |
| citeseerfull | uniform      |          0.4 | frozengrace_gb_mlp_softdist_beta05_loo_shuffled |  46.69 |  2.74 |       5 |

| phase          |   base_positive_cells |   gcn_positive_cells |   shuffled_pass_cells | passed   | dataset      | baseline_method   |   delta_vs_baseline_pp |   positive_vs_baseline |   delta_vs_shuffled_pp |   positive_vs_shuffled |   noninferior_settings |   average_delta_pp |   worst_delta_pp |
|:---------------|----------------------:|---------------------:|----------------------:|:---------|:-------------|:------------------|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------:|
| boundary_entry |                   nan |                  nan |                   nan | False    | amazoncom    | mlp_jocor         |             -0.604613  |                      1 |                5.71161 |                      5 |                    nan |                nan |              nan |
| boundary_entry |                   nan |                  nan |                   nan | False    | attribwiki   | mlp_logitclip     |             -2.66191   |                      0 |                5.33571 |                      5 |                    nan |                nan |              nan |
| boundary_entry |                   nan |                  nan |                   nan | True     | citeseerfull | mlp_jocor         |              1.45897   |                      4 |               14.2519  |                      5 |                    nan |                nan |              nan |
| boundary_entry |                   nan |                  nan |                   nan | False    | coauthorph   | mlp_jocor         |              0.0422448 |                      3 |                4.33133 |                      5 |                    nan |                nan |              nan |
| boundary_entry |                   nan |                  nan |                   nan | False    | coraml       | mlp_small_loss    |              0.696565  |                      3 |                8.4542  |                      5 |                    nan |                nan |              nan |
| boundary_entry |                   nan |                  nan |                   nan | False    | pubmed       | mlp_jocor         |             -5.23585   |                      0 |                9.56308 |                      5 |                    nan |                nan |              nan |

Only CiteSeerFull passed the frozen entry gate. No other boundary dataset was advanced to the full-spectrum stage.

## CiteSeerFull full-spectrum decision

| dataset      | noise_type   |   noise_rate | baseline_source_method   |   baseline_mean |
|:-------------|:-------------|-------------:|:-------------------------|----------------:|
| citeseerfull | instance     |          0.4 | mlp_jocor                |        0.599189 |
| citeseerfull | pair         |          0.4 | mlp_jocor                |        0.511246 |
| citeseerfull | uniform      |          0.2 | mlp_small_loss           |        0.674704 |
| citeseerfull | uniform      |          0.4 | mlp_jocor                |        0.594799 |
| citeseerfull | uniform      |          0.6 | mlp_jocor                |        0.466464 |

| phase                  |   base_positive_cells |   gcn_positive_cells |   shuffled_pass_cells | passed   | dataset      |   baseline_method |   delta_vs_baseline_pp |   positive_vs_baseline |   delta_vs_shuffled_pp |   positive_vs_shuffled |   noninferior_settings |   average_delta_pp |   worst_delta_pp |
|:-----------------------|----------------------:|---------------------:|----------------------:|:---------|:-------------|------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------:|
| boundary_full_spectrum |                   nan |                  nan |                   nan | False    | citeseerfull |               nan |                    nan |                    nan |                    nan |                    nan |                      4 |            1.19284 |         -4.20128 |

CiteSeerFull was non-inferior in four of five settings and had a positive mean cross-setting delta, but failed the pre-registered worst-cell constraint because uniform 0.6 was more than 1 pp below the strongest eligible baseline. It is therefore not added to the formal five-dataset main matrix.

## Decision

- Extend the teacher-distillation table to Coauthor-CS and Amazon-Photo.
- Keep the existing five-dataset main matrix unchanged.
- Report the six final-target boundary diagnostics and the CiteSeerFull high-noise failure without promoting any boundary dataset to the main matrix.
