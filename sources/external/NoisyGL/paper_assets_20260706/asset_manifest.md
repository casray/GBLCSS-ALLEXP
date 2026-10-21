# MLP-17 paper asset manifest

All assets are generated from existing CSV/report artifacts. No new training was run.

| artifact | sources | check | note |
| --- | --- | --- | --- |
| tables\appendix_A_complete_main_table.md | mlp10_aggregate.csv + mlp12_aggregate.csv + p0b | pass |  |
| tables\appendix_A_complete_main_table.tex | mlp10_aggregate.csv + mlp12_aggregate.csv + p0b | pass |  |
| tables\appendix_B_paired_significance.md | p0b_paired_significance.csv | pass |  |
| tables\appendix_B_paired_significance.tex | p0b_paired_significance.csv | pass |  |
| tables\appendix_C_graph_dependent_references.md | phase24/tfr/coauthorcs/2025 comparison CSV scan | pass hash-recorded |  |
| tables\appendix_C_graph_dependent_references.tex | phase24/tfr/coauthorcs/2025 comparison CSV scan | pass hash-recorded |  |
| tables\appendix_D1_parameter_sensitivity.md | mlp13_stageA_grid_summary.csv + mlp13_stageB_summary.csv | pass |  |
| tables\appendix_D1_parameter_sensitivity.tex | mlp13_stageA_grid_summary.csv + mlp13_stageB_summary.csv | pass |  |
| tables\appendix_D2_beta_audit.md | p0a_beta_validation_audit.csv | pass |  |
| tables\appendix_D2_beta_audit.tex | p0a_beta_validation_audit.csv | pass |  |
| figures\appendix_E_feature_noise_curves.pdf | mlp16_feature_noise_aggregate.csv | pass |  |
| figures\appendix_E_feature_noise_curves.png | mlp16_feature_noise_aggregate.csv | pass |  |
| tables\appendix_F_cut_value.md | mlp16_cut_value_aggregate.csv + raw | pass |  |
| tables\appendix_F_cut_value.tex | mlp16_cut_value_aggregate.csv + raw | pass |  |
| figures\appendix_G_tsne_three_coloring.pdf | GRACE H + seed0 uniform0.4 labels + recomputed balls | pass illustrative |  |
| figures\appendix_G_tsne_three_coloring.png | GRACE H + seed0 uniform0.4 labels + recomputed balls | pass illustrative |  |
| figures\appendix_G2_grace_ball_cover.pdf | GRACE H + PCA projection + recomputed default balls | pass illustrative | five-dataset 80% projected-radius ball contour |
| figures\appendix_G2_grace_ball_cover.png | GRACE H + PCA projection + recomputed default balls | pass illustrative | five-dataset 80% projected-radius ball contour |
| figures\appendix_G3_noise_soft_correction.pdf | mechanism schematic derived from LCSS target formula | pass illustrative | shows noisy one-hot softened toward local consensus; no new training |
| figures\appendix_G3_noise_soft_correction.png | mechanism schematic derived from LCSS target formula | pass illustrative | shows noisy one-hot softened toward local consensus; no new training |
| figures\appendix_G3_noise_soft_correction.svg | mechanism schematic derived from LCSS target formula | pass illustrative | shows noisy one-hot softened toward local consensus; no new training |
| figures\appendix_H_labelrate_curves.pdf | mlp6_aggregate.csv | pass; H2 strict monotonic decrease toward high-label end: cora=True, citeseer=False |  |
| figures\appendix_H_labelrate_curves.png | mlp6_aggregate.csv | pass; H2 strict monotonic decrease toward high-label end: cora=True, citeseer=False |  |
| tables/appendix_H_labelrate_monotonicity.md | mlp6_aggregate.csv | pass | H2 strict monotonic decrease toward high-label end: cora=True, citeseer=False |
| tables\table_B_graphfree_family.md | mlp14_aggregate.csv | pass T3/T4-separated |  |
| tables\table_B_graphfree_family.tex | mlp14_aggregate.csv | pass T3/T4-separated |  |
| tables\extension_denoised_teacher_distill.md | mlp9_aggregate.csv + decomposition | pass T3/T4-present |  |
| tables\extension_denoised_teacher_distill.tex | mlp9_aggregate.csv + decomposition | pass T3/T4-present |  |
