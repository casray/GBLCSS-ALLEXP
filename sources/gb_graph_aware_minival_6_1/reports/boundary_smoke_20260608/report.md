# Local-Consistency Validity Boundary Report

## Setting Summary

| dataset | noise | sel_topk | CE | co-teaching | selection gap | noise recall | topk-CE | topk-coteach | C1/C2 note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | pair0.4 | 0.4324 | 0.5396 | 0.4456 | 0.0209 | 0.1475 | -10.72pp | -1.32pp | |

## Label-Free Correlations

- `topk_neighborhood_majority_self_disagreement_vs_selection_gap_pearson`: nan
- `topk_neighborhood_majority_self_disagreement_vs_selection_gap_spearman`: nan
- `topk_neighborhood_majority_self_disagreement_vs_test_delta_pearson`: nan
- `topk_neighborhood_majority_self_disagreement_vs_test_delta_spearman`: nan
- `topk_neighborhood_majority_confidence_vs_selection_gap_pearson`: nan
- `topk_neighborhood_majority_confidence_vs_selection_gap_spearman`: nan
- `topk_neighborhood_majority_confidence_vs_test_delta_pearson`: nan
- `topk_neighborhood_majority_confidence_vs_test_delta_spearman`: nan
- `topk_pairflip_neighborhood_consistency_vs_selection_gap_pearson`: nan
- `topk_pairflip_neighborhood_consistency_vs_selection_gap_spearman`: nan
- `topk_pairflip_neighborhood_consistency_vs_test_delta_pearson`: nan
- `topk_pairflip_neighborhood_consistency_vs_test_delta_spearman`: nan

## Pre-Registered Claims

- `C1_symmetric_positive`: **FAIL** `{"passes": false, "checked_settings": 0}`
- `C2_pair_boundary`: **FAIL** `{"passes": false, "datasets_passing": []}`
- `C3_label_free_warning`: **FAIL** `{"passes": false, "max_abs_pearson_r": NaN}`
- `C4_ball_no_increment`: **FAIL** `{"passes": false, "checked_settings": 1}`
