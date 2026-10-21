# Local-Consistency Validity Boundary Report

## Setting Summary

| dataset | noise | sel_topk | CE | co-teaching | selection gap | noise recall | topk-CE | topk-coteach | C1/C2 note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cora | symmetric0.2 | 0.8117 | 0.8113 | 0.7773 | 0.1608 | 0.8426 | 0.04pp | 3.44pp | |
| Cora | symmetric0.4 | 0.7969 | 0.7935 | 0.7064 | 0.3225 | 0.8839 | 0.35pp | 9.05pp | |
| Cora | symmetric0.6 | 0.6725 | 0.7136 | 0.6646 | 0.0674 | 0.1729 | -4.11pp | 0.79pp | |
| Cora | pair0.2 | 0.7824 | 0.7696 | 0.7414 | 0.1138 | 0.6537 | 1.28pp | 4.10pp | |
| Cora | pair0.4 | 0.6082 | 0.5920 | 0.5225 | 0.0972 | 0.5465 | 1.63pp | 8.57pp | |
| Cora | pair0.6 | 0.2721 | 0.2506 | 0.2448 | -0.1172 | 0.4012 | 2.15pp | 2.73pp | |
| CiteSeer | symmetric0.2 | 0.7106 | 0.7064 | 0.7086 | 0.1085 | 0.6241 | 0.42pp | 0.20pp | |
| CiteSeer | symmetric0.4 | 0.7082 | 0.6979 | 0.6958 | 0.2301 | 0.7233 | 1.03pp | 1.24pp | |
| CiteSeer | symmetric0.6 | 0.6743 | 0.6685 | 0.6668 | 0.1223 | 0.3220 | 0.58pp | 0.75pp | |
| CiteSeer | pair0.2 | 0.7009 | 0.6906 | 0.6970 | 0.0807 | 0.5218 | 1.03pp | 0.40pp | |
| CiteSeer | pair0.4 | 0.6293 | 0.5956 | 0.6521 | 0.1185 | 0.5767 | 3.37pp | -2.29pp | |
| CiteSeer | pair0.6 | 0.3088 | 0.3114 | 0.3284 | -0.0006 | 0.0000 | -0.27pp | -1.97pp | |

## Label-Free Correlations

- `topk_neighborhood_majority_self_disagreement_vs_selection_gap_pearson`: -0.0247
- `topk_neighborhood_majority_self_disagreement_vs_selection_gap_spearman`: 0.0070
- `topk_neighborhood_majority_self_disagreement_vs_test_delta_pearson`: -0.3521
- `topk_neighborhood_majority_self_disagreement_vs_test_delta_spearman`: -0.1678
- `topk_neighborhood_majority_confidence_vs_selection_gap_pearson`: -0.1630
- `topk_neighborhood_majority_confidence_vs_selection_gap_spearman`: -0.1888
- `topk_neighborhood_majority_confidence_vs_test_delta_pearson`: 0.4757
- `topk_neighborhood_majority_confidence_vs_test_delta_spearman`: 0.2657
- `topk_pairflip_neighborhood_consistency_vs_selection_gap_pearson`: -0.1705
- `topk_pairflip_neighborhood_consistency_vs_selection_gap_spearman`: -0.1678
- `topk_pairflip_neighborhood_consistency_vs_test_delta_pearson`: 0.4516
- `topk_pairflip_neighborhood_consistency_vs_test_delta_spearman`: 0.2937

## Pre-Registered Claims

- `C1_symmetric_positive`: **PASS** `{"passes": true, "checked_settings": 2}`
- `C2_pair_boundary`: **PASS** `{"passes": true, "datasets_passing": ["Cora"]}`
- `C3_label_free_warning`: **FAIL** `{"passes": false, "max_abs_pearson_r": 0.47573845537396925}`
- `C4_ball_no_increment`: **PASS** `{"passes": true, "checked_settings": 12}`
