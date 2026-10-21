from test_classifier import test_classifier_accepts_sample_weights, test_classifier_uses_clean_eval_masks
from test_data import test_ratio_split_is_stratified_and_disjoint, test_split_name_distinguishes_ratio_protocol
from test_noise import test_symmetric_noise_only_changes_train_labels, test_zero_noise_keeps_all_labels
from test_smoothing import (
    test_kmeans_matches_gb_group_count,
    test_small_ball_fallback_disables_smoothing,
    test_topk_matches_non_singleton_average_minus_one,
)
from test_summary import test_summary_excludes_old_smoke_rows_by_default
from test_summary import test_summary_keeps_split_protocols_separate
from test_summary import test_summary_includes_weighted_methods_and_paired_controls
from test_weighting import (
    test_kmeans_weighted_ce_matches_gb_group_count,
    test_leave_one_out_support_lowers_local_label_conflict,
    test_reliability_weights_have_train_mean_near_one,
    test_topk_matched_weighted_ce_uses_matched_group_size,
)


def main():
    tests = [
        test_symmetric_noise_only_changes_train_labels,
        test_zero_noise_keeps_all_labels,
        test_ratio_split_is_stratified_and_disjoint,
        test_split_name_distinguishes_ratio_protocol,
        test_small_ball_fallback_disables_smoothing,
        test_kmeans_matches_gb_group_count,
        test_topk_matches_non_singleton_average_minus_one,
        test_classifier_uses_clean_eval_masks,
        test_classifier_accepts_sample_weights,
        test_reliability_weights_have_train_mean_near_one,
        test_leave_one_out_support_lowers_local_label_conflict,
        test_kmeans_weighted_ce_matches_gb_group_count,
        test_topk_matched_weighted_ce_uses_matched_group_size,
        test_summary_excludes_old_smoke_rows_by_default,
        test_summary_keeps_split_protocols_separate,
        test_summary_includes_weighted_methods_and_paired_controls,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
