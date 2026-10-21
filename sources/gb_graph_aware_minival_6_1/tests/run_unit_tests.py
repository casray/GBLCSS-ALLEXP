from test_classifier import (
    test_classifier_accepts_sample_weights,
    test_classifier_optionally_returns_logits_without_changing_default_shape,
    test_classifier_uses_clean_eval_masks,
)
from test_coteaching import (
    test_coteaching_is_deterministic_for_same_seed,
    test_coteaching_returns_metrics_and_respects_keep_ratio,
)
from test_class_failure_diagnostics import (
    test_class_failure_script_reuses_build_method_targets,
    test_confusion_matrix_uses_clean_label_rows_and_prediction_columns,
    test_grace_first_grouping_script_reuses_build_method_targets,
    test_node_profile_stats_handles_empty_masks_with_nan_values,
    test_per_class_ncr_and_cdr_are_computed_from_clean_classes,
    test_quality_filter_script_reuses_build_method_targets,
    test_v23_class_safe_script_reuses_build_method_targets,
)
from test_data import test_ratio_split_is_stratified_and_disjoint, test_split_name_distinguishes_ratio_protocol
from test_graph_aware_ball import (
    test_grace_first_graph_balls_use_embedding_candidates_with_topology_gate,
    test_graph_aware_balls_respect_topology_candidates,
    test_graph_aware_global_candidate_scope_can_cross_components,
    test_shuffle_grouping_preserves_group_sizes,
)
from test_maintable import (
    test_evaluate_main_gate_uses_topk_as_primary_method,
    test_paired_stats_marks_small_effect_not_significant,
    test_paired_stats_reports_t_and_sig_label,
)
from test_gated_diagnostics import (
    test_gated_mask_matches_conflict_gate_condition,
    test_lost_by_alpha_detects_qmix_clean_but_target_stays_noisy,
    test_recomputed_ncr_matches_target_argmax_definition,
    test_required_gated_diagnostic_fields_exist,
    test_summary_reports_mean_and_std_for_numeric_fields,
)
from test_ball_coupling import (
    test_losses_are_differentiable_wrt_h,
    test_no_clean_label_in_signatures,
    test_quality_mask_is_boolean_per_ball,
    test_shuffled_membership_raises_intra_loss,
    test_zero_quality_mask_gives_zero_grad_safe_loss,
)
from test_boundary_summary import (
    test_boundary_correlations_report_label_free_r_values,
    test_boundary_summary_rows_compute_random_clean_rate_gap,
    test_evaluate_boundary_claims_returns_c1_to_c4,
)
from test_noise import (
    test_pair_noise_only_changes_train_labels_to_next_class,
    test_symmetric_noise_only_changes_train_labels,
    test_zero_noise_keeps_all_labels,
)
from test_selection import (
    test_ball_consensus_score_is_unit_range_and_clean_label_free,
    test_ball_density_reliability_is_unit_range_and_clean_label_free,
    test_clean_oracle_selection_precision_is_one,
    test_keep_ratio_schedule_is_fixed_warmup_ramp_hold,
    test_loss_only_selection_ignores_reliability,
    test_selection_training_is_deterministic,
)
from test_failure_signal import (
    test_boundary_warning_metrics_include_prefixed_fields,
    test_failure_signal_functions_are_clean_label_free,
    test_neighborhood_warning_metrics_are_scalar_and_unit_range,
    test_pair_like_neighborhood_has_higher_noisy_confidence_than_symmetric_like,
)
from test_run_noisy_classifier import (
    test_gate_experiment_cli_exposes_grace_first_methods,
    test_grace_gagb_methods_route_to_grace_first_groupings,
    test_repair_control_methods_route_to_requested_grouping,
)
from test_smoothing import (
    test_gagb_method_uses_supplied_graph_aware_grouping,
    test_alpha_override_preserves_gagb_alpha_for_shuffled_control,
    test_gagb_qb_conflict_gate_only_smooths_conflicting_train_nodes,
    test_group_quality_scores_rank_compact_supported_low_entropy_ball_higher,
    test_quality_filter_keeps_high_quality_gated_nodes_and_rejects_low_quality_ball,
    test_quality_filter_empty_eligible_balls_does_not_smooth,
    test_quality_weight_scales_fixed_alpha_by_group_quality,
    test_conflict_gate_alpha07_can_flip_target_argmax_when_qmix_is_strong,
    test_conflict_gate_alpha07_uses_fixed_gate_alpha_not_geometry_scaled_alpha,
    test_conflict_gate_alpha08_uses_fixed_alpha,
    test_class_capped_alpha09_limits_each_qmix_target_class,
    test_class_balanced_alpha09_uses_noisy_train_prior_budget,
    test_boundary_safe_alpha09_uses_teacher_margins_and_confidence,
    test_qsafe_boundary_filters_low_quality_candidates_without_fallback,
    test_qsafe_class_capped_only_budgets_high_quality_candidates,
    test_replay_random_same_count_preserves_source_count_and_seed_determinism,
    test_conflict_gate_margin_replace_only_replaces_when_margin_is_large_enough,
    test_conflict_gate_replace_qmix_uses_qmix_as_target_for_gated_nodes,
    test_repair_controls_accept_shuffled_and_khop_groupings,
    test_gagb_qb_pred_mix_uses_noisy_labels_and_teacher_group_mean,
    test_gagb_qb_teacher_prior_gamma_zero_matches_noisy_distribution,
    test_target_building_and_teacher_training_do_not_accept_clean_labels,
    test_kmeans_gagb_matched_uses_supplied_group_count,
    test_kmeans_matches_gb_group_count,
    test_noisy_teacher_returns_probabilities_without_clean_label_inputs,
    test_shuffled_alpha_matched_reports_target_level_sanity,
    test_small_ball_fallback_disables_smoothing,
    test_topk_gagb_train_alias_and_allnode_match_rules,
    test_topk_gagb_matched_uses_gagb_train_group_size_rule,
    test_topk_matches_non_singleton_average_minus_one,
    test_topk_repair_uses_train_matched_topk_rule,
)
from test_summary import test_summary_excludes_old_smoke_rows_by_default
from test_summary import test_summary_keeps_split_protocols_separate
from test_summary import test_summary_includes_gagb_fair_controls_and_pairings
from test_summary import test_summary_includes_grace_first_gagb_methods_and_pairings
from test_summary import test_summary_includes_quality_filter_methods_and_fields
from test_summary import test_summary_includes_target_repair_methods_and_alpha03_pairings
from test_summary import test_summary_includes_v2_target_methods_and_pairings
from test_summary import test_summary_includes_weighted_methods_and_paired_controls
from test_summary import test_summary_pairs_v21_repair_controls_against_alpha03
from test_weighting import (
    test_gagb_weighted_ce_uses_supplied_grouping,
    test_kmeans_weighted_ce_matches_gb_group_count,
    test_leave_one_out_support_lowers_local_label_conflict,
    test_reliability_weights_have_train_mean_near_one,
    test_topk_matched_weighted_ce_uses_matched_group_size,
)


def main():
    tests = [
        test_symmetric_noise_only_changes_train_labels,
        test_zero_noise_keeps_all_labels,
        test_pair_noise_only_changes_train_labels_to_next_class,
        test_ratio_split_is_stratified_and_disjoint,
        test_split_name_distinguishes_ratio_protocol,
        test_small_ball_fallback_disables_smoothing,
        test_kmeans_matches_gb_group_count,
        test_topk_matches_non_singleton_average_minus_one,
        test_graph_aware_balls_respect_topology_candidates,
        test_graph_aware_global_candidate_scope_can_cross_components,
        test_grace_first_graph_balls_use_embedding_candidates_with_topology_gate,
        test_shuffle_grouping_preserves_group_sizes,
        test_gated_mask_matches_conflict_gate_condition,
        test_recomputed_ncr_matches_target_argmax_definition,
        test_lost_by_alpha_detects_qmix_clean_but_target_stays_noisy,
        test_required_gated_diagnostic_fields_exist,
        test_summary_reports_mean_and_std_for_numeric_fields,
        test_repair_control_methods_route_to_requested_grouping,
        test_grace_gagb_methods_route_to_grace_first_groupings,
        test_gate_experiment_cli_exposes_grace_first_methods,
        test_gagb_method_uses_supplied_graph_aware_grouping,
        test_kmeans_gagb_matched_uses_supplied_group_count,
        test_topk_gagb_matched_uses_gagb_train_group_size_rule,
        test_alpha_override_preserves_gagb_alpha_for_shuffled_control,
        test_noisy_teacher_returns_probabilities_without_clean_label_inputs,
        test_gagb_qb_pred_mix_uses_noisy_labels_and_teacher_group_mean,
        test_gagb_qb_teacher_prior_gamma_zero_matches_noisy_distribution,
        test_gagb_qb_conflict_gate_only_smooths_conflicting_train_nodes,
        test_group_quality_scores_rank_compact_supported_low_entropy_ball_higher,
        test_quality_filter_keeps_high_quality_gated_nodes_and_rejects_low_quality_ball,
        test_quality_filter_empty_eligible_balls_does_not_smooth,
        test_quality_weight_scales_fixed_alpha_by_group_quality,
        test_conflict_gate_alpha07_can_flip_target_argmax_when_qmix_is_strong,
        test_conflict_gate_alpha07_uses_fixed_gate_alpha_not_geometry_scaled_alpha,
        test_conflict_gate_alpha08_uses_fixed_alpha,
        test_class_capped_alpha09_limits_each_qmix_target_class,
        test_class_balanced_alpha09_uses_noisy_train_prior_budget,
        test_boundary_safe_alpha09_uses_teacher_margins_and_confidence,
        test_qsafe_boundary_filters_low_quality_candidates_without_fallback,
        test_qsafe_class_capped_only_budgets_high_quality_candidates,
        test_replay_random_same_count_preserves_source_count_and_seed_determinism,
        test_conflict_gate_replace_qmix_uses_qmix_as_target_for_gated_nodes,
        test_repair_controls_accept_shuffled_and_khop_groupings,
        test_conflict_gate_margin_replace_only_replaces_when_margin_is_large_enough,
        test_target_building_and_teacher_training_do_not_accept_clean_labels,
        test_topk_gagb_train_alias_and_allnode_match_rules,
        test_topk_repair_uses_train_matched_topk_rule,
        test_shuffled_alpha_matched_reports_target_level_sanity,
        test_per_class_ncr_and_cdr_are_computed_from_clean_classes,
        test_confusion_matrix_uses_clean_label_rows_and_prediction_columns,
        test_node_profile_stats_handles_empty_masks_with_nan_values,
        test_class_failure_script_reuses_build_method_targets,
        test_v23_class_safe_script_reuses_build_method_targets,
        test_grace_first_grouping_script_reuses_build_method_targets,
        test_quality_filter_script_reuses_build_method_targets,
        test_classifier_uses_clean_eval_masks,
        test_classifier_accepts_sample_weights,
        test_classifier_optionally_returns_logits_without_changing_default_shape,
        test_reliability_weights_have_train_mean_near_one,
        test_leave_one_out_support_lowers_local_label_conflict,
        test_gagb_weighted_ce_uses_supplied_grouping,
        test_kmeans_weighted_ce_matches_gb_group_count,
        test_topk_matched_weighted_ce_uses_matched_group_size,
        test_summary_excludes_old_smoke_rows_by_default,
        test_summary_keeps_split_protocols_separate,
        test_summary_includes_gagb_fair_controls_and_pairings,
        test_summary_includes_grace_first_gagb_methods_and_pairings,
        test_summary_includes_quality_filter_methods_and_fields,
        test_summary_includes_v2_target_methods_and_pairings,
        test_summary_includes_target_repair_methods_and_alpha03_pairings,
        test_summary_pairs_v21_repair_controls_against_alpha03,
        test_summary_includes_weighted_methods_and_paired_controls,
        test_keep_ratio_schedule_is_fixed_warmup_ramp_hold,
        test_ball_consensus_score_is_unit_range_and_clean_label_free,
        test_ball_density_reliability_is_unit_range_and_clean_label_free,
        test_clean_oracle_selection_precision_is_one,
        test_loss_only_selection_ignores_reliability,
        test_selection_training_is_deterministic,
        test_coteaching_returns_metrics_and_respects_keep_ratio,
        test_coteaching_is_deterministic_for_same_seed,
        test_paired_stats_reports_t_and_sig_label,
        test_paired_stats_marks_small_effect_not_significant,
        test_evaluate_main_gate_uses_topk_as_primary_method,
        test_failure_signal_functions_are_clean_label_free,
        test_neighborhood_warning_metrics_are_scalar_and_unit_range,
        test_pair_like_neighborhood_has_higher_noisy_confidence_than_symmetric_like,
        test_boundary_warning_metrics_include_prefixed_fields,
        test_boundary_summary_rows_compute_random_clean_rate_gap,
        test_boundary_correlations_report_label_free_r_values,
        test_evaluate_boundary_claims_returns_c1_to_c4,
        test_losses_are_differentiable_wrt_h,
        test_no_clean_label_in_signatures,
        test_zero_quality_mask_gives_zero_grad_safe_loss,
        test_shuffled_membership_raises_intra_loss,
        test_quality_mask_is_boolean_per_ball,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
