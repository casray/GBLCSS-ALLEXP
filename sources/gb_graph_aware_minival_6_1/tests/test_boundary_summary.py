import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.summarize_boundary import (
    boundary_summary_rows,
    compute_boundary_correlations,
    evaluate_boundary_claims,
)


def test_boundary_summary_rows_compute_random_clean_rate_gap():
    rows = [
        {"dataset": "Cora", "noise_type": "symmetric", "noise_rate": 0.4, "method": "sel_topk", "seed": 0, "selection_precision": 0.9, "test_acc_at_best_val": 0.8},
        {"dataset": "Cora", "noise_type": "symmetric", "noise_rate": 0.4, "method": "ce", "seed": 0, "test_acc_at_best_val": 0.7},
        {"dataset": "Cora", "noise_type": "symmetric", "noise_rate": 0.4, "method": "coteaching", "seed": 0, "test_acc_at_best_val": 0.6},
    ]
    summary = boundary_summary_rows(rows)
    assert len(summary) == 1
    assert round(summary[0]["selection_precision_minus_random_clean_rate"], 4) == 0.3
    assert round(summary[0]["sel_topk_minus_ce"], 4) == 0.1


def test_boundary_correlations_report_label_free_r_values():
    summary = [
        {"warning": 0.1, "selection_precision_minus_random_clean_rate": 0.8, "sel_topk_minus_ce": 0.1},
        {"warning": 0.2, "selection_precision_minus_random_clean_rate": 0.6, "sel_topk_minus_ce": 0.0},
        {"warning": 0.3, "selection_precision_minus_random_clean_rate": 0.4, "sel_topk_minus_ce": -0.1},
    ]
    corr = compute_boundary_correlations(summary, warning_fields=["warning"])
    assert corr["warning_vs_selection_gap_pearson"] < -0.99
    assert corr["warning_vs_test_delta_pearson"] < -0.99


def test_evaluate_boundary_claims_returns_c1_to_c4():
    setting_rows = [
        {"dataset": "Cora", "noise_type": "symmetric", "noise_rate": 0.4, "topk_minus_coteaching_sig": "SIG", "topk_minus_coteaching_pp": 9.0, "topk_minus_coteaching_wins": 5, "selection_precision_minus_random_clean_rate": 0.3, "dcgb_minus_topk_pp": -1.0, "dcgb_minus_topk_sig": "ns"},
        {"dataset": "CiteSeer", "noise_type": "symmetric", "noise_rate": 0.4, "topk_minus_coteaching_sig": "SIG", "topk_minus_coteaching_pp": 1.2, "topk_minus_coteaching_wins": 5, "selection_precision_minus_random_clean_rate": 0.2, "dcgb_minus_topk_pp": -1.0, "dcgb_minus_topk_sig": "ns"},
        {"dataset": "Cora", "noise_type": "pair", "noise_rate": 0.2, "topk_minus_coteaching_sig": "ns", "topk_minus_coteaching_pp": 0.0, "topk_minus_coteaching_wins": 2, "selection_precision_minus_random_clean_rate": 0.2, "dcgb_minus_topk_pp": -1.0, "dcgb_minus_topk_sig": "ns"},
        {"dataset": "Cora", "noise_type": "pair", "noise_rate": 0.4, "topk_minus_coteaching_sig": "ns", "topk_minus_coteaching_pp": 0.0, "topk_minus_coteaching_wins": 2, "selection_precision_minus_random_clean_rate": 0.1, "dcgb_minus_topk_pp": -1.0, "dcgb_minus_topk_sig": "ns"},
        {"dataset": "Cora", "noise_type": "pair", "noise_rate": 0.6, "topk_minus_coteaching_sig": "ns", "topk_minus_coteaching_pp": 0.0, "topk_minus_coteaching_wins": 2, "selection_precision_minus_random_clean_rate": -0.1, "dcgb_minus_topk_pp": -1.0, "dcgb_minus_topk_sig": "ns"},
    ]
    corr = {"topk_neighborhood_majority_confidence_vs_selection_gap_pearson": -0.7}
    claims = evaluate_boundary_claims(setting_rows, corr)
    assert claims["C1_symmetric_positive"]["passes"] is True
    assert claims["C2_pair_boundary"]["passes"] is True
    assert claims["C3_label_free_warning"]["passes"] is True
    assert claims["C4_ball_no_increment"]["passes"] is True
