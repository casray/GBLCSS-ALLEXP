import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_maintable import evaluate_main_gate, paired_stats


def test_paired_stats_reports_t_and_sig_label():
    stats = paired_stats([0.71, 0.72, 0.73, 0.74, 0.75], [0.70, 0.70, 0.70, 0.70, 0.70])
    assert round(stats["mean_delta"], 4) == 0.03
    assert stats["wins"] == 5
    assert stats["t_value"] > 2.78
    assert stats["sig_label"] == "SIG"


def test_paired_stats_marks_small_effect_not_significant():
    stats = paired_stats([0.71, 0.69, 0.71, 0.69, 0.70], [0.70, 0.70, 0.70, 0.70, 0.70])
    assert stats["wins"] == 2
    assert stats["sig_label"] == "ns"


def test_evaluate_main_gate_uses_topk_as_primary_method():
    per_seed = {
        0: {"sel_topk": 0.72, "coteaching": 0.70, "sel_noball": 0.69, "sel_shuffled": 0.68, "sel_dcgb": 0.71},
        1: {"sel_topk": 0.73, "coteaching": 0.70, "sel_noball": 0.69, "sel_shuffled": 0.68, "sel_dcgb": 0.71},
        2: {"sel_topk": 0.74, "coteaching": 0.70, "sel_noball": 0.69, "sel_shuffled": 0.68, "sel_dcgb": 0.71},
        3: {"sel_topk": 0.75, "coteaching": 0.70, "sel_noball": 0.69, "sel_shuffled": 0.68, "sel_dcgb": 0.71},
        4: {"sel_topk": 0.76, "coteaching": 0.70, "sel_noball": 0.69, "sel_shuffled": 0.68, "sel_dcgb": 0.71},
    }
    gate = evaluate_main_gate(per_seed, [0, 1, 2, 3, 4])
    assert gate["gate_1_topk_beats_coteaching"]["passes"] is True
    assert gate["gate_2_topk_beats_controls"]["passes"] is True
    assert gate["dcgb_minus_topk"]["mean_delta"] < 0.0
