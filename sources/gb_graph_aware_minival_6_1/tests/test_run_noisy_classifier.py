import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate_experiment import DEFAULT_METHODS as GATE_DEFAULT_METHODS
from run_gate_experiment import METHOD_CHOICES as GATE_METHOD_CHOICES
from run_noisy_classifier import _method_grouping


def _route(method: str, groupings: dict) -> dict:
    return _method_grouping(
        method,
        groupings["gb"],
        groupings["gagb"],
        groupings["grace_gagb"],
        groupings["grace_gagb_emb"],
        groupings["grace_gagb_topo"],
        groupings["emb_local"],
        groupings["emb_global"],
        groupings["khop"],
        groupings["shuffled_size"],
        groupings["shuffled_alpha"],
        groupings["grace_gagb_shuffled"],
        groupings.get("dcgb", {"name": "dcgb"}),
        groupings.get("dcgb_emb", {"name": "dcgb_emb"}),
        groupings.get("dcgb_topo", {"name": "dcgb_topo"}),
        groupings.get("dcgb_shuffled", {"name": "dcgb_shuffled"}),
    )


def test_repair_control_methods_route_to_requested_grouping():
    groupings = {
        "gb": {"name": "gb"},
        "gagb": {"name": "gagb"},
        "grace_gagb": {"name": "grace_gagb"},
        "grace_gagb_emb": {"name": "grace_gagb_emb"},
        "grace_gagb_topo": {"name": "grace_gagb_topo"},
        "emb_local": {"name": "emb_local"},
        "emb_global": {"name": "emb_global"},
        "khop": {"name": "khop"},
        "shuffled_size": {"name": "shuffled_size"},
        "shuffled_alpha": {"name": "shuffled_alpha"},
        "grace_gagb_shuffled": {"name": "grace_gagb_shuffled"},
    }

    assert _route("gagb_shuffled_alpha09_matched", groupings) is groupings["shuffled_alpha"]
    assert _route("khop_only_alpha09", groupings) is groupings["khop"]


def test_grace_gagb_methods_route_to_grace_first_groupings():
    groupings = {
        "gb": {"name": "gb"},
        "gagb": {"name": "gagb"},
        "grace_gagb": {"name": "grace_gagb"},
        "grace_gagb_emb": {"name": "grace_gagb_emb"},
        "grace_gagb_topo": {"name": "grace_gagb_topo"},
        "emb_local": {"name": "emb_local"},
        "emb_global": {"name": "emb_global"},
        "khop": {"name": "khop"},
        "shuffled_size": {"name": "shuffled_size"},
        "shuffled_alpha": {"name": "shuffled_alpha"},
        "grace_gagb_shuffled": {"name": "grace_gagb_shuffled"},
    }

    assert _route("grace_gagb", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qb_conflict_gate_alpha09", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qfilter_alpha07_q50", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qweight_alpha07", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qsafe_boundary_alpha09_q50", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qsafe_class_capped_alpha09_q50", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_qsafe_class_balanced_alpha09_q50", groupings) is groupings["grace_gagb"]
    assert _route("gagb_qsafe_boundary_alpha09_q50", groupings) is groupings["gagb"]
    assert _route("khop_only_qsafe_boundary_alpha09_q50", groupings) is groupings["khop"]
    assert _route("gagb_qfilter_alpha07_q50", groupings) is groupings["gagb"]
    assert _route("khop_only_qfilter_alpha07_q50", groupings) is groupings["khop"]
    assert _route("kmeans_grace_gagb_matched", groupings) is groupings["grace_gagb"]
    assert _route("topk_grace_gagb_train_matched", groupings) is groupings["grace_gagb"]
    assert _route("grace_gagb_emb_only", groupings) is groupings["grace_gagb_emb"]
    assert _route("grace_gagb_topo_only", groupings) is groupings["grace_gagb_topo"]
    assert _route("grace_gagb_shuffled_alpha_matched", groupings) is groupings["grace_gagb_shuffled"]
    assert _route("grace_gagb_shuffled_qfilter_alpha07_q50", groupings) is groupings["grace_gagb_shuffled"]
    assert _route("grace_gagb_shuffled_qsafe_boundary_alpha09_q50", groupings) is groupings["grace_gagb_shuffled"]


def test_dcgb_methods_route_to_dcgb_groupings():
    groupings = {
        "gb": {"name": "gb"},
        "gagb": {"name": "gagb"},
        "grace_gagb": {"name": "grace_gagb"},
        "grace_gagb_emb": {"name": "grace_gagb_emb"},
        "grace_gagb_topo": {"name": "grace_gagb_topo"},
        "emb_local": {"name": "emb_local"},
        "emb_global": {"name": "emb_global"},
        "khop": {"name": "khop"},
        "shuffled_size": {"name": "shuffled_size"},
        "shuffled_alpha": {"name": "shuffled_alpha"},
        "grace_gagb_shuffled": {"name": "grace_gagb_shuffled"},
        "dcgb": {"name": "dcgb"},
        "dcgb_emb": {"name": "dcgb_emb"},
        "dcgb_topo": {"name": "dcgb_topo"},
        "dcgb_shuffled": {"name": "dcgb_shuffled"},
    }

    assert _route("dcgb", groupings) is groupings["dcgb"]
    assert _route("dcgb_qsafe_boundary_alpha09_q50", groupings) is groupings["dcgb"]
    assert _route("kmeans_dcgb_matched", groupings) is groupings["dcgb"]
    assert _route("topk_dcgb_train_matched", groupings) is groupings["dcgb"]
    assert _route("dcgb_emb_only", groupings) is groupings["dcgb_emb"]
    assert _route("dcgb_topo_only", groupings) is groupings["dcgb_topo"]
    assert _route("dcgb_shuffled_alpha_matched", groupings) is groupings["dcgb_shuffled"]
    # DCGB names must not steal grace_gagb / gagb routing.
    assert _route("grace_gagb", groupings) is groupings["grace_gagb"]
    assert _route("gagb", groupings) is groupings["gagb"]


def test_gate_experiment_cli_exposes_grace_first_methods():
    expected = {
        "grace_gagb",
        "grace_gagb_emb_only",
        "grace_gagb_topo_only",
        "kmeans_grace_gagb_matched",
        "topk_grace_gagb_train_matched",
        "grace_gagb_shuffled_alpha_matched",
        "grace_gagb_qb_conflict_gate_alpha03",
        "grace_gagb_qb_conflict_gate_alpha07",
        "grace_gagb_qb_conflict_gate_alpha09",
        "grace_gagb_qfilter_alpha07_q50",
        "grace_gagb_qfilter_alpha07_q70",
        "grace_gagb_qweight_alpha07",
        "grace_gagb_qsafe_boundary_alpha09_q50",
        "grace_gagb_qsafe_boundary_alpha09_q70",
        "grace_gagb_qsafe_boundary_alpha07_q50",
        "grace_gagb_qsafe_class_capped_alpha09_q50",
        "grace_gagb_qsafe_class_balanced_alpha09_q50",
        "gagb_qsafe_boundary_alpha09_q50",
        "khop_only_qsafe_boundary_alpha09_q50",
        "grace_gagb_shuffled_qsafe_boundary_alpha09_q50",
        "gagb_qfilter_alpha07_q50",
        "khop_only_qfilter_alpha07_q50",
        "grace_gagb_shuffled_qfilter_alpha07_q50",
    }

    assert expected <= set(GATE_DEFAULT_METHODS)
    assert expected <= set(GATE_METHOD_CHOICES)
