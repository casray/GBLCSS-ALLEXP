from __future__ import annotations

import pytest
from pathlib import Path


def test_table4_g2m_methods_declare_xonly_resource_identity():
    from scripts.run_standard_downstream import METHOD_CAPABILITIES

    expected_status = {
        "ffg2m_reimplemented": "common-backbone paper reimplementation",
        "krd_adapted": "mechanism-faithful reliability adaptation",
        "vqgraph_adapted": "soft-token mechanism adaptation",
    }
    for method, status in expected_status.items():
        capability = METHOD_CAPABILITIES[method]
        assert capability["requires_embedding"] is False
        assert capability["training_uses_graph"] is True
        assert capability["inference_uses_graph"] is False
        assert capability["uses_unlabeled_nodes"] is True
        assert capability["comparison_tier"] == "graph_teacher_x_only_student"
        assert capability["implementation_status"] == status


def _new_row(dataset, method, noise_rate, seed, test=0.5):
    protocol = f"{dataset}-{noise_rate}-{seed}"
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "status": "ok",
        "test": test,
        "valid": test - 0.01,
        "teacher_test": test + 0.01,
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{protocol}",
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "source": "synthetic.csv",
    }


def _complete_new_rows():
    from scripts.summarize_mlp55_g2m_table4_20260812 import (
        DATASETS,
        NEW_METHODS,
        RATES,
        SEEDS,
    )

    return [
        _new_row(dataset, method, rate, seed, test=0.5 + 0.01 * seed)
        for dataset in DATASETS
        for rate in RATES
        for method in NEW_METHODS
        for seed in SEEDS
    ]


def test_validate_new_rows_requires_complete_paired_xonly_grid():
    from scripts.summarize_mlp55_g2m_table4_20260812 import validate_new_rows

    rows = _complete_new_rows()
    audit = validate_new_rows(rows)

    assert len(rows) == 90
    assert len(audit) == 30
    assert all(row["status"] == "PASS" for row in audit)


def test_validate_new_rows_rejects_protocol_hash_mismatch():
    from scripts.summarize_mlp55_g2m_table4_20260812 import validate_new_rows

    rows = _complete_new_rows()
    rows[1]["noise_hash"] = "different-noise"

    with pytest.raises(ValueError, match="noise hash mismatch"):
        validate_new_rows(rows)


def test_validate_new_rows_rejects_validation_label_corruption():
    from scripts.summarize_mlp55_g2m_table4_20260812 import validate_new_rows

    rows = _complete_new_rows()
    rows[0]["modified_val_count"] = 1

    with pytest.raises(ValueError, match="validation/test labels were modified"):
        validate_new_rows(rows)


def test_table4_method_order_is_fixed_before_results():
    from scripts.summarize_mlp55_g2m_table4_20260812 import TABLE4_METHODS

    assert TABLE4_METHODS == (
        "frozengrace_gb_mlp_softdist_beta05_loo",
        "mlp_distill_gcn",
        "ffg2m_reimplemented",
        "krd_adapted",
        "vqgraph_adapted",
        "mlp_distill_softdist_xh",
        "mlp_distill_nrgnn",
    )


def test_normalize_new_row_accepts_runner_boolean_strings():
    from scripts.summarize_mlp55_g2m_table4_20260812 import normalize_new_row

    raw = {key: str(value) for key, value in _new_row("cora", "ffg2m_reimplemented", 0.4, 0).items()}
    raw["training_uses_graph"] = "True"
    raw["inference_uses_graph"] = "False"
    raw["g2m_uses_unlabeled_nodes"] = "1"

    row = normalize_new_row(raw, Path("runner.csv"))

    assert row["training_uses_graph"] == 1
    assert row["inference_uses_graph"] == 0
    assert row["uses_unlabeled_nodes"] == 1
