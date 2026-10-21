import pandas as pd
import pytest

from scripts.summarize_mlp34_pubmed_rhohat_mechanism import (
    _audit_trace_regression,
    _find_lcss_test,
    _saturation_direction_support,
    jaccard_similarity,
    spearman_with_lodo,
    validate_trace_rows,
)


def _trace_rows():
    rows = []
    for dataset in ["a", "b"]:
        for method in ["small_loss", "lcss_soft_small_loss", "lcss_soft_small_loss_shuffled"]:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": 0,
                    "noise_type": "pair",
                    "noise_rate": 0.4,
                    "noise_scope": "train",
                    "split_type": "percent",
                    "train_percent": 0.2,
                    "val_percent": 0.1,
                    "test_percent": 0.7,
                    "split_hash": f"split-{dataset}",
                    "noise_hash": f"noise-{dataset}",
                    "modified_val_count": 0,
                    "modified_test_count": 0,
                    "status": "ok",
                }
            )
    return rows


def test_validate_trace_rows_accepts_complete_paired_protocol():
    validate_trace_rows(
        _trace_rows(),
        expected_datasets={"a", "b"},
        expected_settings={("pair", 0.4)},
    )


def test_validate_trace_rows_rejects_duplicate_or_hash_mismatch():
    rows = _trace_rows()
    with pytest.raises(ValueError, match="duplicate"):
        validate_trace_rows(
            rows + [dict(rows[0])],
            expected_datasets={"a", "b"},
            expected_settings={("pair", 0.4)},
        )

    rows = _trace_rows()
    rows[1]["noise_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        validate_trace_rows(
            rows,
            expected_datasets={"a", "b"},
            expected_settings={("pair", 0.4)},
        )


def test_jaccard_similarity_handles_empty_masks():
    assert jaccard_similarity([False, False], [False, False]) == 1.0
    assert jaccard_similarity([True, False], [True, True]) == 0.5


def test_spearman_reports_leave_one_dataset_out_range():
    rows = [
        {"dataset": "a", "x": 1.0, "y": 1.0},
        {"dataset": "a", "x": 2.0, "y": 2.0},
        {"dataset": "b", "x": 3.0, "y": 3.0},
        {"dataset": "b", "x": 4.0, "y": 4.0},
        {"dataset": "c", "x": 5.0, "y": 5.0},
        {"dataset": "c", "x": 6.0, "y": 6.0},
    ]

    result = spearman_with_lodo(rows, "x", "y")

    assert result["n"] == 6
    assert result["rho"] == 1.0
    assert result["lodo_min"] == 1.0
    assert result["lodo_max"] == 1.0
    assert result["lodo_direction_consistent"] is True


def test_find_lcss_uses_explicit_breadth_source_when_fiveseed_source_lacks_dataset():
    columns = ["dataset", "method", "noise_type", "noise_rate", "seed", "test"]
    empty = pd.DataFrame(columns=columns)
    breadth = pd.DataFrame(
        [["amazonpho", "frozengrace_gb_mlp_softdist_beta05", "uniform", 0.6, 0, 0.66]],
        columns=columns,
    )

    value = _find_lcss_test("amazonpho", "uniform", 0.6, empty, empty, breadth)

    assert value == 0.66


def test_saturation_direction_requires_the_registered_sign():
    assert _saturation_direction_support("ncr", -0.6) is True
    assert _saturation_direction_support("ncr", 0.6) is False
    assert _saturation_direction_support("cdr", 0.6) is True
    assert _saturation_direction_support("mean_margin", 0.6) is False


def test_trace_regression_requires_exact_anchor_metric_and_hashes():
    trace = pd.DataFrame(
        [{
            "dataset": "cora", "method": "small_loss", "seed": 0,
            "noise_type": "pair", "noise_rate": 0.4, "test": 0.5,
            "split_hash": "s", "noise_hash": "n",
        }]
    )
    anchor = pd.DataFrame(
        [{
            "dataset": "cora", "method": "mlp_small_loss", "seed": 0,
            "noise_type": "pair", "noise_rate": 0.4, "test": 0.5,
            "split_hash": "s", "noise_hash": "n",
        }]
    )

    rows = _audit_trace_regression(trace, pd.DataFrame(), anchor, pd.DataFrame())
    assert rows[0]["exact_match"] is True

    trace.loc[0, "test"] = 0.51
    rows = _audit_trace_regression(trace, pd.DataFrame(), anchor, pd.DataFrame())
    assert rows[0]["exact_match"] is False
    assert rows[0]["within_rocm_tolerance"] is True

    trace.loc[0, "test"] = 0.54
    with pytest.raises(ValueError, match="trace regression"):
        _audit_trace_regression(trace, pd.DataFrame(), anchor, pd.DataFrame())
