import pandas as pd
import pytest

from scripts.summarize_mlp57_xteacher import paired_summary, pre_registered_gate


def _paired_rows(deltas):
    return pd.DataFrame(
        {
            "noise_rate": [0.4] * 5 + [0.4] * 5 + [0.4] * 5,
            "dataset": ["cora"] * 5 + ["citeseer"] * 5 + ["corafull"] * 5,
            "seed": list(range(5)) * 3,
            "delta": deltas,
        }
    )


def test_paired_summary_reports_mean_and_direction_count():
    rows = _paired_rows([0.02] * 5 + [0.015] * 4 + [-0.001] + [-0.005] * 5)
    summary = paired_summary(rows)
    cora = summary[summary.dataset == "cora"].iloc[0]
    assert cora.mean_delta_pp == pytest.approx(2.0)
    assert cora.positive_seeds == 5


def test_pre_registered_gate_requires_two_datasets_and_one_pp_average():
    passing = paired_summary(
        _paired_rows([0.02] * 5 + [0.015] * 5 + [-0.002] * 5)
    )
    decision = pre_registered_gate(passing, noise_rate=0.4)
    assert decision["passed"] is True
    assert decision["passing_datasets"] == 2

    weak = passing.copy()
    weak["mean_delta_pp"] = [0.2, 0.2, -0.1]
    decision = pre_registered_gate(weak, noise_rate=0.4)
    assert decision["passed"] is False
