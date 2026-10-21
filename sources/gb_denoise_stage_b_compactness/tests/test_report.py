import pandas as pd
import pytest

from gb_stage_b.report import summarize_results
from gb_stage_b.utils import write_json


def test_summarize_results_writes_summary_and_paired_delta(tmp_path):
    b0 = tmp_path / "Cora" / "B0" / "seed_0" / "run"
    b1 = tmp_path / "Cora" / "B1" / "seed_0" / "run"
    write_json(
        b0 / "metrics.json",
        {"dataset": "Cora", "method": "B0", "seed": 0, "linear_eval_acc": 0.7, "mean_radius": 0.5},
    )
    write_json(
        b1 / "metrics.json",
        {"dataset": "Cora", "method": "B1", "seed": 0, "linear_eval_acc": 0.8, "mean_radius": 0.4},
    )

    out = summarize_results(tmp_path)

    assert out == tmp_path / "stage_b_summary.csv"
    assert out.exists()
    paired = tmp_path / "stage_b_paired_deltas.csv"
    assert paired.exists()
    deltas = pd.read_csv(paired)
    assert deltas.loc[0, "delta_linear_eval_acc"] == pytest.approx(0.1)
    assert deltas.loc[0, "delta_mean_radius"] == pytest.approx(-0.1)


def test_summarize_results_handles_empty_dir(tmp_path):
    out = summarize_results(tmp_path)

    assert out.exists()
    assert (tmp_path / "stage_b_paired_deltas.csv").exists()
