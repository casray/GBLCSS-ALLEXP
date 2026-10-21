from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper_pipeline"
    / "scripts"
    / "summarize_v38_loo_manuscript.py"
)
SPEC = importlib.util.spec_from_file_location("v38_summary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    values = [0.01, 0.04, 0.03]
    adjusted = MODULE.holm_adjust(values)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_paired_statistics_use_sample_standard_deviation() -> None:
    result = MODULE.paired_stats([0.01, 0.02, 0.03, 0.04, 0.05])
    assert result["n"] == 5
    assert result["mean_delta"] == pytest.approx(0.03)
    assert result["std_delta"] == pytest.approx(0.0158113883)
    assert result["positive_seeds"] == 5

