from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_mlp46_loo_clean_control.py"
)
SPEC = importlib.util.spec_from_file_location("mlp46_summary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in MODULE.DATASETS:
        for seed in range(5):
            for method_index, method in enumerate(MODULE.METHODS):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": str(seed),
                        "noise_type": "uniform",
                        "noise_rate": "0.0",
                        "noise_scope": "train",
                        "split_hash": f"split-{dataset}-{seed}",
                        "noise_hash": "clean",
                        "status": "ok",
                        "modified_count": "0",
                        "modified_train_count": "0",
                        "modified_val_count": "0",
                        "modified_test_count": "0",
                        "tie_mode": (
                            "" if method == "mlp_ce" else "distribution_only"
                        ),
                        "leave_one_out": "" if method == "mlp_ce" else "1",
                        "test": str(0.8 - method_index * 0.01 + seed * 0.001),
                    }
                )
    return rows


def test_clean_control_summary_is_complete_and_paired() -> None:
    aggregate, paired, audit = MODULE.summarize(make_rows())
    assert len(aggregate) == 15
    assert len(paired) == 10
    assert len(audit) == 25
    cora_ce = next(
        row
        for row in paired
        if row["dataset"] == "cora" and row["comparison"] == "loo_vs_mlp_ce"
    )
    assert cora_ce["mean_delta"] == pytest.approx(0.02)
    assert cora_ce["positive_seeds"] == 5


def test_clean_control_rejects_old_loo_lineage() -> None:
    rows = make_rows()
    target = next(row for row in rows if row["method"] == MODULE.LOO)
    target["tie_mode"] = "min_index"
    with pytest.raises(ValueError, match="non-v2"):
        MODULE.summarize(rows)

