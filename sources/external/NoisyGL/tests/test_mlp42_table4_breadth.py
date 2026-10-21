from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _row(dataset: str, rate: float, seed: int, method: str, test: float) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": rate,
        "noise_scope": "train",
        "status": "ok",
        "test": test,
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{dataset}-{rate}-{seed}",
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = importlib.import_module("scripts.summarize_mlp42_table4_breadth")
    offsets = {
        summary.GCN: 0.00,
        summary.GCN_XH: 0.01,
        summary.SHUFFLED: 0.02,
        summary.KNN: 0.03,
        summary.BALL: 0.04,
    }
    by_rate: dict[float, list[dict[str, object]]] = {0.4: [], 0.6: []}
    for rate in summary.NOISE_RATES:
        for dataset in summary.DATASETS:
            for seed in summary.SEEDS:
                for method, offset in offsets.items():
                    by_rate[rate].append(_row(dataset, rate, seed, method, 0.70 + offset))
    return pd.DataFrame(by_rate[0.4]), pd.DataFrame(by_rate[0.6])


def test_mlp42_summary_builds_complete_matrix_and_paired_deltas():
    summary = importlib.import_module("scripts.summarize_mlp42_table4_breadth")
    u04, u06 = _frames()

    combined, aggregate, paired = summary.build_outputs(u04, u06)

    assert len(combined) == 100
    assert len(aggregate) == 20
    assert len(paired) == 40
    ball_vs_knn = paired[paired["comparison"].eq("ball_minus_knn")]
    assert len(ball_vs_knn) == 4
    assert ball_vs_knn["mean_delta"].tolist() == pytest.approx([0.01] * 4)
    assert ball_vs_knn["positive_seeds"].tolist() == [5] * 4


def test_mlp42_summary_rejects_hash_mismatch():
    summary = importlib.import_module("scripts.summarize_mlp42_table4_breadth")
    u04, u06 = _frames()
    u06.loc[0, "noise_hash"] = "wrong"

    with pytest.raises(ValueError, match="hash mismatch"):
        summary.build_outputs(u04, u06)


def test_mlp42_summary_rejects_validation_label_contamination():
    summary = importlib.import_module("scripts.summarize_mlp42_table4_breadth")
    u04, u06 = _frames()
    u04.loc[0, "modified_val_count"] = 1

    with pytest.raises(ValueError, match="validation or test"):
        summary.build_outputs(u04, u06)
