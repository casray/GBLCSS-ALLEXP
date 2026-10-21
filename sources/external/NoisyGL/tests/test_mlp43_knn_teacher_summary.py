from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_mlp43_knn_teacher import (
    METHODS,
    aggregate_results,
    merge_teacher_arms,
)


def _row(dataset: str, method: str, rate: float, seed: int, test: float) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": rate,
        "noise_scope": "train",
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{dataset}-{rate}-{seed}",
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "status": "ok",
        "test": test,
    }


def _frames() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    old_rows = []
    knn_rows = []
    for dataset in ("cora", "citeseer", "corafull"):
        for rate in (0.4, 0.6):
            for seed in range(5):
                for method in METHODS:
                    if method == "mlp_distill_knn_xh":
                        continue
                    old_rows.append(_row(dataset, method, rate, seed, 0.5))
                knn = _row(dataset, "mlp_distill_knn_xh", rate, seed, 0.6)
                knn.update(
                    {
                        "knn_exclude_self": 1,
                        "target_beta": 0.0,
                        "soft_target_builder": "knn",
                        "matched_k": 10,
                    }
                )
                knn_rows.append(knn)
    return [pd.DataFrame(old_rows)], pd.DataFrame(knn_rows)


def test_merge_teacher_arms_requires_complete_hash_matched_grid():
    old_frames, corrected = _frames()
    merged = merge_teacher_arms(old_frames, corrected)
    assert len(merged) == 150
    assert set(merged["method"]) == set(METHODS)
    assert (
        merged[merged["method"] == "mlp_distill_knn_xh"]["knn_exclude_self"]
        .astype(float)
        .eq(1)
        .all()
    )


def test_merge_rejects_self_including_knn():
    old_frames, corrected = _frames()
    corrected.loc[0, "knn_exclude_self"] = 0
    with pytest.raises(ValueError, match="exclude_self"):
        merge_teacher_arms(old_frames, corrected)


def test_merge_rejects_hash_mismatch():
    old_frames, corrected = _frames()
    corrected.loc[0, "noise_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_teacher_arms(old_frames, corrected)


def test_aggregate_uses_sample_standard_deviation():
    old_frames, corrected = _frames()
    corrected.loc[
        (corrected["dataset"] == "cora")
        & (corrected["noise_rate"] == 0.4),
        "test",
    ] = [0.5, 0.6, 0.7, 0.8, 0.9]
    merged = merge_teacher_arms(old_frames, corrected)
    aggregate = aggregate_results(merged)
    row = aggregate[
        (aggregate["dataset"] == "cora")
        & (aggregate["noise_rate"] == 0.4)
        & (aggregate["method"] == "mlp_distill_knn_xh")
    ].iloc[0]
    assert row["n"] == 5
    assert row["test_mean"] == pytest.approx(0.7)
    assert row["test_std"] == pytest.approx(0.158113883)
