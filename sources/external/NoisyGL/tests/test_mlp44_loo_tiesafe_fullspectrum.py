import csv
from pathlib import Path

import pytest

from scripts.summarize_mlp44_loo_tiesafe_fullspectrum import (
    ANCHOR_METHODS,
    BASE_METHOD,
    DATASETS,
    LOO_METHOD,
    LOO_SHUFFLED_METHOD,
    SETTINGS,
    build_summary,
)


def _row(dataset, noise_type, noise_rate, seed, method, test, **overrides):
    row = {
        "dataset": dataset,
        "noise_type": noise_type,
        "noise_rate": str(noise_rate),
        "noise_scope": "train",
        "seed": str(seed),
        "method": method,
        "status": "ok",
        "test": str(test),
        "split_hash": f"split-{dataset}-{seed}",
        "noise_hash": f"noise-{dataset}-{noise_type}-{noise_rate}-{seed}",
        "modified_count": "4",
        "modified_train_count": "4",
        "modified_val_count": "0",
        "modified_test_count": "0",
        "tie_mode": "distribution_only",
        "leave_one_out": "1",
        "soft_pseudo_ncr": "0.1",
        "soft_pseudo_cdr": "0.2",
        "alpha_positive_train_ratio": "0.8",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def _complete_rows():
    rows = []
    methods = (*ANCHOR_METHODS, LOO_METHOD, LOO_SHUFFLED_METHOD)
    for dataset_index, dataset in enumerate(DATASETS):
        for setting_index, (noise_type, noise_rate) in enumerate(SETTINGS):
            for seed in range(5):
                base = 0.5 + 0.01 * dataset_index + 0.001 * setting_index
                for method in methods:
                    value = base
                    if method == LOO_METHOD:
                        value += 0.01
                    elif method == LOO_SHUFFLED_METHOD:
                        value -= 0.04
                    rows.append(
                        _row(
                            dataset,
                            noise_type,
                            noise_rate,
                            seed,
                            method,
                            value,
                            leave_one_out=(
                                "1"
                                if method in {LOO_METHOD, LOO_SHUFFLED_METHOD}
                                else "0"
                            ),
                        )
                    )
    return rows


def test_summary_accepts_complete_tiesafe_loo_grid_and_passes_gate():
    result = build_summary(_complete_rows())
    assert len(result.combined) == 750
    assert len(result.setting_deltas) == 25
    assert result.gate["pass"] == 1
    assert result.gate["positive_structure_settings"] == 25
    assert result.gate["datasets_nonnegative_vs_base"] == 5


def test_summary_rejects_mixed_v1_loo_rows():
    rows = _complete_rows()
    target = next(row for row in rows if row["method"] == LOO_METHOD)
    target["tie_mode"] = "min_index"
    with pytest.raises(ValueError, match="tie-safe"):
        build_summary(rows)


def test_summary_rejects_non_loo_shuffled_rows():
    rows = _complete_rows()
    target = next(row for row in rows if row["method"] == LOO_SHUFFLED_METHOD)
    target["leave_one_out"] = "0"
    with pytest.raises(ValueError, match="leave-one-out"):
        build_summary(rows)


def test_summary_rejects_hash_mismatch():
    rows = _complete_rows()
    target = next(row for row in rows if row["method"] == LOO_METHOD)
    target["noise_hash"] = "wrong-noise"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_summary(rows)


def test_summary_rejects_missing_cell():
    rows = _complete_rows()
    rows.pop()
    with pytest.raises(ValueError, match="missing grid cell"):
        build_summary(rows)

