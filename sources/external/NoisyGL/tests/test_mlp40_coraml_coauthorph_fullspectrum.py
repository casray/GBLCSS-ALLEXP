from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _row(
    dataset: str,
    method: str,
    noise_type: str,
    noise_rate: float,
    test: float,
    *,
    hash_suffix: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": 0,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "split_type": "percent",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "status": "ok",
        "test": test,
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}{hash_suffix}",
        "noise_hash": f"noise-{dataset}-{noise_type}-{noise_rate}{hash_suffix}",
    }


def _synthetic_raw() -> pd.DataFrame:
    summary = importlib.import_module("scripts.summarize_mlp40_coraml_coauthorph_fullspectrum")
    rows = []
    settings = {
        "coraml": (("uniform", 0.6), ("pair", 0.4), ("instance", 0.4)),
        "coauthorph": (("pair", 0.4), ("instance", 0.4)),
    }
    for dataset, regimes in settings.items():
        for noise_type, noise_rate in regimes:
            for method_index, method in enumerate(summary.METHODS):
                score = 0.50 - 0.001 * method_index
                if method == summary.OURS:
                    score = 0.52
                elif method == summary.SHUFFLED:
                    score = 0.48
                rows.append(_row(dataset, method, noise_type, noise_rate, score))
    return pd.DataFrame(rows)


def test_mlp40_summary_validates_the_45_cell_matrix_and_applies_frozen_gates():
    summary = importlib.import_module("scripts.summarize_mlp40_coraml_coauthorph_fullspectrum")

    combined, setting_summary, gates = summary.build_summary(_synthetic_raw())

    assert len(combined) == 45
    assert len(setting_summary) == 5
    assert set(setting_summary["best_baseline_method"]) == {"mlp_ce"}
    assert setting_summary["lcss_minus_best"].tolist() == pytest.approx([0.022] * 5)
    assert setting_summary["lcss_minus_shuffled"].tolist() == pytest.approx([0.04] * 5)
    assert gates.set_index("dataset").loc["coraml", "pass"] == 1
    assert gates.set_index("dataset").loc["coauthorph", "pass"] == 1
    assert gates.set_index("dataset").loc["coraml", "total_settings"] == 3
    assert gates.set_index("dataset").loc["coauthorph", "total_settings"] == 2


def test_mlp40_summary_rejects_hash_mismatch_within_a_setting():
    summary = importlib.import_module("scripts.summarize_mlp40_coraml_coauthorph_fullspectrum")
    frame = _synthetic_raw()
    mask = (
        (frame["dataset"] == "coraml")
        & (frame["noise_type"] == "uniform")
        & (frame["method"] == "mlp_ce")
    )
    frame.loc[mask, "noise_hash"] = "wrong-noise-hash"

    with pytest.raises(ValueError, match="hash mismatch"):
        summary.build_summary(frame)


def test_mlp40_summary_rejects_validation_or_test_label_modification():
    summary = importlib.import_module("scripts.summarize_mlp40_coraml_coauthorph_fullspectrum")
    frame = _synthetic_raw()
    frame.loc[0, "modified_test_count"] = 1

    with pytest.raises(ValueError, match="validation or test labels"):
        summary.build_summary(frame)
