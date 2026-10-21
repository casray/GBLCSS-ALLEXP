from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_standard_downstream
from utils.tools import load_conf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "corafull")
LOO_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss_loo"
LOO_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled"


def test_mlp39_predictors_are_registered_with_the_frozen_robust_strategy():
    module = importlib.import_module("predictor.LCSSRobust_Predictor")

    for method, shuffled in ((LOO_COMBO, False), (LOO_SHUFFLED, True)):
        assert run_standard_downstream.PREDICTOR_MODULES[method] == "predictor.LCSSRobust_Predictor"
        predictor = getattr(module, f"{method}_Predictor")
        assert predictor.robust_strategy == "soft_small_loss"
        assert predictor.random_control is shuffled
        assert predictor.use_soft_targets is True
        assert predictor.backbone == "mlp"


def test_mlp39_configs_change_only_the_loo_switch_relative_to_soft_small_loss():
    anchor_method = "frozengrace_gb_mlp_softdist_beta05_small_loss"
    for dataset in DATASETS:
        anchor = load_conf(method=anchor_method, dataset=dataset)
        for method in (LOO_COMBO, LOO_SHUFFLED):
            conf = load_conf(method=method, dataset=dataset)
            assert conf.model["method"] == method
            assert conf.gb["leave_one_out"] is True
            assert conf.gb["target_beta"] == anchor.gb["target_beta"] == 0.5
            assert conf.gb["max_balls"] == anchor.gb["max_balls"] == 64
            assert conf.gb["min_ball_size"] == anchor.gb["min_ball_size"] == 32
            assert conf.gb["split_improvement"] == anchor.gb["split_improvement"] == 0.05
            assert conf.model["forget_rate"] == anchor.model["forget_rate"] == "auto"
            assert conf.model["num_gradual"] == anchor.model["num_gradual"] == 10
            assert conf.model["exponent"] == anchor.model["exponent"] == 1.0


def _row(dataset: str, method: str, test: float, *, hash_suffix: str = "") -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": 0,
        "noise_type": "uniform",
        "noise_rate": 0.4,
        "noise_scope": "train",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
        "status": "ok",
        "test": test,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}{hash_suffix}",
        "noise_hash": f"noise-{dataset}{hash_suffix}",
    }


def _synthetic_frames():
    summary = importlib.import_module("scripts.summarize_mlp39_loo_soft_small_loss")
    smoke_rows = []
    mlp30_rows = []
    mlp36_rows = []
    values = {
        "cora": (0.612, 0.50, 0.600, 0.57, 0.49),
        "citeseer": (0.608, 0.50, 0.600, 0.56, 0.49),
        "corafull": (0.597, 0.50, 0.600, 0.56, 0.49),
    }
    for dataset, (loo_combo, shuffled, default_combo, loo_lcss, small_only) in values.items():
        smoke_rows.extend(
            [
                _row(dataset, summary.LOO_COMBO, loo_combo),
                _row(dataset, summary.LOO_SHUFFLED, shuffled),
            ]
        )
        mlp30_rows.extend(
            [
                _row(dataset, summary.DEFAULT_COMBO, default_combo),
                _row(dataset, summary.SMALL_ONLY, small_only),
            ]
        )
        mlp36_rows.append(_row(dataset, summary.LOO_LCSS, loo_lcss))
    return pd.DataFrame(smoke_rows), pd.DataFrame(mlp30_rows), pd.DataFrame(mlp36_rows)


def test_mlp39_summary_passes_the_preregistered_synthetic_gate():
    summary = importlib.import_module("scripts.summarize_mlp39_loo_soft_small_loss")
    smoke, mlp30, mlp36 = _synthetic_frames()

    combined, deltas, gate = summary.build_summary(smoke, mlp30, mlp36)

    assert len(combined) == 15
    assert len(deltas) == 3
    assert gate["pass"] == 1
    assert gate["positive_vs_default_datasets"] == 2
    assert gate["structure_ge_1pp_datasets"] == 3
    assert gate["mean_loo_increment"] == pytest.approx((0.012 + 0.008 - 0.003) / 3)


def test_mlp39_summary_rejects_anchor_hash_mismatch():
    summary = importlib.import_module("scripts.summarize_mlp39_loo_soft_small_loss")
    smoke, mlp30, mlp36 = _synthetic_frames()
    mlp30.loc[mlp30["dataset"] == "cora", "noise_hash"] = "wrong-noise"

    with pytest.raises(ValueError, match="hash mismatch"):
        summary.build_summary(smoke, mlp30, mlp36)
