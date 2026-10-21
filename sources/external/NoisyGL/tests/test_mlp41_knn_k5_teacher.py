from __future__ import annotations

import importlib

import pandas as pd
import pytest

from scripts import run_ssl_swap_downstream, run_standard_downstream
from utils.tools import load_conf


DATASETS = ("cora", "citeseer", "corafull")
METHOD = "mlp_distill_knn_k5_xh"
ANCHOR = "mlp_distill_knn_xh"


def test_mlp41_k5_teacher_is_registered_without_replacing_matched_k_anchor():
    module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")

    assert run_standard_downstream.PREDICTOR_MODULES[METHOD] == "predictor.FrozenGraceGB_GCN_Predictor"
    assert run_ssl_swap_downstream.PREDICTOR_MODULES[METHOD] == "predictor.FrozenGraceGB_GCN_Predictor"
    predictor = getattr(module, f"{METHOD}_Predictor")
    anchor = getattr(module, f"{ANCHOR}_Predictor")
    assert issubclass(predictor, anchor)
    assert predictor.soft_target_builder == "knn"
    assert predictor.teacher_use_xh is True
    assert predictor.teacher_target == "soft_target"


def test_mlp41_k5_configs_change_only_method_and_matched_k():
    for dataset in DATASETS:
        conf = load_conf(None, METHOD, dataset)
        anchor = load_conf(None, ANCHOR, dataset)

        assert conf.model["method"] == METHOD
        assert conf.gb["matched_k"] == 5
        assert "matched_k" not in anchor.gb
        assert conf.gb["target_beta"] == anchor.gb["target_beta"] == 0.0
        assert conf.gb["soft_target_builder"] == anchor.gb["soft_target_builder"] == "knn"
        assert conf.gb["teacher_use_xh"] == anchor.gb["teacher_use_xh"] is True
        assert conf.gb["teacher_target"] == anchor.gb["teacher_target"] == "soft_target"
        for key in (
            "embedding_path",
            "max_balls",
            "min_ball_size",
            "split_improvement",
            "alpha_baseline",
            "target_mode",
            "teacher_hidden",
            "teacher_n_layer",
            "teacher_dropout",
            "teacher_lr",
            "teacher_weight_decay",
            "teacher_epochs",
            "distill_temperature",
        ):
            assert conf.gb[key] == anchor.gb[key]


def _result_row(
    dataset: str,
    noise_rate: float,
    seed: int,
    method: str,
    test: float,
    *,
    matched_k: int | str = "",
    hash_suffix: str = "",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": noise_rate,
        "noise_scope": "train",
        "status": "ok",
        "test": test,
        "matched_k": matched_k,
        "modified_count": 10,
        "modified_train_count": 10,
        "modified_val_count": 0,
        "modified_test_count": 0,
        "split_hash": f"split-{dataset}-{seed}{hash_suffix}",
        "noise_hash": f"noise-{dataset}-{noise_rate}-{seed}{hash_suffix}",
    }


def _synthetic_summary_frames():
    summary = importlib.import_module("scripts.summarize_mlp41_knn_k5_teacher")
    k5_rows = []
    anchor_rows = []
    for dataset in summary.DATASETS:
        for noise_rate in summary.NOISE_RATES:
            for seed in summary.SEEDS:
                k5_rows.append(_result_row(dataset, noise_rate, seed, METHOD, 0.70, matched_k=5))
                anchor_rows.append(_result_row(dataset, noise_rate, seed, summary.BALL, 0.72))
                anchor_rows.append(_result_row(dataset, noise_rate, seed, summary.MATCHED_KNN, 0.73))
    return pd.DataFrame(k5_rows), pd.DataFrame(anchor_rows)


def test_mlp41_summary_pairs_30_k5_rows_with_ball_and_matched_knn():
    summary = importlib.import_module("scripts.summarize_mlp41_knn_k5_teacher")
    k5, anchors = _synthetic_summary_frames()

    combined, aggregate = summary.build_summary(k5, anchors)

    assert len(combined) == 90
    assert len(aggregate) == 6
    assert aggregate["k5_mean"].tolist() == pytest.approx([0.70] * 6)
    assert aggregate["k5_minus_ball"].tolist() == pytest.approx([-0.02] * 6)
    assert aggregate["k5_minus_matched_knn"].tolist() == pytest.approx([-0.03] * 6)


def test_mlp41_summary_rejects_anchor_hash_mismatch():
    summary = importlib.import_module("scripts.summarize_mlp41_knn_k5_teacher")
    k5, anchors = _synthetic_summary_frames()
    anchors.loc[0, "noise_hash"] = "wrong-noise-hash"

    with pytest.raises(ValueError, match="hash mismatch"):
        summary.build_summary(k5, anchors)
