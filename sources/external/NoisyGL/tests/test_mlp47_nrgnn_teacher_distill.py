from __future__ import annotations

import importlib

import pytest
import torch

from utils.tools import load_conf


METHODS = ("mlp_distill_nrgnn", "mlp_distill_nrgnn_xh")
DATASETS = ("cora", "citeseer", "corafull")


def test_nrgnn_distill_predictors_are_registered() -> None:
    module = importlib.import_module("predictor.NRGNNDistill_Predictor")
    runner = importlib.import_module("scripts.run_standard_downstream")

    assert not module.mlp_distill_nrgnn_Predictor.use_xh_features
    assert module.mlp_distill_nrgnn_xh_Predictor.use_xh_features
    for method in METHODS:
        assert runner.PREDICTOR_MODULES[method] == "predictor.NRGNNDistill_Predictor"
        assert runner.predictor_class_for(method) is getattr(
            module,
            f"{method}_Predictor",
        )


def test_configs_keep_student_x_only_and_freeze_temperature() -> None:
    for method in METHODS:
        for dataset in DATASETS:
            conf = load_conf(None, method, dataset)
            assert conf.model["method"] == method
            assert conf.gb["student_input"] == "x"
            assert conf.gb["student_hidden"] == 64
            assert conf.gb["student_n_layer"] == 2
            assert conf.gb["student_dropout"] == pytest.approx(0.5)
            assert conf.gb["student_lr"] == pytest.approx(1e-2)
            assert conf.gb["student_weight_decay"] == pytest.approx(5e-4)
            assert conf.gb["student_epochs"] == 200
            assert conf.gb["distill_temperature"] == pytest.approx(1.0)
            if method.endswith("_xh"):
                assert "embedding_path" in conf.gb


def test_soft_distillation_loss_uses_probability_targets() -> None:
    module = importlib.import_module("predictor.NRGNNDistill_Predictor")
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    teacher_probs = torch.tensor([[0.75, 0.25], [0.10, 0.90]])

    loss = module.soft_target_distillation_loss(
        logits,
        teacher_probs,
        temperature=1.0,
    )

    expected = -(
        teacher_probs * torch.log_softmax(logits, dim=1)
    ).sum(dim=1).mean()
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_soft_distillation_loss_rejects_invalid_probabilities() -> None:
    module = importlib.import_module("predictor.NRGNNDistill_Predictor")
    logits = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="sum to one"):
        module.soft_target_distillation_loss(
            logits,
            torch.tensor([[0.6, 0.6], [0.5, 0.5]]),
            temperature=1.0,
        )


def test_summary_requires_complete_hash_matched_grid() -> None:
    summary = importlib.import_module(
        "scripts.summarize_mlp47_nrgnn_teacher_distill"
    )
    rows = []
    for dataset in summary.DATASETS:
        for noise_rate in summary.NOISE_RATES:
            for method_index, method in enumerate(summary.METHODS):
                for seed in summary.SEEDS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "noise_rate": noise_rate,
                            "method": method,
                            "seed": seed,
                            "student_test": 0.5 + 0.001 * method_index,
                            "teacher_test": 0.6,
                            "split_hash": f"split-{dataset}-{seed}",
                            "noise_hash": f"noise-{dataset}-{noise_rate}-{seed}",
                            "wall_time_seconds": 1.0,
                        }
                    )

    completeness, audits = summary.completeness_and_protocol(rows)
    assert len(completeness) == 3 * 2 * 7
    assert len(audits) == 3 * 2 * 5
    assert all(row["complete"] == 1 for row in completeness)
    assert all(row["passed"] == 1 for row in audits)
    assert len(summary.paired_deltas(rows)) == 3 * 2 * (6 * 5 + 2 * 5)

    rows[-1]["noise_hash"] = "mismatch"
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        summary.completeness_and_protocol(rows)
