from __future__ import annotations

import importlib

from utils.tools import load_conf


REAL = "mlp_distill_softdist_xh_loo"
SHUFFLED = "mlp_distill_softdist_xh_loo_shuffled"


def test_loo_distill_predictors_are_single_variable_variants():
    module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
    real = getattr(module, f"{REAL}_Predictor")
    shuffled = getattr(module, f"{SHUFFLED}_Predictor")

    assert real.teacher_use_xh is True
    assert real.teacher_target == "soft_target"
    assert real.use_soft_targets is True
    assert real.skip_ball_targets is False
    assert real.random_control is False
    assert shuffled.random_control is True


def test_loo_distill_methods_are_registered():
    runner = importlib.import_module("scripts.run_standard_downstream")

    assert runner.PREDICTOR_MODULES[REAL] == "predictor.FrozenGraceGB_GCN_Predictor"
    assert runner.PREDICTOR_MODULES[SHUFFLED] == "predictor.FrozenGraceGB_GCN_Predictor"


def test_loo_distill_configs_change_only_loo_and_control_flag():
    for dataset in ("cora", "citeseer", "corafull"):
        default = load_conf(None, "mlp_distill_softdist_xh", dataset)
        real = load_conf(None, REAL, dataset)
        shuffled = load_conf(None, SHUFFLED, dataset)

        assert real.gb["leave_one_out"] is True
        assert shuffled.gb["leave_one_out"] is True
        assert float(real.gb["target_beta"]) == float(default.gb["target_beta"]) == 0.0
        assert float(shuffled.gb["target_beta"]) == 0.0
        for key in (
            "embedding_path",
            "max_balls",
            "min_ball_size",
            "split_improvement",
            "min_train_per_ball",
            "alpha_baseline",
            "target_mode",
            "teacher_hidden",
            "teacher_n_layer",
            "teacher_epochs",
            "distill_temperature",
        ):
            assert real.gb[key] == default.gb[key]
            assert shuffled.gb[key] == default.gb[key]


def test_seed0_gate_requires_synergy_and_real_structure():
    summary = importlib.import_module("scripts.summarize_mlp38_loo_distill_smoke")
    rows = [
        {"dataset": "cora", "default": 0.80, "loo": 0.81, "loo_shuffled": 0.78},
        {"dataset": "citeseer", "default": 0.69, "loo": 0.70, "loo_shuffled": 0.68},
        {"dataset": "corafull", "default": 0.59, "loo": 0.589, "loo_shuffled": 0.57},
    ]

    decision = summary.evaluate_gate(rows)

    assert decision["positive_dataset_count"] == 2
    assert decision["real_structure_pass_count"] == 3
    assert decision["decision"] == "expand_5seed"

