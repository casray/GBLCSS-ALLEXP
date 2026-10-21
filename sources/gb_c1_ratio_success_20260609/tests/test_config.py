import builtins
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from gb_stage_b import __version__
from gb_stage_b.config import StageBConfig, load_config
from gb_stage_b.train import resolve_device
from gb_stage_b.utils import ensure_dir, read_json, set_seed, write_json


def test_package_imports():
    assert __version__ == "0.1.0"


def test_load_config_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
dataset: Synthetic
method: B1
ssl_objective: negfree
epochs: 10
warmup_epochs: 4
lambda_gb: 0.005
seed: 0
""",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert isinstance(cfg, StageBConfig)
    assert cfg.dataset == "Synthetic"
    assert cfg.method == "B1"
    assert cfg.ball_space == "H"
    assert cfg.assignment_control == "true"


def test_config_rejects_bad_warmup():
    with pytest.raises(ValueError, match="warmup_epochs"):
        StageBConfig(dataset="Synthetic", epochs=3, warmup_epochs=4)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
dataset: Synthetic
unknown_key: value
""",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="unknown_key"):
        load_config(path)


def test_config_rejects_bad_percentile_and_enum():
    with pytest.raises(ValueError, match="reliable_ball_ad_percentile"):
        StageBConfig(dataset="Synthetic", reliable_ball_ad_percentile=101.0)
    with pytest.raises(ValueError, match="method"):
        StageBConfig(dataset="Synthetic", method="B9")


def test_config_accepts_and_validates_synthetic_graph_fields():
    cfg = StageBConfig(
        dataset="Synthetic",
        num_nodes=120,
        num_classes=4,
        feature_dim=8,
        knn_k=6,
        label_noise=0.15,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    assert cfg.num_nodes == 120
    assert cfg.num_classes == 4
    assert cfg.feature_dim == 8
    assert cfg.knn_k == 6
    assert cfg.label_noise == 0.15


def test_config_accepts_stability_ablation_fields():
    cfg = StageBConfig(
        dataset="Synthetic",
        experiment_group="aug_weak",
        variant="negfree_bn",
        device="auto",
        edge_drop_1=0.1,
        edge_drop_2=0.2,
        feat_mask_1=0.1,
        feat_mask_2=0.2,
        ball_algorithm="gain_split",
        gain_split_min_ball_size=4,
        gain_split_max_depth=8,
        gain_split_min_gain=0.0005,
        results_dir="results_stability",
    )

    assert cfg.experiment_group == "aug_weak"
    assert cfg.variant == "negfree_bn"
    assert cfg.device == "auto"
    assert cfg.ball_algorithm == "gain_split"
    assert cfg.results_dir == "results_stability"


def test_config_accepts_c2_structure_regularization_fields():
    cfg = StageBConfig(
        dataset="Synthetic",
        method="C2_sce_realgb",
        phase2_supervised_loss="sce",
        phase2_sce_alpha=1.0,
        phase2_sce_beta=0.1,
    )

    assert cfg.method == "C2_sce_realgb"
    assert cfg.phase2_supervised_loss == "sce"
    assert cfg.phase2_sce_beta == 0.1


def test_config_accepts_c1_gated_fields():
    cfg = StageBConfig(
        dataset="Synthetic",
        method="C1_gated",
        gate_tau_cov=3,
        gate_tau_conf=0.6,
        gate_alpha=0.3,
    )

    assert cfg.method == "C1_gated"
    assert cfg.gate_tau_cov == 3
    assert cfg.gate_tau_conf == 0.6
    assert cfg.gate_alpha == 0.3


def test_config_accepts_c1_weighted_fields():
    cfg = StageBConfig(
        dataset="Synthetic",
        method="C1_weighted",
        weighted_k_ref=3,
        weighted_floor=0.25,
    )

    assert cfg.method == "C1_weighted"
    assert cfg.weighted_k_ref == 3
    assert cfg.weighted_floor == 0.25


def test_config_accepts_ratio_split_mode():
    cfg = StageBConfig(dataset="Cora", split_mode="ratio", train_ratio=0.2, val_ratio=0.1, test_ratio=0.7)

    assert cfg.split_mode == "ratio"
    assert cfg.train_ratio == 0.2


def test_config_rejects_bad_synthetic_graph_fields():
    with pytest.raises(ValueError, match="label_noise"):
        StageBConfig(dataset="Synthetic", label_noise=1.1)
    with pytest.raises(ValueError, match="knn_k"):
        StageBConfig(dataset="Synthetic", num_nodes=10, knn_k=10)
    with pytest.raises(ValueError, match="sum to 1"):
        StageBConfig(dataset="Synthetic", train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_config_rejects_bad_stability_fields():
    with pytest.raises(ValueError, match="device"):
        StageBConfig(dataset="Synthetic", device="tpu")
    with pytest.raises(ValueError, match="variant"):
        StageBConfig(dataset="Synthetic", variant="unknown")
    with pytest.raises(ValueError, match="ball_algorithm"):
        StageBConfig(dataset="Synthetic", ball_algorithm="labels")
    with pytest.raises(ValueError, match="gain_split_min_ball_size"):
        StageBConfig(dataset="Synthetic", gain_split_min_ball_size=1)


def test_config_rejects_bad_c2_supervised_loss_fields():
    with pytest.raises(ValueError, match="phase2_supervised_loss"):
        StageBConfig(dataset="Synthetic", method="C2_sce_realgb", phase2_supervised_loss="mae")
    with pytest.raises(ValueError, match="phase2_sce_beta"):
        StageBConfig(dataset="Synthetic", method="C2_sce_realgb", phase2_sce_beta=-0.1)


def test_config_rejects_bad_gate_fields():
    with pytest.raises(ValueError, match="gate_tau_cov"):
        StageBConfig(dataset="Synthetic", method="C1_gated", gate_tau_cov=0)
    with pytest.raises(ValueError, match="gate_tau_conf"):
        StageBConfig(dataset="Synthetic", method="C1_gated", gate_tau_conf=1.1)
    with pytest.raises(ValueError, match="gate_alpha"):
        StageBConfig(dataset="Synthetic", method="C1_gated", gate_alpha=-0.1)


def test_config_rejects_bad_weighted_fields():
    with pytest.raises(ValueError, match="weighted_k_ref"):
        StageBConfig(dataset="Synthetic", method="C1_weighted", weighted_k_ref=0)
    with pytest.raises(ValueError, match="weighted_floor"):
        StageBConfig(dataset="Synthetic", method="C1_weighted", weighted_floor=-0.1)
    with pytest.raises(ValueError, match="weighted_floor"):
        StageBConfig(dataset="Synthetic", method="C1_weighted", weighted_floor=1.1)


def test_config_rejects_bad_split_mode():
    with pytest.raises(ValueError, match="split_mode"):
        StageBConfig(dataset="Cora", split_mode="random")


def test_resolve_device_auto_returns_usable_torch_device():
    device = resolve_device("auto")

    assert device.type in {"cuda", "cpu"}


def test_utils_seed_and_dir(tmp_path):
    set_seed(123)
    out = ensure_dir(tmp_path / "nested")
    assert out.exists()
    assert out.is_dir()


def test_set_seed_reproducible_for_random_and_numpy():
    set_seed(123)
    first_random = random.random()
    first_numpy = np.random.rand()

    set_seed(123)

    assert random.random() == first_random
    assert np.random.rand() == first_numpy


def test_set_seed_without_torch(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    original_utils = sys.modules.get("gb_stage_b.utils")
    sys.modules.pop("gb_stage_b.utils", None)
    try:
        utils = importlib.import_module("gb_stage_b.utils")
        utils.set_seed(123)
    finally:
        if original_utils is not None:
            sys.modules["gb_stage_b.utils"] = original_utils
        else:
            sys.modules.pop("gb_stage_b.utils", None)


def test_json_helpers_create_parent_and_round_trip_utf8(tmp_path):
    path = tmp_path / "a" / "b" / "out.json"
    data = {"text": "中文"}

    write_json(path, data)

    assert path.parent.exists()
    text = path.read_text(encoding="utf-8")
    assert '\n  "text":' in text
    assert "中文" in text
    assert read_json(path) == data
