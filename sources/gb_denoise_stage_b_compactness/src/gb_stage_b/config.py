from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StageBConfig:
    dataset: str = "Cora"
    method: str = "B0"
    ssl_objective: str = "negfree"
    experiment_group: str = "default"
    variant: str = "negfree_plain"
    device: str = "auto"
    seed: int = 0
    epochs: int = 1000
    warmup_epochs: int = 300
    hidden_dim: int = 256
    proj_dim: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    edge_drop: float = 0.2
    feat_mask: float = 0.2
    edge_drop_1: float | None = None
    edge_drop_2: float | None = None
    feat_mask_1: float | None = None
    feat_mask_2: float | None = None
    lambda_gb: float = 0.005
    variance_loss_weight: float = 1.0
    ema_decay: float = 0.99
    ball_space: str = "H"
    ball_refresh_interval: int = 0
    ball_algorithm: str = "gain_split"
    gain_split_min_ball_size: int = 8
    gain_split_max_depth: int = 12
    gain_split_min_gain: float = 0.001
    assignment_control: str = "true"
    reliable_ball_min_size: int = 3
    reliable_ball_max_size: int = 256
    reliable_ball_ad_percentile: float = 50.0
    phase2_epochs: int = 200
    phase2_lr: float = 1e-4
    lambda_intra: float = 1.0
    lambda_inter: float = 1.0
    lambda_sce: float = 1.0
    phase2_ball_refresh: int = 20
    phase2_label_smoothing: float = 0.1
    phase2_supervised_loss: str = "ce"
    phase2_sce_alpha: float = 1.0
    phase2_sce_beta: float = 0.1
    phase2_use_label_correction: bool = True
    noise_rate: float = 0.0
    results_dir: str = "results"
    num_nodes: int = 300
    num_classes: int = 3
    feature_dim: int = 16
    knn_k: int = 10
    label_noise: float = 0.2
    split_mode: str = "planetoid"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs must be between 0 and epochs")
        allowed_methods = {
            "B0", "B1", "B2", "B3", "B4",
            "C1", "C1_noball", "C1_topk", "C1_shuffled",
            "C2_ce_noball", "C2_ce_realgb",
            "C2_sce_noball", "C2_sce_realgb", "C2_sce_shuffled",
        }
        if self.method not in allowed_methods:
            raise ValueError(f"method must be one of {', '.join(sorted(allowed_methods))}")
        if self.method.startswith(("C1", "C2")):
            if self.phase2_epochs < 1:
                raise ValueError("phase2_epochs must be positive for C1/C2")
            if self.phase2_lr <= 0:
                raise ValueError("phase2_lr must be positive for C1/C2")
            if self.lambda_intra < 0:
                raise ValueError("lambda_intra must be non-negative")
            if self.lambda_inter < 0:
                raise ValueError("lambda_inter must be non-negative")
            if self.lambda_sce < 0:
                raise ValueError("lambda_sce must be non-negative")
            if self.phase2_ball_refresh < 1:
                raise ValueError("phase2_ball_refresh must be positive")
            if not 0 <= self.phase2_label_smoothing <= 1:
                raise ValueError("phase2_label_smoothing must be in [0, 1]")
            if self.phase2_supervised_loss not in {"ce", "sce"}:
                raise ValueError("phase2_supervised_loss must be ce or sce")
            if self.phase2_sce_alpha < 0:
                raise ValueError("phase2_sce_alpha must be non-negative")
            if self.phase2_sce_beta < 0:
                raise ValueError("phase2_sce_beta must be non-negative")
        if not 0 <= self.noise_rate <= 1:
            raise ValueError("noise_rate must be in [0, 1]")
        if self.ssl_objective not in {"negfree", "infonce"}:
            raise ValueError("ssl_objective must be negfree or infonce")
        if self.variant not in {"negfree_plain", "negfree_bn", "negfree_var", "negfree_ema"}:
            raise ValueError("variant must be one of negfree_plain, negfree_bn, negfree_var, negfree_ema")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.ball_space not in {"H", "Z"}:
            raise ValueError("ball_space must be H or Z")
        if self.ball_algorithm not in {"gain_split", "angular"}:
            raise ValueError("ball_algorithm must be gain_split or angular")
        if self.assignment_control not in {"true", "shuffled", "size_matched_random"}:
            raise ValueError("assignment_control must be true, shuffled, or size_matched_random")
        if self.lambda_gb < 0:
            raise ValueError("lambda_gb must be non-negative")
        if self.variance_loss_weight < 0:
            raise ValueError("variance_loss_weight must be non-negative")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gain_split_min_ball_size < 2:
            raise ValueError("gain_split_min_ball_size must be at least 2")
        if self.gain_split_max_depth < 0:
            raise ValueError("gain_split_max_depth must be non-negative")
        if self.gain_split_min_gain < 0:
            raise ValueError("gain_split_min_gain must be non-negative")
        if not 0 <= self.reliable_ball_ad_percentile <= 100:
            raise ValueError("reliable_ball_ad_percentile must be in [0, 100]")
        for name, value in (
            ("edge_drop", self.edge_drop),
            ("feat_mask", self.feat_mask),
            ("edge_drop_1", self.edge_drop_1),
            ("edge_drop_2", self.edge_drop_2),
            ("feat_mask_1", self.feat_mask_1),
            ("feat_mask_2", self.feat_mask_2),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.knn_k <= 0 or self.knn_k >= self.num_nodes:
            raise ValueError("knn_k must be positive and less than num_nodes")
        if not 0 <= self.label_noise <= 1:
            raise ValueError("label_noise must be in [0, 1]")
        if self.split_mode not in {"planetoid", "ratio"}:
            raise ValueError("split_mode must be planetoid or ratio")
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("train_ratio, val_ratio, and test_ratio must be positive")
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> StageBConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return StageBConfig(**data)
