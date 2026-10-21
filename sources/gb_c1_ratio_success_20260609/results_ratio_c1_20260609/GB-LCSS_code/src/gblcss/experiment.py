from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .data import GraphData, inject_train_label_noise, set_seed
from .distillation import train_consensus_teacher_student
from .granular_ball import build_granular_balls, random_size_matched_assignment
from .targets import build_leave_one_out_targets
from .training import TrainResult, train_x_only_mlp


@dataclass(frozen=True)
class ExperimentOutput:
    result: TrainResult
    diagnostics: dict[str, object]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_gblcss(
    data: GraphData,
    representation: torch.Tensor,
    config: dict,
    *,
    seed: int,
    shuffled: bool = False,
    distill: bool = False,
) -> ExperimentOutput:
    set_seed(seed)
    method = config["method"]
    model = config["model"]
    training = config["training"]
    balls = build_granular_balls(
        representation,
        max_balls=method["max_balls"],
        min_ball_size=method["min_ball_size"],
        split_improvement=method["split_improvement"],
        random_state=seed,
    )
    assignment = (
        random_size_matched_assignment(balls.assignment, seed)
        if shuffled
        else balls.assignment
    )
    labels = data.noisy_y if data.noisy_y is not None else data.y
    targets = build_leave_one_out_targets(
        labels.to(training.get("device", "cpu")),
        assignment,
        data.train_mask,
        data.num_classes,
        min_train_per_ball=method["min_train_per_ball"],
        beta=method["beta"],
    )
    common = {
        "hidden": model["hidden"],
        "layers": model["layers"],
        "dropout": model["dropout"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "device": training.get("device", "cpu"),
    }
    if distill:
        result = train_consensus_teacher_student(
            data,
            targets.soft_target,
            teacher_epochs=training["epochs"],
            student_epochs=training["epochs"],
            temperature=config.get("distillation", {}).get("temperature", 1.0),
            **common,
        )
    else:
        result = train_x_only_mlp(data, targets.soft_target, epochs=training["epochs"], **common)
    return ExperimentOutput(
        result=result,
        diagnostics={**balls.summary, **targets.diagnostics, "random_size_matched": int(shuffled)},
    )


def make_smoke_graph(seed: int = 0) -> tuple[GraphData, torch.Tensor]:
    """Create a deterministic graph with class-local geometry for offline testing."""
    set_seed(seed)
    n_classes, nodes_per_class, feature_dim = 3, 48, 12
    labels = torch.arange(n_classes).repeat_interleave(nodes_per_class)
    centers = F.one_hot(torch.arange(n_classes), num_classes=feature_dim).float() * 3.0
    x = centers[labels] + 0.35 * torch.randn(labels.shape[0], feature_dim)
    representation = centers[labels] + 0.12 * torch.randn(labels.shape[0], feature_dim)
    sources, destinations = [], []
    for class_index in range(n_classes):
        start = class_index * nodes_per_class
        for offset in range(nodes_per_class):
            node = start + offset
            other = start + ((offset + 1) % nodes_per_class)
            sources.extend([node, other])
            destinations.extend([other, node])
    edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    train_mask, val_mask, test_mask = (
        torch.zeros(labels.shape[0], dtype=torch.bool),
        torch.zeros(labels.shape[0], dtype=torch.bool),
        torch.zeros(labels.shape[0], dtype=torch.bool),
    )
    for class_index in range(n_classes):
        start = class_index * nodes_per_class
        train_mask[start : start + 12] = True
        val_mask[start + 12 : start + 20] = True
        test_mask[start + 20 : start + nodes_per_class] = True
    graph = GraphData("synthetic", x, edge_index, labels, train_mask, val_mask, test_mask)
    graph = inject_train_label_noise(graph, "uniform", 0.4, seed)
    return graph, representation
