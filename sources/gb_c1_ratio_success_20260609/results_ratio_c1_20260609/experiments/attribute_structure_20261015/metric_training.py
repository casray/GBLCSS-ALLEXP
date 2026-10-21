from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GB-LCSS_code" / "src"))
from gblcss.data import GraphData
from gblcss.models import MLP
from gblcss.training import accuracy, soft_cross_entropy


@dataclass(frozen=True)
class MetricTrainResult:
    logits: torch.Tensor
    best_epoch: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history: list[dict[str, float | int]]


def evaluate_logits(
    logits: torch.Tensor, labels: torch.Tensor, *, binary: bool = False
) -> dict[str, float]:
    """Recompute metrics from saved logits; binary positive label is class 1."""
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()
    if binary:
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("binary metrics require exactly two classes (two logits)")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("binary metrics require both classes 0 and 1")
    result = {
        "accuracy": accuracy(logits, labels),
        "macro_f1": float(f1_score(labels.numpy(), logits.argmax(1).numpy(), average="macro", zero_division=0)),
    }
    if binary:
        # Preserve the actual float32 softmax probabilities, including saturated ties.
        probability = logits.softmax(dim=1)[:, 1].numpy()
        result["roc_auc"] = float(roc_auc_score(labels.numpy(), probability))
        result["average_precision"] = float(average_precision_score(labels.numpy(), probability))
    return result


def select_best_epoch(scores: Iterable[float]) -> int:
    """Return the one-based earliest epoch attaining the highest finite score."""
    best_score = -math.inf
    best_epoch = None
    for epoch, score in enumerate(scores, start=1):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("validation scores must be finite")
        if score > best_score:
            best_epoch, best_score = epoch, score
    if best_epoch is None:
        raise ValueError("validation scores must not be empty")
    return best_epoch


def train_metric_mlp(
    data: GraphData,
    targets: torch.Tensor,
    *,
    primary_metric: str = "accuracy",
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    device: str = "cpu",
) -> MetricTrainResult:
    """Use the original attribute-only soft-CE protocol with validation selection."""
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    if primary_metric not in {"accuracy", "roc_auc"}:
        raise ValueError("primary_metric must be accuracy or roc_auc")
    if primary_metric == "roc_auc" and data.num_classes != 2:
        raise ValueError("roc_auc selection requires exactly two classes")

    x = data.x.to(device)
    clean = data.y.to(device)
    noisy = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train_mask = (data.train_index if data.train_index is not None else data.train_mask).to(device)
    val_mask = (data.val_index if data.val_index is not None else data.val_mask).to(device)
    targets = targets.to(device)
    if primary_metric == "roc_auc" and set(noisy[val_mask].detach().cpu().tolist()) != {0, 1}:
        raise ValueError("roc_auc validation requires both classes 0 and 1")

    model = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = soft_cross_entropy(logits[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            if primary_metric == "accuracy":
                validation = accuracy(logits[val_mask], noisy[val_mask])
            else:
                validation = evaluate_logits(logits[val_mask], noisy[val_mask], binary=True)["roc_auc"]
        history.append({"epoch": epoch + 1, "validation_score": validation, "loss": float(loss.detach().item())})
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).cpu()
    binary = data.num_classes == 2
    return MetricTrainResult(
        logits=logits,
        best_epoch=int(best_epoch),
        train_metrics=evaluate_logits(logits[data.train_mask], noisy.cpu()[data.train_mask], binary=binary),
        val_metrics=evaluate_logits(logits[data.val_mask], noisy.cpu()[data.val_mask], binary=binary),
        test_metrics=evaluate_logits(logits[data.test_mask], clean.cpu()[data.test_mask], binary=binary),
        history=history,
    )
