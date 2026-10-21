from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import GraphData
from .models import MLP


@dataclass(frozen=True)
class TrainResult:
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    best_epoch: int
    logits: torch.Tensor


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def train_x_only_mlp(
    data: GraphData,
    targets: torch.Tensor,
    *,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    device: str = "cpu",
) -> TrainResult:
    """Train and evaluate an MLP whose forward pass receives node attributes only."""
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    x = data.x.to(device)
    clean = data.y.to(device)
    noisy = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train_mask = (data.train_index if data.train_index is not None else data.train_mask).to(device)
    val_mask = (data.val_index if data.val_index is not None else data.val_mask).to(device)
    targets = targets.to(device)

    model = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation = -1.0
    best_epoch = -1
    best_state = None

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
            validation = accuracy(logits[val_mask], noisy[val_mask])
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).cpu()
    return TrainResult(
        train_accuracy=accuracy(logits[data.train_mask], noisy.cpu()[data.train_mask]),
        validation_accuracy=accuracy(logits[data.val_mask], noisy.cpu()[data.val_mask]),
        test_accuracy=accuracy(logits[data.test_mask], clean.cpu()[data.test_mask]),
        best_epoch=int(best_epoch),
        logits=logits,
    )
