from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from noisy_label_gb_grace.utils import mask_to_index, set_seed


class NoisyLinearTeacher(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_noisy_linear_teacher(
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    seed: int,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = embeddings.detach().float().to(device)
    labels = noisy_labels.detach().long().to(device)
    train_mask = train_mask.detach().bool().to(device)
    train_idx = mask_to_index(train_mask)
    if train_idx.numel() == 0:
        raise ValueError("train_mask must contain at least one node")

    model = NoisyLinearTeacher(x.size(1), int(num_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x) / float(temperature)
        probs = F.softmax(logits, dim=-1).detach().cpu()
        train_pred = logits[train_idx].argmax(dim=-1)
        train_acc = float((train_pred == labels[train_idx]).float().mean().item())

    stats = {
        "teacher_epochs": int(epochs),
        "teacher_lr": float(learning_rate),
        "teacher_weight_decay": float(weight_decay),
        "teacher_temperature": float(temperature),
        "teacher_train_nodes": int(train_idx.numel()),
        "teacher_train_noisy_acc": train_acc,
        "teacher_best_train_loss": float(best_loss),
    }
    return probs, stats
