from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from .data import GraphData
from .models import GCN, MLP
from .training import TrainResult, accuracy, soft_cross_entropy


def train_consensus_teacher_student(
    data: GraphData,
    targets: torch.Tensor,
    *,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    teacher_epochs: int = 200,
    student_epochs: int = 200,
    temperature: float = 1.0,
    device: str = "cpu",
) -> TrainResult:
    """Train a GCN teacher on GB-LCSS targets and distill it to an X-only MLP."""
    if int(teacher_epochs) < 1 or int(student_epochs) < 1:
        raise ValueError("teacher_epochs and student_epochs must be positive")
    x = data.x.to(device)
    edges = data.edge_index.to(device)
    # Use the same sparse matrix orientation and aggregation path as the teacher experiments.
    adjacency = torch.sparse_coo_tensor(
        edges, torch.ones(edges.shape[1], device=device, dtype=x.dtype),
        (data.num_nodes, data.num_nodes),
    ).coalesce()
    clean = data.y.to(device)
    noisy = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train_mask = (data.train_index if data.train_index is not None else data.train_mask).to(device)
    val_mask = (data.val_index if data.val_index is not None else data.val_mask).to(device)
    targets = targets.to(device)

    # The student is initialized before the teacher consumes training randomness.
    student = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    teacher = GCN(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_validation = -1.0
    for _ in range(int(teacher_epochs)):
        teacher.train()
        optimizer.zero_grad()
        logits = teacher(x, adjacency)
        loss = soft_cross_entropy(logits[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        teacher.eval()
        with torch.no_grad():
            logits = teacher(x, adjacency)
            validation = accuracy(logits[val_mask], noisy[val_mask])
        if validation >= best_validation:
            best_validation = validation
            best_state = deepcopy(teacher.state_dict())
    if best_state is not None:
        teacher.load_state_dict(best_state)
    teacher.eval()
    with torch.no_grad():
        teacher_probabilities = F.softmax(teacher(x, adjacency) / max(float(temperature), 1e-6), dim=1)

    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_validation = -1.0
    best_epoch = -1
    for epoch in range(int(student_epochs)):
        student.train()
        optimizer.zero_grad()
        logits = student(x)
        loss = soft_cross_entropy(logits / max(float(temperature), 1e-6), teacher_probabilities)
        loss = loss * float(temperature) ** 2
        loss.backward()
        optimizer.step()
        student.eval()
        with torch.no_grad():
            logits = student(x)
            validation = accuracy(logits[val_mask], noisy[val_mask])
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = deepcopy(student.state_dict())
    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        logits = student(x).cpu()
    return TrainResult(
        train_accuracy=accuracy(logits[data.train_mask], noisy.cpu()[data.train_mask]),
        validation_accuracy=accuracy(logits[data.val_mask], noisy.cpu()[data.val_mask]),
        test_accuracy=accuracy(logits[data.test_mask], clean.cpu()[data.test_mask]),
        best_epoch=int(best_epoch),
        logits=logits,
    )
