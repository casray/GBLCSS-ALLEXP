"""GLNN-style, FF-G2M, KRD and VQGraph protocol adaptations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from ..data import GraphData
from ..models import GCN, MLP
from ..training import TrainResult, accuracy


def ffg2m_low_frequency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    edge_index = edge_index.to(student_logits.device).long().reshape(2, -1)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    nodes = torch.arange(int(num_nodes), device=student_logits.device)
    edge_with_loops = torch.cat([edge_index, torch.stack([nodes, nodes])], dim=1)
    src, dst = edge_with_loops
    return F.kl_div(
        F.log_softmax(student_logits[src], dim=1),
        F.log_softmax(teacher_logits.detach()[dst], dim=1),
        reduction="batchmean", log_target=True,
    )


def ffg2m_high_frequency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    edge_index: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    edge_index = edge_index.to(student_logits.device).long().reshape(2, -1)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] == 0:
        return student_logits.new_zeros(())
    src, dst = edge_index
    tau = max(float(tau), 1e-6)
    student_difference = (student_logits[src] - student_logits[dst]).abs()
    teacher_difference = (teacher_logits.detach()[src] - teacher_logits.detach()[dst]).abs()
    return F.kl_div(
        F.log_softmax(student_difference / tau, dim=1),
        F.log_softmax(teacher_difference / tau, dim=1),
        reduction="batchmean", log_target=True,
    )


def _ffg2m_loss(
    output: torch.Tensor,
    teacher_logits: torch.Tensor,
    edge_index: torch.Tensor,
    noisy: torch.Tensor,
    train_index: torch.Tensor,
    settings: Mapping[str, object],
) -> torch.Tensor:
    label_loss = F.cross_entropy(output[train_index], noisy[train_index])
    low = ffg2m_low_frequency_loss(output, teacher_logits, edge_index, output.shape[0])
    high = ffg2m_high_frequency_loss(
        output, teacher_logits, edge_index,
        tau=float(settings.get("high_frequency_temperature", 1.5)),
    )
    label_weight = float(settings.get("label_weight", 0.4))
    return label_weight * label_loss + (1.0 - label_weight) * (low + high)


def _distillation_loss(
    output: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_prob = F.log_softmax(output / max(float(temperature), 1e-6), dim=1)
    per_node = -(teacher_probs * log_prob).sum(dim=1)
    if weights is None:
        # GLNN multiplies after reduction; the adapted routes multiply before it.
        return per_node.mean() * (temperature ** 2)
    per_node = per_node * (temperature ** 2)
    weights = weights.to(output.device, dtype=output.dtype).detach().clamp_min(0.0)
    if float(weights.sum().detach().cpu()) <= 1e-12:
        weights = torch.ones_like(weights)
    weights = weights / weights.mean().clamp_min(1e-12)
    return (per_node * weights).mean()


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=1)


def _krd_weights(
    teacher: nn.Module,
    features: torch.Tensor,
    adj: torch.Tensor,
    settings: Mapping[str, object],
) -> torch.Tensor:
    teacher.eval()
    perturbations = max(1, int(settings.get("perturbations", 4)))
    noise_std = float(settings.get("perturbation_std", 0.01))
    with torch.no_grad():
        base_entropy = _entropy(F.softmax(teacher(features, adj), dim=1))
        dense_features = features.to_dense() if features.is_sparse else features
        diffs = []
        for _ in range(perturbations):
            perturbed = dense_features + torch.randn_like(dense_features) * noise_std
            probs = F.softmax(teacher(perturbed, adj), dim=1)
            diffs.append((_entropy(probs) - base_entropy).abs())
        instability = torch.stack(diffs, dim=0).mean(dim=0)
        reliability = 1.0 / (instability + 1e-6)
        minimum, maximum = reliability.min(), reliability.max()
        if float((maximum - minimum).detach().cpu()) <= 1e-12:
            reliability = torch.ones_like(reliability)
        else:
            reliability = (reliability - minimum) / (maximum - minimum)
    n = features.shape[0]
    fraction = min(max(float(settings.get("reliable_fraction", 0.5)), 1.0 / max(1, n)), 1.0)
    count = max(1, int(round(float(n) * fraction)))
    selected = torch.topk(reliability, count, largest=True).indices
    weights = torch.zeros_like(reliability)
    weights[selected] = reliability[selected].clamp_min(1e-6)
    return weights.detach()


class _MLPWithHidden(MLP):
    def forward_with_hidden(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = None
        for index, layer in enumerate(self.layers[:-1]):
            features = layer(features)
            hidden = features
            # The source's inplace ReLU also changes the saved hidden tensor.
            features = F.relu(features, inplace=True)
            features = self.batch_norms[index](features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        logits = self.layers[-1](features)
        return logits, logits if hidden is None else hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(features)[0]


class _GCNWithHidden(GCN):
    def forward_with_hidden(
        self, features: torch.Tensor, adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = None
        for layer in self.layers[:-1]:
            features = F.relu(layer(features, adj))
            hidden = features
            features = F.dropout(features, p=self.dropout, training=self.training)
        logits = self.layers[-1](features, adj)
        return logits, logits if hidden is None else hidden


class _VQGraphTeacher(nn.Module):
    def __init__(
        self, base_gcn: _GCNWithHidden, hidden_channels: int,
        codebook_size: int, token_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_gcn = base_gcn
        self.codebook = nn.Parameter(torch.randn(int(codebook_size), int(hidden_channels)) * 0.02)
        self.token_temperature = float(token_temperature)

    def soft_assignments(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.normalize(hidden, p=2, dim=1)
        codebook = F.normalize(self.codebook, p=2, dim=1)
        logits = hidden @ codebook.t()
        return F.softmax(logits / max(self.token_temperature, 1e-6), dim=1)

    def forward(
        self, features: torch.Tensor, adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, hidden = self.base_gcn.forward_with_hidden(features, adj)
        assignments = self.soft_assignments(hidden)
        codebook = F.normalize(self.codebook, p=2, dim=1)
        return logits, hidden, assignments, assignments @ codebook


def _vq_edge_loss(
    quantized: torch.Tensor, edge_index: torch.Tensor, max_edges: int = 4096,
) -> torch.Tensor:
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    if edge_index.shape[1] == 0:
        return quantized.new_zeros(())
    if edge_index.shape[1] > max_edges:
        perm = torch.randperm(edge_index.shape[1], device=quantized.device)[:max_edges]
        edge_index = edge_index[:, perm]
    src, dst = edge_index
    q = F.normalize(quantized, p=2, dim=1)
    pos_logits = (q[src] * q[dst]).sum(dim=1)
    neg_src = torch.randint(0, q.shape[0], (src.numel(),), device=q.device)
    neg_dst = torch.randint(0, q.shape[0], (src.numel(),), device=q.device)
    neg_logits = (q[neg_src] * q[neg_dst]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, targets)


def _vq_token_loss(
    hidden: torch.Tensor, assignments: torch.Tensor,
    codebook: torch.Tensor, temperature: float,
) -> torch.Tensor:
    hidden = F.normalize(hidden, p=2, dim=1)
    logits = hidden @ codebook.to(hidden.device).t()
    log_probs = F.log_softmax(logits / max(float(temperature), 1e-6), dim=1)
    return F.kl_div(log_probs, assignments.to(hidden.device).detach(), reduction="batchmean")


def _fit_teacher(
    teacher: nn.Module,
    features: torch.Tensor,
    adj: torch.Tensor,
    noisy: torch.Tensor,
    train_index: torch.Tensor,
    val_index: torch.Tensor,
    settings: Mapping[str, object],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, int]:
    optimizer = torch.optim.Adam(teacher.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_validation = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        teacher.train()
        optimizer.zero_grad()
        if isinstance(teacher, _VQGraphTeacher):
            logits, hidden, _, quantized = teacher(features, adj)
            node_loss = F.mse_loss(F.normalize(hidden, p=2, dim=1), quantized)
            edge_loss = _vq_edge_loss(
                quantized, adj.indices(), int(settings.get("sampled_edges_per_epoch", 4096)),
            )
            loss = F.cross_entropy(logits[train_index], noisy[train_index])
            loss = loss + float(settings.get("node_weight", 0.001)) * node_loss
            loss = loss + float(settings.get("edge_weight", 0.03)) * edge_loss
        else:
            logits = teacher(features, adj)
            loss = F.cross_entropy(logits[train_index], noisy[train_index])
        loss.backward()
        optimizer.step()
        teacher.eval()
        with torch.no_grad():
            prediction = teacher(features, adj)
            logits = prediction[0] if isinstance(teacher, _VQGraphTeacher) else prediction
            validation = accuracy(logits[val_index], noisy[val_index])
        # Teachers keep the last tied maximum; students keep the first.
        if validation >= best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = deepcopy(teacher.state_dict())
    teacher.load_state_dict(best_state)
    teacher.eval()
    return teacher, best_epoch


def _split_index(data: GraphData, name: str, device: str) -> torch.Tensor:
    index = getattr(data, f"{name}_index")
    if index is None:
        index = getattr(data, f"{name}_mask")
    index = index.to(device)
    if index.dtype == torch.bool:
        index = torch.nonzero(index, as_tuple=False).flatten()
    if index.numel() == 0:
        raise ValueError(f"{name} split must not be empty")
    return index.long().flatten()


def train_transfer_baseline(
    data: GraphData,
    method: str,
    settings: Mapping[str, object],
    *,
    epochs: int = 200,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    device: str = "cpu",
) -> TrainResult:
    """Train a CE GCN teacher followed by an X-only student.

    ``settings`` is the per-method mapping in protocol_adaptations.yaml.
    Optional teacher overrides: teacher_hidden, teacher_layers (or the source
    name teacher_n_layer), teacher_dropout, teacher_epochs, teacher_learning_rate
    (or teacher_lr), teacher_weight_decay. Defaults follow the shared arguments.
    KRD/VQ also accept the source lambda_kd=1 and lambda_ce=0 overrides.

    The caller owns seeding. No RNG reset occurs here. Validation selects the
    last tied teacher maximum and the first student maximum, without early stop.
    train_accuracy retains the source's pre-update, training-mode measurement
    at the selected epoch; validation/test logits are post-update and eval-mode.
    """
    method = method.lower()
    if method not in {"glnn_style", "ffg2m", "krd", "vqgraph"}:
        raise ValueError(f"unsupported graph-transfer baseline: {method}")
    teacher_epochs = int(settings.get("teacher_epochs", epochs))
    teacher_hidden = int(settings.get("teacher_hidden", hidden))
    teacher_layers = int(settings.get("teacher_layers", settings.get("teacher_n_layer", layers)))
    teacher_dropout = float(settings.get("teacher_dropout", dropout))
    for key, value in (("epochs", epochs), ("teacher_epochs", teacher_epochs),
                       ("hidden", hidden), ("teacher_hidden", teacher_hidden),
                       ("layers", layers), ("teacher_layers", teacher_layers)):
        if int(value) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= dropout <= 1 or not 0 <= teacher_dropout <= 1:
        raise ValueError("dropout and teacher_dropout must be between zero and one")
    if data.num_classes < 2:
        raise ValueError("graph-transfer CE routes require at least two classes")
    if method == "vqgraph":
        if int(settings.get("codebook_size", 128)) < 1:
            raise ValueError("codebook_size must be positive")
        if int(settings.get("sampled_edges_per_epoch", 4096)) < 1:
            raise ValueError("sampled_edges_per_epoch must be positive")
        # The adapted source has no projection for mismatched hidden widths.
        student_width = hidden if layers > 1 else data.num_classes
        teacher_width = teacher_hidden if teacher_layers > 1 else data.num_classes
        if student_width != teacher_hidden or teacher_width != teacher_hidden:
            raise ValueError("VQGraph student/teacher hidden widths must match the codebook width")

    x = data.x.to(device)
    noisy = (data.y if data.noisy_y is None else data.noisy_y).to(device)
    clean = data.y.to(device)
    train_index = _split_index(data, "train", device)
    val_index = _split_index(data, "val", device)
    test_index = _split_index(data, "test", device)
    edges = data.edge_index.to(device).long()
    # Match NoisyGL's coalesced, unnormalized sparse COO adjacency, including
    # duplicate-edge weights and its matrix orientation. GCNConv adds loops.
    adj = torch.sparse_coo_tensor(
        edges, torch.ones(edges.shape[1], device=device, dtype=x.dtype),
        (data.num_nodes, data.num_nodes),
    ).coalesce()
    edges = adj.indices()

    # Initialization order is observable: never construct the teacher first.
    student_class = _MLPWithHidden if method == "vqgraph" else MLP
    student = student_class(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    teacher_class = _GCNWithHidden if method == "vqgraph" else GCN
    teacher = teacher_class(x.shape[1], teacher_hidden, data.num_classes, teacher_layers, teacher_dropout)
    token_temperature = float(settings.get("codebook_temperature", 1.0))
    if method == "vqgraph":
        teacher = _VQGraphTeacher(
            teacher, teacher_hidden, int(settings.get("codebook_size", 128)), token_temperature,
        )
    teacher = teacher.to(device)
    teacher, _ = _fit_teacher(
        teacher, x, adj, noisy, train_index, val_index, settings, epochs=teacher_epochs,
        learning_rate=float(settings.get("teacher_learning_rate", settings.get("teacher_lr", learning_rate))),
        weight_decay=float(settings.get("teacher_weight_decay", weight_decay)),
    )
    temperature = float(settings.get("temperature", 1.0))
    with torch.no_grad():
        if method == "vqgraph":
            teacher_logits, _, assignments, _ = teacher(x, adj)
            assignments = assignments.detach()
            codebook = F.normalize(teacher.codebook.detach(), p=2, dim=1)
        else:
            teacher_logits = teacher(x, adj).detach()
        teacher_probs = F.softmax(teacher_logits / max(temperature, 1e-6), dim=1).detach()
    weights = _krd_weights(teacher, x, adj, settings) if method == "krd" else torch.ones(
        data.num_nodes, device=device,
    )
    best_validation = -1.0
    best_train = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(int(epochs)):
        student.train()
        optimizer.zero_grad()
        if method == "vqgraph":
            output, student_hidden = student.forward_with_hidden(x)
        else:
            output = student(x)
        if method == "ffg2m":
            loss = _ffg2m_loss(output, teacher_logits, edges, noisy, train_index, settings)
        elif method == "glnn_style":
            loss = _distillation_loss(output, teacher_probs, temperature)
        else:
            loss = float(settings.get("lambda_kd", 1.0)) * _distillation_loss(
                output, teacher_probs, temperature, weights,
            )
            ce_weight = float(settings.get("lambda_ce", 0.0))
            if ce_weight > 0:
                loss = loss + ce_weight * F.cross_entropy(output[train_index], noisy[train_index])
            if method == "vqgraph":
                loss = loss + float(settings.get("token_weight", 1.0)) * _vq_token_loss(
                    student_hidden, assignments, codebook, token_temperature,
                )
        train_accuracy = accuracy(output[train_index].detach(), noisy[train_index])
        loss.backward()
        optimizer.step()
        student.eval()
        with torch.no_grad():
            validation = accuracy(student(x)[val_index], noisy[val_index])
        if validation > best_validation:
            best_validation = validation
            best_train = train_accuracy
            best_epoch = epoch + 1
            best_state = deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    student.eval()
    with torch.no_grad():
        logits = student(x)
        test_accuracy = accuracy(logits[test_index], clean[test_index])
    return TrainResult(best_train, best_validation, test_accuracy, best_epoch, logits.detach().cpu())
