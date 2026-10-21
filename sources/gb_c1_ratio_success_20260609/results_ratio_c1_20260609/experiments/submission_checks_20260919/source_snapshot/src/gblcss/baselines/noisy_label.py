from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from ..data import GraphData
from ..models import MLP
from ..training import TrainResult, accuracy


def _remember_count(size: int, epoch: int, rate: float, gradual: int = 10) -> int:
    progress = min(1.0, float(epoch + 1) / float(max(1, gradual)))
    current_rate = min(max(float(rate), 0.0), 0.95) * progress
    return max(1, min(int(size), int((1.0 - current_rate) * size)))


def _symmetric_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels)
    probabilities = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float().clamp(min=1e-4, max=1.0)
    reverse_ce = -(probabilities * torch.log(one_hot)).sum(dim=1).mean()
    return float(alpha) * ce + float(beta) * reverse_ce


def _clip_logits(logits: torch.Tensor, tau: float) -> torch.Tensor:
    norms = logits.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    return logits * torch.clamp(float(tau) / norms, max=1.0)


def _jocor_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
    rate: float,
    epoch: int,
    co_lambda: float = 0.1,
    gradual: int = 10,
) -> torch.Tensor:
    ce_a = F.cross_entropy(logits_a, labels, reduction="none")
    ce_b = F.cross_entropy(logits_b, labels, reduction="none")
    probability_a = F.softmax(logits_a, dim=1)
    probability_b = F.softmax(logits_b, dim=1)
    kl_ab = F.kl_div(F.log_softmax(logits_a, dim=1), probability_b, reduction="none").sum(dim=1)
    kl_ba = F.kl_div(F.log_softmax(logits_b, dim=1), probability_a, reduction="none").sum(dim=1)
    per_node = (1.0 - co_lambda) * (ce_a + ce_b) + co_lambda * (kl_ab + kl_ba)
    keep = _remember_count(labels.shape[0], epoch, rate, gradual)
    selected = torch.topk(per_node.detach(), keep, largest=False).indices
    return per_node[selected].mean()


def train_noisy_label_baseline(
    data: GraphData,
    method: str,
    *,
    noise_rate: float,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    device: str = "cpu",
    settings: dict | None = None,
) -> TrainResult:
    """Train one of the seven matched-backbone noisy-label baselines."""
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    method = method.lower()
    supported = {"mlp_ce", "small_loss", "coteaching", "jocor", "sce", "elr", "logitclip"}
    if method not in supported:
        raise ValueError(f"unsupported matched-backbone baseline: {method}")
    settings = settings or {}
    gradual = int(settings.get("gradual_epochs", 10))
    configured_rate = settings.get("forget_rate", "synthetic_noise_rate")
    forget_rate = noise_rate if configured_rate == "synthetic_noise_rate" else float(configured_rate)

    x = data.x.to(device)
    clean = data.y.to(device)
    noisy = (data.noisy_y if data.noisy_y is not None else data.y).to(device)
    train_mask = (data.train_index if data.train_index is not None else data.train_mask).to(device)
    val_mask = (data.val_index if data.val_index is not None else data.val_mask).to(device)
    labels = noisy[train_mask]
    model_a = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device)
    model_b = MLP(x.shape[1], hidden, data.num_classes, layers, dropout).to(device) if method in {"coteaching", "jocor"} else None
    parameters = list(model_a.parameters()) + (list(model_b.parameters()) if model_b is not None else [])
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    history = torch.zeros(data.num_nodes, data.num_classes, device=device) if method == "elr" else None
    best_state = None
    best_validation = -1.0
    best_epoch = -1

    for epoch in range(int(epochs)):
        model_a.train()
        if model_b is not None:
            model_b.train()
        optimizer.zero_grad()
        logits_a = model_a(x)
        train_logits_a = logits_a[train_mask]
        if method == "mlp_ce":
            loss = F.cross_entropy(train_logits_a, labels)
        elif method == "small_loss":
            per_node = F.cross_entropy(train_logits_a, labels, reduction="none")
            keep = _remember_count(labels.shape[0], epoch, forget_rate, gradual)
            selected = torch.topk(per_node.detach(), keep, largest=False).indices
            loss = per_node[selected].mean()
        elif method == "coteaching":
            logits_b = model_b(x)[train_mask]
            loss_a = F.cross_entropy(train_logits_a, labels, reduction="none")
            loss_b = F.cross_entropy(logits_b, labels, reduction="none")
            keep = _remember_count(labels.shape[0], epoch, forget_rate, gradual)
            selected_a = torch.topk(loss_a.detach(), keep, largest=False).indices
            selected_b = torch.topk(loss_b.detach(), keep, largest=False).indices
            loss = loss_a[selected_b].mean() + loss_b[selected_a].mean()
        elif method == "jocor":
            loss = _jocor_loss(train_logits_a, model_b(x)[train_mask], labels, forget_rate, epoch,
                               co_lambda=float(settings.get("co_regularization", 0.1)), gradual=gradual)
        elif method == "sce":
            loss = _symmetric_cross_entropy(train_logits_a, labels,
                                            alpha=float(settings.get("cross_entropy_weight", 0.1)),
                                            beta=float(settings.get("reverse_cross_entropy_weight", 1.0)))
        elif method == "logitclip":
            loss = F.cross_entropy(_clip_logits(train_logits_a, tau=float(settings.get("threshold", 1.0))), labels)
        else:
            probabilities = F.softmax(train_logits_a, dim=1)
            with torch.no_grad():
                ema = float(settings.get("ema", 0.7))
                history[train_mask] = ema * history[train_mask] + (1.0 - ema) * probabilities.detach()
                history[train_mask] /= history[train_mask].sum(dim=1, keepdim=True).clamp_min(1e-12)
            ce = F.cross_entropy(train_logits_a, labels)
            agreement = (history[train_mask] * probabilities).sum(dim=1).clamp(1e-4, 1.0 - 1e-4)
            loss = ce + float(settings.get("regularization_weight", 1.0)) * torch.log(1.0 - agreement).mean()
        loss.backward()
        optimizer.step()

        model_a.eval()
        with torch.no_grad():
            logits = model_a(x)
            validation = accuracy(logits[val_mask], noisy[val_mask])
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = deepcopy(model_a.state_dict())

    if best_state is not None:
        model_a.load_state_dict(best_state)
    model_a.eval()
    with torch.no_grad():
        logits = model_a(x).cpu()
    return TrainResult(
        train_accuracy=accuracy(logits[data.train_mask], noisy.cpu()[data.train_mask]),
        validation_accuracy=accuracy(logits[data.val_mask], noisy.cpu()[data.val_mask]),
        test_accuracy=accuracy(logits[data.test_mask], clean.cpu()[data.test_mask]),
        best_epoch=int(best_epoch),
        logits=logits,
    )
