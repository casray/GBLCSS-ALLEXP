"""GRACE encoder and contrastive objective adapted from the public GRACE implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module | None, layers: int = 2):
        super().__init__()
        if int(layers) < 2:
            raise ValueError("GRACE requires at least two encoder layers")
        channels = [in_channels, *([2 * out_channels] * (int(layers) - 1)), out_channels]
        self.convolutions = nn.ModuleList(
            GCNConv(channels[index], channels[index + 1]) for index in range(int(layers))
        )
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for convolution in self.convolutions:
            x = convolution(x, edge_index)
            x = self.activation(x) if self.activation is not None else x
        return x


class GRACE(nn.Module):
    def __init__(self, encoder: Encoder, hidden: int, projection_hidden: int, temperature: float):
        super().__init__()
        self.encoder = encoder
        self.temperature = float(temperature)
        self.projection_1 = nn.Linear(hidden, projection_hidden)
        self.projection_2 = nn.Linear(projection_hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.projection_2(F.elu(self.projection_1(embedding)))

    def _similarity(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return F.normalize(first, dim=1) @ F.normalize(second, dim=1).t()

    def _semi_loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        within = torch.exp(self._similarity(first, first) / self.temperature)
        between = torch.exp(self._similarity(first, second) / self.temperature)
        denominator = within.sum(dim=1) + between.sum(dim=1) - within.diag()
        return -torch.log(between.diag() / denominator.clamp_min(1e-12))

    def loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = self.projection(first)
        second = self.projection(second)
        return 0.5 * (self._semi_loss(first, second) + self._semi_loss(second, first)).mean()


@dataclass(frozen=True)
class GRACEOutput:
    representation: torch.Tensor
    losses: list[float]


def _drop_features(features: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand(features.shape[1], device=features.device) < float(probability)
    augmented = features.clone()
    augmented[:, mask] = 0
    return augmented


def _row_normalize_features(features: torch.Tensor) -> torch.Tensor:
    shifted = features - features.min(dim=1, keepdim=True).values
    return shifted / shifted.sum(dim=1, keepdim=True).clamp_min(1.0)


def train_grace(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    config: dict,
    *,
    device: str = "cpu",
) -> GRACEOutput:
    """Train the label-agnostic GRACE stage and return row-normalized encoder output."""
    activation_name = str(config.get("activation", "relu")).lower()
    activation = nn.PReLU() if activation_name == "prelu" else nn.ReLU()
    hidden = int(config["hidden_dim"])
    model = GRACE(
        Encoder(features.shape[1], hidden, activation, layers=int(config.get("layers", 2))),
        hidden,
        int(config["projection_dim"]),
        float(config["temperature"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    x = _row_normalize_features(features.float()).to(device)
    edges = edge_index.long().to(device)
    edge_rates = config["edge_drop_rates"]
    feature_rates = config["feature_mask_rates"]
    losses: list[float] = []

    for _ in range(int(config["epochs"])):
        model.train()
        optimizer.zero_grad()
        first_edges, _ = dropout_edge(edges, p=float(edge_rates[0]), training=True)
        second_edges, _ = dropout_edge(edges, p=float(edge_rates[1]), training=True)
        first = model(_drop_features(x, feature_rates[0]), first_edges)
        second = model(_drop_features(x, feature_rates[1]), second_edges)
        loss = model.loss(first, second)
        if not torch.isfinite(loss):
            raise FloatingPointError("GRACE produced a non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        representation = F.normalize(model(x, edges), p=2, dim=1).cpu().float()
    return GRACEOutput(representation=representation, losses=losses)
