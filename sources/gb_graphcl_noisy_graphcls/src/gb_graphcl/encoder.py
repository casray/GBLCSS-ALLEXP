from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class GINEncoder(nn.Module):
    """GIN graph encoder with global mean pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        if x is None:
            raise ValueError("GINEncoder requires node features; load_dataset adds ones when absent")
        h = x.float()
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            h = bn(h)
            h = torch.relu(h)
        return global_mean_pool(h, batch)


class ProjectionHead(nn.Module):
    """Two-layer MLP used only for contrastive training."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)

