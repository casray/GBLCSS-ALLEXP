from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if int(n_layers) == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(int(n_layers) - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for index, layer in enumerate(self.layers[:-1]):
            hidden = F.relu(layer(hidden), inplace=True)
            hidden = self.batch_norms[index](hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layers[-1](hidden)


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.layers = nn.ModuleList()
        if int(n_layers) == 1:
            self.layers.append(GCNConv(in_channels, out_channels, add_self_loops=True))
        else:
            self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=True))
            for _ in range(int(n_layers) - 2):
                self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
            self.layers.append(GCNConv(hidden_channels, out_channels, add_self_loops=True))

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden, edge_index))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.layers[-1](hidden, edge_index)
