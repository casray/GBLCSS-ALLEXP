from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, use_bn: bool = False) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        return self.bn2(self.conv2(h, edge_index))


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_bn: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity(),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageBModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int, variant: str = "negfree_plain") -> None:
        super().__init__()
        use_bn = variant == "negfree_bn"
        self.encoder = GCNEncoder(in_dim, hidden_dim, use_bn=use_bn)
        self.projector = MLP(hidden_dim, hidden_dim, proj_dim, use_bn=use_bn)
        self.predictor = MLP(proj_dim, hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, edge_index)
        z = self.projector(h)
        p = self.predictor(z)
        return h, z, p


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
