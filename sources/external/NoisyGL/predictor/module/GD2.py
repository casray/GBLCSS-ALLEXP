from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops

from predictor.module.GNNs import GCN


def build_graph_views(edge_index: torch.Tensor, n_nodes: int):
    structure, _ = remove_self_loops(edge_index)
    structure = coalesce(structure, num_nodes=int(n_nodes))
    nodes = torch.arange(n_nodes, device=edge_index.device)
    ego = torch.stack([nodes, nodes], dim=0)
    mixed, _ = add_self_loops(structure, num_nodes=int(n_nodes))
    mixed = coalesce(mixed, num_nodes=int(n_nodes))
    return ego, structure, mixed


def prediction_discrepancy(
        first_probability: torch.Tensor,
        second_probability: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(first_probability - second_probability, ord=2, dim=1)


def gmm_low_component_probability(
        discrepancy: torch.Tensor,
        random_seed: int) -> torch.Tensor:
    flat = discrepancy.detach().float().cpu().view(-1)
    if flat.numel() < 4 or float(flat.std(unbiased=False)) < 1e-6:
        return torch.ones_like(discrepancy, dtype=torch.float)
    values = flat.numpy().reshape(-1, 1)
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(random_seed),
        reg_covar=1e-5,
    )
    try:
        model.fit(values)
        clean_component = int(np.argmin(model.means_.reshape(-1)))
        probability = model.predict_proba(values)[:, clean_component]
    except ValueError:
        probability = np.ones(flat.numel(), dtype=np.float32)
    return torch.as_tensor(
        probability,
        dtype=torch.float,
        device=discrepancy.device,
    ).view_as(discrepancy)


def select_purified_nodes(
        probabilities: torch.Tensor,
        noisy_indices: torch.Tensor,
        confidence_threshold: float):
    noisy_indices = torch.as_tensor(
        noisy_indices, dtype=torch.long, device=probabilities.device)
    if noisy_indices.numel() == 0:
        return noisy_indices, noisy_indices.clone()
    confidence, pseudo_label = probabilities.max(dim=1)
    selected = confidence > float(confidence_threshold)
    return noisy_indices[selected], pseudo_label[selected]


def structural_classification_loss(
        logits: torch.Tensor,
        clean_indices: torch.Tensor,
        noisy_labels: torch.Tensor,
        purified_indices: torch.Tensor,
        pseudo_labels: torch.Tensor) -> torch.Tensor:
    indices = clean_indices
    targets = noisy_labels[clean_indices]
    if purified_indices.numel() > 0:
        indices = torch.cat([indices, purified_indices])
        targets = torch.cat([targets, pseudo_labels])
    return F.cross_entropy(logits[indices], targets)


class GD2Model(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            n_classes: int,
            n_layers: int,
            dropout: float):
        super().__init__()
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
            norm_info=None,
            act="F.relu",
            input_layer=False,
            output_layer=False,
            add_self_loops=False,
        )
        self.ego_head = nn.Linear(hidden_channels, n_classes)
        self.graph_head = nn.Linear(hidden_channels, n_classes)

    def encode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(features, edge_index)

    def ego_logits(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.ego_head(self.encode(features, edge_index))

    def graph_logits(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.graph_head(self.encode(features, edge_index))
