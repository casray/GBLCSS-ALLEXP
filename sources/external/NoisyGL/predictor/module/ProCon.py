from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch_geometric.utils import dropout_edge

from predictor.module.GNNs import GCN


def _two_hop_sets(edge_index: torch.Tensor, n_nodes: int) -> list[set[int]]:
    neighbors = [set() for _ in range(n_nodes)]
    edge_cpu = edge_index.detach().cpu()
    for source, target in edge_cpu.t().tolist():
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    reachable = []
    for node in range(n_nodes):
        candidates = set(neighbors[node])
        for neighbor in neighbors[node]:
            candidates.update(neighbors[neighbor])
        candidates.discard(node)
        reachable.append(candidates)
    return reachable


def build_peer_index_exact(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        k: int,
        alpha: float) -> torch.Tensor:
    n_nodes = int(features.shape[0])
    k = min(int(k), max(0, n_nodes - 1))
    if k < 1:
        return torch.empty((n_nodes, 0), dtype=torch.long, device=features.device)
    normalized = F.normalize(features.float(), p=2, dim=1)
    similarity = normalized @ normalized.t()
    weighted = similarity * float(alpha)
    for node, candidates in enumerate(_two_hop_sets(edge_index, n_nodes)):
        if candidates:
            candidate_idx = torch.as_tensor(
                sorted(candidates), dtype=torch.long, device=features.device)
            weighted[node, candidate_idx] = similarity[node, candidate_idx]
    weighted.fill_diagonal_(-torch.inf)
    return weighted.topk(k, dim=1).indices


def _features_to_csr(features: torch.Tensor) -> sp.csr_matrix:
    sparse = features.detach().cpu().to_sparse_coo().coalesce()
    indices = sparse.indices().numpy()
    values = sparse.values().float().numpy()
    return sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(features.shape),
        dtype=np.float32,
    ).tocsr()


def _lsh_buckets(features: sp.csr_matrix, random_seed: int, n_bits: int = 12):
    rng = np.random.RandomState(random_seed)
    planes = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(features.shape[1], n_bits),
    )
    signatures = np.asarray(features @ planes) >= 0
    powers = (1 << np.arange(n_bits, dtype=np.int64))
    codes = signatures.astype(np.int64) @ powers
    buckets = defaultdict(list)
    for node, code in enumerate(codes.tolist()):
        buckets[int(code)].append(node)
    return codes, buckets


def build_peer_index_bounded(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        k: int,
        alpha: float,
        candidate_limit: int,
        random_seed: int) -> torch.Tensor:
    n_nodes = int(features.shape[0])
    k = min(int(k), max(0, n_nodes - 1))
    candidate_limit = min(n_nodes, max(int(candidate_limit), k + 1))
    if candidate_limit >= n_nodes:
        return build_peer_index_exact(features, edge_index, k, alpha)
    if k < 1:
        return torch.empty((n_nodes, 0), dtype=torch.long, device=features.device)

    feature_csr = _features_to_csr(features)
    squared_norm = np.asarray(feature_csr.multiply(feature_csr).sum(axis=1)).reshape(-1)
    norm = np.sqrt(np.maximum(squared_norm, 1e-12))
    two_hop = _two_hop_sets(edge_index, n_nodes)
    codes, buckets = _lsh_buckets(feature_csr, random_seed)
    peers = np.empty((n_nodes, k), dtype=np.int64)

    for node in range(n_nodes):
        structural = two_hop[node]
        candidates = set(structural)
        candidates.update(buckets[int(codes[node])])
        candidates.discard(node)
        rng = np.random.RandomState(random_seed + node)
        if len(candidates) < candidate_limit:
            remaining = np.setdiff1d(
                np.arange(n_nodes, dtype=np.int64),
                np.fromiter(candidates | {node}, dtype=np.int64),
                assume_unique=False,
            )
            add_count = min(candidate_limit - len(candidates), remaining.size)
            if add_count:
                candidates.update(rng.choice(remaining, size=add_count, replace=False).tolist())
        elif len(candidates) > candidate_limit:
            structural_list = sorted(structural)
            budget = max(0, candidate_limit - len(structural_list))
            other = np.asarray(sorted(candidates - structural), dtype=np.int64)
            if other.size > budget:
                other = rng.choice(other, size=budget, replace=False)
            candidates = set(structural_list)
            candidates.update(other.tolist())

        candidate_idx = np.asarray(sorted(candidates), dtype=np.int64)
        dot = (feature_csr[node] @ feature_csr[candidate_idx].T).toarray().reshape(-1)
        cosine = dot / np.maximum(norm[node] * norm[candidate_idx], 1e-12)
        score = cosine * float(alpha)
        structural_mask = np.fromiter(
            (int(candidate) in structural for candidate in candidate_idx),
            dtype=bool,
            count=candidate_idx.size,
        )
        score[structural_mask] = cosine[structural_mask]
        top = np.argpartition(score, -k)[-k:]
        top = top[np.argsort(score[top])[::-1]]
        peers[node] = candidate_idx[top]

    return torch.as_tensor(peers, dtype=torch.long, device=features.device)


def gmm_clean_probability(scores: torch.Tensor, random_seed: int) -> torch.Tensor:
    flat = scores.detach().float().cpu().view(-1)
    if flat.numel() < 4 or float(flat.std(unbiased=False)) < 1e-6:
        return torch.ones_like(scores, dtype=torch.float)
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=int(random_seed),
        reg_covar=1e-5,
    )
    values = flat.numpy().reshape(-1, 1)
    try:
        model.fit(values)
        clean_component = int(np.argmax(model.means_.reshape(-1)))
        probability = model.predict_proba(values)[:, clean_component]
    except ValueError:
        probability = np.ones(flat.numel(), dtype=np.float32)
    return torch.as_tensor(
        probability,
        dtype=torch.float,
        device=scores.device,
    ).view_as(scores)


def update_prototypes_ema(
        prototypes: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        momentum: float,
        n_classes: int) -> torch.Tensor:
    updated = prototypes.detach().clone()
    for class_id in range(int(n_classes)):
        mask = labels == class_id
        if not bool(mask.any()):
            continue
        class_weights = weights[mask].float()
        denominator = class_weights.sum()
        if float(denominator) <= 0:
            continue
        class_center = (
            embeddings[mask] * class_weights.unsqueeze(1)
        ).sum(dim=0) / denominator
        updated[class_id] = (
            float(momentum) * updated[class_id]
            + (1.0 - float(momentum)) * class_center.detach()
        )
    return updated


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, momentum: float) -> None:
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.data.mul_(float(momentum)).add_(
            online_parameter.data, alpha=1.0 - float(momentum))


class ProConEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            n_layers: int,
            dropout: float):
        super().__init__()
        self.gnn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            n_layers=n_layers,
            dropout=dropout,
            norm_info=None,
            act="F.relu",
            input_layer=False,
            output_layer=False,
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn(features, edge_index)


class ProjectionPredictor(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


def make_graph_view(
        features: torch.Tensor,
        edge_index: torch.Tensor,
        feature_drop: float,
        edge_drop: float):
    dropped_edge, _ = dropout_edge(
        edge_index, p=float(edge_drop), force_undirected=True, training=True)
    if feature_drop <= 0:
        return features, dropped_edge
    mask = torch.rand_like(features) >= float(feature_drop)
    return features * mask, dropped_edge


def negative_cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 2.0 - 2.0 * F.cosine_similarity(first, second.detach(), dim=1).mean()


def copy_frozen(module: nn.Module) -> nn.Module:
    target = deepcopy(module)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return target
