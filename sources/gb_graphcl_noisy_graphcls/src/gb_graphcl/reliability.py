from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import Tensor

from .granular_ball import random_group_ids, shuffle_group_ids


def _train_tensors(embeddings: Tensor, noisy_labels: Tensor, train_indices: list[int]):
    idx = torch.tensor(train_indices, dtype=torch.long, device=embeddings.device)
    return embeddings[idx].detach().float(), noisy_labels[idx].detach().long()


def knn_consensus_reliability(embeddings: Tensor, noisy_labels: Tensor, train_indices: list[int], k: int) -> Tensor:
    """Return same-noisy-label ratio among train kNN neighbors for each train sample."""
    train_emb, train_labels = _train_tensors(embeddings, noisy_labels.to(embeddings.device), train_indices)
    n = train_emb.size(0)
    k = max(1, min(int(k), n))
    distances = torch.cdist(train_emb, train_emb)
    nn_idx = torch.topk(distances, k=k, largest=False).indices
    neighbor_labels = train_labels[nn_idx]
    return (neighbor_labels == train_labels.view(-1, 1)).float().mean(dim=1).cpu()


def kmeans_consensus_reliability(
    embeddings: Tensor,
    noisy_labels: Tensor,
    train_indices: list[int],
    n_clusters: int,
) -> Tensor:
    """Return same-noisy-label ratio inside each KMeans cluster."""
    train_emb, train_labels = _train_tensors(embeddings.cpu(), noisy_labels.cpu(), train_indices)
    n = train_emb.size(0)
    n_clusters = max(1, min(int(n_clusters), n))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(train_emb.numpy())
    cluster_ids = torch.tensor(labels, dtype=torch.long)
    rel = torch.empty(n, dtype=torch.float32)
    for cluster in torch.unique(cluster_ids).tolist():
        mask = cluster_ids == cluster
        cluster_labels = train_labels[mask]
        rel[mask] = (cluster_labels.view(-1, 1) == cluster_labels.view(1, -1)).float().mean(dim=1)
    return rel


def gb_consensus_reliability(group_ids: Tensor, noisy_labels: Tensor, train_indices: list[int]) -> Tensor:
    """Return same-noisy-label ratio inside each assigned group."""
    labels = noisy_labels[torch.tensor(train_indices, dtype=torch.long)].detach().cpu().long()
    group_ids = group_ids.detach().cpu().long()
    if len(group_ids) != len(labels):
        raise ValueError("group_ids must have one entry per train sample")
    rel = torch.empty(len(labels), dtype=torch.float32)
    for group in torch.unique(group_ids).tolist():
        mask = group_ids == group
        group_labels = labels[mask]
        rel[mask] = (group_labels.view(-1, 1) == group_labels.view(1, -1)).float().mean(dim=1)
    return rel


def compute_reliability(method: str, embeddings, noisy_labels, train_indices, **kwargs) -> Tensor:
    """Unified reliability entry point."""
    method = method.lower()
    if method == "topk":
        return knn_consensus_reliability(embeddings, noisy_labels, train_indices, kwargs["k"])
    if method == "kmeans":
        return kmeans_consensus_reliability(embeddings, noisy_labels, train_indices, kwargs["n_clusters"])
    if method == "gb":
        return gb_consensus_reliability(kwargs["group_ids"], noisy_labels, train_indices)
    if method == "shuffled":
        return gb_consensus_reliability(shuffle_group_ids(kwargs["group_ids"], kwargs.get("seed", 0)), noisy_labels, train_indices)
    if method == "random":
        group_ids = random_group_ids(len(train_indices), kwargs["group_sizes"], kwargs.get("seed", 0))
        return gb_consensus_reliability(group_ids, noisy_labels, train_indices)
    if method == "none":
        return torch.zeros(len(train_indices), dtype=torch.float32)
    raise ValueError(f"Unknown reliability method {method!r}")


def _distribution_from_group_ids(group_ids: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    group_ids = group_ids.detach().cpu().long()
    labels = labels.detach().cpu().long()
    if len(group_ids) != len(labels):
        raise ValueError("group_ids must have one entry per train sample")
    targets = torch.empty((len(labels), num_classes), dtype=torch.float32)
    for group in torch.unique(group_ids).tolist():
        mask = group_ids == group
        counts = torch.bincount(labels[mask], minlength=num_classes).float()
        dist = counts / counts.sum().clamp_min(1e-12)
        targets[mask] = dist
    return targets


def compute_soft_targets(method: str, embeddings, noisy_labels, train_indices, num_classes: int, **kwargs) -> Tensor:
    """Compute noisy-label neighborhood vote distributions for train samples."""
    method = method.lower()
    train_idx_t = torch.tensor(train_indices, dtype=torch.long)
    train_labels = noisy_labels[train_idx_t].detach().cpu().long()
    if method == "none":
        return F.one_hot(train_labels, num_classes=num_classes).float()

    if method == "topk":
        train_emb, device_labels = _train_tensors(embeddings, noisy_labels.to(embeddings.device), train_indices)
        n = train_emb.size(0)
        k = max(1, min(int(kwargs["k"]), n))
        distances = torch.cdist(train_emb, train_emb)
        nn_idx = torch.topk(distances, k=k, largest=False).indices.cpu()
        labels_cpu = device_labels.detach().cpu().long()
        targets = torch.empty((n, num_classes), dtype=torch.float32)
        for row in range(n):
            counts = torch.bincount(labels_cpu[nn_idx[row]], minlength=num_classes).float()
            targets[row] = counts / counts.sum().clamp_min(1e-12)
        return targets

    if method == "kmeans":
        train_emb, _ = _train_tensors(embeddings.cpu(), noisy_labels.cpu(), train_indices)
        n = train_emb.size(0)
        n_clusters = max(1, min(int(kwargs["n_clusters"]), n))
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(train_emb.numpy())
        return _distribution_from_group_ids(torch.tensor(labels, dtype=torch.long), train_labels, num_classes)

    if method == "gb":
        return _distribution_from_group_ids(kwargs["group_ids"], train_labels, num_classes)

    if method == "shuffled":
        group_ids = shuffle_group_ids(kwargs["group_ids"], kwargs.get("seed", 0))
        return _distribution_from_group_ids(group_ids, train_labels, num_classes)

    if method == "random":
        group_ids = random_group_ids(len(train_indices), kwargs["group_sizes"], kwargs.get("seed", 0))
        return _distribution_from_group_ids(group_ids, train_labels, num_classes)

    raise ValueError(f"Unknown soft target method {method!r}")
