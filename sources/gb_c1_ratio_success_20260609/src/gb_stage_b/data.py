from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from gb_stage_b.config import StageBConfig


def _mask_from_indices(num_nodes: int, indices: np.ndarray) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _stratified_masks(
    labels: np.ndarray,
    config: StageBConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(labels.shape[0])
    num_classes = int(np.max(labels)) + 1 if labels.size else config.num_classes
    class_counts = np.bincount(labels, minlength=num_classes)
    train_size = int(num_nodes * config.train_ratio)
    val_size = int(num_nodes * config.val_ratio)
    test_size = num_nodes - train_size - val_size
    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    if any(size < num_classes for size in split_sizes.values()):
        raise ValueError(
            "stratified split requires each split to have at least one node per class; "
            f"got {split_sizes} for {num_classes} classes"
        )
    if class_counts.min() < 3:
        raise ValueError("stratified split requires at least 3 nodes per class")

    indices = np.arange(labels.shape[0])
    try:
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices,
            labels,
            train_size=train_size,
            stratify=labels,
            random_state=config.seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            stratify=temp_labels,
            random_state=config.seed,
        )
    except ValueError as exc:
        raise ValueError(f"stratified split failed for synthetic graph: {exc}") from exc
    return (
        _mask_from_indices(num_nodes, train_idx),
        _mask_from_indices(num_nodes, val_idx),
        _mask_from_indices(num_nodes, test_idx),
    )


def _knn_edge_index(features: np.ndarray, k: int) -> torch.Tensor:
    neighbors = NearestNeighbors(n_neighbors=min(features.shape[0], k + 2))
    neighbors.fit(features)
    _, indices = neighbors.kneighbors(features)
    edges: list[tuple[int, int]] = []
    for node, row in enumerate(indices):
        dst = [int(neighbor) for neighbor in row if int(neighbor) != node][:k]
        if len(dst) != k:
            raise ValueError("knn_k must leave enough non-self neighbors for every node")
        edges.extend((node, neighbor) for neighbor in dst)
    edges = np.asarray(edges, dtype=np.int64).T
    return torch.as_tensor(edges, dtype=torch.long).contiguous()


def _apply_train_label_noise(
    labels: np.ndarray,
    train_mask: torch.Tensor,
    label_noise: float,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten().cpu().numpy()
    num_noisy = int(round(train_indices.shape[0] * label_noise))
    noise_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    if num_noisy == 0:
        return torch.as_tensor(noisy, dtype=torch.long), noise_mask

    noisy_indices = rng.choice(train_indices, size=num_noisy, replace=False)
    for idx in noisy_indices:
        current = int(labels[idx])
        choices = [cls for cls in range(num_classes) if cls != current]
        noisy[idx] = int(rng.choice(choices))
    noise_mask[torch.as_tensor(noisy_indices, dtype=torch.long)] = True
    return torch.as_tensor(noisy, dtype=torch.long), noise_mask


def make_synthetic_graph(config: StageBConfig) -> Data:
    features, labels = make_blobs(
        n_samples=config.num_nodes,
        n_features=config.feature_dim,
        centers=config.num_classes,
        cluster_std=1.0,
        random_state=config.seed,
    )
    labels = labels.astype(np.int64, copy=False)
    train_mask, val_mask, test_mask = _stratified_masks(labels, config)
    y_noisy, noise_mask = _apply_train_label_noise(
        labels=labels,
        train_mask=train_mask,
        label_noise=config.label_noise,
        num_classes=config.num_classes,
        seed=config.seed,
    )
    y_true = torch.as_tensor(labels, dtype=torch.long)
    return Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=_knn_edge_index(features, config.knn_k),
        y=y_noisy,
        y_true=y_true,
        y_noisy=y_noisy,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        noise_mask=noise_mask,
    )


def synthetic_graph(seed: int = 0, config: StageBConfig | None = None) -> Data:
    if config is None:
        config = StageBConfig(dataset="Synthetic", seed=seed)
    elif config.seed != seed:
        config = StageBConfig(**{**config.__dict__, "seed": seed})
    return make_synthetic_graph(config)


def load_graph(
    name: str,
    seed: int = 0,
    root: str | Path = "data",
    noise_rate: float = 0.0,
    config: StageBConfig | None = None,
) -> Data:
    if name == "Synthetic":
        return synthetic_graph(seed=seed, config=config)
    if name not in {"Cora", "CiteSeer", "PubMed"}:
        raise ValueError("dataset must be Synthetic, Cora, CiteSeer, or PubMed")
    dataset = Planetoid(root=str(Path(root) / name), name=name)
    data = dataset[0]
    if config is not None and config.split_mode == "ratio":
        train_mask, val_mask, test_mask = _stratified_masks(data.y.cpu().numpy(), config)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    if noise_rate > 0:
        num_classes = int(data.y.unique().numel())
        y_true = data.y.clone()
        y_noisy, noise_mask = _apply_train_label_noise(
            labels=data.y.cpu().numpy(),
            train_mask=data.train_mask,
            label_noise=noise_rate,
            num_classes=num_classes,
            seed=seed,
        )
        data.y_true = y_true
        data.y_noisy = y_noisy
        data.y = y_noisy
        data.noise_mask = noise_mask
    return data


def dataset_summary(data: Data) -> dict[str, int | float]:
    train_size = int(data.train_mask.sum().item())
    train_noise_rate = 0.0
    if train_size > 0 and hasattr(data, "noise_mask"):
        train_noise_rate = float((data.noise_mask & data.train_mask).sum().item() / train_size)
    labels = data.y_true if hasattr(data, "y_true") else data.y
    return {
        "num_nodes": int(data.x.shape[0]),
        "num_edges": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(labels.unique().numel()),
        "train_size": train_size,
        "val_size": int(data.val_mask.sum().item()),
        "test_size": int(data.test_mask.sum().item()),
        "train_noise_rate": train_noise_rate,
    }
