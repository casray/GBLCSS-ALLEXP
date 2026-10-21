from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    noisy_y: torch.Tensor | None = None
    train_index: torch.Tensor | None = None
    val_index: torch.Tensor | None = None
    test_index: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _mask(indices: np.ndarray, n_nodes: int) -> torch.Tensor:
    mask = torch.zeros(int(n_nodes), dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def unique_undirected_edge_statistics(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, float]:
    """Count loop-free undirected edges and their endpoint label agreement."""
    source, target = edge_index.detach().cpu().long()
    keep = source != target
    lower = torch.minimum(source[keep], target[keep])
    upper = torch.maximum(source[keep], target[keep])
    if lower.numel() == 0:
        return 0, float("nan")
    pairs = torch.unique(torch.stack([lower, upper], dim=1), dim=0)
    labels = labels.detach().cpu().long()
    homophily = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).float().mean().item()
    return int(pairs.shape[0]), float(homophily)


def percent_split_indices(
    n_nodes: int,
    seed: int,
    train: float = 0.2,
    val: float = 0.1,
    test: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ordered split indices used by the formal experiments."""
    rng = np.random.RandomState(int(seed))
    remaining = np.arange(int(n_nodes))
    train_index = rng.choice(remaining, int(n_nodes * train), replace=False)
    remaining = np.setdiff1d(remaining, train_index)
    val_index = rng.choice(remaining, int(n_nodes * val), replace=False)
    remaining = np.setdiff1d(remaining, val_index)
    test_index = rng.choice(remaining, int(n_nodes * test), replace=False)
    return train_index, val_index, test_index


def percent_split(
    n_nodes: int,
    seed: int,
    train: float = 0.2,
    val: float = 0.1,
    test: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_index, val_index, test_index = percent_split_indices(n_nodes, seed, train, val, test)
    return _mask(train_index, n_nodes), _mask(val_index, n_nodes), _mask(test_index, n_nodes)


def load_public_dataset(name: str, root: str | Path, seed: int) -> GraphData:
    """Load a public PyG dataset with the paper's 20/10/70 split."""
    from torch_geometric.datasets import (
        Amazon,
        AttributedGraphDataset,
        CitationFull,
        Coauthor,
        Planetoid,
    )

    key = str(name).lower().replace("-", "").replace("_", "")
    root = Path(root)
    if key in {"cora", "citeseer", "pubmed"}:
        display = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[key]
        dataset = Planetoid(root / display, name=display)
    elif key == "corafull":
        dataset = CitationFull(root / "CoraFull", name="Cora")
    elif key == "coraml":
        dataset = CitationFull(root / "CoraML", name="cora_ml")
    elif key == "citeseerfull":
        dataset = CitationFull(root / "CiteSeerFull", name="CiteSeer")
    elif key in {"coauthorcs", "coauthorphysics"}:
        display = "CS" if key == "coauthorcs" else "Physics"
        dataset = Coauthor(root / f"Coauthor{display}", name=display)
    elif key in {"amazonphoto", "amazoncomputers"}:
        display = "Photo" if key == "amazonphoto" else "Computers"
        dataset = Amazon(root / f"Amazon{display}", name=display)
    elif key == "attributedwiki":
        dataset = AttributedGraphDataset(root / "AttributedWiki", name="Wiki")
    else:
        raise ValueError(f"unsupported public dataset: {name}")

    raw = dataset[0]
    train_index, val_index, test_index = percent_split_indices(raw.num_nodes, seed)
    return GraphData(
        name=name,
        x=raw.x.float(),
        edge_index=raw.edge_index.long(),
        y=raw.y.long(),
        train_mask=_mask(train_index, raw.num_nodes),
        val_mask=_mask(val_index, raw.num_nodes),
        test_mask=_mask(test_index, raw.num_nodes),
        train_index=torch.as_tensor(train_index, dtype=torch.long),
        val_index=torch.as_tensor(val_index, dtype=torch.long),
        test_index=torch.as_tensor(test_index, dtype=torch.long),
    )


def _independent_transition(n_classes: int, rate: float, kind: str) -> np.ndarray:
    if kind == "uniform":
        transition = np.float64(rate) / np.float64(n_classes - 1) * np.ones((n_classes, n_classes))
        np.fill_diagonal(transition, (np.float64(1.0) - np.float64(rate)) * np.ones(n_classes))
        diagonal = np.arange(n_classes)
        transition[diagonal, diagonal] = (
            transition[diagonal, diagonal] + 1.0 - transition.sum(axis=0)
        )
        return transition
    if kind == "pair":
        transition = (1.0 - rate) * np.eye(n_classes)
        for class_index in range(n_classes):
            transition[class_index, class_index - 1] = rate
        return transition
    raise ValueError(f"unsupported independent noise: {kind}")


def _instance_dependent_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    rate: float,
    seed: int,
    norm_std: float = 0.1,
) -> torch.Tensor:
    set_seed(seed)
    flip_distribution = stats.truncnorm(
        (0.0 - rate) / norm_std,
        (1.0 - rate) / norm_std,
        loc=rate,
        scale=norm_std,
    )
    flip_rates = flip_distribution.rvs(labels.shape[0])
    weights = torch.tensor(
        np.random.randn(n_classes, features.shape[1], n_classes), dtype=torch.float32
    )
    probabilities = []
    for index in range(labels.shape[0]):
        score = features[index].unsqueeze(0).mm(weights[int(labels[index])]).squeeze(0)
        score[int(labels[index])] = -torch.inf
        probability = float(flip_rates[index]) * F.softmax(score, dim=0)
        probability[int(labels[index])] += 1.0 - float(flip_rates[index])
        probabilities.append(probability.numpy())
    # Match the seeded global NumPy stream used to draw the perturbation tensor.
    output = [np.random.choice(np.arange(n_classes), p=probabilities[index]) for index in range(labels.shape[0])]
    return torch.tensor(output, dtype=torch.long)


def inject_train_label_noise(
    data: GraphData, noise_type: str, noise_rate: float, seed: int, *, noise_scope: str = "train"
) -> GraphData:
    """Corrupt training labels, optionally adding independent validation noise."""
    if noise_scope not in {"train", "train_val"}:
        raise ValueError("noise_scope must be train or train_val")
    noisy = data.y.clone()
    train_index = (
        data.train_index
        if data.train_index is not None
        else torch.nonzero(data.train_mask, as_tuple=False).flatten()
    )
    train_labels = data.y[train_index]
    if float(noise_rate) > 0.0:
        if noise_type in {"uniform", "pair"}:
            transition = _independent_transition(data.num_classes, float(noise_rate), noise_type)
            rng = np.random.RandomState(int(seed))
            sampled = []
            for label in train_labels:
                draw = rng.multinomial(1, transition[int(label)], 1)[0]
                sampled.append(int(np.flatnonzero(draw)[0]))
            noisy[train_index] = torch.tensor(sampled, dtype=torch.long)
        elif noise_type == "instance":
            noisy[train_index] = _instance_dependent_labels(
                data.x[train_index], train_labels, data.num_classes, float(noise_rate), int(seed)
            )
        else:
            raise ValueError(f"unsupported noise type: {noise_type}")
    if noise_scope == "train_val":
        validation_graph = replace(data, train_mask=data.val_mask, train_index=data.val_index)
        validation_graph = inject_train_label_noise(
            validation_graph, noise_type, noise_rate, int(seed) + 1000003)
        noisy[data.val_mask] = validation_graph.noisy_y[data.val_mask]
    data.noisy_y = noisy
    return data
