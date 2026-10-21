from __future__ import annotations

from pathlib import Path

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch_geometric.datasets import TUDataset


SUPPORTED_DATASETS = {"NCI1", "PROTEINS", "MUTAG"}


def _ensure_node_features(data):
    if getattr(data, "x", None) is None:
        data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    else:
        data.x = data.x.float()
    return data


def load_dataset(name: str, root: str = "data") -> TUDataset:
    """Load a PyG TUDataset by name."""
    name = name.upper()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}; expected one of {sorted(SUPPORTED_DATASETS)}")
    return TUDataset(root=root, name=name, transform=_ensure_node_features)


def dataset_labels(dataset) -> Tensor:
    labels = [int(dataset[i].y.view(-1)[0].item()) for i in range(len(dataset))]
    return torch.tensor(labels, dtype=torch.long)


def stratified_split(dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 0):
    """Return disjoint train/val/test indices using stratified random splits."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    labels = dataset_labels(dataset).numpy()
    indices = list(range(len(dataset)))
    temp_ratio = 1.0 - train_ratio

    first = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(first.split(indices, labels))

    val_fraction_of_temp = val_ratio / temp_ratio
    temp_labels = labels[temp_idx]
    second = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction_of_temp, random_state=seed)
    val_rel, test_rel = next(second.split(temp_idx, temp_labels))

    train_indices = [int(i) for i in train_idx.tolist()]
    val_indices = [int(temp_idx[i]) for i in val_rel.tolist()]
    test_indices = [int(temp_idx[i]) for i in test_rel.tolist()]
    return train_indices, val_indices, test_indices


def corrupt_symmetric_labels(labels: Tensor, train_indices: list[int], noise_rate: float, seed: int) -> Tensor:
    """Corrupt only train labels with standard symmetric label noise."""
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1]")
    labels = labels.detach().clone().long()
    noisy = labels.clone()
    classes = torch.unique(labels).sort().values
    if len(classes) < 2:
        raise ValueError("symmetric noise requires at least two classes")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    train_tensor = torch.tensor(train_indices, dtype=torch.long)
    flip_mask = torch.rand(len(train_tensor), generator=gen) < noise_rate
    for pos in torch.nonzero(flip_mask, as_tuple=False).view(-1).tolist():
        idx = train_tensor[pos].item()
        current = noisy[idx]
        candidates = classes[classes != current]
        replacement = candidates[torch.randint(len(candidates), (1,), generator=gen).item()]
        noisy[idx] = replacement
    return noisy


def save_split_and_noise(path, train_idx, val_idx, test_idx, noisy_labels, clean_labels):
    """Persist split and noise artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_idx": list(map(int, train_idx)),
            "val_idx": list(map(int, val_idx)),
            "test_idx": list(map(int, test_idx)),
            "noisy_labels": noisy_labels.detach().cpu().long(),
            "clean_labels": clean_labels.detach().cpu().long(),
        },
        path,
    )


def load_split_and_noise(path):
    """Load cached split and noise artifacts."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)

