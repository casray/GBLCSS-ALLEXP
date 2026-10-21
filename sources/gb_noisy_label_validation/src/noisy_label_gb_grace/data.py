from __future__ import annotations

from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Planetoid

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _format_float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def split_name(
    split_mode: str,
    split_seed: int,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> str:
    if split_mode == "planetoid":
        return f"split{split_seed}"
    if split_mode == "ratio":
        if train_ratio is None or val_ratio is None:
            raise ValueError("ratio split requires train_ratio and val_ratio")
        return (
            f"ratio_train{_format_float_token(train_ratio)}"
            f"_val{_format_float_token(val_ratio)}"
            f"_split{split_seed}"
        )
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def apply_split(
    data,
    split_mode: str,
    split_seed: int,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
):
    if split_mode == "planetoid":
        return data
    if split_mode != "ratio":
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    if train_ratio is None or val_ratio is None:
        raise ValueError("ratio split requires train_ratio and val_ratio")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("train_ratio and val_ratio must be in (0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    labels = data.y.detach().cpu().long()
    train_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    val_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    test_mask = torch.zeros(labels.numel(), dtype=torch.bool)
    generator = torch.Generator().manual_seed(int(split_seed))
    num_classes = int(labels.max().item()) + 1

    for class_id in range(num_classes):
        class_idx = (labels == class_id).nonzero(as_tuple=False).view(-1)
        perm = class_idx[torch.randperm(class_idx.numel(), generator=generator)]
        n_total = int(perm.numel())
        n_train = max(1, int(round(n_total * float(train_ratio))))
        n_val = max(1, int(round(n_total * float(val_ratio))))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    split_data = data.clone()
    split_data.train_mask = train_mask.to(data.y.device)
    split_data.val_mask = val_mask.to(data.y.device)
    split_data.test_mask = test_mask.to(data.y.device)
    return split_data


def load_dataset(
    dataset_name: str,
    data_root: Path | None = None,
    device=None,
    split_mode: str = "planetoid",
    split_seed: int = 0,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
):
    assert dataset_name in {"Cora", "CiteSeer", "PubMed", "DBLP"}
    pyg_name = "dblp" if dataset_name == "DBLP" else dataset_name
    root = data_root or PROJECT_ROOT / "data" / dataset_name
    dataset = (CitationFull if pyg_name == "dblp" else Planetoid)(
        str(root),
        pyg_name,
        transform=T.NormalizeFeatures(),
    )
    data = apply_split(dataset[0], split_mode, split_seed, train_ratio, val_ratio)
    if device is not None:
        data = data.to(device)
    return dataset, data


def extract_masks(data) -> dict[str, torch.Tensor]:
    return {
        "train_mask": data.train_mask.detach().cpu().bool(),
        "val_mask": data.val_mask.detach().cpu().bool(),
        "test_mask": data.test_mask.detach().cpu().bool(),
    }
