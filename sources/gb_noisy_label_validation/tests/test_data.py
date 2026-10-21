import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.data import apply_split, split_name


def test_ratio_split_is_stratified_and_disjoint():
    y = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    data = Data(x=torch.randn(30, 4), edge_index=torch.empty((2, 0), dtype=torch.long), y=y)

    split = apply_split(data, "ratio", split_seed=0, train_ratio=0.2, val_ratio=0.1)

    assert int(split.train_mask.sum()) == 6
    assert int(split.val_mask.sum()) == 3
    assert int(split.test_mask.sum()) == 21
    assert not torch.any(split.train_mask & split.val_mask)
    assert not torch.any(split.train_mask & split.test_mask)
    assert not torch.any(split.val_mask & split.test_mask)
    for class_id in range(3):
        class_mask = y == class_id
        assert int((split.train_mask.cpu() & class_mask).sum()) == 2
        assert int((split.val_mask.cpu() & class_mask).sum()) == 1


def test_split_name_distinguishes_ratio_protocol():
    assert split_name("planetoid", 0) == "split0"
    assert split_name("ratio", 0, train_ratio=0.1, val_ratio=0.1) == "ratio_train0p1_val0p1_split0"
