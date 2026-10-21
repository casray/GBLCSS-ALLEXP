import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.noise import corrupt_pair_labels, corrupt_symmetric_labels


def test_symmetric_noise_only_changes_train_labels():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    train_mask = torch.tensor([True, True, True, False, False, False])
    noisy, meta = corrupt_symmetric_labels(labels, train_mask, num_classes=3, noise_rate=1.0, seed=7)
    assert torch.equal(noisy[~train_mask], labels[~train_mask])
    assert torch.all(noisy[train_mask] != labels[train_mask])
    assert meta["num_flipped"] == 3


def test_zero_noise_keeps_all_labels():
    labels = torch.tensor([0, 1, 2])
    train_mask = torch.tensor([True, True, False])
    noisy, meta = corrupt_symmetric_labels(labels, train_mask, num_classes=3, noise_rate=0.0, seed=1)
    assert torch.equal(noisy, labels)
    assert meta["num_flipped"] == 0


def test_pair_noise_only_changes_train_labels_to_next_class():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    train_mask = torch.tensor([True, True, True, False, False, False])
    noisy, meta = corrupt_pair_labels(labels, train_mask, num_classes=3, noise_rate=1.0, seed=3)
    assert torch.equal(noisy[~train_mask], labels[~train_mask])
    assert torch.equal(noisy[train_mask], torch.tensor([1, 2, 0]))
    assert meta["noise_type"] == "pair"
    assert meta["num_flipped"] == 3
