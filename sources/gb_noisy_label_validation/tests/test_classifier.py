import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.evaluation import train_frozen_classifier
from noisy_label_gb_grace.utils import one_hot


def test_classifier_uses_clean_eval_masks():
    h = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    clean = torch.tensor([0, 0, 1, 1, 0, 1])
    noisy = torch.tensor([1, 1, 0, 0, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])
    val_mask = torch.tensor([False, False, False, False, True, False])
    test_mask = torch.tensor([False, False, False, False, False, True])
    result = train_frozen_classifier(
        h,
        clean,
        one_hot(noisy, 2),
        train_mask,
        val_mask,
        test_mask,
        seed=0,
        epochs=5,
        patience=3,
        device=torch.device("cpu"),
    )
    assert "best_val_acc" in result
    assert "test_acc_at_best_val" in result


def test_classifier_accepts_sample_weights():
    h = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    clean = torch.tensor([0, 0, 1, 1, 0, 1])
    noisy = torch.tensor([1, 1, 0, 0, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])
    val_mask = torch.tensor([False, False, False, False, True, False])
    test_mask = torch.tensor([False, False, False, False, False, True])
    sample_weights = torch.tensor([0.5, 1.5, 1.0, 1.0, 99.0, 99.0])
    result = train_frozen_classifier(
        h,
        clean,
        one_hot(noisy, 2),
        train_mask,
        val_mask,
        test_mask,
        seed=0,
        epochs=5,
        patience=3,
        device=torch.device("cpu"),
        sample_weights=sample_weights,
    )
    assert "best_val_acc" in result
    assert "test_acc_at_best_val" in result
