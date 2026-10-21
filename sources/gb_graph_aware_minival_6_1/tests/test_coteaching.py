import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.evaluation.coteaching import train_coteaching


def _toy_inputs():
    embeddings = torch.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.1],
            [0.0, 2.0],
            [0.1, 1.8],
            [1.7, 0.0],
            [0.0, 1.7],
            [2.1, 0.1],
            [0.1, 2.1],
        ],
        dtype=torch.float32,
    )
    clean = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    noisy = clean.clone()
    noisy[1] = 1
    train_mask = torch.tensor([True, True, True, True, True, True, False, False])
    val_mask = torch.tensor([False, False, False, False, False, False, True, False])
    test_mask = torch.tensor([False, False, False, False, False, False, False, True])
    return embeddings, clean, noisy, train_mask, val_mask, test_mask


def test_coteaching_returns_metrics_and_respects_keep_ratio():
    embeddings, clean, noisy, train_mask, val_mask, test_mask = _toy_inputs()
    result = train_coteaching(
        embeddings=embeddings,
        clean_labels=clean,
        noisy_labels=noisy,
        num_classes=2,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=7,
        keep_ratio_final=0.5,
        warmup_epochs=0,
        ramp_epochs=0,
        epochs=3,
        lr=0.05,
        weight_decay=0.0,
        patience=0,
        device=torch.device("cpu"),
    )
    assert 0.0 <= result["test_acc_at_best_val"] <= 1.0
    assert result["selected_count"] == 3
    assert abs(result["selected_coverage"] - 0.5) < 1e-6
    assert 0.0 <= result["selection_precision"] <= 1.0
    assert 0.0 <= result["noise_excluded_recall"] <= 1.0


def test_coteaching_is_deterministic_for_same_seed():
    embeddings, clean, noisy, train_mask, val_mask, test_mask = _toy_inputs()
    kwargs = dict(
        embeddings=embeddings,
        clean_labels=clean,
        noisy_labels=noisy,
        num_classes=2,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=11,
        keep_ratio_final=0.5,
        warmup_epochs=0,
        ramp_epochs=0,
        epochs=3,
        lr=0.05,
        weight_decay=0.0,
        patience=0,
        device=torch.device("cpu"),
    )
    first = train_coteaching(**kwargs)
    second = train_coteaching(**kwargs)
    assert first["test_acc_at_best_val"] == second["test_acc_at_best_val"]
    assert first["selected_count"] == second["selected_count"]
