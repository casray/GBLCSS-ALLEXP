import torch

from gb_graphcl.selection import train_ce_baseline, train_with_reweight


def _toy_problem():
    gen = torch.Generator().manual_seed(7)
    embeddings = torch.randn((60, 16), generator=gen)
    labels = (embeddings[:, 0] > 0).long()
    noisy = labels.clone()
    noisy[::5] = 1 - noisy[::5]
    train_idx = list(range(40))
    val_idx = list(range(40, 50))
    test_idx = list(range(50, 60))
    reliability = torch.linspace(0.0, 1.0, steps=len(train_idx))
    return embeddings, noisy, labels, train_idx, val_idx, test_idx, reliability


def test_reweight_w_min_one_equals_ce():
    embeddings, noisy, clean, train_idx, val_idx, test_idx, reliability = _toy_problem()
    kwargs = {
        "embeddings": embeddings,
        "noisy_labels": noisy,
        "clean_labels": clean,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "num_classes": 2,
        "device": torch.device("cpu"),
        "epochs": 25,
        "lr": 0.02,
        "wd": 0.0,
        "patience": 25,
        "seed": 3,
    }
    ce = train_ce_baseline(**kwargs)
    rw = train_with_reweight(**kwargs, reliability=reliability, warmup=0, ramp=1, w_min=1.0)
    assert abs(ce["test_acc_at_best_val"] - rw["test_acc_at_best_val"]) < 1e-4


def test_reweight_warmup_is_uniform():
    embeddings, noisy, clean, train_idx, val_idx, test_idx, reliability = _toy_problem()
    result = train_with_reweight(
        embeddings,
        noisy,
        clean,
        train_idx,
        val_idx,
        test_idx,
        reliability,
        num_classes=2,
        device=torch.device("cpu"),
        epochs=3,
        lr=0.02,
        wd=0.0,
        patience=3,
        warmup=2,
        ramp=2,
        w_min=0.1,
        seed=4,
        trace=True,
    )
    assert result["trace"][0]["selected_count"] == len(train_idx)
    assert result["trace"][1]["selected_count"] == len(train_idx)

