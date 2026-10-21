from __future__ import annotations

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def linear_eval(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
) -> float:
    del val_mask
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    train_idx = train_mask.detach().cpu().numpy().astype(bool)
    test_idx = test_mask.detach().cpu().numpy().astype(bool)
    if train_idx.sum() == 0 or test_idx.sum() == 0:
        raise ValueError("train_mask and test_mask must select at least one node")

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    return float(accuracy_score(y[test_idx], pred))
