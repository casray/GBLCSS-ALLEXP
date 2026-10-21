from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GBSelectDiagnostics:
    selected_count: int
    selected_ratio_train: float
    selected_noise_rate: float
    selected_clean_count: int
    selected_noisy_count: int


def mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return np.flatnonzero(mask)
    return mask.astype(np.int64)


def ball_disagreement_scores(
    noisy_label: torch.Tensor,
    assignment,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")
    train_idx = mask_to_index_array(train_mask)
    noisy_np = noisy_label.detach().cpu().long().numpy()
    scores = torch.zeros_like(noisy_label, dtype=torch.float, device=device)
    eligible = torch.zeros_like(noisy_label, dtype=torch.bool, device=device)
    min_train_per_ball = int(min_train_per_ball)

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes)).astype(np.float32)
        dist = counts / float(max(1, counts.sum()))
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        node_scores = 1.0 - torch.as_tensor(dist[noisy_np[ball_train]], dtype=torch.float, device=device)
        scores[ball_train_tensor] = node_scores
        eligible[ball_train_tensor] = True
    return scores, eligible


def _clean_subset_from_scores(scores: torch.Tensor, train_idx: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    keep_ratio = min(1.0, max(0.0, float(keep_ratio)))
    selected = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
    if train_idx.numel() == 0 or keep_ratio <= 0.0:
        return selected
    keep_count = int(round(float(train_idx.numel()) * keep_ratio))
    keep_count = min(int(train_idx.numel()), max(1, keep_count))
    train_scores = scores[train_idx].float()
    order = torch.argsort(train_scores, stable=True)
    selected[train_idx[order[:keep_count]]] = True
    return selected


def select_clean_train_mask(
    s_ball: torch.Tensor,
    s_gcn: torch.Tensor,
    train_mask,
    keep_ratio: float = 0.6,
    mode: str = "intersection",
) -> torch.Tensor:
    if s_ball.shape != s_gcn.shape:
        raise ValueError("s_ball and s_gcn must have the same shape")
    mode = str(mode)
    train_idx = torch.as_tensor(mask_to_index_array(train_mask), dtype=torch.long, device=s_ball.device)
    if mode == "intersection":
        ball_clean = _clean_subset_from_scores(s_ball, train_idx, keep_ratio)
        gcn_clean = _clean_subset_from_scores(s_gcn, train_idx, keep_ratio)
        return ball_clean & gcn_clean
    if mode == "gcn_only":
        return _clean_subset_from_scores(s_gcn, train_idx, keep_ratio)
    if mode == "ball_only":
        return _clean_subset_from_scores(s_ball, train_idx, keep_ratio)
    raise ValueError("mode must be 'intersection', 'gcn_only', or 'ball_only'")


def build_gbselect_selection(
    noisy_label: torch.Tensor,
    assignment,
    train_mask,
    n_classes: int,
    s_gcn: torch.Tensor,
    keep_ratio: float = 0.6,
    mode: str = "intersection",
    min_train_per_ball: int = 2,
    clean_label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # clean_label is intentionally ignored. It is accepted only to make
    # leakage-regression tests explicit at the API boundary.
    del clean_label
    s_ball, eligible = ball_disagreement_scores(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
    )
    selected = select_clean_train_mask(s_ball, s_gcn, train_mask, keep_ratio=keep_ratio, mode=mode)
    if float(keep_ratio) >= 1.0:
        return selected, s_ball
    if mode != "gcn_only":
        selected = selected & eligible
    return selected, s_ball


def gbselect_diagnostics(selected_mask, noisy_label, clean_label, train_mask) -> GBSelectDiagnostics:
    selected_mask = selected_mask.bool()
    train_idx = torch.as_tensor(mask_to_index_array(train_mask), dtype=torch.long, device=noisy_label.device)
    selected_train = selected_mask[train_idx]
    selected_idx = train_idx[selected_train]
    selected_count = int(selected_idx.numel())
    if selected_count == 0:
        return GBSelectDiagnostics(
            selected_count=0,
            selected_ratio_train=0.0,
            selected_noise_rate=0.0,
            selected_clean_count=0,
            selected_noisy_count=0,
        )
    noisy_flag = noisy_label[selected_idx].long() != clean_label[selected_idx].long()
    selected_noisy_count = int(noisy_flag.sum().item())
    selected_clean_count = int(selected_count - selected_noisy_count)
    return GBSelectDiagnostics(
        selected_count=selected_count,
        selected_ratio_train=float(selected_count / max(1, int(train_idx.numel()))),
        selected_noise_rate=float(selected_noisy_count / max(1, selected_count)),
        selected_clean_count=selected_clean_count,
        selected_noisy_count=selected_noisy_count,
    )
