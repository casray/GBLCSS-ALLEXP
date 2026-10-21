from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from noisy_label_gb_grace.utils import mask_to_index, one_hot


GATED_DIAGNOSTIC_FIELDS = [
    "dataset",
    "noise_rate",
    "seed",
    "method",
    "num_train",
    "num_noisy_train",
    "train_noise_rate",
    "gated_train_count",
    "gated_train_ratio",
    "gated_noisy_count",
    "gated_clean_count",
    "gated_noisy_ratio",
    "gated_noisy_lift_over_base_noise_rate",
    "qmix_argmax_eq_noisy_rate",
    "qmix_argmax_eq_clean_rate",
    "qmix_argmax_other_rate",
    "qmix_noisy_support_mean",
    "qmix_clean_support_mean",
    "qmix_clean_minus_noisy_mean",
    "qmix_top1_conf_mean",
    "qmix_top1_top2_margin_mean",
    "qmix_top1_minus_noisy_mean",
    "gated_noisy_qmix_argmax_eq_noisy_rate",
    "gated_noisy_qmix_argmax_eq_clean_rate",
    "gated_noisy_qmix_clean_minus_noisy_mean",
    "gated_noisy_qmix_top1_conf_mean",
    "gated_noisy_qmix_top1_top2_margin_mean",
    "target_argmax_change_rate",
    "target_argmax_eq_noisy_rate",
    "target_argmax_eq_clean_rate",
    "target_argmax_other_rate",
    "target_noisy_support_mean",
    "target_clean_support_mean",
    "target_clean_minus_noisy_mean",
    "gated_noisy_target_argmax_change_rate",
    "gated_noisy_target_argmax_eq_noisy_rate",
    "gated_noisy_target_argmax_eq_clean_rate",
    "gated_noisy_target_clean_minus_noisy_mean",
    "gated_alpha_mean",
    "gated_alpha_std",
    "gated_alpha_min",
    "gated_alpha_q25",
    "gated_alpha_q50",
    "gated_alpha_q75",
    "gated_alpha_max",
    "teacher_train_clean_acc",
    "teacher_gated_clean_acc",
    "teacher_gated_noisy_clean_acc",
    "teacher_gated_cleannode_clean_acc",
    "teacher_gated_conf_mean",
    "teacher_gated_noisy_conf_mean",
    "teacher_gated_cleannode_conf_mean",
    "teacher_gated_argmax_eq_noisy_rate",
    "teacher_gated_argmax_eq_clean_rate",
    "gated_ball_clean_purity_mean",
    "gated_ball_noisy_entropy_mean",
    "gated_ball_train_count_mean",
    "gated_ball_size_mean",
    "gated_ball_qmix_entropy_mean",
    "current_reported_ncr",
    "recomputed_target_argmax_ncr",
    "possible_correction_rate_before_alpha",
    "lost_by_alpha_rate",
]


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else math.nan


def _mean(values: torch.Tensor | list[float]) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float()
        return float(values.mean().item()) if values.numel() else math.nan
    return float(np.mean(values)) if values else math.nan


def _std(values: torch.Tensor | list[float]) -> float:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float()
        return float(values.std(unbiased=False).item()) if values.numel() else math.nan
    return float(np.std(np.array(values, dtype=float), ddof=0)) if values else math.nan


def _quantile(values: torch.Tensor, q: float) -> float:
    values = values.detach().cpu().float()
    return float(torch.quantile(values, torch.tensor(float(q))).item()) if values.numel() else math.nan


def _entropy(probs: torch.Tensor) -> float:
    probs = probs.detach().cpu().float()
    if probs.numel() == 0 or float(probs.sum().item()) <= 0.0:
        return math.nan
    probs = probs / probs.sum().clamp_min(1e-12)
    return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())


def _distribution(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    counts = torch.bincount(labels[indices].long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


def _top1_top2_margin(probs: torch.Tensor) -> torch.Tensor:
    if probs.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    if probs.size(1) == 1:
        return probs[:, 0]
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _alpha_for_nodes(
    nodes: torch.Tensor,
    group_id: int,
    radii: torch.Tensor,
    node_distances: torch.Tensor,
    alpha_max: float,
    eps: float,
) -> torch.Tensor:
    radius = float(radii[group_id].item()) if group_id < radii.numel() else 0.0
    ball_conf = 1.0 / (1.0 + radius)
    node_conf = torch.exp(-node_distances[nodes] / (radius + eps))
    return float(alpha_max) * ball_conf * node_conf


def _rate_argmax(rows: torch.Tensor, labels: torch.Tensor) -> float:
    if rows.numel() == 0:
        return math.nan
    return float((rows.argmax(dim=1) == labels).float().mean().item())


def _other_rate(pred: torch.Tensor, noisy: torch.Tensor, clean: torch.Tensor) -> float:
    if pred.numel() == 0:
        return math.nan
    return float(((pred != noisy) & (pred != clean)).float().mean().item())


def _ball_quality_for_gated_nodes(
    gated_idx: torch.Tensor,
    grouping: dict,
    qmix_by_node: torch.Tensor,
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
) -> dict:
    group_ids = grouping["group_ids"].detach().cpu().long()
    clean_purity = []
    noisy_entropy = []
    train_counts = []
    sizes = []
    qmix_entropy = []
    for node in gated_idx.tolist():
        group_id = int(group_ids[node].item())
        group = torch.tensor(grouping["groups"][group_id], dtype=torch.long)
        train_in_group = group[train_mask[group]]
        if train_in_group.numel() == 0:
            continue
        clean_counts = torch.bincount(clean_labels[train_in_group].long(), minlength=num_classes).float()
        noisy_dist = _distribution(noisy_labels, train_in_group, num_classes)
        clean_purity.append(float(clean_counts.max().item() / clean_counts.sum().clamp_min(1.0).item()))
        noisy_entropy.append(_entropy(noisy_dist))
        train_counts.append(float(train_in_group.numel()))
        sizes.append(float(group.numel()))
        qmix_entropy.append(_entropy(qmix_by_node[node]))
    return {
        "gated_ball_clean_purity_mean": _mean(clean_purity),
        "gated_ball_noisy_entropy_mean": _mean(noisy_entropy),
        "gated_ball_train_count_mean": _mean(train_counts),
        "gated_ball_size_mean": _mean(sizes),
        "gated_ball_qmix_entropy_mean": _mean(qmix_entropy),
    }


def compute_gated_node_diagnostics(
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    grouping: dict,
    teacher_probs: torch.Tensor,
    *,
    num_classes: int,
    alpha_max: float,
    dataset: str,
    noise_rate: float,
    seed: int,
    method: str = "gagb_qb_conflict_gate",
    qb_pred_beta: float = 0.5,
    min_train_per_group: int = 3,
    eps: float = 1e-8,
    conflict_noisy_support_max: float = 0.35,
    conflict_target_conf_min: float = 0.55,
    current_reported_ncr: float | None = None,
) -> dict:
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    probs = teacher_probs.detach().cpu().float()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    group_ids = grouping["group_ids"].detach().cpu().long()
    radii = grouping["radii"].detach().cpu().float()
    node_distances = grouping["node_distances"].detach().cpu().float()
    num_nodes = int(clean.numel())

    qmix_by_node = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    alpha_by_node = torch.zeros(num_nodes, dtype=torch.float32)
    gate_mask = torch.zeros(num_nodes, dtype=torch.bool)
    targets = one_hot(noisy, num_classes)

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        if train_in_group.numel() < int(min_train_per_group):
            continue
        q_label = _distribution(noisy, train_in_group, num_classes)
        q_pred = probs[idx].mean(dim=0)
        q_pred = q_pred / q_pred.sum().clamp_min(1e-12)
        q_mix = (1.0 - float(qb_pred_beta)) * q_label + float(qb_pred_beta) * q_pred
        q_mix = q_mix / q_mix.sum().clamp_min(1e-12)
        qmix_by_node[train_in_group] = q_mix

        alpha = _alpha_for_nodes(train_in_group, group_id, radii, node_distances, alpha_max, eps)
        noisy_support = q_mix[noisy[train_in_group]]
        gate = (noisy_support <= float(conflict_noisy_support_max)) & (q_mix.max() >= float(conflict_target_conf_min))
        gated_nodes = train_in_group[gate]
        alpha_by_node[gated_nodes] = alpha[gate]
        gate_mask[gated_nodes] = True
        if gated_nodes.numel():
            gated_alpha = alpha[gate]
            targets[gated_nodes] = (1.0 - gated_alpha).unsqueeze(1) * targets[gated_nodes] + gated_alpha.unsqueeze(1) * q_mix

    train_idx = mask_to_index(train_mask)
    noisy_train_mask = train_mask & (clean != noisy)
    noisy_train_idx = mask_to_index(noisy_train_mask)
    gated_idx = mask_to_index(gate_mask & train_mask)
    gated_noisy_idx = mask_to_index(gate_mask & noisy_train_mask)
    gated_clean_idx = mask_to_index(gate_mask & train_mask & (clean == noisy))

    num_train = int(train_idx.numel())
    num_noisy_train = int(noisy_train_idx.numel())
    train_noise_rate = _safe_rate(num_noisy_train, num_train)
    gated_train_count = int(gated_idx.numel())
    gated_noisy_count = int(gated_noisy_idx.numel())
    gated_clean_count = int(gated_clean_idx.numel())
    gated_noisy_ratio = _safe_rate(gated_noisy_count, gated_train_count)

    qmix_gated = qmix_by_node[gated_idx]
    target_gated = targets[gated_idx]
    qmix_noisy = qmix_by_node[gated_noisy_idx]
    target_noisy = targets[gated_noisy_idx]
    teacher_gated = probs[gated_idx]
    teacher_noisy = probs[gated_noisy_idx]
    teacher_clean = probs[gated_clean_idx]

    qmix_pred = qmix_gated.argmax(dim=1) if gated_idx.numel() else torch.empty(0, dtype=torch.long)
    target_pred = target_gated.argmax(dim=1) if gated_idx.numel() else torch.empty(0, dtype=torch.long)
    qmix_noisy_pred = qmix_noisy.argmax(dim=1) if gated_noisy_idx.numel() else torch.empty(0, dtype=torch.long)
    target_noisy_pred = target_noisy.argmax(dim=1) if gated_noisy_idx.numel() else torch.empty(0, dtype=torch.long)
    all_target_pred = targets.argmax(dim=1)
    all_qmix_pred = qmix_by_node.argmax(dim=1)
    teacher_pred = probs.argmax(dim=1)

    possible_mask = (all_qmix_pred[gated_noisy_idx] == clean[gated_noisy_idx]) if gated_noisy_idx.numel() else torch.empty(0, dtype=torch.bool)
    lost_mask = possible_mask & (all_target_pred[gated_noisy_idx] != clean[gated_noisy_idx]) if gated_noisy_idx.numel() else torch.empty(0, dtype=torch.bool)
    recomputed_ncr = (
        float((all_target_pred[noisy_train_idx] == clean[noisy_train_idx]).float().mean().item())
        if noisy_train_idx.numel()
        else math.nan
    )

    row = {
        "dataset": dataset,
        "noise_rate": float(noise_rate),
        "seed": int(seed),
        "method": method,
        "num_train": num_train,
        "num_noisy_train": num_noisy_train,
        "train_noise_rate": train_noise_rate,
        "gated_train_count": gated_train_count,
        "gated_train_ratio": _safe_rate(gated_train_count, num_train),
        "gated_noisy_count": gated_noisy_count,
        "gated_clean_count": gated_clean_count,
        "gated_noisy_ratio": gated_noisy_ratio,
        "gated_noisy_lift_over_base_noise_rate": gated_noisy_ratio / train_noise_rate if train_noise_rate and not math.isnan(gated_noisy_ratio) else math.nan,
        "qmix_argmax_eq_noisy_rate": _rate_argmax(qmix_gated, noisy[gated_idx]) if gated_idx.numel() else math.nan,
        "qmix_argmax_eq_clean_rate": _rate_argmax(qmix_gated, clean[gated_idx]) if gated_idx.numel() else math.nan,
        "qmix_argmax_other_rate": _other_rate(qmix_pred, noisy[gated_idx], clean[gated_idx]) if gated_idx.numel() else math.nan,
        "qmix_noisy_support_mean": _mean(qmix_gated[torch.arange(gated_idx.numel()), noisy[gated_idx]]) if gated_idx.numel() else math.nan,
        "qmix_clean_support_mean": _mean(qmix_gated[torch.arange(gated_idx.numel()), clean[gated_idx]]) if gated_idx.numel() else math.nan,
        "qmix_clean_minus_noisy_mean": _mean(qmix_gated[torch.arange(gated_idx.numel()), clean[gated_idx]] - qmix_gated[torch.arange(gated_idx.numel()), noisy[gated_idx]]) if gated_idx.numel() else math.nan,
        "qmix_top1_conf_mean": _mean(qmix_gated.max(dim=1).values) if gated_idx.numel() else math.nan,
        "qmix_top1_top2_margin_mean": _mean(_top1_top2_margin(qmix_gated)) if gated_idx.numel() else math.nan,
        "qmix_top1_minus_noisy_mean": _mean(qmix_gated.max(dim=1).values - qmix_gated[torch.arange(gated_idx.numel()), noisy[gated_idx]]) if gated_idx.numel() else math.nan,
        "gated_noisy_qmix_argmax_eq_noisy_rate": _rate_argmax(qmix_noisy, noisy[gated_noisy_idx]) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_qmix_argmax_eq_clean_rate": _rate_argmax(qmix_noisy, clean[gated_noisy_idx]) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_qmix_clean_minus_noisy_mean": _mean(qmix_noisy[torch.arange(gated_noisy_idx.numel()), clean[gated_noisy_idx]] - qmix_noisy[torch.arange(gated_noisy_idx.numel()), noisy[gated_noisy_idx]]) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_qmix_top1_conf_mean": _mean(qmix_noisy.max(dim=1).values) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_qmix_top1_top2_margin_mean": _mean(_top1_top2_margin(qmix_noisy)) if gated_noisy_idx.numel() else math.nan,
        "target_argmax_change_rate": float((target_pred != noisy[gated_idx]).float().mean().item()) if gated_idx.numel() else math.nan,
        "target_argmax_eq_noisy_rate": _rate_argmax(target_gated, noisy[gated_idx]) if gated_idx.numel() else math.nan,
        "target_argmax_eq_clean_rate": _rate_argmax(target_gated, clean[gated_idx]) if gated_idx.numel() else math.nan,
        "target_argmax_other_rate": _other_rate(target_pred, noisy[gated_idx], clean[gated_idx]) if gated_idx.numel() else math.nan,
        "target_noisy_support_mean": _mean(target_gated[torch.arange(gated_idx.numel()), noisy[gated_idx]]) if gated_idx.numel() else math.nan,
        "target_clean_support_mean": _mean(target_gated[torch.arange(gated_idx.numel()), clean[gated_idx]]) if gated_idx.numel() else math.nan,
        "target_clean_minus_noisy_mean": _mean(target_gated[torch.arange(gated_idx.numel()), clean[gated_idx]] - target_gated[torch.arange(gated_idx.numel()), noisy[gated_idx]]) if gated_idx.numel() else math.nan,
        "gated_noisy_target_argmax_change_rate": float((target_noisy_pred != noisy[gated_noisy_idx]).float().mean().item()) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_target_argmax_eq_noisy_rate": _rate_argmax(target_noisy, noisy[gated_noisy_idx]) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_target_argmax_eq_clean_rate": _rate_argmax(target_noisy, clean[gated_noisy_idx]) if gated_noisy_idx.numel() else math.nan,
        "gated_noisy_target_clean_minus_noisy_mean": _mean(target_noisy[torch.arange(gated_noisy_idx.numel()), clean[gated_noisy_idx]] - target_noisy[torch.arange(gated_noisy_idx.numel()), noisy[gated_noisy_idx]]) if gated_noisy_idx.numel() else math.nan,
        "gated_alpha_mean": _mean(alpha_by_node[gated_idx]) if gated_idx.numel() else math.nan,
        "gated_alpha_std": _std(alpha_by_node[gated_idx]) if gated_idx.numel() else math.nan,
        "gated_alpha_min": float(alpha_by_node[gated_idx].min().item()) if gated_idx.numel() else math.nan,
        "gated_alpha_q25": _quantile(alpha_by_node[gated_idx], 0.25) if gated_idx.numel() else math.nan,
        "gated_alpha_q50": _quantile(alpha_by_node[gated_idx], 0.50) if gated_idx.numel() else math.nan,
        "gated_alpha_q75": _quantile(alpha_by_node[gated_idx], 0.75) if gated_idx.numel() else math.nan,
        "gated_alpha_max": float(alpha_by_node[gated_idx].max().item()) if gated_idx.numel() else math.nan,
        "teacher_train_clean_acc": float((teacher_pred[train_idx] == clean[train_idx]).float().mean().item()) if train_idx.numel() else math.nan,
        "teacher_gated_clean_acc": float((teacher_pred[gated_idx] == clean[gated_idx]).float().mean().item()) if gated_idx.numel() else math.nan,
        "teacher_gated_noisy_clean_acc": float((teacher_pred[gated_noisy_idx] == clean[gated_noisy_idx]).float().mean().item()) if gated_noisy_idx.numel() else math.nan,
        "teacher_gated_cleannode_clean_acc": float((teacher_pred[gated_clean_idx] == clean[gated_clean_idx]).float().mean().item()) if gated_clean_idx.numel() else math.nan,
        "teacher_gated_conf_mean": _mean(teacher_gated.max(dim=1).values) if gated_idx.numel() else math.nan,
        "teacher_gated_noisy_conf_mean": _mean(teacher_noisy.max(dim=1).values) if gated_noisy_idx.numel() else math.nan,
        "teacher_gated_cleannode_conf_mean": _mean(teacher_clean.max(dim=1).values) if gated_clean_idx.numel() else math.nan,
        "teacher_gated_argmax_eq_noisy_rate": _rate_argmax(teacher_gated, noisy[gated_idx]) if gated_idx.numel() else math.nan,
        "teacher_gated_argmax_eq_clean_rate": _rate_argmax(teacher_gated, clean[gated_idx]) if gated_idx.numel() else math.nan,
        "current_reported_ncr": float(current_reported_ncr) if current_reported_ncr is not None and not math.isnan(float(current_reported_ncr)) else recomputed_ncr,
        "recomputed_target_argmax_ncr": recomputed_ncr,
        "possible_correction_rate_before_alpha": float(possible_mask.float().mean().item()) if possible_mask.numel() else math.nan,
        "lost_by_alpha_rate": float(lost_mask.float().mean().item()) if lost_mask.numel() else math.nan,
    }
    row.update(_ball_quality_for_gated_nodes(gated_idx, grouping, qmix_by_node, clean, noisy, train_mask, num_classes))
    return {field: row.get(field, math.nan) for field in GATED_DIAGNOSTIC_FIELDS}


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), float(row["noise_rate"]))].append(row)
    summary = []
    for (dataset, noise_rate), group in sorted(groups.items()):
        out = {"dataset": dataset, "noise_rate": noise_rate, "n": len(group)}
        for field in GATED_DIAGNOSTIC_FIELDS:
            if field in {"dataset", "noise_rate", "method"}:
                continue
            values = []
            for row in group:
                value = row.get(field)
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(number):
                    continue
                values.append(number)
            out[f"{field}_mean"] = float(np.mean(values)) if values else math.nan
            out[f"{field}_std"] = float(np.std(np.array(values, dtype=float), ddof=0)) if values else math.nan
        summary.append(out)
    return summary
