from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


def safe_rate(num: int | float, den: int | float) -> float:
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
    return float(np.std(np.asarray(values, dtype=float), ddof=0)) if values else math.nan


def class_confusion_matrix(
    clean_labels: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int,
) -> list[list[int]]:
    clean = clean_labels.detach().cpu().long()
    pred = predictions.detach().cpu().long()
    mask = mask.detach().cpu().bool()
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for idx in mask.nonzero(as_tuple=False).view(-1).tolist():
        row = int(clean[idx].item())
        col = int(pred[idx].item())
        if 0 <= row < num_classes and 0 <= col < num_classes:
            matrix[row][col] += 1
    return matrix


def node_profile_stats(values_by_name: dict[str, torch.Tensor], mask: torch.Tensor, *, prefix: str) -> dict:
    mask = mask.detach().cpu().bool()
    out = {f"{prefix}_count": int(mask.sum().item())}
    for name, values in values_by_name.items():
        selected = values.detach().cpu().float()[mask]
        out[f"{prefix}_{name}_mean"] = _mean(selected)
        out[f"{prefix}_{name}_std"] = _std(selected)
    return out


def per_class_target_diagnostics(
    clean_labels: torch.Tensor,
    noisy_labels: torch.Tensor,
    target_predictions: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    num_classes: int,
    gated_mask: torch.Tensor | None = None,
    teacher_pred: torch.Tensor | None = None,
    test_predictions: torch.Tensor | None = None,
    test_mask: torch.Tensor | None = None,
    values_by_name: dict[str, torch.Tensor] | None = None,
) -> list[dict]:
    clean = clean_labels.detach().cpu().long()
    noisy = noisy_labels.detach().cpu().long()
    target = target_predictions.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    gated = gated_mask.detach().cpu().bool() if gated_mask is not None else torch.zeros_like(train_mask)
    teacher = teacher_pred.detach().cpu().long() if teacher_pred is not None else None
    test_pred = test_predictions.detach().cpu().long() if test_predictions is not None else None
    test_mask = test_mask.detach().cpu().bool() if test_mask is not None else torch.zeros_like(train_mask)
    values_by_name = values_by_name or {}

    rows = []
    for class_id in range(num_classes):
        clean_class = clean == class_id
        noisy_train = train_mask & clean_class & (clean != noisy)
        clean_train = train_mask & clean_class & (clean == noisy)
        train_class = train_mask & clean_class
        target_changed = train_class & (target != noisy)
        corrected = noisy_train & (target == clean)
        damaged = clean_train & (target != clean)
        gated_noisy = gated & noisy_train
        gated_clean = gated & clean_train
        failed_gated = gated_noisy & (target != clean)
        test_class = test_mask & clean_class

        row = {
            "class_id": int(class_id),
            "train_class_count": int(train_class.sum().item()),
            "noisy_train_count": int(noisy_train.sum().item()),
            "clean_train_count": int(clean_train.sum().item()),
            "correction_by_class_ncr": safe_rate(int(corrected.sum().item()), int(noisy_train.sum().item())),
            "clean_damage_by_class_cdr": safe_rate(int(damaged.sum().item()), int(clean_train.sum().item())),
            "target_changed_by_clean_class": int(target_changed.sum().item()),
            "target_changed_by_noisy_class": int((train_mask & (noisy == class_id) & (target != noisy)).sum().item()),
            "gated_noisy_by_class": int(gated_noisy.sum().item()),
            "gated_clean_by_class": int(gated_clean.sum().item()),
            "test_class_count": int(test_class.sum().item()),
            "test_acc_by_class": (
                safe_rate(int(((test_pred == clean) & test_class).sum().item()), int(test_class.sum().item()))
                if test_pred is not None
                else math.nan
            ),
        }
        if teacher is not None:
            gated_class = gated & train_class
            row["teacher_clean_acc_by_class_on_gated_nodes"] = safe_rate(
                int(((teacher == clean) & gated_class).sum().item()),
                int(gated_class.sum().item()),
            )
        else:
            row["teacher_clean_acc_by_class_on_gated_nodes"] = math.nan
        row.update(node_profile_stats(values_by_name, corrected, prefix="corrected"))
        row.update(node_profile_stats(values_by_name, damaged, prefix="damaged"))
        row.update(node_profile_stats(values_by_name, failed_gated, prefix="failed_gated"))
        rows.append(row)
    return rows


def summarize_class_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    key_fields = ("dataset", "noise_rate", "method", "class_id")
    for row in rows:
        groups[tuple(row.get(field) for field in key_fields)].append(row)
    summary = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        out = dict(zip(key_fields, key))
        out["n"] = len(group)
        fields = sorted(set().union(*(row.keys() for row in group)) - set(key_fields) - {"seed"})
        for field in fields:
            values = []
            for row in group:
                value = row.get(field)
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(number):
                    values.append(number)
            out[f"{field}_mean"] = float(np.mean(values)) if values else math.nan
            out[f"{field}_std"] = float(np.std(np.asarray(values, dtype=float), ddof=0)) if values else math.nan
        summary.append(out)
    return summary
