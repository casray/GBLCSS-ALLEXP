from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@dataclass
class SoftTargetResult:
    targets: torch.Tensor
    weights: torch.Tensor
    alpha: torch.Tensor
    gate_mask: torch.Tensor
    ball_targets: torch.Tensor
    node_ball_targets: torch.Tensor
    node_ball_confidence: torch.Tensor
    suspicion: torch.Tensor
    diagnostics: dict


def _mask_to_numpy(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=np.int64)


def _safe_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _safe_auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def build_gb_soft_targets(
    noisy_label,
    probs,
    assignment,
    train_mask,
    n_classes,
    clean_label=None,
    alpha_max=0.5,
    gamma=0.5,
    eta_max=0.4,
    min_train_per_ball=2,
    full_train_per_ball=4,
    label_prior_smoothing=1.0,
    target_mode="mix",
    suspicion_mode="gb",
    suspicion_top_ratio=0.3,
    min_ball_conf=0.3,
    beta=0.5,
    min_weight=0.3,
):
    device = probs.device
    noisy_label = noisy_label.to(device)
    if clean_label is not None:
        clean_label = clean_label.to(device)

    assignment = np.asarray(assignment, dtype=np.int64)
    train_mask_np = _mask_to_numpy(train_mask)
    n_nodes = int(noisy_label.shape[0])
    if assignment.shape[0] != n_nodes:
        raise ValueError("assignment length must match number of labels")

    alpha_max = float(alpha_max)
    gamma = float(gamma)
    eta_max = float(eta_max)
    min_train_per_ball = int(min_train_per_ball)
    full_train_per_ball = max(int(full_train_per_ball), min_train_per_ball)
    label_prior_smoothing = float(label_prior_smoothing)
    target_mode = str(target_mode)
    suspicion_mode = str(suspicion_mode)
    if target_mode not in {"mix", "reweight"}:
        raise ValueError("target_mode must be 'mix' or 'reweight'")
    if suspicion_mode not in {"gb", "no_ball"}:
        raise ValueError("suspicion_mode must be 'gb' or 'no_ball'")
    suspicion_top_ratio = min(max(float(suspicion_top_ratio), 0.0), 1.0)
    min_ball_conf = float(min_ball_conf)
    beta = float(beta)
    min_weight = min(max(float(min_weight), 0.0), 1.0)

    targets = F.one_hot(noisy_label, num_classes=n_classes).float()
    weights = torch.ones(n_nodes, device=device)
    alpha = torch.zeros(n_nodes, device=device)
    node_ball_targets = torch.zeros(n_nodes, n_classes, device=device)
    node_ball_conf = torch.zeros(n_nodes, device=device)

    train_tensor = torch.as_tensor(train_mask_np, dtype=torch.long, device=device)
    train_labels = noisy_label[train_tensor]
    label_prior = torch.bincount(train_labels, minlength=n_classes).float()
    label_prior = (label_prior + 1e-12) / torch.clamp(label_prior.sum(), min=1.0)

    ball_ids = sorted(np.unique(assignment).tolist())
    ball_targets = torch.zeros(len(ball_ids), n_classes, device=device)
    ball_quality_values = []
    ball_coverage_values = []

    for out_ball_id, ball_id in enumerate(ball_ids):
        node_idx_np = np.where(assignment == ball_id)[0]
        train_idx_np = train_mask_np[assignment[train_mask_np] == ball_id]
        node_idx = torch.as_tensor(node_idx_np, dtype=torch.long, device=device)
        train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)

        pbar = probs[node_idx].mean(dim=0)
        if train_idx.numel() > 0:
            counts = torch.bincount(noisy_label[train_idx], minlength=n_classes).float()
            yhat = (counts + label_prior_smoothing * label_prior) / (
                counts.sum() + label_prior_smoothing
            )
        else:
            yhat = label_prior

        if train_idx.numel() < min_train_per_ball:
            coverage = 0.0
        else:
            coverage = min(1.0, train_idx.numel() / float(full_train_per_ball))
        eta = eta_max * coverage
        q_ball = (1.0 - eta) * pbar + eta * yhat
        q_ball = q_ball / torch.clamp(q_ball.sum(), min=1e-12)

        label_conf = yhat.max()
        pred_conf = pbar.max()
        quality = coverage * 0.5 * (label_conf + pred_conf)
        log_classes = max(float(np.log(max(n_classes, 2))), 1e-12)
        entropy = -(q_ball * torch.log(torch.clamp(q_ball, min=1e-12))).sum()
        entropy_conf = torch.clamp(1.0 - entropy / log_classes, min=0.0, max=1.0)
        ball_conf = coverage * entropy_conf * q_ball.max()

        ball_targets[out_ball_id] = q_ball
        node_ball_targets[node_idx] = q_ball
        node_ball_conf[node_idx] = ball_conf
        ball_quality_values.append(_safe_float(quality))
        ball_coverage_values.append(float(coverage))

        if target_mode != "mix" or train_idx.numel() == 0 or coverage <= 0.0:
            continue

        noisy_prob = q_ball[noisy_label[train_idx]]
        node_alpha = alpha_max * quality * (1.0 - noisy_prob)
        node_alpha = torch.clamp(node_alpha, min=0.0, max=alpha_max)
        alpha[train_idx] = node_alpha
        targets[train_idx] = (
            (1.0 - node_alpha).unsqueeze(1) * targets[train_idx]
            + node_alpha.unsqueeze(1) * q_ball.unsqueeze(0)
        )
        weights[train_idx] = torch.clamp(1.0 - gamma * node_alpha, min=1e-6)

    warmup_conf = probs.max(dim=1).values
    noisy_prob = probs[torch.arange(n_nodes, device=device), noisy_label]
    noisy_q = node_ball_targets[torch.arange(n_nodes, device=device), noisy_label]
    if target_mode == "reweight":
        if suspicion_mode == "no_ball":
            suspicion = 1.0 - noisy_prob
        else:
            suspicion = 1.0 - torch.sqrt(torch.clamp(noisy_prob * noisy_q, min=0.0))

        if train_tensor.numel() and suspicion_top_ratio > 0.0:
            train_suspicion = suspicion[train_tensor]
            k = int(np.ceil(train_suspicion.numel() * suspicion_top_ratio))
            k = min(max(k, 1), train_suspicion.numel())
            threshold = torch.topk(train_suspicion, k=k).values.min()
            top_suspicion = train_suspicion >= threshold
            if suspicion_mode == "no_ball":
                conf_gate = torch.ones_like(top_suspicion, dtype=torch.bool)
            else:
                conf_gate = node_ball_conf[train_tensor] >= min_ball_conf
            gated = top_suspicion & conf_gate

            denom = train_suspicion.max() - train_suspicion.min()
            if _safe_float(denom) > 1e-12:
                normalized = (train_suspicion - train_suspicion.min()) / denom
            else:
                normalized = torch.ones_like(train_suspicion)
            node_alpha = torch.clamp(beta * normalized, min=0.0, max=1.0 - min_weight)
            node_alpha = torch.where(gated, node_alpha, torch.zeros_like(node_alpha))
            alpha[train_tensor] = node_alpha
            weights[train_tensor] = torch.clamp(1.0 - node_alpha, min=min_weight, max=1.0)
    else:
        suspicion = 1.0 - warmup_conf * noisy_q

    train_alpha = alpha[train_tensor]
    target_argmax = targets.argmax(dim=1)
    ball_argmax = node_ball_targets.argmax(dim=1)
    gated_mask = train_alpha > 1e-8
    if target_mode == "mix":
        target_changed = gated_mask
    else:
        target_changed = torch.zeros_like(gated_mask, dtype=torch.bool)

    diagnostics = {
        "target_mode": target_mode,
        "suspicion_mode": suspicion_mode,
        "alpha_mean": _safe_float(train_alpha.mean()) if train_alpha.numel() else 0.0,
        "alpha_max": _safe_float(train_alpha.max()) if train_alpha.numel() else 0.0,
        "target_changed_ratio": _safe_float(target_changed.float().mean()) if train_alpha.numel() else 0.0,
        "target_argmax_changed_ratio": _safe_float(
            (target_argmax[train_tensor] != noisy_label[train_tensor]).float().mean()
        ) if train_tensor.numel() else 0.0,
        "effective_coverage": _safe_float(gated_mask.float().mean()) if train_alpha.numel() else 0.0,
        "gated_ratio": _safe_float(gated_mask.float().mean()) if train_alpha.numel() else 0.0,
        "mean_q": float(np.mean(ball_quality_values)) if ball_quality_values else 0.0,
        "mean_train_coverage": float(np.mean(ball_coverage_values)) if ball_coverage_values else 0.0,
        "weight_mean": _safe_float(weights[train_tensor].mean()) if train_tensor.numel() else 1.0,
        "weight_min": _safe_float(weights[train_tensor].min()) if train_tensor.numel() else 1.0,
    }

    if clean_label is not None and train_tensor.numel():
        noise_flag = noisy_label[train_tensor] != clean_label[train_tensor]
        clean_flag = ~noise_flag
        gated_noise = gated_mask & noise_flag
        gated_clean = gated_mask & clean_flag
        if gated_mask.any():
            diagnostics["gated_noisy_precision"] = _safe_float(gated_noise.float().sum() / gated_mask.float().sum())
        else:
            diagnostics["gated_noisy_precision"] = 0.0
        if noise_flag.any():
            diagnostics["gated_noisy_recall"] = _safe_float(gated_noise.float().sum() / noise_flag.float().sum())
        else:
            diagnostics["gated_noisy_recall"] = 0.0
        if clean_flag.any():
            diagnostics["gated_clean_ratio"] = _safe_float(gated_clean.float().sum() / clean_flag.float().sum())
        else:
            diagnostics["gated_clean_ratio"] = 0.0
        if noise_flag.any():
            diagnostics["ncr"] = _safe_float(
                (ball_argmax[train_tensor][noise_flag] == clean_label[train_tensor][noise_flag]).float().mean()
            )
        else:
            diagnostics["ncr"] = 0.0
        if clean_flag.any():
            diagnostics["cdr"] = _safe_float(
                (ball_argmax[train_tensor][clean_flag] != clean_label[train_tensor][clean_flag]).float().mean()
            )
        else:
            diagnostics["cdr"] = 0.0
        diagnostics["suspicion_auroc"] = _safe_auc(
            noise_flag.detach().cpu().numpy().astype(np.int64),
            suspicion[train_tensor].detach().cpu().numpy(),
        )
    else:
        diagnostics["gated_noisy_precision"] = float("nan")
        diagnostics["gated_noisy_recall"] = float("nan")
        diagnostics["gated_clean_ratio"] = float("nan")
        diagnostics["ncr"] = float("nan")
        diagnostics["cdr"] = float("nan")
        diagnostics["suspicion_auroc"] = float("nan")

    return SoftTargetResult(
        targets=targets,
        weights=weights,
        alpha=alpha,
        gate_mask=gated_mask.detach(),
        ball_targets=ball_targets,
        node_ball_targets=node_ball_targets,
        node_ball_confidence=node_ball_conf.detach(),
        suspicion=suspicion,
        diagnostics=diagnostics,
    )
