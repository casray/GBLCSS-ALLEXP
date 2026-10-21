from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.ordinary_gb import (
    build_ordinary_granular_balls,
    build_purity_adaptive_granular_balls,
    build_random_size_matched_assignment,
)


@dataclass
class FrozenGraceTargetResult:
    corrected_label: torch.Tensor
    changed: torch.Tensor
    assignment: np.ndarray
    diagnostics: dict
    soft_target: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    pseudo_label: torch.Tensor | None = None


@dataclass
class BallExpansionDiagnosticResult:
    candidate_mask: torch.Tensor
    pseudo_label: torch.Tensor
    diagnostics: dict


@dataclass
class BallExpansionTargetResult:
    soft_target: torch.Tensor
    candidate_mask: torch.Tensor
    pseudo_label: torch.Tensor
    diagnostics: dict


def load_frozen_embeddings(path: str | Path, expected_nodes: int | None = None) -> torch.Tensor:
    embeddings = torch.load(Path(path), map_location="cpu")
    if isinstance(embeddings, dict):
        for key in ("H", "h", "embeddings", "embedding"):
            if key in embeddings:
                embeddings = embeddings[key]
                break
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("frozen embedding artifact must be a tensor or a dict containing a tensor")
    if embeddings.ndim != 2:
        raise ValueError("frozen embeddings must be a 2-D tensor")
    if expected_nodes is not None and int(embeddings.shape[0]) != int(expected_nodes):
        raise ValueError(f"embedding node count {embeddings.shape[0]} != expected {expected_nodes}")
    return embeddings.float()


def apply_ball_majority_correction(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
    min_majority_confidence: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    corrected = noisy_label.detach().clone()
    changed = torch.zeros_like(noisy_label, dtype=torch.bool, device=device)
    min_train_per_ball = int(min_train_per_ball)
    min_majority_confidence = float(min_majority_confidence)

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        majority_confidence = float(counts.max().item() / max(1, int(counts.sum().item())))
        if majority_confidence < min_majority_confidence:
            continue
        majority = int(torch.argmax(counts).item())
        old = corrected[ball_train_tensor].clone()
        corrected[ball_train_tensor] = majority
        changed[ball_train_tensor] = old != corrected[ball_train_tensor]
    return corrected, changed


def apply_ball_soft_targets(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "binary",
    target_mode: str = "majority",
    prediction_label: torch.Tensor | None = None,
    agreement_beta: float = 0.0,
    topology_alpha_scale: dict[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    soft_target = F.one_hot(noisy_label.long(), num_classes=int(n_classes)).float().to(device)
    alpha = torch.zeros_like(noisy_label, dtype=torch.float, device=device)
    pseudo_label = noisy_label.detach().clone()
    min_train_per_ball = int(min_train_per_ball)
    alpha_baseline = str(alpha_baseline)
    target_mode = str(target_mode)
    if alpha_baseline not in {"binary", "multiclass"}:
        raise ValueError("alpha_baseline must be 'binary' or 'multiclass'")
    if target_mode not in {"majority", "distribution"}:
        raise ValueError("target_mode must be 'majority' or 'distribution'")
    if prediction_label is not None:
        prediction_label = prediction_label.to(device).long()
        if prediction_label.ndim != 1 or int(prediction_label.shape[0]) != int(noisy_label.shape[0]):
            raise ValueError("prediction_label must be a 1-D tensor matching noisy_label length")
    agreement_beta = float(agreement_beta)

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        total = max(1, int(counts.sum().item()))
        margin = float(counts.max().item() / total)
        ball_alpha = _alpha_from_margin(margin, int(n_classes), alpha_baseline)
        majority = int(torch.argmax(counts).item())
        if ball_alpha <= 0.0:
            continue
        node_alpha = torch.full((ball_train_tensor.numel(),), ball_alpha, dtype=torch.float, device=device)
        if topology_alpha_scale is not None:
            node_alpha = torch.clamp(
                node_alpha * float(topology_alpha_scale.get(int(ball_id), 1.0)),
                min=0.0,
                max=1.0,
            )
        if prediction_label is not None and agreement_beta > 0.0:
            agreement = prediction_label[ball_train_tensor] == majority
            scale = torch.where(
                agreement,
                torch.full_like(node_alpha, 1.0 + agreement_beta),
                torch.full_like(node_alpha, max(0.0, 1.0 - agreement_beta)),
            )
            node_alpha = torch.clamp(node_alpha * scale, min=0.0, max=1.0)
        if target_mode == "distribution":
            ball_target = (counts.float() / float(total)).to(device)
            majority_target = ball_target.unsqueeze(0).expand(ball_train_tensor.numel(), -1)
        else:
            majority_target = F.one_hot(
                torch.full_like(noisy_label[ball_train_tensor].long(), majority),
                num_classes=int(n_classes),
            ).float().to(device)
        soft_target[ball_train_tensor] = (
            (1.0 - node_alpha.unsqueeze(1)) * soft_target[ball_train_tensor] + node_alpha.unsqueeze(1) * majority_target
        )
        alpha[ball_train_tensor] = node_alpha
        pseudo_label[ball_train_tensor] = majority
    return soft_target, alpha, pseudo_label


def build_topology_alpha_scale(
    assignment: np.ndarray,
    adj: torch.Tensor,
    floor: float = 0.5,
) -> tuple[dict[int, float], dict]:
    """Build per-ball alpha scales from graph-topology compactness.

    The scale is static and label-free. It only downweights the existing
    SoftDist alpha for balls that are weakly supported by graph edges.
    """
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1:
        raise ValueError("assignment must be a 1-D array")
    floor = float(floor)
    if floor < 0.0 or floor > 1.0:
        raise ValueError("floor must be in [0, 1]")

    if adj.is_sparse:
        sparse_adj = adj.coalesce()
        edge_index = sparse_adj.indices().detach().cpu().numpy()
    else:
        edge_index = torch.nonzero(adj.detach().cpu(), as_tuple=False).t().numpy()
    if edge_index.shape[0] != 2:
        raise ValueError("adj must provide 2-D edge indices")

    row = edge_index[0].astype(np.int64, copy=False)
    col = edge_index[1].astype(np.int64, copy=False)
    valid = (row >= 0) & (row < assignment.shape[0]) & (col >= 0) & (col < assignment.shape[0]) & (row != col)
    row = row[valid]
    col = col[valid]

    raw_conf: dict[int, float] = {}
    for ball_id in sorted(np.unique(assignment).tolist()):
        src_in_ball = assignment[row] == int(ball_id)
        if not np.any(src_in_ball):
            raw_conf[int(ball_id)] = 0.0
            continue
        internal = src_in_ball & (assignment[col] == int(ball_id))
        raw_conf[int(ball_id)] = float(internal.sum() / max(1, int(src_in_ball.sum())))

    values = np.asarray(list(raw_conf.values()), dtype=np.float64)
    if values.size == 0:
        return {}, {
            "topology_alpha": 1,
            "topology_alpha_floor": floor,
            "topology_conf_mean_raw": 0.0,
            "topology_conf_min_raw": 0.0,
            "topology_conf_max_raw": 0.0,
            "topology_scale_mean": 0.0,
        }

    vmin = float(values.min())
    vmax = float(values.max())
    denom = vmax - vmin
    scale_by_ball: dict[int, float] = {}
    for ball_id, raw in raw_conf.items():
        normalized = 1.0 if denom <= 1e-12 else (float(raw) - vmin) / denom
        scale_by_ball[int(ball_id)] = float(floor + (1.0 - floor) * normalized)

    scales = np.asarray(list(scale_by_ball.values()), dtype=np.float64)
    diagnostics = {
        "topology_alpha": 1,
        "topology_alpha_floor": floor,
        "topology_conf_mean_raw": float(values.mean()),
        "topology_conf_min_raw": vmin,
        "topology_conf_max_raw": vmax,
        "topology_scale_mean": float(scales.mean()),
        "topology_scale_min": float(scales.min()),
        "topology_scale_max": float(scales.max()),
    }
    return scale_by_ball, diagnostics


def diagnostics_for_correction(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    corrected_label: torch.Tensor,
    changed: torch.Tensor,
    train_mask,
) -> dict:
    train_idx = torch.as_tensor(_mask_to_index_array(train_mask), dtype=torch.long, device=noisy_label.device)
    if train_idx.numel() == 0:
        return {
            "changed_train_count": 0,
            "changed_train_ratio": 0.0,
            "ncr": 0.0,
            "cdr": 0.0,
            "noisy_train_count": 0,
            "clean_train_count": 0,
        }

    noisy_flag = noisy_label[train_idx] != clean_label[train_idx]
    clean_flag = ~noisy_flag
    changed_train = changed[train_idx]
    changed_noisy = changed_train & noisy_flag
    changed_clean = changed_train & clean_flag
    corrected_to_clean = corrected_label[train_idx] == clean_label[train_idx]
    damaged_clean = corrected_label[train_idx] != clean_label[train_idx]

    noisy_count = int(noisy_flag.sum().item())
    clean_count = int(clean_flag.sum().item())
    ncr = float((changed_noisy & corrected_to_clean).sum().item() / noisy_count) if noisy_count else 0.0
    cdr = float((changed_clean & damaged_clean).sum().item() / clean_count) if clean_count else 0.0
    return {
        "changed_train_count": int(changed_train.sum().item()),
        "changed_train_ratio": float(changed_train.float().mean().item()),
        "ncr": ncr,
        "cdr": cdr,
        "noisy_train_count": noisy_count,
        "clean_train_count": clean_count,
    }


def diagnostics_for_soft_targets(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    soft_target: torch.Tensor,
    alpha: torch.Tensor,
    pseudo_label: torch.Tensor,
    train_mask,
) -> dict:
    train_idx = torch.as_tensor(_mask_to_index_array(train_mask), dtype=torch.long, device=noisy_label.device)
    if train_idx.numel() == 0:
        return {
            "alpha_positive_train_count": 0,
            "alpha_positive_train_ratio": 0.0,
            "mean_alpha_train": 0.0,
            "soft_shift_train_mass": 0.0,
            "soft_pseudo_changed_train_count": 0,
            "soft_pseudo_ncr": 0.0,
            "soft_pseudo_cdr": 0.0,
            "soft_clean_mass_delta": 0.0,
        }

    train_noisy = noisy_label[train_idx].long()
    train_clean = clean_label[train_idx].long()
    train_alpha = alpha[train_idx]
    train_pseudo = pseudo_label[train_idx].long()
    alpha_positive = train_alpha > 0
    pseudo_changed = alpha_positive & (train_pseudo != train_noisy)
    noisy_flag = train_noisy != train_clean
    clean_flag = ~noisy_flag
    pseudo_to_clean = train_pseudo == train_clean
    pseudo_damages_clean = train_pseudo != train_clean
    noisy_count = int(noisy_flag.sum().item())
    clean_count = int(clean_flag.sum().item())

    clean_mass = soft_target[train_idx, train_clean]
    original_clean_mass = (train_noisy == train_clean).float()
    ncr = float((pseudo_changed & noisy_flag & pseudo_to_clean).sum().item() / noisy_count) if noisy_count else 0.0
    cdr = float((pseudo_changed & clean_flag & pseudo_damages_clean).sum().item() / clean_count) if clean_count else 0.0
    soft_shift_mass = float(train_alpha[pseudo_changed].sum().item())
    return {
        "alpha_positive_train_count": int(alpha_positive.sum().item()),
        "alpha_positive_train_ratio": float(alpha_positive.float().mean().item()),
        "mean_alpha_train": float(train_alpha.mean().item()),
        "soft_shift_train_mass": soft_shift_mass,
        "soft_pseudo_changed_train_count": int(pseudo_changed.sum().item()),
        "soft_pseudo_ncr": ncr,
        "soft_pseudo_cdr": cdr,
        "soft_clean_mass_delta": float((clean_mass - original_clean_mass).sum().item()),
    }


def diagnostics_for_early_arbitration(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    alpha: torch.Tensor,
    pseudo_label: torch.Tensor,
    train_mask,
    early_noisy_conf: torch.Tensor,
    tau_protect: float,
) -> dict:
    train_idx = torch.as_tensor(_mask_to_index_array(train_mask), dtype=torch.long, device=noisy_label.device)
    if train_idx.numel() == 0:
        return {
            "tau_protect": float(tau_protect),
            "baseline_changed_train_count": 0,
            "protected_count": 0,
            "protection_precision": 0.0,
            "baseline_ncr": 0.0,
            "protected_ncr": 0.0,
            "ncr_relative_loss": 0.0,
            "baseline_cdr": 0.0,
            "protected_cdr": 0.0,
            "cdr_relative_reduction": 0.0,
        }
    if early_noisy_conf.ndim != 1 or int(early_noisy_conf.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("early_noisy_conf must be a 1-D tensor matching noisy_label length")

    tau_protect = float(tau_protect)
    noisy_train = noisy_label[train_idx].long()
    clean_train = clean_label[train_idx].long()
    pseudo_train = pseudo_label[train_idx].long()
    alpha_train = alpha[train_idx].float()
    conf_train = early_noisy_conf.to(noisy_label.device).float()[train_idx]

    baseline_changed = (alpha_train > 0) & (pseudo_train != noisy_train)
    protected = baseline_changed & (conf_train >= tau_protect)
    effective_changed = baseline_changed & ~protected
    noisy_flag = noisy_train != clean_train
    clean_flag = ~noisy_flag
    baseline_to_clean = pseudo_train == clean_train
    protected_to_clean = pseudo_train == clean_train
    baseline_damages_clean = pseudo_train != clean_train
    protected_damages_clean = pseudo_train != clean_train

    noisy_count = int(noisy_flag.sum().item())
    clean_count = int(clean_flag.sum().item())
    baseline_ncr_count = int((baseline_changed & noisy_flag & baseline_to_clean).sum().item())
    protected_ncr_count = int((effective_changed & noisy_flag & protected_to_clean).sum().item())
    baseline_cdr_count = int((baseline_changed & clean_flag & baseline_damages_clean).sum().item())
    protected_cdr_count = int((effective_changed & clean_flag & protected_damages_clean).sum().item())

    baseline_ncr = float(baseline_ncr_count / noisy_count) if noisy_count else 0.0
    protected_ncr = float(protected_ncr_count / noisy_count) if noisy_count else 0.0
    baseline_cdr = float(baseline_cdr_count / clean_count) if clean_count else 0.0
    protected_cdr = float(protected_cdr_count / clean_count) if clean_count else 0.0
    protected_count = int(protected.sum().item())
    protection_precision = float((protected & clean_flag).sum().item() / protected_count) if protected_count else 0.0
    ncr_relative_loss = float((baseline_ncr - protected_ncr) / baseline_ncr) if baseline_ncr > 0 else 0.0
    cdr_relative_reduction = float((baseline_cdr - protected_cdr) / baseline_cdr) if baseline_cdr > 0 else 0.0

    return {
        "tau_protect": tau_protect,
        "baseline_changed_train_count": int(baseline_changed.sum().item()),
        "protected_count": protected_count,
        "protected_ratio_changed": float(protected_count / max(1, int(baseline_changed.sum().item()))),
        "protection_precision": protection_precision,
        "baseline_ncr": baseline_ncr,
        "protected_ncr": protected_ncr,
        "ncr_relative_loss": ncr_relative_loss,
        "baseline_cdr": baseline_cdr,
        "protected_cdr": protected_cdr,
        "cdr_relative_reduction": cdr_relative_reduction,
        "baseline_ncr_count": baseline_ncr_count,
        "protected_ncr_count": protected_ncr_count,
        "baseline_cdr_count": baseline_cdr_count,
        "protected_cdr_count": protected_cdr_count,
    }


def diagnose_ball_expansion_candidates(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 5,
    margin_threshold: float = 0.6,
) -> BallExpansionDiagnosticResult:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    is_train = np.zeros(assignment.shape[0], dtype=bool)
    is_train[train_idx] = True
    non_train_count = int((~is_train).sum())
    candidate_mask = torch.zeros_like(noisy_label, dtype=torch.bool, device=device)
    pseudo_label = noisy_label.detach().clone()

    noisy_np = noisy_label.detach().cpu().long().numpy()
    clean_np = clean_label.detach().cpu().long().numpy()
    min_train_per_ball = int(min_train_per_ball)
    margin_threshold = float(margin_threshold)
    n_classes = int(n_classes)

    eligible_balls = 0
    selected_balls = 0
    candidate_count = 0
    correct_count = 0
    margin_sum = 0.0
    train_vote_sum = 0

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id)
        ball_train = members[is_train[members]]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        counts = np.bincount(noisy_np[ball_train], minlength=n_classes)
        train_votes = int(counts.sum())
        margin = float(counts.max() / max(1, train_votes))
        if margin < margin_threshold:
            continue
        ball_candidates = members[~is_train[members]]
        if ball_candidates.size == 0:
            continue
        majority = int(np.argmax(counts))
        candidate_tensor = torch.as_tensor(ball_candidates, dtype=torch.long, device=device)
        candidate_mask[candidate_tensor] = True
        pseudo_label[candidate_tensor] = majority
        selected_balls += 1
        candidate_count += int(ball_candidates.size)
        correct_count += int(np.sum(clean_np[ball_candidates] == majority))
        margin_sum += margin
        train_vote_sum += train_votes

    precision = float(correct_count / candidate_count) if candidate_count else 0.0
    coverage_unlabeled = float(candidate_count / max(1, non_train_count))
    diagnostics = {
        "min_train_per_ball": min_train_per_ball,
        "margin_threshold": margin_threshold,
        "eligible_balls": int(eligible_balls),
        "selected_balls": int(selected_balls),
        "candidate_count": int(candidate_count),
        "candidate_correct_count": int(correct_count),
        "coverage_unlabeled": coverage_unlabeled,
        "coverage_total": float(candidate_count / max(1, int(noisy_label.shape[0]))),
        "precision": precision,
        "mean_selected_margin": float(margin_sum / max(1, selected_balls)),
        "mean_selected_train_votes": float(train_vote_sum / max(1, selected_balls)),
    }
    return BallExpansionDiagnosticResult(
        candidate_mask=candidate_mask.detach(),
        pseudo_label=pseudo_label.detach(),
        diagnostics=diagnostics,
    )


def build_ball_expansion_soft_targets(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 10,
    margin_threshold: float = 0.5,
) -> BallExpansionTargetResult:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    is_train = np.zeros(assignment.shape[0], dtype=bool)
    is_train[train_idx] = True
    non_train_count = int((~is_train).sum())
    soft_target = F.one_hot(noisy_label.long(), num_classes=int(n_classes)).float().to(device)
    candidate_mask = torch.zeros_like(noisy_label, dtype=torch.bool, device=device)
    pseudo_label = noisy_label.detach().clone()

    noisy_np = noisy_label.detach().cpu().long().numpy()
    min_train_per_ball = int(min_train_per_ball)
    margin_threshold = float(margin_threshold)
    n_classes = int(n_classes)

    eligible_balls = 0
    selected_balls = 0
    candidate_count = 0
    margin_sum = 0.0
    train_vote_sum = 0

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id)
        ball_train = members[is_train[members]]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        counts = np.bincount(noisy_np[ball_train], minlength=n_classes)
        train_votes = int(counts.sum())
        margin = float(counts.max() / max(1, train_votes))
        if margin < margin_threshold:
            continue
        ball_candidates = members[~is_train[members]]
        if ball_candidates.size == 0:
            continue
        ball_dist = torch.as_tensor(counts / max(1, train_votes), dtype=torch.float, device=device)
        majority = int(np.argmax(counts))
        candidate_tensor = torch.as_tensor(ball_candidates, dtype=torch.long, device=device)
        soft_target[candidate_tensor] = ball_dist
        candidate_mask[candidate_tensor] = True
        pseudo_label[candidate_tensor] = majority
        selected_balls += 1
        candidate_count += int(ball_candidates.size)
        margin_sum += margin
        train_vote_sum += train_votes

    diagnostics = {
        "expansion_min_train_per_ball": min_train_per_ball,
        "expansion_margin_threshold": margin_threshold,
        "expansion_eligible_balls": int(eligible_balls),
        "expansion_selected_balls": int(selected_balls),
        "expansion_candidate_count": int(candidate_count),
        "expansion_coverage_unlabeled": float(candidate_count / max(1, non_train_count)),
        "expansion_coverage_total": float(candidate_count / max(1, int(noisy_label.shape[0]))),
        "expansion_mean_selected_margin": float(margin_sum / max(1, selected_balls)),
        "expansion_mean_selected_train_votes": float(train_vote_sum / max(1, selected_balls)),
    }
    return BallExpansionTargetResult(
        soft_target=soft_target.detach(),
        candidate_mask=candidate_mask.detach(),
        pseudo_label=pseudo_label.detach(),
        diagnostics=diagnostics,
    )


def ball_consistency_loss(
    logits: torch.Tensor,
    assignment: np.ndarray,
    min_ball_size: int = 2,
) -> torch.Tensor:
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(logits.shape[0]):
        raise ValueError("assignment must be a 1-D array matching logits node count")
    probs = F.softmax(logits, dim=1)
    losses = []
    min_ball_size = int(min_ball_size)
    device = logits.device
    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id).astype(np.int64)
        if members.size < min_ball_size:
            continue
        member_tensor = torch.as_tensor(members, dtype=torch.long, device=device)
        ball_probs = probs[member_tensor]
        mean_prob = ball_probs.mean(dim=0, keepdim=True)
        losses.append(((ball_probs - mean_prob) ** 2).sum(dim=1).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def soft_target_cross_entropy(logits: torch.Tensor, soft_target: torch.Tensor, mask=None) -> torch.Tensor:
    if soft_target.ndim != 2 or soft_target.shape != logits.shape:
        raise ValueError("soft_target must be a 2-D tensor matching logits shape")
    if mask is not None:
        logits = logits[mask]
        soft_target = soft_target[mask]
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    log_prob = F.log_softmax(logits, dim=1)
    return -(soft_target * log_prob).sum(dim=1).mean()


def expansion_soft_target_loss(
    logits: torch.Tensor,
    soft_target: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
    lambda_u: float,
) -> torch.Tensor:
    lambda_u = float(lambda_u)
    if lambda_u <= 0.0 or soft_target is None or candidate_mask is None:
        return logits.new_tensor(0.0)
    if int(candidate_mask.sum().item()) == 0:
        return logits.new_tensor(0.0)
    return lambda_u * soft_target_cross_entropy(logits, soft_target, candidate_mask)


def build_frozen_grace_targets(
    embedding_path: str | Path,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    min_train_per_ball: int = 2,
    min_majority_confidence: float = 0.0,
    random_state: int = 0,
    random_control: bool = False,
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    assignment = gb_result.assignment
    if random_control:
        assignment = build_random_size_matched_assignment(assignment, random_state=random_state + 17)
    corrected, changed = apply_ball_majority_correction(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        min_majority_confidence=min_majority_confidence,
    )
    diagnostics = {
        **gb_result.summary,
        **_coverage_diagnostics(noisy_label, assignment, train_mask, n_classes, min_train_per_ball, min_majority_confidence),
        **diagnostics_for_correction(noisy_label, clean_label, corrected, changed, train_mask),
        "random_control": int(random_control),
    }
    return FrozenGraceTargetResult(
        corrected_label=corrected.detach(),
        changed=changed.detach(),
        assignment=assignment,
        diagnostics=diagnostics,
    )


def partition_purity_diagnostics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
) -> dict:
    """Purity diagnostics for any ball partition.

    noisy_purity uses noisy TRAIN labels (the only signal the method sees).
    true_purity_* use clean labels and are DIAGNOSTIC ONLY - they never enter
    training. Comparing the two detects noise-fitting: if noisy purity rises
    while true purity does not, the partition is carving noise structure.
    """
    assignment = np.asarray(assignment, dtype=np.int64)
    train_idx = _mask_to_index_array(train_mask)
    is_train = np.zeros(assignment.shape[0], dtype=bool)
    is_train[train_idx] = True
    noisy_np = noisy_label.detach().cpu().long().numpy()
    clean_np = clean_label.detach().cpu().long().numpy()

    noisy_purity_sum = 0.0
    true_train_purity_sum = 0.0
    eligible_train_total = 0
    true_all_purity_sum = 0.0
    all_member_total = 0
    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id)
        ball_train = members[is_train[members]]
        true_all_counts = np.bincount(clean_np[members], minlength=int(n_classes))
        true_all_purity_sum += float(true_all_counts.max())
        all_member_total += int(members.size)
        if ball_train.size < int(min_train_per_ball):
            continue
        noisy_counts = np.bincount(noisy_np[ball_train], minlength=int(n_classes))
        true_counts = np.bincount(clean_np[ball_train], minlength=int(n_classes))
        noisy_purity_sum += float(noisy_counts.max())
        true_train_purity_sum += float(true_counts.max())
        eligible_train_total += int(ball_train.size)

    eligible_train_total_safe = max(1, eligible_train_total)
    return {
        "noisy_train_purity_weighted": float(noisy_purity_sum / eligible_train_total_safe),
        "true_train_purity_weighted": float(true_train_purity_sum / eligible_train_total_safe),
        "true_all_purity_weighted": float(true_all_purity_sum / max(1, all_member_total)),
        "purity_eligible_train_total": int(eligible_train_total),
    }


def build_frozen_grace_soft_targets(
    embedding_path: str | Path,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "binary",
    target_mode: str = "majority",
    random_state: int = 0,
    random_control: bool = False,
    ball_mode: str = "geometric",
    purity_threshold: float = 0.6,
    min_train_per_child: int = 4,
    min_train_to_split: int = 16,
    adaptive_min_ball_size: int = 8,
    adaptive_max_balls: int = 128,
    min_purity_gain: float = 0.05,
    split_attempts: int = 1,
    adj: torch.Tensor | None = None,
    topology_alpha: bool = False,
    topology_alpha_floor: float = 0.5,
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    ball_mode = str(ball_mode)
    if ball_mode == "adaptive":
        gb_result = build_purity_adaptive_granular_balls(
            embeddings,
            noisy_label=noisy_label,
            train_mask=train_mask,
            n_classes=n_classes,
            purity_threshold=purity_threshold,
            min_train_per_child=min_train_per_child,
            min_train_to_split=min_train_to_split,
            min_ball_size=adaptive_min_ball_size,
            max_balls=adaptive_max_balls,
            min_purity_gain=min_purity_gain,
            split_attempts=split_attempts,
            random_state=random_state,
        )
    elif ball_mode == "geometric":
        gb_result = build_ordinary_granular_balls(
            embeddings,
            max_balls=max_balls,
            min_ball_size=min_ball_size,
            split_improvement=split_improvement,
            random_state=random_state,
        )
    else:
        raise ValueError("ball_mode must be 'geometric' or 'adaptive'")
    assignment = gb_result.assignment
    if random_control:
        assignment = build_random_size_matched_assignment(assignment, random_state=random_state + 17)
    topology_alpha_scale = None
    topology_diagnostics = {"topology_alpha": 0}
    if topology_alpha:
        if adj is None:
            raise ValueError("adj is required when topology_alpha=True")
        topology_alpha_scale, topology_diagnostics = build_topology_alpha_scale(
            assignment=assignment,
            adj=adj,
            floor=topology_alpha_floor,
        )
    soft_target, alpha, pseudo_label = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        topology_alpha_scale=topology_alpha_scale,
    )
    changed = (alpha > 0) & (pseudo_label != noisy_label)
    diagnostics = {
        **gb_result.summary,
        **_soft_coverage_diagnostics(noisy_label, assignment, train_mask, n_classes, min_train_per_ball, alpha_baseline),
        **diagnostics_for_soft_targets(noisy_label, clean_label, soft_target, alpha, pseudo_label, train_mask),
        **partition_purity_diagnostics(noisy_label, clean_label, assignment, train_mask, n_classes, min_train_per_ball),
        **topology_diagnostics,
        "ball_mode": ball_mode,
        "soft_alpha_baseline": str(alpha_baseline),
        "soft_target_mode": str(target_mode),
        "random_control": int(random_control),
    }
    return FrozenGraceTargetResult(
        corrected_label=pseudo_label.detach(),
        changed=changed.detach(),
        assignment=assignment,
        diagnostics=diagnostics,
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
    )


def _mask_to_index_array(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().numpy().astype(np.int64)
        return mask.detach().cpu().numpy().astype(np.int64)
    arr = np.asarray(mask)
    if arr.dtype == np.bool_:
        return np.flatnonzero(arr).astype(np.int64)
    return arr.astype(np.int64)


def _coverage_diagnostics(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int,
    min_majority_confidence: float,
) -> dict:
    train_idx = _mask_to_index_array(train_mask)
    eligible_balls = 0
    confidence_pass_balls = 0
    eligible_train_nodes = 0
    confidence_pass_train_nodes = 0
    device = noisy_label.device
    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        majority_confidence = float(counts.max().item() / max(1, int(counts.sum().item())))
        if majority_confidence >= min_majority_confidence:
            confidence_pass_balls += 1
            confidence_pass_train_nodes += int(ball_train.size)
    total_train = max(1, int(train_idx.size))
    return {
        "min_train_per_ball": int(min_train_per_ball),
        "min_majority_confidence": float(min_majority_confidence),
        "eligible_balls": int(eligible_balls),
        "confidence_pass_balls": int(confidence_pass_balls),
        "eligible_train_nodes": int(eligible_train_nodes),
        "confidence_pass_train_nodes": int(confidence_pass_train_nodes),
        "confidence_pass_train_ratio": float(confidence_pass_train_nodes / total_train),
    }


def _soft_coverage_diagnostics(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int,
    alpha_baseline: str = "binary",
) -> dict:
    train_idx = _mask_to_index_array(train_mask)
    eligible_balls = 0
    eligible_train_nodes = 0
    alpha_positive_balls = 0
    alpha_positive_train_nodes = 0
    device = noisy_label.device
    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        margin = float(counts.max().item() / max(1, int(counts.sum().item())))
        ball_alpha = _alpha_from_margin(margin, int(n_classes), str(alpha_baseline))
        if ball_alpha > 0.0:
            alpha_positive_balls += 1
            alpha_positive_train_nodes += int(ball_train.size)
    total_train = max(1, int(train_idx.size))
    return {
        "min_train_per_ball": int(min_train_per_ball),
        "eligible_balls": int(eligible_balls),
        "eligible_train_nodes": int(eligible_train_nodes),
        "alpha_positive_balls": int(alpha_positive_balls),
        "alpha_positive_train_nodes": int(alpha_positive_train_nodes),
        "alpha_positive_train_ratio_by_ball": float(alpha_positive_train_nodes / total_train),
    }


def _alpha_from_margin(margin: float, n_classes: int, alpha_baseline: str) -> float:
    if alpha_baseline == "multiclass":
        zero_point = 1.0 / max(1, int(n_classes))
    else:
        zero_point = 0.5
    if zero_point >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (float(margin) - zero_point) / (1.0 - zero_point)))
