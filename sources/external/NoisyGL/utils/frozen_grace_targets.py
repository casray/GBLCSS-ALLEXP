from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf, OAS
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from utils.ordinary_gb import (
    build_ordinary_granular_balls,
    build_purity_adaptive_granular_balls,
    build_random_size_matched_assignment,
    build_size_adaptive_ellipsoid_balls,
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


def prepare_ball_features(features: torch.Tensor) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if getattr(features, "is_sparse", False):
        features = features.to_dense()
    if features.ndim != 2:
        raise ValueError("ball features must be a 2-D tensor")
    return F.normalize(features.detach().float(), p=2, dim=1, eps=1e-12)


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
    target_beta: float = 0.0,
    prediction_label: torch.Tensor | None = None,
    agreement_beta: float = 0.0,
    leave_one_out: bool = False,
    tie_mode: str = "min_index",
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
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")
    if prediction_label is not None:
        prediction_label = prediction_label.to(device).long()
        if prediction_label.ndim != 1 or int(prediction_label.shape[0]) != int(noisy_label.shape[0]):
            raise ValueError("prediction_label must be a 1-D tensor matching noisy_label length")
    agreement_beta = float(agreement_beta)
    leave_one_out = bool(leave_one_out)
    tie_mode = str(tie_mode).lower()
    if tie_mode not in {"min_index", "alpha_zero", "distribution_only"}:
        raise ValueError(
            "tie_mode must be 'min_index', 'alpha_zero', or 'distribution_only'"
        )

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        required_support = min_train_per_ball + (1 if leave_one_out else 0)
        if ball_train.size < required_support:
            continue
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        if leave_one_out:
            for node in ball_train_tensor:
                node_counts = counts.clone()
                node_label = noisy_label[node].long()
                node_counts[node_label] -= 1
                total = int(node_counts.sum().item())
                margin = float(node_counts.max().item() / max(1, total))
                node_alpha = _alpha_from_margin(margin, int(n_classes), alpha_baseline)
                if node_alpha <= 0.0:
                    continue
                is_tie = int((node_counts == node_counts.max()).sum().item()) > 1
                if is_tie and tie_mode == "alpha_zero":
                    continue
                majority = int(torch.argmax(node_counts).item())
                if target_mode == "distribution":
                    ball_target = (node_counts.float() / float(total)).to(device)
                    majority_onehot = F.one_hot(
                        torch.tensor(majority, dtype=torch.long, device=device),
                        num_classes=int(n_classes),
                    ).float()
                    effective_target_beta = (
                        0.0
                        if is_tie and tie_mode == "distribution_only"
                        else target_beta
                    )
                    local_target = (
                        (1.0 - effective_target_beta) * ball_target
                        + effective_target_beta * majority_onehot
                    )
                elif is_tie and tie_mode == "distribution_only":
                    local_target = (node_counts.float() / float(total)).to(device)
                else:
                    local_target = F.one_hot(
                        torch.tensor(majority, dtype=torch.long, device=device),
                        num_classes=int(n_classes),
                    ).float()
                soft_target[node] = (
                    (1.0 - node_alpha) * soft_target[node]
                    + node_alpha * local_target
                )
                alpha[node] = node_alpha
                if not (is_tie and tie_mode == "distribution_only"):
                    pseudo_label[node] = majority
            continue
        total = max(1, int(counts.sum().item()))
        margin = float(counts.max().item() / total)
        ball_alpha = _alpha_from_margin(margin, int(n_classes), alpha_baseline)
        majority = int(torch.argmax(counts).item())
        if ball_alpha <= 0.0:
            continue
        is_tie = int((counts == counts.max()).sum().item()) > 1
        if is_tie and tie_mode == "alpha_zero":
            continue
        node_alpha = torch.full((ball_train_tensor.numel(),), ball_alpha, dtype=torch.float, device=device)
        if (
            prediction_label is not None
            and agreement_beta > 0.0
            and not (is_tie and tie_mode == "distribution_only")
        ):
            agreement = prediction_label[ball_train_tensor] == majority
            scale = torch.where(
                agreement,
                torch.full_like(node_alpha, 1.0 + agreement_beta),
                torch.full_like(node_alpha, max(0.0, 1.0 - agreement_beta)),
            )
            node_alpha = torch.clamp(node_alpha * scale, min=0.0, max=1.0)
        if target_mode == "distribution":
            ball_target = (counts.float() / float(total)).to(device)
            majority_onehot = F.one_hot(
                torch.full((1,), majority, dtype=torch.long, device=device),
                num_classes=int(n_classes),
            ).float().squeeze(0)
            effective_target_beta = (
                0.0 if is_tie and tie_mode == "distribution_only" else target_beta
            )
            blended_target = (
                (1.0 - effective_target_beta) * ball_target
                + effective_target_beta * majority_onehot
            )
            majority_target = blended_target.unsqueeze(0).expand(ball_train_tensor.numel(), -1)
        elif is_tie and tie_mode == "distribution_only":
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
        if not (is_tie and tie_mode == "distribution_only"):
            pseudo_label[ball_train_tensor] = majority
    return soft_target, alpha, pseudo_label


def _ball_assignment_hash(assignment: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(assignment, dtype="<i8"))
    shape = ",".join(str(size) for size in canonical.shape)
    header = (
        f"dtype={canonical.dtype.str};shape={shape};order=C\n"
    ).encode("ascii")
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


def _uniform_tie_diagnostics(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int,
    alpha_baseline: str,
    leave_one_out: bool,
    tie_mode: str,
) -> dict:
    assignment = np.asarray(assignment, dtype=np.int64)
    train_idx = _mask_to_index_array(train_mask)
    device = noisy_label.device
    tie_ball_count = 0
    tie_active_ball_count = 0
    tie_affected_node_count = 0
    tie_affected_alpha_sum = 0.0

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        required_support = int(min_train_per_ball) + (1 if leave_one_out else 0)
        if ball_train.size < required_support:
            continue
        ball_train_tensor = torch.as_tensor(
            ball_train, dtype=torch.long, device=device
        )
        counts = torch.bincount(
            noisy_label[ball_train_tensor].long(), minlength=int(n_classes)
        )
        if leave_one_out:
            ball_has_tie = False
            ball_has_active_tie = False
            for node in ball_train_tensor:
                node_counts = counts.clone()
                node_counts[noisy_label[node].long()] -= 1
                is_tie = int(
                    (node_counts == node_counts.max()).sum().item()
                ) > 1
                if not is_tie:
                    continue
                ball_has_tie = True
                total = max(1, int(node_counts.sum().item()))
                margin = float(node_counts.max().item() / total)
                node_alpha = _alpha_from_margin(
                    margin, int(n_classes), str(alpha_baseline)
                )
                if node_alpha <= 0.0:
                    continue
                ball_has_active_tie = True
                tie_affected_node_count += 1
                tie_affected_alpha_sum += node_alpha
            tie_ball_count += int(ball_has_tie)
            tie_active_ball_count += int(ball_has_active_tie)
            continue

        is_tie = int((counts == counts.max()).sum().item()) > 1
        if not is_tie:
            continue
        tie_ball_count += 1
        total = max(1, int(counts.sum().item()))
        margin = float(counts.max().item() / total)
        ball_alpha = _alpha_from_margin(
            margin, int(n_classes), str(alpha_baseline)
        )
        if ball_alpha <= 0.0:
            continue
        tie_active_ball_count += 1
        tie_affected_node_count += int(ball_train.size)
        tie_affected_alpha_sum += float(ball_alpha * int(ball_train.size))

    return {
        "tie_mode": str(tie_mode),
        "tie_ball_count": int(tie_ball_count),
        "tie_active_ball_count": int(tie_active_ball_count),
        "tie_affected_node_count": int(tie_affected_node_count),
        "tie_affected_alpha_sum": float(tie_affected_alpha_sum),
        "ball_assignment_hash": _ball_assignment_hash(assignment),
    }


def build_global_smoothing_targets(
    noisy_label: torch.Tensor,
    train_mask,
    n_classes: int,
    epsilon: float = 0.1,
    distribution: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Build label-smoothing controls without graph, embeddings, or clean labels."""
    device = noisy_label.device
    n_classes = int(n_classes)
    epsilon = float(epsilon)
    distribution = str(distribution).strip().lower()
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if distribution not in {"uniform", "prior"}:
        raise ValueError("distribution must be 'uniform' or 'prior'")

    train_idx_np = _mask_to_index_array(train_mask)
    train_idx = torch.as_tensor(train_idx_np, dtype=torch.long, device=device)
    soft_target = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    alpha = torch.zeros_like(noisy_label, dtype=torch.float, device=device)
    pseudo_label = noisy_label.detach().clone()

    if distribution == "uniform":
        reference = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float, device=device)
    elif train_idx.numel() == 0:
        reference = torch.full((n_classes,), 1.0 / n_classes, dtype=torch.float, device=device)
    else:
        counts = torch.bincount(noisy_label[train_idx].long(), minlength=n_classes).float()
        reference = counts / counts.sum().clamp_min(1.0)

    if train_idx.numel() > 0:
        soft_target[train_idx] = (
            (1.0 - epsilon) * soft_target[train_idx]
            + epsilon * reference.unsqueeze(0)
        )
        alpha[train_idx] = epsilon
        pseudo_label[train_idx] = soft_target[train_idx].argmax(dim=1)

    diagnostics = {
        "target_builder": f"{distribution}_smoothing",
        "smoothing_distribution": distribution,
        "smoothing_epsilon": epsilon,
        "smoothing_reference": reference.detach().cpu().tolist(),
        "alpha_positive_train_count": int((alpha[train_idx] > 0).sum().item()),
        "alpha_positive_train_ratio": float((alpha[train_idx] > 0).float().mean().item())
        if train_idx.numel() > 0
        else 0.0,
        "mean_alpha_train": float(alpha[train_idx].mean().item()) if train_idx.numel() > 0 else 0.0,
    }
    return soft_target, alpha, pseudo_label, diagnostics


def build_global_smoothing_target_result(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    epsilon: float = 0.1,
    distribution: str = "uniform",
) -> FrozenGraceTargetResult:
    soft_target, alpha, pseudo_label, diagnostics = build_global_smoothing_targets(
        noisy_label=noisy_label,
        train_mask=train_mask,
        n_classes=n_classes,
        epsilon=epsilon,
        distribution=distribution,
    )
    diagnostics.update(
        diagnostics_for_soft_targets(
            noisy_label,
            clean_label,
            soft_target,
            alpha,
            pseudo_label,
            train_mask,
        )
    )
    changed = (alpha > 0) & (pseudo_label != noisy_label)
    return FrozenGraceTargetResult(
        corrected_label=pseudo_label.detach(),
        changed=changed.detach(),
        assignment=np.full(int(noisy_label.shape[0]), -1, dtype=np.int64),
        diagnostics=diagnostics,
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
    )


def _reference_ball_assignment_for_matching(
    embeddings: torch.Tensor,
    train_mask,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    random_state: int = 0,
) -> tuple[np.ndarray, dict]:
    gb_result = build_ordinary_granular_balls(
        embeddings,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    assignment = np.asarray(gb_result.assignment, dtype=np.int64)
    train_idx = _mask_to_index_array(train_mask)
    train_counts = []
    for ball_id in sorted(np.unique(assignment).tolist()):
        train_counts.append(int(np.sum(assignment[train_idx] == ball_id)))
    nonzero_train_counts = [count for count in train_counts if count > 0]
    mean_train_per_ball = float(np.mean(nonzero_train_counts)) if nonzero_train_counts else 1.0
    diagnostics = {
        **gb_result.summary,
        "matched_reference_num_balls": int(len(train_counts)),
        "matched_mean_train_per_ball": mean_train_per_ball,
        "matched_zero_train_balls": int(sum(1 for count in train_counts if count == 0)),
    }
    return assignment, diagnostics


def _targets_from_node_distributions(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_idx: np.ndarray,
    distributions: np.ndarray,
    n_classes: int,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.5,
    tie_mode: str = "min_index",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = noisy_label.device
    n_classes = int(n_classes)
    distributions = np.asarray(distributions, dtype=np.float64)
    if distributions.ndim != 2 or int(distributions.shape[1]) != n_classes:
        raise ValueError("distributions must be a 2-D array with n_classes columns")
    if int(distributions.shape[0]) != int(train_idx.size):
        raise ValueError("distributions row count must match train_idx size")
    if alpha_baseline not in {"binary", "multiclass"}:
        raise ValueError("alpha_baseline must be 'binary' or 'multiclass'")
    if target_mode not in {"majority", "distribution"}:
        raise ValueError("target_mode must be 'majority' or 'distribution'")
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")
    tie_mode = str(tie_mode).lower()
    if tie_mode not in {"min_index", "alpha_zero", "distribution_only"}:
        raise ValueError(
            "tie_mode must be 'min_index', 'alpha_zero', or 'distribution_only'"
        )

    row_sum = np.clip(distributions.sum(axis=1, keepdims=True), 1e-12, None)
    distributions = distributions / row_sum
    majority = distributions.argmax(axis=1).astype(np.int64)
    margin = distributions.max(axis=1)
    is_tie = np.isclose(distributions, margin[:, None]).sum(axis=1) > 1
    alpha_np = np.asarray(
        [_alpha_from_margin(float(value), n_classes, alpha_baseline) for value in margin],
        dtype=np.float64,
    )
    if tie_mode == "alpha_zero":
        alpha_np[is_tie] = 0.0
    majority_onehot = np.eye(n_classes, dtype=np.float64)[majority]
    if target_mode == "distribution":
        target_np = (1.0 - target_beta) * distributions + target_beta * majority_onehot
    else:
        target_np = majority_onehot
    if tie_mode == "distribution_only":
        target_np[is_tie] = distributions[is_tie]

    soft_target = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    alpha = torch.zeros_like(noisy_label, dtype=torch.float, device=device)
    pseudo_label = noisy_label.detach().clone()
    if train_idx.size:
        train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        alpha_tensor = torch.as_tensor(alpha_np, dtype=torch.float, device=device)
        target_tensor = torch.as_tensor(target_np, dtype=torch.float, device=device)
        soft_target[train_tensor] = (
            (1.0 - alpha_tensor.unsqueeze(1)) * soft_target[train_tensor]
            + alpha_tensor.unsqueeze(1) * target_tensor
        )
        alpha[train_tensor] = alpha_tensor
        update_pseudo = np.ones(train_idx.size, dtype=bool)
        if tie_mode == "distribution_only":
            update_pseudo[is_tie] = False
        if np.any(update_pseudo):
            update_tensor = torch.as_tensor(
                train_idx[update_pseudo], dtype=torch.long, device=device
            )
            pseudo_label[update_tensor] = torch.as_tensor(
                majority[update_pseudo], dtype=torch.long, device=device
            )

    diagnostics = diagnostics_for_soft_targets(
        noisy_label=noisy_label,
        clean_label=clean_label,
        soft_target=soft_target,
        alpha=alpha,
        pseudo_label=pseudo_label,
        train_mask=train_idx,
    )
    diagnostics.update({
        "soft_alpha_baseline": str(alpha_baseline),
        "soft_target_mode": str(target_mode),
        "target_beta": float(target_beta),
        "tie_mode": str(tie_mode),
        "tie_affected_node_count": int(is_tie.sum()),
    })
    return soft_target.detach(), alpha.detach(), pseudo_label.detach(), diagnostics


def build_label_propagation_soft_targets(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    adj: torch.Tensor,
    n_classes: int,
    num_steps: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.5,
) -> FrozenGraceTargetResult:
    if not getattr(adj, "is_sparse", False):
        adj = adj.to_sparse()
    adj = adj.coalesce().to(noisy_label.device)
    indices = adj.indices()
    values = adj.values().float()
    row_sum = torch.sparse.sum(adj, dim=1).to_dense().float().clamp_min(1e-12)
    norm_values = values / row_sum[indices[0]]
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, adj.shape, device=noisy_label.device).coalesce()

    train_idx = _mask_to_index_array(train_mask)
    n_classes = int(n_classes)
    label_matrix = torch.zeros((int(noisy_label.shape[0]), n_classes), device=noisy_label.device)
    if train_idx.size:
        train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=noisy_label.device)
        label_matrix[train_tensor] = F.one_hot(noisy_label[train_tensor].long(), num_classes=n_classes).float()

    propagated = label_matrix
    for _ in range(max(0, int(num_steps))):
        propagated = torch.sparse.mm(norm_adj, propagated)

    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=noisy_label.device)
    train_distributions = propagated[train_tensor] if train_idx.size else torch.empty((0, n_classes), device=noisy_label.device)
    row_mass = train_distributions.sum(dim=1) if train_idx.size else torch.empty(0, device=noisy_label.device)
    covered = row_mass > 1e-12
    distributions_np = train_distributions.detach().cpu().numpy()
    if train_idx.size:
        noisy_np = noisy_label.detach().cpu().long().numpy()
        for row_id, node_id in enumerate(train_idx):
            if not bool(covered[row_id].detach().cpu()):
                distributions_np[row_id, :] = 0.0
                distributions_np[row_id, int(noisy_np[node_id])] = 1.0

    soft_target, alpha, pseudo_label, diagnostics = _targets_from_node_distributions(
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_idx=train_idx,
        distributions=distributions_np,
        n_classes=n_classes,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        target_beta=target_beta,
    )
    diagnostics.update(
        {
            "target_source": "label_propagation",
            "lp_steps": int(num_steps),
            "lp_train_coverage": float(covered.float().mean().item()) if train_idx.size else 0.0,
            "lp_mean_mass": float(row_mass.mean().item()) if train_idx.size else 0.0,
        }
    )
    assignment = np.arange(int(noisy_label.shape[0]), dtype=np.int64)
    return FrozenGraceTargetResult(
        corrected_label=pseudo_label.detach(),
        changed=pseudo_label.detach().ne(noisy_label.detach()),
        assignment=assignment,
        diagnostics=diagnostics,
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
    )


def build_knn_soft_targets_from_features(
    ball_features: torch.Tensor,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.5,
    random_state: int = 0,
    matched_k: int | None = None,
    tie_mode: str = "min_index",
    exclude_self: bool = False,
) -> FrozenGraceTargetResult:
    embeddings = prepare_ball_features(ball_features).cpu()
    if int(embeddings.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("ball feature node count must match noisy_label length")
    reference_assignment, match_diag = _reference_ball_assignment_for_matching(
        embeddings,
        train_mask=train_mask,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    train_idx = _mask_to_index_array(train_mask)
    n_train = int(train_idx.size)
    n_classes = int(n_classes)
    if n_train == 0:
        distributions = np.zeros((0, n_classes), dtype=np.float64)
        k_value = 0
    else:
        k_value = int(matched_k) if matched_k is not None else int(round(match_diag["matched_mean_train_per_ball"]))
        max_neighbors = n_train - 1 if bool(exclude_self) else n_train
        k_value = max(0, min(k_value, max_neighbors))
        train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
        train_features = embeddings[train_tensor]
        similarity = train_features @ train_features.T
        noisy_np = noisy_label.detach().cpu().long().numpy()
        distributions = np.zeros((n_train, n_classes), dtype=np.float64)
        if k_value == 0:
            distributions[np.arange(n_train), noisy_np[train_idx]] = 1.0
        else:
            if bool(exclude_self):
                similarity.fill_diagonal_(float("-inf"))
            topk = torch.topk(
                similarity, k=k_value, dim=1, largest=True
            ).indices.cpu().numpy()
            for row_id, neighbor_pos in enumerate(topk):
                labels = noisy_np[train_idx[neighbor_pos]]
                counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
                distributions[row_id] = counts / max(1.0, float(counts.sum()))
    soft_target, alpha, pseudo_label, target_diag = _targets_from_node_distributions(
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_idx=train_idx,
        distributions=distributions,
        n_classes=n_classes,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        target_beta=target_beta,
        tie_mode=tie_mode,
    )
    diagnostics = {
        **match_diag,
        **target_diag,
        "target_builder": "knn",
        "matched_k": int(k_value),
        "matched_num_balls": int(match_diag["matched_reference_num_balls"]),
        "ball_feature_source": "memory",
        "knn_exclude_self": int(bool(exclude_self)),
    }
    changed = (alpha > 0) & (pseudo_label != noisy_label)
    return FrozenGraceTargetResult(
        corrected_label=pseudo_label.detach(),
        changed=changed.detach(),
        assignment=reference_assignment,
        diagnostics=diagnostics,
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
    )


def build_kmeans_soft_targets_from_features(
    ball_features: torch.Tensor,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.5,
    random_state: int = 0,
    matched_kmeans_clusters: int | None = None,
    tie_mode: str = "min_index",
    leave_one_out: bool = False,
) -> FrozenGraceTargetResult:
    embeddings = prepare_ball_features(ball_features).cpu()
    if int(embeddings.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("ball feature node count must match noisy_label length")
    _, match_diag = _reference_ball_assignment_for_matching(
        embeddings,
        train_mask=train_mask,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    n_nodes = int(embeddings.shape[0])
    n_clusters = (
        int(matched_kmeans_clusters)
        if matched_kmeans_clusters is not None
        else int(match_diag["matched_reference_num_balls"])
    )
    n_clusters = max(1, min(n_clusters, n_nodes))
    features_np = embeddings.numpy()
    if n_clusters == 1:
        assignment = np.zeros(n_nodes, dtype=np.int64)
    else:
        assignment = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10).fit_predict(features_np)
    soft_target, alpha, pseudo_label = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        target_beta=target_beta,
        tie_mode=tie_mode,
        leave_one_out=leave_one_out,
    )
    changed = (alpha > 0) & (pseudo_label != noisy_label)
    diagnostics = {
        **_assignment_summary(assignment),
        **_soft_coverage_diagnostics(noisy_label, assignment, train_mask, n_classes, min_train_per_ball, alpha_baseline),
        **diagnostics_for_soft_targets(noisy_label, clean_label, soft_target, alpha, pseudo_label, train_mask),
        **partition_purity_diagnostics(noisy_label, clean_label, assignment, train_mask, n_classes, min_train_per_ball),
        **match_diag,
        "target_builder": "kmeans",
        "matched_num_balls": int(match_diag["matched_reference_num_balls"]),
        "kmeans_n_clusters": int(n_clusters),
        "soft_alpha_baseline": str(alpha_baseline),
        "soft_target_mode": str(target_mode),
        "target_beta": float(target_beta),
        "tie_mode": str(tie_mode),
        "leave_one_out": int(bool(leave_one_out)),
        "ball_feature_source": "memory",
    }
    return FrozenGraceTargetResult(
        corrected_label=pseudo_label.detach(),
        changed=changed.detach(),
        assignment=assignment.astype(np.int64),
        diagnostics=diagnostics,
        soft_target=soft_target.detach(),
        alpha=alpha.detach(),
        pseudo_label=pseudo_label.detach(),
    )


def build_frozen_grace_knn_soft_targets(
    embedding_path: str | Path,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    **kwargs,
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    result = build_knn_soft_targets_from_features(
        ball_features=embeddings,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=train_mask,
        n_classes=n_classes,
        **kwargs,
    )
    result.diagnostics["ball_feature_source"] = "grace"
    return result


def build_frozen_grace_kmeans_soft_targets(
    embedding_path: str | Path,
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    train_mask,
    n_classes: int,
    **kwargs,
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    result = build_kmeans_soft_targets_from_features(
        ball_features=embeddings,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=train_mask,
        n_classes=n_classes,
        **kwargs,
    )
    result.diagnostics["ball_feature_source"] = "grace"
    return result


def apply_ball_ellipsoid_soft_targets(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    ball_features: torch.Tensor,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "binary",
    target_mode: str = "majority",
    target_beta: float = 0.0,
    ellipsoid_ridge_scale: float = 0.05,
    ellipsoid_weight_floor: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")
    if not isinstance(ball_features, torch.Tensor):
        ball_features = torch.as_tensor(ball_features)
    if getattr(ball_features, "is_sparse", False):
        ball_features = ball_features.to_dense()
    if ball_features.ndim != 2 or int(ball_features.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("ball_features must be a 2-D tensor matching noisy_label length")

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
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")

    eligible_balls = 0
    eligible_train_nodes = 0
    fallback_balls = 0
    alpha_positive_balls = 0
    alpha_positive_train_nodes = 0
    train_weights: list[torch.Tensor] = []
    effective_support_values: list[float] = []
    effective_support_ratios: list[float] = []

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id).astype(np.int64)
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        weights_cpu, used_fallback = _ellipsoid_train_weights_for_ball(
            ball_features=ball_features,
            members=members,
            ball_train=ball_train,
            ridge_scale=float(ellipsoid_ridge_scale),
            weight_floor=float(ellipsoid_weight_floor),
        )
        if used_fallback:
            fallback_balls += 1
        train_weights.append(weights_cpu.detach().float().cpu())
        weight_sum = float(weights_cpu.sum().item())
        weight_square_sum = float((weights_cpu * weights_cpu).sum().item())
        effective_support = (weight_sum * weight_sum / weight_square_sum) if weight_square_sum > 0 else 0.0
        effective_support_values.append(float(effective_support))
        effective_support_ratios.append(float(effective_support / max(1, int(ball_train.size))))

        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        weights = weights_cpu.to(device=device, dtype=torch.float)
        one_hot = F.one_hot(noisy_label[ball_train_tensor].long(), num_classes=int(n_classes)).float().to(device)
        counts = (one_hot * weights.unsqueeze(1)).sum(dim=0)
        total = float(counts.sum().item())
        if total <= 0.0 or not torch.isfinite(counts).all():
            counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes)).float().to(device)
            total = float(counts.sum().item())
            fallback_balls += int(not used_fallback)
        margin = float(counts.max().item() / max(total, 1e-12))
        ball_alpha = _alpha_from_margin(margin, int(n_classes), alpha_baseline)
        majority = int(torch.argmax(counts).item())
        if ball_alpha <= 0.0:
            continue
        alpha_positive_balls += 1
        alpha_positive_train_nodes += int(ball_train.size)
        node_alpha = torch.full((ball_train_tensor.numel(),), ball_alpha, dtype=torch.float, device=device)
        if target_mode == "distribution":
            ball_target = (counts / max(total, 1e-12)).to(device)
            majority_onehot = F.one_hot(
                torch.full((1,), majority, dtype=torch.long, device=device),
                num_classes=int(n_classes),
            ).float().squeeze(0)
            blended_target = (1.0 - target_beta) * ball_target + target_beta * majority_onehot
            majority_target = blended_target.unsqueeze(0).expand(ball_train_tensor.numel(), -1)
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

    if train_weights:
        all_weights = torch.cat(train_weights)
        mean_weight = float(all_weights.mean().item())
        median_weight = float(all_weights.median().item())
        min_weight = float(all_weights.min().item())
    else:
        mean_weight = median_weight = min_weight = 0.0
    total_train = max(1, int(train_idx.size))
    diagnostics = {
        "min_train_per_ball": int(min_train_per_ball),
        "eligible_balls": int(eligible_balls),
        "eligible_train_nodes": int(eligible_train_nodes),
        "alpha_positive_balls": int(alpha_positive_balls),
        "alpha_positive_train_nodes": int(alpha_positive_train_nodes),
        "alpha_positive_train_ratio_by_ball": float(alpha_positive_train_nodes / total_train),
        "softdist_weighting": "ellipsoid",
        "target_beta": float(target_beta),
        "ellipsoid_ridge_scale": float(ellipsoid_ridge_scale),
        "ellipsoid_weight_floor": float(ellipsoid_weight_floor),
        "ellipsoid_eligible_balls": int(eligible_balls),
        "ellipsoid_fallback_balls": int(fallback_balls),
        "ellipsoid_fallback_ball_ratio": float(fallback_balls / max(1, eligible_balls)),
        "ellipsoid_weighted_train_nodes": int(eligible_train_nodes),
        "ellipsoid_mean_train_weight": mean_weight,
        "ellipsoid_median_train_weight": median_weight,
        "ellipsoid_min_train_weight": min_weight,
        "ellipsoid_effective_support_mean": (
            float(np.mean(effective_support_values)) if effective_support_values else 0.0
        ),
        "ellipsoid_effective_support_ratio_mean": (
            float(np.mean(effective_support_ratios)) if effective_support_ratios else 0.0
        ),
    }
    return soft_target, alpha, pseudo_label, diagnostics


def apply_ball_ellipsoid_pca_shrinkage_soft_targets(
    noisy_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    ball_features: torch.Tensor,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "binary",
    target_mode: str = "majority",
    target_beta: float = 0.0,
    ellipsoid_feature_space: str = "fullH",
    ellipsoid_covariance_estimator: str = "ridge",
    ellipsoid_ridge_scale: float = 0.05,
    ellipsoid_weight_floor: float = 0.1,
    ellipsoid_distance_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    device = noisy_label.device
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")
    if not isinstance(ball_features, torch.Tensor):
        ball_features = torch.as_tensor(ball_features)
    if getattr(ball_features, "is_sparse", False):
        ball_features = ball_features.to_dense()
    if ball_features.ndim != 2 or int(ball_features.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("ball_features must be a 2-D tensor matching noisy_label length")

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
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")

    eligible_balls = 0
    eligible_train_nodes = 0
    fallback_balls = 0
    covariance_failure_balls = 0
    alpha_positive_balls = 0
    alpha_positive_train_nodes = 0
    train_weights: list[torch.Tensor] = []
    effective_support_values: list[float] = []
    effective_support_ratios: list[float] = []
    effective_dims: list[int] = []

    for ball_id in sorted(np.unique(assignment).tolist()):
        members = np.flatnonzero(assignment == ball_id).astype(np.int64)
        ball_train = train_idx[assignment[train_idx] == ball_id]
        if ball_train.size < min_train_per_ball:
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        weights_cpu, status = _ellipsoid_pca_shrinkage_train_weights_for_ball(
            ball_features=ball_features,
            members=members,
            ball_train=ball_train,
            feature_space=str(ellipsoid_feature_space),
            covariance_estimator=str(ellipsoid_covariance_estimator),
            ridge_scale=float(ellipsoid_ridge_scale),
            weight_floor=float(ellipsoid_weight_floor),
            distance_temperature=float(ellipsoid_distance_temperature),
        )
        if status["used_fallback"]:
            fallback_balls += 1
        if status["covariance_failed"]:
            covariance_failure_balls += 1
        effective_dims.append(int(status["effective_dim"]))
        train_weights.append(weights_cpu.detach().float().cpu())
        weight_sum = float(weights_cpu.sum().item())
        weight_square_sum = float((weights_cpu * weights_cpu).sum().item())
        effective_support = (weight_sum * weight_sum / weight_square_sum) if weight_square_sum > 0 else 0.0
        effective_support_values.append(float(effective_support))
        effective_support_ratios.append(float(effective_support / max(1, int(ball_train.size))))

        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        weights = weights_cpu.to(device=device, dtype=torch.float)
        one_hot = F.one_hot(noisy_label[ball_train_tensor].long(), num_classes=int(n_classes)).float().to(device)
        counts = (one_hot * weights.unsqueeze(1)).sum(dim=0)
        total = float(counts.sum().item())
        if total <= 0.0 or not torch.isfinite(counts).all():
            counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes)).float().to(device)
            total = float(counts.sum().item())
            if not status["used_fallback"]:
                fallback_balls += 1
        margin = float(counts.max().item() / max(total, 1e-12))
        ball_alpha = _alpha_from_margin(margin, int(n_classes), alpha_baseline)
        majority = int(torch.argmax(counts).item())
        if ball_alpha <= 0.0:
            continue
        alpha_positive_balls += 1
        alpha_positive_train_nodes += int(ball_train.size)
        node_alpha = torch.full((ball_train_tensor.numel(),), ball_alpha, dtype=torch.float, device=device)
        if target_mode == "distribution":
            ball_target = (counts / max(total, 1e-12)).to(device)
            majority_onehot = F.one_hot(
                torch.full((1,), majority, dtype=torch.long, device=device),
                num_classes=int(n_classes),
            ).float().squeeze(0)
            blended_target = (1.0 - target_beta) * ball_target + target_beta * majority_onehot
            majority_target = blended_target.unsqueeze(0).expand(ball_train_tensor.numel(), -1)
        else:
            majority_target = F.one_hot(
                torch.full_like(noisy_label[ball_train_tensor].long(), majority),
                num_classes=int(n_classes),
            ).float().to(device)
        soft_target[ball_train_tensor] = (
            (1.0 - node_alpha.unsqueeze(1)) * soft_target[ball_train_tensor]
            + node_alpha.unsqueeze(1) * majority_target
        )
        alpha[ball_train_tensor] = node_alpha
        pseudo_label[ball_train_tensor] = majority

    if train_weights:
        all_weights = torch.cat(train_weights)
        mean_weight = float(all_weights.mean().item())
        median_weight = float(all_weights.median().item())
        min_weight = float(all_weights.min().item())
    else:
        mean_weight = median_weight = min_weight = 0.0
    total_train = max(1, int(train_idx.size))
    diagnostics = {
        "min_train_per_ball": int(min_train_per_ball),
        "eligible_balls": int(eligible_balls),
        "eligible_train_nodes": int(eligible_train_nodes),
        "alpha_positive_balls": int(alpha_positive_balls),
        "alpha_positive_train_nodes": int(alpha_positive_train_nodes),
        "alpha_positive_train_ratio_by_ball": float(alpha_positive_train_nodes / total_train),
        "softdist_weighting": "ellipsoid_pca_shrinkage",
        "target_beta": float(target_beta),
        "ellipsoid_requested_feature_space": str(ellipsoid_feature_space),
        "ellipsoid_covariance_estimator": str(ellipsoid_covariance_estimator),
        "ellipsoid_ridge_scale": float(ellipsoid_ridge_scale),
        "ellipsoid_weight_floor": float(ellipsoid_weight_floor),
        "ellipsoid_distance_temperature": float(ellipsoid_distance_temperature),
        "ellipsoid_eligible_balls": int(eligible_balls),
        "ellipsoid_fallback_balls": int(fallback_balls),
        "ellipsoid_fallback_ball_ratio": float(fallback_balls / max(1, eligible_balls)),
        "ellipsoid_covariance_failure_balls": int(covariance_failure_balls),
        "ellipsoid_covariance_failure_ratio": float(covariance_failure_balls / max(1, eligible_balls)),
        "ellipsoid_weighted_train_nodes": int(eligible_train_nodes),
        "ellipsoid_mean_train_weight": mean_weight,
        "ellipsoid_median_train_weight": median_weight,
        "ellipsoid_min_train_weight": min_weight,
        "ellipsoid_effective_support_mean": (
            float(np.mean(effective_support_values)) if effective_support_values else 0.0
        ),
        "ellipsoid_effective_support_ratio_mean": (
            float(np.mean(effective_support_ratios)) if effective_support_ratios else 0.0
        ),
        "ellipsoid_pca_effective_dim_mean": float(np.mean(effective_dims)) if effective_dims else 0.0,
        "ellipsoid_pca_effective_dim_max": int(max(effective_dims)) if effective_dims else 0,
    }
    return soft_target, alpha, pseudo_label, diagnostics


def shuffle_responsibilities_for_control(
    responsibilities: np.ndarray,
    random_state: int = 0,
) -> np.ndarray:
    responsibilities = np.asarray(responsibilities, dtype=np.float64)
    if responsibilities.ndim != 2:
        raise ValueError("responsibilities must be a 2-D array")
    rng = np.random.default_rng(int(random_state) + 17)
    perm = rng.permutation(int(responsibilities.shape[0]))
    if perm.size > 1 and np.array_equal(perm, np.arange(perm.size)):
        perm = np.roll(perm, 1)
    return responsibilities[perm].copy()


def apply_gmm_resp_soft_targets_from_responsibilities(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    responsibilities: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.0,
    gmm_prior: str = "jeffreys",
    gmm_geometry_confidence: str = "max_resp",
    random_state: int = 0,
    random_control: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, np.ndarray]:
    device = noisy_label.device
    responsibilities = _normalize_responsibilities(responsibilities, int(noisy_label.shape[0]))
    if random_control:
        responsibilities = shuffle_responsibilities_for_control(responsibilities, random_state=random_state)
    train_idx = _mask_to_index_array(train_mask)
    n_classes = int(n_classes)
    if alpha_baseline not in {"binary", "multiclass"}:
        raise ValueError("alpha_baseline must be 'binary' or 'multiclass'")
    target_mode = str(target_mode)
    if target_mode not in {"majority", "distribution"}:
        raise ValueError("target_mode must be 'majority' or 'distribution'")
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")
    if str(gmm_prior) != "jeffreys":
        raise ValueError("gmm_prior must be 'jeffreys'")
    if str(gmm_geometry_confidence) not in {"max_resp", "none"}:
        raise ValueError("gmm_geometry_confidence must be 'max_resp' or 'none'")

    noisy_np = noisy_label.detach().cpu().long().numpy()
    n_components = int(responsibilities.shape[1])
    counts = np.zeros((n_components, n_classes), dtype=np.float64)
    if train_idx.size:
        train_resp = responsibilities[train_idx]
        train_labels = noisy_np[train_idx]
        for class_id in range(n_classes):
            counts[:, class_id] = train_resp[train_labels == class_id].sum(axis=0)
    support = counts.sum(axis=1)
    prior = 0.5
    component_dist = (counts + prior) / (support[:, None] + prior * n_classes)
    node_dist = responsibilities @ component_dist
    node_dist = np.clip(node_dist, 0.0, 1.0)
    node_dist = node_dist / np.clip(node_dist.sum(axis=1, keepdims=True), 1e-12, None)

    margin = node_dist.max(axis=1)
    base_alpha = np.asarray(
        [_alpha_from_margin(float(value), n_classes, str(alpha_baseline)) for value in margin],
        dtype=np.float64,
    )
    if str(gmm_geometry_confidence) == "max_resp":
        geometry_conf = responsibilities.max(axis=1)
    else:
        geometry_conf = np.ones(int(responsibilities.shape[0]), dtype=np.float64)
    node_alpha = np.clip(base_alpha * geometry_conf, 0.0, 1.0)
    node_pseudo = node_dist.argmax(axis=1).astype(np.int64)
    majority_onehot = np.eye(n_classes, dtype=np.float64)[node_pseudo]
    if target_mode == "distribution":
        node_target = (1.0 - target_beta) * node_dist + target_beta * majority_onehot
    else:
        node_target = majority_onehot

    soft_target = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    alpha = torch.zeros_like(noisy_label, dtype=torch.float, device=device)
    pseudo_label = noisy_label.detach().clone()
    if train_idx.size:
        train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        node_alpha_tensor = torch.as_tensor(node_alpha[train_idx], dtype=torch.float, device=device)
        node_target_tensor = torch.as_tensor(node_target[train_idx], dtype=torch.float, device=device)
        soft_target[train_tensor] = (
            (1.0 - node_alpha_tensor.unsqueeze(1)) * soft_target[train_tensor]
            + node_alpha_tensor.unsqueeze(1) * node_target_tensor
        )
        alpha[train_tensor] = node_alpha_tensor
        pseudo_label[train_tensor] = torch.as_tensor(node_pseudo[train_idx], dtype=torch.long, device=device)

    assignment = responsibilities.argmax(axis=1).astype(np.int64)
    alpha_positive = alpha[torch.as_tensor(train_idx, dtype=torch.long, device=device)] > 0 if train_idx.size else torch.tensor([], dtype=torch.bool, device=device)
    component_alpha = np.asarray(
        [_alpha_from_margin(float(row.max()), n_classes, str(alpha_baseline)) for row in component_dist],
        dtype=np.float64,
    )
    low_support = support < int(min_train_per_ball)
    max_resp = responsibilities.max(axis=1)
    entropy = -(responsibilities * np.log(np.clip(responsibilities, 1e-12, 1.0))).sum(axis=1)
    dominant_support = support[assignment] if support.size else np.zeros(int(noisy_label.shape[0]), dtype=np.float64)
    diagnostics = {
        "min_train_per_ball": int(min_train_per_ball),
        "eligible_balls": int(np.sum(~low_support)),
        "eligible_train_nodes": int(np.sum(dominant_support[train_idx] >= int(min_train_per_ball))) if train_idx.size else 0,
        "alpha_positive_balls": int(np.sum((~low_support) & (component_alpha > 0.0))),
        "alpha_positive_train_nodes": int(alpha_positive.sum().item()) if train_idx.size else 0,
        "alpha_positive_train_ratio_by_ball": (
            float(alpha_positive.float().mean().item()) if train_idx.size else 0.0
        ),
        "softdist_weighting": "gmm_resp",
        "target_beta": float(target_beta),
        "gmm_fallback": 0,
        "gmm_prior": str(gmm_prior),
        "gmm_geometry_confidence": str(gmm_geometry_confidence),
        "gmm_n_components": int(n_components),
        "gmm_component_support_min": float(np.min(support)) if support.size else 0.0,
        "gmm_component_support_median": float(np.median(support)) if support.size else 0.0,
        "gmm_component_support_mean": float(np.mean(support)) if support.size else 0.0,
        "gmm_low_support_component_ratio": float(np.mean(low_support)) if low_support.size else 0.0,
        "gmm_max_resp_mean": float(np.mean(max_resp)) if max_resp.size else 0.0,
        "gmm_resp_entropy_mean": float(np.mean(entropy)) if entropy.size else 0.0,
        "gmm_noise_auc": _gmm_noise_auc(
            noisy_label=noisy_label,
            clean_label=clean_label,
            node_dist=node_dist,
            train_idx=train_idx,
        ),
        **_gmm_support_bin_diagnostics(
            noisy_label=noisy_label,
            clean_label=clean_label,
            soft_target=soft_target,
            alpha=alpha,
            pseudo_label=pseudo_label,
            train_idx=train_idx,
            dominant_support=dominant_support,
        ),
    }
    return soft_target, alpha, pseudo_label, diagnostics, assignment


def apply_gmm_resp_soft_targets(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    ball_features: torch.Tensor,
    train_mask,
    n_classes: int,
    fallback_assignment: np.ndarray | None = None,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.0,
    gmm_n_components: int | None = None,
    gmm_covariance_type: str = "diag",
    gmm_reg_covar: float = 1e-5,
    gmm_max_iter: int = 100,
    gmm_n_init: int = 1,
    gmm_prior: str = "jeffreys",
    gmm_geometry_confidence: str = "max_resp",
    random_state: int = 0,
    random_control: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, np.ndarray]:
    fit_error = ""
    try:
        features = prepare_ball_features(ball_features).detach().cpu().numpy()
        if features.ndim != 2 or features.shape[0] != int(noisy_label.shape[0]):
            raise ValueError("ball_features must be a 2-D tensor matching noisy_label length")
        if not np.isfinite(features).all():
            raise ValueError("ball_features contain NaN or Inf")
        n_components = int(gmm_n_components) if gmm_n_components is not None else min(64, int(features.shape[0]))
        n_components = max(1, min(n_components, int(features.shape[0])))
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=str(gmm_covariance_type),
            reg_covar=float(gmm_reg_covar),
            max_iter=int(gmm_max_iter),
            n_init=int(gmm_n_init),
            random_state=int(random_state),
        )
        gmm.fit(features)
        responsibilities = gmm.predict_proba(features)
        soft_target, alpha, pseudo_label, diagnostics, assignment = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy_label,
            clean_label=clean_label,
            responsibilities=responsibilities,
            train_mask=train_mask,
            n_classes=n_classes,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            target_beta=target_beta,
            gmm_prior=gmm_prior,
            gmm_geometry_confidence=gmm_geometry_confidence,
            random_state=random_state,
            random_control=random_control,
        )
        diagnostics.update({
            "gmm_covariance_type": str(gmm_covariance_type),
            "gmm_reg_covar": float(gmm_reg_covar),
            "gmm_max_iter": int(gmm_max_iter),
            "gmm_n_init": int(gmm_n_init),
            "gmm_converged": int(bool(getattr(gmm, "converged_", False))),
            "gmm_n_iter": int(getattr(gmm, "n_iter_", 0)),
        })
        return soft_target, alpha, pseudo_label, diagnostics, assignment
    except Exception as exc:
        fit_error = f"{type(exc).__name__}: {exc}"
    fallback = (
        np.asarray(fallback_assignment, dtype=np.int64)
        if fallback_assignment is not None
        else np.zeros(int(noisy_label.shape[0]), dtype=np.int64)
    )
    soft_target, alpha, pseudo_label = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=fallback,
        train_mask=train_mask,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=alpha_baseline,
        target_mode="distribution",
        target_beta=target_beta,
    )
    diagnostics = {
        **_soft_coverage_diagnostics(noisy_label, fallback, train_mask, n_classes, min_train_per_ball, alpha_baseline),
        "softdist_weighting": "gmm_resp",
        "target_beta": float(target_beta),
        "gmm_fallback": 1,
        "gmm_fit_error": fit_error,
        "gmm_prior": str(gmm_prior),
        "gmm_geometry_confidence": str(gmm_geometry_confidence),
        "gmm_n_components": int(gmm_n_components) if gmm_n_components is not None else 0,
        "gmm_covariance_type": str(gmm_covariance_type),
        "gmm_reg_covar": float(gmm_reg_covar),
        "gmm_max_iter": int(gmm_max_iter),
        "gmm_n_init": int(gmm_n_init),
        "gmm_converged": 0,
        "gmm_n_iter": 0,
        "gmm_component_support_min": 0.0,
        "gmm_component_support_median": 0.0,
        "gmm_component_support_mean": 0.0,
        "gmm_low_support_component_ratio": 1.0,
        "gmm_max_resp_mean": 0.0,
        "gmm_resp_entropy_mean": 0.0,
        "gmm_noise_auc": 0.5,
    }
    return soft_target, alpha, pseudo_label, diagnostics, fallback


def _ellipsoid_train_weights_for_ball(
    ball_features: torch.Tensor,
    members: np.ndarray,
    ball_train: np.ndarray,
    ridge_scale: float = 0.05,
    weight_floor: float = 0.1,
) -> tuple[torch.Tensor, bool]:
    members = np.asarray(members, dtype=np.int64)
    ball_train = np.asarray(ball_train, dtype=np.int64)
    if members.size < 2 or ball_train.size == 0:
        return torch.ones(int(ball_train.size), dtype=torch.float), True

    features = ball_features.detach().float().cpu()
    if not torch.isfinite(features[torch.as_tensor(members, dtype=torch.long)]).all():
        return torch.ones(int(ball_train.size), dtype=torch.float), True

    member_pos = {int(node_id): pos for pos, node_id in enumerate(members.tolist())}
    try:
        train_pos = torch.as_tensor([member_pos[int(node_id)] for node_id in ball_train.tolist()], dtype=torch.long)
    except KeyError:
        return torch.ones(int(ball_train.size), dtype=torch.float), True

    member_tensor = torch.as_tensor(members, dtype=torch.long)
    x = features[member_tensor]
    centered = x - x.mean(dim=0, keepdim=True)
    dim = int(centered.shape[1])
    cov = centered.T.matmul(centered) / float(max(1, int(members.size) - 1))
    trace = torch.trace(cov)
    if dim <= 0 or not torch.isfinite(trace) or float(trace.item()) <= 0.0:
        return torch.ones(int(ball_train.size), dtype=torch.float), True
    ridge = float(ridge_scale) * float(trace.item()) / float(dim) + 1e-6
    try:
        inv_cov = torch.linalg.pinv(cov + ridge * torch.eye(dim, dtype=cov.dtype))
        projected = centered.matmul(inv_cov)
        d2_all = (projected * centered).sum(dim=1).clamp_min(0.0)
    except RuntimeError:
        return torch.ones(int(ball_train.size), dtype=torch.float), True
    if not torch.isfinite(d2_all).all():
        return torch.ones(int(ball_train.size), dtype=torch.float), True
    median_d2 = torch.median(d2_all)
    if not torch.isfinite(median_d2) or float(median_d2.item()) <= 0.0:
        return torch.ones(int(ball_train.size), dtype=torch.float), True
    d2_norm = d2_all[train_pos] / median_d2.clamp_min(1e-12)
    floor = max(0.0, min(1.0, float(weight_floor)))
    weights = floor + (1.0 - floor) * torch.exp(-0.5 * d2_norm)
    if not torch.isfinite(weights).all():
        return torch.ones(int(ball_train.size), dtype=torch.float), True
    return weights.float(), False


def _parse_ellipsoid_pca_dim(feature_space: str, raw_dim: int) -> int | None:
    feature_space = str(feature_space).strip()
    if feature_space.lower() == "fullh":
        return None
    if not feature_space.lower().startswith("pca"):
        raise ValueError("ellipsoid_feature_space must be fullH or pca{dim}")
    try:
        requested = int(feature_space[3:])
    except ValueError as exc:
        raise ValueError("ellipsoid_feature_space must be fullH or pca{dim}") from exc
    return max(1, min(int(requested), int(raw_dim)))


def _reduce_ellipsoid_features(
    centered: torch.Tensor,
    requested_dim: int | None,
) -> tuple[torch.Tensor, int]:
    n_samples = int(centered.shape[0])
    raw_dim = int(centered.shape[1])
    if requested_dim is None:
        return centered, raw_dim
    effective_dim = max(1, min(int(requested_dim), raw_dim, max(1, n_samples - 1)))
    if raw_dim <= effective_dim:
        return centered, raw_dim
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        return centered[:, :effective_dim].contiguous(), effective_dim
    components = vh[:effective_dim].T
    reduced = centered.matmul(components)
    return reduced.contiguous(), effective_dim


def _shrinkage_covariance_numpy(
    x_np: np.ndarray,
    covariance_estimator: str,
    ridge_scale: float,
) -> tuple[np.ndarray | None, bool]:
    covariance_estimator = str(covariance_estimator).lower()
    if covariance_estimator not in {"ridge", "oas", "ledoitwolf"}:
        raise ValueError("ellipsoid_covariance_estimator must be ridge, oas, or ledoitwolf")
    try:
        if covariance_estimator == "oas":
            cov = OAS().fit(x_np).covariance_
        elif covariance_estimator == "ledoitwolf":
            cov = LedoitWolf().fit(x_np).covariance_
        else:
            cov = np.cov(x_np, rowvar=False, bias=False)
            if cov.ndim == 0:
                cov = np.asarray([[float(cov)]], dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            return None, True
        if not np.isfinite(cov).all():
            return None, True
        trace = float(np.trace(cov))
        dim = int(cov.shape[0])
        if dim <= 0 or trace <= 0.0 or not np.isfinite(trace):
            return None, True
        ridge = float(ridge_scale) * trace / float(dim) + 1e-6
        return cov + ridge * np.eye(dim, dtype=np.float64), False
    except Exception:
        return None, True


def _ellipsoid_pca_shrinkage_train_weights_for_ball(
    ball_features: torch.Tensor,
    members: np.ndarray,
    ball_train: np.ndarray,
    feature_space: str = "fullH",
    covariance_estimator: str = "ridge",
    ridge_scale: float = 0.05,
    weight_floor: float = 0.1,
    distance_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    members = np.asarray(members, dtype=np.int64)
    ball_train = np.asarray(ball_train, dtype=np.int64)
    status = {
        "used_fallback": False,
        "covariance_failed": False,
        "effective_dim": 0,
    }
    if members.size < 2 or ball_train.size == 0:
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status

    features = ball_features.detach().float().cpu()
    member_tensor = torch.as_tensor(members, dtype=torch.long)
    if not torch.isfinite(features[member_tensor]).all():
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status

    member_pos = {int(node_id): pos for pos, node_id in enumerate(members.tolist())}
    try:
        train_pos = torch.as_tensor([member_pos[int(node_id)] for node_id in ball_train.tolist()], dtype=torch.long)
    except KeyError:
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status

    x = features[member_tensor]
    centered = x - x.mean(dim=0, keepdim=True)
    if int(centered.shape[1]) <= 0:
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    requested_dim = _parse_ellipsoid_pca_dim(feature_space, int(centered.shape[1]))
    reduced, effective_dim = _reduce_ellipsoid_features(centered, requested_dim)
    status["effective_dim"] = int(effective_dim)
    if int(reduced.shape[1]) <= 0 or not torch.isfinite(reduced).all():
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status

    x_np = reduced.detach().cpu().numpy().astype(np.float64, copy=False)
    cov, covariance_failed = _shrinkage_covariance_numpy(
        x_np=x_np,
        covariance_estimator=covariance_estimator,
        ridge_scale=float(ridge_scale),
    )
    status["covariance_failed"] = bool(covariance_failed)
    if cov is None:
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    try:
        inv_cov = np.linalg.pinv(cov)
        d2_all = np.einsum("ij,jk,ik->i", x_np, inv_cov, x_np)
    except Exception:
        status["used_fallback"] = True
        status["covariance_failed"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    if not np.isfinite(d2_all).all():
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    d2_all = np.clip(d2_all, 0.0, None)
    median_d2 = float(np.median(d2_all))
    if not np.isfinite(median_d2) or median_d2 <= 0.0:
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    temperature = max(1e-6, float(distance_temperature))
    d2_norm = d2_all[train_pos.numpy()] / max(median_d2, 1e-12)
    floor = max(0.0, min(1.0, float(weight_floor)))
    weights_np = floor + (1.0 - floor) * np.exp(-0.5 * d2_norm / temperature)
    if not np.isfinite(weights_np).all():
        status["used_fallback"] = True
        return torch.ones(int(ball_train.size), dtype=torch.float), status
    return torch.as_tensor(weights_np, dtype=torch.float), status


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
            "soft_clean_mass_delta_mean": 0.0,
            "soft_target_brier": 0.0,
            "noisy_onehot_brier": 0.0,
            "soft_brier_gain": 0.0,
            "soft_target_argmax_accuracy": 0.0,
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
    clean_onehot = F.one_hot(train_clean, num_classes=int(soft_target.shape[1])).float()
    noisy_onehot = F.one_hot(train_noisy, num_classes=int(soft_target.shape[1])).float()
    target_brier = ((soft_target[train_idx] - clean_onehot) ** 2).sum(dim=1).mean()
    noisy_brier = ((noisy_onehot - clean_onehot) ** 2).sum(dim=1).mean()
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
        "soft_clean_mass_delta_mean": float((clean_mass - original_clean_mass).mean().item()),
        "soft_target_brier": float(target_brier.item()),
        "noisy_onehot_brier": float(noisy_brier.item()),
        "soft_brier_gain": float((noisy_brier - target_brier).item()),
        "soft_target_argmax_accuracy": float((soft_target[train_idx].argmax(dim=1) == train_clean).float().mean().item()),
    }


def support_bin_diagnostics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
) -> list[dict]:
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    soft_target, alpha, pseudo_label = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_idx,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline="multiclass",
        target_mode="distribution",
    )
    device = noisy_label.device
    bin_order = ("2-3", "4-7", "8+")
    totals = {
        name: {
            "support_bin": name,
            "ball_count": 0,
            "train_count": 0,
            "changed_train_count": 0,
            "noisy_train_count": 0,
            "clean_train_count": 0,
            "ncr_count": 0,
            "cdr_count": 0,
            "alpha_sum": 0.0,
            "clean_mass_delta_total": 0.0,
            "changed_clean_mass_delta_total": 0.0,
            "majority_clean_correct_count": 0,
            "margin_sum": 0.0,
        }
        for name in bin_order
    }

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        support = int(ball_train.size)
        if support < int(min_train_per_ball):
            continue
        support_bin = "2-3" if support <= 3 else "4-7" if support <= 7 else "8+"
        nodes = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[nodes].long(), minlength=int(n_classes))
        majority = int(torch.argmax(counts).item())
        margin = float(counts.max().item() / max(1, int(counts.sum().item())))

        noisy_nodes = noisy_label[nodes].long()
        clean_nodes = clean_label[nodes].long()
        pseudo_nodes = pseudo_label[nodes].long()
        alpha_nodes = alpha[nodes]
        changed = (alpha_nodes > 0) & (pseudo_nodes != noisy_nodes)
        noisy_flag = noisy_nodes != clean_nodes
        clean_flag = ~noisy_flag
        clean_mass_delta = (
            soft_target[nodes, clean_nodes] - (noisy_nodes == clean_nodes).float()
        )

        row = totals[support_bin]
        row["ball_count"] += 1
        row["train_count"] += support
        row["changed_train_count"] += int(changed.sum().item())
        row["noisy_train_count"] += int(noisy_flag.sum().item())
        row["clean_train_count"] += int(clean_flag.sum().item())
        row["ncr_count"] += int(
            (changed & noisy_flag & (pseudo_nodes == clean_nodes)).sum().item()
        )
        row["cdr_count"] += int(
            (changed & clean_flag & (pseudo_nodes != clean_nodes)).sum().item()
        )
        row["alpha_sum"] += float(alpha_nodes.sum().item())
        row["clean_mass_delta_total"] += float(clean_mass_delta.sum().item())
        row["changed_clean_mass_delta_total"] += float(
            clean_mass_delta[changed].sum().item()
        )
        row["majority_clean_correct_count"] += int(
            (clean_nodes == majority).sum().item()
        )
        row["margin_sum"] += margin

    rows = []
    for support_bin in bin_order:
        total = totals[support_bin]
        if total["ball_count"] == 0:
            continue
        train_count = int(total["train_count"])
        changed_count = int(total["changed_train_count"])
        noisy_count = int(total["noisy_train_count"])
        clean_count = int(total["clean_train_count"])
        rows.append(
            {
                "support_bin": support_bin,
                "ball_count": int(total["ball_count"]),
                "train_count": train_count,
                "changed_train_count": changed_count,
                "noisy_train_count": noisy_count,
                "clean_train_count": clean_count,
                "ncr_count": int(total["ncr_count"]),
                "cdr_count": int(total["cdr_count"]),
                "ncr": float(total["ncr_count"] / noisy_count) if noisy_count else 0.0,
                "cdr": float(total["cdr_count"] / clean_count) if clean_count else 0.0,
                "mean_alpha": float(total["alpha_sum"] / train_count) if train_count else 0.0,
                "clean_mass_delta_total": float(total["clean_mass_delta_total"]),
                "clean_mass_delta_per_node": (
                    float(total["clean_mass_delta_total"] / train_count) if train_count else 0.0
                ),
                "changed_clean_mass_delta_per_node": (
                    float(total["changed_clean_mass_delta_total"] / changed_count)
                    if changed_count
                    else 0.0
                ),
                "majority_clean_accuracy": (
                    float(total["majority_clean_correct_count"] / train_count)
                    if train_count
                    else 0.0
                ),
                "mean_observed_margin": float(total["margin_sum"] / total["ball_count"]),
            }
        )
    return rows


def vote_support_diagnostics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    assignment: np.ndarray,
    train_mask,
    n_classes: int,
    min_train_per_ball: int = 2,
    alpha_baseline: str = "multiclass",
    target_mode: str = "distribution",
    target_beta: float = 0.5,
    tie_mode: str = "distribution_only",
) -> list[dict]:
    """Summarize target quality by the number of noisy training votes per ball."""
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.shape[0] != int(noisy_label.shape[0]):
        raise ValueError("assignment must be a 1-D array matching noisy_label length")

    train_idx = _mask_to_index_array(train_mask)
    soft_target, alpha, pseudo_label = apply_ball_soft_targets(
        noisy_label=noisy_label,
        assignment=assignment,
        train_mask=train_idx,
        n_classes=n_classes,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        target_beta=target_beta,
        tie_mode=tie_mode,
    )
    device = noisy_label.device
    bin_order = ("2", "3-4", "5-9", "10-19", "20+")
    totals = {
        name: {
            "support_bin": name,
            "ball_count": 0,
            "target_node_count": 0,
            "changed_train_count": 0,
            "noisy_train_count": 0,
            "clean_train_count": 0,
            "ncr_count": 0,
            "cdr_count": 0,
            "alpha_sum": 0.0,
            "clean_mass_delta_total": 0.0,
            "majority_clean_correct_count": 0,
            "margin_sum": 0.0,
            "target_argmax_correct_count": 0,
        }
        for name in bin_order
    }

    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        support = int(ball_train.size)
        if support < int(min_train_per_ball):
            continue
        if support == 2:
            support_bin = "2"
        elif support <= 4:
            support_bin = "3-4"
        elif support <= 9:
            support_bin = "5-9"
        elif support <= 19:
            support_bin = "10-19"
        else:
            support_bin = "20+"

        nodes = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(
            noisy_label[nodes].long(), minlength=int(n_classes)
        )
        majority = int(torch.argmax(counts).item())
        margin = float(counts.max().item() / max(1, int(counts.sum().item())))
        noisy_nodes = noisy_label[nodes].long()
        clean_nodes = clean_label[nodes].long()
        pseudo_nodes = pseudo_label[nodes].long()
        alpha_nodes = alpha[nodes]
        changed = (alpha_nodes > 0) & (pseudo_nodes != noisy_nodes)
        noisy_flag = noisy_nodes != clean_nodes
        clean_flag = ~noisy_flag
        clean_mass_delta = (
            soft_target[nodes, clean_nodes] - clean_flag.float()
        )

        row = totals[support_bin]
        row["ball_count"] += 1
        row["target_node_count"] += support
        row["changed_train_count"] += int(changed.sum().item())
        row["noisy_train_count"] += int(noisy_flag.sum().item())
        row["clean_train_count"] += int(clean_flag.sum().item())
        row["ncr_count"] += int(
            (changed & noisy_flag & (pseudo_nodes == clean_nodes)).sum().item()
        )
        row["cdr_count"] += int(
            (changed & clean_flag & (pseudo_nodes != clean_nodes)).sum().item()
        )
        row["alpha_sum"] += float(alpha_nodes.sum().item())
        row["clean_mass_delta_total"] += float(clean_mass_delta.sum().item())
        row["majority_clean_correct_count"] += int(
            (clean_nodes == majority).sum().item()
        )
        row["margin_sum"] += margin
        row["target_argmax_correct_count"] += int(
            (soft_target[nodes].argmax(dim=1) == clean_nodes).sum().item()
        )

    total_target_nodes = sum(
        int(row["target_node_count"]) for row in totals.values()
    )
    rows = []
    for support_bin in bin_order:
        total = totals[support_bin]
        target_count = int(total["target_node_count"])
        noisy_count = int(total["noisy_train_count"])
        clean_count = int(total["clean_train_count"])
        rows.append(
            {
                "support_bin": support_bin,
                "ball_count": int(total["ball_count"]),
                "target_node_count": target_count,
                "target_node_coverage": (
                    float(target_count / total_target_nodes)
                    if total_target_nodes
                    else 0.0
                ),
                "changed_train_count": int(total["changed_train_count"]),
                "noisy_train_count": noisy_count,
                "clean_train_count": clean_count,
                "ncr": (
                    float(total["ncr_count"] / noisy_count)
                    if noisy_count
                    else 0.0
                ),
                "cdr": (
                    float(total["cdr_count"] / clean_count)
                    if clean_count
                    else 0.0
                ),
                "mean_alpha": (
                    float(total["alpha_sum"] / target_count)
                    if target_count
                    else 0.0
                ),
                "clean_mass_delta_per_node": (
                    float(total["clean_mass_delta_total"] / target_count)
                    if target_count
                    else 0.0
                ),
                "majority_clean_accuracy": (
                    float(total["majority_clean_correct_count"] / target_count)
                    if target_count
                    else 0.0
                ),
                "mean_observed_margin": (
                    float(total["margin_sum"] / total["ball_count"])
                    if total["ball_count"]
                    else 0.0
                ),
                "soft_target_argmax_accuracy": (
                    float(total["target_argmax_correct_count"] / target_count)
                    if target_count
                    else 0.0
                ),
            }
        )
    return rows


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
    target_beta: float = 0.0,
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
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")
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
        if target_beta > 0.0:
            majority_onehot = F.one_hot(
                torch.full((1,), majority, dtype=torch.long, device=device),
                num_classes=n_classes,
            ).float().squeeze(0)
            ball_dist = (1.0 - target_beta) * ball_dist + target_beta * majority_onehot
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
        "expansion_target_beta": float(target_beta),
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


def build_knn_expansion_soft_targets_from_features(
    ball_features: torch.Tensor,
    noisy_label: torch.Tensor,
    train_mask,
    n_classes: int,
    max_balls: int = 64,
    min_ball_size: int = 32,
    split_improvement: float = 0.05,
    min_train_per_ball: int = 10,
    margin_threshold: float = 0.5,
    target_beta: float = 0.0,
    random_state: int = 0,
    matched_k: int | None = None,
) -> BallExpansionTargetResult:
    embeddings = prepare_ball_features(ball_features).cpu()
    if int(embeddings.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError("ball feature node count must match noisy_label length")
    _, match_diag = _reference_ball_assignment_for_matching(
        embeddings,
        train_mask=train_mask,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        random_state=random_state,
    )
    device = noisy_label.device
    n_nodes = int(noisy_label.shape[0])
    n_classes = int(n_classes)
    train_idx = _mask_to_index_array(train_mask)
    is_train = np.zeros(n_nodes, dtype=bool)
    is_train[train_idx] = True
    candidate_idx = np.flatnonzero(~is_train)
    soft_target = F.one_hot(noisy_label.long(), num_classes=n_classes).float().to(device)
    candidate_mask = torch.zeros_like(noisy_label, dtype=torch.bool, device=device)
    pseudo_label = noisy_label.detach().clone()

    min_train_per_ball = int(min_train_per_ball)
    margin_threshold = float(margin_threshold)
    target_beta = float(target_beta)
    if target_beta < 0.0 or target_beta > 1.0:
        raise ValueError("target_beta must be in [0, 1]")

    n_train = int(train_idx.size)
    if n_train == 0 or candidate_idx.size == 0:
        k_value = 0
        selected_count = 0
        margin_mean = 0.0
    else:
        k_value = int(matched_k) if matched_k is not None else int(round(match_diag["matched_mean_train_per_ball"]))
        k_value = max(1, min(k_value, n_train))
        if k_value < min_train_per_ball:
            selected_count = 0
            margin_mean = 0.0
        else:
            train_tensor = torch.as_tensor(train_idx, dtype=torch.long)
            candidate_tensor_cpu = torch.as_tensor(candidate_idx, dtype=torch.long)
            train_features = embeddings[train_tensor]
            candidate_features = embeddings[candidate_tensor_cpu]
            similarity = candidate_features @ train_features.T
            topk = torch.topk(similarity, k=k_value, dim=1, largest=True).indices.cpu().numpy()
            noisy_np = noisy_label.detach().cpu().long().numpy()
            selected = []
            selected_targets = []
            selected_majority = []
            selected_margins = []
            for row_id, neighbor_pos in enumerate(topk):
                labels = noisy_np[train_idx[neighbor_pos]]
                counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
                train_votes = max(1.0, float(counts.sum()))
                margin = float(counts.max() / train_votes)
                if margin < margin_threshold:
                    continue
                dist = counts / train_votes
                majority = int(np.argmax(counts))
                if target_beta > 0.0:
                    majority_onehot = np.eye(n_classes, dtype=np.float64)[majority]
                    dist = (1.0 - target_beta) * dist + target_beta * majority_onehot
                selected.append(int(candidate_idx[row_id]))
                selected_targets.append(dist)
                selected_majority.append(majority)
                selected_margins.append(margin)
            selected_count = int(len(selected))
            margin_mean = float(np.mean(selected_margins)) if selected_margins else 0.0
            if selected:
                selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
                soft_target[selected_tensor] = torch.as_tensor(
                    np.asarray(selected_targets, dtype=np.float64),
                    dtype=torch.float,
                    device=device,
                )
                candidate_mask[selected_tensor] = True
                pseudo_label[selected_tensor] = torch.as_tensor(selected_majority, dtype=torch.long, device=device)

    non_train_count = int(candidate_idx.size)
    diagnostics = {
        **match_diag,
        "expansion_builder": "knn",
        "expansion_min_train_per_ball": int(min_train_per_ball),
        "expansion_margin_threshold": float(margin_threshold),
        "expansion_target_beta": float(target_beta),
        "expansion_matched_k": int(k_value),
        "expansion_eligible_queries": int(candidate_idx.size if k_value >= min_train_per_ball else 0),
        "expansion_selected_queries": int(selected_count),
        "expansion_candidate_count": int(selected_count),
        "expansion_coverage_unlabeled": float(selected_count / max(1, non_train_count)),
        "expansion_coverage_total": float(selected_count / max(1, n_nodes)),
        "expansion_mean_selected_margin": float(margin_mean),
        "expansion_mean_selected_train_votes": float(k_value if selected_count else 0.0),
    }
    return BallExpansionTargetResult(
        soft_target=soft_target.detach(),
        candidate_mask=candidate_mask.detach(),
        pseudo_label=pseudo_label.detach(),
        diagnostics=diagnostics,
    )


def build_frozen_grace_knn_expansion_soft_targets(
    embedding_path: str | Path,
    noisy_label: torch.Tensor,
    train_mask,
    n_classes: int,
    **kwargs,
) -> BallExpansionTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    return build_knn_expansion_soft_targets_from_features(
        ball_features=embeddings,
        noisy_label=noisy_label,
        train_mask=train_mask,
        n_classes=n_classes,
        **kwargs,
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
    ball_mode: str = "geometric",
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(embedding_path, expected_nodes=int(noisy_label.shape[0]))
    ball_mode = str(ball_mode)
    if ball_mode == "geometric":
        gb_result = build_ordinary_granular_balls(
            embeddings,
            max_balls=max_balls,
            min_ball_size=min_ball_size,
            split_improvement=split_improvement,
            random_state=random_state,
        )
    elif ball_mode == "size_adaptive_ellipsoid":
        gb_result = build_size_adaptive_ellipsoid_balls(
            embeddings,
            n_classes=n_classes,
            max_balls=max_balls if max_balls is not None else None,
            min_ball_size=1,
            random_state=random_state,
        )
    else:
        raise ValueError("ball_mode must be 'geometric' or 'size_adaptive_ellipsoid'")
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
        **partition_purity_diagnostics(noisy_label, clean_label, assignment, train_mask, n_classes, min_train_per_ball),
        "random_control": int(random_control),
        "ball_mode": ball_mode,
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


def build_soft_targets_from_features(
    ball_features: torch.Tensor,
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
    target_beta: float = 0.0,
    leave_one_out: bool = False,
    softdist_weighting: str = "uniform",
    ellipsoid_ridge_scale: float = 0.05,
    ellipsoid_weight_floor: float = 0.1,
    ellipsoid_feature_space: str = "fullH",
    ellipsoid_covariance_estimator: str = "ridge",
    ellipsoid_distance_temperature: float = 1.0,
    gmm_n_components: int | None = None,
    gmm_covariance_type: str = "diag",
    gmm_reg_covar: float = 1e-5,
    gmm_max_iter: int = 100,
    gmm_n_init: int = 1,
    gmm_prior: str = "jeffreys",
    gmm_geometry_confidence: str = "max_resp",
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
    tie_mode: str = "min_index",
) -> FrozenGraceTargetResult:
    embeddings = prepare_ball_features(ball_features)
    if int(embeddings.shape[0]) != int(noisy_label.shape[0]):
        raise ValueError(
            f"ball feature node count {embeddings.shape[0]} != expected {noisy_label.shape[0]}"
        )
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
    source_ball_sizes = sorted(np.bincount(np.asarray(assignment, dtype=np.int64)).tolist())
    if random_control:
        assignment = build_random_size_matched_assignment(assignment, random_state=random_state + 17)
    random_control_size_matched = int(
        source_ball_sizes
        == sorted(np.bincount(np.asarray(assignment, dtype=np.int64)).tolist())
    )
    softdist_weighting = str(softdist_weighting)
    if softdist_weighting == "uniform":
        tie_mode = str(tie_mode).lower()
        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=n_classes,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            target_beta=target_beta,
            leave_one_out=leave_one_out,
            tie_mode=tie_mode,
        )
        weighting_diagnostics = {
            **_soft_coverage_diagnostics(
                noisy_label,
                assignment,
                train_mask,
                n_classes,
                min_train_per_ball,
                alpha_baseline,
                leave_one_out=leave_one_out,
            ),
            **_uniform_tie_diagnostics(
                noisy_label=noisy_label,
                assignment=assignment,
                train_mask=train_mask,
                n_classes=n_classes,
                min_train_per_ball=min_train_per_ball,
                alpha_baseline=alpha_baseline,
                leave_one_out=leave_one_out,
                tie_mode=tie_mode,
            ),
            "softdist_weighting": "uniform",
        }
    elif softdist_weighting == "ellipsoid":
        soft_target, alpha, pseudo_label, weighting_diagnostics = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=n_classes,
            ball_features=embeddings,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            target_beta=target_beta,
            ellipsoid_ridge_scale=ellipsoid_ridge_scale,
            ellipsoid_weight_floor=ellipsoid_weight_floor,
        )
    elif softdist_weighting == "ellipsoid_pca_shrinkage":
        soft_target, alpha, pseudo_label, weighting_diagnostics = apply_ball_ellipsoid_pca_shrinkage_soft_targets(
            noisy_label=noisy_label,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=n_classes,
            ball_features=embeddings,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            target_beta=target_beta,
            ellipsoid_feature_space=ellipsoid_feature_space,
            ellipsoid_covariance_estimator=ellipsoid_covariance_estimator,
            ellipsoid_ridge_scale=ellipsoid_ridge_scale,
            ellipsoid_weight_floor=ellipsoid_weight_floor,
            ellipsoid_distance_temperature=ellipsoid_distance_temperature,
        )
    elif softdist_weighting == "gmm_resp":
        soft_target, alpha, pseudo_label, weighting_diagnostics, assignment = apply_gmm_resp_soft_targets(
            noisy_label=noisy_label,
            clean_label=clean_label,
            ball_features=embeddings,
            train_mask=train_mask,
            n_classes=n_classes,
            fallback_assignment=assignment,
            min_train_per_ball=min_train_per_ball,
            alpha_baseline=alpha_baseline,
            target_mode=target_mode,
            target_beta=target_beta,
            gmm_n_components=max_balls if gmm_n_components is None else gmm_n_components,
            gmm_covariance_type=gmm_covariance_type,
            gmm_reg_covar=gmm_reg_covar,
            gmm_max_iter=gmm_max_iter,
            gmm_n_init=gmm_n_init,
            gmm_prior=gmm_prior,
            gmm_geometry_confidence=gmm_geometry_confidence,
            random_state=random_state,
            random_control=random_control,
        )
    else:
        raise ValueError("softdist_weighting must be 'uniform', 'ellipsoid', 'ellipsoid_pca_shrinkage', or 'gmm_resp'")
    changed = (alpha > 0) & (pseudo_label != noisy_label)
    partition_summary = (
        _assignment_summary(assignment)
        if softdist_weighting == "gmm_resp" and int(weighting_diagnostics.get("gmm_fallback", 0)) == 0
        else gb_result.summary
    )
    diagnostics = {
        **partition_summary,
        **weighting_diagnostics,
        **diagnostics_for_soft_targets(noisy_label, clean_label, soft_target, alpha, pseudo_label, train_mask),
        **partition_purity_diagnostics(noisy_label, clean_label, assignment, train_mask, n_classes, min_train_per_ball),
        "ball_mode": ball_mode,
        "soft_alpha_baseline": str(alpha_baseline),
        "soft_target_mode": str(target_mode),
        "target_beta": float(target_beta),
        "leave_one_out": int(leave_one_out),
        "random_control": int(random_control),
        "random_control_size_matched": random_control_size_matched,
        "ball_feature_source": "memory",
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
    target_beta: float = 0.0,
    leave_one_out: bool = False,
    softdist_weighting: str = "uniform",
    ellipsoid_ridge_scale: float = 0.05,
    ellipsoid_weight_floor: float = 0.1,
    ellipsoid_feature_space: str = "fullH",
    ellipsoid_covariance_estimator: str = "ridge",
    ellipsoid_distance_temperature: float = 1.0,
    gmm_n_components: int | None = None,
    gmm_covariance_type: str = "diag",
    gmm_reg_covar: float = 1e-5,
    gmm_max_iter: int = 100,
    gmm_n_init: int = 1,
    gmm_prior: str = "jeffreys",
    gmm_geometry_confidence: str = "max_resp",
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
    tie_mode: str = "min_index",
) -> FrozenGraceTargetResult:
    embeddings = load_frozen_embeddings(
        embedding_path, expected_nodes=int(noisy_label.shape[0])
    )
    result = build_soft_targets_from_features(
        ball_features=embeddings,
        noisy_label=noisy_label,
        clean_label=clean_label,
        train_mask=train_mask,
        n_classes=n_classes,
        max_balls=max_balls,
        min_ball_size=min_ball_size,
        split_improvement=split_improvement,
        min_train_per_ball=min_train_per_ball,
        alpha_baseline=alpha_baseline,
        target_mode=target_mode,
        target_beta=target_beta,
        leave_one_out=leave_one_out,
        softdist_weighting=softdist_weighting,
        ellipsoid_ridge_scale=ellipsoid_ridge_scale,
        ellipsoid_weight_floor=ellipsoid_weight_floor,
        ellipsoid_feature_space=ellipsoid_feature_space,
        ellipsoid_covariance_estimator=ellipsoid_covariance_estimator,
        ellipsoid_distance_temperature=ellipsoid_distance_temperature,
        gmm_n_components=gmm_n_components,
        gmm_covariance_type=gmm_covariance_type,
        gmm_reg_covar=gmm_reg_covar,
        gmm_max_iter=gmm_max_iter,
        gmm_n_init=gmm_n_init,
        gmm_prior=gmm_prior,
        gmm_geometry_confidence=gmm_geometry_confidence,
        random_state=random_state,
        random_control=random_control,
        ball_mode=ball_mode,
        purity_threshold=purity_threshold,
        min_train_per_child=min_train_per_child,
        min_train_to_split=min_train_to_split,
        adaptive_min_ball_size=adaptive_min_ball_size,
        adaptive_max_balls=adaptive_max_balls,
        min_purity_gain=min_purity_gain,
        split_attempts=split_attempts,
        tie_mode=tie_mode,
    )
    result.diagnostics["ball_feature_source"] = "grace"
    return result


def _normalize_responsibilities(responsibilities: np.ndarray, n_nodes: int) -> np.ndarray:
    responsibilities = np.asarray(responsibilities, dtype=np.float64)
    if responsibilities.ndim != 2 or int(responsibilities.shape[0]) != int(n_nodes):
        raise ValueError("responsibilities must be a 2-D array matching noisy_label length")
    if int(responsibilities.shape[1]) <= 0:
        raise ValueError("responsibilities must have at least one component")
    if not np.isfinite(responsibilities).all():
        raise ValueError("responsibilities contain NaN or Inf")
    responsibilities = np.clip(responsibilities, 0.0, None)
    row_sum = responsibilities.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("responsibility rows must have positive mass")
    return responsibilities / row_sum


def _gmm_noise_auc(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    node_dist: np.ndarray,
    train_idx: np.ndarray,
) -> float:
    if train_idx.size == 0:
        return 0.5
    noisy_np = noisy_label.detach().cpu().long().numpy()
    clean_np = clean_label.detach().cpu().long().numpy()
    flags = (noisy_np[train_idx] != clean_np[train_idx]).astype(np.int64)
    if np.unique(flags).size < 2:
        return 0.5
    scores = 1.0 - node_dist[train_idx, noisy_np[train_idx]]
    try:
        return float(roc_auc_score(flags, scores))
    except ValueError:
        return 0.5


def _gmm_support_bin_diagnostics(
    noisy_label: torch.Tensor,
    clean_label: torch.Tensor,
    soft_target: torch.Tensor,
    alpha: torch.Tensor,
    pseudo_label: torch.Tensor,
    train_idx: np.ndarray,
    dominant_support: np.ndarray,
) -> dict:
    if train_idx.size == 0:
        return {}
    device = noisy_label.device
    out = {}
    bins = [
        ("lt2", dominant_support[train_idx] < 2),
        ("2-3", (dominant_support[train_idx] >= 2) & (dominant_support[train_idx] <= 3)),
        ("4-7", (dominant_support[train_idx] >= 4) & (dominant_support[train_idx] <= 7)),
        ("8+", dominant_support[train_idx] >= 8),
    ]
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    noisy_train = noisy_label[train_tensor].long()
    clean_train = clean_label[train_tensor].long()
    pseudo_train = pseudo_label[train_tensor].long()
    alpha_train = alpha[train_tensor]
    changed = (alpha_train > 0) & (pseudo_train != noisy_train)
    noisy_flag = noisy_train != clean_train
    clean_flag = ~noisy_flag
    clean_mass_delta = soft_target[train_tensor, clean_train] - (noisy_train == clean_train).float()

    for name, mask_np in bins:
        if not np.any(mask_np):
            continue
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        prefix = f"gmm_support_bin_{name}_"
        noisy_count = int((noisy_flag & mask).sum().item())
        clean_count = int((clean_flag & mask).sum().item())
        ncr_count = int((changed & noisy_flag & (pseudo_train == clean_train) & mask).sum().item())
        cdr_count = int((changed & clean_flag & (pseudo_train != clean_train) & mask).sum().item())
        out[prefix + "train_count"] = int(mask.sum().item())
        out[prefix + "changed_train_count"] = int((changed & mask).sum().item())
        out[prefix + "ncr"] = float(ncr_count / noisy_count) if noisy_count else 0.0
        out[prefix + "cdr"] = float(cdr_count / clean_count) if clean_count else 0.0
        out[prefix + "mean_alpha"] = float(alpha_train[mask].mean().item())
        out[prefix + "clean_mass_delta"] = float(clean_mass_delta[mask].sum().item())
    return out


def _assignment_summary(assignment: np.ndarray) -> dict:
    assignment = np.asarray(assignment, dtype=np.int64)
    _, counts = np.unique(assignment, return_counts=True)
    if counts.size == 0:
        return {
            "num_balls": 0,
            "mean_ball_size": 0.0,
            "max_ball_size": 0,
            "singleton_ratio": 0.0,
            "mean_radius": 0.0,
        }
    return {
        "num_balls": int(counts.size),
        "mean_ball_size": float(np.mean(counts)),
        "max_ball_size": int(np.max(counts)),
        "singleton_ratio": float(np.mean(counts == 1)),
        "mean_radius": 0.0,
    }


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
    leave_one_out: bool = False,
) -> dict:
    train_idx = _mask_to_index_array(train_mask)
    eligible_balls = 0
    eligible_train_nodes = 0
    alpha_positive_balls = 0
    alpha_positive_train_nodes = 0
    device = noisy_label.device
    for ball_id in sorted(np.unique(assignment).tolist()):
        ball_train = train_idx[assignment[train_idx] == ball_id]
        required_support = int(min_train_per_ball) + (1 if leave_one_out else 0)
        if ball_train.size < required_support:
            continue
        eligible_balls += 1
        eligible_train_nodes += int(ball_train.size)
        ball_train_tensor = torch.as_tensor(ball_train, dtype=torch.long, device=device)
        counts = torch.bincount(noisy_label[ball_train_tensor].long(), minlength=int(n_classes))
        if leave_one_out:
            positive_nodes = 0
            for node in ball_train_tensor:
                node_counts = counts.clone()
                node_counts[noisy_label[node].long()] -= 1
                margin = float(node_counts.max().item() / max(1, int(node_counts.sum().item())))
                if _alpha_from_margin(margin, int(n_classes), str(alpha_baseline)) > 0.0:
                    positive_nodes += 1
            if positive_nodes > 0:
                alpha_positive_balls += 1
                alpha_positive_train_nodes += positive_nodes
            continue
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
        "leave_one_out": int(leave_one_out),
    }


def _alpha_from_margin(margin: float, n_classes: int, alpha_baseline: str) -> float:
    if alpha_baseline == "multiclass":
        zero_point = 1.0 / max(1, int(n_classes))
    else:
        zero_point = 0.5
    if zero_point >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (float(margin) - zero_point) / (1.0 - zero_point)))
