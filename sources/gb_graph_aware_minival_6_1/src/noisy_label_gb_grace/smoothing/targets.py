from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


CONFLICT_GATE_REPAIR_CONFIGS = {
    "gagb_qb_conflict_gate_alpha03": {"alpha_max": 0.3, "target_update_mode": "mix"},
    "grace_gagb_qb_conflict_gate_alpha03": {"alpha_max": 0.3, "target_update_mode": "mix"},
    "gagb_qb_conflict_gate_alpha06": {"alpha_max": 0.6, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "gagb_qb_conflict_gate_alpha07": {"alpha_max": 0.7, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "grace_gagb_qb_conflict_gate_alpha07": {"alpha_max": 0.7, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "gagb_qb_conflict_gate_alpha08": {"alpha_max": 0.8, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "gagb_qb_conflict_gate_alpha09": {"alpha_max": 0.9, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "grace_gagb_qb_conflict_gate_alpha09": {"alpha_max": 0.9, "target_update_mode": "mix", "fixed_gate_alpha": True},
    "grace_gagb_qfilter_alpha07_q50": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_filter_mix",
        "fixed_gate_alpha": True,
        "quality_quantile": 0.5,
    },
    "grace_gagb_qfilter_alpha07_q70": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_filter_mix",
        "fixed_gate_alpha": True,
        "quality_quantile": 0.7,
    },
    "grace_gagb_qweight_alpha07": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_weight_mix",
        "fixed_gate_alpha": True,
    },
    "gagb_qfilter_alpha07_q50": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_filter_mix",
        "fixed_gate_alpha": True,
        "quality_quantile": 0.5,
    },
    "khop_only_qfilter_alpha07_q50": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_filter_mix",
        "fixed_gate_alpha": True,
        "quality_quantile": 0.5,
    },
    "grace_gagb_shuffled_qfilter_alpha07_q50": {
        "alpha_max": 0.7,
        "target_update_mode": "quality_filter_mix",
        "fixed_gate_alpha": True,
        "quality_quantile": 0.5,
    },
    "gagb_qb_conflict_gate_replace_qmix": {"alpha_max": 1.0, "target_update_mode": "replace_qmix"},
    "gagb_class_capped_alpha09": {
        "alpha_max": 0.9,
        "target_update_mode": "class_capped_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
    },
    "gagb_class_balanced_alpha09": {
        "alpha_max": 0.9,
        "target_update_mode": "class_balanced_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
    },
    "gagb_boundary_safe_alpha09": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
    },
    "grace_gagb_qsafe_boundary_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "grace_gagb_qsafe_boundary_alpha09_q70": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.7,
        "quality_gate_mode": "candidate",
    },
    "dcgb_qsafe_boundary_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "grace_gagb_qsafe_boundary_alpha07_q50": {
        "alpha_max": 0.7,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "grace_gagb_qsafe_class_capped_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "class_capped_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "grace_gagb_qsafe_class_balanced_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "class_balanced_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "gagb_qsafe_boundary_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "khop_only_qsafe_boundary_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "grace_gagb_shuffled_qsafe_boundary_alpha09_q50": {
        "alpha_max": 0.9,
        "target_update_mode": "boundary_safe_mix",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.3,
        "qmix_margin_min": 0.2,
        "teacher_conf_min": 0.6,
        "teacher_margin_min": 0.2,
        "quality_quantile": 0.5,
        "quality_gate_mode": "candidate",
    },
    "gagb_replay_random_same_count": {
        "alpha_max": 0.9,
        "target_update_mode": "replay_random_same_count",
        "fixed_gate_alpha": True,
        "fallback_alpha": 0.0,
    },
    "gagb_qb_conflict_gate_sharpen_qmix_t05": {
        "alpha_max": 1.0,
        "target_update_mode": "sharpen_qmix",
        "sharpen_temperature": 0.5,
    },
    "gagb_qb_conflict_gate_margin_replace": {
        "alpha_max": 1.0,
        "target_update_mode": "margin_replace",
        "margin_replace_threshold": 0.2,
    },
}

REPAIR_CONTROL_CONFIGS = {
    "gagb_shuffled_alpha09_matched": {
        "base_method": "gagb_qb_conflict_gate_alpha09",
        "grouping_source": "gagb_shuffled",
    },
    "gagb_shuffled_replace_matched": {
        "base_method": "gagb_qb_conflict_gate_replace_qmix",
        "grouping_source": "gagb_shuffled",
    },
    "kmeans_gagb_matched_alpha09": {
        "base_method": "gagb_qb_conflict_gate_alpha09",
        "grouping_source": "kmeans_gagb_matched",
    },
    "kmeans_gagb_matched_replace": {
        "base_method": "gagb_qb_conflict_gate_replace_qmix",
        "grouping_source": "kmeans_gagb_matched",
    },
    "topk_gagb_train_matched_alpha09": {
        "base_method": "gagb_qb_conflict_gate_alpha09",
        "grouping_source": "topk_gagb_train_matched",
    },
    "topk_gagb_train_matched_replace": {
        "base_method": "gagb_qb_conflict_gate_replace_qmix",
        "grouping_source": "topk_gagb_train_matched",
    },
    "khop_only_alpha09": {
        "base_method": "gagb_qb_conflict_gate_alpha09",
        "grouping_source": "khop_only",
    },
    "khop_only_replace": {
        "base_method": "gagb_qb_conflict_gate_replace_qmix",
        "grouping_source": "khop_only",
    },
}

TEACHER_REPAIR_METHODS = set(CONFLICT_GATE_REPAIR_CONFIGS) | set(REPAIR_CONTROL_CONFIGS)

from noisy_label_gb_grace.balls.granular_ball import groups_to_stats
from noisy_label_gb_grace.utils import mask_to_index, one_hot


def _adjacency_sets(edge_index: torch.Tensor | None, num_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    if edge_index is None:
        return adjacency
    edge_index = edge_index.detach().cpu().long()
    if edge_index.numel() == 0:
        return adjacency
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a [2, num_edges] tensor")
    for src, dst in edge_index.t().tolist():
        src = int(src)
        dst = int(dst)
        if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _distribution(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    counts = torch.bincount(labels[indices].long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


def _alpha_distribution_stats(alphas: torch.Tensor, train_idx: torch.Tensor) -> dict:
    if train_idx.numel() == 0:
        return {
            "alpha_std_train": 0.0,
            "alpha_q25_train": 0.0,
            "alpha_q50_train": 0.0,
            "alpha_q75_train": 0.0,
        }
    train_alpha = alphas[train_idx].float()
    quantiles = torch.quantile(train_alpha, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32))
    return {
        "alpha_std_train": float(train_alpha.std(unbiased=False).item()),
        "alpha_q25_train": float(quantiles[0].item()),
        "alpha_q50_train": float(quantiles[1].item()),
        "alpha_q75_train": float(quantiles[2].item()),
    }


def _groups_from_ids(group_ids: torch.Tensor) -> list[list[int]]:
    groups = []
    for group_id in sorted(group_ids.unique().tolist()):
        groups.append((group_ids == group_id).nonzero(as_tuple=False).view(-1).tolist())
    return groups


def _cluster_geometry(embeddings: torch.Tensor, group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emb = embeddings.detach().cpu().float()
    groups = _groups_from_ids(group_ids)
    centers = []
    radii = []
    node_distances = torch.zeros(emb.size(0), dtype=torch.float32)
    for group in groups:
        idx = torch.tensor(group, dtype=torch.long)
        points = emb[idx]
        center = points.mean(dim=0)
        dist = torch.norm(points - center, dim=1)
        centers.append(center)
        radii.append(float(dist.max().item()) if dist.numel() else 0.0)
        node_distances[idx] = dist.float()
    return torch.stack(centers), torch.tensor(radii, dtype=torch.float32), node_distances


def kmeans_groups(embeddings: torch.Tensor, num_groups: int, seed: int) -> dict:
    emb = embeddings.detach().cpu().float()
    num_groups = max(1, min(int(num_groups), int(emb.size(0))))
    model = KMeans(n_clusters=num_groups, random_state=seed, n_init=10)
    labels = torch.tensor(model.fit_predict(emb.numpy()), dtype=torch.long)
    centers, radii, node_distances = _cluster_geometry(emb, labels)
    groups = _groups_from_ids(labels)
    return {
        "group_ids": labels,
        "groups": groups,
        "centers": centers,
        "radii": radii,
        "node_distances": node_distances,
        "diagnostics": groups_to_stats(labels, groups, radii),
        "config": {"num_groups": int(num_groups), "seed": int(seed)},
    }


def train_node_count_diagnostics(grouping: dict, train_mask: torch.Tensor) -> dict:
    train_mask = train_mask.detach().cpu().bool()
    counts = []
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        counts.append(int(train_mask[idx].sum().item()))
    if not counts:
        return {
            "avg_train_nodes_per_ball": math.nan,
            "balls_with_0_train_nodes": 0,
            "balls_with_1_train_node": 0,
            "balls_with_2_train_nodes": 0,
            "balls_with_3plus_train_nodes": 0,
            "eligible_ball_count": 0,
            "eligible_train_node_count": 0,
        }
    return {
        "avg_train_nodes_per_ball": float(np.mean(counts)),
        "avg_train_nodes_per_nonempty_ball": float(np.mean([c for c in counts if c > 0])) if any(c > 0 for c in counts) else 0.0,
        "balls_with_0_train_nodes": int(sum(1 for c in counts if c == 0)),
        "balls_with_1_train_node": int(sum(1 for c in counts if c == 1)),
        "balls_with_2_train_nodes": int(sum(1 for c in counts if c == 2)),
        "balls_with_3plus_train_nodes": int(sum(1 for c in counts if c >= 3)),
        "eligible_ball_count": int(sum(1 for c in counts if c >= 3)),
        "eligible_train_node_count": int(sum(c for c in counts if c >= 3)),
    }


def build_topk_group_size(
    gb_groups: list[list[int]],
    train_mask: torch.Tensor | None = None,
    min_train_per_group: int = 3,
    match_train_nodes: bool = False,
) -> int:
    if match_train_nodes and train_mask is not None:
        train_mask = train_mask.detach().cpu().bool()
        eligible_neighbor_counts = []
        for group in gb_groups:
            if len(group) <= 1:
                continue
            idx = torch.tensor(group, dtype=torch.long)
            count = int(train_mask[idx].sum().item())
            if count >= min_train_per_group:
                eligible_neighbor_counts.append(count - 1)
        if eligible_neighbor_counts:
            return max(1, int(round(float(np.mean(eligible_neighbor_counts)))))
        return 1
    non_singleton = [len(group) for group in gb_groups if len(group) > 1]
    if not non_singleton:
        return 1
    return max(1, int(round(float(np.mean(non_singleton)) - 1.0)))


def _alpha_for_train_nodes(
    train_nodes: torch.Tensor,
    group_id: int,
    radii: torch.Tensor,
    node_distances: torch.Tensor,
    alpha_max: float,
    eps: float,
    use_confidence: bool,
    override: torch.Tensor | None = None,
) -> torch.Tensor:
    if override is not None:
        return override[train_nodes].float()
    if use_confidence:
        radius = float(radii[group_id].item()) if group_id < radii.numel() else 0.0
        ball_conf = 1.0 / (1.0 + radius)
        node_conf = torch.exp(-node_distances[train_nodes] / (radius + eps))
        return alpha_max * ball_conf * node_conf
    return torch.full((train_nodes.numel(),), float(alpha_max))


def _target_change_stats(
    original_targets: torch.Tensor,
    targets: torch.Tensor,
    alphas: torch.Tensor,
    train_idx: torch.Tensor,
) -> dict:
    changed_mask = (alphas > 0).bool()
    l1_delta = (targets - original_targets).abs().sum(dim=1).float()
    train_delta = l1_delta[train_idx] if train_idx.numel() else torch.empty(0)
    return {
        "target_changed_mask": changed_mask.tolist(),
        "target_l1_delta": l1_delta.tolist(),
        "target_changed_train_count": int(changed_mask[train_idx].sum().item()) if train_idx.numel() else 0,
        "target_l1_delta_mean_train": float(train_delta.mean().item()) if train_delta.numel() else 0.0,
    }


def compute_group_quality_scores(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    grouping: dict,
    num_classes: int,
    min_train_per_ball: int = 3,
) -> dict:
    emb = torch.nn.functional.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    labels = noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    adjacency = _adjacency_sets(edge_index, int(labels.numel()))
    log_classes = math.log(max(2, int(num_classes)))
    quality = []
    semantic = []
    topology = []
    train_support = []
    entropy_support = []
    train_counts = []

    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        if idx.numel() == 0:
            semantic.append(0.0)
            topology.append(0.0)
            train_support.append(0.0)
            entropy_support.append(0.0)
            train_counts.append(0)
            quality.append(0.0)
            continue

        points = emb[idx]
        center = torch.nn.functional.normalize(points.mean(dim=0, keepdim=True), p=2, dim=1).view(-1)
        sem = float(((points @ center).clamp(-1.0, 1.0).add(1.0).mul(0.5)).mean().item())

        members = [int(node) for node in idx.tolist()]
        if len(members) <= 1:
            topo = 0.0
        else:
            internal_edges = 0
            member_set = set(members)
            for node in members:
                internal_edges += sum(1 for neighbor in adjacency[node] if neighbor in member_set)
            possible_directed = len(members) * (len(members) - 1)
            topo = float(internal_edges / max(1, possible_directed))

        train_in_group = idx[train_mask[idx]]
        train_count = int(train_in_group.numel())
        q_train = min(1.0, train_count / max(1, int(min_train_per_ball)))
        if train_count == 0:
            q_entropy = 0.0
        else:
            probs = _distribution(labels, train_in_group, num_classes)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
            q_entropy = max(0.0, 1.0 - entropy / log_classes)

        q = (max(sem, 0.0) ** 0.35) * (max(topo, 0.0) ** 0.25) * (max(q_train, 0.0) ** 0.20) * (max(q_entropy, 0.0) ** 0.20)
        semantic.append(float(sem))
        topology.append(float(topo))
        train_support.append(float(q_train))
        entropy_support.append(float(q_entropy))
        train_counts.append(train_count)
        quality.append(float(q))

    return {
        "group_quality": torch.tensor(quality, dtype=torch.float32),
        "group_semantic": torch.tensor(semantic, dtype=torch.float32),
        "group_topology": torch.tensor(topology, dtype=torch.float32),
        "group_train": torch.tensor(train_support, dtype=torch.float32),
        "group_entropy": torch.tensor(entropy_support, dtype=torch.float32),
        "group_train_count": torch.tensor(train_counts, dtype=torch.long),
    }


def _quality_filter_state(quality_scores: dict | None, min_train_per_group: int, quality_quantile: float | None, quality_weight_alpha: bool) -> dict | None:
    if quality_scores is None:
        return None
    group_quality = quality_scores["group_quality"].detach().cpu().float()
    train_counts = quality_scores["group_train_count"].detach().cpu().long()
    eligible = train_counts >= int(min_train_per_group)
    eligible_values = group_quality[eligible]
    if quality_weight_alpha:
        threshold = math.nan
        selected = eligible & (group_quality > 0)
    elif eligible_values.numel() == 0:
        threshold = math.inf
        selected = torch.zeros_like(eligible)
    else:
        threshold = float(torch.quantile(eligible_values, float(quality_quantile)).item())
        selected = eligible & (group_quality >= threshold)
    rejected = eligible & ~selected

    def _mean(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else math.nan

    basis = eligible if bool(eligible.any().item()) else torch.ones_like(eligible, dtype=torch.bool)
    return {
        "group_quality": group_quality,
        "selected": selected,
        "rejected": rejected,
        "threshold": threshold,
        "selected_gate_count": 0,
        "rejected_gate_count": 0,
        "stats": {
            "quality_threshold": float(threshold),
            "quality_quantile": float(quality_quantile) if quality_quantile is not None else math.nan,
            "quality_weight_alpha": bool(quality_weight_alpha),
            "quality_eligible_ball_count": int(eligible.sum().item()),
            "quality_selected_ball_count": int(selected.sum().item()),
            "quality_rejected_ball_count": int(rejected.sum().item()),
            "quality_selected_mean": _mean(group_quality[selected]),
            "quality_rejected_mean": _mean(group_quality[rejected]),
            "quality_sem_mean": _mean(quality_scores["group_semantic"].detach().cpu().float()[basis]),
            "quality_topo_mean": _mean(quality_scores["group_topology"].detach().cpu().float()[basis]),
            "quality_train_mean": _mean(quality_scores["group_train"].detach().cpu().float()[basis]),
            "quality_entropy_mean": _mean(quality_scores["group_entropy"].detach().cpu().float()[basis]),
        },
    }


def _local_group_targets(
    clean_or_noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    grouping: dict,
    alpha_max: float,
    min_train_per_group: int,
    eps: float,
    use_confidence: bool,
    prior_gamma: float | None = None,
    alpha_override: torch.Tensor | list[float] | None = None,
) -> tuple[torch.Tensor, dict]:
    labels = clean_or_noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    y_soft = one_hot(labels, num_classes)
    y_original = y_soft.clone()
    group_ids = grouping["group_ids"].detach().cpu().long()
    radii = grouping["radii"].detach().cpu().float()
    node_distances = grouping["node_distances"].detach().cpu().float()
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    train_idx = mask_to_index(train_mask)
    group_train_counts = []
    entropies = []
    p_global = _distribution(labels, train_idx, num_classes)
    override = None
    if alpha_override is not None:
        override = torch.tensor(alpha_override, dtype=torch.float32).view(-1)
        if override.numel() != labels.numel():
            raise ValueError("alpha_override must have one value per node")

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        group_train_counts.append(int(train_in_group.numel()))
        if override is None and train_in_group.numel() < min_train_per_group and prior_gamma is None:
            continue
        if train_in_group.numel() == 0:
            continue
        if prior_gamma is None:
            p_group = _distribution(labels, train_in_group, num_classes)
        else:
            counts = torch.bincount(labels[train_in_group].long(), minlength=num_classes).float()
            p_group = (counts + float(prior_gamma) * p_global) / (float(train_in_group.numel()) + float(prior_gamma))
        entropy = float(-(p_group * torch.log(p_group.clamp_min(1e-12))).sum().item())
        entropies.append(entropy)
        train_nodes = train_in_group
        alpha = _alpha_for_train_nodes(train_nodes, group_id, radii, node_distances, alpha_max, eps, use_confidence, override)
        alphas[train_nodes] = alpha
        y_soft[train_nodes] = (1.0 - alpha).unsqueeze(1) * y_soft[train_nodes] + alpha.unsqueeze(1) * p_group

    smoothed_train = train_idx[alphas[train_idx] > 0]
    nonzero_alpha = alphas[smoothed_train]
    train_diag = train_node_count_diagnostics(grouping, train_mask)
    coverage_by_class = _coverage_by_class(labels, train_mask, alphas)
    stats = {
        "effective_smoothing_coverage": float(smoothed_train.numel() / max(1, train_idx.numel())),
        "effective_train_nodes_smoothed": int(smoothed_train.numel()),
        "effective_smoothing_coverage_by_class": json.dumps(coverage_by_class, sort_keys=True),
        "mean_alpha_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "max_alpha_train": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "alpha_mean_all_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "alpha_mean_nonzero": float(nonzero_alpha.mean().item()) if nonzero_alpha.numel() else 0.0,
        "alpha_max_actual": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "groups_with_0_train": int(sum(1 for count in group_train_counts if count == 0)),
        "groups_with_1_train": int(sum(1 for count in group_train_counts if count == 1)),
        "groups_with_2_to_4_train": int(sum(1 for count in group_train_counts if 2 <= count <= 4)),
        "groups_with_5plus_train": int(sum(1 for count in group_train_counts if count >= 5)),
        "groups_with_2_train": int(sum(1 for count in group_train_counts if count == 2)),
        "groups_with_3plus_train": int(sum(1 for count in group_train_counts if count >= 3)),
        "soft_label_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "local_smoothing_variant": "prior" if prior_gamma is not None else "strict",
        "prior_gamma": float(prior_gamma) if prior_gamma is not None else 0.0,
        "alpha_override_used": bool(override is not None),
        "node_alpha": alphas.tolist(),
        **train_diag,
        **_alpha_distribution_stats(alphas, train_idx),
        **_target_change_stats(y_original, y_soft, alphas, train_idx),
    }
    return y_soft, stats


def _teacher_group_targets(
    clean_or_noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    grouping: dict,
    alpha_max: float,
    min_train_per_group: int,
    eps: float,
    teacher_probs: torch.Tensor,
    variant: str,
    qb_pred_beta: float,
    qb_teacher_gamma: float,
    conflict_noisy_support_max: float,
    conflict_target_conf_min: float,
    target_update_mode: str = "mix",
    sharpen_temperature: float = 0.5,
    margin_replace_threshold: float = 0.2,
    fixed_gate_alpha: bool = False,
    quality_scores: dict | None = None,
    quality_quantile: float | None = None,
    quality_weight_alpha: bool = False,
) -> tuple[torch.Tensor, dict]:
    labels = clean_or_noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    probs = teacher_probs.detach().cpu().float()
    if probs.shape != (labels.numel(), num_classes):
        raise ValueError("teacher_probs must have shape [num_nodes, num_classes]")
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    y_soft = one_hot(labels, num_classes)
    y_original = y_soft.clone()
    radii = grouping["radii"].detach().cpu().float()
    node_distances = grouping["node_distances"].detach().cpu().float()
    train_idx = mask_to_index(train_mask)
    p_global = _distribution(labels, train_idx, num_classes)
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    entropies = []
    group_train_counts = []
    groups_used = 0
    conflict_candidates = 0
    conflict_smoothed = 0
    gate_mask_by_node = torch.zeros(labels.numel(), dtype=torch.bool)
    candidate_argmax_by_node = torch.full((labels.numel(),), -1, dtype=torch.long)
    quality_state = _quality_filter_state(quality_scores, min_train_per_group, quality_quantile, quality_weight_alpha)

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        group_train_counts.append(int(train_in_group.numel()))
        if train_in_group.numel() < min_train_per_group:
            continue
        if idx.numel() == 0:
            continue

        q_label = _distribution(labels, train_in_group, num_classes)
        q_pred = probs[idx].mean(dim=0)
        q_pred = q_pred / q_pred.sum().clamp_min(1e-12)
        if variant in {"pred_mix", "conflict_gate"}:
            q_group = (1.0 - float(qb_pred_beta)) * q_label + float(qb_pred_beta) * q_pred
        elif variant == "teacher_prior":
            counts = torch.bincount(labels[train_in_group].long(), minlength=num_classes).float()
            q_group = (counts + float(qb_teacher_gamma) * q_pred) / (float(train_in_group.numel()) + float(qb_teacher_gamma))
        else:
            raise ValueError(f"Unknown teacher target variant: {variant}")
        if float(q_group.sum().item()) <= 0.0:
            q_group = p_global.clone()
        q_group = q_group / q_group.sum().clamp_min(1e-12)
        entropies.append(float(-(q_group * torch.log(q_group.clamp_min(1e-12))).sum().item()))
        groups_used += 1

        train_nodes = train_in_group
        alpha = _alpha_for_train_nodes(train_nodes, group_id, radii, node_distances, alpha_max, eps, True)
        if variant == "conflict_gate":
            if fixed_gate_alpha:
                alpha = torch.full_like(alpha, float(alpha_max))
            noisy_support = q_group[labels[train_nodes]]
            max_conf = q_group.max().repeat(train_nodes.numel())
            gate = (noisy_support <= float(conflict_noisy_support_max)) & (max_conf >= float(conflict_target_conf_min))
            if quality_state is not None:
                selected_group = bool(quality_state["selected"][group_id].item()) if group_id < quality_state["selected"].numel() else False
                group_quality = float(quality_state["group_quality"][group_id].item()) if group_id < quality_state["group_quality"].numel() else 0.0
                if target_update_mode == "quality_filter_mix":
                    if selected_group:
                        quality_state["selected_gate_count"] += int(gate.sum().item())
                    else:
                        quality_state["rejected_gate_count"] += int(gate.sum().item())
                        gate = torch.zeros_like(gate)
                elif target_update_mode == "quality_weight_mix":
                    if selected_group:
                        quality_state["selected_gate_count"] += int(gate.sum().item())
                    else:
                        quality_state["rejected_gate_count"] += int(gate.sum().item())
                    alpha = alpha * float(group_quality)
            conflict_candidates += int(gate.numel())
            q_argmax = int(q_group.argmax().item())
            gated_nodes = train_nodes[gate]
            gate_mask_by_node[gated_nodes] = True
            candidate_argmax_by_node[gated_nodes] = q_argmax
            if target_update_mode in {"mix", "quality_filter_mix", "quality_weight_mix"}:
                conflict_smoothed += int(gate.sum().item())
                alpha = alpha * gate.float()
                alphas[train_nodes] = alpha
                if train_nodes.numel():
                    y_soft[train_nodes] = (1.0 - alpha).unsqueeze(1) * y_soft[train_nodes] + alpha.unsqueeze(1) * q_group
            elif target_update_mode == "replace_qmix":
                conflict_smoothed += int(gated_nodes.numel())
                alphas[gated_nodes] = 1.0
                y_soft[gated_nodes] = q_group
            elif target_update_mode == "sharpen_qmix":
                conflict_smoothed += int(gated_nodes.numel())
                q_update = q_group.pow(1.0 / max(float(sharpen_temperature), 1e-12))
                q_update = q_update / q_update.sum().clamp_min(1e-12)
                alphas[gated_nodes] = 1.0
                y_soft[gated_nodes] = q_update
            elif target_update_mode == "margin_replace":
                top1 = int(q_group.argmax().item())
                top1_support = q_group[top1]
                noisy_support_for_margin = q_group[labels[train_nodes]]
                margin = top1_support - noisy_support_for_margin
                replace_gate = gate & (torch.full_like(labels[train_nodes], top1) != labels[train_nodes]) & (margin >= float(margin_replace_threshold))
                replace_nodes = train_nodes[replace_gate]
                conflict_smoothed += int(replace_nodes.numel())
                gate_mask_by_node[train_nodes[gate & ~replace_gate]] = False
                candidate_argmax_by_node[train_nodes[gate & ~replace_gate]] = -1
                alphas[replace_nodes] = 1.0
                y_soft[replace_nodes] = q_group
            else:
                raise ValueError(f"Unknown target_update_mode: {target_update_mode}")
            continue
        alphas[train_nodes] = alpha
        if train_nodes.numel():
            y_soft[train_nodes] = (1.0 - alpha).unsqueeze(1) * y_soft[train_nodes] + alpha.unsqueeze(1) * q_group

    smoothed_train = train_idx[alphas[train_idx] > 0]
    nonzero_alpha = alphas[smoothed_train]
    train_diag = train_node_count_diagnostics(grouping, train_mask)
    coverage_by_class = _coverage_by_class(labels, train_mask, alphas)
    stats = {
        "effective_smoothing_coverage": float(smoothed_train.numel() / max(1, train_idx.numel())),
        "effective_train_nodes_smoothed": int(smoothed_train.numel()),
        "effective_smoothing_coverage_by_class": json.dumps(coverage_by_class, sort_keys=True),
        "mean_alpha_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "max_alpha_train": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "alpha_mean_all_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "alpha_mean_nonzero": float(nonzero_alpha.mean().item()) if nonzero_alpha.numel() else 0.0,
        "alpha_max_actual": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "groups_with_0_train": int(sum(1 for count in group_train_counts if count == 0)),
        "groups_with_1_train": int(sum(1 for count in group_train_counts if count == 1)),
        "groups_with_2_to_4_train": int(sum(1 for count in group_train_counts if 2 <= count <= 4)),
        "groups_with_5plus_train": int(sum(1 for count in group_train_counts if count >= 5)),
        "groups_with_2_train": int(sum(1 for count in group_train_counts if count == 2)),
        "groups_with_3plus_train": int(sum(1 for count in group_train_counts if count >= 3)),
        "soft_label_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "local_smoothing_variant": f"teacher_{variant}",
        "teacher_target_variant": variant,
        "teacher_qb_pred_beta": float(qb_pred_beta),
        "teacher_qb_gamma": float(qb_teacher_gamma),
        "teacher_groups_used": int(groups_used),
        "target_update_mode": target_update_mode,
        "fixed_gate_alpha": bool(fixed_gate_alpha),
        "sharpen_temperature": float(sharpen_temperature) if target_update_mode == "sharpen_qmix" else math.nan,
        "margin_replace_threshold": float(margin_replace_threshold) if target_update_mode == "margin_replace" else math.nan,
        "conflict_noisy_support_max": float(conflict_noisy_support_max),
        "conflict_target_conf_min": float(conflict_target_conf_min),
        "conflict_gate_candidate_train_nodes": int(conflict_candidates),
        "conflict_gate_smoothed_train_nodes": int(conflict_smoothed),
        "node_alpha": alphas.tolist(),
        "target_update_gate_mask": gate_mask_by_node.tolist(),
        "target_update_candidate_argmax": candidate_argmax_by_node.tolist(),
        **train_diag,
        **_alpha_distribution_stats(alphas, train_idx),
        **_target_change_stats(y_original, y_soft, alphas, train_idx),
    }
    if quality_state is not None:
        stats.update(quality_state["stats"])
        stats["quality_selected_gate_count"] = int(quality_state["selected_gate_count"])
        stats["quality_rejected_gate_count"] = int(quality_state["rejected_gate_count"])
    return y_soft, stats


def _top2_margin_1d(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.detach().cpu().float()
    if probs.dim() != 2 or probs.size(1) == 0:
        return torch.zeros(probs.size(0), dtype=torch.float32)
    if probs.size(1) == 1:
        return probs[:, 0]
    top2 = torch.topk(probs, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _teacher_calibrated_group_targets(
    clean_or_noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    grouping: dict,
    alpha_max: float,
    min_train_per_group: int,
    teacher_probs: torch.Tensor,
    qb_pred_beta: float,
    conflict_noisy_support_max: float,
    conflict_target_conf_min: float,
    target_update_mode: str,
    fallback_alpha: float,
    seed: int,
    qmix_margin_min: float = 0.2,
    teacher_conf_min: float = 0.6,
    teacher_margin_min: float = 0.2,
    quality_scores: dict | None = None,
    quality_quantile: float | None = None,
    quality_gate_mode: str | None = None,
) -> tuple[torch.Tensor, dict]:
    labels = clean_or_noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    probs = teacher_probs.detach().cpu().float()
    if probs.shape != (labels.numel(), num_classes):
        raise ValueError("teacher_probs must have shape [num_nodes, num_classes]")
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    y_soft = one_hot(labels, num_classes)
    y_original = y_soft.clone()
    train_idx = mask_to_index(train_mask)
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    gate_mask_by_node = torch.zeros(labels.numel(), dtype=torch.bool)
    selected_mask_by_node = torch.zeros(labels.numel(), dtype=torch.bool)
    candidate_argmax_by_node = torch.full((labels.numel(),), -1, dtype=torch.long)
    teacher_conf = probs.max(dim=1).values
    teacher_margin = _top2_margin_1d(probs)
    group_train_counts = []
    entropies = []
    candidates = []
    conflict_candidates = 0
    quality_state = None
    if quality_gate_mode == "candidate":
        quality_state = _quality_filter_state(quality_scores, min_train_per_group, quality_quantile, False)

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        group_train_counts.append(int(train_in_group.numel()))
        if train_in_group.numel() < min_train_per_group or idx.numel() == 0:
            continue

        q_label = _distribution(labels, train_in_group, num_classes)
        q_pred = probs[idx].mean(dim=0)
        q_pred = q_pred / q_pred.sum().clamp_min(1e-12)
        q_group = (1.0 - float(qb_pred_beta)) * q_label + float(qb_pred_beta) * q_pred
        q_group = q_group / q_group.sum().clamp_min(1e-12)
        entropies.append(float(-(q_group * torch.log(q_group.clamp_min(1e-12))).sum().item()))

        noisy_support = q_group[labels[train_in_group]]
        max_conf = q_group.max().repeat(train_in_group.numel())
        gate = (noisy_support <= float(conflict_noisy_support_max)) & (max_conf >= float(conflict_target_conf_min))
        conflict_candidates += int(gate.numel())
        q_argmax = int(q_group.argmax().item())
        selected_group = True
        if quality_state is not None:
            selected_group = bool(quality_state["selected"][group_id].item()) if group_id < quality_state["selected"].numel() else False
            if selected_group:
                quality_state["selected_gate_count"] += int(gate.sum().item())
            else:
                quality_state["rejected_gate_count"] += int(gate.sum().item())
                continue
        for local_pos, node in enumerate(train_in_group[gate].tolist()):
            candidates.append({
                "node": int(node),
                "q_group": q_group.clone(),
                "target_class": q_argmax,
                "qmix_margin": float((q_group[q_argmax] - q_group[labels[int(node)]]).item()),
                "teacher_conf": float(teacher_conf[int(node)].item()),
                "teacher_margin": float(teacher_margin[int(node)].item()),
            })

    selected_indices: set[int] = set()
    if target_update_mode == "class_capped_mix":
        cap = int(math.ceil(len(candidates) / max(1, num_classes)))
        by_class: dict[int, list[tuple[float, int]]] = {}
        for idx, item in enumerate(candidates):
            by_class.setdefault(int(item["target_class"]), []).append((float(item["qmix_margin"]), idx))
        for items in by_class.values():
            for _, idx in sorted(items, reverse=True)[:cap]:
                selected_indices.add(idx)
        class_cap_m = cap
        noisy_prior_budget = [math.nan for _ in range(num_classes)]
    elif target_update_mode == "class_balanced_mix":
        train_labels = labels[train_idx] if train_idx.numel() else torch.empty(0, dtype=torch.long)
        counts = torch.bincount(train_labels, minlength=num_classes).float()
        prior = counts / counts.sum().clamp_min(1.0)
        budgets = torch.ceil(float(len(candidates)) * prior).long()
        by_class = {}
        for idx, item in enumerate(candidates):
            by_class.setdefault(int(item["target_class"]), []).append((float(item["qmix_margin"]), idx))
        for cls, items in by_class.items():
            budget = int(budgets[cls].item()) if 0 <= cls < num_classes else 0
            for _, idx in sorted(items, reverse=True)[:budget]:
                selected_indices.add(idx)
        class_cap_m = math.nan
        noisy_prior_budget = [int(v.item()) for v in budgets]
    elif target_update_mode == "boundary_safe_mix":
        for idx, item in enumerate(candidates):
            if (
                float(item["qmix_margin"]) >= float(qmix_margin_min)
                and float(item["teacher_conf"]) >= float(teacher_conf_min)
                and float(item["teacher_margin"]) >= float(teacher_margin_min)
            ):
                selected_indices.add(idx)
        class_cap_m = math.nan
        noisy_prior_budget = [math.nan for _ in range(num_classes)]
    elif target_update_mode == "replay_random_same_count":
        rng = np.random.default_rng(int(seed))
        source = candidates.copy()
        rng.shuffle(source)
        train_nodes = train_idx.tolist()
        rng.shuffle(train_nodes)
        used_nodes: set[int] = set()
        replay_items = []
        for item in source:
            target_class = int(item["target_class"])
            chosen = None
            for node in train_nodes:
                if node not in used_nodes and int(labels[node].item()) != target_class:
                    chosen = int(node)
                    break
            if chosen is None:
                for node in train_nodes:
                    if node not in used_nodes:
                        chosen = int(node)
                        break
            if chosen is None:
                break
            used_nodes.add(chosen)
            replay_items.append({**item, "node": chosen})
        candidates = replay_items
        selected_indices = set(range(len(candidates)))
        class_cap_m = math.nan
        noisy_prior_budget = [math.nan for _ in range(num_classes)]
    else:
        raise ValueError(f"Unknown calibrated target_update_mode: {target_update_mode}")

    high_strength_count = 0
    fallback_count = 0
    for idx, item in enumerate(candidates):
        node = int(item["node"])
        q_group = item["q_group"]
        gate_mask_by_node[node] = True
        candidate_argmax_by_node[node] = int(item["target_class"])
        if idx in selected_indices:
            alpha = float(alpha_max)
            selected_mask_by_node[node] = True
            high_strength_count += 1
        else:
            alpha = float(fallback_alpha)
            fallback_count += 1
        if alpha <= 0.0:
            continue
        alphas[node] = alpha
        y_soft[node] = (1.0 - alpha) * y_soft[node] + alpha * q_group

    smoothed_train = train_idx[alphas[train_idx] > 0]
    nonzero_alpha = alphas[smoothed_train]
    train_diag = train_node_count_diagnostics(grouping, train_mask)
    coverage_by_class = _coverage_by_class(labels, train_mask, alphas)
    stats = {
        "effective_smoothing_coverage": float(smoothed_train.numel() / max(1, train_idx.numel())),
        "effective_train_nodes_smoothed": int(smoothed_train.numel()),
        "effective_smoothing_coverage_by_class": json.dumps(coverage_by_class, sort_keys=True),
        "mean_alpha_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "max_alpha_train": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "alpha_mean_all_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "alpha_mean_nonzero": float(nonzero_alpha.mean().item()) if nonzero_alpha.numel() else 0.0,
        "alpha_max_actual": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "groups_with_0_train": int(sum(1 for count in group_train_counts if count == 0)),
        "groups_with_1_train": int(sum(1 for count in group_train_counts if count == 1)),
        "groups_with_2_to_4_train": int(sum(1 for count in group_train_counts if 2 <= count <= 4)),
        "groups_with_5plus_train": int(sum(1 for count in group_train_counts if count >= 5)),
        "groups_with_2_train": int(sum(1 for count in group_train_counts if count == 2)),
        "groups_with_3plus_train": int(sum(1 for count in group_train_counts if count >= 3)),
        "soft_label_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "local_smoothing_variant": "teacher_conflict_gate_calibrated",
        "teacher_target_variant": "conflict_gate",
        "teacher_qb_pred_beta": float(qb_pred_beta),
        "target_update_mode": target_update_mode,
        "fixed_gate_alpha": True,
        "fallback_alpha": float(fallback_alpha),
        "class_cap_m": float(class_cap_m) if not isinstance(class_cap_m, int) else int(class_cap_m),
        "class_balanced_budget_by_noisy_prior": json.dumps(noisy_prior_budget),
        "qmix_margin_min": float(qmix_margin_min) if target_update_mode == "boundary_safe_mix" else math.nan,
        "teacher_conf_min": float(teacher_conf_min) if target_update_mode == "boundary_safe_mix" else math.nan,
        "teacher_margin_min": float(teacher_margin_min) if target_update_mode == "boundary_safe_mix" else math.nan,
        "conflict_noisy_support_max": float(conflict_noisy_support_max),
        "conflict_target_conf_min": float(conflict_target_conf_min),
        "conflict_gate_candidate_train_nodes": int(conflict_candidates),
        "conflict_gate_smoothed_train_nodes": int(smoothed_train.numel()),
        "calibrated_candidate_count": int(len(candidates)),
        "calibrated_high_strength_count": int(high_strength_count),
        "calibrated_fallback_count": int(fallback_count),
        "node_alpha": alphas.tolist(),
        "target_update_gate_mask": gate_mask_by_node.tolist(),
        "target_update_selected_mask": selected_mask_by_node.tolist(),
        "target_update_candidate_argmax": candidate_argmax_by_node.tolist(),
        **train_diag,
        **_alpha_distribution_stats(alphas, train_idx),
        **_target_change_stats(y_original, y_soft, alphas, train_idx),
    }
    if quality_state is not None:
        stats.update(quality_state["stats"])
        stats["quality_gate_mode"] = str(quality_gate_mode)
        stats["quality_selected_gate_count"] = int(quality_state["selected_gate_count"])
        stats["quality_rejected_gate_count"] = int(quality_state["rejected_gate_count"])
    return y_soft, stats


def _coverage_aware_group_targets(
    clean_or_noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    grouping: dict,
    alpha_max: float,
    min_train_per_group: int,
    eps: float,
    fallback: str,
    prior_gamma: float,
) -> tuple[torch.Tensor, dict]:
    labels = clean_or_noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    group_ids = grouping["group_ids"].detach().cpu().long()
    radii = grouping["radii"].detach().cpu().float()
    node_distances = grouping["node_distances"].detach().cpu().float()
    p_global = _distribution(labels, train_idx, num_classes)

    one_hot_targets = one_hot(labels, num_classes)
    if fallback == "ce":
        y_soft = one_hot_targets.clone()
    elif fallback == "global":
        y_soft = one_hot_targets.clone()
        y_soft[train_idx] = (1.0 - alpha_max) * y_soft[train_idx] + alpha_max * p_global
    else:
        raise ValueError(f"Unknown coverage-aware fallback: {fallback}")

    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    group_train_counts = []
    entropies = []
    confidence_terms = []
    log_classes = math.log(max(2, int(num_classes)))

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        train_count = int(train_in_group.numel())
        group_train_counts.append(train_count)
        if train_count == 0:
            continue

        counts = torch.bincount(labels[train_in_group].long(), minlength=num_classes).float()
        if fallback == "global":
            p_group = (counts + float(prior_gamma) * p_global) / (float(train_count) + float(prior_gamma))
        else:
            p_group = counts / counts.sum().clamp_min(1.0)

        entropy = float(-(p_group * torch.log(p_group.clamp_min(1e-12))).sum().item())
        entropies.append(entropy)
        entropy_conf = max(0.0, 1.0 - entropy / log_classes)
        count_conf = min(1.0, train_count / max(1, int(min_train_per_group)))
        radius = float(radii[group_id].item()) if group_id < radii.numel() else 0.0
        ball_conf = 1.0 / (1.0 + radius)
        train_nodes = train_in_group
        node_conf = torch.exp(-node_distances[train_nodes] / (radius + eps))
        alpha = alpha_max * count_conf * entropy_conf * ball_conf * node_conf
        alphas[train_nodes] = alpha
        confidence_terms.extend(float(v) for v in alpha.tolist())
        y_soft[train_nodes] = (1.0 - alpha).unsqueeze(1) * y_soft[train_nodes] + alpha.unsqueeze(1) * p_group

    smoothed_train = train_idx[alphas[train_idx] > 0]
    nonzero_alpha = alphas[smoothed_train]
    train_diag = train_node_count_diagnostics(grouping, train_mask)
    stats = {
        "effective_smoothing_coverage": float(smoothed_train.numel() / max(1, train_idx.numel())),
        "effective_train_nodes_smoothed": int(smoothed_train.numel()),
        "effective_smoothing_coverage_by_class": json.dumps(_coverage_by_class(labels, train_mask, alphas), sort_keys=True),
        "mean_alpha_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "max_alpha_train": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "alpha_mean_all_train": float(alphas[train_idx].mean().item()) if train_idx.numel() else 0.0,
        "alpha_mean_nonzero": float(nonzero_alpha.mean().item()) if nonzero_alpha.numel() else 0.0,
        "alpha_max_actual": float(alphas[train_idx].max().item()) if train_idx.numel() else 0.0,
        "groups_with_0_train": int(sum(1 for count in group_train_counts if count == 0)),
        "groups_with_1_train": int(sum(1 for count in group_train_counts if count == 1)),
        "groups_with_2_to_4_train": int(sum(1 for count in group_train_counts if 2 <= count <= 4)),
        "groups_with_5plus_train": int(sum(1 for count in group_train_counts if count >= 5)),
        "groups_with_2_train": int(sum(1 for count in group_train_counts if count == 2)),
        "groups_with_3plus_train": int(sum(1 for count in group_train_counts if count >= 3)),
        "soft_label_entropy": float(np.mean(entropies)) if entropies else math.nan,
        "local_smoothing_variant": f"coverage_aware_{fallback}",
        "prior_gamma": float(prior_gamma) if fallback == "global" else 0.0,
        "confidence_mean_nonzero": float(np.mean(confidence_terms)) if confidence_terms else 0.0,
        **train_diag,
        **_alpha_distribution_stats(alphas, train_idx),
    }
    return y_soft, stats


def _global_targets(labels: torch.Tensor, train_mask: torch.Tensor, num_classes: int, alpha_max: float) -> tuple[torch.Tensor, dict]:
    labels = labels.detach().cpu().long()
    train_idx = mask_to_index(train_mask.detach().cpu().bool())
    y_soft = one_hot(labels, num_classes)
    p_global = _distribution(labels, train_idx, num_classes)
    y_soft[train_idx] = (1.0 - alpha_max) * y_soft[train_idx] + alpha_max * p_global
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    alphas[train_idx] = float(alpha_max)
    return y_soft, {
        "effective_smoothing_coverage": 1.0 if train_idx.numel() else 0.0,
        "effective_train_nodes_smoothed": int(train_idx.numel()),
        "effective_smoothing_coverage_by_class": json.dumps(_coverage_by_class(labels, train_mask, alphas), sort_keys=True),
        "mean_alpha_train": float(alpha_max),
        "max_alpha_train": float(alpha_max),
        "alpha_mean_all_train": float(alpha_max),
        "alpha_mean_nonzero": float(alpha_max),
        "alpha_max_actual": float(alpha_max),
        "soft_label_entropy": float(-(p_global * torch.log(p_global.clamp_min(1e-12))).sum().item()),
        **_alpha_distribution_stats(alphas, train_idx),
    }


def _topk_targets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    topk: int,
    alpha_max: float,
    min_train_per_group: int,
    match_rule: str,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, dict]:
    emb = embeddings.detach().cpu().float()
    labels = labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    y_soft = one_hot(labels, num_classes)
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    if train_idx.numel() == 0:
        return y_soft, {"effective_smoothing_coverage": 0.0, "mean_alpha_train": 0.0, "max_alpha_train": 0.0}
    sims = emb[train_idx] @ emb[train_idx].t()
    extra_self = 1 if exclude_self else 0
    k = max(1, min(int(topk) + extra_self, int(train_idx.numel())))
    neighbors = torch.topk(sims, k=k, dim=1).indices
    entropies = []
    for row, node in enumerate(train_idx):
        local_train = train_idx[neighbors[row]]
        if exclude_self:
            local_train = local_train[local_train != node]
        if local_train.numel() == 0:
            continue
        p_local = _distribution(labels, local_train, num_classes)
        alpha = float(alpha_max)
        y_soft[node] = (1.0 - alpha) * y_soft[node] + alpha * p_local
        alphas[node] = alpha
        entropies.append(float(-(p_local * torch.log(p_local.clamp_min(1e-12))).sum().item()))
    return y_soft, {
        "effective_smoothing_coverage": float((alphas[train_idx] > 0).sum().item() / max(1, train_idx.numel())),
        "effective_train_nodes_smoothed": int((alphas[train_idx] > 0).sum().item()),
        "effective_smoothing_coverage_by_class": json.dumps(_coverage_by_class(labels, train_mask, alphas), sort_keys=True),
        "mean_alpha_train": float(alphas[train_idx].mean().item()),
        "max_alpha_train": float(alphas[train_idx].max().item()),
        "alpha_mean_all_train": float(alphas[train_idx].mean().item()),
        "alpha_mean_nonzero": float(alphas[train_idx][alphas[train_idx] > 0].mean().item()) if (alphas[train_idx] > 0).any() else 0.0,
        "alpha_max_actual": float(alphas[train_idx].max().item()),
        "topk": int(topk),
        "topk_includes_self": not exclude_self,
        "topk_excludes_self": exclude_self,
        "topk_candidate_pool": "train_nodes",
        "topk_match_rule": match_rule,
        "soft_label_entropy": float(np.mean(entropies)) if entropies else math.nan,
        **_alpha_distribution_stats(alphas, train_idx),
    }


def _topk_teacher_grouping(
    embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    gb_grouping: dict,
    topk: int,
) -> dict:
    emb = embeddings.detach().cpu().float()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    groups = []
    group_ids = torch.arange(emb.size(0), dtype=torch.long)
    if train_idx.numel() > 0:
        sims = emb[train_idx] @ emb[train_idx].t()
        k = max(1, min(int(topk) + 1, int(train_idx.numel())))
        neighbor_rows = torch.topk(sims, k=k, dim=1).indices
        for row, node in enumerate(train_idx):
            local = train_idx[neighbor_rows[row]]
            local = local[local != node]
            if local.numel() == 0:
                local = torch.tensor([int(node)], dtype=torch.long)
            groups.append(local.long().tolist())
    assigned = set()
    for group in groups:
        assigned.update(int(value) for value in group)
    for node in range(emb.size(0)):
        if node not in assigned:
            groups.append([node])
    group_ids = torch.full((emb.size(0),), -1, dtype=torch.long)
    node_distances = torch.zeros(emb.size(0), dtype=torch.float32)
    centers = []
    radii = []
    for group_id, group in enumerate(groups):
        idx = torch.tensor(group, dtype=torch.long)
        points = emb[idx]
        center = points.mean(dim=0)
        dist = torch.norm(points - center, dim=1)
        centers.append(center)
        radii.append(float(dist.max().item()) if dist.numel() else 0.0)
        group_ids[idx] = group_id
        node_distances[idx] = dist.float()
    centers_tensor = torch.stack(centers) if centers else torch.empty((0, emb.size(1)))
    radii_tensor = torch.tensor(radii, dtype=torch.float32)
    diagnostics = groups_to_stats(group_ids, groups, radii_tensor)
    diagnostics.update({
        "grouping_kind": "topk_gagb_train_matched_repair",
        "topk": int(topk),
        "topk_match_rule": "mean_eligible_gagb_train_nodes_minus_1",
        "match_source": "gagb",
    })
    return {
        "group_ids": group_ids,
        "groups": groups,
        "centers": centers_tensor,
        "radii": radii_tensor,
        "node_distances": node_distances,
        "diagnostics": diagnostics,
    }


def grouping_overlap_diagnostics(gb_grouping: dict, other_grouping: dict, prefix: str) -> dict:
    gb_ids = gb_grouping["group_ids"].detach().cpu().long().numpy()
    other_ids = other_grouping["group_ids"].detach().cpu().long().numpy()
    nmi = float(normalized_mutual_info_score(gb_ids, other_ids))
    return {f"{prefix}_assignment_nmi": nmi}


def topk_gb_neighbor_overlap(
    embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    gb_grouping: dict,
    topk: int,
) -> dict:
    emb = embeddings.detach().cpu().float()
    train_idx = mask_to_index(train_mask.detach().cpu().bool())
    if train_idx.numel() == 0:
        return {"gb_topk_selected_neighbor_jaccard": 0.0, "gb_topk_selected_neighbor_overlap": 0.0}
    sims = emb[train_idx] @ emb[train_idx].t()
    k = max(1, min(int(topk) + 1, int(train_idx.numel())))
    neighbor_rows = torch.topk(sims, k=k, dim=1).indices
    group_ids = gb_grouping["group_ids"].detach().cpu().long()
    jaccards = []
    overlaps = []
    for row, node in enumerate(train_idx):
        selected = set(train_idx[neighbor_rows[row]].tolist())
        selected.discard(int(node))
        same_group = set(train_idx[group_ids[train_idx] == group_ids[node]].tolist())
        same_group.discard(int(node))
        union = selected | same_group
        inter = selected & same_group
        if union:
            jaccards.append(len(inter) / len(union))
        if selected:
            overlaps.append(len(inter) / len(selected))
    return {
        "gb_topk_selected_neighbor_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "gb_topk_selected_neighbor_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def _coverage_by_class(labels: torch.Tensor, train_mask: torch.Tensor, alphas: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    alphas = alphas.detach().cpu().float()
    out = {}
    for cls in sorted(labels[train_mask].unique().tolist()):
        cls_mask = train_mask & (labels == int(cls))
        denom = int(cls_mask.sum().item())
        if denom == 0:
            out[str(int(cls))] = 0.0
        else:
            out[str(int(cls))] = float(((alphas > 0) & cls_mask).sum().item() / denom)
    return out


def build_method_targets(
    method: str,
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    alpha_max: float,
    gb_grouping: dict | None = None,
    edge_index: torch.Tensor | None = None,
    min_train_per_ball: int = 3,
    eps: float = 1e-8,
    seed: int = 0,
    prior_gamma: float = 1.0,
    alpha_override: torch.Tensor | list[float] | None = None,
    alpha_match_reference: dict | None = None,
    teacher_probs: torch.Tensor | None = None,
    qb_pred_beta: float = 0.5,
    qb_teacher_gamma: float = 2.0,
    conflict_noisy_support_max: float = 0.35,
    conflict_target_conf_min: float = 0.55,
) -> tuple[torch.Tensor, dict]:
    method = method.lower()
    labels = noisy_labels.detach().cpu().long()
    if method == "ce":
        train_idx = mask_to_index(train_mask.detach().cpu().bool())
        return one_hot(labels, num_classes), {
            "effective_smoothing_coverage": 0.0,
            "effective_train_nodes_smoothed": 0,
            "effective_smoothing_coverage_by_class": json.dumps(_coverage_by_class(labels, train_mask, torch.zeros(labels.numel())), sort_keys=True),
            "mean_alpha_train": 0.0,
            "max_alpha_train": 0.0,
            "alpha_mean_all_train": 0.0,
            "alpha_mean_nonzero": 0.0,
            "alpha_max_actual": 0.0,
            "alpha_max": 0.0,
            "alpha_std_train": 0.0,
            "alpha_q25_train": 0.0,
            "alpha_q50_train": 0.0,
            "alpha_q75_train": 0.0,
        }
    if method == "global":
        targets, stats = _global_targets(labels, train_mask, num_classes, alpha_max)
        return targets, {**stats, "alpha_max": float(alpha_max)}
    if gb_grouping is None:
        raise ValueError("gb_grouping is required for local smoothing baselines")
    if method in {
        "gb",
        "gagb",
        "emb_only_gagb_local",
        "emb_only_gagb_global",
        "khop_only",
        "gagb_shuffled",
        "gagb_shuffled_size_only",
        "gagb_shuffled_alpha_matched",
        "grace_gagb",
        "grace_gagb_emb_only",
        "grace_gagb_topo_only",
        "grace_gagb_shuffled_alpha_matched",
        "dcgb",
        "dcgb_emb_only",
        "dcgb_topo_only",
        "dcgb_shuffled_alpha_matched",
    }:
        targets, stats = _local_group_targets(
            labels,
            train_mask,
            num_classes,
            gb_grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            use_confidence=True,
            alpha_override=alpha_override,
        )
        if method == "gagb_shuffled_alpha_matched":
            reference_alpha = alpha_match_reference.get("node_alpha") if alpha_match_reference else None
            reference_coverage = alpha_match_reference.get("effective_smoothing_coverage") if alpha_match_reference else None
            if reference_alpha is not None:
                ref = torch.tensor(reference_alpha, dtype=torch.float32)
                cur = torch.tensor(stats["node_alpha"], dtype=torch.float32)
                stats["alpha_match_max_abs_diff"] = float((cur - ref).abs().max().item()) if cur.numel() else 0.0
            else:
                stats["alpha_match_max_abs_diff"] = math.nan
            if reference_coverage is not None:
                stats["coverage_match_abs_diff"] = abs(float(stats["effective_smoothing_coverage"]) - float(reference_coverage))
            else:
                stats["coverage_match_abs_diff"] = math.nan
            reference_changed = alpha_match_reference.get("target_changed_mask") if alpha_match_reference else None
            reference_l1 = alpha_match_reference.get("target_l1_delta") if alpha_match_reference else None
            if reference_changed is not None:
                ref_changed = torch.tensor(reference_changed, dtype=torch.bool)
                cur_changed = torch.tensor(stats["target_changed_mask"], dtype=torch.bool)
                stats["target_changed_mask_mismatch_rate"] = float((cur_changed != ref_changed).float().mean().item()) if cur_changed.numel() else 0.0
            else:
                stats["target_changed_mask_mismatch_rate"] = math.nan
            if reference_l1 is not None:
                ref_l1 = torch.tensor(reference_l1, dtype=torch.float32)
                cur_l1 = torch.tensor(stats["target_l1_delta"], dtype=torch.float32)
                stats["target_l1_delta_mean_abs_diff"] = float((cur_l1 - ref_l1).abs().mean().item()) if cur_l1.numel() else 0.0
            else:
                stats["target_l1_delta_mean_abs_diff"] = math.nan
            stats["alpha_matched_source"] = "gagb"
        if method == "grace_gagb_shuffled_alpha_matched":
            reference_alpha = alpha_match_reference.get("node_alpha") if alpha_match_reference else None
            reference_coverage = alpha_match_reference.get("effective_smoothing_coverage") if alpha_match_reference else None
            if reference_alpha is not None:
                ref = torch.tensor(reference_alpha, dtype=torch.float32)
                cur = torch.tensor(stats["node_alpha"], dtype=torch.float32)
                stats["alpha_match_max_abs_diff"] = float((cur - ref).abs().max().item()) if cur.numel() else 0.0
            else:
                stats["alpha_match_max_abs_diff"] = math.nan
            if reference_coverage is not None:
                stats["coverage_match_abs_diff"] = abs(float(stats["effective_smoothing_coverage"]) - float(reference_coverage))
            else:
                stats["coverage_match_abs_diff"] = math.nan
            stats["alpha_matched_source"] = "grace_gagb"
        if method == "dcgb_shuffled_alpha_matched":
            reference_alpha = alpha_match_reference.get("node_alpha") if alpha_match_reference else None
            reference_coverage = alpha_match_reference.get("effective_smoothing_coverage") if alpha_match_reference else None
            if reference_alpha is not None:
                ref = torch.tensor(reference_alpha, dtype=torch.float32)
                cur = torch.tensor(stats["node_alpha"], dtype=torch.float32)
                stats["alpha_match_max_abs_diff"] = float((cur - ref).abs().max().item()) if cur.numel() else 0.0
            else:
                stats["alpha_match_max_abs_diff"] = math.nan
            if reference_coverage is not None:
                stats["coverage_match_abs_diff"] = abs(float(stats["effective_smoothing_coverage"]) - float(reference_coverage))
            else:
                stats["coverage_match_abs_diff"] = math.nan
            stats["alpha_matched_source"] = "dcgb"
        return targets, {**stats, **gb_grouping["diagnostics"], "alpha_max": float(alpha_max)}
    if method in {"gagb_qb_pred_mix", "gagb_qb_teacher_prior", "gagb_qb_conflict_gate"} | TEACHER_REPAIR_METHODS:
        if teacher_probs is None:
            raise ValueError(f"{method} requires teacher_probs")
        control_config = REPAIR_CONTROL_CONFIGS.get(method)
        effective_method = control_config["base_method"] if control_config else method
        variant = {
            "gagb_qb_pred_mix": "pred_mix",
            "gagb_qb_teacher_prior": "teacher_prior",
            "gagb_qb_conflict_gate": "conflict_gate",
        }.get(effective_method, "conflict_gate")
        repair_config = CONFLICT_GATE_REPAIR_CONFIGS.get(effective_method, {})
        effective_grouping = gb_grouping
        extra_stats = {}
        if control_config and control_config["grouping_source"] == "kmeans_gagb_matched":
            effective_grouping = kmeans_groups(embeddings, gb_grouping["diagnostics"]["ball_count"], seed=seed)
            extra_stats.update(grouping_overlap_diagnostics(gb_grouping, effective_grouping, "gagb_kmeans"))
            extra_stats.update({"match_source": "gagb", "matched_grouping_kind": gb_grouping.get("diagnostics", {}).get("grouping_kind", "gagb")})
        elif control_config and control_config["grouping_source"] == "topk_gagb_train_matched":
            topk = build_topk_group_size(
                gb_grouping["groups"],
                train_mask=train_mask,
                min_train_per_group=min_train_per_ball,
                match_train_nodes=True,
            )
            effective_grouping = _topk_teacher_grouping(embeddings, train_mask, gb_grouping, topk)
            extra_stats.update(topk_gb_neighbor_overlap(embeddings, train_mask, gb_grouping, topk))
        elif control_config:
            extra_stats["repair_control_grouping_source"] = control_config["grouping_source"]
        effective_alpha_max = float(repair_config.get("alpha_max", alpha_max))
        target_update_mode = str(repair_config.get("target_update_mode", "mix"))
        quality_scores = None
        if target_update_mode in {"quality_filter_mix", "quality_weight_mix"} or repair_config.get("quality_gate_mode") == "candidate":
            if edge_index is None:
                raise ValueError(f"{method} requires edge_index for quality filtering")
            quality_scores = compute_group_quality_scores(
                embeddings,
                edge_index,
                labels,
                train_mask,
                effective_grouping,
                num_classes,
                min_train_per_ball=min_train_per_ball,
            )
        if target_update_mode in {"class_capped_mix", "class_balanced_mix", "boundary_safe_mix", "replay_random_same_count"}:
            targets, stats = _teacher_calibrated_group_targets(
                labels,
                train_mask,
                num_classes,
                effective_grouping,
                effective_alpha_max,
                min_train_per_ball,
                teacher_probs,
                qb_pred_beta,
                conflict_noisy_support_max,
                conflict_target_conf_min,
                target_update_mode=target_update_mode,
                fallback_alpha=float(repair_config.get("fallback_alpha", 0.3)),
                seed=seed,
                qmix_margin_min=float(repair_config.get("qmix_margin_min", 0.2)),
                teacher_conf_min=float(repair_config.get("teacher_conf_min", 0.6)),
                teacher_margin_min=float(repair_config.get("teacher_margin_min", 0.2)),
                quality_scores=quality_scores,
                quality_quantile=repair_config.get("quality_quantile"),
                quality_gate_mode=repair_config.get("quality_gate_mode"),
            )
        else:
            targets, stats = _teacher_group_targets(
                labels,
                train_mask,
                num_classes,
                effective_grouping,
                effective_alpha_max,
                min_train_per_ball,
                eps,
                teacher_probs,
                variant,
                qb_pred_beta,
                qb_teacher_gamma,
                conflict_noisy_support_max,
                conflict_target_conf_min,
                target_update_mode=target_update_mode,
                sharpen_temperature=float(repair_config.get("sharpen_temperature", 0.5)),
                margin_replace_threshold=float(repair_config.get("margin_replace_threshold", 0.2)),
                fixed_gate_alpha=bool(repair_config.get("fixed_gate_alpha", False)),
                quality_scores=quality_scores,
                quality_quantile=repair_config.get("quality_quantile"),
                quality_weight_alpha=target_update_mode == "quality_weight_mix",
            )
        return targets, {
            **stats,
            **effective_grouping["diagnostics"],
            **extra_stats,
            "repair_base_method": effective_method,
            "repair_control_grouping_source": control_config["grouping_source"] if control_config else "gagb",
            "alpha_max": effective_alpha_max,
        }
    if method == "gb_prior":
        targets, stats = _local_group_targets(
            labels,
            train_mask,
            num_classes,
            gb_grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            use_confidence=True,
            prior_gamma=prior_gamma,
        )
        return targets, {**stats, **gb_grouping["diagnostics"], "alpha_max": float(alpha_max)}
    if method in {"gb_ca_ce", "gagb_ca_ce"}:
        targets, stats = _coverage_aware_group_targets(
            labels,
            train_mask,
            num_classes,
            gb_grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            fallback="ce",
            prior_gamma=prior_gamma,
        )
        return targets, {**stats, **gb_grouping["diagnostics"], "alpha_max": float(alpha_max)}
    if method == "gb_ca_global":
        targets, stats = _coverage_aware_group_targets(
            labels,
            train_mask,
            num_classes,
            gb_grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            fallback="global",
            prior_gamma=prior_gamma,
        )
        return targets, {**stats, **gb_grouping["diagnostics"], "alpha_max": float(alpha_max)}
    if method in {"kmeans", "kmeans_gagb_matched", "kmeans_grace_gagb_matched", "kmeans_dcgb_matched"}:
        grouping = kmeans_groups(embeddings, gb_grouping["diagnostics"]["ball_count"], seed=seed)
        targets, stats = _local_group_targets(
            labels,
            train_mask,
            num_classes,
            grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            use_confidence=True,
        )
        if method == "kmeans_dcgb_matched":
            match_source = "dcgb"
        elif method == "kmeans_grace_gagb_matched":
            match_source = "grace_gagb"
        elif method == "kmeans_gagb_matched":
            match_source = "gagb"
        else:
            match_source = "gb"
        overlap_stats = grouping_overlap_diagnostics(gb_grouping, grouping, f"{match_source}_kmeans")
        return targets, {
            **stats,
            **grouping["diagnostics"],
            **overlap_stats,
            "matched_grouping_kind": gb_grouping.get("diagnostics", {}).get("grouping_kind", match_source),
            "match_source": match_source,
            "alpha_max": float(alpha_max),
        }
    if method == "topk":
        topk = build_topk_group_size(gb_grouping["groups"])
        targets, stats = _topk_targets(
            embeddings,
            labels,
            train_mask,
            num_classes,
            topk,
            alpha_max,
            min_train_per_ball,
            match_rule="avg_non_singleton_ball_size_minus_1",
            exclude_self=True,
        )
        overlap_stats = topk_gb_neighbor_overlap(embeddings, train_mask, gb_grouping, topk)
        return targets, {**stats, **overlap_stats, "alpha_max": float(alpha_max)}
    if method in {
        "topk_matched",
        "topk_gagb_matched",
        "topk_gagb_train_matched",
        "topk_gagb_allnode_matched",
        "topk_grace_gagb_train_matched",
        "topk_dcgb_train_matched",
    }:
        match_all_nodes = method == "topk_gagb_allnode_matched"
        topk = build_topk_group_size(
            gb_grouping["groups"],
            train_mask=train_mask,
            min_train_per_group=min_train_per_ball,
            match_train_nodes=not match_all_nodes,
        )
        if method == "topk_gagb_allnode_matched":
            match_rule = "gagb_allnode_avg_ball_size_minus_1"
        elif method == "topk_dcgb_train_matched":
            match_rule = "mean_eligible_dcgb_train_nodes_minus_1"
        elif method == "topk_grace_gagb_train_matched":
            match_rule = "mean_eligible_grace_gagb_train_nodes_minus_1"
        elif method in {"topk_gagb_matched", "topk_gagb_train_matched"}:
            match_rule = "mean_eligible_gagb_train_nodes_minus_1"
        else:
            match_rule = "mean_eligible_gb_train_nodes_minus_1"
        targets, stats = _topk_targets(
            embeddings,
            labels,
            train_mask,
            num_classes,
            topk,
            alpha_max,
            min_train_per_ball,
            match_rule=match_rule,
            exclude_self=True,
        )
        overlap_stats = topk_gb_neighbor_overlap(embeddings, train_mask, gb_grouping, topk)
        return targets, {
            **stats,
            **overlap_stats,
            "match_source": (
                "dcgb"
                if method == "topk_dcgb_train_matched"
                else (
                    "grace_gagb"
                    if method == "topk_grace_gagb_train_matched"
                    else ("gagb" if method in {"topk_gagb_matched", "topk_gagb_train_matched", "topk_gagb_allnode_matched"} else "gb")
                )
            ),
            "alpha_max": float(alpha_max),
        }
    raise ValueError(f"Unknown smoothing method: {method}")
