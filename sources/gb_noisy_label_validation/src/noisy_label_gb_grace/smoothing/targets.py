from __future__ import annotations

import json
import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from noisy_label_gb_grace.balls.granular_ball import groups_to_stats
from noisy_label_gb_grace.utils import mask_to_index, one_hot


def _distribution(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    counts = torch.bincount(labels[indices].long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


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
) -> tuple[torch.Tensor, dict]:
    labels = clean_or_noisy_labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    y_soft = one_hot(labels, num_classes)
    group_ids = grouping["group_ids"].detach().cpu().long()
    radii = grouping["radii"].detach().cpu().float()
    node_distances = grouping["node_distances"].detach().cpu().float()
    alphas = torch.zeros(labels.numel(), dtype=torch.float32)
    train_idx = mask_to_index(train_mask)
    group_train_counts = []
    entropies = []
    p_global = _distribution(labels, train_idx, num_classes)

    for group_id, group in enumerate(grouping["groups"]):
        idx = torch.tensor(group, dtype=torch.long)
        train_in_group = idx[train_mask[idx]]
        group_train_counts.append(int(train_in_group.numel()))
        if train_in_group.numel() < min_train_per_group and prior_gamma is None:
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
        if use_confidence:
            radius = float(radii[group_id].item()) if group_id < radii.numel() else 0.0
            ball_conf = 1.0 / (1.0 + radius)
            node_conf = torch.exp(-node_distances[train_nodes] / (radius + eps))
            alpha = alpha_max * ball_conf * node_conf
        else:
            alpha = torch.full((train_nodes.numel(),), float(alpha_max))
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
        **train_diag,
    }
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
    min_train_per_ball: int = 3,
    eps: float = 1e-8,
    seed: int = 0,
    prior_gamma: float = 1.0,
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
        }
    if method == "global":
        targets, stats = _global_targets(labels, train_mask, num_classes, alpha_max)
        return targets, {**stats, "alpha_max": float(alpha_max)}
    if gb_grouping is None:
        raise ValueError("gb_grouping is required for local smoothing baselines")
    if method == "gb":
        targets, stats = _local_group_targets(
            labels,
            train_mask,
            num_classes,
            gb_grouping,
            alpha_max,
            min_train_per_ball,
            eps,
            use_confidence=True,
        )
        return targets, {**stats, **gb_grouping["diagnostics"], "alpha_max": float(alpha_max)}
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
    if method == "gb_ca_ce":
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
    if method == "kmeans":
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
        overlap_stats = grouping_overlap_diagnostics(gb_grouping, grouping, "gb_kmeans")
        return targets, {**stats, **grouping["diagnostics"], **overlap_stats, "alpha_max": float(alpha_max)}
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
    if method == "topk_matched":
        topk = build_topk_group_size(
            gb_grouping["groups"],
            train_mask=train_mask,
            min_train_per_group=min_train_per_ball,
            match_train_nodes=True,
        )
        targets, stats = _topk_targets(
            embeddings,
            labels,
            train_mask,
            num_classes,
            topk,
            alpha_max,
            min_train_per_ball,
            match_rule="mean_eligible_gb_train_nodes_minus_1",
            exclude_self=True,
        )
        overlap_stats = topk_gb_neighbor_overlap(embeddings, train_mask, gb_grouping, topk)
        return targets, {**stats, **overlap_stats, "alpha_max": float(alpha_max)}
    raise ValueError(f"Unknown smoothing method: {method}")
