from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from noisy_label_gb_grace.smoothing.targets import (
    build_topk_group_size,
    grouping_overlap_diagnostics,
    kmeans_groups,
    topk_gb_neighbor_overlap,
)
from noisy_label_gb_grace.utils import mask_to_index


WEIGHTED_METHODS = {"gb_weighted_ce", "kmeans_weighted_ce", "topk_matched_weighted_ce"}


def _normalise(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)


def _angular_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    sim = (left * right).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(sim) / math.pi


def _distribution(labels: torch.Tensor, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros(num_classes, dtype=torch.float32)
    counts = torch.bincount(labels[indices].long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1.0)


def _normalised_entropy(probs: torch.Tensor) -> float:
    if probs.numel() <= 1 or float(probs.sum().item()) <= 0.0:
        return 0.0
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
    return entropy / math.log(float(probs.numel()))


def _group_train_count_distribution(grouping: dict, train_mask: torch.Tensor) -> str:
    counts = []
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        counts.append(int(train_mask[idx].sum().item()))
    return json.dumps(counts)


def _group_entropy_distribution(grouping: dict, labels: torch.Tensor, train_mask: torch.Tensor, num_classes: int) -> str:
    entropies = []
    for group in grouping["groups"]:
        idx = torch.tensor(group, dtype=torch.long)
        local = idx[train_mask[idx]]
        probs = _distribution(labels, local, num_classes)
        entropies.append(_normalised_entropy(probs))
    return json.dumps(entropies)


def _groups_from_topk(embeddings: torch.Tensor, train_mask: torch.Tensor, topk: int) -> dict[int, torch.Tensor]:
    emb = _normalise(embeddings)
    train_idx = mask_to_index(train_mask)
    if train_idx.numel() == 0:
        return {}
    sims = emb[train_idx] @ emb[train_idx].t()
    k = max(1, min(int(topk) + 1, int(train_idx.numel())))
    neighbor_rows = torch.topk(sims, k=k, dim=1).indices
    out = {}
    for row, node in enumerate(train_idx):
        local = train_idx[neighbor_rows[row]]
        local = local[local != node]
        out[int(node.item())] = local
    return out


def _region_geometry(emb: torch.Tensor, node: torch.Tensor, region: torch.Tensor, eps: float) -> float:
    if region.numel() == 0:
        return 0.0
    points = emb[region]
    center = F.normalize(points.mean(dim=0, keepdim=True), p=2, dim=1).view(-1)
    node_distance = _angular_distance(emb[node].view(1, -1), center.view(1, -1))[0]
    region_distances = _angular_distance(points, center.view(1, -1).expand_as(points))
    radius = region_distances.mean().clamp_min(float(eps))
    return float(torch.exp(-node_distance / radius).item())


def _build_group_method_regions(method: str, embeddings: torch.Tensor, gb_grouping: dict, seed: int) -> tuple[dict, dict]:
    if method == "gb_weighted_ce":
        return gb_grouping, dict(gb_grouping["diagnostics"])
    if method == "kmeans_weighted_ce":
        grouping = kmeans_groups(embeddings, gb_grouping["diagnostics"]["ball_count"], seed=seed)
        stats = {**grouping["diagnostics"], **grouping_overlap_diagnostics(gb_grouping, grouping, "gb_kmeans")}
        return grouping, stats
    raise ValueError(f"Unsupported group method: {method}")


def _compute_group_weights(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    grouping: dict,
    min_train_per_region: int,
    raw_weight_min: float,
    final_weight_low: float,
    final_weight_high: float,
    eps: float,
    extra_stats: dict,
) -> tuple[torch.Tensor, dict]:
    emb = _normalise(embeddings)
    labels = labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    group_ids = grouping["group_ids"].detach().cpu().long()
    weights = torch.ones(labels.numel(), dtype=torch.float32)
    raw = torch.ones(labels.numel(), dtype=torch.float32)
    g_cov = torch.zeros(labels.numel(), dtype=torch.float32)
    g_ent = torch.zeros(labels.numel(), dtype=torch.float32)
    g_sup = torch.zeros(labels.numel(), dtype=torch.float32)
    g_geo = torch.zeros(labels.numel(), dtype=torch.float32)

    for node in train_idx:
        gid = int(group_ids[node].item())
        region = (group_ids == gid).nonzero(as_tuple=False).view(-1)
        local_train = region[train_mask[region]]
        local_train = local_train[local_train != node]
        probs = _distribution(labels, local_train, num_classes)
        coverage = min(1.0, float(local_train.numel()) / max(1, int(min_train_per_region)))
        entropy_gate = 1.0 - _normalised_entropy(probs)
        support_gate = float(probs[labels[node]].item()) if probs.numel() else 0.0
        geometry_gate = _region_geometry(emb, node, region, eps)
        gate = coverage * entropy_gate * support_gate * geometry_gate
        g_cov[node] = coverage
        g_ent[node] = entropy_gate
        g_sup[node] = support_gate
        g_geo[node] = geometry_gate
        raw[node] = min(1.0, max(float(raw_weight_min), float(gate)))

    if train_idx.numel() > 0:
        train_raw_mean = raw[train_idx].mean().clamp_min(float(eps))
        weights[train_idx] = (raw[train_idx] / train_raw_mean).clamp(float(final_weight_low), float(final_weight_high))
    stats = _weight_stats(weights, train_idx, raw, g_cov, g_ent, g_sup, g_geo, final_weight_low, final_weight_high)
    stats.update(extra_stats)
    stats.update({
        "weighting_method": "reliability_weighted_ce",
        "min_train_per_region": int(min_train_per_region),
        "raw_weight_min": float(raw_weight_min),
        "final_weight_low": float(final_weight_low),
        "final_weight_high": float(final_weight_high),
        "ball_train_label_count_distribution": _group_train_count_distribution(grouping, train_mask),
        "ball_noisy_label_entropy_distribution": _group_entropy_distribution(grouping, labels, train_mask, num_classes),
    })
    return weights, stats


def _compute_topk_weights(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    gb_grouping: dict,
    min_train_per_region: int,
    raw_weight_min: float,
    final_weight_low: float,
    final_weight_high: float,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    emb = _normalise(embeddings)
    labels = labels.detach().cpu().long()
    train_mask = train_mask.detach().cpu().bool()
    train_idx = mask_to_index(train_mask)
    topk = build_topk_group_size(
        gb_grouping["groups"],
        train_mask=train_mask,
        min_train_per_group=min_train_per_region,
        match_train_nodes=True,
    )
    regions = _groups_from_topk(emb, train_mask, topk)
    weights = torch.ones(labels.numel(), dtype=torch.float32)
    raw = torch.ones(labels.numel(), dtype=torch.float32)
    g_cov = torch.zeros(labels.numel(), dtype=torch.float32)
    g_ent = torch.zeros(labels.numel(), dtype=torch.float32)
    g_sup = torch.zeros(labels.numel(), dtype=torch.float32)
    g_geo = torch.zeros(labels.numel(), dtype=torch.float32)

    for node in train_idx:
        local_train = regions.get(int(node.item()), torch.empty(0, dtype=torch.long))
        probs = _distribution(labels, local_train, num_classes)
        coverage = min(1.0, float(local_train.numel()) / max(1, int(min_train_per_region)))
        entropy_gate = 1.0 - _normalised_entropy(probs)
        support_gate = float(probs[labels[node]].item()) if probs.numel() else 0.0
        geometry_gate = _region_geometry(emb, node, local_train, eps)
        gate = coverage * entropy_gate * support_gate * geometry_gate
        g_cov[node] = coverage
        g_ent[node] = entropy_gate
        g_sup[node] = support_gate
        g_geo[node] = geometry_gate
        raw[node] = min(1.0, max(float(raw_weight_min), float(gate)))

    if train_idx.numel() > 0:
        train_raw_mean = raw[train_idx].mean().clamp_min(float(eps))
        weights[train_idx] = (raw[train_idx] / train_raw_mean).clamp(float(final_weight_low), float(final_weight_high))
    stats = _weight_stats(weights, train_idx, raw, g_cov, g_ent, g_sup, g_geo, final_weight_low, final_weight_high)
    stats.update(topk_gb_neighbor_overlap(emb, train_mask, gb_grouping, topk))
    stats.update({
        "weighting_method": "reliability_weighted_ce",
        "topk": int(topk),
        "topk_match_rule": "mean_eligible_gb_train_nodes_minus_1",
        "topk_candidate_pool": "train_nodes",
        "min_train_per_region": int(min_train_per_region),
        "raw_weight_min": float(raw_weight_min),
        "final_weight_low": float(final_weight_low),
        "final_weight_high": float(final_weight_high),
    })
    return weights, stats


def _weight_stats(
    weights: torch.Tensor,
    train_idx: torch.Tensor,
    raw: torch.Tensor,
    g_cov: torch.Tensor,
    g_ent: torch.Tensor,
    g_sup: torch.Tensor,
    g_geo: torch.Tensor,
    final_weight_low: float,
    final_weight_high: float,
) -> dict:
    if train_idx.numel() == 0:
        return {
            "weight_mean": 0.0,
            "weight_std": 0.0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "clip_low_ratio": 0.0,
            "clip_high_ratio": 0.0,
            "low_weight_ratio": 0.0,
            "raw_weight_mean": 0.0,
            "coverage_gate_mean": 0.0,
            "entropy_gate_mean": 0.0,
            "support_gate_mean": 0.0,
            "geometry_gate_mean": 0.0,
        }
    train_w = weights[train_idx].float()
    return {
        "weight_mean": float(train_w.mean().item()),
        "weight_std": float(train_w.std(unbiased=False).item()),
        "weight_min": float(train_w.min().item()),
        "weight_max": float(train_w.max().item()),
        "clip_low_ratio": float((train_w <= float(final_weight_low) + 1e-8).float().mean().item()),
        "clip_high_ratio": float((train_w >= float(final_weight_high) - 1e-8).float().mean().item()),
        "low_weight_ratio": float((train_w < 1.0).float().mean().item()),
        "raw_weight_mean": float(raw[train_idx].mean().item()),
        "coverage_gate_mean": float(g_cov[train_idx].mean().item()),
        "entropy_gate_mean": float(g_ent[train_idx].mean().item()),
        "support_gate_mean": float(g_sup[train_idx].mean().item()),
        "geometry_gate_mean": float(g_geo[train_idx].mean().item()),
    }


def build_reliability_weights(
    method: str,
    embeddings: torch.Tensor,
    noisy_labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    gb_grouping: dict,
    min_train_per_region: int = 3,
    raw_weight_min: float = 0.3,
    final_weight_low: float = 0.5,
    final_weight_high: float = 1.5,
    eps: float = 1e-8,
    seed: int = 0,
) -> tuple[torch.Tensor, dict]:
    method = method.lower()
    if method not in WEIGHTED_METHODS:
        raise ValueError(f"Unknown weighted CE method: {method}")
    if method in {"gb_weighted_ce", "kmeans_weighted_ce"}:
        grouping, stats = _build_group_method_regions(method, embeddings, gb_grouping, seed)
        weights, weight_stats = _compute_group_weights(
            embeddings,
            noisy_labels,
            train_mask,
            num_classes,
            grouping,
            min_train_per_region,
            raw_weight_min,
            final_weight_low,
            final_weight_high,
            eps,
            stats,
        )
    else:
        weights, weight_stats = _compute_topk_weights(
            embeddings,
            noisy_labels,
            train_mask,
            num_classes,
            gb_grouping,
            min_train_per_region,
            raw_weight_min,
            final_weight_low,
            final_weight_high,
            eps,
        )
    return weights, {**weight_stats, "method_family": "weighted_ce"}
