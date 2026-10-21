from __future__ import annotations

import math
from collections import deque

import torch
import torch.nn.functional as F

from noisy_label_gb_grace.balls.granular_ball import groups_to_stats


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    if edge_index is None:
        return adjacency
    edge_index = edge_index.detach().cpu().long()
    if edge_index.numel() == 0:
        return adjacency
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a [2, num_edges] tensor")
    for src, dst in edge_index.t().tolist():
        if 0 <= src < num_nodes and 0 <= dst < num_nodes and src != dst:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _khop_distances(center: int, adjacency: list[set[int]], max_hops: int, unassigned: set[int]) -> dict[int, int]:
    if center not in unassigned:
        return {}
    distances = {center: 0}
    queue: deque[int] = deque([center])
    while queue:
        node = queue.popleft()
        hop = distances[node]
        if hop >= max_hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor not in unassigned or neighbor in distances:
                continue
            distances[neighbor] = hop + 1
            queue.append(neighbor)
    return distances


def _group_geometry(embeddings: torch.Tensor, groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = []
    radii = []
    node_distances = torch.zeros(embeddings.size(0), dtype=torch.float32)
    for group in groups:
        idx = torch.tensor(group, dtype=torch.long)
        points = embeddings[idx]
        center = points.mean(dim=0)
        distances = torch.norm(points - center, dim=1)
        centers.append(center)
        radii.append(float(distances.max().item()) if distances.numel() else 0.0)
        node_distances[idx] = distances.float()
    if centers:
        return torch.stack(centers), torch.tensor(radii, dtype=torch.float32), node_distances
    return torch.empty((0, embeddings.size(1))), torch.empty(0, dtype=torch.float32), node_distances


def _payload_from_groups(
    embeddings: torch.Tensor,
    groups: list[list[int]],
    config: dict,
    diagnostics_extra: dict | None = None,
) -> dict:
    group_ids = torch.full((embeddings.size(0),), -1, dtype=torch.long)
    normalised_groups = []
    for group_id, group in enumerate(groups):
        unique_group = sorted({int(node) for node in group})
        for node in unique_group:
            group_ids[node] = group_id
        normalised_groups.append(unique_group)
    centers, radii, node_distances = _group_geometry(embeddings, normalised_groups)
    diagnostics = groups_to_stats(group_ids, normalised_groups, radii)
    if diagnostics_extra:
        diagnostics.update(diagnostics_extra)
    return {
        "group_ids": group_ids,
        "groups": normalised_groups,
        "centers": centers,
        "radii": radii,
        "node_distances": node_distances,
        "diagnostics": diagnostics,
        "config": config,
    }


def _connected_component_count(group: list[int], adjacency: list[set[int]]) -> int:
    members = set(int(node) for node in group)
    if not members:
        return 0
    remaining = set(members)
    count = 0
    while remaining:
        count += 1
        start = remaining.pop()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def _row_normalized_adjacency(adjacency: list[set[int]], num_nodes: int) -> torch.Tensor:
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for node, neighbors in enumerate(adjacency):
        matrix[node, node] = 1.0
        for neighbor in neighbors:
            matrix[node, int(neighbor)] = 1.0
    row_sum = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return matrix / row_sum


def _ppr_lite_matrix(
    adjacency: list[set[int]],
    num_nodes: int,
    alpha: float,
    iters: int,
) -> torch.Tensor:
    transition = _row_normalized_adjacency(adjacency, num_nodes)
    eye = torch.eye(num_nodes, dtype=torch.float32)
    ppr = eye.clone()
    restart = float(alpha) * eye
    keep = 1.0 - float(alpha)
    for _ in range(int(iters)):
        ppr = restart + keep * (ppr @ transition)
    ppr.fill_diagonal_(1.0)
    row_max = ppr.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    return (ppr / row_max).clamp(0.0, 1.0)


def _semantic_density(affinity: torch.Tensor, topk: int) -> torch.Tensor:
    num_nodes = int(affinity.size(0))
    if num_nodes <= 1:
        return torch.ones(num_nodes, dtype=torch.float32)
    k = max(1, min(int(topk) + 1, num_nodes))
    values = torch.topk(affinity, k=k, dim=1).values
    if values.size(1) > 1:
        values = values[:, 1:]
    return values.mean(dim=1)


def _khop_score_row(center: int, adjacency: list[set[int]], max_hops: int, unassigned: set[int]) -> torch.Tensor:
    scores = torch.zeros(len(adjacency), dtype=torch.float32)
    distances = _khop_distances(center, adjacency, max_hops, unassigned)
    for node, distance in distances.items():
        scores[int(node)] = 1.0 if int(distance) == 0 else 1.0 / float(distance)
    return scores


def _ball_quality(
    members: list[int],
    affinity: torch.Tensor,
    topo_row: torch.Tensor,
    center: int,
    *,
    use_embedding: bool,
    use_topology: bool,
    comp_power: float,
    topo_power: float,
    size_m0: float,
    topo_eps: float,
) -> tuple[float, float, float]:
    idx = torch.tensor(members, dtype=torch.long)
    if idx.numel() == 0:
        return 0.0, 0.0, 0.0
    center_affinity = affinity[idx][:, idx].mean(dim=1)
    comp = float(center_affinity.mean().item()) if use_embedding else 1.0
    topo_values = topo_row[idx]
    topo = float(topo_values.mean().item()) if use_topology else 1.0
    size_gate = 1.0 - math.exp(-max(0, int(idx.numel()) - 1) / max(float(size_m0), 1e-6))
    quality = size_gate
    if use_embedding:
        quality *= max(comp, 1e-12) ** float(comp_power)
    if use_topology:
        quality *= (float(topo_eps) + max(topo, 0.0)) ** float(topo_power)
    return float(quality), float(comp), float(topo)


def build_grace_first_graph_balls(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    max_hops: int = 2,
    min_ball_size: int = 5,
    max_ball_size: int = 128,
    seed: int = 0,
    mode: str = "grace_first",
    topology_mode: str = "ppr_lite",
    ppr_alpha: float = 0.15,
    ppr_iters: int = 10,
    embedding_top_m: int = 96,
    topology_top_m: int = 64,
    center_topk: int = 32,
    tau_h: float = 0.68,
    tau_h_strong: float = 0.82,
    tau_topology: float = 0.05,
    tau_add: float = 0.35,
    lambda_h: float = 0.75,
    lambda_t: float = 0.25,
    topo_eps: float = 0.05,
    delta_q: float = 0.01,
    size_m0: float = 5.0,
    teacher_probs: torch.Tensor | None = None,
    lambda_p: float = 0.0,
) -> dict:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be a [num_nodes, dim] tensor")
    if mode not in {"grace_first", "emb_only", "topo_only"}:
        raise ValueError("mode must be one of 'grace_first', 'emb_only', or 'topo_only'")
    if topology_mode not in {"ppr_lite", "khop"}:
        raise ValueError("topology_mode must be 'ppr_lite' or 'khop'")
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    if min_ball_size < 1 or max_ball_size < 1:
        raise ValueError("ball sizes must be positive")

    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    num_nodes = int(emb.size(0))
    adjacency = _build_adjacency(edge_index, num_nodes)
    degrees = torch.tensor([len(neighbors) for neighbors in adjacency], dtype=torch.float32)
    cosine = (emb @ emb.t()).clamp(-1.0, 1.0)
    affinity = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)
    density = _semantic_density(affinity, int(center_topk))
    center_scores_all = density * torch.log1p(degrees)

    topology_matrix = None
    resolved_topology_mode = topology_mode
    if topology_mode == "ppr_lite":
        try:
            topology_matrix = _ppr_lite_matrix(adjacency, num_nodes, ppr_alpha, ppr_iters)
        except RuntimeError:
            resolved_topology_mode = "khop"

    use_embedding = mode in {"grace_first", "emb_only"}
    use_topology = mode in {"grace_first", "topo_only"}
    comp_power = float(lambda_h) if use_embedding else 0.0
    topo_power = float(lambda_t) if use_topology else 0.0
    probs = teacher_probs.detach().cpu().float() if teacher_probs is not None and float(lambda_p) > 0.0 else None

    generator = torch.Generator().manual_seed(int(seed))
    random_tie_break = torch.rand(num_nodes, generator=generator)
    unassigned = set(range(num_nodes))
    groups: list[list[int]] = []
    qualities = []
    comps = []
    topos = []
    low_quality_count = 0
    connected_count = 0
    component_counts = []
    accepted_noncenter = 0
    embedding_hits = 0
    topology_supported_hits = 0

    while unassigned:
        available = torch.tensor(sorted(unassigned), dtype=torch.long)
        center_scores = center_scores_all[available] + 1e-6 * random_tie_break[available]
        center = int(available[int(torch.argmax(center_scores).item())].item())
        if topology_matrix is not None:
            topo_row = topology_matrix[center].clone()
        else:
            topo_row = _khop_score_row(center, adjacency, max_hops, unassigned)

        unassigned_list = torch.tensor(sorted(unassigned), dtype=torch.long)
        emb_scores = affinity[center, unassigned_list].clone()
        topo_scores = topo_row[unassigned_list].clone()
        emb_k = max(1, min(int(embedding_top_m) + 1, int(unassigned_list.numel())))
        topo_k = max(1, min(int(topology_top_m) + 1, int(unassigned_list.numel())))
        emb_top_nodes = set(unassigned_list[torch.topk(emb_scores, k=emb_k).indices].tolist())
        topo_top_nodes = set(unassigned_list[torch.topk(topo_scores, k=topo_k).indices].tolist())
        hop_nodes = set(_khop_distances(center, adjacency, max_hops, unassigned))
        candidate_set = sorted((emb_top_nodes | topo_top_nodes | hop_nodes | {center}) & unassigned)
        candidate_nodes = torch.tensor(candidate_set, dtype=torch.long)

        h_sim = affinity[center, candidate_nodes]
        t_sim = topo_row[candidate_nodes].clamp(0.0, 1.0)
        if mode == "emb_only":
            scores = h_sim
            accepted_mask = h_sim >= float(tau_h)
        elif mode == "topo_only":
            scores = t_sim
            accepted_mask = t_sim >= float(tau_topology)
        else:
            scores = (h_sim.clamp_min(1e-12) ** float(lambda_h)) * ((float(topo_eps) + t_sim).clamp_min(1e-12) ** float(lambda_t))
            topo_supported = t_sim >= float(tau_topology)
            accepted_mask = (h_sim >= float(tau_h)) & (topo_supported | (h_sim >= float(tau_h_strong))) & (scores >= float(tau_add))
        if probs is not None:
            pred_sim = (probs[candidate_nodes] * probs[center].view(1, -1)).sum(dim=1).clamp(0.0, 1.0)
            scores = scores * pred_sim.clamp_min(1e-12) ** float(lambda_p)
            accepted_mask = accepted_mask & (pred_sim >= float(tau_topology))

        order = torch.argsort(scores, descending=True).tolist()
        members = [center]
        current_q, _, _ = _ball_quality(
            members,
            affinity,
            topo_row,
            center,
            use_embedding=use_embedding,
            use_topology=use_topology,
            comp_power=comp_power,
            topo_power=topo_power,
            size_m0=size_m0,
            topo_eps=topo_eps,
        )
        for order_idx in order:
            node = int(candidate_nodes[order_idx].item())
            if node == center or node not in unassigned or node in members:
                continue
            if not bool(accepted_mask[order_idx].item()):
                continue
            proposed = members + [node]
            next_q, _, _ = _ball_quality(
                proposed,
                affinity,
                topo_row,
                center,
                use_embedding=use_embedding,
                use_topology=use_topology,
                comp_power=comp_power,
                topo_power=topo_power,
                size_m0=size_m0,
                topo_eps=topo_eps,
            )
            if next_q + float(delta_q) < current_q:
                continue
            members = proposed
            current_q = next_q
            if len(members) >= int(max_ball_size):
                break

        if len(members) < int(min_ball_size):
            low_quality_count += 1
        quality, comp, topo = _ball_quality(
            members,
            affinity,
            topo_row,
            center,
            use_embedding=use_embedding,
            use_topology=use_topology,
            comp_power=comp_power,
            topo_power=topo_power,
            size_m0=size_m0,
            topo_eps=topo_eps,
        )
        qualities.append(quality)
        comps.append(comp)
        topos.append(topo)
        comp_count = _connected_component_count(members, adjacency)
        component_counts.append(comp_count)
        if comp_count <= 1:
            connected_count += 1
        for node in members:
            if int(node) == center:
                continue
            accepted_noncenter += 1
            if int(node) in emb_top_nodes:
                embedding_hits += 1
            if float(topo_row[int(node)].item()) >= float(tau_topology):
                topology_supported_hits += 1
        groups.append(members)
        for node in members:
            unassigned.discard(int(node))

    config = {
        "mode": mode,
        "topology_mode": resolved_topology_mode,
        "requested_topology_mode": topology_mode,
        "max_hops": int(max_hops),
        "min_ball_size": int(min_ball_size),
        "max_ball_size": int(max_ball_size),
        "seed": int(seed),
        "ppr_alpha": float(ppr_alpha),
        "ppr_iters": int(ppr_iters),
        "embedding_top_m": int(embedding_top_m),
        "topology_top_m": int(topology_top_m),
        "center_topk": int(center_topk),
        "tau_h": float(tau_h),
        "tau_h_strong": float(tau_h_strong),
        "tau_topology": float(tau_topology),
        "tau_add": float(tau_add),
        "lambda_h": float(lambda_h),
        "lambda_t": float(lambda_t),
        "topo_eps": float(topo_eps),
        "delta_q": float(delta_q),
        "size_m0": float(size_m0),
        "lambda_p": float(lambda_p),
    }
    diagnostics_extra = {
        "grouping_kind": "grace_gagb" if mode == "grace_first" else f"grace_gagb_{mode}",
        "gfgb_mode": mode,
        "gfgb_topology_mode": resolved_topology_mode,
        "gfgb_quality_mean": float(sum(qualities) / len(qualities)) if qualities else math.nan,
        "gfgb_comp_mean": float(sum(comps) / len(comps)) if comps else math.nan,
        "gfgb_topo_mean": float(sum(topos) / len(topos)) if topos else math.nan,
        "gfgb_connected_ratio": float(connected_count / len(groups)) if groups else math.nan,
        "gfgb_low_quality_ball_ratio": float(low_quality_count / len(groups)) if groups else math.nan,
        "gfgb_embedding_candidate_hit_rate": float(embedding_hits / max(1, accepted_noncenter)),
        "gfgb_topology_supported_rate": float(topology_supported_hits / max(1, accepted_noncenter)),
        "gfgb_component_count_mean": float(sum(component_counts) / len(component_counts)) if component_counts else math.nan,
        "gfgb_embedding_top_m": int(embedding_top_m),
        "gfgb_topology_top_m": int(topology_top_m),
        "gfgb_tau_h": float(tau_h),
        "gfgb_tau_h_strong": float(tau_h_strong),
        "gfgb_tau_topology": float(tau_topology),
        "gfgb_tau_add": float(tau_add),
        "gfgb_delta_q": float(delta_q),
    }
    return _payload_from_groups(emb, groups, config, diagnostics_extra)


def build_graph_aware_balls(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    max_hops: int = 2,
    min_ball_size: int = 5,
    tau_add: float = 0.55,
    lambda_h: float = 0.7,
    lambda_t: float = 0.3,
    max_ball_size: int = 128,
    seed: int = 0,
    teacher_probs: torch.Tensor | None = None,
    lambda_p: float = 0.0,
    candidate_scope: str = "khop",
) -> dict:
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be a [num_nodes, dim] tensor")
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    if max_ball_size < 1:
        raise ValueError("max_ball_size must be >= 1")
    if candidate_scope not in {"khop", "global"}:
        raise ValueError("candidate_scope must be 'khop' or 'global'")

    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    num_nodes = int(emb.size(0))
    adjacency = _build_adjacency(edge_index, num_nodes)
    degrees = torch.tensor([len(neighbors) for neighbors in adjacency], dtype=torch.float32)
    unassigned = set(range(num_nodes))
    groups: list[list[int]] = []
    generator = torch.Generator().manual_seed(int(seed))
    random_tie_break = torch.rand(num_nodes, generator=generator)
    probs = teacher_probs.detach().cpu().float() if teacher_probs is not None else None

    while unassigned:
        candidates_for_center = torch.tensor(sorted(unassigned), dtype=torch.long)
        center_scores = degrees[candidates_for_center] + 1e-6 * random_tie_break[candidates_for_center]
        center = int(candidates_for_center[int(torch.argmax(center_scores).item())].item())
        hop_distances = _khop_distances(center, adjacency, max_hops, unassigned)
        if candidate_scope == "global":
            candidate_set = sorted(unassigned)
            hop_lookup = {node: hop_distances.get(node) for node in candidate_set}
        else:
            if not hop_distances:
                hop_distances = {center: 0}
            candidate_set = sorted(hop_distances)
            hop_lookup = {node: hop_distances[node] for node in candidate_set}

        candidate_nodes = torch.tensor(candidate_set, dtype=torch.long)
        cosine = (emb[candidate_nodes] @ emb[center]).clamp(-1.0, 1.0)
        topo = torch.tensor(
            [
                1.0 if hop_lookup[int(node)] == 0
                else (1.0 / float(hop_lookup[int(node)]) if hop_lookup[int(node)] is not None else 0.0)
                for node in candidate_nodes.tolist()
            ],
            dtype=torch.float32,
        )
        scores = float(lambda_h) * cosine + float(lambda_t) * topo
        if probs is not None and float(lambda_p) > 0.0:
            pred_sim = (probs[candidate_nodes] * probs[center].view(1, -1)).sum(dim=1)
            scores = scores + float(lambda_p) * pred_sim

        order = torch.argsort(scores, descending=True)
        ordered_nodes = candidate_nodes[order].tolist()
        ordered_scores = scores[order].tolist()
        selected = [node for node, score in zip(ordered_nodes, ordered_scores) if node == center or score >= float(tau_add)]
        if len(selected) < int(min_ball_size):
            for node in ordered_nodes:
                if node not in selected:
                    selected.append(node)
                if len(selected) >= int(min_ball_size):
                    break
        selected = selected[: int(max_ball_size)]
        if center not in selected:
            selected.insert(0, center)
            selected = selected[: int(max_ball_size)]

        groups.append(selected)
        for node in selected:
            unassigned.discard(int(node))

    config = {
        "max_hops": int(max_hops),
        "min_ball_size": int(min_ball_size),
        "tau_add": float(tau_add),
        "lambda_h": float(lambda_h),
        "lambda_t": float(lambda_t),
        "max_ball_size": int(max_ball_size),
        "seed": int(seed),
        "lambda_p": float(lambda_p),
        "candidate_scope": candidate_scope,
    }
    diagnostics_extra = {
        "grouping_kind": "graph_aware",
        "graph_aware_max_hops": int(max_hops),
        "graph_aware_tau_add": float(tau_add),
        "graph_aware_lambda_h": float(lambda_h),
        "graph_aware_lambda_t": float(lambda_t),
        "graph_aware_candidate_scope": candidate_scope,
    }
    return _payload_from_groups(emb, groups, config, diagnostics_extra)


def _khop_affinity_matrix(adjacency: list[set[int]], num_nodes: int, max_hops: int) -> torch.Tensor:
    """Full [N, N] topology affinity from BFS hop distance: T[i, i]=1, T[i, j]=1/dist (dist<=max_hops)."""
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for center in range(num_nodes):
        distances = {center: 0}
        queue: deque[int] = deque([center])
        while queue:
            node = queue.popleft()
            hop = distances[node]
            if hop >= max_hops:
                continue
            for neighbor in adjacency[node]:
                if neighbor in distances:
                    continue
                distances[neighbor] = hop + 1
                queue.append(neighbor)
        for node, dist in distances.items():
            matrix[center, int(node)] = 1.0 if dist == 0 else 1.0 / float(dist)
    matrix.fill_diagonal_(1.0)
    return matrix


def _topm_sets(scores: torch.Tensor, top_m: int) -> list[set[int]]:
    """For each row i, the set of top-`top_m` column indices excluding self (i)."""
    num_nodes = int(scores.size(0))
    if num_nodes <= 1:
        return [set() for _ in range(num_nodes)]
    k = max(1, min(int(top_m) + 1, num_nodes))
    idx = torch.topk(scores, k=k, dim=1).indices
    out: list[set[int]] = []
    for i in range(num_nodes):
        members = set(int(j) for j in idx[i].tolist())
        members.discard(i)
        out.append(members)
    return out


def build_dcgb_balls(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    max_hops: int = 2,
    min_ball_size: int = 5,
    max_ball_size: int = 128,
    seed: int = 0,
    mode: str = "dcgb",
    topology_mode: str = "ppr_lite",
    ppr_alpha: float = 0.15,
    ppr_iters: int = 10,
    embedding_top_m: int = 48,
    topology_top_m: int = 32,
    tau_h: float = 0.6,
    tau_topology: float = 0.05,
    topo_percentile: float = 0.5,
    tau_h_strong: float = 0.82,
    lambda_h: float = 0.7,
    lambda_t: float = 0.3,
    topo_eps: float = 0.05,
    delta_q: float = 0.01,
    size_m0: float = 5.0,
    min_core_relax: int = 0,
) -> dict:
    """Dual-view Consistency Granular Balls.

    Reliability is derived from *two-view corroboration* between the GRACE embedding view and the
    graph-topology view: a node only joins a ball if both views agree it belongs. A directional
    embedding top-k control uses a single view and cannot replicate this.

    - Anchors are dual-view consensus interior nodes: highest |EmbTopM(i) ∩ TopoTopM(i)|.
    - Candidate pool is generous (embedding top-M ∪ k-hop) to keep balls substantive; membership is
      then *corroboration-gated*: a node `j` joins anchor `c` only if it is embedding-close
      (`A[c,j] >= tau_h`) AND corroborated by a second view — reciprocal embedding kNN
      (`c in EmbTopM(j)`) OR topology support (`T[c,j] >= tau_topology`). This corroboration cuts the
      cross-class boundary nodes a single-view top-k would absorb, without collapsing to singletons.
    - `mode="emb_only"` keeps only the reciprocity corroboration (embedding view alone);
      `mode="topo_only"` keeps only topology neighbours.

    Clean labels are never used. Returns the standard grouping payload via `_payload_from_groups`.
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must be a [num_nodes, dim] tensor")
    if mode not in {"dcgb", "emb_only", "topo_only"}:
        raise ValueError("mode must be one of 'dcgb', 'emb_only', or 'topo_only'")
    if topology_mode not in {"ppr_lite", "khop"}:
        raise ValueError("topology_mode must be 'ppr_lite' or 'khop'")
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    if min_ball_size < 1 or max_ball_size < 1:
        raise ValueError("ball sizes must be positive")

    emb = F.normalize(embeddings.detach().cpu().float(), p=2, dim=1)
    num_nodes = int(emb.size(0))
    adjacency = _build_adjacency(edge_index, num_nodes)
    cosine = (emb @ emb.t()).clamp(-1.0, 1.0)
    affinity = ((cosine + 1.0) / 2.0).clamp(0.0, 1.0)

    resolved_topology_mode = topology_mode
    topology_matrix = None
    if topology_mode == "ppr_lite":
        try:
            topology_matrix = _ppr_lite_matrix(adjacency, num_nodes, ppr_alpha, ppr_iters)
        except RuntimeError:
            resolved_topology_mode = "khop"
    if topology_matrix is None:
        topology_matrix = _khop_affinity_matrix(adjacency, num_nodes, max_hops)
        resolved_topology_mode = "khop"

    emb_sets = _topm_sets(affinity, embedding_top_m)
    topo_sets = _topm_sets(topology_matrix, topology_top_m)
    denom = max(1, int(embedding_top_m))
    consensus = torch.tensor(
        [len(emb_sets[i] & topo_sets[i]) / denom for i in range(num_nodes)],
        dtype=torch.float32,
    )

    use_embedding = mode in {"dcgb", "emb_only"}
    use_topology = mode in {"dcgb", "topo_only"}
    comp_power = float(lambda_h) if use_embedding else 0.0
    topo_power = float(lambda_t) if use_topology else 0.0

    generator = torch.Generator().manual_seed(int(seed))
    random_tie_break = torch.rand(num_nodes, generator=generator)
    unassigned = set(range(num_nodes))
    groups: list[list[int]] = []
    qualities: list[float] = []
    comps: list[float] = []
    topos: list[float] = []
    component_counts: list[int] = []
    connected_count = 0
    low_quality_count = 0
    mutual_member_total = 0
    accepted_noncenter = 0
    reciprocal_hits = 0
    topology_supported_hits = 0

    while unassigned:
        available = torch.tensor(sorted(unassigned), dtype=torch.long)
        if mode == "topo_only":
            anchor_score = torch.tensor(
                [float(len(topo_sets[int(i)])) for i in available.tolist()],
                dtype=torch.float32,
            )
        else:
            anchor_score = consensus[available]
        anchor_score = anchor_score + 1e-6 * random_tie_break[available]
        center = int(available[int(torch.argmax(anchor_score).item())].item())
        topo_row = topology_matrix[center]

        # Generous candidate pool (embedding top-M ∪ k-hop) keeps balls substantive.
        hop_nodes = set(_khop_distances(center, adjacency, max_hops, unassigned))
        if mode == "topo_only":
            pool = (topo_sets[center] | hop_nodes) & unassigned
        else:
            pool = (emb_sets[center] | hop_nodes) & unassigned

        # Corroboration gate: embedding-close AND agreed by a second view.
        candidate_set = []
        for j in pool:
            if j == center:
                continue
            h = float(affinity[center, j].item())
            t = float(topo_row[j].item())
            reciprocal = (j in emb_sets[center]) and (center in emb_sets[j])
            if mode == "emb_only":
                accept = (h >= float(tau_h)) and reciprocal
            elif mode == "topo_only":
                accept = t >= float(tau_topology)
            else:
                accept = (h >= float(tau_h)) and (reciprocal or (t >= float(tau_topology)))
            if accept:
                candidate_set.append(int(j))

        # Optional relaxation so an isolated anchor still seeds a small ball.
        if mode != "topo_only" and not candidate_set and int(min_core_relax) > 0:
            relax = [
                j for j in emb_sets[center]
                if j in unassigned and float(affinity[center, j].item()) >= float(tau_h_strong)
            ]
            relax.sort(key=lambda j: float(affinity[center, j].item()), reverse=True)
            candidate_set = relax[: max(0, int(min_ball_size) - 1)]

        if candidate_set:
            cand_nodes = torch.tensor(candidate_set, dtype=torch.long)
            h_sim = affinity[center, cand_nodes].clamp_min(1e-12)
            t_sim = topo_row[cand_nodes].clamp(0.0, 1.0)
            if mode == "emb_only":
                scores = h_sim
            elif mode == "topo_only":
                scores = t_sim.clamp_min(1e-12)
            else:
                scores = (h_sim ** float(lambda_h)) * ((float(topo_eps) + t_sim).clamp_min(1e-12) ** float(lambda_t))
            order = torch.argsort(scores, descending=True).tolist()
        else:
            cand_nodes = torch.empty(0, dtype=torch.long)
            order = []

        members = [center]
        current_q, _, _ = _ball_quality(
            members, affinity, topo_row, center,
            use_embedding=use_embedding, use_topology=use_topology,
            comp_power=comp_power, topo_power=topo_power, size_m0=size_m0, topo_eps=topo_eps,
        )
        for order_idx in order:
            node = int(cand_nodes[order_idx].item())
            if node == center or node not in unassigned or node in members:
                continue
            proposed = members + [node]
            next_q, _, _ = _ball_quality(
                proposed, affinity, topo_row, center,
                use_embedding=use_embedding, use_topology=use_topology,
                comp_power=comp_power, topo_power=topo_power, size_m0=size_m0, topo_eps=topo_eps,
            )
            if next_q + float(delta_q) < current_q:
                continue
            members = proposed
            current_q = next_q
            if len(members) >= int(max_ball_size):
                break

        if len(members) < int(min_ball_size):
            low_quality_count += 1
        quality, comp, topo = _ball_quality(
            members, affinity, topo_row, center,
            use_embedding=use_embedding, use_topology=use_topology,
            comp_power=comp_power, topo_power=topo_power, size_m0=size_m0, topo_eps=topo_eps,
        )
        qualities.append(quality)
        comps.append(comp)
        topos.append(topo)
        comp_count = _connected_component_count(members, adjacency)
        component_counts.append(comp_count)
        if comp_count <= 1:
            connected_count += 1
        for node in members:
            if int(node) == center:
                continue
            accepted_noncenter += 1
            if center in emb_sets[int(node)] and int(node) in emb_sets[center]:
                reciprocal_hits += 1
                mutual_member_total += 1
            if float(topo_row[int(node)].item()) >= float(tau_topology):
                topology_supported_hits += 1
        groups.append(members)
        for node in members:
            unassigned.discard(int(node))

    config = {
        "mode": mode,
        "topology_mode": resolved_topology_mode,
        "requested_topology_mode": topology_mode,
        "max_hops": int(max_hops),
        "min_ball_size": int(min_ball_size),
        "max_ball_size": int(max_ball_size),
        "seed": int(seed),
        "ppr_alpha": float(ppr_alpha),
        "ppr_iters": int(ppr_iters),
        "embedding_top_m": int(embedding_top_m),
        "topology_top_m": int(topology_top_m),
        "tau_h": float(tau_h),
        "tau_topology": float(tau_topology),
        "topo_percentile": float(topo_percentile),
        "tau_h_strong": float(tau_h_strong),
        "lambda_h": float(lambda_h),
        "lambda_t": float(lambda_t),
        "topo_eps": float(topo_eps),
        "delta_q": float(delta_q),
        "size_m0": float(size_m0),
        "min_core_relax": int(min_core_relax),
    }
    # Reuse the established gfgb_* diagnostic key names so summary.py wiring applies without edits.
    diagnostics_extra = {
        "grouping_kind": "dcgb" if mode == "dcgb" else f"dcgb_{mode}",
        "dcgb_mode": mode,
        "gfgb_mode": mode,
        "gfgb_topology_mode": resolved_topology_mode,
        "gfgb_quality_mean": float(sum(qualities) / len(qualities)) if qualities else math.nan,
        "gfgb_comp_mean": float(sum(comps) / len(comps)) if comps else math.nan,
        "gfgb_topo_mean": float(sum(topos) / len(topos)) if topos else math.nan,
        "gfgb_connected_ratio": float(connected_count / len(groups)) if groups else math.nan,
        "gfgb_low_quality_ball_ratio": float(low_quality_count / len(groups)) if groups else math.nan,
        "gfgb_embedding_candidate_hit_rate": float(reciprocal_hits / max(1, accepted_noncenter)),
        "gfgb_topology_supported_rate": float(topology_supported_hits / max(1, accepted_noncenter)),
        "gfgb_component_count_mean": float(sum(component_counts) / len(component_counts)) if component_counts else math.nan,
        "gfgb_embedding_top_m": int(embedding_top_m),
        "gfgb_topology_top_m": int(topology_top_m),
        "gfgb_tau_topology": float(tau_topology),
        "gfgb_delta_q": float(delta_q),
        "dcgb_reciprocal_member_rate": float(mutual_member_total / max(1, accepted_noncenter)),
        "dcgb_consensus_mean": float(consensus.mean().item()) if num_nodes else math.nan,
    }
    return _payload_from_groups(emb, groups, config, diagnostics_extra)


def shuffle_grouping(grouping: dict, seed: int = 0, embeddings: torch.Tensor | None = None) -> dict:
    group_ids = grouping["group_ids"].detach().cpu().long()
    num_nodes = int(group_ids.numel())
    sizes = [len(group) for group in grouping["groups"]]
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=generator).tolist()
    groups = []
    cursor = 0
    for size in sizes:
        groups.append(perm[cursor: cursor + int(size)])
        cursor += int(size)
    if embeddings is None:
        emb_dim = int(grouping.get("centers", torch.empty((0, 1))).size(1)) if "centers" in grouping else 1
        embeddings = torch.zeros((num_nodes, emb_dim), dtype=torch.float32)
    emb = embeddings.detach().cpu().float()
    config = {**grouping.get("config", {}), "shuffle_seed": int(seed)}
    diagnostics_extra = {
        "grouping_kind": "graph_aware_shuffled",
        "source_grouping_kind": grouping.get("diagnostics", {}).get("grouping_kind", ""),
    }
    return _payload_from_groups(emb, groups, config, diagnostics_extra)
