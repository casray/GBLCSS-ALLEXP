import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.balls.graph_aware_ball import (
    build_dcgb_balls,
    build_grace_first_graph_balls,
    build_graph_aware_balls,
    shuffle_grouping,
)


def _two_cluster_graph(per_cluster: int = 12, dim: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    cluster_a = torch.randn(per_cluster, dim, generator=generator) + 5.0
    cluster_b = torch.randn(per_cluster, dim, generator=generator) - 5.0
    embeddings = torch.cat([cluster_a, cluster_b], dim=0)
    src, dst = [], []
    for cluster in range(2):
        base = cluster * per_cluster
        for i in range(per_cluster):
            a = base + i
            b = base + (i + 1) % per_cluster
            src += [a, b]
            dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return embeddings, edge_index, per_cluster


def test_graph_aware_balls_respect_topology_candidates():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 0, 2, 1, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    grouping = build_graph_aware_balls(
        embeddings,
        edge_index,
        max_hops=1,
        min_ball_size=1,
        tau_add=0.0,
        lambda_h=1.0,
        lambda_t=1.0,
    )

    left_component = {0, 1, 2}
    right_component = {3, 4, 5}
    for group in grouping["groups"]:
        members = set(group)
        assert members <= left_component or members <= right_component


def test_shuffle_grouping_preserves_group_sizes():
    grouping = {
        "group_ids": torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long),
        "groups": [[0, 1], [2, 3, 4], [5]],
        "radii": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "node_distances": torch.zeros(6),
        "diagnostics": {"ball_count": 3},
    }

    shuffled = shuffle_grouping(grouping, seed=7)

    assert sorted(len(group) for group in shuffled["groups"]) == [1, 2, 3]
    assert shuffled["group_ids"].numel() == grouping["group_ids"].numel()
    assert sorted(shuffled["group_ids"].unique().tolist()) == [0, 1, 2]


def test_graph_aware_global_candidate_scope_can_cross_components():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [0.95, 0.05],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)

    local = build_graph_aware_balls(
        embeddings,
        edge_index,
        max_hops=1,
        min_ball_size=1,
        tau_add=0.8,
        lambda_h=1.0,
        lambda_t=0.0,
        candidate_scope="khop",
    )
    global_grouping = build_graph_aware_balls(
        embeddings,
        edge_index,
        max_hops=1,
        min_ball_size=1,
        tau_add=0.8,
        lambda_h=1.0,
        lambda_t=0.0,
        candidate_scope="global",
    )

    assert local["group_ids"][0].item() != local["group_ids"][3].item()
    assert global_grouping["group_ids"][0].item() == global_grouping["group_ids"][3].item()
    assert int((global_grouping["group_ids"] < 0).sum().item()) == 0
    assert sum(len(group) for group in global_grouping["groups"]) == embeddings.size(0)


def test_grace_first_graph_balls_use_embedding_candidates_with_topology_gate():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    grouping = build_grace_first_graph_balls(
        embeddings,
        edge_index,
        max_hops=1,
        min_ball_size=1,
        max_ball_size=4,
        mode="grace_first",
        topology_mode="khop",
        embedding_top_m=3,
        topology_top_m=1,
        tau_h=0.7,
        tau_h_strong=0.95,
        tau_topology=0.5,
        tau_add=0.1,
        delta_q=1.0,
        seed=0,
    )

    group_ids = grouping["group_ids"]
    assert group_ids[2].item() in {group_ids[0].item(), group_ids[1].item()}
    assert grouping["diagnostics"]["grouping_kind"] == "grace_gagb"
    assert grouping["diagnostics"]["gfgb_topology_mode"] == "khop"
    assert "gfgb_quality_mean" in grouping["diagnostics"]


def test_dcgb_full_coverage_no_orphans():
    embeddings, edge_index, _ = _two_cluster_graph()
    grouping = build_dcgb_balls(embeddings, edge_index, max_hops=2, min_ball_size=5, seed=0)
    group_ids = grouping["group_ids"]
    assert int((group_ids < 0).sum().item()) == 0
    assert sum(len(group) for group in grouping["groups"]) == embeddings.size(0)
    assert grouping["diagnostics"]["grouping_kind"] == "dcgb"
    assert "gfgb_quality_mean" in grouping["diagnostics"]


def test_dcgb_reciprocal_membership_keeps_clusters_pure():
    embeddings, edge_index, per_cluster = _two_cluster_graph()
    grouping = build_dcgb_balls(embeddings, edge_index, max_hops=2, min_ball_size=5, seed=0)
    for group in grouping["groups"]:
        labels = {0 if node < per_cluster else 1 for node in group}
        assert len(labels) == 1, "a DCGB ball crossed the cluster boundary"
    # Reciprocal members should dominate the accepted non-center members.
    assert grouping["diagnostics"]["dcgb_reciprocal_member_rate"] >= 0.9


def test_dcgb_modes_produce_valid_groupings():
    embeddings, edge_index, _ = _two_cluster_graph()
    for mode, kind in (
        ("dcgb", "dcgb"),
        ("emb_only", "dcgb_emb_only"),
        ("topo_only", "dcgb_topo_only"),
    ):
        grouping = build_dcgb_balls(embeddings, edge_index, max_hops=2, min_ball_size=5, seed=0, mode=mode)
        assert int((grouping["group_ids"] < 0).sum().item()) == 0
        assert sum(len(group) for group in grouping["groups"]) == embeddings.size(0)
        assert grouping["diagnostics"]["grouping_kind"] == kind
