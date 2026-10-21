from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def bidirected_with_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique CPU edges with reverse edges and one loop per node."""
    edges = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    reverse = edges.flip(0)
    nodes = torch.arange(int(num_nodes), dtype=torch.long)
    loops = torch.stack([nodes, nodes], dim=0)
    combined = torch.cat([edges, reverse, loops], dim=1)
    return torch.unique(combined.t(), dim=0).t().contiguous()


@dataclass(frozen=True)
class SampledSubgraph:
    global_nodes: torch.Tensor
    edge_index: torch.Tensor
    output_count: int

    @property
    def num_nodes(self) -> int:
        return int(self.global_nodes.numel())


class CpuNeighborSampler:
    """Deterministic DGL-style multi-layer inbound-neighbor sampler."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fanouts: tuple[int, ...] | list[int],
        seed: int,
    ):
        self.num_nodes = int(num_nodes)
        self.fanouts = tuple(int(value) for value in fanouts)
        if not self.fanouts or any(value == 0 or value < -1 for value in self.fanouts):
            raise ValueError("fanouts must contain positive integers or -1 for all neighbors")
        self.seed = int(seed)
        edges = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= self.num_nodes):
            raise ValueError("edge_index contains an invalid node id")
        neighbor_lists: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for source, destination in edges.t().tolist():
            neighbor_lists[destination].append(source)
        self.neighbors = tuple(
            np.asarray(sorted(set(values)), dtype=np.int64) for values in neighbor_lists
        )

    def sample(
        self,
        output_nodes: torch.Tensor,
        *,
        epoch: int,
        batch_id: int,
    ) -> SampledSubgraph:
        outputs = torch.as_tensor(output_nodes, dtype=torch.long, device="cpu").flatten()
        if outputs.numel() == 0:
            raise ValueError("output_nodes must not be empty")
        ordered_outputs = list(dict.fromkeys(int(value) for value in outputs.tolist()))
        if min(ordered_outputs) < 0 or max(ordered_outputs) >= self.num_nodes:
            raise ValueError("output_nodes contains an invalid node id")

        local_nodes = list(ordered_outputs)
        local_position = {node: idx for idx, node in enumerate(local_nodes)}
        seeds = list(ordered_outputs)
        sampled_edges: list[tuple[int, int]] = []
        mixed_seed = (
            self.seed
            + 1_000_003 * int(epoch)
            + 97_409 * int(batch_id)
        ) % (2**63 - 1)
        rng = np.random.default_rng(mixed_seed)

        # DGL expands the last GNN layer first and preserves nested destination
        # prefixes. Keeping output nodes first reproduces the official slicing.
        for fanout in reversed(self.fanouts):
            for destination in seeds:
                candidates = self.neighbors[destination]
                if fanout == -1 or candidates.size <= fanout:
                    selected = candidates
                else:
                    selected = rng.choice(candidates, size=fanout, replace=False)
                for source_value in selected.tolist():
                    source = int(source_value)
                    if source not in local_position:
                        local_position[source] = len(local_nodes)
                        local_nodes.append(source)
                    sampled_edges.append((source, destination))
            seeds = list(local_nodes)

        local_edges: list[tuple[int, int]] = []
        for source, destination in sampled_edges:
            src_local = local_position[source]
            dst_local = local_position[destination]
            local_edges.append((src_local, dst_local))
            local_edges.append((dst_local, src_local))
        edge_tensor = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        return SampledSubgraph(
            global_nodes=torch.tensor(local_nodes, dtype=torch.long),
            edge_index=edge_tensor,
            output_count=len(ordered_outputs),
        )


def local_dense_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    dense = torch.zeros((int(num_nodes), int(num_nodes)), device=device, dtype=dtype)
    values = torch.ones(edges.shape[1], device=device, dtype=dtype)
    dense.index_put_((edges[0], edges[1]), values, accumulate=True)
    return dense


class DGLBothNormGraphConv(nn.Module):
    """PyTorch equivalent of DGL GraphConv with norm='both'."""

    def __init__(self, in_channels: int, out_channels: int, activation=torch.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(int(in_channels), int(out_channels)))
        self.bias = nn.Parameter(torch.empty(int(out_channels)))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edges = edge_index.to(device=features.device, dtype=torch.long)
        source, destination = edges
        num_nodes = int(features.shape[0])
        ones = torch.ones(source.numel(), device=features.device, dtype=features.dtype)
        out_degree = torch.zeros(num_nodes, device=features.device, dtype=features.dtype)
        in_degree = torch.zeros(num_nodes, device=features.device, dtype=features.dtype)
        out_degree.index_add_(0, source, ones)
        in_degree.index_add_(0, destination, ones)
        if bool((in_degree == 0).any()):
            raise ValueError("DGL GraphConv does not allow zero in-degree nodes")

        normalized = features * out_degree.clamp_min(1).rsqrt().unsqueeze(1)
        messages = normalized @ self.weight
        output = torch.zeros(
            (num_nodes, self.weight.shape[1]),
            device=features.device,
            dtype=features.dtype,
        )
        output.index_add_(0, destination, messages[source])
        output = output * in_degree.rsqrt().unsqueeze(1)
        output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
