from __future__ import annotations

import torch


def _generator_for(tensor: torch.Tensor, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device=tensor.device)
    gen.manual_seed(seed)
    return gen


def _check_probability(value: float, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1]")


def drop_edges(edge_index: torch.Tensor, drop_prob: float, seed: int | None = None) -> torch.Tensor:
    _check_probability(drop_prob, "drop_prob")
    if drop_prob == 0:
        return edge_index
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if drop_prob == 1 or edge_index.size(1) == 0:
        return edge_index.new_empty((2, 0))

    keep = torch.rand(
        edge_index.size(1),
        generator=_generator_for(edge_index, seed),
        device=edge_index.device,
    ) >= drop_prob
    return edge_index[:, keep]


def mask_features(x: torch.Tensor, mask_prob: float, seed: int | None = None) -> torch.Tensor:
    _check_probability(mask_prob, "mask_prob")
    if mask_prob == 0:
        return x
    if mask_prob == 1:
        return torch.zeros_like(x)

    mask = torch.rand(
        x.shape,
        generator=_generator_for(x, seed),
        device=x.device,
    ) < mask_prob
    out = x.clone()
    out[mask] = 0
    return out


def make_two_views(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_drop_1: float,
    edge_drop_2: float,
    feat_mask_1: float,
    feat_mask_2: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seed_1 = seed
    seed_2 = None if seed is None else seed + 1
    x1 = mask_features(x, feat_mask_1, seed=seed_1)
    x2 = mask_features(x, feat_mask_2, seed=seed_2)
    e1 = drop_edges(edge_index, edge_drop_1, seed=seed_1)
    e2 = drop_edges(edge_index, edge_drop_2, seed=seed_2)
    return x1, e1, x2, e2
