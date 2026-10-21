from __future__ import annotations

import math

import torch


def _to_sparse_coo(adj: torch.Tensor) -> torch.Tensor:
    if getattr(adj, "is_sparse", False):
        return adj.coalesce()
    return adj.to_sparse().coalesce()


def _as_index_tensor(mask_or_index, device: torch.device) -> torch.Tensor:
    values = torch.as_tensor(mask_or_index, device=device)
    if values.dtype == torch.bool:
        return torch.nonzero(values, as_tuple=False).flatten()
    return values.to(dtype=torch.long).flatten()


def perturb_test_adj(
    adj: torch.Tensor,
    mode: str,
    p: float,
    seed: int,
    test_mask=None,
) -> torch.Tensor:
    """Return a deterministic sparse adjacency perturbation for evaluation only.

    The function treats the incoming sparse COO entries as the inference graph used
    by NoisyGL predictors. It does not mutate ``adj``.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in [0, 1]")
    mode = str(mode).lower()
    if mode in {"none", "clean"} or p == 0.0:
        return _to_sparse_coo(adj).clone().coalesce()

    sparse = _to_sparse_coo(adj)
    device = sparse.device
    indices = sparse.indices()
    values = sparse.values()
    n_nodes = sparse.size(0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    if mode in {"edge_drop", "drop"}:
        non_self = (indices[0] != indices[1]).cpu()
        candidate = torch.nonzero(non_self, as_tuple=False).flatten()
        drop_count = int(math.floor(float(p) * int(candidate.numel())))
        if drop_count <= 0:
            return sparse.clone().coalesce()
        order = torch.randperm(int(candidate.numel()), generator=generator)[:drop_count]
        drop_entries = candidate[order].to(device=indices.device)
        keep = torch.ones(indices.shape[1], dtype=torch.bool, device=indices.device)
        keep[drop_entries] = False
        return torch.sparse_coo_tensor(indices[:, keep], values[keep], sparse.size(), device=device).coalesce()

    if mode in {"isolated", "isolate"}:
        if test_mask is None:
            raise ValueError("test_mask is required for isolated perturbation")
        test_idx = _as_index_tensor(test_mask, device=device)
        isolate_count = int(math.ceil(float(p) * int(test_idx.numel())))
        if isolate_count <= 0:
            return sparse.clone().coalesce()
        order = torch.randperm(int(test_idx.numel()), generator=generator)[:isolate_count].to(device)
        selected = test_idx[order]
        selected_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        selected_mask[selected] = True
        keep = ~(selected_mask[indices[0]] | selected_mask[indices[1]])
        return torch.sparse_coo_tensor(indices[:, keep], values[keep], sparse.size(), device=device).coalesce()

    if mode in {"edge_add", "add"}:
        non_self_count = int((indices[0] != indices[1]).sum().item())
        add_count = int(math.floor(float(p) * non_self_count))
        if add_count <= 0:
            return sparse.clone().coalesce()
        rows = torch.randint(0, n_nodes, (add_count,), generator=generator)
        cols = torch.randint(0, n_nodes, (add_count,), generator=generator)
        self_loop = rows == cols
        attempts = 0
        while bool(self_loop.any()) and attempts < 10:
            cols[self_loop] = torch.randint(0, n_nodes, (int(self_loop.sum()),), generator=generator)
            self_loop = rows == cols
            attempts += 1
        add_indices = torch.stack([rows, cols]).to(device=device, dtype=indices.dtype)
        fill_value = values.float().mean().to(device=device) if values.numel() else torch.tensor(1.0, device=device)
        add_values = torch.full((add_count,), float(fill_value.item()), dtype=values.dtype, device=device)
        all_indices = torch.cat([indices, add_indices], dim=1)
        all_values = torch.cat([values, add_values], dim=0)
        return torch.sparse_coo_tensor(all_indices, all_values, sparse.size(), device=device).coalesce()

    raise ValueError(f"unsupported structure perturbation mode: {mode}")
