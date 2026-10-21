from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge, subgraph


def augment_node_dropping(data: Data, drop_ratio: float = 0.2) -> Data:
    """Randomly drop nodes and incident edges from one graph."""
    out = data.clone()
    num_nodes = int(out.num_nodes)
    if num_nodes <= 1 or drop_ratio <= 0:
        return out
    keep_count = max(1, int(round(num_nodes * (1.0 - drop_ratio))))
    perm = torch.randperm(num_nodes, device=out.edge_index.device)[:keep_count]
    perm = perm.sort().values
    edge_index, edge_attr = subgraph(
        perm,
        out.edge_index,
        edge_attr=getattr(out, "edge_attr", None),
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    out.edge_index = edge_index
    if getattr(out, "edge_attr", None) is not None:
        out.edge_attr = edge_attr
    out.x = out.x[perm] if getattr(out, "x", None) is not None else None
    out.num_nodes = keep_count
    return out


def augment_edge_perturbation(data: Data, perturb_ratio: float = 0.2) -> Data:
    """Randomly delete and add a fraction of edges from one graph."""
    out = data.clone()
    if out.edge_index.numel() == 0 or perturb_ratio <= 0:
        return out
    original_edges = out.edge_index.size(1)
    edge_index, edge_mask = dropout_edge(out.edge_index, p=perturb_ratio, force_undirected=False)
    num_add = int(round(original_edges * perturb_ratio))
    if num_add > 0 and int(out.num_nodes) > 1:
        add_src = torch.randint(int(out.num_nodes), (num_add,), device=edge_index.device)
        add_dst = torch.randint(int(out.num_nodes), (num_add,), device=edge_index.device)
        added = torch.stack([add_src, add_dst], dim=0)
        edge_index = torch.cat([edge_index, added], dim=1)
    out.edge_index = edge_index
    if getattr(out, "edge_attr", None) is not None and edge_mask is not None:
        kept_attr = out.edge_attr[edge_mask]
        if num_add > 0:
            added_attr = torch.zeros((num_add, kept_attr.size(1)), dtype=kept_attr.dtype, device=kept_attr.device)
            kept_attr = torch.cat([kept_attr, added_attr], dim=0)
        out.edge_attr = kept_attr
    return out


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    """SimCLR/GraphCL NT-Xent loss for paired graph views."""
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    batch_size = z1.size(0)
    if batch_size < 2:
        raise ValueError("NT-Xent requires batch_size >= 2")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    targets = torch.arange(batch_size, device=z1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def _augment_batch(batch, aug_fn):
    data_list = [aug_fn(data) for data in batch.to_data_list()]
    return Batch.from_data_list(data_list).to(batch.x.device if batch.x is not None else batch.edge_index.device)


def train_graphcl_epoch(encoder, proj_head, loader, optimizer, device, aug1_fn, aug2_fn, temperature: float = 0.2):
    """Train GraphCL for one epoch."""
    encoder.train()
    proj_head.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        if batch.num_graphs < 2:
            continue
        view1 = _augment_batch(batch, aug1_fn)
        view2 = _augment_batch(batch, aug2_fn)
        optimizer.zero_grad(set_to_none=True)
        z1 = proj_head(encoder(view1.x, view1.edge_index, view1.batch))
        z2 = proj_head(encoder(view2.x, view2.edge_index, view2.batch))
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)
