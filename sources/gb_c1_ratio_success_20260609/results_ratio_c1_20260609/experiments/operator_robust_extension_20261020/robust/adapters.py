"""ROCm adapters for official TFR and ERASE; vendor sources stay unchanged."""
from __future__ import annotations

import ast
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, MessagePassing
from torch_geometric.utils import add_self_loops, softmax

ROOT = Path(__file__).resolve().parent


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    if dim < 0:
        dim += src.ndim
    moved = src.movedim(dim, 0)
    size = int(dim_size) if dim_size is not None else int(index.max()) + 1
    out = src.new_zeros((size,) + moved.shape[1:])
    out.index_add_(0, index, moved)
    if reduce == "mean":
        counts = src.new_zeros(size)
        counts.index_add_(0, index, src.new_ones(index.numel()))
        out = out / counts.clamp_min(1).reshape((size,) + (1,) * (moved.ndim - 1))
    elif reduce not in {"sum", "add"}:
        raise ValueError(f"unsupported scatter reduction: {reduce}")
    return out.movedim(0, dim)


def scatter_add(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim, dim_size, "sum")


def source_definitions(path, names, namespace):
    """Load named definitions unchanged, excluding unavailable/unused imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    selected = [item for item in tree.body if isinstance(item, (ast.ClassDef, ast.FunctionDef))
                and item.name in names]
    if {item.name for item in selected} != set(names):
        raise ValueError(f"missing upstream definitions in {path}")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def tfr_classes():
    namespace = dict(torch=torch, nn=nn, F=F, GCNConv=GCNConv,
                     MessagePassing=MessagePassing, add_self_loops=add_self_loops,
                     scatter=scatter, softmax=softmax)
    source_definitions(ROOT / "vendor/tfr/predictor/module/GNNs.py", {"GCN"}, namespace)
    source_definitions(ROOT / "vendor/tfr/predictor/module/TEST.py", {"MPNN", "ATTLayer"}, namespace)
    return namespace["GCN"], namespace["MPNN"]


def erase_classes():
    from enum import Enum, auto
    from torch_geometric.utils import add_remaining_self_loops
    namespace = dict(torch=torch, nn=nn, F=F, GATConv=GATConv, GATv2Conv=GATv2Conv,
                     Enum=Enum, auto=auto, scatter_add=scatter_add,
                     add_remaining_self_loops=add_remaining_self_loops)
    source_definitions(ROOT / "vendor/erase/scripts/model.py", {"GAT", "GAT_TYPE"}, namespace)
    namespace.update(np=np, compute_sematic_labels=semantic_labels_gpu,
                     label_to_membership=None)
    source_definitions(ROOT / "vendor/erase/scripts/loss.py", {"MaximalCodingRateReduction"}, namespace)
    return namespace["GAT_TYPE"], namespace["MaximalCodingRateReduction"]


@torch.no_grad()
def semantic_labels_gpu(X_train, Y_all, X, train_mask, beta, num_classes):
    labels = Y_all[train_mask].argmax(1)
    prototypes = []
    for i in range(num_classes):
        members = X_train[labels == i]
        if members.shape[0] == 0:
            raise ValueError(f"ERASE upstream prototype class {i} has no members")
        prototypes.append(members.mean(0))
    prototypes = torch.stack(prototypes)
    cosine = F.normalize(X, dim=1) @ F.normalize(prototypes, dim=1).T
    return F.softmax(beta * Y_all + (1 - beta) * cosine.abs(), dim=1)


def normalized_adjacency(edge_index, n_nodes, train_mask=None):
    row, col = edge_index
    if train_mask is not None:
        keep = train_mask[row] & train_mask[col]
        row, col = row[keep], col[keep]
    # Matches upstream normalize_adj_row: A @ diag(1 / row_sum).
    counts = torch.zeros(n_nodes, device=row.device).index_add_(0, row, torch.ones_like(row, dtype=torch.float))
    inv = counts.reciprocal()
    inv[~torch.isfinite(inv)] = 0
    return torch.sparse_coo_tensor(torch.stack([row, col]), inv[col], (n_nodes, n_nodes)).coalesce()


@torch.no_grad()
def erase_preprocess(data, alpha, steps):
    labels = F.one_hot(data.noisy_y, data.num_classes).float()
    labels[~data.train_mask] = 0
    partial = normalized_adjacency(data.edge_index, data.num_nodes, data.train_mask)
    adjacency = normalized_adjacency(data.edge_index, data.num_nodes)
    for _ in range(steps):
        labels = (1 - alpha) * torch.sparse.mm(partial, labels) + alpha * labels
    return labels, adjacency


def fit_gpu_logistic(features, labels, max_iter=1000):
    """Multinomial LBFGS, C=1, matching sklearn's regularized objective."""
    features = F.normalize(features.detach(), dim=1).double()
    classes = torch.unique(labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("logistic probe needs at least two training classes")
    encoded = torch.searchsorted(classes, labels)
    weight = nn.Parameter(features.new_zeros((classes.numel(), features.shape[1])))
    bias = nn.Parameter(features.new_zeros(classes.numel()))
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=max_iter, tolerance_grad=1e-4,
                                tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(features @ weight.T + bias, encoded)
        loss = loss + weight.square().sum() / (2 * features.shape[0])
        loss.backward()
        return loss
    optimizer.step(closure)
    return weight.detach(), bias.detach(), classes


@torch.no_grad()
def probe_probabilities(features, probe, num_classes):
    weight, bias, classes = probe
    logits = F.normalize(features, dim=1).double() @ weight.T + bias
    probs = F.softmax(logits, dim=1)
    result = features.new_zeros((features.shape[0], num_classes))
    result[:, classes] = probs.to(features.dtype)
    return result
