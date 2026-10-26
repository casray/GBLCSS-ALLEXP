"""Label-free candidate tree and nearest-neighbor measure preserving cut."""
from __future__ import annotations

import heapq
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or min(x.shape) == 0:
        raise ValueError('Expected a finite nonempty matrix')
    scale = np.maximum(np.abs(x).max(1, keepdims=True), np.finfo(float).tiny)
    scaled = x / scale
    return scaled / np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), np.finfo(float).tiny)


def neighbors(x, global_ids, k):
    x, ids = np.asarray(x, dtype=np.float64), np.asarray(global_ids)
    if x.ndim != 2 or not np.isfinite(x).all() or len(ids) != len(x) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid neighbor input')
    if not 0 < k < len(x):
        raise ValueError('k must exclude the query')
    result = np.empty((len(x), k), dtype=np.int64)
    # Subtraction gives exact tie behavior for duplicate rows, unlike Gram subtraction.
    for row in range(len(x)):
        delta = x - x[row]
        distance = np.einsum('ij,ij->i', delta, delta)
        distance[row] = np.inf
        result[row] = np.lexsort((ids, distance))[:k]
    return result


def tv_cost(overlap, k, support):
    if k <= 0 or support <= 0:
        raise ValueError('Two nonempty voting measures are required')
    return 1 - np.asarray(overlap) / max(k, support)


def assignment(nodes, leaves, size):
    out = np.full(size, -1, np.int64)
    for group, leaf in enumerate(sorted(leaves, key=lambda j: int(nodes[j]['members'][0]))):
        members = nodes[leaf]['members']
        if (out[members] != -1).any():
            raise ValueError('Overlapping cut')
        out[members] = group
    if (out < 0).any():
        raise ValueError('Incomplete cut')
    return out


def build_tree(x, train, max_leaves=256, snapshot_leaves=64, min_nodes=32, min_train=3, seed=0):
    x = normalize(x)
    train = np.asarray(train, dtype=np.int64)
    if train.ndim != 1 or len(train) != len(np.unique(train)) or len(train) < min_train:
        raise ValueError('Invalid or insufficient training indices')
    if train.min() < 0 or train.max() >= len(x) or min(max_leaves, snapshot_leaves, min_nodes, min_train) < 1:
        raise ValueError('Invalid tree limits')
    train_mask = np.zeros(len(x), bool)
    train_mask[train] = True
    nodes, frontier, queue = [], set(), []

    def append(members, depth):
        members = np.sort(members)
        delta = x[members] - x[members].mean(0)
        node = dict(members=members, children=(), train_count=int(train_mask[members].sum()),
                    sse=float(np.einsum('ij,ij->', delta, delta)), depth=depth)
        nodes.append(node)
        return len(nodes)-1

    def propose(idx):
        node = nodes[idx]
        members = node['members']
        if len(members) < 2*min_nodes or node['train_count'] < 2*min_train or node['sse'] <= 1e-12:
            return
        labels = KMeans(2, init='k-means++', n_init=10, max_iter=300, algorithm='lloyd',
                        random_state=(seed+idx) % (2**32-1)).fit_predict(x[members])
        parts = [members[labels == i] for i in range(2)]
        if any(len(p) < min_nodes or train_mask[p].sum() < min_train for p in parts):
            return
        parts.sort(key=lambda p: int(p[0]))
        cost = 0.
        for part in parts:
            delta = x[part] - x[part].mean(0)
            cost += np.einsum('ij,ij->', delta, delta)
        gain = node['sse'] - cost
        if gain <= 1e-12*max(1., node['sse']):
            return
        heapq.heappush(queue, (-gain, int(members[0]), idx, parts))

    with threadpool_limits(limits=1):
        frontier.add(append(np.arange(len(x)), 0))
        propose(0)
        snapshot = list(frontier) if snapshot_leaves == 1 else None
        while queue and len(frontier) < max_leaves:
            _, _, idx, parts = heapq.heappop(queue)
            child = tuple(append(part, nodes[idx]['depth']+1) for part in parts)
            nodes[idx]['children'] = child
            frontier.remove(idx)
            frontier.update(child)
            if len(frontier) == snapshot_leaves:
                snapshot = list(frontier)
            for j in child:
                propose(j)
    fine = sorted(frontier)
    snapshot = snapshot if snapshot is not None else fine
    return dict(nodes=nodes, fine_leaves=fine, snapshot_leaves=sorted(snapshot),
                fine_assignment=assignment(nodes, fine, len(x)),
                snapshot_assignment=assignment(nodes, snapshot, len(x)))


def node_costs(nodes, train, near):
    train, near = np.asarray(train), np.asarray(near)
    k = near.shape[1]
    costs = np.zeros(len(nodes))
    for idx, node in enumerate(nodes):
        selected = np.isin(train, node['members'])
        rows = np.flatnonzero(selected)
        support = len(rows)-1
        if support <= 0:
            raise ValueError('Tree node has insufficient donors')
        overlap = selected[near[rows]].sum(1)
        costs[idx] = tv_cost(overlap, k, support).sum()
    return costs


def select_cut(nodes, costs):
    costs = np.asarray(costs, dtype=float)
    if len(costs) != len(nodes) or not np.isfinite(costs).all():
        raise ValueError('Invalid costs')
    values = costs.copy()
    split = np.zeros(len(nodes), bool)
    for idx in range(len(nodes)-1, -1, -1):
        children = nodes[idx]['children']
        if children:
            total = sum(values[c] for c in children)
            if total < costs[idx] - 1e-12*nodes[idx]['train_count']:
                values[idx], split[idx] = total, True
    chosen, stack = [], [0]
    while stack:
        idx = stack.pop()
        if split[idx]:
            stack.extend(reversed(nodes[idx]['children']))
        else:
            chosen.append(idx)
    return chosen, float(values[0])


def targets_from_counts(z, counts, beta=.5, min_votes=2):
    if counts.ndim != 2 or len(z) != len(counts) or (counts < 0).any() or not torch.isfinite(counts).all():
        raise ValueError('Invalid counts')
    classes = counts.shape[1]
    if classes < 2 or z.min() < 0 or z.max() >= classes:
        raise ValueError('Invalid classes')
    own = F.one_hot(z.long(), classes).to(counts.dtype)
    support = counts.sum(1)
    vote = counts / support.clamp_min(1)[:, None]
    maximum, winner = counts.max(1)
    alpha = ((maximum/support.clamp_min(1)-1/classes)/(1-1/classes)).clamp(0, 1)
    alpha = torch.where(support >= min_votes, alpha, 0.)
    tie = (counts == maximum[:, None]).sum(1) > 1
    strength = torch.where(tie, 0., beta)
    q = (1-strength[:, None])*vote + strength[:, None]*F.one_hot(winner, classes)
    target = (1-alpha[:, None])*own + alpha[:, None]*q
    return target, alpha


def neighbor_counts(z, near, support, classes):
    z, near, support = np.asarray(z), np.asarray(near), np.asarray(support)
    if near.shape[0] != len(z) or len(support) != len(z) or (support < 0).any() or (support > near.shape[1]).any():
        raise ValueError('Invalid support')
    counts = np.zeros((len(z), classes), dtype=np.float32)
    for row, size in enumerate(support):
        donors = near[row, :int(size)]
        if row in donors or len(np.unique(donors)) != len(donors):
            raise ValueError('Self/duplicate donor')
        counts[row] = np.bincount(z[donors], minlength=classes)
    return counts
