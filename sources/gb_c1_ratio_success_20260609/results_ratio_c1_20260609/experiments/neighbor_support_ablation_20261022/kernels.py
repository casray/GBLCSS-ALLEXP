"""Label-free CPU kernels over an existing binary candidate tree.

Use the old generator's select_cut with support_costs for the support-only DP.
No candidate generation, assignment conversion, or training is performed here.
"""
from __future__ import annotations

from collections.abc import Mapping
import heapq
import math
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite integer >= {minimum}')
    if isinstance(value, Integral):
        result = int(value)
    else:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f'{name} must be a finite integer >= {minimum}')
        result = int(numeric)
    if result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


def _node_list(nodes):
    try:
        result = list(nodes)
    except TypeError as exc:
        raise ValueError('Expected a nonempty sequence of nodes') from exc
    if not result or any(not isinstance(node, Mapping) for node in result):
        raise ValueError('Expected a nonempty sequence of node mappings')
    return result


def _field(node, name):
    try:
        return node[name]
    except KeyError as exc:
        raise ValueError(f'Missing node field: {name}') from exc


def _indices(value, name, ndim=1):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {name}') from exc
    if (result.ndim != ndim or not result.size or result.dtype.kind not in 'iu'
            or (result < 0).any() or (result > np.iinfo(np.intp).max).any()):
        raise ValueError(f'{name} must contain nonnegative integer indices')
    return result.astype(np.intp, copy=False)


def _members(node):
    members = _indices(_field(node, 'members'), 'members')
    if len(np.unique(members)) != len(members):
        raise ValueError('Duplicate node members')
    return members


def support_costs(nodes, k) -> np.ndarray:
    """Return m * abs((m - 1) - k) / max(m - 1, k) per node.

    Counts are finite integers with k >= 1 and train_count >= 2. Only
    train_count is needed; membership and neighbor information are not used.
    """
    nodes = _node_list(nodes)
    k = _integer(k, 'k', 1)
    result = np.empty(len(nodes), dtype=np.float64)
    for idx, node in enumerate(nodes):
        m = _integer(_field(node, 'train_count'), 'train_count', 2)
        s = m-1
        try:
            result[idx] = m * abs(s-k) / max(s, k)
        except OverflowError as exc:
            raise ValueError('Support cost exceeds finite float64 range') from exc
    if not np.isfinite(result).all():
        raise ValueError('Support cost exceeds finite float64 range')
    return result


def cost_components(nodes, train, near) -> tuple[np.ndarray, np.ndarray]:
    """Split old node TV costs into support-size and membership penalties.

    train contains global IDs; near contains row indices into train, with no
    self or duplicate donors. Each node's stored count must match its selected
    training rows. Membership costs sum (min(s, k) - overlap) / max(s, k).
    """
    nodes = _node_list(nodes)
    train = _indices(train, 'train')
    near = _indices(near, 'near', ndim=2)
    k = near.shape[1]
    if (len(np.unique(train)) != len(train) or len(near) != len(train)
            or k >= len(train) or (near >= len(train)).any()):
        raise ValueError('Invalid training IDs or neighbor shape/range')
    if ((near == np.arange(len(train))[:, None]).any()
            or (np.diff(np.sort(near, axis=1), axis=1) == 0).any()):
        raise ValueError('Self or duplicate neighbor donors')
    support = support_costs(nodes, k)
    membership = np.zeros(len(nodes), dtype=np.float64)
    for idx, node in enumerate(nodes):
        selected = np.isin(train, _members(node))
        rows = np.flatnonzero(selected)
        m = len(rows)
        if m < 2 or m != node['train_count']:
            raise ValueError('train_count does not match sufficient training rows')
        s = m-1
        overlap = selected[near[rows]].sum(axis=1)
        membership[idx] = ((min(s, k)-overlap) / max(s, k)).sum()
    return support, membership


def _validated_tree(nodes):
    children, members, sse = [], [], []
    parents = np.zeros(len(nodes), dtype=np.int64)
    for idx, node in enumerate(nodes):
        members.append(_members(node))
        error = _field(node, 'sse')
        if isinstance(error, (bool, np.bool_)) or not isinstance(error, Real):
            raise ValueError('sse must be finite and nonnegative')
        try:
            error = float(error)
        except OverflowError as exc:
            raise ValueError('sse must be finite and nonnegative') from exc
        if not math.isfinite(error) or error < 0:
            raise ValueError('sse must be finite and nonnegative')
        sse.append(error)
        try:
            child = tuple(_integer(c, 'child index') for c in _field(node, 'children'))
        except TypeError as exc:
            raise ValueError('children must be a sequence') from exc
        if len(child) not in (0, 2) or len(set(child)) != len(child):
            raise ValueError('Expected zero or two distinct children')
        for c in child:
            if c >= len(nodes) or c == idx:
                raise ValueError('Invalid child index')
            parents[c] += 1
        children.append(child)
    if parents[0] != 0 or (parents[1:] != 1).any():
        raise ValueError('Nodes must form one tree rooted at index 0')
    seen, stack = set(), [0]
    while stack:
        idx = stack.pop()
        if idx in seen:
            raise ValueError('Cyclic candidate tree')
        seen.add(idx)
        stack.extend(children[idx])
    if len(seen) != len(nodes):
        raise ValueError('Unreachable candidate nodes')
    gains = np.zeros(len(nodes), dtype=np.float64)
    for idx, child in enumerate(children):
        if child:
            combined = np.concatenate([members[c] for c in child])
            if not np.array_equal(np.sort(combined), np.sort(members[idx])):
                raise ValueError('Children must partition their parent members')
            try:
                gains[idx] = sse[idx] - math.fsum(sse[c] for c in child)
            except OverflowError as exc:
                raise ValueError('Nonfinite SSE gain') from exc
            if not math.isfinite(gains[idx]):
                raise ValueError('Nonfinite SSE gain')
    return children, members, gains


def count_matched_cut(nodes, count) -> list[int]:
    """Greedily split the available frontier to exactly count tree leaves.

    Priority is descending SSE gain, then smallest global member, then node
    index. The existing candidate tree must be binary, rooted at node 0, with
    valid member partitions and finite nonnegative SSEs. Zero/negative gains
    remain eligible, since this control matches a count rather than a cost.
    """
    nodes = _node_list(nodes)
    count = _integer(count, 'count', 1)
    children, members, gains = _validated_tree(nodes)
    if count > sum(not child for child in children):
        raise ValueError('Requested leaf count is infeasible')
    leaves, queue = {0}, []

    def enqueue(idx):
        if children[idx]:
            heapq.heappush(queue, (-float(gains[idx]), int(members[idx].min()), idx))

    enqueue(0)
    while len(leaves) < count:
        if not queue:
            raise ValueError('Requested leaf count is infeasible')
        _, _, idx = heapq.heappop(queue)
        leaves.remove(idx)
        leaves.update(children[idx])
        for c in children[idx]:
            enqueue(c)
    return sorted(leaves)
