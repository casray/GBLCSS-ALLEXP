"""CPU-only unlabeled ball trees and exact fixed-budget additive tree cuts."""
from dataclasses import dataclass
import heapq
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


@dataclass
class BallTree:
    members: list[np.ndarray]
    children: np.ndarray
    parent: np.ndarray
    costs: np.ndarray

    @property
    def num_leaves(self):
        return int(np.count_nonzero(self.children[:, 0] < 0))


def _sse(points):
    centered = points - points.mean(axis=0)
    return float(np.einsum("ij,ij->", centered, centered))


def build_tree(z, max_leaves, seed=20261019, min_leaf=2):
    """Expand the highest-SSE feasible leaf, never consulting labels or splits.

    A failed/minimum-size-violating 2-means partition falls back to a stable
    median split along the maximum-variance input axis, including identical z.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not len(z) or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("z must be a nonempty, finite N by D matrix")
    if int(max_leaves) != max_leaves or max_leaves < 1:
        raise ValueError("max_leaves must be a positive integer")
    if int(min_leaf) != min_leaf or min_leaf < 1:
        raise ValueError("min_leaf must be a positive integer")
    max_leaves, min_leaf = int(max_leaves), int(min_leaf)
    members = [np.arange(len(z), dtype=np.int64)]
    children = [[-1, -1]]
    parent = [-1]
    costs = [_sse(z)]
    pending = [(-costs[0], 0)] if len(z) >= 2 * min_leaf else []
    leaf_count = 1
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        while pending and leaf_count < max_leaves:
            _, node = heapq.heappop(pending)
            ids = members[node]
            points = z[ids]
            # The construction seed is independent of all split/noise seeds.
            partition = KMeans(n_clusters=2, n_init=3, max_iter=100,
                               random_state=int(seed), algorithm="lloyd").fit_predict(points)
            left, right = ids[partition == 0], ids[partition == 1]
            if min(len(left), len(right)) < min_leaf:
                axis = int(np.argmax(np.var(points, axis=0)))
                order = np.argsort(points[:, axis], kind="stable")
                middle = len(ids) // 2
                left, right = ids[order[:middle]], ids[order[middle:]]
            child_ids = []
            for subset in (left, right):
                child = len(members)
                child_ids.append(child)
                members.append(np.sort(subset))
                children.append([-1, -1])
                parent.append(node)
                costs.append(_sse(z[subset]))
                if len(subset) >= 2 * min_leaf:
                    heapq.heappush(pending, (-costs[-1], child))
            children[node] = child_ids
            leaf_count += 1
    return BallTree(members, np.asarray(children, dtype=np.int64),
                    np.asarray(parent, dtype=np.int64), np.asarray(costs))


def _label_costs(tree, labels, train_mask, n_classes, beta):
    n = len(tree.members[0])
    labels, mask = np.asarray(labels), np.asarray(train_mask)
    if labels.shape != (n,) or mask.shape != (n,) or mask.dtype != np.bool_:
        raise ValueError("labels and boolean train_mask must each have N entries")
    if n_classes < 2 or int(n_classes) != n_classes:
        raise ValueError("n_classes must be an integer >= 2")
    # Only the permitted support is read; held-out values can be arbitrary.
    supported_labels = labels[mask]
    if (not np.issubdtype(supported_labels.dtype, np.integer)
            or np.any(supported_labels < 0) or np.any(supported_labels >= n_classes)):
        raise ValueError("training labels must be integer class IDs in [0,n_classes)")
    total_support = int(mask.sum())
    prior = 1. - 1. / n_classes
    count = np.empty(len(tree.members), dtype=np.int64)
    impurity = np.empty(len(tree.members))
    inherited = np.empty(len(tree.members), dtype=bool)
    for node, ids in enumerate(tree.members):
        labeled = ids[mask[ids]]
        support = len(labeled)
        count[node] = support
        inherited[node] = support < 2
        if support >= 2:
            frequencies = np.bincount(labels[labeled], minlength=int(n_classes)).astype(np.float64)
            impurity[node] = 1. - np.sum(frequencies * (frequencies - 1)) / (support * (support - 1))
        else:
            ancestor = int(tree.parent[node])
            impurity[node] = impurity[ancestor] if ancestor >= 0 else prior
    costs = beta * count / max(1, total_support) * impurity / prior
    return costs, count, inherited, total_support


def _exact_cut(tree, budget, node_costs):
    """Tree knapsack with vectorized min-plus merges and exact traceback.

    dp[v][k] is the optimum k-leaf cut rooted at v. k=1 selects v; k>=2
    combines both children. The smaller child is the Python loop dimension;
    all choices in the other child are evaluated in a NumPy vector.
    """
    size = len(tree.members)
    dp = [None] * size
    decisions = [None] * size
    capacity = np.ones(size, dtype=np.int64)
    for node in range(size - 1, -1, -1):
        left, right = tree.children[node]
        if left < 0:
            dp[node] = np.array([np.inf, node_costs[node]])
            continue
        left, right = int(left), int(right)
        cap = min(budget, int(capacity[left] + capacity[right]))
        capacity[node] = cap
        values = np.full(cap + 1, np.inf)
        values[1] = node_costs[node]
        choice = np.zeros(cap + 1, dtype=np.int32)
        a, b = dp[left], dp[right]
        swapped = len(a) > len(b)
        small, large = (b, a) if swapped else (a, b)
        for small_k in range(1, min(len(small) - 1, cap - 1) + 1):
            other_max = min(len(large) - 1, cap - small_k)
            candidate = small[small_k] + large[1:other_max + 1]
            destination = values[small_k + 1:small_k + other_max + 1]
            improve = candidate < destination
            np.copyto(destination, candidate, where=improve)
            left_k = np.arange(1, other_max + 1, dtype=np.int32) if swapped else small_k
            np.copyto(choice[small_k + 1:small_k + other_max + 1], left_k, where=improve)
        dp[node], decisions[node] = values, choice
    selected = []
    pending = [(0, budget)]
    while pending:
        node, k = pending.pop()
        if k == 1:
            selected.append(node)
            continue
        left, right = tree.children[node]
        left_k = int(decisions[node][k])
        if left_k < 1 or not np.isfinite(dp[node][k]):
            raise RuntimeError("exact cut traceback encountered an infeasible cardinality")
        pending.extend([(int(right), k - left_k), (int(left), left_k)])
    return sorted(selected)


def select_cut(tree, budget, labels, train_mask, n_classes, rule="R0", beta=1.0):
    """Return a contiguous assignment and diagnostics for exactly budget balls.

    R0 minimizes input-space SSE only. R3/O2 add a training-support-weighted,
    inherited unbiased pair-disagreement cost. O2 differs only in the clean
    training labels supplied by the caller; this API has no held-out oracle.
    """
    if int(budget) != budget or budget < 1:
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    if budget > tree.num_leaves:
        raise ValueError(f"requested budget {budget} exceeds available candidate leaves {tree.num_leaves}")
    if rule not in ("R0", "R3", "O2"):
        raise ValueError(f"unsupported cut rule {rule!r}")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if rule == "R0":
        label_costs = np.zeros(len(tree.members))
        support = np.zeros(len(tree.members), dtype=np.int64)
        inherited = np.zeros(len(tree.members), dtype=bool)
        total_support = 0
    else:
        label_costs, support, inherited, total_support = _label_costs(
            tree, labels, train_mask, n_classes, beta)
    selected = _exact_cut(tree, budget, tree.costs + label_costs)
    assignment = np.full(len(tree.members[0]), -1, dtype=np.int64)
    for cluster, node in enumerate(selected):
        ids = tree.members[node]
        if np.any(assignment[ids] >= 0):
            raise RuntimeError("tree cut overlaps")
        assignment[ids] = cluster
    if len(selected) != budget or np.any(assignment < 0):
        raise RuntimeError("tree cut does not cover N nodes at the requested budget")
    chosen = np.asarray(selected, dtype=np.int64)
    d_total = float(np.sum(tree.costs[chosen]))
    label_total = float(np.sum(label_costs[chosen]))
    diagnostics = {
        "actual_M": len(selected), "requested_M": budget,
        "candidate_leaves": tree.num_leaves,
        "objective_total": d_total + label_total, "D_total": d_total,
        "label_total": label_total, "selected_nodes": selected,
        "effective_rule": rule, "training_label_support": total_support,
        "inherited_label_fraction": float(np.sum(support[chosen] * inherited[chosen]) / max(1, total_support)),
        "label_support_histogram": {"0": int(np.sum(support[chosen] == 0)),
                                    "1": int(np.sum(support[chosen] == 1)),
                                    "2plus": int(np.sum(support[chosen] >= 2))},
        "global_label_fallback": "uniform_class_prior" if rule != "R0" and total_support < 2 else None,
    }
    return assignment, diagnostics
