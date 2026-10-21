"""Read-only, fresh training-target diagnostics for Figure 4 and Appendix I.2."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import GraphData
from .targets import TargetResult, _indices


@torch.no_grad()
def target_metrics(
    data: GraphData,
    assignment: np.ndarray,
    target_result: TargetResult,
    min_support: int = 2,
) -> dict[str, float | int | None]:
    """Measure supplied targets without generating or modifying training targets.

    Brier scores sum squared errors over classes, then average over ALL N
    training nodes. ``corrupted_brier_reduction`` sums (noisy error - soft
    error) on corrupted nodes / N; ``correct_brier_increase`` sums (soft
    error - noisy error) on correct nodes / N. These are contributions, NOT
    subgroup means; their difference equals ``net_brier_reduction``.

    All coverages are fractions of N (not percentages). A node is supported
    when at least ``min_support`` OTHER training nodes occupy its ball.
    Its clean vote margin is p(clean class of this node) minus the largest
    p(other class), using those voters' CLEAN labels, excluding the node.
    Unsupported nodes and zero margins count false in positive coverage.
    Clean labels are used only for these retrospective measurements.

    ``tied_vote_train_nodes`` counts supported noisy LOO histograms with
    multiple maxima; ``tied_target_train_nodes`` additionally requires the
    supplied alpha > 0. Alpha coverage uses the supplied alpha independently
    of support. ``num_balls`` includes all occupied assignment IDs;
    ``train_balls`` includes balls with training nodes; ``supported_balls``
    includes balls with at least min_support + 1 training nodes.

    An empty training set has zero counts and None (JSON null) for means and
    coverages. Missing noisy labels are an error, never replaced with truth.
    All outputs are native JSON scalars; cached target diagnostics are unused.
    """
    if not isinstance(min_support, (int, np.integer)) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if data.noisy_y is None:
        raise ValueError("data.noisy_y is required for target diagnostics")
    assignment = np.asarray(assignment)
    if assignment.shape != (data.num_nodes,):
        raise ValueError("assignment must match the number of nodes")
    soft = target_result.soft_target.detach().cpu().double()
    if soft.ndim != 2 or soft.shape[0] != data.num_nodes or soft.shape[1] < 2:
        raise ValueError("soft_target must have shape (num_nodes, num_classes >= 2)")
    alpha = target_result.alpha.detach().cpu()
    if alpha.shape != (data.num_nodes,):
        raise ValueError("alpha must match the number of nodes")
    train = _indices(data.train_index if data.train_index is not None else data.train_mask)
    n = int(train.size)
    clean = data.y.detach().cpu().long()[train]
    noisy = data.noisy_y.detach().cpu().long()[train]
    n_classes = int(soft.shape[1])
    truth = F.one_hot(clean, n_classes).double()
    observed = F.one_hot(noisy, n_classes).double()
    noisy_error = (observed - truth).square().sum(dim=1)
    soft_error = (soft[train] - truth).square().sum(dim=1)
    corrupted = clean != noisy
    positive_alpha = alpha[train] > 0
    supported = torch.zeros(n, dtype=torch.bool)
    positive_margin = torch.zeros(n, dtype=torch.bool)
    tied = torch.zeros(n, dtype=torch.bool)
    train_assignment = assignment[train]
    train_balls = np.unique(train_assignment)
    supported_balls = 0

    for ball_id in train_balls:
        members = np.flatnonzero(train_assignment == ball_id)
        support = int(members.size) - 1
        if support < min_support:
            continue
        supported_balls += 1
        supported[members] = True
        # Subtract each target node's own vote, not a ball-wide top-two gap.
        clean_votes = truth[members].sum(dim=0) - truth[members]
        own_votes = clean_votes.gather(1, clean[members, None]).squeeze(1)
        clean_votes.scatter_(1, clean[members, None], -1.0)
        margins = (own_votes - clean_votes.max(dim=1).values) / support
        positive_margin[members] = margins > 0
        noisy_votes = observed[members].sum(dim=0) - observed[members]
        maxima = noisy_votes.max(dim=1, keepdim=True).values
        tied[members] = (noisy_votes == maxima).sum(dim=1) > 1

    def total_train_mean(values: torch.Tensor) -> float | None:
        return float(values.sum().item() / n) if n else None

    return {
        "train_nodes": n,
        "corrupted_train_nodes": int(corrupted.sum().item()),
        "correct_train_nodes": int((~corrupted).sum().item()),
        "noisy_onehot_brier": total_train_mean(noisy_error),
        "soft_target_brier": total_train_mean(soft_error),
        "net_brier_reduction": total_train_mean(noisy_error - soft_error),
        "corrupted_brier_reduction": total_train_mean((noisy_error - soft_error)[corrupted]),
        "correct_brier_increase": total_train_mean((soft_error - noisy_error)[~corrupted]),
        "voting_supported_train_nodes": int(supported.sum().item()),
        "voting_support_coverage": total_train_mean(supported),
        "positive_clean_vote_margin_train_nodes": int(positive_margin.sum().item()),
        "positive_clean_vote_margin_coverage": total_train_mean(positive_margin),
        "alpha_positive_train_nodes": int(positive_alpha.sum().item()),
        "alpha_positive_coverage": total_train_mean(positive_alpha),
        "tied_vote_train_nodes": int(tied.sum().item()),
        "tied_target_train_nodes": int((tied & positive_alpha).sum().item()),
        "num_balls": int(np.unique(assignment).size),
        "train_balls": int(train_balls.size),
        "supported_balls": supported_balls,
    }
