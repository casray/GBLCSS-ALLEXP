from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import GraphData
from gblcss.targets import TargetResult, build_leave_one_out_targets


@pytest.fixture
def metrics():
    assert importlib.util.find_spec("gblcss.diagnostics") is not None, "missing diagnostics API"
    return importlib.import_module("gblcss.diagnostics").target_metrics


def graph(clean, noisy, train=None):
    n = len(clean)
    index = torch.tensor(list(range(n)) if train is None else train, dtype=torch.long)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[index] = True
    return GraphData(
        "toy", torch.zeros(n, 2), torch.empty(2, 0, dtype=torch.long),
        torch.tensor(clean), mask, ~mask, ~mask,
        noisy_y=torch.tensor(noisy), train_index=index,
    )


def targets(data, assignment, min_support=2):
    return build_leave_one_out_targets(
        data.noisy_y, np.asarray(assignment), data.train_mask,
        data.num_classes, min_train_per_ball=min_support,
    )


def test_brier_class_sum_and_full_train_weighted_decomposition(metrics):
    data = graph([0, 1, 2, 0, 2], [1, 1, 2, 0, 0], train=[3, 0, 2, 1])
    soft = torch.tensor([[.5, .5, 0], [0, .75, .25], [0, 0, 1], [1, 0, 0], [1, 0, 0]])
    result = TargetResult(soft, torch.zeros(5), data.noisy_y.clone(), {})
    m = metrics(data, np.zeros(5, dtype=int), result)
    assert m["noisy_onehot_brier"] == .5
    assert m["soft_target_brier"] == .15625
    assert m["corrupted_brier_reduction"] == .375
    assert m["correct_brier_increase"] == .03125
    assert m["net_brier_reduction"] == .34375
    assert m["net_brier_reduction"] == pytest.approx(
        m["corrupted_brier_reduction"] - m["correct_brier_increase"]
    )
    assert m["train_nodes"] == 4
    assert m["corrupted_train_nodes"] == 1
    assert m["correct_train_nodes"] == 3
    assert all(type(value) in (int, float, type(None)) for value in m.values())
    json.dumps(m, allow_nan=False)


def test_clean_margin_is_loo_relative_to_target_truth_and_all_train_weighted(metrics):
    data = graph([0, 0, 0, 1, 2, 2], [1, 1, 1, 0, 2, 0], train=[4, 2, 0, 3, 1])
    assignment = np.array([10, 10, 10, 10, 40, 10])
    result = targets(data, assignment)
    m = metrics(data, assignment, result)
    # First three nodes have clean margins 1/3; node 3 has -1.
    # Node 4 has no other training voters; held-out node 5 never votes.
    assert m["positive_clean_vote_margin_coverage"] == pytest.approx(3 / 5)
    assert m["voting_support_coverage"] == pytest.approx(4 / 5)
    assert m["voting_supported_train_nodes"] == 4
    assert m["num_balls"] == 2
    assert m["train_balls"] == 2
    assert m["supported_balls"] == 1
    assert metrics(replace(data, train_index=None), assignment, result) == m
    assert metrics(data, assignment, result, min_support=4)["voting_support_coverage"] == 0


def test_clean_margin_ties_are_not_positive_and_self_vote_is_removed(metrics):
    data = graph([0, 0, 1, 2], [0, 0, 1, 2], train=[0, 1, 2])
    assignment = [0, 0, 0, 0]
    m = metrics(data, assignment, targets(data, assignment))
    # LOO margins are 0, 0, -1, not the ball's top-two margin of 1/3.
    assert m["voting_support_coverage"] == 1
    assert m["positive_clean_vote_margin_coverage"] == 0


def test_empty_voting_sets_do_not_become_ties_or_positive_margins(metrics):
    data = graph([0, 1, 2, 0], [1, 1, 2, 2], train=[0, 1])
    assignment = [5, 9, 5, 12]
    m = metrics(data, assignment, targets(data, assignment), min_support=1)
    assert m["voting_support_coverage"] == 0
    assert m["positive_clean_vote_margin_coverage"] == 0
    assert m["alpha_positive_coverage"] == 0
    assert m["tied_vote_train_nodes"] == m["tied_target_train_nodes"] == 0
    assert m["noisy_onehot_brier"] == m["soft_target_brier"] == 1
    assert m["net_brier_reduction"] == 0
    assert m["num_balls"] == 3
    assert m["train_balls"] == 2
    assert m["supported_balls"] == 0
    json.dumps(m, allow_nan=False)


@pytest.mark.parametrize("clean,expected_active", [([0, 1, 1, 2], 3), ([0, 1, 1, 1], 1)])
def test_tied_votes_are_distinct_from_alpha_positive_tied_targets(metrics, clean, expected_active):
    data = graph(clean, [0, 1, 1, 1], train=[0, 1, 2])
    assignment = [0, 0, 0, 0]
    result = targets(data, assignment)
    m = metrics(data, assignment, result)
    assert m["tied_vote_train_nodes"] == 2
    assert m["alpha_positive_train_nodes"] == expected_active
    assert m["alpha_positive_coverage"] == pytest.approx(expected_active / 3)
    assert m["tied_target_train_nodes"] == expected_active - 1
    assert m["tied_target_train_nodes"] == result.diagnostics["tied_train_nodes"]


def test_class_permutation_preserves_every_metric(metrics):
    data = graph([0, 0, 1, 2, 2, 1], [0, 1, 1, 2, 1, 0], train=[4, 0, 3, 1, 2])
    assignment = np.array([4, 4, 4, 4, 4, 99])
    result = targets(data, assignment)
    permutation = torch.tensor([2, 0, 1])
    permuted = replace(data, y=permutation[data.y], noisy_y=permutation[data.noisy_y])
    permuted_result = targets(permuted, assignment)
    torch.testing.assert_close(permuted_result.soft_target[:, permutation], result.soft_target)
    assert metrics(permuted, assignment, permuted_result) == pytest.approx(metrics(data, assignment, result))


def test_clean_labels_only_affect_diagnostics_and_inputs_are_not_mutated(metrics):
    data = graph([0, 0, 0, 1, 2], [1, 1, 1, 0, 2], train=[0, 1, 2, 3])
    assignment = np.array([0, 0, 0, 0, 1])
    result = targets(data, assignment)
    changed = replace(data, y=torch.tensor([1, 1, 0, 0, 2]))
    regenerated = targets(changed, assignment)
    for field in ("soft_target", "alpha", "pseudo_label"):
        assert torch.equal(getattr(result, field), getattr(regenerated, field))
    before_data, before_result, before_assignment = deepcopy(data), deepcopy(result), assignment.copy()
    m = metrics(data, assignment, result)
    changed_m = metrics(changed, assignment, result)
    assert m["positive_clean_vote_margin_coverage"] == .75
    assert changed_m["positive_clean_vote_margin_coverage"] == 0
    assert m["noisy_onehot_brier"] != changed_m["noisy_onehot_brier"]
    for field in ("y", "noisy_y", "train_mask", "train_index"):
        assert torch.equal(getattr(data, field), getattr(before_data, field))
    for field in ("soft_target", "alpha", "pseudo_label"):
        assert torch.equal(getattr(result, field), getattr(before_result, field))
    assert result.diagnostics == before_result.diagnostics
    np.testing.assert_array_equal(assignment, before_assignment)


def test_heldout_labels_and_cached_diagnostics_are_ignored(metrics):
    data = graph([0, 0, 1, 2, 2], [1, 0, 1, 2, 0], train=[2, 0, 1])
    assignment = [0, 0, 0, 0, 1]
    result = targets(data, assignment)
    expected = metrics(data, assignment, result)
    changed = deepcopy(data)
    changed.y[~data.train_mask] = -1
    changed.noisy_y[~data.train_mask] = -1
    stale = replace(
        result, soft_target=result.soft_target.clone().requires_grad_(),
        diagnostics={"num_balls": -100, "tied_train_nodes": 900},
        pseudo_label=torch.full_like(result.pseudo_label, -1),
    )
    assert metrics(changed, assignment, stale) == expected
    assert stale.soft_target.grad is None


@pytest.mark.parametrize("noisy", [[0, 1, 2], [1, 2, 0]])
def test_empty_corruption_or_correct_subset_has_zero_contribution(metrics, noisy):
    data = graph([0, 1, 2], noisy)
    assignment = [0, 0, 0]
    m = metrics(data, assignment, targets(data, assignment))
    key = "corrupted_brier_reduction" if noisy == [0, 1, 2] else "correct_brier_increase"
    assert m[key] == 0
    assert m["net_brier_reduction"] == pytest.approx(
        m["corrupted_brier_reduction"] - m["correct_brier_increase"]
    )


def test_empty_training_set_returns_null_means_and_zero_counts(metrics):
    data = graph([0, 1, 2], [1, 1, 2], train=[])
    assignment = [0, 0, 1]
    m = metrics(data, assignment, targets(data, assignment))
    assert m["train_nodes"] == m["train_balls"] == m["supported_balls"] == 0
    assert m["num_balls"] == 2
    assert m["noisy_onehot_brier"] is None
    assert m["net_brier_reduction"] is None
    assert m["positive_clean_vote_margin_coverage"] is None
    json.dumps(m, allow_nan=False)


@pytest.mark.parametrize("minimum", [0, -1, 1.5])
def test_invalid_support_is_rejected(metrics, minimum):
    data = graph([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="min_support"):
        metrics(data, [0, 0, 0], targets(data, [0, 0, 0]), min_support=minimum)


def test_missing_noisy_labels_and_misaligned_inputs_are_rejected(metrics):
    data = graph([0, 1, 2], [0, 1, 2])
    result = targets(data, [0, 0, 0])
    with pytest.raises(ValueError, match="noisy_y"):
        metrics(replace(data, noisy_y=None), [0, 0, 0], result)
    with pytest.raises(ValueError, match="assignment"):
        metrics(data, [0, 0], result)
    with pytest.raises(ValueError, match="soft_target"):
        metrics(data, [0, 0, 0], replace(result, soft_target=result.soft_target[:2]))
