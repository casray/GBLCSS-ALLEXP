from __future__ import annotations

import hashlib

import numpy as np
import pytest

from utils.noise_aware_gb import (
    build_crossfit_budgeted_targets,
    build_geometry_candidate_tree,
    build_prefixed_geometry_candidate_tree,
    cut_candidate_tree_fixed_budget,
    label_assisted_split_score,
    make_stratified_crossfit_folds,
)
from utils.ordinary_gb import build_ordinary_granular_balls


def _tree_fixture(max_leaves: int = 8):
    rng = np.random.default_rng(17)
    clusters = [
        rng.normal(loc=(-3.0, -3.0), scale=0.15, size=(20, 2)),
        rng.normal(loc=(-3.0, 3.0), scale=0.15, size=(20, 2)),
        rng.normal(loc=(3.0, -3.0), scale=0.15, size=(20, 2)),
        rng.normal(loc=(3.0, 3.0), scale=0.15, size=(20, 2)),
    ]
    embeddings = np.vstack(clusters).astype(np.float32)
    labels = np.repeat(np.array([0, 0, 1, 1], dtype=np.int64), 20)
    train_idx = np.arange(labels.size, dtype=np.int64)
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=max_leaves,
        min_ball_size=5,
        min_geometry_gain=0.0,
        random_state=3,
    )
    return tree, labels, train_idx


def test_label_assisted_score_is_symmetric_and_bounded():
    left = np.array([0.8, 0.2], dtype=np.float64)
    right = np.array([0.2, 0.8], dtype=np.float64)

    score_lr, jsd_lr = label_assisted_split_score(0.25, left, right)
    score_rl, jsd_rl = label_assisted_split_score(0.25, right, left)

    assert score_lr == pytest.approx(score_rl)
    assert jsd_lr == pytest.approx(jsd_rl)
    assert 0.25 <= score_lr <= 0.5


def test_equal_child_distributions_leave_geometry_score_unchanged():
    score, normalized_jsd = label_assisted_split_score(
        0.25,
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    )

    assert normalized_jsd == pytest.approx(0.0)
    assert score == pytest.approx(0.25)


def test_budgeted_cut_returns_exact_target_ball_count():
    tree, labels, train_idx = _tree_fixture(max_leaves=8)

    result = cut_candidate_tree_fixed_budget(
        tree,
        target_num_balls=4,
        guidance_indices=train_idx,
        guidance_labels=labels,
        n_classes=2,
        label_assisted=True,
    )

    assert result.num_balls == 4
    assert np.unique(result.assignment).size == 4
    assert len(result.selected_split_ids) == 3


def test_budgeted_cut_rejects_unreachable_budget():
    tree, labels, train_idx = _tree_fixture(max_leaves=4)

    with pytest.raises(ValueError, match="cannot reach target_num_balls"):
        cut_candidate_tree_fixed_budget(
            tree,
            target_num_balls=5,
            guidance_indices=train_idx,
            guidance_labels=labels,
            n_classes=2,
        )


def test_prefixed_candidate_tree_reproduces_formal_partition_before_extension():
    rng = np.random.default_rng(31)
    embeddings = np.vstack(
        [
            rng.normal(loc=(-3.0, -3.0), scale=0.2, size=(20, 2)),
            rng.normal(loc=(-3.0, 3.0), scale=0.2, size=(20, 2)),
            rng.normal(loc=(3.0, -3.0), scale=0.2, size=(20, 2)),
            rng.normal(loc=(3.0, 3.0), scale=0.2, size=(20, 2)),
        ]
    ).astype(np.float32)
    formal = build_ordinary_granular_balls(
        embeddings,
        max_balls=4,
        min_ball_size=5,
        split_improvement=0.05,
        random_state=7,
    )
    tree = build_prefixed_geometry_candidate_tree(
        embeddings,
        expected_prefix_leaves=int(formal.summary["num_balls"]),
        prefix_max_leaves=4,
        max_leaves=8,
        min_ball_size=5,
        prefix_split_improvement=0.05,
        random_state=7,
    )
    formal_hash = hashlib.sha256(
        np.ascontiguousarray(formal.assignment, dtype="<i8").tobytes(order="C")
    ).hexdigest()

    assert tree.config["prefix_leaf_count"] == formal.summary["num_balls"]
    assert tree.config["prefix_assignment_hash"] == formal_hash
    assert len(tree.leaf_ids) >= formal.summary["num_balls"]


def test_crossfit_budgeted_targets_exclude_heldout_label_from_guidance_and_votes():
    tree, labels, train_idx = _tree_fixture(max_leaves=8)
    labels = labels.copy()
    labels[[1, 41]] = labels[[41, 1]]
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=labels.shape[0],
    )
    result_a = build_crossfit_budgeted_targets(
        tree,
        labels,
        train_idx,
        folds,
        n_classes=2,
        target_num_balls=4,
        guidance_mode="noisy",
    )
    eligible = train_idx[
        (result_a.alpha[train_idx] > 0.0) & (result_a.alpha[train_idx] < 1.0)
    ]
    assert eligible.size > 0
    node = int(eligible[0])
    changed = labels.copy()
    changed[node] = 1 - changed[node]
    result_b = build_crossfit_budgeted_targets(
        tree,
        changed,
        train_idx,
        folds,
        n_classes=2,
        target_num_balls=4,
        guidance_mode="noisy",
    )

    assert result_a.ball_ids[node] == result_b.ball_ids[node]
    np.testing.assert_allclose(
        result_a.local_distributions[node],
        result_b.local_distributions[node],
    )
    assert result_a.alpha[node] == pytest.approx(result_b.alpha[node])
    assert not np.allclose(result_a.soft_targets[node], result_b.soft_targets[node])


def test_geometry_only_crossfit_partitions_are_label_invariant():
    tree, labels, train_idx = _tree_fixture(max_leaves=8)
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=labels.shape[0],
    )
    result_a = build_crossfit_budgeted_targets(
        tree,
        labels,
        train_idx,
        folds,
        n_classes=2,
        target_num_balls=4,
        guidance_mode="geometry",
    )
    result_b = build_crossfit_budgeted_targets(
        tree,
        labels[::-1].copy(),
        train_idx,
        folds,
        n_classes=2,
        target_num_balls=4,
        guidance_mode="geometry",
    )

    assert (
        result_a.diagnostics["fold_assignment_hashes"]
        == result_b.diagnostics["fold_assignment_hashes"]
    )


def test_crossfit_budgeted_targets_are_finite_normalized_and_exact_budget():
    tree, labels, train_idx = _tree_fixture(max_leaves=8)
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=labels.shape[0],
    )
    result = build_crossfit_budgeted_targets(
        tree,
        labels,
        train_idx,
        folds,
        n_classes=2,
        target_num_balls=4,
        guidance_mode="shuffled",
        random_state=29,
    )

    assert np.all(result.fold_num_balls == 4)
    assert result.diagnostics["exact_budget"] is True
    assert np.isfinite(result.soft_targets[train_idx]).all()
    assert np.all(result.soft_targets[train_idx] >= 0.0)
    np.testing.assert_allclose(result.soft_targets[train_idx].sum(axis=1), 1.0)
