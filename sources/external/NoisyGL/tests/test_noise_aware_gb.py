from __future__ import annotations

import numpy as np

from utils.noise_aware_gb import (
    build_crossfit_targets,
    build_geometry_candidate_tree,
    make_stratified_crossfit_folds,
    prune_candidate_tree,
    shuffle_guidance_within_construction,
)


def _two_cluster_fixture():
    rng = np.random.default_rng(7)
    left = rng.normal(loc=(-2.0, 0.0), scale=0.12, size=(20, 2))
    right = rng.normal(loc=(2.0, 0.0), scale=0.12, size=(20, 2))
    embeddings = np.vstack([left, right]).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    train_idx = np.arange(40, dtype=np.int64)
    return embeddings, labels, train_idx


def test_geometry_candidate_tree_is_label_agnostic():
    embeddings, labels, _ = _two_cluster_fixture()
    tree_a = build_geometry_candidate_tree(
        embeddings,
        max_leaves=8,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=3,
    )
    tree_b = build_geometry_candidate_tree(
        embeddings,
        max_leaves=8,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=3,
    )

    labels[:] = labels[::-1]
    assert tree_a.assignment_hash == tree_b.assignment_hash
    assert tree_a.leaf_ids == tree_b.leaf_ids


def test_bayes_pruning_accepts_label_separation_and_rejects_mixed_children():
    embeddings, labels, train_idx = _two_cluster_fixture()
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=2,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=0,
    )

    separated = prune_candidate_tree(
        tree,
        guidance_indices=train_idx,
        guidance_labels=labels,
        n_classes=2,
        min_guidance_per_child=5,
        log_bf_threshold=np.log(3.0),
    )
    mixed_labels = np.tile(np.array([0, 1], dtype=np.int64), 20)
    mixed = prune_candidate_tree(
        tree,
        guidance_indices=train_idx,
        guidance_labels=mixed_labels,
        n_classes=2,
        min_guidance_per_child=5,
        log_bf_threshold=np.log(3.0),
    )

    assert separated.num_balls == 2
    assert mixed.num_balls == 1


def test_crossfit_excludes_heldout_label_from_partition_and_local_votes():
    embeddings, labels, train_idx = _two_cluster_fixture()
    labels = labels.copy()
    labels[[1, 21]] = labels[[21, 1]]
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=8,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=2,
    )
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=embeddings.shape[0],
    )
    result_a = build_crossfit_targets(
        tree=tree,
        noisy_labels=labels,
        train_indices=train_idx,
        fold_ids=folds,
        n_classes=2,
        guidance_mode="noisy",
    )

    eligible = train_idx[(result_a.alpha[train_idx] > 0.0) & (result_a.alpha[train_idx] < 1.0)]
    assert eligible.size > 0
    node = int(eligible[0])
    mutated = labels.copy()
    mutated[node] = 1 - mutated[node]
    result_b = build_crossfit_targets(
        tree=tree,
        noisy_labels=mutated,
        train_indices=train_idx,
        fold_ids=folds,
        n_classes=2,
        guidance_mode="noisy",
    )

    assert result_a.ball_ids[node] == result_b.ball_ids[node]
    np.testing.assert_allclose(
        result_a.local_distributions[node],
        result_b.local_distributions[node],
    )
    assert result_a.alpha[node] == result_b.alpha[node]
    assert not np.allclose(result_a.soft_targets[node], result_b.soft_targets[node])


def test_labels_outside_train_cannot_change_crossfit_targets():
    embeddings, labels, train_idx = _two_cluster_fixture()
    extra_embeddings = np.array([[0.0, 2.0], [0.0, -2.0]], dtype=np.float32)
    embeddings = np.vstack([embeddings, extra_embeddings])
    all_labels = np.concatenate([labels, np.array([0, 1], dtype=np.int64)])
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=8,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=2,
    )
    folds = make_stratified_crossfit_folds(
        train_idx,
        all_labels,
        n_splits=5,
        random_state=11,
        n_nodes=embeddings.shape[0],
    )
    result_a = build_crossfit_targets(
        tree=tree,
        noisy_labels=all_labels,
        train_indices=train_idx,
        fold_ids=folds,
        n_classes=2,
    )
    changed = all_labels.copy()
    changed[-2:] = changed[-2:][::-1]
    result_b = build_crossfit_targets(
        tree=tree,
        noisy_labels=changed,
        train_indices=train_idx,
        fold_ids=folds,
        n_classes=2,
    )

    np.testing.assert_allclose(
        result_a.soft_targets[train_idx],
        result_b.soft_targets[train_idx],
    )
    np.testing.assert_array_equal(
        result_a.ball_ids[train_idx],
        result_b.ball_ids[train_idx],
    )


def test_crossfit_targets_are_complete_finite_and_normalized():
    embeddings, labels, train_idx = _two_cluster_fixture()
    tree = build_geometry_candidate_tree(
        embeddings,
        max_leaves=8,
        min_ball_size=5,
        min_geometry_gain=0.01,
        random_state=2,
    )
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=embeddings.shape[0],
    )
    result = build_crossfit_targets(
        tree=tree,
        noisy_labels=labels,
        train_indices=train_idx,
        fold_ids=folds,
        n_classes=2,
    )

    assert np.all(result.ball_ids[train_idx] >= 0)
    assert np.isfinite(result.soft_targets[train_idx]).all()
    np.testing.assert_allclose(result.soft_targets[train_idx].sum(axis=1), 1.0)


def test_guidance_shuffle_preserves_each_construction_histogram():
    _, labels, train_idx = _two_cluster_fixture()
    folds = make_stratified_crossfit_folds(
        train_idx,
        labels,
        n_splits=5,
        random_state=11,
        n_nodes=labels.shape[0],
    )
    for heldout_fold in range(5):
        construction = train_idx[folds[train_idx] != heldout_fold]
        shuffled = shuffle_guidance_within_construction(
            construction,
            labels,
            random_state=100 + heldout_fold,
        )
        np.testing.assert_array_equal(
            np.bincount(shuffled, minlength=2),
            np.bincount(labels[construction], minlength=2),
        )
