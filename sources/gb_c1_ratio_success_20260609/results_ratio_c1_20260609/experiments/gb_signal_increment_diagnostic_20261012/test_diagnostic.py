import numpy as np
import pytest
import torch

import run_diagnostic as d


def test_votes_exclude_self_and_handle_singleton():
    assignment = np.array([0, 0, 0, 1])
    train = np.array([0, 1, 3])
    noisy = np.array([0, 1, 1, 0])
    votes, support = d.leave_one_out_votes(assignment, train, noisy, 2)
    np.testing.assert_allclose(votes, [[0, 1], [1, 0], [.5, .5]])
    np.testing.assert_array_equal(support, [1, 1, 0])
    changed = noisy.copy()
    changed[0] = 1
    alternate, _ = d.leave_one_out_votes(assignment, train, changed, 2)
    np.testing.assert_array_equal(alternate[0], votes[0])


def test_region_features_have_equal_definition_and_ignore_nontrain_labels():
    h = d.normalize(np.array([[1., 0], [.9, .1], [0, 1], [.1, .9]]))
    assignment, train = np.array([0, 0, 1, 1]), np.array([0, 2])
    teacher = np.array([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    noisy = np.array([0, 1, 1, 0])
    first = d.region_features(h, assignment, train, noisy, teacher)
    noisy[[1, 3]] = 0
    second = d.region_features(h, assignment, train, noisy, teacher)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, len(d.REGION_NAMES))
    assert np.isfinite(first).all()


def test_dataset_fold_groups_all_seeds_and_noise_rates():
    records = [dict(dataset=name, seed=seed, rate=rate)
               for name in ('cora', 'citeseer') for seed in (1, 2) for rate in (.4, .6)]
    train, evaluation = d.fold_indices(records, 'cora')
    assert len(train) == len(evaluation) == 4
    assert all(records[j]['dataset'] == 'citeseer' for j in train)
    assert all(records[j]['dataset'] == 'cora' for j in evaluation)


def test_coverage_ties_do_not_use_label_order():
    labels = np.array([1, 1, 0, 0, 0])
    scores = np.ones(5)
    values = d.coverage_metrics(labels, scores)
    assert values['precision20'] == pytest.approx(.4)
    assert values['recall20'] == pytest.approx(.2)
    assert values['clean_flag_rate20'] == pytest.approx(.2)
    assert d.coverage_metrics(labels[::-1], scores) == values


def test_fit_preprocessing_does_not_depend_on_evaluation_distribution():
    x = np.array([[-2], [-1], [1], [2]], dtype=float)
    y = np.array([0, 0, 1, 1])
    _, first, scaler = d.fit_predict(x, y, np.ones(4), np.array([[0.]]))
    _, second, other = d.fit_predict(x, y, np.ones(4), np.array([[1000.]]))
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(scaler.mean_, other.mean_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU retrieval test')
def test_exact_gpu_neighbors_exclude_target_and_nontrain_knn():
    h = d.normalize(np.array([[1., 0], [.9, .1], [0., 1], [-1., 0]]))
    train = np.array([0, 1, 2])
    edge = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    indices, cosine, peers = d.exact_neighbors(h, train, edge, k=2, chunk=2)
    assert indices.shape == peers.shape == (3, 2)
    assert np.all(np.isin(indices, train))
    assert not np.any(indices == train[:, None])
    assert not np.any(peers == train[:, None])
    assert np.isfinite(cosine).all()
