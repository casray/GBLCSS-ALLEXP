import numpy as np
import pytest

import screen as s


def test_identical_voter_sets_deduplicated_and_uniform_fallback_abstains():
    mask = np.array([[False, True, True], [True, False, True], [True, True, False]])
    observed = np.array([0, 1, 1])
    one = s.summarize_votes([mask], observed, 2)
    duplicate = s.summarize_votes([mask, mask, mask], observed, 2)
    for key in one:
        np.testing.assert_allclose(one[key], duplicate[key])
    empty = s.summarize_votes([np.zeros_like(mask)], observed, 2)
    assert not empty['candidate'].any()
    assert (empty['distinct_scopes'] == 0).all()


def test_leave_one_out_at_all_scales():
    train = np.arange(4)
    masks = [s.scope_members(a, train) for a in (np.zeros(4), np.array([0,0,0,1]))]
    observed = np.array([0, 1, 1, 0])
    before = s.summarize_votes(masks, observed, 2)
    observed[0] = 1
    after = s.summarize_votes(masks, observed, 2)
    np.testing.assert_array_equal(before['probability'][0], after['probability'][0])
    assert before['Full'][0] == after['Full'][0]


def test_unique_support_not_sum_of_nested_supports():
    coarse = s.scope_members(np.zeros(5), np.arange(5))
    fine = s.scope_members(np.array([0,0,0,1,1]), np.arange(5))
    out = s.summarize_votes([coarse, fine], np.array([0,1,1,0,0]), 2)
    assert out['unique_support'][0] == 4
    assert out['distinct_scopes'][0] == 2
    assert 0 <= out['stability'][0] <= 1


def test_fixed_candidate_predictions_for_rankings_and_tie_selection():
    candidate = np.array([True, True, True, True, False])
    weights, enough = s.selection(np.ones(5), candidate, fraction=.25)
    assert enough
    np.testing.assert_array_equal(weights, [.25,.25,.25,.25,0])
    assert weights.sum() == 1
    labels = np.array([1,1,0,0,0])
    observed = np.zeros(5, dtype=int)
    prediction = np.ones(5, dtype=int)
    result = s.correction_metrics(labels, observed, prediction, weights)
    assert result['precision'] == .5
    assert result['recall'] == .25
    assert result['clean_corruption'] == pytest.approx(1/6)


def test_knn_support_matched_and_self_excluded():
    order = np.array([[1,2], [0,2], [1,0]])
    mask = s.matched_knn_members(order, np.array([0,1,2]))
    np.testing.assert_array_equal(mask.sum(1), [0,1,2])
    assert not np.diag(mask).any()


def test_two_different_scopes_same_wrong_class_can_be_stable():
    a = np.array([[False, True, True, False], [True,False,True,False],
                  [True,True,False,False], [True,True,False,False]])
    b = np.array([[False,True,True,True], [True,False,True,True],
                  [True,True,False,True], [True,True,True,False]])
    out = s.summarize_votes([a,b], np.array([1,1,1,1]), 2)
    assert out['stability'][0] == pytest.approx(1)
    assert out['prediction'][0] == 1
    # Stability makes no claim about the unobserved true class.


def test_insufficient_candidates_are_not_silently_reduced():
    weights, sufficient = s.selection(np.arange(10), np.array([True]+[False]*9), total_fraction=.2)
    assert not sufficient
    assert weights.sum() == 0


def test_split_tree_cuts_are_nested_and_children_cover_parent(monkeypatch):
    monkeypatch.setattr(s, 'BUDGETS', [2, 4])
    monkeypatch.setitem(s.SETTINGS, 'min_child_size', 4)
    rng = np.random.default_rng(7)
    points = np.vstack([np.array([np.cos(a), np.sin(a)]) + rng.normal(0,.001,(8,2))
                        for a in (0., .5, 1., 1.5)])
    cuts, tree = s.gb_tree(s.normalize(points))
    assert len(np.unique(cuts[2])) == 2
    assert len(np.unique(cuts[4])) == 4
    assert s.nested(cuts[2], cuts[4])
    for parent in tree:
        if parent['children']:
            first, second = [tree[c]['members'] for c in parent['children']]
            assert len(np.intersect1d(first,second)) == 0
            np.testing.assert_array_equal(np.sort(np.concatenate([first,second])), np.sort(parent['members']))
