"""Exact small-label audits. No model training or original result modification."""
import itertools
import json
from pathlib import Path
import numpy as np


def impurity(y, groups, classes, loo):
    cost = 0.0
    for group in groups:
        n = len(group)
        if not n:
            continue
        counts = np.bincount(y[group], minlength=classes)
        gini = 1 - np.square(counts / n).sum()
        value = ((1 - 1 / classes) if n == 1 else n / (n - 1) * gini) if loo else gini
        cost += n * value
    return float(cost)


def expectation(y, groups, transition, loo=True):
    classes, n = len(transition), len(y)
    result = second = 0.0
    for outcome in itertools.product(range(classes), repeat=n):
        outcome = np.asarray(outcome)
        prob = np.prod(transition[y, outcome])
        score = impurity(outcome, groups, classes, loo)
        result += prob * score
        second += prob * score ** 2
    return result, max(0.0, second - result ** 2)


def uniform(classes, eta):
    t = np.full((classes, classes), eta / (classes - 1))
    np.fill_diagonal(t, 1 - eta)
    return t


def cycle(classes, eta):
    return (1 - eta) * np.eye(classes) + eta * np.roll(np.eye(classes), 1, axis=1)


def main():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    groups = {
        'coarse': [[0, 1, 2, 3], [4, 5, 6, 7]],
        'good': [[0, 1, 2], [3, 4, 5], [6, 7]],
        'singletons': [[i] for i in range(8)],
        'mixed_pairs': [[0, 3], [1, 4], [2, 6], [5, 7]],
    }
    table = {}
    for name, partition in groups.items():
        clean_loo = impurity(y, partition, 3, True)
        observed, _ = expectation(y, partition, uniform(3, .4))
        formula = .4 ** 2 * clean_loo + (1 - .4 ** 2) * (2 / 3) * len(y)
        assert abs(observed - formula) < 1e-10
        table[name] = {'clean_gini': impurity(y, partition, 3, False),
                       'clean_loo_with_singleton_prior': clean_loo,
                       'uniform_gini': expectation(y, partition, uniform(3, .4), False)[0],
                       'uniform_loo': observed,
                       'pair3_cycle_loo': expectation(y, partition, cycle(3, .4))[0],
                       'formula_error': abs(observed - formula)}

    # A fixed candidate tree with root and two children; even at zero signal,
    # choosing the lower observed score is not choosing a constant.
    t = uniform(3, 2 / 3)
    root, split = [list(range(4))], [[0, 1], [2, 3]]
    mean_root = mean_split = mean_min = probability_split_wins = 0.0
    for z in itertools.product(range(3), repeat=4):
        z = np.asarray(z)
        probability = 1 / 3 ** 4
        a, b = impurity(z, root, 3, True), impurity(z, split, 3, True)
        mean_root += probability * a
        mean_split += probability * b
        mean_min += probability * min(a, b)
        probability_split_wins += probability * (b < a - 1e-12)
    assert abs(mean_root - mean_split) < 1e-12
    assert mean_min < mean_root
    assert probability_split_wins > 0
    pair_mean, pair_variance = expectation(np.array([0, 0]), [[0, 1]], t)
    assert abs(pair_variance - 8 / 9) < 1e-12  # variance of n*U, n=2

    # General cyclic pair noise lacks a single same/different contraction.
    t4 = cycle(4, .4)
    pair_expectation = 1 - t4 @ t4.T
    # Two partitions of y=[A,A,B,C]: adjacent-cross pairs vs pure+opposite.
    y4 = np.array([0, 0, 1, 2])
    p1, p2 = [[0, 2], [1, 3]], [[0, 1], [2, 3]]
    # This particular pair is recorded only as kernel variation, not claimed reversal.
    kernel_off_diagonal = sorted(set(np.round(pair_expectation[~np.eye(4, dtype=bool)], 10)))
    assert len(kernel_off_diagonal) > 1
    t3 = cycle(3, .4)
    assert np.allclose(t3 @ t3.T, .28 * np.eye(3) + .24 * np.ones((3, 3)))

    result = {
        'status': 'passed',
        'scope': 'finite enumeration and algebra; no model experiment',
        'reconstruction_note': 'Explicit constructed labels and partitions reproduce rounded user table; original unpublished script not available.',
        'labels': y.tolist(), 'partitions': groups, 'table': table,
        'zero_signal': {'fixed_root_expected_cost': mean_root,
                        'fixed_split_expected_cost': mean_split,
                        'adaptively_selected_expected_cost': mean_min,
                        'probability_split_has_lower_observed_cost': probability_split_wins,
                        'two_label_ball_cost_mean': pair_mean,
                        'two_label_ball_cost_variance': pair_variance,
                        'two_label_ball_impurity_variance': pair_variance / 4},
        'singleton_counterexample': {'C': 3, 'two_different_labels_cost': 2,
                                    'split_into_singletons_cost': 4 / 3},
        'pair4_off_diagonal_disagreement_values': kernel_off_diagonal,
        'pair3_gram_matrix': (t3 @ t3.T).tolist(),
    }
    output = Path(__file__).with_name('loo_impurity_checks.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
