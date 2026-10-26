import itertools
import unittest

import numpy as np

from kernels import choose_support, loo_counts, majority_probabilities, agreement


class AgreementTests(unittest.TestCase):
    def test_query_label_never_changes_its_own_votes(self):
        labels = np.array([0, 1, 2, 2, 1, 0])
        groups = np.array([0, 0, 0, 1, 1, 1])
        before = loo_counts(labels, groups, 3)
        for node in range(len(labels)):
            altered = labels.copy()
            altered[node] = (labels[node] + 1) % 3
            np.testing.assert_array_equal(
                before[node], loo_counts(altered, groups, 3)[node])

    def test_ties_do_not_favor_observed_label(self):
        probabilities = majority_probabilities(np.array([[2, 2, 0], [1, 1, 1]]))
        np.testing.assert_allclose(probabilities, [[.5, .5, 0], [1/3, 1/3, 1/3]])
        self.assertAlmostEqual(agreement(probabilities, np.array([0, 2])), 5/12)

    def test_symmetric_noise_identity_by_exhaustive_enumeration(self):
        clean = np.array([0, 0, 1])
        classes = 3
        for eta in [0, .4, .6]:
            observed_score = clean_score = total = 0.0
            for observed in itertools.product(range(classes), repeat=3):
                z = np.array(observed)
                probability = np.prod(np.where(z == clean, 1-eta, eta/(classes-1)))
                vote = majority_probabilities(loo_counts(z, np.zeros(3, int), classes))
                observed_score += probability * agreement(vote, z)
                clean_score += probability * agreement(vote, clean)
                total += probability
            self.assertAlmostEqual(total, 1)
            self.assertAlmostEqual(observed_score,
                                   eta/(classes-1)+(1-classes*eta/(classes-1))*clean_score)

    def test_class_renaming_preserves_score(self):
        z = np.array([0, 1, 1, 2, 0, 2])
        groups = np.repeat([0, 1], 3)
        renamed = np.array([2, 0, 1])[z]
        first = agreement(majority_probabilities(loo_counts(z, groups, 3)), z)
        second = agreement(majority_probabilities(loo_counts(renamed, groups, 3)), renamed)
        self.assertEqual(first, second)

    def test_candidate_ties_and_invalid_scores(self):
        self.assertEqual(choose_support({4: .5, 32: .5, 128: .4}), 32)
        self.assertEqual(choose_support({4: .6, 128: .5}), 4)
        with self.assertRaises(ValueError):
            choose_support({4: np.nan})
        with self.assertRaises(ValueError):
            majority_probabilities(np.array([[0, 0]]))


if __name__ == '__main__':
    unittest.main()
