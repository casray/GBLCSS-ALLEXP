"""CPU mechanism contracts; execute directly with Python and unittest."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

try:
    from mechanisms import build_targets, random_size_matched_assignment
except ModuleNotFoundError as exc:
    if exc.name != "mechanisms":
        raise
    build_targets = random_size_matched_assignment = None


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(build_targets, "build_targets API is not implemented")
        self.assertIsNotNone(random_size_matched_assignment,
                             "random_size_matched_assignment API is not implemented")
        self.labels = np.array([0, 1, 1, 2, 0, 2, 1, 2])
        self.assignment = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        self.train = np.array([0, 1, 2, 3, 5])

    def test_a_is_noisy_onehot_in_train_index_order(self):
        train = self.train[::-1]
        target, _, _, _ = build_targets(self.labels, self.assignment, train, 3, "A")
        np.testing.assert_array_equal(target, np.eye(3)[self.labels[train]])

    def test_b_inclusive_training_histogram_and_empty_ball(self):
        target, q, counts, _ = build_targets(self.labels, self.assignment, self.train, 3, "B")
        expected = np.array([[1/3, 2/3, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_allclose(q, expected, atol=1e-7)
        np.testing.assert_array_equal(counts, [3, 1, 1, 0])
        np.testing.assert_array_equal(target, q[self.assignment[self.train]])

    def test_bprime_removes_own_vote_and_singleton_falls_back(self):
        target, _, _, _ = build_targets(self.labels, self.assignment, self.train, 3, "Bprime")
        expected = [[0, 1, 0], [.5, .5, 0], [.5, .5, 0], [0, 0, 1], [0, 0, 1]]
        np.testing.assert_array_equal(target, expected)

    def test_bprime_uses_one_remaining_vote_while_lcss_requires_two(self):
        labels, assignment, train = np.array([0, 1]), np.array([0, 0]), np.arange(2)
        bp = build_targets(labels, assignment, train, 2, "Bprime")[0]
        lcss = build_targets(labels, assignment, train, 2, "LCSS")[0]
        np.testing.assert_array_equal(bp, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(lcss, [[1, 0], [0, 1]])

    def test_lcss_tie_uses_distribution_only(self):
        labels = np.array([2, 0, 0, 1, 1])
        target, _, _, diagnostics = build_targets(labels, np.zeros(5, dtype=int), np.arange(5), 3, "LCSS")
        # Node 0: LOO=(.5,.5,0), alpha=(.5-1/3)/(1-1/3)=.25.
        np.testing.assert_allclose(target[0], [.125, .125, .75], atol=1e-7)
        self.assertEqual(diagnostics["tied_train_nodes"], 1)

    def test_lcss_unique_majority_uses_confidence_and_beta(self):
        labels = np.array([2, 0, 0, 0, 1])
        target = build_targets(labels, np.zeros(5, dtype=int), np.arange(5), 3, "LCSS")[0]
        # LOO=(.75,.25,0), alpha=.625, local=(.875,.125,0).
        np.testing.assert_allclose(target[0], [.546875, .078125, .375], atol=1e-7)

    def test_lcss_matches_original_function_synthetic_and_random(self):
        import torch
        source = Path(__file__).resolve().parents[3] / "GB-LCSS_GitHub/src/gblcss/targets.py"
        spec = importlib.util.spec_from_file_location("original_gblcss_targets", source)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        rng = np.random.default_rng(512)
        cases = [(self.labels, self.assignment, self.train, 3)]
        for classes in (2, 3, 7):
            for _ in range(12):
                n, groups = 71, 9
                assignment = rng.permutation(np.arange(n) % groups)
                cases.append((rng.integers(classes, size=n), assignment,
                              rng.choice(n, 43, replace=False), classes))
        for labels, assignment, train, classes in cases:
            for beta in (0., .5, 1.):
                actual = build_targets(labels, assignment, train, classes, "LCSS", target_beta=beta)
                reference = module.build_leave_one_out_targets(
                    torch.tensor(labels, device="cpu"), assignment, train, classes, beta=beta)
                np.testing.assert_allclose(actual[0], reference.soft_target.numpy()[train], atol=1e-7)
                for key in ("eligible_train_nodes", "tied_train_nodes", "num_balls"):
                    self.assertEqual(actual[3][key], reference.diagnostics[key])

    def test_heldout_labels_do_not_affect_any_return(self):
        heldout = np.setdiff1d(np.arange(len(self.labels)), self.train)
        poisoned = self.labels.astype(float)
        poisoned[heldout] = np.nan
        for mode in ("A", "B", "Bprime", "LCSS"):
            original = build_targets(self.labels, self.assignment, self.train, 3, mode)
            changed = build_targets(poisoned, self.assignment, self.train, 3, mode)
            for a, b in zip(original[:3], changed[:3]):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(original[3], changed[3])

    def test_probabilities_dtypes_diagnostics_and_no_input_mutation(self):
        snapshots = [value.copy() for value in (self.labels, self.assignment, self.train)]
        for mode in ("A", "B", "Bprime", "LCSS"):
            target, q, count, info = build_targets(self.labels, self.assignment, self.train, 3, mode)
            self.assertEqual(target.shape, (5, 3))
            self.assertEqual(q.shape, (4, 3))
            for value in (target, q, count):
                self.assertEqual(value.dtype, np.float32)
                self.assertTrue(np.isfinite(value).all())
                self.assertTrue((value >= 0).all())
            np.testing.assert_allclose(target.sum(1), 1., atol=1e-6)
            np.testing.assert_allclose(q[count > 0].sum(1), 1., atol=1e-6)
            self.assertTrue(np.isfinite(info["target_entropy_mean"]))
            self.assertEqual(info["train_support_total"], 5)
            self.assertEqual(info["empty_train_balls"], 1)
        for original, current in zip(snapshots, (self.labels, self.assignment, self.train)):
            np.testing.assert_array_equal(original, current)

    def test_rejects_invalid_support_assignment_and_labels(self):
        bad_inputs = [
            (self.labels, self.assignment, np.array([], dtype=int), 3),
            (self.labels, self.assignment, np.array([0, 0]), 3),
            (self.labels, self.assignment, np.array([8]), 3),
            (self.labels, self.assignment, np.array([-1]), 3),
            (self.labels, self.assignment, np.array([0.5]), 3),
            (self.labels, self.assignment + 1, self.train, 3),
            (self.labels, np.array([0, 0, 0, 2, 2, 3, 3, 4]), self.train, 3),
            (self.labels, self.assignment[:-1], self.train, 3),
            (self.labels, self.assignment, self.train, 1),
            (self.labels + 3, self.assignment, self.train, 3),
        ]
        for args in bad_inputs:
            with self.subTest(args=args), self.assertRaises(ValueError):
                build_targets(*args, "B")
        with self.assertRaises(ValueError):
            build_targets(self.labels, self.assignment, self.train, 3, "unknown")
        with self.assertRaises(ValueError):
            build_targets(self.labels, self.assignment, self.train, 3, "LCSS", target_beta=1.2)

    def test_size_matched_assignment_exact_sizes_and_seed(self):
        assignment = np.repeat(np.arange(5), [1, 3, 7, 11, 22])
        random = random_size_matched_assignment(assignment, 123)
        np.testing.assert_array_equal(np.bincount(random), np.bincount(assignment))
        np.testing.assert_array_equal(np.unique(random), np.arange(5))
        np.testing.assert_array_equal(random, random_size_matched_assignment(assignment, 123))
        self.assertFalse(np.array_equal(random, assignment))
        self.assertFalse(np.array_equal(random, random_size_matched_assignment(assignment, 124)))
        np.testing.assert_array_equal(assignment, np.repeat(np.arange(5), [1, 3, 7, 11, 22]))

    def test_size_matched_assignment_validates_ids(self):
        for assignment in (np.array([]), np.array([1, 1]), np.array([0, 2]), np.array([-1, 0])):
            with self.assertRaises(ValueError):
                random_size_matched_assignment(assignment, 5)

    def test_weighted_coarse_ce_equals_sum_for_shared_predictions(self):
        _, q, count, _ = build_targets(self.labels, self.assignment, self.train, 3, "B")
        predictions = np.array([[.2, .5, .3], [.1, .2, .7], [.4, .1, .5], [.6, .2, .2]])
        fine_loss = -np.log(predictions[self.assignment[self.train], self.labels[self.train]]).sum()
        coarse_loss = -(count[:, None] * q * np.log(predictions)).sum()
        self.assertAlmostEqual(fine_loss, coarse_loss, places=6)
        # Sum equality implies mean equality only after dividing BOTH by L.
        self.assertAlmostEqual(fine_loss / len(self.train), coarse_loss / count.sum(), places=6)

    def test_unequal_node_predictions_break_coarse_ce_identity(self):
        labels, assignment, train = np.array([0, 1]), np.array([0, 0]), np.arange(2)
        _, q, count, _ = build_targets(labels, assignment, train, 2, "B")
        node_predictions = np.array([[.9, .1], [.1, .9]])
        fine_loss = -np.log(node_predictions[train, labels]).sum()
        coarse_loss = -(count[:, None] * q * np.log(node_predictions.mean(0))).sum()
        self.assertGreater(abs(fine_loss - coarse_loss), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
