import hashlib
import inspect
import unittest

import numpy as np
import torch

from utils.frozen_grace_targets import (
    apply_ball_majority_correction,
    apply_ball_ellipsoid_soft_targets,
    apply_ball_soft_targets,
    ball_consistency_loss,
    build_ball_expansion_soft_targets,
    build_label_propagation_soft_targets,
    build_soft_targets_from_features,
    diagnostics_for_early_arbitration,
    diagnose_ball_expansion_candidates,
    diagnostics_for_correction,
    expansion_soft_target_loss,
    soft_target_cross_entropy,
)


class FrozenGraceTargetsTest(unittest.TestCase):
    def test_uniform_soft_target_apis_expose_tie_mode(self):
        self.assertIn("tie_mode", inspect.signature(apply_ball_soft_targets).parameters)
        self.assertIn("tie_mode", inspect.signature(build_soft_targets_from_features).parameters)

    def test_majority_correction_uses_train_nodes_only(self):
        noisy = torch.tensor([0, 1, 1, 2, 2, 0])
        clean = torch.tensor([0, 1, 1, 2, 0, 0])
        train_mask = np.array([1, 2, 4])
        assignment = np.array([0, 0, 0, 1, 1, 1])

        corrected, changed = apply_ball_majority_correction(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertEqual(corrected.tolist(), [0, 1, 1, 2, 2, 0])
        self.assertEqual(changed.tolist(), [False, False, False, False, False, False])
        diag = diagnostics_for_correction(
            noisy_label=noisy,
            clean_label=clean,
            corrected_label=corrected,
            changed=changed,
            train_mask=train_mask,
        )
        self.assertEqual(diag["changed_train_count"], 0)
        self.assertAlmostEqual(diag["ncr"], 0.0)
        self.assertAlmostEqual(diag["cdr"], 0.0)

    def test_majority_correction_changes_only_eligible_train_nodes(self):
        noisy = torch.tensor([0, 1, 0, 2, 2, 1])
        clean = torch.tensor([1, 1, 1, 2, 2, 1])
        train_mask = np.array([0, 1, 2, 3, 4])
        assignment = np.array([0, 0, 0, 1, 1, 1])

        corrected, changed = apply_ball_majority_correction(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertEqual(corrected.tolist(), [0, 0, 0, 2, 2, 1])
        self.assertEqual(changed.tolist(), [False, True, False, False, False, False])
        diag = diagnostics_for_correction(
            noisy_label=noisy,
            clean_label=clean,
            corrected_label=corrected,
            changed=changed,
            train_mask=train_mask,
        )
        self.assertEqual(diag["changed_train_count"], 1)
        self.assertAlmostEqual(diag["ncr"], 0.0)
        self.assertAlmostEqual(diag["cdr"], 1.0 / 3.0)

    def test_low_coverage_ball_is_not_corrected(self):
        noisy = torch.tensor([0, 1, 1, 2])
        train_mask = np.array([0, 2])
        assignment = np.array([0, 0, 1, 1])

        corrected, changed = apply_ball_majority_correction(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
        )

        self.assertEqual(corrected.tolist(), noisy.tolist())
        self.assertFalse(bool(changed.any()))

    def test_low_confidence_majority_is_not_corrected(self):
        noisy = torch.tensor([0, 1, 1, 2, 2, 2, 0, 2])
        train_mask = np.array([0, 1, 2, 3, 4, 5, 6])
        assignment = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        corrected, changed = apply_ball_majority_correction(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=3,
            min_majority_confidence=0.7,
        )

        self.assertEqual(corrected.tolist(), [0, 1, 1, 2, 2, 2, 2, 2])
        self.assertEqual(changed.tolist(), [False, False, False, False, False, False, True, False])

    def test_soft_targets_interpolate_by_majority_margin(self):
        noisy = torch.tensor([0, 1, 1, 2, 2, 2, 0, 2])
        train_mask = np.array([0, 1, 2, 3, 4, 5, 6])
        assignment = np.array([0, 0, 0, 1, 1, 1, 1, 1])

        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=3,
        )

        self.assertEqual(pseudo_label.tolist(), [1, 1, 1, 2, 2, 2, 2, 2])
        self.assertAlmostEqual(float(alpha[0]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(alpha[1]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(alpha[3]), 0.5, places=5)
        self.assertAlmostEqual(float(alpha[6]), 0.5, places=5)
        self.assertTrue(torch.allclose(soft_target[0], torch.tensor([2.0 / 3.0, 1.0 / 3.0, 0.0])))
        self.assertTrue(torch.allclose(soft_target[1], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.allclose(soft_target[6], torch.tensor([0.5, 0.0, 0.5])))
        self.assertTrue(torch.allclose(soft_target[7], torch.tensor([0.0, 0.0, 1.0])))

    def test_soft_targets_can_use_multiclass_alpha_and_ball_distribution(self):
        noisy = torch.tensor([0, 1, 1, 2, 3])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0, 0])

        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=4,
            min_train_per_ball=3,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        self.assertEqual(pseudo_label.tolist(), [1, 1, 1, 1, 3])
        self.assertAlmostEqual(float(alpha[0]), 1.0 / 3.0, places=5)
        self.assertAlmostEqual(float(alpha[3]), 1.0 / 3.0, places=5)
        self.assertTrue(torch.allclose(soft_target[0], torch.tensor([0.75, 1.0 / 6.0, 1.0 / 12.0, 0.0])))
        self.assertTrue(torch.allclose(soft_target[1], torch.tensor([1.0 / 12.0, 5.0 / 6.0, 1.0 / 12.0, 0.0])))
        self.assertTrue(torch.allclose(soft_target[3], torch.tensor([1.0 / 12.0, 1.0 / 6.0, 0.75, 0.0])))
        self.assertTrue(torch.allclose(soft_target[4], torch.tensor([0.0, 0.0, 0.0, 1.0])))

    def test_soft_target_beta_interpolates_distribution_and_majority(self):
        noisy = torch.tensor([0, 1, 1, 2, 3])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0, 0])

        soft, alpha, pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=4,
            min_train_per_ball=3,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.0,
        )
        half, half_alpha, half_pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=4,
            min_train_per_ball=3,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
        )
        hard_like, hard_alpha, hard_pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=4,
            min_train_per_ball=3,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=1.0,
        )

        ball_dist = torch.tensor([0.25, 0.5, 0.25, 0.0])
        majority = torch.tensor([0.0, 1.0, 0.0, 0.0])
        original = torch.eye(4)[noisy.long()]
        expected_half = (1.0 - alpha.unsqueeze(1)) * original + alpha.unsqueeze(1) * (
            0.5 * ball_dist + 0.5 * majority
        )
        expected_hard = (1.0 - alpha.unsqueeze(1)) * original + alpha.unsqueeze(1) * majority

        self.assertTrue(torch.allclose(half, expected_half, atol=1e-6))
        self.assertTrue(torch.allclose(hard_like, expected_hard, atol=1e-6))
        self.assertTrue(torch.allclose(soft[4], original[4]))
        self.assertTrue(torch.allclose(half[4], original[4]))
        self.assertTrue(torch.allclose(hard_like[4], original[4]))
        self.assertTrue(torch.allclose(half_alpha, alpha))
        self.assertTrue(torch.allclose(hard_alpha, alpha))
        self.assertEqual(half_pseudo.tolist(), pseudo.tolist())
        self.assertEqual(hard_pseudo.tolist(), pseudo.tolist())

    def test_soft_target_beta_rejects_out_of_range_values(self):
        noisy = torch.tensor([0, 1, 1])
        train_mask = np.array([0, 1, 2])
        assignment = np.array([0, 0, 0])

        with self.assertRaises(ValueError):
            apply_ball_soft_targets(
                noisy_label=noisy,
                assignment=assignment,
                train_mask=train_mask,
                n_classes=2,
                target_mode="distribution",
                target_beta=1.5,
            )

    def test_tie_mode_defaults_to_min_index(self):
        noisy = torch.tensor([0, 0, 1, 1, 2])
        train_mask = np.arange(5)
        assignment = np.zeros(5, dtype=np.int64)
        kwargs = {
            "noisy_label": noisy,
            "assignment": assignment,
            "train_mask": train_mask,
            "n_classes": 3,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "target_beta": 0.5,
        }

        default = apply_ball_soft_targets(**kwargs)
        explicit = apply_ball_soft_targets(**kwargs, tie_mode="min_index")

        for default_tensor, explicit_tensor in zip(default, explicit):
            self.assertTrue(torch.equal(default_tensor, explicit_tensor))
        self.assertTrue(
            torch.allclose(
                default[0][2],
                torch.tensor([0.07, 0.92, 0.01]),
                atol=1e-6,
            )
        )
        self.assertEqual(default[2].tolist(), [0, 0, 0, 0, 0])

    def test_alpha_zero_abstains_on_active_majority_tie(self):
        noisy = torch.tensor([0, 0, 1, 1, 2])
        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=np.zeros(5, dtype=np.int64),
            train_mask=np.arange(5),
            n_classes=3,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            tie_mode="alpha_zero",
        )

        self.assertTrue(torch.equal(soft_target, torch.eye(3)[noisy]))
        self.assertTrue(torch.equal(alpha, torch.zeros(5)))
        self.assertTrue(torch.equal(pseudo_label, noisy))

    def test_distribution_only_preserves_distribution_and_alpha_on_tie(self):
        noisy = torch.tensor([0, 0, 1, 1, 2])
        soft_target, alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=np.zeros(5, dtype=np.int64),
            train_mask=np.arange(5),
            n_classes=3,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
            tie_mode="distribution_only",
        )

        expected_alpha = torch.full((5,), 0.1)
        ball_distribution = torch.tensor([0.4, 0.4, 0.2])
        original = torch.eye(3)[noisy]
        expected = (
            (1.0 - expected_alpha.unsqueeze(1)) * original
            + expected_alpha.unsqueeze(1) * ball_distribution
        )
        self.assertTrue(torch.allclose(alpha, expected_alpha, atol=1e-6))
        self.assertTrue(torch.allclose(soft_target, expected, atol=1e-6))

    def test_tie_modes_do_not_change_non_tied_ball(self):
        noisy = torch.tensor([0, 0, 0, 1, 2])
        kwargs = {
            "noisy_label": noisy,
            "assignment": np.zeros(5, dtype=np.int64),
            "train_mask": np.arange(5),
            "n_classes": 3,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "target_beta": 0.5,
        }

        baseline = apply_ball_soft_targets(**kwargs, tie_mode="min_index")
        for tie_mode in ("alpha_zero", "distribution_only"):
            candidate = apply_ball_soft_targets(**kwargs, tie_mode=tie_mode)
            for baseline_tensor, candidate_tensor in zip(baseline, candidate):
                self.assertTrue(torch.equal(baseline_tensor, candidate_tensor))

    def test_safe_tie_modes_are_class_permutation_equivariant(self):
        noisy = torch.tensor([0, 0, 1, 1, 2])
        permutation = torch.tensor([2, 0, 1])
        permuted_noisy = permutation[noisy]
        kwargs = {
            "assignment": np.zeros(5, dtype=np.int64),
            "train_mask": np.arange(5),
            "n_classes": 3,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "target_beta": 0.5,
        }

        for tie_mode in ("alpha_zero", "distribution_only"):
            target, alpha, pseudo_label = apply_ball_soft_targets(
                noisy_label=noisy,
                tie_mode=tie_mode,
                **kwargs,
            )
            permuted_target, permuted_alpha, permuted_pseudo_label = apply_ball_soft_targets(
                noisy_label=permuted_noisy,
                tie_mode=tie_mode,
                **kwargs,
            )

            self.assertTrue(torch.allclose(permuted_target[:, permutation], target, atol=1e-6))
            self.assertTrue(torch.equal(permuted_alpha, alpha))
            self.assertTrue(
                torch.equal(permuted_pseudo_label, permutation[pseudo_label])
            )

    def test_tie_mode_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "tie_mode"):
            apply_ball_soft_targets(
                noisy_label=torch.tensor([0, 1]),
                assignment=np.zeros(2, dtype=np.int64),
                train_mask=np.arange(2),
                n_classes=2,
                tie_mode="average_argmax",
            )

    def test_uniform_builder_reports_tie_diagnostics_and_stable_assignment_hash(self):
        noisy = torch.tensor([0, 0, 1, 1, 2])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.5, 0.5],
            ]
        )
        kwargs = {
            "ball_features": features,
            "noisy_label": noisy,
            "clean_label": noisy.clone(),
            "train_mask": np.arange(5),
            "n_classes": 3,
            "max_balls": 1,
            "min_ball_size": 2,
            "min_train_per_ball": 2,
            "alpha_baseline": "multiclass",
            "target_mode": "distribution",
            "target_beta": 0.5,
            "random_state": 7,
            "softdist_weighting": "uniform",
        }

        results = {
            mode: build_soft_targets_from_features(**kwargs, tie_mode=mode)
            for mode in ("min_index", "alpha_zero", "distribution_only")
        }
        assignment = np.ascontiguousarray(
            np.asarray(results["min_index"].assignment, dtype="<i8")
        )
        header = (
            f"dtype={assignment.dtype.str};"
            f"shape={','.join(str(size) for size in assignment.shape)};"
            "order=C\n"
        ).encode("ascii")
        expected_hash = hashlib.sha256(
            header + assignment.tobytes(order="C")
        ).hexdigest()

        hashes = set()
        for mode, result in results.items():
            diagnostics = result.diagnostics
            self.assertEqual(diagnostics["tie_mode"], mode)
            self.assertEqual(diagnostics["tie_ball_count"], 1)
            self.assertEqual(diagnostics["tie_active_ball_count"], 1)
            self.assertEqual(diagnostics["tie_affected_node_count"], 5)
            self.assertAlmostEqual(diagnostics["tie_affected_alpha_sum"], 0.5)
            self.assertEqual(diagnostics["ball_assignment_hash"], expected_hash)
            hashes.add(diagnostics["ball_assignment_hash"])
        self.assertEqual(len(hashes), 1)

    def test_soft_target_construction_does_not_use_clean_labels(self):
        noisy = torch.tensor([0, 1, 1, 2, 2, 0])
        clean_a = torch.tensor([0, 1, 1, 2, 2, 0])
        clean_b = torch.tensor([2, 0, 0, 1, 1, 2])
        train_mask = np.array([0, 1, 2, 3, 4])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
            ]
        )

        result_a = build_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy,
            clean_label=clean_a,
            train_mask=train_mask,
            n_classes=3,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=1.0,
            random_state=0,
        )
        result_b = build_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy,
            clean_label=clean_b,
            train_mask=train_mask,
            n_classes=3,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=1.0,
            random_state=0,
        )

        self.assertTrue(torch.allclose(result_a.soft_target, result_b.soft_target))
        self.assertTrue(torch.equal(result_a.pseudo_label, result_b.pseudo_label))
        self.assertTrue(torch.equal(result_a.alpha, result_b.alpha))
        self.assertEqual(result_a.diagnostics["target_beta"], 1.0)

    def test_label_propagation_soft_targets_do_not_use_clean_labels(self):
        indices = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ],
            dtype=torch.long,
        )
        adj = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (4, 4)).coalesce()
        noisy = torch.tensor([0, 1, 1, 2])
        clean_a = torch.tensor([0, 1, 1, 2])
        clean_b = torch.tensor([2, 2, 0, 0])
        train_mask = np.array([0, 1, 2])

        result_a = build_label_propagation_soft_targets(
            noisy_label=noisy,
            clean_label=clean_a,
            train_mask=train_mask,
            adj=adj,
            n_classes=3,
            num_steps=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
        )
        result_b = build_label_propagation_soft_targets(
            noisy_label=noisy,
            clean_label=clean_b,
            train_mask=train_mask,
            adj=adj,
            n_classes=3,
            num_steps=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
        )

        self.assertTrue(torch.allclose(result_a.soft_target, result_b.soft_target))
        self.assertTrue(torch.equal(result_a.pseudo_label, result_b.pseudo_label))
        self.assertTrue(torch.allclose(result_a.alpha, result_b.alpha))
        self.assertEqual(result_a.diagnostics["target_source"], "label_propagation")
        self.assertEqual(result_a.diagnostics["lp_steps"], 2)

    def test_ellipsoid_soft_targets_match_uniform_when_mahalanobis_weights_equal(self):
        noisy = torch.tensor([0, 1, 1])
        train_mask = np.array([0, 1, 2])
        assignment = np.array([0, 0, 0])
        ball_features = torch.tensor(
            [
                [1.0, 0.0],
                [-0.5, 0.8660254],
                [-0.5, -0.8660254],
            ]
        )

        uniform_target, uniform_alpha, uniform_pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        ellipsoid_target, ellipsoid_alpha, ellipsoid_pseudo, diag = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        self.assertTrue(torch.allclose(ellipsoid_target, uniform_target, atol=1e-6))
        self.assertTrue(torch.allclose(ellipsoid_alpha, uniform_alpha, atol=1e-6))
        self.assertEqual(ellipsoid_pseudo.tolist(), uniform_pseudo.tolist())
        self.assertEqual(diag["ellipsoid_fallback_balls"], 0)
        self.assertAlmostEqual(diag["ellipsoid_effective_support_ratio_mean"], 1.0, places=5)

    def test_ellipsoid_soft_targets_use_weighted_distribution_from_ball_geometry(self):
        noisy = torch.tensor([0, 1, 1, 0])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0])
        ball_features = torch.tensor(
            [
                [0.0, 0.0],
                [0.05, 0.0],
                [0.10, 0.0],
                [4.0, 4.0],
            ]
        )

        uniform_target, uniform_alpha, _ = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        soft_target, alpha, pseudo_label, diag = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        self.assertEqual(diag["ellipsoid_fallback_balls"], 0)
        self.assertLess(diag["ellipsoid_min_train_weight"], diag["ellipsoid_mean_train_weight"])
        self.assertLess(diag["ellipsoid_effective_support_ratio_mean"], 1.0)
        self.assertEqual(pseudo_label.tolist(), [1, 1, 1, 1])
        self.assertGreater(float(soft_target[0, 1]), float(uniform_target[0, 1]))
        self.assertGreater(float(alpha[0]), float(uniform_alpha[0]))

    def test_ellipsoid_target_beta_hardens_weighted_distribution(self):
        noisy = torch.tensor([0, 1, 1, 0])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0])
        ball_features = torch.tensor(
            [
                [0.0, 0.0],
                [0.05, 0.0],
                [0.10, 0.0],
                [4.0, 4.0],
            ]
        )

        soft, alpha, pseudo, _ = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.0,
        )
        half, half_alpha, half_pseudo, diag = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
        )
        hard_like, hard_alpha, hard_pseudo, _ = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=1.0,
        )

        self.assertTrue(torch.allclose(half, 0.5 * soft + 0.5 * hard_like, atol=1e-6))
        self.assertTrue(torch.allclose(half_alpha, alpha))
        self.assertTrue(torch.allclose(hard_alpha, alpha))
        self.assertEqual(half_pseudo.tolist(), pseudo.tolist())
        self.assertEqual(hard_pseudo.tolist(), pseudo.tolist())
        self.assertAlmostEqual(diag["target_beta"], 0.5)

    def test_ellipsoid_soft_targets_fall_back_to_uniform_on_invalid_covariance(self):
        noisy = torch.tensor([0, 1, 1])
        train_mask = np.array([0, 1, 2])
        assignment = np.array([0, 0, 0])
        ball_features = torch.tensor(
            [
                [float("nan"), 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )

        uniform_target, uniform_alpha, uniform_pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        ellipsoid_target, ellipsoid_alpha, ellipsoid_pseudo, diag = apply_ball_ellipsoid_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=2,
            ball_features=ball_features,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )

        self.assertTrue(torch.all(torch.isfinite(ellipsoid_target)))
        self.assertTrue(torch.allclose(ellipsoid_target, uniform_target, atol=1e-6))
        self.assertTrue(torch.allclose(ellipsoid_alpha, uniform_alpha, atol=1e-6))
        self.assertEqual(ellipsoid_pseudo.tolist(), uniform_pseudo.tolist())
        self.assertEqual(diag["ellipsoid_fallback_balls"], 1)

    def test_build_soft_targets_from_features_reports_ellipsoid_weighting_for_real_and_shuffled(self):
        noisy = torch.tensor([0, 0, 1, 1, 0, 1])
        clean = noisy.clone()
        train_mask = np.array([0, 1, 2, 3])
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )

        real = build_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy,
            clean_label=clean,
            train_mask=train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            random_state=0,
            random_control=False,
            softdist_weighting="ellipsoid",
        )
        shuffled = build_soft_targets_from_features(
            ball_features=features,
            noisy_label=noisy,
            clean_label=clean,
            train_mask=train_mask,
            n_classes=2,
            max_balls=2,
            min_ball_size=2,
            split_improvement=0.0,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            random_state=0,
            random_control=True,
            softdist_weighting="ellipsoid",
        )

        self.assertEqual(real.diagnostics["softdist_weighting"], "ellipsoid")
        self.assertEqual(shuffled.diagnostics["softdist_weighting"], "ellipsoid")
        self.assertIn("ellipsoid_mean_train_weight", real.diagnostics)
        self.assertIn("ellipsoid_mean_train_weight", shuffled.diagnostics)

    def test_dynamic_alpha_uses_prediction_agreement(self):
        noisy = torch.tensor([0, 1, 1, 2])
        prediction = torch.tensor([1, 1, 0, 1])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0])

        soft_target, alpha, pseudo_label = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=4,
            min_train_per_ball=3,
            alpha_baseline="multiclass",
            target_mode="distribution",
            prediction_label=prediction,
            agreement_beta=0.5,
        )

        self.assertEqual(pseudo_label.tolist(), [1, 1, 1, 1])
        self.assertAlmostEqual(float(alpha[0]), 0.5, places=5)
        self.assertAlmostEqual(float(alpha[1]), 0.5, places=5)
        self.assertAlmostEqual(float(alpha[2]), 1.0 / 6.0, places=5)
        self.assertAlmostEqual(float(alpha[3]), 0.5, places=5)
        self.assertTrue(torch.allclose(soft_target[0], torch.tensor([0.625, 0.25, 0.125, 0.0])))
        self.assertTrue(torch.allclose(soft_target[2], torch.tensor([1.0 / 24.0, 11.0 / 12.0, 1.0 / 24.0, 0.0])))

    def test_ball_consistency_loss_penalizes_within_ball_prediction_variance(self):
        logits = torch.log(torch.tensor([
            [0.8, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.1, 0.9],
        ]))
        assignment = np.array([0, 0, 1, 1])

        loss = ball_consistency_loss(logits, assignment, min_ball_size=2)

        self.assertAlmostEqual(float(loss), 0.01, places=5)

    def test_ball_consistency_loss_ignores_singletons(self):
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assignment = np.array([0, 1])

        loss = ball_consistency_loss(logits, assignment, min_ball_size=2)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_soft_target_cross_entropy_matches_manual_value(self):
        logits = torch.log(torch.tensor([
            [0.8, 0.2],
            [0.1, 0.9],
        ]))
        target = torch.tensor([
            [0.75, 0.25],
            [0.0, 1.0],
        ])
        mask = torch.tensor([True, False])

        loss = soft_target_cross_entropy(logits, target, mask)

        expected = -(0.75 * torch.log(torch.tensor(0.8)) + 0.25 * torch.log(torch.tensor(0.2)))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_expansion_diagnostic_selects_only_unlabeled_nodes_by_train_votes(self):
        noisy = torch.tensor([1, 1, 0, 1, 2, 2, 0, 0])
        clean = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        result = diagnose_ball_expansion_candidates(
            noisy_label=noisy,
            clean_label=clean,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
            margin_threshold=0.6,
        )

        self.assertEqual(result.candidate_mask.tolist(), [False, False, False, False, True, True, False, False])
        self.assertEqual(result.pseudo_label.tolist(), [1, 1, 0, 1, 1, 1, 0, 0])
        self.assertEqual(result.diagnostics["candidate_count"], 2)
        self.assertAlmostEqual(result.diagnostics["coverage_unlabeled"], 0.5)
        self.assertAlmostEqual(result.diagnostics["precision"], 1.0)

    def test_expansion_candidate_mask_is_independent_of_clean_labels(self):
        noisy = torch.tensor([0, 0, 1, 1, 2, 2])
        clean_a = torch.tensor([0, 0, 1, 1, 0, 0])
        clean_b = torch.tensor([0, 0, 1, 1, 1, 0])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 1, 1, 0, 1])

        result_a = diagnose_ball_expansion_candidates(
            noisy_label=noisy,
            clean_label=clean_a,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
            margin_threshold=0.9,
        )
        result_b = diagnose_ball_expansion_candidates(
            noisy_label=noisy,
            clean_label=clean_b,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=2,
            margin_threshold=0.9,
        )

        self.assertEqual(result_a.candidate_mask.tolist(), result_b.candidate_mask.tolist())
        self.assertEqual(result_a.pseudo_label.tolist(), result_b.pseudo_label.tolist())
        self.assertNotEqual(result_a.diagnostics["precision"], result_b.diagnostics["precision"])

    def test_expansion_soft_targets_exclude_train_and_use_train_noisy_distribution(self):
        noisy = torch.tensor([1, 1, 1, 2, 0, 0, 2, 2])
        train_mask = np.array([0, 1, 2, 3])
        assignment = np.array([0, 0, 0, 0, 0, 0, 1, 1])

        result = build_ball_expansion_soft_targets(
            noisy_label=noisy,
            assignment=assignment,
            train_mask=train_mask,
            n_classes=3,
            min_train_per_ball=4,
            margin_threshold=0.7,
        )

        self.assertEqual(result.candidate_mask.tolist(), [False, False, False, False, True, True, False, False])
        self.assertTrue(torch.allclose(result.soft_target[4], torch.tensor([0.0, 0.75, 0.25])))
        self.assertTrue(torch.allclose(result.soft_target[5], torch.tensor([0.0, 0.75, 0.25])))
        self.assertEqual(result.pseudo_label.tolist(), [1, 1, 1, 2, 1, 1, 2, 2])

    def test_expansion_soft_target_loss_is_zero_when_lambda_is_zero(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        mask = torch.tensor([True, False])

        loss = expansion_soft_target_loss(logits, target, mask, lambda_u=0.0)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_early_arbitration_protects_high_confidence_original_labels(self):
        noisy = torch.tensor([0, 1, 0, 2, 1])
        clean = torch.tensor([0, 0, 0, 1, 1])
        alpha = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0])
        pseudo = torch.tensor([1, 0, 0, 1, 1])
        train_mask = np.array([0, 1, 2, 3, 4])
        early_noisy_conf = torch.tensor([0.8, 0.2, 0.9, 0.1, 0.9])

        diag = diagnostics_for_early_arbitration(
            noisy_label=noisy,
            clean_label=clean,
            alpha=alpha,
            pseudo_label=pseudo,
            train_mask=train_mask,
            early_noisy_conf=early_noisy_conf,
            tau_protect=0.7,
        )

        self.assertEqual(diag["baseline_changed_train_count"], 3)
        self.assertEqual(diag["protected_count"], 1)
        self.assertAlmostEqual(diag["protection_precision"], 1.0)
        self.assertAlmostEqual(diag["baseline_cdr"], 1.0 / 3.0)
        self.assertAlmostEqual(diag["protected_cdr"], 0.0)
        self.assertAlmostEqual(diag["cdr_relative_reduction"], 1.0)
        self.assertAlmostEqual(diag["baseline_ncr"], 1.0)
        self.assertAlmostEqual(diag["protected_ncr"], 1.0)
        self.assertAlmostEqual(diag["ncr_relative_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
