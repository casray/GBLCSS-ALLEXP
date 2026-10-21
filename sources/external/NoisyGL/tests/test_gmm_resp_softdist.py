import unittest

import numpy as np
import torch

from utils.frozen_grace_targets import (
    apply_ball_soft_targets,
    apply_gmm_resp_soft_targets,
    apply_gmm_resp_soft_targets_from_responsibilities,
    shuffle_responsibilities_for_control,
)


class TestGMMRespSoftDist(unittest.TestCase):
    def test_responsibility_targets_use_jeffreys_shrinkage(self):
        noisy = torch.tensor([0, 1, 0, 1])
        clean = noisy.clone()
        train_mask = np.array([0, 1, 2, 3])
        responsibilities = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )

        soft_target, alpha, pseudo_label, diag, assignment = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean,
            responsibilities=responsibilities,
            train_mask=train_mask,
            n_classes=2,
            alpha_baseline="multiclass",
            gmm_prior="jeffreys",
            gmm_geometry_confidence="max_resp",
        )

        p0 = torch.tensor([2.0 / 2.75, 0.75 / 2.75])
        p1 = torch.tensor([1.0 / 3.25, 2.25 / 3.25])
        p2 = 0.5 * p0 + 0.5 * p1
        self.assertTrue(torch.allclose(soft_target[0], (1.0 - alpha[0]) * torch.tensor([1.0, 0.0]) + alpha[0] * p0))
        self.assertTrue(torch.allclose(soft_target[1], (1.0 - alpha[1]) * torch.tensor([0.0, 1.0]) + alpha[1] * p1))
        self.assertTrue(torch.allclose(soft_target[2], (1.0 - alpha[2]) * torch.tensor([1.0, 0.0]) + alpha[2] * p2))
        self.assertEqual(pseudo_label.tolist(), [0, 1, 0, 1])
        self.assertEqual(assignment.tolist(), [0, 1, 0, 1])
        self.assertAlmostEqual(float(alpha[0]), 2.0 * float(p0.max()) - 1.0, places=5)
        self.assertEqual(diag["softdist_weighting"], "gmm_resp")
        self.assertEqual(diag["gmm_prior"], "jeffreys")
        self.assertEqual(diag["gmm_fallback"], 0)

    def test_max_resp_geometry_confidence_lowers_boundary_alpha(self):
        noisy = torch.tensor([0, 1, 0, 1])
        clean = noisy.clone()
        responsibilities = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )

        _, alpha, _, diag, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
            alpha_baseline="multiclass",
            gmm_geometry_confidence="max_resp",
        )

        self.assertLess(float(alpha[2]), float(alpha[0]))
        self.assertAlmostEqual(diag["gmm_max_resp_mean"], float(np.max(responsibilities, axis=1).mean()))
        self.assertGreater(diag["gmm_resp_entropy_mean"], 0.0)

    def test_target_beta_hardens_responsibility_weighted_distribution(self):
        noisy = torch.tensor([0, 1, 0, 1])
        clean = noisy.clone()
        responsibilities = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )

        soft, alpha, pseudo, _, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.0,
        )
        half, half_alpha, half_pseudo, diag, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            target_beta=0.5,
        )
        hard_like, hard_alpha, hard_pseudo, _, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
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

    def test_clean_labels_do_not_change_target_construction(self):
        noisy = torch.tensor([0, 1, 0, 1])
        clean_a = torch.tensor([0, 1, 0, 1])
        clean_b = torch.tensor([1, 0, 1, 0])
        responsibilities = np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.4, 0.6],
            ],
            dtype=np.float64,
        )

        target_a, alpha_a, pseudo_a, _, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean_a,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
            alpha_baseline="multiclass",
        )
        target_b, alpha_b, pseudo_b, _, _ = apply_gmm_resp_soft_targets_from_responsibilities(
            noisy_label=noisy,
            clean_label=clean_b,
            responsibilities=responsibilities,
            train_mask=np.arange(4),
            n_classes=2,
            alpha_baseline="multiclass",
        )

        self.assertTrue(torch.allclose(target_a, target_b))
        self.assertTrue(torch.allclose(alpha_a, alpha_b))
        self.assertEqual(pseudo_a.tolist(), pseudo_b.tolist())

    def test_shuffled_control_is_deterministic_row_permutation(self):
        responsibilities = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.55, 0.45],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )

        shuffled_a = shuffle_responsibilities_for_control(responsibilities, random_state=7)
        shuffled_b = shuffle_responsibilities_for_control(responsibilities, random_state=7)
        shuffled_c = shuffle_responsibilities_for_control(responsibilities, random_state=8)

        self.assertTrue(np.allclose(shuffled_a, shuffled_b))
        self.assertFalse(np.allclose(shuffled_a, responsibilities))
        self.assertFalse(np.allclose(shuffled_a, shuffled_c))
        self.assertTrue(
            np.allclose(
                np.sort(shuffled_a, axis=0),
                np.sort(responsibilities, axis=0),
            )
        )

    def test_gmm_fit_failure_falls_back_to_uniform_softdist(self):
        noisy = torch.tensor([0, 1, 1, 0])
        clean = noisy.clone()
        train_mask = np.array([0, 1, 2, 3])
        fallback_assignment = np.array([0, 0, 0, 0])
        invalid_features = torch.tensor(
            [
                [float("nan"), 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        uniform_target, uniform_alpha, uniform_pseudo = apply_ball_soft_targets(
            noisy_label=noisy,
            assignment=fallback_assignment,
            train_mask=train_mask,
            n_classes=2,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
        )
        soft_target, alpha, pseudo_label, diag, assignment = apply_gmm_resp_soft_targets(
            noisy_label=noisy,
            clean_label=clean,
            ball_features=invalid_features,
            train_mask=train_mask,
            n_classes=2,
            fallback_assignment=fallback_assignment,
            min_train_per_ball=2,
            alpha_baseline="multiclass",
            target_mode="distribution",
            gmm_n_components=2,
        )

        self.assertTrue(torch.all(torch.isfinite(soft_target)))
        self.assertTrue(torch.allclose(soft_target, uniform_target))
        self.assertTrue(torch.allclose(alpha, uniform_alpha))
        self.assertEqual(pseudo_label.tolist(), uniform_pseudo.tolist())
        self.assertEqual(assignment.tolist(), fallback_assignment.tolist())
        self.assertEqual(diag["gmm_fallback"], 1)

    def test_diagnostic_gate_rejects_when_gmm_does_not_beat_shuffled(self):
        from scripts.diagnose_gmm_resp_softdist import evaluate_diagnostic_gate

        rows = []
        for seed in range(5):
            rows.append(
                {
                    "dataset": "cora",
                    "method": "frozengrace_gb_gcn_softdist_xh",
                    "seed": seed,
                    "soft_clean_mass_delta": 0.10,
                    "soft_pseudo_cdr": 0.20,
                    "soft_pseudo_ncr": 0.30,
                }
            )
            rows.append(
                {
                    "dataset": "cora",
                    "method": "frozengrace_gb_gcn_softdist_xh_gmmresp",
                    "seed": seed,
                    "soft_clean_mass_delta": 0.12,
                    "soft_pseudo_cdr": 0.18,
                    "soft_pseudo_ncr": 0.30,
                    "gmm_converged": 1,
                    "gmm_low_support_component_ratio": 0.10,
                }
            )
            rows.append(
                {
                    "dataset": "cora",
                    "method": "frozengrace_gb_gcn_softdist_xh_gmmresp_shuffled",
                    "seed": seed,
                    "soft_clean_mass_delta": 0.13,
                    "soft_pseudo_cdr": 0.17,
                    "soft_pseudo_ncr": 0.30,
                    "gmm_converged": 1,
                    "gmm_low_support_component_ratio": 0.10,
                }
            )

        gate_passed, decisions = evaluate_diagnostic_gate(rows, min_passing_datasets=1)

        self.assertFalse(gate_passed)
        self.assertFalse(decisions[0]["beats_shuffled"])


if __name__ == "__main__":
    unittest.main()
