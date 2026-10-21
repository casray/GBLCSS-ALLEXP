import unittest

from scripts.diagnose_mlp18_ellipsoid_feasibility import (
    candidate_score,
    evaluate_offline_candidate_gate,
    geometry_passes,
    row_key,
    select_top_candidates,
)


class MLP18DiagnosticScriptTest(unittest.TestCase):
    def test_geometry_gate_requires_noncollapsed_partition_and_nontrivial_weighting(self):
        passing = {
            "num_balls": 32,
            "mean_train_per_ball": 8.0,
            "zero_train_ball_ratio": 0.1,
            "ellipsoid_fallback_ball_ratio": 0.0,
            "ellipsoid_effective_support_ratio_mean": 0.90,
            "target_l1_vs_current": 0.01,
        }
        collapsed = {**passing, "num_balls": 1}
        trivial = {
            **passing,
            "ellipsoid_effective_support_ratio_mean": 0.99,
            "target_l1_vs_current": 0.0,
        }

        self.assertTrue(geometry_passes(passing))
        self.assertFalse(geometry_passes(collapsed))
        self.assertFalse(geometry_passes(trivial))

    def test_offline_gate_uses_real_minus_shuffled_and_fallback(self):
        passing = {
            "positive_seed_count": 4,
            "mass_delta_per_train": 0.002,
            "hard_net": 0.0,
            "target_l1_vs_current": 0.03,
            "ellipsoid_fallback_ball_ratio": 0.0,
        }
        weak_real_minus_shuffled = {**passing, "positive_seed_count": 3}
        weak_target_shift = {**passing, "mass_delta_per_train": 0.0, "hard_net": 0.0}
        bad_fallback = {**passing, "ellipsoid_fallback_ball_ratio": 0.2}

        self.assertTrue(evaluate_offline_candidate_gate(passing))
        self.assertFalse(evaluate_offline_candidate_gate(weak_real_minus_shuffled))
        self.assertFalse(evaluate_offline_candidate_gate(weak_target_shift))
        self.assertFalse(evaluate_offline_candidate_gate(bad_fallback))

    def test_score_formula_is_fixed(self):
        row = {
            "mass_delta_per_train": 0.002,
            "hard_net": 0.015,
            "target_l1_vs_current": 0.04,
            "ellipsoid_fallback_ball_ratio": 0.1,
        }

        self.assertAlmostEqual(candidate_score(row), 1.95)

    def test_select_top_candidates_caps_per_dataset_after_gate(self):
        rows = []
        for idx, score_mass in enumerate([0.001, 0.003, 0.002]):
            rows.append(
                {
                    "dataset": "cora",
                    "candidate_id": f"c{idx}",
                    "positive_seed_count": 5,
                    "mass_delta_per_train": score_mass,
                    "hard_net": 0.02,
                    "target_l1_vs_current": 0.04,
                    "ellipsoid_fallback_ball_ratio": 0.0,
                }
            )
        rows.append(
            {
                "dataset": "citeseer",
                "candidate_id": "bad",
                "positive_seed_count": 1,
                "mass_delta_per_train": 1.0,
                "hard_net": 1.0,
                "target_l1_vs_current": 1.0,
                "ellipsoid_fallback_ball_ratio": 0.0,
            }
        )

        selected = select_top_candidates(rows, per_dataset=2)

        self.assertEqual([row["candidate_id"] for row in selected], ["c1", "c2"])

    def test_row_key_includes_candidate_parameters_for_resume(self):
        row = {
            "dataset": "cora",
            "seed": 0,
            "candidate_id": "x",
            "method": "candidate",
        }

        self.assertEqual(row_key(row), ("cora", 0, "x", "candidate"))


if __name__ == "__main__":
    unittest.main()
