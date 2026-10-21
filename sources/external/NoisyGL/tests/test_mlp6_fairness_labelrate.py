from pathlib import Path
import unittest

from scripts import summarize_mlp6_fairness_labelrate as summary


class MLP6FairnessLabelrateTests(unittest.TestCase):
    def test_expected_grid_sizes_match_plan(self):
        self.assertEqual(len(summary.expected_new_keys()), 265)
        self.assertEqual(len(summary.expected_combined_keys()), 435)

    def test_current_output_names_are_parsed(self):
        self.assertEqual(
            summary.parse_current_table(Path("fairness_uniform0p4_train0p20_seeds_0_1_2_3_4.csv")),
            ("fairness", 0.20),
        )
        self.assertEqual(
            summary.parse_current_table(Path("fairness_uniform0p4_train0p05_seeds_0_1_2_3_4.csv")),
            ("fairness_low", 0.05),
        )
        self.assertEqual(
            summary.parse_current_table(Path("labelrate_train0p35_seeds_0_1_2_3_4.csv")),
            ("labelrate", 0.35),
        )

    def test_runner_uses_rocm_and_registered_methods(self):
        script = Path("scripts/run_mlp6_fairness_labelrate_20260701.ps1").read_text(encoding="utf-8")

        self.assertIn(".venv-rocm\\Scripts\\python.exe", script)
        self.assertIn("mlp_xh_ce,mlp_small_loss_xh,mlp_jocor_xh", script)
        self.assertIn("frozengrace_gb_mlp_softdist_beta05,mlp_small_loss,mlp_jocor,mlp_fine", script)
        self.assertIn("--noise_scope train", script)
        self.assertIn("--split_type percent", script)


if __name__ == "__main__":
    unittest.main()
