from pathlib import Path
import unittest

from scripts import summarize_mlp7_baselines as summary


class MLP7BaselinesTests(unittest.TestCase):
    def test_expected_grid_sizes_match_plan(self):
        self.assertEqual(len(summary.expected_new_keys()), 75)
        self.assertEqual(len(summary.expected_combined_keys()), 825)

    def test_current_output_name_parses_regime(self):
        self.assertEqual(
            summary.regime_from_current_name(Path("main_ct_uniform0p4_seeds_0_1_2_3_4.csv")),
            ("uniform", 0.4),
        )
        self.assertEqual(
            summary.regime_from_current_name(Path("main_ct_instance0p4_seeds_0_1_2_3_4.csv")),
            ("instance", 0.4),
        )

    def test_runner_uses_x_only_standard_runner_for_ct(self):
        script = Path("scripts/run_mlp7_baselines_20260701.ps1").read_text(encoding="utf-8")

        self.assertIn("run_standard_downstream.py", script)
        self.assertIn("--methods mlp_ct", script)
        self.assertIn("--noise_scope train", script)
        self.assertIn("--split_type percent", script)
        self.assertIn("check_grace_encoder_checkpoints.py", script)


if __name__ == "__main__":
    unittest.main()
