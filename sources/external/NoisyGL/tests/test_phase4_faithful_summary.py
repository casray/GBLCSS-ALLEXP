import unittest


class Phase4FaithfulSummaryTests(unittest.TestCase):
    def test_formal_method_scope_excludes_proxies(self):
        from scripts.summarize_phase4_faithful_claims_20260712 import formal_baselines_for_regime

        regular = formal_baselines_for_regime("pair", 0.4)
        uniform04 = formal_baselines_for_regime("uniform", 0.4)

        for method in ("mlp_fine", "mlp_dividemix_lite", "mlp_anl", "mlp_ct"):
            self.assertNotIn(method, regular)
            self.assertNotIn(method, uniform04)
        self.assertNotIn("mlp_fine_official", regular)
        self.assertIn("mlp_fine_official", uniform04)
        self.assertIn("mlp_anl_ce", uniform04)
        self.assertIn("mlp_gmm_ct", uniform04)

    def test_strongest_baseline_requires_complete_paired_seeds(self):
        from scripts.summarize_phase4_faithful_claims_20260712 import summarize_setting

        rows = []
        for seed, ours, complete, incomplete in (
            (0, 0.70, 0.60, 0.99),
            (1, 0.72, 0.62, 0.99),
            (2, 0.71, 0.61, 0.99),
            (3, 0.73, 0.63, None),
            (4, 0.74, 0.64, None),
        ):
            rows.append({"dataset": "cora", "noise_type": "uniform", "noise_rate": 0.4,
                         "method": "ours", "seed": seed, "test": ours})
            rows.append({"dataset": "cora", "noise_type": "uniform", "noise_rate": 0.4,
                         "method": "complete", "seed": seed, "test": complete})
            if incomplete is not None:
                rows.append({"dataset": "cora", "noise_type": "uniform", "noise_rate": 0.4,
                             "method": "incomplete", "seed": seed, "test": incomplete})

        summary, paired = summarize_setting(
            rows,
            ours_method="ours",
            baseline_methods=("complete", "incomplete"),
            expected_seeds=(0, 1, 2, 3, 4),
        )

        self.assertEqual(summary["strongest_method"], "complete")
        self.assertEqual(summary["positive_seeds"], 5)
        self.assertAlmostEqual(summary["mean_delta"], 0.10, places=8)
        self.assertEqual(len(paired), 5)


if __name__ == "__main__":
    unittest.main()
