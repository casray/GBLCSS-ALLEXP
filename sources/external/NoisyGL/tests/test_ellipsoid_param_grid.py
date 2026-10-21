import unittest

from scripts.diagnose_ellipsoid_param_grid import (
    aggregate_grid_rows,
    classify_dataset_settings,
    evaluate_grid_gates,
    paired_grid_delta_rows,
)


class EllipsoidParamGridDiagnosticTest(unittest.TestCase):
    def _row(
        self,
        dataset,
        seed,
        method,
        max_balls,
        split_improvement,
        mass,
        cdr,
        ncr=0.50,
        num_balls=64,
        fallback=0.0,
    ):
        return {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "max_balls": max_balls,
            "split_improvement": split_improvement,
            "soft_clean_mass_delta": mass,
            "soft_pseudo_cdr": cdr,
            "soft_pseudo_ncr": ncr,
            "num_balls": num_balls,
            "ellipsoid_fallback_ball_ratio": fallback,
        }

    def test_aggregate_groups_by_dataset_setting_and_method(self):
        rows = [
            self._row("dblp", 0, "base", 64, 0.05, -10.0, 0.5, num_balls=1),
            self._row("dblp", 1, "base", 64, 0.05, -12.0, 0.5, num_balls=1),
            self._row("dblp", 0, "base", 128, 0.05, -8.0, 0.4, num_balls=128),
            self._row("dblp", 0, "ellipsoid", 64, 0.05, -7.0, 0.3, num_balls=64),
        ]

        aggregate = aggregate_grid_rows(rows)

        keys = {
            (row["dataset"], row["method"], row["max_balls"], row["split_improvement"])
            for row in aggregate
        }
        self.assertEqual(
            keys,
            {
                ("dblp", "base", 64, 0.05),
                ("dblp", "base", 128, 0.05),
                ("dblp", "ellipsoid", 64, 0.05),
            },
        )
        base_64 = next(row for row in aggregate if row["method"] == "base" and row["max_balls"] == 64)
        self.assertEqual(base_64["n"], 2)
        self.assertAlmostEqual(base_64["soft_clean_mass_delta_mean"], -11.0)
        self.assertAlmostEqual(base_64["num_balls_mean"], 1.0)

    def test_paired_deltas_do_not_cross_grid_settings(self):
        rows = [
            self._row("dblp", 0, "base", 64, 0.05, -10.0, 0.5),
            self._row("dblp", 0, "ellipsoid", 64, 0.05, -7.0, 0.4),
            self._row("dblp", 0, "base", 128, 0.05, -20.0, 0.6),
            self._row("dblp", 0, "ellipsoid", 64, 0.01, -1.0, 0.1),
        ]

        deltas = paired_grid_delta_rows(rows, "base", "ellipsoid")

        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["max_balls"], 64)
        self.assertAlmostEqual(deltas[0]["split_improvement"], 0.05)
        self.assertAlmostEqual(deltas[0]["soft_clean_mass_delta_delta"], 3.0)
        self.assertAlmostEqual(deltas[0]["soft_pseudo_cdr_delta"], -0.1)

    def test_gate_is_evaluated_per_dataset_and_setting(self):
        rows = []
        for seed in range(5):
            rows.extend(
                [
                    self._row("dblp", seed, "base", 64, 0.01, -10.0, 0.50),
                    self._row("dblp", seed, "ellipsoid", 64, 0.01, -7.0, 0.49),
                    self._row("dblp", seed, "ellipsoid_shuffled", 64, 0.01, -30.0, 0.80),
                    self._row("dblp", seed, "base", 64, 0.05, -10.0, 0.50, num_balls=1),
                    self._row("dblp", seed, "ellipsoid", 64, 0.05, -10.0, 0.50, num_balls=1),
                    self._row("dblp", seed, "ellipsoid_shuffled", 64, 0.05, -10.0, 0.50, num_balls=1),
                ]
            )

        decisions = evaluate_grid_gates(
            rows,
            reference_method="base",
            candidate_method="ellipsoid",
            shuffled_method="ellipsoid_shuffled",
        )

        by_setting = {(row["max_balls"], row["split_improvement"]): row for row in decisions}
        self.assertTrue(by_setting[(64, 0.01)]["passed"])
        self.assertEqual(by_setting[(64, 0.01)]["positive_seed_count"], 5)
        self.assertFalse(by_setting[(64, 0.05)]["passed"])

    def test_classification_flags_collapse_fix_and_no_offline_value(self):
        decisions = [
            {
                "dataset": "dblp",
                "max_balls": 64,
                "split_improvement": 0.05,
                "passed": False,
                "candidate_num_balls_mean": 1.0,
                "mass_delta_mean": 0.0,
            },
            {
                "dataset": "dblp",
                "max_balls": 64,
                "split_improvement": 0.01,
                "passed": True,
                "candidate_num_balls_mean": 64.0,
                "mass_delta_mean": 3.2,
            },
            {
                "dataset": "pubmed",
                "max_balls": 64,
                "split_improvement": 0.05,
                "passed": False,
                "candidate_num_balls_mean": 64.0,
                "mass_delta_mean": -0.5,
            },
        ]

        classes = classify_dataset_settings(decisions)

        self.assertEqual(classes["dblp"]["classification"], "collapse-fixed")
        self.assertEqual(classes["pubmed"]["classification"], "no-offline-value")


if __name__ == "__main__":
    unittest.main()
