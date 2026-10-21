import unittest


class MLP25CoarseHoldoutSmokeTests(unittest.TestCase):
    def test_mean_train_votes_falls_back_to_split_count(self):
        from scripts.summarize_mlp25_coarse_holdout_smoke_20260712 import effective_mean_train_votes

        self.assertAlmostEqual(
            effective_mean_train_votes({"dataset": "coraml", "num_balls": 30}),
            599 / 30,
        )

    def test_vote_target_cap_matches_preregistered_holdouts(self):
        from scripts.summarize_mlp25_coarse_holdout_smoke_20260712 import vote_target_cap

        self.assertEqual(vote_target_cap(846), 43)
        self.assertEqual(vote_target_cap(599), 30)
        self.assertEqual(vote_target_cap(481), 25)
        self.assertEqual(vote_target_cap(3958), 64)

    def test_seed0_gate_requires_each_candidate_own_shuffled_control(self):
        from scripts.summarize_mlp25_coarse_holdout_smoke_20260712 import smoke_gate

        datasets = ("citeseerfull", "coraml", "attribwiki")
        rows = []
        for dataset in datasets:
            common = {
                "dataset": dataset,
                "seed": 0,
                "status": "ok",
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "noise_scope": "train",
                "train_percent": 0.2,
                "val_percent": 0.1,
                "test_percent": 0.7,
                "modified_count": 10,
                "modified_train_count": 10,
                "modified_val_count": 0,
                "modified_test_count": 0,
                "split_hash": f"split-{dataset}",
                "noise_hash": f"noise-{dataset}",
                "num_balls": 25,
                "mean_train_per_ball": 20.0,
                "alpha_positive_train_count": 100,
            }
            rows.extend(
                [
                    {**common, "arm": "fixed", "test": 0.50},
                    {**common, "arm": "vote_target", "test": 0.52},
                    {**common, "arm": "vote_target_shuffled", "test": 0.49},
                    {**common, "arm": "candidate", "test": 0.52},
                    {**common, "arm": "candidate_shuffled", "test": 0.48},
                ]
            )

        decisions = smoke_gate(rows)
        self.assertEqual(decisions["vote_target"]["pass"], 1)
        self.assertEqual(decisions["candidate"]["pass"], 1)

        rows = [row for row in rows if row["arm"] != "candidate_shuffled"]
        decisions = smoke_gate(rows)
        self.assertEqual(decisions["candidate"]["pass"], 0)
        self.assertIn("missing_pair", decisions["candidate"]["reasons"])

    def test_gate_rejects_one_point_gain_and_large_regression(self):
        from scripts.summarize_mlp25_coarse_holdout_smoke_20260712 import smoke_gate

        datasets = ("citeseerfull", "coraml", "attribwiki")
        rows = []
        for index, dataset in enumerate(datasets):
            common = {
                "dataset": dataset,
                "seed": 0,
                "status": "ok",
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "noise_scope": "train",
                "train_percent": 0.2,
                "val_percent": 0.1,
                "test_percent": 0.7,
                "modified_count": 10,
                "modified_train_count": 10,
                "modified_val_count": 0,
                "modified_test_count": 0,
                "split_hash": f"split-{dataset}",
                "noise_hash": f"noise-{dataset}",
                "num_balls": 25,
                "mean_train_per_ball": 20.0,
                "alpha_positive_train_count": 100,
            }
            gain = (0.02, 0.005, -0.01)[index]
            rows.extend(
                [
                    {**common, "arm": "fixed", "test": 0.50},
                    {**common, "arm": "vote_target", "test": 0.50 + gain},
                    {**common, "arm": "vote_target_shuffled", "test": 0.47},
                    {**common, "arm": "candidate", "test": 0.50 + gain},
                    {**common, "arm": "candidate_shuffled", "test": 0.47},
                ]
            )

        decisions = smoke_gate(rows)
        self.assertEqual(decisions["vote_target"]["pass"], 0)
        self.assertIn("fewer_than_2_of_3_gain_1pp", decisions["vote_target"]["reasons"])
        self.assertIn("regression_over_0p5pp", decisions["vote_target"]["reasons"])


if __name__ == "__main__":
    unittest.main()
