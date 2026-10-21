import unittest


def make_row(dataset, config, test, seed=0, num_balls=8, mean_train=3, alpha_count=10, noisy_purity=0.5):
    max_balls, min_ball_size, split_improvement = config
    return {
        "dataset": dataset,
        "method": "frozengrace_gb_mlp_softdist_beta05",
        "seed": seed,
        "noise_type": "uniform",
        "noise_rate": 0.4,
        "status": "ok",
        "test": test,
        "gb_max_balls": max_balls,
        "gb_min_ball_size": min_ball_size,
        "gb_split_improvement": split_improvement,
        "num_balls": num_balls,
        "mean_train_per_ball": mean_train,
        "alpha_positive_train_count": alpha_count,
        "noisy_train_purity_weighted": noisy_purity,
    }


class MLP13BallParamSensitivityTests(unittest.TestCase):
    def setUp(self):
        from scripts import summarize_mlp13_ballparam_sensitivity as summary

        self.summary = summary
        self.default = summary.DEFAULT_CONFIG
        self.good = (128, 32, 0.01)
        self.spiky = (32, 16, 0.01)
        self.collapsed = (128, 16, 0.1)
        self.datasets = summary.DATASETS

    def _stage_a_rows(self):
        rows = []
        for dataset in self.datasets:
            rows.append(make_row(dataset, self.default, 0.80, noisy_purity=0.45))
            rows.append(make_row(dataset, self.good, 0.812, noisy_purity=0.55))
            rows.append(make_row(dataset, self.collapsed, 0.90, num_balls=1, noisy_purity=0.99))
        rows.append(make_row("cora", self.spiky, 0.93, noisy_purity=0.60))
        for dataset in self.datasets[1:]:
            rows.append(make_row(dataset, self.spiky, 0.765, noisy_purity=0.60))
        return rows

    def test_collapsed_config_is_rejected(self):
        summaries = self.summary.score_stage_a_candidates(self._stage_a_rows())
        collapsed = next(row for row in summaries if row["config_id"] == self.summary.config_id(*self.collapsed))
        self.assertEqual(collapsed["hard_reject_reason"], "collapse_num_balls")
        selected = self.summary.select_stage_b_configs(summaries)
        selected_ids = {row["config_id"] for row in selected}
        self.assertNotIn(self.summary.config_id(*self.collapsed), selected_ids)

    def test_default_config_is_always_selected_for_stage_b(self):
        summaries = self.summary.score_stage_a_candidates(self._stage_a_rows())
        selected = self.summary.select_stage_b_configs(summaries)
        self.assertEqual(selected[0]["config_id"], self.summary.config_id(*self.default))
        self.assertEqual(selected[0]["selected_reason"], "default")

    def test_selection_uses_global_score_not_single_dataset_spike(self):
        summaries = self.summary.score_stage_a_candidates(self._stage_a_rows())
        selected = self.summary.select_stage_b_configs(summaries)
        selected_ids = {row["config_id"] for row in selected}
        self.assertIn(self.summary.config_id(*self.good), selected_ids)
        self.assertNotIn(self.summary.config_id(*self.spiky), selected_ids)

    def test_paired_delta_uses_same_dataset_and_seed(self):
        candidate = self.good
        rows = [
            make_row("cora", self.default, 0.80, seed=0),
            make_row("cora", candidate, 0.81, seed=0),
            make_row("cora", self.default, 0.82, seed=1),
            make_row("cora", candidate, 0.83, seed=1),
            make_row("citeseer", self.default, 0.70, seed=0),
            make_row("citeseer", candidate, 0.69, seed=0),
        ]
        deltas = self.summary.stage_b_paired_deltas(rows, selected_configs=[self.default, candidate])
        by_key = {(row["dataset"], row["seed"], row["config_id"]): row for row in deltas}
        self.assertAlmostEqual(by_key[("cora", 0, self.summary.config_id(*candidate))]["delta_vs_default"], 0.01)
        self.assertAlmostEqual(by_key[("cora", 1, self.summary.config_id(*candidate))]["delta_vs_default"], 0.01)
        self.assertAlmostEqual(by_key[("citeseer", 0, self.summary.config_id(*candidate))]["delta_vs_default"], -0.01)

    def test_missing_mean_train_per_ball_is_derived_from_alpha_count_and_num_balls(self):
        row = make_row("cora", self.default, 0.80, num_balls=5, mean_train="", alpha_count=20)
        normalized = self.summary.normalize(row)
        self.assertEqual(normalized["mean_train_per_ball"], 4.0)
        self.assertEqual(normalized["mean_train_per_ball_source"], "alpha_positive_train_count_over_num_balls")


if __name__ == "__main__":
    unittest.main()
