import importlib
import unittest


class MLP10BaselineRefreshTests(unittest.TestCase):
    def test_selected_logitclip_tau_uses_noisy_valid(self):
        summary = importlib.import_module("scripts.summarize_mlp10_baseline_refresh")
        rows = [
            {"dataset": "cora", "method": "mlp_logitclip", "seed": 0, "status": "ok", "logit_clip_tau": 0.5, "valid": 0.70},
            {"dataset": "cora", "method": "mlp_logitclip", "seed": 0, "status": "ok", "logit_clip_tau": 1.0, "valid": 0.75},
            {"dataset": "cora", "method": "mlp_logitclip", "seed": 0, "status": "ok", "logit_clip_tau": 2.0, "valid": 0.73},
            {"dataset": "citeseer", "method": "mlp_logitclip", "seed": 0, "status": "ok", "logit_clip_tau": 0.5, "valid": 0.60},
            {"dataset": "citeseer", "method": "mlp_logitclip", "seed": 0, "status": "ok", "logit_clip_tau": 1.0, "valid": 0.61},
        ]

        selected = {row["dataset"]: row["selected_tau"] for row in summary.selected_logitclip_tau(rows)}

        self.assertEqual(selected["cora"], 1.0)
        self.assertEqual(selected["citeseer"], 1.0)

    def test_ga2_counts_coauthorcs_seed0_ours_over_ce_and_shuffled(self):
        summary = importlib.import_module("scripts.summarize_mlp10_baseline_refresh")
        new_rows = [
            {"dataset": "coauthorcs", "noise_type": "uniform", "noise_rate": 0.4, "method": summary.OURS, "seed": 0, "status": "ok", "test": 0.80},
            {"dataset": "coauthorcs", "noise_type": "uniform", "noise_rate": 0.4, "method": summary.SHUFFLED, "seed": 0, "status": "ok", "test": 0.78},
        ]
        reuse_rows = [
            {"dataset": "coauthorcs", "noise_type": "uniform", "noise_rate": 0.4, "method": "mlp_ce", "seed": 0, "status": "ok", "test": 0.77},
        ]

        rows = summary.ga2_rows(new_rows, reuse_rows)
        target = [row for row in rows if row["noise_type"] == "uniform" and row["noise_rate"] == 0.4][0]

        self.assertEqual(target["passes"], 1)

    def test_ga3_requires_split_and_noise_hash_match(self):
        summary = importlib.import_module("scripts.summarize_mlp10_baseline_refresh")
        new_rows = [
            {
                "dataset": "coauthorcs",
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "method": summary.OURS,
                "seed": 0,
                "split_hash": "split",
                "noise_hash": "noise",
                "source": "new.csv",
            }
        ]
        reuse_rows = [
            {
                "dataset": "coauthorcs",
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "method": "mlp_ce",
                "seed": 0,
                "split_hash": "split",
                "noise_hash": "noise",
                "source": "old.csv",
            }
        ]

        audit = summary.ga3_rows(new_rows, reuse_rows)

        self.assertEqual(audit[0]["split_match"], 1)
        self.assertEqual(audit[0]["noise_match"], 1)


if __name__ == "__main__":
    unittest.main()
