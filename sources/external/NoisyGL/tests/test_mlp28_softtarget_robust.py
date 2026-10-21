import importlib
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMLP28SoftTargetRobust(unittest.TestCase):
    def test_soft_cross_entropy_is_nodewise_and_normalized(self):
        module = importlib.import_module("predictor.LCSSRobust_Predictor")
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([[0.75, 0.25], [0.1, 0.9]])

        losses = module.soft_cross_entropy_per_sample(logits, targets)

        expected = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)
        self.assertEqual(tuple(losses.shape), (2,))
        self.assertTrue(torch.allclose(losses, expected))

    def test_soft_small_loss_selects_lowest_losses(self):
        module = importlib.import_module("predictor.LCSSRobust_Predictor")
        losses = torch.tensor([0.4, 0.1, 0.3, 0.2])

        selected = module.select_soft_small_loss_indices(
            losses,
            epoch=9,
            noise_rate=0.5,
            num_gradual=10,
            exponent=1.0,
        )

        self.assertEqual(selected.tolist(), [1, 3])

    def test_soft_jocor_joint_loss_uses_soft_targets_and_symmetric_kl(self):
        module = importlib.import_module("predictor.LCSSRobust_Predictor")
        logits1 = torch.tensor([[2.0, 0.0], [0.2, 1.0]], requires_grad=True)
        logits2 = torch.tensor([[1.5, 0.1], [0.0, 1.4]], requires_grad=True)
        targets = torch.tensor([[0.8, 0.2], [0.1, 0.9]])

        losses = module.soft_jocor_per_sample(logits1, logits2, targets, co_lambda=0.1)
        losses.mean().backward()

        ce1 = module.soft_cross_entropy_per_sample(logits1.detach(), targets)
        ce2 = module.soft_cross_entropy_per_sample(logits2.detach(), targets)
        prob1 = torch.softmax(logits1.detach(), dim=1)
        prob2 = torch.softmax(logits2.detach(), dim=1)
        kl12 = torch.nn.functional.kl_div(
            torch.log_softmax(logits1.detach(), dim=1), prob2, reduction="none"
        ).sum(dim=1)
        kl21 = torch.nn.functional.kl_div(
            torch.log_softmax(logits2.detach(), dim=1), prob1, reduction="none"
        ).sum(dim=1)
        expected = 0.9 * (ce1 + ce2) + 0.1 * (kl12 + kl21)
        self.assertTrue(torch.allclose(losses.detach(), expected))
        self.assertIsNotNone(logits1.grad)
        self.assertIsNotNone(logits2.grad)

    def test_predictors_are_registered_and_keep_lcss_target_builder(self):
        runner = importlib.import_module("scripts.run_standard_downstream")
        module = importlib.import_module("predictor.LCSSRobust_Predictor")
        expected = {
            "frozengrace_gb_mlp_softdist_beta05_logitclip": ("logitclip", False),
            "frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled": ("logitclip", True),
            "frozengrace_gb_mlp_softdist_beta05_small_loss": ("soft_small_loss", False),
            "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled": ("soft_small_loss", True),
            "frozengrace_gb_mlp_softdist_beta05_soft_jocor": ("soft_jocor", False),
            "frozengrace_gb_mlp_softdist_beta05_soft_jocor_shuffled": ("soft_jocor", True),
        }
        for method, (strategy, shuffled) in expected.items():
            self.assertEqual(
                runner.PREDICTOR_MODULES[method],
                "predictor.LCSSRobust_Predictor",
            )
            cls = getattr(module, f"{method}_Predictor")
            self.assertEqual(cls.robust_strategy, strategy)
            self.assertEqual(cls.random_control, shuffled)
            self.assertTrue(cls.use_soft_targets)
            self.assertEqual(cls.backbone, "mlp")
            if strategy == "soft_jocor":
                self.assertTrue(cls.uses_two_models)

    def test_configs_exist_for_preregistered_datasets(self):
        methods = (
            "frozengrace_gb_mlp_softdist_beta05_logitclip",
            "frozengrace_gb_mlp_softdist_beta05_logitclip_shuffled",
            "frozengrace_gb_mlp_softdist_beta05_small_loss",
            "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled",
            "frozengrace_gb_mlp_softdist_beta05_soft_jocor",
            "frozengrace_gb_mlp_softdist_beta05_soft_jocor_shuffled",
        )
        for method in methods:
            for dataset in ("cora", "citeseer", "citeseerfull"):
                path = REPO_ROOT / "config" / method / f"{method}_{dataset}.yaml"
                self.assertTrue(path.exists(), str(path))

    def test_gate_requires_synergy_and_real_structure(self):
        summary = importlib.import_module("scripts.summarize_mlp28_softtarget_robust")
        rows = []
        values = {
            "cora": {
                summary.LCSS: 0.50,
                summary.LOGIT_ONLY: 0.51,
                summary.LOGIT_COMBO: 0.53,
                summary.LOGIT_SHUFFLED: 0.50,
                summary.SMALL_ONLY: 0.52,
                summary.SMALL_COMBO: 0.51,
                summary.SMALL_SHUFFLED: 0.49,
            },
            "citeseer": {
                summary.LCSS: 0.40,
                summary.LOGIT_ONLY: 0.42,
                summary.LOGIT_COMBO: 0.43,
                summary.LOGIT_SHUFFLED: 0.41,
                summary.SMALL_ONLY: 0.43,
                summary.SMALL_COMBO: 0.42,
                summary.SMALL_SHUFFLED: 0.40,
            },
            "citeseerfull": {
                summary.LCSS: 0.45,
                summary.LOGIT_ONLY: 0.48,
                summary.LOGIT_COMBO: 0.479,
                summary.LOGIT_SHUFFLED: 0.44,
                summary.SMALL_ONLY: 0.49,
                summary.SMALL_COMBO: 0.47,
                summary.SMALL_SHUFFLED: 0.45,
            },
        }
        for dataset, methods in values.items():
            for method, test in methods.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": "0",
                        "noise_type": "uniform",
                        "noise_rate": "0.6",
                        "noise_scope": "train",
                        "status": "ok",
                        "test": str(test),
                        "modified_val_count": "0",
                        "modified_test_count": "0",
                    }
                )

        _, deltas, gates = summary.build_summary(rows)

        gate_by_strategy = {row["strategy"]: row for row in gates}
        self.assertEqual(gate_by_strategy["logitclip"]["pass"], 1)
        self.assertEqual(gate_by_strategy["soft_small_loss"]["pass"], 0)
        logit_cora = next(
            row for row in deltas if row["strategy"] == "logitclip" and row["dataset"] == "cora"
        )
        self.assertAlmostEqual(logit_cora["synergy_delta"], 0.02)
        self.assertAlmostEqual(logit_cora["structure_delta"], 0.03)

    def test_soft_jocor_gate_penalizes_complex_model_without_one_pp_gain(self):
        summary = importlib.import_module("scripts.summarize_mlp28_soft_jocor")
        rows = []
        for dataset, lcss, jocor, small, soft_jocor, shuffled in (
            ("cora", 0.39, 0.38, 0.45, 0.456, 0.40),
            ("citeseer", 0.38, 0.37, 0.42, 0.425, 0.39),
            ("citeseerfull", 0.40, 0.39, 0.49, 0.492, 0.41),
        ):
            for method, test in (
                (summary.LCSS, lcss),
                (summary.JOCOR_ONLY, jocor),
                (summary.SOFT_SMALL, small),
                (summary.SOFT_JOCOR, soft_jocor),
                (summary.SOFT_JOCOR_SHUFFLED, shuffled),
            ):
                rows.append({"dataset": dataset, "method": method, "seed": "0", "test": str(test)})

        _, deltas, gate = summary.build_summary(rows)

        self.assertEqual(gate["pass"], 0)
        self.assertIn("mean_gain_below_1pp", gate["reasons"])
        self.assertEqual(sum(row["soft_jocor_minus_soft_small"] > 0 for row in deltas), 3)

    def test_soft_small_five_seed_gate_requires_four_of_five_paired_wins(self):
        summary = importlib.import_module("scripts.summarize_mlp28_soft_small_5seed")
        rows = []
        for dataset in ("cora", "citeseer", "citeseerfull"):
            for seed in range(5):
                lcss = 0.40 + 0.001 * seed
                small = 0.39 + 0.001 * seed
                combo = lcss + (0.02 if seed < 4 else -0.001)
                shuffled = combo - 0.03
                for method, test in (
                    (summary.LCSS, lcss),
                    (summary.SMALL_ONLY, small),
                    (summary.SMALL_COMBO, combo),
                    (summary.SMALL_SHUFFLED, shuffled),
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": str(seed),
                            "noise_type": "uniform",
                            "noise_rate": "0.6",
                            "noise_scope": "train",
                            "status": "ok",
                            "test": str(test),
                            "modified_val_count": "0",
                            "modified_test_count": "0",
                            "split_hash": f"split-{dataset}-{seed}",
                            "noise_hash": f"noise-{dataset}-{seed}",
                        }
                    )

        _, deltas, gate = summary.build_summary(rows)

        self.assertEqual(gate["pass"], 1)
        self.assertTrue(all(row["synergy_positive_seeds"] == 4 for row in deltas))


if __name__ == "__main__":
    unittest.main()
