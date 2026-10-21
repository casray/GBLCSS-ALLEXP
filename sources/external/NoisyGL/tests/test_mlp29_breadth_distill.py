import importlib
import unittest
from pathlib import Path

import torch

from utils.tools import load_conf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("corafull", "coauthorcs", "amazonpho", "pubmed", "amazoncom", "coauthorph")


class TestMLP29BreadthDistill(unittest.TestCase):
    def test_robust_teacher_loss_uses_soft_targets_and_small_loss_schedule(self):
        module = importlib.import_module("predictor.LCSSRobustDistill_Predictor")
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 1.0]])
        targets = torch.tensor([[0.9, 0.1], [0.1, 0.9], [1.0, 0.0], [1.0, 0.0]])

        loss, selected, per_sample = module.soft_small_teacher_loss(
            logits,
            targets,
            epoch=9,
            noise_rate=0.5,
            num_gradual=10,
            exponent=1.0,
        )

        self.assertEqual(selected.numel(), 2)
        self.assertTrue(torch.allclose(loss, per_sample[selected].mean()))
        self.assertEqual(selected.tolist(), torch.topk(per_sample, 2, largest=False).indices.tolist())

    def test_new_distill_methods_are_registered(self):
        runner = importlib.import_module("scripts.run_standard_downstream")
        module = importlib.import_module("predictor.LCSSRobustDistill_Predictor")
        expected = {
            "mlp_distill_softdist_beta05_xh": ("ball", False, False),
            "mlp_distill_knn_beta05_xh": ("knn", False, False),
            "mlp_distill_softdist_beta05_small_loss_xh": ("ball", True, False),
            "mlp_distill_softdist_beta05_small_loss_xh_shuffled": ("ball", True, True),
        }
        for method, (builder, robust, shuffled) in expected.items():
            self.assertEqual(runner.PREDICTOR_MODULES[method], "predictor.LCSSRobustDistill_Predictor")
            cls = getattr(module, f"{method}_Predictor")
            self.assertEqual(cls.soft_target_builder, builder)
            self.assertEqual(cls.robust_teacher, robust)
            self.assertEqual(cls.random_control, shuffled)
            self.assertTrue(cls.teacher_use_xh)

    def test_all_direct_and_distill_configs_exist(self):
        direct = (
            "frozengrace_gb_mlp_softdist_beta05_small_loss",
            "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled",
        )
        distill = (
            "mlp_distill_softdist_beta05_xh",
            "mlp_distill_knn_beta05_xh",
            "mlp_distill_softdist_beta05_small_loss_xh",
            "mlp_distill_softdist_beta05_small_loss_xh_shuffled",
        )
        for method in direct + distill:
            for dataset in DATASETS:
                path = REPO_ROOT / "config" / method / f"{method}_{dataset}.yaml"
                self.assertTrue(path.exists(), str(path))
                conf = load_conf(None, method, dataset)
                self.assertEqual(float(conf.gb["target_beta"]), 0.5)

    def test_summary_keeps_direct_and_distill_gates_separate(self):
        summary = importlib.import_module("scripts.summarize_mlp29_breadth_distill")
        direct_rows = []
        distill_rows = []
        for index, dataset in enumerate(DATASETS):
            base = 0.5 + 0.01 * index
            for method, value in (
                (summary.LCSS, base),
                (summary.SMALL_ONLY, base - 0.01),
                (summary.SMALL_COMBO, base + 0.03),
                (summary.SMALL_SHUFFLED, base - 0.02),
            ):
                direct_rows.append({"dataset": dataset, "method": method, "seed": "0", "test": str(value)})
            for method, value in (
                (summary.DISTILL_BARE, base + 0.1),
                (summary.DISTILL_SOFT, base + 0.11),
                (summary.DISTILL_KNN, base + 0.115),
                (summary.DISTILL_ROBUST, base + 0.13),
                (summary.DISTILL_ROBUST_SHUFFLED, base + 0.08),
            ):
                distill_rows.append({"dataset": dataset, "method": method, "seed": "0", "test": str(value)})

        _, _, gates = summary.build_summary(direct_rows, distill_rows)

        by_phase = {row["phase"]: row for row in gates}
        self.assertEqual(by_phase["direct"]["pass"], 1)
        self.assertEqual(by_phase["distill"]["pass"], 1)


if __name__ == "__main__":
    unittest.main()
