import importlib
import unittest

import torch

from utils.tools import load_conf


class TestMLP32RobustStudentDistill(unittest.TestCase):
    def test_joint_loss_combines_kd_and_selected_soft_ce(self):
        module = importlib.import_module("predictor.LCSSRobustStudentDistill_Predictor")
        student_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        teacher_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
        soft_targets = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        train_mask = torch.tensor([True, True, False])

        total, kd, robust, selected, per_sample = module.robust_student_distill_loss(
            student_logits,
            teacher_probs,
            soft_targets,
            train_mask,
            epoch=10,
            noise_rate=0.5,
            temperature=1.0,
            kd_weight=1.0,
            robust_weight=1.0,
            num_gradual=10,
            exponent=1.0,
        )

        self.assertEqual(selected.numel(), 1)
        self.assertEqual(per_sample.numel(), 2)
        self.assertTrue(torch.allclose(total, kd + robust))

    def test_real_and_shuffled_methods_are_registered(self):
        runner = importlib.import_module("scripts.run_standard_downstream")
        expected = {
            "mlp_distill_gcn_xh_lcss_small_loss",
            "mlp_distill_gcn_xh_lcss_small_loss_shuffled",
        }
        self.assertTrue(expected.issubset(runner.PREDICTOR_MODULES))

        module = importlib.import_module("predictor.LCSSRobustStudentDistill_Predictor")
        self.assertFalse(module.mlp_distill_gcn_xh_lcss_small_loss_Predictor.random_control)
        self.assertTrue(module.mlp_distill_gcn_xh_lcss_small_loss_shuffled_Predictor.random_control)
        self.assertTrue(module.mlp_distill_gcn_xh_lcss_small_loss_Predictor.teacher_use_xh)
        self.assertEqual(module.mlp_distill_gcn_xh_lcss_small_loss_Predictor.teacher_target, "ce")

    def test_six_dataset_configs_freeze_student_and_teacher_resources(self):
        datasets = ("cora", "citeseer", "corafull", "amazonpho", "pubmed", "coauthorph")
        methods = (
            "mlp_distill_gcn_xh_lcss_small_loss",
            "mlp_distill_gcn_xh_lcss_small_loss_shuffled",
        )
        for method in methods:
            for dataset in datasets:
                conf = load_conf(None, method, dataset)
                self.assertEqual(conf.gb["student_kd_weight"], 1.0)
                self.assertEqual(conf.gb["student_robust_weight"], 1.0)
                self.assertEqual(conf.gb["student_forget_rate"], "auto")
                self.assertIn("embedding_path", conf.gb)
                self.assertIn("teacher_embedding_path", conf.gb)
                self.assertEqual(conf.gb["teacher_target"], "ce")


if __name__ == "__main__":
    unittest.main()
