import importlib
import unittest

from utils.tools import load_conf


class MLP57XOnlyBallTeacherTests(unittest.TestCase):
    def test_predictor_is_x_input_ball_teacher(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        predictor = module.mlp_distill_softdist_x_Predictor

        self.assertFalse(predictor.teacher_use_xh)
        self.assertTrue(predictor.use_soft_targets)
        self.assertFalse(predictor.skip_ball_targets)
        self.assertEqual(predictor.teacher_target, "soft_target")

    def test_standard_runner_registers_predictor(self):
        runner = importlib.import_module("scripts.run_standard_downstream")
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")

        self.assertEqual(
            runner.PREDICTOR_MODULES["mlp_distill_softdist_x"],
            "predictor.FrozenGraceGB_GCN_Predictor",
        )
        self.assertIs(
            runner.predictor_class_for("mlp_distill_softdist_x"),
            module.mlp_distill_softdist_x_Predictor,
        )

    def test_configs_change_only_teacher_input_identity(self):
        for dataset in ["cora", "citeseer", "corafull"]:
            with self.subTest(dataset=dataset):
                x_conf = load_conf(None, "mlp_distill_softdist_x", dataset)
                xh_conf = load_conf(None, "mlp_distill_softdist_xh", dataset)

                self.assertEqual(x_conf.model["method"], "mlp_distill_softdist_x")
                self.assertFalse(x_conf.gb["teacher_use_xh"])
                self.assertTrue(xh_conf.gb["teacher_use_xh"])
                self.assertEqual(x_conf.gb["teacher_target"], "soft_target")
                self.assertAlmostEqual(float(x_conf.gb["target_beta"]), 0.0)
                self.assertNotIn("use_xh_features", x_conf.model)

                for key, value in xh_conf.model.items():
                    if key != "method":
                        self.assertEqual(x_conf.model[key], value)
                for key, value in xh_conf.training.items():
                    self.assertEqual(x_conf.training[key], value)
                for key, value in xh_conf.gb.items():
                    if key != "teacher_use_xh":
                        self.assertEqual(x_conf.gb[key], value)


if __name__ == "__main__":
    unittest.main()
