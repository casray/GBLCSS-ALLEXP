import importlib
import unittest

from utils.tools import load_conf


REAL = "mlp_distill_formal_lcss_x"
SHUFFLED = "mlp_distill_formal_lcss_x_shuffled"


class MLP58FormalLCSSXTeacherTests(unittest.TestCase):
    def test_predictors_are_registered_x_input_soft_target_teachers(self):
        module = importlib.import_module("predictor.FrozenGraceGB_GCN_Predictor")
        runner = importlib.import_module("scripts.run_standard_downstream")

        for method in [REAL, SHUFFLED]:
            with self.subTest(method=method):
                predictor = getattr(module, f"{method}_Predictor")
                self.assertFalse(predictor.teacher_use_xh)
                self.assertEqual(predictor.teacher_target, "soft_target")
                self.assertEqual(
                    runner.PREDICTOR_MODULES[method],
                    "predictor.FrozenGraceGB_GCN_Predictor",
                )

    def test_configs_use_the_formal_target_and_only_differ_by_random_control(self):
        for dataset in ["cora", "citeseer", "corafull"]:
            with self.subTest(dataset=dataset):
                real = load_conf(None, REAL, dataset)
                shuffled = load_conf(None, SHUFFLED, dataset)

                for conf, method, random_control in [
                    (real, REAL, False),
                    (shuffled, SHUFFLED, True),
                ]:
                    self.assertEqual(conf.model["method"], method)
                    self.assertFalse(conf.gb["teacher_use_xh"])
                    self.assertEqual(conf.gb["teacher_target"], "soft_target")
                    self.assertAlmostEqual(float(conf.gb["target_beta"]), 0.5)
                    self.assertTrue(conf.gb["leave_one_out"])
                    self.assertEqual(conf.gb["tie_mode"], "distribution_only")
                    self.assertEqual(bool(conf.gb["random_control"]), random_control)
                    self.assertNotIn("use_xh_features", conf.model)

                for section in ["training", "dataset"]:
                    self.assertEqual(getattr(real, section), getattr(shuffled, section))
                for key, value in real.model.items():
                    if key != "method":
                        self.assertEqual(value, shuffled.model[key])
                for key, value in real.gb.items():
                    if key != "random_control":
                        self.assertEqual(value, shuffled.gb[key])


if __name__ == "__main__":
    unittest.main()
