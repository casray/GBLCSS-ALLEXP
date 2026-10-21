import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("lint_manuscript.py")
SPEC = importlib.util.spec_from_file_location("lint_manuscript", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManuscriptLintTest(unittest.TestCase):
    def test_flags_defensive_and_internal_language(self):
        text = (
            "本文并不声称该方法适用于所有场景，而是只关注当前协议。"
            "MLP-24 的内部实验进一步证明了这一点。"
        )
        codes = {issue["code"] for issue in MODULE.lint_text(text)}
        self.assertIn("defensive-negation", codes)
        self.assertIn("internal-experiment-id", codes)

    def test_accepts_direct_scientific_statement(self):
        text = (
            "在 uniform 0.4 设置下，GB-LCSS 在五个数据集上取得最高平均精度。"
            "其优势主要出现在 GRACE 表征能够形成稳定局部共识的数据集上。"
        )
        self.assertEqual([], MODULE.lint_text(text))


if __name__ == "__main__":
    unittest.main()
