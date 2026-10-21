import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "paper_pipeline" / "scripts" / "audit_v39_manuscript.py"
MANUSCRIPT = ROOT / "paper_final_zh_v39_nrgnn_teacher_20260730.md"
REGISTRY = (
    ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v39_evidence_registry_20260730.csv"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_v39_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AuditV39ManuscriptTest(unittest.TestCase):
    def test_real_v39_manuscript_and_registry_have_zero_errors(self):
        auditor = load_auditor()
        self.assertEqual(auditor.audit_v39_manuscript(MANUSCRIPT), [])
        self.assertEqual(auditor.audit_v39_registry(REGISTRY), [])

    def test_rejects_obsolete_five_arm_teacher_text(self):
        auditor = load_auditor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(
                "**表 5** 共识教师蒸馏的五臂比较。\n",
                encoding="utf-8",
            )
            errors = auditor.audit_v39_manuscript(path)
            self.assertTrue(
                any("obsolete v38 teacher statement" in error for error in errors)
            )


if __name__ == "__main__":
    unittest.main()
