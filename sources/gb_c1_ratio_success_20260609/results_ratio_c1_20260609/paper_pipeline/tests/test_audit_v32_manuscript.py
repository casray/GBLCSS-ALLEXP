import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_v32_manuscript.py"
)
MANUSCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper_final_zh_v32_submission_candidate_20260729.md"
)


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_v32_manuscript_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AuditV32ManuscriptTest(unittest.TestCase):
    def test_real_v32_has_zero_errors(self):
        auditor = load_auditor()
        errors = auditor.audit_manuscript(MANUSCRIPT_PATH)
        self.assertEqual(errors, [])

    def test_rejects_legacy_claim_and_pubmed_in_body(self):
        auditor = load_auditor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(
                "# 稿件\n\n旧结论为 14/15。\n\n"
                "正文 PubMed 结果。\n\n# 补充材料\n",
                encoding="utf-8",
            )
            errors = auditor.audit_manuscript(path)
            self.assertTrue(any("14/15" in item for item in errors))
            self.assertTrue(any("PubMed" in item for item in errors))

    def test_rejects_missing_resource_identity(self):
        auditor = load_auditor()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.md"
            path.write_text(
                "# 稿件\n\n25/25 24/25 23/25 distribution_only\n\n"
                "# 补充材料\n\nPubMed 边界案例",
                encoding="utf-8",
            )
            errors = auditor.audit_manuscript(path)
            self.assertTrue(any("临时适配实现" in item for item in errors))
            self.assertTrue(any("PosteL" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
