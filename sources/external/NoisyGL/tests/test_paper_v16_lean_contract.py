from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "docs" / "paper_final_zh_v16_lean_20260720.md"


def _paper() -> str:
    return MANUSCRIPT.read_text(encoding="utf-8")


def test_post_freeze_scope_and_dual_counts_survive_slimming():
    paper = _paper()
    exact_scope = "在 β 与全部构造参数冻结后完成全噪声谱评测的"

    assert paper.count(exact_scope) >= 3
    for count in ("14/15", "13/15", "24/25", "23/25"):
        assert paper.count(count) >= 3
    assert "held-out" not in paper
    assert "留出数据集" not in paper


def test_protocol_loo_and_fixed_h_disclosures_survive_slimming():
    paper = _paper()

    assert "P_{ij}=\\rho/(C-1)" in paper
    assert "i\\rightarrow(i-1)\\bmod C" in paper
    assert "q_v\\sim\\mathcal{N}(\\rho,0.1^2)" in paper
    assert "本文保留预注册的默认目标，不以 LOO 事后替换主方法" in paper
    assert "主表五种子误差条仍条件于单份 H" in paper
    assert "Coauthor-CS 和 Amazon-Photo 也缺少可重建原始 H" in paper


def test_submission_draft_excludes_internal_advisor_figure_section():
    paper = _paper()

    assert "## 导师审阅附录图精选" not in paper
    assert "**附录 G** 表征空间可视化" in paper


def test_html_builder_accepts_v16_source_and_output(tmp_path: Path):
    script = REPO_ROOT / "scripts" / "build_paper_html_embedded_20260706.py"
    output = tmp_path / "paper-v16.html"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--src",
            str(MANUSCRIPT),
            "--out",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rendered = output.read_text(encoding="utf-8")
    assert "14/15" in rendered
    assert "主表五种子误差条仍条件于单份 H" in rendered
    assert "图 1 LCSS 从训练期图结构到仅特征推理的整体流程" in rendered
    assert "data:image/png;base64," in rendered
    assert "paper_assets_20260706" not in rendered
