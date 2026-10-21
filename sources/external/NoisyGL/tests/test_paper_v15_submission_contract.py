from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "docs" / "paper_final_zh_v15_submission_candidate_20260716.md"


def _paper() -> str:
    return MANUSCRIPT.read_text(encoding="utf-8")


def test_post_freeze_results_are_not_mislabeled_as_held_out():
    paper = _paper()
    exact_scope = "在 β 与全部构造参数冻结后完成全噪声谱评测的三个数据集"

    assert paper.count(exact_scope) >= 3
    assert "held-out" not in paper
    assert "留出数据集" not in paper


def test_confirmatory_and_descriptive_counts_are_reported_together():
    paper = _paper()

    for count in ("14/15", "13/15", "24/25", "23/25"):
        assert paper.count(count) >= 3
    assert "参数冻结后的条件评测" in paper
    assert "开发筛选后的描述性结果" in paper


def test_noise_protocol_and_loo_decision_are_explicit():
    paper = _paper()

    assert "P_{ij}=\\rho/(C-1)" in paper
    assert "i\\rightarrow(i-1)\\bmod C" in paper
    assert "q_v\\sim\\mathcal{N}(\\rho,0.1^2)" in paper
    assert "本文保留预注册的默认目标，不以 LOO 事后替换主方法" in paper


def test_grace_seed_variance_result_and_remaining_scope_are_reported():
    paper = _paper()

    assert "三预训练种子×三下游种子" in paper
    assert "0.53、0.07 与 0.44 个百分点" in paper
    assert "11.43、11.61 与 4.73 个百分点" in paper
    assert "0.247--0.365" in paper
    assert "主表五种子误差条仍条件于单份 H" in paper
    assert "Coauthor-CS 与 Amazon-Photo 仍缺少" in paper
    assert "三个核心数据集还缺少可重建 H" not in paper


def test_html_builder_accepts_explicit_v15_source_and_output(tmp_path: Path):
    script = REPO_ROOT / "scripts" / "build_paper_html_embedded_20260706.py"
    output = tmp_path / "paper-v15.html"

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
    assert "三预训练种子×三下游种子" in rendered
    assert "0.53、0.07 与 0.44 个百分点" in rendered
    assert "data:image/png;base64," in rendered
    assert "paper_assets_20260706" not in rendered
