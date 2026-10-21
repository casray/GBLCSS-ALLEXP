#!/usr/bin/env python3
"""Audit the v32 manuscript for evidence-lineage and scope regressions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT = (
    PROJECT_ROOT / "paper_final_zh_v32_submission_candidate_20260729.md"
)

BANNED = (
    "14/15",
    "+7.16",
    "+7.83",
    "+2.87",
    "沿用 NoisyGL 协议",
    "无需干净标签",
    "全面优于 GSDN",
    "GB-LCSS 全面优于 GSDN",
    "表 0",
    "C:\\Users\\",
    "/Users/",
)

REQUIRED = (
    "25/25",
    "24/25",
    "23/25",
    "7.84",
    "7.25",
    "2.96",
    "5.20",
    "7.22",
    "4.20",
    "PosteL",
    "ALS",
    "SALS",
    "临时适配实现",
    "distribution_only",
    "transductive",
    "验证标签噪声敏感性",
    "# 补充材料",
    "附录 E PubMed 边界案例",
    "Hoeffding",
)


def audit_manuscript(path: Path) -> list[str]:
    if not path.is_file():
        return [f"manuscript missing: {path}"]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    for phrase in BANNED:
        if phrase in text:
            errors.append(f"banned phrase present: {phrase}")

    for phrase in REQUIRED:
        if phrase not in text:
            errors.append(f"required evidence or identity missing: {phrase}")

    if "# 补充材料" not in text:
        return errors
    body, supplement = text.split("# 补充材料", 1)
    if "PubMed" in body:
        errors.append("PubMed appears before the supplementary boundary section")
    if "PubMed" not in supplement:
        errors.append("PubMed appendix evidence is missing")

    table_a_start = text.find("### 5.1 同资源抗噪方法")
    table_b_start = text.find("### 5.2 结构辅助 X-only 方法")
    if table_a_start < 0 or table_b_start < 0:
        errors.append("resource-separated Table A/Table B is missing")
    elif "GSDN" in text[table_a_start:table_b_start]:
        errors.append("GSDN leaked into same-resource Table A")

    if "gsdn_faithful_core" in body:
        errors.append("pending GSDN implementation appears in the manuscript body")

    for appendix in ("附录 A", "附录 B", "附录 C", "附录 D", "附录 E", "附录 F", "附录 G"):
        if appendix not in supplement:
            errors.append(f"missing {appendix}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = audit_manuscript(args.manuscript)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)
    print("v32 manuscript audit: 0 errors")


if __name__ == "__main__":
    main()
