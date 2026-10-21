#!/usr/bin/env python3
"""Audit the v39 manuscript and robust-teacher evidence amendment."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_v38_manuscript import audit_manuscript, audit_registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "paper_final_zh_v39_nrgnn_teacher_20260730.md"
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v39_evidence_registry_20260730.csv"
)

REQUIRED_V39 = (
    "图教师蒸馏的七臂比较",
    "| NRGNN 教师 | 81.07±0.55 | **70.87±1.06** | 57.71±0.58 | **74.81±1.17** | 66.62±1.27 | **53.55±1.00** |",
    "| NRGNN[\\(X\\Vert H\\)] 教师 | 80.37±1.29 | 70.43±0.98 | 56.77±0.59 | 74.50±2.25 | **67.75±1.79** | 52.78±0.62 |",
    "成熟鲁棒图教师在高噪声下能够提供比球式共识教师更有效的蒸馏目标",
    "CoraFull 上约为 5 个百分点",
    "60/60 条 NRGNN 鲁棒教师蒸馏记录",
)

BANNED_V39 = (
    "共识教师蒸馏的五臂比较",
    "共识教师蒸馏进一步缩小了仅属性学生与图模型之间的精度差距",
)


def audit_v39_manuscript(path: Path) -> list[str]:
    errors = audit_manuscript(path)
    if not path.is_file():
        return errors
    text = path.read_text(encoding="utf-8")
    for phrase in REQUIRED_V39:
        if phrase not in text:
            errors.append(f"required v39 robust-teacher statement missing: {phrase}")
    for phrase in BANNED_V39:
        if phrase in text:
            errors.append(f"obsolete v38 teacher statement present: {phrase}")

    section = text.split("**表 5**", 1)[1].split("## 7 结论", 1)[0]
    teacher_rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ")
        and not line.startswith("| 教师 ")
        and not line.startswith("|---")
    ]
    if len(teacher_rows) != 7:
        errors.append(f"Table 5 must contain seven teacher rows, found {len(teacher_rows)}")
    return errors


def audit_v39_registry(path: Path) -> list[str]:
    errors = audit_registry(path)
    if not path.is_file():
        return errors
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = {row["evidence_id"] for row in rows}
    if "V39_NRGNN_TEACHER_DISTILL" not in ids:
        errors.append("V39 robust-teacher evidence entry is missing")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    errors = audit_v39_manuscript(args.manuscript)
    errors.extend(audit_v39_registry(args.registry))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)
    print("v39 manuscript and robust-teacher evidence audit: 0 errors")


if __name__ == "__main__":
    main()
