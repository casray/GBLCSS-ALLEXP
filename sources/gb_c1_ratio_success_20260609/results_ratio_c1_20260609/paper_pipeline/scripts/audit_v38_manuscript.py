#!/usr/bin/env python3
"""Audit the formal LOO v38 manuscript and its frozen evidence registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "paper_final_zh_v38_loo_main_20260730.md"
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v38_evidence_registry_20260730.csv"
)
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

REQUIRED = (
    "固定预算粒球局部共识软监督",
    "Noisy-Label Transductive Node Classification with Graph-Free Inference",
    "leave_one_out=true",
    "tie_mode=distribution_only",
    "p_{j,-v}",
    "q_{j,-v}",
    "\\alpha_{j,-v}",
    "T_{j,-v}=T_j\\setminus\\{v\\}",
    "25/25、25/25 和 24/25",
    "24/25 个设置中取得七个同骨干基线中的最高或并列最高均值",
    "8.69、7.74 和 3.46",
    "57.49±0.76",
    "55.24±2.04",
    "37.39±0.51",
    "90.71±0.59",
    "82.12±1.08",
    "4/5 个数据集上距最佳替代实现不超过 1 个百分点",
    "在干净监督下，额外目标平滑并非普遍有利",
    "含自票前驱目标",
    "该诊断采用含自票的并列安全前驱目标",
)

BANNED = (
    "7.841",
    "7.251",
    "2.963",
    "7.84、7.25",
    "LOO 消融",
    "LOO 变体",
    "默认方法允许自票",
    "采用未包含并列安全分支",
    "旧目标规则",
    "min_index",
    "self-inclusive",
    "C:\\Users\\",
    "/Users/",
    "；",
)

REGISTRY_COLUMNS = {
    "evidence_id",
    "manuscript_role",
    "manuscript_location",
    "source_path",
    "source_sha256",
    "method_or_bundle",
    "dataset_scope",
    "noise_scope",
    "tie_mode",
    "protocol",
    "row_count",
    "headline_allowed",
    "implementation_status",
    "limitations",
}


def _section(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0]


def audit_manuscript(path: Path) -> list[str]:
    if not path.is_file():
        return [f"manuscript missing: {path}"]

    text = path.read_text(encoding="utf-8")
    errors: list[str] = []

    for phrase in REQUIRED:
        if phrase not in text:
            errors.append(f"required formal-LOO statement missing: {phrase}")
    for phrase in BANNED:
        if phrase in text:
            errors.append(f"banned legacy or private phrase present: {phrase}")

    tags = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", text)]
    expected_tags = list(range(1, 24))
    if tags != expected_tags:
        errors.append(f"formula tags are not exactly 1-23 in order: {tags}")

    appendix_b = _section(text, "## 附录 B 配对统计", "## 附录 C")
    b_rows = re.findall(
        r"^\| (?:Cora|CiteSeer|CoraFull|Coauthor-CS|Amazon-Photo) \|",
        appendix_b,
        flags=re.MULTILINE,
    )
    if len(b_rows) != 75:
        errors.append(f"Appendix B must contain 75 paired-stat rows, found {len(b_rows)}")

    appendix_h = _section(text, "## 附录 H", "## 附录 I")
    h_rows = re.findall(
        r"^\| (?:Cora|CiteSeer|CoraFull|Coauthor-CS|Amazon-Photo) \|",
        appendix_h,
        flags=re.MULTILINE,
    )
    if len(h_rows) != 25:
        errors.append(f"Appendix H must contain 25 spectrum rows, found {len(h_rows)}")

    if "# 补充材料" not in text:
        errors.append("supplement is missing")
    else:
        supplement = text.split("# 补充材料", 1)[1]
        for appendix in tuple("ABCDEFGHI"):
            if f"附录 {appendix}" not in supplement:
                errors.append(f"missing Appendix {appendix}")

    cited = {
        int(value)
        for group in re.findall(r"\[([0-9,\-\s]+)\]", text)
        for value in re.findall(r"\d+", group)
    }
    references = {
        int(value)
        for value in re.findall(r"^\[(\d+)\]\s", text, flags=re.MULTILINE)
    }
    missing_references = sorted(cited - references)
    if missing_references:
        errors.append(f"citation numbers without references: {missing_references}")
    if references:
        expected_references = set(range(1, max(references) + 1))
        if references != expected_references:
            errors.append(
                "reference numbering is not contiguous: "
                f"missing={sorted(expected_references - references)}"
            )

    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_registry(
    path: Path,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
) -> list[str]:
    if not path.is_file():
        return [f"evidence registry missing: {path}"]

    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(REGISTRY_COLUMNS - columns)
        if missing_columns:
            return [f"registry columns missing: {missing_columns}"]
        rows = list(reader)

    evidence_ids = [row["evidence_id"] for row in rows]
    duplicates = sorted(
        evidence_id
        for evidence_id in set(evidence_ids)
        if evidence_ids.count(evidence_id) > 1
    )
    if duplicates:
        errors.append(f"duplicate evidence_id values: {duplicates}")

    required_ids = {
        "V38_LOO_FULLSPECTRUM",
        "V38_LOO_GROUPING",
        "V38_GROUPING_COST",
        "V38_CLEAN_CONTROL",
        "V38_FORMAL_STATS",
    }
    missing_ids = sorted(required_ids - set(evidence_ids))
    if missing_ids:
        errors.append(f"formal v38 evidence entries missing: {missing_ids}")

    for row in rows:
        evidence_id = row["evidence_id"]
        expected_hash = row["source_sha256"].strip()
        expected_count = row["row_count"].strip()
        if expected_hash.lower() in {"", "not frozen"}:
            if row["headline_allowed"] != "0":
                errors.append(f"{evidence_id}: unfrozen evidence cannot support headlines")
            continue

        source = Path(row["source_path"])
        if not source.is_absolute():
            project_candidate = PROJECT_ROOT / source
            workspace_candidate = workspace_root / source
            source = (
                project_candidate
                if project_candidate.is_file()
                else workspace_candidate
            )
        if not source.is_file():
            errors.append(f"{evidence_id}: source missing: {source}")
            continue

        actual_hash = _sha256(source)
        if actual_hash != expected_hash.upper():
            errors.append(
                f"{evidence_id}: SHA256 mismatch: "
                f"expected={expected_hash.upper()} actual={actual_hash}"
            )
        if not expected_count.isdigit():
            errors.append(f"{evidence_id}: non-numeric row_count: {expected_count}")
            continue
        actual_count = _csv_row_count(source)
        if actual_count != int(expected_count):
            errors.append(
                f"{evidence_id}: row_count mismatch: "
                f"expected={expected_count} actual={actual_count}"
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = audit_manuscript(args.manuscript)
    errors.extend(audit_registry(args.registry))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)
    print("v38 formal-LOO manuscript and evidence audit: 0 errors")


if __name__ == "__main__":
    main()
