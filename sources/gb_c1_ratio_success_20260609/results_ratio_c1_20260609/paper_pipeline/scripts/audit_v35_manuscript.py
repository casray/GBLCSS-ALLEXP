#!/usr/bin/env python3
"""Audit the v35 manuscript for evidence, scope, and citation regressions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANUSCRIPT = (
    PROJECT_ROOT / "paper_final_zh_v35_cold_review_revised_20260729.md"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "v35_evidence_registry_20260729.csv"
)
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parents[1]

BANNED = (
    "采用未包含并列安全分支",
    "该补充实验同样早于并列安全分支",
    "旧目标规则",
    "PubMed 边界",
    "55.59±1.50",
    "57.15±0.84",
    "55.21±2.34",
    "71.90±0.88",
    "75.53±0.87",
    "59.40±1.73",
    "80.99±1.02",
    "| kNN 共识教师 |",
    "实际墙钟时间、显存和吞吐量",
    "C:\\Users\\",
    "/Users/",
    "；",
)

REQUIRED = (
    "固定图上的转导式",
    "25/25",
    "24/25",
    "23/25",
    "distribution_only",
    "56.28±1.54",
    "53.52±2.71",
    "57.49±0.76",
    "55.24±2.04",
    "57.21±1.19",
    "40.50±0.51",
    "2–4 张训练票",
    "s_{\\min}=10",
    "0.48、0.17 和 0.46",
    "PosteL",
    "ALS",
    "SALS",
    "10.1609/aaai.v40i26.39321",
    "10.1137/1.9781611977653.ch7",
    "arXiv:2112.00499",
    "附录 I 关键实现参数",
    "v35_evidence_registry_20260729.csv",
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
            errors.append(f"required evidence or scope statement missing: {phrase}")

    if "# 补充材料" not in text:
        errors.append("supplement is missing")
        return errors

    body, supplement = text.split("# 补充材料", 1)
    if "PubMed" not in supplement:
        errors.append("supplementary PubMed row is missing")
    if "PubMed 边界" in body or "PubMed 的边界" in body:
        errors.append("PubMed is singled out as a special boundary in the main text")

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

    expected_references = set(range(1, 63))
    if references != expected_references:
        errors.append(
            "reference numbering is not contiguous 1-62: "
            f"missing={sorted(expected_references - references)}, "
            f"extra={sorted(references - expected_references)}"
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
    duplicate_ids = sorted(
        evidence_id
        for evidence_id in set(evidence_ids)
        if evidence_ids.count(evidence_id) > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate evidence_id values: {duplicate_ids}")

    for row in rows:
        evidence_id = row["evidence_id"]
        if row["implementation_status"].startswith("deferred"):
            if row["headline_allowed"] != "0":
                errors.append(
                    f"{evidence_id}: deferred evidence cannot be headline evidence"
                )

        expected_hash = row["source_sha256"].strip()
        expected_count = row["row_count"].strip()
        if expected_hash.lower() in {"", "not frozen"}:
            if row["headline_allowed"] != "0":
                errors.append(
                    f"{evidence_id}: unfrozen evidence cannot be headline evidence"
                )
            continue

        source = Path(row["source_path"])
        if not source.is_absolute():
            source = workspace_root / source
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
    print("v35 manuscript and evidence-registry audit: 0 errors")


if __name__ == "__main__":
    main()
