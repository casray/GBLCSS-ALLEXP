#!/usr/bin/env python3
"""Compare two tie-safe full-grid runs at seed-row resolution."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Sequence


KEY_FIELDS = ("dataset", "regime", "method", "seed")
EXACT_FIELDS = (
    "train",
    "valid",
    "test",
    "split_hash",
    "noise_hash",
    "modified_count",
    "modified_train_count",
    "modified_val_count",
    "modified_test_count",
    "feature_embedding_hash",
    "ball_embedding_hash",
    "ball_assignment_hash",
    "tie_mode",
    "tie_ball_count",
    "tie_active_ball_count",
    "tie_affected_node_count",
    "tie_affected_alpha_sum",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp936"):
        try:
            return list(
                csv.DictReader(io.StringIO(payload.decode(encoding), newline=""))
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"cannot decode CSV as UTF-8 or CP936: {path}")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _key(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row.get("dataset", "")).strip().lower(),
        str(row.get("regime", "")).strip().lower(),
        str(row.get("method", "")).strip().lower(),
        int(row.get("seed", -1)),
    )


def _index(rows: Sequence[dict]) -> tuple[dict[tuple, dict], set[tuple]]:
    indexed: dict[tuple, dict] = {}
    duplicates: set[tuple] = set()
    for row in rows:
        key = _key(row)
        if key in indexed:
            duplicates.add(key)
        indexed[key] = row
    return indexed, duplicates


def compare_rows(
    first_rows: Sequence[dict], second_rows: Sequence[dict]
) -> tuple[list[dict], dict]:
    first, first_duplicates = _index(first_rows)
    second, second_duplicates = _index(second_rows)
    all_keys = sorted(set(first) | set(second))
    audit: list[dict] = []
    matched = mismatched = missing_first = missing_second = 0
    for key in all_keys:
        base = dict(zip(KEY_FIELDS, key))
        left = first.get(key)
        right = second.get(key)
        if left is None:
            missing_first += 1
            audit.append(
                {
                    **base,
                    "row_status": "MISSING_FIRST",
                    "mismatched_fields": "",
                }
            )
            continue
        if right is None:
            missing_second += 1
            audit.append(
                {
                    **base,
                    "row_status": "MISSING_SECOND",
                    "mismatched_fields": "",
                }
            )
            continue
        mismatched_fields = [
            field
            for field in EXACT_FIELDS
            if str(left.get(field, "")) != str(right.get(field, ""))
        ]
        if mismatched_fields:
            mismatched += 1
            status = "MISMATCH"
        else:
            matched += 1
            status = "MATCH"
        audit.append(
            {
                **base,
                "row_status": status,
                "mismatched_fields": ",".join(mismatched_fields),
                **{
                    f"first_{field}": left.get(field, "")
                    for field in mismatched_fields
                },
                **{
                    f"second_{field}": right.get(field, "")
                    for field in mismatched_fields
                },
            }
        )
    status = (
        "PASS"
        if not (
            first_duplicates
            or second_duplicates
            or mismatched
            or missing_first
            or missing_second
        )
        else "FAIL"
    )
    summary = {
        "status": status,
        "first_rows": len(first_rows),
        "second_rows": len(second_rows),
        "first_unique_keys": len(first),
        "second_unique_keys": len(second),
        "first_duplicate_keys": len(first_duplicates),
        "second_duplicate_keys": len(second_duplicates),
        "matched_rows": matched,
        "mismatched_rows": mismatched,
        "missing_in_first": missing_first,
        "missing_in_second": missing_second,
    }
    return audit, summary


def render_report(summary: dict, first: Path, second: Path) -> str:
    lines = [
        "# Tie-safe full-grid 重复运行核验",
        "",
        f"- 状态：**{summary['status']}**",
        f"- 第一次运行：`{first}`",
        f"- 第二次运行：`{second}`",
        f"- 行数：{summary['first_rows']} / {summary['second_rows']}",
        f"- 唯一键：{summary['first_unique_keys']} / {summary['second_unique_keys']}",
        f"- 完全匹配：{summary['matched_rows']}",
        f"- 字段漂移：{summary['mismatched_rows']}",
        f"- 第一次缺失：{summary['missing_in_first']}",
        f"- 第二次缺失：{summary['missing_in_second']}",
        f"- 重复键：{summary['first_duplicate_keys']} / {summary['second_duplicate_keys']}",
        "",
        "比较字段包括 train/valid/test、split/noise hash、污染计数、"
        "embedding/assignment hash、tie mode 和 tie diagnostics。",
        "两次使用相同 seed；PASS 表示确定性重复，不增加独立样本量。",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    first_path = args.first_run / "raw_combined.csv"
    second_path = args.second_run / "raw_combined.csv"
    audit, summary = compare_rows(read_csv(first_path), read_csv(second_path))
    if (
        summary["first_rows"] != args.expected_rows
        or summary["second_rows"] != args.expected_rows
    ):
        summary["status"] = "FAIL"
        summary["expected_rows"] = args.expected_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "repeat_row_audit.csv", audit)
    (args.output_dir / "repeat_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "repeat_report.md").write_text(
        render_report(summary, args.first_run, args.second_run),
        encoding="utf-8",
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
