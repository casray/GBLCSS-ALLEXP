#!/usr/bin/env python3
"""Audit the MLP-59 manuscript amendment against its result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOISYGL = PROJECT_ROOT.parents[1] / "external" / "NoisyGL"
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "paper_final_zh_v60_mlp59_extension_20260814.md"
DEFAULT_RUN_DIR = NOISYGL / "runs" / "mlp59_final_extensions_20260814"
V59 = PROJECT_ROOT / "paper_final_zh_v59_pretranslation_revised_20260814.md"
V59_SHA256 = "0A785668C3D02311C2C4BD4F46BC18242353F8799C5779E601E599472F23FAD8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _formatted(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}±{100.0 * std:.2f}"


def _audit_markdown_tables(text: str) -> list[str]:
    errors: list[str] = []
    block: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines() + [""], start=1):
        if line.startswith("|"):
            block.append((number, line))
            continue
        if block:
            counts = {row.count("|") for _, row in block}
            if len(counts) != 1:
                errors.append(
                    f"table starting at line {block[0][0]} has inconsistent pipe counts: {sorted(counts)}"
                )
            block = []
    return errors


def audit(manuscript: Path, run_dir: Path) -> list[str]:
    errors: list[str] = []
    text = manuscript.read_text(encoding="utf-8")
    v59_text = V59.read_text(encoding="utf-8")

    if _sha256(V59) != V59_SHA256:
        errors.append("frozen v59 hash changed")
    old_refs = v59_text.split("## 参考文献", 1)[1].split("## 附录 A", 1)[0]
    new_refs = text.split("## 参考文献", 1)[1].split("## 附录 A", 1)[0]
    if old_refs != new_refs:
        errors.append("reference section changed relative to v59")
    if "；" in text:
        errors.append("Chinese semicolon remains in manuscript")
    if "| NRGNN 教师" in text:
        errors.append("obsolete NRGNN teacher row remains")
    if "表 4 的六个设置" in text:
        errors.append("stale six-cell Table 4 statement remains")

    completeness = pd.read_csv(run_dir / "mlp59_completeness.csv")
    if not completeness["passed"].all():
        errors.append("MLP-59 completeness audit contains failures")
    gates = pd.read_csv(run_dir / "mlp59_gate_decisions.csv")
    teacher_gate = gates[gates["phase"] == "teacher"]
    full_gate = gates[gates["phase"] == "boundary_full_spectrum"]
    if len(teacher_gate) != 1 or not bool(teacher_gate.iloc[0]["passed"]):
        errors.append("teacher gate did not pass")
    if len(full_gate) != 1 or bool(full_gate.iloc[0]["passed"]):
        errors.append("CiteSeerFull full-spectrum gate status is not the expected failure")

    aggregate = pd.read_csv(run_dir / "mlp59_aggregate.csv")
    teacher_methods = [
        "frozengrace_gb_mlp_softdist_beta05_loo",
        "mlp_distill_gcn",
        "ffg2m_reimplemented",
        "krd_adapted",
        "vqgraph_adapted",
        "mlp_distill_formal_lcss_x",
    ]
    for dataset in ["coauthorcs", "amazonpho"]:
        for rate in [0.4, 0.6]:
            for method in teacher_methods:
                rows = aggregate[
                    (aggregate["dataset"] == dataset)
                    & (aggregate["method"] == method)
                    & (aggregate["noise_type"] == "uniform")
                    & (aggregate["noise_rate"] == rate)
                ].drop_duplicates(["dataset", "method", "noise_type", "noise_rate"])
                if len(rows) != 1:
                    errors.append(f"missing unique aggregate row: {dataset}/{rate}/{method}")
                    continue
                value = _formatted(float(rows.iloc[0]["mean"]), float(rows.iloc[0]["std"]))
                if value not in text:
                    errors.append(f"Table 4 value missing from manuscript: {dataset}/{rate}/{method}={value}")

    required_boundary_values = [
        "72.60±0.66", "76.20±0.56", "56.26±2.61", "65.17±1.42",
        "60.94±1.08", "93.13±0.25", "70.71±0.65", "42.45±3.08",
        "55.02±4.07", "61.49±3.01", "-5.24", "+14.25", "-4.20",
    ]
    for value in required_boundary_values:
        if value not in text:
            errors.append(f"Appendix I value missing: {value}")

    reference_numbers = [int(value) for value in re.findall(r"^\[(\d+)\]", new_refs, re.MULTILINE)]
    if reference_numbers != list(range(1, max(reference_numbers) + 1)):
        errors.append("reference list is not contiguous")
    citation_numbers: set[int] = set()
    body = text.split("## 参考文献", 1)[0]
    for content in re.findall(r"\[([0-9,\-\s]+)\]", body):
        for part in content.split(","):
            part = part.strip()
            if "--" in part:
                start, end = (int(value) for value in part.split("--", 1))
                citation_numbers.update(range(start, end + 1))
            elif part:
                citation_numbers.add(int(part))
    if citation_numbers and max(citation_numbers) > max(reference_numbers):
        errors.append("citation exceeds reference-list range")

    tags = [int(value) for value in re.findall(r"\\tag\{(\d+)\}", text)]
    if len(tags) != len(set(tags)) or tags != list(range(1, max(tags) + 1)):
        errors.append("equation tags are duplicated or non-contiguous")
    errors.extend(_audit_markdown_tables(text))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    errors = audit(args.manuscript, args.run_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("v60 manuscript and MLP-59 evidence audit: 0 errors")


if __name__ == "__main__":
    main()
