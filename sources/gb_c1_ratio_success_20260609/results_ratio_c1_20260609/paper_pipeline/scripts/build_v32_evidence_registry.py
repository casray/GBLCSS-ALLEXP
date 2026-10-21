#!/usr/bin/env python3
"""Build the evidence registry used by the v32 manuscript.

The registry separates current tie-safe evidence from historical diagnostics
and provisional baseline ports. It intentionally does not rank the pending
GSDN reimplementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parents[1]
NOISYGL_ROOT = EXPERIMENT_ROOT / "external" / "NoisyGL"

DEFAULT_TIE_SAFE_PATH = (
    NOISYGL_ROOT
    / "runs"
    / "tie_safe_full_grid_20260727_v2_immutable"
    / "raw_combined.csv"
)
DEFAULT_NEAREST_PATH = (
    NOISYGL_ROOT
    / "runs"
    / "nearest_work_full_grid_20260728"
    / "five_method_raw_combined.csv"
)
if not DEFAULT_NEAREST_PATH.is_file():
    DEFAULT_NEAREST_PATH = (
        NOISYGL_ROOT
        / "runs"
        / "nearest_work_full_grid_20260728"
        / "five_method_evidence.csv"
    )
if not DEFAULT_NEAREST_PATH.is_file():
    DEFAULT_NEAREST_PATH = (
        NOISYGL_ROOT
        / "runs"
        / "nearest_work_full_grid_20260728"
        / "raw_combined.csv"
    )

DEFAULT_OLD_GSDN_PATHS = (
    NOISYGL_ROOT
    / "runs"
    / "mlp14_graphfree_family_20260703"
    / "mlp14_combined_raw.csv",
    NOISYGL_ROOT
    / "runs"
    / "mlp21_dataset_replacement_20260710"
    / "mlp21_raw.csv",
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "00_evidence"
    / "v32_evidence_registry.csv"
)

TIE_SAFE_METHODS = (
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "mlp_ce",
    "mlp_jocor",
)
RECENT_METHODS = (
    "gsdn_faithful_core",
    "postel_mlp_noipl_rm",
    "als_mlp_rm",
    "sals_mlp_adapted",
)
HEADLINE_RECENT_METHODS = (
    "postel_mlp_noipl_rm",
    "als_mlp_rm",
    "sals_mlp_adapted",
)

FIELDNAMES = (
    "evidence_id",
    "manuscript_role",
    "source_path",
    "source_sha256",
    "method",
    "dataset_scope",
    "noise_scope",
    "tie_mode",
    "protocol",
    "row_count",
    "headline_allowed",
    "implementation_status",
    "limitations",
)

HISTORICAL_SOURCES = (
    (
        "HIST_LOO",
        "mechanism_ablation",
        NOISYGL_ROOT
        / "runs"
        / "mlp36_review_defense_20260715"
        / "mlp36_raw.csv",
        "loo_and_smoothing_ablation",
        "Historical mechanism evidence; not a current headline comparison.",
    ),
    (
        "HIST_KNN",
        "grouping_ablation",
        NOISYGL_ROOT
        / "runs"
        / "mlp11_k_sensitivity_20260702"
        / "mlp11_combined_raw.csv",
        "knn_grouping_ablation",
        "Historical grouping diagnostic; kNN is not a current main-method claim.",
    ),
    (
        "HIST_H_ACCESS",
        "resource_reference",
        NOISYGL_ROOT
        / "runs"
        / "grace_h_ablation_20260704"
        / "grace_h_ablation_uniform0p4_seeds_0_1_2_3_4.csv",
        "h_access_reference",
        "Uses H at inference and is not an X-only same-resource comparison.",
    ),
    (
        "HIST_DISTILL",
        "distillation_extension",
        NOISYGL_ROOT
        / "runs"
        / "submission_integrity_v12_20260712"
        / "table4_recomputed.csv",
        "consensus_teacher_distillation",
        "Historical extension evidence; not a formal comparison against GSDN.",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    for root, prefix in (
        (PROJECT_ROOT.resolve(), "."),
        (NOISYGL_ROOT.resolve(), "external/NoisyGL"),
    ):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return str(Path(prefix) / relative).replace("\\", "/")
    return path.name


def unique_values(rows: Iterable[dict[str, str]], field: str) -> list[str]:
    return sorted(
        {
            str(row.get(field, "")).strip()
            for row in rows
            if str(row.get(field, "")).strip()
        }
    )


def protocol_text(rows: Sequence[dict[str, str]]) -> str:
    split = "percent 0.2/0.1/0.7"
    if rows and all(
        key in rows[0]
        for key in ("train_percent", "val_percent", "test_percent")
    ):
        split = (
            "percent "
            f"{rows[0].get('train_percent', '')}/"
            f"{rows[0].get('val_percent', '')}/"
            f"{rows[0].get('test_percent', '')}"
        )
    seeds = unique_values(rows, "seed")
    return (
        f"{split}; train-label noise; clean validation; transductive; "
        f"{len(seeds)} paired seeds"
    )


def validate_tie_safe(rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("tie-safe evidence is empty")
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError("non-ok row in current tie-safe evidence")
        method = row.get("method", "")
        if method.startswith("frozengrace_gb_"):
            if row.get("tie_mode") != "distribution_only":
                raise ValueError(
                    "legacy tie_mode cannot enter current evidence registry"
                )


def method_registry_row(
    *,
    evidence_id: str,
    role: str,
    source_path: Path,
    method: str,
    rows: Sequence[dict[str, str]],
    headline_allowed: bool,
    implementation_status: str,
    limitations: str,
    tie_mode: str = "n/a",
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "manuscript_role": role,
        "source_path": display_path(source_path),
        "source_sha256": sha256_file(source_path),
        "method": method,
        "dataset_scope": ",".join(unique_values(rows, "dataset")),
        "noise_scope": ",".join(unique_values(rows, "noise_scope")) or "train",
        "tie_mode": tie_mode,
        "protocol": protocol_text(rows),
        "row_count": str(len(rows)),
        "headline_allowed": "1" if headline_allowed else "0",
        "implementation_status": implementation_status,
        "limitations": limitations,
    }


def historical_registry_rows() -> list[dict[str, str]]:
    rows = []
    for evidence_id, role, path, method, limitations in HISTORICAL_SOURCES:
        if not path.is_file():
            continue
        source_rows = read_csv(path)
        rows.append(
            method_registry_row(
                evidence_id=evidence_id,
                role=role,
                source_path=path,
                method=method,
                rows=source_rows,
                headline_allowed=False,
                implementation_status="historical diagnostic",
                limitations=limitations,
            )
        )
    return rows


def build_registry(
    *,
    tie_safe_path: Path,
    nearest_path: Path,
    old_gsdn_paths: Sequence[Path] = (),
) -> list[dict[str, str]]:
    tie_safe_path = Path(tie_safe_path)
    nearest_path = Path(nearest_path)
    tie_rows = read_csv(tie_safe_path)
    nearest_rows = read_csv(nearest_path)
    validate_tie_safe(tie_rows)

    registry = []
    tie_counts = Counter(row.get("method", "") for row in tie_rows)
    for method in TIE_SAFE_METHODS:
        method_rows = [row for row in tie_rows if row.get("method") == method]
        if not method_rows:
            raise ValueError(f"missing tie-safe method: {method}")
        registry.append(
            method_registry_row(
                evidence_id=f"CURRENT_TIE_SAFE_{method}",
                role="current_target_only",
                source_path=tie_safe_path,
                method=method,
                rows=method_rows,
                headline_allowed=True,
                implementation_status="current tie-safe implementation",
                limitations=(
                    "Headline use is limited to the frozen five-dataset, "
                    "five-regime grid."
                ),
                tie_mode=(
                    "distribution_only"
                    if method.startswith("frozengrace_gb_")
                    else "n/a"
                ),
            )
        )
    if sum(tie_counts[method] for method in TIE_SAFE_METHODS) != len(tie_rows):
        raise ValueError("unexpected methods in tie-safe current evidence")

    for method in RECENT_METHODS:
        method_rows = [
            row for row in nearest_rows if row.get("method") == method
        ]
        if not method_rows:
            raise ValueError(f"missing recent-work method: {method}")
        is_headline = method in HEADLINE_RECENT_METHODS
        if method == "gsdn_faithful_core":
            role = "excluded_pending_reimplementation"
            status = "superseded equation-grounded adaptation"
            limitations = (
                "Excluded from v32 claims pending user reimplementation; "
                "pending user reimplementation must be evaluated separately."
            )
        else:
            role = "current_graph_assisted_xonly"
            status = "common-protocol adapted implementation"
            limitations = (
                "Adapted to the project MLP and protocol; not an official "
                "end-to-end reproduction."
            )
        registry.append(
            method_registry_row(
                evidence_id=f"RECENT_{method}",
                role=role,
                source_path=nearest_path,
                method=method,
                rows=method_rows,
                headline_allowed=is_headline,
                implementation_status=status,
                limitations=limitations,
            )
        )

    old_gsdn_rows_combined = []
    old_gsdn_sources = []
    for path in old_gsdn_paths:
        path = Path(path)
        if not path.is_file():
            continue
        method_rows = [
            row for row in read_csv(path) if row.get("method") == "gsdn"
        ]
        if method_rows:
            old_gsdn_rows_combined.extend(method_rows)
            old_gsdn_sources.append(path)
    if old_gsdn_rows_combined:
        source = old_gsdn_sources[0]
        registry.append(
            method_registry_row(
                evidence_id="PROVISIONAL_OLD_GSDN",
                role="provisional_graph_assisted_xonly",
                source_path=source,
                method="gsdn",
                rows=old_gsdn_rows_combined,
                headline_allowed=False,
                implementation_status="provisional adapted implementation",
                limitations=(
                    "Temporary baseline requested by the user. It is neither "
                    "an official nor the final faithful GSDN implementation."
                ),
            )
        )

    registry.extend(historical_registry_rows())
    return registry


def write_registry(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(materialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tie-safe",
        type=Path,
        default=DEFAULT_TIE_SAFE_PATH,
    )
    parser.add_argument(
        "--nearest",
        type=Path,
        default=DEFAULT_NEAREST_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_registry(
        tie_safe_path=args.tie_safe,
        nearest_path=args.nearest,
        old_gsdn_paths=DEFAULT_OLD_GSDN_PATHS,
    )
    write_registry(args.output, rows)
    print(f"wrote {len(rows)} evidence records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
