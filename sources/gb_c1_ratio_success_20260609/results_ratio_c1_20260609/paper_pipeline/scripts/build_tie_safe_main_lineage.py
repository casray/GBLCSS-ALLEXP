#!/usr/bin/env python3
"""Build the 20-cell uniform-0.4 lineage for the tie-safe main table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Sequence


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
METHODS = (
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "mlp_ce",
    "mlp_jocor",
)
SOFTDIST_METHODS = frozenset(METHODS[:2])
SEEDS = (0, 1, 2, 3, 4)
REGIME = "uniform0.4"
EXPECTED_TIE_MODE = "distribution_only"
TOLERANCE = 1e-12
GENERATION_ID = "TIE_SAFE_MAIN_LINEAGE_UNIFORM0P4_20260727"
GENERATOR_VERSION = "1.0.0"
MANUSCRIPT_TABLE_ID = "Table 1 (same-round tie-safe four-method replacement)"
CLAIM_ID = "CLM-TIE-SAFE-MAIN-UNIFORM0P4"

METHOD_LABELS = {
    "frozengrace_gb_mlp_softdist_beta05": "GB-LCSS",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "Randomized grouping",
    "mlp_ce": "MLP+CE",
    "mlp_jocor": "JoCoR",
}
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp936"):
        try:
            return list(
                csv.DictReader(io.StringIO(payload.decode(encoding), newline=""))
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8/cp936",
        payload,
        0,
        len(payload),
        f"cannot decode CSV as UTF-8 or CP936: {path}",
    )


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("refusing to write an empty lineage")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _normalized(value) -> str:
    return str(value).strip().lower()


def _require_files(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing required input: {path}")


def _index_rows(
    rows: Sequence[dict],
    fields: Sequence[str],
) -> dict[tuple, list[dict]]:
    index: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(_normalized(row.get(field, "")) for field in fields)
        index.setdefault(key, []).append(row)
    return index


def _validate_raw_cell(
    rows: Sequence[dict],
    dataset: str,
    method: str,
) -> tuple[float, float]:
    if len(rows) != 5:
        raise ValueError(
            "raw cell must contain exactly five unique seeds "
            f"dataset={dataset} method={method}; rows={len(rows)}"
        )
    try:
        seeds = [int(row["seed"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"raw cell has invalid seed dataset={dataset} method={method}"
        ) from exc
    if len(set(seeds)) != 5 or set(seeds) != set(SEEDS):
        raise ValueError(
            "raw cell must contain exactly five unique seeds 0-4 "
            f"dataset={dataset} method={method}; observed={sorted(seeds)}"
        )
    values: list[float] = []
    for row in rows:
        if _normalized(row.get("status")) != "ok":
            raise ValueError(
                f"raw status is not ok dataset={dataset} method={method} "
                f"seed={row.get('seed')}"
            )
        try:
            value = float(row["test"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"raw test is invalid dataset={dataset} method={method}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"raw test is non-finite dataset={dataset} method={method}"
            )
        values.append(value)
        tie_mode = _normalized(row.get("tie_mode", ""))
        if method in SOFTDIST_METHODS:
            if tie_mode != EXPECTED_TIE_MODE:
                raise ValueError(
                    "SoftDist tie_mode mismatch "
                    f"dataset={dataset} method={method} seed={row['seed']} "
                    f"observed={row.get('tie_mode')!r}"
                )
        elif tie_mode not in {"", "not_applicable"}:
            raise ValueError(
                "baseline tie_mode must be empty or NOT_APPLICABLE "
                f"dataset={dataset} method={method} seed={row['seed']}"
            )
    return statistics.mean(values), statistics.stdev(values)


def _validate_aggregate_cell(
    rows: Sequence[dict],
    dataset: str,
    method: str,
    computed_mean: float,
    computed_std: float,
) -> None:
    if len(rows) != 1:
        raise ValueError(
            f"aggregate cell must be unique dataset={dataset} method={method}; "
            f"rows={len(rows)}"
        )
    row = rows[0]
    try:
        aggregate_n = int(row["n"])
        aggregate_seeds = {
            int(seed.strip())
            for seed in str(row["seeds"]).split(",")
            if seed.strip()
        }
        aggregate_mean = float(row["test_mean"])
        aggregate_std = float(row["test_sample_std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"aggregate fields are invalid dataset={dataset} method={method}"
        ) from exc
    if aggregate_n != 5 or aggregate_seeds != set(SEEDS):
        raise ValueError(
            f"aggregate coverage mismatch dataset={dataset} method={method}"
        )
    mean_error = abs(aggregate_mean - computed_mean)
    std_error = abs(aggregate_std - computed_std)
    if mean_error > TOLERANCE or std_error > TOLERANCE:
        raise ValueError(
            "aggregate mismatch "
            f"dataset={dataset} method={method} "
            f"mean_error={mean_error:.17g} std_error={std_error:.17g}"
        )


def _validate_protocol_dataset(
    rows: Sequence[dict],
    dataset: str,
) -> None:
    if len(rows) != 5:
        raise ValueError(
            f"protocol coverage mismatch dataset={dataset}; rows={len(rows)}"
        )
    try:
        seeds = [int(row["seed"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"protocol seed invalid dataset={dataset}") from exc
    if len(set(seeds)) != 5 or set(seeds) != set(SEEDS):
        raise ValueError(
            f"protocol seed coverage mismatch dataset={dataset}"
        )
    failures = [
        row
        for row in rows
        if _normalized(row.get("protocol_status")) != "pass"
    ]
    if failures:
        raise ValueError(
            f"protocol is not PASS dataset={dataset} "
            f"seeds={[row.get('seed') for row in failures]}"
        )


def _source_metadata(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_lineage(
    *,
    raw_path: Path,
    aggregate_path: Path,
    protocol_path: Path,
    environment_path: Path,
    output_path: Path,
) -> list[dict]:
    raw_path = Path(raw_path).resolve()
    aggregate_path = Path(aggregate_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    environment_path = Path(environment_path).resolve()
    output_path = Path(output_path).resolve()
    _require_files(
        (raw_path, aggregate_path, protocol_path, environment_path)
    )

    raw_rows = read_csv(raw_path)
    aggregate_rows = read_csv(aggregate_path)
    protocol_rows = read_csv(protocol_path)
    try:
        environment_payload = json.loads(
            environment_path.read_text(encoding="utf-8-sig")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"environment fingerprint is not valid JSON: {environment_path}"
        ) from exc
    if not isinstance(environment_payload, dict):
        raise ValueError("environment fingerprint must be a JSON object")

    raw_index = _index_rows(raw_rows, ("dataset", "regime", "method"))
    aggregate_index = _index_rows(
        aggregate_rows, ("dataset", "regime", "method")
    )
    protocol_index = _index_rows(protocol_rows, ("dataset", "regime"))

    source = {
        "raw": _source_metadata(raw_path),
        "aggregate": _source_metadata(aggregate_path),
        "protocol": _source_metadata(protocol_path),
        "environment": _source_metadata(environment_path),
    }
    for dataset in DATASETS:
        _validate_protocol_dataset(
            protocol_index.get((dataset, REGIME), []),
            dataset,
        )

    records: list[dict] = []
    for method in METHODS:
        for dataset in DATASETS:
            key = (dataset, REGIME, method)
            cell_raw = raw_index.get(key, [])
            computed_mean, computed_std = _validate_raw_cell(
                cell_raw, dataset, method
            )
            _validate_aggregate_cell(
                aggregate_index.get(key, []),
                dataset,
                method,
                computed_mean,
                computed_std,
            )
            tie_mode = (
                EXPECTED_TIE_MODE
                if method in SOFTDIST_METHODS
                else "NOT_APPLICABLE"
            )
            raw_filter = (
                f'dataset=="{dataset}" AND regime=="{REGIME}" AND '
                f'method=="{method}" AND seed IN (0,1,2,3,4)'
            )
            aggregate_filter = (
                f'dataset=="{dataset}" AND regime=="{REGIME}" AND '
                f'method=="{method}"'
            )
            protocol_filter = (
                f'dataset=="{dataset}" AND regime=="{REGIME}" AND '
                "seed IN (0,1,2,3,4)"
            )
            cell_id = (
                f"T1-TIE-SAFE:{METHOD_LABELS[method]}:"
                f"{DATASET_LABELS[dataset]}"
            )
            claim = (
                f"uniform0.4 five-seed test accuracy mean±sample std for "
                f"{METHOD_LABELS[method]} on {DATASET_LABELS[dataset]}"
            )
            boundary = (
                "same-round four-method uniform0.4 comparison; "
                "five downstream seeds; SoftDist cells use "
                "distribution_only; uncertainty excludes unrerun "
                "pretraining seeds; no claim beyond tested datasets, "
                "noise regime, methods, or X-only protocol"
            )
            records.append(
                {
                    "generation": GENERATION_ID,
                    "generation_id": GENERATION_ID,
                    "version": GENERATOR_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "table": MANUSCRIPT_TABLE_ID,
                    "manuscript_table_id": MANUSCRIPT_TABLE_ID,
                    "cell": cell_id,
                    "cell_id": cell_id,
                    "claim": claim,
                    "claim_id": CLAIM_ID,
                    "supersedes": (
                        "paper_final_zh_v30_information_sciences.md "
                        f"historical mixed-lineage Table 1 cell "
                        f"{METHOD_LABELS[method]} × {DATASET_LABELS[dataset]}"
                    ),
                    "dataset": dataset,
                    "regime": REGIME,
                    "method": method,
                    "tie_mode": tie_mode,
                    "seeds": "0,1,2,3,4",
                    "n": 5,
                    "metric": "test_accuracy",
                    "unit": (
                        "test accuracy (%); computed mean/std stored as "
                        "proportions"
                    ),
                    "displayed_value": (
                        f"{100.0 * computed_mean:.2f}±"
                        f"{100.0 * computed_std:.2f}"
                    ),
                    "computed_mean": computed_mean,
                    "computed_sample_std": computed_std,
                    "computed_mean_percent": 100.0 * computed_mean,
                    "computed_sample_std_percent": 100.0 * computed_std,
                    "raw_path": source["raw"]["path"],
                    "raw_size_bytes": source["raw"]["size_bytes"],
                    "raw_sha256": source["raw"]["sha256"],
                    "raw_filter_expression": raw_filter,
                    "raw_aggregation_formula": (
                        "select exact seeds 0-4; mean=arithmetic_mean(test); "
                        "std=sample_stdev(test,ddof=1); "
                        "display=100*mean±100*std formatted .2f"
                    ),
                    "raw_status": "EXACT_5_SEEDS_MATCH",
                    "aggregate_path": source["aggregate"]["path"],
                    "aggregate_size_bytes": source["aggregate"]["size_bytes"],
                    "aggregate_sha256": source["aggregate"]["sha256"],
                    "aggregate_filter_expression": aggregate_filter,
                    "aggregate_formula": (
                        "require n=5, seeds=0,1,2,3,4; compare test_mean and "
                        "test_sample_std to raw recomputation at atol=1e-12"
                    ),
                    "aggregate_status": "MATCH_TOL_1E-12",
                    "protocol_path": source["protocol"]["path"],
                    "protocol_size_bytes": source["protocol"]["size_bytes"],
                    "protocol_sha256": source["protocol"]["sha256"],
                    "protocol_filter_expression": protocol_filter,
                    "protocol_formula": (
                        "require exactly five unique protocol rows for seeds "
                        "0-4 and protocol_status==PASS for every row"
                    ),
                    "protocol_status": "PASS_5_OF_5",
                    "environment_path": source["environment"]["path"],
                    "environment_size_bytes": source["environment"][
                        "size_bytes"
                    ],
                    "environment_sha256": source["environment"]["sha256"],
                    "environment_filter_expression": "whole JSON object",
                    "environment_formula": (
                        "parse as JSON object; bind exact file bytes by SHA-256"
                    ),
                    "environment_status": "HASHED_JSON_VALID",
                    "lineage_status": "CLOSED",
                    "boundary": boundary,
                    "claim_boundary": boundary,
                }
            )

    if len(records) != 20:
        raise AssertionError(f"expected 20 lineage cells, found {len(records)}")
    if any(row["lineage_status"] != "CLOSED" for row in records):
        raise AssertionError("all 20 lineage cells must be CLOSED")
    write_csv(output_path, records)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_lineage(
        raw_path=args.raw,
        aggregate_path=args.aggregate,
        protocol_path=args.protocol_audit,
        environment_path=args.environment,
        output_path=args.output,
    )
    print(f"WROTE {args.output} rows={len(rows)} closed=20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
