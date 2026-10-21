#!/usr/bin/env python3
"""Run the frozen nearest-work comparison grid."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence


class Regime(NamedTuple):
    key: str
    noise_type: str
    noise_rate: float


class CommandSpec(NamedTuple):
    dataset: str
    regime: Regime
    methods: tuple[str, ...]
    expected_rows: int
    raw_path: Path
    immutable_path: Path
    stdout_path: Path
    stderr_path: Path
    command: tuple[str, ...]


DATASETS = ("cora", "citeseer", "coauthorcs", "corafull", "amazonpho")
REGIMES = (
    Regime("uniform0.2", "uniform", 0.2),
    Regime("uniform0.4", "uniform", 0.4),
    Regime("uniform0.6", "uniform", 0.6),
    Regime("pair0.4", "pair", 0.4),
    Regime("instance0.4", "instance", 0.4),
)
SEEDS = (0, 1, 2, 3, 4)
METHODS = (
    "gsdn_faithful_core",
    "postel_mlp_noipl_rm",
    "als_mlp_rm",
    "sals_mlp_adapted",
)
IMPLEMENTATION_STATUS = {
    "gsdn_faithful_core": "equation-grounded common-protocol adaptation",
    "postel_mlp_noipl_rm": "official_code_faithful_target_project_mlp",
    "als_mlp_rm": "mechanism_faithful_project_mlp_adaptation",
    "sals_mlp_adapted": "mechanism_faithful_project_mlp_adaptation",
}
CAPABILITY_REGISTRY = {
    method: {
        "requires_embedding": False,
        "training_uses_graph": True,
        "inference_uses_graph": False,
        "comparison_tier": "same_resource_x_only",
        "implementation_status": IMPLEMENTATION_STATUS[method],
    }
    for method in METHODS
}
COMMON_REQUIRED_FIELDS = (
    "dataset",
    "regime",
    "method",
    "seed",
    "noise_type",
    "noise_rate",
    "noise_scope",
    "split_type",
    "train_percent",
    "val_percent",
    "test_percent",
    "split_hash",
    "noise_hash",
    "config_id",
    "modified_count",
    "modified_train_count",
    "modified_val_count",
    "modified_test_count",
    "status",
    "valid",
    "test",
)
PROTOCOL_EQUAL_FIELDS = (
    "noise_type",
    "noise_rate",
    "noise_scope",
    "split_type",
    "train_percent",
    "val_percent",
    "test_percent",
    "split_hash",
    "noise_hash",
    "modified_count",
    "modified_train_count",
    "modified_val_count",
    "modified_test_count",
)
OURS_V2_METHOD = "frozengrace_gb_mlp_softdist_beta05"
COMPARISON_SOURCE = "tie_safe_full_grid_v2_immutable"
INDEPENDENT_EVIDENCE_METHODS = (OURS_V2_METHOD,) + METHODS[1:]
FIVE_METHODS = (OURS_V2_METHOD,) + METHODS
ZERO_TOLERANCE = 1e-12
ALPHA = 0.05
T_CRITICAL_95_DF4 = 2.7764451051977987


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    payload = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp936"):
        try:
            return list(
                csv.DictReader(
                    io.StringIO(payload.decode(encoding), newline="")
                )
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"cannot decode CSV as UTF-8 or CP936: {path}")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = [dict(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(materialized)


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalized(value) -> str:
    return str(value).strip().lower()


def _nonempty(row: dict, field: str) -> bool:
    return str(row.get(field, "")).strip() != ""


def _planned_keys() -> list[tuple[str, str, str, int]]:
    return [
        (dataset, regime.key, method, seed)
        for dataset in DATASETS
        for regime in REGIMES
        for method in METHODS
        for seed in SEEDS
    ]


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment {name!r} in {path}")


def validate_capability_registry(noisygl_root: Path) -> dict[str, dict]:
    standard_runner = (
        Path(noisygl_root) / "scripts" / "run_standard_downstream.py"
    )
    observed_all = _literal_assignment(standard_runner, "METHOD_CAPABILITIES")
    missing = [method for method in METHODS if method not in observed_all]
    observed = {
        method: observed_all[method]
        for method in METHODS
        if method in observed_all
    }
    if missing or observed != CAPABILITY_REGISTRY:
        raise ValueError(
            "capability registry mismatch: "
            f"missing={missing} expected={CAPABILITY_REGISTRY!r} "
            f"observed={observed!r}"
        )
    return observed


def _write_or_verify_input_snapshots(
    sources: Sequence[Path],
    output_dir: Path,
) -> str:
    sources = [Path(source).resolve() for source in sources]
    output_dir = Path(output_dir).resolve()
    snapshot_dir = output_dir / "input_snapshots"
    manifest_path = output_dir / "input_snapshot_manifest.csv"
    expected = []
    for index, source in enumerate(sources, start=1):
        if not source.is_file():
            raise FileNotFoundError(f"missing snapshot input: {source}")
        target = snapshot_dir / f"{index:02d}_{source.name}"
        expected.append(
            {
                "source_path": str(source),
                "snapshot_path": str(target.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )

    if manifest_path.exists():
        observed = read_csv(manifest_path)
        if len(observed) != len(expected):
            raise ValueError(
                "input snapshot mismatch: "
                f"expected rows={len(expected)} observed={len(observed)}"
            )
        expected_paths = {
            Path(row["snapshot_path"]).resolve() for row in expected
        }
        observed_paths = (
            {path.resolve() for path in snapshot_dir.iterdir()}
            if snapshot_dir.exists()
            else set()
        )
        if observed_paths != expected_paths:
            raise ValueError(
                "input snapshot mismatch: snapshot file set differs"
            )
        for index, (expected_row, observed_row) in enumerate(
            zip(expected, observed),
            start=1,
        ):
            for field in (
                "source_path",
                "snapshot_path",
                "size_bytes",
                "sha256",
            ):
                if str(observed_row.get(field, "")) != str(expected_row[field]):
                    raise ValueError(
                        "input snapshot mismatch: "
                        f"row={index} field={field} "
                        f"expected={expected_row[field]!r} "
                        f"observed={observed_row.get(field, '')!r}"
                    )
            snapshot = Path(expected_row["snapshot_path"])
            if sha256_file(snapshot) != expected_row["sha256"]:
                raise ValueError(
                    f"input snapshot mismatch: snapshot bytes changed: {snapshot}"
                )
        return "verified_existing"

    if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
        raise ValueError(
            "input snapshot mismatch: snapshots exist without manifest"
        )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source, row in zip(sources, expected):
        Path(row["snapshot_path"]).write_bytes(source.read_bytes())
    write_csv(manifest_path, expected)
    return "created"


def deterministic_training_env(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "PYTHONHASHSEED": "0",
        }
    )
    return env


def _collect_environment_fingerprint(
    python: Path,
    noisygl_root: Path,
) -> dict:
    env = deterministic_training_env()
    payload = {
        "captured_at_utc": utc_now(),
        "orchestrator_python": sys.executable,
        "orchestrator_python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "training_python": str(Path(python).resolve()),
        "noisygl_root": str(Path(noisygl_root).resolve()),
        "training_subprocess_environment_flags": {
            key: env[key]
            for key in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_DYNAMIC",
                "MKL_DYNAMIC",
            )
        },
    }
    try:
        payload["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        scoped_diff = subprocess.run(
            ["git", "diff", "--binary", "--", str(noisygl_root)],
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            timeout=60,
        ).stdout
        payload["git_scoped_diff_sha256"] = sha256_bytes(scoped_diff)
    except Exception as exc:
        payload["git_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _write_environment_fingerprint(
    python: Path,
    noisygl_root: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir).resolve()
    payload = _collect_environment_fingerprint(python, noisygl_root)
    initial_path = output_dir / "environment_fingerprint.json"
    if not initial_path.exists():
        payload["lineage_role"] = "initial"
        write_json(initial_path, payload)
        return initial_path
    payload["lineage_role"] = "resume"
    payload["initial_fingerprint_path"] = str(initial_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    resume_path = (
        output_dir / f"environment_fingerprint.resume_{timestamp}.json"
    )
    write_json(resume_path, payload)
    return resume_path


def _preserve_immutable_raw(source: Path, immutable: Path) -> str:
    source = Path(source)
    immutable = Path(immutable)
    source_bytes = source.read_bytes()
    if immutable.exists():
        if immutable.read_bytes() != source_bytes:
            raise ValueError(
                "immutable raw mismatch: existing bytes differ from "
                f"working output: {immutable}"
            )
        return "verified_existing"
    immutable.parent.mkdir(parents=True, exist_ok=True)
    immutable.write_bytes(source_bytes)
    return "created"


def _row_key(row: dict) -> tuple[str, str, str, int]:
    return (
        _normalized(row.get("dataset", "")),
        _normalized(row.get("regime", "")),
        _normalized(row.get("method", "")),
        int(row.get("seed", -1)),
    )


def _validate_native_row(
    row: dict,
    spec: CommandSpec,
) -> tuple[str, int]:
    method = _normalized(row.get("method", ""))
    try:
        seed = int(row.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("protocol drift: invalid seed") from exc
    key = (spec.dataset, spec.regime.key, method, seed)
    if _normalized(row.get("dataset", "")) != spec.dataset:
        raise ValueError(f"protocol drift: dataset mismatch at {key}")
    if method not in METHODS or seed not in SEEDS:
        raise ValueError(f"protocol drift: method/seed mismatch at {key}")
    if _normalized(row.get("noise_type", "")) != spec.regime.noise_type:
        raise ValueError(f"protocol drift: noise_type mismatch at {key}")
    try:
        observed_rate = float(row.get("noise_rate", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"protocol drift: invalid noise_rate at {key}") from exc
    if not math.isclose(
        observed_rate,
        spec.regime.noise_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"protocol drift: noise_rate mismatch at {key}")
    if _normalized(row.get("noise_scope", "")) != "train":
        raise ValueError(f"protocol drift: noise_scope mismatch at {key}")
    for field in (
        "split_hash",
        "noise_hash",
        "config_id",
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
        "status",
        "valid",
        "test",
    ):
        if not _nonempty(row, field):
            raise ValueError(f"protocol drift: empty {field} at {key}")
    if _normalized(row["status"]) != "ok":
        raise ValueError(f"protocol drift: non-ok status at {key}")
    try:
        metrics = (float(row["valid"]), float(row["test"]))
        counts = {
            field: int(row[field])
            for field in (
                "modified_count",
                "modified_train_count",
                "modified_val_count",
                "modified_test_count",
            )
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"protocol drift: invalid numeric field at {key}") from exc
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in metrics
    ):
        raise ValueError(f"protocol drift: invalid metric at {key}")
    if any(value < 0 for value in counts.values()):
        raise ValueError(f"protocol drift: negative mutation count at {key}")
    if counts["modified_count"] != counts["modified_train_count"]:
        raise ValueError(f"protocol drift: non-train mutation count at {key}")
    if counts["modified_val_count"] != 0:
        raise ValueError(f"protocol drift: validation labels modified at {key}")
    if counts["modified_test_count"] != 0:
        raise ValueError(f"protocol drift: test labels modified at {key}")
    return method, seed


def _verify_or_set_text(
    row: dict,
    field: str,
    expected: str,
    key: tuple,
) -> None:
    if _nonempty(row, field):
        if _normalized(row[field]) != _normalized(expected):
            raise ValueError(f"protocol drift: {field} mismatch at {key}")
    else:
        row[field] = expected


def _verify_or_set_float(
    row: dict,
    field: str,
    expected: float,
    key: tuple,
) -> None:
    if _nonempty(row, field):
        try:
            observed = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"protocol drift: invalid {field} at {key}"
            ) from exc
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"protocol drift: {field} mismatch at {key}")
    else:
        row[field] = expected


def _finalize_command_output(spec: CommandSpec) -> str:
    immutable_status = _preserve_immutable_raw(
        spec.raw_path,
        spec.immutable_path,
    )
    rows = read_csv(spec.immutable_path)
    if len(rows) != spec.expected_rows:
        raise ValueError(
            "protocol drift: immutable row count mismatch "
            f"expected={spec.expected_rows} observed={len(rows)}"
        )
    observed: set[tuple[str, int]] = set()
    for row in rows:
        method, seed = _validate_native_row(row, spec)
        native_key = (method, seed)
        if native_key in observed:
            raise ValueError(f"protocol drift: duplicate native row {native_key}")
        observed.add(native_key)
        key = (spec.dataset, spec.regime.key, method, seed)
        _verify_or_set_text(row, "regime", spec.regime.key, key)
        _verify_or_set_text(row, "split_type", "percent", key)
        _verify_or_set_float(row, "train_percent", 0.2, key)
        _verify_or_set_float(row, "val_percent", 0.1, key)
        _verify_or_set_float(row, "test_percent", 0.7, key)
    expected = {
        (method, seed)
        for method in METHODS
        for seed in SEEDS
    }
    if observed != expected:
        raise ValueError(
            "protocol drift: immutable method/seed coverage mismatch"
        )
    write_csv(spec.raw_path, rows)
    return immutable_status


def immutable_raw_file_complete(spec: CommandSpec) -> bool:
    if not spec.immutable_path.exists():
        return False
    try:
        rows = read_csv(spec.immutable_path)
        if len(rows) != spec.expected_rows:
            return False
        observed = {
            _validate_native_row(row, spec)
            for row in rows
        }
    except (OSError, UnicodeError, ValueError):
        return False
    expected = {
        (method, seed)
        for method in METHODS
        for seed in SEEDS
    }
    return observed == expected


def raw_file_complete(spec: CommandSpec) -> bool:
    if not spec.raw_path.exists():
        return False
    try:
        rows = read_csv(spec.raw_path)
        if len(rows) != spec.expected_rows:
            return False
        observed = set()
        for row in rows:
            if any(not _nonempty(row, field) for field in COMMON_REQUIRED_FIELDS):
                return False
            method, seed = _validate_native_row(row, spec)
            key = (spec.dataset, spec.regime.key, method, seed)
            if _row_key(row) != key:
                return False
            if _normalized(row["split_type"]) != "percent":
                return False
            expected_ratios = {
                "train_percent": 0.2,
                "val_percent": 0.1,
                "test_percent": 0.7,
            }
            if any(
                not math.isclose(
                    float(row[field]),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for field, expected in expected_ratios.items()
            ):
                return False
            if (method, seed) in observed:
                return False
            observed.add((method, seed))
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    return len(observed) == spec.expected_rows


def _validate_protocol_group(
    dataset: str,
    regime: Regime,
    seed: int,
    group: Sequence[dict],
) -> None:
    methods = {_normalized(row.get("method", "")) for row in group}
    if len(group) != len(METHODS) or methods != set(METHODS):
        raise ValueError(
            "protocol drift: method coverage mismatch "
            f"dataset={dataset} regime={regime.key} seed={seed}"
        )
    reference = group[0]
    for row in group:
        key = (dataset, regime.key, row.get("method"), seed)
        if any(not _nonempty(row, field) for field in COMMON_REQUIRED_FIELDS):
            raise ValueError(f"protocol drift: empty required field at {key}")
        if _row_key(row) != (
            dataset,
            regime.key,
            _normalized(row["method"]),
            seed,
        ):
            raise ValueError(f"protocol drift: row key mismatch at {key}")
        spec = CommandSpec(
            dataset=dataset,
            regime=regime,
            methods=METHODS,
            expected_rows=len(METHODS) * len(SEEDS),
            raw_path=Path(),
            immutable_path=Path(),
            stdout_path=Path(),
            stderr_path=Path(),
            command=(),
        )
        _validate_native_row(row, spec)
        if _normalized(row["split_type"]) != "percent":
            raise ValueError(f"protocol drift: split_type at {key}")
        for field, expected in {
            "train_percent": 0.2,
            "val_percent": 0.1,
            "test_percent": 0.7,
        }.items():
            if not math.isclose(
                float(row[field]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"protocol drift: {field} at {key}")
        for field in PROTOCOL_EQUAL_FIELDS:
            if str(row.get(field, "")) != str(reference.get(field, "")):
                raise ValueError(
                    "protocol drift: cross-method mismatch "
                    f"field={field} at {key}"
                )


def protocol_audit(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        try:
            key = (
                _normalized(row.get("dataset", "")),
                _normalized(row.get("regime", "")),
                int(row.get("seed", -1)),
            )
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    output = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for seed in SEEDS:
                group = grouped.get((dataset, regime.key, seed), [])
                try:
                    _validate_protocol_group(
                        dataset,
                        regime,
                        seed,
                        group,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    status, error = "FAIL", str(exc)
                else:
                    status, error = "PASS", ""
                output.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "seed": seed,
                        "expected_method_rows": len(METHODS),
                        "observed_method_rows": len(group),
                        "protocol_status": status,
                        "error": error,
                    }
                )
    return output


def build_completeness(rows: Sequence[dict]) -> list[dict]:
    observed: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        try:
            observed.setdefault(_row_key(row), []).append(row)
        except (TypeError, ValueError):
            continue
    output = []
    for key in _planned_keys():
        matches = observed.get(key, [])
        if not matches:
            status, raw_status = "missing", ""
        elif len(matches) > 1:
            status = "duplicate"
            raw_status = "|".join(str(row.get("status", "")) for row in matches)
        else:
            raw_status = _normalized(matches[0].get("status", ""))
            status = "ok" if raw_status == "ok" else "failed"
        output.append(
            {
                "dataset": key[0],
                "regime": key[1],
                "method": key[2],
                "seed": key[3],
                "completion_status": status,
                "raw_status": raw_status,
            }
        )
    return output


def validate_protocol_consistency(rows: Sequence[dict]) -> None:
    expected = set(_planned_keys())
    counts: dict[tuple[str, str, str, int], int] = {}
    for row in rows:
        try:
            key = _row_key(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"protocol drift: invalid row key: {exc}") from exc
        counts[key] = counts.get(key, 0) + 1
    missing = expected - set(counts)
    extra = set(counts) - expected
    duplicates = {key for key, count in counts.items() if count != 1}
    if missing or extra or duplicates:
        raise ValueError(
            "protocol drift: 500-row coverage mismatch "
            f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]} "
            f"duplicates={sorted(duplicates)[:3]}"
        )
    failures = [
        row
        for row in protocol_audit(rows)
        if row["protocol_status"] != "PASS"
    ]
    if failures:
        raise ValueError(failures[0]["error"])


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _continued_fraction_beta(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 3e-14:
        d = 3e-14
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        m2 = 2 * iteration
        aa = (
            iteration
            * (b - iteration)
            * x
            / ((qam + m2) * (a + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < 3e-14:
            d = 3e-14
        c = 1.0 + aa / c
        if abs(c) < 3e-14:
            c = 3e-14
        d = 1.0 / d
        result *= d * c
        aa = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + m2) * (qap + m2))
        )
        d = 1.0 + aa * d
        if abs(d) < 3e-14:
            d = 3e-14
        c = 1.0 + aa / c
        if abs(c) < 3e-14:
            c = 3e-14
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= 3e-14:
            break
    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_fraction_beta(a, b, x) / a
    return 1.0 - front * _continued_fraction_beta(b, a, 1.0 - x) / b


def paired_t_test(values: Sequence[float]) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    mean = statistics.mean(values)
    std = _sample_std(values)
    if std <= ZERO_TOLERANCE:
        if abs(mean) <= ZERO_TOLERANCE:
            return 0.0, 1.0
        return math.copysign(math.inf, mean), 0.0
    t_stat = mean / (std / math.sqrt(len(values)))
    df = len(values) - 1
    x = df / (df + t_stat * t_stat)
    p_value = _regularized_beta(x, df / 2.0, 0.5)
    return t_stat, min(max(p_value, 0.0), 1.0)


def _apply_holm(rows: list[dict]) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda pair: float(pair[1]["p_value_two_sided"]),
    )
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            (total - rank + 1) * float(row["p_value_two_sided"]),
        )
        running = max(running, adjusted)
        rows[index]["holm_rank"] = rank
        rows[index]["holm_family_size"] = total
        rows[index]["holm_adjusted_p"] = running
        rows[index]["holm_significant_0p05"] = int(running < ALPHA)


def summarize_rows(rows: Sequence[dict]) -> list[dict]:
    validate_protocol_consistency(rows)
    lookup = {_row_key(row): row for row in rows}
    aggregate = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for method in METHODS:
                values = [
                    float(
                        lookup[
                            (dataset, regime.key, method, seed)
                        ]["test"]
                    )
                    for seed in SEEDS
                ]
                aggregate.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "noise_type": regime.noise_type,
                        "noise_rate": regime.noise_rate,
                        "method": method,
                        "n": len(values),
                        "seeds": ",".join(str(seed) for seed in SEEDS),
                        "test_mean": statistics.mean(values),
                        "test_sample_std": _sample_std(values),
                    }
                )
    return aggregate


def _frozen_ours_lookup(
    ours_rows: Sequence[dict],
) -> dict[tuple[str, str, int], dict]:
    expected = {
        (dataset, regime.key, seed)
        for dataset in DATASETS
        for regime in REGIMES
        for seed in SEEDS
    }
    lookup = {}
    for row in ours_rows:
        key = (
            _normalized(row.get("dataset", "")),
            _normalized(row.get("regime", "")),
            int(row.get("seed", -1)),
        )
        if _normalized(row.get("method", "")) != OURS_V2_METHOD:
            raise ValueError(
                "cross-directory lineage mismatch: unexpected ours method"
            )
        if key in lookup:
            raise ValueError(
                "cross-directory lineage mismatch: duplicate ours key"
            )
        if _normalized(row.get("status", "")) != "ok":
            raise ValueError(
                "cross-directory lineage mismatch: non-ok ours row"
            )
        if not _nonempty(row, "split_hash") or not _nonempty(row, "noise_hash"):
            raise ValueError(
                "cross-directory lineage mismatch: missing ours hash"
            )
        lookup[key] = row
    if set(lookup) != expected:
        raise ValueError(
            "cross-directory lineage mismatch: ours coverage is not 125 rows"
        )
    return lookup


def summarize_vs_frozen_ours(
    rows: Sequence[dict],
    ours_rows: Sequence[dict],
) -> tuple[list[dict], list[dict]]:
    validate_protocol_consistency(rows)
    nearest = {_row_key(row): row for row in rows}
    ours = _frozen_ours_lookup(ours_rows)
    deltas = []
    tests = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for comparator in METHODS:
                values = []
                for seed in SEEDS:
                    comparator_row = nearest[
                        (dataset, regime.key, comparator, seed)
                    ]
                    ours_row = ours[(dataset, regime.key, seed)]
                    for field in (
                        "split_hash",
                        "noise_hash",
                        "modified_count",
                        "modified_train_count",
                        "modified_val_count",
                        "modified_test_count",
                    ):
                        if str(comparator_row.get(field, "")) != str(
                            ours_row.get(field, "")
                        ):
                            raise ValueError(
                                "cross-directory lineage mismatch: "
                                f"dataset={dataset} regime={regime.key} "
                                f"seed={seed} comparator={comparator} "
                                f"field={field}"
                            )
                    ours_test = float(ours_row["test"])
                    comparator_test = float(comparator_row["test"])
                    delta = ours_test - comparator_test
                    values.append(delta)
                    deltas.append(
                        {
                            "dataset": dataset,
                            "regime": regime.key,
                            "comparator": comparator,
                            "seed": seed,
                            "ours_method": OURS_V2_METHOD,
                            "ours_test": ours_test,
                            "comparator_test": comparator_test,
                            "delta": delta,
                            "delta_pp": 100.0 * delta,
                            "split_hash": ours_row["split_hash"],
                            "noise_hash": ours_row["noise_hash"],
                            "comparison_source": COMPARISON_SOURCE,
                        }
                    )
                mean = statistics.mean(values)
                std = _sample_std(values)
                half_width = (
                    T_CRITICAL_95_DF4 * std / math.sqrt(len(values))
                )
                t_stat, p_value = paired_t_test(values)
                tests.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "comparator": comparator,
                        "ours_method": OURS_V2_METHOD,
                        "comparison_source": COMPARISON_SOURCE,
                        "n": len(values),
                        "paired_delta_mean": mean,
                        "paired_delta_sample_std": std,
                        "paired_delta_ci95_low": mean - half_width,
                        "paired_delta_ci95_high": mean + half_width,
                        "t_statistic": t_stat,
                        "degrees_of_freedom": len(values) - 1,
                        "p_value_two_sided": p_value,
                        "positive_seed_count": sum(
                            value > ZERO_TOLERANCE for value in values
                        ),
                        "negative_seed_count": sum(
                            value < -ZERO_TOLERANCE for value in values
                        ),
                        "zero_seed_count": sum(
                            abs(value) <= ZERO_TOLERANCE for value in values
                        ),
                    }
                )
    for comparator in METHODS:
        family = [
            row for row in tests if row["comparator"] == comparator
        ]
        _apply_holm(family)
    return deltas, tests


def _command_for(
    python: Path,
    noisygl_root: Path,
    output_path: Path,
    dataset: str,
    regime: Regime,
    device: str,
) -> tuple[str, ...]:
    return (
        str(python),
        str(noisygl_root / "scripts" / "run_standard_downstream.py"),
        "--datasets",
        dataset,
        "--methods",
        ",".join(METHODS),
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--noise_type",
        regime.noise_type,
        "--noise_rate",
        str(regime.noise_rate),
        "--noise_scope",
        "train",
        "--device",
        device,
        "--split_type",
        "percent",
        "--train_percent",
        "0.2",
        "--val_percent",
        "0.1",
        "--test_percent",
        "0.7",
        "--config_overrides",
        json.dumps({}, sort_keys=True),
        "--output",
        str(output_path),
    )


def build_command_specs(
    *,
    python: Path,
    noisygl_root: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> list[CommandSpec]:
    specs = []
    for dataset in DATASETS:
        for regime in REGIMES:
            stem = f"{dataset}_{regime.key}_nearest_work_seeds_0_4"
            raw_path = output_dir / "raw" / f"{stem}.csv"
            specs.append(
                CommandSpec(
                    dataset=dataset,
                    regime=regime,
                    methods=METHODS,
                    expected_rows=len(METHODS) * len(SEEDS),
                    raw_path=raw_path,
                    immutable_path=output_dir
                    / "raw_immutable"
                    / f"{stem}.csv",
                    stdout_path=output_dir / "logs" / f"{stem}.stdout.log",
                    stderr_path=output_dir / "logs" / f"{stem}.stderr.log",
                    command=_command_for(
                        python,
                        noisygl_root,
                        raw_path,
                        dataset,
                        regime,
                        device,
                    ),
                )
            )
    return specs


def _command_output_complete(spec: CommandSpec) -> bool:
    if not raw_file_complete(spec) or not immutable_raw_file_complete(spec):
        return False
    try:
        immutable_rows = read_csv(spec.immutable_path)
        working_rows = read_csv(spec.raw_path)
        immutable = {}
        for row in immutable_rows:
            key = _validate_native_row(row, spec)
            if key in immutable:
                return False
            immutable[key] = row
        working = {}
        for row in working_rows:
            key = _validate_native_row(row, spec)
            if key in working:
                return False
            working[key] = row
        if set(immutable) != set(working):
            return False
        for key, immutable_row in immutable.items():
            working_row = working[key]
            if any(
                str(working_row.get(field, "")) != str(value)
                for field, value in immutable_row.items()
            ):
                return False
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    return True


def _validate_frozen_v2_source(path: Path) -> list[dict]:
    rows = read_csv(path)
    if len(rows) != 500:
        raise ValueError(
            "cross-directory lineage mismatch: frozen v2 raw_combined "
            f"must have 500 rows, observed={len(rows)}"
        )
    v2_methods = {
        OURS_V2_METHOD,
        "frozengrace_gb_mlp_softdist_beta05_shuffled",
        "mlp_ce",
        "mlp_jocor",
    }
    keys = set()
    for row in rows:
        try:
            key = (
                _normalized(row.get("dataset", "")),
                _normalized(row.get("regime", "")),
                _normalized(row.get("method", "")),
                int(row.get("seed", -1)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cross-directory lineage mismatch: invalid v2 key"
            ) from exc
        if key in keys:
            raise ValueError(
                "cross-directory lineage mismatch: duplicate v2 row"
            )
        keys.add(key)
        if (
            key[0] not in DATASETS
            or key[1] not in {regime.key for regime in REGIMES}
            or key[2] not in v2_methods
            or key[3] not in SEEDS
        ):
            raise ValueError(
                "cross-directory lineage mismatch: unexpected v2 key"
            )
        if _normalized(row.get("status", "")) != "ok":
            raise ValueError(
                "cross-directory lineage mismatch: non-ok v2 row"
            )
        if _normalized(row.get("noise_scope", "")) != "train":
            raise ValueError(
                "cross-directory lineage mismatch: v2 is not train-only"
            )
        if int(row.get("modified_val_count", -1)) != 0:
            raise ValueError(
                "cross-directory lineage mismatch: v2 validation mutation"
            )
        if int(row.get("modified_test_count", -1)) != 0:
            raise ValueError(
                "cross-directory lineage mismatch: v2 test mutation"
            )
    ours_rows = [
        row
        for row in rows
        if _normalized(row.get("method", "")) == OURS_V2_METHOD
    ]
    _frozen_ours_lookup(ours_rows)
    return ours_rows


def _required_input_sources(
    noisygl_root: Path,
    amendment_path: Path,
    ours_v2_raw: Path,
) -> list[Path]:
    sources = [
        Path(amendment_path),
        Path(__file__).resolve(),
        noisygl_root / "scripts" / "run_standard_downstream.py",
        noisygl_root / "predictor" / "GSDNFaithful_Predictor.py",
        noisygl_root
        / "predictor"
        / "NearestWorkSoftTargets_Predictor.py",
        noisygl_root / "utils" / "dataloader.py",
        noisygl_root / "utils" / "labelnoise.py",
        noisygl_root / "utils" / "tools.py",
    ]
    for method in METHODS:
        for dataset in DATASETS:
            sources.append(
                noisygl_root
                / "config"
                / method
                / f"{method}_{dataset}.yaml"
            )
    for dataset in DATASETS:
        sources.append(
            noisygl_root / "config" / "_dataset" / f"{dataset}.yaml"
        )
    sources.append(Path(ours_v2_raw))
    return [source.resolve() for source in sources]


def _validate_paths(
    python: Path,
    noisygl_root: Path,
    amendment_path: Path,
    ours_v2_dir: Path,
) -> tuple[Path, list[dict]]:
    ours_v2_raw = ours_v2_dir / "raw_combined.csv"
    required = {
        "python": python,
        "NoisyGL root": noisygl_root,
        "frozen amendment": amendment_path,
        "frozen v2 raw_combined": ours_v2_raw,
    }
    for label, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing required {label}: {path}")
    sources = _required_input_sources(
        noisygl_root,
        amendment_path,
        ours_v2_raw,
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"missing frozen input: {source}")
    validate_capability_registry(noisygl_root)
    ours_rows = _validate_frozen_v2_source(ours_v2_raw)
    return ours_v2_raw, ours_rows


def _write_immutable_raw_manifest(
    output_dir: Path,
    specs: Sequence[CommandSpec],
) -> None:
    rows = []
    for spec in specs:
        path = spec.immutable_path
        rows.append(
            {
                "dataset": spec.dataset,
                "regime": spec.regime.key,
                "path": str(path.resolve()),
                "exists": int(path.exists()),
                "complete": int(immutable_raw_file_complete(spec)),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    write_csv(output_dir / "immutable_raw_manifest.csv", rows)


def _collect_rows(specs: Sequence[CommandSpec]) -> list[dict]:
    rows = []
    for spec in specs:
        if not spec.raw_path.exists():
            continue
        try:
            rows.extend(read_csv(spec.raw_path))
        except (OSError, UnicodeError):
            continue
    return rows


def _protocol_value_equal(field: str, left, right) -> bool:
    if field in {
        "noise_rate",
        "train_percent",
        "val_percent",
        "test_percent",
    }:
        try:
            return math.isclose(
                float(left),
                float(right),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False
    if field in {
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
    }:
        try:
            return int(left) == int(right)
        except (TypeError, ValueError):
            return False
    return str(left) == str(right)


def _validate_independent_evidence_group(
    dataset: str,
    regime: Regime,
    seed: int,
    group: Sequence[dict],
) -> None:
    methods = {_normalized(row.get("method", "")) for row in group}
    if (
        len(group) != len(INDEPENDENT_EVIDENCE_METHODS)
        or methods != set(INDEPENDENT_EVIDENCE_METHODS)
    ):
        raise ValueError(
            "protocol drift: independent evidence method coverage mismatch "
            f"dataset={dataset} regime={regime.key} seed={seed}"
        )
    reference = group[0]
    for row in group:
        method = _normalized(row.get("method", ""))
        key = (dataset, regime.key, method, seed)
        if any(not _nonempty(row, field) for field in COMMON_REQUIRED_FIELDS):
            raise ValueError(
                "protocol drift: empty independent evidence field "
                f"at {key}"
            )
        if _row_key(row) != key:
            raise ValueError(
                f"protocol drift: independent evidence key mismatch at {key}"
            )
        if _normalized(row["noise_type"]) != regime.noise_type:
            raise ValueError(
                f"protocol drift: independent evidence noise_type at {key}"
            )
        if not _protocol_value_equal(
            "noise_rate",
            row["noise_rate"],
            regime.noise_rate,
        ):
            raise ValueError(
                f"protocol drift: independent evidence noise_rate at {key}"
            )
        if _normalized(row["noise_scope"]) != "train":
            raise ValueError(
                f"protocol drift: independent evidence noise_scope at {key}"
            )
        if _normalized(row["split_type"]) != "percent":
            raise ValueError(
                f"protocol drift: independent evidence split_type at {key}"
            )
        for field, expected in {
            "train_percent": 0.2,
            "val_percent": 0.1,
            "test_percent": 0.7,
        }.items():
            if not _protocol_value_equal(field, row[field], expected):
                raise ValueError(
                    f"protocol drift: independent evidence {field} at {key}"
                )
        try:
            valid = float(row["valid"])
            test = float(row["test"])
            counts = {
                field: int(row[field])
                for field in (
                    "modified_count",
                    "modified_train_count",
                    "modified_val_count",
                    "modified_test_count",
                )
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "protocol drift: invalid independent evidence numeric field "
                f"at {key}"
            ) from exc
        if (
            _normalized(row["status"]) != "ok"
            or not math.isfinite(valid)
            or not math.isfinite(test)
            or not 0.0 <= valid <= 1.0
            or not 0.0 <= test <= 1.0
        ):
            raise ValueError(
                f"protocol drift: invalid independent evidence result at {key}"
            )
        if (
            counts["modified_count"] != counts["modified_train_count"]
            or counts["modified_val_count"] != 0
            or counts["modified_test_count"] != 0
        ):
            raise ValueError(
                "protocol drift: independent evidence changed non-train "
                f"labels at {key}"
            )
        for field in PROTOCOL_EQUAL_FIELDS:
            if not _protocol_value_equal(
                field,
                row.get(field, ""),
                reference.get(field, ""),
            ):
                raise ValueError(
                    "protocol drift: independent evidence lineage mismatch "
                    f"field={field} at {key}"
                )


def _independent_evidence_keys() -> list[tuple[str, str, str, int]]:
    return [
        (dataset, regime.key, method, seed)
        for dataset in DATASETS
        for regime in REGIMES
        for method in INDEPENDENT_EVIDENCE_METHODS
        for seed in SEEDS
    ]


def _build_independent_evidence(
    native_rows: Sequence[dict],
    ours_rows: Sequence[dict],
) -> list[dict]:
    validate_protocol_consistency(native_rows)
    recent_rows = [
        dict(row)
        for row in native_rows
        if _normalized(row.get("method", "")) in set(METHODS[1:])
    ]
    evidence = recent_rows + [dict(row) for row in ours_rows]
    expected = set(_independent_evidence_keys())
    observed = [_row_key(row) for row in evidence]
    if (
        len(recent_rows) != 375
        or len(ours_rows) != 125
        or len(evidence) != 500
        or set(observed) != expected
        or len(set(observed)) != len(observed)
    ):
        raise ValueError(
            "protocol drift: independent evidence must contain exactly "
            "375 recent-work rows plus 125 frozen ours-v2 rows"
        )
    failures = [
        row
        for row in _independent_protocol_audit(evidence)
        if row["protocol_status"] != "PASS"
    ]
    if failures:
        raise ValueError(failures[0]["error"])
    return evidence


def _independent_completeness(rows: Sequence[dict]) -> list[dict]:
    observed: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        try:
            observed.setdefault(_row_key(row), []).append(row)
        except (TypeError, ValueError):
            continue
    output = []
    for key in _independent_evidence_keys():
        matches = observed.get(key, [])
        if not matches:
            status, raw_status = "missing", ""
        elif len(matches) > 1:
            status = "duplicate"
            raw_status = "|".join(
                str(row.get("status", "")) for row in matches
            )
        else:
            raw_status = _normalized(matches[0].get("status", ""))
            status = "ok" if raw_status == "ok" else "failed"
        output.append(
            {
                "dataset": key[0],
                "regime": key[1],
                "method": key[2],
                "seed": key[3],
                "completion_status": status,
                "raw_status": raw_status,
            }
        )
    return output


def _independent_protocol_audit(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        try:
            key = (
                _normalized(row.get("dataset", "")),
                _normalized(row.get("regime", "")),
                int(row.get("seed", -1)),
            )
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    output = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for seed in SEEDS:
                group = grouped.get((dataset, regime.key, seed), [])
                try:
                    _validate_independent_evidence_group(
                        dataset,
                        regime,
                        seed,
                        group,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    status, error = "FAIL", str(exc)
                else:
                    status, error = "PASS", ""
                output.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "seed": seed,
                        "expected_method_rows": len(
                            INDEPENDENT_EVIDENCE_METHODS
                        ),
                        "observed_method_rows": len(group),
                        "protocol_status": status,
                        "error": error,
                    }
                )
    return output


def _summarize_independent_evidence(
    rows: Sequence[dict],
) -> list[dict]:
    observed = [_row_key(row) for row in rows]
    expected = set(_independent_evidence_keys())
    if (
        len(rows) != 500
        or set(observed) != expected
        or len(set(observed)) != len(observed)
    ):
        raise ValueError(
            "protocol drift: cannot aggregate incomplete independent evidence"
        )
    failures = [
        row
        for row in _independent_protocol_audit(rows)
        if row["protocol_status"] != "PASS"
    ]
    if failures:
        raise ValueError(failures[0]["error"])
    lookup = {_row_key(row): row for row in rows}
    output = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for method in INDEPENDENT_EVIDENCE_METHODS:
                values = [
                    float(lookup[(dataset, regime.key, method, seed)]["test"])
                    for seed in SEEDS
                ]
                output.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "noise_type": regime.noise_type,
                        "noise_rate": regime.noise_rate,
                        "method": method,
                        "n": len(values),
                        "seeds": ",".join(str(seed) for seed in SEEDS),
                        "test_mean": statistics.mean(values),
                        "test_sample_std": _sample_std(values),
                    }
                )
    return output


def _build_five_method_evidence(
    native_rows: Sequence[dict],
    ours_rows: Sequence[dict],
) -> list[dict]:
    rows = [dict(row) for row in native_rows] + [
        dict(row) for row in ours_rows
    ]
    expected = {
        (dataset, regime.key, method, seed)
        for dataset in DATASETS
        for regime in REGIMES
        for method in FIVE_METHODS
        for seed in SEEDS
    }
    observed = [_row_key(row) for row in rows]
    if (
        len(rows) != 625
        or set(observed) != expected
        or len(set(observed)) != len(observed)
    ):
        raise ValueError(
            "protocol drift: five-method evidence must contain exactly "
            "500 recent-work rows plus 125 frozen ours-v2 rows"
        )
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (
            _normalized(row["dataset"]),
            _normalized(row["regime"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, []).append(row)
    for dataset in DATASETS:
        for regime in REGIMES:
            for seed in SEEDS:
                group = grouped[(dataset, regime.key, seed)]
                methods = {
                    _normalized(row.get("method", "")) for row in group
                }
                if len(group) != 5 or methods != set(FIVE_METHODS):
                    raise ValueError(
                        "protocol drift: five-method coverage mismatch "
                        f"dataset={dataset} regime={regime.key} seed={seed}"
                    )
                reference = group[0]
                for row in group:
                    for field in PROTOCOL_EQUAL_FIELDS:
                        if not _protocol_value_equal(
                            field,
                            row.get(field, ""),
                            reference.get(field, ""),
                        ):
                            raise ValueError(
                                "protocol drift: five-method lineage mismatch "
                                f"field={field} dataset={dataset} "
                                f"regime={regime.key} seed={seed}"
                            )
    return rows


def _summarize_five_method_evidence(
    rows: Sequence[dict],
) -> list[dict]:
    lookup = {_row_key(row): row for row in rows}
    output = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for method in FIVE_METHODS:
                values = [
                    float(lookup[(dataset, regime.key, method, seed)]["test"])
                    for seed in SEEDS
                ]
                output.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "noise_type": regime.noise_type,
                        "noise_rate": regime.noise_rate,
                        "method": method,
                        "n": len(values),
                        "seeds": ",".join(str(seed) for seed in SEEDS),
                        "test_mean": statistics.mean(values),
                        "test_sample_std": _sample_std(values),
                    }
                )
    if len(output) != 125:
        raise ValueError(
            "protocol drift: five-method aggregate must contain 125 rows"
        )
    return output


def _render_report(
    native_rows: Sequence[dict],
    evidence: Sequence[dict],
    five_method_evidence: Sequence[dict],
    completeness: Sequence[dict],
    audit: Sequence[dict],
    aggregate: Sequence[dict],
    five_method_aggregate: Sequence[dict],
    deltas: Sequence[dict],
    tests: Sequence[dict],
) -> str:
    completion_counts = {
        status: sum(
            row["completion_status"] == status for row in completeness
        )
        for status in ("ok", "missing", "failed", "duplicate")
    }
    lines = [
        "# Nearest-work full-grid evidence report",
        "",
        f"- Generated at (UTC): {utc_now()}",
        f"- Native recent-work rows: {len(native_rows)}/500.",
        (
            "- Independent evidence rows: "
            f"{len(evidence)}/500 (375 PosteL/ALS/SALS + 125 frozen ours-v2)."
        ),
        (
            "- Five-method manuscript rows: "
            f"{len(five_method_evidence)}/625 "
            "(500 recent-work + 125 frozen ours-v2)."
        ),
        (
            "- Completion status: "
            f"{json.dumps(completion_counts, sort_keys=True)}."
        ),
        (
            "- Protocol groups passing: "
            f"{sum(row['protocol_status'] == 'PASS' for row in audit)}/125."
        ),
        f"- Aggregate rows: {len(aggregate)}/100.",
        (
            "- Five-method aggregate rows: "
            f"{len(five_method_aggregate)}/125."
        ),
        f"- Cross-vs-LCSS paired deltas: {len(deltas)}/500.",
        f"- Cross-vs-LCSS paired tests: {len(tests)}/100.",
        "",
        "## Claim boundary",
        "",
        (
            "- This report is mechanically generated evidence. Performance "
            "claims require a complete 500-row grid, 125/125 protocol PASS, "
            "and review of effect sizes, uncertainty, and per-seed deltas."
        ),
        (
            "- The frozen ours-v2 source remains a separate read-only lineage; "
            "it is not rewritten by this runner."
        ),
        (
            "- Any missing row, failed status, duplicate key, split/noise hash "
            "mismatch, or validation/test label mutation invalidates closure."
        ),
    ]
    return "\n".join(lines) + "\n"


def summarize_output(
    output_dir: Path,
    specs: Sequence[CommandSpec],
    ours_v2_raw: Path,
    *,
    expected_ours_v2_sha256: str | None = None,
) -> int:
    incomplete = [
        f"{spec.dataset}/{spec.regime.key}"
        for spec in specs
        if not _command_output_complete(spec)
    ]
    if incomplete:
        raise ValueError(
            "protocol drift: exact native output pair is incomplete: "
            + ", ".join(incomplete[:3])
        )
    native_rows = _collect_rows(specs)
    if len(native_rows) != 500:
        raise ValueError(
            "protocol drift: native recent-work grid must have 500 rows, "
            f"observed={len(native_rows)}"
        )
    validate_protocol_consistency(native_rows)
    ours_rows = _validate_frozen_v2_source(ours_v2_raw)
    if (
        expected_ours_v2_sha256 is not None
        and sha256_file(ours_v2_raw) != expected_ours_v2_sha256
    ):
        raise ValueError(
            "cross-directory lineage mismatch: frozen ours-v2 hash changed "
            "during orchestration"
        )
    evidence = _build_independent_evidence(native_rows, ours_rows)
    completeness = _independent_completeness(evidence)
    audit = _independent_protocol_audit(evidence)
    if any(row["completion_status"] != "ok" for row in completeness):
        raise ValueError(
            "protocol drift: independent evidence completeness is not 500/500"
        )
    if any(row["protocol_status"] != "PASS" for row in audit):
        raise ValueError(
            "protocol drift: independent evidence protocol audit is not 125/125"
        )
    aggregate = _summarize_independent_evidence(evidence)
    five_method_evidence = _build_five_method_evidence(
        native_rows,
        ours_rows,
    )
    five_method_aggregate = _summarize_five_method_evidence(
        five_method_evidence
    )
    deltas, tests = summarize_vs_frozen_ours(native_rows, ours_rows)
    write_csv(output_dir / "recent_work_raw_combined.csv", native_rows)
    write_csv(output_dir / "raw_combined.csv", evidence)
    write_csv(
        output_dir / "five_method_raw_combined.csv",
        five_method_evidence,
    )
    write_csv(output_dir / "completeness.csv", completeness)
    write_csv(output_dir / "protocol_audit.csv", audit)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_csv(
        output_dir / "five_method_aggregate.csv",
        five_method_aggregate,
    )
    write_csv(output_dir / "cross_vs_lcss_paired_deltas.csv", deltas)
    write_csv(output_dir / "cross_vs_lcss_paired_tests.csv", tests)
    (output_dir / "report.md").write_text(
        _render_report(
            native_rows,
            evidence,
            five_method_evidence,
            completeness,
            audit,
            aggregate,
            five_method_aggregate,
            deltas,
            tests,
        ),
        encoding="utf-8",
    )
    return 0


def orchestrate(
    *,
    python: Path,
    noisygl_root: Path,
    output_dir: Path,
    amendment_path: Path,
    ours_v2_dir: Path,
    device: str,
    dry_run: bool,
    summarize_only: bool,
) -> int:
    python = Path(python).resolve()
    noisygl_root = Path(noisygl_root).resolve()
    output_dir = Path(output_dir).resolve()
    amendment_path = Path(amendment_path).resolve()
    ours_v2_dir = Path(ours_v2_dir).resolve()
    ours_v2_raw, _ = _validate_paths(
        python,
        noisygl_root,
        amendment_path,
        ours_v2_dir,
    )
    for folder in ("raw", "raw_immutable", "logs"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    snapshot_sources = _required_input_sources(
        noisygl_root,
        amendment_path,
        ours_v2_raw,
    )
    _write_or_verify_input_snapshots(snapshot_sources, output_dir)
    snapshot_manifest = output_dir / "input_snapshot_manifest.csv"
    snapshot_manifest_sha256 = sha256_file(snapshot_manifest)
    ours_v2_raw_sha256 = sha256_file(ours_v2_raw)
    specs = build_command_specs(
        python=python,
        noisygl_root=noisygl_root,
        output_dir=output_dir,
        device=device,
    )
    deterministic_env = deterministic_training_env({})
    env_json = json.dumps(deterministic_env, sort_keys=True)
    manifest = []
    for spec in specs:
        command_json = json.dumps(spec.command, ensure_ascii=False)
        manifest.append(
            {
                "dataset": spec.dataset,
                "regime": spec.regime.key,
                "noise_type": spec.regime.noise_type,
                "noise_rate": spec.regime.noise_rate,
                "methods": ",".join(spec.methods),
                "expected_rows": spec.expected_rows,
                "raw_path": str(spec.raw_path),
                "immutable_raw_path": str(spec.immutable_path),
                "stdout_path": str(spec.stdout_path),
                "stderr_path": str(spec.stderr_path),
                "command_json": command_json,
                "command_sha256": sha256_bytes(
                    command_json.encode("utf-8")
                ),
                "deterministic_env_json": env_json,
                "deterministic_env_sha256": sha256_bytes(
                    env_json.encode("utf-8")
                ),
                "input_snapshot_manifest_sha256": (
                    snapshot_manifest_sha256
                ),
                "ours_v2_raw_path": str(ours_v2_raw),
                "ours_v2_raw_sha256": ours_v2_raw_sha256,
                "status": (
                    "complete"
                    if _command_output_complete(spec)
                    else "planned"
                ),
                "returncode": "",
            }
        )
    write_csv(output_dir / "command_manifest.csv", manifest)
    if dry_run:
        return 0
    _write_environment_fingerprint(python, noisygl_root, output_dir)
    if summarize_only:
        return summarize_output(
            output_dir,
            specs,
            ours_v2_raw,
            expected_ours_v2_sha256=ours_v2_raw_sha256,
        )
    training_env = deterministic_training_env()
    for spec, manifest_row in zip(specs, manifest):
        if _command_output_complete(spec):
            manifest_row["status"] = "skipped_exact_complete"
            continue
        spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with spec.stdout_path.open(
                "w",
                encoding="utf-8",
            ) as stdout_handle, spec.stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr_handle:
                result = subprocess.run(
                    spec.command,
                    cwd=noisygl_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                    env=training_env,
                )
        except Exception as exc:
            manifest_row["status"] = "launch_failed"
            manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            write_csv(output_dir / "command_manifest.csv", manifest)
            raise
        manifest_row["returncode"] = result.returncode
        if result.returncode != 0:
            manifest_row["status"] = "failed"
            write_csv(output_dir / "command_manifest.csv", manifest)
            raise RuntimeError(
                "nearest-work training command failed: "
                f"dataset={spec.dataset} regime={spec.regime.key} "
                f"returncode={result.returncode}"
            )
        try:
            immutable_status = _finalize_command_output(spec)
            if not _command_output_complete(spec):
                raise ValueError(
                    "protocol drift: finalized native output pair is not "
                    f"exact-complete for {spec.dataset}/{spec.regime.key}"
                )
        except Exception as exc:
            manifest_row["status"] = "invalid_output"
            manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            write_csv(output_dir / "command_manifest.csv", manifest)
            _write_immutable_raw_manifest(output_dir, specs)
            raise
        manifest_row["immutable_status"] = immutable_status
        manifest_row["status"] = "complete"
        write_csv(output_dir / "command_manifest.csv", manifest)
    write_csv(output_dir / "command_manifest.csv", manifest)
    if any(not _command_output_complete(spec) for spec in specs):
        _write_immutable_raw_manifest(output_dir, specs)
        raise ValueError(
            "protocol drift: one or more native outputs are not exact-complete"
        )
    _write_immutable_raw_manifest(output_dir, specs)
    return summarize_output(
        output_dir,
        specs,
        ours_v2_raw,
        expected_ours_v2_sha256=ours_v2_raw_sha256,
    )


def default_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[2]
    repository_root = project_root.parents[1]
    noisygl_root = repository_root / "external" / "NoisyGL"
    experiment_dir = project_root / "paper_pipeline" / "04_experiments"
    return {
        "noisygl_root": noisygl_root,
        "output_dir": noisygl_root
        / "runs"
        / "nearest_work_full_grid_20260728",
        "amendment_path": experiment_dir
        / "nearest_work_full_grid_experiment_amendment_03.md",
        "ours_v2_dir": noisygl_root
        / "runs"
        / "tie_safe_full_grid_20260727_v2_immutable",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Explicit training interpreter path (including ROCm venvs).",
    )
    parser.add_argument(
        "--noisygl-root",
        type=Path,
        default=defaults["noisygl_root"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
    )
    parser.add_argument(
        "--amendment-path",
        type=Path,
        default=defaults["amendment_path"],
    )
    parser.add_argument(
        "--ours-v2-dir",
        type=Path,
        default=defaults["ours_v2_dir"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return orchestrate(
        python=args.python,
        noisygl_root=args.noisygl_root,
        output_dir=args.output_dir,
        amendment_path=args.amendment_path,
        ours_v2_dir=args.ours_v2_dir,
        device=args.device,
        dry_run=args.dry_run,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
