#!/usr/bin/env python3
"""Run and audit the frozen 500-row tie-safe full grid.

Each dataset/regime is launched twice because NoisyGL applies one override
dictionary to every method in a command: one command for the two SoftDist
methods with ``gb.tie_mode=distribution_only``, and one command for the two
baselines with no override.
"""

from __future__ import annotations

import argparse
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
    group: str
    methods: tuple[str, str]
    raw_path: Path
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
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MLP_CE = "mlp_ce"
MLP_JOCOR = "mlp_jocor"
SOFTDIST_METHODS = (OURS, SHUFFLED)
BASELINE_METHODS = (MLP_CE, MLP_JOCOR)
METHODS = SOFTDIST_METHODS + BASELINE_METHODS
COMPARATORS = (SHUFFLED, MLP_CE, MLP_JOCOR)
GROUP_METHODS = {
    "softdist": SOFTDIST_METHODS,
    "baseline": BASELINE_METHODS,
}
FROZEN_LEDGER_SHA256 = (
    "73CEC921C092EC398FC13EC4E1C4A85A43BC3792298F702930222E1524ADCB90"
)
TIE_MODE = "distribution_only"
ALPHA = 0.05
ZERO_TOLERANCE = 1e-15

COMMON_REQUIRED_FIELDS = (
    "dataset",
    "regime",
    "method",
    "base_method",
    "ssl_method",
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
SOFTDIST_REQUIRED_FIELDS = (
    "tie_mode",
    "feature_embedding_path",
    "feature_embedding_hash",
    "ball_embedding_path",
    "ball_embedding_hash",
    "ball_assignment_hash",
    "tie_ball_count",
    "tie_active_ball_count",
    "tie_affected_node_count",
    "tie_affected_alpha_sum",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_training_env(base: dict[str, str] | None = None) -> dict[str, str]:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nonempty(row: dict, field: str) -> bool:
    return str(row.get(field, "")).strip() != ""


def _normalized(value) -> str:
    return str(value).strip().lower()


def _regime_lookup() -> dict[str, Regime]:
    return {regime.key: regime for regime in REGIMES}


def _planned_keys() -> list[tuple[str, str, str, int]]:
    return [
        (dataset, regime.key, method, seed)
        for dataset in DATASETS
        for regime in REGIMES
        for method in METHODS
        for seed in SEEDS
    ]


def _raw_path(
    output_dir: Path, dataset: str, regime: Regime, group: str
) -> Path:
    return output_dir / "raw" / f"{dataset}_{regime.key}_{group}_seeds_0_4.csv"


def _command_for(
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    output_path: Path,
    dataset: str,
    regime: Regime,
    group: str,
) -> tuple[str, ...]:
    methods = GROUP_METHODS[group]
    overrides = {"gb.tie_mode": TIE_MODE} if group == "softdist" else {}
    return (
        str(python),
        str(noisygl_root / "scripts" / "run_ssl_swap_downstream.py"),
        "--embedding_root",
        str(embedding_root),
        "--datasets",
        dataset,
        "--ssl_methods",
        "grace",
        "--base_methods",
        ",".join(methods),
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--noise_type",
        regime.noise_type,
        "--noise_rate",
        str(regime.noise_rate),
        "--noise_scope",
        "train",
        "--device",
        "cuda:0",
        "--split_type",
        "percent",
        "--train_percent",
        "0.2",
        "--val_percent",
        "0.1",
        "--test_percent",
        "0.7",
        "--config_overrides",
        json.dumps(overrides, sort_keys=True),
        "--output",
        str(output_path),
    )


def build_command_specs(
    *,
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    output_dir: Path,
) -> list[CommandSpec]:
    specs: list[CommandSpec] = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for group in ("softdist", "baseline"):
                raw_path = _raw_path(output_dir, dataset, regime, group)
                log_stem = f"{dataset}_{regime.key}_{group}"
                specs.append(
                    CommandSpec(
                        dataset=dataset,
                        regime=regime,
                        group=group,
                        methods=GROUP_METHODS[group],
                        raw_path=raw_path,
                        stdout_path=output_dir
                        / "logs"
                        / f"{log_stem}.stdout.log",
                        stderr_path=output_dir
                        / "logs"
                        / f"{log_stem}.stderr.log",
                        command=_command_for(
                            python,
                            noisygl_root,
                            embedding_root,
                            raw_path,
                            dataset,
                            regime,
                            group,
                        ),
                    )
                )
    return specs


def _row_key(row: dict) -> tuple[str, str, str, int]:
    return (
        _normalized(row.get("dataset", "")),
        _normalized(row.get("regime", "")),
        _normalized(row.get("method", row.get("base_method", ""))),
        int(row.get("seed", -1)),
    )


def raw_file_complete(
    path: Path, dataset: str, regime: Regime, group: str
) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv(path)
    except (OSError, csv.Error, UnicodeError, ValueError):
        return False
    if len(rows) != 10:
        return False
    expected = {
        (dataset, regime.key, method, seed)
        for method in GROUP_METHODS[group]
        for seed in SEEDS
    }
    observed: set[tuple[str, str, str, int]] = set()
    for row in rows:
        required = COMMON_REQUIRED_FIELDS
        if group == "softdist":
            required += SOFTDIST_REQUIRED_FIELDS
        if any(not _nonempty(row, field) for field in required):
            return False
        if _normalized(row.get("status")) != "ok":
            return False
        if group == "softdist" and _normalized(row.get("tie_mode")) != TIE_MODE:
            return False
        try:
            metrics = (float(row["valid"]), float(row["test"]))
            key = _row_key(row)
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics):
            return False
        if key in observed:
            return False
        observed.add(key)
    return observed == expected


def _validate_softdist_row(row: dict, key: tuple[str, str, str, int]) -> None:
    for field in SOFTDIST_REQUIRED_FIELDS:
        if not _nonempty(row, field):
            raise ValueError(
                f"protocol drift: softdist field {field} is empty at {key}"
            )
    if _normalized(row["tie_mode"]) != TIE_MODE:
        raise ValueError(
            f"protocol drift: tie_mode={row['tie_mode']!r} at {key}, "
            f"expected {TIE_MODE!r}"
        )


def _validate_protocol_group(
    dataset: str,
    regime: Regime,
    seed: int,
    group: Sequence[dict],
) -> None:
    expected_methods = set(METHODS)
    observed_methods = {_normalized(row.get("method", "")) for row in group}
    if len(group) != 4 or observed_methods != expected_methods:
        raise ValueError(
            "protocol drift: method coverage mismatch "
            f"dataset={dataset} regime={regime.key} seed={seed} "
            f"observed={sorted(observed_methods)}"
        )
    reference = next(row for row in group if _normalized(row["method"]) == OURS)
    for row in group:
        method = _normalized(row["method"])
        key = (dataset, regime.key, method, seed)
        if _normalized(row.get("status")) != "ok":
            raise ValueError(f"protocol drift: non-ok row at {key}")
        for field in COMMON_REQUIRED_FIELDS:
            if not _nonempty(row, field):
                raise ValueError(f"protocol drift: empty field {field} at {key}")
        metrics = (float(row["valid"]), float(row["test"]))
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics
        ):
            raise ValueError(f"protocol drift: non-finite/out-of-range metric at {key}")
        for field in (
            "modified_count",
            "modified_train_count",
            "modified_val_count",
            "modified_test_count",
        ):
            value = int(row[field])
            if value < 0:
                raise ValueError(f"protocol drift: negative {field} at {key}")
        if _normalized(row["noise_type"]) != regime.noise_type:
            raise ValueError(f"protocol drift: noise_type at {key}")
        if not math.isclose(
            float(row["noise_rate"]), regime.noise_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"protocol drift: noise_rate at {key}")
        if _normalized(row["noise_scope"]) != "train":
            raise ValueError(f"protocol drift: noise_scope at {key}")
        if _normalized(row["split_type"]) != "percent":
            raise ValueError(f"protocol drift: split_type at {key}")
        expected_ratios = {
            "train_percent": 0.2,
            "val_percent": 0.1,
            "test_percent": 0.7,
        }
        for field, expected in expected_ratios.items():
            if not math.isclose(
                float(row[field]), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"protocol drift: {field} at {key}")
        for field in PROTOCOL_EQUAL_FIELDS:
            if str(row.get(field, "")) != str(reference.get(field, "")):
                raise ValueError(
                    "protocol drift: "
                    f"dataset={dataset} regime={regime.key} seed={seed} "
                    f"method={method} field={field} "
                    f"reference={reference.get(field)!r} "
                    f"observed={row.get(field)!r}"
                )
        if method in SOFTDIST_METHODS:
            _validate_softdist_row(row, key)
    softdist_rows = [
        row for row in group if _normalized(row["method"]) in SOFTDIST_METHODS
    ]
    embedding_fields = (
        "feature_embedding_path",
        "feature_embedding_hash",
        "ball_embedding_path",
        "ball_embedding_hash",
    )
    for field in embedding_fields:
        observed = {str(row[field]).strip() for row in softdist_rows}
        if len(observed) != 1:
            raise ValueError(
                "protocol drift: SoftDist embedding mismatch "
                f"dataset={dataset} regime={regime.key} seed={seed} "
                f"field={field} observed={sorted(observed)}"
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
    audit: list[dict] = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for seed in SEEDS:
                key = (dataset, regime.key, seed)
                group = grouped.get(key, [])
                try:
                    _validate_protocol_group(dataset, regime, seed, group)
                    status = "PASS"
                    error = ""
                except (KeyError, TypeError, ValueError) as exc:
                    status = "FAIL"
                    error = str(exc)
                audit.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "seed": seed,
                        "expected_method_rows": 4,
                        "observed_method_rows": len(group),
                        "protocol_status": status,
                        "error": error,
                    }
                )
    return audit


def validate_protocol_consistency(rows: Sequence[dict]) -> None:
    expected_keys = set(_planned_keys())
    observed: dict[tuple[str, str, str, int], int] = {}
    for row in rows:
        try:
            key = _row_key(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"protocol drift: invalid row key: {exc}") from exc
        observed[key] = observed.get(key, 0) + 1
    missing = sorted(expected_keys - set(observed))
    extra = sorted(set(observed) - expected_keys)
    duplicates = sorted(key for key, count in observed.items() if count != 1)
    if missing or extra or duplicates:
        raise ValueError(
            "protocol drift: 500-row coverage mismatch "
            f"missing={missing[:5]} extra={extra[:5]} duplicates={duplicates[:5]}"
        )
    failures = [row for row in protocol_audit(rows) if row["protocol_status"] != "PASS"]
    if failures:
        raise ValueError(failures[0]["error"])
    for dataset in DATASETS:
        dataset_softdist = [
            row
            for row in rows
            if _normalized(row.get("dataset", "")) == dataset
            and _normalized(row.get("method", "")) in SOFTDIST_METHODS
        ]
        for field in (
            "feature_embedding_path",
            "feature_embedding_hash",
            "ball_embedding_path",
            "ball_embedding_hash",
        ):
            observed_values = {
                str(row.get(field, "")).strip() for row in dataset_softdist
            }
            if len(observed_values) != 1:
                raise ValueError(
                    "protocol drift: SoftDist embedding changed across the "
                    f"frozen grid dataset={dataset} field={field} "
                    f"observed={sorted(observed_values)}"
                )


def build_completeness(rows: Sequence[dict]) -> list[dict]:
    observed: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        try:
            observed.setdefault(_row_key(row), []).append(row)
        except (TypeError, ValueError):
            continue
    output: list[dict] = []
    for dataset, regime, method, seed in _planned_keys():
        matches = observed.get((dataset, regime, method, seed), [])
        if not matches:
            status = "missing"
            raw_status = ""
        elif len(matches) > 1:
            status = "duplicate"
            raw_status = "|".join(str(row.get("status", "")) for row in matches)
        else:
            raw_status = _normalized(matches[0].get("status", ""))
            status = "ok" if raw_status == "ok" else "failed"
        output.append(
            {
                "dataset": dataset,
                "regime": regime,
                "method": method,
                "seed": seed,
                "completion_status": status,
                "raw_status": raw_status,
            }
        )
    return output


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
        return (0.0, 1.0) if abs(mean) <= ZERO_TOLERANCE else (
            math.copysign(math.inf, mean),
            0.0,
        )
    t_stat = mean / (std / math.sqrt(len(values)))
    df = len(values) - 1
    x = df / (df + t_stat * t_stat)
    p_value = _regularized_beta(x, df / 2.0, 0.5)
    return t_stat, min(max(p_value, 0.0), 1.0)


def _apply_holm(rows: list[dict]) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda pair: float(pair[1]["p_value_two_sided"])
    )
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered, start=1):
        adjusted = min(1.0, (total - rank + 1) * float(row["p_value_two_sided"]))
        running = max(running, adjusted)
        rows[original_index]["holm_rank"] = rank
        rows[original_index]["holm_family_size"] = total
        rows[original_index]["holm_adjusted_p"] = running
        rows[original_index]["holm_significant_0p05"] = int(running < ALPHA)


def summarize_rows(
    rows: Sequence[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    validate_protocol_consistency(rows)
    lookup = {_row_key(row): row for row in rows}
    aggregate: list[dict] = []
    deltas: list[dict] = []
    paired_tests: list[dict] = []
    for dataset in DATASETS:
        for regime in REGIMES:
            for method in METHODS:
                values = [
                    float(lookup[(dataset, regime.key, method, seed)]["test"])
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
            ours_values = [
                float(lookup[(dataset, regime.key, OURS, seed)]["test"])
                for seed in SEEDS
            ]
            for comparator in COMPARATORS:
                comparator_values = [
                    float(lookup[(dataset, regime.key, comparator, seed)]["test"])
                    for seed in SEEDS
                ]
                paired_values = [
                    ours - other
                    for ours, other in zip(ours_values, comparator_values)
                ]
                for seed, ours, other, delta in zip(
                    SEEDS, ours_values, comparator_values, paired_values
                ):
                    source = lookup[(dataset, regime.key, OURS, seed)]
                    deltas.append(
                        {
                            "dataset": dataset,
                            "regime": regime.key,
                            "comparator": comparator,
                            "seed": seed,
                            "ours_test": ours,
                            "comparator_test": other,
                            "delta": delta,
                            "delta_pp": 100.0 * delta,
                            "split_hash": source["split_hash"],
                            "noise_hash": source["noise_hash"],
                        }
                    )
                t_stat, p_value = paired_t_test(paired_values)
                paired_tests.append(
                    {
                        "dataset": dataset,
                        "regime": regime.key,
                        "comparator": comparator,
                        "n": len(paired_values),
                        "paired_delta_mean": statistics.mean(paired_values),
                        "paired_delta_sample_std": _sample_std(paired_values),
                        "t_statistic": t_stat,
                        "degrees_of_freedom": len(paired_values) - 1,
                        "p_value_two_sided": p_value,
                        "positive_seed_count": sum(
                            value > ZERO_TOLERANCE for value in paired_values
                        ),
                        "negative_seed_count": sum(
                            value < -ZERO_TOLERANCE for value in paired_values
                        ),
                        "zero_seed_count": sum(
                            abs(value) <= ZERO_TOLERANCE for value in paired_values
                        ),
                    }
                )
    for comparator in COMPARATORS:
        family = [
            row for row in paired_tests if row["comparator"] == comparator
        ]
        _apply_holm(family)
    return aggregate, deltas, paired_tests


def _inject_fixed_fields(
    path: Path,
    dataset: str,
    regime: Regime,
    group: str,
) -> None:
    rows = read_csv(path)
    hash_cache: dict[str, str] = {}
    for row in rows:
        row["dataset"] = dataset
        row["regime"] = regime.key
        row["method"] = row.get("method") or row.get("base_method", "")
        row["base_method"] = row.get("base_method") or row["method"]
        row["ssl_method"] = row.get("ssl_method") or "grace"
        row["noise_type"] = regime.noise_type
        row["noise_rate"] = regime.noise_rate
        row["noise_scope"] = "train"
        row["split_type"] = "percent"
        row["train_percent"] = 0.2
        row["val_percent"] = 0.1
        row["test_percent"] = 0.7
        if group == "softdist":
            row["tie_mode"] = TIE_MODE
            for field in ("feature_embedding_path", "ball_embedding_path"):
                embedding_path = str(row.get(field, "")).strip()
                if not embedding_path:
                    raise ValueError(
                        f"missing {field} in SoftDist output: {path}"
                    )
                if embedding_path not in hash_cache:
                    resolved = Path(embedding_path)
                    if not resolved.exists():
                        raise FileNotFoundError(
                            f"missing recorded embedding: {resolved}"
                        )
                    hash_cache[embedding_path] = sha256_file(resolved)
                row[field.replace("_path", "_hash")] = hash_cache[embedding_path]
    write_csv(path, rows)


def _collect_rows(specs: Sequence[CommandSpec]) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        if spec.raw_path.exists():
            try:
                rows.extend(read_csv(spec.raw_path))
            except (OSError, csv.Error, UnicodeError):
                continue
    return rows


def _validate_paths(
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    plan_path: Path,
    amendment_path: Path,
    ledger_path: Path,
) -> None:
    required = {
        "python": python,
        "noisygl_root": noisygl_root,
        "downstream runner": noisygl_root
        / "scripts"
        / "run_ssl_swap_downstream.py",
        "embedding_root": embedding_root,
        "frozen plan": plan_path,
        "frozen amendment": amendment_path,
        "protocol ledger": ledger_path,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing required {label}: {path}")
    ledger_hash = sha256_file(ledger_path)
    if ledger_hash != FROZEN_LEDGER_SHA256:
        raise ValueError(
            "frozen protocol ledger hash mismatch: "
            f"expected={FROZEN_LEDGER_SHA256} observed={ledger_hash}"
        )
    for method in METHODS:
        for dataset in DATASETS:
            config = (
                noisygl_root
                / "config"
                / method
                / f"{method}_{dataset}.yaml"
            )
            if not config.exists():
                raise FileNotFoundError(f"missing config: {config}")
            if method in SOFTDIST_METHODS:
                text = config.read_text(encoding="utf-8")
                if "tie_mode:" not in text:
                    raise ValueError(
                        f"config does not expose gb.tie_mode: {config}"
                    )


def _snapshot_inputs(
    noisygl_root: Path,
    output_dir: Path,
    plan_path: Path,
    amendment_path: Path,
    ledger_path: Path,
) -> None:
    sources = [
        plan_path,
        amendment_path,
        ledger_path,
        noisygl_root / "scripts" / "run_ssl_swap_downstream.py",
        noisygl_root / "utils" / "frozen_grace_targets.py",
        noisygl_root / "predictor" / "FrozenGraceGB_GCN_Predictor.py",
        Path(__file__).resolve(),
    ]
    for method in METHODS:
        for dataset in DATASETS:
            sources.append(
                noisygl_root
                / "config"
                / method
                / f"{method}_{dataset}.yaml"
            )
    snapshot_dir = output_dir / "input_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for index, source in enumerate(sources, start=1):
        if not source.exists():
            raise FileNotFoundError(f"missing snapshot input: {source}")
        target = snapshot_dir / f"{index:02d}_{source.name}"
        target.write_bytes(source.read_bytes())
        manifest.append(
            {
                "source_path": str(source.resolve()),
                "snapshot_path": str(target.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    write_csv(output_dir / "input_snapshot_manifest.csv", manifest)


def _environment_fingerprint(
    python: Path, noisygl_root: Path, output_dir: Path
) -> None:
    env = deterministic_training_env()
    payload = {
        "captured_at_utc": utc_now(),
        "orchestrator_python": sys.executable,
        "orchestrator_python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "training_python": str(python.resolve()),
        "noisygl_root": str(noisygl_root.resolve()),
        "training_subprocess_environment_flags": {
            key: env.get(key, "")
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
        runtime = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json,platform,sys,torch;"
                    "print(json.dumps({"
                    "'python':sys.version,'platform':platform.platform(),"
                    "'torch':torch.__version__,"
                    "'cuda_available':torch.cuda.is_available(),"
                    "'device':torch.cuda.get_device_name(0) "
                    "if torch.cuda.is_available() else 'NO_GPU'"
                    "},sort_keys=True))"
                ),
            ],
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        payload["training_runtime"] = json.loads(runtime.stdout.strip())
    except Exception as exc:
        payload["training_runtime_error"] = f"{type(exc).__name__}: {exc}"
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
    write_json(output_dir / "environment_fingerprint.json", payload)


def _render_report(
    completeness: Sequence[dict],
    audit: Sequence[dict],
    aggregate: Sequence[dict],
    tests: Sequence[dict],
) -> str:
    completion_counts = {
        status: sum(
            row["completion_status"] == status for row in completeness
        )
        for status in ("ok", "missing", "failed", "duplicate")
    }
    lines = [
        "# Tie-safe 完整网格运行报告",
        "",
        f"- 生成时间（UTC）：{utc_now()}",
        "- 冻结范围：5 datasets × 5 regimes × 5 seeds × 4 methods = 500 rows。",
        "- 命令拆分：25 个 SoftDist 双方法命令 + 25 个 baseline 双方法命令。",
        f"- 完成状态：{json.dumps(completion_counts, ensure_ascii=False, sort_keys=True)}。",
        f"- 协议 PASS：{sum(row['protocol_status'] == 'PASS' for row in audit)}/125。",
        f"- 聚合行：{len(aggregate)}/100；配对检验行：{len(tests)}/75。",
        "- Holm family：每个 comparator 独立校正 25 个 dataset×regime tests。",
        "",
        "## Holm 结果",
        "",
        "| comparator | significant / 25 |",
        "|---|---:|",
    ]
    for comparator in COMPARATORS:
        family = [row for row in tests if row["comparator"] == comparator]
        lines.append(
            f"| {comparator} | "
            f"{sum(int(row['holm_significant_0p05']) for row in family)} / {len(family)} |"
        )
    lines.extend(
        [
            "",
            "## Claim 边界",
            "",
            "- 只有 500/500 行完整且 125/125 协议组 PASS 时，聚合结果才可进入新主表。",
            "- 配对 t 检验和 Holm 校正不替代效应量、sample std 或逐 seed 差值。",
            "- 真实球与 shuffled 球的 assignment hash 允许不同；两者均必须非空。",
            "- 任一 missing、failed、duplicate 或协议漂移均保留并拒绝主表闭合。",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_output(
    output_dir: Path, specs: Sequence[CommandSpec] | None = None
) -> None:
    if specs is None:
        defaults = default_paths()
        specs = build_command_specs(
            python=Path(sys.executable),
            noisygl_root=defaults["noisygl_root"],
            embedding_root=defaults["embedding_root"],
            output_dir=output_dir,
        )
    rows = _collect_rows(specs)
    completeness = build_completeness(rows)
    audit = protocol_audit(rows)
    write_csv(output_dir / "raw_combined.csv", rows)
    write_csv(output_dir / "completeness.csv", completeness)
    write_csv(output_dir / "protocol_audit.csv", audit)
    try:
        validate_protocol_consistency(rows)
    except ValueError:
        write_csv(output_dir / "aggregate.csv", [])
        write_csv(output_dir / "paired_deltas.csv", [])
        write_csv(output_dir / "paired_tests.csv", [])
        (output_dir / "report.md").write_text(
            _render_report(completeness, audit, [], []), encoding="utf-8"
        )
        raise
    aggregate, deltas, tests = summarize_rows(rows)
    write_csv(output_dir / "aggregate.csv", aggregate)
    write_csv(output_dir / "paired_deltas.csv", deltas)
    write_csv(output_dir / "paired_tests.csv", tests)
    (output_dir / "report.md").write_text(
        _render_report(completeness, audit, aggregate, tests),
        encoding="utf-8",
    )


def orchestrate(
    *,
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    output_dir: Path,
    plan_path: Path,
    amendment_path: Path,
    ledger_path: Path,
    dry_run: bool,
    summarize_only: bool,
) -> int:
    python = Path(python).resolve()
    noisygl_root = Path(noisygl_root).resolve()
    embedding_root = Path(embedding_root).resolve()
    output_dir = Path(output_dir).resolve()
    plan_path = Path(plan_path).resolve()
    amendment_path = Path(amendment_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    _validate_paths(
        python,
        noisygl_root,
        embedding_root,
        plan_path,
        amendment_path,
        ledger_path,
    )
    for folder in ("raw", "logs"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    _snapshot_inputs(
        noisygl_root,
        output_dir,
        plan_path,
        amendment_path,
        ledger_path,
    )
    specs = build_command_specs(
        python=python,
        noisygl_root=noisygl_root,
        embedding_root=embedding_root,
        output_dir=output_dir,
    )
    deterministic_env = deterministic_training_env({})
    env_json = json.dumps(deterministic_env, sort_keys=True)
    manifest: list[dict] = []
    for spec in specs:
        command_json = json.dumps(spec.command, ensure_ascii=False)
        manifest.append(
            {
                "dataset": spec.dataset,
                "regime": spec.regime.key,
                "noise_type": spec.regime.noise_type,
                "noise_rate": spec.regime.noise_rate,
                "group": spec.group,
                "methods": ",".join(spec.methods),
                "expected_rows": 10,
                "raw_path": str(spec.raw_path),
                "stdout_path": str(spec.stdout_path),
                "stderr_path": str(spec.stderr_path),
                "command_json": command_json,
                "command_sha256": sha256_bytes(command_json.encode("utf-8")),
                "deterministic_env_json": env_json,
                "deterministic_env_sha256": sha256_bytes(
                    env_json.encode("utf-8")
                ),
                "status": (
                    "complete"
                    if raw_file_complete(
                        spec.raw_path,
                        spec.dataset,
                        spec.regime,
                        spec.group,
                    )
                    else "planned"
                ),
                "returncode": "",
            }
        )
    write_csv(output_dir / "command_manifest.csv", manifest)
    if dry_run:
        return 0

    _environment_fingerprint(python, noisygl_root, output_dir)
    if not summarize_only:
        for spec, manifest_row in zip(specs, manifest):
            if raw_file_complete(
                spec.raw_path, spec.dataset, spec.regime, spec.group
            ):
                manifest_row["status"] = "skipped_exact_complete"
                continue
            spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with spec.stdout_path.open(
                "w", encoding="utf-8"
            ) as stdout_handle, spec.stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                result = subprocess.run(
                    list(spec.command),
                    cwd=noisygl_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                    env=deterministic_training_env(),
                )
            manifest_row["returncode"] = result.returncode
            if result.returncode != 0:
                manifest_row["status"] = "failed"
                write_csv(output_dir / "command_manifest.csv", manifest)
                continue
            try:
                _inject_fixed_fields(
                    spec.raw_path,
                    spec.dataset,
                    spec.regime,
                    spec.group,
                )
                if not raw_file_complete(
                    spec.raw_path,
                    spec.dataset,
                    spec.regime,
                    spec.group,
                ):
                    raise ValueError("output is not exact 10-row complete")
            except Exception as exc:
                manifest_row["status"] = "invalid_output"
                manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            else:
                manifest_row["status"] = "complete"
            write_csv(output_dir / "command_manifest.csv", manifest)
        write_csv(output_dir / "command_manifest.csv", manifest)
    summarize_output(output_dir, specs=specs)
    return 0


def default_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[2]
    repository_root = project_root.parents[1]
    noisygl_root = repository_root / "external" / "NoisyGL"
    experiment_dir = (
        project_root / "paper_pipeline" / "04_experiments"
    )
    return {
        "project_root": project_root,
        "noisygl_root": noisygl_root,
        "embedding_root": repository_root
        / "gb_graph_aware_minival_6_1"
        / "checkpoints"
        / "embeddings_v2",
        "output_dir": noisygl_root / "runs" / "tie_safe_full_grid_20260727",
        "plan_path": experiment_dir
        / "tie_safe_representative_experiment_plan.md",
        "amendment_path": experiment_dir
        / "tie_safe_representative_experiment_amendment_01.md",
        "ledger_path": project_root
        / "paper_pipeline"
        / "00_evidence"
        / "protocol_ledger.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--noisygl-root", type=Path, default=defaults["noisygl_root"]
    )
    parser.add_argument(
        "--embedding-root", type=Path, default=defaults["embedding_root"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=defaults["output_dir"]
    )
    parser.add_argument(
        "--plan-path", type=Path, default=defaults["plan_path"]
    )
    parser.add_argument(
        "--amendment-path", type=Path, default=defaults["amendment_path"]
    )
    parser.add_argument(
        "--ledger-path", type=Path, default=defaults["ledger_path"]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return orchestrate(
        python=args.python,
        noisygl_root=args.noisygl_root,
        embedding_root=args.embedding_root,
        output_dir=args.output_dir,
        plan_path=args.plan_path,
        amendment_path=args.amendment_path,
        ledger_path=args.ledger_path,
        dry_run=args.dry_run,
        summarize_only=args.summarize_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
