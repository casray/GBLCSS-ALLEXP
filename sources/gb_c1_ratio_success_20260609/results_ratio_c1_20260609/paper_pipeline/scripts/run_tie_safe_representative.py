#!/usr/bin/env python3
"""Run and summarize the frozen tie-safe representative experiment.

The experiment design lives in
paper_pipeline/04_experiments/tie_safe_representative_experiment_plan.md.
This script intentionally fixes the datasets, noise regime, seeds, method,
and tie modes. It is an auditable runner, not a hyperparameter-search tool.
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
from typing import Iterable, Sequence


DATASETS = ("cora", "corafull")
TIE_MODES = ("min_index", "alpha_zero", "distribution_only")
SEEDS = (0, 1, 2, 3, 4)
METHOD = "frozengrace_gb_mlp_softdist_beta05"
T_CRITICAL_DF4_95 = 2.7764451051977987
ZERO_TOLERANCE = 1e-12
MATERIALITY_THRESHOLD = 0.01

ROLE_BY_MODE = {
    "min_index": "historical_control",
    "alpha_zero": "mechanism_ablation",
    "distribution_only": "fixed_primary_repair",
}

REQUIRED_RAW_FIELDS = (
    "dataset",
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
    "modified_count",
    "modified_train_count",
    "modified_val_count",
    "modified_test_count",
    "feature_embedding_path",
    "ball_embedding_path",
    "status",
    "valid",
    "test",
    "tie_mode",
    "tie_ball_count",
    "tie_active_ball_count",
    "tie_affected_node_count",
    "tie_affected_alpha_sum",
    "ball_assignment_hash",
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
    "feature_embedding_path",
    "ball_embedding_path",
    "ball_assignment_hash",
)


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            text = payload.decode(encoding)
            return list(csv.DictReader(io.StringIO(text, newline="")))
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


def _nonempty(row: dict[str, str], key: str) -> bool:
    return key in row and str(row[key]).strip() != ""


def _normalized_dataset(value: str) -> str:
    return str(value).strip().lower()


def _normalized_mode(value: str) -> str:
    return str(value).strip().lower()


def raw_file_complete(path: Path, dataset: str, tie_mode: str) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_csv(path)
    except (OSError, csv.Error, UnicodeError):
        return False
    if len(rows) != len(SEEDS):
        return False
    seen: set[int] = set()
    for row in rows:
        if any(not _nonempty(row, field) for field in REQUIRED_RAW_FIELDS):
            return False
        if _normalized_dataset(row["dataset"]) != dataset:
            return False
        if _normalized_mode(row["tie_mode"]) != tie_mode:
            return False
        if str(row["status"]).strip().lower() != "ok":
            return False
        try:
            seed = int(row["seed"])
            float(row["test"])
            float(row["valid"])
        except (TypeError, ValueError):
            return False
        if seed not in SEEDS or seed in seen:
            return False
        seen.add(seed)
    return seen == set(SEEDS)


def _group_by_dataset_seed(rows: Sequence[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (_normalized_dataset(row["dataset"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def validate_protocol_consistency(rows: Sequence[dict]) -> None:
    expected_keys = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    grouped = _group_by_dataset_seed(rows)
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise ValueError(f"protocol drift: dataset/seed coverage mismatch missing={missing} extra={extra}")
    for key, group in sorted(grouped.items()):
        modes = {_normalized_mode(row["tie_mode"]) for row in group}
        if len(group) != len(TIE_MODES) or modes != set(TIE_MODES):
            raise ValueError(f"protocol drift: mode coverage mismatch at {key}: {sorted(modes)}")
        reference = next(row for row in group if _normalized_mode(row["tie_mode"]) == "min_index")
        for row in group:
            for field in PROTOCOL_EQUAL_FIELDS:
                if str(row.get(field, "")) != str(reference.get(field, "")):
                    raise ValueError(
                        "protocol drift: "
                        f"dataset={key[0]} seed={key[1]} mode={row['tie_mode']} "
                        f"field={field} reference={reference.get(field)!r} observed={row.get(field)!r}"
                    )


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _direction_counts(values: Sequence[float]) -> tuple[int, int, int]:
    positive = sum(value > ZERO_TOLERANCE for value in values)
    negative = sum(value < -ZERO_TOLERANCE for value in values)
    zero = len(values) - positive - negative
    return positive, negative, zero


def summarize_rows(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    validate_protocol_consistency(rows)
    lookup = {
        (_normalized_dataset(row["dataset"]), _normalized_mode(row["tie_mode"]), int(row["seed"])): row
        for row in rows
    }
    summaries: list[dict] = []
    paired_rows: list[dict] = []
    for dataset in DATASETS:
        control = [float(lookup[(dataset, "min_index", seed)]["test"]) for seed in SEEDS]
        for mode in TIE_MODES:
            values = [float(lookup[(dataset, mode, seed)]["test"]) for seed in SEEDS]
            summary = {
                "dataset": dataset,
                "tie_mode": mode,
                "role": ROLE_BY_MODE[mode],
                "n": len(values),
                "seeds": ",".join(str(seed) for seed in SEEDS),
                "test_mean": statistics.mean(values),
                "test_std": _sample_std(values),
                "valid_mean": statistics.mean(
                    float(lookup[(dataset, mode, seed)]["valid"]) for seed in SEEDS
                ),
                "valid_std": _sample_std(
                    [float(lookup[(dataset, mode, seed)]["valid"]) for seed in SEEDS]
                ),
                "full_grid_required_after_integrity_pass": 1,
            }
            if mode == "min_index":
                summary.update(
                    {
                        "paired_delta_mean": 0.0,
                        "paired_delta_std": 0.0,
                        "paired_delta_ci95_low": 0.0,
                        "paired_delta_ci95_high": 0.0,
                        "positive_seed_count": 0,
                        "negative_seed_count": 0,
                        "zero_seed_count": len(SEEDS),
                        "statistical_detectability": 0,
                        "practical_materiality": 0,
                    }
                )
            else:
                deltas = [value - base for value, base in zip(values, control)]
                mean_delta = statistics.mean(deltas)
                std_delta = _sample_std(deltas)
                half_width = T_CRITICAL_DF4_95 * std_delta / math.sqrt(len(deltas))
                ci_low = mean_delta - half_width
                ci_high = mean_delta + half_width
                positive, negative, zero = _direction_counts(deltas)
                detectable = int(ci_low > 0.0 or ci_high < 0.0)
                same_direction = max(positive, negative)
                material = int(
                    abs(mean_delta) >= MATERIALITY_THRESHOLD
                    and same_direction >= 4
                )
                summary.update(
                    {
                        "paired_delta_mean": mean_delta,
                        "paired_delta_std": std_delta,
                        "paired_delta_ci95_low": ci_low,
                        "paired_delta_ci95_high": ci_high,
                        "positive_seed_count": positive,
                        "negative_seed_count": negative,
                        "zero_seed_count": zero,
                        "statistical_detectability": detectable,
                        "practical_materiality": material,
                    }
                )
                for seed, delta in zip(SEEDS, deltas):
                    paired_rows.append(
                        {
                            "dataset": dataset,
                            "tie_mode": mode,
                            "role": ROLE_BY_MODE[mode],
                            "seed": seed,
                            "control_test": lookup[(dataset, "min_index", seed)]["test"],
                            "safe_test": lookup[(dataset, mode, seed)]["test"],
                            "delta": delta,
                            "delta_pp": 100.0 * delta,
                            "split_hash": lookup[(dataset, mode, seed)]["split_hash"],
                            "noise_hash": lookup[(dataset, mode, seed)]["noise_hash"],
                            "ball_assignment_hash": lookup[(dataset, mode, seed)][
                                "ball_assignment_hash"
                            ],
                        }
                    )
            summaries.append(summary)
    return summaries, paired_rows


def _raw_path(output_dir: Path, dataset: str, tie_mode: str) -> Path:
    return output_dir / "raw" / f"{dataset}_uniform0p6_{tie_mode}_seeds_0_4.csv"


def _command_for(
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    output_path: Path,
    dataset: str,
    tie_mode: str,
) -> list[str]:
    return [
        str(python),
        str(noisygl_root / "scripts" / "run_ssl_swap_downstream.py"),
        "--embedding_root",
        str(embedding_root),
        "--datasets",
        dataset,
        "--ssl_methods",
        "grace",
        "--base_methods",
        METHOD,
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--noise_type",
        "uniform",
        "--noise_rate",
        "0.6",
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
        json.dumps({"gb.tie_mode": tie_mode}, sort_keys=True),
        "--output",
        str(output_path),
    ]


def _validate_paths(
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    plan_path: Path,
) -> None:
    required = {
        "python": python,
        "noisygl_root": noisygl_root,
        "runner": noisygl_root / "scripts" / "run_ssl_swap_downstream.py",
        "embedding_root": embedding_root,
        "plan": plan_path,
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing required {label}: {path}")
    config_root = noisygl_root / "config" / METHOD
    for dataset in DATASETS:
        config = config_root / f"{METHOD}_{dataset}.yaml"
        if not config.exists():
            raise FileNotFoundError(f"missing config: {config}")
        text = config.read_text(encoding="utf-8")
        if "tie_mode:" not in text:
            raise ValueError(f"config does not expose gb.tie_mode: {config}")


def _snapshot_inputs(
    noisygl_root: Path,
    output_dir: Path,
    plan_path: Path,
) -> None:
    snapshot_dir = output_dir / "config_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    files = [plan_path]
    amendment = plan_path.with_name("tie_safe_representative_experiment_amendment_01.md")
    if amendment.exists():
        files.append(amendment)
    config_root = noisygl_root / "config" / METHOD
    files.extend(config_root / f"{METHOD}_{dataset}.yaml" for dataset in DATASETS)
    files.extend(
        [
            noisygl_root / "utils" / "frozen_grace_targets.py",
            noisygl_root / "predictor" / "FrozenGraceGB_GCN_Predictor.py",
            noisygl_root / "scripts" / "run_ssl_swap_downstream.py",
            Path(__file__).resolve(),
        ]
    )
    manifest: list[dict] = []
    for index, source in enumerate(files, start=1):
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
    python: Path,
    noisygl_root: Path,
    output_dir: Path,
) -> None:
    training_env = deterministic_training_env()
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
            key: training_env.get(key, "")
            for key in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_LAUNCH_BLOCKING",
                "HIP_VISIBLE_DEVICES",
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
    command = [
        str(python),
        "-c",
        (
            "import json,platform,sys,torch;"
            "print(json.dumps({"
            "'python':sys.version,'platform':platform.platform(),"
            "'torch':torch.__version__,"
            "'cuda_available':torch.cuda.is_available(),"
            "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU'"
            "},sort_keys=True))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env=training_env,
        )
        payload["training_runtime"] = json.loads(result.stdout.strip())
    except Exception as exc:  # environment capture failure is retained, never hidden
        payload["training_runtime_error"] = f"{type(exc).__name__}: {exc}"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "--", str(noisygl_root)],
            cwd=noisygl_root,
            check=True,
            capture_output=True,
            timeout=60,
        ).stdout
        payload["git_commit"] = commit
        payload["git_scoped_diff_sha256"] = sha256_bytes(diff)
    except Exception as exc:
        payload["git_error"] = f"{type(exc).__name__}: {exc}"
    write_json(output_dir / "environment_fingerprint.json", payload)


def _inject_fixed_protocol_fields(path: Path, dataset: str, tie_mode: str) -> None:
    rows = read_csv(path)
    for row in rows:
        row["dataset"] = dataset
        row["tie_mode"] = tie_mode
        row.setdefault("method", METHOD)
        row.setdefault("base_method", METHOD)
        row.setdefault("ssl_method", "grace")
        row["noise_type"] = "uniform"
        row["noise_rate"] = "0.6"
        row["noise_scope"] = "train"
        row["split_type"] = "percent"
        row["train_percent"] = "0.2"
        row["val_percent"] = "0.1"
        row["test_percent"] = "0.7"
    write_csv(path, rows)


def _collect_complete_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        for tie_mode in TIE_MODES:
            path = _raw_path(output_dir, dataset, tie_mode)
            if not raw_file_complete(path, dataset, tie_mode):
                raise ValueError(f"incomplete raw result: {path}")
            rows.extend(read_csv(path))
    if len(rows) != len(DATASETS) * len(TIE_MODES) * len(SEEDS):
        raise ValueError(f"expected 30 rows, found {len(rows)}")
    return rows


def _historical_reproduction(
    rows: Sequence[dict],
    ledger_path: Path | None,
) -> list[dict]:
    if ledger_path is None or not ledger_path.exists():
        return []
    ledger = read_csv(ledger_path)
    historical = {
        (_normalized_dataset(row["dataset"]), int(row["seed"])): row
        for row in ledger
        if row.get("method") == METHOD
        and row.get("noise_type") == "uniform"
        and row.get("noise_rate") == "0.6"
        and _normalized_dataset(row.get("dataset", "")) in DATASETS
    }
    controls = {
        (_normalized_dataset(row["dataset"]), int(row["seed"])): row
        for row in rows
        if _normalized_mode(row["tie_mode"]) == "min_index"
    }
    output: list[dict] = []
    for key in sorted(controls):
        observed = controls[key]
        reference = historical.get(key)
        if reference is None:
            output.append(
                {
                    "dataset": key[0],
                    "seed": key[1],
                    "status": "GAP_NO_HISTORICAL_ROW",
                }
            )
            continue
        test_delta = float(observed["test"]) - float(reference["test"])
        comparisons = {}
        for field in ("split_hash", "noise_hash"):
            expected = str(reference.get(field, ""))
            if expected in {"", "GAP"}:
                comparisons[f"{field}_match"] = "GAP"
            else:
                comparisons[f"{field}_match"] = int(str(observed.get(field, "")) == expected)
        output.append(
            {
                "dataset": key[0],
                "seed": key[1],
                "historical_test": reference["test"],
                "rerun_test": observed["test"],
                "test_delta": test_delta,
                "test_within_1e_6": int(abs(test_delta) <= 1e-6),
                **comparisons,
                "historical_protocol_status": reference.get("protocol_status", ""),
                "status": (
                    "PASS"
                    if abs(test_delta) <= 1e-6
                    and all(value in {1, "GAP"} for value in comparisons.values())
                    else "FAIL"
                ),
            }
        )
    return output


def _render_report(
    summaries: Sequence[dict],
    historical: Sequence[dict],
    output_dir: Path,
) -> str:
    lines = [
        "# Tie-safe 代表性实验报告",
        "",
        f"- 生成时间（UTC）：{utc_now()}",
        "- 状态：代表性 30 行已完成并通过跨模式协议一致性检查。",
        "- 固定主修复：`distribution_only`；`alpha_zero` 仅为机制消融。",
        "- 统计区间：n=5 配对差、df=4、未校正 nominal 95% t 区间。",
        "- Claim 边界：CI 包含 0 不证明等价或非劣。",
        "",
        "## 汇总",
        "",
        "| dataset | mode | role | test mean±std | paired Δ mean [95% CI] | detectable | material |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {dataset} | {tie_mode} | {role} | {mean:.4f}±{std:.4f} | "
            "{delta:+.4f} [{low:+.4f}, {high:+.4f}] | {detect} | {material} |".format(
                dataset=row["dataset"],
                tie_mode=row["tie_mode"],
                role=row["role"],
                mean=float(row["test_mean"]),
                std=float(row["test_std"]),
                delta=float(row["paired_delta_mean"]),
                low=float(row["paired_delta_ci95_low"]),
                high=float(row["paired_delta_ci95_high"]),
                detect=row["statistical_detectability"],
                material=row["practical_materiality"],
            )
        )
    lines.extend(
        [
            "",
            "## 历史 min_index 复现",
            "",
            f"- 对照行数：{len(historical)}。",
            f"- PASS：{sum(row.get('status') == 'PASS' for row in historical)}。",
            f"- FAIL：{sum(row.get('status') == 'FAIL' for row in historical)}。",
            f"- GAP：{sum(str(row.get('status', '')).startswith('GAP') for row in historical)}。",
            "",
            "## 固定后续",
            "",
            "- 只要数学测试与本次协议完整性通过，完整 25-setting × 5-seed × 4-method 网格无条件执行。",
            "- 代表性 test accuracy 不改变主修复、消融角色或扩展决定。",
            "",
            "## 产物",
            "",
            f"- `{output_dir / 'raw_combined.csv'}`",
            f"- `{output_dir / 'paired_deltas.csv'}`",
            f"- `{output_dir / 'summary.csv'}`",
            f"- `{output_dir / 'historical_reproduction.csv'}`",
            f"- `{output_dir / 'command_manifest.csv'}`",
            f"- `{output_dir / 'input_snapshot_manifest.csv'}`",
            f"- `{output_dir / 'environment_fingerprint.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_output(
    output_dir: Path,
    ledger_path: Path | None = None,
) -> None:
    rows = _collect_complete_rows(output_dir)
    validate_protocol_consistency(rows)
    summaries, paired = summarize_rows(rows)
    historical = _historical_reproduction(rows, ledger_path)
    write_csv(output_dir / "raw_combined.csv", rows)
    write_csv(output_dir / "paired_deltas.csv", paired)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "historical_reproduction.csv", historical)
    (output_dir / "report.md").write_text(
        _render_report(summaries, historical, output_dir),
        encoding="utf-8",
    )


def orchestrate(
    *,
    python: Path,
    noisygl_root: Path,
    embedding_root: Path,
    output_dir: Path,
    plan_path: Path,
    dry_run: bool,
    summarize_only: bool,
    ledger_path: Path | None = None,
) -> int:
    python = Path(python).resolve()
    noisygl_root = Path(noisygl_root).resolve()
    embedding_root = Path(embedding_root).resolve()
    output_dir = Path(output_dir).resolve()
    plan_path = Path(plan_path).resolve()
    ledger_path = Path(ledger_path).resolve() if ledger_path is not None else None
    _validate_paths(python, noisygl_root, embedding_root, plan_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    _snapshot_inputs(noisygl_root, output_dir, plan_path)

    manifest: list[dict] = []
    deterministic_env = deterministic_training_env({})
    deterministic_env_json = json.dumps(deterministic_env, sort_keys=True)
    for dataset in DATASETS:
        for tie_mode in TIE_MODES:
            raw_path = _raw_path(output_dir, dataset, tie_mode)
            stdout_path = output_dir / "logs" / f"{dataset}_{tie_mode}.stdout.log"
            stderr_path = output_dir / "logs" / f"{dataset}_{tie_mode}.stderr.log"
            command = _command_for(
                python,
                noisygl_root,
                embedding_root,
                raw_path,
                dataset,
                tie_mode,
            )
            manifest.append(
                {
                    "dataset": dataset,
                    "tie_mode": tie_mode,
                    "role": ROLE_BY_MODE[tie_mode],
                    "raw_path": str(raw_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "command_json": json.dumps(command, ensure_ascii=False),
                    "command_sha256": sha256_bytes(
                        json.dumps(command, ensure_ascii=False).encode("utf-8")
                    ),
                    "deterministic_env_json": deterministic_env_json,
                    "deterministic_env_sha256": sha256_bytes(
                        deterministic_env_json.encode("utf-8")
                    ),
                    "status": (
                        "complete"
                        if raw_file_complete(raw_path, dataset, tie_mode)
                        else "planned"
                    ),
                }
            )
    write_csv(output_dir / "command_manifest.csv", manifest)

    if dry_run:
        return 0
    _environment_fingerprint(python, noisygl_root, output_dir)
    if not summarize_only:
        for row in manifest:
            dataset = row["dataset"]
            tie_mode = row["tie_mode"]
            raw_path = Path(row["raw_path"])
            if raw_file_complete(raw_path, dataset, tie_mode):
                continue
            command = json.loads(row["command_json"])
            stdout_path = Path(row["stdout_path"])
            stderr_path = Path(row["stderr_path"])
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                result = subprocess.run(
                    command,
                    cwd=noisygl_root,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    check=False,
                    env=deterministic_training_env(),
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"training failed dataset={dataset} tie_mode={tie_mode} "
                    f"exit={result.returncode}; see {stderr_path}"
                )
            _inject_fixed_protocol_fields(raw_path, dataset, tie_mode)
            if not raw_file_complete(raw_path, dataset, tie_mode):
                raise ValueError(
                    f"training output is incomplete dataset={dataset} "
                    f"tie_mode={tie_mode}: {raw_path}"
                )
    summarize_output(output_dir, ledger_path=ledger_path)
    return 0


def default_paths() -> dict[str, Path]:
    project_root = Path(__file__).resolve().parents[2]
    repository_root = project_root.parents[1]
    noisygl_root = repository_root / "external" / "NoisyGL"
    return {
        "project_root": project_root,
        "noisygl_root": noisygl_root,
        "embedding_root": repository_root
        / "gb_graph_aware_minival_6_1"
        / "checkpoints"
        / "embeddings_v2",
        "output_dir": noisygl_root / "runs" / "tie_safe_representative_20260727",
        "plan_path": project_root
        / "paper_pipeline"
        / "04_experiments"
        / "tie_safe_representative_experiment_plan.md",
        "ledger_path": project_root
        / "paper_pipeline"
        / "00_evidence"
        / "protocol_ledger.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--noisygl-root", type=Path, default=defaults["noisygl_root"])
    parser.add_argument("--embedding-root", type=Path, default=defaults["embedding_root"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--plan-path", type=Path, default=defaults["plan_path"])
    parser.add_argument("--ledger-path", type=Path, default=defaults["ledger_path"])
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
        dry_run=args.dry_run,
        summarize_only=args.summarize_only,
        ledger_path=args.ledger_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
