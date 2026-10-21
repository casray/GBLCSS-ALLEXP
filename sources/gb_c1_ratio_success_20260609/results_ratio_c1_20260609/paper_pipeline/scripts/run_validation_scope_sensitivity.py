#!/usr/bin/env python3
"""Run and audit clean- versus noisy-validation sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence


class CommandSpec(NamedTuple):
    dataset: str
    group: str
    seed: int
    methods: tuple[str, str]
    expected_rows: int
    raw_path: Path
    immutable_path: Path
    stdout_path: Path
    stderr_path: Path
    command: tuple[str, ...]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_NOISYGL_ROOT = EXPERIMENT_ROOT / "external" / "NoisyGL"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_NOISYGL_ROOT
    / "runs"
    / "validation_scope_sensitivity_20260729"
)
PLAN_PATH = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "validation_scope_sensitivity_plan_20260729.md"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "paper_pipeline"
    / "04_experiments"
    / "validation_scope_sensitivity_report.md"
)

DATASETS = ("cora", "citeseer", "corafull")
SEEDS = (0, 1, 2, 3, 4)
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
MLP_CE = "mlp_ce"
MLP_JOCOR = "mlp_jocor"
POSTEL = "postel_mlp_noipl_rm"
OLD_GSDN = "gsdn"
SOFTDIST_METHODS = (OURS, SHUFFLED)
METHODS = SOFTDIST_METHODS + (MLP_CE, MLP_JOCOR, POSTEL, OLD_GSDN)
GROUP_METHODS = {
    "softdist": SOFTDIST_METHODS,
    "target_baselines": (MLP_CE, MLP_JOCOR),
    "graph_assisted": (POSTEL, OLD_GSDN),
}
COMPARATORS = (SHUFFLED, MLP_CE, MLP_JOCOR, POSTEL, OLD_GSDN)
TIE_MODE = "distribution_only"
FINGERPRINT_DERIVED_SEED_OFFSET = 1000003
T_CRITICAL_95_DF4 = 2.7764451051977987


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_training_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PYTHONHASHSEED": "0",
    })
    return env


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
            return list(csv.DictReader(io.StringIO(payload.decode(encoding))))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "utf-8/cp936",
        payload,
        0,
        len(payload),
        f"cannot decode CSV: {path}",
    )


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)


def build_command_specs(
        *,
        python: Path,
        noisygl_root: Path,
        output_dir: Path,
        seeds: Sequence[int] = SEEDS) -> list[CommandSpec]:
    runner = noisygl_root / "scripts" / "run_standard_downstream.py"
    specs = []
    for dataset in DATASETS:
        for group, methods in GROUP_METHODS.items():
            overrides = (
                {"gb.tie_mode": TIE_MODE}
                if group == "softdist"
                else {}
            )
            for seed in seeds:
                stem = f"{dataset}_{group}_seed{seed}"
                raw_path = output_dir / "raw" / f"{stem}.csv"
                immutable_path = (
                    output_dir / "raw_immutable" / f"{stem}.csv"
                )
                command = (
                    str(python),
                    str(runner),
                    "--datasets",
                    dataset,
                    "--methods",
                    ",".join(methods),
                    "--seeds",
                    str(seed),
                    "--noise_type",
                    "uniform",
                    "--noise_rate",
                    "0.4",
                    "--noise_scope",
                    "train_val",
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
                    str(raw_path),
                    "--stop_on_error",
                )
                specs.append(CommandSpec(
                    dataset=dataset,
                    group=group,
                    seed=int(seed),
                    methods=methods,
                    expected_rows=len(methods),
                    raw_path=raw_path,
                    immutable_path=immutable_path,
                    stdout_path=output_dir / "logs" / f"{stem}.stdout.log",
                    stderr_path=output_dir / "logs" / f"{stem}.stderr.log",
                    command=command,
                ))
    return specs


def validate_spec_rows(spec: CommandSpec, rows: Sequence[dict[str, str]]) -> None:
    if len(rows) != spec.expected_rows:
        raise ValueError(
            f"{spec.raw_path} expected {spec.expected_rows} rows, got {len(rows)}"
        )
    observed = {(row.get("method"), int(row.get("seed", -1))) for row in rows}
    expected = {(method, spec.seed) for method in spec.methods}
    if observed != expected:
        raise ValueError(f"unexpected method/seed rows in {spec.raw_path}")
    for row in rows:
        if row.get("dataset") != spec.dataset:
            raise ValueError(f"dataset mismatch in {spec.raw_path}")
        if row.get("status") != "ok":
            raise ValueError(f"failed row in {spec.raw_path}: {row.get('error')}")
        if row.get("noise_scope") != "train_val":
            raise ValueError(f"wrong noise_scope in {spec.raw_path}")
        if row.get("noise_type") != "uniform":
            raise ValueError(f"wrong noise_type in {spec.raw_path}")
        if not math.isclose(float(row.get("noise_rate", "nan")), 0.4):
            raise ValueError(f"wrong noise_rate in {spec.raw_path}")
        if row.get("method") in SOFTDIST_METHODS:
            if row.get("tie_mode") != TIE_MODE:
                raise ValueError(f"wrong tie_mode in {spec.raw_path}")


def is_exact_complete(spec: CommandSpec) -> bool:
    if not spec.raw_path.is_file():
        return False
    try:
        validate_spec_rows(spec, read_csv(spec.raw_path))
    except (ValueError, OSError, KeyError):
        return False
    return True


def freeze_raw(spec: CommandSpec) -> None:
    spec.immutable_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.immutable_path.is_file():
        if sha256_file(spec.raw_path) != sha256_file(spec.immutable_path):
            raise ValueError(f"immutable raw mismatch: {spec.immutable_path}")
        return
    shutil.copy2(spec.raw_path, spec.immutable_path)


def run_spec(spec: CommandSpec, env: dict[str, str]) -> str:
    if is_exact_complete(spec):
        freeze_raw(spec)
        return "skipped_exact_complete"
    if spec.raw_path.exists():
        raise ValueError(f"incomplete raw output already exists: {spec.raw_path}")
    spec.raw_path.parent.mkdir(parents=True, exist_ok=True)
    spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        spec.stdout_path.open("w", encoding="utf-8") as stdout,
        spec.stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            list(spec.command),
            cwd=str(Path(spec.command[1]).resolve().parents[1]),
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(spec.command)}"
        )
    rows = read_csv(spec.raw_path)
    validate_spec_rows(spec, rows)
    freeze_raw(spec)
    return "completed"


def audit_noisy_rows(
        rows: Sequence[dict[str, str]],
        *,
        seeds: Sequence[int] = SEEDS) -> list[dict[str, str]]:
    expected = len(DATASETS) * len(METHODS) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} noisy rows, got {len(rows)}")
    keys = [
        (row.get("dataset"), row.get("method"), int(row.get("seed", -1)))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dataset-method-seed rows")

    audit = []
    for dataset in DATASETS:
        for seed in seeds:
            group = [
                row
                for row in rows
                if row.get("dataset") == dataset
                and int(row.get("seed", -1)) == int(seed)
            ]
            if {row.get("method") for row in group} != set(METHODS):
                raise ValueError(f"incomplete method group: {dataset} seed {seed}")
            for field in (
                "split_hash",
                "train_noise_hash",
                "modified_train_count",
            ):
                values = {row.get(field, "") for row in group}
                if len(values) != 1 or "" in values:
                    raise ValueError(
                        f"{field} mismatch: {dataset} seed {seed}"
                    )
            for row in group:
                if row.get("status") != "ok":
                    raise ValueError(f"failed noisy-validation row: {row}")
                if int(row.get("modified_val_count", "0")) <= 0:
                    raise ValueError(
                        f"validation labels were not corrupted: {row}"
                    )
                if int(row.get("modified_test_count", "0")) != 0:
                    raise ValueError(f"test labels were corrupted: {row}")
                if row.get("method") in SOFTDIST_METHODS:
                    if row.get("tie_mode") != TIE_MODE:
                        raise ValueError(f"tie_mode mismatch: {row}")
                value = float(row.get("test", "nan"))
                if not math.isfinite(value):
                    raise ValueError(f"non-finite test metric: {row}")
            audit.append({
                "dataset": dataset,
                "seed": str(seed),
                "method_count": str(len(group)),
                "split_hash": group[0]["split_hash"],
                "train_noise_hash": group[0]["train_noise_hash"],
                "modified_train_count": group[0]["modified_train_count"],
                "modified_val_count_min": str(min(
                    int(row["modified_val_count"]) for row in group
                )),
                "modified_test_count_max": str(max(
                    int(row["modified_test_count"]) for row in group
                )),
                "status": "PASS",
            })
    return audit


def _row_matches_uniform04(row: dict[str, str], method: str) -> bool:
    try:
        rate = float(row.get("noise_rate", "nan"))
    except ValueError:
        return False
    return (
        row.get("dataset") in DATASETS
        and row.get("method") == method
        and row.get("noise_type") == "uniform"
        and math.isclose(rate, 0.4)
        and int(row.get("seed", -1)) in SEEDS
        and row.get("status") == "ok"
    )


def load_old_gsdn_raw_rows(noisygl_root: Path) -> list[dict[str, str]]:
    summary_paths = (
        noisygl_root
        / "runs"
        / "mlp14_graphfree_family_20260703"
        / "mlp14_combined_raw.csv",
        noisygl_root
        / "runs"
        / "mlp21_dataset_replacement_20260710"
        / "mlp21_raw.csv",
    )
    summary_rows = []
    for path in summary_paths:
        summary_rows.extend(read_csv(path))
    selected = [
        row for row in summary_rows if _row_matches_uniform04(row, OLD_GSDN)
    ]
    by_key = {}
    source_cache = {}
    for summary in selected:
        key = (summary["dataset"], int(summary["seed"]))
        source = Path(summary["source"])
        if source not in source_cache:
            source_cache[source] = read_csv(source)
        matches = [
            row
            for row in source_cache[source]
            if _row_matches_uniform04(row, OLD_GSDN)
            and row["dataset"] == key[0]
            and int(row["seed"]) == key[1]
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot trace old GSDN raw row: {key}")
        by_key[key] = matches[0]
    if len(by_key) != len(DATASETS) * len(SEEDS):
        raise ValueError(
            f"expected 15 traced old GSDN rows, got {len(by_key)}"
        )
    return [by_key[key] for key in sorted(by_key)]


def load_clean_reference_rows(
        *,
        noisygl_root: Path,
        fingerprint_rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    fingerprint_map = {
        (row["dataset"], int(row["seed"])): row for row in fingerprint_rows
    }
    tie_rows = read_csv(
        noisygl_root
        / "runs"
        / "tie_safe_full_grid_20260727_v2_immutable"
        / "raw_combined.csv"
    )
    nearest_rows = read_csv(
        noisygl_root
        / "runs"
        / "nearest_work_full_grid_20260728"
        / "five_method_raw_combined.csv"
    )
    selected = [
        row
        for row in tie_rows
        if any(_row_matches_uniform04(row, method) for method in (
            OURS,
            SHUFFLED,
            MLP_CE,
            MLP_JOCOR,
        ))
    ]
    selected.extend(
        row for row in nearest_rows if _row_matches_uniform04(row, POSTEL)
    )
    selected.extend(load_old_gsdn_raw_rows(noisygl_root))
    if len(selected) != len(DATASETS) * len(METHODS) * len(SEEDS):
        raise ValueError(f"expected 90 clean reference rows, got {len(selected)}")

    normalized = []
    for row in selected:
        key = (row["dataset"], int(row["seed"]))
        fingerprint = fingerprint_map.get(key)
        if fingerprint is None:
            raise ValueError(f"missing protocol fingerprint: {key}")
        if row.get("split_hash") != fingerprint.get("split_hash"):
            raise ValueError(f"legacy split hash mismatch: {key}")
        if row.get("noise_hash") != fingerprint.get("noise_hash"):
            raise ValueError(f"legacy noise hash mismatch: {key}")
        clean = dict(row)
        clean.update({
            "protocol": "clean_validation",
            "train_noise_hash": fingerprint["train_noise_hash"],
            "val_noise_hash": fingerprint["val_noise_hash"],
            "train_noise_hash_provenance": (
                "deterministically regenerated and matched to legacy full hash"
            ),
            "implementation_status": (
                "provisional adapted implementation"
                if row["method"] == OLD_GSDN
                else row.get("implementation_status", "")
            ),
        })
        normalized.append(clean)
    return sorted(
        normalized,
        key=lambda row: (
            row["dataset"],
            METHODS.index(row["method"]),
            int(row["seed"]),
        ),
    )


def aggregate_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped = {}
    for row in rows:
        key = (row["protocol"], row["dataset"], row["method"])
        grouped.setdefault(key, []).append(float(row["test"]))
    output = []
    for (protocol, dataset, method), values in sorted(grouped.items()):
        output.append({
            "protocol": protocol,
            "dataset": dataset,
            "method": method,
            "n": str(len(values)),
            "test_mean": f"{statistics.mean(values):.12g}",
            "test_sample_std": (
                f"{statistics.stdev(values):.12g}"
                if len(values) > 1
                else "0"
            ),
        })
    return output


def paired_rows(rows: Sequence[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    lookup = {
        (
            row["protocol"],
            row["dataset"],
            row["method"],
            int(row["seed"]),
        ): float(row["test"])
        for row in rows
    }
    deltas = []
    tests = []
    protocols = ("clean_validation", "noisy_validation")
    for protocol in protocols:
        for dataset in DATASETS:
            for comparator in COMPARATORS:
                values = []
                for seed in SEEDS:
                    ours = lookup[(protocol, dataset, OURS, seed)]
                    baseline = lookup[(protocol, dataset, comparator, seed)]
                    delta = ours - baseline
                    values.append(delta)
                    deltas.append({
                        "protocol": protocol,
                        "dataset": dataset,
                        "ours": OURS,
                        "baseline": comparator,
                        "seed": str(seed),
                        "delta": f"{delta:.12g}",
                        "delta_pp": f"{100.0 * delta:.12g}",
                    })
                mean = statistics.mean(values)
                std = statistics.stdev(values)
                half_width = T_CRITICAL_95_DF4 * std / math.sqrt(len(values))
                tests.append({
                    "protocol": protocol,
                    "dataset": dataset,
                    "ours": OURS,
                    "baseline": comparator,
                    "n": str(len(values)),
                    "mean_delta_pp": f"{100.0 * mean:.12g}",
                    "sample_std_pp": f"{100.0 * std:.12g}",
                    "ci95_low_pp": f"{100.0 * (mean - half_width):.12g}",
                    "ci95_high_pp": f"{100.0 * (mean + half_width):.12g}",
                    "positive_seeds": str(sum(value > 0 for value in values)),
                })
    return deltas, tests


def kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    position_a = {value: index for index, value in enumerate(order_a)}
    position_b = {value: index for index, value in enumerate(order_b)}
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1:]:
            a = position_a[left] - position_a[right]
            b = position_b[left] - position_b[right]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0


def protocol_interactions(
        aggregate: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    lookup = {
        (row["protocol"], row["dataset"], row["method"]):
        float(row["test_mean"])
        for row in aggregate
    }
    output = []
    for dataset in DATASETS:
        clean_order = sorted(
            METHODS,
            key=lambda method: lookup[
                ("clean_validation", dataset, method)
            ],
            reverse=True,
        )
        noisy_order = sorted(
            METHODS,
            key=lambda method: lookup[
                ("noisy_validation", dataset, method)
            ],
            reverse=True,
        )
        tau = kendall_tau(clean_order, noisy_order)
        top2_overlap = len(set(clean_order[:2]) & set(noisy_order[:2]))
        for method in METHODS:
            clean = lookup[("clean_validation", dataset, method)]
            noisy = lookup[("noisy_validation", dataset, method)]
            output.append({
                "dataset": dataset,
                "method": method,
                "clean_validation_mean": f"{clean:.12g}",
                "noisy_validation_mean": f"{noisy:.12g}",
                "noisy_minus_clean_pp": f"{100.0 * (noisy - clean):.12g}",
                "kendall_tau": f"{tau:.12g}",
                "top2_overlap": str(top2_overlap),
                "clean_rank": str(clean_order.index(method) + 1),
                "noisy_rank": str(noisy_order.index(method) + 1),
            })
    return output


def evaluate_gates(paired_tests: Sequence[dict[str, str]]) -> dict[str, object]:
    noisy = [
        row for row in paired_tests if row["protocol"] == "noisy_validation"
    ]
    by_comparator = {}
    for row in noisy:
        by_comparator.setdefault(row["baseline"], []).append(row)
    shuffled = by_comparator[SHUFFLED]
    mlp_ce = by_comparator[MLP_CE]
    gate_shuffled = all(
        float(row["mean_delta_pp"]) > 0
        and int(row["positive_seeds"]) >= 4
        for row in shuffled
    )
    gate_mlp_ce = sum(
        float(row["mean_delta_pp"]) > 0
        and int(row["positive_seeds"]) >= 4
        for row in mlp_ce
    ) >= 2
    gate_floor = all(
        float(row["mean_delta_pp"]) >= -0.5 for row in shuffled
    )
    return {
        "gb_vs_shuffled_all_three": gate_shuffled,
        "gb_vs_mlp_ce_two_of_three": gate_mlp_ce,
        "gb_vs_shuffled_floor": gate_floor,
        "primary_pass": gate_shuffled and gate_mlp_ce and gate_floor,
    }


def write_report(
        path: Path,
        *,
        aggregate: Sequence[dict[str, str]],
        paired_tests: Sequence[dict[str, str]],
        interactions: Sequence[dict[str, str]],
        gates: dict[str, object],
        row_count: int) -> None:
    lines = [
        "# 验证标签噪声敏感性实验报告",
        "",
        "## 完整性",
        "",
        f"- noisy-validation 正式行：{row_count}/90。",
        "- 测试标签污染：0。",
        "- 旧 GSDN 身份：provisional adapted implementation。",
        "",
        "## 预注册判定",
        "",
        f"- GB-LCSS vs shuffled 三数据集：{gates['gb_vs_shuffled_all_three']}。",
        f"- GB-LCSS vs MLP+CE 至少二数据集：{gates['gb_vs_mlp_ce_two_of_three']}。",
        f"- shuffled 下界：{gates['gb_vs_shuffled_floor']}。",
        f"- 主要机制结论：{'通过' if gates['primary_pass'] else '未通过'}。",
        "",
        "## 配对结果",
        "",
        "| 协议 | 数据集 | 对照 | 均值差(pp) | 正向 seeds |",
        "|---|---|---|---:|---:|",
    ]
    for row in paired_tests:
        if row["baseline"] not in (SHUFFLED, MLP_CE, POSTEL, OLD_GSDN):
            continue
        lines.append(
            f"| {row['protocol']} | {row['dataset']} | {row['baseline']} | "
            f"{float(row['mean_delta_pp']):.2f} | {row['positive_seeds']}/5 |"
        )
    lines.extend([
        "",
        "## 排序敏感性",
        "",
        "| 数据集 | Kendall tau | 前二重合数 |",
        "|---|---:|---:|",
    ])
    seen = set()
    for row in interactions:
        dataset = row["dataset"]
        if dataset in seen:
            continue
        seen.add(dataset)
        lines.append(
            f"| {dataset} | {float(row['kendall_tau']):.3f} | "
            f"{row['top2_overlap']}/2 |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "本实验检验相对排序对验证标签污染的敏感性，不把绝对精度下降直接解释为方法失效。旧 `gsdn` 只作临时 adapted 参照，不能支持对 GSDN 的最终强弱结论。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_fingerprint_export(
        *,
        python: Path,
        noisygl_root: Path,
        output_dir: Path) -> Path:
    output = output_dir / "protocol_fingerprints_train_only.csv"
    command = [
        str(python),
        str(noisygl_root / "scripts" / "export_noise_fingerprints.py"),
        "--datasets",
        ",".join(DATASETS),
        "--seeds",
        ",".join(map(str, SEEDS)),
        "--noise_type",
        "uniform",
        "--noise_rate",
        "0.4",
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
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=str(noisygl_root),
        env=deterministic_training_env(),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("protocol fingerprint export failed")
    rows = read_csv(output)
    if len(rows) != 15:
        raise ValueError(f"expected 15 protocol fingerprints, got {len(rows)}")
    return output


def write_manifest(
        path: Path,
        specs: Sequence[CommandSpec],
        statuses: dict[tuple[str, str, int], str]) -> None:
    rows = []
    plan_sha = sha256_file(PLAN_PATH) if PLAN_PATH.is_file() else ""
    for spec in specs:
        key = (spec.dataset, spec.group, spec.seed)
        rows.append({
            "dataset": spec.dataset,
            "group": spec.group,
            "seed": str(spec.seed),
            "methods": ",".join(spec.methods),
            "expected_rows": str(spec.expected_rows),
            "raw_path": str(spec.raw_path),
            "immutable_path": str(spec.immutable_path),
            "command_json": json.dumps(spec.command),
            "status": statuses.get(key, "planned"),
            "raw_sha256": (
                sha256_file(spec.raw_path) if spec.raw_path.is_file() else ""
            ),
            "immutable_sha256": (
                sha256_file(spec.immutable_path)
                if spec.immutable_path.is_file()
                else ""
            ),
            "plan_sha256": plan_sha,
        })
    write_csv(path, rows)


def combine_immutable(specs: Sequence[CommandSpec]) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        if not spec.immutable_path.is_file():
            raise ValueError(f"missing immutable output: {spec.immutable_path}")
        source_rows = read_csv(spec.immutable_path)
        validate_spec_rows(spec, source_rows)
        rows.extend(source_rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--noisygl-root",
        type=Path,
        default=DEFAULT_NOISYGL_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    )
    if not set(seeds).issubset(set(SEEDS)):
        raise ValueError(f"seeds must be a subset of {SEEDS}")
    specs = build_command_specs(
        python=args.python,
        noisygl_root=args.noisygl_root,
        output_dir=args.output_dir,
        seeds=seeds,
    )
    statuses = {}
    if args.dry_run:
        write_manifest(
            args.output_dir / "command_manifest.csv",
            specs,
            statuses,
        )
        print(
            f"DRY RUN planned_commands={len(specs)} "
            f"planned_rows={sum(spec.expected_rows for spec in specs)}"
        )
        return 0

    fingerprint_path = run_fingerprint_export(
        python=args.python,
        noisygl_root=args.noisygl_root,
        output_dir=args.output_dir,
    )
    env = deterministic_training_env()
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] {spec.dataset} "
            f"{spec.group} seed={spec.seed}",
            flush=True,
        )
        status = run_spec(spec, env)
        statuses[(spec.dataset, spec.group, spec.seed)] = status
        write_manifest(
            args.output_dir / "command_manifest.csv",
            specs,
            statuses,
        )

    noisy_rows = combine_immutable(specs)
    audit = audit_noisy_rows(noisy_rows, seeds=seeds)
    for row in noisy_rows:
        row["protocol"] = "noisy_validation"
        if row["method"] == OLD_GSDN:
            row["implementation_status"] = (
                "provisional adapted implementation"
            )
    write_csv(args.output_dir / "raw_combined.csv", noisy_rows)
    write_csv(args.output_dir / "protocol_audit.csv", audit)
    write_csv(
        args.output_dir / "immutable_raw_manifest.csv",
        [{
            "path": str(spec.immutable_path),
            "sha256": sha256_file(spec.immutable_path),
        } for spec in specs],
    )

    if set(seeds) == set(SEEDS):
        fingerprints = read_csv(fingerprint_path)
        clean_rows = load_clean_reference_rows(
            noisygl_root=args.noisygl_root,
            fingerprint_rows=fingerprints,
        )
        write_csv(args.output_dir / "clean_reference_rows.csv", clean_rows)
        combined = clean_rows + noisy_rows
        aggregate = aggregate_rows(combined)
        paired_deltas, paired_tests = paired_rows(combined)
        interactions = protocol_interactions(aggregate)
        gates = evaluate_gates(paired_tests)
        write_csv(args.output_dir / "combined_protocol_rows.csv", combined)
        write_csv(args.output_dir / "aggregate.csv", aggregate)
        write_csv(args.output_dir / "paired_deltas.csv", paired_deltas)
        write_csv(args.output_dir / "paired_tests.csv", paired_tests)
        write_csv(
            args.output_dir / "protocol_interactions.csv",
            interactions,
        )
        write_csv(
            args.output_dir / "completeness.csv",
            [{
                "protocol": "noisy_validation",
                "expected_rows": "90",
                "actual_rows": str(len(noisy_rows)),
                "status": "PASS" if len(noisy_rows) == 90 else "FAIL",
            }, {
                "protocol": "clean_validation",
                "expected_rows": "90",
                "actual_rows": str(len(clean_rows)),
                "status": "PASS" if len(clean_rows) == 90 else "FAIL",
            }],
        )
        write_report(
            REPORT_PATH,
            aggregate=aggregate,
            paired_tests=paired_tests,
            interactions=interactions,
            gates=gates,
            row_count=len(noisy_rows),
        )
        shutil.copy2(
            REPORT_PATH,
            args.output_dir / "validation_scope_sensitivity_report.md",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
