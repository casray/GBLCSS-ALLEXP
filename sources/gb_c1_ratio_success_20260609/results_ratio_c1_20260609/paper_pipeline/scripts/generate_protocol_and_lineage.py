#!/usr/bin/env python3
"""Generate immutable protocol and manuscript-table lineage assets.

This script only reads existing CSV/Markdown assets. It does not train models,
modify experiment results, or update the material/claim ledgers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GAP = "GAP"
NA = "NOT_APPLICABLE"
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
FORMAL_BASELINES = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_sce",
    "mlp_elr",
    "mlp_logitclip",
)
PRIMARY_METHOD_MAP = {
    OURS: "lcss",
    SHUFFLED: "shuffled",
    "mlp_ce": "ce",
    "mlp_jocor": "jocor",
}
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
    "pubmed": "PubMed",
    "amazoncom": "Amazon-Computers",
    "coraml": "Cora_ML",
    "attribwiki": "Attributed-Wiki",
    "citeseerfull": "CiteSeerFull",
    "coauthorph": "Coauthor-Physics",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def normalized_rate(value: Any) -> str:
    return f"{float(value):g}"


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row["dataset"]).lower(),
        str(row["noise_type"]).lower(),
        normalized_rate(row["noise_rate"]),
        str(row["method"]),
        int(row["seed"]),
    )


def source_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row["dataset"]).lower(),
        str(row.get("noise_type", "uniform")).lower(),
        normalized_rate(row.get("noise_rate", "0.4")),
        str(row["method"]),
        int(row["seed"]),
    )


def source_test(row: dict[str, str]) -> float:
    for field in ("test", "test_acc", "student_test"):
        if str(row.get(field, "")).strip():
            return float(row[field])
    raise ValueError(f"source row has no test field: {row}")


def direct(row: dict[str, str], field: str) -> str:
    value = str(row.get(field, "")).strip()
    return value if value else GAP


def exact_filter(
    dataset: str,
    noise_type: str,
    noise_rate: str | float,
    method: str,
    seed: int | str | None = None,
) -> str:
    parts = [
        f'dataset=="{dataset}"',
        f'noise_type=="{noise_type}"',
        f"noise_rate=={normalized_rate(noise_rate)}",
        f'method=="{method}"',
    ]
    if seed is not None:
        parts.append(f"seed=={int(seed)}")
    return " AND ".join(parts)


def canonical_path(path: Path, workspace_root: Path, noisy_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace_root).as_posix()
    except ValueError:
        pass
    try:
        return (
            "../../external/NoisyGL/"
            + resolved.relative_to(noisy_root).as_posix()
        )
    except ValueError:
        return str(resolved)


def clean_cell(value: str) -> str:
    return value.strip().replace("**", "")


def parse_markdown_table(manuscript: Path, table_id: int) -> tuple[list[str], list[dict[str, str]]]:
    text = manuscript.read_text(encoding="utf-8")
    marker = f"**表 {table_id}**"
    if marker not in text:
        raise ValueError(f"manuscript table marker not found: {marker}")
    lines = text.split(marker, 1)[1].splitlines()
    table_lines: list[str] = []
    started = False
    for line in lines:
        if line.lstrip().startswith("|"):
            started = True
            table_lines.append(line.strip())
        elif started:
            break
    if len(table_lines) < 3:
        raise ValueError(f"malformed markdown table: {marker}")

    def cells(line: str) -> list[str]:
        return [clean_cell(item) for item in line.strip("|").split("|")]

    headers = cells(table_lines[0])
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = cells(line)
        if len(values) != len(headers):
            raise ValueError(f"malformed row in {marker}: {line}")
        rows.append(dict(zip(headers, values)))
    return headers, rows


def stats(rows: Iterable[dict[str, Any]], value_field: str = "test") -> tuple[int, float, float]:
    values = [float(row[value_field]) for row in rows]
    if not values:
        raise ValueError("cannot aggregate zero rows")
    return (
        len(values),
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def pct_stats(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}±{100.0 * std:.2f}"


def signed_pp(value: float) -> str:
    rendered = f"{100.0 * value:+.2f}"
    return "−" + rendered[1:] if rendered.startswith("-") else rendered


def import_audit_module(noisy_root: Path):
    path = noisy_root / "scripts" / "audit_submission_integrity_20260710.py"
    spec = importlib.util.spec_from_file_location("submission_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import audit module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_protocol_ledger(
    workspace_root: Path,
    noisy_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_path = (
        noisy_root
        / "runs"
        / "phase4_faithful_claims_20260712"
        / "phase4_selected_raw.csv"
    )
    primary_path = (
        noisy_root
        / "runs"
        / "mlp36_review_defense_20260715"
        / "primary_statistics_raw.csv"
    )
    selected_rows = read_csv(selected_path)
    primary_rows = read_csv(primary_path)
    primary_index = {
        (
            row["dataset"].lower(),
            row["noise_type"].lower(),
            normalized_rate(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        ): row
        for row in primary_rows
    }
    if len(primary_index) != len(primary_rows):
        raise ValueError("duplicate primary_statistics_raw unique key")

    selected_keys = [row_key(row) for row in selected_rows]
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError("duplicate phase4_selected_raw unique key")

    source_cache: dict[Path, tuple[list[dict[str, str]], dict[tuple[Any, ...], list[dict[str, str]]]]] = {}
    selected_hash = sha256(selected_path)
    ledger: list[dict[str, Any]] = []
    primary_matches = 0

    for index, selected in enumerate(selected_rows, 1):
        source_path = (noisy_root / selected["source"]).resolve()
        if source_path not in source_cache:
            source_rows = read_csv(source_path)
            source_index: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
            for row in source_rows:
                source_index[source_row_key(row)].append(row)
            source_cache[source_path] = (source_rows, source_index)
        _, source_index = source_cache[source_path]
        key = row_key(selected)
        candidates = source_index.get(key, [])
        target_test = float(selected["test"])
        matches = [row for row in candidates if abs(source_test(row) - target_test) <= 1e-12]
        if len(matches) != 1:
            raise ValueError(
                f"expected one source match for {key}, found {len(matches)} in {source_path}"
            )
        source = matches[0]

        dataset, noise_type, noise_rate, method, seed = key
        primary_method = PRIMARY_METHOD_MAP.get(method, NA)
        in_primary = primary_method != NA
        primary_match = NA
        if in_primary:
            primary_key = (dataset, noise_type, noise_rate, primary_method, seed)
            primary = primary_index.get(primary_key)
            if primary is None:
                raise ValueError(f"missing primary statistics row: {primary_key}")
            primary_match = str(abs(float(primary["test"]) - target_test) <= 1e-12).upper()
            if primary_match != "TRUE":
                raise ValueError(f"primary statistics test mismatch: {primary_key}")
            primary_matches += 1

        fields = {
            field: direct(source, field)
            for field in (
                "noise_scope",
                "split_hash",
                "noise_hash",
                "config_id",
                "modified_count",
                "modified_train_count",
                "modified_val_count",
                "modified_test_count",
                "actual_train_noise_rate",
            )
        }
        if noise_type == "instance":
            instance_realization_hash = fields["noise_hash"]
            instance_generator_seed = GAP
            instance_generator_hash = GAP
        else:
            instance_realization_hash = NA
            instance_generator_seed = NA
            instance_generator_hash = NA

        required = (
            "noise_scope",
            "split_hash",
            "noise_hash",
            "modified_train_count",
            "modified_val_count",
            "modified_test_count",
            "actual_train_noise_rate",
        )
        required_gaps = [field for field in required if fields[field] == GAP]
        if noise_type == "instance" and instance_realization_hash == GAP:
            required_gaps.append("instance_realization_hash")
        additional_gaps = []
        if fields["config_id"] == GAP:
            additional_gaps.append("config_id")
        if noise_type == "instance":
            additional_gaps.extend(("instance_generator_seed", "instance_generator_hash"))

        ledger.append(
            {
                "protocol_row_id": f"PR-{index:04d}",
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "seed": seed,
                "method": method,
                "primary_statistics_method": primary_method,
                "in_primary_statistics": str(in_primary).upper(),
                "test": selected["test"],
                **fields,
                "instance_generator_seed": instance_generator_seed,
                "instance_generator_hash": instance_generator_hash,
                "instance_realization_hash": instance_realization_hash,
                "selection_path": canonical_path(selected_path, workspace_root, noisy_root),
                "selection_size_bytes": selected_path.stat().st_size,
                "selection_sha256": selected_hash,
                "source_path": canonical_path(source_path, workspace_root, noisy_root),
                "source_size_bytes": source_path.stat().st_size,
                "source_sha256": sha256(source_path),
                "source_row_filter": exact_filter(
                    dataset, noise_type, noise_rate, method, seed
                ),
                "value_formula": "direct copy of exact source-row test; no aggregation",
                "source_test_match": "TRUE",
                "primary_statistics_test_match": primary_match,
                "required_gap_fields": ";".join(required_gaps) if required_gaps else "",
                "additional_gap_fields": ";".join(additional_gaps) if additional_gaps else "",
                "protocol_status": (
                    "PARTIAL_GAP" if required_gaps else "CLOSED_REQUIRED_FIELDS"
                ),
            }
        )

    if primary_matches != len(primary_rows):
        raise ValueError(
            f"primary statistics coverage mismatch: {primary_matches}/{len(primary_rows)}"
        )

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        grouped[
            (
                row["dataset"],
                row["noise_type"],
                row["noise_rate"],
                int(row["seed"]),
            )
        ].append(row)
    hash_mismatches = 0
    for rows in grouped.values():
        for field in ("split_hash", "noise_hash"):
            values = {row[field] for row in rows if row[field] != GAP}
            if len(values) > 1:
                hash_mismatches += 1
    if hash_mismatches:
        raise ValueError(f"paired hash mismatches: {hash_mismatches}")

    metrics = {
        "rows": len(ledger),
        "primary_rows": sum(row["in_primary_statistics"] == "TRUE" for row in ledger),
        "closed_rows": sum(
            row["protocol_status"] == "CLOSED_REQUIRED_FIELDS" for row in ledger
        ),
        "closed_primary_rows": sum(
            row["in_primary_statistics"] == "TRUE"
            and row["protocol_status"] == "CLOSED_REQUIRED_FIELDS"
            for row in ledger
        ),
        "split_noise_rows": sum(
            row["split_hash"] != GAP and row["noise_hash"] != GAP for row in ledger
        ),
        "pollution_count_rows": sum(
            all(
                row[field] != GAP
                for field in (
                    "modified_train_count",
                    "modified_val_count",
                    "modified_test_count",
                )
            )
            for row in ledger
        ),
        "instance_rows": sum(row["noise_type"] == "instance" for row in ledger),
        "instance_realization_rows": sum(
            row["noise_type"] == "instance"
            and row["instance_realization_hash"] != GAP
            for row in ledger
        ),
        "instance_generator_rows": sum(
            row["noise_type"] == "instance"
            and row["instance_generator_seed"] != GAP
            for row in ledger
        ),
        "hash_mismatches": hash_mismatches,
    }
    setting_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        setting_groups[(row["dataset"], row["noise_type"], row["noise_rate"])].append(row)
    metrics["settings"] = len(setting_groups)
    metrics["closed_settings"] = sum(
        all(row["protocol_status"] == "CLOSED_REQUIRED_FIELDS" for row in rows)
        for rows in setting_groups.values()
    )
    metrics["split_noise_settings"] = sum(
        all(row["split_hash"] != GAP and row["noise_hash"] != GAP for row in rows)
        for rows in setting_groups.values()
    )
    metrics["pollution_count_settings"] = sum(
        all(
            row["modified_train_count"] != GAP
            and row["modified_val_count"] != GAP
            and row["modified_test_count"] != GAP
            for row in rows
        )
        for rows in setting_groups.values()
    )
    return ledger, metrics


def source_provenance(
    rows: list[dict[str, Any]],
    workspace_root: Path,
    noisy_root: Path,
) -> tuple[str, str, str]:
    paths = sorted({Path(str(row["source"])).resolve() for row in rows}, key=str)
    return (
        " | ".join(canonical_path(path, workspace_root, noisy_root) for path in paths),
        " | ".join(sha256(path) for path in paths),
        " | ".join(str(path.stat().st_size) for path in paths),
    )


def lineage_record(
    *,
    table_id: int,
    external_table_id: str,
    row_label: str,
    column_label: str,
    dataset: str,
    noise_type: str,
    noise_rate: str,
    method: str,
    cell_kind: str,
    unit: str,
    manuscript_value: str,
    computed_value: str,
    n: str | int,
    mean: str | float,
    std: str | float,
    raw_path: str,
    raw_sha256: str,
    raw_size_bytes: str,
    filter_expression: str,
    aggregation_formula: str,
    aggregate_path: str = NA,
    aggregate_sha256: str = NA,
    aggregate_filter_expression: str = NA,
    upstream_path: str = NA,
    upstream_sha256: str = NA,
    protocol_note: str = "",
    gap_reason: str = "",
) -> dict[str, Any]:
    if raw_path == GAP:
        status = "GAP"
        match_status = NA
    else:
        match_status = "MATCH" if manuscript_value == computed_value else "MISMATCH"
        status = "CLOSED" if match_status == "MATCH" else "MISMATCH"
    return {
        "manuscript_table_id": f"Table {table_id}",
        "external_table_id": external_table_id,
        "cell_id": f"T{table_id}:{row_label}:{column_label}",
        "row_label": row_label,
        "column_label": column_label,
        "dataset": dataset,
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "method": method,
        "cell_kind": cell_kind,
        "unit": unit,
        "manuscript_value": manuscript_value,
        "computed_value": computed_value,
        "n": n,
        "mean": mean,
        "std": std,
        "raw_path": raw_path,
        "raw_sha256": raw_sha256,
        "raw_size_bytes": raw_size_bytes,
        "filter_expression": filter_expression,
        "aggregation_formula": aggregation_formula,
        "aggregate_path": aggregate_path,
        "aggregate_sha256": aggregate_sha256,
        "aggregate_filter_expression": aggregate_filter_expression,
        "upstream_path": upstream_path,
        "upstream_sha256": upstream_sha256,
        "protocol_note": protocol_note,
        "gap_reason": gap_reason,
        "match_status": match_status,
        "lineage_status": status,
    }


def build_table_lineage(
    workspace_root: Path,
    noisy_root: Path,
    manuscript: Path,
    protocol_ledger: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase4_path = (
        noisy_root
        / "runs"
        / "phase4_faithful_claims_20260712"
        / "phase4_selected_raw.csv"
    )
    phase4_rows = read_csv(phase4_path)
    protocol_index = {
        (
            row["dataset"],
            row["noise_type"],
            row["noise_rate"],
            row["method"],
            int(row["seed"]),
        ): row
        for row in protocol_ledger
    }
    core_aggregate = (
        noisy_root
        / "runs"
        / "submission_integrity_v12_20260712"
        / "core_main_recomputed.csv"
    )
    table3_aggregate = core_aggregate.parent / "table3_recomputed.csv"
    table5_aggregate = core_aggregate.parent / "table4_recomputed.csv"
    table6_aggregate = core_aggregate.parent / "table5_recomputed.csv"

    records: list[dict[str, Any]] = []

    # Table 1
    _, manuscript_rows = parse_markdown_table(manuscript, 1)
    manuscript_by_method = {row["方法"]: row for row in manuscript_rows}
    method_map = {
        "MLP+CE": "mlp_ce",
        "Small-loss": "mlp_small_loss",
        "Co-teaching [17]": "mlp_coteaching",
        "JoCoR [18]": "mlp_jocor",
        "SCE [19]": "mlp_sce",
        "ELR [20]": "mlp_elr",
        "LogitClip [22]": "mlp_logitclip",
        "随机化分组": SHUFFLED,
        "GB-LCSS": OURS,
    }
    label_to_dataset = {
        "Cora": "cora",
        "CiteSeer": "citeseer",
        "CoraFull": "corafull",
        "Coauthor-CS": "coauthorcs",
        "Amazon-Photo": "amazonpho",
    }
    for row_label, method in method_map.items():
        manuscript_row = manuscript_by_method[row_label]
        for column_label, dataset in label_to_dataset.items():
            seed_rows = [
                row
                for row in phase4_rows
                if row["dataset"] == dataset
                and row["noise_type"] == "uniform"
                and normalized_rate(row["noise_rate"]) == "0.4"
                and row["method"] == method
            ]
            n, mean, std = stats(seed_rows)
            computed = pct_stats(mean, std)
            source_paths = sorted(
                {
                    protocol_index[
                        (dataset, "uniform", "0.4", method, int(row["seed"]))
                    ]["source_path"]
                    for row in seed_rows
                }
            )
            source_hashes = sorted(
                {
                    protocol_index[
                        (dataset, "uniform", "0.4", method, int(row["seed"]))
                    ]["source_sha256"]
                    for row in seed_rows
                }
            )
            protocol_gaps = sorted(
                {
                    protocol_index[
                        (dataset, "uniform", "0.4", method, int(row["seed"]))
                    ]["required_gap_fields"]
                    for row in seed_rows
                    if protocol_index[
                        (dataset, "uniform", "0.4", method, int(row["seed"]))
                    ]["required_gap_fields"]
                }
            )
            records.append(
                lineage_record(
                    table_id=1,
                    external_table_id="core_main_recomputed.csv / phase4_selected_raw.csv",
                    row_label=row_label,
                    column_label=column_label,
                    dataset=dataset,
                    noise_type="uniform",
                    noise_rate="0.4",
                    method=method,
                    cell_kind="mean_std",
                    unit="test accuracy (%)",
                    manuscript_value=manuscript_row[column_label],
                    computed_value=computed,
                    n=n,
                    mean=mean,
                    std=std,
                    raw_path=canonical_path(phase4_path, workspace_root, noisy_root),
                    raw_sha256=sha256(phase4_path),
                    raw_size_bytes=str(phase4_path.stat().st_size),
                    filter_expression=exact_filter(dataset, "uniform", "0.4", method),
                    aggregation_formula=(
                        "mean=100*arithmetic_mean(test); "
                        "std=100*sample_stdev(test,ddof=1); format .2f±.2f"
                    ),
                    aggregate_path=canonical_path(
                        core_aggregate, workspace_root, noisy_root
                    ),
                    aggregate_sha256=sha256(core_aggregate),
                    aggregate_filter_expression=(
                        f'dataset=="{dataset}" AND noise_type=="uniform" '
                        f'AND noise_rate==0.4 AND method=="{method}"'
                    ),
                    upstream_path=" | ".join(source_paths),
                    upstream_sha256=" | ".join(source_hashes),
                    protocol_note=(
                        "protocol ledger GAP: " + " | ".join(protocol_gaps)
                        if protocol_gaps
                        else "protocol ledger required fields closed"
                    ),
                )
            )

    # Table 2
    _, manuscript_rows = parse_markdown_table(manuscript, 2)
    breadth_sources = {
        "pubmed": (
            noisy_root
            / "runs"
            / "mlp10_baseline_refresh_20260702"
            / "mlp10_combined_raw.csv",
            "test",
            True,
        ),
        "amazoncom": (
            noisy_root
            / "runs"
            / "mlp21_dataset_replacement_20260710"
            / "mlp21_raw.csv",
            "test",
            True,
        ),
        "coraml": (
            noisy_root
            / "runs"
            / "mlp23_dataset_breadth_20260710"
            / "mlp23_raw.csv",
            "test",
            False,
        ),
        "attribwiki": (
            noisy_root
            / "runs"
            / "mlp23_dataset_breadth_20260710"
            / "mlp23_raw.csv",
            "test",
            False,
        ),
        "citeseerfull": (
            noisy_root
            / "runs"
            / "mlp23_dataset_breadth_20260710"
            / "mlp23_raw.csv",
            "test",
            False,
        ),
        "coauthorph": (
            noisy_root
            / "runs"
            / "mlp23_dataset_breadth_20260710"
            / "mlp23_raw.csv",
            "test",
            False,
        ),
    }
    dataset_by_label = {label: dataset for dataset, label in DATASET_LABELS.items()}
    method_display = {
        "mlp_ce": "MLP+CE",
        "mlp_small_loss": "Small-loss",
        "mlp_coteaching": "Co-teaching",
        "mlp_jocor": "JoCoR",
        "mlp_sce": "SCE",
        "mlp_elr": "ELR",
        "mlp_logitclip": "LogitClip",
    }
    for manuscript_row in manuscript_rows:
        dataset = dataset_by_label[manuscript_row["数据集"]]
        source_path, value_field, has_noise_fields = breadth_sources[dataset]
        source_rows = read_csv(source_path)

        def select(method: str) -> list[dict[str, str]]:
            rows = [
                row
                for row in source_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            if has_noise_fields:
                rows = [
                    row
                    for row in rows
                    if row.get("noise_type", "uniform") == "uniform"
                    and normalized_rate(row.get("noise_rate", "0.4")) == "0.4"
                ]
            return rows

        ours_rows = select(OURS)
        shuffled_rows = select(SHUFFLED)
        baseline_stats = {
            method: stats(select(method), value_field) for method in FORMAL_BASELINES
        }
        missing_candidates = [
            method for method, result in baseline_stats.items() if result[0] != 5
        ]
        if missing_candidates:
            raise ValueError(
                f"Table 2 formal candidate coverage gap in {dataset}: {missing_candidates}"
            )
        best_method, (best_n, best_mean, best_std) = max(
            baseline_stats.items(), key=lambda item: item[1][1]
        )
        ours_n, ours_mean, ours_std = stats(ours_rows, value_field)
        shuffled_n, shuffled_mean, shuffled_std = stats(shuffled_rows, value_field)
        common = {
            "table_id": 2,
            "external_table_id": "breadth seed-level raw assets",
            "row_label": manuscript_row["数据集"],
            "dataset": dataset,
            "noise_type": "uniform",
            "noise_rate": "0.4",
            "raw_path": canonical_path(source_path, workspace_root, noisy_root),
            "raw_sha256": sha256(source_path),
            "raw_size_bytes": str(source_path.stat().st_size),
            "aggregate_path": NA,
            "aggregate_sha256": NA,
            "aggregate_filter_expression": NA,
            "protocol_note": (
                "raw file has explicit noise_type/noise_rate"
                if has_noise_fields
                else "GAP: mlp23_raw.csv has no noise_type/noise_rate columns"
            ),
        }
        base_filter = f'dataset=="{dataset}"'
        if has_noise_fields:
            base_filter += ' AND noise_type=="uniform" AND noise_rate==0.4'
        records.extend(
            [
                lineage_record(
                    **common,
                    column_label="GB-LCSS",
                    method=OURS,
                    cell_kind="mean_std",
                    unit="test accuracy (%)",
                    manuscript_value=manuscript_row["GB-LCSS"],
                    computed_value=pct_stats(ours_mean, ours_std),
                    n=ours_n,
                    mean=ours_mean,
                    std=ours_std,
                    filter_expression=base_filter + f' AND method=="{OURS}"',
                    aggregation_formula=(
                        f"mean=100*arithmetic_mean({value_field}); "
                        f"std=100*sample_stdev({value_field},ddof=1); format .2f±.2f"
                    ),
                ),
                lineage_record(
                    **common,
                    column_label="最强对照",
                    method=best_method,
                    cell_kind="selected_mean_std",
                    unit="test accuracy (%)",
                    manuscript_value=manuscript_row["最强对照"],
                    computed_value=(
                        f"{method_display[best_method]} {pct_stats(best_mean, best_std)}"
                    ),
                    n=best_n,
                    mean=best_mean,
                    std=best_std,
                    filter_expression=(
                        base_filter
                        + f" AND method in {json.dumps(list(FORMAL_BASELINES), ensure_ascii=False)}"
                    ),
                    aggregation_formula=(
                        f"for each formal baseline: mean=arithmetic_mean({value_field}); "
                        "select argmax(mean); format label + 100*mean±100*sample_stdev(ddof=1)"
                    ),
                ),
                lineage_record(
                    **common,
                    column_label="随机化分组",
                    method=SHUFFLED,
                    cell_kind="mean_std",
                    unit="test accuracy (%)",
                    manuscript_value=manuscript_row["随机化分组"],
                    computed_value=pct_stats(shuffled_mean, shuffled_std),
                    n=shuffled_n,
                    mean=shuffled_mean,
                    std=shuffled_std,
                    filter_expression=base_filter + f' AND method=="{SHUFFLED}"',
                    aggregation_formula=(
                        f"mean=100*arithmetic_mean({value_field}); "
                        f"std=100*sample_stdev({value_field},ddof=1); format .2f±.2f"
                    ),
                ),
                lineage_record(
                    **common,
                    column_label="GB-LCSS−最强对照",
                    method=f"{OURS}-minus-{best_method}",
                    cell_kind="paired_mean_delta",
                    unit="percentage points",
                    manuscript_value=manuscript_row["GB-LCSS−最强对照"],
                    computed_value=signed_pp(ours_mean - best_mean),
                    n=5,
                    mean=100.0 * (ours_mean - best_mean),
                    std=NA,
                    filter_expression=(
                        base_filter
                        + f' AND method in ["{OURS}","{best_method}"] AND seed in [0,1,2,3,4]'
                    ),
                    aggregation_formula=(
                        f"100*arithmetic_mean({value_field}[{OURS},seed]"
                        f"-{value_field}[{best_method},seed]); format signed .2f"
                    ),
                ),
                lineage_record(
                    **common,
                    column_label="GB-LCSS−随机化分组",
                    method=f"{OURS}-minus-{SHUFFLED}",
                    cell_kind="paired_mean_delta",
                    unit="percentage points",
                    manuscript_value=manuscript_row["GB-LCSS−随机化分组"],
                    computed_value=signed_pp(ours_mean - shuffled_mean),
                    n=5,
                    mean=100.0 * (ours_mean - shuffled_mean),
                    std=NA,
                    filter_expression=(
                        base_filter
                        + f' AND method in ["{OURS}","{SHUFFLED}"] AND seed in [0,1,2,3,4]'
                    ),
                    aggregation_formula=(
                        f"100*arithmetic_mean({value_field}[{OURS},seed]"
                        f"-{value_field}[{SHUFFLED},seed]); format signed .2f"
                    ),
                ),
            ]
        )

    # Tables 3, 5 and 6 use the existing audited source-chain loader.
    audit = import_audit_module(noisy_root)
    audited_specs = [
        (
            3,
            audit.load_table3_rows(noisy_root),
            table3_aggregate,
            "table3_recomputed.csv",
            {
                "MLP+CE": "mlp_ce",
                "Graph-MLP (protocol port)": "graphmlp_reimplemented",
                "GSDN-inspired (mechanism adaptation)": "gsdn",
                "GB-LCSS": OURS,
                "GLNN 式蒸馏": "mlp_distill_gcn",
                "FF-G2M (common-backbone reimplementation)": "ffg2m_reimplemented",
                "KRD-inspired reliability proxy": "krd_adapted",
                "VQGraph-inspired token proxy": "vqgraph_adapted",
                "GB-LCSS 蒸馏扩展（球式共识教师）": "mlp_distill_softdist_xh",
            },
            label_to_dataset,
        ),
        (
            5,
            audit.load_table4_rows(noisy_root),
            table5_aggregate,
            "external table4_recomputed.csv → manuscript Table 5",
            {
                "GCN(X)+CE（GLNN 式）": "mlp_distill_gcn",
                "GCN[X‖H]+CE": "mlp_distill_gcn_xh",
                "随机分组共识教师": "mlp_distill_softdist_xh_shuffled",
                "kNN（票数匹配）共识教师": "mlp_distill_knn_xh",
                "球式共识教师（本文）": "mlp_distill_softdist_xh",
            },
            {
                "Cora u0.4": ("cora", "0.4"),
                "CiteSeer u0.4": ("citeseer", "0.4"),
                "CoraFull u0.4": ("corafull", "0.4"),
                "Cora u0.6": ("cora", "0.6"),
                "CiteSeer u0.6": ("citeseer", "0.6"),
                "CoraFull u0.6": ("corafull", "0.6"),
            },
        ),
        (
            6,
            audit.load_table5_rows(noisy_root),
            table6_aggregate,
            "external table5_recomputed.csv → manuscript Table 6",
            {
                "GCN(X)": "gcn",
                "GCN[X‖H]": "gcn_xh_ce",
                "NRGNN": "nrgnn",
                "NRGNN[X‖H]": "nrgnn_xh",
                "TFR-GCN": "tfr_gcn",
                "ProCon": "procon",
                "GD2": "gd2",
                "GB-LCSS 蒸馏扩展（学生）": "mlp_distill_softdist_xh",
            },
            label_to_dataset,
        ),
    ]
    table_source_manifest_path = (
        noisy_root
        / "runs"
        / "submission_integrity_v12_20260712"
        / "tables3_5_evidence_manifest.csv"
    )
    manifest_rows = read_csv(table_source_manifest_path)
    manifest_index = {
        Path(row["path"]).resolve(): row
        for row in manifest_rows
    }
    audited_source_paths = {
        Path(str(row["source"])).resolve()
        for spec in audited_specs
        for row in spec[1]
    }
    if set(manifest_index) != audited_source_paths:
        missing = sorted(str(path) for path in audited_source_paths - set(manifest_index))
        extra = sorted(str(path) for path in set(manifest_index) - audited_source_paths)
        raise ValueError(
            f"tables3_5 manifest/source mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    manifest_matches = 0
    for path, manifest_row in manifest_index.items():
        if (
            int(manifest_row["size_bytes"]) == path.stat().st_size
            and manifest_row["sha256"].upper() == sha256(path)
        ):
            manifest_matches += 1
        else:
            raise ValueError(f"tables3_5 manifest drift: {path}")

    for (
        table_id,
        audited_rows,
        aggregate_path,
        external_id,
        methods,
        columns,
    ) in audited_specs:
        _, manuscript_rows = parse_markdown_table(manuscript, table_id)
        row_key_name = "方法" if table_id in (3, 6) else "教师"
        manuscript_index = {row[row_key_name]: row for row in manuscript_rows}
        for row_label, method in methods.items():
            manuscript_row = manuscript_index[row_label]
            for column_label, dataset_spec in columns.items():
                if isinstance(dataset_spec, tuple):
                    dataset, noise_rate = dataset_spec
                else:
                    dataset, noise_rate = dataset_spec, "0.4"
                seed_rows = [
                    row
                    for row in audited_rows
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["noise_type"] == "uniform"
                    and normalized_rate(row["noise_rate"]) == noise_rate
                ]
                n, mean, std = stats(seed_rows)
                raw_paths, raw_hashes, raw_sizes = source_provenance(
                    seed_rows, workspace_root, noisy_root
                )
                records.append(
                    lineage_record(
                        table_id=table_id,
                        external_table_id=external_id,
                        row_label=row_label,
                        column_label=column_label,
                        dataset=dataset,
                        noise_type="uniform",
                        noise_rate=noise_rate,
                        method=method,
                        cell_kind="mean_std",
                        unit="test accuracy (%)",
                        manuscript_value=manuscript_row[column_label],
                        computed_value=pct_stats(mean, std),
                        n=n,
                        mean=mean,
                        std=std,
                        raw_path=raw_paths,
                        raw_sha256=raw_hashes,
                        raw_size_bytes=raw_sizes,
                        filter_expression=exact_filter(
                            dataset, "uniform", noise_rate, method
                        )
                        + " AND seed in [0,1,2,3,4]",
                        aggregation_formula=(
                            "mean=100*arithmetic_mean(test); "
                            "std=100*sample_stdev(test,ddof=1); format .2f±.2f"
                        ),
                        aggregate_path=canonical_path(
                            aggregate_path, workspace_root, noisy_root
                        ),
                        aggregate_sha256=sha256(aggregate_path),
                        aggregate_filter_expression=(
                            f'dataset=="{dataset}" AND noise_type=="uniform" '
                            f'AND noise_rate=={noise_rate} AND method=="{method}"'
                        ),
                        protocol_note=(
                            "existing audit loader supplies train-only scope, split/noise "
                            "hashes, pollution counts, and exact source-chain match"
                        ),
                    )
                )

        # Explicit GAP rows for nonnumeric resource/deployment metadata columns.
        metadata_column = "训练资源" if table_id == 3 else ("推理需图" if table_id == 6 else None)
        if metadata_column:
            for row_label, method in methods.items():
                manuscript_value = manuscript_index[row_label][metadata_column]
                records.append(
                    lineage_record(
                        table_id=table_id,
                        external_table_id=external_id,
                        row_label=row_label,
                        column_label=metadata_column,
                        dataset=NA,
                        noise_type=NA,
                        noise_rate=NA,
                        method=method,
                        cell_kind="method_metadata",
                        unit=NA,
                        manuscript_value=manuscript_value,
                        computed_value=GAP,
                        n=NA,
                        mean=NA,
                        std=NA,
                        raw_path=GAP,
                        raw_sha256=GAP,
                        raw_size_bytes=GAP,
                        filter_expression=GAP,
                        aggregation_formula=GAP,
                        gap_reason=(
                            "no per-cell config/resource manifest connects this manuscript "
                            "metadata declaration to a fixed path/hash"
                        ),
                    )
                )

    # Table 4
    latency_path = (
        noisy_root
        / "runs"
        / "mlp36_review_defense_20260715"
        / "latency_memory_seed0.csv"
    )
    latency_rows = read_csv(latency_path)
    _, manuscript_rows = parse_markdown_table(manuscript, 4)
    latency_method_map = {
        "MLP+CE": "mlp_ce",
        "GB-LCSS": OURS,
        "GCN": "gcn",
    }
    metrics = {
        "参数量 (K)": (
            "thousand parameters",
            lambda row: float(row["parameter_count"]) / 1000.0,
            lambda value: f"{value:.1f}",
            "parameter_count/1000; format .1f",
        ),
        "FP32 参数 (MiB)": (
            "MiB",
            lambda row: float(row["parameter_count"]) * 4.0 / (1024.0**2),
            lambda value: f"{value:.2f}",
            "parameter_count*4/1048576; format .2f",
        ),
        "下游墙钟 (s)": (
            "seconds",
            lambda row: float(row["downstream_wall_seconds"]),
            lambda value: f"{value:.2f}",
            "direct downstream_wall_seconds; format .2f",
        ),
        "构球 (s)": (
            "seconds",
            lambda row: float(row["ball_build_seconds"]),
            lambda value: "0" if value == 0 else f"{value:.2f}",
            "direct ball_build_seconds; zero rendered as 0, else format .2f",
        ),
        "推理 (ms/全图)": (
            "milliseconds per full graph",
            lambda row: float(row["fullgraph_latency_ms"]),
            lambda value: f"{value:.3f}",
            "direct fullgraph_latency_ms; format .3f",
        ),
        "吞吐 (M 节点/s)": (
            "million nodes per second",
            lambda row: float(row["nodes"])
            / (float(row["fullgraph_latency_ms"]) * 1000.0),
            lambda value: f"{value:.2f}",
            "nodes/(fullgraph_latency_ms*1000); format .2f",
        ),
        "峰值推理显存 (MB)": (
            "MB",
            lambda row: float(row["inference_peak_allocated_mb"]),
            lambda value: f"{value:.1f}",
            "direct inference_peak_allocated_mb; format .1f",
        ),
    }
    for manuscript_row in manuscript_rows:
        dataset = dataset_by_label[manuscript_row["数据集"]]
        method = latency_method_map[manuscript_row["方法"]]
        matches = [
            row
            for row in latency_rows
            if row["dataset"] == dataset and row["method"] == method and int(row["seed"]) == 0
        ]
        if len(matches) != 1:
            raise ValueError(f"Table 4 latency row mismatch: {dataset}/{method}")
        raw = matches[0]
        for column_label, (unit, compute, render, formula) in metrics.items():
            value = compute(raw)
            records.append(
                lineage_record(
                    table_id=4,
                    external_table_id="latency_memory_seed0.csv",
                    row_label=f'{manuscript_row["数据集"]} / {manuscript_row["方法"]}',
                    column_label=column_label,
                    dataset=dataset,
                    noise_type="uniform",
                    noise_rate="0.4",
                    method=method,
                    cell_kind="seed0_measurement_or_derived",
                    unit=unit,
                    manuscript_value=manuscript_row[column_label],
                    computed_value=render(value),
                    n=1,
                    mean=value,
                    std=NA,
                    raw_path=canonical_path(latency_path, workspace_root, noisy_root),
                    raw_sha256=sha256(latency_path),
                    raw_size_bytes=str(latency_path.stat().st_size),
                    filter_expression=(
                        f'dataset=="{dataset}" AND method=="{method}" AND seed==0'
                    ),
                    aggregation_formula=formula,
                    protocol_note=(
                        "seed 0; inference_repeats=100 in raw. GAP: raw/report gives one "
                        "ROCm GPU but no exact GPU model, ROCm, PyTorch, or PyG versions."
                    ),
                )
            )

    cell_ids = [row["cell_id"] for row in records]
    if len(set(cell_ids)) != len(cell_ids):
        duplicates = [cell_id for cell_id in set(cell_ids) if cell_ids.count(cell_id) > 1]
        raise ValueError(f"duplicate manuscript cell ids: {duplicates[:5]}")

    metrics_out = {
        "rows": len(records),
        "closed_rows": sum(row["lineage_status"] == "CLOSED" for row in records),
        "gap_rows": sum(row["lineage_status"] == "GAP" for row in records),
        "mismatch_rows": sum(row["lineage_status"] == "MISMATCH" for row in records),
        "numeric_rows": sum(row["cell_kind"] != "method_metadata" for row in records),
        "numeric_closed_rows": sum(
            row["cell_kind"] != "method_metadata"
            and row["lineage_status"] == "CLOSED"
            for row in records
        ),
        "table_source_manifest_rows": len(manifest_rows),
        "table_source_manifest_matches": manifest_matches,
        "audited_table_source_paths": len(audited_source_paths),
        "by_table": {},
    }
    for table_id in range(1, 7):
        table_rows = [
            row for row in records if row["manuscript_table_id"] == f"Table {table_id}"
        ]
        metrics_out["by_table"][table_id] = {
            "rows": len(table_rows),
            "closed": sum(row["lineage_status"] == "CLOSED" for row in table_rows),
            "gap": sum(row["lineage_status"] == "GAP" for row in table_rows),
            "mismatch": sum(row["lineage_status"] == "MISMATCH" for row in table_rows),
        }
    return records, metrics_out


def global_holm_sensitivity(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["paired_t_p"]))
    adjusted = [0.0] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order, 1):
        candidate = min(1.0, (total - rank + 1) * float(rows[index]["paired_t_p"]))
        running = max(running, candidate)
        adjusted[index] = running
    by_comparison = {}
    for comparison in sorted({row["comparison"] for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row["comparison"] == comparison
        ]
        by_comparison[comparison] = sum(adjusted[index] < 0.05 for index in indices)
    return {
        "total": total,
        "significant": sum(value < 0.05 for value in adjusted),
        "by_comparison": by_comparison,
        "formula": (
            "sort all 75 paired_t_p ascending; raw Holm=(75-rank+1)*p; "
            "adjusted=max(previous,raw Holm), clipped at 1"
        ),
    }


def asset_row(path: Path, workspace_root: Path, noisy_root: Path) -> str:
    return (
        f"| `{canonical_path(path, workspace_root, noisy_root)}` | "
        f"{path.stat().st_size} | `{sha256(path)}` |"
    )


def build_gap_report(
    workspace_root: Path,
    noisy_root: Path,
    manuscript: Path,
    protocol_path: Path,
    lineage_path: Path,
    protocol_metrics: dict[str, Any],
    lineage_metrics: dict[str, Any],
) -> str:
    phase4_path = (
        noisy_root
        / "runs"
        / "phase4_faithful_claims_20260712"
        / "phase4_selected_raw.csv"
    )
    primary_raw = (
        noisy_root
        / "runs"
        / "mlp36_review_defense_20260715"
        / "primary_statistics_raw.csv"
    )
    primary_holm = primary_raw.with_name("primary_statistics_holm.csv")
    formal_main = (
        workspace_root
        / "latest_baselines_trainonly_20260622_continuation"
        / "formal_main_raw.csv"
    )
    table_manifest = (
        noisy_root
        / "runs"
        / "submission_integrity_v12_20260712"
        / "tables3_5_evidence_manifest.csv"
    )
    latency_path = (
        noisy_root
        / "runs"
        / "mlp36_review_defense_20260715"
        / "latency_memory_seed0.csv"
    )
    beta_audit = (
        noisy_root
        / "runs"
        / "mlp15_review_defense_20260704"
        / "p0a_beta_validation_audit.csv"
    )
    beta_decision = (
        noisy_root
        / "runs"
        / "mlp_beta_hardness_5seed_20260701"
        / "mlp_beta_hardness_decision.csv"
    )
    beta_report = beta_decision.with_name("mlp_beta_hardness_report.md")
    holm = global_holm_sensitivity(primary_holm)

    table_lines = []
    for table_id, metrics in lineage_metrics["by_table"].items():
        table_lines.append(
            f"| Table {table_id} | {metrics['rows']} | {metrics['closed']} | "
            f"{metrics['gap']} | {metrics['mismatch']} |"
        )

    return f"""# Protocol and manuscript-table lineage gap report

## 1. Scope and verdict

本次只读取既有资产，不训练、不联网、不修改任何原始结果、论文正文、`material_manifest.md`、claim ledger 或 issue traceability。输出覆盖 `{manuscript.name}` **Tables 1–6 的数据单元格**，并把 Table 3 的“训练资源”与 Table 6 的“推理需图”元数据单元格作为显式 GAP 记录。

- **PKG-P1-06：PARTIAL。** 1,200 条 phase4 method/seed 行均已绑定到真实选择文件、真实 source CSV、source SHA-256、精确过滤键和直接取值公式；但完整必需协议字段只闭合 {protocol_metrics['closed_rows']}/{protocol_metrics['rows']} 行（{100.0 * protocol_metrics['closed_rows'] / protocol_metrics['rows']:.2f}%）。
- **PKG-P1-07 / EXP-10 的数值 lineage：PARTIAL。** Tables 1–6 的 {lineage_metrics['numeric_closed_rows']}/{lineage_metrics['numeric_rows']} 个数据单元格（{100.0 * lineage_metrics['numeric_closed_rows'] / lineage_metrics['numeric_rows']:.2f}%）与稿件匹配且有 path/hash/filter/formula；数值 mismatch 为 {lineage_metrics['mismatch_rows']}。
- **完整逐 cell（含资源元数据）lineage：PARTIAL。** {lineage_metrics['closed_rows']}/{lineage_metrics['rows']}（{100.0 * lineage_metrics['closed_rows'] / lineage_metrics['rows']:.2f}%）闭合；{lineage_metrics['gap_rows']} 个元数据 cell 明确为 GAP；数值 mismatch 为 {lineage_metrics['mismatch_rows']}。
- 这不等于 encoder-to-result 端到端复现闭合。MET-04 所列源码入口、环境锁定、全部配置映射和缺失 checkpoint 问题仍然存在。

## 2. Protocol ledger coverage

| Audit item | Closed | Target | Coverage |
|---|---:|---:|---:|
| source path/bytes/SHA-256 + exact source-row test match | {protocol_metrics['rows']} | {protocol_metrics['rows']} | 100.00% |
| split_hash + noise_hash（row） | {protocol_metrics['split_noise_rows']} | {protocol_metrics['rows']} | {100.0 * protocol_metrics['split_noise_rows'] / protocol_metrics['rows']:.2f}% |
| modified train/val/test counts（row） | {protocol_metrics['pollution_count_rows']} | {protocol_metrics['rows']} | {100.0 * protocol_metrics['pollution_count_rows'] / protocol_metrics['rows']:.2f}% |
| all required protocol fields（row） | {protocol_metrics['closed_rows']} | {protocol_metrics['rows']} | {100.0 * protocol_metrics['closed_rows'] / protocol_metrics['rows']:.2f}% |
| primary-statistics subset all required fields | {protocol_metrics['closed_primary_rows']} | {protocol_metrics['primary_rows']} | {100.0 * protocol_metrics['closed_primary_rows'] / protocol_metrics['primary_rows']:.2f}% |
| split/noise hashes complete（setting） | {protocol_metrics['split_noise_settings']} | {protocol_metrics['settings']} | {100.0 * protocol_metrics['split_noise_settings'] / protocol_metrics['settings']:.2f}% |
| pollution counts complete（setting） | {protocol_metrics['pollution_count_settings']} | {protocol_metrics['settings']} | {100.0 * protocol_metrics['pollution_count_settings'] / protocol_metrics['settings']:.2f}% |
| all required fields complete（setting） | {protocol_metrics['closed_settings']} | {protocol_metrics['settings']} | {100.0 * protocol_metrics['closed_settings'] / protocol_metrics['settings']:.2f}% |
| instance overall realization hash | {protocol_metrics['instance_realization_rows']} | {protocol_metrics['instance_rows']} | {100.0 * protocol_metrics['instance_realization_rows'] / protocol_metrics['instance_rows']:.2f}% |
| explicit instance generator seed/hash | {protocol_metrics['instance_generator_rows']} | {protocol_metrics['instance_rows']} | {100.0 * protocol_metrics['instance_generator_rows'] / protocol_metrics['instance_rows']:.2f}% |

验证结果：

- phase4 unique key `(dataset,noise_type,noise_rate,method,seed)`：{protocol_metrics['rows']} 个，无重复。
- `primary_statistics_raw.csv` 的 500 行全部精确连接，test 值容差 `1e-12` 内一致。
- 对已有的 split/noise hash，同一 `(dataset,setting,seed)` 内未发现冲突；冲突数 {protocol_metrics['hash_mismatches']}。
- `source_sha256` 是对当前真实 source CSV 重新计算，不从 manifest 抄写。

## 3. Table lineage coverage

| Manuscript table | Tracked cells | Closed | GAP | Mismatch |
|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

表号映射已显式固化：

- external `table3_recomputed.csv` → manuscript Table 3；
- external `table4_recomputed.csv` → manuscript Table 5；
- external `table5_recomputed.csv` → manuscript Table 6；
- manuscript Table 4 直接回到 `latency_memory_seed0.csv`，FP32 MiB 与吞吐量公式已逐 cell 物化在 lineage CSV。
- `tables3_5_evidence_manifest.csv` 的 {lineage_metrics['table_source_manifest_matches']}/{lineage_metrics['table_source_manifest_rows']} 个 path/bytes/SHA-256 与当前文件重算一致，并且恰好覆盖 audited source-chain 的 {lineage_metrics['audited_table_source_paths']} 个不同源文件。

## 4. Unclosed gaps

### GAP-PROTOCOL-01: CoraFull lacks split/noise hashes

`mlp21_raw.csv` 有 `noise_scope` 与修改计数，但没有 `split_hash`、`noise_hash`。因此 CoraFull 的 method/seed 行不能独立证明 shared realization。没有从其他方法或 setting 借值。

### GAP-PROTOCOL-02: Cora/CiteSeer/Coauthor-CS lack pollution counts in selected source

`mlp10_combined_raw.csv` 有 split/noise hash 与 `actual_train_noise_rate`，但没有 `modified_*_count`。因此这些行不因 hash 已知而被伪装成“完整协议闭合”。

### GAP-PROTOCOL-03: instance generator provenance

已有 `noise_hash` 时将其原样复制为 `instance_realization_hash`；没有把 downstream seed 推断成 generator seed。所有 instance 行的显式 `instance_generator_seed/hash` 仍为 GAP；CoraFull 连 overall realization hash 也缺失。

### GAP-FREEZE-01: freeze boundary remains non-temporal

现有 beta 资产真实记录候选值/验证结果与“beta=0.5 作为 conservative fixed default”的决策：

- `{canonical_path(beta_audit, workspace_root, noisy_root)}`；
- `{canonical_path(beta_decision, workspace_root, noisy_root)}`；
- `{canonical_path(beta_report, workspace_root, noisy_root)}`。

但仍缺：冻结时间或 commit、Cora/CiteSeer 如何合并为唯一选择规则、`B_max/n_min/delta/s_min` 的完整候选网格与选择记录、哪些 encoder 参数允许 dataset-specific 的正式表。稿件不能据此把“全部构造参数冻结”升级为已审计事实。

### GAP-TABLE-01: nonnumeric resource metadata

Table 3 的 9 个“训练资源”cell 和 Table 6 的 8 个“推理需图”cell 没有逐 cell 配置/资源 manifest。lineage CSV 逐一写为 `GAP`，未用论文文本自证论文文本。

### GAP-TABLE-02: Table 4 environment metadata

Table 4 的数值 cell 均有直接 raw/formula；FP32 MiB 使用 `parameter_count*4/1048576`，吞吐量使用节点数除以 full-graph latency。raw 明确 seed 0、`inference_repeats=100`、一个 ROCm GPU；现有 latency raw/report 未给 exact GPU model、ROCm、PyTorch、PyG 版本或同步/warm-up 细节，因此跨环境复现实验仍未闭合。

### GAP-TABLE-03: breadth-source protocol columns

`mlp23_raw.csv` 的四个扩展数据集数值可按 dataset/method/seed 精确聚合，但文件自身没有 `noise_type/noise_rate` 字段。对应 cell 的数值来源可追踪；“该 raw 行独立声明 uniform 0.4”仍是协议 GAP。

### GAP-REPRO-01: end-to-end reproduction

本次没有创建不存在的源码入口、环境锁文件、全数据集 config-ID 映射或 checkpoint。现有资产支持 seed-row/aggregate replay，不自动支持从原始图重建全部 H 并复跑 Tables 1–6。

## 5. Existing 75-test Holm sensitivity

基于 `{canonical_path(primary_holm, workspace_root, noisy_root)}` 中现有 75 个 `paired_t_p`，按单一 75-test Holm 家族机械重算：

- 全局显著：{holm['significant']}/{holm['total']}；
- LCSS vs CE：{holm['by_comparison']['lcss_vs_ce']}/25（原分家族为 24/25）；
- LCSS vs JoCoR：{holm['by_comparison']['lcss_vs_jocor']}/25（原分家族为 8/25）；
- LCSS vs shuffled：{holm['by_comparison']['lcss_vs_shuffled']}/25（原分家族为 25/25）。

公式：`{holm['formula']}`。这是既有 p 值的敏感性重算，不是新实验，也不把三个原作者定义族伪装成外部预注册。

## 6. Key audited inputs

| Path | Bytes | SHA-256 |
|---|---:|---|
{asset_row(manuscript, workspace_root, noisy_root)}
{asset_row(phase4_path, workspace_root, noisy_root)}
{asset_row(primary_raw, workspace_root, noisy_root)}
{asset_row(primary_holm, workspace_root, noisy_root)}
{asset_row(formal_main, workspace_root, noisy_root)}
{asset_row(table_manifest, workspace_root, noisy_root)}
{asset_row(latency_path, workspace_root, noisy_root)}
{asset_row(beta_audit, workspace_root, noisy_root)}
{asset_row(beta_decision, workspace_root, noisy_root)}

## 7. Generated assets

| Path | Bytes | SHA-256 |
|---|---:|---|
{asset_row(protocol_path, workspace_root, noisy_root)}
{asset_row(lineage_path, workspace_root, noisy_root)}

生成/复核命令：

```powershell
py paper_pipeline/scripts/generate_protocol_and_lineage.py --manuscript paper_final_zh_v30_information_sciences.md
py paper_pipeline/scripts/generate_protocol_and_lineage.py --manuscript paper_final_zh_v30_information_sciences.md --verify-only
```
"""


def verify_existing(path: Path, expected_rows: list[dict[str, Any]]) -> None:
    actual_rows = read_csv(path)
    if actual_rows != [
        {key: str(value) for key, value in row.items()} for row in expected_rows
    ]:
        raise ValueError(f"existing generated asset differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="recompute in memory and compare with existing outputs",
    )
    parser.add_argument(
        "--manuscript",
        default="paper_final_zh_v30_information_sciences.md",
        help="workspace-relative manuscript whose Tables 1–6 are audited",
    )
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    noisy_root = (workspace_root.parents[1] / "external" / "NoisyGL").resolve()
    manuscript = (workspace_root / args.manuscript).resolve()
    if manuscript.parent != workspace_root.resolve():
        raise ValueError("--manuscript must resolve directly inside the workspace root")
    if not manuscript.is_file():
        raise FileNotFoundError(manuscript)
    evidence_dir = workspace_root / "paper_pipeline" / "00_evidence"
    protocol_path = evidence_dir / "protocol_ledger.csv"
    lineage_path = evidence_dir / "manuscript_table_lineage.csv"
    report_path = evidence_dir / "protocol_and_lineage_gap_report.md"

    protocol_rows, protocol_metrics = build_protocol_ledger(
        workspace_root, noisy_root
    )
    lineage_rows, lineage_metrics = build_table_lineage(
        workspace_root, noisy_root, manuscript, protocol_rows
    )
    protocol_fields = list(protocol_rows[0].keys())
    lineage_fields = list(lineage_rows[0].keys())

    if args.verify_only:
        verify_existing(protocol_path, protocol_rows)
        verify_existing(lineage_path, lineage_rows)
        expected_report = build_gap_report(
            workspace_root,
            noisy_root,
            manuscript,
            protocol_path,
            lineage_path,
            protocol_metrics,
            lineage_metrics,
        )
        if report_path.read_text(encoding="utf-8") != expected_report:
            raise ValueError(f"existing generated asset differs: {report_path}")
        print(
            f"VERIFIED protocol={len(protocol_rows)} rows; "
            f"lineage={len(lineage_rows)} rows; report=exact; "
            f"mismatches={lineage_metrics['mismatch_rows']}"
        )
        return

    write_csv(protocol_path, protocol_rows, protocol_fields)
    write_csv(lineage_path, lineage_rows, lineage_fields)
    report = build_gap_report(
        workspace_root,
        noisy_root,
        manuscript,
        protocol_path,
        lineage_path,
        protocol_metrics,
        lineage_metrics,
    )
    report_path.write_text(report, encoding="utf-8")
    print(
        f"WROTE protocol={len(protocol_rows)} rows; "
        f"lineage={len(lineage_rows)} rows; mismatches={lineage_metrics['mismatch_rows']}"
    )


if __name__ == "__main__":
    main()
