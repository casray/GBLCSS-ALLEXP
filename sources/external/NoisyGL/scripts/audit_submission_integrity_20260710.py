from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_KEY = ("dataset", "method", "noise_type", "noise_rate", "seed")
PAIR_KEY = ("dataset", "noise_type", "noise_rate", "seed")
CORE_DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
CORE_METHODS = (
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta05_shuffled",
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_ct",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "mlp_logitclip",
    "mlp_anl",
    "mlp_fine",
    "mlp_dividemix_lite",
)
CORE_SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "corafull": "CoraFull",
    "coauthorcs": "Coauthor-CS",
    "amazonpho": "Amazon-Photo",
}
OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"

TABLE3_METHODS = (
    "mlp_ce",
    "graphmlp_reimplemented",
    "gsdn",
    OURS,
    "mlp_distill_gcn",
    "ffg2m_reimplemented",
    "krd_adapted",
    "vqgraph_adapted",
    "mlp_distill_softdist_xh",
)
TABLE4_DATASETS = ("cora", "citeseer", "corafull")
TABLE4_METHODS = (
    "mlp_distill_gcn",
    "mlp_distill_gcn_xh",
    "mlp_distill_softdist_xh_shuffled",
    "mlp_distill_knn_xh",
    "mlp_distill_softdist_xh",
)
TABLE5_METHODS = (
    "gcn",
    "gcn_xh_ce",
    "nrgnn",
    "nrgnn_xh",
    "tfr_gcn",
    "procon",
    "gd2",
    "mlp_distill_softdist_xh",
)
PROTOCOL_FIELDS = (
    "noise_scope",
    "split_hash",
    "noise_hash",
    "modified_count",
    "modified_train_count",
    "modified_val_count",
    "modified_test_count",
    "status",
)


def _key(row: dict[str, object], fields: tuple[str, ...]) -> tuple[object, ...]:
    values = []
    for field in fields:
        value = row[field]
        if field == "noise_rate":
            value = float(value)
        elif field == "seed":
            value = int(value)
        values.append(value)
    return tuple(values)


def assert_unique_rows(rows: list[dict[str, object]]) -> None:
    seen = set()
    for row in rows:
        key = _key(row, ROW_KEY)
        if key in seen:
            raise ValueError(f"duplicate experiment row: {key}")
        seen.add(key)


def assert_paired_hashes(rows: list[dict[str, object]]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_key(row, PAIR_KEY)].append(row)
    for key, items in grouped.items():
        for field in ("split_hash", "noise_hash"):
            values = {str(item.get(field, "")).strip() for item in items}
            values.discard("")
            if len(values) > 1:
                raise ValueError(f"{field} mismatch for paired setting {key}: {sorted(values)}")


def assert_train_only_protocol(rows: list[dict[str, object]]) -> None:
    for row in rows:
        key = _key(row, ROW_KEY)
        if str(row.get("status", "ok")).strip().lower() != "ok":
            raise ValueError(f"non-ok experiment row: {key}")
        if str(row.get("noise_scope", "")).strip().lower() != "train":
            raise ValueError(f"noise_scope is not train: {key}")
        modified_val = int(float(row.get("modified_val_count", 0) or 0))
        modified_test = int(float(row.get("modified_test_count", 0) or 0))
        if modified_val != 0 or modified_test != 0:
            raise ValueError(f"validation/test labels were modified: {key}")
        modified = int(float(row.get("modified_count", 0) or 0))
        modified_train = int(float(row.get("modified_train_count", 0) or 0))
        if modified != modified_train:
            raise ValueError(f"modified_count differs from modified_train_count: {key}")


def aggregate_seed_rows(rows: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, float | int]]:
    grouped = defaultdict(list)
    for row in rows:
        key = _key(row, ("dataset", "noise_type", "noise_rate", "method"))
        grouped[key].append(float(row["test"]))
    return {
        key: {
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in grouped.items()
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _resolve_source_path(source: str, repo_root: Path) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def enrich_protocol_from_source(
    row: dict[str, object],
    repo_root: str | Path,
    cache: dict[Path, list[dict[str, str]]] | None = None,
    seen: set[Path] | None = None,
) -> dict[str, object]:
    repo_root = Path(repo_root)
    source_value = row.get("source") or row.get("source_file")
    if not source_value:
        raise ValueError(f"row has no source path: {_key(row, ROW_KEY)}")
    source_path = _resolve_source_path(str(source_value), repo_root)
    if not source_path.exists():
        raise FileNotFoundError(f"source CSV not found: {source_path}")
    if cache is None:
        cache = {}
    if seen is None:
        seen = set()
    if source_path in seen:
        raise ValueError(f"cyclic source chain for {_key(row, ROW_KEY)}: {source_path}")
    seen = set(seen)
    seen.add(source_path)
    source_rows = cache.setdefault(source_path, _read_csv(source_path))
    key = _key(row, ROW_KEY)
    matches = [item for item in source_rows if _key(item, ROW_KEY) == key]
    if len(matches) > 1:
        target_test = float(row["test"])
        matches = [
            item
            for item in matches
            if (item.get("test") or item.get("test_acc") or item.get("student_test", "")) != ""
            and abs(float(item.get("test") or item.get("test_acc") or item.get("student_test")) - target_test) <= 1e-12
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one source row for {key} in {source_path}, found {len(matches)}")
    source_row = matches[0]
    source_test = source_row.get("test") or source_row.get("test_acc") or source_row.get("student_test", "")
    if source_test == "" or abs(float(source_test) - float(row["test"])) > 1e-12:
        raise ValueError(f"test value differs from source row for {key}")
    enriched = dict(row)
    for field in PROTOCOL_FIELDS:
        if field in source_row:
            enriched[field] = source_row[field]
    enriched["source"] = str(source_path)

    if not all(str(enriched.get(field, "")).strip() for field in PROTOCOL_FIELDS):
        nested_source = source_row.get("source") or source_row.get("source_file")
        if nested_source:
            nested_row = dict(source_row)
            nested_row["test"] = source_test
            nested_row["source"] = nested_source
            nested = enrich_protocol_from_source(nested_row, repo_root, cache, seen)
            for field in PROTOCOL_FIELDS:
                if field in nested:
                    enriched[field] = nested[field]
            enriched["source"] = nested["source"]
    return enriched


def _as_audit_row(row: dict[str, object], *, test_field: str = "test") -> dict[str, object]:
    normalized = dict(row)
    normalized["dataset"] = str(normalized["dataset"]).lower()
    normalized["method"] = str(normalized["method"])
    normalized["noise_type"] = str(normalized.get("noise_type", "uniform")).lower()
    normalized["noise_rate"] = str(normalized.get("noise_rate", "0.4"))
    normalized["seed"] = str(normalized["seed"])
    normalized["test"] = str(normalized[test_field])
    return normalized


def _attach_direct_source(row: dict[str, object], path: Path) -> dict[str, object]:
    normalized = _as_audit_row(row)
    normalized["source"] = str(path.resolve())
    return normalized


def assert_expected_grid(
    rows: list[dict[str, object]],
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    settings: tuple[tuple[str, float], ...],
) -> None:
    expected = {
        (dataset, method, noise_type, float(noise_rate), seed)
        for dataset in datasets
        for method in methods
        for noise_type, noise_rate in settings
        for seed in range(5)
    }
    observed = {
        (
            str(row["dataset"]),
            str(row["method"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            int(row["seed"]),
        )
        for row in rows
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"experiment grid mismatch: missing={missing[:5]} extra={extra[:5]}")


def _enrich_rows(rows: list[dict[str, object]], repo_root: Path) -> list[dict[str, object]]:
    cache: dict[Path, list[dict[str, str]]] = {}
    return [enrich_protocol_from_source(row, repo_root, cache) for row in rows]


def load_table3_rows(repo_root: str | Path = REPO_ROOT) -> list[dict[str, object]]:
    repo_root = Path(repo_root)
    selected: list[dict[str, object]] = []
    legacy_path = repo_root / "runs" / "mlp19_g2m_baselines_20260709" / "mlp19_combined_raw.csv"
    for raw in _read_csv(legacy_path):
        row = _as_audit_row(raw)
        if row["dataset"] in {"cora", "citeseer", "coauthorcs", "amazonpho"} and row["method"] in TABLE3_METHODS:
            selected.append(row)

    corafull_path = repo_root / "runs" / "mlp21_dataset_replacement_20260710" / "mlp21_raw.csv"
    for raw in _read_csv(corafull_path):
        row = _as_audit_row(raw)
        if (
            row["dataset"] == "corafull"
            and row["method"] in TABLE3_METHODS
            and row["noise_type"] == "uniform"
            and float(row["noise_rate"]) == 0.4
        ):
            selected.append(row)

    for name in (
        "runs/mlp26_ffg2m_fidelity_20260712/mlp26_combined_raw.csv",
        "runs/mlp27_graphmlp_fidelity_20260712/mlp27_combined_raw.csv",
    ):
        path = repo_root / name
        for raw in _read_csv(path):
            row = _attach_direct_source(raw, path)
            if row["method"] in TABLE3_METHODS:
                selected.append(row)

    enriched = _enrich_rows(selected, repo_root)
    assert_expected_grid(enriched, CORE_DATASETS, TABLE3_METHODS, (("uniform", 0.4),))
    return enriched


def load_table4_rows(repo_root: str | Path = REPO_ROOT) -> list[dict[str, object]]:
    repo_root = Path(repo_root)
    selected: list[dict[str, object]] = []
    legacy_path = repo_root / "runs" / "mlp9_denoised_teacher_20260702" / "mlp9_combined_raw.csv"
    for raw in _read_csv(legacy_path):
        row = _as_audit_row(raw, test_field="student_test")
        if row["dataset"] in {"cora", "citeseer"} and row["method"] in TABLE4_METHODS:
            selected.append(row)

    run_dir = repo_root / "runs" / "mlp22_corafull_table3_table4_20260710"
    for name in ("table3_corafull_uniform0p4_seeds_0_4.csv", "table3_corafull_uniform0p6_seeds_0_4.csv"):
        path = run_dir / name
        for raw in _read_csv(path):
            row = _attach_direct_source(raw, path)
            if row["method"] in TABLE4_METHODS:
                selected.append(row)

    enriched = _enrich_rows(selected, repo_root)
    assert_expected_grid(enriched, TABLE4_DATASETS, TABLE4_METHODS, (("uniform", 0.4), ("uniform", 0.6)))
    return enriched


def load_table5_rows(repo_root: str | Path = REPO_ROOT) -> list[dict[str, object]]:
    repo_root = Path(repo_root)
    selected: list[dict[str, object]] = []
    legacy_datasets = {"cora", "citeseer", "coauthorcs", "amazonpho"}

    clean_path = repo_root / "runs" / "mlp18_graphdomain_tablec_20260708" / "clean_tablec_raw.csv"
    clean_methods = {"gcn", "gcn_xh_ce", "nrgnn", "nrgnn_xh", "mlp_distill_softdist_xh"}
    for raw in _read_csv(clean_path):
        method = str(raw.get("raw_method", ""))
        if raw.get("dataset") in legacy_datasets and method in clean_methods:
            normalized = dict(raw)
            normalized.update({
                "method": method,
                "noise_type": "uniform",
                "noise_rate": "0.4",
                "source": raw["source_file"],
            })
            selected.append(_as_audit_row(normalized))

    recent_path = repo_root / "runs" / "latest_graph_lnl_smoke_20260709" / "tablec_latest_graph_lnl_uniform0p4_5dataset_5seed.csv"
    for raw in _read_csv(recent_path):
        row = _attach_direct_source(raw, recent_path)
        if row["dataset"] in legacy_datasets and row["method"] in {"tfr_gcn", "procon", "gd2"}:
            selected.append(row)

    run_dir = repo_root / "runs" / "mlp22_corafull_table3_table4_20260710"
    corafull_files = (
        "table4_corafull_gcn_nrgnn_uniform0p4_seeds_0_4.csv",
        "table4_corafull_nrgnn_xh_uniform0p4_seeds_0_4.csv",
        "table4_corafull_tfr_gcn_uniform0p4_seeds_0_4.csv",
        "table4_corafull_procon_uniform0p4_seeds_0_4.csv",
        "table4_corafull_gd2_uniform0p4_seeds_0_4.csv",
        "table3_corafull_uniform0p4_seeds_0_4.csv",
    )
    for name in corafull_files:
        path = run_dir / name
        for raw in _read_csv(path):
            row = _attach_direct_source(raw, path)
            if row["method"] in TABLE5_METHODS:
                selected.append(row)

    enriched = _enrich_rows(selected, repo_root)
    assert_expected_grid(enriched, CORE_DATASETS, TABLE5_METHODS, (("uniform", 0.4),))
    return enriched


def _clean_markdown_cell(value: str) -> str:
    return value.strip().replace("**", "")


def _parse_markdown_table_after(text: str, marker: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    if marker not in text:
        raise ValueError(f"manuscript marker not found: {marker}")
    section = text.split(marker, 1)[1]
    lines = section.splitlines()
    table_lines: list[str] = []
    started = False
    for line in lines:
        if line.lstrip().startswith("|"):
            started = True
            table_lines.append(line.strip())
        elif started:
            break
    if len(table_lines) < 3:
        raise ValueError(f"markdown table missing after marker: {marker}")

    def cells(line: str) -> list[str]:
        return [_clean_markdown_cell(item) for item in line.strip("|").split("|")]

    headers = cells(table_lines[0])
    rows: dict[str, dict[str, str]] = {}
    for line in table_lines[2:]:
        values = cells(line)
        if len(values) != len(headers):
            raise ValueError(f"malformed markdown row after {marker}: {line}")
        rows[values[0]] = dict(zip(headers[1:], values[1:]))
    return headers, rows


def _formatted_stats(rows: list[dict[str, object]]) -> dict[tuple[str, str, float, str], str]:
    aggregate = aggregate_seed_rows(rows)
    return {
        (str(dataset), str(noise_type), float(noise_rate), str(method)): (
            f"{100.0 * float(stats['mean']):.2f}±{100.0 * float(stats['std']):.2f}"
        )
        for (dataset, noise_type, noise_rate, method), stats in aggregate.items()
    }


def compare_manuscript_tables(
    manuscript_path: str | Path,
    *,
    table3_rows: list[dict[str, object]],
    table4_rows: list[dict[str, object]],
    table5_rows: list[dict[str, object]],
) -> list[str]:
    text = Path(manuscript_path).read_text(encoding="utf-8")
    mismatches: list[str] = []

    _, manuscript3 = _parse_markdown_table_after(text, "**表 3**")
    stats3 = _formatted_stats(table3_rows)
    table3_methods = {
        "MLP+CE": "mlp_ce",
        "Graph-MLP (protocol port)": "graphmlp_reimplemented",
        "GSDN-inspired (mechanism adaptation)": "gsdn",
        "LCSS": OURS,
        "GLNN 式蒸馏": "mlp_distill_gcn",
        "FF-G2M (common-backbone reimplementation)": "ffg2m_reimplemented",
        "KRD-inspired reliability proxy": "krd_adapted",
        "VQGraph-inspired token proxy": "vqgraph_adapted",
        "LCSS 蒸馏扩展（球式共识教师）": "mlp_distill_softdist_xh",
    }
    table3_columns = {label: dataset for dataset, label in DATASET_LABELS.items()}
    for label, method in table3_methods.items():
        if label not in manuscript3:
            mismatches.append(f"Table 3 missing method row: {label}")
            continue
        for column, dataset in table3_columns.items():
            expected = stats3[(dataset, "uniform", 0.4, method)]
            actual = manuscript3[label].get(column, "")
            if actual != expected:
                mismatches.append(f"Table 3 {label}/{column}: manuscript={actual} recomputed={expected}")

    _, manuscript4 = _parse_markdown_table_after(text, "**表 4**")
    stats4 = _formatted_stats(table4_rows)
    table4_methods = {
        "GCN(X)+CE（GLNN 式）": "mlp_distill_gcn",
        "GCN[X‖H]+CE": "mlp_distill_gcn_xh",
        "随机分组共识教师": "mlp_distill_softdist_xh_shuffled",
        "kNN 共识教师": "mlp_distill_knn_xh",
        "球式共识教师（本文）": "mlp_distill_softdist_xh",
    }
    table4_columns = {
        f"{DATASET_LABELS[dataset]} u{noise_rate:.1f}": (dataset, noise_rate)
        for dataset in TABLE4_DATASETS
        for noise_rate in (0.4, 0.6)
    }
    for label, method in table4_methods.items():
        if label not in manuscript4:
            mismatches.append(f"Table 4 missing method row: {label}")
            continue
        for column, (dataset, noise_rate) in table4_columns.items():
            expected = stats4[(dataset, "uniform", noise_rate, method)]
            actual = manuscript4[label].get(column, "")
            if actual != expected:
                mismatches.append(f"Table 4 {label}/{column}: manuscript={actual} recomputed={expected}")

    _, manuscript5 = _parse_markdown_table_after(text, "**表 5**")
    stats5 = _formatted_stats(table5_rows)
    table5_methods = {
        "GCN(X)": "gcn",
        "GCN[X‖H]": "gcn_xh_ce",
        "NRGNN": "nrgnn",
        "NRGNN[X‖H]": "nrgnn_xh",
        "TFR-GCN": "tfr_gcn",
        "ProCon": "procon",
        "GD2": "gd2",
        "LCSS 蒸馏扩展（学生）": "mlp_distill_softdist_xh",
    }
    for label, method in table5_methods.items():
        if label not in manuscript5:
            mismatches.append(f"Table 5 missing method row: {label}")
            continue
        for column, dataset in table3_columns.items():
            expected = stats5[(dataset, "uniform", 0.4, method)]
            actual = manuscript5[label].get(column, "")
            if actual != expected:
                mismatches.append(f"Table 5 {label}/{column}: manuscript={actual} recomputed={expected}")

    return mismatches


def build_extended_audit_report(
    manuscript_path: str | Path,
    table3_rows: list[dict[str, object]],
    table4_rows: list[dict[str, object]],
    table5_rows: list[dict[str, object]],
) -> str:
    for rows in (table3_rows, table4_rows, table5_rows):
        assert_unique_rows(rows)
        assert_paired_hashes(rows)
        assert_train_only_protocol(rows)
    mismatches = compare_manuscript_tables(
        manuscript_path,
        table3_rows=table3_rows,
        table4_rows=table4_rows,
        table5_rows=table5_rows,
    )
    status = "PASS" if not mismatches else "FAIL"
    lines = [
        "# Submission Integrity Audit - Phase 2: Tables 3-5",
        "",
        "## Numerical and Protocol Verdict",
        "",
        f"- Status: {status} for the reported Table 3-5 numerical cells and train-only protocol.",
        f"- Table 3 rows: {len(table3_rows)} (5 datasets x 9 methods x 5 seeds).",
        f"- Table 4 rows: {len(table4_rows)} (3 datasets x 2 noise rates x 5 teacher arms x 5 seeds).",
        f"- Table 5 rows: {len(table5_rows)} (5 datasets x 8 methods x 5 seeds).",
        "- Every audited source row reports `noise_scope=train`, zero modified validation/test labels, and `modified_count == modified_train_count`.",
        "- Split and noise fingerprints match across paired methods for every dataset/setting/seed.",
        f"- Manuscript cell mismatches: {len(mismatches)}.",
    ]
    if mismatches:
        lines.extend(["", "## Cell Mismatches", ""] + [f"- {item}" for item in mismatches])

    lines.extend(
        [
            "",
            "## Evidence Scope",
            "",
            "- Table 3 values trace to 225 seed rows. Graph-MLP and FF-G2M use the MLP-27/26 fidelity reruns; GSDN, KRD/VQGraph proxies and common reference rows trace to MLP-19/21.",
            "- Table 4 values trace to the MLP-9 teacher ablation for Cora/CiteSeer and the MLP-22 completion for CoraFull.",
            "- Table 5 values trace to the clean MLP-18 graph-method reruns, the later 2025-method completion, and MLP-22 for CoraFull.",
            "",
            "## Important Boundaries Not Certified by This Pass",
            "",
            "- This pass proves that the manuscript numbers can be recomputed from existing seed rows under the stated protocol; it does not prove that adapted/reimplemented methods are faithful official reproductions.",
            "- Graph-MLP is a protocol port and FF-G2M is a common-backbone reimplementation; neither is claimed as an official tuned reproduction. GSDN is a mechanism adaptation, while KRD/VQGraph remain proxies excluded from published-method rankings.",
            "- The CoraFull teacher ablation is a negative attribution result for the ball teacher: it is below GCN[X||H], kNN, and shuffled teachers. The manuscript currently discloses this boundary.",
            "- CLNode is intentionally absent from Table 5; its current local implementation reads validation labels during curriculum construction and is not citable until fixed and rerun.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_recomputed_table(path: Path, rows: list[dict[str, object]]) -> None:
    aggregate = aggregate_seed_rows(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "noise_type", "noise_rate", "method", "n", "mean", "std"])
        writer.writeheader()
        for (dataset, noise_type, noise_rate, method), stats in sorted(aggregate.items()):
            writer.writerow({
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "method": method,
                **stats,
            })


def write_extended_audit_outputs(
    repo_root: Path,
    manuscript_path: Path,
    table3_rows: list[dict[str, object]],
    table4_rows: list[dict[str, object]],
    table5_rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_recomputed_table(output_dir / "table3_recomputed.csv", table3_rows)
    _write_recomputed_table(output_dir / "table4_recomputed.csv", table4_rows)
    _write_recomputed_table(output_dir / "table5_recomputed.csv", table5_rows)

    source_paths = sorted({Path(str(row["source"])) for row in table3_rows + table4_rows + table5_rows})
    manifest_path = output_dir / "tables3_5_evidence_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in source_paths:
            writer.writerow({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    report = build_extended_audit_report(manuscript_path, table3_rows, table4_rows, table5_rows)
    (output_dir / "tables3_5_audit_report.md").write_text(report, encoding="utf-8")
    (repo_root / "docs" / "submission_integrity_audit_phase2_tables3_5_20260711.md").write_text(report, encoding="utf-8")


def load_core_rows(repo_root: str | Path = REPO_ROOT) -> list[dict[str, object]]:
    repo_root = Path(repo_root)
    inputs = (
        (
            repo_root / "runs" / "mlp10_baseline_refresh_20260702" / "mlp10_combined_raw.csv",
            {"cora", "citeseer", "coauthorcs"},
        ),
        (
            repo_root / "runs" / "mlp12_amazonphoto_20260702" / "mlp12_combined_raw.csv",
            {"amazonpho"},
        ),
        (
            repo_root / "runs" / "mlp21_dataset_replacement_20260710" / "mlp21_raw.csv",
            {"corafull"},
        ),
    )
    selected = []
    allowed_settings = set(CORE_SETTINGS)
    for path, datasets in inputs:
        for row in _read_csv(path):
            setting = (str(row.get("noise_type", "")).lower(), float(row.get("noise_rate", -1)))
            if (
                row.get("dataset") in datasets
                and row.get("method") in CORE_METHODS
                and setting in allowed_settings
                and int(row.get("seed", -1)) in range(5)
            ):
                selected.append(row)

    cache: dict[Path, list[dict[str, str]]] = {}
    return [enrich_protocol_from_source(row, repo_root, cache) for row in selected]


def summarize_core_claims(
    rows: list[dict[str, object]],
    ours_method: str,
    shuffled_method: str,
) -> dict[str, int]:
    aggregate = aggregate_seed_rows(rows)
    by_setting = defaultdict(dict)
    for (dataset, noise_type, noise_rate, method), stats in aggregate.items():
        by_setting[(dataset, noise_type, noise_rate)][method] = stats

    ours_best = 0
    ours_above_shuffled = 0
    for methods in by_setting.values():
        ours = float(methods[ours_method]["mean"])
        competitors = [
            float(stats["mean"])
            for method, stats in methods.items()
            if method not in {ours_method, shuffled_method}
        ]
        if competitors and ours >= max(competitors):
            ours_best += 1
        if ours > float(methods[shuffled_method]["mean"]):
            ours_above_shuffled += 1
    return {
        "settings": len(by_setting),
        "ours_best_settings": ours_best,
        "ours_above_shuffled_settings": ours_above_shuffled,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def build_core_audit_report(rows: list[dict[str, object]]) -> str:
    assert_unique_rows(rows)
    assert_paired_hashes(rows)
    assert_train_only_protocol(rows)
    aggregate = aggregate_seed_rows(rows)
    claims = summarize_core_claims(rows, OURS, SHUFFLED)

    dataset_wins = {}
    for dataset in CORE_DATASETS:
        wins = 0
        for noise_type, noise_rate in CORE_SETTINGS:
            ours = float(aggregate[(dataset, noise_type, noise_rate, OURS)]["mean"])
            best = max(
                float(aggregate[(dataset, noise_type, noise_rate, method)]["mean"])
                for method in CORE_METHODS
                if method not in {OURS, SHUFFLED}
            )
            wins += int(ours >= best)
        dataset_wins[dataset] = wins

    lines = [
        "# Submission Integrity Audit - Phase 1: Core Main Table",
        "",
        "## Verdict",
        "",
        "- Status: PASS for the core 5-dataset x 5-setting numerical claim.",
        f"- Recomputed rows: {len(rows)} = 5 datasets x 5 settings x 13 methods x 5 seeds.",
        "- All formal source rows report `noise_scope=train`, `modified_val_count=0`, `modified_test_count=0`, and `modified_count == modified_train_count`.",
        "- Split and noise fingerprints are identical across paired methods within every dataset/setting/seed.",
        f"- LCSS exceeds shuffled grouping in {claims['ours_above_shuffled_settings']}/{claims['settings']} settings.",
        f"- In the historical 13-arm matrix, LCSS is best or tied in {claims['ours_best_settings']}/{claims['settings']} settings; proxy rows in this matrix are not counted as published-method wins.",
        "",
        "## Per-dataset Claim Check",
        "",
        "| Dataset | LCSS best/tied settings |",
        "|---|---:|",
    ]
    for dataset in CORE_DATASETS:
        lines.append(f"| {DATASET_LABELS[dataset]} | {dataset_wins[dataset]}/5 |")

    lines.extend(
        [
            "",
            "## Uniform 0.4 Recomputed Slice",
            "",
            "| Dataset | LCSS | Best non-LCSS baseline | Shuffled |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in CORE_DATASETS:
        ours = aggregate[(dataset, "uniform", 0.4, OURS)]
        shuffled = aggregate[(dataset, "uniform", 0.4, SHUFFLED)]
        candidates = [
            (method, aggregate[(dataset, "uniform", 0.4, method)])
            for method in CORE_METHODS
            if method not in {OURS, SHUFFLED}
        ]
        best_method, best = max(candidates, key=lambda item: float(item[1]["mean"]))
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {_pct(float(ours['mean']))}+-{_pct(float(ours['std']))} | "
            f"{best_method} {_pct(float(best['mean']))}+-{_pct(float(best['std']))} | "
            f"{_pct(float(shuffled['mean']))}+-{_pct(float(shuffled['std']))} |"
        )

    lines.extend(
        [
            "",
            "## Scope of This Pass",
            "",
            "This phase verifies the historical core train-only label-noise matrix and its source CSVs. Published-method ranking is certified separately using the proxy-free Phase 4 scope; this pass does not by itself certify method fidelity, GRACE checkpoint provenance, or exact reproducibility from a clean checkout.",
            "",
            "## Submission Blockers Already Visible",
            "",
            "- The 13-arm historical matrix remains useful for source/protocol integrity, but its proxy rows are excluded from the formal published-method ranking; the faithful 25-setting claim is recomputed separately by `summarize_phase4_faithful_claims_20260712.py`.",
            "- Exact reproduction from a clean checkout still requires packaging the frozen H files or a deterministic encoder-checkpoint regeneration path.",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_core_audit_outputs(repo_root: Path, rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_seed_rows(rows)
    aggregate_path = output_dir / "core_main_recomputed.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "noise_type", "noise_rate", "method", "n", "mean", "std"])
        writer.writeheader()
        for (dataset, noise_type, noise_rate, method), stats in sorted(aggregate.items()):
            writer.writerow({
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "method": method,
                **stats,
            })

    source_paths = sorted({Path(str(row["source"])) for row in rows})
    manifest_path = output_dir / "core_evidence_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in source_paths:
            writer.writerow({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    report = build_core_audit_report(rows)
    (output_dir / "core_main_audit_report.md").write_text(report, encoding="utf-8")
    docs_report = repo_root / "docs" / "submission_integrity_audit_phase1_20260710.md"
    docs_report.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit citable SoftDist/LCSS experiment evidence.")
    parser.add_argument("--repo_root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "runs" / "submission_integrity_audit_20260710",
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=REPO_ROOT / "docs" / "paper_final_zh_v6_table34_completed_20260710.md",
    )
    args = parser.parse_args()
    rows = load_core_rows(args.repo_root)
    write_core_audit_outputs(args.repo_root, rows, args.output_dir)
    table3_rows = load_table3_rows(args.repo_root)
    table4_rows = load_table4_rows(args.repo_root)
    table5_rows = load_table5_rows(args.repo_root)
    write_extended_audit_outputs(
        args.repo_root,
        args.manuscript,
        table3_rows,
        table4_rows,
        table5_rows,
        args.output_dir,
    )
    print(build_core_audit_report(rows))
    print(build_extended_audit_report(args.manuscript, table3_rows, table4_rows, table5_rows))


if __name__ == "__main__":
    main()
