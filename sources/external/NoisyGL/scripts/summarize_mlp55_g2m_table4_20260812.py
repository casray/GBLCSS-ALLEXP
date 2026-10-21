from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp55_g2m_table4_20260812"
DEFAULT_BASE_RAW = (
    REPO_ROOT
    / "runs"
    / "mlp44_loo_tiesafe_fullspectrum_20260730"
    / "mlp44_combined_raw.csv"
)
DEFAULT_TEACHER_RAW = (
    REPO_ROOT
    / "runs"
    / "mlp54_teacher_noise_spectrum_20260812"
    / "mlp54_full_spectrum_raw.csv"
)

DATASETS = ("cora", "citeseer", "corafull")
RATES = (0.4, 0.6)
SEEDS = tuple(range(5))
NEW_METHODS = ("ffg2m_reimplemented", "krd_adapted", "vqgraph_adapted")

BASE = "frozengrace_gb_mlp_softdist_beta05_loo"
GLNN = "mlp_distill_gcn"
BALL = "mlp_distill_softdist_xh"
NRGNN = "mlp_distill_nrgnn"
TABLE4_METHODS = (
    BASE,
    GLNN,
    "ffg2m_reimplemented",
    "krd_adapted",
    "vqgraph_adapted",
    BALL,
    NRGNN,
)

METHOD_LABELS = {
    BASE: "Base GB-LCSS (without distillation)",
    GLNN: "GCN(X)+CE teacher (GLNN-style)",
    "ffg2m_reimplemented": "FF-G2M (common-backbone reimplementation)",
    "krd_adapted": "KRD-inspired reliability adaptation",
    "vqgraph_adapted": "VQGraph-inspired token adaptation",
    BALL: "Granular-ball consensus teacher",
    NRGNN: "NRGNN teacher",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _int(row: dict, key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes"}:
        return 1
    if normalized in {"false", "no"}:
        return 0
    return int(float(value))


def _float(row: dict, key: str, default: float | None = None):
    value = row.get(key, "")
    return default if value in (None, "") else float(value)


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def normalize_new_row(row: dict[str, str], source: Path) -> dict[str, object]:
    return {
        "dataset": str(row.get("dataset", "")).lower(),
        "method": str(row.get("method", "")),
        "seed": _int(row, "seed", -1),
        "noise_type": str(row.get("noise_type", "")).lower(),
        "noise_rate": _float(row, "noise_rate", float("nan")),
        "noise_scope": str(row.get("noise_scope", "")),
        "status": str(row.get("status", "")),
        "test": _float(row, "test"),
        "valid": _float(row, "valid"),
        "teacher_test": _float(row, "teacher_best_test"),
        "teacher_valid": _float(row, "teacher_best_val"),
        "split_hash": str(row.get("split_hash", "")),
        "noise_hash": str(row.get("noise_hash", "")),
        "modified_count": _int(row, "modified_count", -1),
        "modified_train_count": _int(row, "modified_train_count", -1),
        "modified_val_count": _int(row, "modified_val_count", -1),
        "modified_test_count": _int(row, "modified_test_count", -1),
        "training_uses_graph": _int(row, "training_uses_graph", -1),
        "inference_uses_graph": _int(row, "inference_uses_graph", -1),
        "uses_unlabeled_nodes": _int(row, "g2m_uses_unlabeled_nodes", -1),
        "implementation_status": str(row.get("implementation_status", "")),
        "wall_time_seconds": _float(row, "wall_time_seconds"),
        "source": str(source),
    }


def load_new_rows(run_dir: Path) -> list[dict[str, object]]:
    paths = sorted(run_dir.glob("smoke_*.csv")) + sorted(run_dir.glob("formal_*.csv"))
    rows: list[dict[str, object]] = []
    for path in paths:
        for row in read_csv(path):
            if row.get("method") in NEW_METHODS:
                rows.append(normalize_new_row(row, path))
    return deduplicate(rows)


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple, dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["noise_type"],
            float(row["noise_rate"]),
            int(row["seed"]),
        )
        if key in unique:
            raise ValueError(f"duplicate experiment row: {key}")
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (
        DATASETS.index(row["dataset"]) if row["dataset"] in DATASETS else 999,
        float(row["noise_rate"]),
        TABLE4_METHODS.index(row["method"]) if row["method"] in TABLE4_METHODS else 999,
        int(row["seed"]),
    ))


def validate_new_rows(
    rows: list[dict[str, object]],
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict[str, object]]:
    expected = {
        (dataset, method, "uniform", rate, seed)
        for dataset in DATASETS
        for rate in RATES
        for method in NEW_METHODS
        for seed in seeds
    }
    observed: set[tuple] = set()
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["method"], row["noise_type"],
            float(row["noise_rate"]), int(row["seed"]),
        )
        if key in observed:
            raise ValueError(f"duplicate experiment row: {key}")
        observed.add(key)
        if key not in expected:
            raise ValueError(f"unexpected experiment row: {key}")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {key}")
        if row.get("noise_scope") != "train":
            raise ValueError(f"noise scope is not train-only: {key}")
        if _int(row, "modified_val_count", -1) != 0 or _int(row, "modified_test_count", -1) != 0:
            raise ValueError(f"validation/test labels were modified: {key}")
        if _int(row, "modified_count", -1) != _int(row, "modified_train_count", -2):
            raise ValueError(f"modified label count mismatch: {key}")
        if not row.get("split_hash") or not row.get("noise_hash"):
            raise ValueError(f"missing protocol hash: {key}")
        if _int(row, "training_uses_graph", -1) != 1:
            raise ValueError(f"training graph identity missing: {key}")
        if _int(row, "inference_uses_graph", -1) != 0:
            raise ValueError(f"student inference is not X-only: {key}")
        if _int(row, "uses_unlabeled_nodes", -1) != 1:
            raise ValueError(f"unlabeled-node teacher identity missing: {key}")
        if not _finite(row.get("test")) or not _finite(row.get("teacher_test")):
            raise ValueError(f"non-finite result: {key}")
        groups[(row["dataset"], float(row["noise_rate"]), int(row["seed"]))].append(row)

    missing = expected - observed
    if missing:
        raise ValueError(f"missing experiment rows: {sorted(missing)[:5]}")

    audit: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        split_hashes = {row["split_hash"] for row in items}
        noise_hashes = {row["noise_hash"] for row in items}
        if len(split_hashes) != 1:
            raise ValueError(f"split hash mismatch: {key}")
        if len(noise_hashes) != 1:
            raise ValueError(f"noise hash mismatch: {key}")
        audit.append({
            "dataset": key[0],
            "noise_type": "uniform",
            "noise_rate": key[1],
            "seed": key[2],
            "method_count": len(items),
            "split_hash": next(iter(split_hashes)),
            "noise_hash": next(iter(noise_hashes)),
            "status": "PASS",
        })
    return audit


def _normalize_context_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = row.get("method", "")
    if method not in (BASE, GLNN, BALL, NRGNN):
        return None
    dataset = row.get("dataset", "").lower()
    noise_type = row.get("noise_type", "").lower()
    noise_rate = _float(row, "noise_rate")
    if dataset not in DATASETS or noise_type != "uniform" or noise_rate not in RATES:
        return None
    test = _float(row, "student_test")
    if test is None:
        test = _float(row, "test")
    return {
        "dataset": dataset,
        "method": method,
        "seed": _int(row, "seed", -1),
        "noise_type": noise_type,
        "noise_rate": noise_rate,
        "noise_scope": row.get("noise_scope", "train"),
        "status": row.get("status", "ok") or "ok",
        "test": test,
        "valid": _float(row, "student_valid", _float(row, "valid")),
        "teacher_test": _float(row, "teacher_test", _float(row, "teacher_best_test")),
        "teacher_valid": _float(row, "teacher_valid", _float(row, "teacher_best_val")),
        "split_hash": row.get("split_hash", ""),
        "noise_hash": row.get("noise_hash", ""),
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 0 if method == BASE else 1,
        "implementation_status": "audited existing table-4 evidence",
        "wall_time_seconds": _float(row, "wall_time_seconds"),
        "source": str(source),
    }


def load_context_rows(base_raw: Path, teacher_raw: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in (base_raw, teacher_raw):
        for raw in read_csv(source):
            row = _normalize_context_row(raw, source)
            if row is not None:
                rows.append(row)
    return deduplicate(rows)


def validate_table4_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    expected = {
        (dataset, method, rate, seed)
        for dataset in DATASETS
        for rate in RATES
        for method in TABLE4_METHODS
        for seed in SEEDS
    }
    observed = {
        (row["dataset"], row["method"], float(row["noise_rate"]), int(row["seed"]))
        for row in rows
    }
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        raise ValueError(f"table-4 grid mismatch: missing={len(missing)} extra={len(extra)}")

    audit: list[dict[str, object]] = []
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or not _finite(row.get("test")):
            raise ValueError(f"invalid table-4 row: {row['dataset']} {row['method']} {row['seed']}")
        grouped[(row["dataset"], float(row["noise_rate"]), int(row["seed"]))].append(row)
    for key, items in sorted(grouped.items()):
        split_hashes = {row["split_hash"] for row in items}
        noise_hashes = {row["noise_hash"] for row in items}
        if len(split_hashes) != 1 or len(noise_hashes) != 1:
            raise ValueError(f"context/new protocol hash mismatch: {key}")
        audit.append({
            "dataset": key[0], "noise_type": "uniform", "noise_rate": key[1],
            "seed": key[2], "method_count": len(items),
            "split_hash": next(iter(split_hashes)), "noise_hash": next(iter(noise_hashes)),
            "status": "PASS",
        })
    return audit


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], float(row["noise_rate"]), row["method"])].append(float(row["test"]))
    out: list[dict[str, object]] = []
    for dataset in DATASETS:
        for rate in RATES:
            for method in TABLE4_METHODS:
                values = grouped[(dataset, rate, method)]
                out.append({
                    "dataset": dataset,
                    "noise_type": "uniform",
                    "noise_rate": rate,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                })
    return out


def completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    observed = {
        (row["dataset"], row["method"], float(row["noise_rate"]), int(row["seed"]))
        for row in rows
    }
    out: list[dict[str, object]] = []
    for dataset in DATASETS:
        for rate in RATES:
            for method in NEW_METHODS:
                seeds = [seed for seed in SEEDS if (dataset, method, rate, seed) in observed]
                out.append({
                    "dataset": dataset, "noise_type": "uniform", "noise_rate": rate,
                    "method": method, "expected": 5, "observed": len(seeds),
                    "missing": " ".join(str(seed) for seed in SEEDS if seed not in seeds),
                })
    return out


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test"])
        for row in rows
    }
    out: list[dict[str, object]] = []
    for method in NEW_METHODS:
        for baseline in (GLNN, BALL, NRGNN):
            for dataset in DATASETS:
                for rate in RATES:
                    deltas = [
                        values[(dataset, rate, method, seed)] - values[(dataset, rate, baseline, seed)]
                        for seed in SEEDS
                    ]
                    out.append({
                        "method": method, "baseline": baseline, "dataset": dataset,
                        "noise_type": "uniform", "noise_rate": rate, "n": 5,
                        "mean_delta": statistics.mean(deltas),
                        "std_delta": statistics.stdev(deltas),
                        "positive_count": sum(delta > 0 for delta in deltas),
                        "seed_deltas": " ".join(f"{delta:.8f}" for delta in deltas),
                    })
    return out


def _cell(row: dict[str, object]) -> str:
    return f"{100 * float(row['mean']):.2f}+-{100 * float(row['std']):.2f}"


def table4_markdown(aggregate_rows: list[dict[str, object]]) -> str:
    lookup = {
        (row["method"], row["dataset"], float(row["noise_rate"])): row
        for row in aggregate_rows
    }
    columns = [(dataset, rate) for rate in RATES for dataset in DATASETS]
    header = "| Training route | Student inference | " + " | ".join(
        f"{dataset} u{rate:.1f}" for dataset, rate in columns
    ) + " |"
    divider = "|---|:---:|" + "---:|" * len(columns)
    lines = [header, divider]
    for method in TABLE4_METHODS:
        cells = [_cell(lookup[(method, dataset, rate)]) for dataset, rate in columns]
        lines.append(f"| {METHOD_LABELS[method]} | X-only | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    new_rows: list[dict[str, object]],
    agg_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    table_md: str,
) -> None:
    new_agg = [row for row in agg_rows if row["method"] in NEW_METHODS]
    lines = [
        "# MLP-55 GNN-to-MLP Table 4 Report",
        "",
        "## Protocol",
        "",
        "- Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only noise, percent 0.2/0.1/0.7.",
        "- Five paired seeds; every dataset--noise--seed independently retrains teacher and student.",
        "- All students use X-only inference.",
        "- FF-G2M is a common-backbone paper reimplementation; KRD and VQGraph rows are mechanism adaptations.",
        "",
        "## New Methods",
        "",
        "| Method | Dataset | Noise | Mean+-std (%) |",
        "|---|---|---:|---:|",
    ]
    for row in new_agg:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['dataset']} | {row['noise_rate']:.1f} | {_cell(row)} |"
        )
    lines.extend(["", "## Table 4", "", table_md.rstrip(), "", "## Paired Deltas", ""])
    lines.extend(["| Method | Baseline | Dataset | Noise | Delta (pp) | Positive seeds |", "|---|---|---|---:|---:|---:|"])
    for row in delta_rows:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {METHOD_LABELS[row['baseline']]} | {row['dataset']} | "
            f"{row['noise_rate']:.1f} | {100 * float(row['mean_delta']):+.2f} | "
            f"{row['positive_count']}/5 |"
        )
    lines.extend(["", "## Completeness", "", f"- New formal records: {len(new_rows)}/90."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--base_raw", default=str(DEFAULT_BASE_RAW))
    parser.add_argument("--teacher_raw", default=str(DEFAULT_TEACHER_RAW))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    new_rows = load_new_rows(run_dir)
    new_audit = validate_new_rows(new_rows)
    context_rows = load_context_rows(Path(args.base_raw), Path(args.teacher_raw))
    table_rows = deduplicate(context_rows + new_rows)
    table_audit = validate_table4_rows(table_rows)
    agg_rows = aggregate(table_rows)
    delta_rows = paired_deltas(table_rows)
    complete_rows = completeness(new_rows)
    table_md = table4_markdown(agg_rows)

    write_csv(run_dir / "mlp55_raw.csv", new_rows)
    write_csv(run_dir / "mlp55_aggregate.csv", [row for row in agg_rows if row["method"] in NEW_METHODS])
    write_csv(run_dir / "mlp55_paired_deltas.csv", delta_rows)
    write_csv(run_dir / "mlp55_completeness.csv", complete_rows)
    write_csv(run_dir / "mlp55_protocol_audit.csv", new_audit)
    write_csv(run_dir / "mlp55_table4.csv", agg_rows)
    write_csv(run_dir / "mlp55_table4_protocol_audit.csv", table_audit)
    (run_dir / "mlp55_table4.md").write_text(table_md, encoding="utf-8")
    write_report(run_dir / "mlp55_report.md", new_rows, agg_rows, delta_rows, table_md)
    print(f"MLP55_NEW_ROWS={len(new_rows)}")
    print(f"MLP55_TABLE4_ROWS={len(table_rows)}")
    print(f"MLP55_PROTOCOL_GROUPS={len(table_audit)}")
    print(f"WROTE {run_dir / 'mlp55_report.md'}")


if __name__ == "__main__":
    main()
