from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp14_graphfree_family_20260703"
MLP_MAIN_DIR = REPO_ROOT / "runs" / "mlp_maintable_20260701"
MLP10_DIR = REPO_ROOT / "runs" / "mlp10_baseline_refresh_20260702"
MLP12_DIR = REPO_ROOT / "runs" / "mlp12_amazonphoto_20260702"
MLP9_DIR = REPO_ROOT / "runs" / "mlp9_denoised_teacher_20260702"

DATASETS = ("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
METHODS = ("gsdn", "graphmlp")
REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

OURS = "frozengrace_gb_mlp_softdist_beta05"
GLNN = "mlp_distill_gcn"
GLNN_XH = "mlp_distill_gcn_xh"
DISTILL = "mlp_distill_knn_xh"
GCN_REF = "gcn"
GCN_XH_REF = "gcn_xh_ce"
REF_METHODS = ("mlp_ce", OURS, GCN_REF, GCN_XH_REF, GLNN, GLNN_XH, DISTILL)

METHOD_LABELS = {
    "mlp_ce": "Vanilla MLP",
    "gsdn": "GSDN adapted",
    "graphmlp": "Graph-MLP adapted",
    OURS: "Ours beta=0.5 train-only",
    GCN_REF: "GCN(X)+CE",
    GCN_XH_REF: "GCN[X||H]+CE",
    GLNN: "GLNN-style GCN teacher",
    GLNN_XH: "GLNN-style GCN[X||H] teacher",
    DISTILL: "Ours-distill kNN teacher",
}

RESOURCE_ROWS = [
    {
        "method": "mlp_ce",
        "training_uses_graph": 0,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 0,
        "uses_all_edges": 0,
        "resource_note": "X-only supervised MLP baseline",
    },
    {
        "method": "graphmlp",
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "NContrast-style graph regularization during training",
    },
    {
        "method": "gsdn",
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "Feature/label neighbor self-distillation during training",
    },
    {
        "method": OURS,
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "Frozen GRACE-H preprocessing uses the full graph; downstream MLP is X-only",
    },
    {
        "method": GCN_REF,
        "training_uses_graph": 1,
        "inference_uses_graph": 1,
        "uses_unlabeled_nodes": 0,
        "uses_all_edges": 1,
        "resource_note": "Reference message-passing model; not graph-free at inference",
    },
    {
        "method": GCN_XH_REF,
        "training_uses_graph": 1,
        "inference_uses_graph": 1,
        "uses_unlabeled_nodes": 0,
        "uses_all_edges": 1,
        "resource_note": "Reference message-passing model with frozen-H input",
    },
    {
        "method": GLNN,
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "GCN(X) teacher distills to X-only student",
    },
    {
        "method": GLNN_XH,
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "GCN[X||H] teacher distills to X-only student",
    },
    {
        "method": DISTILL,
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "uses_all_edges": 1,
        "resource_note": "GCN[X||H] teacher distills to X-only student",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _float(value, default=""):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _matches_train_percent(row: dict[str, str], expected: float = 0.2) -> bool:
    value = row.get("train_percent")
    if value is None or str(value).strip() == "":
        return True
    return math.isclose(float(value), expected, abs_tol=1e-12)


def normalize_graphfree(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in METHODS:
        return None
    test = row.get("test") or row.get("test_acc") or ""
    if test == "":
        return None
    return {
        "source_group": "mlp14",
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(row["seed"]),
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "status": row.get("status", "ok").strip().lower(),
        "error": row.get("error", ""),
        "valid": _float(row.get("valid")),
        "test": float(test),
        "graph_batch_size": _float(row.get("graph_batch_size")),
        "lambda_feature": _float(row.get("lambda_feature")),
        "lambda_label": _float(row.get("lambda_label")),
        "lambda_negative": _float(row.get("lambda_negative")),
        "lambda_contrast": _float(row.get("lambda_contrast")),
        "tau": _float(row.get("tau")),
        "source": str(source),
    }


def normalize_reference(row: dict[str, str], source: Path, source_group: str) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in REF_METHODS:
        return None
    test = row.get("test") or row.get("student_test") or row.get("test_acc") or ""
    if test == "":
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    return {
        "source_group": source_group,
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(row["seed"]),
        "noise_type": row.get("noise_type", "uniform").strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "status": "ok",
        "error": "",
        "valid": _float(row.get("valid")),
        "test": float(test),
        "graph_batch_size": "",
        "lambda_feature": "",
        "lambda_label": "",
        "lambda_negative": "",
        "lambda_contrast": "",
        "tau": "",
        "source": str(source),
    }


def normalize_teacher_reference(row: dict[str, str], source: Path) -> list[dict[str, object]]:
    method = (row.get("method") or "").strip()
    if method not in {GLNN, GLNN_XH}:
        return []
    test = row.get("teacher_test") or ""
    if test == "":
        return []
    mapped_method = GCN_REF if method == GLNN else GCN_XH_REF
    return [{
        "source_group": "mlp9_teacher",
        "dataset": row["dataset"].strip().lower(),
        "method": mapped_method,
        "method_label": METHOD_LABELS[mapped_method],
        "seed": int(row["seed"]),
        "noise_type": row.get("noise_type", "uniform").strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "status": "ok",
        "error": "",
        "valid": _float(row.get("teacher_best_val")),
        "test": float(test),
        "graph_batch_size": "",
        "lambda_feature": "",
        "lambda_label": "",
        "lambda_negative": "",
        "lambda_contrast": "",
        "tau": "",
        "source": str(source),
    }]


def load_graphfree_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp14_"):
            continue
        for raw in read_csv(path):
            row = normalize_graphfree(raw, path)
            if row is not None:
                rows.append(row)
    return deduplicate(rows)


def load_reference_rows(mlp_main_dir: Path, mlp10_dir: Path, mlp12_dir: Path, mlp9_dir: Path, run_dir: Path | None = None) -> list[dict[str, object]]:
    rows = []
    for raw in read_csv(mlp_main_dir / "mlp_maintable_combined_raw.csv"):
        if not _matches_train_percent(raw, 0.2):
            continue
        row = normalize_reference(raw, mlp_main_dir / "mlp_maintable_combined_raw.csv", "mlp_main")
        if row and row["dataset"] in {"cora", "citeseer", "pubmed"} and row["method"] in {"mlp_ce", OURS}:
            rows.append(row)
    for raw in read_csv(mlp10_dir / "mlp10_combined_raw.csv"):
        row = normalize_reference(raw, mlp10_dir / "mlp10_combined_raw.csv", "mlp10")
        if row and row["dataset"] == "coauthorcs" and row["method"] in {"mlp_ce", OURS}:
            rows.append(row)
    for raw in read_csv(mlp12_dir / "mlp12_combined_raw.csv"):
        row = normalize_reference(raw, mlp12_dir / "mlp12_combined_raw.csv", "mlp12")
        if row and row["dataset"] == "amazonpho" and row["method"] in {"mlp_ce", OURS}:
            rows.append(row)
    for raw in read_csv(mlp9_dir / "mlp9_combined_raw.csv"):
        row = normalize_reference(raw, mlp9_dir / "mlp9_combined_raw.csv", "mlp9")
        if row and row["method"] in {GLNN, GLNN_XH, DISTILL}:
            rows.append(row)
        rows.extend(normalize_teacher_reference(raw, mlp9_dir / "mlp9_combined_raw.csv"))
    if run_dir is not None:
        for path in sorted(run_dir.glob("gcn_reference_*.csv")):
            for raw in read_csv(path):
                row = normalize_reference(raw, path, "gcn_reference")
                if row and row["method"] in {GCN_REF, GCN_XH_REF}:
                    rows.append(row)
    return deduplicate(rows)


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row["noise_type"],
            float(row["noise_rate"]),
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test"]), float(row["test"]), abs_tol=1e-12):
                # Prefer the explicit graph-free run over reference duplicates; references
                # should otherwise be internally consistent.
                if row["source_group"] == "mlp14":
                    unique[key] = row
                continue
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] in {"", "ok"}:
            grouped[(row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], row["method_label"])].append(row)
    out = []
    for (dataset, noise_type, noise_rate, method, label), items in sorted(grouped.items()):
        values = [float(row["test"]) for row in items]
        out.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "method_label": label,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seeds": " ".join(str(row["seed"]) for row in sorted(items, key=lambda r: int(r["seed"]))),
        })
    return out


def smoke_gate_rows(graphfree_rows: list[dict[str, object]], reference_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ref = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), int(row["seed"])): row
        for row in reference_rows
        if row["method"] == "mlp_ce"
    }
    out = []
    for method in METHODS:
        for dataset in DATASETS:
            for noise_type, noise_rate in (("uniform", 0.4), ("uniform", 0.6)):
                row = next(
                    (
                        item for item in graphfree_rows
                        if item["method"] == method
                        and item["dataset"] == dataset
                        and item["noise_type"] == noise_type
                        and float(item["noise_rate"]) == noise_rate
                        and int(item["seed"]) == 0
                    ),
                    None,
                )
                ce = ref.get((dataset, noise_type, noise_rate, 0))
                status_ok = row is not None and row["status"] in {"", "ok"} and math.isfinite(float(row["test"]))
                ce_gate = True
                delta_vs_ce = ""
                if noise_type == "uniform" and math.isclose(noise_rate, 0.4):
                    ce_gate = ce is not None and row is not None and float(row["test"]) >= float(ce["test"]) - 0.02
                    if ce is not None and row is not None:
                        delta_vs_ce = float(row["test"]) - float(ce["test"])
                out.append({
                    "method": method,
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "seed": 0,
                    "test": "" if row is None else row["test"],
                    "mlp_ce_test": "" if ce is None else ce["test"],
                    "delta_vs_mlp_ce": delta_vs_ce,
                    "status_ok": int(status_ok),
                    "ce_gate": int(ce_gate),
                    "passes": int(status_ok and ce_gate),
                })
    return out


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test"])
        for row in rows
        if row["status"] in {"", "ok"}
    }
    out = []
    for method in METHODS:
        for baseline in (OURS, DISTILL):
            for dataset in DATASETS:
                for noise_type, noise_rate in REGIMES:
                    deltas = []
                    seeds = []
                    for seed in SEEDS:
                        left = values.get((dataset, noise_type, noise_rate, method, seed))
                        right = values.get((dataset, noise_type, noise_rate, baseline, seed))
                        if left is None or right is None:
                            continue
                        deltas.append(left - right)
                        seeds.append(seed)
                    if not deltas:
                        continue
                    out.append({
                        "method": method,
                        "baseline": baseline,
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": noise_rate,
                        "n": len(deltas),
                        "mean_delta": statistics.mean(deltas),
                        "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        "positive_count": sum(delta > 0 for delta in deltas),
                        "seeds": " ".join(map(str, seeds)),
                        "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                    })
    return out


def completeness_rows(graphfree_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    observed = {
        (row["dataset"], row["method"], row["noise_type"], float(row["noise_rate"]), int(row["seed"]))
        for row in graphfree_rows
        if row["status"] in {"", "ok"}
    }
    out = []
    for dataset in DATASETS:
        for method in METHODS:
            for noise_type, noise_rate in REGIMES:
                seeds = sorted(seed for seed in SEEDS if (dataset, method, noise_type, noise_rate, seed) in observed)
                out.append({
                    "dataset": dataset,
                    "method": method,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "expected": len(SEEDS),
                    "observed": len(seeds),
                    "missing": " ".join(str(seed) for seed in SEEDS if seed not in seeds),
                    "seeds": " ".join(str(seed) for seed in seeds),
                })
    return out


def hyperparam_rows(graphfree_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {}
    for row in graphfree_rows:
        key = (row["method"], row["dataset"])
        if key in seen:
            continue
        seen[key] = {
            "method": row["method"],
            "dataset": row["dataset"],
            "selection_protocol": "fixed_default_adapted_to_protocol",
            "graph_batch_size": row.get("graph_batch_size", ""),
            "lambda_feature": row.get("lambda_feature", ""),
            "lambda_label": row.get("lambda_label", ""),
            "lambda_negative": row.get("lambda_negative", ""),
            "lambda_contrast": row.get("lambda_contrast", ""),
            "tau": row.get("tau", ""),
        }
    return sorted(seen.values(), key=lambda row: (row["method"], row["dataset"]))


def _fmt(value) -> str:
    if value == "":
        return "NA"
    return f"{100.0 * float(value):.2f}"


def _fmt_delta(value) -> str:
    if value == "":
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def write_report(path: Path, agg_rows, smoke_rows, delta_rows, complete_rows, hyper_rows) -> None:
    missing = [row for row in complete_rows if int(row["observed"]) != int(row["expected"])]
    smoke_pass = all(int(row["passes"]) == 1 for row in smoke_rows) if smoke_rows else False
    by_key = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"]): row
        for row in agg_rows
    }
    lines = [
        "# MLP-14 Graph-free Family Report",
        "",
        "## Material Passport",
        "- Table role: Table B / graph-free family positioning only.",
        "- Protocol: train-only label noise, percent split 0.2/0.1/0.7.",
        "- Methods: adapted GSDN and adapted Graph-MLP with X-only MLP inference.",
        "- These methods use graph structure during training and are not added to main Table A.",
        f"- Smoke gate: {'pass' if smoke_pass else 'not-pass'}",
        "",
        "## Resource Annotation",
        "| Method | train graph | infer graph | unlabeled nodes | all edges | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in RESOURCE_ROWS:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['training_uses_graph']} | {row['inference_uses_graph']} | "
            f"{row['uses_unlabeled_nodes']} | {row['uses_all_edges']} | {row['resource_note']} |"
        )
    lines.extend([
        "",
        "## Smoke Gate",
        "| Method | Dataset | Noise | Test | MLP CE | Delta vs CE | status | CE gate | pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in smoke_rows:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['noise_type']}{row['noise_rate']} | "
            f"{_fmt(row['test'])} | {_fmt(row['mlp_ce_test'])} | {_fmt_delta(row['delta_vs_mlp_ce'])} | "
            f"{row['status_ok']} | {row['ce_gate']} | {row['passes']} |"
        )
    lines.extend([
        "",
        "## Graph-free Family Table B",
        "| Dataset | Noise | MLP | Graph-MLP | GSDN | Ours train-only | GCN | GCN[X||H] | GLNN | GLNN[X||H] | Ours-distill |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    ordered_methods = ("mlp_ce", "graphmlp", "gsdn", OURS, GCN_REF, GCN_XH_REF, GLNN, GLNN_XH, DISTILL)
    for dataset in DATASETS:
        for noise_type, noise_rate in REGIMES:
            values = []
            for method in ordered_methods:
                row = by_key.get((dataset, noise_type, noise_rate, method))
                values.append("NA" if row is None else f"{_fmt(row['mean'])}+/-{_fmt(row['std'])} ({row['n']})")
            lines.append(f"| {dataset} | {noise_type}{noise_rate} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Paired Deltas",
        "| Method | Baseline | Dataset | Noise | n | Delta | Pos |",
        "|---|---|---|---|---:|---:|---:|",
    ])
    for row in delta_rows:
        stable = int(int(row["n"]) == 5 and int(row["positive_count"]) >= 4)
        lines.append(
            f"| {row['method']} | {row['baseline']} | {row['dataset']} | {row['noise_type']}{row['noise_rate']} | "
            f"{row['n']} | {_fmt_delta(row['mean_delta'])} | {row['positive_count']}/{row['n']} stable={stable} |"
        )
    lines.extend([
        "",
        "## Hyperparameter Record",
        "| Method | Dataset | protocol | batch | lf | ll | ln | lc | tau |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in hyper_rows:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['selection_protocol']} | {row['graph_batch_size']} | "
            f"{row['lambda_feature']} | {row['lambda_label']} | {row['lambda_negative']} | "
            f"{row['lambda_contrast']} | {row['tau']} |"
        )
    lines.extend([
        "",
        "## Completeness",
        f"- Expected graph-free rows: {len(DATASETS) * len(METHODS) * len(REGIMES) * len(SEEDS)}",
        f"- Missing dataset/method/regime groups: {len(missing)}",
    ])
    if missing:
        for row in missing[:50]:
            lines.append(f"- missing {row['dataset']} {row['method']} {row['noise_type']}{row['noise_rate']}: {row['missing']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--mlp_main_dir", default=str(MLP_MAIN_DIR))
    parser.add_argument("--mlp10_dir", default=str(MLP10_DIR))
    parser.add_argument("--mlp12_dir", default=str(MLP12_DIR))
    parser.add_argument("--mlp9_dir", default=str(MLP9_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    graphfree_rows = load_graphfree_rows(run_dir)
    reference_rows = load_reference_rows(Path(args.mlp_main_dir), Path(args.mlp10_dir), Path(args.mlp12_dir), Path(args.mlp9_dir), run_dir)
    all_rows = deduplicate(graphfree_rows + reference_rows)
    agg = aggregate(all_rows)
    smoke = smoke_gate_rows(graphfree_rows, reference_rows)
    deltas = paired_deltas(all_rows)
    complete = completeness_rows(graphfree_rows)
    hyper = hyperparam_rows(graphfree_rows)

    write_csv(run_dir / "mlp14_graphfree_raw.csv", graphfree_rows)
    write_csv(run_dir / "mlp14_reference_raw.csv", reference_rows)
    write_csv(run_dir / "mlp14_combined_raw.csv", all_rows)
    write_csv(run_dir / "mlp14_aggregate.csv", agg)
    write_csv(run_dir / "mlp14_smoke_gate.csv", smoke)
    write_csv(run_dir / "mlp14_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp14_completeness.csv", complete)
    write_csv(run_dir / "mlp14_hyperparams.csv", hyper)
    write_csv(run_dir / "mlp14_resource_annotations.csv", RESOURCE_ROWS)
    write_report(run_dir / "mlp14_report.md", agg, smoke, deltas, complete, hyper)

    smoke_pass = all(int(row["passes"]) == 1 for row in smoke) if smoke else False
    missing_count = sum(1 for row in complete if int(row["observed"]) != int(row["expected"]))
    print(f"MLP14_GRAPHFREE_ROWS={len(graphfree_rows)}")
    print(f"MLP14_REFERENCE_ROWS={len(reference_rows)}")
    print(f"MLP14_SMOKE={'pass' if smoke_pass else 'not-pass'}")
    print(f"MLP14_MISSING_GROUPS={missing_count}")
    print(f"WROTE {run_dir / 'mlp14_report.md'}")


if __name__ == "__main__":
    main()
