from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp19_g2m_baselines_20260709"
DEFAULT_GRAPHFREE_TABLE = REPO_ROOT / "runs" / "mlp14_graphfree_family_20260703" / "mlp14_combined_raw.csv"

DATASETS = ("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
SEEDS = tuple(range(5))
METHODS = ("ffg2m_adapted", "krd_adapted", "vqgraph_adapted")
OURS = "frozengrace_gb_mlp_softdist_beta05"
DISTILL = "mlp_distill_softdist_xh"
GRAPHFREE_CONTEXT = ("mlp_ce", "graphmlp", "gsdn", OURS, "mlp_distill_gcn", DISTILL)

METHOD_LABELS = {
    "mlp_ce": "Vanilla MLP",
    "graphmlp": "Graph-MLP adapted",
    "gsdn": "GSDN adapted",
    OURS: "Ours beta=0.5 train-only",
    "mlp_distill_gcn": "GLNN-style GCN teacher",
    DISTILL: "LCSS-distill ball teacher",
    "ffg2m_adapted": "FF-G2M adapted",
    "krd_adapted": "KRD adapted",
    "vqgraph_adapted": "VQGraph adapted",
}

RESOURCE_ROWS = [
    {
        "method": "ffg2m_adapted",
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "adapted": 1,
        "resource_note": "GCN teacher plus low/high-frequency adapted distillation to X-only MLP",
    },
    {
        "method": "krd_adapted",
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "adapted": 1,
        "resource_note": "GCN teacher reliability estimated from entropy stability, then distilled to X-only MLP",
    },
    {
        "method": "vqgraph_adapted",
        "training_uses_graph": 1,
        "inference_uses_graph": 0,
        "uses_unlabeled_nodes": 1,
        "adapted": 1,
        "resource_note": "GCN teacher plus adapted structure tokenizer/codebook and soft token distillation to X-only MLP",
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


def _float_or_blank(value):
    if value is None or str(value).strip() == "":
        return ""
    return float(value)


def _normalize_row(row: dict[str, str], source: Path, allowed_methods: tuple[str, ...]) -> dict[str, object] | None:
    method = (row.get("method") or "").strip()
    if method not in allowed_methods:
        return None
    status = (row.get("status") or "ok").strip().lower()
    test = _float_or_blank(row.get("test") or row.get("student_test"))
    if status == "ok" and test == "":
        return None
    return {
        "dataset": row.get("dataset", "").strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": int(row.get("seed", 0)),
        "noise_type": row.get("noise_type", "uniform").strip().lower(),
        "noise_rate": float(row.get("noise_rate", 0.4)),
        "status": status,
        "error": row.get("error", ""),
        "train": _float_or_blank(row.get("train")),
        "valid": _float_or_blank(row.get("valid")),
        "test": test,
        "teacher_best_test": _float_or_blank(row.get("teacher_best_test")),
        "teacher_best_val": _float_or_blank(row.get("teacher_best_val")),
        "g2m_strategy": row.get("g2m_strategy", ""),
        "source": str(source),
    }


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
        unique[key] = row
    return sorted(unique.values(), key=lambda r: (r["dataset"], r["noise_type"], float(r["noise_rate"]), r["method"], int(r["seed"])))


def load_mlp19_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith("mlp19_"):
            continue
        for raw in read_csv(path):
            row = _normalize_row(raw, path, METHODS)
            if row is not None:
                rows.append(row)
    return deduplicate(rows)


def load_graphfree_context(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        for raw in read_csv(path):
            if raw.get("noise_type", "").strip().lower() != "uniform":
                continue
            if float(raw.get("noise_rate", 0.0)) != 0.4:
                continue
            if raw.get("dataset", "").strip().lower() not in DATASETS:
                continue
            row = _normalize_row(raw, path, GRAPHFREE_CONTEXT)
            if row is not None:
                rows.append(row)
    return deduplicate(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and row["test"] != "":
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


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test"])
        for row in rows
        if row["status"] == "ok" and row["test"] != "" and row["noise_type"] == "uniform" and float(row["noise_rate"]) == 0.4
    }
    out = []
    for method in METHODS:
        for baseline in (OURS, DISTILL, "graphmlp", "gsdn"):
            for dataset in DATASETS:
                deltas = []
                seeds = []
                for seed in SEEDS:
                    left = values.get((dataset, method, seed))
                    right = values.get((dataset, baseline, seed))
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
                    "noise_type": "uniform",
                    "noise_rate": 0.4,
                    "n": len(deltas),
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "positive_count": sum(delta > 0 for delta in deltas),
                    "seeds": " ".join(map(str, seeds)),
                    "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                })
    return out


def completeness(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    observed = {(row["dataset"], row["method"], int(row["seed"])) for row in rows}
    out = []
    for dataset in DATASETS:
        for method in METHODS:
            seeds = sorted(seed for seed in SEEDS if (dataset, method, seed) in observed)
            out.append({
                "dataset": dataset,
                "method": method,
                "expected": len(SEEDS),
                "observed": len(seeds),
                "missing": " ".join(str(seed) for seed in SEEDS if seed not in seeds),
                "seeds": " ".join(str(seed) for seed in seeds),
            })
    return out


def _pct(value):
    if value == "":
        return "NA"
    return f"{100.0 * float(value):.2f}"


def _pct_pm(row):
    if row is None:
        return "NA"
    return f"{_pct(row['mean'])}+/-{_pct(row['std'])} ({row['n']})"


def write_report(path: Path, rows, agg_rows, delta_rows, complete_rows) -> None:
    agg = {(row["dataset"], row["method"]): row for row in agg_rows if row["noise_type"] == "uniform" and float(row["noise_rate"]) == 0.4}
    not_runnable = [row for row in rows if row["status"] == "not_runnable"]
    lines = [
        "# MLP-19 GNN-to-MLP Baselines Report",
        "",
        "## Material Passport",
        "- Table role: Table B / graph-free GNN-to-MLP family expansion only.",
        "- Protocol: Cora/CiteSeer/PubMed/Coauthor-CS/Amazon-Photo, uniform0.4, train-only noise, percent split 0.2/0.1/0.7.",
        "- New runnable methods are adapted to this repository protocol, not official reproduction.",
        "- VQGraph is implemented as an adapted core mechanism, not a line-by-line official reproduction.",
        "",
        "## Resource Annotation",
        "| Method | train graph | infer graph | unlabeled nodes | adapted | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in RESOURCE_ROWS:
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['training_uses_graph']} | {row['inference_uses_graph']} | "
            f"{row['uses_unlabeled_nodes']} | {row['adapted']} | {row['resource_note']} |"
        )
    lines.extend([
        "",
        "## Uniform0.4 Table B Slice",
        "| Dataset | MLP | Graph-MLP | GSDN | Ours train-only | GLNN | LCSS-distill | FF-G2M adapted | KRD adapted | VQGraph adapted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    methods = ("mlp_ce", "graphmlp", "gsdn", OURS, "mlp_distill_gcn", DISTILL, "ffg2m_adapted", "krd_adapted", "vqgraph_adapted")
    for dataset in DATASETS:
        values = [_pct_pm(agg.get((dataset, method))) for method in methods]
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Paired Deltas",
        "| Method | Baseline | Dataset | n | Delta | Positive |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in delta_rows:
        lines.append(
            f"| {row['method']} | {row['baseline']} | {row['dataset']} | {row['n']} | "
            f"{100.0 * float(row['mean_delta']):+.2f}pp | {row['positive_count']}/{row['n']} |"
        )
    lines.extend([
        "",
        "## Not Runnable Records",
        "| Dataset | Method | Seeds | Reason |",
        "|---|---|---|---|",
    ])
    by_dataset = defaultdict(list)
    for row in not_runnable:
        by_dataset[row["dataset"]].append(row)
    if not by_dataset:
        lines.append("| - | - | - | none |")
    else:
        for dataset, items in sorted(by_dataset.items()):
            seeds = " ".join(str(row["seed"]) for row in sorted(items, key=lambda r: int(r["seed"])))
            reason = items[0]["error"]
            lines.append(f"| {dataset} | {items[0]['method']} | {seeds} | {reason} |")
    lines.extend([
        "",
        "## Completeness",
        "| Dataset | Method | expected | observed | missing |",
        "|---|---|---:|---:|---|",
    ])
    for row in complete_rows:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['expected']} | {row['observed']} | {row['missing']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--graphfree_table", default=str(DEFAULT_GRAPHFREE_TABLE))
    parser.add_argument(
        "--distill_tables",
        default="runs/mlp9_denoised_teacher_20260702/mlp9_combined_raw.csv,runs/mlp16_distill_tableb_completion_20260707/distill_tableb_uniform0p4_seeds_0_1_2_3_4.csv",
        help="Comma-separated raw distillation tables that contain mlp_distill_gcn and mlp_distill_softdist_xh rows.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    mlp19_rows = load_mlp19_rows(run_dir)
    context_paths = [Path(args.graphfree_table)]
    context_paths.extend(Path(item) for item in args.distill_tables.split(",") if item.strip())
    context_rows = load_graphfree_context(context_paths)
    combined = deduplicate(context_rows + mlp19_rows)
    agg = aggregate(combined)
    deltas = paired_deltas(combined)
    complete = completeness(mlp19_rows)

    write_csv(run_dir / "mlp19_raw.csv", mlp19_rows)
    write_csv(run_dir / "mlp19_combined_raw.csv", combined)
    write_csv(run_dir / "mlp19_aggregate.csv", agg)
    write_csv(run_dir / "mlp19_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp19_completeness.csv", complete)
    write_csv(run_dir / "mlp19_resource_annotations.csv", RESOURCE_ROWS)
    write_report(run_dir / "mlp19_report.md", combined, agg, deltas, complete)

    missing = sum(1 for row in complete if int(row["observed"]) != int(row["expected"]))
    not_runnable = sum(1 for row in mlp19_rows if row["status"] == "not_runnable")
    print(f"MLP19_ROWS={len(mlp19_rows)}")
    print(f"MLP19_CONTEXT_ROWS={len(context_rows)}")
    print(f"MLP19_NOT_RUNNABLE_ROWS={not_runnable}")
    print(f"MLP19_MISSING_GROUPS={missing}")
    print(f"WROTE {run_dir / 'mlp19_report.md'}")


if __name__ == "__main__":
    main()
