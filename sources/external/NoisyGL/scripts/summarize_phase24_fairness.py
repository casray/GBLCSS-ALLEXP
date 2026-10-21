from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METHOD_LABELS = {
    "gcn": "GCN+CE",
    "nrgnn": "NRGNN",
    "gcn_xh_ce": "GCN+CE [X||H]",
    "nrgnn_xh": "NRGNN [X||H]",
    "tfr": "TFR (CIKM 2025)",
    "tfr_gcn": "TFR-GCN (CIKM 2025)",
    "frozengrace_gb_gcn_softdist_xh": "SoftDist [X||H]",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "SoftDist [X||H] shuffled",
}

FAIRNESS_ROWS = [
    ("nrgnn", "No", "No", "strong NoisyGL baseline"),
    ("nrgnn_xh", "Yes", "No", "H contribution to NRGNN"),
    ("gcn_xh_ce", "Yes", "No", "H contribution to plain CE"),
    ("frozengrace_gb_gcn_softdist_xh", "Yes", "Yes", "full method"),
]


def _test_value(row: dict) -> float:
    if row.get("test_acc") not in (None, ""):
        return float(row["test_acc"])
    return float(row["test"])


def load_raw_rows(paths: list[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method not in METHOD_LABELS:
                    continue
                rows.append({
                    "dataset": row["dataset"].lower(),
                    "method": method,
                    "seed": int(row["seed"]),
                    "test": _test_value(row),
                    "source": str(path),
                })
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(float(row["test"]))
    out = []
    for (dataset, method), values in sorted(grouped.items()):
        values = list(values)
        out.append({
            "dataset": dataset,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "seeds": " ".join(f"{value:.4f}" for value in values),
        })
    return out


def paired_deltas(rows: list[dict], comparisons: list[tuple[str, str]]) -> list[dict]:
    by_key = {(row["dataset"], row["method"], row["seed"]): float(row["test"]) for row in rows}
    datasets = sorted({row["dataset"] for row in rows})
    out = []
    for dataset in datasets:
        seeds = sorted({row["seed"] for row in rows if row["dataset"] == dataset})
        for left, right in comparisons:
            deltas = []
            for seed in seeds:
                left_key = (dataset, left, seed)
                right_key = (dataset, right, seed)
                if left_key in by_key and right_key in by_key:
                    deltas.append(by_key[left_key] - by_key[right_key])
            if not deltas:
                continue
            out.append({
                "dataset": dataset,
                "comparison": f"{left}_minus_{right}",
                "left_label": METHOD_LABELS.get(left, left),
                "right_label": METHOD_LABELS.get(right, right),
                "n": len(deltas),
                "mean_delta": mean(deltas),
                "std_delta": stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_seed_count": sum(delta > 0 for delta in deltas),
                "deltas": " ".join(f"{delta:.4f}" for delta in deltas),
            })
    return out


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}pp"


def write_report(path: str | Path, aggregate: list[dict], deltas: list[dict], args) -> None:
    by_summary = {(row["dataset"], row["method"]): row for row in aggregate}
    path = Path(path)
    lines = [
        "# Phase 24: NRGNN [X||H] Fairness Control",
        "",
        "## Protocol",
        "",
        f"- New run CSV: `{args.nrgnn_xh_raw}`",
        f"- Existing NoisyGL anchors: `{args.base_raw}`",
        f"- Existing XH anchors: `{args.xh_raw}`",
        "- Split: `percent`, train/val/test = `0.2/0.1/0.7`",
        "- Noise: `uniform`, noise rate = `0.4`",
        "- Seeds: `0,1,2,3,4`",
        "",
        "## Fairness Matrix",
        "",
        "| Dataset | Method | H? | Ball? | Test acc | Role |",
        "|---|---|---|---|---:|---|",
    ]
    for dataset in sorted({row["dataset"] for row in aggregate}):
        for method, use_h, use_ball, role in FAIRNESS_ROWS:
            row = by_summary.get((dataset, method))
            if row is None:
                continue
            lines.append(
                f"| {dataset} | {METHOD_LABELS[method]} | {use_h} | {use_ball} | "
                f"{_fmt(row['mean'])} ± {_fmt(row['std'])} | {role} |"
            )
    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "| Dataset | Comparison | Mean delta | Positive seeds | Per-seed deltas |",
        "|---|---|---:|---:|---|",
    ])
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {row['left_label']} - {row['right_label']} | "
            f"{_fmt_pp(row['mean_delta'])} | {row['positive_seed_count']}/{row['n']} | {row['deltas']} |"
        )

    soft_vs_nrgnn_xh = {
        row["dataset"]: row
        for row in deltas
        if row["comparison"] == "frozengrace_gb_gcn_softdist_xh_minus_nrgnn_xh"
    }
    lines.extend([
        "",
        "## Decision",
        "",
        "- Cora: SoftDist [X||H] is effectively tied with NRGNN [X||H]; the mean paired delta is "
        f"{_fmt_pp(soft_vs_nrgnn_xh['cora']['mean_delta'])} with "
        f"{soft_vs_nrgnn_xh['cora']['positive_seed_count']}/{soft_vs_nrgnn_xh['cora']['n']} positive seeds.",
        "- CiteSeer: NRGNN [X||H] is stronger; the mean paired delta for SoftDist [X||H] minus NRGNN [X||H] is "
        f"{_fmt_pp(soft_vs_nrgnn_xh['citeseer']['mean_delta'])}.",
        "- Interpretation: Phase 24 does not support a strong claim that SoftDist beats a fully H-augmented NRGNN. "
        "It does support the narrower, honest claim that SoftDist remains competitive on Cora while being much simpler, "
        "and that H alone explains a substantial part of the CiteSeer gap.",
        "",
        "## Artifacts",
        "",
        f"- Aggregate: `{path.parent / 'phase24_fairness_aggregate.csv'}`",
        f"- Paired deltas: `{path.parent / 'phase24_fairness_paired_deltas.csv'}`",
        f"- Combined raw rows: `{path.parent / 'phase24_fairness_combined_raw.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 24 NRGNN [X||H] fairness control.")
    parser.add_argument("--base_raw", default="runs/softdist_5seed_rocm_20260610/softdist_5seed_rocm_raw_summary_20260610.csv")
    parser.add_argument("--xh_raw", default="runs/xh_5seed_rocm_20260611/xh_5seed_raw_summary_20260611.csv")
    parser.add_argument("--nrgnn_xh_raw", default="runs/phase24_nrgnn_xh_20260616/nrgnn_xh_5seed.csv")
    parser.add_argument("--output_dir", default="runs/phase24_nrgnn_xh_20260616")
    args = parser.parse_args()

    rows = load_raw_rows([args.base_raw, args.xh_raw, args.nrgnn_xh_raw])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows, [
        ("nrgnn_xh", "nrgnn"),
        ("frozengrace_gb_gcn_softdist_xh", "nrgnn_xh"),
        ("frozengrace_gb_gcn_softdist_xh", "gcn_xh_ce"),
        ("frozengrace_gb_gcn_softdist_xh", "frozengrace_gb_gcn_softdist_xh_shuffled"),
        ("gcn_xh_ce", "gcn"),
    ])
    write_csv(output_dir / "phase24_fairness_combined_raw.csv", rows)
    write_csv(output_dir / "phase24_fairness_aggregate.csv", aggregate)
    write_csv(output_dir / "phase24_fairness_paired_deltas.csv", deltas)
    write_report(output_dir / "phase24_fairness_report.md", aggregate, deltas, args)
    print(f"WROTE {output_dir / 'phase24_fairness_report.md'}")


if __name__ == "__main__":
    main()
