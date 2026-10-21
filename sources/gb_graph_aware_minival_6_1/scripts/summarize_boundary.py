"""Summarize validity-boundary experiments for local-consistency selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.utils import save_csv, save_json


WARNING_FIELDS = [
    "topk_neighborhood_majority_self_disagreement",
    "topk_neighborhood_majority_confidence",
    "topk_pairflip_neighborhood_consistency",
]


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def _std(values: list[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if len(vals) <= 1:
        return 0.0
    mean = _mean(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) <= 1:
        return math.nan
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx == 0.0 or vy == 0.0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


def _rank(values: list[float]) -> list[float]:
    order = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        rank = (i + j) / 2.0
        for _, idx in order[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) <= 1:
        return math.nan
    xvals, yvals = zip(*pairs)
    return _pearson(_rank(list(xvals)), _rank(list(yvals)))


def _group_rows(rows: list[dict]) -> dict[tuple[str, str, float], list[dict]]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["noise_type"], _to_float(row["noise_rate"]))].append(row)
    return dict(grouped)


def boundary_summary_rows(rows: list[dict]) -> list[dict]:
    """Create one row per dataset/noise/rate with selection-quality and method deltas."""
    out = []
    for (dataset, noise_type, noise_rate), local in sorted(_group_rows(rows).items()):
        methods = defaultdict(list)
        for row in local:
            methods[row["method"]].append(row)
        topk = methods.get("sel_topk", [])
        if not topk:
            continue
        ce_vals = [_to_float(r["test_acc_at_best_val"]) for r in methods.get("ce", [])]
        coteach_vals = [_to_float(r["test_acc_at_best_val"]) for r in methods.get("coteaching", [])]
        dcgb_vals = [_to_float(r["test_acc_at_best_val"]) for r in methods.get("sel_dcgb", [])]
        topk_vals = [_to_float(r["test_acc_at_best_val"]) for r in topk]
        random_clean_rate = 1.0 - float(noise_rate)
        row = {
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": float(noise_rate),
            "random_clean_rate": random_clean_rate,
            "sel_topk_mean": _mean(topk_vals),
            "ce_mean": _mean(ce_vals),
            "coteaching_mean": _mean(coteach_vals),
            "sel_dcgb_mean": _mean(dcgb_vals),
            "selection_precision_mean": _mean([_to_float(r.get("selection_precision")) for r in topk]),
            "selection_precision_std": _std([_to_float(r.get("selection_precision")) for r in topk]),
            "noise_excluded_recall_mean": _mean([_to_float(r.get("noise_excluded_recall")) for r in topk]),
        }
        row["selection_precision_minus_random_clean_rate"] = row["selection_precision_mean"] - random_clean_rate
        row["sel_topk_minus_ce"] = row["sel_topk_mean"] - row["ce_mean"]
        row["sel_topk_minus_coteaching"] = row["sel_topk_mean"] - row["coteaching_mean"]
        row["sel_dcgb_minus_topk"] = row["sel_dcgb_mean"] - row["sel_topk_mean"]
        for field in WARNING_FIELDS:
            row[field] = _mean([_to_float(r.get(field)) for r in topk])
        out.append(row)
    return out


def compute_boundary_correlations(summary_rows: list[dict], warning_fields: list[str] | None = None) -> dict[str, float]:
    warning_fields = warning_fields or WARNING_FIELDS
    out = {}
    for field in warning_fields:
        xs = [_to_float(row.get(field)) for row in summary_rows]
        selection_gap = [_to_float(row.get("selection_precision_minus_random_clean_rate")) for row in summary_rows]
        test_delta = [_to_float(row.get("sel_topk_minus_ce")) for row in summary_rows]
        out[f"{field}_vs_selection_gap_pearson"] = _pearson(xs, selection_gap)
        out[f"{field}_vs_selection_gap_spearman"] = _spearman(xs, selection_gap)
        out[f"{field}_vs_test_delta_pearson"] = _pearson(xs, test_delta)
        out[f"{field}_vs_test_delta_spearman"] = _spearman(xs, test_delta)
    return out


def evaluate_boundary_claims(setting_rows: list[dict], correlations: dict[str, float]) -> dict[str, dict]:
    sym04 = [
        row
        for row in setting_rows
        if row["noise_type"] == "symmetric" and abs(float(row["noise_rate"]) - 0.4) < 1e-9
    ]
    c1_pass = bool(sym04) and all(
        row.get("topk_minus_coteaching_sig") == "SIG"
        and _to_float(row.get("topk_minus_coteaching_pp")) > 0
        and int(row.get("topk_minus_coteaching_wins", 0)) >= 4
        for row in sym04
    )

    pair_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in setting_rows:
        if row["noise_type"] == "pair":
            pair_by_dataset[row["dataset"]].append(row)
    pair_pass_datasets = []
    for dataset, rows in pair_by_dataset.items():
        rows = sorted(rows, key=lambda r: float(r["noise_rate"]))
        rates = [float(r["noise_rate"]) for r in rows]
        gaps = [_to_float(r.get("selection_precision_minus_random_clean_rate")) for r in rows]
        if len(rows) >= 3 and rates == sorted(rates):
            monotonic = all(gaps[i] >= gaps[i + 1] - 1e-12 for i in range(len(gaps) - 1))
            rate06_nonpositive = any(abs(float(r["noise_rate"]) - 0.6) < 1e-9 and _to_float(r.get("selection_precision_minus_random_clean_rate")) <= 0 for r in rows)
            if monotonic and rate06_nonpositive:
                pair_pass_datasets.append(dataset)
    c2_pass = bool(pair_pass_datasets)

    max_abs_r = math.nan
    if correlations:
        vals = [abs(_to_float(v)) for key, v in correlations.items() if key.endswith("_pearson") and not math.isnan(_to_float(v))]
        max_abs_r = max(vals) if vals else math.nan
    c3_pass = not math.isnan(max_abs_r) and max_abs_r >= 0.6

    dcgb_rows = [row for row in setting_rows if "dcgb_minus_topk_pp" in row]
    c4_pass = bool(dcgb_rows) and all(
        _to_float(row.get("dcgb_minus_topk_pp")) <= 0 or row.get("dcgb_minus_topk_sig") == "ns" for row in dcgb_rows
    )

    return {
        "C1_symmetric_positive": {"passes": c1_pass, "checked_settings": len(sym04)},
        "C2_pair_boundary": {"passes": c2_pass, "datasets_passing": pair_pass_datasets},
        "C3_label_free_warning": {"passes": c3_pass, "max_abs_pearson_r": max_abs_r},
        "C4_ball_no_increment": {"passes": c4_pass, "checked_settings": len(dcgb_rows)},
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _plot_outputs(summary_rows: list[dict], correlations: dict[str, float], out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: validity boundary
    plt.figure(figsize=(8, 5))
    for dataset in sorted({r["dataset"] for r in summary_rows}):
        for noise_type in ["symmetric", "pair"]:
            rows = sorted(
                [r for r in summary_rows if r["dataset"] == dataset and r["noise_type"] == noise_type],
                key=lambda r: float(r["noise_rate"]),
            )
            if not rows:
                continue
            plt.plot(
                [r["noise_rate"] for r in rows],
                [r["selection_precision_minus_random_clean_rate"] for r in rows],
                marker="o",
                label=f"{dataset}-{noise_type}",
            )
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Noise rate")
    plt.ylabel("Selection precision - random clean rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figure_A_validity_boundary.png", dpi=200)
    plt.close()

    # Figure B: best available label-free warning by abs Pearson
    pearsons = {k: v for k, v in correlations.items() if k.endswith("_vs_selection_gap_pearson")}
    if pearsons:
        best_key = max(pearsons, key=lambda k: abs(_to_float(pearsons[k])) if not math.isnan(_to_float(pearsons[k])) else -1)
        field = best_key.replace("_vs_selection_gap_pearson", "")
        xs = [_to_float(r.get(field)) for r in summary_rows]
        ys = [_to_float(r.get("selection_precision_minus_random_clean_rate")) for r in summary_rows]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel(field)
        plt.ylabel("Selection precision - random clean rate")
        plt.title(f"Pearson r={pearsons[best_key]:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir / "figure_B_label_free_warning.png", dpi=200)
        plt.close()

    # Figure C: selection quality vs downstream
    plt.figure(figsize=(6, 5))
    for row in summary_rows:
        plt.scatter(row["selection_precision_minus_random_clean_rate"], row["sel_topk_minus_ce"])
        plt.text(row["selection_precision_minus_random_clean_rate"], row["sel_topk_minus_ce"], f"{row['dataset']} {row['noise_type']}{row['noise_rate']}", fontsize=7)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.axvline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Selection precision - random clean rate")
    plt.ylabel("sel_topk - CE")
    plt.tight_layout()
    plt.savefig(out_dir / "figure_C_selection_vs_downstream.png", dpi=200)
    plt.close()


def write_boundary_report(out_dir: Path, setting_rows: list[dict], correlations: dict[str, float], claims: dict):
    lines = [
        "# Local-Consistency Validity Boundary Report",
        "",
        "## Setting Summary",
        "",
        "| dataset | noise | sel_topk | CE | co-teaching | selection gap | noise recall | topk-CE | topk-coteach | C1/C2 note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row['dataset']} | {row['noise_type']}{row['noise_rate']} | {row.get('sel_topk_mean', math.nan):.4f} | "
            f"{row.get('ce_mean', math.nan):.4f} | {row.get('coteaching_mean', math.nan):.4f} | "
            f"{row.get('selection_precision_minus_random_clean_rate', math.nan):.4f} | "
            f"{row.get('noise_excluded_recall_mean', math.nan):.4f} | "
            f"{row.get('sel_topk_minus_ce', math.nan) * 100.0:.2f}pp | "
            f"{row.get('sel_topk_minus_coteaching', math.nan) * 100.0:.2f}pp | |"
        )
    lines.extend(["", "## Label-Free Correlations", ""])
    for key, value in correlations.items():
        lines.append(f"- `{key}`: {value:.4f}" if not math.isnan(_to_float(value)) else f"- `{key}`: nan")
    lines.extend(["", "## Pre-Registered Claims", ""])
    for key, value in claims.items():
        lines.append(f"- `{key}`: **{'PASS' if value['passes'] else 'FAIL'}** `{json.dumps(value, ensure_ascii=False)}`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(args.rows)
    setting_rows = boundary_summary_rows(rows)
    correlations = compute_boundary_correlations(setting_rows)
    claims = evaluate_boundary_claims(setting_rows, correlations)
    save_csv(args.out_dir / "boundary_setting_summary.csv", setting_rows)
    save_json(args.out_dir / "boundary_correlations.json", correlations)
    save_json(args.out_dir / "boundary_claims.json", claims)
    _plot_outputs(setting_rows, correlations, args.out_dir)
    write_boundary_report(args.out_dir, setting_rows, correlations, claims)
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
