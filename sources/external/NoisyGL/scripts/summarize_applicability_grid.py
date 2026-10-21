from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset


DATASET_ORDER = [
    "cora",
    "citeseer",
    "pubmed",
    "amazonpho",
    "dblp",
    "coauthorcs",
    "amazoncom",
    "coauthorph",
    "wikics",
]

DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "amazonpho": "Amazon-Photo",
    "dblp": "DBLP",
    "coauthorcs": "Coauthor-CS",
    "amazoncom": "Amazon-Computers",
    "coauthorph": "Coauthor-Physics",
    "wikics": "WikiCS",
}

METHOD_LABELS = {
    "gcn": "GCN+CE",
    "gcn_clean": "GCN clean",
    "gcn_xh_ce": "GCN+CE [X||H]",
    "nrgnn": "NRGNN",
    "nrgnn_xh": "NRGNN [X||H]",
    "frozengrace_gb_gcn_softdist_xh": "SoftDist [X||H]",
    "frozengrace_gb_gcn_softdist_xh_shuffled": "SoftDist [X||H] shuffled",
    "gbselect": "GBSelect",
    "gbselect_gcn_only": "GBSelect GCN-only",
    "gbselect_shuffled": "GBSelect shuffled",
}

PAIRS = [
    ("frozengrace_gb_gcn_softdist_xh", "gcn_xh_ce", "softdist_vs_gcn_xh"),
    ("frozengrace_gb_gcn_softdist_xh", "frozengrace_gb_gcn_softdist_xh_shuffled", "softdist_vs_shuffled"),
    ("gbselect", "gbselect_gcn_only", "gbselect_vs_gcn_only"),
    ("gbselect", "gbselect_shuffled", "gbselect_vs_shuffled"),
    ("gbselect", "gcn_xh_ce", "gbselect_vs_gcn_xh"),
    ("gcn_clean", "gcn", "noise_harm"),
]


def _dataset_key(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "amazonphoto": "amazonpho",
        "amazon-photo": "amazonpho",
        "coauthor-cs": "coauthorcs",
        "coauthorcs": "coauthorcs",
        "coauthor-physics": "coauthorph",
        "coauthorphysics": "coauthorph",
        "wiki-cs": "wikics",
    }
    return aliases.get(key, key)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _method_from_row(row: dict[str, str], source_path: Path) -> str:
    method = row.get("method") or row.get("base_method") or ""
    name = source_path.name.lower()
    noise_rate = _safe_float(row.get("noise_rate"))
    if method == "gcn" and (noise_rate == 0.0 or ("clean" in name and "gcn" in name)):
        return "gcn_clean"
    if method:
        return method
    if "clean" in name and "gcn" in name:
        return "gcn_clean"
    return ""


def _test_from_row(row: dict[str, str]) -> float | None:
    return _safe_float(row.get("test") or row.get("test_acc") or row.get("accuracy"))


def collect_seed_results(paths: list[Path]) -> dict[tuple[str, str], dict[int, float]]:
    values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for path in paths:
        for row in _read_rows(path):
            status = str(row.get("status", "ok")).lower()
            if status not in {"", "ok"}:
                continue
            dataset = _dataset_key(row.get("dataset", ""))
            method = _method_from_row(row, path)
            seed = _safe_int(row.get("seed"))
            test = _test_from_row(row)
            if not dataset or not method or seed is None or test is None:
                continue
            values[(dataset, method)][seed] = test
    return values


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    try:
        from scipy.stats import spearmanr

        corr = spearmanr(x, y).correlation
        return float(corr) if corr is not None and math.isfinite(float(corr)) else math.nan
    except Exception:
        return _pearson(_rankdata(x), _rankdata(y))


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def aggregate_rows(values: dict[tuple[str, str], dict[int, float]]) -> list[dict[str, object]]:
    rows = []
    for (dataset, method), by_seed in sorted(values.items()):
        tests = [by_seed[seed] for seed in sorted(by_seed)]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n": len(tests),
                "mean": _mean(tests),
                "std": _std(tests),
                "seeds": " ".join(f"{value:.4f}" for value in tests),
            }
        )
    return rows


def paired_delta_rows(values: dict[tuple[str, str], dict[int, float]]) -> list[dict[str, object]]:
    rows = []
    datasets = sorted({dataset for dataset, _ in values}, key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 999)
    for dataset in datasets:
        for left, right, name in PAIRS:
            left_values = values.get((dataset, left), {})
            right_values = values.get((dataset, right), {})
            seeds = sorted(set(left_values) & set(right_values))
            if not seeds:
                continue
            deltas = [left_values[seed] - right_values[seed] for seed in seeds]
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": name,
                    "left_method": left,
                    "right_method": right,
                    "n": len(deltas),
                    "mean_delta": _mean(deltas),
                    "std_delta": _std(deltas),
                    "positive_seed_count": sum(delta > 0 for delta in deltas),
                    "deltas": " ".join(f"{delta:.4f}" for delta in deltas),
                }
            )
    return rows


def _edge_homophily(data) -> float:
    adj = data.adj.coalesce()
    idx = adj.indices().detach().cpu()
    values = adj.values().detach().cpu().float()
    labels = data.labels.detach().cpu().long()
    row = idx[0]
    col = idx[1]
    non_self = row != col
    row = row[non_self]
    col = col[non_self]
    values = values[non_self]
    if values.numel() == 0:
        return math.nan
    same = labels[row] == labels[col]
    return float(values[same].sum().item() / values.sum().item())


def load_metadata(datasets: list[str], device: str) -> dict[str, dict[str, object]]:
    meta = {}
    for dataset in datasets:
        try:
            data = _load_dataset(
                dataset,
                device=device,
                split_type="percent",
                train_percent=0.2,
                val_percent=0.1,
                test_percent=0.7,
            )
            meta[dataset] = {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "nodes": int(data.n_nodes),
                "classes": int(data.n_classes),
                "edges": int(data.n_edges),
                "homophily": _edge_homophily(data),
            }
        except Exception as exc:
            meta[dataset] = {
                "dataset": dataset,
                "label": DATASET_LABELS.get(dataset, dataset),
                "nodes": "",
                "classes": "",
                "edges": "",
                "homophily": "",
                "metadata_error": f"{type(exc).__name__}: {exc}",
            }
    return meta


def load_orthogonality(paths: list[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in paths:
        for row in _read_rows(path):
            dataset = _dataset_key(row.get("dataset", ""))
            if not dataset:
                continue
            metrics = out.setdefault(dataset, {})
            for key in [
                "ball_auc_in_gcn_clean50",
                "ball_auc_in_gcn_clean50_mean",
                "ball_auc",
                "ball_auc_mean",
                "true_all_purity_weighted",
                "true_all_purity_weighted_mean",
                "num_balls",
                "num_balls_mean",
            ]:
                value = _safe_float(row.get(key))
                if value is not None:
                    metrics[key.replace("_mean", "")] = value
    return out


def _delta_lookup(delta_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["dataset"]), str(row["comparison"])): row for row in delta_rows}


def applicability_rows(
    datasets: list[str],
    metadata: dict[str, dict[str, object]],
    delta_rows: list[dict[str, object]],
    orthogonality: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    deltas = _delta_lookup(delta_rows)
    rows = []
    for dataset in datasets:
        row = dict(metadata.get(dataset, {"dataset": dataset, "label": DATASET_LABELS.get(dataset, dataset)}))
        for _, _, name in PAIRS:
            delta = deltas.get((dataset, name))
            if delta:
                row[f"{name}_mean"] = delta["mean_delta"]
                row[f"{name}_positive_seeds"] = delta["positive_seed_count"]
                row[f"{name}_n"] = delta["n"]
        ortho = orthogonality.get(dataset, {})
        row["orthogonal_auc"] = ortho.get("ball_auc_in_gcn_clean50", "")
        row["ball_auc"] = ortho.get("ball_auc", "")
        row["true_ball_purity"] = ortho.get("true_all_purity_weighted", "")
        row["num_balls"] = ortho.get("num_balls", "")
        soft = row.get("softdist_vs_gcn_xh_mean")
        shuf = row.get("softdist_vs_shuffled_mean")
        gb = row.get("gbselect_vs_gcn_only_mean")
        if isinstance(soft, (int, float)) and isinstance(shuf, (int, float)):
            row["softdist_success"] = bool(float(soft) > 0.0 and float(shuf) > 0.0)
        if isinstance(gb, (int, float)):
            row["gbselect_success"] = bool(float(gb) > 0.0)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def correlation_rows(app_rows: list[dict[str, object]], target: str) -> list[dict[str, object]]:
    variables = [
        "nodes",
        "classes",
        "homophily",
        "num_balls",
        "noise_harm_mean",
        "orthogonal_auc",
        "ball_auc",
        "true_ball_purity",
    ]
    rows = []
    for variable in variables:
        x = []
        y = []
        included = []
        for row in app_rows:
            xv = _safe_float(row.get(variable))
            yv = _safe_float(row.get(target))
            if xv is None or yv is None:
                continue
            x.append(xv)
            y.append(yv)
            included.append(str(row.get("dataset", "")))
        rows.append(
            {
                "target": target,
                "variable": variable,
                "n": len(x),
                "spearman": _spearman(x, y),
                "pearson": _pearson(x, y) if len(x) >= 2 else math.nan,
                "datasets": " ".join(included),
            }
        )
    return rows


def _fmt(value, digits=4) -> str:
    number = _safe_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def write_report(path: Path, app_rows: list[dict[str, object]], corr_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Phase 37 Applicability Grid",
        "",
        "Protocol target: percent split 0.2/0.1/0.7, uniform noise 0.4, seeds 0-4 where available.",
        "Clean labels are used only for diagnostic variables such as homophily and noise harm.",
        "",
        "## Grid",
        "",
        "| Dataset | N | C | h | noise harm | orthogonal AUC | SoftDist-GCN[XH] | SoftDist-shuffled | GBSelect-GCN-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in app_rows:
        lines.append(
            "| {dataset} | {nodes} | {classes} | {h} | {harm} | {auc} | {soft} | {shuf} | {gb} |".format(
                dataset=row.get("label") or row.get("dataset"),
                nodes=row.get("nodes", "NA"),
                classes=row.get("classes", "NA"),
                h=_fmt(row.get("homophily")),
                harm=_fmt(row.get("noise_harm_mean")),
                auc=_fmt(row.get("orthogonal_auc")),
                soft=_fmt(row.get("softdist_vs_gcn_xh_mean")),
                shuf=_fmt(row.get("softdist_vs_shuffled_mean")),
                gb=_fmt(row.get("gbselect_vs_gcn_only_mean")),
            )
        )
    lines.extend(["", "## Correlations", "", "| Target | Variable | n | Spearman | Pearson |", "|---|---|---:|---:|---:|"])
    for row in corr_rows:
        lines.append(
            f"| {row['target']} | {row['variable']} | {row['n']} | {_fmt(row['spearman'])} | {_fmt(row['pearson'])} |"
        )
    deployable = [
        row
        for row in corr_rows
        if row["target"] == "softdist_vs_gcn_xh_mean" and row["variable"] in {"nodes", "classes", "num_balls"}
    ]
    strong_deployable = [
        row
        for row in deployable
        if _safe_float(row.get("spearman")) is not None and abs(float(row["spearman"])) >= 0.6 and int(row["n"]) >= 7
    ]
    lines.extend(["", "## Adaptive-Layer Gate", ""])
    lines.append(
        "`orthogonal_auc`, `ball_auc`, `true_ball_purity`, `noise_harm`, and homophily are diagnostic variables here; "
        "they use clean labels or full dataset labels for analysis and cannot directly drive a deploy-time adaptive layer."
    )
    if strong_deployable:
        names = ", ".join(f"{row['variable']} (rho={_fmt(row['spearman'])})" for row in strong_deployable)
        lines.append(
            f"Deployable metadata proxy/proxies currently show a strong rank correlation with SoftDist gain: {names}. "
            "This is only a candidate trigger and still needs validation on complete rows."
        )
    else:
        lines.append(
            "No strong deployable trigger is established yet. If this remains true after WikiCS/Physics rows are filled, "
            "an adaptive layer would risk becoming dataset-level cherry-picking."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_result_inputs() -> list[Path]:
    return [
        REPO_ROOT / "runs" / "phase24_nrgnn_xh_20260616" / "phase24_fairness_combined_raw.csv",
        REPO_ROOT / "runs" / "phase10_pubmed_20260612" / "phase10_pubmed_standard_summary.csv",
        REPO_ROOT / "runs" / "amazonphoto_20260612" / "amazonphoto_standard_summary.csv",
        REPO_ROOT / "runs" / "dblp_20260616" / "dblp_gcn_seed0_download_smoke.csv",
        REPO_ROOT / "runs" / "figures_20260617" / "phase36_aggregate.csv",
        REPO_ROOT / "runs" / "coauthorcs_20260616" / "softdist_xh_5seed.csv",
        REPO_ROOT / "runs" / "coauthorcs_20260616" / "gbselect_5seed.csv",
        REPO_ROOT / "runs" / "coauthorcs_baseline_20260617" / "coauthorcs_baselines_5seed.csv",
        REPO_ROOT / "runs" / "amazoncom_20260616" / "softdist_xh_5seed.csv",
        REPO_ROOT / "runs" / "amazoncom_20260616" / "gbselect_5seed.csv",
        REPO_ROOT / "runs" / "amazoncom_baseline_20260617" / "amazoncom_baselines_5seed.csv",
        REPO_ROOT / "runs" / "gbselect_20260616" / "dblp_5seed.csv",
        REPO_ROOT / "runs" / "gbselect_20260616" / "dblp_keep07_5seed.csv",
        REPO_ROOT / "runs" / "gbselect_20260616" / "dblp_keep07_shuffled_5seed.csv",
        REPO_ROOT / "runs" / "gbselect_20260616" / "pubmed_keep07_5seed.csv",
        REPO_ROOT / "runs" / "gbselect_20260616" / "amazonpho_keep07_5seed.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "new_dataset_standard.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "new_dataset_xh_softdist.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "new_dataset_gbselect.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_standard_seed0.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_nrgnn_seed0.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_xh_softdist_seed0.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_gbselect_seed0.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_xh_softdist_5seed.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_gbselect_5seed.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_gcn_5seed.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "wikics_gcn_clean_5seed.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "coauthorph_gcn_noisy_seed0.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "clean_gcn_seed0_all.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "clean_gcn.csv",
    ]


def default_orthogonality_inputs() -> list[Path]:
    return [
        REPO_ROOT / "runs" / "biggraph_orthogonality_diag_20260616" / "orthogonality_summary.csv",
        REPO_ROOT / "runs" / "applicability_grid_20260617" / "orthogonality_summary.csv",
        REPO_ROOT
        / "runs"
        / "applicability_grid_20260617"
        / "orthogonality_wikics_seed0"
        / "orthogonality_summary.csv",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 37 applicability grid variables and deltas.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "applicability_grid_20260617"))
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--inputs", nargs="*", default=None)
    parser.add_argument("--orthogonality_inputs", nargs="*", default=None)
    parser.add_argument("--metadata_device", default="cpu")
    args = parser.parse_args()

    datasets = [_dataset_key(item) for item in args.datasets.split(",") if item.strip()]
    inputs = [Path(path) for path in args.inputs] if args.inputs else default_result_inputs()
    orthogonality_inputs = (
        [Path(path) for path in args.orthogonality_inputs] if args.orthogonality_inputs else default_orthogonality_inputs()
    )
    output_dir = Path(args.output_dir)

    values = collect_seed_results(inputs)
    aggregate = aggregate_rows(values)
    deltas = paired_delta_rows(values)
    metadata = load_metadata(datasets, args.metadata_device)
    orthogonality = load_orthogonality(orthogonality_inputs)
    app_rows = applicability_rows(datasets, metadata, deltas, orthogonality)
    corr = correlation_rows(app_rows, "softdist_vs_gcn_xh_mean")
    corr.extend(correlation_rows(app_rows, "gbselect_vs_gcn_only_mean"))

    write_csv(output_dir / "applicability_method_aggregate.csv", aggregate)
    write_csv(output_dir / "applicability_paired_deltas.csv", deltas)
    write_csv(output_dir / "applicability_grid_summary.csv", app_rows)
    write_csv(output_dir / "applicability_correlations.csv", corr)
    write_report(output_dir / "correlation_analysis.md", app_rows, corr)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
