from __future__ import annotations

from pathlib import Path

import pandas as pd

from gb_stage_b.utils import read_json


DELTA_KEYS = ["linear_eval_acc", "mean_radius", "mean_average_angular_distance", "effective_rank"]
GROUP_KEYS = ["experiment_group", "dataset", "seed", "ssl_objective", "variant"]
SUMMARY_KEYS = [
    "mean_radius",
    "mean_average_angular_distance",
    "label_purity_diagnostic",
    "linear_eval_acc",
    "effective_rank",
    "feature_std_mean",
]


def _metric_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*/seed_*/*/metrics.json"))


def summarize_results(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    rows = []
    for path in _metric_files(results_dir):
        row = read_json(path)
        row["run_dir"] = str(path.parent)
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = results_dir / "stage_b_summary.csv"
    paired_path = results_dir / "stage_b_paired_deltas.csv"
    if df.empty:
        df.to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(paired_path, index=False)
        return summary_path

    df.to_csv(summary_path, index=False)
    grouped_path = results_dir / "stage_b_grouped_summary.csv"
    available_group_keys = [key for key in GROUP_KEYS if key in df.columns]
    available_summary_keys = [key for key in SUMMARY_KEYS if key in df.columns]
    if available_group_keys and available_summary_keys:
        grouped = df.groupby(available_group_keys, dropna=False)[available_summary_keys].agg(["mean", "std"])
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
        grouped.reset_index().to_csv(grouped_path, index=False)
    else:
        pd.DataFrame().to_csv(grouped_path, index=False)
    deltas = []
    for (dataset, seed), group in df.groupby(["dataset", "seed"], dropna=False):
        base = group[group["method"] == "B0"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            if row["method"] == "B0":
                continue
            item = {"dataset": dataset, "seed": seed, "method": row["method"]}
            for key in DELTA_KEYS:
                if key in row.index and key in base_row.index and pd.notna(row[key]) and pd.notna(base_row[key]):
                    item[f"delta_{key}"] = row[key] - base_row[key]
            deltas.append(item)
    pd.DataFrame(deltas).to_csv(paired_path, index=False)
    return summary_path
