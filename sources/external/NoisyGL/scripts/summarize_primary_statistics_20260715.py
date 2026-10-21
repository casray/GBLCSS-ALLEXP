from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_mlp36_review_defense import holm_adjust, paired_statistics


METHOD_MAP = {
    "frozengrace_gb_mlp_softdist_beta05": "lcss",
    "frozengrace_gb_mlp_softdist_beta05_shuffled": "shuffled",
    "mlp_ce": "ce",
    "mlp_jocor": "jocor",
}
COMPARISONS = {
    "lcss_vs_ce": ("lcss", "ce"),
    "lcss_vs_shuffled": ("lcss", "shuffled"),
    "lcss_vs_jocor": ("lcss", "jocor"),
}
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "test" not in result and "test_acc" in result:
        result["test"] = result["test_acc"]
    required = {"dataset", "method", "noise_type", "noise_rate", "seed", "test"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"missing canonical columns: {sorted(missing)}")
    result = result[result["method"].isin(METHOD_MAP)].copy()
    result["method"] = result["method"].map(METHOD_MAP)
    result["seed"] = result["seed"].astype(int)
    result["noise_rate"] = result["noise_rate"].astype(float)
    result["test"] = result["test"].astype(float)
    return result[["dataset", "noise_type", "noise_rate", "seed", "method", "test"]]


def load_primary_records(repo_root: Path) -> pd.DataFrame:
    main = pd.read_csv(repo_root / "runs/mlp_maintable_20260701/mlp_maintable_combined_raw.csv")
    main = main[
        main["dataset"].isin(("cora", "citeseer"))
        & (main["table"] == "main")
        & np.isclose(main["train_percent"].astype(float), 0.2)
    ]

    coauthor = pd.read_csv(repo_root / "runs/mlp10_baseline_refresh_20260702/mlp10_combined_raw.csv")
    coauthor = coauthor[(coauthor["dataset"] == "coauthorcs") & (coauthor["status"] == "ok")]

    amazon = pd.read_csv(repo_root / "runs/mlp12_amazonphoto_20260702/mlp12_combined_raw.csv")
    amazon = amazon[(amazon["dataset"] == "amazonpho") & (amazon["status"] == "ok")]

    corafull = pd.read_csv(repo_root / "runs/mlp21_dataset_replacement_20260710/mlp21_raw.csv")
    corafull = corafull[(corafull["dataset"] == "corafull") & (corafull["status"] == "ok")]

    records = pd.concat(
        [_canonicalize(frame) for frame in (main, coauthor, amazon, corafull)],
        ignore_index=True,
    )
    records = records[
        records["dataset"].isin(DATASETS)
        & records.apply(
            lambda row: (row["noise_type"], float(row["noise_rate"])) in SETTINGS,
            axis=1,
        )
    ].copy()
    if records.duplicated(["dataset", "noise_type", "noise_rate", "seed", "method"]).any():
        raise ValueError("duplicate primary-statistics row")
    expected = len(DATASETS) * len(SETTINGS) * len(METHOD_MAP) * 5
    if len(records) != expected:
        raise ValueError(f"expected {expected} primary rows, found {len(records)}")
    return records.sort_values(
        ["dataset", "noise_type", "noise_rate", "method", "seed"]
    ).reset_index(drop=True)


def compute_primary_statistics(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    setting_columns = ["dataset", "noise_type", "noise_rate"]
    for setting, group in records.groupby(setting_columns, sort=True):
        for comparison, (left_method, right_method) in COMPARISONS.items():
            left = group[group["method"] == left_method].set_index("seed")["test"]
            right = group[group["method"] == right_method].set_index("seed")["test"]
            common = left.index.intersection(right.index)
            if len(common) != 5 or set(common) != set(range(5)):
                raise ValueError(
                    f"exactly 5 paired seeds are required for {setting} {comparison}"
                )
            stats = paired_statistics(
                left.loc[sorted(common)].to_numpy(),
                right.loc[sorted(common)].to_numpy(),
            )
            rows.append(
                {
                    "dataset": setting[0],
                    "noise_type": setting[1],
                    "noise_rate": float(setting[2]),
                    "comparison": comparison,
                    **stats,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no primary comparisons were produced")
    result["holm_p"] = np.nan
    for comparison, indices in result.groupby("comparison").groups.items():
        adjusted = holm_adjust(result.loc[indices, "paired_t_p"].to_numpy())
        result.loc[indices, "holm_p"] = adjusted
    return result.sort_values(
        ["comparison", "dataset", "noise_type", "noise_rate"]
    ).reset_index(drop=True)


def write_report(stats: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Primary comparison family statistics",
        "",
        "- Five paired seeds per setting.",
        "- Holm correction is applied separately within each 25-setting comparison family.",
        "- Remaining pairwise tests are exploratory and are not included here.",
        "",
        "| Comparison | Mean delta across settings (pp) | Holm p<0.05 | >=4/5 positive |",
        "|---|---:|---:|---:|",
    ]
    for comparison, group in stats.groupby("comparison", sort=False):
        lines.append(
            f"| {comparison} | {100 * group['mean_delta'].mean():+.2f} | "
            f"{int((group['holm_p'] < 0.05).sum())}/{len(group)} | "
            f"{int((group['positive_seeds'] >= 4).sum())}/{len(group)} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/mlp36_review_defense_20260715"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_primary_records(repo_root)
    stats = compute_primary_statistics(records)
    records.to_csv(output_dir / "primary_statistics_raw.csv", index=False)
    stats.to_csv(output_dir / "primary_statistics_holm.csv", index=False)
    write_report(stats, output_dir / "primary_statistics_report.md")


if __name__ == "__main__":
    main()
