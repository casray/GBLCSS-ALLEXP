from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull")
SOURCES = ("raw_x", "adj2_x")
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"


def summarize(raw_path: Path, main_grid_path: Path, output_dir: Path) -> None:
    raw_all = pd.read_csv(raw_path)
    raw_all.to_csv(output_dir / "representation_source_all_executed.csv", index=False)
    source = raw_all[
        raw_all.dataset.isin(DATASETS)
        & raw_all.representation_source.isin(SOURCES)
        & (raw_all.seed == 0)
    ].copy()
    source = source.drop_duplicates(
        ["dataset", "representation_source", "seed"], keep="last"
    )
    expected = {(dataset, source_name) for dataset in DATASETS for source_name in SOURCES}
    actual = set(zip(source.dataset, source.representation_source))
    if len(source) != 6 or actual != expected:
        raise ValueError(f"incomplete seed0 representation grid: {actual}")
    if not (source.status == "ok").all():
        raise ValueError("non-ok representation row")
    if not (source.tie_mode == "distribution_only").all():
        raise ValueError("non-tie-safe representation row")
    if not (pd.to_numeric(source.leave_one_out) == 1).all():
        raise ValueError("non-LOO representation row")
    if not (pd.to_numeric(source.num_balls) == 1).all():
        raise ValueError("a registered seed0 source did not collapse")

    main = pd.read_csv(main_grid_path)
    grace = main[
        main.dataset.isin(DATASETS)
        & (main.method == LOO_METHOD)
        & (main.noise_type == "uniform")
        & np.isclose(main.noise_rate, 0.4)
        & main.seed.isin(range(5))
    ].copy()
    if len(grace) != 15 or not (pd.to_numeric(grace.leave_one_out) == 1).all():
        raise ValueError("incomplete formal GRACE LOO reference")
    grace_aggregate = (
        grace.groupby("dataset")["test"].agg(["mean", "std", "count"]).reset_index()
    )
    source.to_csv(output_dir / "representation_source_formal_seed0.csv", index=False)
    grace_aggregate.to_csv(output_dir / "representation_grace_aggregate.csv", index=False)

    lines = [
        "# Formal LOO representation-source diagnostic",
        "",
        "All target rows use distribution_only and leave_one_out=true.",
        "Raw X and A2X are stopped at seed0 because both form one ball on all three datasets.",
        "",
        "| dataset | GRACE H (5 seeds) | raw X seed0 | A2X seed0 | raw/A2X balls |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        grace_row = grace_aggregate[grace_aggregate.dataset == dataset].iloc[0]
        raw_row = source[
            (source.dataset == dataset) & (source.representation_source == "raw_x")
        ].iloc[0]
        adj_row = source[
            (source.dataset == dataset) & (source.representation_source == "adj2_x")
        ].iloc[0]
        if raw_row.split_hash != grace[grace.dataset == dataset].set_index("seed").loc[0, "split_hash"]:
            raise ValueError(f"split hash mismatch for {dataset}/raw_x")
        if adj_row.noise_hash != grace[grace.dataset == dataset].set_index("seed").loc[0, "noise_hash"]:
            raise ValueError(f"noise hash mismatch for {dataset}/adj2_x")
        lines.append(
            f"| {dataset} | {100*grace_row['mean']:.2f}+/-{100*grace_row['std']:.2f} | "
            f"{100*float(raw_row.test):.2f} | {100*float(adj_row.test):.2f} | 1/1 |"
        )
    (output_dir / "representation_source_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--main-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize(args.raw, args.main_grid, args.output_dir)


if __name__ == "__main__":
    main()
