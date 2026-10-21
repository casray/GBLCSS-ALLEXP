from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
GB_METHOD = "frozengrace_gb_mlp_softdist_beta05"
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"
LABEL_FREE_SOURCES = ("raw_x", "adj2_x")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).replace({np.nan: ""})


def validate(
    frame: pd.DataFrame,
    *,
    require_leave_one_out: bool = False,
) -> pd.DataFrame:
    required = {
        "dataset",
        "representation_source",
        "seed",
        "test",
        "status",
        "tie_mode",
        "num_balls",
        "split_hash",
        "noise_hash",
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = frame.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    if not (work["status"] == "ok").all():
        raise ValueError("non-ok source row present")
    if not (work["tie_mode"] == "distribution_only").all():
        raise ValueError("non-tie-safe source row present")
    if require_leave_one_out:
        if "leave_one_out" not in work.columns:
            raise ValueError("leave_one_out column is missing")
        if not (pd.to_numeric(work["leave_one_out"]) == 1).all():
            raise ValueError("non-LOO source row present")
    if not (pd.to_numeric(work["modified_val_count"]) == 0).all():
        raise ValueError("validation label leakage")
    if not (pd.to_numeric(work["modified_test_count"]) == 0).all():
        raise ValueError("test label leakage")
    if not (
        pd.to_numeric(work["modified_count"])
        == pd.to_numeric(work["modified_train_count"])
    ).all():
        raise ValueError("noise was not train-only")
    return work


def load_grace(
    path: Path,
    *,
    method: str = GB_METHOD,
    require_leave_one_out: bool = False,
) -> pd.DataFrame:
    frame = _read(path)
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["seed"] = pd.to_numeric(frame["seed"]).astype(int)
    frame["noise_rate"] = pd.to_numeric(frame["noise_rate"])
    selected = frame[
        frame["dataset"].isin(DATASETS)
        & (frame["method"] == method)
        & (frame["noise_type"] == "uniform")
        & np.isclose(frame["noise_rate"], 0.4)
        & frame["seed"].isin(range(5))
    ].copy()
    if len(selected) != 25:
        raise ValueError(f"expected 25 GRACE rows, found {len(selected)}")
    if require_leave_one_out:
        if "leave_one_out" not in selected.columns:
            raise ValueError("GRACE reference lacks leave_one_out")
        if not (pd.to_numeric(selected["leave_one_out"]) == 1).all():
            raise ValueError("GRACE reference contains non-LOO rows")
    return selected


def summarize(source: pd.DataFrame, grace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregates: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for dataset in DATASETS:
        grace_dataset = grace[grace["dataset"] == dataset].set_index("seed")
        grace_values = pd.to_numeric(grace_dataset["test"]).to_numpy(dtype=float)
        aggregates.append(
            {
                "dataset": dataset,
                "representation_source": "grace_h",
                "n": len(grace_values),
                "test_mean": grace_values.mean(),
                "test_std": grace_values.std(ddof=1),
                "mean_num_balls": pd.to_numeric(grace_dataset["num_balls"]).mean(),
                "collapsed_seeds": int(
                    (pd.to_numeric(grace_dataset["num_balls"]) <= 1).sum()
                ),
                "label_free": 1,
            }
        )
        for representation_source in LABEL_FREE_SOURCES:
            subset = source[
                (source["dataset"] == dataset)
                & (source["representation_source"] == representation_source)
            ].set_index("seed")
            if subset.empty or 0 not in subset.index:
                raise ValueError(f"missing seed0 smoke: {dataset}/{representation_source}")
            expected_seeds = (
                set(range(5))
                if int(float(subset.loc[0, "num_balls"])) > 1
                else {0}
            )
            if set(subset.index) != expected_seeds:
                raise ValueError(
                    f"source violates preregistered expansion rule: "
                    f"{dataset}/{representation_source}, seeds={sorted(subset.index)}"
                )
            values = pd.to_numeric(subset["test"]).to_numpy(dtype=float)
            num_balls = pd.to_numeric(subset["num_balls"]).to_numpy(dtype=float)
            aggregates.append(
                {
                    "dataset": dataset,
                    "representation_source": representation_source,
                    "n": len(values),
                    "test_mean": values.mean(),
                    "test_std": values.std(ddof=1) if len(values) > 1 else "",
                    "mean_num_balls": num_balls.mean(),
                    "collapsed_seeds": int((num_balls <= 1).sum()),
                    "label_free": 1,
                }
            )
            for seed in sorted(expected_seeds):
                if subset.loc[seed, "split_hash"] != grace_dataset.loc[seed, "split_hash"]:
                    raise ValueError(f"split hash mismatch: {dataset}/{representation_source}/{seed}")
                if subset.loc[seed, "noise_hash"] != grace_dataset.loc[seed, "noise_hash"]:
                    raise ValueError(f"noise hash mismatch: {dataset}/{representation_source}/{seed}")
                paired.append(
                    {
                        "dataset": dataset,
                        "representation_source": representation_source,
                        "seed": seed,
                        "source_test": float(subset.loc[seed, "test"]),
                        "grace_test": float(grace_dataset.loc[seed, "test"]),
                        "delta_source_minus_grace": float(subset.loc[seed, "test"])
                        - float(grace_dataset.loc[seed, "test"]),
                    }
                )
    for row in source[
        source["representation_source"] == "gcn_hidden_noisy_seed0"
    ].itertuples(index=False):
        aggregates.append(
            {
                "dataset": row.dataset,
                "representation_source": "gcn_hidden_noisy_seed0",
                "n": 1,
                "test_mean": float(row.test),
                "test_std": "",
                "mean_num_balls": float(row.num_balls),
                "collapsed_seeds": int(float(row.num_balls) <= 1),
                "label_free": 0,
            }
        )
    return pd.DataFrame(aggregates), pd.DataFrame(paired)


def write_report(path: Path, aggregate: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# MLP-43 Representation-Source Attribution",
        "",
        "All arms share the same split, noise, ball parameters, tie-safe target "
        "rule, and X-only MLP. Only the grouping representation changes. "
        "`adj2_x` denotes two applications of the stored normalized adjacency "
        "to X. A source is expanded to five seeds only when seed0 does not "
        "collapse to one ball.",
        "",
        "| Dataset | Source | Test | Balls | Collapsed seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        std_text = f" +/- {100 * float(row.test_std):.2f}" if row.test_std != "" else ""
        lines.append(
            f"| {row.dataset} | {row.representation_source} | "
            f"{100 * float(row.test_mean):.2f}{std_text} | "
            f"{float(row.mean_num_balls):.1f} | {int(row.collapsed_seeds)} |"
        )
    lines.extend(["", "## Paired label-free comparison against GRACE", ""])
    for (dataset, source), group in paired.groupby(
        ["dataset", "representation_source"], sort=True
    ):
        delta = group["delta_source_minus_grace"].to_numpy(dtype=float)
        lines.append(
            f"- {dataset}/{source}: n={len(delta)}, {100 * delta.mean():+.2f}pp, "
            f"{int((delta > 0).sum())}/{len(delta)} positive seeds."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- This table attributes performance to the tested grouping "
            "representations; it is not a proof that GRACE is uniquely optimal.",
            "- One-ball seed0 sources are reported as geometric failures and are "
            "not expanded solely to obtain repeated classifier scores.",
            "- The supervised GCN-hidden rows are seed-0 diagnostics only and use "
            "noisy-label supervision, so they are not ranked with the label-free "
            "five-seed sources.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--main-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-leave-one-out", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_loo = bool(args.require_leave_one_out)
    source = validate(_read(args.raw), require_leave_one_out=require_loo)
    grace = load_grace(
        args.main_grid,
        method=LOO_METHOD if require_loo else GB_METHOD,
        require_leave_one_out=require_loo,
    )
    aggregate, paired = summarize(source, grace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output_dir / "representation_source_aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "representation_source_paired.csv", index=False)
    write_report(args.output_dir / "representation_source_report.md", aggregate, paired)
    print("wrote representation-source summary")


if __name__ == "__main__":
    main()
