from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


DATASETS = ("cora", "citeseer", "corafull")
H_SEEDS = (0, 1, 2)
DOWNSTREAM_SEEDS = (0, 1, 2)
METHOD = "frozengrace_gb_mlp_softdist_beta05"
LOO_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"
CE = "mlp_ce"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).replace({np.nan: ""})


def validate_raw(
    frame: pd.DataFrame,
    *,
    method: str = METHOD,
    require_leave_one_out: bool = False,
) -> pd.DataFrame:
    required = {
        "dataset",
        "method",
        "seed",
        "h_seed",
        "test",
        "status",
        "noise_scope",
        "noise_type",
        "noise_rate",
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
        "split_hash",
        "noise_hash",
        "tie_mode",
        "h_sha256",
        "h_num_nodes",
        "h_dim",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = frame.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    work["h_seed"] = pd.to_numeric(work["h_seed"]).astype(int)
    formal = work[
        work["dataset"].isin(DATASETS)
        & (work["method"] == method)
        & work["seed"].isin(DOWNSTREAM_SEEDS)
        & work["h_seed"].isin(H_SEEDS)
    ].copy()
    expected = {
        (dataset, h_seed, seed)
        for dataset in DATASETS
        for h_seed in H_SEEDS
        for seed in DOWNSTREAM_SEEDS
    }
    actual = set(zip(formal["dataset"], formal["h_seed"], formal["seed"]))
    if len(formal) != 27 or actual != expected:
        raise ValueError(f"incomplete 3x3 grid: rows={len(formal)}")
    if not (formal["status"] == "ok").all():
        raise ValueError("non-ok row present")
    if not (formal["noise_scope"] == "train").all():
        raise ValueError("non-train noise row present")
    if not (formal["noise_type"] == "uniform").all():
        raise ValueError("non-uniform noise row present")
    if not np.allclose(pd.to_numeric(formal["noise_rate"]), 0.4):
        raise ValueError("wrong noise rate")
    if not (formal["tie_mode"] == "distribution_only").all():
        raise ValueError("non-tie-safe row present")
    if require_leave_one_out:
        if "leave_one_out" not in formal.columns:
            raise ValueError("leave_one_out column is missing")
        if not (pd.to_numeric(formal["leave_one_out"]) == 1).all():
            raise ValueError("non-LOO row present")
    if not (pd.to_numeric(formal["modified_val_count"]) == 0).all():
        raise ValueError("validation label leakage")
    if not (pd.to_numeric(formal["modified_test_count"]) == 0).all():
        raise ValueError("test label leakage")
    if not (
        pd.to_numeric(formal["modified_count"])
        == pd.to_numeric(formal["modified_train_count"])
    ).all():
        raise ValueError("noise was not restricted to train")
    return formal


def load_ce_reference(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["seed"] = pd.to_numeric(frame["seed"]).astype(int)
    frame["noise_rate"] = pd.to_numeric(frame["noise_rate"])
    selected = frame[
        frame["dataset"].isin(DATASETS)
        & (frame["method"] == CE)
        & (frame["noise_type"] == "uniform")
        & np.isclose(frame["noise_rate"], 0.4)
        & frame["seed"].isin(DOWNSTREAM_SEEDS)
    ].copy()
    if len(selected) != 9:
        raise ValueError(f"expected 9 CE reference rows, found {len(selected)}")
    return selected


def summarize_variance(
    frame: pd.DataFrame,
    datasets: tuple[str, ...] = DATASETS,
) -> pd.DataFrame:
    work = frame.copy()
    work["test"] = pd.to_numeric(work["test"])
    work["h_seed"] = pd.to_numeric(work["h_seed"]).astype(int)
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        subset = work[work["dataset"] == dataset]
        h_means: list[float] = []
        downstream_stds: list[float] = []
        for h_seed in H_SEEDS:
            values = subset[subset["h_seed"] == h_seed]["test"].to_numpy(dtype=float)
            if len(values) != 3:
                raise ValueError(f"incomplete h-seed cell: {dataset}/{h_seed}")
            h_means.append(float(values.mean()))
            downstream_stds.append(float(values.std(ddof=1)))
        h_std = float(np.std(h_means, ddof=1))
        downstream_rms = float(np.sqrt(np.mean(np.square(downstream_stds))))
        rows.append(
            {
                "dataset": dataset,
                "h_seed_mean_0": h_means[0],
                "h_seed_mean_1": h_means[1],
                "h_seed_mean_2": h_means[2],
                "h_seed_std": h_std,
                "downstream_rms_std": downstream_rms,
                "h_to_downstream_std_ratio": (
                    h_std / downstream_rms if downstream_rms > 0 else math.inf
                ),
                "dispersion_pass": bool(h_std <= 2.0 * downstream_rms),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_h_seed(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["test"] = pd.to_numeric(work["test"])
    return (
        work.groupby(["dataset", "h_seed"], as_index=False)
        .agg(n=("test", "size"), test_mean=("test", "mean"), test_std=("test", "std"))
        .sort_values(["dataset", "h_seed"])
    )


def paired_delta_ce(frame: pd.DataFrame, ce: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        ce_rows = ce[ce["dataset"] == dataset].set_index("seed")
        for h_seed in H_SEEDS:
            lcss = frame[
                (frame["dataset"] == dataset) & (frame["h_seed"] == h_seed)
            ].set_index("seed")
            for seed in DOWNSTREAM_SEEDS:
                if lcss.loc[seed, "split_hash"] != ce_rows.loc[seed, "split_hash"]:
                    raise ValueError(f"CE split hash mismatch: {dataset}/{h_seed}/{seed}")
                if lcss.loc[seed, "noise_hash"] != ce_rows.loc[seed, "noise_hash"]:
                    raise ValueError(f"CE noise hash mismatch: {dataset}/{h_seed}/{seed}")
                delta = float(lcss.loc[seed, "test"]) - float(ce_rows.loc[seed, "test"])
                rows.append(
                    {
                        "dataset": dataset,
                        "h_seed": h_seed,
                        "seed": seed,
                        "lcss_test": float(lcss.loc[seed, "test"]),
                        "ce_test": float(ce_rows.loc[seed, "test"]),
                        "delta_lcss_minus_ce": delta,
                    }
                )
    return pd.DataFrame(rows)


def assignment_agreement(geometry: pd.DataFrame) -> pd.DataFrame:
    work = geometry.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    work["h_seed"] = pd.to_numeric(work["h_seed"]).astype(int)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in DOWNSTREAM_SEEDS:
            subset = work[(work["dataset"] == dataset) & (work["seed"] == seed)]
            if len(subset) != 3:
                raise ValueError(f"incomplete geometry: {dataset}/{seed}")
            assignments = {
                int(row.h_seed): np.load(Path(str(row.assignment_path)))
                for row in subset.itertuples(index=False)
            }
            for first, second in ((0, 1), (0, 2), (1, 2)):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "h_seed_pair": f"{first}-{second}",
                        "adjusted_rand_index": adjusted_rand_score(
                            assignments[first], assignments[second]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    variance: pd.DataFrame,
    aggregate: pd.DataFrame,
    deltas: pd.DataFrame,
    agreements: pd.DataFrame,
    leave_one_out: bool = False,
) -> None:
    target_rule = "`tie_mode=distribution_only`"
    if leave_one_out:
        target_rule += " and `leave_one_out=true`"
    lines = [
        "# MLP-43 Tie-Safe GRACE-Seed Stability",
        "",
        "Protocol: three archived GRACE seeds x three downstream seeds on Cora, "
        f"CiteSeer, and CoraFull. The downstream target rule is {target_rule}.",
        "",
        "| Dataset | H0 | H1 | H2 | H-seed SD | downstream RMS SD | ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {
        (str(row.dataset), int(row.h_seed)): row
        for row in aggregate.itertuples(index=False)
    }
    for row in variance.itertuples(index=False):
        h_rows = [by_key[(row.dataset, h_seed)] for h_seed in H_SEEDS]
        lines.append(
            f"| {row.dataset} | "
            + " | ".join(
                f"{100 * item.test_mean:.2f} +/- {100 * item.test_std:.2f}"
                for item in h_rows
            )
            + f" | {100 * row.h_seed_std:.2f}pp | "
            f"{100 * row.downstream_rms_std:.2f}pp | "
            f"{row.h_to_downstream_std_ratio:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered interpretation",
            "",
        ]
    )
    for dataset in DATASETS:
        h_deltas = (
            deltas[deltas["dataset"] == dataset]
            .groupby("h_seed")["delta_lcss_minus_ce"]
            .mean()
        )
        variance_row = variance[variance["dataset"] == dataset].iloc[0]
        lines.append(
            f"- {dataset}: LCSS-CE is positive for "
            f"{int((h_deltas > 0).sum())}/3 H seeds; "
            f"cross-H dispersion gate="
            f"{'PASS' if bool(variance_row['dispersion_pass']) else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            f"Cross-H assignment ARI mean={agreements['adjusted_rand_index'].mean():.3f}, "
            f"range={agreements['adjusted_rand_index'].min():.3f}-"
            f"{agreements['adjusted_rand_index'].max():.3f}.",
            "",
            "Passing supports only stability across the three archived pretraining "
            "seeds. It does not remove the transductive fixed-graph scope.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--main-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-leave-one-out", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = LOO_METHOD if args.require_leave_one_out else METHOD
    raw = validate_raw(
        read_csv(args.raw),
        method=method,
        require_leave_one_out=bool(args.require_leave_one_out),
    )
    ce = load_ce_reference(args.main_grid)
    geometry = read_csv(args.geometry)
    variance = summarize_variance(raw)
    aggregate = aggregate_by_h_seed(raw)
    deltas = paired_delta_ce(raw, ce)
    agreements = assignment_agreement(geometry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variance.to_csv(args.output_dir / "grace_seed_tiesafe_variance.csv", index=False)
    aggregate.to_csv(args.output_dir / "grace_seed_tiesafe_aggregate.csv", index=False)
    deltas.to_csv(args.output_dir / "grace_seed_tiesafe_paired_ce.csv", index=False)
    agreements.to_csv(
        args.output_dir / "grace_seed_tiesafe_assignment_agreement.csv", index=False
    )
    write_report(
        args.output_dir / "grace_seed_tiesafe_report.md",
        variance,
        aggregate,
        deltas,
        agreements,
        leave_one_out=bool(args.require_leave_one_out),
    )
    print("wrote tie-safe GRACE-seed summary")


if __name__ == "__main__":
    main()
