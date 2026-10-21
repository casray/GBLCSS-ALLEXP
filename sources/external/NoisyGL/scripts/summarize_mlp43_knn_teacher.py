from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SEEDS = tuple(range(5))
METHODS = (
    "mlp_distill_gcn",
    "mlp_distill_gcn_xh",
    "mlp_distill_softdist_xh",
    "mlp_distill_knn_xh",
    "mlp_distill_softdist_xh_shuffled",
)
KNN_METHOD = "mlp_distill_knn_xh"
KEY = ["dataset", "noise_rate", "seed", "method"]


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["dataset"] = work["dataset"].astype(str).str.lower()
    work["method"] = work["method"].astype(str)
    work["noise_rate"] = pd.to_numeric(work["noise_rate"])
    work["seed"] = pd.to_numeric(work["seed"]).astype(int)
    return work


def _select_protocol(frame: pd.DataFrame) -> pd.DataFrame:
    work = _normalize(frame)
    return work[
        work["dataset"].isin(DATASETS)
        & work["method"].isin(METHODS)
        & work["seed"].isin(SEEDS)
        & work["noise_rate"].apply(
            lambda value: any(np.isclose(value, rate) for rate in NOISE_RATES)
        )
        & (work["noise_type"].astype(str) == "uniform")
        & (work["noise_scope"].astype(str) == "train")
    ].copy()


def _validate_common(frame: pd.DataFrame) -> None:
    required = {
        *KEY,
        "noise_type",
        "noise_scope",
        "split_hash",
        "noise_hash",
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
        "status",
        "test",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if not (frame["status"].astype(str) == "ok").all():
        raise ValueError("non-ok teacher row present")
    if not (pd.to_numeric(frame["modified_val_count"]) == 0).all():
        raise ValueError("validation label leakage")
    if not (pd.to_numeric(frame["modified_test_count"]) == 0).all():
        raise ValueError("test label leakage")
    if not (
        pd.to_numeric(frame["modified_count"])
        == pd.to_numeric(frame["modified_train_count"])
    ).all():
        raise ValueError("noise was not restricted to train labels")
    if pd.to_numeric(frame["test"], errors="coerce").isna().any():
        raise ValueError("missing test accuracy")


def merge_teacher_arms(
    old_frames: list[pd.DataFrame], corrected_knn: pd.DataFrame
) -> pd.DataFrame:
    old = _select_protocol(pd.concat(old_frames, ignore_index=True, sort=False))
    old = old[old["method"] != KNN_METHOD].copy()
    corrected = _select_protocol(corrected_knn)
    corrected = corrected[corrected["method"] == KNN_METHOD].copy()
    _validate_common(old)
    _validate_common(corrected)

    for column in ("knn_exclude_self", "target_beta", "soft_target_builder"):
        if column not in corrected.columns:
            raise ValueError(f"corrected kNN is missing {column}")
    if not pd.to_numeric(corrected["knn_exclude_self"]).eq(1).all():
        raise ValueError("corrected kNN must use exclude_self=true")
    if not np.isclose(pd.to_numeric(corrected["target_beta"]), 0.0).all():
        raise ValueError("corrected kNN teacher must use target_beta=0")
    if not corrected["soft_target_builder"].astype(str).eq("knn").all():
        raise ValueError("corrected kNN teacher must use the knn builder")

    expected_old = len(DATASETS) * len(NOISE_RATES) * len(SEEDS) * (len(METHODS) - 1)
    expected_knn = len(DATASETS) * len(NOISE_RATES) * len(SEEDS)
    for name, frame, expected in (
        ("reusable teacher arms", old, expected_old),
        ("corrected kNN arm", corrected, expected_knn),
    ):
        duplicated = frame.duplicated(KEY, keep=False)
        if duplicated.any():
            duplicate_keys = frame.loc[duplicated, KEY].drop_duplicates().to_dict("records")
            raise ValueError(f"duplicate {name}: {duplicate_keys[:3]}")
        if len(frame) != expected:
            raise ValueError(f"expected {expected} {name} rows, found {len(frame)}")

    merged = pd.concat([old, corrected], ignore_index=True, sort=False)
    if len(merged) != 150 or merged.duplicated(KEY).any():
        raise ValueError("teacher grid is not a complete 150-row five-arm grid")
    counts = merged.groupby(["dataset", "noise_rate", "seed"]).size()
    if not counts.eq(len(METHODS)).all():
        raise ValueError("teacher cell does not contain all five arms")
    for cell, group in merged.groupby(["dataset", "noise_rate", "seed"]):
        if group["split_hash"].astype(str).nunique() != 1:
            raise ValueError(f"split hash mismatch: {cell}")
        if group["noise_hash"].astype(str).nunique() != 1:
            raise ValueError(f"noise hash mismatch: {cell}")
    return merged.sort_values(KEY).reset_index(drop=True)


def aggregate_results(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in merged.groupby(["dataset", "noise_rate", "method"], sort=True):
        values = pd.to_numeric(group["test"]).to_numpy(dtype=float)
        rows.append(
            {
                "dataset": keys[0],
                "noise_rate": keys[1],
                "method": keys[2],
                "n": len(values),
                "test_mean": values.mean(),
                "test_std": values.std(ddof=1),
                "seeds": ",".join(map(str, sorted(group["seed"].astype(int)))),
            }
        )
    return pd.DataFrame(rows)


def paired_results(merged: pd.DataFrame) -> pd.DataFrame:
    pivot = merged.pivot(
        index=["dataset", "noise_rate", "seed"], columns="method", values="test"
    )
    comparisons = {
        "ball_minus_knn": (
            "mlp_distill_softdist_xh",
            "mlp_distill_knn_xh",
        ),
        "ball_minus_gcn_xh": (
            "mlp_distill_softdist_xh",
            "mlp_distill_gcn_xh",
        ),
        "ball_minus_shuffled": (
            "mlp_distill_softdist_xh",
            "mlp_distill_softdist_xh_shuffled",
        ),
        "knn_minus_gcn_xh": (
            "mlp_distill_knn_xh",
            "mlp_distill_gcn_xh",
        ),
    }
    rows = []
    for (dataset, noise_rate), group in pivot.groupby(level=[0, 1]):
        for name, (left, right) in comparisons.items():
            delta = (group[left] - group[right]).to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "noise_rate": noise_rate,
                    "comparison": name,
                    "n": len(delta),
                    "mean_delta": delta.mean(),
                    "std_delta": delta.std(ddof=1),
                    "positive_seeds": int((delta > 0).sum()),
                    "negative_seeds": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path, aggregate: pd.DataFrame, paired: pd.DataFrame, merged: pd.DataFrame
) -> None:
    labels = {
        "mlp_distill_gcn": "GCN(X)+CE (GLNN-style)",
        "mlp_distill_gcn_xh": "GCN[X||H]+CE",
        "mlp_distill_softdist_xh": "ball-consensus teacher",
        "mlp_distill_knn_xh": "kNN-consensus teacher (exclude self)",
        "mlp_distill_softdist_xh_shuffled": "shuffled-consensus teacher",
    }
    lines = [
        "# MLP-43 Corrected kNN-Teacher Audit",
        "",
        "Protocol: Cora/CiteSeer/CoraFull, uniform 0.4/0.6, train-only "
        "label noise, percent split 0.2/0.1/0.7, seeds 0--4. The corrected "
        "kNN teacher excludes the query node. The other four teacher arms "
        "are reused only after split/noise-hash equality is verified.",
        "",
        "| Dataset | Noise | Teacher | Student test (%) |",
        "|---|---:|---|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.noise_rate:.1f} | {labels[row.method]} | "
            f"{100 * row.test_mean:.2f} +/- {100 * row.test_std:.2f} |"
        )
    lines.extend(["", "## Paired deltas", ""])
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.dataset}/u{row.noise_rate:.1f}/{row.comparison}: "
            f"{100 * row.mean_delta:+.2f}pp, "
            f"{row.positive_seeds}/{row.n} positive seeds."
        )
    knn = merged[merged["method"] == KNN_METHOD]
    lines.extend(["", "## Matched-k and protocol audit", ""])
    for row in knn.sort_values(["noise_rate", "dataset", "seed"]).itertuples(index=False):
        lines.append(
            f"- {row.dataset}/u{row.noise_rate:.1f}/seed{row.seed}: "
            f"k={int(float(row.matched_k))}, exclude_self=1, "
            f"split={row.split_hash[:12]}, noise={row.noise_hash[:12]}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The corrected kNN arm is a pointwise local-consensus alternative, "
            "not evidence that the granular-ball partition is uniquely optimal.",
            "- All five arms use identical per-seed splits and noisy labels. "
            "Only the teacher construction differs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-source", type=Path, action="append", required=True)
    parser.add_argument("--corrected-knn", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_frames = [pd.read_csv(path) for path in args.old_source]
    corrected = pd.concat(
        [pd.read_csv(path) for path in args.corrected_knn],
        ignore_index=True,
        sort=False,
    )
    merged = merge_teacher_arms(old_frames, corrected)
    aggregate = aggregate_results(merged)
    paired = paired_results(merged)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "knn_teacher_corrected_combined_raw.csv", index=False)
    aggregate.to_csv(
        args.output_dir / "knn_teacher_corrected_aggregate.csv", index=False
    )
    paired.to_csv(
        args.output_dir / "knn_teacher_corrected_paired_deltas.csv", index=False
    )
    write_report(
        args.output_dir / "knn_teacher_corrected_report.md",
        aggregate,
        paired,
        merged,
    )
    print("wrote corrected kNN-teacher summary")


if __name__ == "__main__":
    main()
