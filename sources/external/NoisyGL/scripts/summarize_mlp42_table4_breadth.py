from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp42_table4_breadth_20260717"
DEFAULT_U04 = (
    REPO_ROOT
    / "runs"
    / "mlp16_distill_tableb_completion_20260707"
    / "distill_tableb_uniform0p4_seeds_0_1_2_3_4.csv"
)

GCN = "mlp_distill_gcn"
GCN_XH = "mlp_distill_gcn_xh"
SHUFFLED = "mlp_distill_softdist_xh_shuffled"
KNN = "mlp_distill_knn_xh"
BALL = "mlp_distill_softdist_xh"
METHODS = (GCN, GCN_XH, SHUFFLED, KNN, BALL)
DATASETS = ("coauthorcs", "amazonpho")
NOISE_RATES = (0.4, 0.6)
SEEDS = tuple(range(5))

METHOD_LABELS = {
    GCN: "GCN(X)+CE (GLNN-style)",
    GCN_XH: "GCN[X||H]+CE",
    SHUFFLED: "Shuffled-consensus teacher",
    KNN: "Vote-matched kNN-consensus teacher",
    BALL: "Ball-consensus teacher",
}


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["method"] = frame["method"].astype(str)
    frame["noise_type"] = frame["noise_type"].astype(str).str.lower()
    frame["noise_scope"] = frame["noise_scope"].astype(str).str.lower()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["noise_rate"] = pd.to_numeric(frame["noise_rate"], errors="raise")
    frame["test"] = pd.to_numeric(frame["test"], errors="raise")
    for column in (
        "modified_count",
        "modified_train_count",
        "modified_val_count",
        "modified_test_count",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def _select(frame: pd.DataFrame, noise_rate: float) -> pd.DataFrame:
    frame = _normalize(frame)
    return frame[
        frame["dataset"].isin(DATASETS)
        & frame["method"].isin(METHODS)
        & frame["seed"].isin(SEEDS)
        & frame["noise_type"].eq("uniform")
        & frame["noise_rate"].eq(noise_rate)
        & frame["noise_scope"].eq("train")
    ].copy()


def _validate_rate(frame: pd.DataFrame, noise_rate: float) -> None:
    expected_rows = len(DATASETS) * len(METHODS) * len(SEEDS)
    if len(frame) != expected_rows:
        raise ValueError(f"uniform {noise_rate} incomplete: expected {expected_rows}, found {len(frame)}")
    if frame.duplicated(["dataset", "noise_rate", "seed", "method"]).any():
        raise ValueError(f"uniform {noise_rate} contains duplicate cells")
    if not frame["status"].astype(str).str.lower().eq("ok").all():
        raise ValueError(f"uniform {noise_rate} contains failed rows")
    if not frame["test"].map(math.isfinite).all() or not frame["test"].between(0.0, 1.0).all():
        raise ValueError(f"uniform {noise_rate} contains invalid test accuracy")
    if frame[["modified_val_count", "modified_test_count"]].to_numpy().max() != 0:
        raise ValueError(f"uniform {noise_rate} modified validation or test labels")
    if not frame["modified_count"].eq(frame["modified_train_count"]).all():
        raise ValueError(f"uniform {noise_rate} modified labels outside train")

    for key, group in frame.groupby(["dataset", "noise_rate", "seed"], sort=True):
        if len(group) != len(METHODS) or set(group["method"]) != set(METHODS):
            raise ValueError(f"paired method set mismatch for {key}")
        for column in ("split_hash", "noise_hash"):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"hash mismatch for {key}: {column}")

    if noise_rate == 0.6 and "matched_k" in frame.columns:
        knn = frame[frame["method"].eq(KNN)]
        values = pd.to_numeric(knn["matched_k"], errors="raise")
        if values.isna().any() or not (values > 0).all() or not (values % 1 == 0).all():
            raise ValueError("uniform 0.6 kNN rows contain invalid matched_k")


def build_outputs(
    uniform04_frame: pd.DataFrame,
    uniform06_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uniform04 = _select(uniform04_frame, 0.4)
    uniform06 = _select(uniform06_frame, 0.6)
    _validate_rate(uniform04, 0.4)
    _validate_rate(uniform06, 0.6)
    combined = pd.concat([uniform04, uniform06], ignore_index=True, sort=False)

    aggregate_rows = []
    paired_rows = []
    comparison_names = {
        (GCN_XH, GCN): "gcn_xh_minus_gcn",
        (SHUFFLED, GCN): "shuffled_minus_gcn",
        (KNN, GCN): "knn_minus_gcn",
        (BALL, GCN): "ball_minus_gcn",
        (SHUFFLED, GCN_XH): "shuffled_minus_gcn_xh",
        (KNN, GCN_XH): "knn_minus_gcn_xh",
        (BALL, GCN_XH): "ball_minus_gcn_xh",
        (KNN, SHUFFLED): "knn_minus_shuffled",
        (BALL, SHUFFLED): "ball_minus_shuffled",
        (BALL, KNN): "ball_minus_knn",
    }
    for (dataset, noise_rate), group in combined.groupby(["dataset", "noise_rate"], sort=True):
        pivot = group.pivot(index="seed", columns="method", values="test").sort_index()
        for method in METHODS:
            values = pivot[method]
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "noise_rate": float(noise_rate),
                    "method": method,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )
        for left, right in combinations(METHODS, 2):
            pair = (right, left)
            delta = pivot[pair[0]] - pivot[pair[1]]
            paired_rows.append(
                {
                    "dataset": dataset,
                    "noise_rate": float(noise_rate),
                    "comparison": comparison_names[pair],
                    "left_method": pair[0],
                    "right_method": pair[1],
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)),
                    "positive_seeds": int((delta > 0).sum()),
                    "zero_seeds": int((delta == 0).sum()),
                    "seed_deltas": ";".join(f"{seed}:{value:.8f}" for seed, value in delta.items()),
                }
            )

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["dataset", "noise_rate", "method"]
    ).reset_index(drop=True)
    paired = pd.DataFrame(paired_rows).sort_values(
        ["dataset", "noise_rate", "comparison"]
    ).reset_index(drop=True)
    combined = combined.sort_values(["dataset", "noise_rate", "seed", "method"]).reset_index(drop=True)
    return combined, aggregate, paired


def build_report(aggregate: pd.DataFrame, paired: pd.DataFrame, combined: pd.DataFrame) -> str:
    lookup = aggregate.set_index(["dataset", "noise_rate", "method"])
    lines = [
        "# MLP-42 Table 4 Breadth Report",
        "",
        "Protocol: Coauthor-CS/Amazon-Photo, train-only uniform 0.4/0.6, percent 0.2/0.1/0.7, seeds 0-4.",
        "The kNN teacher uses the same vote-matched diagnostic definition as the frozen Table 4; k is not selected by validation/test accuracy.",
        "",
    ]
    for noise_rate in NOISE_RATES:
        lines.extend(
            [
                f"## Uniform {noise_rate:.1f}",
                "",
                "| Teacher | Coauthor-CS | Amazon-Photo |",
                "|---|---:|---:|",
            ]
        )
        for method in METHODS:
            cells = []
            for dataset in DATASETS:
                row = lookup.loc[(dataset, noise_rate, method)]
                cells.append(f"{100 * row['mean']:.2f}+-{100 * row['std']:.2f}")
            lines.append(f"| {METHOD_LABELS[method]} | {cells[0]} | {cells[1]} |")
        lines.append("")

    lines.extend(
        [
            "## Paired attribution",
            "",
            "| Dataset | Noise | Ball-GCN(X) | Ball-GCN[X||H] | Ball-shuffled | Ball-kNN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    paired_lookup = paired.set_index(["dataset", "noise_rate", "comparison"])
    for dataset in DATASETS:
        for noise_rate in NOISE_RATES:
            cells = []
            for comparison in (
                "ball_minus_gcn",
                "ball_minus_gcn_xh",
                "ball_minus_shuffled",
                "ball_minus_knn",
            ):
                row = paired_lookup.loc[(dataset, noise_rate, comparison)]
                cells.append(f"{100 * row['mean_delta']:+.2f}pp ({int(row['positive_seeds'])}/5)")
            lines.append(f"| {dataset} | {noise_rate:.1f} | " + " | ".join(cells) + " |")

    if "matched_k" in combined.columns:
        k_rows = combined[(combined["noise_rate"].eq(0.6)) & combined["method"].eq(KNN)]
        lines.extend(["", "## Vote-matched k audit", ""])
        for dataset in DATASETS:
            values = sorted(set(pd.to_numeric(k_rows[k_rows["dataset"].eq(dataset)]["matched_k"]).astype(int)))
            lines.append(f"- {dataset}: {values}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an exploratory breadth extension. It does not modify the frozen v15 manuscript unless the author later elects to revise Table 4. "
            "Vote-matched kNN controls effective consensus size and must not be described as an accuracy-tuned kNN method.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_uniform06(run_dir: Path) -> pd.DataFrame:
    paths = (
        run_dir / "uniform0p6_seed0.csv",
        run_dir / "uniform0p6_seeds_1_2_3_4.csv",
        run_dir / "uniform0p6_coauthorcs_missing.csv",
        run_dir / "uniform0p6_amazonpho_missing.csv",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the MLP-42 Table 4 breadth experiment.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--uniform04", type=Path, default=DEFAULT_U04)
    args = parser.parse_args()

    uniform04 = pd.read_csv(args.uniform04)
    uniform06 = _load_uniform06(args.run_dir)
    combined, aggregate, paired = build_outputs(uniform04, uniform06)
    combined.to_csv(args.run_dir / "mlp42_combined.csv", index=False, encoding="utf-8")
    aggregate.to_csv(args.run_dir / "mlp42_aggregate.csv", index=False, encoding="utf-8")
    paired.to_csv(args.run_dir / "mlp42_paired_deltas.csv", index=False, encoding="utf-8")
    report = build_report(aggregate, paired, combined)
    (args.run_dir / "mlp42_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
