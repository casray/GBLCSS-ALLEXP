from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp41_knn_k5_teacher_20260717"
DEFAULT_MLP9_DIR = REPO_ROOT / "runs" / "mlp9_denoised_teacher_20260702"
DEFAULT_MLP22_DIR = REPO_ROOT / "runs" / "mlp22_corafull_table3_table4_20260710"

K5 = "mlp_distill_knn_k5_xh"
BALL = "mlp_distill_softdist_xh"
MATCHED_KNN = "mlp_distill_knn_xh"
METHODS = (K5, BALL, MATCHED_KNN)
DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SEEDS = tuple(range(5))


def read_csv_compat(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    for encoding in ("utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"cannot decode {path}")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "method" not in frame.columns and "base_method" in frame.columns:
        frame["method"] = frame["base_method"]
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


def _select(frame: pd.DataFrame, methods: set[str]) -> pd.DataFrame:
    frame = _normalize(frame)
    return frame[
        frame["dataset"].isin(DATASETS)
        & frame["method"].isin(methods)
        & frame["seed"].isin(SEEDS)
        & frame["noise_type"].eq("uniform")
        & frame["noise_rate"].isin(NOISE_RATES)
        & frame["noise_scope"].eq("train")
    ].copy()


def _validate_rows(frame: pd.DataFrame, expected_methods: set[str], expected_rows: int, source: str) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"{source} incomplete: expected {expected_rows} rows, found {len(frame)}")
    if set(frame["method"]) != expected_methods:
        raise ValueError(f"{source} method set mismatch")
    if not frame["status"].astype(str).str.lower().eq("ok").all():
        raise ValueError(f"{source} contains failed rows")
    if not frame["test"].map(math.isfinite).all() or not frame["test"].between(0.0, 1.0).all():
        raise ValueError(f"{source} contains invalid test accuracy")
    if frame[["modified_val_count", "modified_test_count"]].to_numpy().max() != 0:
        raise ValueError(f"{source} modified validation or test labels")
    if not frame["modified_count"].eq(frame["modified_train_count"]).all():
        raise ValueError(f"{source} modified labels outside train")
    if frame.duplicated(["dataset", "noise_rate", "seed", "method"]).any():
        raise ValueError(f"{source} contains duplicate cells")


def build_summary(k5_frame: pd.DataFrame, anchor_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    k5 = _select(k5_frame, {K5})
    anchors = _select(anchor_frame, {BALL, MATCHED_KNN})
    _validate_rows(k5, {K5}, 30, "k=5 runs")
    _validate_rows(anchors, {BALL, MATCHED_KNN}, 60, "teacher anchors")
    if set(pd.to_numeric(k5["matched_k"], errors="raise").astype(int)) != {5}:
        raise ValueError("k=5 runs contain a non-5 matched_k")

    combined = pd.concat([k5, anchors], ignore_index=True, sort=False)
    for key, group in combined.groupby(["dataset", "noise_rate", "seed"], sort=True):
        if set(group["method"]) != set(METHODS) or len(group) != 3:
            raise ValueError(f"paired method set mismatch for {key}")
        for column in ("split_hash", "noise_hash"):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"hash mismatch for {key}: {column}")

    rows = []
    for (dataset, noise_rate), group in combined.groupby(["dataset", "noise_rate"], sort=True):
        pivot = group.pivot(index="seed", columns="method", values="test").sort_index()
        k5_values = pivot[K5]
        ball_values = pivot[BALL]
        matched_values = pivot[MATCHED_KNN]
        rows.append(
            {
                "dataset": dataset,
                "noise_rate": float(noise_rate),
                "n": len(pivot),
                "k5_mean": float(k5_values.mean()),
                "k5_std": float(k5_values.std(ddof=1)),
                "ball_mean": float(ball_values.mean()),
                "ball_std": float(ball_values.std(ddof=1)),
                "matched_knn_mean": float(matched_values.mean()),
                "matched_knn_std": float(matched_values.std(ddof=1)),
                "k5_minus_ball": float((k5_values - ball_values).mean()),
                "k5_vs_ball_positive": int((k5_values > ball_values).sum()),
                "k5_minus_matched_knn": float((k5_values - matched_values).mean()),
                "k5_vs_matched_positive": int((k5_values > matched_values).sum()),
            }
        )
    aggregate = pd.DataFrame(rows).sort_values(["dataset", "noise_rate"]).reset_index(drop=True)
    return combined.sort_values(["dataset", "noise_rate", "seed", "method"]).reset_index(drop=True), aggregate


def build_report(aggregate: pd.DataFrame) -> str:
    lines = [
        "# MLP-41 Fixed k=5 kNN Teacher Report",
        "",
        "Protocol: Cora/CiteSeer/CoraFull, train-only uniform 0.4/0.6, percent 0.2/0.1/0.7, seeds 0-4.",
        "",
        "| Dataset | Noise | kNN k=5 teacher | Ball teacher | Original matched-k teacher | k5-ball | k5-matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']:.1f} | "
            f"{100 * row['k5_mean']:.2f}+-{100 * row['k5_std']:.2f} | "
            f"{100 * row['ball_mean']:.2f}+-{100 * row['ball_std']:.2f} | "
            f"{100 * row['matched_knn_mean']:.2f}+-{100 * row['matched_knn_std']:.2f} | "
            f"{100 * row['k5_minus_ball']:+.2f}pp ({row['k5_vs_ball_positive']}/5) | "
            f"{100 * row['k5_minus_matched_knn']:+.2f}pp ({row['k5_vs_matched_positive']}/5) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This experiment compares the ball teacher with a deliberately fixed local kNN scale (k=5). "
            "It does not replace the historical evidence that scale-matched kNN can equal or exceed the ball teacher.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_inputs(run_dir: Path, mlp9_dir: Path, mlp22_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    k5_paths = (
        run_dir / "uniform0p4_seed0.csv",
        run_dir / "uniform0p4_seeds_1_2_3_4.csv",
        run_dir / "uniform0p6_seed0.csv",
        run_dir / "uniform0p6_seeds_1_2_3_4.csv",
    )
    mlp9_paths = (
        mlp9_dir / "mlp9_uniform0p4_seed0.csv",
        mlp9_dir / "mlp9_uniform0p4_seeds_1_2_3_4.csv",
        mlp9_dir / "mlp9_uniform0p6_seed0.csv",
        mlp9_dir / "mlp9_uniform0p6_seeds_1_2_3_4.csv",
    )
    mlp22_paths = (
        mlp22_dir / "table3_corafull_uniform0p4_seeds_0_4.csv",
        mlp22_dir / "table3_corafull_uniform0p6_seeds_0_4.csv",
    )
    for path in (*k5_paths, *mlp9_paths, *mlp22_paths):
        if not path.exists():
            raise FileNotFoundError(path)
    k5 = pd.concat([read_csv_compat(path) for path in k5_paths], ignore_index=True, sort=False)
    anchors = pd.concat([read_csv_compat(path) for path in (*mlp9_paths, *mlp22_paths)], ignore_index=True, sort=False)
    return k5, anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed k=5 kNN teacher distillation.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mlp9-dir", type=Path, default=DEFAULT_MLP9_DIR)
    parser.add_argument("--mlp22-dir", type=Path, default=DEFAULT_MLP22_DIR)
    args = parser.parse_args()
    k5, anchors = _load_inputs(args.run_dir, args.mlp9_dir, args.mlp22_dir)
    combined, aggregate = build_summary(k5, anchors)
    combined.to_csv(args.run_dir / "mlp41_combined.csv", index=False, encoding="utf-8")
    aggregate.to_csv(args.run_dir / "mlp41_aggregate.csv", index=False, encoding="utf-8")
    report = build_report(aggregate)
    (args.run_dir / "mlp41_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
