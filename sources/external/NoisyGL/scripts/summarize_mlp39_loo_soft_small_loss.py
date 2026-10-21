from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cora", "citeseer", "corafull")
LOO_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss_loo"
LOO_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_small_loss_loo_shuffled"
DEFAULT_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
LOO_LCSS = "frozengrace_gb_mlp_softdist_beta05_loo"
SMALL_ONLY = "mlp_small_loss"
METHODS = (LOO_COMBO, LOO_SHUFFLED, DEFAULT_COMBO, LOO_LCSS, SMALL_ONLY)


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
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["method"] = frame["method"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["noise_rate"] = pd.to_numeric(frame["noise_rate"], errors="raise")
    frame["test"] = pd.to_numeric(frame["test"], errors="raise")
    for column in ("modified_val_count", "modified_test_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def _select(frame: pd.DataFrame, methods: set[str]) -> pd.DataFrame:
    frame = _normalize(frame)
    selected = frame[
        frame["dataset"].isin(DATASETS)
        & frame["method"].isin(methods)
        & (frame["seed"] == 0)
        & (frame["noise_type"].astype(str).str.lower() == "uniform")
        & (frame["noise_rate"].sub(0.4).abs() < 1e-12)
        & (frame["noise_scope"].astype(str).str.lower() == "train")
    ].copy()
    if "train_percent" in selected.columns:
        train_percent = pd.to_numeric(selected["train_percent"], errors="coerce")
        selected = selected[train_percent.sub(0.2).abs() < 1e-12]
    return selected


def _validate_rows(frame: pd.DataFrame, expected_methods: set[str], expected_rows: int, source: str) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"{source} incomplete: expected {expected_rows} rows, found {len(frame)}")
    if set(frame["method"]) != expected_methods:
        raise ValueError(f"{source} method set mismatch: {sorted(set(frame['method']))}")
    if set(frame["dataset"]) != set(DATASETS):
        raise ValueError(f"{source} dataset set mismatch: {sorted(set(frame['dataset']))}")
    if frame.duplicated(["dataset", "method", "seed"]).any():
        raise ValueError(f"{source} contains duplicate cells")
    if not frame["status"].astype(str).str.lower().eq("ok").all():
        raise ValueError(f"{source} contains failed rows")
    if frame[["modified_val_count", "modified_test_count"]].to_numpy().max() != 0:
        raise ValueError(f"{source} modified validation or test labels")
    if not frame["test"].between(0.0, 1.0).all():
        raise ValueError(f"{source} contains invalid test accuracy")


def build_summary(
    smoke: pd.DataFrame,
    mlp30: pd.DataFrame,
    mlp36: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    smoke = _select(smoke, {LOO_COMBO, LOO_SHUFFLED})
    mlp30 = _select(mlp30, {DEFAULT_COMBO, SMALL_ONLY})
    mlp36 = _select(mlp36, {LOO_LCSS})
    _validate_rows(smoke, {LOO_COMBO, LOO_SHUFFLED}, 6, "smoke")
    _validate_rows(mlp30, {DEFAULT_COMBO, SMALL_ONLY}, 6, "MLP-30 anchors")
    _validate_rows(mlp36, {LOO_LCSS}, 3, "MLP-36 anchors")

    shared_columns = sorted(set(smoke.columns) | set(mlp30.columns) | set(mlp36.columns))
    combined = pd.concat(
        [frame.reindex(columns=shared_columns) for frame in (smoke, mlp30, mlp36)],
        ignore_index=True,
    )
    for dataset, group in combined.groupby("dataset", sort=True):
        for column in ("split_hash", "noise_hash"):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"hash mismatch for {dataset}: {column}")

    pivot = combined.pivot(index="dataset", columns="method", values="test")
    rows = []
    for dataset in DATASETS:
        values = pivot.loc[dataset]
        rows.append(
            {
                "dataset": dataset,
                "loo_combo": float(values[LOO_COMBO]),
                "default_combo": float(values[DEFAULT_COMBO]),
                "loo_lcss": float(values[LOO_LCSS]),
                "small_loss": float(values[SMALL_ONLY]),
                "loo_shuffled_combo": float(values[LOO_SHUFFLED]),
                "loo_minus_default": float(values[LOO_COMBO] - values[DEFAULT_COMBO]),
                "small_loss_on_loo_gain": float(values[LOO_COMBO] - values[LOO_LCSS]),
                "loo_combo_minus_small": float(values[LOO_COMBO] - values[SMALL_ONLY]),
                "structure_delta": float(values[LOO_COMBO] - values[LOO_SHUFFLED]),
            }
        )
    deltas = pd.DataFrame(rows)
    positive = int((deltas["loo_minus_default"] > 0).sum())
    mean_increment = float(deltas["loo_minus_default"].mean())
    minimum_increment = float(deltas["loo_minus_default"].min())
    structure = int((deltas["structure_delta"] >= 0.01 - 1e-12).sum())
    reasons = []
    if positive < 2:
        reasons.append("fewer_than_2_of_3_positive_vs_default")
    if mean_increment < 0.005 - 1e-12:
        reasons.append("mean_increment_below_0p5pp")
    if minimum_increment < -0.005 - 1e-12:
        reasons.append("dataset_regression_below_minus_0p5pp")
    if structure < 2:
        reasons.append("fewer_than_2_of_3_structure_deltas_ge_1pp")
    gate = {
        "pass": int(not reasons),
        "positive_vs_default_datasets": positive,
        "mean_loo_increment": mean_increment,
        "minimum_loo_increment": minimum_increment,
        "structure_ge_1pp_datasets": structure,
        "reasons": "pass" if not reasons else ";".join(reasons),
    }
    return combined.sort_values(["dataset", "method"]).reset_index(drop=True), deltas, gate


def build_report(deltas: pd.DataFrame, gate: dict[str, object]) -> str:
    lines = [
        "# MLP-39 LOO-LCSS + Soft-Small-Loss Seed0 Smoke",
        "",
        "| Dataset | LOO combo | Default combo | LOO-LCSS | Small-loss | LOO shuffled | LOO-default | LOO structure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {100*row['loo_combo']:.2f} | {100*row['default_combo']:.2f} | "
            f"{100*row['loo_lcss']:.2f} | {100*row['small_loss']:.2f} | "
            f"{100*row['loo_shuffled_combo']:.2f} | {100*row['loo_minus_default']:+.2f}pp | "
            f"{100*row['structure_delta']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Pass: {gate['pass']}",
            f"- Positive vs default: {gate['positive_vs_default_datasets']}/3",
            f"- Mean LOO conditional increment: {100*float(gate['mean_loo_increment']):+.2f}pp",
            f"- Minimum dataset increment: {100*float(gate['minimum_loo_increment']):+.2f}pp",
            f"- Structure deltas >=1pp: {gate['structure_ge_1pp_datasets']}/3",
            f"- Reasons: {gate['reasons']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the MLP-39 six-run seed0 smoke.")
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument(
        "--mlp30",
        type=Path,
        default=REPO_ROOT / "runs" / "mlp30_noise_rate_grid_seed0_20260712" / "mlp30_combined_raw.csv",
    )
    parser.add_argument(
        "--mlp36",
        type=Path,
        default=REPO_ROOT / "runs" / "mlp36_review_defense_20260715" / "mlp36_raw.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined, deltas, gate = build_summary(
        read_csv_compat(args.smoke),
        read_csv_compat(args.mlp30),
        read_csv_compat(args.mlp36),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "mlp39_combined.csv", index=False, encoding="utf-8")
    deltas.to_csv(args.output_dir / "mlp39_deltas.csv", index=False, encoding="utf-8")
    pd.DataFrame([gate]).to_csv(args.output_dir / "mlp39_gate.csv", index=False, encoding="utf-8")
    report = build_report(deltas, gate)
    (args.output_dir / "mlp39_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
