from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp40_coraml_coauthorph_fullspectrum_20260717"

OURS = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
BASELINES = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_sce",
    "mlp_elr",
    "mlp_logitclip",
)
METHODS = (OURS, SHUFFLED, *BASELINES)
EXPECTED_SETTINGS = {
    "coraml": (("uniform", 0.6), ("pair", 0.4), ("instance", 0.4)),
    "coauthorph": (("pair", 0.4), ("instance", 0.4)),
}


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


def _setting_is_expected(dataset: str, noise_type: str, noise_rate: float) -> bool:
    return any(
        noise_type == expected_type and math.isclose(noise_rate, expected_rate, abs_tol=1e-12)
        for expected_type, expected_rate in EXPECTED_SETTINGS.get(dataset, ())
    )


def _validate_protocol(frame: pd.DataFrame) -> None:
    if not frame["noise_scope"].eq("train").all():
        raise ValueError("non-train-only noise row found")
    optional_protocol = {
        "split_type": "percent",
        "train_percent": 0.2,
        "val_percent": 0.1,
        "test_percent": 0.7,
    }
    for column, expected in optional_protocol.items():
        if column not in frame.columns:
            continue
        if isinstance(expected, float):
            values = pd.to_numeric(frame[column], errors="raise")
            if not values.sub(expected).abs().lt(1e-12).all():
                raise ValueError(f"protocol mismatch in {column}")
        elif not frame[column].astype(str).str.lower().eq(expected).all():
            raise ValueError(f"protocol mismatch in {column}")


def _validate_rows(frame: pd.DataFrame) -> None:
    if len(frame) != 45:
        raise ValueError(f"incomplete seed0 matrix: expected 45 rows, found {len(frame)}")
    if not frame["status"].astype(str).str.lower().eq("ok").all():
        raise ValueError("failed run found")
    if not frame["test"].map(math.isfinite).all() or not frame["test"].between(0.0, 1.0).all():
        raise ValueError("invalid test accuracy found")
    if frame[["modified_val_count", "modified_test_count"]].to_numpy().max() != 0:
        raise ValueError("modified validation or test labels")
    if not frame["modified_count"].eq(frame["modified_train_count"]).all():
        raise ValueError("modified labels are not confined to the train split")
    if frame.duplicated(["dataset", "noise_type", "noise_rate", "seed", "method"]).any():
        raise ValueError("duplicate seed0 cell found")

    observed_settings = {
        (row.dataset, row.noise_type, float(row.noise_rate))
        for row in frame[["dataset", "noise_type", "noise_rate"]].itertuples(index=False)
    }
    expected_settings = {
        (dataset, noise_type, noise_rate)
        for dataset, settings in EXPECTED_SETTINGS.items()
        for noise_type, noise_rate in settings
    }
    if observed_settings != expected_settings:
        raise ValueError(f"setting set mismatch: {sorted(observed_settings)}")

    for key, group in frame.groupby(["dataset", "noise_type", "noise_rate", "seed"], sort=True):
        if set(group["method"]) != set(METHODS) or len(group) != len(METHODS):
            raise ValueError(f"method set mismatch for {key}")
        for column in ("split_hash", "noise_hash"):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"hash mismatch for {key}: {column}")


def _build_gates(setting_summary: pd.DataFrame) -> pd.DataFrame:
    gates = []
    coraml = setting_summary[setting_summary["dataset"] == "coraml"]
    coraml_reasons = []
    coraml_positive = int((coraml["lcss_minus_best"] > 0).sum())
    coraml_mean = float(coraml["lcss_minus_best"].mean())
    coraml_min = float(coraml["lcss_minus_best"].min())
    coraml_structure = int((coraml["lcss_minus_shuffled"] >= 0.01 - 1e-12).sum())
    if coraml_positive < 2:
        coraml_reasons.append("fewer_than_2_of_3_positive_vs_best")
    if coraml_mean < 0.01 - 1e-12:
        coraml_reasons.append("mean_delta_below_1pp")
    if coraml_min < -0.01 - 1e-12:
        coraml_reasons.append("setting_regression_below_minus_1pp")
    if coraml_structure < 3:
        coraml_reasons.append("fewer_than_3_of_3_structure_deltas_ge_1pp")
    gates.append(
        {
            "dataset": "coraml",
            "pass": int(not coraml_reasons),
            "total_settings": len(coraml),
            "positive_vs_best_settings": coraml_positive,
            "required_positive_settings": 2,
            "mean_lcss_minus_best": coraml_mean,
            "minimum_lcss_minus_best": coraml_min,
            "structure_ge_1pp_settings": coraml_structure,
            "required_structure_settings": 3,
            "reasons": "pass" if not coraml_reasons else ";".join(coraml_reasons),
        }
    )

    physics = setting_summary[setting_summary["dataset"] == "coauthorph"]
    physics_reasons = []
    physics_positive = int((physics["lcss_minus_best"] > 0).sum())
    physics_mean = float(physics["lcss_minus_best"].mean())
    physics_min = float(physics["lcss_minus_best"].min())
    physics_structure = int((physics["lcss_minus_shuffled"] >= 0.01 - 1e-12).sum())
    if physics_positive < 2:
        physics_reasons.append("fewer_than_2_of_2_positive_vs_best")
    if physics_mean < 0.005 - 1e-12:
        physics_reasons.append("mean_delta_below_0p5pp")
    if physics_structure < 2:
        physics_reasons.append("fewer_than_2_of_2_structure_deltas_ge_1pp")
    gates.append(
        {
            "dataset": "coauthorph",
            "pass": int(not physics_reasons),
            "total_settings": len(physics),
            "positive_vs_best_settings": physics_positive,
            "required_positive_settings": 2,
            "mean_lcss_minus_best": physics_mean,
            "minimum_lcss_minus_best": physics_min,
            "structure_ge_1pp_settings": physics_structure,
            "required_structure_settings": 2,
            "reasons": "pass" if not physics_reasons else ";".join(physics_reasons),
        }
    )
    return pd.DataFrame(gates)


def build_summary(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _normalize(raw)
    frame = frame[
        frame["dataset"].isin(EXPECTED_SETTINGS)
        & frame["method"].isin(METHODS)
        & (frame["seed"] == 0)
    ].copy()
    frame = frame[
        frame.apply(
            lambda row: _setting_is_expected(
                str(row["dataset"]), str(row["noise_type"]), float(row["noise_rate"])
            ),
            axis=1,
        )
    ].copy()
    _validate_protocol(frame)
    _validate_rows(frame)

    rows = []
    for (dataset, noise_type, noise_rate), group in frame.groupby(
        ["dataset", "noise_type", "noise_rate"], sort=True
    ):
        values = group.set_index("method")["test"]
        best_method = max(BASELINES, key=lambda method: float(values.loc[method]))
        rows.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": float(noise_rate),
                "lcss": float(values.loc[OURS]),
                "shuffled": float(values.loc[SHUFFLED]),
                "best_baseline_method": best_method,
                "best_baseline": float(values.loc[best_method]),
                "lcss_minus_best": float(values.loc[OURS] - values.loc[best_method]),
                "lcss_minus_shuffled": float(values.loc[OURS] - values.loc[SHUFFLED]),
            }
        )
    setting_summary = pd.DataFrame(rows).sort_values(
        ["dataset", "noise_type", "noise_rate"]
    ).reset_index(drop=True)
    gates = _build_gates(setting_summary)
    return (
        frame.sort_values(["dataset", "noise_type", "noise_rate", "method"]).reset_index(drop=True),
        setting_summary,
        gates,
    )


def build_report(setting_summary: pd.DataFrame, gates: pd.DataFrame) -> str:
    lines = [
        "# MLP-40 Cora_ML / Coauthor-Physics Missing-Spectrum Seed0 Report",
        "",
        "Protocol: percent 0.2/0.1/0.7, train-only noise, seed 0, frozen LCSS and baseline configs.",
        "",
        "| Dataset | Setting | LCSS | Best formal baseline | Shuffled | LCSS-best | LCSS-shuffled |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in setting_summary.to_dict("records"):
        setting = f"{row['noise_type']}{row['noise_rate']:.1f}"
        best = f"{row['best_baseline_method']} {100 * row['best_baseline']:.2f}"
        lines.append(
            f"| {row['dataset']} | {setting} | {100 * row['lcss']:.2f} | {best} | "
            f"{100 * row['shuffled']:.2f} | {100 * row['lcss_minus_best']:+.2f}pp | "
            f"{100 * row['lcss_minus_shuffled']:+.2f}pp |"
        )
    lines.extend(["", "## Frozen Seed0 Gates", ""])
    for gate in gates.to_dict("records"):
        lines.extend(
            [
                f"### {gate['dataset']}",
                f"- Pass: {gate['pass']}",
                f"- Positive vs best: {gate['positive_vs_best_settings']}/{gate['total_settings']} "
                f"(required >= {gate['required_positive_settings']})",
                f"- Mean LCSS-best: {100 * gate['mean_lcss_minus_best']:+.2f}pp",
                f"- Minimum LCSS-best: {100 * gate['minimum_lcss_minus_best']:+.2f}pp",
                f"- Structure deltas >=1pp: {gate['structure_ge_1pp_settings']}/{gate['required_structure_settings']}",
                f"- Reasons: {gate['reasons']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "Neither dataset passes its frozen seed0 expansion gate. Seeds 1-4 are not run. "
            "Cora_ML remains a breadth/boundary dataset; Coauthor-Physics remains a uniform-spectrum boundary dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MLP-40 missing-spectrum seed0 runs.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.run_dir.glob("*_seed0.csv"))
    if len(input_paths) != 5:
        raise ValueError(f"expected five seed0 input files, found {len(input_paths)}")
    raw = pd.concat([read_csv_compat(path) for path in input_paths], ignore_index=True, sort=False)
    combined, setting_summary, gates = build_summary(raw)
    combined.to_csv(args.run_dir / "mlp40_seed0_combined.csv", index=False, encoding="utf-8")
    setting_summary.to_csv(args.run_dir / "mlp40_seed0_summary.csv", index=False, encoding="utf-8")
    gates.to_csv(args.run_dir / "mlp40_gate_decisions.csv", index=False, encoding="utf-8")
    report = build_report(setting_summary, gates)
    (args.run_dir / "mlp40_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
