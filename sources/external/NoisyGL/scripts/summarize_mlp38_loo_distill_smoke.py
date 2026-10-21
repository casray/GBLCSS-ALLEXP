from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull")
METHODS = {
    "gcn_xh": "mlp_distill_gcn_xh",
    "default": "mlp_distill_softdist_xh",
    "loo": "mlp_distill_softdist_xh_loo",
    "loo_shuffled": "mlp_distill_softdist_xh_loo_shuffled",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp38_loo_distill_smoke_20260717"


def _float(value, default=0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _int(value, default=0) -> int:
    if value is None or str(value).strip() == "":
        return int(default)
    return int(float(value))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    deltas = [_float(row["loo"]) - _float(row["default"]) for row in rows]
    structure = [_float(row["loo"]) - _float(row["loo_shuffled"]) for row in rows]
    positive_count = sum(delta > 0.0 for delta in deltas)
    mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
    structure_count = sum(delta >= 0.01 for delta in structure)
    no_large_regression = bool(deltas) and all(delta >= -0.005 for delta in deltas)
    passed = (
        len(rows) == len(DATASETS)
        and positive_count >= 2
        and mean_delta >= 0.005
        and no_large_regression
        and structure_count >= 2
    )
    return {
        "decision": "expand_5seed" if passed else "stop_seed0",
        "positive_dataset_count": positive_count,
        "mean_delta_loo_vs_default": mean_delta,
        "no_regression_over_0p5pp": int(no_large_regression),
        "real_structure_pass_count": structure_count,
    }


def protocol_errors(raw_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    expected = {(dataset, method) for dataset in DATASETS for method in METHODS.values()}
    observed: set[tuple[str, str]] = set()
    for row in raw_rows:
        dataset = str(row.get("dataset", "")).lower()
        method = str(row.get("method", ""))
        observed.add((dataset, method))
        if str(row.get("status", "ok")).lower() not in {"", "ok"}:
            errors.append(f"failed:{dataset}:{method}")
        if str(row.get("noise_type", "")).lower() != "uniform" or not math.isclose(
            _float(row.get("noise_rate")), 0.4, abs_tol=1e-12
        ):
            errors.append(f"wrong_noise:{dataset}:{method}")
        if str(row.get("noise_scope", "")).lower() != "train" or _int(row.get("seed")) != 0:
            errors.append(f"wrong_scope_or_seed:{dataset}:{method}")
        for key, expected_value in (("train_percent", 0.2), ("val_percent", 0.1), ("test_percent", 0.7)):
            if key in row and str(row.get(key, "")).strip() and not math.isclose(
                _float(row.get(key)), expected_value, abs_tol=1e-12
            ):
                errors.append(f"wrong_split:{dataset}:{method}:{key}")
        if _int(row.get("modified_val_count")) != 0 or _int(row.get("modified_test_count")) != 0:
            errors.append(f"label_leakage:{dataset}:{method}")
    for dataset, method in sorted(expected - observed):
        errors.append(f"missing:{dataset}:{method}")
    for dataset, method in sorted(observed - expected):
        errors.append(f"unexpected:{dataset}:{method}")
    return sorted(set(errors))


def build_summary(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    indexed = {
        (str(row.get("dataset", "")).lower(), str(row.get("method", ""))): row
        for row in raw_rows
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        result: dict[str, object] = {"dataset": dataset}
        for arm, method in METHODS.items():
            result[arm] = _float(indexed[(dataset, method)].get("test"))
        result["delta_loo_default"] = _float(result["loo"]) - _float(result["default"])
        result["delta_loo_gcn_xh"] = _float(result["loo"]) - _float(result["gcn_xh"])
        result["delta_loo_shuffled"] = _float(result["loo"]) - _float(result["loo_shuffled"])
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the MLP-38 LOO-LCSS distillation smoke.")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    input_path = args.input or args.run_dir / "seed0.csv"
    raw_rows = read_csv(input_path)
    errors = protocol_errors(raw_rows)
    if errors:
        raise ValueError("Invalid MLP-38 experiment rows: " + ";".join(errors))
    summary = build_summary(raw_rows)
    gate = evaluate_gate(summary)
    write_csv(args.run_dir / "mlp38_summary.csv", summary)

    lines = [
        "# MLP-38 LOO-LCSS Distillation Seed0 Smoke",
        "",
        "| Dataset | GCN[X||H] CE | Default LCSS teacher | LOO-LCSS teacher | LOO shuffled | LOO-default | LOO-shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {100 * _float(row['gcn_xh']):.2f} | "
            f"{100 * _float(row['default']):.2f} | {100 * _float(row['loo']):.2f} | "
            f"{100 * _float(row['loo_shuffled']):.2f} | "
            f"{100 * _float(row['delta_loo_default']):+.2f}pp | "
            f"{100 * _float(row['delta_loo_shuffled']):+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gate",
            "",
            f"- Decision: `{gate['decision']}`",
            f"- Positive LOO-default datasets: {gate['positive_dataset_count']}/3",
            f"- Mean LOO-default: {100 * _float(gate['mean_delta_loo_vs_default']):+.2f}pp",
            f"- No regression over 0.5pp: {gate['no_regression_over_0p5pp']}",
            f"- LOO real-shuffled >=1pp datasets: {gate['real_structure_pass_count']}/3",
            "",
            "Seed0 is a directional smoke only; it is not evidence of statistical stability.",
        ]
    )
    (args.run_dir / "mlp38_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
