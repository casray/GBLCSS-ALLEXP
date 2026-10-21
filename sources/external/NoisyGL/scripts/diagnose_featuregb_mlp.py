from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import _load_dataset
from utils.frozen_grace_targets import build_soft_targets_from_features
from utils.labelnoise import label_process
from utils.tools import load_conf, setup_seed


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _gb_value(model_conf, key, default):
    gb_conf = getattr(model_conf, "gb", {})
    if isinstance(gb_conf, dict):
        value = gb_conf.get(key, default)
    else:
        value = getattr(gb_conf, key, default)
    return default if value is None else value


def evaluate_offline_gate(rows: list[dict]) -> tuple[bool, list[dict]]:
    grouped = {}
    for row in rows:
        grouped[(str(row["dataset"]).lower(), str(row["control"]).lower())] = row
    decisions = []
    for dataset in sorted({key[0] for key in grouped}):
        real = grouped.get((dataset, "real"))
        shuffled = grouped.get((dataset, "shuffled"))
        if real is None or shuffled is None:
            decisions.append(
                {
                    "dataset": dataset,
                    "true_all_purity_gap": "",
                    "noisy_train_purity_gap": "",
                    "eligible_train_coverage": "",
                    "passed": False,
                }
            )
            continue
        true_gap = float(real["true_all_purity_weighted"]) - float(
            shuffled["true_all_purity_weighted"]
        )
        noisy_gap = float(real["noisy_train_purity_weighted"]) - float(
            shuffled["noisy_train_purity_weighted"]
        )
        coverage = float(real["eligible_train_coverage"])
        decisions.append(
            {
                "dataset": dataset,
                "true_all_purity_gap": true_gap,
                "noisy_train_purity_gap": noisy_gap,
                "eligible_train_coverage": coverage,
                "passed": (
                    true_gap >= 0.10
                    and noisy_gap >= 0.05
                    and coverage >= 0.95
                ),
            }
        )
    return (
        len(decisions) == 2 and all(bool(row["passed"]) for row in decisions),
        decisions,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def diagnose_dataset(args, dataset_name: str) -> list[dict]:
    setup_seed(args.seed)
    dataset = dataset_name.lower()
    data = _load_dataset(
        dataset,
        device=args.device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, modified_mask = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        random_seed=args.seed,
        debug=False,
    )
    conf = load_conf(None, "featuregb_mlp_softdist", dataset)
    train_count = max(1, int(np.asarray(data.train_masks).size))
    rows = []
    for control, random_control in (("real", False), ("shuffled", True)):
        result = build_soft_targets_from_features(
            ball_features=data.feats,
            noisy_label=noisy_label,
            clean_label=data.labels,
            train_mask=data.train_masks,
            n_classes=data.n_classes,
            max_balls=_gb_value(conf, "max_balls", 64),
            min_ball_size=_gb_value(conf, "min_ball_size", 32),
            split_improvement=_gb_value(conf, "split_improvement", 0.05),
            min_train_per_ball=_gb_value(conf, "min_train_per_ball", 2),
            alpha_baseline=_gb_value(conf, "alpha_baseline", "multiclass"),
            target_mode=_gb_value(conf, "target_mode", "distribution"),
            random_state=args.seed,
            random_control=random_control,
        )
        sizes = sorted(np.bincount(result.assignment).tolist())
        rows.append(
            {
                "dataset": dataset,
                "control": control,
                "seed": int(args.seed),
                "noise_type": args.noise_type,
                "noise_rate": float(args.noise_rate),
                "actual_train_noise_rate": float(
                    np.intersect1d(
                        np.asarray(data.train_masks, dtype=np.int64),
                        np.asarray(modified_mask, dtype=np.int64),
                    ).size
                    / train_count
                ),
                "eligible_train_coverage": float(
                    result.diagnostics["purity_eligible_train_total"] / train_count
                ),
                "ball_size_signature": " ".join(map(str, sizes)),
                **result.diagnostics,
            }
        )
    return rows


def _build_report(rows: list[dict], decisions: list[dict], passed: bool) -> str:
    lines = [
        "# FeatureGB-MLP Offline Diagnostic",
        "",
        "- Ball features: raw X, dense float row-L2 normalized.",
        "- Parameters: max_balls=64, min_ball_size=32, split_improvement=0.05.",
        "- Clean labels are used only for diagnostic purity/NCR/CDR.",
        "",
        "## Mechanism Diagnostics",
        "",
        "| Dataset | Control | Balls | True-all purity | Noisy-train purity | NCR | CDR | Alpha positive | mean alpha | Eligible coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['control']} | {int(row['num_balls'])} | "
            f"{float(row['true_all_purity_weighted']):.4f} | "
            f"{float(row['noisy_train_purity_weighted']):.4f} | "
            f"{float(row['soft_pseudo_ncr']):.4f} | "
            f"{float(row['soft_pseudo_cdr']):.4f} | "
            f"{float(row['alpha_positive_train_ratio']):.4f} | "
            f"{float(row['mean_alpha_train']):.4f} | "
            f"{float(row['eligible_train_coverage']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered Gate",
            "",
        "| Dataset | true-all gap | noisy-train gap | eligible coverage | Gate |",
        "|---|---:|---:|---:|---|",
        ]
    )
    for row in decisions:
        lines.append(
            f"| {row['dataset']} | {float(row['true_all_purity_gap']):+.4f} | "
            f"{float(row['noisy_train_purity_gap']):+.4f} | "
            f"{float(row['eligible_train_coverage']):.4f} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            f"Overall gate: {'PASS' if passed else 'FAIL'}.",
            "",
            (
                "A single-ball partition was observed; real and shuffled controls "
                "therefore carry the same grouping when fixed X-space geometry does "
                "not accept any split."
                if any(
                    str(row.get("control")) == "real"
                    and int(row.get("num_balls", 0)) == 1
                    for row in rows
                )
                else "No single-ball partition was observed."
            ),
            "",
            (
                "Seed-0 downstream: ALLOWED."
                if passed
                else "Seed-0 downstream: NOT RUN under the preregistered stopping rule."
            ),
            "",
            "## Claim Boundary",
            "",
            "- This diagnostic does not establish an H-free SoftDist pipeline.",
            "- The observed failure is specific to the fixed X-space geometry and preregistered parameters.",
            "- It does not rule out topology-aware or other label-free ball construction, which requires a separate preregistration.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Run the preregistered X-space granular-ball offline diagnostic."
    )
    parser.add_argument("--datasets", default="cora,citeseer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "runs" / "featuregb_mlp_diagnostic_20260618"),
    )
    args = parser.parse_args()

    rows = []
    for dataset in _parse_str_list(args.datasets):
        rows.extend(diagnose_dataset(args, dataset))
    passed, decisions = evaluate_offline_gate(rows)
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "featuregb_offline_raw.csv", rows)
    _write_csv(output_dir / "featuregb_offline_gate.csv", decisions)
    (output_dir / "featuregb_offline_report.md").write_text(
        _build_report(rows, decisions, passed),
        encoding="utf-8",
    )
    print(f"OFFLINE_GATE={'PASS' if passed else 'FAIL'}")
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
