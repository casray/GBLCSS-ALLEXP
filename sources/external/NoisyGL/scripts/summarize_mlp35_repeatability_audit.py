from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mlp35_repeatability_audit import SETTING_SPECS


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_repeatability_summary(rows, *, setting_specs=SETTING_SPECS, expected_replicates=(0, 1, 2)):
    keyed = {}
    for source in rows:
        row = dict(source)
        setting_id = str(row.get("setting_id", ""))
        if setting_id not in setting_specs:
            continue
        replicate = int(row.get("replicate", -1))
        method = str(row.get("method", ""))
        spec = setting_specs[setting_id]
        expected_methods = (spec["baseline_method"], spec["combo_method"]) if "methods" not in spec else spec["methods"]
        if replicate not in expected_replicates or method not in expected_methods:
            continue
        key = (setting_id, replicate, method)
        if key in keyed:
            raise ValueError(f"duplicate repeatability cell: {key}")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok repeatability row: {key}")
        if row.get("noise_scope") != "train":
            raise ValueError(f"wrong noise scope: {key}")
        if int(row.get("seed", -1)) != 0:
            raise ValueError(f"repeatability audit requires seed0: {key}")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError(f"validation/test labels were modified: {key}")
        if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
            raise ValueError(f"noise is not train-only: {key}")
        row["replicate"] = replicate
        row["test"] = float(row["test"])
        keyed[key] = row

    expected_keys = set()
    for setting_id, spec in setting_specs.items():
        methods = (spec["baseline_method"], spec["combo_method"]) if "methods" not in spec else spec["methods"]
        for replicate in expected_replicates:
            for method in methods:
                expected_keys.add((setting_id, int(replicate), method))
    missing = expected_keys - set(keyed)
    if missing:
        raise ValueError(f"missing repeatability cells: {sorted(missing)}")

    for setting_id, spec in setting_specs.items():
        methods = (spec["baseline_method"], spec["combo_method"]) if "methods" not in spec else spec["methods"]
        all_rows = [
            keyed[(setting_id, int(replicate), method)]
            for replicate in expected_replicates
            for method in methods
        ]
        hashes = {(row["split_hash"], row["noise_hash"]) for row in all_rows}
        if len(hashes) != 1:
            raise ValueError(f"hash mismatch: {setting_id}")
        for method in methods:
            config_ids = {
                keyed[(setting_id, int(replicate), method)]["config_id"]
                for replicate in expected_replicates
            }
            if len(config_ids) != 1:
                raise ValueError(f"config hash mismatch: {setting_id} {method}")

    method_rows = []
    paired_rows = []
    for setting_id, spec in setting_specs.items():
        methods = (spec["baseline_method"], spec["combo_method"]) if "methods" not in spec else spec["methods"]
        baseline_method, combo_method = methods
        for method in methods:
            values = [
                keyed[(setting_id, int(replicate), method)]["test"]
                for replicate in expected_replicates
            ]
            method_rows.append(
                {
                    "setting_id": setting_id,
                    "dataset": spec["dataset"],
                    "method": method,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "range": max(values) - min(values),
                    "values": json.dumps(values),
                }
            )
        deltas = [
            keyed[(setting_id, int(replicate), combo_method)]["test"]
            - keyed[(setting_id, int(replicate), baseline_method)]["test"]
            for replicate in expected_replicates
        ]
        same_sign = all(value > 0 for value in deltas) or all(value < 0 for value in deltas)
        paired_rows.append(
            {
                "setting_id": setting_id,
                "dataset": spec["dataset"],
                "baseline_method": baseline_method,
                "combo_method": combo_method,
                "n": len(deltas),
                "delta_mean": statistics.mean(deltas),
                "delta_std": statistics.pstdev(deltas),
                "delta_minimum": min(deltas),
                "delta_maximum": max(deltas),
                "delta_range": max(deltas) - min(deltas),
                "positive_replicates": sum(value > 0 for value in deltas),
                "same_sign": bool(same_sign),
                "deltas": json.dumps(deltas),
            }
        )
    maximum_method_range = max(row["range"] for row in method_rows)
    same_sign_settings = sum(row["same_sign"] for row in paired_rows)
    reasons = []
    if maximum_method_range > 0.01:
        reasons.append("method_range_over_1pp")
    if same_sign_settings != len(paired_rows):
        reasons.append("paired_delta_sign_not_reproducible")
    gate = {
        "pass": not reasons,
        "maximum_method_range": maximum_method_range,
        "same_sign_settings": same_sign_settings,
        "n_settings": len(paired_rows),
        "reasons": reasons,
    }
    return method_rows, paired_rows, gate


def _report(path, method_rows, paired_rows, gate):
    lines = [
        "# MLP-35 Same-Seed ROCm Repeatability Audit",
        "",
        "> Ranges below are within-seed rerun spreads, not hardware variance estimates.",
        "",
        "## Per-method spread",
        "",
        "| Setting | Method | Mean | Std | Min | Max | Range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['setting_id']} | {row['method']} | {100*row['mean']:.2f} | "
            f"{100*row['std']:.2f} | {100*row['minimum']:.2f} | {100*row['maximum']:.2f} | "
            f"{100*row['range']:.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Paired-delta spread",
            "",
            "| Setting | Baseline | Mean delta | Min | Max | Range | Positive | Same sign |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row['setting_id']} | {row['baseline_method']} | {100*row['delta_mean']:+.2f}pp | "
            f"{100*row['delta_minimum']:+.2f}pp | {100*row['delta_maximum']:+.2f}pp | "
            f"{100*row['delta_range']:.2f}pp | {row['positive_replicates']}/{row['n']} | "
            f"{int(row['same_sign'])} |"
        )
    lines.extend(["", "## Gate", "", "```json", json.dumps(gate, indent=2), "```", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    method_rows, paired_rows, gate = build_repeatability_summary(_read(args.input))
    output = Path(args.output_dir)
    _write(output / "mlp35_repeatability_summary.csv", method_rows)
    _write(output / "mlp35_repeatability_paired.csv", paired_rows)
    (output / "mlp35_repeatability_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    _report(output / "mlp35_repeatability_report.md", method_rows, paired_rows, gate)


if __name__ == "__main__":
    main()
