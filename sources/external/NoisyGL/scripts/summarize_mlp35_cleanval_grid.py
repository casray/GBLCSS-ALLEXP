from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


METHOD = "frozengrace_gb_mlp_softdist_beta05_small_loss"
LCSS_METHOD = "frozengrace_gb_mlp_softdist_beta05"
DEFAULT_DATASETS = ("cora", "citeseer", "corafull", "amazonpho", "pubmed")
DEFAULT_NOISE_RATES = (0.2, 0.4, 0.6)
DEFAULT_CANDIDATE_RATES = (0.0, 0.2, 0.4, 0.6)
DEFAULT_FIXED_RATE = 0.5


def _rate(value):
    return round(float(value), 10)


def _parse_list(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_by_clean_validation(rows, candidate_rates=DEFAULT_CANDIDATE_RATES):
    allowed = {_rate(value) for value in candidate_rates}
    candidates = [row for row in rows if _rate(row["requested_forget_rate"]) in allowed]
    if not candidates:
        raise ValueError("no clean-validation candidates")
    return max(
        candidates,
        key=lambda row: (float(row["valid"]), -_rate(row["requested_forget_rate"])),
    )


def audit_forget0_against_lcss(candidate_rows, anchor_rows):
    anchors = {}
    for source in anchor_rows:
        if source.get("method") != LCSS_METHOD:
            continue
        key = (str(source["dataset"]), _rate(source["noise_rate"]), int(source["seed"]))
        if key in anchors:
            raise ValueError(f"duplicate LCSS anchor: {key}")
        anchors[key] = source
    forget0_rows = {}
    for candidate in candidate_rows:
        if _rate(candidate.get("requested_forget_rate", math.nan)) != 0.0:
            continue
        key = (
            str(candidate["dataset"]),
            _rate(candidate["noise_rate"]),
            int(candidate["seed"]),
        )
        if key in forget0_rows:
            raise ValueError(f"duplicate forget-rate-0 row: {key}")
        forget0_rows[key] = candidate
    anchors = {key: value for key, value in anchors.items() if key in forget0_rows}
    output = []
    for key, anchor in anchors.items():
        if key not in forget0_rows:
            raise ValueError(f"missing forget-rate-0 row for LCSS anchor: {key}")
        candidate = forget0_rows[key]
        if anchor.get("status") != "ok":
            raise ValueError(f"non-ok LCSS anchor: {key}")
        if (candidate["split_hash"], candidate["noise_hash"]) != (
            anchor["split_hash"],
            anchor["noise_hash"],
        ):
            raise ValueError(f"LCSS anchor hash mismatch: {key}")
        output.append(
            {
                "dataset": key[0],
                "noise_rate": key[1],
                "seed": key[2],
                "split_hash": candidate["split_hash"],
                "noise_hash": candidate["noise_hash"],
                "forget0_test": float(candidate["test"]),
                "lcss_test": float(anchor["test"]),
                "forget0_minus_lcss": float(candidate["test"]) - float(anchor["test"]),
            }
        )
    if not output:
        raise ValueError("no forget-rate-0 rows available for LCSS audit")
    output.sort(key=lambda row: (row["dataset"], row["noise_rate"], row["seed"]))
    return output


def validate_candidate_rows(
    rows,
    *,
    expected_datasets,
    expected_noise_rates,
    expected_seeds,
    candidate_rates=DEFAULT_CANDIDATE_RATES,
    fixed_rate=DEFAULT_FIXED_RATE,
):
    all_rates = tuple(candidate_rates) + (fixed_rate,)
    keyed = {}
    normalized = []
    for source in rows:
        row = dict(source)
        dataset = str(row.get("dataset", ""))
        noise_rate = _rate(row.get("noise_rate", math.nan))
        seed = int(row.get("seed", -1))
        forget_rate = _rate(row.get("requested_forget_rate", math.nan))
        if dataset not in expected_datasets or noise_rate not in {_rate(v) for v in expected_noise_rates}:
            continue
        if seed not in expected_seeds or forget_rate not in {_rate(v) for v in all_rates}:
            continue
        key = (dataset, noise_rate, seed, forget_rate)
        if key in keyed:
            raise ValueError(f"duplicate grid cell: {key}")
        if row.get("status") != "ok":
            raise ValueError(f"non-ok row: {key}")
        if row.get("method") != METHOD:
            raise ValueError(f"wrong method: {key}")
        if row.get("noise_type") != "uniform" or row.get("noise_scope") != "train":
            raise ValueError(f"wrong noise protocol: {key}")
        if row.get("split_type") != "percent":
            raise ValueError(f"wrong split type: {key}")
        if abs(float(row["train_percent"]) - 0.2) > 1e-12:
            raise ValueError(f"wrong train percent: {key}")
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError(f"validation/test labels were modified: {key}")
        if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
            raise ValueError(f"noise is not train-only: {key}")
        active = row.get("forget_rate", "")
        if active != "" and abs(float(active) - forget_rate) > 1e-12:
            raise ValueError(f"active forget rate mismatch: {key}")
        row.update(
            {
                "dataset": dataset,
                "noise_rate": noise_rate,
                "seed": seed,
                "requested_forget_rate": forget_rate,
                "valid": float(row["valid"]),
                "test": float(row["test"]),
            }
        )
        keyed[key] = row
        normalized.append(row)

    expected_keys = {
        (dataset, _rate(noise_rate), int(seed), _rate(forget_rate))
        for dataset in expected_datasets
        for noise_rate in expected_noise_rates
        for seed in expected_seeds
        for forget_rate in all_rates
    }
    missing = expected_keys - set(keyed)
    if missing:
        raise ValueError(f"missing grid cells: {sorted(missing)[:5]} (n={len(missing)})")
    extra = set(keyed) - expected_keys
    if extra:
        raise ValueError(f"unexpected grid cells: {sorted(extra)[:5]}")

    for dataset in expected_datasets:
        for noise_rate in expected_noise_rates:
            for seed in expected_seeds:
                members = [
                    keyed[(dataset, _rate(noise_rate), int(seed), _rate(forget_rate))]
                    for forget_rate in all_rates
                ]
                hashes = {(row["split_hash"], row["noise_hash"]) for row in members}
                if len(hashes) != 1:
                    raise ValueError(
                        f"hash mismatch: dataset={dataset} noise_rate={noise_rate} seed={seed}"
                    )
    normalized.sort(
        key=lambda row: (
            expected_datasets.index(row["dataset"]),
            expected_noise_rates.index(row["noise_rate"]),
            row["seed"],
            row["requested_forget_rate"],
        )
    )
    return normalized, keyed


def _seed0_gate(selected):
    deltas = [row["selected_minus_oracle"] for row in selected]
    required = math.ceil(0.8 * len(deltas))
    reasons = []
    if statistics.mean(deltas) < -0.01:
        reasons.append("mean_selected_minus_oracle_below_minus_1pp")
    if sum(value >= -0.01 for value in deltas) < required:
        reasons.append("fewer_than_80pct_within_1pp")
    if min(deltas) < -0.02:
        reasons.append("setting_regression_over_2pp")
    return {
        "complete": True,
        "pass": not reasons,
        "n_settings": len(deltas),
        "required_within_1pp": required,
        "within_1pp": sum(value >= -0.01 for value in deltas),
        "mean_selected_minus_oracle": statistics.mean(deltas),
        "minimum_selected_minus_oracle": min(deltas),
        "reasons": reasons,
    }


def _aggregate(selected, expected_datasets, expected_noise_rates):
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["dataset"], row["noise_rate"])].append(row)
    output = []
    for dataset in expected_datasets:
        for noise_rate in expected_noise_rates:
            members = grouped[(dataset, _rate(noise_rate))]
            rates = [row["selected_forget_rate"] for row in members]
            rate_counts = Counter(rates)
            output.append(
                {
                    "dataset": dataset,
                    "noise_rate": _rate(noise_rate),
                    "n": len(members),
                    "selected_test_mean": statistics.mean(row["selected_test"] for row in members),
                    "selected_test_std": statistics.pstdev(row["selected_test"] for row in members),
                    "oracle_test_mean": statistics.mean(row["oracle_test"] for row in members),
                    "oracle_test_std": statistics.pstdev(row["oracle_test"] for row in members),
                    "fixed0p5_test_mean": statistics.mean(row["fixed0p5_test"] for row in members),
                    "fixed0p5_test_std": statistics.pstdev(row["fixed0p5_test"] for row in members),
                    "forget0_test_mean": statistics.mean(row["forget0_test"] for row in members),
                    "forget0_test_std": statistics.pstdev(row["forget0_test"] for row in members),
                    "selected_minus_oracle_mean": statistics.mean(
                        row["selected_minus_oracle"] for row in members
                    ),
                    "selected_minus_oracle_std": statistics.pstdev(
                        row["selected_minus_oracle"] for row in members
                    ),
                    "selected_minus_fixed0p5_mean": statistics.mean(
                        row["selected_minus_fixed0p5"] for row in members
                    ),
                    "selected_minus_forget0_mean": statistics.mean(
                        row["selected_minus_forget0"] for row in members
                    ),
                    "selected_rate_mode": min(
                        (rate for rate, count in rate_counts.items() if count == max(rate_counts.values())),
                        default=math.nan,
                    ),
                    "selected_rate_counts": json.dumps(rate_counts, sort_keys=True),
                }
            )
    return output


def _fiveseed_gates(aggregate):
    oracle_deltas = [row["selected_minus_oracle_mean"] for row in aggregate]
    fixed_deltas = [row["selected_minus_fixed0p5_mean"] for row in aggregate]
    oracle_reasons = []
    if sum(value >= -0.01 for value in oracle_deltas) < math.ceil(0.8 * len(oracle_deltas)):
        oracle_reasons.append("fewer_than_80pct_setting_means_within_1pp")
    if statistics.mean(oracle_deltas) < -0.01:
        oracle_reasons.append("global_mean_below_minus_1pp")
    if min(oracle_deltas) < -0.02:
        oracle_reasons.append("setting_mean_regression_over_2pp")
    fixed_reasons = []
    if sum(value >= 0 for value in fixed_deltas) < math.ceil(0.6 * len(fixed_deltas)):
        fixed_reasons.append("fewer_than_60pct_nonnegative_vs_fixed0p5")
    if statistics.mean(fixed_deltas) < 0.005:
        fixed_reasons.append("global_gain_vs_fixed0p5_below_0p5pp")
    return {
        "oracle_substitution": {
            "pass": not oracle_reasons,
            "within_1pp": sum(value >= -0.01 for value in oracle_deltas),
            "n_settings": len(oracle_deltas),
            "global_mean": statistics.mean(oracle_deltas),
            "minimum": min(oracle_deltas),
            "reasons": oracle_reasons,
        },
        "fixed_default_value": {
            "pass": not fixed_reasons,
            "nonnegative_settings": sum(value >= 0 for value in fixed_deltas),
            "n_settings": len(fixed_deltas),
            "global_mean": statistics.mean(fixed_deltas),
            "reasons": fixed_reasons,
        },
    }


def build_cleanval_summary(
    rows,
    *,
    expected_datasets=DEFAULT_DATASETS,
    expected_noise_rates=DEFAULT_NOISE_RATES,
    expected_seeds=(0,),
    candidate_rates=DEFAULT_CANDIDATE_RATES,
    fixed_rate=DEFAULT_FIXED_RATE,
):
    candidates, keyed = validate_candidate_rows(
        rows,
        expected_datasets=tuple(expected_datasets),
        expected_noise_rates=tuple(_rate(v) for v in expected_noise_rates),
        expected_seeds=tuple(int(v) for v in expected_seeds),
        candidate_rates=tuple(_rate(v) for v in candidate_rates),
        fixed_rate=_rate(fixed_rate),
    )
    selected = []
    for dataset in expected_datasets:
        for noise_rate in expected_noise_rates:
            for seed in expected_seeds:
                setting_rows = [
                    keyed[(dataset, _rate(noise_rate), int(seed), _rate(forget_rate))]
                    for forget_rate in tuple(candidate_rates) + (fixed_rate,)
                ]
                choice = select_by_clean_validation(setting_rows, candidate_rates)
                oracle = keyed[(dataset, _rate(noise_rate), int(seed), _rate(noise_rate))]
                fixed = keyed[(dataset, _rate(noise_rate), int(seed), _rate(fixed_rate))]
                forget0 = keyed[(dataset, _rate(noise_rate), int(seed), 0.0)]
                selected.append(
                    {
                        "dataset": dataset,
                        "noise_rate": _rate(noise_rate),
                        "seed": int(seed),
                        "split_hash": choice["split_hash"],
                        "noise_hash": choice["noise_hash"],
                        "selected_forget_rate": _rate(choice["requested_forget_rate"]),
                        "selected_valid": float(choice["valid"]),
                        "selected_test": float(choice["test"]),
                        "oracle_forget_rate": _rate(noise_rate),
                        "oracle_valid": float(oracle["valid"]),
                        "oracle_test": float(oracle["test"]),
                        "fixed0p5_valid": float(fixed["valid"]),
                        "fixed0p5_test": float(fixed["test"]),
                        "forget0_valid": float(forget0["valid"]),
                        "forget0_test": float(forget0["test"]),
                        "selected_minus_oracle": float(choice["test"]) - float(oracle["test"]),
                        "selected_minus_fixed0p5": float(choice["test"]) - float(fixed["test"]),
                        "selected_minus_forget0": float(choice["test"]) - float(forget0["test"]),
                    }
                )
    aggregate = _aggregate(selected, tuple(expected_datasets), tuple(expected_noise_rates))
    seed0_rows = [row for row in selected if row["seed"] == 0]
    gates = {"seed0": _seed0_gate(seed0_rows)}
    if len(tuple(expected_seeds)) == 5:
        gates["fiveseed"] = _fiveseed_gates(aggregate)
    return candidates, selected, aggregate, gates


def _report(path, selected, aggregate, gates, lcss_audit=None):
    lines = [
        "# MLP-35 Clean-Validation Forget-Rate Selection",
        "",
        "> The selected rate uses clean validation accuracy only. Test accuracy is read after selection.",
        "",
        "## Per-seed selections",
        "",
        "| Dataset | Noise | Seed | Selected rate | Selected | Oracle | Fixed 0.5 | Forget 0 | Sel-Oracle | Sel-Fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']:.1f} | {row['seed']} | "
            f"{row['selected_forget_rate']:.1f} | {100*row['selected_test']:.2f} | "
            f"{100*row['oracle_test']:.2f} | {100*row['fixed0p5_test']:.2f} | "
            f"{100*row['forget0_test']:.2f} | {100*row['selected_minus_oracle']:+.2f}pp | "
            f"{100*row['selected_minus_fixed0p5']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Dataset/noise aggregates",
            "",
            "| Dataset | Noise | n | Selected | Oracle | Fixed 0.5 | Sel-Oracle | Sel-Fixed | Rate counts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']:.1f} | {row['n']} | "
            f"{100*row['selected_test_mean']:.2f}+/-{100*row['selected_test_std']:.2f} | "
            f"{100*row['oracle_test_mean']:.2f}+/-{100*row['oracle_test_std']:.2f} | "
            f"{100*row['fixed0p5_test_mean']:.2f}+/-{100*row['fixed0p5_test_std']:.2f} | "
            f"{100*row['selected_minus_oracle_mean']:+.2f}pp | "
            f"{100*row['selected_minus_fixed0p5_mean']:+.2f}pp | {row['selected_rate_counts']} |"
        )
    counts_by_true_rate = defaultdict(Counter)
    exact_rate = 0
    for row in selected:
        counts_by_true_rate[row["noise_rate"]][row["selected_forget_rate"]] += 1
        exact_rate += int(abs(row["noise_rate"] - row["selected_forget_rate"]) < 1e-12)
    worst = min(selected, key=lambda row: row["selected_minus_oracle"])
    lines.extend(
        [
            "",
            "## Selection diagnostics",
            "",
            f"- Exact true-rate selections: {exact_rate}/{len(selected)}. This is a training-strength selector, not a noise-rate estimator.",
            f"- Worst per-seed selected-oracle delta: {100*worst['selected_minus_oracle']:+.2f}pp "
            f"({worst['dataset']} u{worst['noise_rate']:.1f} seed {worst['seed']}, selected={worst['selected_forget_rate']:.1f}).",
        ]
    )
    for true_rate in sorted(counts_by_true_rate):
        lines.append(
            f"- True rate {true_rate:.1f}: selected counts "
            f"{json.dumps(counts_by_true_rate[true_rate], sort_keys=True)}."
        )
    if lcss_audit:
        max_abs = max(abs(row["forget0_minus_lcss"]) for row in lcss_audit)
        exact = sum(abs(row["forget0_minus_lcss"]) < 1e-12 for row in lcss_audit)
        lines.extend(
            [
                "",
                "## Forget-0 versus historical LCSS audit",
                "",
                f"All {len(lcss_audit)} rows match split/noise hashes; {exact}/{len(lcss_audit)} test values are exact; "
                f"maximum absolute rerun delta is {100*max_abs:.2f}pp. Forget-0 is therefore reported as the "
                "same-mechanism zero-selection arm, not as bitwise-identical historical LCSS.",
            ]
        )
    lines.extend(["", "## Gates", "", "```json", json.dumps(gates, indent=2), "```", ""])
    if "fiveseed" in gates:
        oracle_pass = gates["fiveseed"]["oracle_substitution"]["pass"]
        fixed_pass = gates["fiveseed"]["fixed_default_value"]["pass"]
        if oracle_pass and fixed_pass:
            decision = "continue_clean_val_tuned_variant"
        elif oracle_pass:
            decision = "retain_as_oracle_rate_relaxation_only"
        else:
            decision = "abandon_clean_val_selection"
        lines.extend(
            [
                "## Decision",
                "",
                f"`{decision}`. The supported claim is conditional on access to a clean validation set; "
                "the method is not fully noise-rate-free.",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--noise_rates", default="0.2,0.4,0.6")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--lcss_anchor")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(_read(path))
    datasets = _parse_list(args.datasets)
    noise_rates = _parse_list(args.noise_rates, float)
    seeds = _parse_list(args.seeds, int)
    candidates, selected, aggregate, gates = build_cleanval_summary(
        rows,
        expected_datasets=datasets,
        expected_noise_rates=noise_rates,
        expected_seeds=seeds,
    )
    output = Path(args.output_dir)
    _write(output / "mlp35_cleanval_candidates.csv", candidates)
    _write(output / "mlp35_cleanval_selected.csv", selected)
    _write(output / "mlp35_cleanval_aggregate.csv", aggregate)
    (output / "mlp35_cleanval_gate.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    lcss_audit = None
    if args.lcss_anchor:
        lcss_audit = audit_forget0_against_lcss(candidates, _read(args.lcss_anchor))
        _write(output / "mlp35_forget0_lcss_regression.csv", lcss_audit)
    _report(output / "mlp35_report.md", selected, aggregate, gates, lcss_audit)


if __name__ == "__main__":
    main()
