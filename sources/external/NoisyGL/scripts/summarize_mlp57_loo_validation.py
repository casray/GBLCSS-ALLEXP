from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DATASETS = ("cora", "citeseer", "corafull")
SEEDS = tuple(range(5))
OURS = "frozengrace_gb_mlp_softdist_beta05_loo"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_loo_shuffled"
CONTROLS = ("mlp_ce", "mlp_jocor", "postel_mlp_noipl_rm", "gsdn")
METHODS = (OURS, SHUFFLED) + CONTROLS


def kendall_tau(
    order_a: Sequence[str],
    order_b: Sequence[str],
    universe: Sequence[str] = METHODS,
) -> float:
    position_a = {value: index for index, value in enumerate(order_a)}
    position_b = {value: index for index, value in enumerate(order_b)}
    concordant = 0
    discordant = 0
    for left_index, left in enumerate(universe):
        for right in universe[left_index + 1 :]:
            product = (position_a[left] - position_a[right]) * (
                position_b[left] - position_b[right]
            )
            concordant += int(product > 0)
            discordant += int(product < 0)
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0


def _validate_new(path: Path, protocol: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame.method.isin((OURS, SHUFFLED))
        & frame.dataset.isin(DATASETS)
        & frame.seed.isin(SEEDS)
    ].copy()
    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in (OURS, SHUFFLED)
        for seed in SEEDS
    }
    actual = set(zip(selected.dataset, selected.method, selected.seed))
    if len(selected) != 30 or actual != expected:
        raise ValueError(f"incomplete {protocol} LOO grid: {len(selected)} rows")
    if not (selected.status == "ok").all():
        raise ValueError(f"non-ok row in {protocol}")
    if not (selected.tie_mode == "distribution_only").all():
        raise ValueError(f"non-tie-safe row in {protocol}")
    if not (pd.to_numeric(selected.leave_one_out) == 1).all():
        raise ValueError(f"non-LOO row in {protocol}")
    if not (pd.to_numeric(selected.modified_test_count) == 0).all():
        raise ValueError(f"test-label leakage in {protocol}")
    expected_scope = "train" if protocol == "clean_validation" else "train_val"
    if not (selected.noise_scope == expected_scope).all():
        raise ValueError(f"wrong noise scope in {protocol}")
    if protocol == "clean_validation":
        if not (pd.to_numeric(selected.modified_val_count) == 0).all():
            raise ValueError("clean-validation rows contain validation noise")
    elif not (pd.to_numeric(selected.modified_val_count) > 0).all():
        raise ValueError("noisy-validation rows do not contain validation noise")
    selected["protocol"] = protocol
    return selected


def _controls(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame.method.isin(CONTROLS)
        & frame.dataset.isin(DATASETS)
        & frame.seed.isin(SEEDS)
        & frame.protocol.isin(("clean_validation", "noisy_validation"))
    ].copy()
    expected = {
        (protocol, dataset, method, seed)
        for protocol in ("clean_validation", "noisy_validation")
        for dataset in DATASETS
        for method in CONTROLS
        for seed in SEEDS
    }
    actual = set(zip(selected.protocol, selected.dataset, selected.method, selected.seed))
    if len(selected) != 120 or actual != expected:
        raise ValueError(f"incomplete legacy control grid: {len(selected)} rows")
    return selected


def summarize(clean_path: Path, noisy_path: Path, controls_path: Path, output: Path) -> None:
    clean = _validate_new(clean_path, "clean_validation")
    noisy = _validate_new(noisy_path, "noisy_validation")
    controls = _controls(controls_path)
    combined = pd.concat([clean, noisy, controls], ignore_index=True, sort=False)
    if len(combined) != 180:
        raise ValueError(f"expected 180 rows, found {len(combined)}")
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "loo_validation_combined.csv", index=False)

    aggregate = (
        combined.groupby(["protocol", "dataset", "method"], sort=True)["test"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.to_csv(output / "loo_validation_aggregate.csv", index=False)

    lookup = combined.set_index(["protocol", "dataset", "method", "seed"])["test"]
    paired_rows = []
    for protocol in ("clean_validation", "noisy_validation"):
        for dataset in DATASETS:
            for baseline in (SHUFFLED,) + CONTROLS:
                for seed in SEEDS:
                    delta = float(lookup.loc[(protocol, dataset, OURS, seed)]) - float(
                        lookup.loc[(protocol, dataset, baseline, seed)]
                    )
                    paired_rows.append(
                        {
                            "protocol": protocol,
                            "dataset": dataset,
                            "baseline": baseline,
                            "seed": seed,
                            "delta": delta,
                            "delta_pp": 100 * delta,
                        }
                    )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "loo_validation_paired_deltas.csv", index=False)
    paired_summary = (
        paired.groupby(["protocol", "dataset", "baseline"], sort=True)["delta"]
        .agg(
            mean_delta="mean",
            std_delta="std",
            positive_seeds=lambda values: int((values > 0).sum()),
            n="count",
        )
        .reset_index()
    )
    paired_summary["mean_delta_pp"] = 100 * paired_summary.mean_delta
    paired_summary.to_csv(output / "loo_validation_paired_summary.csv", index=False)

    interactions = []
    means = aggregate.set_index(["protocol", "dataset", "method"])["mean"]
    for dataset in DATASETS:
        clean_order = tuple(
            sorted(
                METHODS,
                key=lambda method: means.loc[("clean_validation", dataset, method)],
                reverse=True,
            )
        )
        noisy_order = tuple(
            sorted(
                METHODS,
                key=lambda method: means.loc[("noisy_validation", dataset, method)],
                reverse=True,
            )
        )
        interactions.append(
            {
                "dataset": dataset,
                "kendall_tau": kendall_tau(clean_order, noisy_order),
                "top2_overlap": len(set(clean_order[:2]) & set(noisy_order[:2])),
                "clean_order": ">".join(clean_order),
                "noisy_order": ">".join(noisy_order),
            }
        )
    interactions = pd.DataFrame(interactions)
    interactions.to_csv(output / "loo_validation_interactions.csv", index=False)

    noisy_paired = paired_summary[paired_summary.protocol == "noisy_validation"]
    shuffled = noisy_paired[noisy_paired.baseline == SHUFFLED]
    ce = noisy_paired[noisy_paired.baseline == "mlp_ce"]
    gate = bool(
        (shuffled.mean_delta_pp > 0).all()
        and (shuffled.positive_seeds >= 4).all()
        and int(((ce.mean_delta_pp > 0) & (ce.positive_seeds >= 4)).sum()) >= 2
    )
    report = [
        "# Formal LOO validation-protocol sensitivity",
        "",
        "Both real and shuffled GB targets use distribution_only and leave_one_out=true.",
        f"Completeness: {len(combined)}/180 rows. Test-label contamination: 0. Gate passed: {gate}.",
        "",
        "## LOO paired results",
        "",
    ]
    for row in paired_summary.itertuples(index=False):
        report.append(
            f"- {row.protocol} {row.dataset} vs {row.baseline}: "
            f"{row.mean_delta_pp:+.2f}pp, {row.positive_seeds}/{row.n} positive seeds."
        )
    report.extend(["", "## Ranking sensitivity", ""])
    for row in interactions.itertuples(index=False):
        report.append(
            f"- {row.dataset}: Kendall tau={row.kendall_tau:.3f}, "
            f"top-2 overlap={row.top2_overlap}/2."
        )
    (output / "loo_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--noisy", type=Path, required=True)
    parser.add_argument("--legacy-controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args.clean, args.noisy, args.legacy_controls, args.output_dir)


if __name__ == "__main__":
    main()
