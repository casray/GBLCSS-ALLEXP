from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path


for _name, _value in {
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}.items():
    os.environ[_name] = _value


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _split_fingerprint,
)
from utils.frozen_grace_targets import (
    build_frozen_grace_soft_targets,
    vote_support_diagnostics,
)
from utils.tools import setup_seed


DEFAULT_DATASETS = "cora,citeseer,corafull,coauthorcs,amazonpho"


def _parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _main_grid_index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    selected = {}
    for row in rows:
        if row.get("method") != "frozengrace_gb_mlp_softdist_beta05":
            continue
        if row.get("noise_type") != "uniform":
            continue
        if abs(float(row.get("noise_rate", -1)) - 0.4) > 1e-12:
            continue
        if row.get("noise_scope") != "train":
            continue
        key = (row["dataset"].lower(), int(row["seed"]))
        if key in selected:
            raise ValueError(f"duplicate main-grid row: {key}")
        selected[key] = row
    return selected


def _aggregate(raw_rows: list[dict]) -> list[dict]:
    metric_names = [
        "ball_count",
        "target_node_count",
        "target_node_coverage",
        "changed_train_count",
        "noisy_train_count",
        "clean_train_count",
        "ncr",
        "cdr",
        "mean_alpha",
        "clean_mass_delta_per_node",
        "majority_clean_accuracy",
        "mean_observed_margin",
        "soft_target_argmax_accuracy",
    ]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in raw_rows:
        grouped.setdefault((row["dataset"], row["support_bin"]), []).append(row)

    aggregates = []
    for (dataset, support_bin), rows in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "support_bin": support_bin,
            "n_seeds": len(rows),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            out[f"{metric}_mean"] = statistics.fmean(values)
            out[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregates.append(out)
    return aggregates


def diagnose(args, dataset: str, seed: int, anchor: dict) -> list[dict]:
    setup_seed(seed)
    data = _load_dataset(
        dataset,
        device=args.device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    noisy_label, modified_mask = _build_noisy_labels(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        seed=seed,
        noise_scope="train",
        train_mask=data.train_masks,
        val_mask=data.val_masks,
    )
    split_hash = _split_fingerprint(
        data.train_masks, data.val_masks, data.test_masks
    )
    noise_hash = _noise_fingerprint(data.labels, noisy_label)
    if split_hash != anchor["split_hash"]:
        raise ValueError(
            f"split hash mismatch for {dataset} seed {seed}: "
            f"{split_hash} != {anchor['split_hash']}"
        )
    if noise_hash != anchor["noise_hash"]:
        raise ValueError(
            f"noise hash mismatch for {dataset} seed {seed}: "
            f"{noise_hash} != {anchor['noise_hash']}"
        )

    embedding_path = Path(
        anchor.get("ball_embedding_path")
        or anchor.get("feature_embedding_path")
        or anchor.get("embedding_path")
    )
    result = build_frozen_grace_soft_targets(
        embedding_path=embedding_path,
        noisy_label=noisy_label,
        clean_label=data.labels,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
        random_state=seed,
    )
    if result.diagnostics.get("ball_assignment_hash") != anchor.get(
        "ball_assignment_hash"
    ):
        raise ValueError(
            f"ball assignment hash mismatch for {dataset} seed {seed}: "
            f"{result.diagnostics.get('ball_assignment_hash')} != "
            f"{anchor.get('ball_assignment_hash')}"
        )

    train_set = set(int(value) for value in data.train_masks)
    modified_set = set(int(value) for value in modified_mask)
    if not modified_set.issubset(train_set):
        raise ValueError(f"non-train label modification detected for {dataset}")

    rows = vote_support_diagnostics(
        noisy_label=noisy_label,
        clean_label=data.labels,
        assignment=result.assignment,
        train_mask=data.train_masks,
        n_classes=data.n_classes,
        min_train_per_ball=2,
        alpha_baseline="multiclass",
        target_mode="distribution",
        target_beta=0.5,
        tie_mode="distribution_only",
    )
    for row in rows:
        row.update(
            {
                "dataset": dataset,
                "seed": seed,
                "noise_type": "uniform",
                "noise_rate": 0.4,
                "noise_scope": "train",
                "split_hash": split_hash,
                "noise_hash": noise_hash,
                "embedding_path": str(embedding_path),
                "ball_assignment_hash": result.diagnostics.get(
                    "ball_assignment_hash", ""
                ),
                "tie_mode": "distribution_only",
                "min_train_per_ball": 2,
                "modified_count": len(modified_set),
                "modified_train_count": len(modified_set),
                "modified_val_count": 0,
                "modified_test_count": 0,
            }
        )
    return rows


def _write_report(path: Path, aggregates: list[dict]) -> None:
    lines = [
        "# MLP-43 Vote-Support Diagnostic",
        "",
        "Protocol: uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, "
        "tie-safe `distribution_only` targets.",
        "",
        "| Dataset | Votes | Node coverage | Mean alpha | NCR | CDR | Target accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['dataset']} | {row['support_bin']} | "
            f"{100 * float(row['target_node_coverage_mean']):.2f}% | "
            f"{float(row['mean_alpha_mean']):.3f} | "
            f"{float(row['ncr_mean']):.3f} | "
            f"{float(row['cdr_mean']):.3f} | "
            f"{100 * float(row['soft_target_argmax_accuracy_mean']):.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit GB-LCSS target reliability by effective vote count."
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--main_grid",
        default=str(
            REPO_ROOT
            / "runs"
            / "tie_safe_full_grid_20260727_v2_immutable"
            / "raw_combined.csv"
        ),
    )
    parser.add_argument(
        "--run_dir",
        default=str(REPO_ROOT / "runs" / "mlp43_cold_review_defense_20260729"),
    )
    args = parser.parse_args()

    main_grid = _main_grid_index(_read_csv(Path(args.main_grid)))
    raw_rows = []
    for dataset in _parse_list(args.datasets):
        for seed in _parse_seeds(args.seeds):
            key = (dataset, seed)
            if key not in main_grid:
                raise KeyError(f"missing immutable main-grid anchor: {key}")
            print(f"VOTE_SUPPORT dataset={dataset} seed={seed}")
            raw_rows.extend(diagnose(args, dataset, seed, main_grid[key]))

    run_dir = Path(args.run_dir)
    aggregates = _aggregate(raw_rows)
    _write_csv(run_dir / "vote_support_raw.csv", raw_rows)
    _write_csv(run_dir / "vote_support_aggregate.csv", aggregates)
    _write_report(run_dir / "vote_support_report.md", aggregates)
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
