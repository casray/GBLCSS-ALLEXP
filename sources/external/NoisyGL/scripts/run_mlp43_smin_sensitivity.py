from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


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

from scripts import run_ssl_swap_downstream as ssl_runner


DEFAULT_DATASETS = "cora,citeseer,corafull,coauthorcs,amazonpho"
FROZEN_SMIN_GRID = (2, 3, 5, 10)


def _parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_smins(value: str) -> list[int]:
    requested = []
    for item in str(value).split(","):
        if not item.strip():
            continue
        parsed = int(item.strip())
        if parsed not in FROZEN_SMIN_GRID:
            raise ValueError(
                f"s_min must be one of {FROZEN_SMIN_GRID}, got {parsed}"
            )
        if parsed not in requested:
            requested.append(parsed)
    return requested


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def _anchor_rows(
    main_grid_rows: list[dict], datasets: set[str], seeds: set[int]
) -> list[dict]:
    rows = []
    for source in main_grid_rows:
        if source.get("method") != "frozengrace_gb_mlp_softdist_beta05":
            continue
        if source.get("dataset", "").lower() not in datasets:
            continue
        if int(source.get("seed", -1)) not in seeds:
            continue
        if source.get("noise_type") != "uniform":
            continue
        if abs(float(source.get("noise_rate", -1)) - 0.4) > 1e-12:
            continue
        if source.get("noise_scope") != "train":
            continue
        row = dict(source)
        row.update(
            {
                "s_min": 2,
                "min_train_per_ball": 2,
                "source": "immutable_tie_safe_v2_main_grid",
            }
        )
        rows.append(row)
    return rows


def _runner_args(
    *,
    embedding_root: Path,
    device: str,
    s_min: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_root=str(embedding_root),
        ball_embedding_root=None,
        feature_ssl_method=None,
        noise_type="uniform",
        noise_rate=0.4,
        noise_scope="train",
        device=device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        gb_max_balls=None,
        gb_max_balls_policy=None,
        gb_min_ball_size=None,
        gb_split_improvement=None,
        gb_keep_ratio=None,
        gb_lambda_u=None,
        config_overrides={
            "gb.tie_mode": "distribution_only",
            "gb.min_train_per_ball": int(s_min),
        },
        debug=False,
    )


def _aggregate(raw_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    by_seed: dict[tuple[str, int, int], dict] = {}
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        dataset = row["dataset"].lower()
        s_min = int(row["s_min"])
        seed = int(row["seed"])
        grouped.setdefault((dataset, s_min), []).append(row)
        by_seed[(dataset, s_min, seed)] = row

    aggregates = []
    for (dataset, s_min), rows in sorted(grouped.items()):
        tests = [float(row["test"]) for row in rows]
        coverage = []
        for row in rows:
            explicit = row.get("alpha_positive_train_ratio", "")
            if explicit not in {"", None}:
                coverage.append(float(explicit))
                continue
            train_count = 0.0
            modified_train = float(row.get("modified_train_count", 0.0) or 0.0)
            actual_rate = float(row.get("actual_train_noise_rate", 0.0) or 0.0)
            if actual_rate > 0.0:
                train_count = modified_train / actual_rate
            positive = float(row.get("alpha_positive_train_count", 0.0) or 0.0)
            coverage.append(positive / train_count if train_count > 0 else 0.0)
        aggregates.append(
            {
                "dataset": dataset,
                "s_min": s_min,
                "n_seeds": len(rows),
                "test_mean": statistics.fmean(tests),
                "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
                "target_coverage_mean": statistics.fmean(coverage),
                "target_coverage_std": (
                    statistics.stdev(coverage) if len(coverage) > 1 else 0.0
                ),
            }
        )

    deltas = []
    datasets = sorted({key[0] for key in by_seed})
    smins = sorted({key[1] for key in by_seed if key[1] != 2})
    seeds = sorted({key[2] for key in by_seed})
    for dataset in datasets:
        for s_min in smins:
            paired = []
            for seed in seeds:
                anchor = by_seed.get((dataset, 2, seed))
                candidate = by_seed.get((dataset, s_min, seed))
                if anchor is None or candidate is None:
                    continue
                delta = float(candidate["test"]) - float(anchor["test"])
                paired.append(delta)
                deltas.append(
                    {
                        "dataset": dataset,
                        "s_min": s_min,
                        "seed": seed,
                        "test_delta_vs_smin2": delta,
                        "split_hash_match": int(
                            candidate.get("split_hash") == anchor.get("split_hash")
                        ),
                        "noise_hash_match": int(
                            candidate.get("noise_hash") == anchor.get("noise_hash")
                        ),
                    }
                )
            if paired:
                deltas.append(
                    {
                        "dataset": dataset,
                        "s_min": s_min,
                        "seed": "aggregate",
                        "test_delta_vs_smin2": statistics.fmean(paired),
                        "positive_seed_count": sum(value > 0 for value in paired),
                        "paired_seed_count": len(paired),
                        "split_hash_match": int(
                            all(
                                row["split_hash_match"] == 1
                                for row in deltas
                                if row["dataset"] == dataset
                                and row["s_min"] == s_min
                                and row["seed"] != "aggregate"
                            )
                        ),
                        "noise_hash_match": int(
                            all(
                                row["noise_hash_match"] == 1
                                for row in deltas
                                if row["dataset"] == dataset
                                and row["s_min"] == s_min
                                and row["seed"] != "aggregate"
                            )
                        ),
                    }
                )
    return aggregates, deltas


def _write_report(path: Path, aggregates: list[dict], deltas: list[dict]) -> None:
    delta_index = {
        (row["dataset"], int(row["s_min"])): row
        for row in deltas
        if row["seed"] == "aggregate"
    }
    lines = [
        "# MLP-43 Minimum-Vote Sensitivity",
        "",
        "Only the minimum number of noisy training votes required to activate a "
        "ball target is changed. The main method remains `s_min=2`.",
        "",
        "| Dataset | s_min | Accuracy | Target coverage | Delta vs 2 | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        key = (row["dataset"], int(row["s_min"]))
        delta = delta_index.get(key)
        delta_text = (
            f"{100 * float(delta['test_delta_vs_smin2']):+.2f}"
            if delta is not None
            else "0.00"
        )
        seed_text = (
            f"{delta.get('positive_seed_count', 0)}/{delta.get('paired_seed_count', 0)}"
            if delta is not None
            else "-"
        )
        lines.append(
            f"| {row['dataset']} | {row['s_min']} | "
            f"{100 * float(row['test_mean']):.2f} +/- "
            f"{100 * float(row['test_std']):.2f} | "
            f"{100 * float(row['target_coverage_mean']):.2f}% | "
            f"{delta_text} | {seed_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen s_min sensitivity for tie-safe GB-LCSS."
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--smins", default="2,3,5,10")
    parser.add_argument("--device", default="cuda:0")
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

    datasets = _parse_list(args.datasets)
    seeds = _parse_seeds(args.seeds)
    smins = parse_smins(args.smins)
    main_grid_rows = _read_csv(Path(args.main_grid))
    anchors = _anchor_rows(main_grid_rows, set(datasets), set(seeds))
    expected_anchors = len(datasets) * len(seeds)
    if len(anchors) != expected_anchors:
        raise ValueError(
            f"expected {expected_anchors} s_min=2 anchors, found {len(anchors)}"
        )
    embedding_paths = [
        Path(row.get("ball_embedding_path") or row["embedding_path"])
        for row in anchors
    ]
    embedding_roots = {path.parents[2] for path in embedding_paths}
    if len(embedding_roots) != 1:
        raise ValueError(f"multiple embedding roots found: {embedding_roots}")
    embedding_root = next(iter(embedding_roots))

    run_dir = Path(args.run_dir)
    raw_path = run_dir / "smin_raw.csv"
    existing = _read_csv(raw_path)
    rows_by_key = {
        (row["dataset"].lower(), int(row["s_min"]), int(row["seed"])): row
        for row in existing
    }
    for anchor in anchors:
        key = (anchor["dataset"].lower(), 2, int(anchor["seed"]))
        rows_by_key[key] = anchor

    for dataset in datasets:
        for s_min in smins:
            if s_min == 2:
                continue
            runner_args = _runner_args(
                embedding_root=embedding_root,
                device=args.device,
                s_min=s_min,
            )
            for seed in seeds:
                key = (dataset, s_min, seed)
                if key in rows_by_key and rows_by_key[key].get("status") == "ok":
                    print(f"SKIP dataset={dataset} s_min={s_min} seed={seed}")
                    continue
                print(f"SMIN dataset={dataset} s_min={s_min} seed={seed}")
                row = ssl_runner.run_one(
                    runner_args,
                    dataset_name=dataset,
                    ssl_method="grace",
                    base_method="frozengrace_gb_mlp_softdist_beta05",
                    seed=seed,
                )
                row.update({"s_min": s_min, "source": "mlp43_new_run"})
                anchor = rows_by_key[(dataset, 2, seed)]
                if row.get("split_hash") != anchor.get("split_hash"):
                    raise ValueError(f"split hash mismatch for {key}")
                if row.get("noise_hash") != anchor.get("noise_hash"):
                    raise ValueError(f"noise hash mismatch for {key}")
                if int(float(row.get("modified_val_count", 0) or 0)) != 0:
                    raise ValueError(f"validation label leakage for {key}")
                if int(float(row.get("modified_test_count", 0) or 0)) != 0:
                    raise ValueError(f"test label leakage for {key}")
                rows_by_key[key] = row
                ordered = [
                    rows_by_key[item]
                    for item in sorted(rows_by_key)
                    if item[0] in datasets and item[1] in smins and item[2] in seeds
                ]
                _write_csv(raw_path, ordered)

    raw_rows = [
        rows_by_key[item]
        for item in sorted(rows_by_key)
        if item[0] in datasets and item[1] in smins and item[2] in seeds
    ]
    aggregates, deltas = _aggregate(raw_rows)
    _write_csv(raw_path, raw_rows)
    _write_csv(run_dir / "smin_aggregate.csv", aggregates)
    _write_csv(run_dir / "smin_paired_deltas.csv", deltas)
    _write_report(run_dir / "smin_report.md", aggregates, deltas)
    print(
        json.dumps(
            {
                "rows": len(raw_rows),
                "datasets": datasets,
                "seeds": seeds,
                "smins": smins,
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
