from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp43_cold_review_defense_20260729"
DEFAULT_MAIN_GRID = (
    REPO_ROOT / "runs" / "tie_safe_full_grid_20260727_v2_immutable" / "raw_combined.csv"
)
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
SEEDS = tuple(range(5))
GB_METHOD = "frozengrace_gb_mlp_softdist_beta05"
ALTERNATIVE_METHODS = (
    "knn_mlp_softdist_beta05",
    "kmeans_mlp_softdist_beta05",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing numeric field {key!r}: {row}")
    return float(value)


def _validate_protocol(row: dict[str, str]) -> None:
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        raise ValueError(f"Non-success row: {row}")
    if row.get("noise_type", "").strip().lower() != "uniform":
        raise ValueError(f"Unexpected noise type: {row}")
    if abs(_number(row, "noise_rate") - 0.4) > 1e-12:
        raise ValueError(f"Unexpected noise rate: {row}")
    if row.get("noise_scope", "").strip().lower() != "train":
        raise ValueError(f"Unexpected noise scope: {row}")
    if int(_number(row, "modified_val_count")) != 0:
        raise ValueError(f"Validation labels were modified: {row}")
    if int(_number(row, "modified_test_count")) != 0:
        raise ValueError(f"Test labels were modified: {row}")
    if int(_number(row, "modified_count")) != int(_number(row, "modified_train_count")):
        raise ValueError(f"Noise count is not train-only: {row}")


def _load_alternatives(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    selected: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in read_csv(path):
        dataset = row.get("dataset", "").strip().lower()
        method = row.get("method", "").strip()
        if dataset not in DATASETS or method not in ALTERNATIVE_METHODS:
            continue
        _validate_protocol(row)
        seed = int(_number(row, "seed"))
        if seed not in SEEDS:
            continue
        if row.get("tie_mode", "").strip() != "distribution_only":
            raise ValueError(f"Alternative is not tie-safe: {row}")
        if method == "knn_mlp_softdist_beta05" and row.get(
            "knn_exclude_self", ""
        ).strip() not in {"1", "true", "True"}:
            raise ValueError(f"kNN query node was not excluded: {row}")
        key = (dataset, method, seed)
        if key in selected:
            raise ValueError(f"Duplicate alternative row: {key}")
        selected[key] = row
    expected = len(DATASETS) * len(ALTERNATIVE_METHODS) * len(SEEDS)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} alternative rows, found {len(selected)}")
    return selected


def _load_gb(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in read_csv(path):
        dataset = row.get("dataset", "").strip().lower()
        method = row.get("method", "").strip()
        if dataset not in DATASETS or method != GB_METHOD:
            continue
        if row.get("noise_type", "").strip().lower() != "uniform":
            continue
        if abs(float(row.get("noise_rate", "nan")) - 0.4) > 1e-12:
            continue
        _validate_protocol(row)
        seed = int(_number(row, "seed"))
        if seed not in SEEDS:
            continue
        if row.get("tie_mode", "").strip() != "distribution_only":
            raise ValueError(f"GB anchor is not tie-safe v2: {row}")
        key = (dataset, seed)
        if key in selected:
            raise ValueError(f"Duplicate GB anchor row: {key}")
        selected[key] = row
    expected = len(DATASETS) * len(SEEDS)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} GB rows, found {len(selected)}")
    return selected


def summarize(
    alternatives: dict[tuple[str, str, int], dict[str, str]],
    gb_rows: dict[tuple[str, int], dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    paired: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    build_times: dict[tuple[str, str], list[float]] = defaultdict(list)

    for dataset in DATASETS:
        gb_values = [_number(gb_rows[(dataset, seed)], "test") for seed in SEEDS]
        grouped[(dataset, GB_METHOD)].extend(gb_values)
        for method in ALTERNATIVE_METHODS:
            for seed in SEEDS:
                alt = alternatives[(dataset, method, seed)]
                gb = gb_rows[(dataset, seed)]
                if alt.get("split_hash", "") != gb.get("split_hash", ""):
                    raise ValueError(f"Split hash mismatch for {dataset}/{method}/seed{seed}")
                if alt.get("noise_hash", "") != gb.get("noise_hash", ""):
                    raise ValueError(f"Noise hash mismatch for {dataset}/{method}/seed{seed}")
                alt_test = _number(alt, "test")
                gb_test = _number(gb, "test")
                grouped[(dataset, method)].append(alt_test)
                build_times[(dataset, method)].append(
                    _number(alt, "ball_build_time_seconds")
                )
                paired.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "alternative_test": alt_test,
                        "gb_test": gb_test,
                        "delta_alternative_minus_gb": alt_test - gb_test,
                        "split_hash": alt.get("split_hash", ""),
                        "noise_hash": alt.get("noise_hash", ""),
                        "tie_mode": alt.get("tie_mode", ""),
                        "knn_exclude_self": alt.get("knn_exclude_self", ""),
                        "ball_build_time_seconds": _number(
                            alt, "ball_build_time_seconds"
                        ),
                    }
                )

    aggregate: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in (GB_METHOD, *ALTERNATIVE_METHODS):
            values = grouped[(dataset, method)]
            row: dict[str, object] = {
                "dataset": dataset,
                "method": method,
                "n": len(values),
                "test_mean": statistics.mean(values),
                "test_std": statistics.stdev(values),
            }
            if method != GB_METHOD:
                deltas = [
                    float(item["delta_alternative_minus_gb"])
                    for item in paired
                    if item["dataset"] == dataset and item["method"] == method
                ]
                row.update(
                    {
                        "delta_vs_gb_mean": statistics.mean(deltas),
                        "delta_vs_gb_std": statistics.stdev(deltas),
                        "positive_seeds_vs_gb": sum(delta > 0 for delta in deltas),
                        "mean_build_time_seconds": statistics.mean(
                            build_times[(dataset, method)]
                        ),
                    }
                )
            aggregate.append(row)
    return aggregate, paired


def write_report(path: Path, aggregate: list[dict[str, object]]) -> None:
    by_key = {(row["dataset"], row["method"]): row for row in aggregate}
    lines = [
        "# MLP-43 Corrected Grouping Alternatives",
        "",
        "Protocol: uniform 0.4, train-only noise, percent split 0.2/0.1/0.7, "
        "seeds 0-4. All kNN rows exclude the query node and all grouping "
        "alternatives use `tie_mode=distribution_only`.",
        "",
        "| Dataset | GB-LCSS | kNN | kNN-GB | k-means | k-means-GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        gb = by_key[(dataset, GB_METHOD)]
        knn = by_key[(dataset, "knn_mlp_softdist_beta05")]
        kmeans = by_key[(dataset, "kmeans_mlp_softdist_beta05")]
        lines.append(
            "| {dataset} | {gb_mean:.2f} +/- {gb_std:.2f} | "
            "{knn_mean:.2f} +/- {knn_std:.2f} | {knn_delta:+.2f} "
            "({knn_pos}/5) | {km_mean:.2f} +/- {km_std:.2f} | "
            "{km_delta:+.2f} ({km_pos}/5) |".format(
                dataset=dataset,
                gb_mean=100 * float(gb["test_mean"]),
                gb_std=100 * float(gb["test_std"]),
                knn_mean=100 * float(knn["test_mean"]),
                knn_std=100 * float(knn["test_std"]),
                knn_delta=100 * float(knn["delta_vs_gb_mean"]),
                knn_pos=int(knn["positive_seeds_vs_gb"]),
                km_mean=100 * float(kmeans["test_mean"]),
                km_std=100 * float(kmeans["test_std"]),
                km_delta=100 * float(kmeans["delta_vs_gb_mean"]),
                km_pos=int(kmeans["positive_seeds_vs_gb"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- The corrected kNN result is not explained by self-neighbor leakage.",
            "- Granular balls are not an accuracy-unique grouping mechanism in the "
            "frozen GRACE space; kNN is stronger on several datasets and k-means "
            "is competitive.",
            "- The manuscript should attribute the main effect to local-consensus "
            "soft supervision and present granular balls as a fixed-budget "
            "implementation, not as an irreplaceable accuracy source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RUN_DIR / "grouping_alternatives_5seed.csv",
    )
    parser.add_argument("--main_grid", type=Path, default=DEFAULT_MAIN_GRID)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alternatives = _load_alternatives(args.input)
    gb_rows = _load_gb(args.main_grid)
    aggregate, paired = summarize(alternatives, gb_rows)
    write_csv(args.run_dir / "grouping_alternatives_aggregate.csv", aggregate)
    write_csv(args.run_dir / "grouping_alternatives_paired_deltas.csv", paired)
    write_report(args.run_dir / "grouping_alternatives_report.md", aggregate)
    print(f"wrote {len(aggregate)} aggregate rows and {len(paired)} paired rows")


if __name__ == "__main__":
    main()
