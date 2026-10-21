from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull")
NOISE_RATES = (0.4, 0.6)
SEEDS = tuple(range(5))

GLNN = "mlp_distill_gcn"
GCN_XH = "mlp_distill_gcn_xh"
RANDOM = "mlp_distill_softdist_xh_shuffled"
KNN = "mlp_distill_knn_xh"
BALL = "mlp_distill_softdist_xh"
NRGNN = "mlp_distill_nrgnn"
NRGNN_XH = "mlp_distill_nrgnn_xh"

METHODS = (GLNN, GCN_XH, RANDOM, KNN, BALL, NRGNN, NRGNN_XH)
NEW_METHODS = (NRGNN, NRGNN_XH)
METHOD_LABELS = {
    GLNN: "GCN(X)+CE (GLNN-style)",
    GCN_XH: "GCN[X||H]+CE",
    RANDOM: "Random-group consensus teacher",
    KNN: "kNN consensus teacher (exclude self)",
    BALL: "Granular-ball consensus teacher",
    NRGNN: "NRGNN teacher",
    NRGNN_XH: "NRGNN[X||H] teacher",
}

COMPARISONS = (
    ("nrgnn_vs_glnn", NRGNN, GLNN),
    ("nrgnn_xh_vs_gcn_xh", NRGNN_XH, GCN_XH),
    ("nrgnn_vs_ball", NRGNN, BALL),
    ("nrgnn_xh_vs_ball", NRGNN_XH, BALL),
    ("nrgnn_vs_knn", NRGNN, KNN),
    ("nrgnn_xh_vs_knn", NRGNN_XH, KNN),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    return default if value is None or value == "" else int(float(value))


def _as_float(
    row: dict[str, str],
    key: str,
    default: float | None = None,
) -> float | None:
    value = row.get(key)
    return default if value is None or value == "" else float(value)


def validate_new_row(row: dict[str, str]) -> None:
    if row.get("status") != "ok":
        raise ValueError(f"non-ok new row: {row.get('dataset')} {row.get('method')}")
    if row.get("dataset") not in DATASETS or row.get("method") not in NEW_METHODS:
        raise ValueError("unexpected dataset or method in MLP-47 row")
    if row.get("noise_type") != "uniform":
        raise ValueError("MLP-47 requires uniform noise")
    if float(row["noise_rate"]) not in NOISE_RATES:
        raise ValueError("MLP-47 requires noise rate 0.4 or 0.6")
    if row.get("noise_scope") != "train":
        raise ValueError("MLP-47 requires train-only label noise")
    if _as_int(row, "seed", -1) not in SEEDS:
        raise ValueError("MLP-47 requires seeds 0--4")
    if _as_int(row, "modified_val_count") != 0:
        raise ValueError("validation labels were modified")
    if _as_int(row, "modified_test_count") != 0:
        raise ValueError("test labels were modified")
    if _as_int(row, "modified_count") != _as_int(row, "modified_train_count"):
        raise ValueError("label modifications are not restricted to train")
    if _as_int(row, "student_inference_uses_graph", -1) != 0:
        raise ValueError("student is not graph-free at inference")
    if _as_int(row, "student_input_dim", -1) <= 0:
        raise ValueError("missing student input dimension")
    expected_xh = int(row["method"] == NRGNN_XH)
    if _as_int(row, "teacher_use_xh", -1) != expected_xh:
        raise ValueError("teacher resource identity does not match method")
    row_sum_error = _as_float(row, "teacher_probability_row_sum_max_error")
    if row_sum_error is None or row_sum_error > 1e-5:
        raise ValueError("invalid teacher probability normalization")


def normalize_new_row(row: dict[str, str], source: Path) -> dict[str, object]:
    validate_new_row(row)
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": row["noise_scope"],
        "seed": int(row["seed"]),
        "method": row["method"],
        "method_label": METHOD_LABELS[row["method"]],
        "student_test": float(row["test"]),
        "student_valid": float(row["valid"]),
        "teacher_test": float(row["teacher_best_test"]),
        "teacher_valid": float(row["teacher_best_val"]),
        "split_hash": row["split_hash"],
        "noise_hash": row["noise_hash"],
        "wall_time_seconds": float(row["wall_time_seconds"]),
        "teacher_training_seconds": float(row["teacher_training_seconds"]),
        "student_training_seconds": float(row["student_training_seconds"]),
        "student_parameter_count": _as_int(row, "student_parameter_count"),
        "teacher_parameter_count": _as_int(row, "teacher_parameter_count"),
        "teacher_use_xh": _as_int(row, "teacher_use_xh"),
        "student_input_dim": _as_int(row, "student_input_dim"),
        "teacher_input_dim": _as_int(row, "teacher_input_dim"),
        "student_inference_uses_graph": 0,
        "source": str(source),
    }


def normalize_reference_row(row: dict[str, str], source: Path) -> dict[str, object]:
    method = row.get("method", "")
    if method not in METHODS[:5]:
        raise ValueError(f"unexpected reference method: {method}")
    if row.get("status", "ok") not in {"", "ok"}:
        raise ValueError(f"non-ok reference row: {row.get('dataset')} {method}")
    if row.get("dataset") not in DATASETS:
        raise ValueError(f"unexpected reference dataset: {row.get('dataset')}")
    if row.get("noise_type") != "uniform":
        raise ValueError("reference row does not use uniform noise")
    if float(row["noise_rate"]) not in NOISE_RATES:
        raise ValueError("reference row has unexpected noise rate")
    if row.get("noise_scope") != "train":
        raise ValueError("reference row does not use train-only noise")
    if _as_int(row, "modified_val_count") or _as_int(row, "modified_test_count"):
        raise ValueError("reference row modified validation/test labels")
    if method == KNN and _as_int(row, "knn_exclude_self", -1) != 1:
        raise ValueError("kNN reference did not exclude the query node")
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": row["noise_scope"],
        "seed": int(row["seed"]),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "student_test": float(row["test"]),
        "student_valid": float(row["valid"]),
        "teacher_test": _as_float(row, "teacher_best_test"),
        "teacher_valid": _as_float(row, "teacher_best_val"),
        "split_hash": row["split_hash"],
        "noise_hash": row["noise_hash"],
        "wall_time_seconds": _as_float(row, "wall_time_seconds"),
        "teacher_training_seconds": _as_float(row, "teacher_training_seconds"),
        "student_training_seconds": _as_float(row, "student_training_seconds"),
        "student_parameter_count": _as_int(row, "student_parameter_count"),
        "teacher_parameter_count": _as_int(row, "teacher_parameter_count"),
        "teacher_use_xh": _as_int(row, "teacher_use_xh"),
        "student_input_dim": _as_int(row, "student_input_dim"),
        "teacher_input_dim": _as_int(row, "teacher_input_dim"),
        "student_inference_uses_graph": 0,
        "source": str(source),
    }


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[str, float, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            float(row["noise_rate"]),
            str(row["method"]),
            int(row["seed"]),
        )
        if key in unique:
            previous = unique[key]
            if abs(float(previous["student_test"]) - float(row["student_test"])) > 1e-12:
                raise ValueError(f"conflicting duplicate row: {key}")
            continue
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: (
            float(row["noise_rate"]),
            str(row["dataset"]),
            str(row["method"]),
            int(row["seed"]),
        ),
    )


def completeness_and_protocol(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    keyed = {
        (
            str(row["dataset"]),
            float(row["noise_rate"]),
            str(row["method"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    completeness: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for dataset in DATASETS:
        for rate in NOISE_RATES:
            for method in METHODS:
                missing = [
                    seed
                    for seed in SEEDS
                    if (dataset, rate, method, seed) not in keyed
                ]
                completeness.append(
                    {
                        "dataset": dataset,
                        "noise_rate": rate,
                        "method": method,
                        "expected_n": len(SEEDS),
                        "observed_n": len(SEEDS) - len(missing),
                        "missing_seeds": ",".join(map(str, missing)),
                        "complete": int(not missing),
                    }
                )
                if missing:
                    raise ValueError(
                        f"incomplete MLP-47 comparison: {dataset} {rate} "
                        f"{method} missing {missing}"
                    )
            for seed in SEEDS:
                paired = [
                    keyed[(dataset, rate, method, seed)]
                    for method in METHODS
                ]
                split_hashes = {str(row["split_hash"]) for row in paired}
                noise_hashes = {str(row["noise_hash"]) for row in paired}
                passed = len(split_hashes) == 1 and len(noise_hashes) == 1
                audits.append(
                    {
                        "dataset": dataset,
                        "noise_rate": rate,
                        "seed": seed,
                        "split_hash_count": len(split_hashes),
                        "noise_hash_count": len(noise_hashes),
                        "passed": int(passed),
                    }
                )
                if not passed:
                    raise ValueError(
                        f"protocol hash mismatch: {dataset} {rate} seed {seed}"
                    )
    return completeness, audits


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["dataset"]), float(row["noise_rate"]), str(row["method"]))
        ].append(row)
    output: list[dict[str, object]] = []
    for (dataset, rate, method), group in sorted(grouped.items()):
        student = [float(row["student_test"]) for row in group]
        teacher = [
            float(row["teacher_test"])
            for row in group
            if row["teacher_test"] is not None
        ]
        wall = [
            float(row["wall_time_seconds"])
            for row in group
            if row["wall_time_seconds"] is not None
        ]
        teacher_wall = [
            float(row["teacher_training_seconds"])
            for row in group
            if row["teacher_training_seconds"] is not None
        ]
        student_wall = [
            float(row["student_training_seconds"])
            for row in group
            if row["student_training_seconds"] is not None
        ]
        output.append(
            {
                "dataset": dataset,
                "noise_rate": rate,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": len(student),
                "student_mean": statistics.mean(student),
                "student_std": statistics.stdev(student),
                "teacher_mean": statistics.mean(teacher) if teacher else "",
                "teacher_std": statistics.stdev(teacher) if len(teacher) > 1 else "",
                "mean_wall_time_seconds": statistics.mean(wall) if wall else "",
                "mean_teacher_training_seconds": (
                    statistics.mean(teacher_wall) if teacher_wall else ""
                ),
                "mean_student_training_seconds": (
                    statistics.mean(student_wall) if student_wall else ""
                ),
            }
        )
    return output


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keyed = {
        (
            str(row["dataset"]),
            float(row["noise_rate"]),
            str(row["method"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for dataset in DATASETS:
        for rate in NOISE_RATES:
            for name, left_method, right_method in COMPARISONS:
                for seed in SEEDS:
                    left = keyed[(dataset, rate, left_method, seed)]
                    right = keyed[(dataset, rate, right_method, seed)]
                    output.append(
                        {
                            "dataset": dataset,
                            "noise_rate": rate,
                            "comparison": name,
                            "left_method": left_method,
                            "right_method": right_method,
                            "seed": seed,
                            "left_test": left["student_test"],
                            "right_test": right["student_test"],
                            "delta": float(left["student_test"])
                            - float(right["student_test"]),
                        }
                    )
            for method in NEW_METHODS:
                for seed in SEEDS:
                    row = keyed[(dataset, rate, method, seed)]
                    output.append(
                        {
                            "dataset": dataset,
                            "noise_rate": rate,
                            "comparison": f"{method}_student_vs_own_teacher",
                            "left_method": method,
                            "right_method": f"{method}_teacher",
                            "seed": seed,
                            "left_test": row["student_test"],
                            "right_test": row["teacher_test"],
                            "delta": float(row["student_test"])
                            - float(row["teacher_test"]),
                        }
                    )
    return output


def _fmt(mean: float, std: float) -> str:
    return f"{100 * mean:.2f}±{100 * std:.2f}"


def write_report(
    path: Path,
    aggregate_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> None:
    aggregate_key = {
        (str(row["dataset"]), float(row["noise_rate"]), str(row["method"])): row
        for row in aggregate_rows
    }
    lines = [
        "# MLP-47 NRGNN Robust-Teacher Distillation Report",
        "",
        "All values use five paired seeds. NRGNN teachers are retrained independently",
        "inside each seed; every student is a two-layer X-only MLP.",
        "",
    ]
    for rate in NOISE_RATES:
        lines.extend(
            [
                f"## Uniform {rate}",
                "",
                "| Method | Cora | CiteSeer | CoraFull |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            cells = []
            for dataset in DATASETS:
                row = aggregate_key[(dataset, rate, method)]
                cells.append(_fmt(float(row["student_mean"]), float(row["student_std"])))
            lines.append(f"| {METHOD_LABELS[method]} | {' | '.join(cells)} |")
        lines.append("")

    lines.extend(["## Paired comparisons", ""])
    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in delta_rows:
        grouped[
            (str(row["dataset"]), float(row["noise_rate"]), str(row["comparison"]))
        ].append(float(row["delta"]))
    lines.extend(
        [
            "| Rate | Dataset | Comparison | Mean delta | Positive seeds |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for (dataset, rate, comparison), values in sorted(grouped.items()):
        lines.append(
            f"| {rate} | {dataset} | {comparison} | "
            f"{100 * statistics.mean(values):+.2f}pp | "
            f"{sum(value > 0 for value in values)}/{len(values)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This experiment reports all pre-registered outcomes. It tests whether a",
            "mature robust graph teacher can replace the consensus teacher under the",
            "same X-only inference requirement; it does not change the base GB-LCSS",
            "main-table claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--reference_raw", type=Path, required=True)
    args = parser.parse_args()

    new_rows: list[dict[str, object]] = []
    for path in sorted(args.run_dir.glob("*.csv")):
        if path.name.startswith("mlp47_"):
            continue
        for row in read_csv(path):
            if row.get("method") in NEW_METHODS:
                new_rows.append(normalize_new_row(row, path))
    new_rows = deduplicate(new_rows)

    reference_rows = [
        normalize_reference_row(row, args.reference_raw)
        for row in read_csv(args.reference_raw)
        if row.get("dataset") in DATASETS
        and float(row.get("noise_rate", -1)) in NOISE_RATES
        and row.get("method") in METHODS[:5]
    ]
    combined = deduplicate(reference_rows + new_rows)
    completeness, audits = completeness_and_protocol(combined)
    aggregate_rows = aggregate(combined)
    delta_rows = paired_deltas(combined)

    write_csv(args.run_dir / "mlp47_combined_raw.csv", combined)
    write_csv(args.run_dir / "mlp47_aggregate.csv", aggregate_rows)
    write_csv(args.run_dir / "mlp47_paired_deltas.csv", delta_rows)
    write_csv(args.run_dir / "mlp47_protocol_audit.csv", audits)
    write_csv(args.run_dir / "mlp47_completeness.csv", completeness)
    write_report(args.run_dir / "mlp47_report.md", aggregate_rows, delta_rows)


if __name__ == "__main__":
    main()
