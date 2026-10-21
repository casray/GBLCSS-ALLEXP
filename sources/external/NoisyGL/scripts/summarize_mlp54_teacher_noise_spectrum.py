from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull")
SETTINGS = (("uniform", 0.2), ("pair", 0.4), ("instance", 0.4))
REFERENCE_SETTINGS = (("uniform", 0.4), ("uniform", 0.6))
FULL_SETTINGS = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)
SEEDS = tuple(range(5))

GLNN = "mlp_distill_gcn"
RANDOM = "mlp_distill_softdist_xh_shuffled"
KNN = "mlp_distill_knn_xh"
BALL = "mlp_distill_softdist_xh"
NRGNN = "mlp_distill_nrgnn"
METHODS = (GLNN, RANDOM, KNN, BALL, NRGNN)

METHOD_LABELS = {
    GLNN: "GCN(X)+CE (GLNN-style)",
    RANDOM: "Random-group consensus teacher",
    KNN: "kNN consensus teacher (exclude self)",
    BALL: "Granular-ball consensus teacher",
    NRGNN: "NRGNN teacher",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value == "" else int(float(value))


def _float(row: dict[str, str], key: str, default: float | None = None):
    value = row.get(key, "")
    return default if value == "" else float(value)


def validate_row(row: dict[str, str]) -> None:
    identity = (row.get("noise_type"), float(row.get("noise_rate", "nan")))
    if row.get("status") != "ok":
        raise ValueError(f"non-ok row: {row.get('dataset')} {row.get('method')}")
    if row.get("dataset") not in DATASETS or row.get("method") not in METHODS:
        raise ValueError("unexpected dataset or method")
    if identity not in SETTINGS:
        raise ValueError(f"unexpected noise setting: {identity}")
    if row.get("noise_scope") != "train":
        raise ValueError("noise is not restricted to training labels")
    if _int(row, "seed", -1) not in SEEDS:
        raise ValueError("unexpected seed")
    if _int(row, "modified_val_count") != 0 or _int(row, "modified_test_count") != 0:
        raise ValueError("validation or test labels were modified")
    if _int(row, "modified_count") != _int(row, "modified_train_count"):
        raise ValueError("modified count is inconsistent with train-only noise")
    if not row.get("split_hash") or not row.get("noise_hash"):
        raise ValueError("missing protocol hash")
    if row["method"] == KNN and _int(row, "knn_exclude_self", -1) != 1:
        raise ValueError("kNN teacher did not exclude the query node")
    if row["method"] == NRGNN:
        if _int(row, "student_inference_uses_graph", -1) != 0:
            raise ValueError("NRGNN student is not X-only at inference")
        error = _float(row, "teacher_probability_row_sum_max_error")
        if error is None or error > 1e-5:
            raise ValueError("invalid NRGNN teacher probabilities")


def normalize(row: dict[str, str], source: Path) -> dict[str, object]:
    validate_row(row)
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": row["noise_scope"],
        "seed": _int(row, "seed"),
        "method": row["method"],
        "method_label": METHOD_LABELS[row["method"]],
        "student_test": float(row["test"]),
        "student_valid": float(row["valid"]),
        "teacher_test": _float(row, "teacher_best_test"),
        "teacher_valid": _float(row, "teacher_best_val"),
        "split_hash": row["split_hash"],
        "noise_hash": row["noise_hash"],
        "knn_exclude_self": _int(row, "knn_exclude_self", 0),
        "student_inference_uses_graph": 0,
        "student_inference_input": "X-only",
        "wall_time_seconds": _float(row, "wall_time_seconds"),
        "teacher_training_seconds": _float(row, "teacher_training_seconds"),
        "student_training_seconds": _float(row, "student_training_seconds"),
        "source": str(source),
    }


def deduplicate(
    rows: list[dict[str, object]],
    settings=SETTINGS,
) -> list[dict[str, object]]:
    unique: dict[tuple[str, str, float, str, int], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["noise_type"]),
            float(row["noise_rate"]),
            str(row["method"]),
            int(row["seed"]),
        )
        if key in unique:
            if float(unique[key]["student_test"]) != float(row["student_test"]):
                raise ValueError(f"conflicting duplicate: {key}")
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (
        settings.index((str(row["noise_type"]), float(row["noise_rate"]))),
        DATASETS.index(str(row["dataset"])),
        METHODS.index(str(row["method"])),
        int(row["seed"]),
    ))


def audit_grid(rows: list[dict[str, object]], settings=SETTINGS):
    keyed = {
        (
            str(row["dataset"]), str(row["noise_type"]), float(row["noise_rate"]),
            str(row["method"]), int(row["seed"]),
        ): row
        for row in rows
    }
    completeness: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for noise_type, rate in settings:
        for dataset in DATASETS:
            for method in METHODS:
                missing = [seed for seed in SEEDS if (
                    dataset, noise_type, rate, method, seed,
                ) not in keyed]
                completeness.append({
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": rate,
                    "method": method,
                    "expected_n": 5,
                    "observed_n": 5 - len(missing),
                    "missing_seeds": ",".join(map(str, missing)),
                    "complete": int(not missing),
                })
                if missing:
                    raise ValueError(
                        f"incomplete grid: {dataset} {noise_type} {rate} "
                        f"{method} missing {missing}"
                    )
            for seed in SEEDS:
                paired = [keyed[(dataset, noise_type, rate, method, seed)] for method in METHODS]
                split_hashes = {str(row["split_hash"]) for row in paired}
                noise_hashes = {str(row["noise_hash"]) for row in paired}
                x_only = all(int(row["student_inference_uses_graph"]) == 0 for row in paired)
                passed = len(split_hashes) == len(noise_hashes) == 1 and x_only
                audits.append({
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "noise_rate": rate,
                    "seed": seed,
                    "split_hash_count": len(split_hashes),
                    "noise_hash_count": len(noise_hashes),
                    "all_students_x_only": int(x_only),
                    "passed": int(passed),
                })
                if not passed:
                    raise ValueError(
                        f"protocol mismatch: {dataset} {noise_type} {rate} seed {seed}"
                    )
    return completeness, audits


def aggregate(rows: list[dict[str, object]], settings=SETTINGS) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["dataset"]), str(row["noise_type"]),
            float(row["noise_rate"]), str(row["method"]),
        )].append(float(row["student_test"]))
    output = []
    for (dataset, noise_type, rate, method), values in sorted(
        grouped.items(), key=lambda item: (
            settings.index((item[0][1], item[0][2])),
            DATASETS.index(item[0][0]), METHODS.index(item[0][3]),
        )
    ):
        output.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": rate,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n": len(values),
            "student_mean": statistics.mean(values),
            "student_std": statistics.stdev(values),
        })
    return output


def paired_deltas(rows: list[dict[str, object]], settings=SETTINGS) -> list[dict[str, object]]:
    keyed = {
        (
            str(row["dataset"]), str(row["noise_type"]), float(row["noise_rate"]),
            str(row["method"]), int(row["seed"]),
        ): float(row["student_test"])
        for row in rows
    }
    output = []
    for noise_type, rate in settings:
        for dataset in DATASETS:
            for comparison in (GLNN, RANDOM, KNN, NRGNN):
                for seed in SEEDS:
                    ball = keyed[(dataset, noise_type, rate, BALL, seed)]
                    other = keyed[(dataset, noise_type, rate, comparison, seed)]
                    output.append({
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": rate,
                        "comparison": f"ball_vs_{comparison}",
                        "left_method": BALL,
                        "right_method": comparison,
                        "seed": seed,
                        "left_test": ball,
                        "right_test": other,
                        "delta": ball - other,
                    })
    return output


def _fmt(mean: float, std: float) -> str:
    return f"{100 * mean:.2f}±{100 * std:.2f}"


def report(path: Path, aggregate_rows, delta_rows, settings=SETTINGS) -> None:
    lookup = {
        (row["dataset"], row["noise_type"], row["noise_rate"], row["method"]): row
        for row in aggregate_rows
    }
    lines = [
        "# MLP-54 Teacher Noise-Spectrum Report", "",
        "All students are identical two-layer X-only MLPs. Teacher routes are",
        "retrained within each seed; no setting is selected by outcome.", "",
    ]
    win_counts = {method: 0 for method in METHODS}
    for noise_type, rate in settings:
        title = f"{noise_type.title()} {rate}"
        lines.extend([
            f"## {title}", "",
            "| Teacher | Student inference | Cora | CiteSeer | CoraFull |",
            "|---|---|---:|---:|---:|",
        ])
        for method in METHODS:
            cells = []
            for dataset in DATASETS:
                row = lookup[(dataset, noise_type, rate, method)]
                cells.append(_fmt(float(row["student_mean"]), float(row["student_std"])))
            lines.append(f"| {METHOD_LABELS[method]} | X-only | {' | '.join(cells)} |")
        for dataset in DATASETS:
            means = {
                method: float(lookup[(dataset, noise_type, rate, method)]["student_mean"])
                for method in METHODS
            }
            best = max(means.values())
            for method, value in means.items():
                if abs(value - best) <= 1e-12:
                    win_counts[method] += 1
        lines.append("")

    lines.extend([
        "## Mean-cell wins", "",
        f"| Teacher | Best/tied-best cells (out of {len(settings) * len(DATASETS)} cells) |",
        "|---|---:|",
    ])
    for method in METHODS:
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{win_counts[method]}/{len(settings) * len(DATASETS)} |"
        )

    grouped: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    for row in delta_rows:
        grouped[(
            row["noise_type"], row["noise_rate"], row["dataset"], row["comparison"],
        )].append(float(row["delta"]))
    lines.extend([
        "", "## Ball-teacher paired deltas", "",
        "| Setting | Dataset | Comparison | Mean delta (pp) | Positive seeds |",
        "|---|---|---|---:|---:|",
    ])
    for key, values in sorted(grouped.items(), key=lambda item: (
        settings.index((item[0][0], item[0][1])),
        DATASETS.index(item[0][2]), item[0][3],
    )):
        noise_type, rate, dataset, comparison = key
        lines.append(
            f"| {noise_type} {rate} | {dataset} | {comparison} | "
            f"{100 * statistics.mean(values):+.2f} | "
            f"{sum(value > 0 for value in values)}/5 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_reference(row: dict[str, str], source: Path) -> dict[str, object]:
    setting = (row.get("noise_type"), float(row.get("noise_rate", "nan")))
    method = row.get("method", "")
    if row.get("dataset") not in DATASETS or setting not in REFERENCE_SETTINGS:
        raise ValueError("unexpected reference identity")
    if method not in (GLNN, RANDOM, KNN, BALL):
        raise ValueError(f"unexpected reference method: {method}")
    if row.get("status") != "ok" or row.get("noise_scope") != "train":
        raise ValueError("invalid reference protocol")
    if _int(row, "modified_val_count") != 0 or _int(row, "modified_test_count") != 0:
        raise ValueError("reference validation or test labels were modified")
    if _int(row, "modified_count") != _int(row, "modified_train_count"):
        raise ValueError("reference modified count is inconsistent")
    if method == KNN and _int(row, "knn_exclude_self", -1) != 1:
        raise ValueError("reference kNN teacher did not exclude self")
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": "train",
        "seed": _int(row, "seed"),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "student_test": float(row["test"]),
        "student_valid": float(row["valid"]),
        "teacher_test": _float(row, "teacher_best_test"),
        "teacher_valid": _float(row, "teacher_best_val"),
        "split_hash": row["split_hash"],
        "noise_hash": row["noise_hash"],
        "knn_exclude_self": _int(row, "knn_exclude_self", 0),
        "student_inference_uses_graph": 0,
        "student_inference_input": "X-only",
        "source": str(source),
    }


def normalize_nrgnn_reference(row: dict[str, str], source: Path) -> dict[str, object]:
    setting = (row.get("noise_type"), float(row.get("noise_rate", "nan")))
    if row.get("dataset") not in DATASETS or setting not in REFERENCE_SETTINGS:
        raise ValueError("unexpected NRGNN reference identity")
    if row.get("method") != NRGNN:
        raise ValueError(f"unexpected NRGNN reference method: {row.get('method')}")
    if row.get("noise_scope") != "train":
        raise ValueError("invalid NRGNN reference protocol")
    if _int(row, "student_inference_uses_graph", -1) != 0:
        raise ValueError("NRGNN reference student is not X-only")
    return {
        "dataset": row["dataset"],
        "noise_type": row["noise_type"],
        "noise_rate": float(row["noise_rate"]),
        "noise_scope": "train",
        "seed": _int(row, "seed"),
        "method": NRGNN,
        "method_label": METHOD_LABELS[NRGNN],
        "student_test": float(row["student_test"]),
        "student_valid": float(row["student_valid"]),
        "teacher_test": _float(row, "teacher_test"),
        "teacher_valid": _float(row, "teacher_valid"),
        "split_hash": row["split_hash"],
        "noise_hash": row["noise_hash"],
        "knn_exclude_self": 0,
        "student_inference_uses_graph": 0,
        "student_inference_input": "X-only",
        "source": str(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--existing_reference", type=Path)
    parser.add_argument("--existing_nrgnn", type=Path)
    args = parser.parse_args()

    rows = []
    for source in args.inputs:
        rows.extend(normalize(row, source) for row in read_csv(source))
    rows = deduplicate(rows)
    completeness, audits = audit_grid(rows)
    aggregate_rows = aggregate(rows)
    delta_rows = paired_deltas(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mlp54_new_raw.csv", rows)
    write_csv(args.output_dir / "mlp54_new_aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "mlp54_new_paired_deltas.csv", delta_rows)
    write_csv(args.output_dir / "mlp54_completeness.csv", completeness)
    write_csv(args.output_dir / "mlp54_protocol_audit.csv", audits)
    report(args.output_dir / "mlp54_report.md", aggregate_rows, delta_rows)

    if bool(args.existing_reference) != bool(args.existing_nrgnn):
        raise ValueError("both existing reference files must be supplied together")
    if args.existing_reference:
        full_rows = list(rows)
        for row in read_csv(args.existing_reference):
            if row.get("method") in (GLNN, RANDOM, KNN, BALL):
                full_rows.append(normalize_reference(row, args.existing_reference))
        for row in read_csv(args.existing_nrgnn):
            if row.get("method") == NRGNN:
                full_rows.append(normalize_nrgnn_reference(row, args.existing_nrgnn))
        full_rows = deduplicate(full_rows, FULL_SETTINGS)
        full_completeness, full_audits = audit_grid(full_rows, FULL_SETTINGS)
        full_aggregate = aggregate(full_rows, FULL_SETTINGS)
        full_deltas = paired_deltas(full_rows, FULL_SETTINGS)
        write_csv(args.output_dir / "mlp54_full_spectrum_raw.csv", full_rows)
        write_csv(args.output_dir / "mlp54_full_spectrum_aggregate.csv", full_aggregate)
        write_csv(args.output_dir / "mlp54_full_spectrum_paired_deltas.csv", full_deltas)
        write_csv(args.output_dir / "mlp54_full_spectrum_completeness.csv", full_completeness)
        write_csv(args.output_dir / "mlp54_full_spectrum_protocol_audit.csv", full_audits)
        report(
            args.output_dir / "mlp54_full_spectrum_report.md",
            full_aggregate,
            full_deltas,
            FULL_SETTINGS,
        )


if __name__ == "__main__":
    main()
