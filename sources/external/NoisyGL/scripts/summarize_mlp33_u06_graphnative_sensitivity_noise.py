import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull", "amazonpho", "pubmed")
SENSITIVITY_DATASETS = ("cora", "citeseer", "corafull")
NOISE_TYPES = ("pair", "instance")
SEEDS = tuple(range(5))

LCSS = "frozengrace_gb_mlp_softdist_beta05"
SMALL_ONLY = "mlp_small_loss"
COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss"
SHUFFLED_COMBO = "frozengrace_gb_mlp_softdist_beta05_small_loss_shuffled"
NOISE_METHODS = (LCSS, SMALL_ONLY, COMBO, SHUFFLED_COMBO)
GRAPH_METHODS = ("nrgnn", "nrgnn_xh", "rtgnn")

# These are the pre-existing u0.6 X-only distillation anchors used in the
# seed-0 go/no-go comparison. The selection is frozen before MLP-33.
DISTILL_ANCHOR = {
    "cora": "mlp_distill_knn_xh",
    "citeseer": "mlp_distill_softdist_xh",
    "corafull": "mlp_distill_gcn_xh",
    "amazonpho": "mlp_distill_gcn_xh",
    "pubmed": "mlp_distill_softdist_xh",
}


def _protocol_check(row, *, noise_type, noise_rate, seed):
    if row.get("status") != "ok":
        raise ValueError(f"non-ok row: {row.get('dataset')} {row.get('method')} seed={seed}")
    if row.get("noise_type") != noise_type or abs(float(row["noise_rate"]) - noise_rate) > 1e-12:
        raise ValueError("wrong noise protocol")
    if row.get("noise_scope") != "train":
        raise ValueError("noise must be train-only")
    if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
        raise ValueError("validation/test labels were modified")
    if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
        raise ValueError("modified labels are not restricted to train")
    if int(row["seed"]) != seed:
        raise ValueError("seed mismatch")


def build_noise_type_summary(rows):
    keyed = {}
    raw = []
    for row in rows:
        dataset = row.get("dataset")
        method = row.get("method")
        noise_type = row.get("noise_type")
        if dataset not in DATASETS or method not in NOISE_METHODS or noise_type not in NOISE_TYPES:
            continue
        seed = int(row["seed"])
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed: {seed}")
        _protocol_check(row, noise_type=noise_type, noise_rate=0.4, seed=seed)
        key = (noise_type, dataset, method, seed)
        if key in keyed:
            raise ValueError(f"duplicate grid cell: {key}")
        normalized = dict(row)
        normalized["seed"] = seed
        normalized["test"] = float(row["test"])
        keyed[key] = normalized
        raw.append(normalized)

    for noise_type in NOISE_TYPES:
        for dataset in DATASETS:
            for seed in SEEDS:
                hashes = set()
                for method in NOISE_METHODS:
                    key = (noise_type, dataset, method, seed)
                    if key not in keyed:
                        raise ValueError(f"missing grid cell: {key}")
                    hashes.add((keyed[key]["split_hash"], keyed[key]["noise_hash"]))
                if len(hashes) != 1:
                    raise ValueError(
                        f"protocol hash mismatch: noise_type={noise_type} dataset={dataset} seed={seed}"
                    )

    raw.sort(
        key=lambda row: (
            NOISE_TYPES.index(row["noise_type"]),
            DATASETS.index(row["dataset"]),
            NOISE_METHODS.index(row["method"]),
            row["seed"],
        )
    )
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["noise_type"], row["dataset"], row["method"])].append(row["test"])

    aggregate = []
    for noise_type in NOISE_TYPES:
        for dataset in DATASETS:
            for method in NOISE_METHODS:
                values = grouped[(noise_type, dataset, method)]
                aggregate.append(
                    {
                        "noise_type": noise_type,
                        "dataset": dataset,
                        "method": method,
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "std": statistics.pstdev(values),
                    }
                )

    seed_deltas = []
    for noise_type in NOISE_TYPES:
        for dataset in DATASETS:
            for seed in SEEDS:
                value = lambda method: keyed[(noise_type, dataset, method, seed)]["test"]
                lcss = value(LCSS)
                small = value(SMALL_ONLY)
                combo = value(COMBO)
                shuffled = value(SHUFFLED_COMBO)
                seed_deltas.append(
                    {
                        "noise_type": noise_type,
                        "dataset": dataset,
                        "seed": seed,
                        "lcss": lcss,
                        "small_only": small,
                        "combo": combo,
                        "shuffled_combo": shuffled,
                        "combo_minus_lcss": combo - lcss,
                        "combo_minus_small_only": combo - small,
                        "synergy_delta": combo - max(lcss, small),
                        "structure_delta": combo - shuffled,
                    }
                )

    paired = []
    for noise_type in NOISE_TYPES:
        for dataset in DATASETS:
            current = [
                row
                for row in seed_deltas
                if row["noise_type"] == noise_type and row["dataset"] == dataset
            ]
            synergy = [row["synergy_delta"] for row in current]
            structure = [row["structure_delta"] for row in current]
            paired.append(
                {
                    "noise_type": noise_type,
                    "dataset": dataset,
                    "combo_minus_lcss": statistics.mean(
                        row["combo_minus_lcss"] for row in current
                    ),
                    "combo_minus_small_only": statistics.mean(
                        row["combo_minus_small_only"] for row in current
                    ),
                    "synergy_delta": statistics.mean(synergy),
                    "synergy_std": statistics.pstdev(synergy),
                    "synergy_positive_seeds": sum(value > 0 for value in synergy),
                    "structure_delta": statistics.mean(structure),
                    "structure_std": statistics.pstdev(structure),
                    "structure_positive_seeds": sum(value > 0 for value in structure),
                }
            )

    gates = []
    for noise_type in NOISE_TYPES:
        current = [row for row in paired if row["noise_type"] == noise_type]
        synergy_passes = sum(
            row["synergy_delta"] > 0 and row["synergy_positive_seeds"] >= 4 for row in current
        )
        structure_passes = sum(
            row["structure_delta"] >= 0.01 and row["structure_positive_seeds"] >= 4
            for row in current
        )
        mean_synergy = statistics.mean(row["synergy_delta"] for row in current)
        minimum_synergy = min(row["synergy_delta"] for row in current)
        reasons = []
        if synergy_passes < 3:
            reasons.append("fewer_than_3_of_5_paired_synergy_passes")
        if mean_synergy < 0.01:
            reasons.append("mean_synergy_below_1pp")
        if minimum_synergy < -0.005:
            reasons.append("dataset_regression_over_0p5pp")
        if structure_passes < 3:
            reasons.append("fewer_than_3_of_5_structure_passes")
        gates.append(
            {
                "noise_type": noise_type,
                "pass": int(not reasons),
                "qualifying_synergy_datasets": synergy_passes,
                "mean_synergy_delta": mean_synergy,
                "minimum_synergy_delta": minimum_synergy,
                "qualifying_structure_datasets": structure_passes,
                "reasons": ";".join(reasons),
            }
        )
    return raw, aggregate, seed_deltas, paired, gates


def evaluate_graphnative_gate(comparisons):
    qualifying = sum(row["best_xonly_minus_best_graph"] >= -0.02 for row in comparisons)
    return {
        "pass": int(qualifying >= 3),
        "qualifying_datasets": qualifying,
        "required_datasets": 3,
        "threshold": -0.02,
        "decision": "expand_5seed_and_coauthorph" if qualifying >= 3 else "stop_after_seed0",
    }


def build_graphnative_summary(graph_rows, direct_scores, distill_scores):
    by_dataset = defaultdict(dict)
    for row in graph_rows:
        dataset = row.get("dataset")
        method = row.get("method")
        if dataset not in DATASETS or method not in GRAPH_METHODS:
            continue
        _protocol_check(row, noise_type="uniform", noise_rate=0.6, seed=0)
        if method in by_dataset[dataset]:
            raise ValueError(f"duplicate graph-native row: {dataset} {method}")
        by_dataset[dataset][method] = float(row["test"])
    comparisons = []
    for dataset in DATASETS:
        if "nrgnn" not in by_dataset[dataset] or "nrgnn_xh" not in by_dataset[dataset]:
            raise ValueError(f"missing NRGNN arm: {dataset}")
        graph_method, graph_score = max(by_dataset[dataset].items(), key=lambda item: item[1])
        direct = float(direct_scores[dataset])
        distill_method, distill_score = distill_scores[dataset]
        if distill_score >= direct:
            xonly_method, xonly_score = distill_method, distill_score
        else:
            xonly_method, xonly_score = COMBO, direct
        comparisons.append(
            {
                "dataset": dataset,
                "direct_combo": direct,
                "distill_method": distill_method,
                "distill_score": distill_score,
                "nrgnn": by_dataset[dataset]["nrgnn"],
                "nrgnn_xh": by_dataset[dataset]["nrgnn_xh"],
                "rtgnn": by_dataset[dataset].get("rtgnn", ""),
                "best_graph_method": graph_method,
                "best_graph_score": graph_score,
                "best_xonly_method": xonly_method,
                "best_xonly_score": xonly_score,
                "best_xonly_minus_best_graph": xonly_score - graph_score,
            }
        )
    return comparisons, evaluate_graphnative_gate(comparisons)


def build_misspec_summary(scores):
    rows = []
    for dataset in SENSITIVITY_DATASETS:
        correct = float(scores[0.6][dataset])
        for assumed in (0.4, 0.5, 0.6):
            test = float(scores[assumed][dataset])
            rows.append(
                {
                    "dataset": dataset,
                    "true_noise_rate": 0.6,
                    "assumed_noise_rate": assumed,
                    "test": test,
                    "delta_vs_correct": test - correct,
                }
            )
    return rows


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _standard_scores(rows, method, datasets):
    result = {}
    for row in rows:
        if (
            row.get("dataset") in datasets
            and row.get("method") == method
            and int(row["seed"]) == 0
            and row.get("noise_type") == "uniform"
            and abs(float(row["noise_rate"]) - 0.6) < 1e-12
        ):
            _protocol_check(row, noise_type="uniform", noise_rate=0.6, seed=0)
            result[row["dataset"]] = float(row["test"])
    missing = set(datasets) - set(result)
    if missing:
        raise ValueError(f"missing standard scores: {sorted(missing)}")
    return result


def _distill_scores(mlp9_rows, mlp29_rows):
    result = {}
    for dataset, method in DISTILL_ANCHOR.items():
        source = mlp9_rows if dataset in ("cora", "citeseer", "pubmed") else mlp29_rows
        value_key = "student_test" if source is mlp9_rows else "test"
        matches = [
            row
            for row in source
            if row.get("dataset") == dataset
            and row.get("method") == method
            and int(row["seed"]) == 0
            and row.get("noise_type") == "uniform"
            and abs(float(row["noise_rate"]) - 0.6) < 1e-12
        ]
        if len(matches) != 1:
            raise ValueError(f"missing/duplicate distillation anchor: {dataset} {method}")
        result[dataset] = (method, float(matches[0][value_key]))
    return result


def _report(path, aggregate, paired, noise_gates, graph_rows, graph_gate, misspec):
    lookup = {(row["noise_type"], row["dataset"], row["method"]): row for row in aggregate}
    lines = ["# MLP-33 High-Noise Graph-Native, Sensitivity, and Noise-Type Report", ""]
    lines.extend(
        [
            "## Uniform 0.6 graph-native seed-0 comparison",
            "",
            "| Dataset | Direct combo | Frozen X-only distill | NRGNN | NRGNN[X||H] | RTGNN | Best X-only - best graph |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in graph_rows:
        rtgnn = "NA" if row["rtgnn"] == "" else f"{100*row['rtgnn']:.2f}"
        lines.append(
            f"| {row['dataset']} | {100*row['direct_combo']:.2f} | {100*row['distill_score']:.2f} "
            f"({row['distill_method']}) | {100*row['nrgnn']:.2f} | {100*row['nrgnn_xh']:.2f} | "
            f"{rtgnn} | {100*row['best_xonly_minus_best_graph']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            f"Gate: {graph_gate['qualifying_datasets']}/5 datasets are within 2pp or win; "
            f"pass={graph_gate['pass']}; decision={graph_gate['decision']}.",
            "",
            "## Forget-rate misspecification at true uniform 0.6 (seed 0)",
            "",
            "| Dataset | Assumed rate | Accuracy | Delta vs correct 0.6 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in misspec:
        lines.append(
            f"| {row['dataset']} | {row['assumed_noise_rate']:.1f} | {100*row['test']:.2f} | "
            f"{100*row['delta_vs_correct']:+.2f}pp |"
        )

    for noise_type in NOISE_TYPES:
        lines.extend(
            [
                "",
                f"## {noise_type.title()} 0.4 five-seed comparison",
                "",
                "| Dataset | LCSS | Small-loss | Combination | Shuffled combination | Synergy | Positive | Real-shuffled | Positive |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in paired if item["noise_type"] == noise_type]:
            dataset = row["dataset"]
            fmt = lambda method: (
                f"{100*lookup[(noise_type, dataset, method)]['mean']:.2f}+/-"
                f"{100*lookup[(noise_type, dataset, method)]['std']:.2f}"
            )
            lines.append(
                f"| {dataset} | {fmt(LCSS)} | {fmt(SMALL_ONLY)} | {fmt(COMBO)} | "
                f"{fmt(SHUFFLED_COMBO)} | {100*row['synergy_delta']:+.2f}pp | "
                f"{row['synergy_positive_seeds']}/5 | {100*row['structure_delta']:+.2f}pp | "
                f"{row['structure_positive_seeds']}/5 |"
            )
        gate = next(row for row in noise_gates if row["noise_type"] == noise_type)
        lines.extend(
            [
                "",
                f"Gate: pass={gate['pass']}; synergy={gate['qualifying_synergy_datasets']}/5; "
                f"mean={100*gate['mean_synergy_delta']:+.2f}pp; "
                f"minimum={100*gate['minimum_synergy_delta']:+.2f}pp; "
                f"structure={gate['qualifying_structure_datasets']}/5; "
                f"reasons={gate['reasons'] or 'pass'}.",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--direct_anchor", required=True)
    parser.add_argument("--distill_small", required=True)
    parser.add_argument("--distill_large", required=True)
    args = parser.parse_args()
    output = Path(args.run_dir)

    noise_rows = []
    for name in (
        "pair0p4_seed0.csv",
        "pair0p4_seeds_1_2_3_4.csv",
        "instance0p4_seed0.csv",
        "instance0p4_seeds_1_2_3_4.csv",
    ):
        noise_rows.extend(_read(output / name))
    raw, aggregate, seed_deltas, paired, noise_gates = build_noise_type_summary(noise_rows)

    graph_input = _read(output / "u06_nrgnn_seed0.csv") + _read(output / "u06_rtgnn_seed0.csv")
    direct_anchor_rows = _read(args.direct_anchor)
    direct_scores = _standard_scores(direct_anchor_rows, COMBO, DATASETS)
    distill_scores = _distill_scores(_read(args.distill_small), _read(args.distill_large))
    graph_rows, graph_gate = build_graphnative_summary(graph_input, direct_scores, distill_scores)

    sensitivity_scores = {0.6: {dataset: direct_scores[dataset] for dataset in SENSITIVITY_DATASETS}}
    for assumed, name in ((0.4, "u06_assumed0p4_seed0.csv"), (0.5, "u06_assumed0p5_seed0.csv")):
        sensitivity_scores[assumed] = _standard_scores(
            _read(output / name), COMBO, SENSITIVITY_DATASETS
        )
    misspec = build_misspec_summary(sensitivity_scores)

    _write(output / "mlp33_noise_raw.csv", raw)
    _write(output / "mlp33_noise_aggregate.csv", aggregate)
    _write(output / "mlp33_noise_seed_deltas.csv", seed_deltas)
    _write(output / "mlp33_noise_paired_deltas.csv", paired)
    _write(output / "mlp33_noise_gates.csv", noise_gates)
    _write(output / "mlp33_graphnative_seed0.csv", graph_rows)
    _write(output / "mlp33_graphnative_gate.csv", [graph_gate])
    _write(output / "mlp33_forget_rate_sensitivity.csv", misspec)
    _report(
        output / "mlp33_report.md",
        aggregate,
        paired,
        noise_gates,
        graph_rows,
        graph_gate,
        misspec,
    )


if __name__ == "__main__":
    main()
