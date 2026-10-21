from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp8_replacement_expansion_20260702"
MLP5_RAW = REPO_ROOT / "runs" / "mlp_maintable_20260701" / "mlp_maintable_combined_raw.csv"

SOFTDIST = "frozengrace_gb_mlp_softdist_beta05"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_shuffled"
KNN = "knn_mlp_softdist_beta05"
KMEANS = "kmeans_mlp_softdist_beta05"
MLP_CE = "mlp_ce"
EXPAND = "frozengrace_gb_mlp_softdist_beta05_expand"
EXPAND_SHUFFLED = "frozengrace_gb_mlp_softdist_beta05_expand_shuffled"
KNN_EXPAND = "knn_mlp_softdist_beta05_expand"
GLNN = "mlp_distill_gcn"
P0_METHODS = (SOFTDIST, KNN, KMEANS, SHUFFLED)
P0_DATASETS = ("cora", "citeseer", "pubmed")
P0_SEEDS = tuple(range(5))
P1_DATASETS = ("cora", "citeseer", "pubmed")
P1_NOISE_RATES = (0.4, 0.6)
P1_SEEDS = (0,)
P1_LAMBDAS = (0.1, 0.3, 1.0)

METHOD_LABELS = {
    SOFTDIST: "true ball beta=0.5",
    KNN: "kNN-in-H beta=0.5",
    KMEANS: "kmeans-H beta=0.5",
    SHUFFLED: "shuffled ball beta=0.5",
    MLP_CE: "MLP CE",
    EXPAND: "true ball beta=0.5 + expansion",
    EXPAND_SHUFFLED: "shuffled ball beta=0.5 + expansion",
    KNN_EXPAND: "kNN-in-H beta=0.5 + expansion",
    GLNN: "MLP distill noisy GCN teacher",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _float_or_blank(value):
    if value is None or str(value).strip() == "":
        return ""
    return float(value)


def load_mlp5_anchor_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(MLP5_RAW):
        method = row.get("method", "")
        if method not in {SOFTDIST, SHUFFLED}:
            continue
        if row.get("table") != "main":
            continue
        if row.get("dataset") not in P0_DATASETS:
            continue
        if row.get("noise_type") != "uniform" or abs(float(row.get("noise_rate", 0)) - 0.4) > 1e-9:
            continue
        if abs(float(row.get("train_percent", 0)) - 0.2) > 1e-9:
            continue
        rows.append({
            "phase": "p0",
            "dataset": row["dataset"],
            "method": method,
            "variant": method,
            "method_label": METHOD_LABELS[method],
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "seed": int(row["seed"]),
            "test_acc": float(row["test_acc"]),
            "source": row.get("source", str(MLP5_RAW)),
            "target_builder": "ball" if method == SOFTDIST else "shuffled",
            "matched_k": "",
            "matched_num_balls": "",
            "kmeans_n_clusters": "",
            "lambda_u": "",
            "expansion_candidate_count": "",
            "expansion_coverage_unlabeled": "",
            "teacher_best_test": "",
        })
    return rows


def load_mlp5_p1_anchor_rows() -> list[dict[str, object]]:
    rows = []
    for row in read_csv(MLP5_RAW):
        method = row.get("method", "")
        if method not in {SOFTDIST, SHUFFLED, MLP_CE}:
            continue
        if row.get("table") != "main":
            continue
        if row.get("dataset") not in P1_DATASETS:
            continue
        if row.get("noise_type") != "uniform":
            continue
        noise_rate = float(row.get("noise_rate", 0))
        if noise_rate not in P1_NOISE_RATES:
            continue
        if int(row.get("seed", -1)) not in P1_SEEDS:
            continue
        if abs(float(row.get("train_percent", 0)) - 0.2) > 1e-9:
            continue
        rows.append({
            "phase": "p1",
            "dataset": row["dataset"],
            "method": method,
            "variant": method,
            "method_label": METHOD_LABELS[method],
            "noise_type": "uniform",
            "noise_rate": noise_rate,
            "seed": int(row["seed"]),
            "test_acc": float(row["test_acc"]),
            "source": row.get("source", str(MLP5_RAW)),
            "target_builder": "ball" if method == SOFTDIST else ("shuffled" if method == SHUFFLED else ""),
            "matched_k": "",
            "matched_num_balls": "",
            "kmeans_n_clusters": "",
            "lambda_u": "",
            "expansion_candidate_count": "",
            "expansion_coverage_unlabeled": "",
            "teacher_best_test": "",
        })
    return rows


def load_p0_new_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("p0_replacement_*.csv")):
        for row in read_csv(path):
            method = row.get("method") or row.get("base_method")
            if method not in {KNN, KMEANS}:
                continue
            if row.get("status", "ok").lower() not in {"", "ok"}:
                continue
            rows.append({
                "phase": "p0",
                "dataset": row["dataset"].lower(),
                "method": method,
                "variant": method,
                "method_label": METHOD_LABELS[method],
                "noise_type": row["noise_type"].lower(),
                "noise_rate": float(row["noise_rate"]),
                "seed": int(row["seed"]),
                "test_acc": float(row["test"]),
                "source": str(path),
                "target_builder": row.get("target_builder") or row.get("soft_target_builder") or "",
                "matched_k": _float_or_blank(row.get("matched_k")),
                "matched_num_balls": _float_or_blank(row.get("matched_num_balls") or row.get("matched_reference_num_balls")),
                "kmeans_n_clusters": _float_or_blank(row.get("kmeans_n_clusters")),
                "lambda_u": "",
                "expansion_candidate_count": "",
                "expansion_coverage_unlabeled": "",
                "teacher_best_test": "",
            })
    return rows


def _lambda_variant(method: str, lambda_u) -> str:
    if lambda_u == "" or lambda_u is None:
        return method
    return f"{method}_lambda{float(lambda_u):g}"


def load_p1_new_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    p1_methods = {EXPAND, EXPAND_SHUFFLED, KNN_EXPAND, GLNN}
    for path in sorted(run_dir.glob("p1_*.csv")):
        for row in read_csv(path):
            method = row.get("method") or row.get("base_method")
            if method not in p1_methods:
                continue
            if row.get("status", "ok").lower() not in {"", "ok"}:
                continue
            lambda_u = _float_or_blank(row.get("lambda_u"))
            variant = _lambda_variant(method, lambda_u)
            label = METHOD_LABELS[method]
            if lambda_u != "":
                label = f"{label} (lambda={float(lambda_u):g})"
            rows.append({
                "phase": "p1",
                "dataset": row["dataset"].lower(),
                "method": method,
                "variant": variant,
                "method_label": label,
                "noise_type": row["noise_type"].lower(),
                "noise_rate": float(row["noise_rate"]),
                "seed": int(row["seed"]),
                "test_acc": float(row["test"]),
                "source": str(path),
                "target_builder": row.get("target_builder") or row.get("soft_target_builder") or "",
                "matched_k": _float_or_blank(row.get("matched_k")),
                "matched_num_balls": _float_or_blank(row.get("matched_num_balls") or row.get("matched_reference_num_balls")),
                "kmeans_n_clusters": _float_or_blank(row.get("kmeans_n_clusters")),
                "lambda_u": lambda_u,
                "expansion_candidate_count": _float_or_blank(row.get("expansion_candidate_count")),
                "expansion_coverage_unlabeled": _float_or_blank(row.get("expansion_coverage_unlabeled")),
                "teacher_best_test": _float_or_blank(row.get("teacher_best_test")),
            })
    return rows


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique = {}
    for row in rows:
        key = (
            row["phase"],
            row["dataset"],
            row["noise_type"],
            row["noise_rate"],
            row.get("variant", row["method"]),
            row["seed"],
        )
        if key in unique and abs(float(unique[key]["test_acc"]) - float(row["test_acc"])) > 1e-12:
            raise ValueError(f"conflicting duplicate row for {key}")
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (row["phase"], row["dataset"], row.get("variant", row["method"]), row["seed"]))


def expected_keys() -> set[tuple[object, ...]]:
    return {
        ("p0", dataset, "uniform", 0.4, method, seed)
        for dataset in P0_DATASETS
        for method in P0_METHODS
        for seed in P0_SEEDS
    }


def observed_keys(rows: list[dict[str, object]]) -> set[tuple[object, ...]]:
    return {
        (row["phase"], row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"]))
        for row in rows
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    aux = {}
    for row in rows:
        key = (
            row["phase"],
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            row.get("variant", row["method"]),
            row["method_label"],
        )
        groups[key].append(float(row["test_acc"]))
        aux.setdefault(key, []).append(row)
    output = []
    for (phase, dataset, noise_type, noise_rate, method, variant, label), values in groups.items():
        group_rows = aux[(phase, dataset, noise_type, noise_rate, method, variant, label)]
        matched_k_values = [row["matched_k"] for row in group_rows if row["matched_k"] != ""]
        matched_num_values = [row["matched_num_balls"] for row in group_rows if row["matched_num_balls"] != ""]
        kmeans_values = [row["kmeans_n_clusters"] for row in group_rows if row["kmeans_n_clusters"] != ""]
        lambda_values = [row["lambda_u"] for row in group_rows if row.get("lambda_u", "") != ""]
        expansion_count_values = [row["expansion_candidate_count"] for row in group_rows if row.get("expansion_candidate_count", "") != ""]
        expansion_cov_values = [row["expansion_coverage_unlabeled"] for row in group_rows if row.get("expansion_coverage_unlabeled", "") != ""]
        output.append({
            "phase": phase,
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": noise_rate,
            "method": method,
            "variant": variant,
            "method_label": label,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "matched_k_mean": "" if not matched_k_values else statistics.mean(matched_k_values),
            "matched_num_balls_mean": "" if not matched_num_values else statistics.mean(matched_num_values),
            "kmeans_n_clusters_mean": "" if not kmeans_values else statistics.mean(kmeans_values),
            "lambda_u": "" if not lambda_values else statistics.mean(lambda_values),
            "expansion_candidate_count_mean": "" if not expansion_count_values else statistics.mean(expansion_count_values),
            "expansion_coverage_unlabeled_mean": "" if not expansion_cov_values else statistics.mean(expansion_cov_values),
        })
    return sorted(output, key=lambda row: (row["phase"], row["dataset"], row["noise_rate"], row["variant"]))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test_acc"])
        for row in rows
        if row["phase"] == "p0"
    }
    output = []
    for dataset in P0_DATASETS:
        for baseline in (KNN, KMEANS, SHUFFLED):
            deltas = []
            seeds = []
            for seed in P0_SEEDS:
                left = (dataset, SOFTDIST, seed)
                right = (dataset, baseline, seed)
                if left in values and right in values:
                    seeds.append(seed)
                    deltas.append(values[left] - values[right])
            if not deltas:
                continue
            output.append({
                "phase": "p0",
                "dataset": dataset,
                "comparison": f"ball_minus_{baseline}",
                "baseline": baseline,
                "n": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_count": sum(1 for value in deltas if value > 0),
                "seeds": ",".join(str(seed) for seed in seeds),
            })
    return output


def p1_gate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    p1_rows = [row for row in rows if row["phase"] == "p1"]
    values = {
        (
            row["dataset"],
            float(row["noise_rate"]),
            row["method"],
            "" if row.get("lambda_u", "") == "" else float(row["lambda_u"]),
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in p1_rows
    }
    output = []
    for noise_rate in P1_NOISE_RATES:
        for lambda_u in P1_LAMBDAS:
            g1_deltas = []
            g3_deltas = []
            g4_shuf_deltas = []
            g4_knn_deltas = []
            datasets_g1 = []
            for dataset in P1_DATASETS:
                base = values.get((dataset, noise_rate, SOFTDIST, "", 0))
                exp = values.get((dataset, noise_rate, EXPAND, lambda_u, 0))
                shuf = values.get((dataset, noise_rate, EXPAND_SHUFFLED, lambda_u, 0))
                knn = values.get((dataset, noise_rate, KNN_EXPAND, lambda_u, 0))
                glnn = values.get((dataset, noise_rate, GLNN, "", 0))
                if base is not None and exp is not None:
                    g1_deltas.append(exp - base)
                    datasets_g1.append(dataset)
                if glnn is not None and exp is not None:
                    g3_deltas.append(exp - glnn)
                if shuf is not None and exp is not None:
                    g4_shuf_deltas.append(exp - shuf)
                if knn is not None and exp is not None:
                    g4_knn_deltas.append(exp - knn)
            output.append({
                "phase": "p1",
                "noise_rate": noise_rate,
                "lambda_u": lambda_u,
                "gate": "G1_expansion_minus_train_only",
                "n": len(g1_deltas),
                "mean_delta": "" if not g1_deltas else statistics.mean(g1_deltas),
                "positive_count": sum(1 for value in g1_deltas if value > 0),
                "pass_count": sum(1 for value in g1_deltas if value >= 0.03),
                "datasets": ",".join(datasets_g1),
                "decision": "pass" if len(g1_deltas) >= 3 and sum(1 for value in g1_deltas if value >= 0.03) >= 2 else "not-pass",
            })
            for gate, deltas in [
                ("G3_expansion_minus_glnn", g3_deltas),
                ("G4_expansion_minus_shuffled_expansion", g4_shuf_deltas),
                ("G4_expansion_minus_knn_expansion", g4_knn_deltas),
            ]:
                output.append({
                    "phase": "p1",
                    "noise_rate": noise_rate,
                    "lambda_u": lambda_u,
                    "gate": gate,
                    "n": len(deltas),
                    "mean_delta": "" if not deltas else statistics.mean(deltas),
                    "positive_count": sum(1 for value in deltas if value > 0),
                    "pass_count": sum(1 for value in deltas if value > 0),
                    "datasets": ",".join(P1_DATASETS),
                    "decision": "pass" if len(deltas) >= 3 and sum(1 for value in deltas if value > 0) >= 2 else "not-pass",
                })
    return output


def p0_decision(delta_rows: list[dict[str, object]]) -> tuple[str, list[str]]:
    by_key = {(row["dataset"], row["baseline"]): row for row in delta_rows}
    winning_datasets = []
    replacement_notes = []
    for dataset in P0_DATASETS:
        knn = by_key.get((dataset, KNN))
        kmeans = by_key.get((dataset, KMEANS))
        if not knn or not kmeans:
            replacement_notes.append(f"{dataset}: missing paired replacement rows")
            continue
        knn_pass = float(knn["mean_delta"]) >= 0.01 and int(knn["positive_count"]) >= 4
        kmeans_pass = float(kmeans["mean_delta"]) >= 0.01 and int(kmeans["positive_count"]) >= 4
        if knn_pass and kmeans_pass:
            winning_datasets.append(dataset)
        else:
            replacement_notes.append(
                f"{dataset}: ball-knn={float(knn['mean_delta']):+.4f} ({knn['positive_count']}/5), "
                f"ball-kmeans={float(kmeans['mean_delta']):+.4f} ({kmeans['positive_count']}/5)"
            )
    if len(winning_datasets) >= 2:
        return "p0_ball_has_independent_value", winning_datasets
    return "p0_replacement_not_rejected", replacement_notes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def write_report(path: Path, rows, agg_rows, delta_rows, missing) -> None:
    decision, notes = p0_decision(delta_rows)
    lines = [
        "# MLP-8 Replacement + Expansion Report",
        "",
        "## Material Passport",
        "- Artifact type: experiment result",
        "- Phase covered in this report: P0 replacement control",
        "- Protocol: X-only weak MLP, train-only label noise, percent split 0.2/0.1/0.7",
        "- Reused anchor: MLP-5 beta=0.5 and shuffled rows",
        "",
        "## Completeness",
        f"- Expected P0 rows: {len(expected_keys())}",
        f"- Observed P0 rows: {len(rows)}",
        f"- Missing P0 rows: {len(missing)}",
        "",
        "## P0 Aggregate",
        "| Dataset | Method | Mean +/- Std | matched k | matched K |",
        "|---|---|---:|---:|---:|",
    ]
    for row in agg_rows:
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | "
            f"{fmt(row['mean'])}+/-{fmt(row['std'])} | "
            f"{row.get('matched_k_mean', '')} | "
            f"{row.get('matched_num_balls_mean') or row.get('kmeans_n_clusters_mean') or ''} |"
        )
    lines.extend([
        "",
        "## P0 Paired Deltas: true ball - replacement",
        "| Dataset | Comparison | Mean delta | Wins | Decision aid |",
        "|---|---|---:|---:|---|",
    ])
    for row in delta_rows:
        aid = "pass" if float(row["mean_delta"]) >= 0.01 and int(row["positive_count"]) >= 4 else "not-pass"
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {fmt(row['mean_delta'])}pp | "
            f"{row['positive_count']}/{row['n']} | {aid} |"
        )
    lines.extend([
        "",
        "## P0 Verdict",
        f"Verdict: **{decision}**",
    ])
    for note in notes:
        lines.append(f"- {note}")
    lines.extend([
        "",
        "## Claim Boundary",
        "- P0 tests only the train-target role of granular balls.",
        "- If replacement is not rejected, it does not automatically invalidate P1 expansion/teacher roles.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_p1_report(path: Path, p1_rows, p1_agg_rows, p1_gate_rows_) -> None:
    if not p1_rows:
        return
    lines = [
        "",
        "## P1 Expansion + GLNN Seed0",
        "- Phase covered in this section: P1 seed0 lambda scan",
        "- Protocol: X-only weak MLP, train-only uniform label noise, percent split 0.2/0.1/0.7",
        "- Gate note: G1 uses a global lambda per noise rate, not a per-dataset cherry-picked lambda.",
        "",
        "### P1 Aggregate",
        "| Dataset | Noise | Method | Mean | lambda | Expansion coverage |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in p1_agg_rows:
        lines.append(
            f"| {row['dataset']} | {row['noise_rate']} | {row['method_label']} | "
            f"{fmt(row['mean'])} | {row.get('lambda_u', '')} | "
            f"{row.get('expansion_coverage_unlabeled_mean', '')} |"
        )
    lines.extend([
        "",
        "### P1 Gates",
        "| Noise | lambda | Gate | Mean delta | Positive | Pass-count | Decision |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in p1_gate_rows_:
        mean_delta = row["mean_delta"]
        mean_text = "" if mean_delta == "" else f"{100.0 * float(mean_delta):+.2f}pp"
        lines.append(
            f"| {row['noise_rate']} | {row['lambda_u']} | {row['gate']} | {mean_text} | "
            f"{row['positive_count']}/{row['n']} | {row['pass_count']}/{row['n']} | {row['decision']} |"
        )
    lines.extend([
        "",
        "### P1 Claim Boundary",
        "- P1 is seed0 only unless a gate passes; it is a smoke test, not a stable paper table.",
        "- GLNN here is an MLP-adapted noisy-GCN-teacher distillation baseline, not an official GLNN reproduction.",
        "- Clean labels are used only through final test accuracy and offline reporting, not in target construction.",
    ])
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    p0_rows = deduplicate(load_mlp5_anchor_rows() + load_p0_new_rows(run_dir))
    p1_rows = deduplicate(load_mlp5_p1_anchor_rows() + load_p1_new_rows(run_dir))
    all_rows = deduplicate(p0_rows + p1_rows)
    missing = sorted(expected_keys() - observed_keys(p0_rows))
    p0_agg_rows = aggregate(p0_rows)
    delta_rows = paired_deltas(p0_rows)
    p1_agg_rows = aggregate(p1_rows)
    p1_gates = p1_gate_rows(p1_rows)
    write_csv(run_dir / "mlp8_combined_raw.csv", all_rows)
    write_csv(run_dir / "mlp8_p0_combined_raw.csv", p0_rows)
    write_csv(run_dir / "mlp8_p0_aggregate.csv", p0_agg_rows)
    write_csv(run_dir / "mlp8_p0_paired_deltas.csv", delta_rows)
    write_csv(run_dir / "mlp8_p1_combined_raw.csv", p1_rows)
    write_csv(run_dir / "mlp8_p1_aggregate.csv", p1_agg_rows)
    write_csv(run_dir / "mlp8_p1_gates.csv", p1_gates)
    write_csv(
        run_dir / "mlp8_p0_completeness.csv",
        [{"scope": "p0", "expected": len(expected_keys()), "observed": len(p0_rows), "missing": len(missing)}],
    )
    write_report(run_dir / "mlp8_report.md", p0_rows, p0_agg_rows, delta_rows, missing)
    append_p1_report(run_dir / "mlp8_report.md", p1_rows, p1_agg_rows, p1_gates)
    print(f"MLP8_P0_EXPECTED={len(expected_keys())}")
    print(f"MLP8_P0_OBSERVED={len(p0_rows)}")
    print(f"MLP8_P0_MISSING={len(missing)}")
    print(f"MLP8_P1_OBSERVED={len(p1_rows)}")


if __name__ == "__main__":
    main()
