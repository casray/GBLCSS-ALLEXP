from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp45_loo_grouping_pareto_20260730"
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")

GB_METHOD = "frozengrace_gb_mlp_softdist_beta05_loo"
KNN_METHOD = "knn_mlp_softdist_beta05"
KMEANS_METHOD = "kmeans_mlp_softdist_beta05_loo"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _accuracy_rows() -> list[dict]:
    gb = [
        row
        for row in read_csv(
            REPO_ROOT
            / "runs"
            / "mlp44_loo_tiesafe_fullspectrum_20260730"
            / "mlp44_combined_raw.csv"
        )
        if row["dataset"] in DATASETS
        and row["method"] == GB_METHOD
        and row["noise_type"] == "uniform"
        and abs(float(row["noise_rate"]) - 0.4) < 1e-12
    ]
    knn = [
        row
        for row in read_csv(
            REPO_ROOT
            / "runs"
            / "mlp43_cold_review_defense_20260729"
            / "grouping_alternatives_5seed.csv"
        )
        if row["dataset"] in DATASETS and row["method"] == KNN_METHOD
    ]
    kmeans = []
    for filename in (
        "kmeans_loo_smoke_seed0.csv",
        "kmeans_loo_seeds_1_2_3_4.csv",
    ):
        kmeans.extend(read_csv(RUN_DIR / filename))
    rows = gb + knn + kmeans

    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in (GB_METHOD, KNN_METHOD, KMEANS_METHOD)
        for seed in range(5)
    }
    keyed = {}
    for row in rows:
        key = (row["dataset"], row["method"], int(row["seed"]))
        if key in keyed:
            raise ValueError(f"duplicate accuracy row: {key}")
        keyed[key] = row
        if row.get("status") != "ok":
            raise ValueError(f"failed accuracy row: {key}")
        if row.get("tie_mode") != "distribution_only":
            raise ValueError(f"non-tie-safe row: {key}")
        if row["method"] == GB_METHOD and row.get("leave_one_out") != "1":
            raise ValueError(f"GB row is not LOO: {key}")
        if row["method"] == KNN_METHOD and row.get("knn_exclude_self") != "1":
            raise ValueError(f"kNN row includes the query node: {key}")
        if row["method"] == KMEANS_METHOD and row.get("leave_one_out") != "1":
            raise ValueError(f"k-means row is not LOO: {key}")
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"accuracy completeness failure missing={missing} extra={extra}")

    for dataset in DATASETS:
        for seed in range(5):
            selected = [
                keyed[(dataset, method, seed)]
                for method in (GB_METHOD, KNN_METHOD, KMEANS_METHOD)
            ]
            if len({row["split_hash"] for row in selected}) != 1:
                raise ValueError(f"split hash mismatch: {dataset}, seed={seed}")
            if len({row["noise_hash"] for row in selected}) != 1:
                raise ValueError(f"noise hash mismatch: {dataset}, seed={seed}")
    return [keyed[key] for key in sorted(keyed)]


def main():
    accuracy = _accuracy_rows()
    cost = read_csv(RUN_DIR / "grouping_cost_benchmark.csv")
    write_csv(RUN_DIR / "mlp45_accuracy_raw.csv", accuracy)

    grouped = defaultdict(list)
    for row in accuracy:
        grouped[(row["dataset"], row["method"])].append(float(row["test"]))
    aggregate = []
    for (dataset, method), values in sorted(grouped.items()):
        aggregate.append({
            "dataset": dataset,
            "method": method,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "seeds": len(values),
        })
    write_csv(RUN_DIR / "mlp45_accuracy_aggregate.csv", aggregate)

    keyed = {
        (row["dataset"], row["method"], int(row["seed"])): float(row["test"])
        for row in accuracy
    }
    deltas = []
    dataset_summary = []
    within_one_pp = 0
    best_gaps = []
    for dataset in DATASETS:
        gb = [keyed[(dataset, GB_METHOD, seed)] for seed in range(5)]
        knn = [keyed[(dataset, KNN_METHOD, seed)] for seed in range(5)]
        kmeans = [keyed[(dataset, KMEANS_METHOD, seed)] for seed in range(5)]
        knn_delta = [knn[i] - gb[i] for i in range(5)]
        kmeans_delta = [kmeans[i] - gb[i] for i in range(5)]
        best_alt_mean = max(statistics.mean(knn), statistics.mean(kmeans))
        gap_to_best = statistics.mean(gb) - best_alt_mean
        best_gaps.append(gap_to_best)
        within_one_pp += int(gap_to_best >= -0.01)
        dataset_summary.append({
            "dataset": dataset,
            "loo_gb_mean": statistics.mean(gb),
            "loo_gb_std": statistics.stdev(gb),
            "exact_knn_mean": statistics.mean(knn),
            "exact_knn_std": statistics.stdev(knn),
            "loo_kmeans_mean": statistics.mean(kmeans),
            "loo_kmeans_std": statistics.stdev(kmeans),
            "gb_minus_best_alternative": gap_to_best,
            "gb_vs_knn_positive_seeds": sum(delta < 0 for delta in knn_delta),
            "gb_vs_kmeans_positive_seeds": sum(
                delta < 0 for delta in kmeans_delta
            ),
        })
        for seed in range(5):
            deltas.append({
                "dataset": dataset,
                "seed": seed,
                "gb_minus_exact_knn": gb[seed] - knn[seed],
                "gb_minus_loo_kmeans": gb[seed] - kmeans[seed],
            })
    write_csv(RUN_DIR / "mlp45_accuracy_paired_deltas.csv", deltas)
    write_csv(RUN_DIR / "mlp45_accuracy_dataset_summary.csv", dataset_summary)

    cost_by_key = {(row["dataset"], row["method"]): row for row in cost}
    faster = 0
    lower_memory = 0
    lower_storage = 0
    for dataset in DATASETS:
        gb = cost_by_key[(dataset, "loo_gb")]
        knn = cost_by_key[(dataset, "exact_knn")]
        faster += int(
            float(gb["end_to_end_structure_seconds"])
            < float(knn["end_to_end_structure_seconds"])
        )
        lower_memory += int(
            float(gb["peak_rss_delta_bytes"])
            < float(knn["peak_rss_delta_bytes"])
        )
        lower_storage += int(
            float(gb["structure_storage_bytes"])
            < float(knn["structure_storage_bytes"])
        )

    mean_gap = statistics.mean(best_gaps)
    accuracy_ok = mean_gap >= -0.01 and within_one_pp >= 3
    cost_ok = faster >= 4 or lower_memory >= 4
    if accuracy_ok and cost_ok:
        decision = "fixed_budget_pareto_supported"
    elif mean_gap < -0.015 and faster < 4 and lower_memory < 4:
        decision = "alternative_dominates"
    else:
        decision = "implementation_profile_only"

    decision_row = {
        "decision": decision,
        "mean_gb_minus_best_alternative": mean_gap,
        "datasets_within_1pp": within_one_pp,
        "gb_faster_than_exact_knn": faster,
        "gb_lower_peak_memory_than_exact_knn": lower_memory,
        "gb_lower_structure_storage_than_exact_knn": lower_storage,
        "approximate_knn_available": 0,
    }
    write_csv(RUN_DIR / "mlp45_decision.csv", [decision_row])

    names = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "corafull": "CoraFull",
        "coauthorcs": "Coauthor-CS",
        "amazonpho": "Amazon-Photo",
    }
    lines = [
        "# MLP-45 LOO Grouping Accuracy-Cost Pareto Report",
        "",
        "All accuracy rows use uniform 0.4, train-only noise, five paired seeds, "
        "tie-safe targets, and the same frozen GRACE representation. GB and "
        "k-means use leave-one-out votes; kNN excludes the query node.",
        "",
        "| Dataset | LOO-GB | Exact kNN | LOO-k-means | GB-best alternative |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in dataset_summary:
        lines.append(
            f"| {names[row['dataset']]} | "
            f"{100*row['loo_gb_mean']:.2f}+/-{100*row['loo_gb_std']:.2f} | "
            f"{100*row['exact_knn_mean']:.2f}+/-{100*row['exact_knn_std']:.2f} | "
            f"{100*row['loo_kmeans_mean']:.2f}+/-{100*row['loo_kmeans_std']:.2f} | "
            f"{100*row['gb_minus_best_alternative']:+.2f}pp |"
        )
    lines.extend([
        "",
        "## Cost summary against exact kNN",
        "",
        f"- GB has lower end-to-end construction time on {faster}/5 datasets.",
        f"- GB has lower measured peak RSS on {lower_memory}/5 datasets.",
        f"- GB has lower grouping-structure storage on {lower_storage}/5 datasets.",
        "- No maintained approximate-kNN package was available in the frozen "
        "environment, so no approximate-kNN timing is claimed.",
        "",
        "## Preregistered decision",
        "",
        f"- Decision: `{decision}`.",
        f"- Mean GB gap to the best alternative: {100*mean_gap:+.2f}pp.",
        f"- Datasets within 1pp of the best alternative: {within_one_pp}/5.",
        "",
        "Granular balls remain an auditable fixed-budget compression of local "
        "consensus, but this experiment does not establish a general accuracy "
        "or runtime advantage over exact kNN.",
        "",
    ])
    (RUN_DIR / "mlp45_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        f"decision={decision}; mean_gap_pp={100*mean_gap:+.2f}; "
        f"within_1pp={within_one_pp}/5"
    )


if __name__ == "__main__":
    main()
