import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DATASETS = ("cora", "citeseer", "corafull", "amazonpho", "pubmed", "coauthorph")
DIRECT = "frozengrace_gb_mlp_softdist_beta05_small_loss"
PLAIN = "mlp_distill_gcn_xh"
NRGNN = "nrgnn"
NRGNN_XH = "nrgnn_xh"
METHODS = (DIRECT, PLAIN, NRGNN, NRGNN_XH)
SEEDS = tuple(range(5))
RESOURCE = {
    DIRECT: ("GRACE preprocessing", "no", "no", "X-only direct"),
    PLAIN: ("GCN[X||H] teacher", "no", "no", "X-only distilled student"),
    NRGNN: ("yes", "yes", "no", "graph-native robust GNN"),
    NRGNN_XH: ("yes", "yes", "yes", "graph-native robust GNN with H"),
}


def build_summary(rows):
    keyed = {}
    selected = []
    for row in rows:
        if row.get("dataset") not in DATASETS or row.get("method") not in METHODS:
            continue
        if row.get("status") != "ok":
            continue
        if row.get("noise_type") != "uniform" or abs(float(row["noise_rate"]) - 0.4) > 1e-12:
            continue
        if row.get("noise_scope") != "train":
            raise ValueError("resource table requires train-only noise")
        seed = int(row["seed"])
        if seed not in SEEDS:
            continue
        if int(row.get("modified_val_count") or 0) or int(row.get("modified_test_count") or 0):
            raise ValueError("validation/test labels were modified")
        if int(row.get("modified_count") or 0) != int(row.get("modified_train_count") or 0):
            raise ValueError("modified labels are not restricted to train")
        key = (row["dataset"], row["method"], seed)
        if key in keyed:
            raise ValueError(f"duplicate row: {key}")
        normalized = dict(row)
        normalized["seed"] = seed
        normalized["test"] = float(row["test"])
        keyed[key] = normalized
        selected.append(normalized)

    for dataset in DATASETS:
        for seed in SEEDS:
            hashes = set()
            for method in METHODS:
                key = (dataset, method, seed)
                if key not in keyed:
                    raise ValueError(f"missing row: {key}")
                hashes.add((keyed[key]["split_hash"], keyed[key]["noise_hash"]))
            if len(hashes) != 1:
                raise ValueError(f"protocol hash mismatch: {dataset} seed={seed}")

    selected.sort(key=lambda row: (DATASETS.index(row["dataset"]), METHODS.index(row["method"]), row["seed"]))
    grouped = defaultdict(list)
    times = defaultdict(list)
    for row in selected:
        key = (row["dataset"], row["method"])
        grouped[key].append(row["test"])
        if row.get("wall_time_seconds") not in (None, ""):
            times[key].append(float(row["wall_time_seconds"]))
    aggregate = []
    for dataset in DATASETS:
        for method in METHODS:
            values = grouped[(dataset, method)]
            training_graph, inference_graph, h_inference, note = RESOURCE[method]
            aggregate.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values),
                    "mean_wall_time_seconds": statistics.mean(times[(dataset, method)]) if times[(dataset, method)] else "",
                    "training_graph": training_graph,
                    "inference_graph": inference_graph,
                    "h_at_inference": h_inference,
                    "resource_note": note,
                }
            )

    paired = []
    for dataset in DATASETS:
        plain_nrgnn = []
        plain_nrgnn_xh = []
        direct_nrgnn = []
        for seed in SEEDS:
            value = lambda method: keyed[(dataset, method, seed)]["test"]
            plain_nrgnn.append(value(PLAIN) - value(NRGNN))
            plain_nrgnn_xh.append(value(PLAIN) - value(NRGNN_XH))
            direct_nrgnn.append(value(DIRECT) - value(NRGNN))
        paired.append(
            {
                "dataset": dataset,
                "plain_minus_nrgnn": statistics.mean(plain_nrgnn),
                "plain_minus_nrgnn_positive": sum(value > 0 for value in plain_nrgnn),
                "plain_minus_nrgnn_xh": statistics.mean(plain_nrgnn_xh),
                "plain_minus_nrgnn_xh_positive": sum(value > 0 for value in plain_nrgnn_xh),
                "direct_minus_nrgnn": statistics.mean(direct_nrgnn),
                "direct_minus_nrgnn_positive": sum(value > 0 for value in direct_nrgnn),
            }
        )
    return selected, aggregate, paired


def _read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report(path, aggregate, paired):
    lookup = {(row["dataset"], row["method"]): row for row in aggregate}
    lines = [
        "# MLP-32 Uniform0.4 Cross-Resource Five-Seed Table",
        "",
        "| Dataset | Direct robust X-only | Plain distill X-only | NRGNN | NRGNN[X||H] | Distill-NRGNN | Positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for delta in paired:
        dataset = delta["dataset"]
        fmt = lambda method: f"{100*lookup[(dataset, method)]['mean']:.2f}+/-{100*lookup[(dataset, method)]['std']:.2f}"
        lines.append(
            f"| {dataset} | {fmt(DIRECT)} | {fmt(PLAIN)} | {fmt(NRGNN)} | {fmt(NRGNN_XH)} | "
            f"{100*delta['plain_minus_nrgnn']:+.2f}pp | {delta['plain_minus_nrgnn_positive']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Resource profiles",
            "",
            "| Method | Training graph | Inference graph | H at inference |",
            "|---|---|---|---|",
        ]
    )
    for method in METHODS:
        training_graph, inference_graph, h_inference, _ = RESOURCE[method]
        lines.append(f"| {method} | {training_graph} | {inference_graph} | {h_inference} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(_read(path))
    combined, aggregate, paired = build_summary(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "resource_table_raw.csv", combined)
    _write(output / "resource_table_aggregate.csv", aggregate)
    _write(output / "resource_table_paired.csv", paired)
    _report(output / "resource_table_report.md", aggregate, paired)


if __name__ == "__main__":
    main()
