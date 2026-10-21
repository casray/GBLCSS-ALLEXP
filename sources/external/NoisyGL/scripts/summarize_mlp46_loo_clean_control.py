from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "mlp46_loo_clean_control_20260730"
RAW = RUN_DIR / "clean_uniform0p0.csv"
DATASETS = ("cora", "citeseer", "corafull", "coauthorcs", "amazonpho")
METHODS = (
    "frozengrace_gb_mlp_softdist_beta05_loo",
    "frozengrace_gb_mlp_softdist_beta05_loo_shuffled",
    "mlp_ce",
)
LOO = METHODS[0]
COMPARISONS = (
    ("loo_vs_mlp_ce", LOO, "mlp_ce"),
    ("loo_vs_loo_shuffled", LOO, METHODS[1]),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict[str, str]]) -> tuple[
    dict[tuple[str, int, str], dict[str, str]], list[dict[str, object]]
]:
    keyed: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        dataset = row["dataset"].lower()
        method = row["method"]
        seed = int(float(row["seed"]))
        if dataset not in DATASETS or method not in METHODS or seed not in range(5):
            continue
        if row["status"].lower() != "ok":
            raise ValueError(f"failed row: {dataset}, {method}, {seed}")
        if row["noise_type"].lower() != "uniform" or float(row["noise_rate"]) != 0:
            raise ValueError(f"wrong clean protocol: {dataset}, {method}, {seed}")
        if row["noise_scope"].lower() != "train":
            raise ValueError(f"wrong noise scope: {dataset}, {method}, {seed}")
        for field in (
            "modified_count",
            "modified_train_count",
            "modified_val_count",
            "modified_test_count",
        ):
            if int(float(row[field])) != 0:
                raise ValueError(f"clean-label modification in {field}: {row}")
        if method != "mlp_ce":
            if row.get("tie_mode", "") != "distribution_only":
                raise ValueError(f"non-v2 LOO row: {row}")
            if row.get("leave_one_out", "").lower() not in {"1", "true"}:
                raise ValueError(f"non-LOO row: {row}")
        key = (dataset, seed, method)
        if key in keyed:
            raise ValueError(f"duplicate row: {key}")
        keyed[key] = row

    audit: list[dict[str, object]] = []
    for dataset in DATASETS:
        for seed in range(5):
            group = [keyed[(dataset, seed, method)] for method in METHODS]
            hashes = {(row["split_hash"], row["noise_hash"]) for row in group}
            if len(hashes) != 1:
                raise ValueError(f"hash mismatch: {dataset}, seed {seed}")
            audit.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "complete": 1,
                    "hash_match": 1,
                    "split_hash": group[0]["split_hash"],
                    "noise_hash": group[0]["noise_hash"],
                }
            )
    if len(keyed) != len(DATASETS) * len(METHODS) * 5:
        raise ValueError(f"expected 75 rows, found {len(keyed)}")
    return keyed, audit


def summarize(rows: list[dict[str, str]]) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    keyed, audit = validate(rows)
    aggregate: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            values = [
                float(keyed[(dataset, seed, method)]["test"]) for seed in range(5)
            ]
            aggregate.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                }
            )
        for name, left, right in COMPARISONS:
            deltas = [
                float(keyed[(dataset, seed, left)]["test"])
                - float(keyed[(dataset, seed, right)]["test"])
                for seed in range(5)
            ]
            paired.append(
                {
                    "dataset": dataset,
                    "comparison": name,
                    "mean_delta": statistics.mean(deltas),
                    "std_delta": statistics.stdev(deltas),
                    "positive_seeds": sum(delta > 0 for delta in deltas),
                    **{f"seed_{seed}_delta": delta for seed, delta in enumerate(deltas)},
                }
            )
    return aggregate, paired, audit


def main() -> None:
    aggregate, paired, audit = summarize(read_csv(RAW))
    write_csv(RUN_DIR / "mlp46_aggregate.csv", aggregate)
    write_csv(RUN_DIR / "mlp46_paired_deltas.csv", paired)
    write_csv(RUN_DIR / "mlp46_protocol_audit.csv", audit)

    agg = {(r["dataset"], r["method"]): r for r in aggregate}
    delta = {(r["dataset"], r["comparison"]): r for r in paired}
    lines = [
        "# MLP-46 Tie-Safe LOO Clean-Label Control",
        "",
        "All values use five paired seeds and the final "
        "`distribution_only + leave_one_out=true` target.",
        "",
        "| Dataset | LOO-GB-LCSS | LOO-shuffled | MLP+CE | "
        "LOO-CE (pp) | LOO-shuffled (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        values = []
        for method in METHODS:
            row = agg[(dataset, method)]
            values.append(f"{100*row['mean']:.2f}±{100*row['std']:.2f}")
        ce_delta = 100 * delta[(dataset, "loo_vs_mlp_ce")]["mean_delta"]
        shuffled_delta = 100 * delta[
            (dataset, "loo_vs_loo_shuffled")
        ]["mean_delta"]
        lines.append(
            f"| {dataset} | {values[0]} | {values[1]} | {values[2]} | "
            f"{ce_delta:+.2f} | {shuffled_delta:+.2f} |"
        )
    lines.extend(
        [
            "",
            "This is a descriptive clean-label control. It is not used to "
            "tune the method or to alter the MLP-44 promotion gate.",
        ]
    )
    (RUN_DIR / "mlp46_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

