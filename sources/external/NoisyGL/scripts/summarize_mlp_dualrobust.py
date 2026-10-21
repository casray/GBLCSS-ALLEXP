from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp_dualrobust_20260701"
LEGACY_LABEL_DIR = REPO_ROOT / "runs" / "mlp_noiserobust_20260701"

METHODS = (
    "mlp_ce",
    "mlp_small_loss",
    "mlp_coteaching",
    "mlp_jocor",
    "mlp_elr",
    "mlp_sce",
    "frozengrace_gb_mlp_softdist",
    "frozengrace_gb_mlp_softdist_shuffled",
)
ABLATION_METHODS = (
    "mlp_ce",
    "featuregb_mlp_softdist",
    "frozengrace_gb_mlp_hard",
    "frozengrace_gb_mlp_softdist",
    "frozengrace_gb_mlp_softdist_shuffled",
)
METHOD_LABELS = {
    "mlp_ce": "MLP+CE",
    "mlp_small_loss": "MLP+small-loss",
    "mlp_coteaching": "MLP+co-teaching",
    "mlp_jocor": "MLP+JoCoR",
    "mlp_elr": "MLP+ELR",
    "mlp_sce": "MLP+SCE",
    "frozengrace_gb_mlp_softdist": "SoftDist-MLP",
    "frozengrace_gb_mlp_softdist_shuffled": "SoftDist shuffled",
    "featuregb_mlp_softdist": "w/o GRACE (X balls)",
    "frozengrace_gb_mlp_hard": "hard majority",
}
SOFTDIST = "frozengrace_gb_mlp_softdist"
ROBUST_BASELINES = ("mlp_small_loss", "mlp_coteaching", "mlp_jocor", "mlp_elr", "mlp_sce")
DATASETS = ("cora", "citeseer", "coauthorcs")
REGIMES = (
    ("uniform", 0.2),
    ("uniform", 0.4),
    ("uniform", 0.6),
    ("pair", 0.4),
    ("instance", 0.4),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value):
    if value is None or not str(value).strip():
        return ""
    return float(value)


def normalize_label_row(row: dict[str, str], source: str | Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in set(METHODS) | set(ABLATION_METHODS):
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    if not row.get("noise_type"):
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "noise_type": row["noise_type"].strip().lower(),
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "actual_train_noise_rate": _optional_float(row.get("actual_train_noise_rate")),
        "source": str(source),
    }


def normalize_structure_row(row: dict[str, str], source: str | Path) -> dict[str, object] | None:
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    test_text = (row.get("test") or "").strip()
    if not test_text:
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": row["method"].strip(),
        "seed": int(row["seed"]),
        "structure_mode": row["structure_mode"].strip(),
        "structure_p": float(row["structure_p"]),
        "test_acc": float(test_text),
        "delta_from_clean": float(row.get("delta_from_clean") or 0.0),
        "source": str(source),
    }


def _csv_files(run_dirs: list[Path]) -> list[Path]:
    files = []
    for run_dir in run_dirs:
        if run_dir.exists():
            files.extend(sorted(run_dir.glob("*.csv")))
    return files


def load_label_rows(run_dirs: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in _csv_files(run_dirs):
        if path.name.startswith(("mlp_dualrobust_", "structure_")):
            continue
        for row in read_csv(path):
            normalized = normalize_label_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows, key_fields=("dataset", "noise_type", "noise_rate", "method", "seed"))


def load_structure_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("structure*.csv")):
        for row in read_csv(path):
            normalized = normalize_structure_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows, key_fields=("dataset", "method", "seed", "structure_mode", "structure_p"))


def deduplicate(rows: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} from "
                    f"{previous.get('source', '')} vs {row['test_acc']} from {row.get('source', '')}"
                )
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda item: tuple(str(item[field]) for field in key_fields))


def aggregate(rows: list[dict[str, object]], fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups[key].append(float(row["test_acc"]))
    output = []
    for key, values in groups.items():
        item = {field: key[i] for i, field in enumerate(fields)}
        item.update(
            {
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
        output.append(item)
    return sorted(output, key=lambda item: tuple(str(item[field]) for field in fields))


def paired_label_deltas(rows: list[dict[str, object]], methods: tuple[str, ...]) -> list[dict[str, object]]:
    values = {
        (
            row["dataset"],
            row["noise_type"],
            float(row["noise_rate"]),
            row["method"],
            int(row["seed"]),
        ): float(row["test_acc"])
        for row in rows
        if row["method"] in methods
    }
    settings = sorted({key[:3] for key in values})
    output = []
    for dataset, noise_type, noise_rate in settings:
        for baseline in methods:
            if baseline == SOFTDIST:
                continue
            deltas = []
            seeds = []
            for seed in range(5):
                left = (dataset, noise_type, noise_rate, SOFTDIST, seed)
                right = (dataset, noise_type, noise_rate, baseline, seed)
                if left in values and right in values:
                    deltas.append(values[left] - values[right])
                    seeds.append(seed)
            if deltas:
                output.append(
                    {
                        "dataset": dataset,
                        "noise_type": noise_type,
                        "noise_rate": noise_rate,
                        "baseline": baseline,
                        "baseline_label": METHOD_LABELS[baseline],
                        "n": len(deltas),
                        "mean_delta": statistics.mean(deltas),
                        "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        "positive_count": sum(delta > 0 for delta in deltas),
                        "seeds": " ".join(map(str, seeds)),
                        "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                    }
                )
    return output


def expected_label_keys() -> set[tuple[str, str, float, str, int]]:
    return {
        (dataset, noise_type, noise_rate, method, seed)
        for dataset in DATASETS
        for noise_type, noise_rate in REGIMES
        for method in METHODS
        for seed in range(5)
    }


def expected_ablation_keys() -> set[tuple[str, str, float, str, int]]:
    return {
        (dataset, "uniform", 0.4, method, seed)
        for dataset in DATASETS
        for method in ABLATION_METHODS
        for seed in range(5)
    }


def expected_structure_keys() -> set[tuple[str, str, int, str, float]]:
    methods = ("gcn", "gin", SOFTDIST)
    clean = {
        (dataset, method, seed, "clean", 0.0)
        for dataset in DATASETS
        for method in methods
        for seed in range(5)
    }
    perturbed = {
        (dataset, method, seed, mode, p)
        for dataset in DATASETS
        for method in methods
        for seed in range(5)
        for mode in ("edge_drop", "isolated")
        for p in (0.25, 0.5, 0.75)
    }
    return clean | perturbed


def missing_label_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, float, str, int]]:
    observed = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"]))
        for row in rows
        if row["method"] in METHODS
    }
    return expected_label_keys() - observed


def missing_ablation_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, float, str, int]]:
    observed = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["method"], int(row["seed"]))
        for row in rows
        if row["method"] in ABLATION_METHODS
    }
    return expected_ablation_keys() - observed


def missing_structure_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, int, str, float]]:
    observed = {
        (row["dataset"], row["method"], int(row["seed"]), row["structure_mode"], float(row["structure_p"]))
        for row in rows
    }
    return expected_structure_keys() - observed


def judge_c1(deltas: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (row["dataset"], row["noise_type"], float(row["noise_rate"]), row["baseline"]): row
        for row in deltas
    }
    new_settings = [
        ("cora", "instance", 0.4),
        ("citeseer", "instance", 0.4),
        *[("coauthorcs", noise_type, noise_rate) for noise_type, noise_rate in REGIMES],
    ]
    output = []
    for dataset, noise_type, noise_rate in new_settings:
        wins = 0
        complete = 0
        for baseline in ROBUST_BASELINES:
            row = lookup.get((dataset, noise_type, noise_rate, baseline))
            if row and int(row["n"]) == 5:
                complete += 1
                if float(row["mean_delta"]) >= 0 and int(row["positive_count"]) >= 3:
                    wins += 1
        output.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "robust_baseline_wins": wins,
                "robust_baseline_complete": complete,
                "setting_pass": int(complete == len(ROBUST_BASELINES) and wins >= 3),
            }
        )
    return output


def judge_c2(structure_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate_rows = aggregate(structure_rows, ("dataset", "method", "structure_mode", "structure_p"))
    lookup = {
        (row["dataset"], row["method"], row["structure_mode"], float(row["structure_p"])): row
        for row in aggregate_rows
    }
    output = []
    for dataset in DATASETS:
        for mode in ("edge_drop", "isolated"):
            mlp_key = (dataset, SOFTDIST, mode, 0.75)
            mlp_delta = lookup.get(mlp_key, {}).get("mean", None)
            clean_key = (dataset, SOFTDIST, "clean", 0.0)
            clean_mean = lookup.get(clean_key, {}).get("mean", None)
            if mlp_delta is not None and clean_mean is not None:
                mlp_drop = float(mlp_delta) - float(clean_mean)
            else:
                mlp_drop = None
            gnn_drops = []
            for method in ("gcn", "gin"):
                key = (dataset, method, mode, 0.75)
                clean = lookup.get((dataset, method, "clean", 0.0), {}).get("mean", None)
                perturbed = lookup.get(key, {}).get("mean", None)
                if clean is not None and perturbed is not None:
                    gnn_drops.append(float(perturbed) - float(clean))
            output.append(
                {
                    "dataset": dataset,
                    "structure_mode": mode,
                    "softdist_drop_at_0p75": "" if mlp_drop is None else mlp_drop,
                    "min_gnn_drop_at_0p75": "" if not gnn_drops else min(gnn_drops),
                    "c2_pass": int(mlp_drop is not None and abs(mlp_drop) < 1e-9 and gnn_drops and min(gnn_drops) <= -0.02),
                }
            )
    return output


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


def _fmt(value, scale=100.0):
    if value == "" or value is None:
        return "NA"
    return f"{scale * float(value):.2f}"


def build_report(label_agg, label_deltas, c1_rows, structure_agg, c2_rows, ablation_deltas, missing) -> str:
    lines = [
        "# MLP Dual-Robust Report",
        "",
        "## Material Passport",
        "",
        "- Artifact type: experiment result",
        "- Protocol: percent split 0.2/0.1/0.7, seeds 0-4, train-only label noise, ROCm `cuda:0`",
        "- Label-noise table: X-only MLP backbone for all robust strategies",
        "- Structure-noise table: clean-label training, inference-time graph perturbation only",
        "- Claim discipline: clean labels are evaluation-only; structural noise is not used during training",
        "",
        "## Completeness",
        "",
        f"- Label main table missing rows: {len(missing['label'])} / {len(expected_label_keys())}",
        f"- Ablation missing rows: {len(missing['ablation'])} / {len(expected_ablation_keys())}",
        f"- Structure table missing rows: {len(missing['structure'])} / {len(expected_structure_keys())}",
        "",
        "## Main Table 1: Label Noise Robustness",
        "",
        "| Dataset | Noise | Method | n | Test acc mean±std |",
        "|---|---:|---|---:|---:|",
    ]
    method_order = {method: i for i, method in enumerate(METHODS)}
    for row in sorted(
        [item for item in label_agg if item["method"] in METHODS],
        key=lambda item: (item["dataset"], item["noise_type"], float(item["noise_rate"]), method_order[item["method"]]),
    ):
        lines.append(
            "| {dataset} | {noise}{rate:.1f} | {method} | {n} | {mean}±{std} |".format(
                dataset=row["dataset"],
                noise=row["noise_type"],
                rate=float(row["noise_rate"]),
                method=METHOD_LABELS[row["method"]],
                n=row["n"],
                mean=_fmt(row["mean"]),
                std=_fmt(row["std"]),
            )
        )
    lines.extend(
        [
            "",
            "## C1 Label Robustness Gate",
            "",
            "| Dataset | Noise | Robust baselines beaten | Complete | Pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in c1_rows:
        lines.append(
            "| {dataset} | {noise}{rate:.1f} | {wins}/5 | {complete}/5 | {passed} |".format(
                dataset=row["dataset"],
                noise=row["noise_type"],
                rate=float(row["noise_rate"]),
                wins=row["robust_baseline_wins"],
                complete=row["robust_baseline_complete"],
                passed=row["setting_pass"],
            )
        )
    complete_c1 = [row for row in c1_rows if row["robust_baseline_complete"] == len(ROBUST_BASELINES)]
    c1_pass_count = sum(int(row["setting_pass"]) for row in complete_c1)
    lines.append("")
    lines.append(
        f"- C1 summary: {c1_pass_count}/{len(complete_c1)} complete new settings pass "
        "(pass means SoftDist beats at least 3/5 robust MLP strategies with paired seed support)."
    )
    lines.extend(
        [
            "",
            "## Main Table 2: Structure Noise Robustness",
            "",
            "| Dataset | Method | Mode | p | n | Test acc mean±std |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        structure_agg,
        key=lambda item: (item["dataset"], item["method"], item["structure_mode"], float(item["structure_p"])),
    ):
        lines.append(
            "| {dataset} | {method} | {mode} | {p:.2f} | {n} | {mean}±{std} |".format(
                dataset=row["dataset"],
                method=METHOD_LABELS.get(row["method"], row["method"]),
                mode=row["structure_mode"],
                p=float(row["structure_p"]),
                n=row["n"],
                mean=_fmt(row["mean"]),
                std=_fmt(row["std"]),
            )
        )
    lines.extend(
        [
            "",
            "## C2 Structure Robustness Gate",
            "",
            "| Dataset | Mode | SoftDist drop @0.75 | Worst GNN drop @0.75 | Pass |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in c2_rows:
        lines.append(
            "| {dataset} | {mode} | {mlp} | {gnn} | {passed} |".format(
                dataset=row["dataset"],
                mode=row["structure_mode"],
                mlp=_fmt(row["softdist_drop_at_0p75"]),
                gnn=_fmt(row["min_gnn_drop_at_0p75"]),
                passed=row["c2_pass"],
            )
        )
    lines.extend(
        [
            "",
            "## Ablation: SoftDist vs Components",
            "",
            "| Dataset | Baseline removed/changed | n | SoftDist delta mean±std | Seedwise wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        [item for item in ablation_deltas if item["noise_type"] == "uniform" and float(item["noise_rate"]) == 0.4],
        key=lambda item: (item["dataset"], item["baseline"]),
    ):
        lines.append(
            "| {dataset} | {baseline} | {n} | {mean}±{std} | {wins}/5 |".format(
                dataset=row["dataset"],
                baseline=row["baseline_label"],
                n=row["n"],
                mean=_fmt(row["mean_delta"]),
                std=_fmt(row["std_delta"]),
                wins=row["positive_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- C1 decides whether label-robustness extends beyond the MLP-1 Cora/CiteSeer grid.",
            "- C2 tests the structural half of the dual-robust framing; SoftDist-MLP should be invariant because inference ignores adjacency.",
            "- C3 is supported only if GRACE balls, soft targets, and real ball structure all beat their removals on paired seeds.",
        ]
    )
    if missing["label"] or missing["ablation"] or missing["structure"]:
        lines.extend(
            [
                "",
                "## Missing Keys",
                "",
                f"- Label first 20: {sorted(missing['label'])[:20]}",
                f"- Ablation first 20: {sorted(missing['ablation'])[:20]}",
                f"- Structure first 20: {sorted(missing['structure'])[:20]}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize MLP-2 dual-robust experiments.")
    parser.add_argument("--run_dir", default=str(RUN_DIR))
    parser.add_argument("--legacy_label_dir", default=str(LEGACY_LABEL_DIR))
    parser.add_argument("--require_complete", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    legacy_dir = Path(args.legacy_label_dir)
    label_rows = load_label_rows([legacy_dir, run_dir])
    structure_rows = load_structure_rows(run_dir)
    label_agg = aggregate(label_rows, ("dataset", "noise_type", "noise_rate", "method"))
    label_deltas = paired_label_deltas(label_rows, METHODS)
    ablation_deltas = paired_label_deltas(label_rows, ABLATION_METHODS)
    structure_agg = aggregate(structure_rows, ("dataset", "method", "structure_mode", "structure_p"))
    c1_rows = judge_c1(label_deltas)
    c2_rows = judge_c2(structure_rows)
    missing = {
        "label": missing_label_keys(label_rows),
        "ablation": missing_ablation_keys(label_rows),
        "structure": missing_structure_keys(structure_rows),
    }

    write_csv(run_dir / "mlp_dualrobust_label_aggregate.csv", label_agg)
    write_csv(run_dir / "mlp_dualrobust_label_paired_deltas.csv", label_deltas)
    write_csv(run_dir / "mlp_dualrobust_c1_gate.csv", c1_rows)
    write_csv(run_dir / "mlp_dualrobust_structure_aggregate.csv", structure_agg)
    write_csv(run_dir / "mlp_dualrobust_c2_gate.csv", c2_rows)
    write_csv(run_dir / "mlp_dualrobust_ablation_paired_deltas.csv", ablation_deltas)
    report = build_report(label_agg, label_deltas, c1_rows, structure_agg, c2_rows, ablation_deltas, missing)
    (run_dir / "mlp_dualrobust_report.md").write_text(report, encoding="utf-8")
    print(
        "SUMMARY rows_label={} rows_structure={} missing_label={} missing_ablation={} missing_structure={}".format(
            len(label_rows),
            len(structure_rows),
            len(missing["label"]),
            len(missing["ablation"]),
            len(missing["structure"]),
        )
    )
    if args.require_complete and any(missing.values()):
        raise SystemExit("Dual-robust experiment grid is incomplete")


if __name__ == "__main__":
    main()
