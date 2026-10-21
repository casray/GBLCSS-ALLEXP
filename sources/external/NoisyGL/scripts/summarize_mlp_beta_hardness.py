from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "runs" / "mlp_beta_hardness_20260701"

METHODS = (
    "frozengrace_gb_mlp_softdist",
    "frozengrace_gb_mlp_softdist_beta05",
    "frozengrace_gb_mlp_softdist_beta1",
    "frozengrace_gb_mlp_softdist_beta1_shuffled",
)
REAL_METHODS = METHODS[:3]
SOFT = "frozengrace_gb_mlp_softdist"
HARD = "frozengrace_gb_mlp_softdist_beta1"
SHUFFLED = "frozengrace_gb_mlp_softdist_beta1_shuffled"
METHOD_LABELS = {
    "frozengrace_gb_mlp_softdist": "beta=0 soft distribution",
    "frozengrace_gb_mlp_softdist_beta05": "beta=0.5 hybrid",
    "frozengrace_gb_mlp_softdist_beta1": "beta=1 hard-majority target",
    "frozengrace_gb_mlp_softdist_beta1_shuffled": "beta=1 shuffled balls",
}
METHOD_BETA = {
    "frozengrace_gb_mlp_softdist": 0.0,
    "frozengrace_gb_mlp_softdist_beta05": 0.5,
    "frozengrace_gb_mlp_softdist_beta1": 1.0,
    "frozengrace_gb_mlp_softdist_beta1_shuffled": 1.0,
}
NOISE_RATES = (0.2, 0.4)


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as stream:
        return list(csv.DictReader(stream))


def normalize_row(row: dict[str, str], source: Path) -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or "").strip()
    if method not in METHODS:
        return None
    if row.get("status", "ok").strip().lower() not in {"", "ok"}:
        return None
    if (row.get("noise_type") or "").strip().lower() != "uniform":
        return None
    test_text = (row.get("test") or row.get("test_acc") or "").strip()
    if not test_text:
        return None
    return {
        "dataset": row["dataset"].strip().lower(),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "target_beta": METHOD_BETA[method],
        "noise_type": "uniform",
        "noise_rate": float(row["noise_rate"]),
        "seed": int(row["seed"]),
        "test_acc": float(test_text),
        "source": str(source),
    }


def deduplicate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = (row["dataset"], row["noise_rate"], row["method"], row["seed"])
        if key in unique:
            previous = unique[key]
            if not math.isclose(float(previous["test_acc"]), float(row["test_acc"]), abs_tol=1e-12):
                raise ValueError(
                    f"conflicting duplicate for {key}: {previous['test_acc']} from "
                    f"{previous.get('source', '')} vs {row['test_acc']} from {row.get('source', '')}"
                )
            continue
        unique[key] = row
    return sorted(unique.values(), key=lambda item: (str(item["dataset"]), float(item["noise_rate"]), str(item["method"]), int(item["seed"])))


def load_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(run_dir.glob("*.csv")):
        if path.name.startswith(("mlp_beta_hardness_", "paired_", "decision_")):
            continue
        for row in read_csv(path):
            normalized = normalize_row(row, path)
            if normalized is not None:
                rows.append(normalized)
    return deduplicate(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    betas: dict[str, float] = {}
    for row in rows:
        key = (row["dataset"], row["noise_type"], row["noise_rate"], row["method"])
        groups[key].append(float(row["test_acc"]))
        labels[str(row["method"])] = str(row["method_label"])
        betas[str(row["method"])] = float(row["target_beta"])
    output = []
    for key, values in groups.items():
        dataset, noise_type, noise_rate, method = key
        output.append(
            {
                "dataset": dataset,
                "noise_type": noise_type,
                "noise_rate": noise_rate,
                "method": method,
                "method_label": labels[str(method)],
                "target_beta": betas[str(method)],
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda item: (str(item["dataset"]), float(item["noise_rate"]), str(item["method"])))


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (row["dataset"], float(row["noise_rate"]), row["method"], int(row["seed"])): float(row["test_acc"])
        for row in rows
    }
    settings = sorted({(row["dataset"], float(row["noise_rate"])) for row in rows})
    comparisons = (
        ("beta05_minus_soft", "frozengrace_gb_mlp_softdist_beta05", SOFT),
        ("beta1_minus_soft", HARD, SOFT),
        ("beta1_minus_beta05", HARD, "frozengrace_gb_mlp_softdist_beta05"),
        ("beta1_real_minus_shuffled", HARD, SHUFFLED),
    )
    output = []
    for dataset, noise_rate in settings:
        seeds = sorted({seed for d, r, _, seed in values if d == dataset and math.isclose(r, noise_rate)})
        for name, left_method, right_method in comparisons:
            deltas = []
            used_seeds = []
            for seed in seeds:
                left = (dataset, noise_rate, left_method, seed)
                right = (dataset, noise_rate, right_method, seed)
                if left in values and right in values:
                    deltas.append(values[left] - values[right])
                    used_seeds.append(seed)
            if deltas:
                output.append(
                    {
                        "dataset": dataset,
                        "noise_type": "uniform",
                        "noise_rate": noise_rate,
                        "comparison": name,
                        "left_method": left_method,
                        "right_method": right_method,
                        "n": len(deltas),
                        "mean_delta": statistics.mean(deltas),
                        "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                        "positive_count": sum(delta > 0 for delta in deltas),
                        "seeds": " ".join(str(seed) for seed in used_seeds),
                        "seed_deltas": " ".join(f"{delta:.6f}" for delta in deltas),
                    }
                )
    return output


def _delta_lookup(deltas: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, object]]:
    return {
        (row["dataset"], float(row["noise_rate"]), row["comparison"]): row
        for row in deltas
    }


def _stable_positive(delta_row: dict[str, object] | None, threshold: int) -> bool:
    if not delta_row:
        return False
    return (
        int(delta_row["n"]) >= threshold
        and int(delta_row["positive_count"]) >= threshold
        and float(delta_row["mean_delta"]) >= 0.0
    )


def _stable_negative(delta_row: dict[str, object] | None, threshold: int) -> bool:
    if not delta_row:
        return False
    n = int(delta_row["n"])
    positive = int(delta_row["positive_count"])
    return n >= threshold and (n - positive) >= threshold and float(delta_row["mean_delta"]) <= 0.0


def decision_rows(
    agg: list[dict[str, object]],
    datasets: list[str],
    deltas: list[dict[str, object]] | None = None,
    stability_threshold: int = 4,
) -> list[dict[str, object]]:
    values = {
        (row["dataset"], float(row["noise_rate"]), row["method"]): float(row["mean"])
        for row in agg
    }
    delta_values = _delta_lookup(deltas or [])
    output = []
    for dataset in datasets:
        for noise_rate in NOISE_RATES:
            setting = {"dataset": dataset, "noise_type": "uniform", "noise_rate": noise_rate}
            real_values = {
                method: values.get((dataset, noise_rate, method))
                for method in REAL_METHODS
            }
            if any(value is None for value in real_values.values()):
                setting.update({"best_method": "", "best_beta": "", "beta1_minus_soft": "", "beta05_minus_soft": "", "beta1_real_minus_shuffled": "", "decision": "missing"})
                output.append(setting)
                continue
            best_method = max(real_values, key=lambda method: float(real_values[method]))
            beta1_minus_soft = float(real_values[HARD]) - float(real_values[SOFT])
            beta05_minus_soft = float(real_values["frozengrace_gb_mlp_softdist_beta05"]) - float(real_values[SOFT])
            shuffled_value = values.get((dataset, noise_rate, SHUFFLED))
            real_minus_shuffled = "" if shuffled_value is None else float(real_values[HARD]) - float(shuffled_value)
            beta1_soft_delta = delta_values.get((dataset, noise_rate, "beta1_minus_soft"))
            beta05_soft_delta = delta_values.get((dataset, noise_rate, "beta05_minus_soft"))
            beta1_beta05_delta = delta_values.get((dataset, noise_rate, "beta1_minus_beta05"))
            beta1_shuffled_delta = delta_values.get((dataset, noise_rate, "beta1_real_minus_shuffled"))
            beta1_ge_soft_stable = _stable_positive(beta1_soft_delta, stability_threshold)
            beta05_ge_soft_stable = _stable_positive(beta05_soft_delta, stability_threshold)
            beta1_ge_beta05_stable = _stable_positive(beta1_beta05_delta, stability_threshold)
            beta05_ge_beta1_stable = _stable_negative(beta1_beta05_delta, stability_threshold)
            beta1_gt_shuffled_stable = _stable_positive(beta1_shuffled_delta, stability_threshold)
            if beta1_ge_soft_stable and beta1_ge_beta05_stable and beta1_gt_shuffled_stable:
                decision = "hard_preferred_5seed"
            elif beta1_minus_soft > 0 and (real_minus_shuffled == "" or real_minus_shuffled > 0):
                decision = "hardening_positive_seed0"
            elif beta05_ge_soft_stable and beta05_ge_beta1_stable:
                decision = "mid_preferred_5seed"
            elif beta05_minus_soft > 0 and best_method == "frozengrace_gb_mlp_softdist_beta05":
                decision = "mid_beta_positive_seed0"
            elif float(real_values[SOFT]) >= max(float(real_values[method]) for method in REAL_METHODS if method != SOFT):
                decision = "soft_retained_seed0"
            else:
                decision = "mixed_seed0"
            setting.update(
                {
                    "best_method": best_method,
                    "best_beta": METHOD_BETA[best_method],
                    "soft": real_values[SOFT],
                    "beta05": real_values["frozengrace_gb_mlp_softdist_beta05"],
                    "beta1": real_values[HARD],
                    "beta1_shuffled": "" if shuffled_value is None else shuffled_value,
                    "beta1_minus_soft": beta1_minus_soft,
                    "beta05_minus_soft": beta05_minus_soft,
                    "beta1_real_minus_shuffled": real_minus_shuffled,
                    "beta1_minus_soft_positive_count": "" if beta1_soft_delta is None else int(beta1_soft_delta["positive_count"]),
                    "beta05_minus_soft_positive_count": "" if beta05_soft_delta is None else int(beta05_soft_delta["positive_count"]),
                    "beta1_minus_beta05_positive_count": "" if beta1_beta05_delta is None else int(beta1_beta05_delta["positive_count"]),
                    "beta1_real_minus_shuffled_positive_count": "" if beta1_shuffled_delta is None else int(beta1_shuffled_delta["positive_count"]),
                    "paired_n": "" if beta1_soft_delta is None else int(beta1_soft_delta["n"]),
                    "beta1_ge_soft_stable": int(beta1_ge_soft_stable),
                    "beta05_ge_soft_stable": int(beta05_ge_soft_stable),
                    "beta1_ge_beta05_stable": int(beta1_ge_beta05_stable),
                    "beta05_ge_beta1_stable": int(beta05_ge_beta1_stable),
                    "beta1_gt_shuffled_stable": int(beta1_gt_shuffled_stable),
                    "decision": decision,
                }
            )
            output.append(setting)
    return output


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


def fmt(value: object, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _overall_conclusion(decisions: list[dict[str, object]], datasets: list[str], seed_count: int) -> tuple[str, list[str]]:
    complete_settings = [row for row in decisions if row.get("decision") != "missing"]
    if not complete_settings:
        return "No complete setting was available, so the beta spectrum cannot be judged.", []
    if seed_count <= 1:
        beta1_wins = sum(1 for row in complete_settings if row.get("best_method") == HARD)
        beta05_wins = sum(1 for row in complete_settings if row.get("best_method") == "frozengrace_gb_mlp_softdist_beta05")
        soft_wins = sum(1 for row in complete_settings if row.get("best_method") == SOFT)
        hard_positive = sum(1 for row in complete_settings if row.get("decision") == "hardening_positive_seed0")
        shuffled_positive = sum(
            1
            for row in complete_settings
            if row.get("beta1_real_minus_shuffled", "") != "" and float(row["beta1_real_minus_shuffled"]) > 0
        )
        details = [
            f"Best beta counts among real balls: beta=0: {soft_wins}, beta=0.5: {beta05_wins}, beta=1: {beta1_wins}.",
            f"Hardening-positive settings with beta=1 > beta=0 and beta=1 > shuffled: {hard_positive}.",
            f"Real beta=1 above shuffled settings: {shuffled_positive}/{len(complete_settings)}.",
        ]
        if beta1_wins >= 4 and hard_positive >= 4 and shuffled_positive >= 4:
            conclusion = (
                "Seed0 supports hardening as the next main-method candidate, but this still requires 5-seed confirmation."
            )
        elif beta05_wins >= 4 and beta05_wins >= max(beta1_wins, soft_wins):
            conclusion = "Seed0 favors an intermediate target; treat beta=0.5 as a candidate only after 5-seed confirmation."
        elif soft_wins >= 4 and soft_wins >= max(beta1_wins, beta05_wins):
            conclusion = "Seed0 does not justify replacing beta=0."
        else:
            conclusion = "Seed0 is heterogeneous across datasets or noise rates; do not choose a fixed beta from this smoke alone."
        return conclusion, details

    hard_datasets = 0
    mid_datasets = 0
    for dataset in datasets:
        dataset_rows = [row for row in complete_settings if row["dataset"] == dataset]
        if len(dataset_rows) == len(NOISE_RATES) and all(int(row.get("beta1_ge_soft_stable", 0)) == 1 for row in dataset_rows):
            hard_datasets += 1
        if len(dataset_rows) == len(NOISE_RATES) and all(
            int(row.get("beta05_ge_soft_stable", 0)) == 1 and int(row.get("beta05_ge_beta1_stable", 0)) == 1
            for row in dataset_rows
        ):
            mid_datasets += 1
    hard_noise04 = sum(
        1
        for row in complete_settings
        if math.isclose(float(row["noise_rate"]), 0.4)
        and int(row.get("beta1_ge_soft_stable", 0)) == 1
        and int(row.get("beta1_ge_beta05_stable", 0)) == 1
    )
    mid_noise02 = sum(
        1
        for row in complete_settings
        if math.isclose(float(row["noise_rate"]), 0.2)
        and int(row.get("beta05_ge_soft_stable", 0)) == 1
        and int(row.get("beta05_ge_beta1_stable", 0)) == 1
    )
    shuffled_stable = sum(1 for row in complete_settings if int(row.get("beta1_gt_shuffled_stable", 0)) == 1)
    half_above_soft_stable = sum(1 for row in complete_settings if int(row.get("beta05_ge_soft_stable", 0)) == 1)
    beta1_wins = sum(1 for row in complete_settings if row.get("best_method") == HARD)
    beta05_wins = sum(1 for row in complete_settings if row.get("best_method") == "frozengrace_gb_mlp_softdist_beta05")
    soft_wins = sum(1 for row in complete_settings if row.get("best_method") == SOFT)
    details = [
        f"Stable hard datasets (beta=1 >= beta=0 for both noise rates): {hard_datasets}/{len(datasets)}.",
        f"Stable half-hard datasets (beta=0.5 >= beta=0 and beta=0.5 >= beta=1 for both noise rates): {mid_datasets}/{len(datasets)}.",
        f"Stable hard preference at uniform0.4: {hard_noise04}/{len(datasets)} datasets.",
        f"Stable half-hard preference at uniform0.2: {mid_noise02}/{len(datasets)} datasets.",
        f"Stable beta=0.5 > beta=0 settings: {half_above_soft_stable}/{len(complete_settings)}.",
        f"Stable beta=1 real > shuffled settings: {shuffled_stable}/{len(complete_settings)}.",
        f"Mean-best beta counts: beta=0: {soft_wins}, beta=0.5: {beta05_wins}, beta=1: {beta1_wins}.",
    ]
    if hard_datasets >= 2:
        conclusion = "5-seed supports a fixed hard-majority target as the main-method candidate."
    elif mid_datasets >= 2:
        conclusion = "5-seed supports fixed beta=0.5 as the main-method candidate."
    elif hard_noise04 >= 2 and mid_noise02 >= 2:
        conclusion = "5-seed supports a dataset/noise-dependent hardness pattern; trigger adaptive-hardness smoke instead of choosing one fixed beta."
    elif half_above_soft_stable == len(complete_settings):
        conclusion = (
            "5-seed does not give a stable fixed-hard or adaptive-hardness law. Use beta=0.5 as the conservative "
            "fixed default because it is the only hardness that stably improves over beta=0 in every setting; keep "
            "beta=1 as an aggressive/mid-noise variant rather than the default."
        )
    else:
        conclusion = "5-seed does not give a stable hardness law; choose the cross-setting average best beta conservatively and do not pursue adaptive hardness yet."
    return conclusion, details


def write_report(
    path: Path,
    agg: list[dict[str, object]],
    deltas: list[dict[str, object]],
    decisions: list[dict[str, object]],
    datasets: list[str],
    seeds: list[int] | None = None,
) -> None:
    seeds = seeds or [0]
    seed_count = len(seeds)
    agg_values = {(row["dataset"], float(row["noise_rate"]), row["method"]): row for row in agg}
    decision_values = {(row["dataset"], float(row["noise_rate"])): row for row in decisions}

    matrix_rows = []
    for dataset in datasets:
        for rate in NOISE_RATES:
            row = [dataset, fmt(rate, 1)]
            for method in METHODS:
                item = agg_values.get((dataset, rate, method))
                row.append("" if item is None else f"{fmt(item['mean'])} +/- {fmt(item['std'])}")
            decision = decision_values.get((dataset, rate), {})
            n_text = "" if decision.get("paired_n", "") == "" else f"/{decision.get('paired_n')}"
            row.extend([
                fmt(decision.get("beta1_minus_soft", "")),
                "" if decision.get("beta1_minus_soft_positive_count", "") == "" else f"{decision.get('beta1_minus_soft_positive_count')}{n_text}",
                fmt(decision.get("beta1_real_minus_shuffled", "")),
                "" if decision.get("beta1_real_minus_shuffled_positive_count", "") == "" else f"{decision.get('beta1_real_minus_shuffled_positive_count')}{n_text}",
                str(decision.get("decision", "")),
            ])
            matrix_rows.append(row)

    complete_settings = sum(1 for row in decisions if row.get("decision") != "missing")
    conclusion, detail_lines = _overall_conclusion(decisions, datasets, seed_count)
    report_label = "5-Seed" if seed_count > 1 else "Seed0"
    caution = (
        "Caution: this is a seed0 smoke test. Any main-method change needs 5-seed confirmation before it can support a paper claim."
        if seed_count <= 1
        else "Caution: these are 5-seed paired results, but any final paper claim still needs alignment with the full MLP robustness table."
    )

    lines = [
        f"# MLP Beta Hardness {report_label} Report",
        "",
        f"Protocol: SoftDist-MLP beta spectrum, uniform noise rates 0.2 and 0.4, percent split 0.2/0.1/0.7, seeds={','.join(map(str, seeds))}.",
        "",
        "## Accuracy Matrix",
        "",
        markdown_table(
            matrix_rows,
            [
                "dataset",
                "noise",
                "beta=0",
                "beta=0.5",
                "beta=1",
                "beta=1 shuffled",
                "beta1-soft",
                "beta1-soft +/n",
                "beta1-shuf",
                "beta1-shuf +/n",
                "decision",
            ],
        ),
        "",
        "## Directional Counts",
        "",
        f"- Complete settings: {complete_settings}/{len(datasets) * len(NOISE_RATES)}.",
        *[f"- {line}" for line in detail_lines],
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        caution,
        "",
        "## Outputs",
        "",
        "- `mlp_beta_hardness_aggregate.csv`",
        "- `mlp_beta_hardness_paired_deltas.csv`",
        "- `mlp_beta_hardness_decision.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_complete(rows: list[dict[str, object]], datasets: list[str], seeds: list[int]) -> None:
    expected = {
        (dataset, rate, method, seed)
        for dataset in datasets
        for rate in NOISE_RATES
        for method in METHODS
        for seed in seeds
    }
    observed = {
        (row["dataset"], float(row["noise_rate"]), row["method"], int(row["seed"]))
        for row in rows
        if row["dataset"] in datasets and float(row["noise_rate"]) in NOISE_RATES
    }
    missing = sorted(expected - observed)
    if missing:
        preview = ", ".join(str(item) for item in missing[:10])
        raise SystemExit(f"missing {len(missing)} expected rows; first missing: {preview}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MLP beta hardness smoke results.")
    parser.add_argument("--run_dir", default=str(RUN_DIR))
    parser.add_argument("--datasets", default="cora,citeseer,pubmed")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--require_complete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    datasets = parse_list(args.datasets)
    seeds = parse_seed_list(args.seeds)
    rows = [row for row in load_rows(run_dir) if row["dataset"] in datasets and float(row["noise_rate"]) in NOISE_RATES]
    if args.require_complete:
        validate_complete(rows, datasets, seeds)
    agg = aggregate(rows)
    deltas = paired_deltas(rows)
    decisions = decision_rows(agg, datasets, deltas=deltas, stability_threshold=min(4, len(seeds)))
    write_csv(run_dir / "mlp_beta_hardness_aggregate.csv", agg)
    write_csv(run_dir / "mlp_beta_hardness_paired_deltas.csv", deltas)
    write_csv(run_dir / "mlp_beta_hardness_decision.csv", decisions)
    write_report(run_dir / "mlp_beta_hardness_report.md", agg, deltas, decisions, datasets, seeds=seeds)
    print(f"WROTE {run_dir / 'mlp_beta_hardness_report.md'}")


if __name__ == "__main__":
    main()
