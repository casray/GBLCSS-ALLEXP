from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "mlp13_ballparam_sensitivity_20260703"

METHOD = "frozengrace_gb_mlp_softdist_beta05"
DATASETS = ("cora", "citeseer", "pubmed", "coauthorcs", "amazonpho")
SEEDS = tuple(range(5))
DEFAULT_CONFIG = (64, 32, 0.05)
MAX_NON_DEFAULT_STAGE_B = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _float(value, default=""):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _int(value, default=""):
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def config_id(max_balls: int, min_ball_size: int, split_improvement: float) -> str:
    return f"mb{int(max_balls)}_min{int(min_ball_size)}_si{float(split_improvement):g}"


def _config_from_row(row: dict) -> tuple[int, int, float]:
    return (
        int(float(row.get("gb_max_balls", DEFAULT_CONFIG[0]))),
        int(float(row.get("gb_min_ball_size", DEFAULT_CONFIG[1]))),
        float(row.get("gb_split_improvement", DEFAULT_CONFIG[2])),
    )


def normalize(row: dict, source: Path | str = "") -> dict[str, object] | None:
    method = (row.get("method") or row.get("base_method") or METHOD).strip()
    test = row.get("test") or row.get("test_acc") or ""
    if method != METHOD or test == "":
        return None
    max_balls, min_ball_size, split_improvement = _config_from_row(row)
    num_balls = _float(row.get("num_balls"))
    alpha_positive_train_count = _float(row.get("alpha_positive_train_count"))
    mean_train_per_ball = _float(row.get("mean_train_per_ball"))
    mean_train_per_ball_source = "raw"
    if mean_train_per_ball == "" and num_balls != "" and alpha_positive_train_count != "" and float(num_balls) > 0:
        mean_train_per_ball = float(alpha_positive_train_count) / float(num_balls)
        mean_train_per_ball_source = "alpha_positive_train_count_over_num_balls"
    return {
        "dataset": str(row["dataset"]).strip().lower(),
        "method": method,
        "seed": int(float(row["seed"])),
        "noise_type": str(row.get("noise_type", "uniform")).strip().lower(),
        "noise_rate": float(row.get("noise_rate", 0.4)),
        "status": str(row.get("status", "ok")).strip().lower(),
        "valid": _float(row.get("valid")),
        "test": float(test),
        "gb_max_balls": max_balls,
        "gb_min_ball_size": min_ball_size,
        "gb_split_improvement": split_improvement,
        "config_id": config_id(max_balls, min_ball_size, split_improvement),
        "num_balls": num_balls,
        "mean_ball_size": _float(row.get("mean_ball_size")),
        "mean_train_per_ball": mean_train_per_ball,
        "mean_train_per_ball_source": mean_train_per_ball_source,
        "noisy_train_purity_weighted": _float(row.get("noisy_train_purity_weighted")),
        "true_train_purity_weighted": _float(row.get("true_train_purity_weighted")),
        "alpha_positive_train_count": alpha_positive_train_count,
        "mean_alpha_train": _float(row.get("mean_alpha_train")),
        "split_hash": row.get("split_hash", ""),
        "noise_hash": row.get("noise_hash", ""),
        "source": str(source),
    }


def load_stage_rows(run_dir: Path, stage: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob(f"{stage}_*.csv")):
        if path.name.startswith("mlp13_"):
            continue
        for raw in read_csv(path):
            row = normalize(raw, path)
            if row is not None and row["status"] in {"", "ok"}:
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _min_metric(items: list[dict[str, object]], key: str) -> float | str:
    values = [float(row[key]) for row in items if row.get(key) != ""]
    return min(values) if values else ""


def _mean_metric(items: list[dict[str, object]], key: str) -> float | str:
    values = [float(row[key]) for row in items if row.get(key) != ""]
    return statistics.mean(values) if values else ""


def score_stage_a_candidates(rows: list[dict]) -> list[dict[str, object]]:
    normed = [row if "config_id" in row else normalize(row) for row in rows]
    normed = [row for row in normed if row is not None and row["status"] in {"", "ok"} and int(row["seed"]) == 0]
    default_id = config_id(*DEFAULT_CONFIG)
    default_by_dataset = {
        row["dataset"]: row
        for row in normed
        if row["config_id"] == default_id and row["dataset"] in DATASETS
    }

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normed:
        if row["dataset"] in DATASETS:
            grouped[row["config_id"]].append(row)

    out: list[dict[str, object]] = []
    for cid, items in sorted(grouped.items()):
        by_dataset = {row["dataset"]: row for row in items}
        deltas = []
        dataset_delta_fields = {}
        for dataset in DATASETS:
            row = by_dataset.get(dataset)
            default = default_by_dataset.get(dataset)
            if row is None or default is None:
                dataset_delta_fields[f"delta_{dataset}"] = ""
                continue
            delta = float(row["test"]) - float(default["test"])
            deltas.append(delta)
            dataset_delta_fields[f"delta_{dataset}"] = delta

        max_balls, min_ball_size, split_improvement = _config_from_row(items[0])
        min_num_balls = _min_metric(items, "num_balls")
        min_mean_train = _min_metric(items, "mean_train_per_ball")
        min_alpha_count = _min_metric(items, "alpha_positive_train_count")
        mean_noisy_purity = _mean_metric(items, "noisy_train_purity_weighted")
        complete_dataset_count = len({row["dataset"] for row in items})
        positive_dataset_count = sum(delta > 0 for delta in deltas)
        loss_gt_2pp_count = sum(delta < -0.02 for delta in deltas)

        hard_reject_reason = ""
        if complete_dataset_count < len(DATASETS):
            hard_reject_reason = "incomplete_datasets"
        elif min_num_balls != "" and float(min_num_balls) <= 1:
            hard_reject_reason = "collapse_num_balls"
        elif min_mean_train != "" and float(min_mean_train) < 2:
            hard_reject_reason = "low_mean_train_per_ball"
        elif min_alpha_count != "" and float(min_alpha_count) <= 0:
            hard_reject_reason = "zero_alpha_positive_train_count"
        elif loss_gt_2pp_count >= 2:
            hard_reject_reason = "loss_gt_2pp_on_at_least_2_datasets"

        degeneracy_reject_reason = ""
        if complete_dataset_count < len(DATASETS):
            degeneracy_reject_reason = "incomplete_datasets"
        elif min_num_balls != "" and float(min_num_balls) <= 1:
            degeneracy_reject_reason = "collapse_num_balls"
        elif min_mean_train != "" and float(min_mean_train) < 2:
            degeneracy_reject_reason = "low_mean_train_per_ball"
        elif min_alpha_count != "" and float(min_alpha_count) <= 0:
            degeneracy_reject_reason = "zero_alpha_positive_train_count"

        out.append({
            "config_id": cid,
            "gb_max_balls": max_balls,
            "gb_min_ball_size": min_ball_size,
            "gb_split_improvement": split_improvement,
            "is_default": int(cid == default_id),
            "complete_dataset_count": complete_dataset_count,
            "avg_delta_vs_default": _mean(deltas),
            "positive_dataset_count": positive_dataset_count,
            "loss_gt_2pp_count": loss_gt_2pp_count,
            "min_num_balls": min_num_balls,
            "min_mean_train_per_ball": min_mean_train,
            "min_alpha_positive_train_count": min_alpha_count,
            "mean_noisy_train_purity_weighted": mean_noisy_purity,
            "hard_reject_reason": "" if cid == default_id else hard_reject_reason,
            "degeneracy_reject_reason": "" if cid == default_id else degeneracy_reject_reason,
            **dataset_delta_fields,
        })
    return sorted(out, key=lambda row: row["config_id"])


def _candidate_sort_key(row: dict[str, object]):
    avg_delta = float(row["avg_delta_vs_default"]) if row["avg_delta_vs_default"] != "" else -999.0
    mean_purity = float(row["mean_noisy_train_purity_weighted"]) if row["mean_noisy_train_purity_weighted"] != "" else -1.0
    return (
        -avg_delta,
        -int(row["positive_dataset_count"]),
        -mean_purity,
        int(row["gb_max_balls"]),
        int(row["gb_min_ball_size"]),
        float(row["gb_split_improvement"]),
    )


def select_stage_b_configs(
    summaries: list[dict[str, object]],
    max_non_default: int = MAX_NON_DEFAULT_STAGE_B,
) -> list[dict[str, object]]:
    default_id = config_id(*DEFAULT_CONFIG)
    by_id = {row["config_id"]: row for row in summaries}
    selected: list[dict[str, object]] = []
    default_row = dict(by_id.get(default_id, {
        "config_id": default_id,
        "gb_max_balls": DEFAULT_CONFIG[0],
        "gb_min_ball_size": DEFAULT_CONFIG[1],
        "gb_split_improvement": DEFAULT_CONFIG[2],
    }))
    default_row["selected_reason"] = "default"
    selected.append(default_row)

    non_default = [row for row in summaries if row["config_id"] != default_id]
    passing = [row for row in non_default if row.get("hard_reject_reason", "") == ""]
    if passing:
        for row in sorted(passing, key=_candidate_sort_key)[:max_non_default]:
            chosen = dict(row)
            chosen["selected_reason"] = "selected_hard_pass"
            selected.append(chosen)
        return selected

    fallback = [row for row in non_default if row.get("degeneracy_reject_reason", "") == ""]
    if fallback:
        chosen = dict(sorted(fallback, key=_candidate_sort_key)[0])
        chosen["selected_reason"] = "fallback_non_degenerate"
        selected.append(chosen)
    return selected


def selected_config_tuples(selected_rows: list[dict[str, object]]) -> list[tuple[int, int, float]]:
    return [
        (int(row["gb_max_balls"]), int(row["gb_min_ball_size"]), float(row["gb_split_improvement"]))
        for row in selected_rows
    ]


def filter_selected_rows(rows: list[dict[str, object]], selected_configs: list[tuple[int, int, float]]) -> list[dict[str, object]]:
    selected_ids = {config_id(*config) for config in selected_configs}
    return [row for row in rows if row["config_id"] in selected_ids]


def stage_b_paired_deltas(rows: list[dict], selected_configs: list[tuple[int, int, float]]) -> list[dict[str, object]]:
    normed = [row if "config_id" in row else normalize(row) for row in rows]
    normed = [row for row in normed if row is not None and row["status"] in {"", "ok"}]
    selected_ids = {config_id(*config) for config in selected_configs}
    default_id = config_id(*DEFAULT_CONFIG)
    default_by_dataset_seed = {
        (row["dataset"], int(row["seed"])): row
        for row in normed
        if row["config_id"] == default_id and row["dataset"] in DATASETS
    }
    out: list[dict[str, object]] = []
    for row in sorted(normed, key=lambda r: (DATASETS.index(r["dataset"]), int(r["seed"]), r["config_id"])):
        if row["config_id"] not in selected_ids or row["dataset"] not in DATASETS:
            continue
        default = default_by_dataset_seed.get((row["dataset"], int(row["seed"])))
        if default is None:
            continue
        out.append({
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "config_id": row["config_id"],
            "gb_max_balls": row["gb_max_balls"],
            "gb_min_ball_size": row["gb_min_ball_size"],
            "gb_split_improvement": row["gb_split_improvement"],
            "test": float(row["test"]),
            "default_test": float(default["test"]),
            "delta_vs_default": float(row["test"]) - float(default["test"]),
        })
    return out


def aggregate_stage_b(rows: list[dict[str, object]], selected_configs: list[tuple[int, int, float]]) -> list[dict[str, object]]:
    selected_ids = {config_id(*config) for config in selected_configs}
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] in DATASETS and row["config_id"] in selected_ids and row["status"] in {"", "ok"}:
            grouped[(row["dataset"], row["config_id"])].append(row)
    delta_lookup = {
        (row["dataset"], int(row["seed"]), row["config_id"]): float(row["delta_vs_default"])
        for row in stage_b_paired_deltas(rows, selected_configs)
    }
    out: list[dict[str, object]] = []
    for (dataset, cid), items in sorted(grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), item[0][1])):
        tests = [float(row["test"]) for row in items]
        deltas = [delta_lookup[(row["dataset"], int(row["seed"]), row["config_id"])] for row in items if (row["dataset"], int(row["seed"]), row["config_id"]) in delta_lookup]
        max_balls, min_ball_size, split_improvement = _config_from_row(items[0])
        out.append({
            "dataset": dataset,
            "config_id": cid,
            "gb_max_balls": max_balls,
            "gb_min_ball_size": min_ball_size,
            "gb_split_improvement": split_improvement,
            "n": len(items),
            "test_mean": statistics.mean(tests),
            "test_std": _std(tests),
            "mean_delta_vs_default": _mean(deltas),
            "positive_seed_count": sum(delta > 0 for delta in deltas),
            "seeds": " ".join(str(row["seed"]) for row in sorted(items, key=lambda r: int(r["seed"]))),
            "num_balls_mean": _mean_metric(items, "num_balls"),
            "mean_train_per_ball_mean": _mean_metric(items, "mean_train_per_ball"),
            "noisy_train_purity_weighted_mean": _mean_metric(items, "noisy_train_purity_weighted"),
            "true_train_purity_weighted_mean": _mean_metric(items, "true_train_purity_weighted"),
            "alpha_positive_train_count_mean": _mean_metric(items, "alpha_positive_train_count"),
            "mean_alpha_train_mean": _mean_metric(items, "mean_alpha_train"),
        })
    return out


def stage_b_config_gates(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["config_id"]].append(row)
    default_id = config_id(*DEFAULT_CONFIG)
    default_by_dataset = {row["dataset"]: row for row in grouped.get(default_id, [])}
    best_by_dataset: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        candidates = [row for row in summary_rows if row["dataset"] == dataset]
        if candidates:
            best_by_dataset[dataset] = sorted(candidates, key=lambda row: -float(row["test_mean"]))[0]

    out: list[dict[str, object]] = []
    for cid, items in sorted(grouped.items()):
        deltas = [float(row["mean_delta_vs_default"]) for row in items if row["mean_delta_vs_default"] != ""]
        non_negative_dataset_count = sum(delta >= 0 for delta in deltas)
        min_delta = min(deltas) if deltas else ""
        avg_delta = statistics.mean(deltas) if deltas else ""
        pass_new_default = int(cid != default_id and avg_delta != "" and avg_delta >= 0.005 and non_negative_dataset_count >= 4 and float(min_delta) >= -0.005)
        out.append({
            "config_id": cid,
            "gb_max_balls": items[0]["gb_max_balls"],
            "gb_min_ball_size": items[0]["gb_min_ball_size"],
            "gb_split_improvement": items[0]["gb_split_improvement"],
            "dataset_count": len(items),
            "avg_mean_delta_vs_default": avg_delta,
            "non_negative_dataset_count": non_negative_dataset_count,
            "min_dataset_delta": min_delta,
            "passes_candidate_new_default": pass_new_default,
        })

    default_close_count = 0
    for dataset, default_row in default_by_dataset.items():
        best = best_by_dataset.get(dataset)
        if best is None:
            continue
        gap = float(best["test_mean"]) - float(default_row["test_mean"])
        if gap <= 0.01 or gap <= float(best["test_std"]):
            default_close_count += 1
    out.append({
        "config_id": "DEFAULT_ROBUSTNESS_GATE",
        "gb_max_balls": DEFAULT_CONFIG[0],
        "gb_min_ball_size": DEFAULT_CONFIG[1],
        "gb_split_improvement": DEFAULT_CONFIG[2],
        "dataset_count": len(default_by_dataset),
        "avg_mean_delta_vs_default": "",
        "non_negative_dataset_count": default_close_count,
        "min_dataset_delta": "",
        "passes_candidate_new_default": int(default_close_count >= 4),
    })
    return out


def completeness_rows(rows: list[dict[str, object]], selected_configs: list[tuple[int, int, float]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    selected_ids = {config_id(*config) for config in selected_configs}
    for row in rows:
        if row["dataset"] in DATASETS and row["config_id"] in selected_ids:
            grouped[(row["dataset"], row["config_id"])].add(int(row["seed"]))
    out = []
    for dataset in DATASETS:
        for cid in sorted(selected_ids):
            seeds = grouped.get((dataset, cid), set())
            out.append({
                "dataset": dataset,
                "config_id": cid,
                "expected": len(SEEDS),
                "observed": len(seeds),
                "missing": " ".join(str(seed) for seed in SEEDS if seed not in seeds),
                "seeds": " ".join(str(seed) for seed in sorted(seeds)),
            })
    return out


def _fmt(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):.2f}"


def _fmt_delta(value) -> str:
    if value == "":
        return ""
    return f"{100.0 * float(value):+.2f}pp"


def write_report(
    path: Path,
    stage_a_summary: list[dict[str, object]],
    selected: list[dict[str, object]],
    stage_b_summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    gates: list[dict[str, object]],
    complete: list[dict[str, object]],
) -> None:
    candidate_passes = [row for row in gates if row["config_id"] != "DEFAULT_ROBUSTNESS_GATE" and int(row["passes_candidate_new_default"]) == 1]
    default_gate = next((row for row in gates if row["config_id"] == "DEFAULT_ROBUSTNESS_GATE"), None)
    missing = [row for row in complete if int(row["observed"]) != int(row["expected"])]
    if candidate_passes:
        decision = "candidate_unified_default_needs_noise_spectrum_validation"
    elif default_gate and int(default_gate["passes_candidate_new_default"]) == 1:
        decision = "keep_default_low_sensitivity"
    else:
        decision = "appendix_sensitivity_only_no_default_change"

    lines = [
        "# MLP-13 Ball-Parameter Sensitivity Report",
        "",
        "## Material Passport",
        "- Method: `frozengrace_gb_mlp_softdist_beta05`.",
        "- Protocol: train-only uniform 0.4, percent split 0.2/0.1/0.7.",
        "- Scope: fixed-rule + self-stopping ball construction sensitivity only.",
        "- No dataset-specific tuning is allowed.",
        "",
        f"Decision: `{decision}`",
        "",
        "## Stage A Selected Configs",
        "| Config | max_balls | min_ball_size | split_improvement | reason | seed0 avg delta | positive datasets | reject |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['config_id']} | {row['gb_max_balls']} | {row['gb_min_ball_size']} | "
            f"{row['gb_split_improvement']} | {row['selected_reason']} | "
            f"{_fmt_delta(row.get('avg_delta_vs_default', ''))} | {row.get('positive_dataset_count', '')} | "
            f"{row.get('hard_reject_reason', '')} |"
        )
    lines.extend([
        "",
        "## Stage A Grid Summary",
        "| Config | seed0 avg delta | positive | loss >2pp | min balls | mean train/ball min | alpha count min | noisy purity mean | reject |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted(stage_a_summary, key=_candidate_sort_key):
        lines.append(
            f"| {row['config_id']} | {_fmt_delta(row['avg_delta_vs_default'])} | "
            f"{row['positive_dataset_count']} | {row['loss_gt_2pp_count']} | {row['min_num_balls']} | "
            f"{row['min_mean_train_per_ball']} | {row['min_alpha_positive_train_count']} | "
            f"{row['mean_noisy_train_purity_weighted']} | {row['hard_reject_reason']} |"
        )
    lines.extend([
        "",
        "## Stage B Summary",
        "| Dataset | Config | n | test mean±std | delta vs default | pos seeds | balls | train/ball | noisy purity | true purity | alpha count | mean alpha |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in stage_b_summary:
        lines.append(
            f"| {row['dataset']} | {row['config_id']} | {row['n']} | "
            f"{_fmt(row['test_mean'])}+-{_fmt(row['test_std'])} | {_fmt_delta(row['mean_delta_vs_default'])} | "
            f"{row['positive_seed_count']}/{row['n']} | {row['num_balls_mean']} | {row['mean_train_per_ball_mean']} | "
            f"{row['noisy_train_purity_weighted_mean']} | {row['true_train_purity_weighted_mean']} | "
            f"{row['alpha_positive_train_count_mean']} | {row['mean_alpha_train_mean']} |"
        )
    lines.extend([
        "",
        "## Stage B Gates",
        "| Config | avg delta | non-negative datasets | min dataset delta | pass candidate new default |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in gates:
        lines.append(
            f"| {row['config_id']} | {_fmt_delta(row['avg_mean_delta_vs_default'])} | "
            f"{row['non_negative_dataset_count']} | {_fmt_delta(row['min_dataset_delta'])} | "
            f"{row['passes_candidate_new_default']} |"
        )
    lines.extend([
        "",
        "## Paired Delta Rows",
        "| Dataset | Seed | Config | test | default | delta |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in paired:
        if row["config_id"] == config_id(*DEFAULT_CONFIG):
            continue
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {row['config_id']} | "
            f"{_fmt(row['test'])} | {_fmt(row['default_test'])} | {_fmt_delta(row['delta_vs_default'])} |"
        )
    lines.extend([
        "",
        "## Completeness",
        f"- Missing selected dataset/config seed groups: {len(missing)}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stage_a_rows = load_stage_rows(run_dir, "stageA")
    stage_b_new_rows = load_stage_rows(run_dir, "stageB")
    stage_a_summary = score_stage_a_candidates(stage_a_rows)
    selected = select_stage_b_configs(stage_a_summary)
    selected_configs = selected_config_tuples(selected)
    selected_stage_a_seed0 = filter_selected_rows(stage_a_rows, selected_configs)
    stage_b_rows = selected_stage_a_seed0 + filter_selected_rows(stage_b_new_rows, selected_configs)
    paired = stage_b_paired_deltas(stage_b_rows, selected_configs)
    stage_b_summary = aggregate_stage_b(stage_b_rows, selected_configs)
    gates = stage_b_config_gates(stage_b_summary)
    complete = completeness_rows(stage_b_rows, selected_configs)

    write_csv(run_dir / "mlp13_stageA_raw.csv", stage_a_rows)
    write_csv(run_dir / "mlp13_stageA_grid_summary.csv", stage_a_summary)
    write_csv(run_dir / "mlp13_stageA_selected_configs.csv", selected)
    write_json(run_dir / "mlp13_stageA_selected_configs.json", selected)
    write_csv(run_dir / "mlp13_stageB_raw.csv", stage_b_rows)
    write_csv(run_dir / "mlp13_stageB_paired_deltas.csv", paired)
    write_csv(run_dir / "mlp13_stageB_summary.csv", stage_b_summary)
    write_csv(run_dir / "mlp13_stageB_config_gates.csv", gates)
    write_csv(run_dir / "mlp13_completeness.csv", complete)
    write_report(run_dir / "mlp13_report.md", stage_a_summary, selected, stage_b_summary, paired, gates, complete)

    print(f"MLP13_STAGE_A_ROWS={len(stage_a_rows)}")
    print(f"MLP13_SELECTED_CONFIGS={','.join(row['config_id'] for row in selected)}")
    print(f"MLP13_STAGE_B_ROWS={len(stage_b_rows)}")
    print(f"WROTE {run_dir / 'mlp13_report.md'}")


if __name__ == "__main__":
    main()
